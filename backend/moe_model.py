"""
Tiny Mixture-of-Experts (MoE) Transformer — built from scratch.

Architecture
────────────
  Input tokens
       │
  Embedding + Positional Encoding
       │
  ┌────┴────┐
  │  MoE    │ × N_LAYERS
  │  Block  │   Each block: LayerNorm → MultiHeadAttention → LayerNorm → MoE FFN
  └────┬────┘
       │
  LayerNorm → Linear → Softmax   (next-token prediction)

Each **MoE FFN** contains:
  • A *router* (linear projection → softmax) that scores every expert
  • K *expert* feed-forward networks (small 2-layer MLPs)
  • A top-k selection that activates only a few experts per token

This is the same pattern used by Mixtral-8x7B, Qwen-MoE, and other
production MoE models — just at a scale where every neuron is visible.
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MoEConfig:
    vocab_size: int = 65          # tiny Shakespeare character vocab
    block_size: int = 256         # max context length
    n_embd: int = 192             # embedding dimension
    n_head: int = 6               # attention heads
    n_layer: int = 3              # transformer layers
    n_expert: int = 8             # experts per MoE layer
    top_k: int = 2                # experts activated per token
    expert_dim: int = 384         # hidden dim inside each expert
    dropout: float = 0.1
    bias: bool = False


class ExpertFFN(nn.Module):
    """A single expert: small two-layer MLP with GELU activation."""

    def __init__(self, config: MoEConfig):
        super().__init__()
        self.w1 = nn.Linear(config.n_embd, config.expert_dim, bias=config.bias)
        self.w2 = nn.Linear(config.expert_dim, config.n_embd, bias=config.bias)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.act(self.w1(x)))


class Router(nn.Module):
    """Learned gating network that assigns tokens to experts."""

    def __init__(self, config: MoEConfig):
        super().__init__()
        self.gate = nn.Linear(config.n_embd, config.n_expert, bias=False)
        self.top_k = config.top_k
        self.n_expert = config.n_expert

    def forward(self, x: torch.Tensor):
        logits = self.gate(x)                               # (B*T, n_expert)
        probs = F.softmax(logits, dim=-1)                   # router probabilities
        topk_probs, topk_indices = probs.topk(self.top_k, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)  # renormalize
        return topk_probs, topk_indices, probs


class MoELayer(nn.Module):
    """Mixture-of-Experts feed-forward layer."""

    def __init__(self, config: MoEConfig):
        super().__init__()
        self.experts = nn.ModuleList([ExpertFFN(config) for _ in range(config.n_expert)])
        self.router = Router(config)
        self.n_expert = config.n_expert
        self.top_k = config.top_k
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor):
        B, T, D = x.shape
        x_flat = x.view(-1, D)                             # (N, D)  where N = B*T

        topk_probs, topk_indices, router_probs = self.router(x_flat)

        # Compute all experts in one vectorised pass (fast on GPU/MPS)
        all_expert_out = torch.stack(
            [expert(x_flat) for expert in self.experts], dim=1
        )                                                   # (N, n_expert, D)

        # Gather only the top-k expert outputs
        idx_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, D)  # (N, top_k, D)
        selected = torch.gather(all_expert_out, 1, idx_expanded)     # (N, top_k, D)

        # Weighted sum of selected experts
        out = (topk_probs.unsqueeze(-1) * selected).sum(dim=1)       # (N, D)

        out = self.dropout(out)

        # Load-balancing auxiliary loss (Switch Transformer style).
        # Encourages the router to distribute tokens evenly across experts.
        # f_i = fraction of tokens routed to expert i
        # p_i = mean router probability for expert i
        # loss = n_expert * sum(f_i * p_i)  — minimised when uniform
        N = x_flat.shape[0]
        one_hot = F.one_hot(topk_indices, self.n_expert).float()      # (N, top_k, E)
        tokens_per_expert = one_hot.sum(dim=1).sum(dim=0) / N         # (E,)
        mean_probs = router_probs.mean(dim=0)                         # (E,)
        aux_loss = self.n_expert * (tokens_per_expert * mean_probs).sum()

        return out.view(B, T, D), aux_loss, {
            "router_probs": router_probs.view(B, T, -1).detach(),
            "topk_indices": topk_indices.view(B, T, -1).detach(),
            "topk_probs": topk_probs.view(B, T, -1).detach(),
        }


class CausalSelfAttention(nn.Module):
    def __init__(self, config: MoEConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.dropout_p = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor):
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.dropout_p if self.training else 0.0,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, D)
        return self.resid_dropout(self.proj(y))


class MoEBlock(nn.Module):
    """Transformer block: Attention + MoE FFN (with residual connections)."""

    def __init__(self, config: MoEConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.moe = MoELayer(config)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln1(x))
        moe_out, aux_loss, moe_info = self.moe(self.ln2(x))
        x = x + moe_out
        return x, aux_loss, moe_info


class TinyMoETransformer(nn.Module):
    """Complete character-level MoE language model."""

    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([MoEBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.tok_emb.weight = self.head.weight  # weight tying

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        _, T = idx.shape
        assert T <= self.config.block_size, f"Sequence length {T} > block_size {self.config.block_size}"

        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = self.drop(tok + pos)

        all_moe_info = []
        total_aux_loss = 0.0
        for block in self.blocks:
            x, aux_loss, moe_info = block(x)
            all_moe_info.append(moe_info)
            total_aux_loss = total_aux_loss + aux_loss

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss = ce_loss + 0.01 * total_aux_loss   # small weight so it guides without dominating

        return logits, loss, all_moe_info

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 0.8, top_k: int = 40):
        all_moe_info_steps = []
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _, moe_info = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
            all_moe_info_steps.append(moe_info)
        return idx, all_moe_info_steps
