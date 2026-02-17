"""
FastAPI backend for Expert Lens.

Endpoints:
  POST /api/infer    — run text through the MoE model, return routing data
  GET  /api/model-info — return architecture metadata for visualization
  GET  /api/expert-profile — profile what each expert specializes in
"""

from dotenv import load_dotenv
load_dotenv(dotenv_path="../.env")

import os
import logging

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from moe_model import MoEConfig, TinyMoETransformer

app = FastAPI(title="Expert Lens API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

PRETRAINED_DIR = os.path.join(os.path.dirname(__file__), "pretrained")

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"

# Registry: model_type -> { model, config, encode_fn, decode_fn, token_type }
models: dict[str, dict] = {}


def _load_char_model():
    """Load the character-level model."""
    model_path = os.path.join(PRETRAINED_DIR, "model.pt")
    meta_path = os.path.join(PRETRAINED_DIR, "meta.pt")
    if not os.path.exists(model_path):
        print("Character model not found — skipping.")
        return
    meta = torch.load(meta_path, map_location="cpu", weights_only=False)
    stoi = meta["stoi"]
    itos = meta["itos"]
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    cfg = checkpoint["config"]
    mdl = TinyMoETransformer(cfg).to(device)
    mdl.load_state_dict(checkpoint["model_state_dict"])
    mdl.eval()
    models["char"] = {
        "model": mdl,
        "config": cfg,
        "encode": lambda text: [stoi.get(ch, 0) for ch in text],
        "decode": lambda ids: [itos.get(i, "?") for i in ids],
        "tokenize_display": lambda text: list(text),
        "decode_display": lambda ids: [itos.get(i, "?") for i in ids],
        "token_type": "char",
    }
    print(f"Char model loaded on {device} (val_loss={checkpoint.get('val_loss', '?')})")


def _load_bpe_model():
    """Load the BPE-tokenized model."""
    model_path = os.path.join(PRETRAINED_DIR, "model_bpe.pt")
    tokenizer_path = os.path.join(PRETRAINED_DIR, "bpe_tokenizer.json")
    if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
        print("BPE model not found — skipping. Run `python train_bpe.py` to train.")
        return
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    cfg = checkpoint["config"]
    mdl = TinyMoETransformer(cfg).to(device)
    mdl.load_state_dict(checkpoint["model_state_dict"])
    mdl.eval()
    def _bpe_tokenize_display(text: str) -> list[str]:
        """Get display strings for input tokens, preserving whitespace."""
        enc = tokenizer.encode(text)
        offsets = enc.offsets
        result = []
        prev_end = 0
        for start, end in offsets:
            # Include any gap (whitespace) before this token
            result.append(text[prev_end:end])
            prev_end = end
        return result

    def _bpe_decode_display(ids: list[int]) -> list[str]:
        """Decode generated token ids, preserving whitespace via incremental decode."""
        if not ids:
            return []
        result = []
        prev = ""
        for i in range(1, len(ids) + 1):
            full = tokenizer.decode(ids[:i])
            result.append(full[len(prev):])
            prev = full
        return result

    models["bpe"] = {
        "model": mdl,
        "config": cfg,
        "encode": lambda text: tokenizer.encode(text).ids,
        "decode": _bpe_decode_display,
        "tokenize_display": _bpe_tokenize_display,
        "decode_display": _bpe_decode_display,
        "token_type": "bpe",
        "tokenizer": tokenizer,
    }
    print(f"BPE model loaded on {device} (val_loss={checkpoint.get('val_loss', '?')})")


def load_models():
    _load_char_model()
    _load_bpe_model()
    if not models:
        raise FileNotFoundError("No pretrained models found. Run train.py or train_bpe.py first.")


def _get_model(model_type: str = "char") -> dict:
    """Get a model registry entry, defaulting to char."""
    if model_type in models:
        return models[model_type]
    return next(iter(models.values()))


# Backward-compatible globals (used by expert_profile)
model = None
config = None


class InferRequest(BaseModel):
    text: str
    max_new_tokens: int = 64
    temperature: float = 0.8
    model_type: str = "char"


class InferResponse(BaseModel):
    input_tokens: list[str]
    generated_tokens: list[str]
    layers: list[dict]
    generated_layers: list[list[dict]]


@app.on_event("startup")
def startup():
    global model, config
    load_models()
    # Set backward-compatible globals to char model (used by expert_profile)
    default = _get_model("char")
    model = default["model"]
    config = default["config"]


@app.get("/api/model-info")
def model_info(model_type: str = "char"):
    reg = _get_model(model_type)
    mdl, cfg = reg["model"], reg["config"]
    total_params = sum(p.numel() for p in mdl.parameters())

    expert_params_each = 2 * cfg.n_embd * cfg.expert_dim + cfg.expert_dim + cfg.n_embd
    total_expert_params = cfg.n_layer * cfg.n_expert * expert_params_each
    active_expert_params = cfg.n_layer * cfg.top_k * expert_params_each
    shared_params = total_params - total_expert_params
    active_params = shared_params + active_expert_params

    return {
        "n_layer": cfg.n_layer,
        "n_expert": cfg.n_expert,
        "top_k": cfg.top_k,
        "n_embd": cfg.n_embd,
        "n_head": cfg.n_head,
        "expert_dim": cfg.expert_dim,
        "vocab_size": cfg.vocab_size,
        "block_size": cfg.block_size,
        "n_params": total_params,
        "n_active_params": active_params,
        "model_type": reg["token_type"],
        "available_models": list(models.keys()),
    }


def _encode(text: str) -> list[int]:
    return [stoi.get(ch, 0) for ch in text]


def _decode(ids: list[int]) -> list[str]:
    return [itos.get(i, "?") for i in ids]


def _moe_info_to_dict(moe_info_list):
    """Convert a list of per-layer moe_info dicts to serializable format."""
    layers = []
    for layer_info in moe_info_list:
        layers.append({
            "router_probs": layer_info["router_probs"][0].cpu().tolist(),
            "topk_indices": layer_info["topk_indices"][0].cpu().tolist(),
            "topk_probs": layer_info["topk_probs"][0].cpu().tolist(),
        })
    return layers


@app.post("/api/infer", response_model=InferResponse)
def infer(req: InferRequest):
    reg = _get_model(req.model_type)
    mdl, cfg = reg["model"], reg["config"]
    encode_fn, decode_fn = reg["encode"], reg["decode_display"]
    tokenize_display = reg["tokenize_display"]

    text = req.text[-cfg.block_size :]
    if not text:
        text = " "

    ids = encode_fn(text)
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        _, _, prefill_info = mdl(idx)

        generated_idx, gen_moe_steps = mdl.generate(
            idx, max_new_tokens=req.max_new_tokens, temperature=req.temperature
        )

    input_tokens = tokenize_display(text)
    gen_ids = generated_idx[0, len(ids) :].tolist()
    generated_tokens = decode_fn(gen_ids)

    prefill_layers = _moe_info_to_dict(prefill_info)

    generated_layers = []
    for step_info in gen_moe_steps:
        generated_layers.append(_moe_info_to_dict(step_info))

    return InferResponse(
        input_tokens=input_tokens,
        generated_tokens=generated_tokens,
        layers=prefill_layers,
        generated_layers=generated_layers,
    )


from samples import DOMAIN_SAMPLES, SAMPLE_TEXTS

VOWELS = set("aeiouAEIOU")

DEMO_SENTENCE = "The little cat jumped over a big, shiny red ball."


def _classify_char(ch: str, pos: int, text: str):
    """Classify a character into simple, human-readable categories."""
    result = {"char_type": "other", "word_pos": None}

    if ch == " ":
        result["char_type"] = "space"
    elif ch in ".!?":
        result["char_type"] = "sentence_end"
    elif ch in ",;:":
        result["char_type"] = "pause"
    elif ch in "\"'()":
        result["char_type"] = "quote"
    elif ch.isupper():
        result["char_type"] = "uppercase"
    elif ch.islower():
        result["char_type"] = "vowel" if ch in VOWELS else "consonant"

    if ch.isalpha():
        prev_alpha = pos > 0 and text[pos - 1].isalpha()
        next_alpha = pos < len(text) - 1 and text[pos + 1].isalpha()
        if not prev_alpha and next_alpha:
            result["word_pos"] = "first"
        elif prev_alpha and not next_alpha:
            result["word_pos"] = "last"
        elif not prev_alpha and not next_alpha:
            result["word_pos"] = "single"
        else:
            result["word_pos"] = "middle"

    return result


def _simple_role_name(char_type_counts, word_pos_counts, total):
    """Generate a plain-English role name anyone can understand."""
    if total == 0:
        return "Inactive"

    ct = {k: v / total * 100 for k, v in char_type_counts.items()}
    wp = {k: v / total * 100 for k, v in word_pos_counts.items()}

    # --- Check for clear dominant signals ---
    if ct.get("space", 0) > 35:
        return "The Spaces Expert"
    if ct.get("sentence_end", 0) > 12:
        return "The Sentence Ender"
    if ct.get("pause", 0) > 10:
        return "The Comma & Pause Expert"
    if ct.get("quote", 0) > 8:
        return "The Dialogue Expert"

    # --- Word position + vowel/consonant ---
    vowel_pct = ct.get("vowel", 0)
    cons_pct = ct.get("consonant", 0)
    first_pct = wp.get("first", 0)
    last_pct = wp.get("last", 0)
    mid_pct = wp.get("middle", 0)
    upper_pct = ct.get("uppercase", 0)

    # Strong positional signal
    if first_pct > 28 and first_pct > last_pct and first_pct > mid_pct:
        if upper_pct > 10:
            return "The Capital Starter"
        if cons_pct > vowel_pct:
            return "The Word Opener"
        return "The First Letter Expert"

    if last_pct > 28 and last_pct > first_pct and last_pct > mid_pct:
        if vowel_pct > cons_pct:
            return "The Soft Ending Expert"
        return "The Word Finisher"

    if mid_pct > 35:
        if vowel_pct > cons_pct + 5:
            return "The Inner Vowel Expert"
        if cons_pct > vowel_pct + 5:
            return "The Inner Consonant Expert"
        return "The Middle Letter Expert"

    # Just vowel/consonant dominant
    if vowel_pct > cons_pct + 10:
        if last_pct > first_pct:
            return "The Vowel & Ending Expert"
        return "The Vowel Expert"
    if cons_pct > vowel_pct + 10:
        if first_pct > last_pct:
            return "The Consonant & Opener Expert"
        return "The Consonant Expert"

    # Fallback
    return "The General Expert"


def _make_roles_unique(experts_out):
    """Ensure every expert in a layer has a unique role name."""
    seen = {}
    for e in experts_out:
        name = e["role"]
        if name in seen:
            seen[name] += 1
            top_chars = [c["char"] for c in e["top_chars"] if c["char"].strip()][:3]
            if top_chars:
                e["role"] = f'{name} ({", ".join(top_chars)})'
            else:
                e["role"] = f'{name} #{seen[name]}'
        else:
            seen[name] = 1


def _get_example_words(text: str, expert_id: int, topk_indices):
    """For a given expert, find words and mark which characters it handles."""
    words = []
    i = 0
    while i < len(text):
        if text[i].isalpha():
            start = i
            while i < len(text) and text[i].isalpha():
                i += 1
            word = text[start:i]
            highlights = []
            for ci in range(start, i):
                if ci < topk_indices.shape[0]:
                    handled = expert_id in topk_indices[ci].tolist()
                    highlights.append(handled)
                else:
                    highlights.append(False)
            if any(highlights):
                words.append({"word": word, "highlights": highlights})
        else:
            i += 1
    return words


@app.get("/api/expert-profile")
def expert_profile(model_type: str = "char"):
    """Run sample texts through the model, aggregate routing, return per-expert token profiles."""
    from collections import Counter

    reg = _get_model(model_type)
    mdl, cfg = reg["model"], reg["config"]
    encode_fn = reg["encode"]
    is_bpe = reg["token_type"] == "bpe"

    n_layers = cfg.n_layer
    n_experts = cfg.n_expert

    # Per layer, per expert: counters
    char_type_counts: list[list[Counter]] = [
        [Counter() for _ in range(n_experts)] for _ in range(n_layers)
    ]
    word_pos_counts: list[list[Counter]] = [
        [Counter() for _ in range(n_experts)] for _ in range(n_layers)
    ]
    expert_chars: list[list[Counter]] = [
        [Counter() for _ in range(n_experts)] for _ in range(n_layers)
    ]
    expert_totals: list[list[int]] = [
        [0] * n_experts for _ in range(n_layers)
    ]

    for text, _domain in DOMAIN_SAMPLES:
        ids = encode_fn(text)
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            _, _, moe_info = mdl(idx)

        if is_bpe:
            tokens_display = reg["tokenize_display"](text)
        else:
            tokens_display = list(text)

        for li, layer_info in enumerate(moe_info):
            topk_idx = layer_info["topk_indices"][0]  # (T, top_k)
            for ti in range(topk_idx.shape[0]):
                if ti >= len(tokens_display):
                    continue
                tok_str = tokens_display[ti]
                if not is_bpe:
                    info = _classify_char(tok_str, ti, text)
                else:
                    info = {"char_type": "subword", "word_pos": None}
                for ki in range(topk_idx.shape[1]):
                    eid = topk_idx[ti, ki].item()
                    expert_totals[li][eid] += 1
                    expert_chars[li][eid][tok_str] += 1
                    char_type_counts[li][eid][info["char_type"]] += 1
                    if info["word_pos"]:
                        word_pos_counts[li][eid][info["word_pos"]] += 1

    # --- Color-coded demo sentence per layer ---
    demo_ids = encode_fn(DEMO_SENTENCE)
    demo_idx = torch.tensor([demo_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        _, _, demo_moe_info = mdl(demo_idx)

    if is_bpe:
        demo_tokens = reg["tokenize_display"](DEMO_SENTENCE)
    else:
        demo_tokens = list(DEMO_SENTENCE)

    demo_sentences = []
    for li, layer_info in enumerate(demo_moe_info):
        topk_idx = layer_info["topk_indices"][0]  # (T, top_k)
        chars = []
        for ti, tok in enumerate(demo_tokens):
            if ti < topk_idx.shape[0]:
                primary_expert = topk_idx[ti, 0].item()
                chars.append({"char": tok, "expert": primary_expert})
            else:
                chars.append({"char": tok, "expert": -1})
        demo_sentences.append(chars)

    # --- Build per-expert example words/tokens from demo sentence ---
    demo_example_words: list[list[list[dict]]] = []
    for li, layer_info in enumerate(demo_moe_info):
        topk_idx = layer_info["topk_indices"][0]
        layer_examples = []
        for ei in range(n_experts):
            if is_bpe:
                # For BPE: show which tokens this expert handles
                token_examples = []
                for ti, tok in enumerate(demo_tokens):
                    if ti < topk_idx.shape[0] and ei in topk_idx[ti].tolist():
                        token_examples.append({"word": tok, "highlights": [True] * len(tok)})
                layer_examples.append(token_examples[:5])
            else:
                words = _get_example_words(DEMO_SENTENCE, ei, topk_idx)
                layer_examples.append(words[:5])
        demo_example_words.append(layer_examples)

    llm_errors = []

    # Build per-expert stats (used by both LLM and programmatic labeling)
    layers_out = []
    for li in range(n_layers):
        experts_raw = []
        for ei in range(n_experts):
            total = expert_totals[li][ei]
            top_chars = expert_chars[li][ei].most_common(6)
            ct_pcts = {k: v / total * 100 for k, v in char_type_counts[li][ei].items()} if total else {}
            wp_pcts = {k: v / total * 100 for k, v in word_pos_counts[li][ei].items()} if total else {}

            experts_raw.append({
                "expert_id": ei,
                "total_activations": total,
                "top_chars": [
                    {"char": ch, "count": c, "pct": round(c / total * 100, 1) if total > 0 else 0}
                    for ch, c in top_chars
                ],
                "example_words": demo_example_words[li][ei],
                "char_type_pcts": ct_pcts,
                "word_pos_pcts": wp_pcts,
            })

        # Try LLM labeling, fall back to programmatic
        llm_labels = None
        try:
            from llm_labeler import label_experts_with_llm
            llm_labels = label_experts_with_llm(experts_raw, li)
            logging.info("LLM labeling succeeded for layer %d", li)
        except Exception as exc:
            msg = f"LLM labeling failed for layer {li}: {exc}"
            logging.warning("%s — using programmatic fallback", msg)
            llm_errors.append(msg)

        experts_out = []
        for ei, raw in enumerate(experts_raw):
            role = _simple_role_name(char_type_counts[li][ei], word_pos_counts[li][ei], raw["total_activations"])
            title = role
            description = ""

            domain = ""
            if llm_labels:
                lbl = llm_labels[ei]
                title = lbl.get("title", role)
                domain = lbl.get("domain", "")
                description = lbl.get("description", "")

            experts_out.append({
                "expert_id": ei,
                "role": title,
                "domain": domain,
                "description": description,
                "total_activations": raw["total_activations"],
                "top_chars": raw["top_chars"],
                "example_words": raw["example_words"],
            })

        if not llm_labels:
            _make_roles_unique(experts_out)
        layers_out.append(experts_out)

    return {
        "layers": layers_out,
        "demo_sentence": DEMO_SENTENCE,
        "demo_layers": demo_sentences,
        "warnings": llm_errors,
        "sample_count": len(DOMAIN_SAMPLES),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
