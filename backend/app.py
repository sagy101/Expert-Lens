"""
FastAPI backend for Expert Lens.

Endpoints:
  POST /api/infer    — run text through the MoE model, return routing data
  GET  /api/model-info — return architecture metadata for visualization
"""

import os

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
MODEL_PATH = os.path.join(PRETRAINED_DIR, "model.pt")
META_PATH = os.path.join(PRETRAINED_DIR, "meta.pt")

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"

model = None
config = None
stoi = {}
itos = {}


def load_model():
    global model, config, stoi, itos
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No pretrained model found at {MODEL_PATH}. Run `python train.py` first."
        )
    meta = torch.load(META_PATH, map_location="cpu", weights_only=False)
    stoi = meta["stoi"]
    itos = meta["itos"]

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = TinyMoETransformer(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Model loaded on {device} (val_loss={checkpoint.get('val_loss', '?')})")


class InferRequest(BaseModel):
    text: str
    max_new_tokens: int = 64
    temperature: float = 0.8


class InferResponse(BaseModel):
    input_tokens: list[str]
    generated_tokens: list[str]
    layers: list[dict]
    generated_layers: list[list[dict]]


@app.on_event("startup")
def startup():
    load_model()


@app.get("/api/model-info")
def model_info():
    return {
        "n_layer": config.n_layer,
        "n_expert": config.n_expert,
        "top_k": config.top_k,
        "n_embd": config.n_embd,
        "n_head": config.n_head,
        "expert_dim": config.expert_dim,
        "vocab_size": config.vocab_size,
        "block_size": config.block_size,
        "n_params": sum(p.numel() for p in model.parameters()),
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
    text = req.text[-config.block_size :]
    if not text:
        text = " "

    ids = _encode(text)
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        _, _, prefill_info = model(idx)

        generated_idx, gen_moe_steps = model.generate(
            idx, max_new_tokens=req.max_new_tokens, temperature=req.temperature
        )

    input_tokens = list(text)
    gen_ids = generated_idx[0, len(ids) :].tolist()
    generated_tokens = _decode(gen_ids)

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
