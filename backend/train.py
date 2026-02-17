"""
Train the tiny MoE transformer on TinyStories.

TinyStories (Microsoft Research, 2023) is a dataset of simple short stories
specifically designed to produce coherent text from very small language models.

Usage:
    python train.py                      # train from scratch
    python train.py --resume             # resume from checkpoint
    python train.py --epochs 60          # custom epoch count

Data splits:
    80% train · 10% validation (model selection) · 10% test (final evaluation)
"""

import os
import time
import argparse
import subprocess

import torch
from tqdm import tqdm
from moe_model import MoEConfig, TinyMoETransformer

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DATA_PATH = os.path.join(DATA_DIR, "tinystories.txt")
PRETRAINED_DIR = os.path.join(os.path.dirname(__file__), "pretrained")
MODEL_PATH = os.path.join(PRETRAINED_DIR, "model.pt")
META_PATH = os.path.join(PRETRAINED_DIR, "meta.pt")

TINYSTORIES_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
MAX_CHARS = 10_000_000  # use first ~10M chars (~2M tokens) — enough for a tiny model


def download_data():
    if os.path.exists(DATA_PATH):
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    raw_path = os.path.join(DATA_DIR, "raw_tinystories.txt")
    if not os.path.exists(raw_path):
        print("Downloading TinyStories dataset (this may take a minute)...")
        subprocess.run(["curl", "-L", TINYSTORIES_URL, "-o", raw_path], check=True)
    print("Preparing dataset (first 10M chars)...")
    with open(raw_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read(MAX_CHARS)
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved {len(text):,} characters to {DATA_PATH}")


def load_data():
    download_data()
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    return data, stoi, itos, len(chars)


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix]).to(device)
    return x, y


@torch.no_grad()
def evaluate(model, data, block_size, batch_size, device, n_batches=50):
    """Evaluate average loss on a data split."""
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = get_batch(data, block_size, batch_size, device)
        _, loss, _ = model(x, y)
        losses.append(loss.item())
    return sum(losses) / len(losses)


def train(args):
    data, stoi, itos, vocab_size = load_data()

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")

    # 80 / 10 / 10 split
    n = len(data)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    train_data = data[:n_train]
    val_data = data[n_train : n_train + n_val]
    test_data = data[n_train + n_val :]
    print(f"Data splits — train: {len(train_data):,}  val: {len(val_data):,}  test: {len(test_data):,}")

    config = MoEConfig(vocab_size=vocab_size)

    if args.resume and os.path.exists(MODEL_PATH):
        print("Resuming from checkpoint...")
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        model = TinyMoETransformer(config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint.get("epoch", 0)
    else:
        model = TinyMoETransformer(config).to(device)
        start_epoch = 0

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    batch_size = 64
    block_size = config.block_size
    best_val_loss = float("inf")

    # AMP — automatic mixed precision for faster training
    amp_dtype = torch.float16 if device == "cuda" else torch.bfloat16
    use_amp = device in ("cuda", "mps")
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))  # GradScaler only helps on CUDA

    epoch_bar = tqdm(range(start_epoch, args.epochs), desc="Training", unit="epoch")

    for epoch in epoch_bar:
        model.train()
        total_loss = 0
        n_batches = 200
        epoch_start = time.time()

        batch_bar = tqdm(range(n_batches), desc=f"  Epoch {epoch+1}/{args.epochs}", leave=False, unit="batch")
        for batch_idx in batch_bar:
            x, y = get_batch(train_data, block_size, batch_size, device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                _, loss, _ = model(x, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            batch_bar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{total_loss/(batch_idx+1):.4f}")

        scheduler.step()
        avg_train_loss = total_loss / n_batches
        epoch_time = time.time() - epoch_start

        avg_val_loss = evaluate(model, val_data, block_size, batch_size, device, n_batches=20)

        epoch_bar.set_postfix(
            train=f"{avg_train_loss:.4f}",
            val=f"{avg_val_loss:.4f}",
            lr=f"{scheduler.get_last_lr()[0]:.6f}",
            t=f"{epoch_time:.1f}s",
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(PRETRAINED_DIR, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "epoch": epoch + 1,
                    "val_loss": avg_val_loss,
                },
                MODEL_PATH,
            )
            torch.save({"stoi": stoi, "itos": itos, "vocab_size": vocab_size}, META_PATH)
            tqdm.write(f"  ✓ Saved best model (val_loss={avg_val_loss:.4f})")

    # ── Final test evaluation ──────────────────────────────────────────
    print(f"\nTraining complete. Best val_loss={best_val_loss:.4f}")
    print("Loading best checkpoint for test evaluation...")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss = evaluate(model, test_data, block_size, batch_size, device, n_batches=50)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Model saved to {MODEL_PATH}")

    # ── Sample generation ──────────────────────────────────────────────
    model.eval()
    prompt = "Once upon a time"
    idx = torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long, device=device)
    generated, _ = model.generate(idx, max_new_tokens=300)
    text = "".join([itos[i] for i in generated[0].tolist()])
    print(f"\nSample generation:\n{'='*60}\n{text}\n{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train tiny MoE transformer on TinyStories")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()
    train(args)
