"""
Train BPE tokenizer on TinyStories, then train MoE model with subword tokens.

Usage:
    python train_bpe.py                    # train tokenizer + model from scratch
    python train_bpe.py --resume           # resume model training
    python train_bpe.py --tokenizer-only   # only train the BPE tokenizer
    python train_bpe.py --epochs 30        # custom epoch count
"""

import os
import time
import argparse

import torch
from tqdm import tqdm
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Metaspace as MetaspacePreTokenizer
from tokenizers.decoders import Metaspace as MetaspaceDecoder

from moe_model import MoEConfig, TinyMoETransformer

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DATA_PATH = os.path.join(DATA_DIR, "tinystories.txt")
PRETRAINED_DIR = os.path.join(os.path.dirname(__file__), "pretrained")
TOKENIZER_PATH = os.path.join(PRETRAINED_DIR, "bpe_tokenizer.json")
MODEL_PATH = os.path.join(PRETRAINED_DIR, "model_bpe.pt")
META_PATH = os.path.join(PRETRAINED_DIR, "meta_bpe.pt")

BPE_VOCAB_SIZE = 512


def train_tokenizer():
    """Train a BPE tokenizer on the TinyStories data."""
    if os.path.exists(TOKENIZER_PATH):
        print(f"BPE tokenizer already exists at {TOKENIZER_PATH}")
        return Tokenizer.from_file(TOKENIZER_PATH)

    print(f"Training BPE tokenizer with vocab_size={BPE_VOCAB_SIZE}...")
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = MetaspacePreTokenizer()
    tokenizer.decoder = MetaspaceDecoder()

    trainer = BpeTrainer(
        vocab_size=BPE_VOCAB_SIZE,
        special_tokens=["[UNK]", "[PAD]"],
        min_frequency=2,
    )
    tokenizer.train([DATA_PATH], trainer)

    os.makedirs(PRETRAINED_DIR, exist_ok=True)
    tokenizer.save(TOKENIZER_PATH)
    print(f"Saved BPE tokenizer ({tokenizer.get_vocab_size()} tokens) to {TOKENIZER_PATH}")
    return tokenizer


def load_data(tokenizer):
    """Load and tokenize data using BPE."""
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    print("Encoding text with BPE tokenizer...")
    encoded = tokenizer.encode(text)
    data = torch.tensor(encoded.ids, dtype=torch.long)
    print(f"Encoded {len(text):,} chars → {len(data):,} tokens (compression: {len(text)/len(data):.1f}x)")
    return data


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix]).to(device)
    return x, y


@torch.no_grad()
def evaluate(model, data, block_size, batch_size, device, n_batches=50):
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = get_batch(data, block_size, batch_size, device)
        _, loss, _ = model(x, y)
        losses.append(loss.item())
    return sum(losses) / len(losses)


def train(args):
    # Step 1: Train or load tokenizer
    tokenizer = train_tokenizer()
    if args.tokenizer_only:
        return

    vocab_size = tokenizer.get_vocab_size()
    print(f"BPE vocab size: {vocab_size}")

    # Step 2: Load and encode data
    data = load_data(tokenizer)

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

    # Same architecture, just different vocab_size
    config = MoEConfig(vocab_size=vocab_size)
    config.tokenizer_type = "bpe"

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

    amp_dtype = torch.float16 if device == "cuda" else torch.bfloat16
    use_amp = device in ("cuda", "mps")
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))

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
            torch.save({"vocab_size": vocab_size, "tokenizer_type": "bpe"}, META_PATH)
            tqdm.write(f"  ✓ Saved best model (val_loss={avg_val_loss:.4f})")

    # Final test evaluation
    print(f"\nTraining complete. Best val_loss={best_val_loss:.4f}")
    print("Loading best checkpoint for test evaluation...")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss = evaluate(model, test_data, block_size, batch_size, device, n_batches=50)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Model saved to {MODEL_PATH}")

    # Sample generation
    model.eval()
    prompt = "Once upon a time"
    ids = tokenizer.encode(prompt).ids
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    generated, _ = model.generate(idx, max_new_tokens=100)
    tokens = generated[0].tolist()
    text = tokenizer.decode(tokens)
    print(f"\nSample generation:\n{'='*60}\n{text}\n{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BPE MoE transformer on TinyStories")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--tokenizer-only", action="store_true", help="Only train the tokenizer")
    args = parser.parse_args()
    train(args)
