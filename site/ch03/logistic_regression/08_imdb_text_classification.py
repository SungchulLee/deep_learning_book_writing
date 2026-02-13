
#!/usr/bin/env python3
"""
Minimal IMDb text classification project (from scratch, no torchtext).
- Downloads the Large Movie Review Dataset (aclImdb) to ./data by default
- Builds a simple vocabulary
- Trains an average-embedding classifier (PyTorch)
Usage (example):
    python main.py --epochs 2 --batch_size 64 --max_len 256
"""
from dataclasses import dataclass
from pathlib import Path
import argparse
import torch

from imdb.download_data import download_imdb
from imdb.load_data import make_dataloaders
from imdb.model import AverageEmbeddingsClassifier
from imdb.train import train_loop, evaluate_loop, save_checkpoint

@dataclass
class Config:
    data_dir: Path = Path("./data")
    epochs: int = 2
    batch_size: int = 64
    lr: float = 1e-3
    seed: int = 0
    embed_dim: int = 128
    max_vocab_size: int = 30000
    min_freq: int = 2
    max_len: int = 256
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: Path = Path("./save")

def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=Path, default=Path("./data"))
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--max_vocab_size", type=int, default=30000)
    p.add_argument("--min_freq", type=int, default=2)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save_dir", type=Path, default=Path("./save"))
    args = p.parse_args()
    return Config(**vars(args))

def main():
    cfg = parse_args()
    cfg.save_dir.mkdir(parents=True, exist_ok=True)

    # 1) Ensure the dataset lives under ./data (or cfg.data_dir) exactly as requested
    root = download_imdb(cfg.data_dir)

    # 2) Build dataloaders + vocab
    train_loader, val_loader, test_loader, vocab, pad_idx = make_dataloaders(
        root=root,
        batch_size=cfg.batch_size,
        max_vocab_size=cfg.max_vocab_size,
        min_freq=cfg.min_freq,
        max_len=cfg.max_len,
        num_workers=cfg.num_workers,
        seed=cfg.seed
    )

    print(f"Vocab size: {len(vocab)}  (pad_idx={pad_idx})")
    device = torch.device(cfg.device)

    # 3) Build model
    model = AverageEmbeddingsClassifier(vocab_size=len(vocab), embed_dim=cfg.embed_dim, pad_idx=pad_idx).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    # 4) Train
    best_val_acc = 0.0
    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = train_loop(model, train_loader, optim, criterion, device)
        val_loss, val_acc = evaluate_loop(model, val_loader, criterion, device)
        print(f"[Epoch {epoch:02d}] Train loss={train_loss:.4f} acc={train_acc:.3f} | "
              f"Val loss={val_loss:.4f} acc={val_acc:.3f}")
        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(cfg.save_dir / "imdb_avgemb_best.pt", model, vocab, pad_idx, cfg)

    # 5) Final test
    test_loss, test_acc = evaluate_loop(model, test_loader, criterion, device)
    print(f"[Test] loss={test_loss:.4f} acc={test_acc:.3f}")

    # Save last
    save_checkpoint(cfg.save_dir / "imdb_avgemb_last.pt", model, vocab, pad_idx, cfg)

if __name__ == "__main__":
    main()
