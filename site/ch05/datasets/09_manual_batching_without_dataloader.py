#!/usr/bin/env python3
# ========================================================
# 12_dataset_07_manual_batching_without_dataloader.py
# ========================================================
"""
Manual batching (no DataLoader):
- Shows how a map-style Dataset is indexed to build mini-batches.
- We shuffle indices, slice them per batch, fetch samples, and stack.
- The last batch may be smaller than batch_size (no drop-last here).

Tip:
- DataLoader automates this (shuffling, batching, workers, pin_memory, etc.).
- This script is just the minimal mechanics.
"""

import torch
from torch.utils.data import Dataset

class TinyXY(Dataset):
    def __init__(self, n=10, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.X = torch.randn(n, 2, generator=g)              # features [N, 2]
        self.y = (self.X.sum(dim=1) > 0).long()              # labels   [N]

    def __len__(self):  # dataset size
        return self.X.shape[0]

    def __getitem__(self, i):  # index -> (x_i, y_i)
        return self.X[i], self.y[i]

def manual_batches(ds: Dataset, batch_size=4, shuffle=True, seed=123):
    n = len(ds)
    idx = torch.arange(n)                                   # indices 0..N-1
    if shuffle:
        g = torch.Generator().manual_seed(seed)             # deterministic shuffle
        idx = idx[torch.randperm(n, generator=g)]
    for start in range(0, n, batch_size):
        ids = idx[start:start+batch_size].tolist()          # indices for this batch
        batch = [ds[i] for i in ids]                        # [(x0,y0), (x1,y1), ...]
        # zip(*batch) -> (x0,x1,...), (y0,y1,...) ; stack into tensors:
        Xb, yb = map(torch.stack, zip(*batch))              # Xb:[B,2], yb:[B]
        yield Xb, yb

def main():
    ds = TinyXY(n=9)
    for Xb, yb in manual_batches(ds, batch_size=4, shuffle=False):
        print("Xb", tuple(Xb.shape), "yb", tuple(yb.shape))

if __name__ == "__main__":
    main()
