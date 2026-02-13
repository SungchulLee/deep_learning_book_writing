#!/usr/bin/env python3
# ========================================================
# 12_dataset_02_map_style_with_init.py
# ========================================================
"""
Map-style Dataset (with __init__)

Concept:
- Treat the dataset as a mapping f: {0, ..., N-1} -> sample.
- You MUST implement:
    __len__()       → returns N (dataset size)
    __getitem__(i)  → returns the i-th sample
- __init__ is optional, but typically used to pass data, paths, transforms, etc.
- Not an iterator: you don't call next(ds). Use indexing or a DataLoader.

Tips:
- Keep __getitem__ side-effect free and fast (DataLoader may call it in parallel).
- Return tensors (or tuples of tensors) so DataLoader can collate batches.
"""

import torch
from torch.utils.data import Dataset

# --------------------------------------------------------
# Minimal implementation of a Map-style Dataset
# --------------------------------------------------------
class MinimalMapDataset(Dataset):
    def __init__(self, n=8):
        # Store tensor data: [0, 1, 2, ..., n-1]
        # In real use, you'd load/prepare data here and keep references.
        self.data = torch.arange(n)

    def __len__(self):
        # Number of samples
        return self.data.numel()

    def __getitem__(self, idx):
        # Map index -> sample (0-D tensor here)
        # If you need (input, target), return a tuple instead.
        return self.data[idx]

# --------------------------------------------------------
# Quick sanity check
# --------------------------------------------------------
def main():
    ds = MinimalMapDataset(n=5)
    print(f"{len(ds) = }")  # Expect 5
    # Indexing like a list/ndarray; .item() converts 0-D tensor -> Python int
    for i in range(len(ds)):
        print(f"ds[{i}] = {ds[i].item()}")

if __name__ == "__main__":
    main()