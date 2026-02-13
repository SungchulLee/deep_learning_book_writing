#!/usr/bin/env python3
# ========================================================
# 12_dataset_04_map_style_in_ram_memory.py
# ========================================================
"""
Map-style Dataset (RAM): keep tensors in memory.
- Implements __len__ (size) and __getitem__ (random access by index).
- Useful when data already fits in memory as tensors (no disk I/O in __getitem__).
"""

from typing import Tuple
import torch
from torch.utils.data import Dataset
from collections.abc import Iterable, Iterator

class InMemoryDataset(Dataset):
    """Keep features/labels in RAM as tensors (no disk access during indexing)."""
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        assert len(X) == len(y)
        self.X, self.y = X, y  # references to already-loaded tensors

    def __len__(self) -> int:
        # Number of samples
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Return (feature_i, target_i) directly from RAM
        return self.X[idx], self.y[idx]

def check_indexed_collection(ds: Dataset, name: str, show_idx):
    print(f"\n[{name}] len={len(ds)}")
    # NOTE:
    # - Dataset doesn't define __iter__; isinstance(ds, Iterable) can be False,
    #   yet iter(ds) still works via the sequence protocol (__len__ + __getitem__).
    print("Iterable? ", isinstance(ds, Iterable))
    # - Not an Iterator: no __next__; next(ds) should fail.
    print("Iterator? ", isinstance(ds, Iterator))  # should be False
    for i in show_idx:
        x, y = ds[i]
        print(f"  ds[{i:2d}] -> x.shape={tuple(x.shape)}, y={y}")

def main():
    # X: 8 evenly spaced points in [-1, 1], shape [8, 1]
    X = torch.linspace(-1, 1, 8).unsqueeze(1)          # [8, 1]
    # y: integer targets from the linear rule 3x + 1
    y = (3 * X + 1.0).squeeze(1).round().long()        # [8]

    # Yes: X and y are already in RAM.
    # This dataset is a thin wrapper that exposes them via the Dataset API,
    # so __getitem__ just slices those in-memory tensors (no file reads).
    ds = InMemoryDataset(X, y)

    check_indexed_collection(ds, "InMemoryDataset (RAM)", [0, 5, 2, 7])

    # Show iterable vs iterator behavior
    it = iter(ds)                    # creates a sequence iterator over indices 0..len-1
    print("next(iter(ds)) ->", next(it))
    try:
        next(ds)                     # ds is not an iterator; this should raise
    except TypeError as e:
        print("next(ds) raises TypeError:", e)

if __name__ == "__main__":
    main()