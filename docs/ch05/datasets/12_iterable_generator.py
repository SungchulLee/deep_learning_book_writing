#!/usr/bin/env python3
# ========================================================
# 12_dataset_10_iterable_generator.py
# ========================================================
"""
IterableDataset via a simple generator.

Notes:
- We re-seed inside __iter__, so each pass yields the SAME sequence (demo).
- For training, vary the seed per epoch/worker to avoid repeats.
"""

import torch
from torch.utils.data import IterableDataset

class RandomStream_Generator(IterableDataset):
    def __init__(self, total=5, seed=0):
        self.total = total
        self.seed = seed

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed)  # reseeded each iteration
        for _ in range(self.total):
            yield torch.randn(3, generator=g)  # one sample (shape [3])

def main():
    ds = RandomStream_Generator(total=4, seed=42)
    print("pass 1:")
    for s in ds:
        print(s.tolist())
    print("pass 2 (same seq):")
    for s in ds:
        print(s.tolist())

if __name__ == "__main__":
    main()
