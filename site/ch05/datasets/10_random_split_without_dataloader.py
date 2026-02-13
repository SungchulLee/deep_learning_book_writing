#!/usr/bin/env python3
# ========================================================
# 12_dataset_08_random_split_without_dataloader.py
# ========================================================
"""
random_split / Subset on any map-style Dataset.

- random_split(ds, lengths, generator):
    Splits a Dataset into non-overlapping Subsets whose lengths sum to len(ds).
    Uses the given RNG for reproducibility.

- Subset(ds, indices):
    Lightweight wrapper that references items in `ds` at `indices`
    (no data copy; original indexing is preserved).
"""

import torch
from torch.utils.data import Dataset, random_split, Subset

class SeqDs(Dataset):
    def __init__(self, n=10):
        self.data = torch.arange(n)            # [0, 1, ..., n-1]
    def __len__(self):
        return len(self.data)                  # dataset size
    def __getitem__(self, i):
        return self.data[i]                    # index -> sample

def main():
    ds = SeqDs(10)
    g = torch.Generator().manual_seed(7)       # deterministic split
    train, val = random_split(ds, [7, 3], generator=g)

    # .indices are positions in the ORIGINAL dataset `ds`
    print("train indices:", train.indices)
    print("val indices  :", val.indices)

    # Subset again (no copy): take first 3 elements from the train subset
    sub = Subset(train, indices=train.indices[:3])
    print("subset len =", len(sub))
    print([sub[i].item() for i in range(len(sub))])

if __name__ == "__main__":
    main()

