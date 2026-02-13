#!/usr/bin/env python3
# ========================================================
# 12_dataset_01_map_style_minimal.py
# ========================================================
"""
Map-style Dataset (minimal example)

Core idea:
- Treat the dataset like a mapping f: {0, ..., N-1} -> sample.
- You MUST implement:
    __len__()       → return number of samples N
    __getitem__(i)  → return the i-th sample (random-access)

Notes:
- __getitem__ should be pure (no side effects) and fast.
- Return tensors (or tuples of tensors) so DataLoader can collate them.
"""

import torch
from torch.utils.data import Dataset

# --------------------------------------------------------
# Example Dataset class without an explicit __init__
# --------------------------------------------------------
class NoInitDataset(Dataset):
    # Class attribute shared by all instances.
    # (If you mutate it via one instance, it affects all.)
    # In real projects you'd typically do: self.data = ... in __init__.
    data = torch.tensor([10, 20, 30, 40])

    def __len__(self):
        # Dataset size (number of valid indices)
        return self.data.numel()

    def __getitem__(self, idx):
        # Map index -> sample (here: a 0-D tensor)
        # DataLoader will call this repeatedly for different indices.
        return self.data[idx]

# --------------------------------------------------------
# Quick smoke test
# --------------------------------------------------------
def main():
    ds = NoInitDataset()             # No __init__ args needed here
    print("len =", len(ds))          # Expect: 4
    # Materialize all items; .item() converts 0-D tensors to Python ints
    print([ds[i].item() for i in range(len(ds))])

if __name__ == "__main__":
    main()