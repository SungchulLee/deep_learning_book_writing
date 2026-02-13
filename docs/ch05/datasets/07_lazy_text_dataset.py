#!/usr/bin/env python3
# ========================================================
# 12_dataset_05_lazy_text_dataset.py
# ========================================================
"""
Map-style Dataset (DISK, lazy): store file paths; load on demand in __getitem__.

Why lazy?
- Small RAM footprint (keeps only paths in memory).
- Each __getitem__ does I/O; slower per sample, but parallelizable across **CPU**
  worker processes via DataLoader(num_workers>0). (DataLoader workers always run
  on CPU; for GPU training, load on CPU then move batches to CUDA in the loop.)

Notes:
- This Dataset is NOT an iterator (no __next__); it supports random access via indices.
- Even without __iter__, `iter(ds)` works via the **sequence protocol**:
  Python repeatedly calls ds[0], ds[1], ... until __getitem__ raises IndexError.
- Demo creates temp files and does not clean them up (fine for a demo; see tempfile.TemporaryDirectory for auto-cleanup).
"""

from typing import List
from collections.abc import Iterable, Iterator
import os, tempfile
import torch
from torch.utils.data import Dataset

def _make_temp_text_files(n: int = 6) -> tuple[str, List[str]]:
    """Create a temp folder with tiny text files for the demo; returns (dir, paths)."""
    tmpdir = tempfile.mkdtemp(prefix="indexed_collection_demo_")
    paths: List[str] = []
    for i in range(n):
        p = os.path.join(tmpdir, f"sample_{i}.txt")
        with open(p, "w") as f:
            f.write(f"value:{i}\n")  # very simple payload
        paths.append(p)
    return tmpdir, paths

class LazyTextDataset(Dataset):
    """Holds file paths; reads/parses file content in __getitem__ (on-demand)."""
    def __init__(self, paths: List[str]):
        self.paths = list(paths)  # just the paths; no file data loaded here

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Load and parse exactly when the sample is requested.
        path = self.paths[idx]
        with open(path, "r") as f:
            line = f.readline().strip()
        val = int(line.split(":")[1])  # parse trailing int from "value:X"
        return torch.tensor(val, dtype=torch.long)

def check_indexed_collection(ds: Dataset, name: str, show_idx):
    print(f"\n[{name}] len={len(ds)}")
    # Iterable? checks for __iter__; map-style Datasets often omit it.
    # Python can still iterate via the sequence protocol (__len__ + __getitem__).
    print("Iterable? ", isinstance(ds, Iterable))
    # Not an Iterator: no __next__ defined; next(ds) should raise.
    print("Iterator? ", isinstance(ds, Iterator))  # should be False
    for i in show_idx:
        print(f"  ds[{i:2d}] -> {ds[i]}")

def main():
    d, paths = _make_temp_text_files(n=6)
    print(f"(temp files in: {d})")
    ds = LazyTextDataset(paths)
    check_indexed_collection(ds, "LazyTextDataset (DISK, on-demand)", [5, 1, 3])

    # Show iterable vs iterator behavior
    it = iter(ds)  # sequence iterator over indices 0..len-1 (uses __len__/__getitem__)
    print("next(iter(ds)) ->", next(it))
    try:
        next(ds)    # ds itself is not an iterator
    except TypeError as e:
        print("next(ds) raises TypeError:", e)

if __name__ == "__main__":
    main()