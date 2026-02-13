#!/usr/bin/env python3
# ========================================================
# 12_dataset_06_memmap_dataset.py
# ========================================================
"""
Map-style Dataset (memmap): wrap a NumPy memmap to avoid loading the whole file into RAM.

Why memmap?
- OS-backed, on-demand paging (good for arrays larger than RAM).
- Random access by index; minimal memory footprint per sample.
- Works well with DataLoader(num_workers>0) to parallelize disk I/O on CPU.

Notes:
- from_numpy() shares memory with the underlying np.ndarray (no copy). Here we open
  the memmap in read-only mode ("r"), so tensors are effectively read-only views.
- If dtype is already float32, .float() is a no-op (no extra copy).
- With multi-process DataLoader on some platforms (e.g., Windows spawn), it's safer to
  store the file path and lazily open memmap in each worker on first __getitem__ call.
"""

from typing import Tuple
from collections.abc import Iterable, Iterator
import os, tempfile
import numpy as np
import torch
from torch.utils.data import Dataset

def _create_memmap_file(shape=(10, 3), dtype="float32") -> tuple[str, Tuple[int, ...], str]:
    """Create a temp .dat backed by NumPy memmap and fill with a simple pattern."""
    tmpdir = tempfile.mkdtemp(prefix="indexed_collection_mmap_")
    path = os.path.join(tmpdir, "array.dat")
    mm = np.memmap(path, mode="w+", dtype=dtype, shape=shape)
    for i in range(shape[0]):
        mm[i] = np.arange(shape[1], dtype=dtype) + i  # row i = [i, i+1, i+2, ...]
    mm.flush()  # ensure data is on disk
    return path, shape, dtype

class MemmapDataset(Dataset):
    """Wrap a NumPy memmap file (no full load into RAM)."""
    def __init__(self, path: str, shape: Tuple[int, ...], dtype: str = "float32"):
        # Read-only mapping; change to mode="r+" if you need writable views.
        self.mm = np.memmap(path, mode="r", dtype=dtype, shape=shape)

    def __len__(self) -> int:
        return self.mm.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        # memmap slicing yields an ndarray view (no copy).
        # torch.from_numpy shares memory with that ndarray (zero-copy).
        arr = np.asarray(self.mm[idx])      # view into the mmap'd region
        return torch.from_numpy(arr).float()

def check_indexed_collection(ds: Dataset, name: str, show_idx):
    print(f"\n[{name}] len={len(ds)}")
    print("Iterable? ", isinstance(ds, Iterable))
    print("Iterator? ", isinstance(ds, Iterator))  # should be False
    for i in show_idx:
        sample = ds[i]
        print(f"  ds[{i:2d}] -> tensor.shape={tuple(sample.shape)}, values={sample.tolist()}")

def main():
    path, shape, dtype = _create_memmap_file(shape=(10, 3))
    ds = MemmapDataset(path, shape, dtype)
    check_indexed_collection(ds, "MemmapDataset (memory-mapped)", [0, 4, 9])

    # Iteration works via the sequence protocol (__len__ + __getitem__)
    it = iter(ds)
    print("next(iter(ds)) ->", next(it))
    try:
        next(ds)  # dataset itself is not an iterator
    except TypeError as e:
        print("next(ds) raises TypeError:", e)

if __name__ == "__main__":
    main()