#!/usr/bin/env python3
# ========================================================
# 12_dataset_11_iterable_iterator_obj.py
# ========================================================
"""
Iterable-style Dataset that returns an explicit iterator object.

Key points:
- IterableDataset supplies __iter__ (a stream), not random index access.
- Here __iter__ returns a NEW iterator object each time (state lives in that object).
- We re-seed the RNG inside the iterator → every fresh pass yields the SAME sequence
  (good for a deterministic demo). In real training, vary the seed per epoch/worker.

DataLoader behavior:
- For IterableDataset, DataLoader does NOT use a sampler; it consumes your iterator(s).
- If you need sharding across workers, handle it inside __iter__ using
  torch.utils.data.get_worker_info() (not shown here).
"""

import torch
from torch.utils.data import IterableDataset

class _RandomStreamIterator:
    """Stateful iterator with its own RNG and cursor."""
    def __init__(self, total, seed):
        self.total = total      # how many samples to emit
        self.i = 0              # position (advanced by __next__)
        self.g = torch.Generator().manual_seed(seed)  # deterministic stream

    def __iter__(self):
        return self             # iter(iterator) must return itself

    def __next__(self):
        if self.i >= self.total:
            raise StopIteration
        self.i += 1
        # Each call produces one sample (shape [3]) from this iterator's RNG
        return torch.randn(3, generator=self.g)


class RandomStream_Iterable(IterableDataset):
    """IterableDataset whose __iter__ returns a fresh _RandomStreamIterator."""
    def __init__(self, total=5, seed=0):
        self.total = total
        self.seed = seed

    def __iter__(self):
        # New iterator each time → iteration state doesn't leak across passes.
        # Since we pass the same seed, each pass reproduces the same sequence.
        return _RandomStreamIterator(self.total, self.seed)


def main():
    ds = RandomStream_Iterable(total=4, seed=42)

    print("pass 1:")
    for s in ds:
        print(s.tolist())

    print("pass 2 (same seq due to identical seeding):")
    for s in ds:
        print(s.tolist())

    # NOTE:
    # - To get different sequences per epoch/worker, add an epoch parameter or use
    #   get_worker_info() in __iter__ and tweak the seed accordingly.

if __name__ == "__main__":
    main()

