#!/usr/bin/env python3
# ========================================================
# 12_dataset_14_iterable_dataloader_workers_demo.py
# ========================================================
"""
Demonstrate worker-aware sharding + lightweight shuffling with DataLoader(num_workers>0).

Concepts shown:
- IterableDataset: DataLoader does NOT sample; it just consumes __iter__ from each worker.
- Sharding (mod-split): worker w of W processes items whose global index i satisfies i % W == w.
- Per-worker RNG: base_seed + c * worker_id → independent random streams per worker.
- In-worker shuffle: tiny buffer (size=4) is randomly permuted before yielding (local shuffle).

Notes:
- Increase the buffer size for stronger local mixing, or use a shuffle buffer/block shuffle
  (see other examples) if you need better global mixing on streams.
- For epoch-to-epoch randomness, vary base_seed per epoch (or add an epoch argument).
- batch_size=None ⇒ DataLoader yields one sample at a time (no collation).
"""

import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info


class WorkerAwareStream(IterableDataset):
    def __init__(self, n=20, base_seed=0):
        self.n, self.base_seed = n, base_seed

    def __iter__(self):
        info = get_worker_info()
        if info is None:
            wid, nw = 0, 1                     # single-process path
        else:
            wid, nw = info.id, info.num_workers

        # Per-worker RNG (same base, different streams)
        g = torch.Generator().manual_seed(self.base_seed + 1000 * wid)

        # Shard by modulo; then do a small in-worker shuffle with a short buffer.
        buf = []
        for i in range(self.n):
            if i % nw == wid:                  # disjoint shard for this worker
                # Emit a noisy float tied to the index (demo payload)
                buf.append(i + torch.randn(1, generator=g).item())

            # Local shuffle when buffer accumulates enough items
            if len(buf) >= 4:
                perm = torch.randperm(len(buf), generator=g).tolist()
                for j in perm:
                    yield buf[j]
                buf.clear()

        # Flush remainder (last partial buffer)
        if buf:
            perm = torch.randperm(len(buf), generator=g).tolist()
            for j in perm:
                yield buf[j]


def main():
    ds = WorkerAwareStream(n=20, base_seed=7)

    print("Single-process iteration:")
    for x in ds:
        print(x)

    print("\nDataLoader with 2 workers:")
    loader = DataLoader(ds, batch_size=None, num_workers=2)
    for x in loader:
        print(x)


if __name__ == "__main__":
    main()
