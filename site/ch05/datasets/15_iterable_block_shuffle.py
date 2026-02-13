#!/usr/bin/env python3
# ========================================================
# 12_dataset_13_iterable_block_shuffle.py
# ========================================================
"""
Block shuffle for IterableDataset (finite stream).

Idea:
- Read a block of K elements from the source stream, randomly permute that block,
  yield all K items, then repeat for the next block.
- Memory footprint is O(K) (much smaller than materializing the whole dataset).
- Great for locality/cache friendliness; less global mixing than a full shuffle.

Trade-offs:
- Inter-block order is preserved (only items *within* each block are permuted).
  Increase block_size for stronger mixing; block_size >= N approximates a full shuffle.
- Works best when the stream is finite or chunkable; for truly unbounded streams,
  consider a shuffle buffer (see 12_dataset_11_iterable_shuffle_buffer.py).

Multi-worker notes:
- DataLoader does NOT sample IterableDataset; you must shard the stream yourself.
- We shard by modulo on a conceptual global index: worker w of W gets items with i % W == w.
- RNG is seeded per worker so different workers produce different permutations.
- For different orders across epochs, vary the seed each epoch.

Complexity:
- Time: O(N), Memory: O(block_size).
"""

import torch
from torch.utils.data import IterableDataset, get_worker_info


class BlockShuffle(IterableDataset):
    def __init__(self, src: IterableDataset, block_size: int = 1024, seed: int = 0):
        self.src = src
        self.block_size = max(1, int(block_size))
        self.seed = seed

    def __iter__(self):
        info = get_worker_info()
        wid = info.id if info else 0
        rng = torch.Generator().manual_seed(self.seed + 2024 * wid)

        # Shard the source across workers so each sees a disjoint subset.
        def _shard(it):
            if info is None:
                for i, x in enumerate(it):
                    yield x
            else:
                for i, x in enumerate(it):
                    if i % info.num_workers == wid:
                        yield x

        src_iter = iter(_shard(self.src))
        block = []
        while True:
            # Fill a block (handles the final partial block as well)
            block.clear()
            try:
                for _ in range(self.block_size):
                    block.append(next(src_iter))
            except StopIteration:
                # Source exhausted; proceed with whatever was collected
                pass

            if not block:
                break  # nothing left to yield

            if len(block) > 1:
                # Permute items *within* the block
                perm = torch.randperm(len(block), generator=rng).tolist()
                for j in perm:
                    yield block[j]
            else:
                # Single leftover element
                yield block[0]


# --- demo ---------------------------------------------------------------------
class Stream(torch.utils.data.IterableDataset):
    """Finite toy stream emitting n noisy scalars for demonstration."""
    def __init__(self, n=12, seed=0):
        self.n, self.seed = n, seed
    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed)
        for _ in range(self.n):
            yield torch.randn(1, generator=g).item()

def main():
    src = Stream(n=12, seed=0)
    shuf = BlockShuffle(src, block_size=5, seed=77)
    print("block-shuffled:")
    for x in shuf:
        print(x)

if __name__ == "__main__":
    main()

