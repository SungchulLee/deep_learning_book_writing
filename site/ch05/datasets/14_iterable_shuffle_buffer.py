#!/usr/bin/env python3
# ========================================================
# 12_dataset_12_iterable_shuffle_buffer.py
# ========================================================
"""
Streaming shuffle for IterableDataset using a fixed-size buffer.

What this does (high level):
- Maintains a buffer of size K (buffer_size).
- Repeatedly:
    1) Randomly pick an element from the buffer and yield it.
    2) Refill the buffer with the next source item (if any).
- This approximates a shuffle *online* without loading the full dataset.

Why/when to use:
- Works for unknown or very large streams (can't hold all items in RAM).
- Larger K → better mixing (but more RAM). K=1 means "no shuffle".

Caveats:
- The very beginning and very end of the stream are less well-mixed
  (buffer warm-up/drain). Increase K to reduce this effect.
- This is not a perfect global permutation; it's an *online* approximation.
- For multi-worker DataLoader, each worker must get a disjoint shard and an
  independent RNG seed (handled below).

Reproducibility / multi-worker:
- We derive the RNG seed from (base seed + worker_id * constant).
- Optional sharding via modulo ensures workers see disjoint items:
  worker w out of W sees items whose global index i satisfies i % W == w.

Tip:
- To vary the order across epochs, add an `epoch` parameter and modify the seed.
"""

import torch
from torch.utils.data import IterableDataset, get_worker_info


class ShuffleBuffer(IterableDataset):
    def __init__(self, src: IterableDataset, buffer_size: int = 1024, seed: int = 0):
        self.src = src
        self.buffer_size = max(1, int(buffer_size))
        self.seed = seed

    def __iter__(self):
        info = get_worker_info()
        wid = info.id if info else 0
        # Worker-aware RNG so each worker produces a different shuffle
        rng = torch.Generator().manual_seed(self.seed + 1337 * wid)

        # ---- Worker sharding (disjoint items per worker) ---------------------
        # For item i in the (conceptual) global order, only yield it to worker
        # wid if i % num_workers == wid. Single-process path returns all items.
        def _shard(it):
            if info is None:
                for i, x in enumerate(it):
                    yield x
            else:
                for i, x in enumerate(it):
                    if i % info.num_workers == wid:
                        yield x

        src_iter = iter(_shard(self.src))

        # ---- Fill the buffer (warm-up) ---------------------------------------
        buf = []
        try:
            for _ in range(self.buffer_size):
                buf.append(next(src_iter))
        except StopIteration:
            # Source shorter than buffer_size: proceed with whatever we have
            pass

        # ---- Streaming shuffle loop -----------------------------------------
        while buf:
            # Pick a random index in [0, len(buf)-1]
            j = int(torch.randint(len(buf), (1,), generator=rng))
            # Yield the chosen element and remove it from the buffer
            yield buf.pop(j)

            # Refill (if the source still has items); otherwise we just keep draining
            try:
                buf.append(next(src_iter))
            except StopIteration:
                pass


# --- demo ---------------------------------------------------------------------
class Stream(torch.utils.data.IterableDataset):
    """Finite toy stream emitting n noisy scalars for demonstration."""
    def __init__(self, n=10, seed=0):
        self.n, self.seed = n, seed
    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed)
        for _ in range(self.n):
            yield torch.randn(1, generator=g).item()


def main():
    src = Stream(n=10, seed=0)

    print("Shuffled (buffer_size=4, seed=123):")
    shuf = ShuffleBuffer(src, buffer_size=4, seed=123)
    for x in shuf:
        print(x)

    print("\nShuffled again (same seed → same order):")
    for x in ShuffleBuffer(Stream(10, 0), buffer_size=4, seed=123):
        print(x)


if __name__ == "__main__":
    main()

