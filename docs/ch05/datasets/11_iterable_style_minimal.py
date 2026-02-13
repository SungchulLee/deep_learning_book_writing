#!/usr/bin/env python3
# ========================================================
# 12_dataset_09_iterable_style_minimal.py
# ========================================================
"""
Iterable-style Dataset:
- Implements __iter__ to stream samples; no random indexing by design.
- Length may be unknown; __len__ is optional.
- DataLoader will NOT use a sampler for IterableDataset; it just consumes your iterator.
- If you need shuffling/epoch control/worker partitioning, implement it inside __iter__.

Re-iteration:
- In didactic examples you might re-seed inside __iter__ so each loop reproduces
  the SAME sequence.
- In practice, vary the seed per epoch/worker (see torch.utils.data.get_worker_info).

FYI — Two dataset flavors in PyTorch:
- Map-style (torch.utils.data.Dataset): __len__ + __getitem__(i), works with
  Sampler (shuffle=True), random_split, Subset.
- Iterable-style (torch.utils.data.IterableDataset): __iter__ (stream);
  DataLoader does not sample; you handle shuffling/partitioning yourself.
"""