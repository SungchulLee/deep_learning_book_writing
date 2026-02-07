# Collate Functions

## Overview

The collate function merges a list of individual samples into a batched tensor. PyTorch's `default_collate` handles the common case, but custom collate functions are needed for variable-length data, nested structures, or domain-specific batching logic.

## Default Collate Behavior

`default_collate` recursively stacks tensors, concatenates numpy arrays, and preserves Python scalars:

```python
from torch.utils.data.dataloader import default_collate

samples = [(torch.randn(3, 32, 32), 0),
           (torch.randn(3, 32, 32), 1),
           (torch.randn(3, 32, 32), 0)]

batch = default_collate(samples)
# batch[0].shape = torch.Size([3, 3, 32, 32])  # stacked images
# batch[1] = tensor([0, 1, 0])                  # stacked labels
```

This works when all samples have identical shapes. When they don't, `default_collate` raises an error.

## Custom Collate for Variable-Length Sequences

```python
def pad_collate(batch):
    """Pad variable-length sequences to the longest in the batch."""
    sequences, labels = zip(*batch)
    lengths = [len(seq) for seq in sequences]
    max_len = max(lengths)

    padded = torch.zeros(len(sequences), max_len, sequences[0].size(-1))
    for i, seq in enumerate(sequences):
        padded[i, :len(seq)] = seq

    lengths = torch.tensor(lengths)
    labels = torch.tensor(labels)
    return padded, lengths, labels

loader = DataLoader(dataset, batch_size=32, collate_fn=pad_collate)
```

The returned `lengths` tensor enables packed sequence processing with `nn.utils.rnn.pack_padded_sequence`.

## Custom Collate for Dictionary Samples

When datasets return dictionaries, the collate function must handle each key:

```python
def dict_collate(batch):
    """Collate a list of dictionaries into a dictionary of batched tensors."""
    return {
        key: torch.stack([sample[key] for sample in batch])
        if isinstance(batch[0][key], torch.Tensor)
        else [sample[key] for sample in batch]
        for key in batch[0]
    }
```

## Filtering Invalid Samples

A collate function can filter out `None` values from samples that failed to load:

```python
def filter_collate(batch):
    """Remove None samples and collate the rest."""
    batch = [sample for sample in batch if sample is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)
```

## Financial Time Series Collate

For time series with different numbers of features or missing data:

```python
def timeseries_collate(batch):
    """Collate time series with masking for missing values."""
    features, targets = zip(*batch)
    max_len = max(f.size(0) for f in features)
    n_features = features[0].size(1)

    padded = torch.zeros(len(features), max_len, n_features)
    mask = torch.zeros(len(features), max_len, dtype=torch.bool)

    for i, f in enumerate(features):
        padded[i, :f.size(0)] = f
        mask[i, :f.size(0)] = True

    targets = torch.stack(targets)
    return padded, mask, targets
```

## Key Takeaways

- `default_collate` stacks same-shape samples automatically.
- Custom collate functions handle variable-length data via padding, dictionary samples, and invalid sample filtering.
- Always return length or mask tensors alongside padded data so downstream modules can ignore padding.
