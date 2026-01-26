# Custom Collate Functions

## Learning Objectives

By the end of this section, you will be able to:

- Understand when and why default collation fails
- Implement padding strategies for variable-length sequences
- Create attention masks for Transformer models
- Handle multi-modal data with custom collate functions
- Apply batch-level transformations

## What Is collate_fn?

The `collate_fn` defines how individual samples are combined into a batch. The DataLoader calls this function with a list of samples fetched from the dataset.

### Default Behavior

The default collate function (`default_collate`) performs:

1. **Stacks tensors**: Samples with identical shapes are stacked along a new batch dimension
2. **Recursively handles collections**: Tuples, lists, and dicts are processed element-wise
3. **Preserves non-tensor types**: Strings, numbers, etc. are collected into lists

```python
# Default collation for uniform tensors
samples = [
    (torch.tensor([1, 2, 3]), torch.tensor(0)),
    (torch.tensor([4, 5, 6]), torch.tensor(1)),
]

# Default collate stacks:
# features → torch.tensor([[1, 2, 3], [4, 5, 6]])  # [2, 3]
# labels → torch.tensor([0, 1])                     # [2]
```

### When Default Collate Fails

Default collation fails when samples have **different shapes**:

```python
# Variable-length sequences cause errors
samples = [
    (torch.tensor([1, 2, 3]), torch.tensor(0)),      # length 3
    (torch.tensor([4, 5, 6, 7, 8]), torch.tensor(1)), # length 5
]

# default_collate will raise:
# RuntimeError: stack expects each tensor to be equal size
```

## Variable-Length Sequences

Text, audio, and time series data often have varying lengths. Custom collate functions handle this through padding.

### Simple Padding Collate

```python
from typing import List, Tuple
import torch

def simple_pad_collate(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    """
    Pad all sequences to the maximum length in the batch.
    
    Args:
        batch: List of (sequence, label) tuples
        
    Returns:
        padded_seqs: [batch_size, max_len] padded sequences
        lengths: [batch_size] original lengths
        labels: [batch_size] labels
    """
    sequences, labels = zip(*batch)
    
    # Record original lengths
    lengths = torch.tensor([len(seq) for seq in sequences])
    
    # Find max length in this batch
    max_len = lengths.max().item()
    
    # Create padded tensor (filled with zeros)
    batch_size = len(sequences)
    padded_seqs = torch.zeros(batch_size, max_len, dtype=sequences[0].dtype)
    
    # Copy each sequence into padded tensor
    for i, seq in enumerate(sequences):
        padded_seqs[i, :len(seq)] = seq
    
    # Stack labels
    labels = torch.stack(labels)
    
    return padded_seqs, lengths, labels


# Usage
loader = DataLoader(
    dataset,
    batch_size=32,
    collate_fn=simple_pad_collate
)

for padded_seqs, lengths, labels in loader:
    print(f"Padded shape: {padded_seqs.shape}")
    print(f"Lengths: {lengths}")
```

### Using pad_sequence

PyTorch provides `pad_sequence` for efficient padding:

```python
from torch.nn.utils.rnn import pad_sequence

def pytorch_pad_collate(batch):
    """
    Use PyTorch's optimized pad_sequence.
    """
    sequences, labels = zip(*batch)
    
    # pad_sequence expects list of tensors
    # batch_first=True → [batch, seq_len]
    # batch_first=False → [seq_len, batch]
    padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    
    lengths = torch.tensor([len(seq) for seq in sequences])
    labels = torch.stack(labels)
    
    return padded, lengths, labels
```

**pad_sequence parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_first` | False | Output shape: True→[B,L], False→[L,B] |
| `padding_value` | 0.0 | Value used for padding |

### Attention Masks for Transformers

Transformer models require attention masks to ignore padding tokens:

```python
def masked_pad_collate(batch):
    """
    Create padded sequences with attention masks.
    
    Returns:
        padded: [batch_size, max_len] padded sequences
        attention_mask: [batch_size, max_len] boolean mask
        labels: [batch_size] labels
    """
    sequences, labels = zip(*batch)
    
    # Pad sequences
    padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    
    # Create attention mask
    # True where we have real tokens, False for padding
    batch_size, max_len = padded.shape
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    
    for i, seq in enumerate(sequences):
        attention_mask[i, :len(seq)] = True
    
    labels = torch.stack(labels)
    
    return padded, attention_mask, labels


# Usage with Transformers
for input_ids, attention_mask, labels in loader:
    # Pass to BERT/GPT
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
    )
```

**Attention mask interpretation:**

```
Sequence: [CLS] Hello world [PAD] [PAD]
Mask:        1     1     1     0     0

Where 1 = attend, 0 = ignore
```

### Pack Padded Sequence for RNNs

For RNNs, use `pack_padded_sequence` for efficiency:

```python
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

def rnn_collate(batch):
    """Collate for efficient RNN processing."""
    sequences, labels = zip(*batch)
    
    # Sort by length (descending) - required by pack_padded_sequence
    lengths = torch.tensor([len(seq) for seq in sequences])
    sorted_indices = lengths.argsort(descending=True)
    
    sorted_seqs = [sequences[i] for i in sorted_indices]
    sorted_labels = torch.stack([labels[i] for i in sorted_indices])
    sorted_lengths = lengths[sorted_indices]
    
    padded = pad_sequence(sorted_seqs, batch_first=True)
    
    return padded, sorted_lengths, sorted_labels


# Usage in model forward pass
class RNNModel(nn.Module):
    def forward(self, x, lengths):
        # Pack for efficient processing
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=True
        )
        
        # RNN processes variable lengths efficiently
        packed_output, (hidden, cell) = self.lstm(packed)
        
        # Unpack if you need full output
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        return hidden[-1]  # Last hidden state
```

## Multi-Modal Data

Complex datasets often contain multiple data types that need different handling:

```python
def multimodal_collate(batch):
    """
    Handle datasets with images, text, and metadata.
    
    Each sample is a dict:
    {
        'image': [C, H, W] tensor,
        'text': [seq_len] tensor,
        'metadata': dict
    }
    """
    images = []
    texts = []
    metadatas = []
    
    for sample in batch:
        images.append(sample['image'])
        texts.append(sample['text'])
        metadatas.append(sample['metadata'])
    
    # Images: same size, just stack
    images_batch = torch.stack(images)
    
    # Text: variable length, pad
    texts_batch = pad_sequence(texts, batch_first=True, padding_value=0)
    text_lengths = torch.tensor([len(t) for t in texts])
    
    # Metadata: keep as list of dicts (or convert to batch dict)
    
    return {
        'images': images_batch,
        'texts': texts_batch,
        'text_lengths': text_lengths,
        'metadata': metadatas
    }


# Usage
for batch in loader:
    images = batch['images']        # [B, C, H, W]
    texts = batch['texts']          # [B, max_len]
    lengths = batch['text_lengths'] # [B]
    meta = batch['metadata']        # List of dicts
```

### Converting Metadata to Batch Dict

For structured metadata, you might want a batch dictionary:

```python
def collate_metadata(metadatas: List[dict]) -> dict:
    """
    Convert list of dicts to dict of batched values.
    
    Input: [{'id': 1, 'score': 0.5}, {'id': 2, 'score': 0.8}]
    Output: {'id': tensor([1, 2]), 'score': tensor([0.5, 0.8])}
    """
    keys = metadatas[0].keys()
    
    batch_dict = {}
    for key in keys:
        values = [m[key] for m in metadatas]
        
        if isinstance(values[0], (int, float)):
            batch_dict[key] = torch.tensor(values)
        else:
            batch_dict[key] = values  # Keep as list
    
    return batch_dict
```

## Batch-Level Transformations

Collate functions can apply transformations that require batch context:

### Mixup Augmentation

```python
def mixup_collate(batch, alpha=0.2):
    """
    Apply Mixup augmentation at batch level.
    
    Mixup creates virtual training examples by linear interpolation:
    x̃ = λx_i + (1-λ)x_j
    ỹ = λy_i + (1-λ)y_j
    """
    images, labels = zip(*batch)
    
    images = torch.stack(images)  # [B, C, H, W]
    labels = torch.tensor(labels)  # [B]
    
    # Sample mixing coefficient
    lam = np.random.beta(alpha, alpha)
    
    # Random shuffle for pairing
    batch_size = images.size(0)
    index = torch.randperm(batch_size)
    
    # Mix images
    mixed_images = lam * images + (1 - lam) * images[index]
    
    # Return both label sets (for computing mixed loss)
    return mixed_images, labels, labels[index], lam


# Usage in training
for images, labels_a, labels_b, lam in loader:
    outputs = model(images)
    loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
```

### Dynamic Batching by Tokens

For NLP, you might want to batch by total tokens rather than sample count:

```python
class TokenBatchCollate:
    """
    Create batches based on total token count, not sample count.
    
    This ensures consistent memory usage regardless of sequence lengths.
    """
    
    def __init__(self, max_tokens=4096, pad_token_id=0):
        self.max_tokens = max_tokens
        self.pad_token_id = pad_token_id
    
    def __call__(self, batch):
        sequences, labels = zip(*batch)
        
        # Sort by length for efficient packing
        sorted_pairs = sorted(
            zip(sequences, labels),
            key=lambda x: len(x[0]),
            reverse=True
        )
        sequences, labels = zip(*sorted_pairs)
        
        padded = pad_sequence(
            sequences,
            batch_first=True,
            padding_value=self.pad_token_id
        )
        
        lengths = torch.tensor([len(s) for s in sequences])
        labels = torch.stack(labels)
        
        return padded, lengths, labels


# Note: This collate receives a fixed-size batch
# For true dynamic batching, use a custom BatchSampler
```

## Handling Special Cases

### None Values

Sometimes samples should be skipped:

```python
def skip_none_collate(batch):
    """
    Filter out None samples (e.g., corrupted data).
    """
    # Remove None samples
    batch = [sample for sample in batch if sample is not None]
    
    if len(batch) == 0:
        return None
    
    # Standard collation on filtered batch
    return default_collate(batch)


# In training loop
for batch in loader:
    if batch is None:
        continue
    # Process batch...
```

### Nested Data Structures

For complex nested structures:

```python
def nested_collate(batch):
    """
    Handle arbitrarily nested data structures.
    """
    elem = batch[0]
    
    if isinstance(elem, torch.Tensor):
        # Stack if same shape, else pad
        shapes = [s.shape for s in batch]
        if all(s == shapes[0] for s in shapes):
            return torch.stack(batch)
        else:
            return pad_sequence(batch, batch_first=True)
    
    elif isinstance(elem, dict):
        return {key: nested_collate([d[key] for d in batch]) for key in elem}
    
    elif isinstance(elem, (list, tuple)):
        transposed = zip(*batch)
        return type(elem)(nested_collate(samples) for samples in transposed)
    
    else:
        return batch  # Return as list
```

## Performance Considerations

### Avoiding Copies

```python
def efficient_collate(batch):
    """
    Minimize memory copies during collation.
    """
    sequences, labels = zip(*batch)
    
    lengths = [len(seq) for seq in sequences]
    max_len = max(lengths)
    batch_size = len(sequences)
    
    # Pre-allocate output tensor
    dtype = sequences[0].dtype
    device = sequences[0].device
    
    padded = torch.zeros(
        batch_size, max_len,
        dtype=dtype, device=device
    )
    
    # Use narrow for efficient in-place assignment
    for i, (seq, length) in enumerate(zip(sequences, lengths)):
        padded.narrow(0, i, 1).narrow(1, 0, length).copy_(seq.unsqueeze(0))
    
    return padded, torch.tensor(lengths), torch.stack(labels)
```

### Parallel Collation

For expensive collation, consider offloading to workers:

```python
# Collation happens in workers, not main process
loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,
    collate_fn=expensive_collate  # Runs in worker processes
)
```

## Summary

| Scenario | Collate Strategy |
|----------|-----------------|
| Uniform tensors | Default (no custom needed) |
| Variable-length sequences | pad_sequence |
| Transformer models | Padding + attention mask |
| RNN models | pack_padded_sequence |
| Multi-modal data | Dictionary output |
| Batch augmentation | Mixup/CutMix in collate |
| Token batching | Custom batch sampler + collate |

## Complete Example

```python
class TextClassificationCollate:
    """
    Production-ready collate for text classification.
    """
    
    def __init__(self, pad_token_id=0, max_length=512):
        self.pad_token_id = pad_token_id
        self.max_length = max_length
    
    def __call__(self, batch):
        input_ids_list = []
        labels_list = []
        
        for sample in batch:
            input_ids = sample['input_ids']
            
            # Truncate if needed
            if len(input_ids) > self.max_length:
                input_ids = input_ids[:self.max_length]
            
            input_ids_list.append(input_ids)
            labels_list.append(sample['label'])
        
        # Pad sequences
        input_ids = pad_sequence(
            input_ids_list,
            batch_first=True,
            padding_value=self.pad_token_id
        )
        
        # Create attention mask
        attention_mask = (input_ids != self.pad_token_id).long()
        
        # Stack labels
        labels = torch.stack(labels_list)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


# Usage
collate_fn = TextClassificationCollate(pad_token_id=0, max_length=128)
loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
```

## Practice Exercises

1. **Multi-length Collate**: Implement a collate function that handles samples with multiple variable-length fields (e.g., question and answer texts).

2. **Bucketing**: Implement a collate function that groups similarly-lengthed sequences to minimize padding waste.

3. **CutMix**: Implement CutMix augmentation as a collate function for image classification.

## What's Next

The next section covers **Multiprocessing**, explaining how DataLoader uses worker processes for parallel data loading and how to optimize for maximum throughput.
