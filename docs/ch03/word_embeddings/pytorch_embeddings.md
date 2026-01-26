# Embedding Layers in PyTorch

## Overview

PyTorch provides powerful and flexible tools for working with word embeddings through the `nn.Embedding` module. This section covers the practical aspects of using embedding layers in PyTorch, including creation, initialization, freezing/fine-tuning, and integration with downstream models. We'll also explore best practices for handling vocabularies, padding, and out-of-vocabulary tokens.

## Learning Objectives

By the end of this section, you will:

- Master the `nn.Embedding` API and its parameters
- Initialize embeddings randomly or from pre-trained vectors
- Freeze, fine-tune, and partially update embeddings
- Handle padding and unknown tokens properly
- Build complete models that use embedding layers
- Understand memory and performance considerations

## The nn.Embedding Module

### Basic Usage

`nn.Embedding` is a lookup table that maps integer indices to dense vectors:

```python
import torch
import torch.nn as nn

# Create an embedding layer
# vocab_size: number of unique tokens
# embedding_dim: size of each embedding vector
embedding = nn.Embedding(num_embeddings=1000, embedding_dim=128)

print(f"Embedding weight shape: {embedding.weight.shape}")
# Output: torch.Size([1000, 128])

# Look up embeddings for indices
indices = torch.tensor([1, 5, 3, 7])
vectors = embedding(indices)
print(f"Output shape: {vectors.shape}")
# Output: torch.Size([4, 128])

# Batch lookup
batch_indices = torch.tensor([[1, 2, 3], [4, 5, 6]])  # (batch_size=2, seq_len=3)
batch_vectors = embedding(batch_indices)
print(f"Batch output shape: {batch_vectors.shape}")
# Output: torch.Size([2, 3, 128])
```

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `num_embeddings` | Size of vocabulary (number of rows) | Required |
| `embedding_dim` | Dimension of each embedding | Required |
| `padding_idx` | Index to fill with zeros (not updated) | None |
| `max_norm` | Max norm for embeddings (renormalize if exceeded) | None |
| `norm_type` | p of p-norm for max_norm | 2.0 |
| `scale_grad_by_freq` | Scale gradients by inverse frequency | False |
| `sparse` | Use sparse gradients (for large vocabularies) | False |

### Padding Index

When working with variable-length sequences, use `padding_idx` to ensure padding tokens don't affect your model:

```python
# Create embedding with padding at index 0
PAD_IDX = 0
embedding = nn.Embedding(
    num_embeddings=1000,
    embedding_dim=128,
    padding_idx=PAD_IDX
)

# The padding embedding is always zero
print(f"Padding vector: {embedding.weight[PAD_IDX]}")
# Output: tensor([0., 0., 0., ...])

# It stays zero even after training
indices = torch.tensor([0, 1, 2, 0, 3])
vectors = embedding(indices)
print(f"Embedding at position 0 (PAD): {vectors[0].sum().item()}")  # 0.0
print(f"Embedding at position 1: {vectors[1].sum().item()}")  # Non-zero
```

## Initialization Strategies

### Default Initialization

By default, PyTorch initializes embeddings from a normal distribution:

```python
embedding = nn.Embedding(1000, 128)
print(f"Mean: {embedding.weight.mean():.4f}")
print(f"Std: {embedding.weight.std():.4f}")
# Approximately: Mean ≈ 0, Std ≈ 1
```

### Custom Initialization

```python
# Uniform initialization
embedding = nn.Embedding(1000, 128)
nn.init.uniform_(embedding.weight, -0.1, 0.1)

# Xavier/Glorot initialization
nn.init.xavier_uniform_(embedding.weight)

# Normal with specific std
nn.init.normal_(embedding.weight, mean=0, std=0.01)

# Zeros (not recommended for training from scratch)
nn.init.zeros_(embedding.weight)

# Custom scaling based on embedding dimension
embedding_dim = 128
nn.init.uniform_(embedding.weight, -1/embedding_dim**0.5, 1/embedding_dim**0.5)
```

### Loading Pre-trained Embeddings

```python
import numpy as np

def load_pretrained_embeddings(embedding_layer, pretrained_vectors, word_to_idx):
    """
    Load pre-trained embeddings into an nn.Embedding layer.
    
    Args:
        embedding_layer: nn.Embedding instance
        pretrained_vectors: Dict mapping words to numpy arrays
        word_to_idx: Dict mapping words to indices in embedding layer
    
    Returns:
        Number of words loaded
    """
    loaded = 0
    embedding_dim = embedding_layer.embedding_dim
    
    for word, idx in word_to_idx.items():
        if word in pretrained_vectors:
            vector = pretrained_vectors[word]
            if len(vector) == embedding_dim:
                embedding_layer.weight.data[idx] = torch.tensor(vector)
                loaded += 1
    
    print(f"Loaded {loaded}/{len(word_to_idx)} pre-trained embeddings")
    return loaded


# Example: Loading GloVe embeddings
def load_glove(filepath, embedding_dim=100):
    """Load GloVe embeddings from file."""
    embeddings = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype=np.float32)
            if len(vector) == embedding_dim:
                embeddings[word] = vector
    return embeddings

# Usage:
# glove_embeddings = load_glove('glove.6B.100d.txt', 100)
# load_pretrained_embeddings(embedding, glove_embeddings, word_to_idx)
```

### Handling Unknown Words

For words not in pre-trained vocabulary, use special initialization:

```python
def initialize_with_pretrained(vocab, pretrained_path, embedding_dim):
    """
    Create embedding layer with pre-trained vectors and special token handling.
    """
    # Special tokens
    PAD_TOKEN = '<PAD>'
    UNK_TOKEN = '<UNK>'
    
    # Build vocabulary with special tokens
    word_to_idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for word in vocab:
        if word not in word_to_idx:
            word_to_idx[word] = len(word_to_idx)
    
    vocab_size = len(word_to_idx)
    
    # Create embedding layer
    embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
    
    # Initialize with small random values
    nn.init.normal_(embedding.weight, mean=0, std=0.1)
    
    # Set padding to zero
    embedding.weight.data[0] = torch.zeros(embedding_dim)
    
    # Load pre-trained vectors
    pretrained = load_glove(pretrained_path, embedding_dim)
    
    found = 0
    for word, idx in word_to_idx.items():
        if word in pretrained:
            embedding.weight.data[idx] = torch.tensor(pretrained[word])
            found += 1
    
    # UNK embedding: average of all found embeddings
    if found > 0:
        found_indices = [idx for word, idx in word_to_idx.items() 
                        if word in pretrained]
        unk_vector = embedding.weight.data[found_indices].mean(dim=0)
        embedding.weight.data[word_to_idx[UNK_TOKEN]] = unk_vector
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"Pre-trained coverage: {found}/{vocab_size} ({100*found/vocab_size:.1f}%)")
    
    return embedding, word_to_idx
```

## Freezing and Fine-tuning

### Freeze Embeddings (Feature Extraction)

```python
# Method 1: Set requires_grad to False
embedding = nn.Embedding(1000, 128)
embedding.weight.requires_grad = False

# Method 2: Create as frozen from the start
embedding = nn.Embedding(1000, 128)
embedding.weight.requires_grad_(False)

# Method 3: Using from_pretrained with freeze=True
pretrained_weights = torch.randn(1000, 128)
embedding = nn.Embedding.from_pretrained(pretrained_weights, freeze=True)
print(f"requires_grad: {embedding.weight.requires_grad}")  # False
```

### Fine-tune Embeddings

```python
# Unfreeze embeddings for fine-tuning
embedding.weight.requires_grad_(True)

# Or create unfrozen from pre-trained
embedding = nn.Embedding.from_pretrained(pretrained_weights, freeze=False)
```

### Partial Fine-tuning

```python
class PartiallyFrozenEmbedding(nn.Module):
    """
    Embedding where some tokens are frozen and others are trainable.
    Useful for keeping pre-trained embeddings for common words
    while training embeddings for domain-specific vocabulary.
    """
    
    def __init__(self, pretrained_weights, frozen_indices):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            pretrained_weights, freeze=False
        )
        self.frozen_indices = set(frozen_indices)
        
        # Store original frozen embeddings
        self.register_buffer(
            'frozen_weights',
            pretrained_weights[list(frozen_indices)].clone()
        )
    
    def forward(self, x):
        # Get embeddings
        embeddings = self.embedding(x)
        
        # During training, restore frozen embeddings
        if self.training:
            with torch.no_grad():
                for i, idx in enumerate(self.frozen_indices):
                    self.embedding.weight.data[idx] = self.frozen_weights[i]
        
        return embeddings
```

### Differential Learning Rates

```python
# Different learning rates for embeddings vs. other parameters
embedding = nn.Embedding(1000, 128)
other_layer = nn.Linear(128, 64)

# Slower learning for pre-trained embeddings
optimizer = torch.optim.Adam([
    {'params': embedding.parameters(), 'lr': 1e-5},  # Small LR for embeddings
    {'params': other_layer.parameters(), 'lr': 1e-3}  # Normal LR for other layers
])
```

## Building Complete Models

### Text Classification with Embeddings

```python
class TextClassifier(nn.Module):
    """
    Simple text classifier using embeddings.
    
    Architecture:
        tokens → embedding → mean pooling → MLP → class logits
    """
    
    def __init__(self, vocab_size, embedding_dim, num_classes, 
                 hidden_dim=128, dropout=0.5, padding_idx=0):
        super().__init__()
        
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x, lengths=None):
        """
        Args:
            x: Token indices (batch_size, max_seq_len)
            lengths: Actual sequence lengths (batch_size,) for masking
        
        Returns:
            Class logits (batch_size, num_classes)
        """
        # Get embeddings: (batch, seq_len, embed_dim)
        embedded = self.embedding(x)
        
        if lengths is not None:
            # Create mask for padding
            mask = torch.arange(x.size(1), device=x.device)[None, :] < lengths[:, None]
            mask = mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
            
            # Masked mean pooling
            embedded = embedded * mask
            pooled = embedded.sum(dim=1) / lengths.unsqueeze(-1).float()
        else:
            # Simple mean pooling (assumes no padding or padding_idx handles it)
            pooled = embedded.mean(dim=1)
        
        # Classify
        logits = self.classifier(pooled)
        
        return logits


# Example usage
model = TextClassifier(
    vocab_size=10000,
    embedding_dim=128,
    num_classes=5,
    padding_idx=0
)

# Dummy input
batch_size, seq_len = 32, 50
x = torch.randint(0, 10000, (batch_size, seq_len))
lengths = torch.randint(10, seq_len, (batch_size,))

logits = model(x, lengths)
print(f"Output shape: {logits.shape}")  # torch.Size([32, 5])
```

### N-gram Language Model

```python
class NGramLanguageModel(nn.Module):
    """
    N-gram language model with embeddings.
    
    Predicts next word from previous n-1 words.
    """
    
    def __init__(self, vocab_size, embedding_dim, context_size, hidden_dim=128):
        super().__init__()
        
        self.context_size = context_size
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        self.network = nn.Sequential(
            nn.Linear(context_size * embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size)
        )
    
    def forward(self, context):
        """
        Args:
            context: Previous words (batch_size, context_size)
        
        Returns:
            Next word logits (batch_size, vocab_size)
        """
        # Embed context words: (batch, context_size, embed_dim)
        embedded = self.embeddings(context)
        
        # Flatten: (batch, context_size * embed_dim)
        embedded = embedded.view(embedded.size(0), -1)
        
        # Predict next word
        logits = self.network(embedded)
        
        return logits


# Training example
def train_language_model(model, data, epochs=10, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        total_loss = 0
        for context, target in data:
            optimizer.zero_grad()
            logits = model(context)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(data):.4f}")
```

## Memory and Performance

### Sparse Embeddings

For large vocabularies with infrequent updates, use sparse gradients:

```python
# Sparse gradients - only update accessed embeddings
embedding = nn.Embedding(1000000, 128, sparse=True)

# Must use optimizers that support sparse gradients
optimizer = torch.optim.SparseAdam(embedding.parameters(), lr=0.001)
# or
optimizer = torch.optim.SGD(embedding.parameters(), lr=0.01)
```

### Memory Estimation

```python
def estimate_embedding_memory(vocab_size, embedding_dim, dtype=torch.float32):
    """Estimate memory usage of embedding layer."""
    bytes_per_element = {
        torch.float32: 4,
        torch.float16: 2,
        torch.int8: 1
    }
    
    total_bytes = vocab_size * embedding_dim * bytes_per_element[dtype]
    
    return {
        'bytes': total_bytes,
        'KB': total_bytes / 1024,
        'MB': total_bytes / (1024 ** 2),
        'GB': total_bytes / (1024 ** 3)
    }

# Example
memory = estimate_embedding_memory(100000, 300)
print(f"100K vocab × 300 dim: {memory['MB']:.1f} MB")
# Output: 100K vocab × 300 dim: 114.4 MB
```

### Embedding Compression

```python
class CompressedEmbedding(nn.Module):
    """
    Compressed embedding using low-rank factorization.
    
    Instead of V × D matrix, uses V × K and K × D matrices (K << D).
    """
    
    def __init__(self, vocab_size, embedding_dim, rank):
        super().__init__()
        self.vocab_to_low = nn.Embedding(vocab_size, rank)
        self.low_to_high = nn.Linear(rank, embedding_dim, bias=False)
    
    def forward(self, x):
        low_dim = self.vocab_to_low(x)
        return self.low_to_high(low_dim)

# Memory comparison
vocab_size, embed_dim, rank = 100000, 300, 64

full_params = vocab_size * embed_dim
compressed_params = vocab_size * rank + rank * embed_dim

print(f"Full embedding: {full_params:,} parameters")
print(f"Compressed (rank={rank}): {compressed_params:,} parameters")
print(f"Compression ratio: {full_params/compressed_params:.1f}x")
```

## Best Practices

### Vocabulary Management

```python
class Vocabulary:
    """Robust vocabulary management for embeddings."""
    
    def __init__(self, pad_token='<PAD>', unk_token='<UNK>',
                 bos_token='<BOS>', eos_token='<EOS>'):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        
        # Initialize with special tokens
        self.token_to_idx = {}
        self.idx_to_token = {}
        
        # Add special tokens in order
        for token in [pad_token, unk_token, bos_token, eos_token]:
            if token:
                self._add_token(token)
        
        self.pad_idx = self.token_to_idx.get(pad_token, None)
        self.unk_idx = self.token_to_idx.get(unk_token, None)
    
    def _add_token(self, token):
        if token not in self.token_to_idx:
            idx = len(self.token_to_idx)
            self.token_to_idx[token] = idx
            self.idx_to_token[idx] = token
    
    def build(self, texts, min_freq=1, max_size=None):
        """Build vocabulary from texts."""
        from collections import Counter
        
        # Count tokens
        counter = Counter()
        for text in texts:
            counter.update(text.split())
        
        # Filter by frequency and size
        tokens = [t for t, c in counter.most_common(max_size) if c >= min_freq]
        
        for token in tokens:
            self._add_token(token)
        
        return self
    
    def __len__(self):
        return len(self.token_to_idx)
    
    def encode(self, text):
        """Convert text to indices."""
        tokens = text.split()
        return [self.token_to_idx.get(t, self.unk_idx) for t in tokens]
    
    def decode(self, indices):
        """Convert indices to text."""
        return ' '.join(self.idx_to_token.get(i, self.unk_token) for i in indices)
```

### Complete Pipeline

```python
def create_embedding_pipeline(texts, labels, pretrained_path=None,
                             embedding_dim=128, min_freq=2, max_vocab=50000):
    """
    Complete pipeline for creating embeddings and training data.
    
    Args:
        texts: List of text strings
        labels: List of labels
        pretrained_path: Path to pre-trained embeddings (optional)
        embedding_dim: Embedding dimension
        min_freq: Minimum token frequency
        max_vocab: Maximum vocabulary size
    
    Returns:
        embedding_layer, vocabulary, encoded_data
    """
    # Build vocabulary
    vocab = Vocabulary().build(texts, min_freq=min_freq, max_size=max_vocab)
    print(f"Vocabulary size: {len(vocab)}")
    
    # Create embedding layer
    embedding = nn.Embedding(
        num_embeddings=len(vocab),
        embedding_dim=embedding_dim,
        padding_idx=vocab.pad_idx
    )
    
    # Load pre-trained if available
    if pretrained_path:
        pretrained = load_glove(pretrained_path, embedding_dim)
        loaded = load_pretrained_embeddings(embedding, pretrained, vocab.token_to_idx)
    
    # Encode texts
    encoded_texts = [vocab.encode(text) for text in texts]
    
    # Pad sequences
    max_len = max(len(t) for t in encoded_texts)
    padded = torch.zeros(len(texts), max_len, dtype=torch.long)
    lengths = torch.zeros(len(texts), dtype=torch.long)
    
    for i, enc in enumerate(encoded_texts):
        length = len(enc)
        padded[i, :length] = torch.tensor(enc)
        lengths[i] = length
    
    return embedding, vocab, (padded, lengths, torch.tensor(labels))
```

## Key Takeaways

!!! success "Main Concepts"

    1. **nn.Embedding** is a learnable lookup table for converting indices to vectors
    2. **padding_idx** ensures padding tokens have zero embeddings
    3. **Pre-trained embeddings** can be loaded with `from_pretrained()` or manual assignment
    4. **Freezing** embeddings prevents updates; useful for feature extraction
    5. **Sparse gradients** help with large vocabularies

!!! tip "Best Practices"

    - Always use `padding_idx` for variable-length sequences
    - Initialize unknown tokens with average of known embeddings
    - Use lower learning rates when fine-tuning pre-trained embeddings
    - Consider low-rank factorization for memory-constrained settings
    - Build a robust vocabulary class to handle special tokens

## References

- PyTorch Documentation: nn.Embedding
- Collobert, R., & Weston, J. (2008). "A Unified Architecture for Natural Language Processing"
- Kim, Y. (2014). "Convolutional Neural Networks for Sentence Classification"
