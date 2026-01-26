# Text Preprocessing for RNNs

## The Preprocessing Pipeline

Before text can enter a neural network, it must undergo a systematic transformation from raw characters to numerical tensors. This pipeline—tokenization, vocabulary building, encoding, and padding—forms the foundation of all NLP applications.

## Tokenization

Tokenization segments text into discrete units (tokens) that become the atomic elements of our sequence.

### Word-Level Tokenization

The simplest approach splits on whitespace and punctuation:

```python
import re

def word_tokenize(text):
    """Basic word tokenizer."""
    # Lowercase and split on non-alphanumeric characters
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    return tokens

text = "The cat sat on the mat. It was a sunny day!"
tokens = word_tokenize(text)
# ['the', 'cat', 'sat', 'on', 'the', 'mat', 'it', 'was', 'a', 'sunny', 'day']
```

**Advantages**: Intuitive, preserves semantic units
**Disadvantages**: Large vocabulary, cannot handle unknown words

### Character-Level Tokenization

Treats each character as a token:

```python
def char_tokenize(text):
    """Character-level tokenizer."""
    return list(text.lower())

text = "Hello!"
tokens = char_tokenize(text)
# ['h', 'e', 'l', 'l', 'o', '!']
```

**Advantages**: Small vocabulary (~100 characters), handles any text
**Disadvantages**: Very long sequences, harder to learn semantics

### Subword Tokenization (BPE)

Modern NLP uses subword methods that balance vocabulary size with semantic meaning:

```python
# Using HuggingFace tokenizers
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# Train BPE tokenizer
tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = Whitespace()
trainer = BpeTrainer(vocab_size=5000, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]"])
tokenizer.train(files=["corpus.txt"], trainer=trainer)

# Tokenize
output = tokenizer.encode("unbelievably")
# Might produce: ['un', 'believ', 'ably']
```

## Building Vocabulary

A vocabulary maps tokens to integer indices:

```python
class Vocabulary:
    """Maps tokens to indices and vice versa."""
    
    def __init__(self, min_freq=1):
        self.token2idx = {}
        self.idx2token = {}
        self.token_counts = {}
        self.min_freq = min_freq
        
        # Special tokens
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.sos_token = '<SOS>'
        self.eos_token = '<EOS>'
        
        # Add special tokens first
        for token in [self.pad_token, self.unk_token, self.sos_token, self.eos_token]:
            self._add_token(token)
    
    def _add_token(self, token):
        if token not in self.token2idx:
            idx = len(self.token2idx)
            self.token2idx[token] = idx
            self.idx2token[idx] = token
    
    def build_from_corpus(self, texts):
        """Build vocabulary from list of texts."""
        # Count tokens
        for text in texts:
            for token in word_tokenize(text):
                self.token_counts[token] = self.token_counts.get(token, 0) + 1
        
        # Add tokens meeting frequency threshold
        for token, count in sorted(self.token_counts.items(), key=lambda x: -x[1]):
            if count >= self.min_freq:
                self._add_token(token)
    
    def encode(self, text):
        """Convert text to list of indices."""
        tokens = word_tokenize(text)
        return [self.token2idx.get(t, self.token2idx[self.unk_token]) for t in tokens]
    
    def decode(self, indices):
        """Convert indices back to text."""
        return ' '.join(self.idx2token.get(i, self.unk_token) for i in indices)
    
    def __len__(self):
        return len(self.token2idx)

# Usage
vocab = Vocabulary(min_freq=2)
vocab.build_from_corpus(["I love machine learning", "Machine learning is amazing"])
print(f"Vocabulary size: {len(vocab)}")
```

### Special Tokens

| Token | Purpose |
|-------|---------|
| `<PAD>` | Padding for batch processing (usually index 0) |
| `<UNK>` | Unknown/out-of-vocabulary words |
| `<SOS>` | Start of sequence marker |
| `<EOS>` | End of sequence marker |
| `<SEP>` | Separator between sequences |
| `<CLS>` | Classification token (BERT-style) |

## Sequence Encoding

Convert text to numerical sequences:

```python
def text_to_sequence(text, vocab, max_length=None, add_sos=False, add_eos=False):
    """
    Convert text to padded sequence of indices.
    
    Args:
        text: Input string
        vocab: Vocabulary object
        max_length: Maximum sequence length (pads/truncates)
        add_sos: Add start-of-sequence token
        add_eos: Add end-of-sequence token
    
    Returns:
        List of token indices
    """
    # Encode text
    sequence = vocab.encode(text)
    
    # Add special tokens
    if add_sos:
        sequence = [vocab.token2idx[vocab.sos_token]] + sequence
    if add_eos:
        sequence = sequence + [vocab.token2idx[vocab.eos_token]]
    
    # Pad or truncate
    if max_length:
        pad_idx = vocab.token2idx[vocab.pad_token]
        if len(sequence) < max_length:
            sequence = sequence + [pad_idx] * (max_length - len(sequence))
        else:
            sequence = sequence[:max_length]
    
    return sequence
```

## Padding Strategies

### Post-Padding (Most Common)

Pad after the actual content:

```
"hello world" → [5, 12, 0, 0, 0]  (pad_idx=0)
```

```python
import torch
from torch.nn.utils.rnn import pad_sequence

sequences = [
    torch.tensor([1, 2, 3]),
    torch.tensor([4, 5]),
    torch.tensor([6, 7, 8, 9])
]

# Pad sequences to same length
padded = pad_sequence(sequences, batch_first=True, padding_value=0)
# tensor([[1, 2, 3, 0],
#         [4, 5, 0, 0],
#         [6, 7, 8, 9]])
```

### Pre-Padding

Pad before the content (sometimes better for RNNs):

```
"hello world" → [0, 0, 0, 5, 12]
```

```python
def pre_pad(sequences, max_len, pad_value=0):
    """Pad sequences at the beginning."""
    padded = []
    for seq in sequences:
        padding = [pad_value] * (max_len - len(seq))
        padded.append(padding + seq)
    return torch.tensor(padded)
```

### Dynamic Padding

Pad to the maximum length in each batch (more efficient):

```python
def collate_fn(batch):
    """Custom collate function for DataLoader."""
    sequences, labels = zip(*batch)
    
    # Sort by length (required for pack_padded_sequence)
    lengths = [len(seq) for seq in sequences]
    sorted_indices = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    
    sequences = [sequences[i] for i in sorted_indices]
    labels = [labels[i] for i in sorted_indices]
    lengths = [lengths[i] for i in sorted_indices]
    
    # Pad sequences
    padded = pad_sequence([torch.tensor(s) for s in sequences], batch_first=True)
    
    return padded, torch.tensor(labels), lengths
```

## Packed Sequences

For variable-length sequences, packed representations avoid unnecessary computation:

```python
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Padded sequences: (batch=3, max_len=5, features=10)
padded_input = torch.randn(3, 5, 10)
lengths = [5, 3, 2]  # Actual lengths

# Pack for efficient RNN processing
packed = pack_padded_sequence(padded_input, lengths, batch_first=True, enforce_sorted=True)

# Process with RNN
rnn = nn.LSTM(10, 20, batch_first=True)
packed_output, (h_n, c_n) = rnn(packed)

# Unpack back to padded tensor
output, output_lengths = pad_packed_sequence(packed_output, batch_first=True)
```

## Word Embeddings

### Learnable Embeddings

```python
import torch.nn as nn

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx  # Padding vectors stay zero
        )
    
    def forward(self, x):
        # x: (batch, seq_len) - token indices
        return self.embedding(x)  # (batch, seq_len, embed_dim)

# Create encoder
encoder = TextEncoder(vocab_size=10000, embed_dim=100)

# Input: batch of 32 sequences, max length 50
x = torch.randint(0, 10000, (32, 50))
embeddings = encoder(x)  # (32, 50, 100)
```

### Pre-trained Embeddings (GloVe, Word2Vec)

```python
import numpy as np

def load_pretrained_embeddings(vocab, embedding_path, embed_dim):
    """
    Load pre-trained embeddings for vocabulary.
    
    Args:
        vocab: Vocabulary object
        embedding_path: Path to embeddings file (GloVe format)
        embed_dim: Embedding dimension
    
    Returns:
        Embedding matrix of shape (vocab_size, embed_dim)
    """
    # Initialize randomly
    embeddings = np.random.normal(0, 0.1, (len(vocab), embed_dim))
    embeddings[vocab.token2idx[vocab.pad_token]] = 0  # Zero for padding
    
    # Load pre-trained vectors
    with open(embedding_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip().split(' ')
            word = parts[0]
            if word in vocab.token2idx:
                vector = np.array(parts[1:], dtype=np.float32)
                embeddings[vocab.token2idx[word]] = vector
    
    return torch.tensor(embeddings, dtype=torch.float32)

# Usage
pretrained = load_pretrained_embeddings(vocab, 'glove.6B.100d.txt', 100)
embedding_layer = nn.Embedding.from_pretrained(pretrained, freeze=False)
```

## Complete Preprocessing Pipeline

```python
import torch
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    """Complete text dataset with preprocessing."""
    
    def __init__(self, texts, labels, vocab=None, max_length=100):
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        
        # Build or use provided vocabulary
        if vocab is None:
            self.vocab = Vocabulary(min_freq=2)
            self.vocab.build_from_corpus(texts)
        else:
            self.vocab = vocab
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Convert to sequence
        sequence = text_to_sequence(text, self.vocab, self.max_length)
        
        return torch.tensor(sequence), torch.tensor(label)

# Example usage
texts = [
    "This movie was fantastic!",
    "Terrible waste of time.",
    "Great acting and direction.",
    "I fell asleep halfway through."
]
labels = [1, 0, 1, 0]  # 1 = positive, 0 = negative

dataset = TextDataset(texts, labels, max_length=20)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

for batch_sequences, batch_labels in dataloader:
    print(f"Sequences: {batch_sequences.shape}")  # (2, 20)
    print(f"Labels: {batch_labels}")
    break
```

## Data Augmentation for Text

```python
import random

def augment_text(text, p=0.1):
    """Simple text augmentation techniques."""
    words = word_tokenize(text)
    augmented = []
    
    for word in words:
        r = random.random()
        if r < p:
            # Random deletion
            continue
        elif r < 2 * p:
            # Random swap with next word
            augmented.append(word)
            if augmented and len(augmented) > 1:
                augmented[-1], augmented[-2] = augmented[-2], augmented[-1]
        else:
            augmented.append(word)
    
    return ' '.join(augmented)
```

## Summary

The text preprocessing pipeline transforms raw text into numerical tensors suitable for RNN processing:

1. **Tokenization**: Segment text into tokens (words, characters, or subwords)
2. **Vocabulary**: Map tokens to integer indices
3. **Encoding**: Convert text to index sequences
4. **Padding**: Handle variable lengths
5. **Embedding**: Convert indices to dense vectors

Key considerations:
- Choose tokenization granularity based on task and vocabulary size
- Handle unknown tokens gracefully with `<UNK>`
- Use packed sequences for efficiency
- Consider pre-trained embeddings for better initialization
