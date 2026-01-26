# FastText: Subword Embeddings

## Overview

FastText, developed by Facebook AI Research (Bojanowski et al., 2017), extends Word2Vec by incorporating **subword information** through character n-grams. This approach addresses key limitations of traditional word embeddings: the inability to handle out-of-vocabulary (OOV) words and poor representations for rare words. FastText represents each word as a bag of character n-grams, enabling meaningful embeddings even for unseen words based on their morphological structure.

## Learning Objectives

By the end of this section, you will:

- Understand the limitations of word-level embeddings
- Learn how character n-grams capture morphological information
- Implement FastText's subword representation in PyTorch
- Handle out-of-vocabulary words effectively
- Apply FastText to morphologically rich languages

## Motivation: The OOV Problem

### Limitations of Word-Level Embeddings

Traditional word embeddings (Word2Vec, GloVe) face several challenges:

| Problem | Description | Example |
|---------|-------------|---------|
| **OOV Words** | No embedding for unseen words | "unforgettable" not in training |
| **Rare Words** | Poor embeddings for low-frequency words | "serendipitous" seen 3 times |
| **Morphology** | No sharing between related forms | "run", "runs", "running" are separate |
| **Misspellings** | Cannot handle typos | "teh" vs "the" |
| **New Words** | Cannot adapt to neologisms | "COVID-19", "selfie" |

### The Morphological Solution

Many languages have rich morphological systems where word meanings relate to their structure:

**English:**
- "unhappiness" = "un-" (negation) + "happy" + "-ness" (noun)
- "internationalization" = "inter-" + "nation" + "-al" + "-ize" + "-ation"

**German (compound words):**
- "Handschuh" (glove) = "Hand" (hand) + "Schuh" (shoe)
- "Krankenhaus" (hospital) = "Kranken" (sick) + "Haus" (house)

**Finnish (agglutinative):**
- "talossanikin" = "talo" (house) + "-ssa" (in) + "-ni" (my) + "-kin" (also)
- = "also in my house"

## Character N-grams

### Definition

A character n-gram is a contiguous sequence of $n$ characters. FastText uses n-grams with $n$ typically ranging from 3 to 6.

**Special markers:** FastText adds `<` and `>` to mark word boundaries:
- Word "where" becomes `<where>`

### Extracting N-grams

For the word "where" with $n \in \{3, 4, 5, 6\}$:

| n | N-grams |
|---|---------|
| 3 | `<wh`, `whe`, `her`, `ere`, `re>` |
| 4 | `<whe`, `wher`, `here`, `ere>` |
| 5 | `<wher`, `where`, `here>` |
| 6 | `<where`, `where>` |

Plus the special full word token: `<where>`

```python
def extract_ngrams(word, min_n=3, max_n=6):
    """
    Extract character n-grams from a word.
    
    Args:
        word: Input word
        min_n: Minimum n-gram length
        max_n: Maximum n-gram length
    
    Returns:
        Set of n-grams including the full word
    """
    # Add boundary markers
    word_with_markers = f"<{word}>"
    ngrams = set()
    
    # Extract n-grams of each length
    for n in range(min_n, max_n + 1):
        for i in range(len(word_with_markers) - n + 1):
            ngram = word_with_markers[i:i+n]
            ngrams.add(ngram)
    
    # Add the full word as a special token
    ngrams.add(word_with_markers)
    
    return ngrams

# Example
word = "where"
ngrams = extract_ngrams(word, min_n=3, max_n=6)
print(f"N-grams for '{word}':")
for n in range(3, 7):
    n_grams_of_length_n = [ng for ng in ngrams if len(ng) == n]
    if n_grams_of_length_n:
        print(f"  {n}-grams: {sorted(n_grams_of_length_n)}")
```

**Output:**
```
N-grams for 'where':
  3-grams: ['<wh', 'ere', 'her', 're>', 'whe']
  4-grams: ['<whe', 'ere>', 'here', 'wher']
  5-grams: ['<wher', 'here>', 'where']
  6-grams: ['<where', 'where>']
```

## FastText Model Architecture

### Word Representation

In FastText, a word $w$ is represented as the sum of its n-gram embeddings:

$$\mathbf{v}_w = \sum_{g \in \mathcal{G}_w} \mathbf{z}_g$$

Where:
- $\mathcal{G}_w$: Set of n-grams for word $w$ (including the word itself)
- $\mathbf{z}_g \in \mathbb{R}^d$: Embedding for n-gram $g$

### Skip-gram with Subwords

FastText uses the Skip-gram objective but with subword-augmented representations:

$$\mathcal{L} = -\sum_{(w, c) \in \mathcal{D}} \left[ \log \sigma(\mathbf{u}_c^\top \mathbf{v}_w) + \sum_{i=1}^{k} \mathbb{E}_{n \sim P_n} \log \sigma(-\mathbf{u}_n^\top \mathbf{v}_w) \right]$$

Where:
- $\mathbf{v}_w = \sum_{g \in \mathcal{G}_w} \mathbf{z}_g$: Word embedding as sum of n-grams
- $\mathbf{u}_c$: Context word embedding
- $k$: Number of negative samples

### Handling Large N-gram Vocabularies

The number of possible n-grams is huge. FastText uses **hashing**:

1. Hash each n-gram to a bucket: $h(g) \in \{1, \ldots, B\}$
2. Use $B$ embedding vectors (typically $B = 2,000,000$)
3. Different n-grams may share the same embedding (collision)

```python
def hash_ngram(ngram, num_buckets=2000000):
    """Hash n-gram to bucket index using FNV-1a hash."""
    FNV_PRIME = 0x01000193
    FNV_OFFSET = 0x811c9dc5
    
    h = FNV_OFFSET
    for char in ngram.encode('utf-8'):
        h ^= char
        h = (h * FNV_PRIME) & 0xFFFFFFFF
    
    return h % num_buckets
```

## PyTorch Implementation

### FastText Model

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

class FastTextModel(nn.Module):
    """
    FastText model with subword embeddings.
    
    Each word is represented as the sum of its character n-gram embeddings.
    """
    
    def __init__(self, vocab_size, num_buckets, embedding_dim, min_n=3, max_n=6):
        """
        Args:
            vocab_size: Number of words in vocabulary
            num_buckets: Number of hash buckets for n-grams
            embedding_dim: Dimension of embeddings
            min_n: Minimum n-gram length
            max_n: Maximum n-gram length
        """
        super(FastTextModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.num_buckets = num_buckets
        self.embedding_dim = embedding_dim
        self.min_n = min_n
        self.max_n = max_n
        
        # Word embeddings (for known words)
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # N-gram embeddings (hashed)
        self.ngram_embeddings = nn.Embedding(num_buckets, embedding_dim)
        
        # Context embeddings (for Skip-gram)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Initialize
        init_range = 0.5 / embedding_dim
        nn.init.uniform_(self.word_embeddings.weight, -init_range, init_range)
        nn.init.uniform_(self.ngram_embeddings.weight, -init_range, init_range)
        nn.init.uniform_(self.context_embeddings.weight, -init_range, init_range)
    
    def get_ngram_indices(self, word):
        """Get hashed indices for all n-grams of a word."""
        ngrams = extract_ngrams(word, self.min_n, self.max_n)
        indices = [hash_ngram(ng, self.num_buckets) for ng in ngrams]
        return torch.tensor(indices, dtype=torch.long)
    
    def get_word_embedding(self, word, word_idx=None):
        """
        Get embedding for a word (sum of n-gram embeddings).
        
        Args:
            word: The word string
            word_idx: Index in vocabulary (None for OOV)
        
        Returns:
            Word embedding tensor
        """
        ngram_indices = self.get_ngram_indices(word)
        ngram_embs = self.ngram_embeddings(ngram_indices)
        
        # Sum n-gram embeddings
        word_emb = ngram_embs.sum(dim=0)
        
        # Add word embedding if in vocabulary
        if word_idx is not None:
            word_emb = word_emb + self.word_embeddings.weight[word_idx]
        
        return word_emb
    
    def forward(self, center_words, center_word_strs, context_words, negative_words):
        """
        Compute FastText loss with negative sampling.
        
        Args:
            center_words: (batch_size,) center word indices
            center_word_strs: List of center word strings
            context_words: (batch_size,) positive context word indices
            negative_words: (batch_size, num_neg) negative word indices
        
        Returns:
            Loss value
        """
        batch_size = center_words.size(0)
        
        # Get center word embeddings (including n-grams)
        center_embs = []
        for i, (word_idx, word_str) in enumerate(zip(center_words, center_word_strs)):
            emb = self.get_word_embedding(word_str, word_idx.item())
            center_embs.append(emb)
        center_embs = torch.stack(center_embs)  # (batch, dim)
        
        # Get context embeddings
        context_embs = self.context_embeddings(context_words)  # (batch, dim)
        negative_embs = self.context_embeddings(negative_words)  # (batch, num_neg, dim)
        
        # Positive score
        positive_score = torch.sum(center_embs * context_embs, dim=1)
        positive_loss = F.logsigmoid(positive_score)
        
        # Negative scores
        # (batch, dim, 1)
        center_embs_expanded = center_embs.unsqueeze(2)
        # (batch, num_neg)
        negative_scores = torch.bmm(negative_embs, center_embs_expanded).squeeze(2)
        negative_loss = F.logsigmoid(-negative_scores).sum(dim=1)
        
        # Total loss
        loss = -(positive_loss + negative_loss).mean()
        
        return loss


class FastTextVocab:
    """Vocabulary manager for FastText."""
    
    def __init__(self, min_count=5):
        self.min_count = min_count
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.word_counts = defaultdict(int)
    
    def build(self, tokens):
        """Build vocabulary from tokens."""
        # Count words
        for token in tokens:
            self.word_counts[token] += 1
        
        # Filter by minimum count
        idx = 0
        for word, count in self.word_counts.items():
            if count >= self.min_count:
                self.word_to_idx[word] = idx
                self.idx_to_word[idx] = word
                idx += 1
        
        print(f"Vocabulary size: {len(self.word_to_idx)}")
        return self
    
    def __len__(self):
        return len(self.word_to_idx)
    
    def __contains__(self, word):
        return word in self.word_to_idx
```

### Training FastText

```python
def train_fasttext(tokens, embedding_dim=100, num_buckets=2000000, 
                   min_n=3, max_n=6, window_size=5, num_negatives=5,
                   epochs=5, batch_size=512, lr=0.025, min_count=5):
    """
    Train FastText model.
    
    Args:
        tokens: List of tokens
        embedding_dim: Embedding dimension
        num_buckets: Hash buckets for n-grams
        min_n, max_n: N-gram range
        window_size: Context window size
        num_negatives: Negative samples per positive
        epochs: Training epochs
        batch_size: Batch size
        lr: Learning rate
        min_count: Minimum word frequency
    
    Returns:
        Trained model, vocabulary
    """
    # Build vocabulary
    vocab = FastTextVocab(min_count=min_count).build(tokens)
    vocab_size = len(vocab)
    
    # Create model
    model = FastTextModel(vocab_size, num_buckets, embedding_dim, min_n, max_n)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    # Prepare noise distribution for negative sampling
    word_freqs = np.array([vocab.word_counts[vocab.idx_to_word[i]] 
                          for i in range(vocab_size)], dtype=np.float64)
    word_freqs = word_freqs ** 0.75
    noise_dist = word_freqs / word_freqs.sum()
    
    # Generate training pairs
    print("Generating training pairs...")
    pairs = []
    for i in range(window_size, len(tokens) - window_size):
        center = tokens[i]
        if center not in vocab:
            continue
        
        center_idx = vocab.word_to_idx[center]
        
        for j in range(-window_size, window_size + 1):
            if j != 0:
                context = tokens[i + j]
                if context in vocab:
                    context_idx = vocab.word_to_idx[context]
                    pairs.append((center_idx, center, context_idx))
    
    print(f"Training pairs: {len(pairs)}")
    
    # Training loop
    losses = []
    for epoch in range(epochs):
        np.random.shuffle(pairs)
        total_loss = 0
        num_batches = 0
        
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i+batch_size]
            if len(batch) < batch_size // 2:
                continue
            
            # Prepare batch
            center_indices = torch.tensor([p[0] for p in batch], dtype=torch.long)
            center_words = [p[1] for p in batch]
            context_indices = torch.tensor([p[2] for p in batch], dtype=torch.long)
            
            # Sample negatives
            negatives = np.random.choice(
                vocab_size, 
                size=(len(batch), num_negatives),
                p=noise_dist
            )
            negative_indices = torch.tensor(negatives, dtype=torch.long)
            
            # Forward and backward
            optimizer.zero_grad()
            loss = model(center_indices, center_words, context_indices, negative_indices)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return model, vocab, losses
```

## Handling OOV Words

### The Key Advantage

FastText can generate embeddings for any word, including those never seen during training:

```python
def get_oov_embedding(model, word):
    """
    Get embedding for an out-of-vocabulary word.
    
    The embedding is computed purely from character n-grams.
    """
    return model.get_word_embedding(word, word_idx=None)

# Example: Get embedding for an unseen word
model = FastTextModel(vocab_size=1000, num_buckets=10000, embedding_dim=100)

# These words might not be in vocabulary
oov_words = ["unforgettableness", "COVID-19", "blockchain", "deeplearning"]

for word in oov_words:
    emb = get_oov_embedding(model, word)
    print(f"'{word}': embedding shape = {emb.shape}, norm = {emb.norm().item():.4f}")
```

### Morphological Generalization

FastText captures morphological relationships through shared n-grams:

| Word 1 | Word 2 | Shared N-grams | Relationship |
|--------|--------|----------------|--------------|
| "happy" | "unhappy" | "happ", "appy", "ppy>" | Negation |
| "run" | "running" | "run", "<run", "run>" | Verb form |
| "nation" | "national" | "natio", "ation", "tion" | Derivation |

## Comparison with Word2Vec

| Aspect | Word2Vec | FastText |
|--------|----------|----------|
| **Word Unit** | Whole word | N-grams + word |
| **OOV Handling** | Cannot | Can generate |
| **Rare Words** | Poor | Better (shared n-grams) |
| **Morphology** | Not captured | Captured |
| **Vocabulary Size** | Fixed | Flexible |
| **Training Speed** | Faster | Slower (more embeddings) |
| **Memory** | Lower | Higher |

### When to Use FastText

**Use FastText when:**
- Working with morphologically rich languages (German, Finnish, Turkish)
- OOV words are common (social media, technical domains)
- Rare words matter for your task
- Typos/misspellings need to be handled

**Use Word2Vec when:**
- Memory is constrained
- Training speed is critical
- Working with languages with limited morphology
- All words of interest are in vocabulary

## Advanced: Subword Pooling Strategies

### Sum Pooling (Default)

$$\mathbf{v}_w = \sum_{g \in \mathcal{G}_w} \mathbf{z}_g$$

### Mean Pooling

$$\mathbf{v}_w = \frac{1}{|\mathcal{G}_w|} \sum_{g \in \mathcal{G}_w} \mathbf{z}_g$$

### Attention Pooling

$$\mathbf{v}_w = \sum_{g \in \mathcal{G}_w} \alpha_g \mathbf{z}_g$$

Where $\alpha_g$ are learned attention weights.

```python
class FastTextWithAttention(nn.Module):
    """FastText with attention-based n-gram pooling."""
    
    def __init__(self, num_buckets, embedding_dim):
        super().__init__()
        self.ngram_embeddings = nn.Embedding(num_buckets, embedding_dim)
        self.attention = nn.Linear(embedding_dim, 1)
    
    def get_word_embedding(self, ngram_indices):
        # Get n-gram embeddings
        ngram_embs = self.ngram_embeddings(ngram_indices)  # (num_ngrams, dim)
        
        # Compute attention weights
        attn_scores = self.attention(ngram_embs).squeeze(-1)  # (num_ngrams,)
        attn_weights = F.softmax(attn_scores, dim=0)  # (num_ngrams,)
        
        # Weighted sum
        word_emb = (attn_weights.unsqueeze(1) * ngram_embs).sum(dim=0)
        
        return word_emb
```

## Using Pre-trained FastText

Facebook provides pre-trained FastText embeddings for 157 languages:

```python
import gensim.downloader as api

# Download pre-trained FastText
# (Note: requires gensim library)
# fasttext_model = api.load('fasttext-wiki-news-subwords-300')

# Alternative: Load from .bin file
from gensim.models import FastText

# Load pre-trained model
# model = FastText.load_fasttext_format('cc.en.300.bin')

# Get embedding (works for OOV too!)
# embedding = model.wv['unforgettableness']
```

## Evaluation Example

```python
def evaluate_analogy(model, vocab, word_a, word_b, word_c, top_k=5):
    """
    Evaluate word analogy: a is to b as c is to ?
    
    Uses FastText embeddings (including OOV handling).
    """
    # Get embeddings (works even for OOV words)
    def get_emb(word):
        word_idx = vocab.word_to_idx.get(word, None)
        return model.get_word_embedding(word, word_idx)
    
    emb_a = get_emb(word_a)
    emb_b = get_emb(word_b)
    emb_c = get_emb(word_c)
    
    # Target: b - a + c
    target = emb_b - emb_a + emb_c
    target = target / target.norm()
    
    # Find most similar
    similarities = []
    for idx in range(len(vocab)):
        word = vocab.idx_to_word[idx]
        if word in {word_a, word_b, word_c}:
            continue
        
        emb = get_emb(word)
        emb = emb / emb.norm()
        sim = torch.dot(target, emb).item()
        similarities.append((word, sim))
    
    # Sort and return top-k
    similarities.sort(key=lambda x: -x[1])
    return similarities[:top_k]


# Test morphological analogies
print("Morphological Analogies:")
print("happy : unhappy :: possible : ?")
# Expected: impossible

print("\nVerb Form Analogies:")
print("run : running :: walk : ?")
# Expected: walking
```

## Key Takeaways

!!! success "Main Concepts"

    1. **FastText** uses character n-grams to represent words
    2. **OOV handling** is automatic through n-gram composition
    3. **Morphological relationships** are captured via shared n-grams
    4. **Hashing** keeps the n-gram vocabulary manageable
    5. **Trade-off**: Better generalization vs. more memory/computation

!!! tip "Best Practices"

    - Use n-gram range 3-6 for most languages
    - Increase `min_n` for languages with longer words
    - Use pre-trained models when available
    - Consider memory constraints for large vocabularies

## References

- Bojanowski, P., et al. (2017). "Enriching Word Vectors with Subword Information"
- Mikolov, T., et al. (2018). "Advances in Pre-Training Distributed Word Representations"
- FastText Library: https://fasttext.cc/
