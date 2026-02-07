# FastText

## Overview

FastText, developed by Facebook AI Research (Bojanowski et al., 2017), extends Word2Vec by incorporating **subword information** through character n-grams. This approach addresses key limitations of traditional word embeddings: the inability to handle out-of-vocabulary (OOV) words and poor representations for rare words. FastText represents each word as a bag of character n-grams, enabling meaningful embeddings even for unseen words based on their morphological structure.

## Learning Objectives

By the end of this section, you will:

- Understand why word-level embeddings fail on morphologically rich languages
- Learn the FastText model architecture and how it extends Skip-gram
- Implement FastText training in PyTorch
- Compare FastText with Word2Vec on key trade-offs
- Use pre-trained FastText models for downstream tasks

## Motivation: The OOV Problem

Traditional word embeddings (Word2Vec, GloVe) treat each word as an atomic unit, leading to several practical challenges:

| Problem | Description | Example |
|---------|-------------|---------|
| **OOV Words** | No embedding for unseen words | "unforgettable" not in training vocab |
| **Rare Words** | Poor embeddings for low-frequency words | "serendipitous" seen only 3 times |
| **Morphology** | No parameter sharing between related forms | "run", "runs", "running" are independent |
| **Misspellings** | Cannot handle typos | "teh" vs "the" |
| **New Words** | Cannot adapt to neologisms | "COVID-19", "selfie" |

### The Morphological Solution

Many languages have rich morphological systems where meaning relates directly to word structure:

**English:** "unhappiness" = "un-" (negation) + "happy" + "-ness" (noun formation)

**German (compounding):** "Krankenhaus" (hospital) = "Kranken" (sick) + "Haus" (house)

**Finnish (agglutination):** "talossanikin" = "talo" (house) + "-ssa" (in) + "-ni" (my) + "-kin" (also) = "also in my house"

FastText captures these structural patterns by decomposing words into character n-grams.

## FastText Model Architecture

### Word Representation

In FastText, a word $w$ is represented as the **sum** of its character n-gram embeddings:

$$\mathbf{v}_w = \sum_{g \in \mathcal{G}_w} \mathbf{z}_g$$

Where:

- $\mathcal{G}_w$: Set of n-grams for word $w$ (including the word itself)
- $\mathbf{z}_g \in \mathbb{R}^d$: Embedding for n-gram $g$

### Skip-gram with Subwords

FastText uses the Skip-gram objective but with subword-augmented center word representations:

$$\mathcal{L} = -\sum_{(w, c) \in \mathcal{D}} \left[ \log \sigma(\mathbf{u}_c^\top \mathbf{v}_w) + \sum_{i=1}^{k} \mathbb{E}_{n \sim P_n} \log \sigma(-\mathbf{u}_n^\top \mathbf{v}_w) \right]$$

Where $\mathbf{v}_w = \sum_{g \in \mathcal{G}_w} \mathbf{z}_g$ is composed from n-gram embeddings, and the context embeddings $\mathbf{u}_c$ remain word-level (not decomposed into n-grams).

### Handling Large N-gram Vocabularies

The number of possible n-grams is vast. FastText uses **hashing** to manage this:

1. Hash each n-gram to a bucket: $h(g) \in \{1, \ldots, B\}$
2. Use $B$ embedding vectors (typically $B = 2{,}000{,}000$)
3. Different n-grams may share the same embedding (hash collision)

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

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

class FastTextModel(nn.Module):
    """
    FastText model with subword embeddings.
    
    Each word is represented as the sum of its character n-gram embeddings
    plus (optionally) a word-level embedding for known vocabulary items.
    """
    
    def __init__(self, vocab_size, num_buckets, embedding_dim, min_n=3, max_n=6):
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
        
        # Context embeddings (for Skip-gram prediction)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Initialize
        init_range = 0.5 / embedding_dim
        nn.init.uniform_(self.word_embeddings.weight, -init_range, init_range)
        nn.init.uniform_(self.ngram_embeddings.weight, -init_range, init_range)
        nn.init.uniform_(self.context_embeddings.weight, -init_range, init_range)
    
    def get_ngram_indices(self, word):
        """Get hashed indices for all n-grams of a word."""
        from fasttext_utils import extract_ngrams  # see Subword Embeddings section
        ngrams = extract_ngrams(word, self.min_n, self.max_n)
        indices = [hash_ngram(ng, self.num_buckets) for ng in ngrams]
        return torch.tensor(indices, dtype=torch.long)
    
    def get_word_embedding(self, word, word_idx=None):
        """
        Get embedding for a word (sum of n-gram embeddings).
        
        Args:
            word: The word string
            word_idx: Index in vocabulary (None for OOV words)
        
        Returns:
            Word embedding tensor of shape (embedding_dim,)
        """
        ngram_indices = self.get_ngram_indices(word)
        ngram_embs = self.ngram_embeddings(ngram_indices)
        
        # Sum n-gram embeddings
        word_emb = ngram_embs.sum(dim=0)
        
        # Add word-level embedding if word is in vocabulary
        if word_idx is not None:
            word_emb = word_emb + self.word_embeddings.weight[word_idx]
        
        return word_emb
    
    def forward(self, center_words, center_word_strs, context_words, negative_words):
        """
        Compute FastText loss with negative sampling.
        
        Args:
            center_words: (batch_size,) center word indices
            center_word_strs: List of center word strings (for n-gram extraction)
            context_words: (batch_size,) positive context word indices
            negative_words: (batch_size, num_neg) negative word indices
        """
        batch_size = center_words.size(0)
        
        # Get subword-augmented center word embeddings
        center_embs = []
        for word_idx, word_str in zip(center_words, center_word_strs):
            emb = self.get_word_embedding(word_str, word_idx.item())
            center_embs.append(emb)
        center_embs = torch.stack(center_embs)  # (batch, dim)
        
        # Get context embeddings
        context_embs = self.context_embeddings(context_words)    # (batch, dim)
        negative_embs = self.context_embeddings(negative_words)  # (batch, k, dim)
        
        # Positive score
        positive_score = torch.sum(center_embs * context_embs, dim=1)
        positive_loss = F.logsigmoid(positive_score)
        
        # Negative scores
        center_embs_expanded = center_embs.unsqueeze(2)           # (batch, dim, 1)
        negative_scores = torch.bmm(negative_embs, center_embs_expanded).squeeze(2)
        negative_loss = F.logsigmoid(-negative_scores).sum(dim=1)
        
        loss = -(positive_loss + negative_loss).mean()
        return loss
```

## Comparison with Word2Vec

| Aspect | Word2Vec | FastText |
|--------|----------|----------|
| **Word Unit** | Whole word | N-grams + word |
| **OOV Handling** | Cannot produce embeddings | Generates from n-grams |
| **Rare Words** | Poor quality | Better (shared n-grams) |
| **Morphology** | Not captured | Captured via subword sharing |
| **Vocabulary Size** | Fixed at training time | Flexible (any word) |
| **Training Speed** | Faster | Slower (more embeddings to update) |
| **Memory** | Lower ($|V| \times d$) | Higher ($|V| \times d + B \times d$) |

### When to Use FastText

**Use FastText when:**

- Working with morphologically rich languages (German, Finnish, Turkish)
- OOV words are common (social media text, technical domains)
- Rare words matter for the downstream task
- Typos and misspellings need to be handled gracefully

**Use Word2Vec when:**

- Memory is constrained
- Training speed is critical
- Working with languages with limited morphology
- All words of interest are in the training vocabulary

## Using Pre-trained FastText

Facebook provides pre-trained FastText embeddings for 157 languages:

```python
import gensim.downloader as api

# Download pre-trained FastText (300-dimensional)
# fasttext_model = api.load('fasttext-wiki-news-subwords-300')

# Alternative: Load from .bin file
from gensim.models import FastText

# model = FastText.load_fasttext_format('cc.en.300.bin')

# Get embedding — works for OOV words too
# embedding = model.wv['unforgettableness']
```

## Key Takeaways

!!! success "Main Concepts"

    1. **FastText** extends Skip-gram by representing words as sums of character n-gram embeddings
    2. **OOV handling** is automatic — any word can be embedded via its n-grams
    3. **Hashing** keeps the n-gram embedding table manageable ($B = 2M$ buckets)
    4. **Morphological relationships** are captured through shared n-grams between related words
    5. **Trade-off**: Better generalization and OOV coverage at the cost of more memory and slower training

!!! tip "Best Practices"

    - Use n-gram range 3–6 for most languages
    - Increase `min_n` for languages with longer average word length
    - Use pre-trained models when available (157 languages from Facebook)
    - Consider memory constraints: the n-gram hash table adds significant overhead

## References

- Bojanowski, P., et al. (2017). "Enriching Word Vectors with Subword Information"
- Mikolov, T., et al. (2018). "Advances in Pre-Training Distributed Word Representations"
- FastText Library: https://fasttext.cc/
