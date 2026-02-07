# Word2Vec

## Overview

Word2Vec, introduced by Mikolov et al. (2013), is a family of models for learning word embeddings from large text corpora. The two main architectures are **Skip-gram** (predicting context from a center word) and **CBOW** (Continuous Bag of Words, predicting a center word from context). These models revolutionized NLP by producing embeddings that capture semantic relationships through simple, efficient training objectives.

## Learning Objectives

By the end of this section, you will:

- Understand the prediction-based approach to learning word embeddings
- Define context windows and their role in both architectures
- Compare the strengths and trade-offs of Skip-gram vs. CBOW
- Understand the computational bottleneck of the full softmax objective

## The Distributional Hypothesis Revisited

Word2Vec operationalizes the distributional hypothesis through prediction tasks:

> Words that appear in similar contexts should have similar representations.

**Key Insight:** Rather than explicitly counting co-occurrences (as in LSA or GloVe), Word2Vec learns embeddings by predicting words from their contexts (or vice versa). The prediction task forces the model to encode contextual relationships into the embedding vectors.

## Context Windows

Both architectures use a **context window** of size $k$ to define which words are related. For a sequence $w_1, w_2, \ldots, w_T$ and center word $w_t$, the context consists of words within $k$ positions:

$$\text{Context}(w_t) = \{w_{t-k}, \ldots, w_{t-1}, w_{t+1}, \ldots, w_{t+k}\}$$

**Example with $k=2$:**

```
Sentence: "The quick brown fox jumps over the lazy dog"
Center word: "fox" (position 4)
Context window: ["quick", "brown", "jumps", "over"]
```

The choice of $k$ controls what type of relationships the embeddings capture:

| Window Size | Captures | Example |
|-------------|----------|---------|
| Small ($k = 2\text{–}3$) | Syntactic relationships | Verb forms, POS patterns |
| Large ($k = 5\text{–}10$) | Semantic relationships | Topical similarity |

## Two Embedding Matrices

A critical architectural detail: Word2Vec uses **two separate embedding matrices** — one for center words ($\mathbf{V} \in \mathbb{R}^{|V| \times d}$) and one for context words ($\mathbf{U} \in \mathbb{R}^{|V| \times d}$):

- $\mathbf{v}_{w} \in \mathbb{R}^d$: Embedding of word $w$ when it appears as a **center** word
- $\mathbf{u}_{w} \in \mathbb{R}^d$: Embedding of word $w$ when it appears as a **context** word

After training, the standard practice is to use either the center embeddings alone or the average $(\mathbf{v}_w + \mathbf{u}_w) / 2$ as the final word representation.

## Skip-gram vs. CBOW at a Glance

The two architectures are mirror images of each other:

```
Skip-gram:                          CBOW:
Input (center word)                 Input (context words)
    "fox"                           ["quick", "brown", "jumps", "over"]
      ↓                                   ↓
  embedding                         embeddings → average
      ↓                                   ↓
  linear → softmax                  linear → softmax
      ↓                                   ↓
Output (context words)              Output (center word)
["quick", "brown", "jumps", "over"]     "fox"
```

| Aspect | Skip-gram | CBOW |
|--------|-----------|------|
| **Prediction Task** | Context from center word | Center word from context |
| **Training Examples** | One per (center, context) pair | One per center word |
| **Speed** | Slower (more examples per position) | Faster (fewer examples) |
| **Rare Words** | Better performance | Worse performance |
| **Frequent Words** | Good performance | Better performance |
| **Dataset Size** | Better for smaller datasets | Better for larger datasets |
| **Memory** | Higher (more gradient updates) | Lower (batched context) |

### When to Use Each

**Use Skip-gram when:**

- Working with smaller datasets
- Rare words are important for the downstream task
- Embedding quality is more important than training speed

**Use CBOW when:**

- Working with large datasets
- Training speed is critical
- Frequent words are more important for the task

## The Softmax Bottleneck

Both models compute a softmax over the full vocabulary to obtain word probabilities:

$$P(w_o | w_c) = \frac{\exp(\mathbf{u}_{w_o}^\top \mathbf{v}_{w_c})}{\sum_{w \in V} \exp(\mathbf{u}_w^\top \mathbf{v}_{w_c})}$$

The denominator requires computing dot products with **all** $|V|$ words:

| Vocabulary Size | Operations per Update |
|----------------|----------------------|
| 10,000 | 10,000 |
| 100,000 | 100,000 |
| 1,000,000 | 1,000,000 |

For a corpus of billions of words, this $O(|V|)$ computation per training example is prohibitively expensive.

**Solutions** (covered in subsequent sections):

1. **Negative Sampling:** Approximate the full softmax with binary classification over a small number of sampled "negative" examples
2. **Hierarchical Softmax:** Organize the vocabulary as a binary tree, reducing complexity to $O(\log |V|)$

## Complete Training Example

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import Counter

# Sample text (Shakespeare sonnet excerpt)
text = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise."""

# Tokenize
tokens = text.lower().split()
print(f"Total tokens: {len(tokens)}")
print(f"Sample tokens: {tokens[:10]}")

# Build vocabulary
word_counts = Counter(tokens)
word_to_ix = {word: idx for idx, (word, _) in enumerate(word_counts.most_common())}
ix_to_word = {idx: word for word, idx in word_to_ix.items()}
vocab_size = len(word_to_ix)

print(f"Vocabulary size: {vocab_size}")
```

### Finding Similar Words

After training either model, we use cosine similarity to find semantically related words:

```python
def find_similar_words(embeddings_weight, word, word_to_ix, ix_to_word, top_k=5):
    """Find most similar words using cosine similarity."""
    if word not in word_to_ix:
        return []
    
    word_idx = word_to_ix[word]
    word_embedding = embeddings_weight[word_idx]
    
    # Compute similarities with all words
    similarities = F.cosine_similarity(
        word_embedding.unsqueeze(0),
        embeddings_weight,
        dim=1
    )
    
    # Exclude the word itself
    similarities[word_idx] = -1
    
    # Get top-k
    top_sim, top_indices = torch.topk(similarities, k=top_k)
    
    results = []
    for sim, idx in zip(top_sim, top_indices):
        results.append((ix_to_word[idx.item()], sim.item()))
    
    return results
```

## Gradient Intuition

### Skip-gram Gradient

For the softmax loss with center word $c$ and context word $o$:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{v}_c} = -\mathbf{u}_o + \sum_{w=1}^{|V|} P(w|c) \cdot \mathbf{u}_w$$

**Interpretation:** The gradient pushes the center embedding toward the true context word ($-\mathbf{u}_o$) and away from the expected (probability-weighted average) context words.

### CBOW Gradient

For averaged context embedding $\hat{\mathbf{v}}$ and center word $t$:

$$\frac{\partial \mathcal{L}}{\partial \hat{\mathbf{v}}} = -\mathbf{u}_t + \sum_{w=1}^{|V|} P(w|\text{context}) \cdot \mathbf{u}_w$$

The gradient is distributed equally to all context word embeddings:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{v}_{c_j}} = \frac{1}{2k} \frac{\partial \mathcal{L}}{\partial \hat{\mathbf{v}}}$$

## Key Takeaways

!!! success "Main Concepts"

    1. **Word2Vec** learns embeddings through prediction tasks rather than counting
    2. **Two architectures**: Skip-gram (center → context) and CBOW (context → center)
    3. Both models use **two embedding matrices** (center and context)
    4. **Context window size** controls syntactic vs. semantic emphasis
    5. The **softmax bottleneck** makes naive training infeasible for large vocabularies

!!! warning "Common Pitfalls"

    - Forgetting to shuffle training data between epochs
    - Using embedding dimensions that are too large for the dataset size
    - Not normalizing embeddings before similarity computation
    - Ignoring the softmax bottleneck for large vocabularies

## References

- Mikolov, T., et al. (2013). "Efficient Estimation of Word Representations in Vector Space"
- Mikolov, T., et al. (2013). "Distributed Representations of Words and Phrases and their Compositionality"
- Goldberg, Y., & Levy, O. (2014). "word2vec Explained"
