# One-Hot Encoding

## Overview

Traditional approaches to representing words in machine learning use **one-hot encoding**, where each word is a sparse vector with a single 1 and all other entries 0. Understanding why this representation fails — and how the distributional hypothesis motivates dense alternatives — provides essential context for all embedding methods that follow.

## Learning Objectives

By the end of this section, you will:

- Understand the mathematical definition and properties of one-hot encoding
- Identify the limitations of sparse symbolic representations
- Grasp the distributional hypothesis and its implications for learned representations
- Contrast one-hot encoding with distributed representations

## Mathematical Definition

For a vocabulary $V$ of size $|V|$, the one-hot representation of word $w_i$ is:

$$\mathbf{e}_i \in \{0, 1\}^{|V|}, \quad e_{i,j} = \begin{cases} 1 & \text{if } j = i \\ 0 & \text{otherwise} \end{cases}$$

Each word occupies a unique axis in a $|V|$-dimensional space, forming an orthonormal basis.

```python
import torch
import torch.nn as nn

# Small vocabulary for demonstration
vocabulary = ["cat", "dog", "bird", "fish", "lion"]
vocab_size = len(vocabulary)

# Create word-to-index mapping
word_to_ix = {word: i for i, word in enumerate(vocabulary)}
print(f"Vocabulary: {vocabulary}")
print(f"Word to index mapping: {word_to_ix}")

# One-hot encoding for "cat"
word = "cat"
word_idx = word_to_ix[word]

# Create one-hot vector
one_hot = torch.zeros(vocab_size)
one_hot[word_idx] = 1
print(f"\nOne-hot vector for '{word}': {one_hot}")
print(f"Vector size: {one_hot.shape[0]}")
print(f"Number of non-zero elements: {one_hot.count_nonzero().item()}")
```

**Output:**
```
Vocabulary: ['cat', 'dog', 'bird', 'fish', 'lion']
Word to index mapping: {'cat': 0, 'dog': 1, 'bird': 2, 'fish': 3, 'lion': 4}

One-hot vector for 'cat': tensor([1., 0., 0., 0., 0.])
Vector size: 5
Number of non-zero elements: 1
```

## Limitations of One-Hot Encoding

| Problem | Description |
|---------|-------------|
| **High Dimensionality** | Vector size equals vocabulary size (typically 50K–500K words) |
| **Extreme Sparsity** | Only one non-zero element per vector |
| **No Semantic Information** | All words are equidistant from each other |
| **No Generalization** | Cannot leverage word similarities for transfer learning |

### The Semantic Distance Problem

With one-hot encoding, the cosine similarity between any two different words is always 0:

$$\cos(\mathbf{e}_i, \mathbf{e}_j) = \frac{\mathbf{e}_i \cdot \mathbf{e}_j}{\|\mathbf{e}_i\| \|\mathbf{e}_j\|} = 0 \quad \text{for } i \neq j$$

This means "cat" is equally distant from "dog" (a fellow mammal) as it is from "algorithm" (completely unrelated). A model using one-hot inputs must learn all semantic relationships entirely from the task-specific training data, with no structural prior about word similarity.

```python
import torch.nn.functional as F

# One-hot vectors for three words
cat_oh = torch.tensor([1., 0., 0., 0., 0.])
dog_oh = torch.tensor([0., 1., 0., 0., 0.])
fish_oh = torch.tensor([0., 0., 0., 1., 0.])

# Cosine similarities are all zero
cos = nn.CosineSimilarity(dim=0)
print(f"cat ↔ dog:  {cos(cat_oh, dog_oh).item():.4f}")   # 0.0000
print(f"cat ↔ fish: {cos(cat_oh, fish_oh).item():.4f}")   # 0.0000
```

## The Distributional Hypothesis

The foundation of all learned word representations rests on the **distributional hypothesis**, attributed to linguist J.R. Firth (1957):

> *"You shall know a word by the company it keeps."*

This hypothesis states that words appearing in similar contexts tend to have similar meanings. If "dog" and "cat" frequently appear in similar contexts (e.g., "the ___ is sleeping", "feed the ___"), they should have similar representations.

The distributional hypothesis motivates a paradigm shift from hand-crafted symbolic representations to **distributed representations** — dense, low-dimensional vectors learned from data where semantic similarity is encoded geometrically.

## From One-Hot to Distributed Representations

A **distributed representation** maps each word to a dense, continuous vector in $\mathbb{R}^d$ where $d \ll |V|$:

$$f: V \rightarrow \mathbb{R}^d$$

**Key Properties:**

1. **Dense**: All dimensions carry non-zero information
2. **Low-dimensional**: Typically $d \in [50, 300]$, vs. $|V| \approx 100{,}000$
3. **Learned**: Parameters optimized during training
4. **Meaningful**: Semantic relationships encoded as geometric proximity

### Comparison

| Aspect | One-Hot | Distributed |
|--------|---------|-------------|
| **Dimensionality** | $|V|$ (e.g., 100,000) | $d$ (e.g., 300) |
| **Sparsity** | Extremely sparse (single 1) | Dense (all non-zero) |
| **Similarity** | Always 0 for different words | Reflects semantic similarity |
| **Memory** | $O(|V|)$ per word | $O(d)$ per word |
| **Generalization** | None | Leverages word relationships |
| **Learned** | Fixed (deterministic) | Trained from data |

### PyTorch: One-Hot vs. Embedding Lookup

```python
import torch
import torch.nn as nn

vocab_size = 5
embedding_dim = 3

# Method 1: One-hot + linear projection (mathematically equivalent to embedding)
one_hot = torch.zeros(vocab_size)
one_hot[0] = 1  # "cat"
linear = nn.Linear(vocab_size, embedding_dim, bias=False)
projected = linear(one_hot)

# Method 2: Embedding lookup (efficient implementation)
embedding = nn.Embedding(vocab_size, embedding_dim)
looked_up = embedding(torch.tensor([0]))

print(f"One-hot + Linear shape: {projected.shape}")    # (3,)
print(f"Embedding lookup shape: {looked_up.shape}")     # (1, 3)

# Both are equivalent: an embedding lookup is a matrix multiplication
# with a one-hot vector, but avoids the sparse multiplication overhead
```

!!! info "Embedding as Efficient One-Hot Projection"

    An embedding lookup `nn.Embedding(V, d)` is mathematically equivalent to multiplying a one-hot vector by a weight matrix $\mathbf{W} \in \mathbb{R}^{|V| \times d}$. The embedding layer simply skips the redundant multiplication and directly indexes the row, making it $O(d)$ instead of $O(|V| \cdot d)$.

## Initialization of Embeddings

Before training, embedding weights are initialized from a random distribution. The choice of initialization affects convergence:

```python
# PyTorch default: Uniform from standard range
embedding = nn.Embedding(vocab_size, embedding_dim)

# Normal initialization (common alternative)
embedding_normal = nn.Embedding(vocab_size, embedding_dim)
nn.init.normal_(embedding_normal.weight, mean=0, std=0.1)

# Xavier/Glorot initialization
embedding_xavier = nn.Embedding(vocab_size, embedding_dim)
nn.init.xavier_uniform_(embedding_xavier.weight)
```

## Key Takeaways

!!! success "Main Concepts"

    1. **One-hot encoding** represents words as sparse, orthogonal vectors with no semantic information
    2. **The distributional hypothesis** states that words in similar contexts have similar meanings
    3. **Distributed representations** encode semantics through dense, learned vectors in low-dimensional space
    4. **Embedding layers** are efficient implementations of the one-hot-to-dense projection
    5. **Cosine similarity** between one-hot vectors is always 0, motivating learned representations

## References

- Firth, J. R. (1957). "A Synopsis of Linguistic Theory, 1930–1955"
- Rumelhart, D. E., et al. (1986). "Learning representations by back-propagating errors"
- Bengio, Y., et al. (2003). "A Neural Probabilistic Language Model"
