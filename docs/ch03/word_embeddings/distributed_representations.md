# Distributed Representations

## Overview

Distributed representations are dense, low-dimensional vector representations of discrete objects (such as words) where semantic properties are encoded across multiple dimensions rather than in single, isolated features. This foundational concept underlies all modern word embedding techniques and represents a paradigm shift from sparse, symbolic representations to continuous vector spaces.

## Learning Objectives

By the end of this section, you will:

- Understand the limitations of one-hot encoding for representing words
- Grasp the concept of distributed representations and their advantages
- Learn how semantic relationships emerge from distributional properties
- Implement basic embedding lookups in PyTorch
- Understand the role of embedding dimensions in capturing word semantics

## From Symbolic to Distributed Representations

### The Problem with One-Hot Encoding

Traditional approaches to representing words in machine learning used **one-hot encoding**, where each word is represented as a sparse vector with a single 1 and all other entries as 0.

**Mathematical Definition:**

For a vocabulary $V$ of size $|V|$, the one-hot representation of word $w_i$ is:

$$\mathbf{e}_i \in \{0, 1\}^{|V|}, \quad e_{i,j} = \begin{cases} 1 & \text{if } j = i \\ 0 & \text{otherwise} \end{cases}$$

**Example with a small vocabulary:**

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

### Limitations of One-Hot Encoding

| Problem | Description |
|---------|-------------|
| **High Dimensionality** | Vector size equals vocabulary size (typically 50K-500K words) |
| **Extreme Sparsity** | Only one non-zero element per vector |
| **No Semantic Information** | All words are equidistant from each other |
| **No Generalization** | Cannot leverage word similarities |

**The Semantic Distance Problem:**

With one-hot encoding, the cosine similarity between any two different words is always 0:

$$\cos(\mathbf{e}_i, \mathbf{e}_j) = \frac{\mathbf{e}_i \cdot \mathbf{e}_j}{\|\mathbf{e}_i\| \|\mathbf{e}_j\|} = 0 \quad \text{for } i \neq j$$

This means "cat" is equally distant from "dog" (a fellow mammal) as it is from "algorithm" (completely unrelated).

### The Distributional Hypothesis

The foundation of distributed representations rests on the **distributional hypothesis**, attributed to linguist J.R. Firth (1957):

> *"You shall know a word by the company it keeps."*

This hypothesis states that words appearing in similar contexts tend to have similar meanings. If "dog" and "cat" frequently appear in similar contexts (e.g., "the ___ is sleeping", "feed the ___"), they should have similar representations.

## Distributed Representations

### Definition and Properties

A **distributed representation** maps each word to a dense, continuous vector in $\mathbb{R}^d$ where $d \ll |V|$:

$$f: V \rightarrow \mathbb{R}^d$$

**Key Properties:**

1. **Dense**: All dimensions have non-zero values
2. **Low-dimensional**: Typically $d \in [50, 300]$
3. **Learned**: Parameters optimized during training
4. **Meaningful**: Semantic relationships encoded geometrically

### The Embedding Matrix

Word embeddings are stored in an **embedding matrix** $\mathbf{E} \in \mathbb{R}^{|V| \times d}$:

$$\mathbf{E} = \begin{bmatrix} \mathbf{e}_1^T \\ \mathbf{e}_2^T \\ \vdots \\ \mathbf{e}_{|V|}^T \end{bmatrix}$$

where $\mathbf{e}_i \in \mathbb{R}^d$ is the embedding for word $i$.

**Embedding Lookup:**

Given a word index $i$, the embedding is retrieved as the $i$-th row of $\mathbf{E}$:

$$\text{Embed}(i) = \mathbf{E}[i, :] = \mathbf{e}_i$$

### PyTorch Implementation

```python
import torch
import torch.nn as nn

# Configuration
vocab_size = 5
embedding_dim = 3  # Small for visualization; typically 50-300

# Create embedding layer (a learnable lookup table)
embeddings = nn.Embedding(vocab_size, embedding_dim)

print("Embedding Layer Configuration:")
print(f"  Vocabulary size: {vocab_size}")
print(f"  Embedding dimension: {embedding_dim}")
print(f"  Total parameters: {vocab_size * embedding_dim}")

print(f"\nEmbedding matrix shape: {embeddings.weight.shape}")
print(f"Embedding matrix:\n{embeddings.weight}")
```

**Output:**
```
Embedding Layer Configuration:
  Vocabulary size: 5
  Embedding dimension: 3
  Total parameters: 15

Embedding matrix shape: torch.Size([5, 3])
Embedding matrix:
tensor([[-0.2846,  0.9728, -0.4561],
        [ 1.2405,  0.3189, -0.8912],
        [-0.5431,  0.1288,  0.7643],
        [ 0.4521, -0.6789,  0.2341],
        [-0.1234,  0.5678, -0.9012]], requires_grad=True)
```

### Embedding Lookup Operations

```python
# Look up embedding for a single word
word_to_ix = {"cat": 0, "dog": 1, "bird": 2, "fish": 3, "lion": 4}

word = "cat"
word_idx = word_to_ix[word]
word_tensor = torch.tensor([word_idx], dtype=torch.long)

# Retrieve embedding
cat_embedding = embeddings(word_tensor)

print(f"Word: '{word}'")
print(f"Index: {word_idx}")
print(f"Embedding vector: {cat_embedding}")
print(f"Shape: {cat_embedding.shape}")  # (1, 3)

# Batch lookup for multiple words
words = ["cat", "dog", "lion"]
word_indices = [word_to_ix[w] for w in words]
word_tensor = torch.tensor(word_indices, dtype=torch.long)

batch_embeddings = embeddings(word_tensor)

print(f"\nBatch lookup for {words}:")
print(f"Shape: {batch_embeddings.shape}")  # (3, 3)
for i, word in enumerate(words):
    print(f"  {word}: {batch_embeddings[i].detach().numpy()}")
```

## Measuring Similarity in Embedding Space

### Cosine Similarity

The most common metric for comparing word embeddings is **cosine similarity**:

$$\text{sim}(\mathbf{u}, \mathbf{v}) = \cos(\theta) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|} = \frac{\sum_{i=1}^{d} u_i v_i}{\sqrt{\sum_{i=1}^{d} u_i^2} \sqrt{\sum_{i=1}^{d} v_i^2}}$$

**Properties:**
- Range: $[-1, 1]$
- 1: Vectors point in the same direction (similar)
- 0: Vectors are orthogonal (unrelated)
- -1: Vectors point in opposite directions (dissimilar)

```python
import torch.nn.functional as F

# Get embeddings for comparison
cat_emb = embeddings.weight[word_to_ix["cat"]]
dog_emb = embeddings.weight[word_to_ix["dog"]]
fish_emb = embeddings.weight[word_to_ix["fish"]]

# Compute cosine similarity
cos = nn.CosineSimilarity(dim=0)

cat_dog_sim = cos(cat_emb, dog_emb)
cat_fish_sim = cos(cat_emb, fish_emb)

print("Cosine Similarity (random initialization):")
print(f"  cat ↔ dog:  {cat_dog_sim.item():.4f}")
print(f"  cat ↔ fish: {cat_fish_sim.item():.4f}")

print("\nNote: These are random! After training, we expect:")
print("  - cat ↔ dog to be higher (both mammals)")
print("  - cat ↔ fish to be lower (different categories)")
```

### Euclidean Distance

An alternative metric is **Euclidean distance**:

$$d(\mathbf{u}, \mathbf{v}) = \|\mathbf{u} - \mathbf{v}\|_2 = \sqrt{\sum_{i=1}^{d} (u_i - v_i)^2}$$

```python
# Euclidean distance
cat_dog_dist = torch.norm(cat_emb - dog_emb)
cat_fish_dist = torch.norm(cat_emb - fish_emb)

print("Euclidean Distance:")
print(f"  cat ↔ dog:  {cat_dog_dist.item():.4f}")
print(f"  cat ↔ fish: {cat_fish_dist.item():.4f}")
```

## Why Distributed Representations Work

### Geometric Interpretation

Distributed representations capture semantic relationships through **geometric properties** of the embedding space:

1. **Similarity as Proximity**: Similar words cluster together
2. **Analogies as Parallelograms**: Relationships form consistent directions
3. **Linear Substructures**: Semantic categories form linear patterns

**The Famous Word Analogy:**

$$\text{king} - \text{man} + \text{woman} \approx \text{queen}$$

This works because the relationship "male → female" is encoded as a consistent direction in the embedding space.

### Information Compression

Distributed representations achieve **efficient compression** of word relationships:

| Representation | Parameters | Relationships Encoded |
|---------------|------------|----------------------|
| One-hot | $|V|$ per word | None |
| Co-occurrence matrix | $|V|^2$ | Explicit co-occurrences |
| Embeddings | $d$ per word | Implicit, generalized |

For $|V| = 100,000$ and $d = 300$:
- One-hot: 100,000 dimensions per word
- Co-occurrence: 10 billion parameters total
- Embeddings: 300 dimensions per word, 30 million total

## Comparison: One-Hot vs. Distributed

| Aspect | One-Hot | Distributed |
|--------|---------|-------------|
| **Dimensionality** | $|V|$ (e.g., 100,000) | $d$ (e.g., 300) |
| **Sparsity** | Extremely sparse (one 1) | Dense (all non-zero) |
| **Similarity** | Always 0 for different words | Reflects semantic similarity |
| **Memory** | $O(|V|)$ per word | $O(d)$ per word |
| **Generalization** | None | Leverages word relationships |
| **Learned** | Fixed (deterministic) | Trained from data |

## Practical Considerations

### Choosing Embedding Dimension

The embedding dimension $d$ balances expressiveness and efficiency:

| Dataset Size | Recommended $d$ | Rationale |
|--------------|-----------------|-----------|
| Small (<1M tokens) | 50-100 | Prevent overfitting |
| Medium (1M-100M) | 100-200 | Balance capacity/generalization |
| Large (>100M) | 200-300 | Capture nuanced relationships |

### Initialization

Embedding weights are typically initialized from a standard distribution:

```python
# PyTorch default: Uniform from [-sqrt(3), sqrt(3)]
nn.Embedding(vocab_size, embedding_dim)

# Alternative: Normal initialization
embedding = nn.Embedding(vocab_size, embedding_dim)
nn.init.normal_(embedding.weight, mean=0, std=0.1)

# Xavier/Glorot initialization
nn.init.xavier_uniform_(embedding.weight)
```

## Key Takeaways

!!! success "Main Concepts"

    1. **Distributed representations** encode words as dense, low-dimensional vectors
    2. **Semantic similarity** is captured through geometric proximity
    3. **Embedding layers** are learnable lookup tables that map indices to vectors
    4. **Cosine similarity** is the standard metric for comparing embeddings
    5. **Training** adjusts embeddings so similar words have similar vectors

!!! tip "Best Practices"

    - Use embedding dimensions between 50-300 depending on vocabulary and task
    - Initialize embeddings randomly before training
    - Consider using pre-trained embeddings for small datasets
    - Normalize embeddings when computing similarity scores

## Next Steps

In the following sections, we will explore specific algorithms for learning word embeddings:

1. **Word2Vec** (Skip-gram and CBOW)
2. **Negative Sampling** for efficient training
3. **GloVe** for global vector representations
4. **FastText** for subword information
5. **PyTorch Embedding Layers** for practical implementation

## References

- Bengio, Y., et al. (2003). "A Neural Probabilistic Language Model"
- Mikolov, T., et al. (2013). "Efficient Estimation of Word Representations in Vector Space"
- Rumelhart, D. E., et al. (1986). "Learning representations by back-propagating errors"
