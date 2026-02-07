# Word Embeddings Overview

## Introduction

Word embeddings are dense, low-dimensional vector representations that encode words as continuous vectors in $\mathbb{R}^d$, where semantic relationships between words are captured through geometric properties of the embedding space. This chapter provides comprehensive coverage of word embedding techniques, from foundational one-hot encoding through classical prediction-based methods to modern contextualized approaches.

The central idea is elegantly simple: map each word $w$ in a vocabulary $V$ to a vector $\mathbf{e}_w \in \mathbb{R}^d$ such that semantically related words occupy nearby regions of the vector space. The result is a representational framework where arithmetic on vectors corresponds to linguistic relationships:

$$\mathbf{v}_{\text{king}} - \mathbf{v}_{\text{man}} + \mathbf{v}_{\text{woman}} \approx \mathbf{v}_{\text{queen}}$$

## Learning Path

The sections in this chapter are organized into three parts that build progressively from foundational concepts through core algorithms to modern methods.

**Part I — Foundations:**

1. **One-Hot Encoding** — Why sparse, symbolic representations fail and how the distributional hypothesis motivates dense embeddings
2. **Word2Vec** — The neural prediction framework that revolutionized NLP
3. **Skip-gram Model** — Predicting context from a center word; better for rare words
4. **CBOW Model** — Predicting center word from context; faster training on large corpora
5. **Negative Sampling** — Efficient training via binary classification approximation

**Part II — Static Embedding Methods:**

6. **GloVe** — Combining global co-occurrence statistics with local context windows
7. **FastText** — Extending Word2Vec with character n-gram composition for morphology
8. **Subword Embeddings** — Handling OOV words, morphologically rich languages, and rare tokens through subword decomposition

**Part III — Modern Methods:**

9. **Contextual Embeddings** — From static to dynamic representations with ELMo, BERT, GPT, and the transformer architecture
10. **Embedding Visualization** — Dimensionality reduction techniques (PCA, t-SNE, UMAP) for understanding and evaluating embedding spaces

## The Embedding Matrix

All embedding methods ultimately produce an **embedding matrix** $\mathbf{E} \in \mathbb{R}^{|V| \times d}$:

$$\mathbf{E} = \begin{bmatrix} \mathbf{e}_1^T \\ \mathbf{e}_2^T \\ \vdots \\ \mathbf{e}_{|V|}^T \end{bmatrix}$$

where $\mathbf{e}_i \in \mathbb{R}^d$ is the embedding for word $i$. Given a word index $i$, the embedding is retrieved as the $i$-th row:

$$\text{Embed}(i) = \mathbf{E}[i, :] = \mathbf{e}_i$$

In PyTorch, this lookup is implemented via `nn.Embedding`:

```python
import torch
import torch.nn as nn

vocab_size = 10000
embedding_dim = 300

# Learnable lookup table: maps integer indices to dense vectors
embedding = nn.Embedding(vocab_size, embedding_dim)

# Look up embeddings for a batch of word indices
word_indices = torch.tensor([42, 7, 1024])
vectors = embedding(word_indices)  # shape: (3, 300)
```

## Measuring Similarity

The most common metric for comparing word embeddings is **cosine similarity**:

$$\text{sim}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|} = \frac{\sum_{i=1}^{d} u_i v_i}{\sqrt{\sum_{i=1}^{d} u_i^2} \sqrt{\sum_{i=1}^{d} v_i^2}}$$

| Value | Interpretation |
|-------|---------------|
| $\approx 1$ | Vectors point in the same direction (semantically similar) |
| $\approx 0$ | Vectors are orthogonal (unrelated) |
| $\approx -1$ | Vectors point in opposite directions (dissimilar) |

An alternative is **Euclidean distance** $d(\mathbf{u}, \mathbf{v}) = \|\mathbf{u} - \mathbf{v}\|_2$, though cosine similarity is generally preferred because it is invariant to vector magnitude.

## Taxonomy of Embedding Methods

| Method | Type | Core Idea | Strengths |
|--------|------|-----------|-----------|
| **One-Hot** | Sparse | Indicator vector per word | Simple, deterministic |
| **Word2Vec** | Prediction-based | Context prediction via neural network | Good semantic capture |
| **GloVe** | Count-based | Weighted log co-occurrence regression | Global statistics + local context |
| **FastText** | Subword | Character n-gram composition | OOV handling, morphology |
| **ELMo** | Contextualized | Bidirectional LSTM language model | Context-sensitive, layer combination |
| **BERT** | Contextualized | Masked language model + transformer | Deep bidirectional context |
| **GPT** | Contextualized | Autoregressive transformer | Strong generation capabilities |

## Choosing Embedding Dimension

The embedding dimension $d$ balances expressiveness and efficiency:

| Dataset Size | Recommended $d$ | Rationale |
|--------------|-----------------|-----------|
| Small (<1M tokens) | 50–100 | Prevent overfitting |
| Medium (1M–100M) | 100–200 | Balance capacity and generalization |
| Large (>100M) | 200–300 | Capture nuanced relationships |

## Information Compression

Distributed representations achieve efficient compression of word relationships:

| Representation | Parameters per Word | Relationships Encoded |
|---------------|--------------------|-----------------------|
| One-hot | $|V|$ | None |
| Co-occurrence matrix | $|V|$ (one row) | Explicit co-occurrences |
| Embeddings | $d$ | Implicit, generalized |

For $|V| = 100{,}000$ and $d = 300$: one-hot vectors require 100,000 dimensions per word, while embeddings need only 300 dimensions — a 333× reduction that simultaneously encodes richer semantic information.

## Dependencies

Code examples throughout this chapter require:

- PyTorch
- NumPy
- scikit-learn (for PCA, t-SNE)
- SciPy (for SVD)
- Matplotlib (for visualization)
- Hugging Face Transformers (for BERT, GPT examples in the contextual embeddings section)

## References

- Bengio, Y., et al. (2003). "A Neural Probabilistic Language Model"
- Mikolov, T., et al. (2013). "Efficient Estimation of Word Representations in Vector Space"
- Pennington, J., Socher, R., & Manning, C. D. (2014). "GloVe: Global Vectors for Word Representation"
- Bojanowski, P., et al. (2017). "Enriching Word Vectors with Subword Information"
- Vaswani, A., et al. (2017). "Attention Is All You Need"
- Peters, M. E., et al. (2018). "Deep Contextualized Word Representations"
- Devlin, J., et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
