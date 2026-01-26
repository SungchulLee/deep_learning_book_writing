# GloVe: Global Vectors for Word Representation

## Overview

GloVe (Global Vectors for Word Representation), introduced by Pennington et al. (2014), is a word embedding method that combines the benefits of global matrix factorization techniques (like LSA) with local context window methods (like Word2Vec). By directly modeling word co-occurrence statistics, GloVe produces embeddings that capture both syntactic and semantic relationships with a more interpretable objective function.

## Learning Objectives

By the end of this section, you will:

- Understand the co-occurrence matrix and its properties
- Derive the GloVe objective function from first principles
- Implement GloVe training in PyTorch
- Compare GloVe with Word2Vec approaches
- Apply appropriate weighting schemes for rare and frequent words

## Motivation: Bridging Global and Local Methods

### Global Matrix Factorization (LSA)

Latent Semantic Analysis factorizes the term-document matrix using SVD:

$$\mathbf{X} \approx \mathbf{U} \mathbf{\Sigma} \mathbf{V}^\top$$

**Advantages:** Captures global statistics
**Disadvantages:** Poor on word analogy tasks, dominated by frequent words

### Local Context Window (Word2Vec)

Word2Vec learns from local context windows through prediction tasks.

**Advantages:** Good on analogy tasks, captures syntactic patterns
**Disadvantages:** Doesn't directly exploit global co-occurrence statistics

### GloVe: The Best of Both Worlds

GloVe directly models the **co-occurrence matrix** while optimizing for local context relationships.

## Co-occurrence Statistics

### The Co-occurrence Matrix

Given a corpus, the **co-occurrence matrix** $\mathbf{X}$ counts how often words appear together within a context window:

$$X_{ij} = \text{count of word } j \text{ appearing in context of word } i$$

**Example:**

For the sentence "the cat sat on the mat" with window size 1:

|       | the | cat | sat | on | mat |
|-------|-----|-----|-----|----|-----|
| the   | 0   | 1   | 0   | 1  | 1   |
| cat   | 1   | 0   | 1   | 0  | 0   |
| sat   | 0   | 1   | 0   | 1  | 0   |
| on    | 1   | 0   | 1   | 0  | 1   |
| mat   | 1   | 0   | 0   | 1  | 0   |

### Co-occurrence Probabilities

Define the probability of word $j$ appearing in the context of word $i$:

$$P_{ij} = P(j|i) = \frac{X_{ij}}{X_i}$$

where $X_i = \sum_k X_{ik}$ is the total count of contexts for word $i$.

## The GloVe Insight

### Ratio of Co-occurrence Probabilities

The key insight of GloVe is that **ratios of co-occurrence probabilities** encode semantic relationships.

**Example:** Consider words "ice", "steam", "water", and "fashion":

| Probe word $k$ | $P(k|\text{ice})$ | $P(k|\text{steam})$ | $\frac{P(k|\text{ice})}{P(k|\text{steam})}$ |
|----------------|-------------------|---------------------|---------------------------------------------|
| solid | $1.9 \times 10^{-4}$ | $2.2 \times 10^{-5}$ | 8.9 |
| gas | $6.6 \times 10^{-5}$ | $7.8 \times 10^{-4}$ | 0.085 |
| water | $3.0 \times 10^{-3}$ | $2.2 \times 10^{-3}$ | 1.36 |
| fashion | $1.7 \times 10^{-5}$ | $1.8 \times 10^{-5}$ | 0.96 |

**Interpretation:**
- Ratio >> 1: Word $k$ is more associated with "ice"
- Ratio << 1: Word $k$ is more associated with "steam"
- Ratio ≈ 1: Word $k$ is equally related (or unrelated) to both

### The Goal

We want embeddings $\mathbf{w}_i$ and $\mathbf{w}_j$ such that their dot product relates to the log co-occurrence:

$$\mathbf{w}_i^\top \tilde{\mathbf{w}}_j \approx \log X_{ij}$$

## Mathematical Derivation

### Starting Point

We want a function $F$ that captures the ratio relationship:

$$F(\mathbf{w}_i, \mathbf{w}_j, \tilde{\mathbf{w}}_k) = \frac{P_{ik}}{P_{jk}}$$

### Symmetry Requirements

Words and contexts should be interchangeable, so $F$ should be symmetric. This leads to:

$$F((\mathbf{w}_i - \mathbf{w}_j)^\top \tilde{\mathbf{w}}_k) = \frac{P_{ik}}{P_{jk}}$$

### Exponential Form

Using $F = \exp$:

$$\exp((\mathbf{w}_i - \mathbf{w}_j)^\top \tilde{\mathbf{w}}_k) = \frac{P_{ik}}{P_{jk}}$$

$$\exp(\mathbf{w}_i^\top \tilde{\mathbf{w}}_k - \mathbf{w}_j^\top \tilde{\mathbf{w}}_k) = \frac{P_{ik}}{P_{jk}}$$

This implies:

$$\mathbf{w}_i^\top \tilde{\mathbf{w}}_k = \log P_{ik} = \log X_{ik} - \log X_i$$

### Adding Bias Terms

To handle the asymmetry from $\log X_i$, we add bias terms:

$$\mathbf{w}_i^\top \tilde{\mathbf{w}}_j + b_i + \tilde{b}_j = \log X_{ij}$$

## GloVe Objective Function

### Weighted Least Squares

The GloVe objective minimizes:

$$J = \sum_{i,j=1}^{|V|} f(X_{ij}) \left( \mathbf{w}_i^\top \tilde{\mathbf{w}}_j + b_i + \tilde{b}_j - \log X_{ij} \right)^2$$

### Weighting Function

The weighting function $f(x)$ handles the imbalance between rare and frequent co-occurrences:

$$f(x) = \begin{cases} 
(x/x_{\max})^\alpha & \text{if } x < x_{\max} \\
1 & \text{otherwise}
\end{cases}$$

**Typical values:** $x_{\max} = 100$, $\alpha = 0.75$

**Properties of $f(x)$:**
1. $f(0) = 0$: Zero co-occurrences contribute nothing
2. $f(x)$ is non-decreasing: More frequent pairs get more weight
3. $f(x)$ saturates at 1: Very frequent pairs don't dominate

```python
import numpy as np
import matplotlib.pyplot as plt

def glove_weight(x, x_max=100, alpha=0.75):
    """GloVe weighting function."""
    return np.where(x < x_max, (x / x_max) ** alpha, 1.0)

# Visualize
x = np.linspace(0, 150, 1000)
weights = glove_weight(x)

plt.figure(figsize=(10, 4))
plt.plot(x, weights, linewidth=2)
plt.axhline(y=1, color='r', linestyle='--', alpha=0.5)
plt.axvline(x=100, color='g', linestyle='--', alpha=0.5)
plt.xlabel('Co-occurrence count $X_{ij}$')
plt.ylabel('Weight $f(X_{ij})$')
plt.title('GloVe Weighting Function')
plt.grid(True, alpha=0.3)
plt.show()
```

## PyTorch Implementation

### Building the Co-occurrence Matrix

```python
import torch
import torch.nn as nn
import numpy as np
from collections import Counter, defaultdict
from scipy.sparse import lil_matrix, csr_matrix

def build_cooccurrence_matrix(tokens, vocab, window_size=5, distance_weighting=True):
    """
    Build word co-occurrence matrix from tokens.
    
    Args:
        tokens: List of tokens
        vocab: Dictionary mapping words to indices
        window_size: Context window size on each side
        distance_weighting: Weight by 1/distance (standard in GloVe)
    
    Returns:
        Sparse co-occurrence matrix
    """
    vocab_size = len(vocab)
    cooc = lil_matrix((vocab_size, vocab_size), dtype=np.float64)
    
    for i, center_word in enumerate(tokens):
        if center_word not in vocab:
            continue
        center_idx = vocab[center_word]
        
        # Look at context words
        start = max(0, i - window_size)
        end = min(len(tokens), i + window_size + 1)
        
        for j in range(start, end):
            if j == i:
                continue
            
            context_word = tokens[j]
            if context_word not in vocab:
                continue
            context_idx = vocab[context_word]
            
            # Weight by distance (closer words get higher weight)
            distance = abs(j - i)
            if distance_weighting:
                weight = 1.0 / distance
            else:
                weight = 1.0
            
            cooc[center_idx, context_idx] += weight
    
    return csr_matrix(cooc)


def build_vocab(tokens, min_count=1, max_vocab=None):
    """Build vocabulary from tokens."""
    word_counts = Counter(tokens)
    
    # Filter by minimum count
    filtered_words = [(word, count) for word, count in word_counts.most_common() 
                      if count >= min_count]
    
    # Limit vocabulary size
    if max_vocab is not None:
        filtered_words = filtered_words[:max_vocab]
    
    vocab = {word: idx for idx, (word, count) in enumerate(filtered_words)}
    counts = {word: count for word, count in filtered_words}
    
    return vocab, counts
```

### GloVe Model

```python
class GloVe(nn.Module):
    """
    GloVe model for learning word embeddings.
    
    Learns embeddings by minimizing weighted squared error
    between dot products and log co-occurrence counts.
    """
    
    def __init__(self, vocab_size, embedding_dim):
        super(GloVe, self).__init__()
        
        # Word embeddings (center words)
        self.w_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Context embeddings
        self.c_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Bias terms
        self.w_bias = nn.Embedding(vocab_size, 1)
        self.c_bias = nn.Embedding(vocab_size, 1)
        
        # Initialize with small random values
        init_range = 0.5 / embedding_dim
        nn.init.uniform_(self.w_embeddings.weight, -init_range, init_range)
        nn.init.uniform_(self.c_embeddings.weight, -init_range, init_range)
        nn.init.zeros_(self.w_bias.weight)
        nn.init.zeros_(self.c_bias.weight)
    
    def forward(self, center_idx, context_idx):
        """
        Compute predicted log co-occurrence.
        
        Args:
            center_idx: (batch_size,) center word indices
            context_idx: (batch_size,) context word indices
        
        Returns:
            (batch_size,) predicted values
        """
        # Get embeddings
        w = self.w_embeddings(center_idx)  # (batch, dim)
        c = self.c_embeddings(context_idx)  # (batch, dim)
        
        # Get biases
        w_b = self.w_bias(center_idx).squeeze()  # (batch,)
        c_b = self.c_bias(context_idx).squeeze()  # (batch,)
        
        # Compute dot product plus biases
        prediction = torch.sum(w * c, dim=1) + w_b + c_b
        
        return prediction
    
    def get_embedding(self, word_idx):
        """Get combined embedding (average of word and context)."""
        w = self.w_embeddings.weight[word_idx]
        c = self.c_embeddings.weight[word_idx]
        return ((w + c) / 2).detach()


def glove_loss(predictions, log_cooc, weights):
    """
    GloVe weighted squared error loss.
    
    Args:
        predictions: Model predictions
        log_cooc: Log of co-occurrence counts
        weights: f(X_ij) weights
    
    Returns:
        Weighted MSE loss
    """
    diff = predictions - log_cooc
    weighted_loss = weights * (diff ** 2)
    return weighted_loss.mean()
```

### Training Loop

```python
def train_glove(tokens, embedding_dim=100, window_size=5, epochs=50, 
                batch_size=512, lr=0.05, x_max=100, alpha=0.75, min_count=5):
    """
    Train GloVe embeddings.
    
    Args:
        tokens: List of tokens
        embedding_dim: Dimension of embeddings
        window_size: Context window size
        epochs: Number of training epochs
        batch_size: Mini-batch size
        lr: Learning rate
        x_max: Maximum count for weighting function
        alpha: Exponent for weighting function
        min_count: Minimum word frequency
    
    Returns:
        Trained model, vocab, losses
    """
    # Build vocabulary
    vocab, counts = build_vocab(tokens, min_count=min_count)
    vocab_size = len(vocab)
    ix_to_word = {idx: word for word, idx in vocab.items()}
    
    print(f"Vocabulary size: {vocab_size}")
    
    # Build co-occurrence matrix
    print("Building co-occurrence matrix...")
    cooc_matrix = build_cooccurrence_matrix(tokens, vocab, window_size)
    
    # Extract non-zero entries
    print("Extracting training data...")
    rows, cols = cooc_matrix.nonzero()
    values = np.array(cooc_matrix[rows, cols]).flatten()
    
    # Convert to tensors
    center_indices = torch.tensor(rows, dtype=torch.long)
    context_indices = torch.tensor(cols, dtype=torch.long)
    cooc_values = torch.tensor(values, dtype=torch.float32)
    
    # Compute weights and log values
    weights = torch.tensor(glove_weight(values, x_max, alpha), dtype=torch.float32)
    log_cooc = torch.log(cooc_values)
    
    print(f"Non-zero co-occurrences: {len(values)}")
    
    # Initialize model
    model = GloVe(vocab_size, embedding_dim)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
    
    # Training
    n_batches = (len(values) + batch_size - 1) // batch_size
    losses = []
    
    print("Training...")
    for epoch in range(epochs):
        # Shuffle data
        perm = torch.randperm(len(values))
        center_indices = center_indices[perm]
        context_indices = context_indices[perm]
        log_cooc = log_cooc[perm]
        weights = weights[perm]
        
        total_loss = 0
        
        for i in range(0, len(values), batch_size):
            # Get batch
            batch_center = center_indices[i:i+batch_size]
            batch_context = context_indices[i:i+batch_size]
            batch_log_cooc = log_cooc[i:i+batch_size]
            batch_weights = weights[i:i+batch_size]
            
            # Forward pass
            optimizer.zero_grad()
            predictions = model(batch_center, batch_context)
            
            # Compute loss
            loss = glove_loss(predictions, batch_log_cooc, batch_weights)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / n_batches
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return model, vocab, ix_to_word, losses
```

## GloVe vs. Word2Vec

### Theoretical Comparison

| Aspect | GloVe | Word2Vec |
|--------|-------|----------|
| **Training Data** | Co-occurrence matrix | Context windows |
| **Objective** | Weighted least squares | Softmax/Negative sampling |
| **Global Statistics** | Directly modeled | Implicitly captured |
| **Interpretability** | Log-bilinear relationship | Prediction-based |
| **Efficiency** | Single pass over matrix | Multiple passes over corpus |

### Connection to Word2Vec

Levy & Goldberg (2014) showed that Skip-gram with negative sampling implicitly factorizes a shifted PMI matrix, while GloVe directly factorizes the log co-occurrence matrix.

Both can be viewed as **matrix factorization** methods:

$$\mathbf{W} \mathbf{C}^\top \approx \mathbf{M}$$

Where $\mathbf{M}$ is:
- **Word2Vec:** Shifted PMI matrix
- **GloVe:** Log co-occurrence matrix

### Practical Performance

| Task | GloVe | Word2Vec |
|------|-------|----------|
| Word analogy | Slightly better | Good |
| Word similarity | Good | Good |
| Downstream NLP | Comparable | Comparable |
| Training speed | Faster (single pass) | Slower (multiple epochs) |

## Word Analogy with GloVe

GloVe embeddings excel at word analogies through vector arithmetic:

$$\mathbf{v}_{\text{king}} - \mathbf{v}_{\text{man}} + \mathbf{v}_{\text{woman}} \approx \mathbf{v}_{\text{queen}}$$

```python
def analogy(model, vocab, ix_to_word, word_a, word_b, word_c, top_k=5):
    """
    Solve analogy: a is to b as c is to ?
    
    Example: king is to man as queen is to woman
             analogy("king", "man", "queen") -> "woman"
    """
    if word_a not in vocab or word_b not in vocab or word_c not in vocab:
        return None
    
    # Get embeddings
    emb_a = model.get_embedding(vocab[word_a])
    emb_b = model.get_embedding(vocab[word_b])
    emb_c = model.get_embedding(vocab[word_c])
    
    # Compute target vector: b - a + c
    target = emb_b - emb_a + emb_c
    
    # Get all embeddings
    all_embeddings = (model.w_embeddings.weight + model.c_embeddings.weight) / 2
    
    # Compute cosine similarity
    target_norm = target / target.norm()
    all_norm = all_embeddings / all_embeddings.norm(dim=1, keepdim=True)
    similarities = torch.matmul(all_norm, target_norm)
    
    # Exclude input words
    exclude = {vocab[word_a], vocab[word_b], vocab[word_c]}
    for idx in exclude:
        similarities[idx] = -1
    
    # Get top-k
    top_sim, top_idx = torch.topk(similarities, k=top_k)
    
    return [(ix_to_word[idx.item()], sim.item()) 
            for idx, sim in zip(top_idx, top_sim)]
```

## Hyperparameter Guidelines

| Parameter | Typical Value | Notes |
|-----------|---------------|-------|
| Embedding dim | 50-300 | 300 for large corpora |
| Window size | 5-15 | Larger captures more semantic |
| $x_{\max}$ | 100 | Can tune based on corpus |
| $\alpha$ | 0.75 | Rarely changed |
| Learning rate | 0.05 | For Adagrad optimizer |
| Epochs | 50-100 | Until convergence |
| Min count | 5 | Filter rare words |

## Key Takeaways

!!! success "Main Concepts"

    1. **GloVe** combines global matrix factorization with local context methods
    2. **Co-occurrence ratios** capture semantic relationships
    3. **Weighted least squares** handles word frequency imbalance
    4. **The objective** is interpretable: dot product ≈ log co-occurrence
    5. **Performance** is comparable to Word2Vec with different trade-offs

!!! tip "Best Practices"

    - Use larger window sizes (10-15) for semantic relationships
    - Use smaller window sizes (5) for syntactic relationships
    - Always apply the weighting function to handle rare/frequent words
    - Average word and context embeddings for final vectors

## References

- Pennington, J., Socher, R., & Manning, C. D. (2014). "GloVe: Global Vectors for Word Representation"
- Levy, O., & Goldberg, Y. (2014). "Neural Word Embedding as Implicit Matrix Factorization"
- Levy, O., Goldberg, Y., & Dagan, I. (2015). "Improving Distributional Similarity with Lessons Learned from Word Embeddings"
