# Negative Sampling

## Overview

Negative Sampling is an efficient training technique for word embeddings that approximates the full softmax objective by converting the multi-class classification problem into multiple binary classification problems. Instead of computing probabilities over the entire vocabulary, negative sampling only considers a small number of "negative" examples for each training instance, dramatically reducing computational cost while maintaining embedding quality.

## Learning Objectives

By the end of this section, you will:

- Understand the computational bottleneck of the full softmax
- Derive the negative sampling objective mathematically
- Implement negative sampling efficiently in PyTorch
- Design effective noise distributions for sampling
- Understand the theoretical connection to PMI (Pointwise Mutual Information)

## The Softmax Bottleneck

### Computational Complexity

Recall the Skip-gram softmax probability:

$$P(w_o | w_c) = \frac{\exp(\mathbf{u}_{w_o}^\top \mathbf{v}_{w_c})}{\sum_{w=1}^{|V|} \exp(\mathbf{u}_w^\top \mathbf{v}_{w_c})}$$

The denominator requires computing dot products with all $|V|$ words. For a corpus of billions of words, this $O(|V|)$ computation per update is prohibitively expensive.

### Gradient Computation Cost

The gradient involves computing the expected context embedding:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{v}_c} = -\mathbf{u}_o + \sum_{w=1}^{|V|} P(w|c) \cdot \mathbf{u}_w$$

This requires the full softmax distribution at every training step — infeasible for large vocabularies.

## Negative Sampling Intuition

### From Multi-class to Binary Classification

**Key Insight:** Instead of asking "which word is correct among all $|V|$ words?", we ask: "Is this (center, context) pair real or fake?"

**Training Signal:**

- **Positive pairs:** Real (center, context) pairs from data → label 1
- **Negative pairs:** Fake pairs with randomly sampled "noise" words → label 0

### Geometric Interpretation

Negative sampling pushes:

- **Positive pairs closer:** Increase dot product $\mathbf{u}_o^\top \mathbf{v}_c$
- **Negative pairs apart:** Decrease dot product $\mathbf{u}_k^\top \mathbf{v}_c$

```
Before training:              After training:
    ○ noise_1                     ○ noise_1
    ○ center                      
      ○ context                       ● center ↔ context
    ○ noise_2                     ○ noise_2
    
(all similar)                 (context close, noise far)
```

## Mathematical Formulation

### Negative Sampling Objective

For a positive pair $(w_c, w_o)$ and $k$ negative samples $w_1, \ldots, w_k$ drawn from noise distribution $P_n(w)$:

$$\mathcal{L}_{\text{NEG}} = -\log \sigma(\mathbf{u}_{w_o}^\top \mathbf{v}_{w_c}) - \sum_{i=1}^{k} \mathbb{E}_{w_i \sim P_n} \left[ \log \sigma(-\mathbf{u}_{w_i}^\top \mathbf{v}_{w_c}) \right]$$

Where $\sigma(x) = \frac{1}{1 + e^{-x}}$ is the sigmoid function.

**Simplified (using sampled negatives):**

$$\mathcal{L}_{\text{NEG}} = -\log \sigma(\mathbf{u}_{w_o}^\top \mathbf{v}_{w_c}) - \sum_{i=1}^{k} \log \sigma(-\mathbf{u}_{w_i}^\top \mathbf{v}_{w_c})$$

### Interpretation

| Term | Meaning | Goal |
|------|---------|------|
| $-\log \sigma(\mathbf{u}_o^\top \mathbf{v}_c)$ | Positive pair score | Maximize (push together) |
| $-\log \sigma(-\mathbf{u}_i^\top \mathbf{v}_c)$ | Negative pair score | Maximize (push apart) |

### Connection to Binary Cross-Entropy

The objective is equivalent to binary cross-entropy with label 1 for positive pairs and label 0 for negative pairs:

$$\mathcal{L} = -[y \log(\hat{y}) + (1-y) \log(1-\hat{y})]$$

Where $\hat{y} = \sigma(\mathbf{u}^\top \mathbf{v})$.

## Noise Distribution

### The Unigram Distribution

The noise distribution $P_n(w)$ determines which words are sampled as negatives.

$$P_n(w) = \frac{\text{count}(w)}{\sum_{w'} \text{count}(w')}$$

### The 3/4 Power Trick

Mikolov et al. found that raising word frequencies to the power of $3/4$ works better empirically:

$$P_n(w) = \frac{\text{count}(w)^{0.75}}{\sum_{w'} \text{count}(w')^{0.75}}$$

**Why 3/4?** This reduces the dominance of very frequent words and gives rare words a better chance of being sampled as negatives, improving overall embedding quality.

| Word | Count | Unigram $P$ | Smoothed $P^{0.75}$ |
|------|-------|-------------|---------------------|
| "the" | 1000 | 0.50 | 0.35 |
| "cat" | 100 | 0.05 | 0.08 |
| "aardvark" | 10 | 0.005 | 0.015 |

```python
import numpy as np
from collections import Counter

def create_noise_distribution(tokens, power=0.75):
    """
    Create the noise distribution for negative sampling.
    
    Args:
        tokens: List of tokens in corpus
        power: Smoothing power (default 0.75 per Mikolov et al.)
    
    Returns:
        Array of sampling probabilities indexed by word
    """
    word_counts = Counter(tokens)
    words = list(word_counts.keys())
    counts = np.array([word_counts[w] for w in words], dtype=np.float64)
    
    # Apply power smoothing
    counts_powered = counts ** power
    
    # Normalize to probabilities
    probs = counts_powered / counts_powered.sum()
    
    return words, probs
```

## PyTorch Implementation

### Skip-gram with Negative Sampling

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter

class SkipGramNegativeSampling(nn.Module):
    """
    Skip-gram model with Negative Sampling.
    
    Key difference from softmax Skip-gram:
    - Uses binary classification instead of multi-class
    - Only updates embeddings for sampled words
    - Complexity: O(k) instead of O(|V|)
    """
    
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramNegativeSampling, self).__init__()
        
        # Center word embeddings
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Context word embeddings
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Initialize with small random values
        nn.init.uniform_(self.center_embeddings.weight, -0.5/embedding_dim, 0.5/embedding_dim)
        nn.init.uniform_(self.context_embeddings.weight, -0.5/embedding_dim, 0.5/embedding_dim)
    
    def forward(self, center_words, context_words, negative_words):
        """
        Compute negative sampling loss.
        
        Args:
            center_words: (batch_size,) center word indices
            context_words: (batch_size,) positive context word indices
            negative_words: (batch_size, num_negatives) negative word indices
        
        Returns:
            Negative sampling loss (scalar)
        """
        # Get embeddings
        center_emb = self.center_embeddings(center_words)       # (batch, dim)
        context_emb = self.context_embeddings(context_words)     # (batch, dim)
        negative_emb = self.context_embeddings(negative_words)   # (batch, k, dim)
        
        # Positive score: dot product of center and context
        positive_score = torch.sum(center_emb * context_emb, dim=1)  # (batch,)
        positive_loss = F.logsigmoid(positive_score)
        
        # Negative scores: dot product of center and each negative
        center_emb_expanded = center_emb.unsqueeze(2)            # (batch, dim, 1)
        negative_scores = torch.bmm(negative_emb, center_emb_expanded).squeeze(2)  # (batch, k)
        negative_loss = F.logsigmoid(-negative_scores).sum(dim=1)
        
        # Total loss (negate because we maximize log-likelihood)
        loss = -(positive_loss + negative_loss).mean()
        
        return loss
    
    def get_embedding(self, word_idx):
        """Get the learned embedding (average of center and context)."""
        center = self.center_embeddings.weight[word_idx]
        context = self.context_embeddings.weight[word_idx]
        return ((center + context) / 2).detach()
```

### Negative Sampler

```python
class NegativeSampler:
    """Efficient negative sampler with smoothed unigram distribution."""
    
    def __init__(self, word_counts, power=0.75):
        self.vocab_size = len(word_counts)
        
        # Compute smoothed probabilities
        counts = np.array(list(word_counts.values()), dtype=np.float64)
        counts_powered = counts ** power
        self.probs = counts_powered / counts_powered.sum()
    
    def sample(self, batch_size, num_negatives):
        """
        Sample negative word indices.
        
        Returns:
            Tensor of shape (batch_size, num_negatives) with word indices
        """
        total_samples = batch_size * num_negatives
        samples = np.random.choice(
            self.vocab_size, 
            size=total_samples, 
            p=self.probs
        )
        return torch.tensor(samples.reshape(batch_size, num_negatives), dtype=torch.long)
```

### Efficient Batch Processing with DataLoader

```python
class SkipGramDataset(torch.utils.data.Dataset):
    """Dataset for Skip-gram training with negative sampling."""
    
    def __init__(self, token_indices, window_size, num_negatives, noise_probs):
        self.token_indices = token_indices
        self.window_size = window_size
        self.num_negatives = num_negatives
        self.noise_probs = noise_probs
        self.vocab_size = len(noise_probs)
        
        # Pre-generate all (center, context) pairs
        self.pairs = []
        for i in range(window_size, len(token_indices) - window_size):
            center = token_indices[i]
            for j in range(-window_size, window_size + 1):
                if j != 0:
                    context = token_indices[i + j]
                    self.pairs.append((center, context))
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        center, context = self.pairs[idx]
        
        # Sample negatives on-the-fly for diversity
        negatives = np.random.choice(
            self.vocab_size,
            size=self.num_negatives,
            p=self.noise_probs
        )
        
        return (
            torch.tensor(center, dtype=torch.long),
            torch.tensor(context, dtype=torch.long),
            torch.tensor(negatives, dtype=torch.long)
        )
```

## Theoretical Analysis

### Connection to PMI

Levy and Goldberg (2014) showed that Skip-gram with negative sampling implicitly factorizes a **shifted PMI matrix**:

$$\mathbf{u}_w^\top \mathbf{v}_c \approx \text{PMI}(w, c) - \log k$$

Where PMI (Pointwise Mutual Information) is:

$$\text{PMI}(w, c) = \log \frac{P(w, c)}{P(w) P(c)} = \log \frac{\text{count}(w, c) \cdot |D|}{\text{count}(w) \cdot \text{count}(c)}$$

**Implication:** Word2Vec embeddings encode co-occurrence statistics, connecting the prediction-based and count-based approaches.

### Optimal Number of Negatives

| Corpus Size | Recommended $k$ |
|------------|-----------------|
| Small (< 10M tokens) | 5–10 |
| Medium (10M–100M) | 5 |
| Large (> 100M) | 2–5 |

Larger corpora need fewer negatives because each positive example is seen more times during training.

## Gradient Analysis

**Gradient for center embedding $\mathbf{v}_c$:**

$$\frac{\partial \mathcal{L}}{\partial \mathbf{v}_c} = -\sigma(-\mathbf{u}_o^\top \mathbf{v}_c) \, \mathbf{u}_o + \sum_{i=1}^k \sigma(\mathbf{u}_{w_i}^\top \mathbf{v}_c) \, \mathbf{u}_{w_i}$$

- First term: Pulls center toward positive context (weighted by current error)
- Second term: Pushes center away from negative samples

**Gradient for positive context embedding $\mathbf{u}_o$:**

$$\frac{\partial \mathcal{L}}{\partial \mathbf{u}_o} = -\sigma(-\mathbf{u}_o^\top \mathbf{v}_c) \, \mathbf{v}_c$$

**Gradient for negative context embedding $\mathbf{u}_{w_i}$:**

$$\frac{\partial \mathcal{L}}{\partial \mathbf{u}_{w_i}} = \sigma(\mathbf{u}_{w_i}^\top \mathbf{v}_c) \, \mathbf{v}_c$$

## Comparison: Softmax vs. Negative Sampling

| Aspect | Full Softmax | Negative Sampling |
|--------|--------------|-------------------|
| **Complexity** | $O(|V|)$ per update | $O(k)$ per update |
| **Gradient Updates** | All words | Only sampled words |
| **Memory** | Full embedding matrix | Sparse updates |
| **Theoretical Basis** | Maximum likelihood | NCE approximation |
| **Quality** | Optimal (given enough data) | Near-optimal |

## Key Takeaways

!!! success "Main Concepts"

    1. **Negative sampling** converts multi-class to binary classification
    2. **Noise distribution** uses word frequency raised to power 0.75
    3. **Computational complexity** reduces from $O(|V|)$ to $O(k)$ per update
    4. **Theoretical basis** connects to PMI matrix factorization (Levy & Goldberg, 2014)
    5. **Quality** is comparable to full softmax with proper hyperparameters

!!! tip "Best Practices"

    1. **Number of negatives:** Use 5–20 for small datasets, 2–5 for large
    2. **Noise distribution:** Always use the 0.75 power smoothing
    3. **Subsampling:** Remove frequent words with probability $1 - \sqrt{t/f(w)}$
    4. **Learning rate:** Start with 0.025 and linearly decay
    5. **Final embeddings:** Average center and context embeddings

!!! warning "Common Mistakes"

    - Sampling the positive word as a negative (small but measurable impact)
    - Not shuffling training data between epochs
    - Using too few negatives for small datasets
    - Not applying subsampling for frequent words

## References

- Mikolov, T., et al. (2013). "Distributed Representations of Words and Phrases and their Compositionality"
- Levy, O., & Goldberg, Y. (2014). "Neural Word Embedding as Implicit Matrix Factorization"
- Gutmann, M., & Hyvärinen, A. (2010). "Noise-contrastive estimation"
