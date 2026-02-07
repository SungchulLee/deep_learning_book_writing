# t-SNE

Probabilistic neighbor embedding with heavy-tailed distributions for visualization.

---

## Overview

**t-Distributed Stochastic Neighbor Embedding (t-SNE)** models pairwise similarities as probability distributions — Gaussian in high-dimensional space and Student-t in low-dimensional space — then minimizes the KL divergence between them. The heavy-tailed Student-t distribution alleviates the **crowding problem**, making t-SNE exceptionally effective at revealing cluster structure.

---

## Motivation: The Crowding Problem

When mapping from high to low dimensions, there is far less "room" to faithfully represent moderate distances. Points at moderate distances in high-D get crushed together in low-D, obscuring cluster structure.

**SNE (Gaussian in both spaces):** Moderate-distance points crowd into a dense mass.

**t-SNE (Student-t in low-D):** The heavy tail allows moderate-distance points to spread out, revealing clusters.

---

## Algorithm

### Step 1: High-Dimensional Affinities

For each pair $(i, j)$, define the conditional probability that $i$ would pick $j$ as a neighbor under a Gaussian centered at $\mathbf{x}_i$:

$$p_{j|i} = \frac{\exp(-\|\mathbf{x}_i - \mathbf{x}_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|\mathbf{x}_i - \mathbf{x}_k\|^2 / 2\sigma_i^2)}$$

Symmetrize to get joint probabilities:

$$p_{ij} = \frac{p_{j|i} + p_{i|j}}{2n}$$

### Step 2: Bandwidth Selection via Perplexity

Each $\sigma_i$ is chosen so that the conditional distribution $P_i$ has a specified **perplexity**:

$$\text{Perp}(P_i) = 2^{H(P_i)} = 2^{-\sum_j p_{j|i} \log_2 p_{j|i}}$$

Perplexity is an effective number of neighbors. Typical values: 5–50.

### Step 3: Low-Dimensional Affinities (Student-t)

In the embedding space, use a Student-t distribution with 1 degree of freedom (Cauchy):

$$q_{ij} = \frac{(1 + \|\mathbf{y}_i - \mathbf{y}_j\|^2)^{-1}}{\sum_{k \neq l} (1 + \|\mathbf{y}_k - \mathbf{y}_l\|^2)^{-1}}$$

### Step 4: Minimize KL Divergence

$$C = \text{KL}(P \| Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}$$

### Gradient

$$\frac{\partial C}{\partial \mathbf{y}_i} = 4 \sum_{j} (p_{ij} - q_{ij})(\mathbf{y}_i - \mathbf{y}_j)(1 + \|\mathbf{y}_i - \mathbf{y}_j\|^2)^{-1}$$

The gradient has an intuitive interpretation: points with $p_{ij} > q_{ij}$ (too far apart in embedding) are attracted; points with $p_{ij} < q_{ij}$ (too close) are repelled.

---

## Why Student-t?

### Tail Comparison

| Distance | Gaussian $\exp(-r^2)$ | Student-t $(1+r^2)^{-1}$ |
|----------|-----------------------|---------------------------|
| $r = 1$ | 0.37 | 0.50 |
| $r = 3$ | 0.00012 | 0.10 |
| $r = 10$ | $\approx 0$ | 0.0099 |

The Student-t distribution assigns much more probability to moderate and large distances. This means:

- **Nearby points** are modeled similarly by both distributions
- **Distant points** can spread further apart in the embedding without large KL cost
- **Clusters separate clearly** because inter-cluster gaps can be wider

---

## Implementation

```python
import numpy as np

def compute_pairwise_affinities(X, perplexity=30.0, tol=1e-5):
    """
    Compute symmetric pairwise affinities P with binary search
    for bandwidth sigma_i to match target perplexity.
    
    Args:
        X: Data matrix [n, d]
        perplexity: Target perplexity (effective neighbor count)
        tol: Tolerance for perplexity match
    
    Returns:
        P: Symmetric affinity matrix [n, n]
    """
    n = X.shape[0]
    
    # Squared Euclidean distances
    D = np.sum((X[:, None] - X[None, :]) ** 2, axis=2)
    
    P = np.zeros((n, n))
    target_entropy = np.log(perplexity)
    
    for i in range(n):
        # Binary search for sigma_i
        lo, hi = 1e-10, 1e4
        
        for _ in range(50):  # Max iterations
            sigma = (lo + hi) / 2
            
            # Conditional probabilities
            exp_d = np.exp(-D[i] / (2 * sigma ** 2))
            exp_d[i] = 0
            
            sum_exp = exp_d.sum()
            if sum_exp == 0:
                lo = sigma
                continue
            
            p_i = exp_d / sum_exp
            
            # Shannon entropy
            H = -np.sum(p_i[p_i > 0] * np.log(p_i[p_i > 0]))
            
            if H > target_entropy:
                hi = sigma
            else:
                lo = sigma
            
            if abs(H - target_entropy) < tol:
                break
        
        P[i] = p_i
    
    # Symmetrize
    P = (P + P.T) / (2 * n)
    P = np.maximum(P, 1e-12)
    
    return P


def tsne(X, n_components=2, perplexity=30.0, n_iter=1000, 
         lr=200.0, momentum=0.8, seed=42):
    """
    t-SNE via gradient descent.
    
    Args:
        X: Data matrix [n, d]
        n_components: Embedding dimensionality (typically 2)
        perplexity: Effective number of neighbors
        n_iter: Number of gradient descent iterations
        lr: Learning rate
        momentum: Momentum coefficient
    
    Returns:
        Y: Embedded coordinates [n, n_components]
        kl_history: KL divergence per iteration
    """
    np.random.seed(seed)
    n = X.shape[0]
    
    # High-dimensional affinities
    P = compute_pairwise_affinities(X, perplexity)
    
    # Early exaggeration (multiply P by 4 for first 250 iterations)
    P_exag = P * 4.0
    
    # Initialize embedding
    Y = 0.01 * np.random.randn(n, n_components)
    Y_prev = Y.copy()
    kl_history = []
    
    for t in range(n_iter):
        # Use exaggerated P for early iterations
        P_t = P_exag if t < 250 else P
        
        # Low-dimensional affinities (Student-t)
        diff = Y[:, None] - Y[None, :]          # [n, n, k]
        dist_sq = np.sum(diff ** 2, axis=2)      # [n, n]
        Q_num = (1 + dist_sq) ** (-1)
        np.fill_diagonal(Q_num, 0)
        Q = Q_num / Q_num.sum()
        Q = np.maximum(Q, 1e-12)
        
        # Gradient
        PQ_diff = P_t - Q
        grad = np.zeros_like(Y)
        for i in range(n):
            grad[i] = 4 * np.sum(
                (PQ_diff[i, :, None] * diff[i]) * 
                Q_num[i, :, None], axis=0
            )
        
        # Update with momentum
        Y_new = Y - lr * grad + momentum * (Y - Y_prev)
        Y_prev = Y.copy()
        Y = Y_new
        
        # Center
        Y -= Y.mean(axis=0)
        
        # KL divergence
        kl = np.sum(P_t * np.log(P_t / Q))
        kl_history.append(kl)
    
    return Y, kl_history
```

---

## Perplexity

Perplexity is the single most important hyperparameter. It controls the effective neighborhood size used to compute affinities.

### Effect of Perplexity

| Perplexity | Behavior |
|------------|----------|
| Low (5–10) | Tight local clusters, may fragment |
| Medium (30–50) | Balanced local/global structure |
| High (100+) | Smoother, more global, slower |

### Guidelines

- Always try **multiple perplexity values** (e.g., 5, 30, 50, 100)
- Rule of thumb: perplexity $\approx n^{1/2}$ to $n / 50$
- Perplexity must be less than $n$
- Different perplexities can reveal different structure scales

---

## scikit-learn Interface

```python
from sklearn.manifold import TSNE

# Basic usage
tsne = TSNE(
    n_components=2,
    perplexity=30,
    n_iter=1000,
    learning_rate='auto',
    init='pca',
    random_state=42
)
Y = tsne.fit_transform(X)
print(f"KL divergence: {tsne.kl_divergence_:.4f}")
```

### Key Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `perplexity` | 30 | Effective neighborhood size |
| `n_iter` | 1000 | Optimization iterations |
| `learning_rate` | `'auto'` | Step size ($n/12$ when auto) |
| `early_exaggeration` | 12.0 | Factor for early P amplification |
| `init` | `'pca'` | Initialization strategy |
| `metric` | `'euclidean'` | Input distance metric |

---

## Optimization Tricks

### Early Exaggeration

Multiply $P$ by a factor (typically 4–12) during early iterations. This creates tighter clusters early, helping the algorithm find global structure before refining local detail.

### PCA Initialization

Initialize $\mathbf{Y}$ from PCA rather than random. Preserves global structure and leads to more reproducible results.

### Barnes-Hut Approximation

For large $n$, the $O(n^2)$ gradient computation is expensive. Barnes-Hut t-SNE uses a tree structure to approximate repulsive forces in $O(n \log n)$.

```python
# Barnes-Hut (default for n > 500 in sklearn)
tsne_bh = TSNE(method='barnes_hut', angle=0.5)  # angle controls accuracy

# Exact (for small datasets)
tsne_exact = TSNE(method='exact')
```

---

## Interpreting t-SNE Plots

### What You CAN Infer

- **Cluster membership:** Points in the same cluster are genuinely similar
- **Relative cluster separation:** Well-separated clusters are distinct in high-D

### What You CANNOT Infer

- **Inter-cluster distances:** Distances between clusters are NOT meaningful
- **Cluster sizes:** Apparent cluster size reflects density, not spread
- **Global geometry:** The overall layout is arbitrary

### Common Pitfalls

| Pitfall | Why It Happens |
|---------|---------------|
| Seeing clusters that don't exist | Perplexity too low → fragmenting |
| Missing real clusters | Perplexity too high → merging |
| Different runs give different layouts | Non-convex optimization, random init |
| Reading meaning into distances between clusters | KL divergence doesn't preserve global distances |

---

## Application: Market Regime Visualization

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def market_regime_tsne(returns, window=60, step=10,
                       perplexity=30):
    """
    Visualize market regimes by embedding rolling
    statistics with t-SNE.
    
    Args:
        returns: Return matrix [n_periods, n_assets]
        window: Rolling window size
        step: Step size
        perplexity: t-SNE perplexity
    """
    n_periods = returns.shape[0]
    features = []
    timestamps = []
    
    for t in range(0, n_periods - window, step):
        chunk = returns[t:t+window]
        
        # Feature vector: means, vols, correlations
        means = chunk.mean(axis=0)
        vols = chunk.std(axis=0)
        corr = np.corrcoef(chunk.T)
        upper_corr = corr[np.triu_indices_from(corr, k=1)]
        
        features.append(np.concatenate([means, vols, upper_corr]))
        timestamps.append(t + window)
    
    features = np.array(features)
    timestamps = np.array(timestamps)
    
    # t-SNE embedding
    tsne = TSNE(n_components=2, perplexity=perplexity,
                random_state=42)
    Y = tsne.fit_transform(features)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(Y[:, 0], Y[:, 1], c=timestamps,
                         cmap='viridis', s=30, alpha=0.7)
    
    # Connect consecutive windows
    ax.plot(Y[:, 0], Y[:, 1], 'k-', alpha=0.15, linewidth=0.5)
    
    plt.colorbar(scatter, label='Time Index')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title('Market Regime Map (t-SNE)')
    plt.tight_layout()
    plt.show()


# Example
np.random.seed(42)
n_periods, n_assets = 1000, 10
returns = np.random.randn(n_periods, n_assets) * 0.01

# Inject regime change at t=500 (higher vol, higher corr)
returns[500:] *= 2.0
returns[500:] += 0.3 * np.random.randn(500, 1)

market_regime_tsne(returns, window=60, step=5, perplexity=20)
```

---

## t-SNE with PyTorch

```python
import torch

def tsne_torch(X, n_components=2, perplexity=30.0, 
               n_iter=1000, lr=200.0, seed=42):
    """
    t-SNE with PyTorch autograd for gradient computation.
    
    Args:
        X: Data matrix [n, d] (numpy)
        n_components: Embedding dimension
        perplexity: Target perplexity
        n_iter: Iterations
        lr: Learning rate
    
    Returns:
        Y: Embedding [n, n_components] (numpy)
    """
    torch.manual_seed(seed)
    
    # Compute P in numpy (binary search doesn't benefit from GPU)
    P = compute_pairwise_affinities(X, perplexity)
    P_tensor = torch.tensor(P, dtype=torch.float32)
    
    n = X.shape[0]
    Y = torch.randn(n, n_components, requires_grad=True) * 0.01
    
    optimizer = torch.optim.Adam([Y], lr=lr)
    
    for t in range(n_iter):
        optimizer.zero_grad()
        
        # Student-t affinities
        diff = Y.unsqueeze(0) - Y.unsqueeze(1)
        dist_sq = (diff ** 2).sum(dim=-1)
        Q_num = (1 + dist_sq).pow(-1)
        Q_num.fill_diagonal_(0)
        Q = Q_num / Q_num.sum()
        Q = torch.clamp(Q, min=1e-12)
        
        # KL divergence
        P_eff = P_tensor * 4.0 if t < 250 else P_tensor
        kl = (P_eff * torch.log(P_eff / Q)).sum()
        
        kl.backward()
        optimizer.step()
        
        # Center
        Y.data -= Y.data.mean(dim=0)
    
    return Y.detach().numpy()
```

---

## Complexity Analysis

| Step | Exact | Barnes-Hut |
|------|-------|------------|
| Affinities $P$ | $O(n^2 d)$ | $O(n^2 d)$ |
| Each gradient step | $O(n^2)$ | $O(n \log n)$ |
| **Total** ($T$ iterations) | $O(n^2 T)$ | $O(n \log n \cdot T)$ |

---

## Summary

| Concept | Key Point |
|---------|-----------|
| **High-D affinities** | Gaussian: $p_{ij} \propto \exp(-\|x_i - x_j\|^2 / 2\sigma_i^2)$ |
| **Low-D affinities** | Student-t: $q_{ij} \propto (1 + \|y_i - y_j\|^2)^{-1}$ |
| **Objective** | Minimize $\text{KL}(P \| Q)$ |
| **Key parameter** | Perplexity (effective neighborhood size) |
| **Strength** | Excellent cluster visualization |
| **Weakness** | No global distance preservation, non-deterministic |
| **Complexity** | $O(n^2)$ exact, $O(n \log n)$ Barnes-Hut per step |

---

## What's Next

t-SNE excels at visualization but is slow, non-parametric, and does not preserve global structure. **UMAP** addresses these issues with a topological framework that is faster, supports out-of-sample mapping, and better balances local and global structure preservation.
