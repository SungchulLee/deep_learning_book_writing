# Embedding Visualization

## Overview

Visualizing word embeddings is essential for understanding their structure, verifying that similar words cluster together, debugging training issues, and communicating results. Because embeddings live in high-dimensional space ($d = 50\text{–}300$), we need dimensionality reduction techniques to project them into 2D or 3D for human inspection. This section covers the three main approaches — **PCA**, **t-SNE**, and **UMAP** — along with practical visualization recipes for analogies, clusters, and training diagnostics.

## Learning Objectives

By the end of this section, you will:

- Apply PCA, t-SNE, and UMAP for embedding visualization
- Implement embedding scatter plots with word annotations
- Visualize word analogies as geometric parallelograms
- Plot training loss curves for embedding models
- Understand the trade-offs between visualization methods

## Dimensionality Reduction Techniques

### PCA (Principal Component Analysis)

PCA finds orthogonal directions of maximum variance and projects embeddings onto the top-$k$ principal components.

**Properties:**

- **Linear** projection
- **Preserves global structure** (distances between distant points)
- **Fast** computation ($O(nd^2)$ for $n$ embeddings of dimension $d$)
- **Deterministic** (same input always produces the same output)
- **Interpretable** axes (principal components explain decreasing proportions of variance)

**When to use:** Quick overview of global embedding structure, initial exploration, comparing before/after training.

### t-SNE (t-distributed Stochastic Neighbor Embedding)

t-SNE models pairwise similarities as probability distributions and minimizes the KL-divergence between high-dimensional and low-dimensional representations.

**Properties:**

- **Non-linear** mapping
- **Preserves local neighborhoods** (nearby points stay nearby)
- **Slow** for large datasets ($O(n^2)$ naive, $O(n \log n)$ with Barnes-Hut)
- **Stochastic** (different runs produce different layouts)
- **Hyperparameter-sensitive** (perplexity controls neighborhood size)

**When to use:** Discovering clusters of semantically related words, examining local structure.

### UMAP (Uniform Manifold Approximation and Projection)

UMAP constructs a topological representation of the data and finds a low-dimensional embedding that preserves this topology.

**Properties:**

- **Non-linear** mapping
- **Preserves both local and global structure**
- **Faster** than t-SNE for large datasets
- **More stable** across runs than t-SNE
- Requires the `umap-learn` library

**When to use:** Large embedding vocabularies, when both cluster separation and inter-cluster distance matter.

### Comparison

| Property | PCA | t-SNE | UMAP |
|----------|-----|-------|------|
| **Type** | Linear | Non-linear | Non-linear |
| **Global structure** | Preserved | Distorted | Partially preserved |
| **Local structure** | Approximate | Excellent | Excellent |
| **Speed** | Fast | Slow | Moderate |
| **Deterministic** | Yes | No | No |
| **Scalability** | Excellent | Poor (>10K) | Good |

## PyTorch Implementation

### General Visualization Function

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def visualize_embeddings(embeddings, words, word_to_idx, method='pca', 
                         perplexity=30, figsize=(12, 10)):
    """
    Visualize word embeddings in 2D using PCA or t-SNE.
    
    Args:
        embeddings: Embedding weight tensor of shape (vocab_size, embed_dim)
        words: List of words to visualize
        word_to_idx: Dictionary mapping words to indices
        method: 'pca' or 'tsne'
        perplexity: t-SNE perplexity (ignored for PCA)
        figsize: Figure size
    
    Returns:
        2D coordinates array of shape (num_words, 2)
    """
    # Filter to words in vocabulary
    valid_words = [w for w in words if w in word_to_idx]
    indices = [word_to_idx[w] for w in valid_words]
    
    if len(indices) == 0:
        print("No valid words found in vocabulary")
        return None
    
    # Get embeddings for selected words
    if isinstance(embeddings, torch.Tensor):
        selected_emb = embeddings[indices].detach().numpy()
    else:
        selected_emb = embeddings[indices]
    
    # Dimensionality reduction
    if method == 'pca':
        reducer = PCA(n_components=2)
        coords = reducer.fit_transform(selected_emb)
        variance_explained = reducer.explained_variance_ratio_
        title = f'Word Embeddings (PCA — {variance_explained[0]:.1%} + {variance_explained[1]:.1%} variance)'
    elif method == 'tsne':
        perplexity = min(perplexity, len(indices) - 1)
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        coords = reducer.fit_transform(selected_emb)
        title = f'Word Embeddings (t-SNE, perplexity={perplexity})'
    else:
        raise ValueError(f"Unknown method: {method}. Use 'pca' or 'tsne'.")
    
    # Plot
    plt.figure(figsize=figsize)
    plt.scatter(coords[:, 0], coords[:, 1], alpha=0.7, s=100)
    
    # Annotate points with word labels
    for i, word in enumerate(valid_words):
        plt.annotate(word, (coords[i, 0], coords[i, 1]),
                    fontsize=10, alpha=0.8,
                    xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return coords
```

### Visualizing Word Analogies

Word analogies form parallelograms in embedding space. Visualizing them verifies that the expected geometric relationships hold:

```python
def visualize_analogy(embeddings, word_to_idx, word_a, word_b, word_c, word_d):
    """
    Visualize word analogy as a parallelogram: a:b :: c:d
    
    Example: king:queen :: man:woman
    """
    words = [word_a, word_b, word_c, word_d]
    
    if not all(w in word_to_idx for w in words):
        print("Some words not in vocabulary")
        return
    
    # Get embeddings
    indices = [word_to_idx[w] for w in words]
    if isinstance(embeddings, torch.Tensor):
        selected_emb = embeddings[indices].detach().numpy()
    else:
        selected_emb = embeddings[indices]
    
    # Reduce to 2D with PCA
    pca = PCA(n_components=2)
    coords = pca.fit_transform(selected_emb)
    
    # Plot parallelogram
    plt.figure(figsize=(10, 8))
    
    # Draw edges
    plt.plot([coords[0, 0], coords[1, 0]], [coords[0, 1], coords[1, 1]], 
             'b-', linewidth=2, label=f'{word_a} → {word_b}')
    plt.plot([coords[2, 0], coords[3, 0]], [coords[2, 1], coords[3, 1]], 
             'b--', linewidth=2, label=f'{word_c} → {word_d}')
    plt.plot([coords[0, 0], coords[2, 0]], [coords[0, 1], coords[2, 1]], 
             'r-', linewidth=2, alpha=0.5)
    plt.plot([coords[1, 0], coords[3, 0]], [coords[1, 1], coords[3, 1]], 
             'r-', linewidth=2, alpha=0.5)
    
    # Plot points
    plt.scatter(coords[:, 0], coords[:, 1], s=200, zorder=5)
    
    # Annotate
    for i, word in enumerate(words):
        plt.annotate(word, (coords[i, 0], coords[i, 1]),
                    fontsize=12, fontweight='bold',
                    xytext=(10, 10), textcoords='offset points')
    
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(f'Word Analogy: {word_a}:{word_b} :: {word_c}:{word_d}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
```

### Cluster Visualization with Color Coding

```python
def visualize_word_clusters(embeddings, word_to_idx, word_groups, method='pca',
                            figsize=(14, 10)):
    """
    Visualize semantic clusters with color coding.
    
    Args:
        embeddings: Embedding weight tensor
        word_to_idx: Word to index mapping
        word_groups: Dict mapping group_name -> list of words
                     e.g., {"animals": ["cat", "dog"], "colors": ["red", "blue"]}
        method: 'pca' or 'tsne'
    """
    all_words = []
    group_labels = []
    colors = plt.cm.Set2(np.linspace(0, 1, len(word_groups)))
    
    for group_name, words in word_groups.items():
        for w in words:
            if w in word_to_idx:
                all_words.append(w)
                group_labels.append(group_name)
    
    if not all_words:
        print("No valid words found")
        return
    
    indices = [word_to_idx[w] for w in all_words]
    if isinstance(embeddings, torch.Tensor):
        selected_emb = embeddings[indices].detach().numpy()
    else:
        selected_emb = embeddings[indices]
    
    # Reduce dimensions
    if method == 'pca':
        coords = PCA(n_components=2).fit_transform(selected_emb)
    else:
        perp = min(30, len(all_words) - 1)
        coords = TSNE(n_components=2, perplexity=perp, random_state=42).fit_transform(selected_emb)
    
    # Plot with colors per group
    plt.figure(figsize=figsize)
    group_names = list(word_groups.keys())
    
    for i, (word, label) in enumerate(zip(all_words, group_labels)):
        color_idx = group_names.index(label)
        plt.scatter(coords[i, 0], coords[i, 1], c=[colors[color_idx]], s=100, alpha=0.8)
        plt.annotate(word, (coords[i, 0], coords[i, 1]),
                    fontsize=9, alpha=0.8,
                    xytext=(5, 5), textcoords='offset points')
    
    # Legend
    for idx, name in enumerate(group_names):
        plt.scatter([], [], c=[colors[idx]], label=name, s=100)
    plt.legend(loc='best', fontsize=11)
    
    plt.title(f'Word Embedding Clusters ({method.upper()})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
```

### Training Loss Visualization

```python
def plot_training_loss(losses, title="Embedding Training Loss"):
    """Plot training loss curve."""
    plt.figure(figsize=(10, 5))
    plt.plot(losses, linewidth=2, color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
```

## Visualization Best Practices

!!! tip "Guidelines for Effective Visualization"

    1. **Choose the right method:** PCA for global overview, t-SNE for local clusters, UMAP for both
    2. **Limit the number of words:** 50–200 words produce readable plots; more requires interactive tools
    3. **Use semantic groupings:** Color-code words by category to verify cluster quality
    4. **Try multiple perplexity values:** t-SNE is sensitive to perplexity; try 5, 30, and 50
    5. **Report variance explained:** For PCA, state how much variance the top 2 components capture
    6. **Be cautious with t-SNE distances:** Inter-cluster distances in t-SNE are not meaningful

!!! warning "Common Pitfalls"

    - **Over-interpreting t-SNE:** Cluster sizes and distances between clusters are artifacts, not meaningful
    - **Too many words:** Dense plots become unreadable — use interactive visualization (e.g., Plotly) for large vocabularies
    - **Ignoring variance:** PCA with low variance explained (e.g., <20%) means the 2D projection is lossy
    - **Single run of t-SNE:** Run multiple times to ensure stable cluster patterns

## Key Takeaways

!!! success "Main Concepts"

    1. **PCA** provides fast, deterministic, global-structure-preserving projections
    2. **t-SNE** excels at revealing local clustering structure but distorts global distances
    3. **UMAP** balances local and global structure with better scalability than t-SNE
    4. **Analogy visualization** as parallelograms verifies geometric embedding properties
    5. **Cluster visualization** with color coding validates that semantic categories are separated

## References

- Maaten, L. v. d., & Hinton, G. (2008). "Visualizing Data using t-SNE." JMLR.
- McInnes, L., Healy, J., & Melville, J. (2018). "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction."
- Wattenberg, M., Viégas, F., & Johnson, I. (2016). "How to Use t-SNE Effectively." Distill.
