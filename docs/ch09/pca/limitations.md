# Limitations of Linear Methods

Why we need nonlinear dimensionality reduction.

---

## Fundamental Limitation

PCA finds **linear subspaces**. Real data often lies on **nonlinear manifolds**.

---

## Example: Swiss Roll

```python
from sklearn.datasets import make_swiss_roll
import matplotlib.pyplot as plt

# Generate Swiss roll data
X, color = make_swiss_roll(n_samples=1000, noise=0.1)

# 3D manifold embedded in 3D
# Intrinsic dimensionality is 2 (unrolled rectangle)
# But PCA cannot "unroll" it!
```

### PCA Failure

```
Original (3D):          PCA (2D):
                        
   ╭──────╮             ●●●●●●
  ╭──────╮│            ●●●●●●
 ╭──────╮│╯           ●●●●●●
 │●●●●●●│╯           ●●●●●●
 ╰──────╯            ●●●●●●
                     (overlapping!)
```

PCA projects to a plane, causing points to overlap that are far apart on the manifold.

---

## Types of Nonlinear Structure

### 1. Curved Manifolds

- Swiss roll, S-curve
- PCA: Projects to flat subspace, loses structure

### 2. Clusters on Manifolds

- Data clusters on a curved surface
- PCA: May merge distinct clusters

### 3. Hierarchical Structure

- Images: edges → textures → objects
- PCA: Single linear transformation, no hierarchy

---

## Quantitative Comparison

```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Swiss roll experiment
X_pca = PCA(n_components=2).fit_transform(X)
X_tsne = TSNE(n_components=2).fit_transform(X)

# Measure: Do neighbors in original space stay neighbors?
# PCA: Poor neighborhood preservation
# t-SNE/UMAP: Better neighborhood preservation
```

---

## What Autoencoders Offer

### Nonlinear Encoder/Decoder

$$z = f_{\text{encoder}}(x) \quad \text{(nonlinear)}$$
$$\hat{x} = g_{\text{decoder}}(z) \quad \text{(nonlinear)}$$

### Benefits

| Aspect | PCA | Autoencoder |
|--------|-----|-------------|
| **Manifolds** | Flat subspaces | Curved manifolds |
| **Features** | Linear combinations | Nonlinear features |
| **Hierarchy** | None | Multiple layers |
| **Flexibility** | Fixed form | Learnable |

---

## When PCA Still Works

Despite limitations, PCA is appropriate when:

1. **Data is approximately linear**
2. **Interpretability is crucial** (loadings have meaning)
3. **Small datasets** (autoencoders may overfit)
4. **Fast computation** needed (no training required)
5. **Baseline comparison** for other methods

---

## Transition to Autoencoders

### From Linear to Nonlinear

```python
# Linear autoencoder (≈ PCA)
encoder = nn.Linear(784, 32)
decoder = nn.Linear(32, 784)

# Nonlinear autoencoder (can learn manifolds)
encoder = nn.Sequential(
    nn.Linear(784, 256), nn.ReLU(),
    nn.Linear(256, 32)
)
decoder = nn.Sequential(
    nn.Linear(32, 256), nn.ReLU(),
    nn.Linear(256, 784)
)
```

Adding nonlinearities enables learning **arbitrary continuous mappings**.

---

## Summary

| Limitation | Consequence | Solution |
|------------|-------------|----------|
| **Linear only** | Cannot capture curves | Nonlinear autoencoders |
| **Global optimization** | Misses local structure | Deep architectures |
| **Single transformation** | No feature hierarchy | Stacked layers |
| **Fixed basis** | Limited expressiveness | Learned representations |

---

## Key Takeaway

PCA is optimal for **linear** dimensionality reduction. For real-world data with **nonlinear structure**, autoencoders and VAEs provide more expressive alternatives.

This motivates the study of autoencoders (Section 9.2) and variational autoencoders (Section 9.3).
