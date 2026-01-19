# Limitations of Linear Methods

Understanding why PCA fails on nonlinear data and motivating autoencoders.

---

## Overview

**Key Insight:** Real-world data often lies on nonlinear manifolds that linear methods cannot capture.

**Motivation:** These limitations drive the need for nonlinear dimensionality reduction via autoencoders.

**Time:** ~20 minutes  
**Level:** Intermediate

---

## The Linearity Assumption

### What PCA Assumes

PCA assumes data lies near a **linear subspace**:

$$x \approx \mu + Wz$$

where $W$ defines a $k$-dimensional hyperplane through the data.

### When This Fails

**Nonlinear Manifolds:** Data may lie on curved surfaces (e.g., Swiss roll, sphere).

**Multiple Clusters:** Data with distinct clusters needs local linear approximations.

**Complex Dependencies:** Nonlinear relationships between features.

---

## Classic Examples of PCA Failure

### 1. Swiss Roll

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# Generate Swiss roll
n = 1500
t = 1.5 * np.pi * (1 + 2 * np.random.rand(n))
x = t * np.cos(t)
y = 21 * np.random.rand(n)
z = t * np.sin(t)
X = np.column_stack([x, y, z])
color = t  # Color by position on manifold

# PCA projection
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Visualization shows PCA fails to "unroll" the manifold
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=color, cmap='viridis', s=5)
axes[0].set_title('PCA Projection (fails to preserve structure)')
```

**Problem:** PCA projects onto a plane, mixing points that are far apart on the manifold.

### 2. Concentric Circles

```python
# Inner and outer circles
n = 500
theta = 2 * np.pi * np.random.rand(n)
r_inner = 1 + 0.1 * np.random.randn(n // 2)
r_outer = 3 + 0.1 * np.random.randn(n // 2)

X_inner = np.column_stack([r_inner * np.cos(theta[:n//2]), 
                            r_inner * np.sin(theta[:n//2])])
X_outer = np.column_stack([r_outer * np.cos(theta[n//2:]), 
                            r_outer * np.sin(theta[n//2:])])
X = np.vstack([X_inner, X_outer])
```

**Problem:** PCA cannot separate the circles—they overlap when projected to 1D.

### 3. XOR Pattern

```python
# XOR-like data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])  # XOR labels
```

**Problem:** No linear projection separates the classes.

---

## Theoretical Limitations

### Variance ≠ Information

PCA maximizes variance, but variance doesn't always capture meaningful structure:

| Data Structure | Variance Direction | Meaningful Direction |
|----------------|-------------------|---------------------|
| Swiss roll | Across the roll | Along the roll |
| Noisy manifold | Noise direction | Manifold tangent |
| Clusters | Between clusters | Within-cluster structure |

### Reconstruction Error Bounds

For nonlinear manifolds, the best linear approximation has error:

$$\mathcal{L}_{linear}^* \geq \mathcal{L}_{nonlinear}^*$$

The gap can be arbitrarily large depending on manifold curvature.

---

## What Autoencoders Offer

### Nonlinear Encoding

$$z = f_\theta(x) \quad \text{(nonlinear encoder)}$$
$$\hat{x} = g_\phi(z) \quad \text{(nonlinear decoder)}$$

### Universal Approximation

With sufficient capacity, autoencoders can approximate any continuous mapping, enabling them to:

1. **Unroll** curved manifolds
2. **Separate** entangled clusters  
3. **Capture** nonlinear dependencies

### Comparison

| Aspect | PCA | Autoencoder |
|--------|-----|-------------|
| Mapping | Linear | Nonlinear |
| Manifolds | Hyperplanes only | Arbitrary smooth manifolds |
| Optimization | Closed-form | Gradient descent |
| Computation | Fast | Slower |
| Interpretability | High (eigenvectors) | Lower (learned weights) |

---

## When to Use PCA vs Autoencoders

### Use PCA When:
- Data is approximately linear
- Interpretability is important
- Computational resources are limited
- Quick baseline is needed

### Use Autoencoders When:
- Data lies on nonlinear manifolds
- High reconstruction quality is needed
- Sufficient training data is available
- Computational resources permit

---

## Exercises

### Exercise 1: Swiss Roll Comparison
Implement both PCA and an autoencoder on Swiss roll data. Compare reconstruction errors and visualize projections.

### Exercise 2: Measure Nonlinearity
Design a metric to quantify how "nonlinear" a dataset is. Test on various synthetic datasets.

### Exercise 3: Hybrid Approach
Implement PCA preprocessing followed by a shallow autoencoder. Compare with deep autoencoder alone.

---

## Summary

| Limitation | Description | Autoencoder Solution |
|------------|-------------|---------------------|
| **Linearity** | Can only find hyperplanes | Learns curved manifolds |
| **Global** | Single linear projection | Local nonlinear mappings |
| **Variance focus** | May miss structure | Learns task-relevant features |

---

## Next: Autoencoders

The next section introduces autoencoders for nonlinear dimensionality reduction.
