# PCA Fundamentals

Linear dimensionality reduction through variance maximization.

---

## Overview

**Principal Component Analysis (PCA)** is the most widely used technique for linear dimensionality reduction. Given data in $\mathbb{R}^d$, PCA finds an orthogonal set of $k < d$ directions along which the data exhibits maximum variance, then projects onto this lower-dimensional subspace. The result is an optimal linear compression in the sense of minimizing mean squared reconstruction error.

Understanding PCA is essential background for autoencoders and variational autoencoders, which generalize PCA to nonlinear settings.

---

## Learning Objectives

After completing this section, you will be able to:

- Formulate PCA as a constrained variance-maximization problem and derive the eigenvector solution
- Implement PCA via eigendecomposition and SVD, understanding the trade-offs of each approach
- Quantify information loss through reconstruction error and explained variance ratios
- Interpret loadings (principal directions) and scores (projected coordinates) in applied settings
- Connect PCA to linear autoencoders and understand why nonlinear extensions are necessary
- Apply PCA to practical problems including image compression, denoising, and feature extraction

---

## Geometric Intuition

PCA answers a simple geometric question: given a cloud of points in high-dimensional space, what is the best low-dimensional "flat" (affine subspace) to project them onto?

"Best" here means preserving as much of the data's spread as possible. The first principal component captures the direction of greatest variance; the second captures the most variance orthogonal to the first; and so on. Each successive direction explains progressively less variance.

For centered data $\mathbf{X} \in \mathbb{R}^{n \times d}$ with covariance $\boldsymbol{\Sigma} = \frac{1}{n}\mathbf{X}^T\mathbf{X}$, the variance of projections onto a unit vector $\mathbf{v}$ is:

$$\operatorname{Var}(\mathbf{X}\mathbf{v}) = \mathbf{v}^T \boldsymbol{\Sigma} \mathbf{v}$$

Maximizing this subject to $\|\mathbf{v}\| = 1$ yields the eigenvector of $\boldsymbol{\Sigma}$ with the largest eigenvalue. The full PCA solution consists of the top-$k$ eigenvectors.

---

## Key Equations at a Glance

| Concept | Formula |
|---------|---------|
| **Covariance matrix** | $\boldsymbol{\Sigma} = \frac{1}{n}\mathbf{X}^T\mathbf{X}$ |
| **Eigenproblem** | $\boldsymbol{\Sigma}\mathbf{v} = \lambda \mathbf{v}$ |
| **Projection (scores)** | $\mathbf{z} = \mathbf{W}^T \mathbf{x}$ |
| **Loadings** | $\mathbf{W} = [\mathbf{v}_1, \ldots, \mathbf{v}_k]$ |
| **Reconstruction** | $\hat{\mathbf{x}} = \mathbf{W}\mathbf{W}^T \mathbf{x}$ |
| **Reconstruction error** | $\sum_{i=k+1}^{d} \lambda_i$ |
| **SVD decomposition** | $\mathbf{X} = \mathbf{U}\mathbf{S}\mathbf{V}^T$ |
| **Explained variance ratio** | $\text{EVR}_k = \lambda_k / \sum_{i=1}^d \lambda_i$ |

---

## Preprocessing

Before computing principal components, data must be preprocessed. The covariance matrix depends on the scale and location of features, so failing to preprocess can produce misleading results.

### Mean Centering (Required)

Subtract the sample mean from each observation:

$$\boldsymbol{\mu} = \frac{1}{n}\sum_{i=1}^n \mathbf{x}^{(i)}, \qquad \mathbf{x}^{(i)} \leftarrow \mathbf{x}^{(i)} - \boldsymbol{\mu}$$

Mean centering ensures that the covariance matrix captures variance rather than the location of the data cloud. Without centering, the first principal component would simply point toward the data centroid.

### Feature Scaling (Recommended)

When features have different units or vastly different magnitudes, standardize each feature to unit variance:

$$\sigma_j^2 = \frac{1}{n}\sum_{i=1}^n \left(x_j^{(i)}\right)^2, \qquad x_j^{(i)} \leftarrow x_j^{(i)} / \sigma_j$$

Without scaling, PCA on a dataset with features measured in meters and kilometers would be dominated by the kilometer-scale features. Standardization makes PCA equivalent to eigendecomposition of the **correlation matrix** rather than the covariance matrix.

**Exception:** When all features share the same units and scale (e.g., pixel intensities in images), scaling may not be necessary and can even be counterproductive.

```python
def preprocess(X):
    """Center and scale data for PCA."""
    mu = X.mean(axis=0)
    X_centered = X - mu

    sigma = X_centered.std(axis=0)
    X_scaled = X_centered / (sigma + 1e-10)  # Avoid division by zero

    return X_scaled, mu, sigma
```

---

## Loadings and Scores

PCA produces two fundamental outputs with specific names in the statistical literature.

### Loadings (Principal Directions)

**Loadings** are the coefficients that define each principal component as a linear combination of original features. For the $k$-th principal component:

$$\text{Loading}_k = \mathbf{v}_k = [v_{k1}, v_{k2}, \ldots, v_{kd}]^T$$

Each loading coefficient $v_{kj}$ represents the contribution of feature $j$ to component $k$. Large absolute values indicate strong influence; the sign indicates the direction of contribution. Since eigenvectors are unit vectors, we have $\|\mathbf{v}_k\| = 1$.

The full loading matrix $\mathbf{W} = [\mathbf{v}_1, \ldots, \mathbf{v}_k] \in \mathbb{R}^{d \times k}$ has principal components as columns.

### Scores (Projected Coordinates)

**Scores** are the coordinates of data points in the principal component space. For sample $\mathbf{x}^{(i)}$ and component $k$:

$$z_{ik} = {\mathbf{x}^{(i)}}^T \mathbf{v}_k = \sum_{j=1}^d x_j^{(i)} v_{kj}$$

The full score matrix is:

$$\mathbf{Z} = \mathbf{X} \mathbf{W} \in \mathbb{R}^{n \times k}$$

Scores have two important properties: they are **uncorrelated** ($\operatorname{Cov}(z_i, z_j) = 0$ for $i \neq j$), and the variance of the $k$-th score equals the $k$-th eigenvalue ($\operatorname{Var}(z_k) = \lambda_k$).

### Projection and Reconstruction

The relationship between loadings and scores gives us both projection (encoding) and reconstruction (decoding):

$$\text{Projection:} \quad \mathbf{Z} = \mathbf{X}\mathbf{W}$$

$$\text{Reconstruction:} \quad \hat{\mathbf{X}} = \mathbf{Z}\mathbf{W}^T = \mathbf{X}\mathbf{W}\mathbf{W}^T$$

Each reconstructed sample is a sum of weighted principal directions:

$$\hat{\mathbf{x}}^{(i)} = \sum_{k=1}^{K} z_{ik} \, \mathbf{v}_k = \sum_{k=1}^{K} \left({\mathbf{x}^{(i)}}^T \mathbf{v}_k\right) \mathbf{v}_k$$

### Scaled Loadings (Correlation Loadings)

In some applications, loadings are scaled by the square root of the corresponding eigenvalue:

$$\text{Scaled Loading}_{kj} = v_{kj} \cdot \sqrt{\lambda_k}$$

Scaled loadings represent the **correlation** between original features and principal components. They are useful for interpretation because the sum of squared scaled loadings for each feature equals the communality (proportion of variance explained by the retained components).

```python
def correlation_loadings(loadings, eigenvalues):
    """Scale loadings to represent correlations with PCs."""
    return loadings * np.sqrt(eigenvalues)[:, np.newaxis]
```

---

## Reconstruction Error and Choosing $k$

### Quantifying Information Loss

When we keep only $k$ out of $d$ principal components, we lose information. The reconstruction error quantifies this loss:

$$\mathcal{E}_k = \frac{1}{n}\sum_{i=1}^n \left\|\mathbf{x}^{(i)} - \hat{\mathbf{x}}^{(i)}\right\|^2 = \sum_{j=k+1}^{d} \lambda_j$$

**The total reconstruction error equals the sum of discarded eigenvalues.** This elegant result follows directly from the fact that eigenvectors form an orthonormal basis: the error decomposes neatly into the variance along each discarded direction.

### Optimality

PCA minimizes reconstruction error among **all** rank-$k$ linear projections:

$$\mathbf{W}^* = \arg\min_{\mathbf{W}} \sum_{i=1}^n \left\|\mathbf{x}^{(i)} - \mathbf{W}\mathbf{W}^T \mathbf{x}^{(i)}\right\|^2 \quad \text{s.t.} \quad \mathbf{W}^T \mathbf{W} = \mathbf{I}$$

No other linear projection of the same rank can achieve lower error.

### Explained Variance Ratio

The fraction of total variance captured by component $k$:

$$\text{EVR}_k = \frac{\lambda_k}{\sum_{i=1}^d \lambda_i}$$

The cumulative explained variance by the first $k$ components:

$$\text{Cumulative EVR}_k = \frac{\sum_{i=1}^k \lambda_i}{\sum_{i=1}^d \lambda_i}$$

### Choosing the Number of Components

Three common strategies for selecting $k$:

**Variance threshold.** Choose the smallest $k$ such that cumulative EVR exceeds a desired threshold (commonly 90% or 95%):

```python
def choose_n_components(eigenvalues, threshold=0.95):
    """Find minimum k for desired explained variance."""
    total = eigenvalues.sum()
    cumsum = np.cumsum(eigenvalues)
    k = np.searchsorted(cumsum / total, threshold) + 1
    return k
```

**Scree plot.** Plot eigenvalues versus component index and look for an "elbow" — a sharp drop-off after which eigenvalues are approximately flat. Components before the elbow capture signal; those after capture noise.

**Reconstruction error budget.** Set a maximum acceptable per-sample MSE and choose the smallest $k$ that stays under this budget:

```python
def choose_by_error(eigenvalues, max_error):
    """Find minimum k for acceptable reconstruction error."""
    cumsum_discarded = eigenvalues.sum() - np.cumsum(eigenvalues)
    k = np.searchsorted(-cumsum_discarded, -max_error) + 1
    return k
```

```python
def plot_scree(eigenvalues):
    """Plot eigenvalues and cumulative explained variance."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(eigenvalues, 'o-')
    ax1.set_xlabel('Component')
    ax1.set_ylabel('Eigenvalue')
    ax1.set_title('Scree Plot')

    cumsum = np.cumsum(eigenvalues) / eigenvalues.sum()
    ax2.plot(cumsum, 'o-')
    ax2.axhline(0.95, color='r', linestyle='--', label='95%')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Explained Variance')
    ax2.legend()

    plt.tight_layout()
    plt.show()
```

---

## Connection to Linear Autoencoders

PCA has a precise equivalence to a linear autoencoder trained with MSE loss.

### Architecture

A linear autoencoder consists of:

- **Encoder:** $\mathbf{z} = \mathbf{W}_e^T \mathbf{x}$ (no bias, no activation)
- **Decoder:** $\hat{\mathbf{x}} = \mathbf{W}_d \mathbf{z}$ (no bias, no activation)

The training objective is:

$$\mathcal{L} = \frac{1}{n}\sum_{i=1}^n \left\|\mathbf{x}^{(i)} - \mathbf{W}_d \mathbf{W}_e^T \mathbf{x}^{(i)}\right\|^2$$

### Equivalence Theorem

At convergence, a linear autoencoder trained with MSE loss satisfies:

1. The encoder weights span the same subspace as the top-$k$ eigenvectors of $\boldsymbol{\Sigma}$
2. The optimal solution has tied weights: $\mathbf{W}_d = \mathbf{W}_e$
3. The reconstruction equals the PCA reconstruction
4. The loss equals the PCA reconstruction error

The matrix $\mathbf{W}_d \mathbf{W}_e^T$ converges to $\mathbf{W}\mathbf{W}^T$ where $\mathbf{W}$ contains the principal components.

```python
import torch
import torch.nn as nn

class LinearAutoencoder(nn.Module):
    """Linear autoencoder equivalent to PCA.

    With MSE loss, no activations, and no bias, training
    converges to the PCA solution.
    """
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, latent_dim, bias=False)
        self.decoder = nn.Linear(latent_dim, input_dim, bias=False)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
```

| Aspect | PCA (Analytical) | Linear Autoencoder |
|--------|------------------|--------------------|
| **Method** | Eigendecomposition / SVD | Gradient descent |
| **Speed** | One-shot (fast) | Iterative (slower) |
| **Exactness** | Exact solution | Converges to PCA |
| **GPU support** | Limited | Native |
| **Extensibility** | Fixed linear form | Easy to add nonlinearity |

---

## Limitations of PCA

PCA finds **linear subspaces**. Real data often lies on **nonlinear manifolds**, and this mismatch represents PCA's fundamental limitation.

### Failure on Nonlinear Structure

Consider the Swiss roll: a 2D surface embedded in 3D space. PCA projects onto a flat 2D plane, causing points that are far apart on the manifold to overlap in the projection:

```python
from sklearn.datasets import make_swiss_roll

X, color = make_swiss_roll(n_samples=1000, noise=0.1)
# Intrinsic dimensionality is 2, but PCA cannot "unroll" it
```

### Types of Structure PCA Misses

**Curved manifolds** (Swiss roll, S-curve): PCA projects to flat subspaces, collapsing the manifold structure. **Clusters on manifolds**: PCA may merge distinct clusters that lie on a curved surface. **Hierarchical features** (edges → textures → objects in images): PCA applies a single linear transformation, capturing no hierarchy.

### When PCA Still Works

Despite these limitations, PCA is appropriate when: the data is approximately linear; interpretability of components is important; the dataset is small (autoencoders may overfit); computation must be fast; or a baseline is needed for comparison with nonlinear methods.

### Transition to Nonlinear Methods

Adding nonlinear activation functions transforms a linear autoencoder into a nonlinear one capable of learning curved manifolds:

```python
# Linear (≈ PCA)
encoder = nn.Linear(784, 32)
decoder = nn.Linear(32, 784)

# Nonlinear (can learn manifolds)
encoder = nn.Sequential(
    nn.Linear(784, 256), nn.ReLU(),
    nn.Linear(256, 32)
)
decoder = nn.Sequential(
    nn.Linear(32, 256), nn.ReLU(),
    nn.Linear(256, 784)
)
```

| Aspect | PCA | Nonlinear Autoencoder |
|--------|-----|-----------------------|
| **Manifolds** | Flat subspaces only | Curved manifolds |
| **Features** | Linear combinations | Nonlinear features |
| **Hierarchy** | None | Multiple layers |
| **Solution** | Analytical (exact) | Learned (approximate) |

---

## Practical Applications

### Application 1: Dimensionality Reduction (2D → 1D)

A minimal example showing how PCA projects correlated 2D data onto a 1D line:

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

np.random.seed(0)

# Generate correlated 2D data
x = np.random.normal(size=(200,))
y = 0.5 * x + 2 + 0.1 * np.random.normal(size=(200,))
X = np.column_stack([x, y])

# Reduce to 1D and reconstruct
pca = PCA(n_components=1).fit(X)
X_pca = pca.transform(X)
X_reconstructed = pca.inverse_transform(X_pca)

print(f"Original shape:      {X.shape}")           # (200, 2)
print(f"Projected shape:     {X_pca.shape}")        # (200, 1)
print(f"Reconstructed shape: {X_reconstructed.shape}")  # (200, 2)

# Plot original vs. projected data
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(X[:, 0], X[:, 1], alpha=0.3, label="Original")
ax.scatter(X_reconstructed[:, 0], X_reconstructed[:, 1],
           color='red', s=20, label="Projected")
ax.legend()
plt.show()
```

### Application 2: MNIST Compression

Compress 784-dimensional handwritten digit images while retaining 95% of the variance:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tensorflow.keras.datasets import mnist

(X_train, _), (X_test, _) = mnist.load_data()
X_train = X_train.reshape(-1, 784).astype(np.float32)
X_test = X_test.reshape(-1, 784).astype(np.float32)

# PCA with 95% variance retention
pca = PCA(n_components=0.95, svd_solver='full').fit(X_train)
X_reduced = pca.transform(X_test)
X_recovered = pca.inverse_transform(X_reduced)

print(f"Original dim:    {X_test.shape[1]}")       # 784
print(f"Reduced dim:     {X_reduced.shape[1]}")     # ~150
print(f"Compression:     {784 / X_reduced.shape[1]:.1f}x")
```

### Application 3: Noise Filtering

PCA denoises data by projecting onto the high-variance subspace, discarding low-variance components that capture mostly noise:

```python
# Add Gaussian noise to MNIST
X_noisy = X_train + 10.0 * np.random.normal(size=X_train.shape)

# Fit PCA on noisy data, keeping 90% variance
pca = PCA(n_components=0.9, svd_solver='full').fit(X_noisy)
X_filtered = pca.inverse_transform(pca.transform(X_noisy))

print(f"Components used for denoising: {pca.n_components_}")
```

The key insight is that signal variance concentrates in the top components while noise variance spreads evenly across all components. Truncating removes disproportionately more noise than signal.

### Application 4: EigenFaces

PCA applied to face images produces **eigenfaces** — the principal component directions in face space:

```python
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA

faces = fetch_lfw_people(min_faces_per_person=60)
print(f"Dataset: {faces.data.shape}")  # (n_people, 62*47)

# Fit PCA with 150 components
pca = PCA(n_components=150, svd_solver='randomized').fit(faces.data)

# Each component is an "eigenface"
eigenface_0 = pca.components_[0].reshape(62, 47)

# Reconstruct faces from 150 components
components = pca.transform(faces.data)
reconstructed = pca.inverse_transform(components)
```

Each eigenface captures a mode of variation across the face dataset (lighting direction, head pose, expression). Any face can be approximately represented as a weighted sum of eigenfaces.

### Application 5: PCA as Linear Autoencoder in PyTorch

Training a linear autoencoder with MSE loss converges to the PCA solution:

```python
import torch
from torch import nn, optim
from torchvision import datasets, transforms

torch.manual_seed(0)

class PCAAutoencoder(nn.Module):
    """Linear autoencoder (no activation, no bias) ≡ PCA."""
    def __init__(self, input_dim=784, latent_dim=20):
        super().__init__()
        self.encoder = nn.Linear(input_dim, latent_dim, bias=False)
        self.decoder = nn.Linear(latent_dim, input_dim, bias=False)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.decoder(self.encoder(out))
        return out.view(x.size())

# Training
transform = transforms.ToTensor()
train_data = datasets.MNIST('./data', train=True, transform=transform,
                             download=True)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64,
                                            shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PCAAutoencoder(784, 20).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

for epoch in range(100):
    epoch_loss = 0.0
    for batch_X, _ in train_loader:
        batch_X = batch_X.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(batch_X), batch_X)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: loss={epoch_loss / len(train_loader):.4f}")
```

---

## Interpreting PCA Results

### Biplot

A biplot overlays scores (sample positions) and loadings (feature arrows) on the same axes, enabling simultaneous interpretation:

```python
import matplotlib.pyplot as plt

def biplot(scores, loadings, feature_names, labels=None):
    """PCA biplot: scores as scatter, loadings as arrows."""
    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(scores[:, 0], scores[:, 1],
                         c=labels, alpha=0.6, s=50)

    scale = np.abs(scores).max() * 0.8
    for load, name in zip(loadings.T, feature_names):
        ax.arrow(0, 0, load[0] * scale, load[1] * scale,
                 head_width=0.05 * scale, fc='red', ec='red', alpha=0.7)
        ax.text(load[0] * scale * 1.1, load[1] * scale * 1.1,
                name, fontsize=10, ha='center')

    ax.axhline(0, color='gray', ls='--', alpha=0.3)
    ax.axvline(0, color='gray', ls='--', alpha=0.3)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('PCA Biplot')
    plt.tight_layout()
    plt.show()
```

### Interpreting Loadings

Inspect which features contribute most to each component:

```python
def interpret_loadings(loadings, feature_names, n_top=5):
    """Show top contributing features per component."""
    for k in range(loadings.shape[0]):
        abs_load = np.abs(loadings[k])
        top_idx = np.argsort(abs_load)[::-1][:n_top]
        print(f"\n=== PC{k+1} ===")
        for idx in top_idx:
            print(f"  {feature_names[idx]}: {loadings[k, idx]:.4f}")
```

---

## Quick Reference Implementation

```python
import torch
import numpy as np

def pca(X, n_components):
    """
    PCA via eigendecomposition.

    Args:
        X: Data matrix [n_samples, n_features]
        n_components: Number of principal components

    Returns:
        W: Principal components (loadings) [n_features, n_components]
        Z: Projected data (scores) [n_samples, n_components]
        eigenvalues: Variance explained per component
    """
    # Center data
    X_centered = X - X.mean(dim=0)

    # Covariance matrix
    cov = X_centered.T @ X_centered / (X.shape[0] - 1)

    # Eigendecomposition (eigh for symmetric matrices)
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)

    # Sort by descending eigenvalue
    idx = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[idx[:n_components]]
    W = eigenvectors[:, idx[:n_components]]

    # Project (compute scores)
    Z = X_centered @ W

    return W, Z, eigenvalues
```

---

## Summary

PCA provides the foundational framework for linear dimensionality reduction. Its key properties — variance maximization, minimum reconstruction error, and analytical solvability — make it both a practical tool and a theoretical baseline. The equivalence between PCA and linear autoencoders establishes the bridge to deep generative models: autoencoders and VAEs can be understood as nonlinear generalizations of PCA.

The subsequent sections derive PCA rigorously from the variance-maximization perspective, detail the eigendecomposition and SVD computational approaches, and extend to probabilistic and kernel formulations.
