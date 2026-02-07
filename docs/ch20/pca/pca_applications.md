# PCA Applications

Practical implementations with scikit-learn and PyTorch, from toy examples to production pipelines.

---

## Overview

This section collects complete, runnable PCA applications that bridge the gap between the mathematical foundations developed in earlier sections and real-world usage. Each application illustrates a different facet of PCA: geometric projection, compression, denoising, feature extraction, and financial factor modeling. All examples include full data loading, fitting, and visualization code.

---

## Learning Objectives

After completing this section, you will be able to:

- Apply PCA with scikit-learn to real datasets and interpret the outputs
- Implement PCA as a linear autoencoder in PyTorch and verify convergence to the analytical solution
- Use PCA for image compression, noise filtering, and feature extraction (eigenfaces)
- Decompose yield curves into interpretable principal components (level, slope, curvature)
- Choose between sklearn's analytical PCA and PyTorch's learned PCA based on problem requirements

---

## PCA with scikit-learn: Quick Reference

The `sklearn.decomposition.PCA` class wraps the full pipeline — centering, SVD, projection, and reconstruction — into a consistent API:

```python
import numpy as np
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
Z = pca.fit_transform(X)            # Fit and project
X_reconstructed = pca.inverse_transform(Z)  # Reconstruct

# Key attributes after fitting
print(f"Components (loadings): {pca.components_.shape}")      # [k, d]
print(f"Explained variance:    {pca.explained_variance_}")     # [k]
print(f"Explained var ratio:   {pca.explained_variance_ratio_}")  # [k]
print(f"Singular values:       {pca.singular_values_}")        # [k]
print(f"Mean (for centering):  {pca.mean_.shape}")             # [d]
print(f"Noise variance:        {pca.noise_variance_}")         # scalar
```

The `n_components` parameter accepts either an integer (exact number of components) or a float in $(0, 1)$ (minimum explained variance ratio). The `svd_solver` parameter controls the algorithm: `'full'` for exact SVD, `'randomized'` for large datasets, `'arpack'` for sparse data.

---

## Application 1: Geometric Projection (2D → 1D)

The simplest PCA application: projecting correlated 2D data onto its principal axis.

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

np.random.seed(0)


def generate_correlated_data(n=200):
    """Generate 2D data with strong linear correlation."""
    x = np.random.normal(size=(n,))
    y = 0.5 * x + 2 + 0.1 * np.random.normal(size=(n,))
    return np.column_stack([x, y])


def plot_projection(X, X_reconstructed, pca):
    """Overlay original data, projected points, and principal axis."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Original data
    ax.scatter(X[:, 0], X[:, 1], alpha=0.3, s=30, label="Original data")

    # Projected points (on the principal axis)
    ax.scatter(X_reconstructed[:, 0], X_reconstructed[:, 1],
               color='red', s=20, label="Projected onto PC1")

    # Draw lines from original to projected
    for i in range(0, len(X), 10):
        ax.plot([X[i, 0], X_reconstructed[i, 0]],
                [X[i, 1], X_reconstructed[i, 1]],
                'gray', alpha=0.3, linewidth=0.5)

    # Draw principal axis
    mean = pca.mean_
    pc1 = pca.components_[0]
    scale = 3 * pca.singular_values_[0] / np.sqrt(len(X))
    ax.annotate('', xy=mean + scale * pc1, xytext=mean - scale * pc1,
                arrowprops=dict(arrowstyle='<->', color='darkred', lw=2))

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title('PCA Projection: 2D → 1D')
    ax.legend()
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()


X = generate_correlated_data()

pca = PCA(n_components=1).fit(X)
Z = pca.transform(X)                    # [200, 1] scores
X_reconstructed = pca.inverse_transform(Z)  # [200, 2] back in original space

print(f"Original shape:      {X.shape}")              # (200, 2)
print(f"Projected shape:     {Z.shape}")               # (200, 1)
print(f"Reconstructed shape: {X_reconstructed.shape}")  # (200, 2)
print(f"Explained variance:  {pca.explained_variance_ratio_[0]:.4f}")

plot_projection(X, X_reconstructed, pca)
```

The gray lines connecting original points to their projections illustrate the **minimum reconstruction error** property: PCA minimizes the sum of squared perpendicular distances to the projection subspace.

---

## Application 2: MNIST Compression

Compress 784-dimensional handwritten digit images while retaining 95% of the total variance:

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from tensorflow.keras.datasets import mnist


def load_mnist():
    """Load MNIST and flatten to [n, 784]."""
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(-1, 784).astype(np.float32)
    X_test = X_test.reshape(-1, 784).astype(np.float32)
    return X_train, y_train, X_test, y_test


def plot_comparison(X_original, X_recovered, n_images=10):
    """Side-by-side comparison of original and compressed images."""
    fig, axes = plt.subplots(2, n_images, figsize=(n_images * 1.5, 3),
                             subplot_kw={'xticks': [], 'yticks': []})
    for i in range(n_images):
        axes[0, i].imshow(X_original[i].reshape(28, 28), cmap='binary')
        axes[1, i].imshow(X_recovered[i].reshape(28, 28), cmap='binary')
    axes[0, 0].set_ylabel('Original', fontsize=12)
    axes[1, 0].set_ylabel('Compressed', fontsize=12)
    plt.suptitle('MNIST: Original vs. PCA-Compressed', fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_cumulative_variance(pca):
    """Plot cumulative explained variance with key thresholds."""
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(cumvar, linewidth=2)
    for threshold in [0.90, 0.95, 0.99]:
        k = np.searchsorted(cumvar, threshold) + 1
        ax.axhline(threshold, color='gray', ls='--', alpha=0.5)
        ax.axvline(k, color='gray', ls='--', alpha=0.5)
        ax.annotate(f'{threshold:.0%} → {k} components',
                    xy=(k, threshold), fontsize=9,
                    xytext=(k + 20, threshold - 0.03))
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Cumulative Explained Variance')
    ax.set_title('MNIST: Variance Explained by PCA Components')
    plt.tight_layout()
    plt.show()


X_train, y_train, X_test, y_test = load_mnist()

# Fit PCA retaining 95% variance
pca = PCA(n_components=0.95, svd_solver='full').fit(X_train)
X_reduced = pca.transform(X_test)
X_recovered = pca.inverse_transform(X_reduced)

print(f"Original dimension:   {X_test.shape[1]}")          # 784
print(f"Compressed dimension: {X_reduced.shape[1]}")        # ~154
print(f"Compression ratio:    {784 / X_reduced.shape[1]:.1f}x")
print(f"Reconstruction MSE:   {np.mean((X_test - X_recovered)**2):.2f}")

plot_comparison(X_test, X_recovered)
plot_cumulative_variance(pca)
```

MNIST digits live in a 784-dimensional pixel space, but most of this space is empty (background pixels) or redundant (correlated neighboring pixels). PCA discovers that roughly 150 directions capture 95% of the total pixel variance — a compression ratio exceeding 5×.

---

## Application 3: Noise Filtering

PCA denoises data by exploiting the observation that signal variance concentrates in the top components while noise variance spreads uniformly across all components. Projecting onto the high-variance subspace retains most of the signal while discarding disproportionately more noise.

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from tensorflow.keras.datasets import mnist

np.random.seed(0)


def plot_digit_grid(data, title, n_rows=4, n_cols=10):
    """Display a grid of digit images."""
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows),
                             subplot_kw={'xticks': [], 'yticks': []},
                             gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        if i < len(data):
            ax.imshow(data[i].reshape(28, 28),
                      cmap='binary', interpolation='nearest')
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


(X_train, _), _ = mnist.load_data()
X_train = X_train.reshape(-1, 784).astype(np.float32)

# Add substantial Gaussian noise
noise_level = 50.0
X_noisy = X_train + noise_level * np.random.normal(size=X_train.shape)

# Denoise with PCA (retain 90% variance of noisy data)
pca = PCA(n_components=0.90, svd_solver='full').fit(X_noisy)
X_denoised = pca.inverse_transform(pca.transform(X_noisy))

print(f"Components used:   {pca.n_components_}")
print(f"Noisy MSE:         {np.mean((X_train - X_noisy)**2):.1f}")
print(f"Denoised MSE:      {np.mean((X_train - X_denoised)**2):.1f}")

plot_digit_grid(X_train[:40], "Original Digits")
plot_digit_grid(X_noisy[:40], f"Noisy Digits (σ = {noise_level})")
plot_digit_grid(X_denoised[:40], f"PCA-Denoised ({pca.n_components_} components)")
```

**Why this works.** For isotropic Gaussian noise with variance $\sigma^2_\text{noise}$, each eigenvalue of the noisy covariance is inflated by $\sigma^2_\text{noise}$. The top eigenvalues are dominated by signal; the bottom eigenvalues are dominated by noise. Truncation removes the noise-dominated directions while keeping the signal-dominated ones.

---

## Application 4: EigenFaces

PCA applied to face images produces **eigenfaces** — the principal directions of variation in face space. These capture systematic modes like lighting direction, head pose, and expression.

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA

np.random.seed(0)


def plot_eigenfaces(pca, image_shape=(62, 47), n_rows=3, n_cols=8):
    """Display the top eigenfaces (principal components)."""
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5),
                             subplot_kw={'xticks': [], 'yticks': []},
                             gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow(pca.components_[i].reshape(image_shape), cmap='bone')
    plt.suptitle(f'Top {n_rows * n_cols} Eigenfaces', fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_reconstruction_comparison(faces, pca, n_faces=10):
    """Compare original faces with PCA reconstructions."""
    components = pca.transform(faces.data[:n_faces])
    reconstructed = pca.inverse_transform(components)

    fig, axes = plt.subplots(2, n_faces, figsize=(n_faces * 1.5, 3),
                             subplot_kw={'xticks': [], 'yticks': []})
    for i in range(n_faces):
        axes[0, i].imshow(faces.data[i].reshape(62, 47), cmap='binary_r')
        axes[1, i].imshow(reconstructed[i].reshape(62, 47), cmap='binary_r')
    axes[0, 0].set_ylabel('Original', fontsize=11)
    axes[1, 0].set_ylabel(f'{pca.n_components_}-dim\nreconstruction', fontsize=10)
    plt.suptitle('Face Reconstruction from Eigenface Basis', fontsize=14)
    plt.tight_layout()
    plt.show()


# Load face dataset
faces = fetch_lfw_people(min_faces_per_person=60)
print(f"Dataset shape: {faces.data.shape}")     # (n_people, 2914)
print(f"Image shape:   {faces.images.shape}")    # (n_people, 62, 47)
print(f"Identities:    {len(faces.target_names)}")

# Fit PCA with 150 components
pca = PCA(n_components=150, svd_solver='randomized').fit(faces.data)

print(f"\nVariance explained by 150 components: "
      f"{pca.explained_variance_ratio_.sum():.1%}")

# Visualize
plot_eigenfaces(pca)
plot_reconstruction_comparison(faces, pca)

# Cumulative variance plot
cumvar = np.cumsum(pca.explained_variance_ratio_)
plt.figure(figsize=(8, 4))
plt.plot(cumvar, linewidth=2)
plt.xlabel('Number of Eigenfaces')
plt.ylabel('Cumulative Explained Variance')
plt.title('Variance Captured by Eigenface Basis')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

**Interpretation.** The first eigenface typically captures overall brightness (a "DC component"). Subsequent eigenfaces isolate progressively finer modes of variation: lighting direction, left-right asymmetry, glasses versus no glasses, smiling versus neutral. Any face in the dataset can be approximately represented as:

$$\hat{\mathbf{x}}_\text{face} = \boldsymbol{\mu}_\text{face} + \sum_{k=1}^{150} z_k \, \mathbf{v}_k$$

where $z_k$ are the face's coordinates in the eigenface basis.

---

## Application 5: Linear Autoencoder in PyTorch

A complete PyTorch implementation that learns PCA via gradient descent. Training a linear autoencoder (no activations, no bias) with MSE loss converges to the same solution as analytical PCA.

```python
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torchvision import datasets, transforms

torch.manual_seed(0)


class PCAAutoencoder(nn.Module):
    """Linear autoencoder equivalent to PCA.

    Key constraints that ensure PCA equivalence:
    - No activation functions (purely linear)
    - No bias terms
    - MSE reconstruction loss
    """
    def __init__(self, input_dim=784, latent_dim=20):
        super().__init__()
        self.encoder = nn.Linear(input_dim, latent_dim, bias=False)
        self.decoder = nn.Linear(latent_dim, input_dim, bias=False)

    def forward(self, x):
        out = x.view(x.size(0), -1)        # Flatten
        out = self.encoder(out)              # Encode
        out = self.decoder(out)              # Decode
        return out.view(x.size())            # Reshape

    def encode(self, x):
        """Get latent codes (scores)."""
        return self.encoder(x.view(x.size(0), -1))


def get_data_loaders(batch_size=64):
    """Load MNIST with DataLoaders."""
    transform = transforms.ToTensor()
    train_data = datasets.MNIST('./data', train=True,
                                transform=transform, download=True)
    test_data = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def train_model(model, train_loader, test_loader, device,
                num_epochs=100, lr=0.01):
    """Training loop with periodic evaluation."""
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, test_losses = [], []
    best_test_loss = float('inf')
    patience, max_patience = 0, 50

    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_loss = 0.0
        for batch_X, _ in train_loader:
            batch_X = batch_X.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(batch_X), batch_X)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_loader))

        # Evaluation every 10 epochs
        if epoch % 10 == 0:
            model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for batch_X, _ in test_loader:
                    batch_X = batch_X.to(device)
                    test_loss += loss_fn(model(batch_X), batch_X).item()
            test_loss /= len(test_loader)
            test_losses.append(test_loss)

            marker = " *" if test_loss < best_test_loss else ""
            print(f"Epoch {epoch:3d}: train={train_losses[-1]:.5f}  "
                  f"test={test_loss:.5f}{marker}")

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                patience = 0
            else:
                patience += 1
            if patience >= max_patience:
                print("Early stopping.")
                break

    return train_losses, test_losses


def visualize_reconstructions(model, test_loader, device, n_images=8):
    """Display original vs. reconstructed images."""
    model.eval()
    batch_X, _ = next(iter(test_loader))
    batch_X = batch_X[:n_images].to(device)

    with torch.no_grad():
        reconstructed = model(batch_X).cpu()
    originals = batch_X.cpu()

    fig, axes = plt.subplots(2, n_images, figsize=(n_images * 1.5, 3),
                             subplot_kw={'xticks': [], 'yticks': []})
    for i in range(n_images):
        axes[0, i].imshow(originals[i].squeeze(), cmap='binary')
        axes[1, i].imshow(reconstructed[i].squeeze(), cmap='binary')
    axes[0, 0].set_ylabel('Original', fontsize=11)
    axes[1, 0].set_ylabel('Reconstructed', fontsize=11)
    plt.suptitle('Linear Autoencoder (PCA) Reconstruction', fontsize=14)
    plt.tight_layout()
    plt.show()


# --- Main ---
latent_dim = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}, Latent dim: {latent_dim}")

train_loader, test_loader = get_data_loaders()
model = PCAAutoencoder(784, latent_dim).to(device)

train_losses, test_losses = train_model(
    model, train_loader, test_loader, device, num_epochs=100)

visualize_reconstructions(model, test_loader, device)
```

---

## Empirical Verification: Analytical PCA vs. Linear Autoencoder

A direct comparison confirms that the learned linear autoencoder converges to the PCA reconstruction error:

```python
import numpy as np
import torch
import torch.nn as nn

np.random.seed(42)
torch.manual_seed(42)

# Generate random data
n, d, k = 1000, 50, 10
X = np.random.randn(n, d).astype(np.float32)
X_centered = X - X.mean(axis=0)

# --- Analytical PCA ---
cov = X_centered.T @ X_centered / n
eigenvalues = np.sort(np.linalg.eigh(cov)[0])[::-1]
pca_error = eigenvalues[k:].sum()

# --- Linear Autoencoder ---
X_t = torch.tensor(X_centered)

model = nn.Sequential(
    nn.Linear(d, k, bias=False),
    nn.Linear(k, d, bias=False)
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

for epoch in range(5000):
    optimizer.zero_grad()
    loss = ((X_t - model(X_t)) ** 2).mean()
    loss.backward()
    optimizer.step()

ae_error = loss.item() * d  # Convert MSE → total error per sample

print(f"Analytical PCA error:      {pca_error:.6f}")
print(f"Linear autoencoder error:  {ae_error:.6f}")
print(f"Relative difference:       {abs(pca_error - ae_error) / pca_error:.2e}")
```

The two errors should agree to within optimizer tolerance (typically $< 1\%$ relative error with sufficient training).

---

## sklearn vs. PyTorch Comparison

| Aspect | sklearn PCA | PyTorch Linear Autoencoder |
|--------|-------------|----------------------------|
| **Algorithm** | Analytical SVD | Iterative gradient descent |
| **Speed** | One-shot (fast) | Many epochs (slower) |
| **Exactness** | Exact global optimum | Converges to PCA solution |
| **GPU support** | CPU only (limited) | Native GPU acceleration |
| **Memory** | Full SVD of $\mathbf{X}$ | Mini-batch compatible |
| **Extensibility** | Fixed linear form | Add nonlinearity → autoencoder |
| **Missing data** | Not supported | Can mask loss terms |
| **Streaming data** | `IncrementalPCA` available | Natural with SGD |

**Use sklearn PCA when** you need the exact solution, your data fits in memory, and you don't need GPU acceleration. This covers most practical applications.

**Use a PyTorch linear autoencoder when** you are learning PCA concepts, plan to extend to nonlinear autoencoders, need GPU acceleration for very large datasets, or need mini-batch processing for data that doesn't fit in memory.

---

## Quantitative Finance Application: Yield Curve Decomposition

PCA applied to yield curves is one of the most established applications in fixed-income analytics. Daily changes in the yield curve can be decomposed into a small number of interpretable factors.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

np.random.seed(42)

# --- Simulate yield curve data ---
# In practice, use actual Treasury yield data from FRED or Bloomberg
n_days = 1000
maturities = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
n_maturities = len(maturities)

# Simulate realistic yield curve dynamics
# Three latent factors: level, slope, curvature
t = np.linspace(0, 1, n_maturities)
level_loading = np.ones(n_maturities)
slope_loading = 1 - 2 * t
curvature_loading = 4 * t * (1 - t)

# Random factor realizations
factors = np.column_stack([
    np.random.normal(0, 0.05, n_days),    # Level changes
    np.random.normal(0, 0.03, n_days),    # Slope changes
    np.random.normal(0, 0.015, n_days),   # Curvature changes
])
loadings_true = np.column_stack([level_loading, slope_loading, curvature_loading])

# Yield changes = factor realizations × loadings + idiosyncratic noise
delta_yields = factors @ loadings_true.T + 0.005 * np.random.randn(n_days, n_maturities)

# --- PCA on yield changes ---
pca = PCA(n_components=5).fit(delta_yields)

print("Explained variance by component:")
for i, evr in enumerate(pca.explained_variance_ratio_[:5]):
    print(f"  PC{i+1}: {evr:.1%}")
print(f"  Total (3 PCs): {pca.explained_variance_ratio_[:3].sum():.1%}")

# --- Plot factor loadings ---
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
factor_names = ['Level (PC1)', 'Slope (PC2)', 'Curvature (PC3)']
colors = ['#2196F3', '#FF5722', '#4CAF50']

for i, (ax, name, color) in enumerate(zip(axes, factor_names, colors)):
    loading = pca.components_[i]
    # Sign convention: PC1 should have positive loadings (parallel shift up)
    if loading.mean() < 0:
        loading = -loading
    ax.plot(maturities, loading, 'o-', color=color, linewidth=2, markersize=6)
    ax.set_xlabel('Maturity (years)')
    ax.set_ylabel('Loading')
    ax.set_title(f'{name} ({pca.explained_variance_ratio_[i]:.1%})')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linewidth=0.5)

plt.suptitle('PCA Yield Curve Factor Loadings', fontsize=14)
plt.tight_layout()
plt.show()
```

**Interpretation.** The three canonical yield curve factors are well-established in fixed-income literature:

**PC1 (Level):** Roughly uniform loadings across maturities. Represents a parallel shift of the entire yield curve. Typically explains 85–95% of daily yield variation. In risk management, this is the dominant exposure for bond portfolios.

**PC2 (Slope):** Positive loadings at the short end, negative at the long end (or vice versa). Represents a steepening or flattening of the curve. Typically explains 3–8% of variation. Trades that exploit this factor include curve flatteners and steepeners.

**PC3 (Curvature):** Positive loadings at intermediate maturities, negative at both ends (a "butterfly" shape). Represents bowing of the curve. Typically explains 1–3% of variation. Butterfly trades are designed to capture this factor.

Three components typically explain over 95% of yield curve movements, making PCA a cornerstone of fixed-income risk decomposition, relative-value trading, and curve interpolation.

---

## Summary

| Application | Purpose | Key Insight |
|-------------|---------|-------------|
| **2D → 1D projection** | Geometric visualization | Projection minimizes perpendicular distances |
| **MNIST compression** | Dimensionality reduction | 95% variance in ~150 of 784 components |
| **Noise filtering** | Denoising | Signal in top components, noise spread across all |
| **EigenFaces** | Feature extraction | Principal components = interpretable face modes |
| **Linear autoencoder** | Learned PCA | MSE + linear + no bias → converges to PCA |
| **Yield curve decomposition** | Factor modeling | Level, slope, curvature explain >95% of variation |

Each application demonstrates a different property of PCA: variance maximization (projection), optimal compression (MNIST), signal-noise separation (denoising), interpretable basis extraction (eigenfaces), equivalence to neural methods (autoencoder), and domain-specific factor modeling (yield curves).
