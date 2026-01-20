# PCA as Linear Autoencoder

The connection between classical PCA and neural network autoencoders.

---

## The Linear Autoencoder

### Architecture

A **linear autoencoder** has:
- Encoder: $z = W_e^T x$ (linear, no activation)
- Decoder: $\hat{x} = W_d z$ (linear, no activation)

### Loss Function

$$\mathcal{L} = \frac{1}{n}\sum_{i=1}^n \|x_i - W_d W_e^T x_i\|^2$$

---

## Equivalence to PCA

### Theorem

For a linear autoencoder trained with MSE loss:
1. Optimal encoder weights span the PCA subspace
2. Reconstruction equals PCA reconstruction
3. Loss equals PCA reconstruction error

**Key insight:** $W_d W_e^T$ converges to $W W^T$ where $W$ contains principal components.

### Proof Sketch

At optimum, $W_d = W_e$ (tied weights) and columns of $W_e$ are orthonormal eigenvectors of the covariance matrix.

---

## Implementation Comparison

### PCA (Analytical)

```python
def pca_analytical(X, k):
    """PCA via eigendecomposition."""
    X_centered = X - X.mean(axis=0)
    cov = X_centered.T @ X_centered / (X.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    idx = np.argsort(eigenvalues)[::-1][:k]
    W = eigenvectors[:, idx]
    
    Z = X_centered @ W
    X_recon = Z @ W.T + X.mean(axis=0)
    
    return X_recon, W
```

### Linear Autoencoder (Learned)

```python
import torch
import torch.nn as nn

class LinearAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, latent_dim, bias=False)
        self.decoder = nn.Linear(latent_dim, input_dim, bias=False)
    
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def train_linear_ae(X, latent_dim, epochs=1000, lr=0.01):
    """Train linear autoencoder."""
    X_tensor = torch.tensor(X, dtype=torch.float32)
    X_centered = X_tensor - X_tensor.mean(dim=0)
    
    model = LinearAutoencoder(X.shape[1], latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        recon = model(X_centered)
        loss = ((X_centered - recon) ** 2).mean()
        loss.backward()
        optimizer.step()
    
    return model
```

---

## Empirical Verification

```python
# Compare PCA and linear autoencoder
X = np.random.randn(1000, 50)  # Random data
k = 10

# PCA
X_pca, W_pca = pca_analytical(X, k)
error_pca = np.mean((X - X_pca) ** 2)

# Linear autoencoder
model = train_linear_ae(X, k, epochs=5000)
X_centered = torch.tensor(X - X.mean(axis=0), dtype=torch.float32)
X_ae = model(X_centered).detach().numpy() + X.mean(axis=0)
error_ae = np.mean((X - X_ae) ** 2)

print(f"PCA error: {error_pca:.6f}")
print(f"Linear AE error: {error_ae:.6f}")
# Should be approximately equal!
```

---

## Why Use Autoencoders Then?

### Advantages of Autoencoders

| Aspect | Linear AE / PCA | Nonlinear AE |
|--------|-----------------|--------------|
| **Expressiveness** | Linear subspace only | Arbitrary manifolds |
| **Complex data** | Poor for curves, clusters | Can capture nonlinear structure |
| **Flexibility** | Fixed architecture | Arbitrary encoder/decoder |

### When PCA Suffices

- Data lies near a linear subspace
- Interpretability is important
- Computational efficiency is critical
- Small datasets

### When Autoencoders Excel

- Nonlinear structure in data
- Image/audio/text data
- Deep feature hierarchies
- Generative modeling (VAEs)

---

## Summary

| Concept | Key Point |
|---------|-----------|
| **Linear AE** | Equivalent to PCA |
| **Loss** | Both minimize MSE reconstruction |
| **Solution** | Both find principal subspace |
| **Nonlinear AE** | Generalizes to curved manifolds |

---

## Bridge to Nonlinear Methods

Adding nonlinearities (ReLU, sigmoid) allows autoencoders to:

1. **Capture nonlinear structure** in data
2. **Learn hierarchical features**
3. **Model complex distributions** (with VAEs)

This motivates the transition from PCA to autoencoders to VAEs.
