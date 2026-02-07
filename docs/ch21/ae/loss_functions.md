# Autoencoder Loss Functions

Reconstruction losses, regularization objectives, and specialized training criteria for autoencoder variants.

---

## Overview

**What you'll learn:**

- Reconstruction loss functions: MSE, BCE, MAE
- Sparsity penalties: L1 regularization and KL divergence
- Contractive penalty: Jacobian norm regularization
- Denoising objective: learning from corrupted inputs
- How loss function choice affects learned representations

---

## Part 1: Reconstruction Loss Functions

All autoencoders share a common foundation — minimizing the difference between input $x$ and reconstruction $\hat{x}$. The choice of loss function encodes assumptions about the data distribution.

### Mean Squared Error (MSE)

$$\mathcal{L}_{MSE} = \frac{1}{n} \sum_{i=1}^{n} \|x_i - \hat{x}_i\|^2$$

**Properties:**

- Penalizes large errors more than small errors (quadratic)
- Assumes Gaussian noise model: $p(x|\hat{x}) \propto \exp(-\|x - \hat{x}\|^2 / 2\sigma^2)$
- Tends to produce **blurry** reconstructions (averages over modes)
- Works with any output activation

### Binary Cross-Entropy (BCE)

$$\mathcal{L}_{BCE} = -\frac{1}{n} \sum_{i=1}^{n} [x_i \log(\hat{x}_i) + (1-x_i) \log(1-\hat{x}_i)]$$

**Properties:**

- Natural for binary/normalized data in $[0, 1]$
- Requires **sigmoid** output activation
- Interprets pixel values as Bernoulli probabilities
- Often gives **sharper** reconstructions than MSE

### Mean Absolute Error (MAE / L1)

$$\mathcal{L}_{MAE} = \frac{1}{n} \sum_{i=1}^{n} |x_i - \hat{x}_i|$$

**Properties:**

- More robust to outliers than MSE (linear penalty)
- Produces sparser gradients (constant magnitude)
- Can lead to sharper edges in reconstructions
- Assumes Laplace noise model

### Comparison and Implementation

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class AutoencoderWithAnalysis(nn.Module):
    """Autoencoder with methods for latent space analysis."""
    
    def __init__(self, input_dim=784, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


def compare_loss_functions(model_class, train_loader, test_loader, device):
    """Compare different reconstruction loss functions."""
    
    losses = {
        'MSE': nn.MSELoss(),
        'BCE': nn.BCELoss(),
        'L1': nn.L1Loss()
    }
    
    results = {}
    
    for loss_name, criterion in losses.items():
        print(f"\nTraining with {loss_name} loss...")
        
        model = model_class().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        train_losses = []
        for epoch in range(10):
            model.train()
            epoch_loss = 0
            for images, _ in train_loader:
                images = images.view(images.size(0), -1).to(device)
                
                optimizer.zero_grad()
                recon, _ = model(images)
                loss = criterion(recon, images)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            train_losses.append(epoch_loss / len(train_loader))
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            test_images, _ = next(iter(test_loader))
            test_images = test_images[:10].view(10, -1).to(device)
            recon, _ = model(test_images)
        
        results[loss_name] = {
            'model': model,
            'train_losses': train_losses,
            'reconstructions': recon.cpu().numpy()
        }
    
    return results
```

### Optimal Latent Dimension Search

```python
def find_optimal_latent_dim(train_loader, test_loader, device, 
                            dims=[2, 4, 8, 16, 32, 64, 128, 256]):
    """Find optimal latent dimension by reconstruction error."""
    
    results = []
    
    for dim in dims:
        model = AutoencoderWithAnalysis(latent_dim=dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Train
        for epoch in range(15):
            model.train()
            for images, _ in train_loader:
                images = images.view(images.size(0), -1).to(device)
                optimizer.zero_grad()
                recon, _ = model(images)
                loss = criterion(recon, images)
                loss.backward()
                optimizer.step()
        
        # Evaluate
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.view(images.size(0), -1).to(device)
                recon, _ = model(images)
                test_loss += criterion(recon, images).item()
        
        test_loss /= len(test_loader)
        results.append({'dim': dim, 'error': test_loss})
        print(f"Latent dim {dim}: Test MSE = {test_loss:.6f}")
    
    return results
```

---

## Part 2: Sparsity Penalties

Sparse autoencoders add a penalty that encourages most latent activations to be zero, leading to more interpretable and overcomplete representations.

### Standard vs Sparse Autoencoder Loss

| Type | Loss Function |
|------|---------------|
| Standard | $\mathcal{L} = \|x - f(x)\|^2$ |
| Sparse (L1) | $\mathcal{L} = \|x - f(x)\|^2 + \lambda \sum_j |h_j|$ |
| Sparse (KL) | $\mathcal{L} = \|x - f(x)\|^2 + \beta \sum_j \text{KL}(\rho \| \hat{\rho}_j)$ |

### L1 Regularization

$$\mathcal{L} = \|x - f(x)\|^2 + \lambda \sum_j |h_j|$$

Where:

- $h_j$ is the activation of neuron $j$ in the latent layer
- $\lambda$ is sparsity regularization strength
- $\sum_j |h_j|$ encourages many activations to be exactly zero

**Why sparsity helps:**

1. **Selective feature activation** — each input activates only relevant features
2. **Interpretable representations** — features correspond to meaningful patterns
3. **Robustness to noise** — sparse codes are more stable
4. **Better generalization** — prevents overfitting

### KL Divergence Sparsity

$$\mathcal{L} = \|x - f(x)\|^2 + \beta \sum_j \text{KL}(\rho \| \hat{\rho}_j)$$

Where:

- $\rho$ is the target sparsity level (e.g., 0.05 means 5% average activation)
- $\hat{\rho}_j = \frac{1}{n}\sum_{i=1}^n h_j(x_i)$ is the average activation of neuron $j$
- $\text{KL}(\rho \| \hat{\rho}_j) = \rho \log\frac{\rho}{\hat{\rho}_j} + (1-\rho) \log\frac{1-\rho}{1-\hat{\rho}_j}$

The KL divergence is minimized when $\hat{\rho}_j = \rho$, providing **precise control** over the target activation level.

### Implementation

```python
class SparseAutoencoder_L1(nn.Module):
    """
    Sparse Autoencoder using L1 regularization on latent activations.
    
    Loss = Reconstruction Loss + λ × L1(latent activations)
    
    The L1 penalty encourages many latent activations to be exactly zero.
    """
    
    def __init__(self, input_dim: int = 784, latent_dim: int = 128):
        super(SparseAutoencoder_L1, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder: Often larger latent_dim for overcomplete representations
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
            nn.ReLU()  # ReLU naturally promotes sparsity
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent


def l1_loss(latent: torch.Tensor) -> torch.Tensor:
    """
    Compute L1 penalty on latent activations.
    
    L1(h) = Σᵢⱼ |hᵢⱼ|
    
    Encourages sparsity by penalizing non-zero activations.
    """
    return torch.mean(torch.abs(latent))


class SparseAutoencoder_KL(nn.Module):
    """
    Sparse Autoencoder using KL divergence sparsity constraint.
    
    Constrains the average activation of each neuron to be close
    to a target sparsity level ρ (e.g., 0.05).
    """
    
    def __init__(self, input_dim: int = 784, latent_dim: int = 128):
        super(SparseAutoencoder_KL, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder with Sigmoid for KL divergence (outputs in [0,1])
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
            nn.Sigmoid()  # Required for KL divergence
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent


def kl_divergence_loss(latent: torch.Tensor, rho: float = 0.05) -> torch.Tensor:
    """
    Compute KL divergence sparsity penalty.
    
    For each neuron j, we want its average activation ρ̂ⱼ ≈ ρ.
    
    KL(ρ || ρ̂ⱼ) = ρ log(ρ/ρ̂ⱼ) + (1-ρ) log((1-ρ)/(1-ρ̂ⱼ))
    
    Minimized when ρ̂ⱼ = ρ.
    """
    # Average activation for each neuron across batch
    rho_hat = torch.mean(latent, dim=0)
    
    # Avoid log(0)
    eps = 1e-8
    rho_hat = torch.clamp(rho_hat, eps, 1 - eps)
    
    # KL divergence for each neuron
    kl = rho * torch.log(rho / rho_hat) + \
         (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))
    
    return torch.sum(kl)


def train_sparse_autoencoder(
    model, train_loader, optimizer, device, epoch,
    sparsity_type='l1', sparsity_weight=0.001, rho=0.05
):
    """
    Train sparse autoencoder for one epoch.
    
    Total Loss = Reconstruction Loss + Sparsity Penalty
    """
    model.train()
    
    total_loss_sum = 0.0
    recon_loss_sum = 0.0
    sparsity_loss_sum = 0.0
    num_batches = 0
    
    recon_criterion = nn.MSELoss()
    
    for batch_idx, (images, _) in enumerate(train_loader):
        images = images.view(images.size(0), -1).to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        reconstructed, latent = model(images)
        
        # Reconstruction loss
        recon_loss = recon_criterion(reconstructed, images)
        
        # Sparsity penalty
        if sparsity_type == 'l1':
            sparsity_loss = l1_loss(latent)
        elif sparsity_type == 'kl':
            sparsity_loss = kl_divergence_loss(latent, rho)
        
        # Total loss
        total_loss = recon_loss + sparsity_weight * sparsity_loss
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        total_loss_sum += total_loss.item()
        recon_loss_sum += recon_loss.item()
        sparsity_loss_sum += sparsity_loss.item()
        num_batches += 1
    
    return (total_loss_sum / num_batches, 
            recon_loss_sum / num_batches, 
            sparsity_loss_sum / num_batches)
```

### Sparsity Metrics

| Metric | Definition |
|--------|------------|
| **Population sparsity** | For each sample, what fraction of neurons are active? |
| **Lifetime sparsity** | For each neuron, what fraction of samples activate it? |

```python
def analyze_sparsity(model, test_loader, device, num_samples=1000):
    """Analyze sparsity of learned representations."""
    model.eval()
    
    all_activations = []
    
    with torch.no_grad():
        for images, _ in test_loader:
            if len(all_activations) * test_loader.batch_size >= num_samples:
                break
            images = images.view(images.size(0), -1).to(device)
            _, latent = model(images)
            all_activations.append(latent.cpu().numpy())
    
    all_activations = np.concatenate(all_activations, axis=0)[:num_samples]
    
    # Define "active" as activation > threshold
    threshold = 0.1
    active = all_activations > threshold
    
    # Population sparsity: average fraction of active neurons per sample
    population_sparsity = np.mean(np.mean(active, axis=1))
    
    # Lifetime sparsity: fraction of samples that activate each neuron
    lifetime_sparsity = np.mean(active, axis=0)
    
    return population_sparsity, lifetime_sparsity
```

### L1 vs KL Comparison

| Method | Mechanism | Pros | Cons |
|--------|-----------|------|------|
| **L1** | Penalize $\|h\|_1$ | Simple, fast | May not reach exact target sparsity |
| **KL** | Penalize divergence from target $\rho$ | Precise control over sparsity | Requires sigmoid activation |

---

## Part 3: Contractive Penalty

A **contractive autoencoder (CAE)** adds a penalty on the Frobenius norm of the encoder's Jacobian, encouraging the encoder to be insensitive to input perturbations.

### The Contractive Penalty

$$\mathcal{L} = \|x - g(f(x))\|^2 + \lambda \|J_f(x)\|_F^2$$

where:

- $J_f(x) = \frac{\partial f(x)}{\partial x} \in \mathbb{R}^{k \times d}$: Jacobian matrix of the encoder
- $\|J_f\|_F^2 = \sum_{ij} J_{ij}^2$: Frobenius norm squared

### Intuition

| Component | Effect |
|-----------|--------|
| Reconstruction loss | Learn to reconstruct inputs |
| Jacobian penalty | Make encoder insensitive to input perturbations |

The Jacobian penalty encourages **local invariance** (small input changes → small latent changes), **robust representations** (ignore noise, capture essential structure), and **flat manifolds** (latent space locally constant along noise directions).

### Connection to Denoising Autoencoders

For small Gaussian noise with variance $\sigma^2$, denoising autoencoders approximately minimize:

$$\mathcal{L}_{DAE} \approx \|x - g(f(x))\|^2 + \sigma^2 \|J_f(x)\|_F^2$$

**Key insight:** Denoising with Gaussian noise implicitly applies a contractive penalty!

| Aspect | Denoising AE | Contractive AE |
|--------|--------------|----------------|
| Regularization | Via corrupted inputs | Via explicit Jacobian penalty |
| Computation | Forward pass with noise | Requires Jacobian computation |
| Flexibility | Different noise types | Direct control over contraction |
| Interpretation | Learn to denoise | Minimize encoder sensitivity |

### Implementation

```python
from torch.autograd import grad


class ContractiveAutoencoder(nn.Module):
    """Contractive Autoencoder with Jacobian penalty."""
    
    def __init__(self, input_dim=784, latent_dim=64):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder with sigmoid for bounded outputs
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Sigmoid(),
            nn.Linear(256, 128),
            nn.Sigmoid(),
            nn.Linear(128, latent_dim),
            nn.Sigmoid()  # Bounded [0,1] for stable Jacobian
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


def compute_jacobian_penalty(model, x):
    """
    Compute the Frobenius norm squared of the encoder Jacobian.
    
    J_f(x)_ij = ∂z_i / ∂x_j
    ||J_f||_F^2 = Σ_ij (∂z_i / ∂x_j)^2
    """
    x = x.requires_grad_(True)
    z = model.encode(x)
    
    # Compute Jacobian column by column
    jacobian_norm_sq = 0.0
    
    for i in range(z.shape[1]):
        # Gradient of z_i with respect to x
        grad_outputs = torch.zeros_like(z)
        grad_outputs[:, i] = 1.0
        
        jacobian_col = grad(
            outputs=z,
            inputs=x,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Sum of squares of this column
        jacobian_norm_sq = jacobian_norm_sq + torch.sum(jacobian_col ** 2)
    
    return jacobian_norm_sq / x.shape[0]  # Average over batch


def train_contractive_autoencoder(
    model, train_loader, device, 
    lambda_contractive=0.1, num_epochs=15
):
    """
    Train contractive autoencoder.
    
    Loss = Reconstruction + λ × ||J_f||_F^2
    """
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    recon_criterion = nn.MSELoss()
    
    history = {'recon_loss': [], 'contractive_loss': [], 'total_loss': []}
    
    for epoch in range(num_epochs):
        model.train()
        
        epoch_recon = 0
        epoch_contractive = 0
        epoch_total = 0
        
        for images, _ in train_loader:
            images = images.view(images.size(0), -1).to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            recon, z = model(images)
            
            # Reconstruction loss
            recon_loss = recon_criterion(recon, images)
            
            # Contractive penalty
            contractive_loss = compute_jacobian_penalty(model, images)
            
            # Total loss
            total_loss = recon_loss + lambda_contractive * contractive_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            epoch_recon += recon_loss.item()
            epoch_contractive += contractive_loss.item()
            epoch_total += total_loss.item()
        
        n_batches = len(train_loader)
        history['recon_loss'].append(epoch_recon / n_batches)
        history['contractive_loss'].append(epoch_contractive / n_batches)
        history['total_loss'].append(epoch_total / n_batches)
        
        print(f"Epoch {epoch+1}: Recon={epoch_recon/n_batches:.6f}, "
              f"Contract={epoch_contractive/n_batches:.6f}")
    
    return history
```

### Geometric Interpretation

The contractive penalty encourages:

1. **Flat latent manifold:** Encoder output varies slowly with input
2. **Noise directions contracted:** Non-manifold directions compressed
3. **Data manifold preserved:** Important variations retained

The trade-off:

$$\text{Low } \lambda \to \text{Better reconstruction, less robustness}$$
$$\text{High } \lambda \to \text{More robustness, worse reconstruction}$$

---

## Part 4: Denoising Objective

The denoising autoencoder uses a fundamentally different training objective: reconstruct **clean** data from **corrupted** inputs.

### Standard vs Denoising Objective

| Autoencoder Type | Training Objective |
|------------------|-------------------|
| Standard | Minimize $\|x - f(x)\|^2$ |
| Denoising | Minimize $\|x - f(\tilde{x})\|^2$ |

where $\tilde{x} = \text{corrupt}(x)$ is the noisy input and the loss is computed against the **clean** original. This forces the network to learn robust features that can recover the underlying signal from corrupted observations.

### Corruption Strategies

| Strategy | Formula | Description | Use Case |
|----------|---------|-------------|----------|
| Gaussian | $\tilde{x} = x + \epsilon$, $\epsilon \sim \mathcal{N}(0, \sigma^2)$ | Additive white noise | General purpose |
| Salt-and-pepper | Random pixels → 0 or 1 | Impulse noise | Document/sensor data |
| Masking | Random pixels → 0 | Dropout-like | Occlusion robustness |
| Structured | Block/region masking | Spatial coherent noise | Inpainting |

### Noise Implementations

```python
def add_noise(images: torch.Tensor, noise_factor: float = 0.1) -> torch.Tensor:
    """
    Add Gaussian noise to images.
    
    Corrupted image: x̃ = x + ε where ε ~ N(0, σ²)
    """
    noise = torch.randn_like(images) * noise_factor
    noisy_images = images + noise
    return torch.clamp(noisy_images, 0.0, 1.0)


def add_salt_pepper_noise(images: torch.Tensor, 
                          noise_prob: float = 0.2) -> torch.Tensor:
    """Add salt-and-pepper noise to images."""
    noisy_images = images.clone()
    
    noise_mask = torch.rand_like(images) < noise_prob
    salt_mask = torch.rand_like(images) > 0.5
    
    noisy_images[noise_mask & salt_mask] = 1.0   # Salt
    noisy_images[noise_mask & ~salt_mask] = 0.0  # Pepper
    
    return noisy_images


def add_masking_noise(images: torch.Tensor, 
                      mask_prob: float = 0.3) -> torch.Tensor:
    """
    Add masking noise by randomly setting pixels to zero.
    Related to masked autoencoders (MAE) in modern vision transformers.
    """
    mask = (torch.rand_like(images) > mask_prob).float()
    return images * mask
```

### Score Matching Connection

For small Gaussian noise with variance $\sigma^2$, denoising autoencoders implicitly estimate the **score function**:

$$\nabla_x \log p(x) \approx \frac{1}{\sigma^2}(f(\tilde{x}) - \tilde{x})$$

This connects denoising autoencoders to score-based generative models, diffusion models, and energy-based models.

---

## Part 5: Complete Regularized Training

Bringing all loss components together:

```python
def train_regularized_autoencoder(
    model, train_loader, optimizer, device, epoch,
    regularization='none',       # 'none', 'l1', 'kl', 'contractive', 'denoising'
    reg_weight=0.001,            # Weight for regularization term
    rho=0.05,                    # Target sparsity for KL
    noise_factor=0.1             # Noise level for denoising
):
    """
    Unified training function supporting all regularization types.
    
    Total Loss = Reconstruction Loss + reg_weight × Regularization Term
    """
    model.train()
    
    total_loss_sum = 0.0
    recon_loss_sum = 0.0
    reg_loss_sum = 0.0
    num_batches = 0
    
    recon_criterion = nn.MSELoss()
    
    for batch_idx, (images, _) in enumerate(train_loader):
        images = images.view(images.size(0), -1).to(device)
        
        optimizer.zero_grad()
        
        # Prepare input (corrupt for denoising, clean otherwise)
        if regularization == 'denoising':
            input_images = add_noise(images, noise_factor)
        else:
            input_images = images
        
        # Forward pass
        reconstructed, latent = model(input_images)
        
        # Reconstruction loss (always against CLEAN images)
        recon_loss = recon_criterion(reconstructed, images)
        
        # Regularization term
        if regularization == 'l1':
            reg_loss = l1_loss(latent)
        elif regularization == 'kl':
            reg_loss = kl_divergence_loss(latent, rho)
        elif regularization == 'contractive':
            reg_loss = compute_jacobian_penalty(model, input_images)
        else:
            reg_loss = torch.tensor(0.0, device=device)
        
        # Total loss
        total_loss = recon_loss + reg_weight * reg_loss
        
        total_loss.backward()
        optimizer.step()
        
        total_loss_sum += total_loss.item()
        recon_loss_sum += recon_loss.item()
        reg_loss_sum += reg_loss.item()
        num_batches += 1
    
    return (total_loss_sum / num_batches,
            recon_loss_sum / num_batches,
            reg_loss_sum / num_batches)
```

---

## Visualizing Learned Features

Sparse autoencoders learn features that are often more interpretable than dense autoencoders:

```python
def visualize_learned_features(model, num_features=64):
    """
    Visualize learned features by decoding one-hot latent vectors.
    
    For sparse autoencoders, features are often more interpretable,
    showing localized patterns.
    """
    model.eval()
    
    latent_dim = model.latent_dim
    num_features = min(num_features, latent_dim)
    
    features = []
    with torch.no_grad():
        for i in range(num_features):
            # Create one-hot vector
            latent = torch.zeros(1, latent_dim)
            latent[0, i] = 1.0  # Activate only neuron i
            
            # Decode to image space
            feature = model.decoder(latent)
            features.append(feature.cpu().numpy().reshape(28, 28))
    
    # Visualize in grid
    grid_size = int(np.ceil(np.sqrt(num_features)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten()
    
    for i in range(num_features):
        axes[i].imshow(features[i], cmap='gray')
        axes[i].axis('off')
    
    plt.suptitle('Learned Features (Decoder Basis)')
    plt.savefig('learned_features.png', dpi=150)
    plt.show()
```

---

## Exercises

### Exercise 1: Loss Function Comparison

Train autoencoders with MSE, BCE, and L1 losses. Compare reconstruction quality visually and numerically. Which produces the sharpest outputs?

### Exercise 2: Sparsity Weight Tuning

Train models with different sparsity weights:

```python
l1_weights = [0.0001, 0.001, 0.01, 0.1, 1.0]
kl_weights = [0.001, 0.01, 0.1, 1.0, 10.0]
```

How does sparsity weight affect the trade-off between reconstruction quality and sparsity? Do stronger constraints lead to more interpretable features?

### Exercise 3: L1 vs KL Comparison

Train two models with similar effective sparsity: L1 with $\lambda = 0.01$ and KL with $\beta = 0.1$, $\rho = 0.05$. Compare training dynamics, final sparsity levels, and feature quality.

### Exercise 4: Contractive λ Tuning

Train contractive autoencoders with $\lambda \in \{0.001, 0.01, 0.1, 1.0\}$. Plot reconstruction error vs encoder sensitivity. What is the optimal trade-off?

### Exercise 5: Noise Level Analysis (Denoising)

Train denoising autoencoders with different noise levels:

```python
noise_factors = [0.1, 0.2, 0.3, 0.4, 0.5]
```

Is there an optimal noise level for training? Can a model trained on high noise denoise low noise?

### Exercise 6: Noise Type Robustness

Train three separate models, each on one noise type (Gaussian, salt-and-pepper, masking). Test each model on all noise types. Does training on one noise type generalize to others?

---

## Summary

| Loss / Penalty | Formula | Effect | Use Case |
|----------------|---------|--------|----------|
| **MSE** | $\|x - \hat{x}\|^2$ | Gaussian assumption, blurry | Continuous data |
| **BCE** | $-[x\log\hat{x} + (1-x)\log(1-\hat{x})]$ | Sharper, binary assumption | Normalized images |
| **MAE** | $\|x - \hat{x}\|_1$ | Robust to outliers | Noisy data |
| **L1 Sparsity** | $\lambda\sum\|h_j\|$ | Forces zero activations | Interpretable features |
| **KL Sparsity** | $\beta\sum\text{KL}(\rho\|\hat{\rho}_j)$ | Precise sparsity control | Overcomplete AE |
| **Contractive** | $\lambda\|J_f\|_F^2$ | Input-insensitive encoder | Robust manifold learning |
| **Denoising** | Corrupt input, reconstruct clean | Implicit regularization | Robust features |

**Key Insight:** The choice of loss function and regularization fundamentally determines what the autoencoder learns. Reconstruction losses encode assumptions about data noise, while regularization terms shape the geometry and interpretability of the latent space.
