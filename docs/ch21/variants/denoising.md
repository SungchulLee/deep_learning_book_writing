# Denoising Autoencoder

Learn robust representations by reconstructing clean data from corrupted inputs.

---

## Overview

**What you'll learn:**

- Adding structured noise to inputs during training
- Learning to denoise: reconstruct clean targets from corrupted observations
- Noise types: Gaussian, salt-and-pepper, masking
- Comparison with standard autoencoders for robustness and feature quality
- Theoretical connection to contractive autoencoders and score matching

---

## Mathematical Foundation

### Standard vs Denoising Autoencoder

| Autoencoder Type | Training Objective |
|------------------|-------------------|
| Standard | Minimize $\|x - f(x)\|^2$ |
| Denoising | Minimize $\|x - f(\tilde{x})\|^2$ |

where $x$ is the clean input, $\tilde{x} = \text{corrupt}(x)$ is the noisy input, and $f(\tilde{x})$ is the reconstruction from the noisy input.

**Critical:** The loss is computed between the reconstruction and the *clean* original, not the noisy input. This forces the network to learn robust features that can recover the underlying signal from corrupted observations.

### Why Denoising Works

1. **Prevents identity mapping:** The network cannot simply copy the input because the input is corrupted
2. **Learns robust features:** Must capture underlying data structure to denoise
3. **Implicit regularization:** Equivalent to a contractive penalty for small noise (theoretical result by Alain & Bengio, 2014)
4. **Better generalization:** Features transfer better to downstream tasks

### Common Corruption Strategies

| Strategy | Formula | Description | Use Case |
|----------|---------|-------------|----------|
| Gaussian | $\tilde{x} = x + \epsilon$, $\epsilon \sim \mathcal{N}(0, \sigma^2)$ | Additive white noise | General purpose |
| Salt-and-pepper | Random pixels → 0 or 1 | Impulse noise | Document/sensor data |
| Masking | Random pixels → 0 | Dropout-like | Occlusion robustness |
| Structured | Block/region masking | Spatially coherent noise | Inpainting |

---

## Part 1: Noise Corruption Functions

```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def add_noise(images: torch.Tensor, noise_factor: float = 0.1) -> torch.Tensor:
    """
    Add Gaussian noise to images.
    
    Corrupted image: x̃ = x + ε where ε ~ N(0, σ²)
    
    Args:
        images: Clean images tensor
        noise_factor: Standard deviation of Gaussian noise
        
    Returns:
        Noisy images (clamped to [0, 1] range)
    """
    noise = torch.randn_like(images) * noise_factor
    noisy_images = images + noise
    return torch.clamp(noisy_images, 0.0, 1.0)


def add_salt_pepper_noise(images: torch.Tensor, 
                          noise_prob: float = 0.2) -> torch.Tensor:
    """
    Add salt-and-pepper noise to images.
    Randomly sets pixels to either 0 (pepper) or 1 (salt).
    """
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

---

## Part 2: Convolutional Denoising Autoencoder

```python
class ConvDenoisingAutoencoder(nn.Module):
    """
    Convolutional Denoising Autoencoder for MNIST.
    
    The architecture is identical to a standard conv autoencoder.
    The key difference is entirely in training: input is noisy,
    target is clean.
    """
    
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)
```

---

## Part 3: Fully-Connected Denoising Autoencoder

```python
class FCDenoisingAutoencoder(nn.Module):
    """
    Denoising Autoencoder with fully-connected layers.
    
    Same architecture as basic autoencoder — the key difference
    is in training with corrupted inputs.
    """
    
    def __init__(self, input_dim: int = 784, latent_dim: int = 64):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent
```

---

## Part 4: Training

```python
def train_denoising_ae(model, train_loader, device, noise_fn, noise_param,
                       num_epochs=10, learning_rate=0.001, is_conv=False):
    """
    Train denoising autoencoder.
    
    CRITICAL DIFFERENCE from standard autoencoder:
    - Input to network: noisy images
    - Target for loss: clean images
    
    This forces the network to learn to remove noise rather than copy input.
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss() if not is_conv else nn.BCELoss()
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for images, _ in train_loader:
            if is_conv:
                clean = images.to(device)
                noisy = noise_fn(clean, noise_param)
            else:
                clean = images.view(images.size(0), -1).to(device)
                noisy = noise_fn(clean, noise_param)
            
            optimizer.zero_grad()
            
            if is_conv:
                reconstructed = model(noisy)
            else:
                reconstructed, _ = model(noisy)
            
            # Loss against CLEAN images — not noisy!
            loss = criterion(reconstructed, clean)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.6f}")
    
    return model
```

---

## Part 5: Comparison with Standard Autoencoder

```python
def compare_with_standard_ae(denoising_model, standard_model, 
                             test_loader, device, noise_fn, noise_param,
                             is_conv=False):
    """
    Compare denoising AE with standard AE on noisy inputs.
    
    Demonstrates that denoising autoencoders learn more robust
    representations and handle noisy inputs better.
    """
    denoising_model.eval()
    standard_model.eval()
    
    images, _ = next(iter(test_loader))
    if is_conv:
        images = images[:5].to(device)
    else:
        images = images[:5].view(5, -1).to(device)
    
    noisy_images = noise_fn(images, noise_param)
    
    with torch.no_grad():
        if is_conv:
            denoising_recon = denoising_model(noisy_images)
            standard_recon = standard_model(noisy_images)
        else:
            denoising_recon, _ = denoising_model(noisy_images)
            standard_recon, _ = standard_model(noisy_images)
    
    mse_denoising = torch.mean((images - denoising_recon) ** 2).item()
    mse_standard = torch.mean((images - standard_recon) ** 2).item()
    
    print(f"Denoising AE - MSE vs clean: {mse_denoising:.6f}")
    print(f"Standard AE - MSE vs clean: {mse_standard:.6f}")
    print(f"Improvement: {mse_standard/mse_denoising:.2f}x better")


def compare_noise_levels(model, test_loader, device, is_conv=True):
    """
    Compare denoising quality across different noise levels.
    Tests generalization: Can a model trained on one noise level
    handle different noise levels at test time?
    """
    model.eval()
    
    images, _ = next(iter(test_loader))
    image = images[0:1].to(device)
    
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]
    
    fig, axes = plt.subplots(2, len(noise_levels), figsize=(15, 5))
    
    with torch.no_grad():
        for i, noise_factor in enumerate(noise_levels):
            noisy = add_noise(image, noise_factor)
            denoised = model(noisy)
            
            axes[0, i].imshow(noisy.cpu().squeeze(), cmap='gray')
            axes[0, i].set_title(f'σ={noise_factor}')
            axes[0, i].axis('off')
            
            axes[1, i].imshow(denoised.cpu().squeeze(), cmap='gray')
            axes[1, i].axis('off')
    
    axes[0, 0].set_ylabel('Noisy')
    axes[1, 0].set_ylabel('Denoised')
    
    plt.tight_layout()
    plt.savefig('noise_level_comparison.png', dpi=150)
    plt.show()
```

---

## Part 6: Feature Quality Analysis

```python
def analyze_feature_quality(denoising_model, standard_model, 
                           test_loader, device):
    """
    Compare quality of learned features for downstream classification.
    
    Better features should cluster by class and be more separable.
    """
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score
    
    denoising_model.eval()
    standard_model.eval()
    
    denoising_features = []
    standard_features = []
    labels = []
    
    with torch.no_grad():
        for images, targets in test_loader:
            images_flat = images.view(images.size(0), -1).to(device)
            
            denoising_z = denoising_model.encoder(images_flat)
            standard_z = standard_model.encoder(images_flat)
            
            denoising_features.append(denoising_z.cpu())
            standard_features.append(standard_z.cpu())
            labels.append(targets)
    
    denoising_features = torch.cat(denoising_features).numpy()
    standard_features = torch.cat(standard_features).numpy()
    labels = torch.cat(labels).numpy()
    
    # KNN classification accuracy as proxy for feature quality
    knn_denoising = KNeighborsClassifier(n_neighbors=5)
    knn_standard = KNeighborsClassifier(n_neighbors=5)
    
    acc_denoising = cross_val_score(knn_denoising, denoising_features, labels, cv=5)
    acc_standard = cross_val_score(knn_standard, standard_features, labels, cv=5)
    
    print(f"Denoising AE features - KNN accuracy: {acc_denoising.mean():.4f}")
    print(f"Standard AE features - KNN accuracy: {acc_standard.mean():.4f}")
```

---

## Theoretical Connections

### Score Matching Interpretation

For small Gaussian noise with variance $\sigma^2$, denoising autoencoders implicitly estimate the score function:

$$\nabla_x \log p(x) \approx \frac{1}{\sigma^2}(f(\tilde{x}) - \tilde{x})$$

This connects denoising autoencoders to score-based generative models, diffusion models, and energy-based models.

### Contractive Regularization

The denoising objective acts as implicit regularization on the encoder's Jacobian:

$$\mathcal{L}_{\text{denoise}} \approx \mathcal{L}_{\text{recon}} + \lambda \|J_f(x)\|_F^2$$

where $J_f(x) = \frac{\partial f}{\partial x}$ is the encoder's Jacobian and $\|\cdot\|_F$ is the Frobenius norm. This means denoising with Gaussian noise implicitly applies a contractive penalty, providing a deep theoretical link between the two approaches.

---

## Quantitative Finance Application

Denoising autoencoders are highly relevant in quantitative finance:

- **Signal extraction:** Financial time series are inherently noisy. A denoising AE trained on historical returns can learn to separate signal from microstructure noise, producing cleaner features for alpha generation
- **Robust factor models:** Factors learned via denoising are more stable across market regimes than those from standard PCA or vanilla autoencoders
- **Missing data imputation:** Masking noise during training naturally teaches the model to reconstruct missing observations — useful for illiquid instruments or asynchronous data

---

## Exercises

### Exercise 1: Noise Level Analysis
Train denoising autoencoders with different noise levels (`σ ∈ {0.1, 0.2, 0.3, 0.4, 0.5}`). Is there an optimal noise level? Can a model trained on high noise denoise low noise effectively?

### Exercise 2: Noise Type Robustness
Train three models on Gaussian, salt-and-pepper, and masking noise respectively. Test each on all noise types. Which noise type leads to the most robust features?

### Exercise 3: Feature Transfer
Compare representations from standard vs denoising autoencoders using KNN classification accuracy. Do denoising features transfer better to downstream tasks?

### Exercise 4: Progressive Denoising
Implement iterative denoising: start with a heavily corrupted image and pass through the denoiser multiple times. Does iterative application improve results?

---

## Summary

| Aspect | Standard AE | Denoising AE |
|--------|-------------|--------------|
| Input | Clean images | Corrupted images |
| Target | Same as input | Clean images |
| Learning | Copy input | Remove corruption |
| Features | May overfit | More robust |
| Identity risk | Present | Eliminated by noise |
| Use case | Compression | Denoising, robust features |

**Key Insight:** By training to reconstruct clean data from corrupted inputs, denoising autoencoders learn features that capture the underlying structure of the data rather than superficial patterns or noise. The theoretical equivalence to contractive regularization provides a principled understanding of why this approach works.

---

## References

1. Vincent, P., et al. (2008). "Extracting and composing robust features with denoising autoencoders." *ICML*.
2. Vincent, P., et al. (2010). "Stacked Denoising Autoencoders: Learning Useful Representations." *JMLR*.
3. Alain, G., & Bengio, Y. (2014). "What regularized auto-encoders learn from the data-generating distribution." *JMLR*.
