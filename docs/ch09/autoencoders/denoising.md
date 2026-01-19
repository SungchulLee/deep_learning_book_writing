# Denoising Autoencoder

Learn robust representations by reconstructing clean data from corrupted inputs.

---

## Overview

**Key Concepts:**

- Adding structured noise to inputs during training
- Learning to denoise and reconstruct
- Comparison with standard autoencoders
- Robustness to noise and corruption
- Applications in image denoising

**Time:** ~40 minutes  
**Level:** Beginner-Intermediate

---

## Mathematical Foundation

| Autoencoder Type | Objective |
|------------------|-----------|
| Standard | Minimize $\|x - f(x)\|^2$ |
| Denoising | Minimize $\|x - f(\tilde{x})\|^2$ |

Where:
- $x$ is the clean input
- $\tilde{x} = \text{corrupt}(x)$ is the noisy input
- $f(\tilde{x})$ is the reconstruction from noisy input

The model learns to map corrupted inputs back to clean outputs, forcing it to learn more robust and meaningful features rather than just copying the input.

### Common Corruption Strategies

| Strategy | Formula | Description |
|----------|---------|-------------|
| Gaussian noise | $\tilde{x} = x + \epsilon$, $\epsilon \sim \mathcal{N}(0, \sigma^2)$ | Add random Gaussian noise |
| Salt-and-pepper | Random pixels → 0 or 1 | Impulse noise |
| Masking | Random pixels → 0 | Dropout-like corruption |

---

## Part 1: Noise Corruption Functions

### Gaussian Noise

```python
def add_gaussian_noise(images: torch.Tensor, noise_factor: float = 0.3) -> torch.Tensor:
    """
    Add Gaussian noise to images.
    Corrupted image: x̃ = x + ε where ε ~ N(0, σ²)
    """
    noise = torch.randn_like(images) * noise_factor
    noisy_images = images + noise
    noisy_images = torch.clamp(noisy_images, 0.0, 1.0)
    return noisy_images
```

### Salt-and-Pepper Noise

```python
def add_salt_pepper_noise(images: torch.Tensor, noise_prob: float = 0.2) -> torch.Tensor:
    """
    Add salt-and-pepper noise to images.
    Randomly set pixels to either 0 (pepper) or 1 (salt).
    """
    noisy_images = images.clone()
    
    # Generate random mask for corruption
    noise_mask = torch.rand_like(images) < noise_prob
    
    # For corrupted pixels, randomly choose salt (1) or pepper (0)
    salt_mask = torch.rand_like(images) > 0.5
    
    noisy_images[noise_mask & salt_mask] = 1.0   # Salt
    noisy_images[noise_mask & ~salt_mask] = 0.0  # Pepper
    
    return noisy_images
```

### Masking Noise

```python
def add_masking_noise(images: torch.Tensor, mask_prob: float = 0.3) -> torch.Tensor:
    """
    Add masking noise by randomly setting pixels to zero.
    Similar to dropout but applied to input pixels.
    """
    mask = (torch.rand_like(images) > mask_prob).float()
    noisy_images = images * mask
    return noisy_images
```

---

## Part 2: Denoising Autoencoder Architecture

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

class DenoisingAutoencoder(nn.Module):
    """
    Denoising Autoencoder with same architecture as basic autoencoder.
    
    The key difference is in training: we corrupt inputs but reconstruct
    clean originals, forcing the model to learn robust representations.
    
    Architecture: 784 → 256 → 128 → 64 → 128 → 256 → 784
    """
    
    def __init__(self, input_dim: int = 784, latent_dim: int = 64):
        super(DenoisingAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder: Maps noisy input to latent representation
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
            nn.ReLU()
        )
        
        # Decoder: Reconstructs clean image from latent representation
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor):
        """Forward pass: encode noisy input and decode to clean output."""
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent
```

---

## Part 3: Training Function with Noise Corruption

The key difference from standard autoencoders: **loss is computed between reconstruction and CLEAN images**, not noisy inputs.

```python
def train_denoising_autoencoder(
    model, train_loader, criterion, optimizer, device, epoch, noise_fn, noise_param
):
    """
    Train denoising autoencoder for one epoch.
    
    Training Process:
    1. Load clean images
    2. Create corrupted versions
    3. Pass corrupted images through encoder-decoder
    4. Compute loss between reconstruction and CLEAN images
    5. Backpropagate and update weights
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (images, _) in enumerate(train_loader):
        # Flatten clean images
        clean_images = images.view(images.size(0), -1).to(device)
        
        # Add noise to create corrupted inputs
        noisy_images = noise_fn(clean_images, noise_param)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass: Encode noisy input, decode to clean output
        reconstructed, _ = model(noisy_images)
        
        # IMPORTANT: Loss is between reconstruction and CLEAN images
        # This forces the model to learn to denoise
        loss = criterion(reconstructed, clean_images)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches
```

---

## Part 4: Visualization Functions

### Visualize Denoising Results

Shows three rows: clean → noisy → reconstructed

```python
def visualize_denoising_results(model, test_loader, device, noise_fn, noise_param, num_images=10):
    """Visualize denoising results: clean → noisy → reconstructed."""
    model.eval()
    
    images, labels = next(iter(test_loader))
    images = images[:num_images]
    
    clean_images = images.view(images.size(0), -1).to(device)
    noisy_images = noise_fn(clean_images, noise_param)
    
    with torch.no_grad():
        reconstructed, _ = model(noisy_images)
    
    clean_np = clean_images.cpu().numpy().reshape(-1, 28, 28)
    noisy_np = noisy_images.cpu().numpy().reshape(-1, 28, 28)
    reconstructed_np = reconstructed.cpu().numpy().reshape(-1, 28, 28)
    
    fig, axes = plt.subplots(3, num_images, figsize=(15, 5))
    
    for i in range(num_images):
        axes[0, i].imshow(clean_np[i], cmap='gray')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(noisy_np[i], cmap='gray')
        axes[1, i].axis('off')
        
        axes[2, i].imshow(reconstructed_np[i], cmap='gray')
        axes[2, i].axis('off')
    
    plt.savefig('denoising_results.png', dpi=150)
    plt.show()
```

### Compare Noise Types

```python
def compare_noise_types(model, test_loader, device):
    """Compare different types of noise and reconstruction quality."""
    model.eval()
    
    images, labels = next(iter(test_loader))
    clean_image = images[0:1].view(1, -1).to(device)
    
    noise_types = [
        ('Gaussian', add_gaussian_noise, 0.3),
        ('Salt-Pepper', add_salt_pepper_noise, 0.2),
        ('Masking', add_masking_noise, 0.3)
    ]
    
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    clean_np = clean_image.cpu().numpy().reshape(28, 28)
    
    with torch.no_grad():
        for row, (noise_name, noise_fn, noise_param) in enumerate(noise_types):
            noisy_image = noise_fn(clean_image, noise_param)
            reconstructed, _ = model(noisy_image)
            
            noisy_np = noisy_image.cpu().numpy().reshape(28, 28)
            recon_np = reconstructed.cpu().numpy().reshape(28, 28)
            
            # Calculate MSE
            mse_noisy = np.mean((clean_np - noisy_np) ** 2)
            mse_recon = np.mean((clean_np - recon_np) ** 2)
            
            # Plot: clean, noisy, reconstructed, error map
            axes[row, 0].imshow(clean_np, cmap='gray')
            axes[row, 1].imshow(noisy_np, cmap='gray')
            axes[row, 2].imshow(recon_np, cmap='gray')
            axes[row, 3].imshow(np.abs(clean_np - recon_np), cmap='hot')
    
    plt.savefig('noise_comparison.png', dpi=150)
    plt.show()
```

---

## Part 5: Compare with Standard Autoencoder

```python
def compare_with_standard_autoencoder(
    denoising_model, standard_model, test_loader, device, noise_fn, noise_param, num_images=5
):
    """
    Compare denoising autoencoder with standard autoencoder on noisy inputs.
    
    Demonstrates that denoising autoencoders learn more robust
    representations and handle noisy inputs better than standard AEs.
    """
    denoising_model.eval()
    standard_model.eval()
    
    images, _ = next(iter(test_loader))
    images = images[:num_images]
    clean_images = images.view(images.size(0), -1).to(device)
    noisy_images = noise_fn(clean_images, noise_param)
    
    with torch.no_grad():
        denoising_recon, _ = denoising_model(noisy_images)
        standard_recon, _ = standard_model(noisy_images)
    
    # Visualize: clean, noisy, denoising AE output, standard AE output
    # Denoising AE typically shows much better reconstruction on noisy inputs
```

---

## Part 6: Main Execution

```python
def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Hyperparameters
    input_dim = 784
    latent_dim = 64
    batch_size = 128
    learning_rate = 0.001
    num_epochs = 10
    
    # Noise configuration
    noise_type = 'gaussian'  # Options: 'gaussian', 'salt_pepper', 'masking'
    noise_param = 0.3
    
    # Select noise function
    noise_functions = {
        'gaussian': add_gaussian_noise,
        'salt_pepper': add_salt_pepper_noise,
        'masking': add_masking_noise
    }
    noise_fn = noise_functions[noise_type]
    
    # Load data
    train_loader, test_loader = load_mnist_data(batch_size)
    
    # Initialize model
    model = DenoisingAutoencoder(input_dim, latent_dim).to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(1, num_epochs + 1):
        train_loss = train_denoising_autoencoder(
            model, train_loader, criterion, optimizer,
            device, epoch, noise_fn, noise_param
        )
        print(f"Epoch {epoch} - Loss: {train_loss:.6f}")
    
    # Visualizations
    visualize_denoising_results(model, test_loader, device, noise_fn, noise_param)
    compare_noise_types(model, test_loader, device)
    
    # Save model
    torch.save(model.state_dict(), 'denoising_autoencoder.pth')

if __name__ == "__main__":
    main()
```

---

## Exercises

### Exercise 1: Noise Level Analysis

Train denoising autoencoders with different noise levels:

```python
noise_factors = [0.1, 0.2, 0.3, 0.4, 0.5]
```

**Questions:**
- How does noise level affect learned representations?
- Is there an optimal noise level for training?
- Can a model trained on high noise denoise low noise?

### Exercise 2: Noise Type Robustness

Train three separate models, each on one noise type:
- Model A: Gaussian noise
- Model B: Salt-and-pepper noise
- Model C: Masking noise

Test each model on ALL noise types.

**Questions:**
- Does training on one noise type generalize to others?
- Which noise type leads to most robust features?

### Exercise 3: Feature Analysis

Compare representations learned by standard vs. denoising autoencoders:

- Visualize filters/weights of first layer
- Compute correlation between learned features
- Use latent representations for classification task

**Questions:**
- Are denoising features more robust?
- Do denoising features transfer better to downstream tasks?

### Exercise 4: Progressive Denoising

Implement iterative denoising:

1. Start with heavily corrupted image
2. Pass through decoder multiple times
3. At each step, add small noise and re-denoise

**Questions:**
- Does iterative denoising improve results?
- How many iterations are optimal?

### Advanced Challenge: Blind Denoising

Train a model that can handle unknown noise types:

1. Create dataset with mixed noise types
2. Train single model on all noise types
3. Test on unseen noise combinations

Can you build a universal denoiser?

---

## Summary

| Aspect | Standard AE | Denoising AE |
|--------|-------------|--------------|
| Input | Clean images | Corrupted images |
| Target | Same as input | Clean images |
| Learning | Copy input | Remove corruption |
| Features | May overfit | More robust |
| Use case | Compression | Denoising, robust features |

**Key Insight:** By training to reconstruct clean data from corrupted inputs, denoising autoencoders learn features that capture the underlying structure of the data rather than superficial patterns or noise.
