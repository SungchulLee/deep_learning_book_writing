# Encoder-Decoder Architecture

A comprehensive introduction to autoencoder architecture and the encoder-decoder framework.

---

## Overview

**What you'll learn:**

- Basic autoencoder architecture
- Encoder-decoder structure
- Training process for reconstruction
- Visualization of learned representations

**Time:** ~45 minutes  
**Level:** Beginner

---

## Mathematical Foundation

An autoencoder learns two functions:

- **Encoder:** $f_\theta: X \to Z$ where $Z$ is the latent space
- **Decoder:** $g_\phi: Z \to \hat{X}$ where $\hat{X}$ is the reconstruction

**Objective:** Minimize reconstruction loss

$$\mathcal{L}(\theta, \phi) = \frac{1}{n} \sum_{i=1}^{n} \| x_i - g_\phi(f_\theta(x_i)) \|^2$$

**For MNIST:**

| Component | Dimension |
|-----------|-----------|
| Input $x$ | $\mathbb{R}^{784}$ (28×28 flattened) |
| Latent $z$ | $\mathbb{R}^{64}$ |
| Output $\hat{x}$ | $\mathbb{R}^{784}$ |

**Architecture:**

```
Encoder: 784 → 256 → 128 → 64
Decoder: 64 → 128 → 256 → 784
```

---

## Part 1: Autoencoder Architecture

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

class SimpleAutoencoder(nn.Module):
    """
    A simple fully-connected autoencoder for MNIST images.
    
    Architecture:
    - Encoder: Progressively reduces dimensionality
    - Bottleneck: Compressed latent representation
    - Decoder: Mirrors encoder to reconstruct input
    """
    
    def __init__(self, input_dim: int = 784, latent_dim: int = 64):
        super(SimpleAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder: Compresses input to latent representation
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
            nn.ReLU()
        )
        
        # Decoder: Reconstructs input from latent representation
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()  # Maps to [0, 1] for pixel values
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input into latent representation."""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to reconstruct input."""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor):
        """Full forward pass through autoencoder."""
        z = self.encode(x)
        x_reconstructed = self.decode(z)
        return x_reconstructed, z
```

### Why These Design Choices?

- **ReLU activation:** Non-linearity allows learning complex patterns
- **Sigmoid output:** Ensures output is in [0, 1] range (matching normalized images)
- **Symmetric architecture:** Decoder mirrors encoder for balanced capacity

---

## Part 2: Data Loading and Preprocessing

```python
def load_mnist_data(batch_size: int = 128):
    """
    Load and preprocess MNIST dataset.
    
    Preprocessing:
    1. Convert images to tensors
    2. Normalize to [0, 1] range
    3. Flatten 28×28 images to 784-dimensional vectors
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    
    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    return train_loader, test_loader
```

---

## Part 3: Training Function

```python
def train_autoencoder(model, train_loader, criterion, optimizer, device, epoch):
    """
    Train autoencoder for one epoch.
    
    Training Process:
    1. Forward pass: Encode and decode input
    2. Compute reconstruction loss
    3. Backward pass: Compute gradients
    4. Update weights using optimizer
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (images, _) in enumerate(train_loader):
        # Flatten: (batch_size, 1, 28, 28) → (batch_size, 784)
        images = images.view(images.size(0), -1).to(device)
        
        optimizer.zero_grad()
        reconstructed, latent = model(images)
        loss = criterion(reconstructed, images)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def evaluate_autoencoder(model, test_loader, criterion, device):
    """Evaluate autoencoder on test set."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.view(images.size(0), -1).to(device)
            reconstructed, _ = model(images)
            loss = criterion(reconstructed, images)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches
```

---

## Part 4: Visualization Functions

### Visualize Reconstructions

```python
def visualize_reconstructions(model, test_loader, device, num_images=10):
    """Visualize original images and their reconstructions."""
    model.eval()
    
    images, labels = next(iter(test_loader))
    images = images[:num_images]
    images_flat = images.view(images.size(0), -1).to(device)
    
    with torch.no_grad():
        reconstructed, _ = model(images_flat)
    
    images_np = images.cpu().numpy()
    reconstructed_np = reconstructed.cpu().numpy().reshape(-1, 28, 28)
    
    fig, axes = plt.subplots(2, num_images, figsize=(15, 3))
    
    for i in range(num_images):
        axes[0, i].imshow(images_np[i].squeeze(), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(reconstructed_np[i], cmap='gray')
        axes[1, i].axis('off')
    
    axes[0, 0].set_ylabel('Original')
    axes[1, 0].set_ylabel('Reconstructed')
    plt.tight_layout()
    plt.savefig('autoencoder_reconstructions.png', dpi=150)
    plt.show()
```

### Interpolate in Latent Space

```python
def interpolate_in_latent_space(model, test_loader, device, num_steps=10):
    """
    Interpolate between two images in latent space.
    
    Process:
    1. Encode two images to get z1 and z2
    2. Linear interpolation: z_t = (1-t)z1 + t*z2 for t ∈ [0, 1]
    3. Decode each interpolated point
    """
    model.eval()
    
    images, labels = next(iter(test_loader))
    img1 = images[0:1].view(1, -1).to(device)
    img2 = images[1:2].view(1, -1).to(device)
    
    with torch.no_grad():
        z1 = model.encode(img1)
        z2 = model.encode(img2)
        
        interpolated_images = []
        for t in np.linspace(0, 1, num_steps):
            z_interpolated = (1 - t) * z1 + t * z2
            img_interpolated = model.decode(z_interpolated)
            interpolated_images.append(img_interpolated.cpu().numpy())
    
    fig, axes = plt.subplots(1, num_steps, figsize=(15, 2))
    for i in range(num_steps):
        axes[i].imshow(interpolated_images[i].reshape(28, 28), cmap='gray')
        axes[i].axis('off')
    
    plt.suptitle('Interpolation in Latent Space')
    plt.savefig('latent_interpolation.png', dpi=150)
    plt.show()
```

---

## Part 5: Main Execution

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
    
    print(f"Compression ratio: {input_dim / latent_dim:.2f}x")
    
    train_loader, test_loader = load_mnist_data(batch_size)
    model = SimpleAutoencoder(input_dim, latent_dim).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(1, num_epochs + 1):
        train_loss = train_autoencoder(
            model, train_loader, criterion, optimizer, device, epoch
        )
        test_loss = evaluate_autoencoder(model, test_loader, criterion, device)
        print(f"Epoch {epoch} - Train: {train_loss:.6f}, Test: {test_loss:.6f}")
    
    visualize_reconstructions(model, test_loader, device)
    interpolate_in_latent_space(model, test_loader, device)
    
    torch.save(model.state_dict(), 'simple_autoencoder.pth')

if __name__ == "__main__":
    main()
```

---

## Exercises

### Exercise 1: Architecture Exploration

Modify the autoencoder architecture:

a) Change `latent_dim` to 2, 32, 128, and compare reconstruction quality  
b) Add more layers (e.g., 784→512→256→128→64)  
c) Try different activation functions (LeakyReLU, ELU)

### Exercise 2: Loss Function Analysis

Try different loss functions: `nn.L1Loss()`, `nn.BCELoss()`, combined MSE + L1

### Exercise 3: Latent Space Exploration

a) Train with `latent_dim=2` and visualize the 2D space  
b) Sample random points and decode them  
c) Perform arithmetic: $z_{new} = z_1 + z_2 - z_3$

---

## Summary

| Component | Description |
|-----------|-------------|
| **Encoder** | Maps input to lower-dimensional latent space |
| **Decoder** | Reconstructs input from latent representation |
| **Loss** | MSE between input and reconstruction |
| **Training** | Minimize reconstruction error via gradient descent |

---

## Next: Latent Space and Reconstruction Loss

The next section explores latent space properties and reconstruction loss in detail.
