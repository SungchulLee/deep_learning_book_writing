# Convolutional Autoencoder

Use convolutional layers to preserve spatial structure in image data for better reconstruction.

---

## Overview

**Key Concepts:**

- Convolutional encoder with max pooling
- Transposed convolutions (deconvolutions) for upsampling
- Preserving spatial structure
- Parameter efficiency compared to fully connected
- Application to image reconstruction and compression

**Time:** ~55 minutes  
**Level:** Intermediate

---

## Mathematical Foundation

### Architecture Comparison

| Type | Data Flow |
|------|-----------|
| Standard AE | $x \in \mathbb{R}^d \to z \in \mathbb{R}^k \to \hat{x} \in \mathbb{R}^d$ |
| Convolutional AE | $X \in \mathbb{R}^{H \times W \times C} \to Z \in \mathbb{R}^{h \times w \times c} \to \hat{X} \in \mathbb{R}^{H \times W \times C}$ |

Spatial dimensions are progressively reduced/increased through convolution + pooling / transposed convolution.

### Encoder Operations

**Convolution:** Extracts spatial features

$$\text{Output size} = \left\lfloor \frac{\text{input\_size} + 2 \times \text{padding} - \text{kernel\_size}}{\text{stride}} + 1 \right\rfloor$$

**Max Pooling:** Reduces spatial dimensions

$$\text{Output size} = \left\lfloor \frac{\text{input\_size}}{\text{pool\_size}} \right\rfloor$$

### Decoder Operations

**Transposed Convolution (ConvTranspose2d):** Upsamples

$$\text{Output size} = (\text{input\_size} - 1) \times \text{stride} - 2 \times \text{padding} + \text{kernel\_size} + \text{output\_padding}$$

### Advantages over Fully Connected

1. **Preserves spatial structure** — nearby pixels remain related
2. **Fewer parameters** — weight sharing across spatial locations
3. **Translation invariance** — features detected anywhere in image
4. **Better for image data** — exploits 2D structure

---

## Part 1: Convolutional Autoencoder Architecture

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

class ConvAutoencoder(nn.Module):
    """
    Convolutional Autoencoder for MNIST (28×28 images).
    
    Architecture:
    Encoder:
        28×28×1 → Conv(16) → 28×28×16 → MaxPool → 14×14×16
        14×14×16 → Conv(32) → 14×14×32 → MaxPool → 7×7×32
        7×7×32 → Conv(64) → 7×7×64 → MaxPool → 3×3×64
        
    Latent: 3×3×64 = 576-dimensional
    
    Decoder:
        3×3×64 → ConvTranspose(64) → 7×7×64
        7×7×64 → ConvTranspose(32) → 14×14×32
        14×14×32 → ConvTranspose(16) → 28×28×16
        28×28×16 → Conv(1) → 28×28×1
    """
    
    def __init__(self, latent_channels: int = 64):
        super(ConvAutoencoder, self).__init__()
        
        self.latent_channels = latent_channels
        
        # ENCODER
        self.encoder = nn.Sequential(
            # Layer 1: 1×28×28 → 16×14×14
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Layer 2: 16×14×14 → 32×7×7
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Layer 3: 32×7×7 → 64×3×3
            nn.Conv2d(32, latent_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(latent_channels),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # DECODER
        self.decoder = nn.Sequential(
            # Layer 1: 64×3×3 → 64×7×7
            nn.ConvTranspose2d(latent_channels, 64, kernel_size=3, 
                             stride=2, padding=0, output_padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            
            # Layer 2: 64×7×7 → 32×14×14
            nn.ConvTranspose2d(64, 32, kernel_size=3,
                             stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            
            # Layer 3: 32×14×14 → 16×28×28
            nn.ConvTranspose2d(32, 16, kernel_size=3,
                             stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            
            # Output: 16×28×28 → 1×28×28
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        """Encode input image to latent representation."""
        return self.encoder(x)
    
    def decode(self, z):
        """Decode latent representation to reconstructed image."""
        return self.decoder(z)
    
    def forward(self, x):
        """Full forward pass: encode then decode."""
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed, latent
```

### Key Design Choices

- **Progressively increase channels** while reducing spatial dimensions
- **ReLU activations** in hidden layers
- **Sigmoid output** for [0, 1] range
- **Batch normalization** for stable training

---

## Part 2: Deeper Convolutional Autoencoder

```python
class DeepConvAutoencoder(nn.Module):
    """
    Deeper convolutional autoencoder with more layers.
    
    Encoder: 28×28×1 → 14×14×32 → 7×7×64 → 3×3×128 → 1×1×256
    Decoder: 1×1×256 → 3×3×128 → 7×7×64 → 14×14×32 → 28×28×1
    """
    
    def __init__(self):
        super(DeepConvAutoencoder, self).__init__()
        
        # Encoder blocks
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2)  # → 14×14
        )
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2)  # → 7×7
        )
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2)  # → 3×3
        )
        
        self.enc4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=0),  # → 1×1 (bottleneck)
            nn.ReLU(), nn.BatchNorm2d(256)
        )
        
        # Decoder blocks (mirror of encoder)
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, padding=0),  # → 3×3
            nn.ReLU(), nn.BatchNorm2d(128)
        )
        
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=0),  # → 7×7
            nn.ReLU(), nn.BatchNorm2d(64)
        )
        
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32)
        )
        
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(), nn.BatchNorm2d(16),
            nn.Conv2d(16, 1, 3, padding=1), nn.Sigmoid()
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        latent = self.enc4(e3)
        
        # Decoder
        d4 = self.dec4(latent)
        d3 = self.dec3(d4)
        d2 = self.dec2(d3)
        reconstructed = self.dec1(d2)
        
        return reconstructed, latent
```

---

## Part 3: Training Functions

**Note:** Unlike FC autoencoders, we don't flatten the images. Images remain in `(batch, channels, height, width)` format.

```python
def train_conv_autoencoder(model, train_loader, criterion, optimizer, device, epoch):
    """Train convolutional autoencoder for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (images, _) in enumerate(train_loader):
        # Images already in correct shape: (batch, 1, 28, 28)
        images = images.to(device)
        
        optimizer.zero_grad()
        reconstructed, _ = model(images)
        loss = criterion(reconstructed, images)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def evaluate_conv_autoencoder(model, test_loader, criterion, device):
    """Evaluate convolutional autoencoder on test set."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            reconstructed, _ = model(images)
            loss = criterion(reconstructed, images)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches
```

---

## Part 4: Visualization Functions

### Reconstructions

```python
def visualize_conv_reconstructions(model, test_loader, device, num_images=10):
    """Visualize original and reconstructed images."""
    model.eval()
    
    images, labels = next(iter(test_loader))
    images = images[:num_images].to(device)
    
    with torch.no_grad():
        reconstructed, _ = model(images)
    
    images_np = images.cpu().numpy()
    reconstructed_np = reconstructed.cpu().numpy()
    
    fig, axes = plt.subplots(2, num_images, figsize=(15, 3))
    
    for i in range(num_images):
        axes[0, i].imshow(images_np[i, 0], cmap='gray')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(reconstructed_np[i, 0], cmap='gray')
        axes[1, i].axis('off')
    
    plt.savefig('conv_reconstructions.png', dpi=150)
    plt.show()
```

### Feature Maps

```python
def visualize_feature_maps(model, test_loader, device):
    """Visualize feature maps from encoder layers."""
    model.eval()
    
    images, _ = next(iter(test_loader))
    image = images[0:1].to(device)
    
    # Extract feature maps from encoder
    feature_maps = []
    x = image
    
    with torch.no_grad():
        for layer in model.encoder:
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                feature_maps.append(x.cpu().numpy())
    
    # Visualize
    num_layers = len(feature_maps)
    fig, axes = plt.subplots(num_layers, 8, figsize=(16, 2 * num_layers))
    
    for layer_idx, fmap in enumerate(feature_maps):
        num_channels = min(8, fmap.shape[1])
        for channel_idx in range(num_channels):
            ax = axes[layer_idx, channel_idx]
            ax.imshow(fmap[0, channel_idx], cmap='viridis')
            ax.axis('off')
    
    plt.suptitle('Encoder Feature Maps')
    plt.savefig('feature_maps.png', dpi=150)
    plt.show()
```

---

## Part 5: Parameter Comparison

```python
def count_parameters(model):
    """Count total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compare_model_sizes():
    """Compare parameter counts between FC and Conv autoencoders."""
    
    # FC autoencoder (784 → 256 → 128 → 64 → 128 → 256 → 784)
    fc_encoder = nn.Sequential(
        nn.Linear(784, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 64), nn.ReLU()
    )
    fc_decoder = nn.Sequential(
        nn.Linear(64, 128), nn.ReLU(),
        nn.Linear(128, 256), nn.ReLU(),
        nn.Linear(256, 784), nn.Sigmoid()
    )
    
    # Conv autoencoder
    conv_ae = ConvAutoencoder()
    
    fc_params = count_parameters(fc_encoder) + count_parameters(fc_decoder)
    conv_params = count_parameters(conv_ae)
    
    print(f"FC Autoencoder:   {fc_params:,} parameters")
    print(f"Conv Autoencoder: {conv_params:,} parameters")
    print(f"Reduction:        {fc_params / conv_params:.2f}x fewer params")
```

**Typical output:**
- FC: ~440K parameters
- Conv: ~80K parameters
- Conv uses ~5x fewer parameters!

---

## Part 6: Main Execution

```python
def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Hyperparameters
    batch_size = 128
    learning_rate = 0.001
    num_epochs = 10
    
    # Compare model sizes
    compare_model_sizes()
    
    # Load data
    train_loader, test_loader = load_mnist_data(batch_size)
    
    # Initialize model
    model = ConvAutoencoder(latent_channels=64).to(device)
    print(f"Total parameters: {count_parameters(model):,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(1, num_epochs + 1):
        train_loss = train_conv_autoencoder(
            model, train_loader, criterion, optimizer, device, epoch
        )
        test_loss = evaluate_conv_autoencoder(
            model, test_loader, criterion, device
        )
        print(f"Epoch {epoch} - Train: {train_loss:.6f}, Test: {test_loss:.6f}")
    
    # Visualizations
    visualize_conv_reconstructions(model, test_loader, device)
    visualize_feature_maps(model, test_loader, device)
    
    torch.save(model.state_dict(), 'conv_autoencoder.pth')

if __name__ == "__main__":
    main()
```

---

## Exercises

### Exercise 1: Architecture Exploration

Modify the convolutional autoencoder:

a) Different number of channels: [8, 16, 32, 64, 128]  
b) Different numbers of layers (2, 3, 4, 5)  
c) Different kernel sizes (3×3, 5×5, 7×7)

**Questions:**
- How do these affect reconstruction quality?
- What's the trade-off between parameters and performance?

### Exercise 2: Stride vs Pooling

Compare two downsampling strategies:

a) Max pooling after convolution  
b) Strided convolutions (stride=2)

**Questions:**
- Which gives better reconstructions?
- How do parameter counts compare?

### Exercise 3: Color Images (CIFAR-10)

Adapt the conv autoencoder for CIFAR-10 (32×32×3):

a) Modify input/output channels (1 → 3)  
b) Adjust architecture for larger images  
c) Train and evaluate

### Exercise 4: Skip Connections

Implement skip connections (like U-Net):

a) Concatenate encoder features to decoder  
b) Compare with standard autoencoder  
c) Measure reconstruction quality

### Exercise 5: Compression Ratio Analysis

Train models with different compression ratios:

```python
compression_ratios = [2, 4, 8, 16, 32, 64]
```

Plot: compression ratio vs. reconstruction quality

---

## Summary

| Aspect | FC Autoencoder | Conv Autoencoder |
|--------|----------------|------------------|
| Input format | Flattened vector | 2D image tensor |
| Parameters | ~440K | ~80K |
| Spatial structure | Lost | Preserved |
| Translation invariance | No | Yes |
| Best for | Tabular data | Image data |

**Key Insight:** Convolutional autoencoders exploit the 2D structure of images through weight sharing and local connectivity, achieving better reconstructions with fewer parameters than fully connected autoencoders.
