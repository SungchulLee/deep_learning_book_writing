# Convolutional Autoencoder

Use convolutional layers to preserve spatial structure in image data, achieving better reconstruction with fewer parameters.

---

## Overview

**What you'll learn:**

- Convolutional encoder with max pooling for spatial downsampling
- Transposed convolutions for upsampling in the decoder
- Preserving spatial structure vs flattening approaches
- Parameter efficiency of weight sharing
- Deep convolutional architectures with batch normalization
- Hierarchical feature learning through stacked convolutional layers

---

## Mathematical Foundation

### Architecture Comparison

| Type | Data Flow |
|------|-----------|
| Fully Connected | $x \in \mathbb{R}^d \to z \in \mathbb{R}^k \to \hat{x} \in \mathbb{R}^d$ |
| Convolutional | $X \in \mathbb{R}^{H \times W \times C} \to Z \in \mathbb{R}^{h \times w \times c} \to \hat{X} \in \mathbb{R}^{H \times W \times C}$ |

The key difference: convolutional autoencoders preserve the 2D spatial structure of images throughout the network, rather than flattening to 1D.

### Encoder Operations

**Convolution:** Extracts local spatial features

$$\text{Output size} = \left\lfloor \frac{\text{input} + 2 \times \text{padding} - \text{kernel}}{\text{stride}} + 1 \right\rfloor$$

**Max Pooling:** Reduces spatial dimensions while preserving important features

$$\text{Output size} = \left\lfloor \frac{\text{input}}{\text{pool\_size}} \right\rfloor$$

### Decoder Operations

**Transposed Convolution (ConvTranspose2d):** Upsamples spatial dimensions

$$H_{\text{out}} = (H_{\text{in}} - 1) \times \text{stride} - 2 \times \text{padding} + \text{kernel} + \text{output\_padding}$$

### Advantages over Fully Connected

| Aspect | FC Autoencoder | Conv Autoencoder |
|--------|----------------|------------------|
| **Spatial structure** | Lost (flattened) | Preserved |
| **Parameters** | ~440K (MNIST) | ~80K (MNIST) |
| **Translation invariance** | No | Yes |
| **Weight sharing** | No | Yes |
| **Best for** | Tabular data | Image data |

---

## Part 1: Simple Convolutional Autoencoder

```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class SimpleConvAutoencoder(nn.Module):
    """
    Simple Convolutional Autoencoder for MNIST.
    
    Encoder:
        Conv2d: 1×28×28 → 16×28×28 (same padding)
        MaxPool2d: 16×28×28 → 16×14×14
        
    Decoder:
        ConvTranspose2d: 16×14×14 → 1×28×28
    
    Images stay as 2D tensors throughout — no flattening needed.
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

## Part 2: Deep Convolutional Autoencoder

For more complex data or better reconstruction quality, use multiple encoding stages with batch normalization:

```python
class DeepConvAutoencoder(nn.Module):
    """
    Deeper convolutional autoencoder with multiple encoding stages.
    
    Encoder: 28×28×1 → 14×14×16 → 7×7×32 → 3×3×64
    Decoder: 3×3×64 → 7×7×64 → 14×14×32 → 28×28×16 → 28×28×1
    
    Uses BatchNorm for training stability and progressive
    feature extraction at multiple spatial resolutions.
    """
    
    def __init__(self, latent_channels: int = 64):
        super().__init__()
        
        self.latent_channels = latent_channels
        
        # Encoder
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
        
        # Decoder
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
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed, latent
```

---

## Part 3: Deep Fully-Connected (Stacked) Autoencoder

For non-image data or when a flat latent vector is needed, deep fully-connected autoencoders learn hierarchical features through stacked layers:

```python
class DeepAutoencoder(nn.Module):
    """
    Deep autoencoder with multiple encoding and decoding layers.
    
    Architecture: 784 → 512 → 256 → 128 → 32 (bottleneck)
                  32 → 128 → 256 → 512 → 784
    
    Compression ratio: 784/32 = 24.5x
    
    Each layer captures increasingly abstract features:
    - h₁: Low-level features (edges, textures)
    - h₂: Mid-level features (parts, patterns)
    - z:  High-level abstract representation
    """
    
    def __init__(self):
        super(DeepAutoencoder, self).__init__()
        
        self.latent_dim = 32
        
        # Encoder: Progressive dimensionality reduction
        self.encoder = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            
            nn.Linear(128, 32),
            nn.ReLU()
        )
        
        # Decoder: Mirror of encoder
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            
            nn.Linear(512, 784),
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
    
    def get_layer_outputs(self, x):
        """Get outputs from all encoder layers for visualization."""
        activations = []
        h = x
        
        for layer in self.encoder:
            h = layer(h)
            if isinstance(layer, nn.ReLU):
                activations.append(h.detach())
        
        return activations
```

---

## Part 4: Training Deep Autoencoders

```python
def train_deep_autoencoder(model, train_loader, criterion, optimizer, 
                           device, epoch):
    """Train deep autoencoder for one epoch with gradient clipping."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (images, _) in enumerate(train_loader):
        images = images.view(images.size(0), -1).to(device)
        
        optimizer.zero_grad()
        reconstructed, _ = model(images)
        loss = criterion(reconstructed, images)
        loss.backward()
        
        # Gradient clipping for deep networks
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches
```

### Key Training Techniques for Deep Networks

| Technique | Purpose | When to Use |
|-----------|---------|-------------|
| **Gradient clipping** | Prevents exploding gradients | Always for deep networks |
| **Batch normalization** | Stabilizes training, reduces internal covariate shift | Standard for 3+ layers |
| **Dropout** | Regularization against overfitting | When capacity exceeds data complexity |
| **Learning rate scheduling** | Adapts LR during training | For fine-tuning convergence |

---

## Part 5: Transposed Convolution Details

### Understanding ConvTranspose2d

Transposed convolutions perform the opposite spatial transformation of regular convolutions:

```python
# Regular convolution: reduces spatial size
# Input: 28×28, Output: 14×14 (with stride=2)
nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

# Transposed convolution: increases spatial size
# Input: 14×14, Output: 28×28 (with stride=2)
nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, 
                   padding=1, output_padding=1)
```

### Output Size Calculation

For ConvTranspose2d:

$$H_{\text{out}} = (H_{\text{in}} - 1) \times \text{stride} - 2 \times \text{padding} + \text{kernel} + \text{output\_padding}$$

**Example:** 14×14 → 28×28

$$H_{\text{out}} = (14 - 1) \times 2 - 2 \times 1 + 3 + 1 = 26 - 2 + 3 + 1 = 28 \checkmark$$

### Alternative: Upsample + Conv

Some architectures use bilinear upsampling followed by convolution to avoid checkerboard artifacts:

```python
# Transposed convolution approach
nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)

# Upsample + conv approach (avoids checkerboard artifacts)
nn.Sequential(
    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
    nn.Conv2d(64, 32, kernel_size=3, padding=1)
)
```

---

## Part 6: Parameter Comparison

```python
def compare_model_sizes():
    """
    Compare parameter counts between FC and Conv autoencoders.
    Demonstrates the efficiency of weight sharing in CNNs.
    """
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
    
    conv_ae = DeepConvAutoencoder()
    
    fc_params = sum(p.numel() for p in fc_encoder.parameters()) + \
                sum(p.numel() for p in fc_decoder.parameters())
    conv_params = sum(p.numel() for p in conv_ae.parameters())
    
    print(f"FC Autoencoder:   {fc_params:,} parameters")
    print(f"Conv Autoencoder: {conv_params:,} parameters")
    print(f"Reduction:        {fc_params / conv_params:.2f}x fewer params")
```

**Typical result:** Conv AE achieves better reconstruction with **~5x fewer parameters** than FC AE on image data.

---

## Part 7: Visualization

```python
def visualize_feature_maps(model, test_loader, device):
    """
    Visualize feature maps from encoder layers.
    Shows what spatial features the conv layers detect.
    """
    model.eval()
    
    images, _ = next(iter(test_loader))
    image = images[0:1].to(device)
    
    with torch.no_grad():
        encoder_output = model.encoder(image)
    
    feature_maps = encoder_output[0].cpu().numpy()
    
    num_maps = min(16, feature_maps.shape[0])
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    
    for i, ax in enumerate(axes.flat):
        if i < num_maps:
            ax.imshow(feature_maps[i], cmap='viridis')
        ax.axis('off')
    
    plt.suptitle('Encoder Feature Maps')
    plt.tight_layout()
    plt.savefig('feature_maps.png', dpi=150)
    plt.show()


def visualize_hierarchical_features(model, test_loader, device):
    """
    Visualize activations at different layers of a deep FC autoencoder.
    Shows how representations become more abstract in deeper layers.
    """
    model.eval()
    
    images, labels = next(iter(test_loader))
    image = images[0:1].view(1, -1).to(device)
    label = labels[0].item()
    
    with torch.no_grad():
        layer_outputs = model.get_layer_outputs(image)
        reconstructed, _ = model(image)
    
    num_layers = len(layer_outputs) + 2
    fig, axes = plt.subplots(1, num_layers, figsize=(3 * num_layers, 3))
    
    # Original image
    axes[0].imshow(image.cpu().reshape(28, 28), cmap='gray')
    axes[0].set_title(f'Input (digit {label})')
    axes[0].axis('off')
    
    # Layer activations
    for i, activation in enumerate(layer_outputs):
        act_np = activation.cpu().numpy().flatten()
        size = int(np.ceil(np.sqrt(len(act_np))))
        padded = np.zeros(size * size)
        padded[:len(act_np)] = act_np
        
        axes[i + 1].imshow(padded.reshape(size, size), cmap='viridis')
        axes[i + 1].set_title(f'Layer {i + 1} ({len(act_np)} dim)')
        axes[i + 1].axis('off')
    
    # Reconstructed
    axes[-1].imshow(reconstructed.cpu().reshape(28, 28), cmap='gray')
    axes[-1].set_title('Reconstructed')
    axes[-1].axis('off')
    
    plt.suptitle('Hierarchical Feature Representations')
    plt.tight_layout()
    plt.savefig('hierarchical_features.png', dpi=150)
    plt.show()
```

---

## Historical Note: Layer-wise Pretraining

Before modern optimization techniques (ReLU, batch normalization, Adam), deep autoencoders required **greedy layer-wise pretraining** (Hinton & Salakhutdinov, 2006):

1. Train first autoencoder: $x \to h_1 \to x$
2. Fix encoder₁, train second: $h_1 \to h_2 \to h_1$
3. Continue for all layers
4. Stack all encoders, fine-tune end-to-end

Modern techniques have largely eliminated this need, but the concept remains important for understanding the historical development of deep learning.

---

## Exercises

### Exercise 1: Architecture Exploration
Modify the convolutional autoencoder with different channel counts, layer depths, and kernel sizes. How do these affect reconstruction quality and parameter count?

### Exercise 2: Stride vs Pooling
Compare max pooling after convolution vs strided convolutions for downsampling. Which gives better reconstructions?

### Exercise 3: Color Images (CIFAR-10)
Adapt the conv autoencoder for CIFAR-10 (32×32×3): change input/output channels from 1 to 3 and adjust architecture for the larger spatial size.

### Exercise 4: Skip Connections
Implement U-Net style skip connections that concatenate encoder features to decoder layers. How do residuals affect reconstruction quality?

### Exercise 5: Depth vs Width
Compare a wide-and-shallow architecture (784 → 1024 → 32 → 1024 → 784) with a narrow-and-deep one (784 → 256 → 128 → 64 → 32 → ... mirror). Use similar parameter counts for fair comparison.

---

## Summary

### Convolutional vs Fully-Connected

| Aspect | FC Autoencoder | Conv Autoencoder |
|--------|----------------|------------------|
| Input format | Flattened vector (784) | 2D tensor (1×28×28) |
| Parameters | ~440K | ~80K |
| Spatial structure | Lost | Preserved |
| Translation invariance | No | Yes |
| Weight sharing | No | Yes |
| Best for | Tabular data | Image data |

### Shallow vs Deep

| Aspect | Shallow AE | Deep AE |
|--------|------------|---------|
| Layers | 1–2 per side | 3+ per side |
| Features | Single level | Hierarchical |
| Compression | Limited | Aggressive |
| Training | Easy | Requires batch norm, dropout, gradient clipping |
| Expressivity | Limited | High |

**Key Insight:** Convolutional autoencoders exploit the 2D structure of images through weight sharing and local connectivity, achieving better reconstructions with fewer parameters. Deep architectures (whether convolutional or fully-connected) learn hierarchical representations where each layer captures increasingly abstract features. Modern techniques enable end-to-end training without the historical need for layer-wise pretraining.
