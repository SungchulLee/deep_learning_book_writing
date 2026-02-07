# Autoencoder Architecture

A comprehensive guide to autoencoder architecture design, from basic encoder-decoder structures to convolutional and deep variants.

---

## Overview

**What you'll learn:**

- Basic autoencoder architecture and design principles
- Encoder-decoder structure and information flow
- Modular implementation patterns for production use
- Convolutional autoencoders for image data
- Deep (stacked) autoencoders for hierarchical feature learning
- Architecture design choices and trade-offs

---

## Mathematical Foundation

An autoencoder learns two functions that together approximate the identity:

**Encoder:** $f_\theta: \mathcal{X} \to \mathcal{Z}$ where $\mathcal{Z}$ is the latent space

**Decoder:** $g_\phi: \mathcal{Z} \to \hat{\mathcal{X}}$ where $\hat{\mathcal{X}}$ is the reconstruction

**Objective:** Minimize reconstruction loss

$$\mathcal{L}(\theta, \phi) = \frac{1}{n} \sum_{i=1}^{n} \| x_i - g_\phi(f_\theta(x_i)) \|^2$$

### Architecture Dimensions (MNIST Example)

| Component | Dimension | Description |
|-----------|-----------|-------------|
| Input $x$ | $\mathbb{R}^{784}$ | 28×28 flattened |
| Latent $z$ | $\mathbb{R}^{20}$ | Compressed representation |
| Output $\hat{x}$ | $\mathbb{R}^{784}$ | Reconstruction |

**Compression Ratio:** $784/20 = 39.2\times$

### Standard Architecture Pattern

```
Encoder: 784 → 400 → 20 (latent)
Decoder: 20 → 400 → 784
```

The decoder mirrors the encoder, creating a symmetric hourglass structure.

---

## Part 1: Fully-Connected Autoencoder

### Modular Project Structure

```
ae/
├── global_name_space.py   # Configuration and hyperparameters
├── load_data.py           # Data loading utilities
├── model.py               # Autoencoder architecture
├── train.py               # Training and evaluation loops
├── utils.py               # Visualization utilities
└── main.py                # Entry point
```

### Configuration Module

```python
# global_name_space.py
"""
Central configuration for autoencoder experiments.
All hyperparameters and paths defined in one place.
"""
import argparse
import os
import torch

parser = argparse.ArgumentParser(description='AE MNIST Example')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=128,
                    help='input batch size for training')
parser.add_argument('--data_folder', type=str, default="./data",
                    help='path to dataset')
parser.add_argument('--path', type=str, default="./model",
                    help='path to save model weights')
parser.add_argument('--path_reconstructed_images', type=str, 
                    default="./reconstructed_images",
                    help='path to save reconstruction visualizations')
parser.add_argument('--path_generated_images', type=str, 
                    default="./generated_images",
                    help='path to save generated samples')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='enable CUDA training')
parser.add_argument('--mps', action='store_true', default=True,
                    help='enable macOS GPU training')
parser.add_argument('--log_interval', type=int, default=10,
                    help='batches between logging')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed for reproducibility')

ARGS = parser.parse_args()

# Set random seed for reproducibility
torch.manual_seed(ARGS.seed)

# Create output directories
os.makedirs(ARGS.path, exist_ok=True)
os.makedirs(ARGS.path_reconstructed_images, exist_ok=True)
os.makedirs(ARGS.path_generated_images, exist_ok=True)

# Device configuration with fallback chain: CUDA → MPS → CPU
ARGS.cuda = ARGS.cuda and torch.cuda.is_available()
ARGS.mps = ARGS.mps and torch.backends.mps.is_available()

if ARGS.cuda:
    ARGS.device = torch.device("cuda")
elif ARGS.mps:
    ARGS.device = torch.device("mps")
else:
    ARGS.device = torch.device("cpu")

# DataLoader configuration
# pin_memory speeds up CPU→GPU transfer for CUDA
ARGS.train_kwargs = {'batch_size': ARGS.batch_size, 'shuffle': True}
ARGS.test_kwargs = {'batch_size': ARGS.batch_size, 'shuffle': False}

if ARGS.cuda:
    cuda_kwargs = {'num_workers': 1, 'pin_memory': True}
    ARGS.train_kwargs.update(cuda_kwargs)
    ARGS.test_kwargs.update(cuda_kwargs)
```

### Data Loading Module

```python
# load_data.py
"""
Data loading utilities for MNIST dataset.
"""
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from global_name_space import ARGS


def load_data():
    """
    Load MNIST dataset with minimal preprocessing.
    
    Returns:
        train_loader: DataLoader for training set (60,000 images)
        test_loader: DataLoader for test set (10,000 images)
    
    Note: ToTensor() automatically scales [0, 255] → [0, 1]
    """
    transform = transforms.ToTensor()

    train_data = datasets.MNIST(
        ARGS.data_folder, 
        train=True, 
        download=True, 
        transform=transform
    )
    test_data = datasets.MNIST(
        ARGS.data_folder, 
        train=False, 
        download=False,  # Already downloaded with train
        transform=transform
    )

    train_loader = DataLoader(train_data, **ARGS.train_kwargs)
    test_loader = DataLoader(test_data, **ARGS.test_kwargs)

    return train_loader, test_loader
```

### Model Architecture

```python
# model.py
"""
Autoencoder architecture for MNIST.

Architecture:
    Encoder: 784 → 400 → 20
    Decoder: 20 → 400 → 784
    
The latent dimension of 20 provides ~39x compression while
maintaining reasonable reconstruction quality.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AE(nn.Module):
    """
    Simple fully-connected autoencoder.
    
    Design choices:
    - ReLU activations in hidden layers for non-linearity
    - Sigmoid output to match [0,1] pixel range
    - Symmetric encoder/decoder architecture
    """
    
    def __init__(self, input_dim: int = 784, hidden_dim: int = 400, 
                 latent_dim: int = 20):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Encoder layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder layers
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compress input to latent representation.
        
        Args:
            x: Input tensor of shape (batch, 784)
            
        Returns:
            Latent representation of shape (batch, latent_dim)
        """
        h = F.relu(self.fc1(x))
        return self.fc2(h)  # No activation on latent (can be any real value)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct input from latent representation.
        
        Args:
            z: Latent tensor of shape (batch, latent_dim)
            
        Returns:
            Reconstruction of shape (batch, 784)
        """
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))  # Sigmoid ensures output in [0,1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass: encode then decode.
        
        Args:
            x: Input tensor of shape (batch, 1, 28, 28) or (batch, 784)
            
        Returns:
            Reconstruction of shape (batch, 784)
        """
        # Flatten if needed: (batch, 1, 28, 28) → (batch, 784)
        z = self.encode(x.view(-1, 784))
        return self.decode(z)
```

### Simple Self-Contained Implementation

For quick experimentation, here is a self-contained version:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


class SimpleAutoencoder(nn.Module):
    """
    A simple fully-connected autoencoder for MNIST images.
    
    Architecture:
    - Encoder: Progressively reduces dimensionality
    - Bottleneck: Compressed latent representation
    - Decoder: Mirrors encoder to reconstruct input
    """
    
    def __init__(self, input_dim: int = 784, latent_dim: int = 64):
        super().__init__()
        
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


def main():
    # Setup
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Hyperparameters
    batch_size = 128
    learning_rate = 0.001
    num_epochs = 10
    latent_dim = 64
    
    print(f"Compression ratio: {784 / latent_dim:.2f}x")
    
    # Data
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST('./data', train=True, download=True, 
                                   transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Model
    model = SimpleAutoencoder(latent_dim=latent_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0
        
        for images, _ in train_loader:
            images = images.view(images.size(0), -1).to(device)
            
            optimizer.zero_grad()
            reconstructed, _ = model(images)
            loss = criterion(reconstructed, images)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}/{num_epochs}, Loss: {avg_loss:.6f}")
    
    print("Training complete!")


if __name__ == "__main__":
    main()
```

---

## Part 2: Convolutional Autoencoder

Convolutional autoencoders preserve the 2D spatial structure of images throughout the network, rather than flattening to 1D. This yields better reconstruction with fewer parameters.

### Architecture Comparison

| Type | Data Flow |
|------|-----------|
| Fully Connected | $x \in \mathbb{R}^d \to z \in \mathbb{R}^k \to \hat{x} \in \mathbb{R}^d$ |
| Convolutional | $X \in \mathbb{R}^{H \times W \times C} \to Z \in \mathbb{R}^{h \times w \times c} \to \hat{X} \in \mathbb{R}^{H \times W \times C}$ |

### Encoder Operations

**Convolution:** Extracts local spatial features

$$\text{Output size} = \left\lfloor \frac{\text{input} + 2 \times \text{padding} - \text{kernel}}{\text{stride}} + 1 \right\rfloor$$

**Max Pooling:** Reduces spatial dimensions while preserving important features

$$\text{Output size} = \left\lfloor \frac{\text{input}}{\text{pool\_size}} \right\rfloor$$

### Decoder Operations

**Transposed Convolution (ConvTranspose2d):** Upsamples spatial dimensions

$$\text{Output size} = (\text{input} - 1) \times \text{stride} - 2 \times \text{padding} + \text{kernel} + \text{output\_padding}$$

### FC vs Conv Comparison

| Aspect | FC Autoencoder | Conv Autoencoder |
|--------|----------------|------------------|
| **Spatial structure** | Lost (flattened) | Preserved |
| **Parameters** | ~440K (MNIST) | ~80K (MNIST) |
| **Translation invariance** | No | Yes |
| **Weight sharing** | No | Yes |
| **Best for** | Tabular data | Image data |

### Simple Convolutional Autoencoder

```python
# model.py
"""
Convolutional Autoencoder for MNIST.

Architecture:
    Encoder: 28×28×1 → Conv(16) → 28×28×16 → MaxPool → 14×14×16
    Decoder: 14×14×16 → ConvTranspose → 28×28×1

Key insight: Images stay as 2D tensors throughout, preserving
spatial relationships that would be lost with flattening.
"""
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    """
    Simple Convolutional Autoencoder.
    
    Encoder:
        Conv2d: 1×28×28 → 16×28×28 (same padding)
        MaxPool2d: 16×28×28 → 16×14×14
        
    Decoder:
        ConvTranspose2d: 16×14×14 → 1×28×28
    """
    
    def __init__(self):
        super().__init__()

        # Encoder: Extract features and downsample
        self.encoder = nn.Sequential(
            # Conv: 1×28×28 → 16×28×28
            # padding=1 with kernel=3 preserves spatial size
            nn.Conv2d(
                in_channels=1, 
                out_channels=16, 
                kernel_size=3, 
                stride=1, 
                padding=1
            ),
            nn.ReLU(),
            # Pool: 16×28×28 → 16×14×14
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Decoder: Upsample to original size
        self.decoder = nn.Sequential(
            # ConvTranspose: 16×14×14 → 1×28×28
            nn.ConvTranspose2d(
                in_channels=16, 
                out_channels=1, 
                kernel_size=2, 
                stride=2
            ),
            nn.Sigmoid()  # Output in [0, 1] for pixel values
        )

    def forward(self, x):
        """
        Forward pass preserves 2D structure.
        
        Args:
            x: Input images (batch, 1, 28, 28)
            
        Returns:
            Reconstructed images (batch, 1, 28, 28)
        """
        x = self.encoder(x)
        return self.decoder(x)
```

### Deeper Convolutional Autoencoder

For more complex data or better reconstruction quality:

```python
class DeepConvAutoencoder(nn.Module):
    """
    Deeper convolutional autoencoder with multiple encoding stages.
    
    Encoder: 28×28×1 → 14×14×32 → 7×7×64 → 3×3×128
    Decoder: 3×3×128 → 7×7×64 → 14×14×32 → 28×28×1
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

### Transposed Convolution Details

Transposed convolutions (sometimes called "deconvolutions") perform the opposite spatial transformation of regular convolutions:

```python
# Regular convolution: reduces spatial size
# Input: 28×28, Output: 14×14 (with stride=2)
nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

# Transposed convolution: increases spatial size
# Input: 14×14, Output: 28×28 (with stride=2)
nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, 
                   padding=1, output_padding=1)
```

**Alternative — Upsample + Conv** (avoids checkerboard artifacts):

```python
nn.Sequential(
    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
    nn.Conv2d(64, 32, kernel_size=3, padding=1)
)
```

---

## Part 3: Deep (Stacked) Autoencoder

Deep autoencoders learn **hierarchical feature representations** through multiple encoding/decoding layers.

### Shallow vs Deep Architecture

| Type | Data Flow |
|------|-----------|
| Shallow | $x \to h \to z \to h' \to \hat{x}$ |
| Deep | $x \to h_1 \to h_2 \to \cdots \to z \to \cdots \to h'_2 \to h'_1 \to \hat{x}$ |

Each layer $h_i$ represents progressively more abstract features:

- $h_1$: Low-level features (edges, textures)
- $h_2$: Mid-level features (parts, patterns)
- $z$: High-level abstract representation
- Decoder mirrors encoder in reverse

### Deep Autoencoder Implementation

```python
class DeepAutoencoder(nn.Module):
    """
    Deep autoencoder with multiple encoding and decoding layers.
    
    Architecture: 784 → 512 → 256 → 128 → 32 (bottleneck)
                  32 → 128 → 256 → 512 → 784
    
    Creates a narrow bottleneck with aggressive compression (24.5x).
    """
    
    def __init__(self):
        super(DeepAutoencoder, self).__init__()
        
        # Encoder: Progressive dimensionality reduction
        self.encoder = nn.Sequential(
            # Layer 1: 784 → 512
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            
            # Layer 2: 512 → 256
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            
            # Layer 3: 256 → 128
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            
            # Bottleneck: 128 → 32
            nn.Linear(128, 32),
            nn.ReLU()
        )
        
        # Decoder: Mirror of encoder
        self.decoder = nn.Sequential(
            # Expand from bottleneck: 32 → 128
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            
            # Layer 3: 128 → 256
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            
            # Layer 2: 256 → 512
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            
            # Output: 512 → 784
            nn.Linear(512, 784),
            nn.Sigmoid()
        )
        
        self.latent_dim = 32
    
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

### Greedy Layer-Wise Pretraining (Historical)

Before modern optimization techniques (ReLU, batch norm, Adam), deep autoencoders required **greedy layer-wise pretraining**:

1. Train first autoencoder: $x \to h_1 \to x$
2. Fix encoder₁, train second: $h_1 \to h_2 \to h_1$
3. Continue for all layers
4. Stack all encoders, fine-tune end-to-end

Modern approaches typically do not need this, but understanding the concept is valuable.

```python
class StackedAutoencoder:
    """
    Greedy layer-wise pretraining for deep autoencoders.
    Historical approach (2006-2012) before modern optimization.
    """
    
    def __init__(self, layer_dims):
        """
        Parameters:
        -----------
        layer_dims : List[int]
            Dimensions for each layer, e.g., [784, 512, 256, 128, 32]
        """
        self.layer_dims = layer_dims
        self.autoencoders = []
        self.encoders = []
        
        # Create autoencoder for each layer pair
        for i in range(len(layer_dims) - 1):
            in_dim = layer_dims[i]
            out_dim = layer_dims[i + 1]
            
            ae = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, in_dim),
                nn.Sigmoid() if i == 0 else nn.ReLU()
            )
            self.autoencoders.append(ae)
    
    def pretrain_layer(self, layer_idx, data_loader, device, epochs=5):
        """Pretrain a single layer."""
        ae = self.autoencoders[layer_idx].to(device)
        optimizer = optim.Adam(ae.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        print(f"\nPretraining layer {layer_idx + 1}...")
        
        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0
            
            for images, _ in data_loader:
                images = images.view(images.size(0), -1).to(device)
                
                # If not first layer, encode through previous layers
                if layer_idx > 0:
                    with torch.no_grad():
                        for prev_ae in self.autoencoders[:layer_idx]:
                            encoder = nn.Sequential(*list(prev_ae.children())[:2])
                            images = encoder(images)
                
                # Train current autoencoder
                optimizer.zero_grad()
                reconstructed = ae(images)
                loss = criterion(reconstructed, images)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            print(f"  Epoch {epoch + 1}/{epochs}, "
                  f"Loss: {total_loss / num_batches:.6f}")
        
        # Extract and save encoder
        encoder = nn.Sequential(*list(ae.children())[:2])
        self.encoders.append(encoder)
```

---

## Part 4: Flexible Autoencoder for Experimentation

This implementation supports both undercomplete and overcomplete configurations:

```python
class FlexibleAutoencoder(nn.Module):
    """
    Autoencoder that can be undercomplete or overcomplete.
    """
    
    def __init__(self, input_dim=784, latent_dim=64, hidden_dim=256):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z
    
    @property
    def compression_ratio(self):
        return self.input_dim / self.latent_dim
    
    @property
    def is_undercomplete(self):
        return self.latent_dim < self.input_dim
```

---

## Part 5: Architecture Design Choices

### Activation Functions

| Choice | Reason |
|--------|--------|
| **ReLU in hidden layers** | Non-linearity for learning complex patterns |
| **Sigmoid output** | Ensures output in [0, 1] matching normalized images |
| **Symmetric architecture** | Balanced capacity for encoding and decoding |
| **Progressive narrowing** | Gradual compression prevents information loss |

```python
# Hidden layers: ReLU (standard choice)
nn.ReLU()  # Fast, sparse gradients, works well in practice

# Output layer options:
nn.Sigmoid()  # Data in [0, 1], use with MSE or BCE
nn.Tanh()     # Data in [-1, 1], use with MSE
None          # Unbounded data, use with MSE
```

### Loss Function Selection

| Loss | Output Activation | Data Range | Best For |
|------|-------------------|------------|----------|
| MSE | Sigmoid or None | [0, 1] or unbounded | Continuous data |
| BCE | Sigmoid | [0, 1] | Binary/normalized images |
| MAE | Any | Any | Robust to outliers |

### Shallow vs Deep Comparison

| Aspect | Shallow AE | Deep AE |
|--------|------------|---------|
| Layers | 1-2 per side | 3+ per side |
| Features | Single level | Hierarchical |
| Compression | Limited | Aggressive |
| Training | Easy | Requires regularization |
| Expressivity | Limited | High |

### Key Training Techniques for Deep Networks

1. **Gradient clipping** — prevents exploding gradients
2. **Batch normalization** — stabilizes training
3. **Dropout** — regularization for deep networks
4. **Learning rate scheduling** — adapts LR during training

---

## Visualization Utilities

```python
# utils.py
"""
Visualization utilities for autoencoder analysis.
"""
import matplotlib.pyplot as plt
import torch


def visualize_reconstructions(model, test_loader, device, num_images=10):
    """
    Compare original images with their reconstructions.
    """
    model.eval()
    
    images, _ = next(iter(test_loader))
    images = images[:num_images].to(device)
    
    with torch.no_grad():
        reconstructed = model(images).view(-1, 28, 28).cpu()
    
    images = images.cpu().squeeze()
    
    fig, axes = plt.subplots(2, num_images, figsize=(15, 3))
    
    for i in range(num_images):
        axes[0, i].imshow(images[i], cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(reconstructed[i], cmap='gray')
        axes[1, i].axis('off')
    
    axes[0, 0].set_ylabel('Original')
    axes[1, 0].set_ylabel('Reconstructed')
    
    plt.tight_layout()
    plt.savefig('reconstruction_comparison.png', dpi=150)
    plt.show()


def visualize_feature_maps(model, test_loader, device):
    """
    Visualize feature maps from convolutional encoder layers.
    Shows what spatial features the conv layers detect.
    """
    model.eval()
    
    images, _ = next(iter(test_loader))
    image = images[0:1].to(device)
    
    # Get intermediate activations
    with torch.no_grad():
        encoder_output = model.encoder(image)
    
    # Visualize feature maps
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
```

---

## Exercises

### Exercise 1: Architecture Exploration

Modify the autoencoder architecture:

a) Change `latent_dim` to 2, 16, 32, 128 and compare reconstruction quality
b) Add more layers (e.g., 784 → 512 → 256 → 128 → 64)
c) Try different activation functions (LeakyReLU, ELU)

**Questions:** How does latent dimension affect reconstruction quality? Is there a "sweet spot" for compression?

### Exercise 2: Latent Space Exploration

With `latent_dim=2`:

a) Train the autoencoder
b) Visualize the 2D latent space colored by digit class
c) Sample random points and decode them
d) Perform arithmetic: $z_{new} = z_1 + z_2 - z_3$

### Exercise 3: Interpolation

Given two images $x_1$ and $x_2$:

1. Encode to get $z_1$ and $z_2$
2. Interpolate: $z_t = (1-t) z_1 + t z_2$ for $t \in [0, 1]$
3. Decode each $z_t$
4. Visualize the interpolation sequence

### Exercise 4: Stride vs Pooling (Conv AE)

Compare two downsampling strategies:

```python
# Option A: Conv + Pool
nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2))

# Option B: Strided Conv
nn.Sequential(nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU())
```

Which gives better reconstructions? How do parameter counts compare?

### Exercise 5: Depth vs Performance

Train autoencoders with different depths, all with `latent_dim=32`:

- **Shallow:** 784 → 128 → 32 → 128 → 784
- **Medium:** 784 → 512 → 256 → 32 → 256 → 512 → 784
- **Deep:** 784 → 512 → 384 → 256 → 128 → 32 → ... (mirror)

Is deeper always better? What is the optimal depth for MNIST?

---

## Summary

| Component | Description |
|-----------|-------------|
| **Encoder** | Maps input to lower-dimensional latent space |
| **Decoder** | Reconstructs input from latent representation |
| **FC AE** | Flattened input, suitable for tabular data |
| **Conv AE** | Preserves spatial structure, fewer parameters for images |
| **Deep AE** | Hierarchical features, aggressive compression |
| **Loss** | MSE or BCE between input and reconstruction |
| **Training** | Minimize reconstruction error via gradient descent |

**Key Insight:** The bottleneck architecture forces the network to learn a compressed representation that captures the most important features of the data. Convolutional variants exploit spatial structure for efficiency, while deeper architectures learn richer hierarchical representations.
