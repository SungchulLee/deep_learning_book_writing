# DCGAN (Deep Convolutional GAN)

DCGAN, introduced by Radford et al. in 2016, established architectural guidelines that enabled stable training of convolutional GANs. These principles remain foundational for modern GAN architectures.

## Historical Significance

Before DCGAN, training GANs with convolutional layers was notoriously unstable. DCGAN provided a set of architectural guidelines that dramatically improved training stability and sample quality.

## Architectural Guidelines

### The Five DCGAN Principles

1. **Replace pooling with strided convolutions**
   - Discriminator: strided convolutions for downsampling
   - Generator: transposed convolutions for upsampling

2. **Use batch normalization**
   - In both G and D
   - Exception: G output layer, D input layer

3. **Remove fully connected hidden layers**
   - Use global average pooling or fully convolutional
   - Connect latent vector directly to convolutions

4. **Generator activations**
   - ReLU for all hidden layers
   - Tanh for output layer

5. **Discriminator activations**
   - LeakyReLU (slope 0.2) for all layers

## Generator Architecture

### Structure Overview

```
Input: z ∈ ℝ^100 (latent vector)
    ↓
Project and Reshape: z → (512, 4, 4)
    ↓
ConvTranspose2d: (512, 4, 4) → (256, 8, 8)
BatchNorm2d + ReLU
    ↓
ConvTranspose2d: (256, 8, 8) → (128, 16, 16)
BatchNorm2d + ReLU
    ↓
ConvTranspose2d: (128, 16, 16) → (64, 32, 32)
BatchNorm2d + ReLU
    ↓
ConvTranspose2d: (64, 32, 32) → (3, 64, 64)
Tanh
    ↓
Output: image ∈ [-1, 1]^(3×64×64)
```

### PyTorch Implementation

```python
import torch
import torch.nn as nn

class DCGANGenerator(nn.Module):
    """
    DCGAN Generator for 64x64 images.
    
    Transforms latent vector z to image using transposed convolutions.
    Follows all DCGAN architectural guidelines.
    """
    
    def __init__(self, latent_dim=100, feature_maps=64, image_channels=3):
        """
        Args:
            latent_dim: Dimension of input latent vector
            feature_maps: Base number of feature maps (scales by 2x)
            image_channels: Output image channels (1=grayscale, 3=RGB)
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        ngf = feature_maps  # Number of generator feature maps
        
        self.main = nn.Sequential(
            # Input: z (latent_dim x 1 x 1)
            # Output: (ngf*8) x 4 x 4
            nn.ConvTranspose2d(latent_dim, ngf * 8, 
                              kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            
            # (ngf*8) x 4 x 4 → (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4,
                              kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            
            # (ngf*4) x 8 x 8 → (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2,
                              kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            
            # (ngf*2) x 16 x 16 → ngf x 32 x 32
            nn.ConvTranspose2d(ngf * 2, ngf,
                              kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            
            # ngf x 32 x 32 → image_channels x 64 x 64
            nn.ConvTranspose2d(ngf, image_channels,
                              kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()  # Output in [-1, 1] to match normalized images
        )
    
    def forward(self, z):
        """
        Generate image from latent vector.
        
        Args:
            z: Latent vector, shape (batch_size, latent_dim) or 
               (batch_size, latent_dim, 1, 1)
        
        Returns:
            Generated image, shape (batch_size, channels, 64, 64)
        """
        # Reshape if needed
        if z.dim() == 2:
            z = z.view(z.size(0), z.size(1), 1, 1)
        
        return self.main(z)
```

### Generator for 28x28 (MNIST)

```python
class DCGANGeneratorMNIST(nn.Module):
    """DCGAN Generator for 28x28 images (MNIST)."""
    
    def __init__(self, latent_dim=100, feature_maps=64):
        super().__init__()
        
        self.latent_dim = latent_dim
        ngf = feature_maps
        
        # Project and reshape: z → (ngf*8, 7, 7)
        self.project = nn.Sequential(
            nn.Linear(latent_dim, ngf * 8 * 7 * 7),
            nn.BatchNorm1d(ngf * 8 * 7 * 7),
            nn.ReLU(True)
        )
        
        self.conv = nn.Sequential(
            # (ngf*8) x 7 x 7 → (ngf*4) x 14 x 14
            nn.ConvTranspose2d(ngf * 8, ngf * 4,
                              kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            
            # (ngf*4) x 14 x 14 → (ngf*2) x 28 x 28
            nn.ConvTranspose2d(ngf * 4, ngf * 2,
                              kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            
            # (ngf*2) x 28 x 28 → 1 x 28 x 28
            nn.Conv2d(ngf * 2, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, z):
        if z.dim() == 4:
            z = z.view(z.size(0), -1)
        
        x = self.project(z)
        x = x.view(x.size(0), -1, 7, 7)
        return self.conv(x)
```

## Discriminator Architecture

### Structure Overview

```
Input: image ∈ [-1, 1]^(3×64×64)
    ↓
Conv2d: (3, 64, 64) → (64, 32, 32)
LeakyReLU(0.2)
    ↓
Conv2d: (64, 32, 32) → (128, 16, 16)
BatchNorm2d + LeakyReLU(0.2)
    ↓
Conv2d: (128, 16, 16) → (256, 8, 8)
BatchNorm2d + LeakyReLU(0.2)
    ↓
Conv2d: (256, 8, 8) → (512, 4, 4)
BatchNorm2d + LeakyReLU(0.2)
    ↓
Conv2d: (512, 4, 4) → (1, 1, 1)
Sigmoid
    ↓
Output: probability ∈ [0, 1]
```

### PyTorch Implementation

```python
class DCGANDiscriminator(nn.Module):
    """
    DCGAN Discriminator for 64x64 images.
    
    Classifies images as real/fake using strided convolutions.
    Follows all DCGAN architectural guidelines.
    """
    
    def __init__(self, image_channels=3, feature_maps=64):
        """
        Args:
            image_channels: Input image channels (1=grayscale, 3=RGB)
            feature_maps: Base number of feature maps (scales by 2x)
        """
        super().__init__()
        
        ndf = feature_maps  # Number of discriminator feature maps
        
        self.main = nn.Sequential(
            # Input: image_channels x 64 x 64
            # No BatchNorm on first layer (DCGAN guideline)
            nn.Conv2d(image_channels, ndf,
                     kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # Output: ndf x 32 x 32
            
            # ndf x 32 x 32 → (ndf*2) x 16 x 16
            nn.Conv2d(ndf, ndf * 2,
                     kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # (ndf*2) x 16 x 16 → (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4,
                     kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # (ndf*4) x 8 x 8 → (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 4, ndf * 8,
                     kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # (ndf*8) x 4 x 4 → 1 x 1 x 1
            nn.Conv2d(ndf * 8, 1,
                     kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        """
        Classify image as real or fake.
        
        Args:
            img: Input image, shape (batch_size, channels, 64, 64)
        
        Returns:
            Probability of being real, shape (batch_size, 1)
        """
        output = self.main(img)
        return output.view(-1, 1)
```

## Spatial Size Calculations

### Transposed Convolution (Upsampling)

$$H_{out} = (H_{in} - 1) \times \text{stride} - 2 \times \text{padding} + \text{kernel\_size}$$

### Standard Convolution (Downsampling)

$$H_{out} = \lfloor \frac{H_{in} + 2 \times \text{padding} - \text{kernel\_size}}{\text{stride}} \rfloor + 1$$

### Common Patterns

| Operation | Kernel | Stride | Padding | Effect |
|-----------|--------|--------|---------|--------|
| ConvTranspose2d | 4 | 2 | 1 | 2× upsample |
| Conv2d | 4 | 2 | 1 | 2× downsample |
| ConvTranspose2d | 4 | 1 | 0 | 1→4 (initial) |
| Conv2d | 4 | 1 | 0 | 4→1 (final) |

## Weight Initialization

DCGAN specifies particular weight initialization:

```python
def weights_init_dcgan(m):
    """
    Initialize weights following DCGAN recommendations.
    
    - Conv/ConvTranspose: Normal(0, 0.02)
    - BatchNorm: weight ~ Normal(1, 0.02), bias = 0
    """
    classname = m.__class__.__name__
    
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Apply initialization
generator.apply(weights_init_dcgan)
discriminator.apply(weights_init_dcgan)
```

## Training Configuration

### Recommended Hyperparameters

```python
config = {
    'latent_dim': 100,
    'feature_maps': 64,
    'lr': 0.0002,
    'beta1': 0.5,
    'beta2': 0.999,
    'batch_size': 128,
    'n_epochs': 25,
}
```

### Complete Training Script

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

def train_dcgan(config, device='cuda'):
    """Train DCGAN on a dataset."""
    
    # Data
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])
    
    dataset = datasets.MNIST(root='./data', train=True, 
                            download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'],
                           shuffle=True, num_workers=4)
    
    # Networks
    G = DCGANGenerator(config['latent_dim'], config['feature_maps'], 1).to(device)
    D = DCGANDiscriminator(1, config['feature_maps']).to(device)
    
    G.apply(weights_init_dcgan)
    D.apply(weights_init_dcgan)
    
    # Optimizers
    g_optimizer = optim.Adam(G.parameters(), lr=config['lr'], 
                            betas=(config['beta1'], config['beta2']))
    d_optimizer = optim.Adam(D.parameters(), lr=config['lr'],
                            betas=(config['beta1'], config['beta2']))
    
    # Loss
    criterion = nn.BCELoss()
    
    # Fixed noise for visualization
    fixed_noise = torch.randn(64, config['latent_dim'], 1, 1, device=device)
    
    # Training loop
    for epoch in range(config['n_epochs']):
        for i, (real_images, _) in enumerate(dataloader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            
            # Labels
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)
            
            # ==================
            # Train Discriminator
            # ==================
            d_optimizer.zero_grad()
            
            # Real images
            d_real = D(real_images)
            d_loss_real = criterion(d_real, real_labels)
            
            # Fake images
            z = torch.randn(batch_size, config['latent_dim'], 1, 1, device=device)
            fake_images = G(z)
            d_fake = D(fake_images.detach())
            d_loss_fake = criterion(d_fake, fake_labels)
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()
            
            # ===============
            # Train Generator
            # ===============
            g_optimizer.zero_grad()
            
            d_fake = D(fake_images)
            g_loss = criterion(d_fake, real_labels)
            
            g_loss.backward()
            g_optimizer.step()
            
            # Logging
            if i % 100 == 0:
                print(f'Epoch [{epoch}/{config["n_epochs"]}] '
                      f'Batch [{i}/{len(dataloader)}] '
                      f'D_loss: {d_loss.item():.4f} '
                      f'G_loss: {g_loss.item():.4f}')
        
        # Save samples
        with torch.no_grad():
            fake_samples = G(fixed_noise)
            fake_samples = (fake_samples + 1) / 2  # Denormalize
            save_image(fake_samples, f'samples_epoch_{epoch}.png', nrow=8)
    
    return G, D
```

## Why DCGAN Works

### Strided Convolutions vs Pooling

Pooling (max/avg) discards spatial information. Strided convolutions learn what to discard:

```python
# Pooling: fixed operation
x = nn.MaxPool2d(2)(x)  # Loses information

# Strided conv: learned downsampling
x = nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1)(x)  # Learns what matters
```

### Batch Normalization Benefits

1. **Stabilizes activations**: Prevents internal covariate shift
2. **Enables higher learning rates**: Gradients are better conditioned
3. **Regularization effect**: Reduces need for dropout

### LeakyReLU in Discriminator

Standard ReLU creates sparse gradients (dead neurons). LeakyReLU maintains gradient flow:

```python
# ReLU: gradient = 0 for x < 0
relu = nn.ReLU()

# LeakyReLU: gradient = 0.2 for x < 0
leaky = nn.LeakyReLU(0.2)  # Small but non-zero gradient
```

## Summary

| Principle | Generator | Discriminator |
|-----------|-----------|---------------|
| **Downsampling/Upsampling** | ConvTranspose stride=2 | Conv stride=2 |
| **Batch Normalization** | All except output | All except input |
| **Activation (hidden)** | ReLU | LeakyReLU(0.2) |
| **Activation (output)** | Tanh | Sigmoid |
| **Weight Init** | Normal(0, 0.02) | Normal(0, 0.02) |

DCGAN's architectural guidelines established the foundation for stable GAN training and remain influential in modern architectures like StyleGAN, BigGAN, and many others.

## Complete Implementation

The following provides a production-ready DCGAN implementation with multi-dataset support and checkpointing.

## Configuration

```python
"""
global_name_space.py

DCGAN configuration supporting multiple datasets.
"""

import random
import os
import argparse
import torch

parser = argparse.ArgumentParser()

# =============================================================================
# Dataset Configuration
# =============================================================================
parser.add_argument('--dataset', type=str, default="mnist",
                    choices=['cifar10', 'lsun', 'mnist', 'imagenet', 'folder', 'lfw', 'fake'],
                    help='Dataset to train on')
parser.add_argument('--dataroot', default="./data",
                    help='Path to dataset')
parser.add_argument('--workers', type=int, default=2,
                    help='Number of data loading workers')

# =============================================================================
# Image Configuration
# =============================================================================
parser.add_argument('--imageSize', type=int, default=64,
                    help='Height/width of input images')
parser.add_argument('--nz', type=int, default=100,
                    help='Size of latent vector z')
parser.add_argument('--ngf', type=int, default=64,
                    help='Generator feature map multiplier')
parser.add_argument('--ndf', type=int, default=64,
                    help='Discriminator feature map multiplier')

# =============================================================================
# Training Configuration
# =============================================================================
parser.add_argument('--niter', type=int, default=25,
                    help='Number of training epochs')
parser.add_argument('--batchSize', type=int, default=64,
                    help='Training batch size')
parser.add_argument('--lr', type=float, default=0.0002,
                    help='Learning rate')
parser.add_argument('--beta1', type=float, default=0.5,
                    help='Adam beta1')

# =============================================================================
# Model Checkpoints
# =============================================================================
parser.add_argument('--ngpu', type=int, default=1,
                    help='Number of GPUs')
parser.add_argument('--netG', default='',
                    help='Path to Generator checkpoint (for resuming)')
parser.add_argument('--netD', default='',
                    help='Path to Discriminator checkpoint (for resuming)')
parser.add_argument('--outf', default='./fake_images',
                    help='Output folder for generated images')
parser.add_argument('--modelf', default='./model',
                    help='Output folder for model checkpoints')

# =============================================================================
# Hardware
# =============================================================================
parser.add_argument('--cuda', action='store_true', default=False,
                    help='Enable CUDA')
parser.add_argument('--mps', action='store_true', default=False,
                    help='Enable Apple Silicon GPU')
parser.add_argument('--manualSeed', type=int,
                    help='Manual seed for reproducibility')
parser.add_argument('--dry-run', action='store_true',
                    help='Run single training cycle for testing')

opt = parser.parse_args()

# Handle dry-run mode
if opt.dry_run:
    opt.niter = 1

# Create output directories
os.makedirs(opt.outf, exist_ok=True)
os.makedirs(opt.modelf, exist_ok=True)

# Set random seed
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

# Configure device
opt.use_mps = opt.mps and torch.backends.mps.is_available()
if opt.cuda:
    device = torch.device("cuda:0")
elif opt.use_mps:
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Hardware warnings
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: CUDA available but not enabled. Use --cuda")
elif torch.backends.mps.is_available() and not opt.mps:
    print("WARNING: Apple Silicon GPU available. Use --mps")
```

## Multi-Dataset Data Loading

```python
"""
load_data.py

Flexible data loading supporting multiple image datasets.
"""

import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torch

from global_name_space import opt, device


def load_data():
    """
    Load dataset based on configuration.
    
    Supports:
    - MNIST (1 channel, grayscale)
    - CIFAR-10 (3 channels, RGB)
    - ImageNet/folder datasets (3 channels, RGB)
    - LSUN scenes (3 channels, RGB)
    - Fake data (for testing)
    
    Returns:
        dataloader: DataLoader for training
        nc: Number of image channels
        classes: Class names (if applicable)
    """
    
    if opt.dataset in ['imagenet', 'folder', 'lfw']:
        # Custom folder of images
        classes = None
        dataset = datasets.ImageFolder(
            root=opt.dataroot,
            transform=transforms.Compose([
                transforms.Resize(opt.imageSize),
                transforms.CenterCrop(opt.imageSize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        )
        nc = 3
        
    elif opt.dataset == 'lsun':
        # LSUN scene dataset
        classes = [c + '_train' for c in opt.classes.split(',')]
        dataset = datasets.LSUN(
            root=opt.dataroot, 
            classes=classes,
            transform=transforms.Compose([
                transforms.Resize(opt.imageSize),
                transforms.CenterCrop(opt.imageSize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        )
        nc = 3
        
    elif opt.dataset == 'cifar10':
        # CIFAR-10
        classes = None
        dataset = datasets.CIFAR10(
            root=opt.dataroot, 
            download=True,
            transform=transforms.Compose([
                transforms.Resize(opt.imageSize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        )
        nc = 3
        
    elif opt.dataset == 'mnist':
        # MNIST (grayscale)
        classes = None
        dataset = datasets.MNIST(
            root=opt.dataroot, 
            download=True,
            transform=transforms.Compose([
                transforms.Resize(opt.imageSize),
                transforms.ToTensor(),
                # Single channel normalization
                transforms.Normalize((0.5,), (0.5,)),
            ])
        )
        nc = 1
        
    elif opt.dataset == 'fake':
        # Fake data for testing pipeline
        classes = None
        dataset = datasets.FakeData(
            image_size=(3, opt.imageSize, opt.imageSize),
            transform=transforms.ToTensor()
        )
        nc = 3
    
    assert dataset, f"Unknown dataset: {opt.dataset}"
    
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=opt.batchSize,
        shuffle=True, 
        num_workers=int(opt.workers)
    )
    
    return dataloader, nc, classes


def show_batch_images(dataloader, n_images=10):
    """Display sample images from dataloader."""
    images, labels = next(iter(dataloader))
    
    fig, axes = plt.subplots(1, min(opt.batchSize, n_images), figsize=(12, 3))
    
    for i, (image, label) in enumerate(zip(images, labels)):
        if i >= n_images:
            break
            
        # Unnormalize
        image = image / 2 + 0.5
        image = image.permute(1, 2, 0).numpy().squeeze()
        
        axes[i].imshow(image, cmap='gray' if image.ndim == 2 else None)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    dataloader, nc, classes = load_data()
    print(f"Dataset: {opt.dataset}, Channels: {nc}")
    show_batch_images(dataloader)
```

## DCGAN Model Architecture

```python
"""
model.py

DCGAN Generator and Discriminator with convolutional architectures.
Follows all DCGAN architectural guidelines.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from global_name_space import opt, device
from load_data import load_data


def weights_init(m):
    """
    Initialize weights following DCGAN recommendations.
    
    - Convolutional layers: Normal(0, 0.02)
    - BatchNorm: weight ~ Normal(1, 0.02), bias = 0
    
    This initialization helps stabilize training by ensuring
    activations start in a reasonable range.
    """
    classname = m.__class__.__name__
    
    if classname.find('Conv') != -1:
        # Convolutional layers
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        
    elif classname.find('BatchNorm') != -1:
        # BatchNorm layers
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias.data)


class Generator(nn.Module):
    """
    DCGAN Generator using transposed convolutions.
    
    Architecture (for 64x64 output):
        z (nz×1×1) → ngf*8×4×4 → ngf*4×8×8 → ngf*2×16×16 → ngf×32×32 → nc×64×64
    
    Each block:
        ConvTranspose2d → BatchNorm → ReLU
    
    Output block:
        ConvTranspose2d → Tanh
    """
    
    def __init__(self, nc):
        """
        Args:
            nc: Number of output channels (1 for grayscale, 3 for RGB)
        """
        super().__init__()
        
        nz = int(opt.nz)    # Latent dimension
        ngf = int(opt.ngf)  # Feature map multiplier
        self.ngpu = int(opt.ngpu)
        
        self.main = nn.Sequential(
            # =================================================================
            # Block 1: z → ngf*8 × 4 × 4
            # Input is latent vector z, going into a convolution
            # =================================================================
            nn.ConvTranspose2d(
                in_channels=nz,
                out_channels=ngf * 8,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False  # No bias when using BatchNorm
            ),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            
            # =================================================================
            # Block 2: ngf*8 × 4 × 4 → ngf*4 × 8 × 8
            # =================================================================
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            
            # =================================================================
            # Block 3: ngf*4 × 8 × 8 → ngf*2 × 16 × 16
            # =================================================================
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            
            # =================================================================
            # Block 4: ngf*2 × 16 × 16 → ngf × 32 × 32
            # =================================================================
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            
            # =================================================================
            # Output: ngf × 32 × 32 → nc × 64 × 64
            # No BatchNorm on output layer (DCGAN guideline)
            # =================================================================
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, input):
        """
        Generate images from latent vectors.
        
        Args:
            input: Latent vectors, shape (batch, nz, 1, 1)
            
        Returns:
            Generated images, shape (batch, nc, 64, 64)
        """
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class Discriminator(nn.Module):
    """
    DCGAN Discriminator using strided convolutions.
    
    Architecture (for 64x64 input):
        nc×64×64 → ndf×32×32 → ndf*2×16×16 → ndf*4×8×8 → ndf*8×4×4 → 1×1×1
    
    Each block:
        Conv2d → BatchNorm → LeakyReLU
    
    Input block:
        Conv2d → LeakyReLU (no BatchNorm on input)
    
    Output block:
        Conv2d → Sigmoid
    """
    
    def __init__(self, nc):
        """
        Args:
            nc: Number of input channels (1 for grayscale, 3 for RGB)
        """
        super().__init__()
        
        ndf = int(opt.ndf)  # Feature map multiplier
        self.ngpu = int(opt.ngpu)
        
        self.main = nn.Sequential(
            # =================================================================
            # Input: nc × 64 × 64 → ndf × 32 × 32
            # No BatchNorm on input (DCGAN guideline)
            # =================================================================
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # =================================================================
            # Block 2: ndf × 32 × 32 → ndf*2 × 16 × 16
            # =================================================================
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # =================================================================
            # Block 3: ndf*2 × 16 × 16 → ndf*4 × 8 × 8
            # =================================================================
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # =================================================================
            # Block 4: ndf*4 × 8 × 8 → ndf*8 × 4 × 4
            # =================================================================
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # =================================================================
            # Output: ndf*8 × 4 × 4 → 1 × 1 × 1
            # =================================================================
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, input):
        """
        Classify images as real or fake.
        
        Args:
            input: Images, shape (batch, nc, 64, 64)
            
        Returns:
            Probabilities, shape (batch,)
        """
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        
        # Flatten output
        return output.view(-1, 1).squeeze(1)


def show_generated_samples(netG, n_samples=10):
    """Generate and display samples from the generator."""
    netG.eval()
    
    with torch.no_grad():
        z = torch.randn(n_samples, opt.nz, 1, 1, device=device)
        samples = netG(z)
    
    fig, axes = plt.subplots(1, n_samples, figsize=(12, 3))
    
    for i, image in enumerate(samples):
        image = image.cpu().numpy()
        image = np.transpose(image, (1, 2, 0))
        
        # Handle grayscale vs RGB
        if image.shape[-1] == 1:
            image = image.squeeze()
            axes[i].imshow(image, cmap='gray')
        else:
            image = (image + 1) / 2  # Unnormalize
            axes[i].imshow(image)
        
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    netG.train()


if __name__ == "__main__":
    dataloader, nc, _ = load_data()
    
    netG = Generator(nc).to(device)
    netD = Discriminator(nc).to(device)
    
    # Apply weight initialization
    netG.apply(weights_init)
    netD.apply(weights_init)
    
    print(f"Generator parameters: {sum(p.numel() for p in netG.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in netD.parameters()):,}")
    
    show_generated_samples(netG)
```

## Training with Checkpointing

```python
"""
main.py

DCGAN training with checkpointing and resume capability.
"""

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.utils as vutils

from global_name_space import opt, device
from load_data import load_data
from model import Generator, Discriminator, weights_init


def train(netG, netD, dataloader, optimizerG, optimizerD, criterion, 
          real_label, fake_label, start_epoch):
    """
    Train DCGAN with logging and checkpointing.
    
    Args:
        netG, netD: Generator and Discriminator networks
        dataloader: Training data loader
        optimizerG, optimizerD: Optimizers for G and D
        criterion: Loss function (BCELoss)
        real_label, fake_label: Label values (1 and 0)
        start_epoch: Starting epoch (for resuming)
    """
    
    # Fixed noise for visualizing progress
    fixed_noise = torch.randn(opt.batchSize, opt.nz, 1, 1, device=device)
    
    for epoch in range(start_epoch, opt.niter):
        for i, data in enumerate(dataloader, 0):
            
            # =================================================================
            # (1) Update Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
            # =================================================================
            
            # Train with real images
            netD.zero_grad()
            real_cpu = data[0].to(device)
            batch_size = real_cpu.size(0)
            
            # Create labels
            label = torch.full(
                (batch_size,), 
                real_label,
                dtype=real_cpu.dtype, 
                device=device
            )
            
            # Forward pass with real images
            output = netD(real_cpu)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()  # Average D output on real images
            
            # Train with fake images
            noise = torch.randn(batch_size, opt.nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            
            # Forward pass with fake images (detached to not train G)
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()  # Average D output on fake images
            
            # Total discriminator loss
            errD = errD_real + errD_fake
            optimizerD.step()
            
            # =================================================================
            # (2) Update Generator: maximize log(D(G(z)))
            # =================================================================
            
            netG.zero_grad()
            label.fill_(real_label)  # Fake labels are real for generator cost
            
            # Forward pass (D should see fakes as real)
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()  # D output after G update
            
            optimizerG.step()
            
            # =================================================================
            # Logging
            # =================================================================
            print(f'[{epoch}/{opt.niter}][{i}/{len(dataloader)}] '
                  f'Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} '
                  f'D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}')
            
            # Save sample images
            if i % 100 == 0:
                vutils.save_image(
                    real_cpu,
                    f'real_samples.png',
                    normalize=True
                )
                
                with torch.no_grad():
                    fake_display = netG(fixed_noise)
                vutils.save_image(
                    fake_display.detach(),
                    f'{opt.outf}/fake_samples_epoch_{epoch:03d}.png',
                    normalize=True
                )
            
            if opt.dry_run:
                break
        
        # Save checkpoints after each epoch
        torch.save(netG.state_dict(), f'{opt.modelf}/netG_epoch_{epoch}.pth')
        torch.save(netD.state_dict(), f'{opt.modelf}/netD_epoch_{epoch}.pth')


def main():
    """Main training pipeline."""
    
    # Enable cuDNN benchmarking for faster training
    cudnn.benchmark = True
    
    # Label values
    real_label = 1
    fake_label = 0
    
    # Load data
    dataloader, nc, classes = load_data()
    
    # Initialize models
    netG = Generator(nc).to(device)
    netD = Discriminator(nc).to(device)
    
    # Apply DCGAN weight initialization
    netG.apply(weights_init)
    netD.apply(weights_init)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
        netD.load_state_dict(torch.load(opt.netD))
        # Extract epoch number from filename
        start_epoch = int(opt.netG.split("_")[-1].split(".")[0]) + 1
        print(f"Resuming from epoch {start_epoch}")
    
    # Loss function
    criterion = nn.BCELoss()
    
    # Optimizers with DCGAN hyperparameters
    optimizerD = optim.Adam(
        netD.parameters(), 
        lr=opt.lr, 
        betas=(opt.beta1, 0.999)
    )
    optimizerG = optim.Adam(
        netG.parameters(), 
        lr=opt.lr, 
        betas=(opt.beta1, 0.999)
    )
    
    # Train
    train(netG, netD, dataloader, optimizerG, optimizerD, 
          criterion, real_label, fake_label, start_epoch)


if __name__ == "__main__":
    main()
```

## Understanding Transposed Convolution Dimensions

The DCGAN generator uses transposed convolutions for upsampling. Understanding the output dimensions is crucial:

$$H_{out} = (H_{in} - 1) \times \text{stride} - 2 \times \text{padding} + \text{kernel\_size}$$

For the common upsampling pattern (kernel=4, stride=2, padding=1):

$$H_{out} = (H_{in} - 1) \times 2 - 2 \times 1 + 4 = 2H_{in}$$

This doubles the spatial dimensions at each layer.

## Architecture Dimensions

| Layer | Operation | Input | Output |
|-------|-----------|-------|--------|
| G Block 1 | ConvT(nz, ngf*8, k=4, s=1, p=0) | nz×1×1 | ngf*8×4×4 |
| G Block 2 | ConvT(ngf*8, ngf*4, k=4, s=2, p=1) | ngf*8×4×4 | ngf*4×8×8 |
| G Block 3 | ConvT(ngf*4, ngf*2, k=4, s=2, p=1) | ngf*4×8×8 | ngf*2×16×16 |
| G Block 4 | ConvT(ngf*2, ngf, k=4, s=2, p=1) | ngf*2×16×16 | ngf×32×32 |
| G Output | ConvT(ngf, nc, k=4, s=2, p=1) | ngf×32×32 | nc×64×64 |

| Layer | Operation | Input | Output |
|-------|-----------|-------|--------|
| D Block 1 | Conv(nc, ndf, k=4, s=2, p=1) | nc×64×64 | ndf×32×32 |
| D Block 2 | Conv(ndf, ndf*2, k=4, s=2, p=1) | ndf×32×32 | ndf*2×16×16 |
| D Block 3 | Conv(ndf*2, ndf*4, k=4, s=2, p=1) | ndf*2×16×16 | ndf*4×8×8 |
| D Block 4 | Conv(ndf*4, ndf*8, k=4, s=2, p=1) | ndf*4×8×8 | ndf*8×4×4 |
| D Output | Conv(ndf*8, 1, k=4, s=1, p=0) | ndf*8×4×4 | 1×1×1 |

## Summary

| DCGAN Principle | Implementation |
|-----------------|----------------|
| Strided convolutions | `stride=2` for up/downsampling |
| Batch normalization | All layers except G output, D input |
| No fully connected | Direct conv from z to feature maps |
| ReLU in G | `nn.ReLU(True)` |
| LeakyReLU in D | `nn.LeakyReLU(0.2)` |
| Tanh output | `nn.Tanh()` for [-1, 1] range |
| Weight init | Normal(0, 0.02) |

This implementation provides a solid foundation for generating 64×64 images and can be adapted for different resolutions by adjusting the number of convolutional blocks.
