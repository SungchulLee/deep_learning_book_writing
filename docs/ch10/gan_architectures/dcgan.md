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
