# Introduction to Generative Adversarial Networks

Generative Adversarial Networks (GANs), introduced by Ian Goodfellow et al. in 2014, represent one of the most influential advances in deep learning. GANs leverage a game-theoretic framework where two neural networks compete: a **generator** that creates synthetic samples and a **discriminator** that distinguishes real from fake. This adversarial process drives both networks toward equilibrium, producing remarkably realistic outputs.

Unlike likelihood-based approaches (VAEs, Flows, Diffusion models), GANs learn through **adversarial competition** between two neural networks without ever explicitly computing probability densities.

## The Core Intuition

Consider an analogy from art forgery:

**The Forger (Generator)**: Learns to create fake paintings that look authentic
**The Expert (Discriminator)**: Learns to distinguish genuine art from forgeries

As the expert improves at detecting fakes, the forger must create more convincing forgeries. This adversarial dynamic drives both parties to continuously improve. Eventually, the forger produces paintings indistinguishable from authentic masterpieces.

## The Two Players

## The Generator

### Role and Purpose

The generator $G$ is a neural network that transforms random noise into synthetic data samples:

$$G: \mathcal{Z} \rightarrow \mathcal{X}$$

where:
- $\mathcal{Z}$ is the latent space (typically $\mathbb{R}^d$ with $d$ = 100)
- $\mathcal{X}$ is the data space (e.g., images, audio)
- $z \sim p_z(z)$ is the input noise vector
- $G(z) = \hat{x}$ is the generated sample

### Input: The Latent Vector

The latent vector $z$ serves as the "seed" for generation:

```python
import torch

def sample_latent(batch_size, latent_dim, device='cpu'):
    """Sample latent vectors from the prior distribution."""
    # Standard Gaussian prior (most common)
    z = torch.randn(batch_size, latent_dim, device=device)
    return z
```

**Common prior distributions:**

| Distribution | Formula | Use Case |
|--------------|---------|----------|
| Standard Gaussian | $z \sim \mathcal{N}(0, I)$ | Most common, works well |
| Uniform | $z \sim U(-1, 1)^d$ | Sometimes used |
| Truncated Gaussian | $z \sim \mathcal{N}(0, I), \|z\| \leq c$ | Better sample quality |

### Generator Architecture (MLP)

For simple data (2D points, small vectors):

```python
import torch.nn as nn

class MLPGenerator(nn.Module):
    """Simple MLP generator for low-dimensional data."""
    
    def __init__(self, latent_dim=100, hidden_dim=256, output_dim=2):
        super().__init__()
        
        self.model = nn.Sequential(
            # Latent to hidden
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(True),
            
            # Hidden layers
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            
            # Output layer
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, z):
        """Generate data from latent vector."""
        return self.model(z)
```

### Generator Architecture (Convolutional)

For images, use transposed convolutions to upsample:

```python
class ConvGenerator(nn.Module):
    """
    Convolutional generator for image generation.
    
    Upsamples latent vector to image using transposed convolutions.
    """
    
    def __init__(self, latent_dim=100, feature_maps=64, image_channels=1):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Project and reshape: z -> (feature_maps*8, 4, 4)
        self.project = nn.Sequential(
            nn.Linear(latent_dim, feature_maps * 8 * 4 * 4),
            nn.BatchNorm1d(feature_maps * 8 * 4 * 4),
            nn.ReLU(True)
        )
        
        # Upsample: 4x4 -> 8x8 -> 16x16 -> 32x32
        self.conv_layers = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4,
                              kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            
            # 8x8 -> 16x16
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2,
                              kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(feature_maps * 2, feature_maps,
                              kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            
            # 32x32 -> 32x32 (final)
            nn.Conv2d(feature_maps, image_channels,
                     kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()  # Output in [-1, 1]
        )
    
    def forward(self, z):
        """Generate image from latent vector."""
        # Project and reshape
        x = self.project(z)
        x = x.view(x.size(0), -1, 4, 4)
        
        # Upsample to image
        return self.conv_layers(x)
```

### Key Generator Design Principles

1. **Use ReLU activations** in hidden layers (allows sparse representations)
2. **Use Tanh activation** for output (matches [-1, 1] normalized images)
3. **Use Batch Normalization** in all layers except output
4. **No pooling layers**—use transposed convolutions for upsampling

## The Discriminator

### Role and Purpose

The discriminator $D$ is a binary classifier that distinguishes real from generated data:

$$D: \mathcal{X} \rightarrow [0, 1]$$

where:
- $D(x) \approx 1$ means "x is probably real"
- $D(x) \approx 0$ means "x is probably fake"

### Discriminator Architecture (MLP)

For simple data:

```python
class MLPDiscriminator(nn.Module):
    """Simple MLP discriminator for low-dimensional data."""
    
    def __init__(self, input_dim=2, hidden_dim=256):
        super().__init__()
        
        self.model = nn.Sequential(
            # Input to hidden
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Hidden layers
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Output: probability of being real
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """Classify input as real or fake."""
        return self.model(x)
```

### Discriminator Architecture (Convolutional)

For images, use strided convolutions to downsample:

```python
class ConvDiscriminator(nn.Module):
    """
    Convolutional discriminator for image classification.
    
    Downsamples image to scalar probability using strided convolutions.
    """
    
    def __init__(self, image_channels=1, feature_maps=64):
        super().__init__()
        
        self.model = nn.Sequential(
            # 32x32 -> 16x16 (no batch norm on first layer)
            nn.Conv2d(image_channels, feature_maps,
                     kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16x16 -> 8x8
            nn.Conv2d(feature_maps, feature_maps * 2,
                     kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 8x8 -> 4x4
            nn.Conv2d(feature_maps * 2, feature_maps * 4,
                     kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 4x4 -> 1x1
            nn.Conv2d(feature_maps * 4, 1,
                     kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """Classify image as real or fake."""
        return self.model(x).view(-1, 1)
```

### Key Discriminator Design Principles

1. **Use LeakyReLU activations** (slope 0.2) to avoid sparse gradients
2. **Use Sigmoid activation** for output (probability interpretation)
3. **No Batch Normalization** on input layer
4. **No pooling layers**—use strided convolutions for downsampling



## Architectural Symmetry

## Architectural Symmetry

The generator and discriminator often have symmetric architectures:

```
Generator (Upsampling):          Discriminator (Downsampling):
z (100,)                         image (C, H, W)
    ↓ Project                        ↓ Conv stride=2
(512, 4, 4)                      (64, H/2, W/2)
    ↓ ConvT stride=2                 ↓ Conv stride=2
(256, 8, 8)                      (128, H/4, W/4)
    ↓ ConvT stride=2                 ↓ Conv stride=2
(128, 16, 16)                    (256, H/8, W/8)
    ↓ ConvT stride=2                 ↓ Conv stride=2
(64, 32, 32)                     (512, H/16, W/16)
    ↓ Conv                           ↓ Conv
image (C, 32, 32)                probability (1,)
```



## Weight Initialization

## Weight Initialization

Proper initialization is crucial for GAN training:

```python
def weights_init(m):
    """
    Initialize network weights following DCGAN recommendations.
    
    - Conv/ConvTranspose: Normal(0, 0.02)
    - BatchNorm: weight=1, bias=0
    """
    classname = m.__class__.__name__
    
    if classname.find('Conv') != -1:
        # Conv and ConvTranspose layers
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        
    elif classname.find('BatchNorm') != -1:
        # BatchNorm layers
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Apply initialization
generator.apply(weights_init)
discriminator.apply(weights_init)
```



## Feature Matching

## Feature Matching

An intermediate technique where the generator is trained to match discriminator feature statistics:

```python
class DiscriminatorWithFeatures(nn.Module):
    """Discriminator that also outputs intermediate features."""
    
    def __init__(self, image_channels=1, feature_maps=64):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(image_channels, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.classifier = nn.Sequential(
            nn.Conv2d(feature_maps * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x, return_features=False):
        """Forward pass with optional feature output."""
        features = self.features(x)
        output = self.classifier(features).view(-1, 1)
        
        if return_features:
            return output, features.view(features.size(0), -1)
        return output

def feature_matching_loss(real_features, fake_features):
    """Match mean feature statistics."""
    return torch.mean((real_features.mean(0) - fake_features.mean(0)) ** 2)
```



## Interaction During Training

## Interaction During Training

The generator and discriminator have a delicate relationship:

### Gradient Flow

```
Training Discriminator:            Training Generator:
real_data ──→ D ──→ d_real         z ──→ G ──→ fake ──→ D ──→ d_fake
              ↓                              ↑           ↓
fake_data ──→ D ──→ d_fake                   └─ gradient flows back
(detached)    ↓
              loss
              ↓
         update D only                  update G only
```

### Balance Considerations

| Scenario | Symptom | Solution |
|----------|---------|----------|
| D too strong | G loss stays high, poor samples | Reduce D learning rate, fewer D steps |
| D too weak | D loss near random (0.693) | More D steps, increase D capacity |
| G too strong | D always fooled | Increase D capacity |
| Both weak | No learning | Check architecture, learning rates |



## Monitoring G and D

## Monitoring G and D

```python
def monitor_gan_training(d_real, d_fake, g_loss, d_loss):
    """
    Monitor GAN training health.
    
    Healthy signs:
    - D(real) ~ 0.7-0.9 (not 1.0)
    - D(fake) ~ 0.1-0.3 (not 0.0)
    - Both losses oscillate but trend down
    """
    d_real_mean = d_real.mean().item()
    d_fake_mean = d_fake.mean().item()
    
    # Warning signs
    if d_real_mean > 0.99:
        print("Warning: D is too confident on real data")
    if d_fake_mean < 0.01:
        print("Warning: D is too confident on fake data")
    if d_real_mean < 0.5:
        print("Warning: D is failing on real data")
    if d_fake_mean > 0.9:
        print("Warning: G may be collapsing or D failing")
    
    return {
        'D(real)': d_real_mean,
        'D(fake)': d_fake_mean,
        'D_loss': d_loss.item(),
        'G_loss': g_loss.item()
    }
```



## Complete Example: Simple GAN for MNIST

The following implementation demonstrates a complete, minimal GAN trained on MNIST using fully-connected layers. This serves as a pedagogical starting point before moving to convolutional architectures.

### Configuration

```python
import argparse
import os
import torch

# Key hyperparameters for GAN training
config = {
    'latent_dim': 100,     # Dimension of latent space z
    'img_size': 28,        # Height/width of generated images
    'channels': 1,         # Number of image channels (grayscale)
    'batch_size': 64,      # Training batch size
    'epochs': 200,         # Number of training epochs
    'lr': 2e-4,            # Learning rate for Adam optimizer
    'b1': 0.5,             # Adam beta1 (lower than default for GAN stability)
    'b2': 0.999,           # Adam beta2
}
```

### MLP Generator and Discriminator

```python
import numpy as np
import torch.nn as nn

class Generator(nn.Module):
    """
    MLP Generator: z (100) → 128 → 256 → 512 → 1024 → 784 (28×28).
    
    Uses BatchNorm + LeakyReLU in hidden layers, Tanh output for [-1, 1] range.
    """
    
    def __init__(self, latent_dim=100, img_shape=(1, 28, 28)):
        super().__init__()
        self.img_shape = img_shape
        
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, momentum=0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )
    
    def forward(self, z):
        img = self.model(z)
        return img.view(img.size(0), *self.img_shape)


class Discriminator(nn.Module):
    """
    MLP Discriminator: 784 → 512 → 256 → 1.
    
    Uses LeakyReLU throughout, Sigmoid output for probability.
    """
    
    def __init__(self, img_shape=(1, 28, 28)):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        return self.model(img.view(img.size(0), -1))
```

### Training Loop

```python
import torch.optim as optim
from torchvision.utils import save_image

def train_simple_gan(generator, discriminator, dataloader, config, device):
    criterion = nn.BCELoss()
    
    optimizer_G = optim.Adam(generator.parameters(), lr=config['lr'], betas=(config['b1'], config['b2']))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=config['lr'], betas=(config['b1'], config['b2']))
    
    for epoch in range(config['epochs']):
        for i, (imgs, _) in enumerate(dataloader):
            batch_size = imgs.size(0)
            real_imgs = imgs.to(device)
            
            valid = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)
            
            # --- Train Generator ---
            optimizer_G.zero_grad()
            z = torch.randn(batch_size, config['latent_dim'], device=device)
            gen_imgs = generator(z)
            g_loss = criterion(discriminator(gen_imgs), valid)
            g_loss.backward()
            optimizer_G.step()
            
            # --- Train Discriminator ---
            optimizer_D.zero_grad()
            real_loss = criterion(discriminator(real_imgs), valid)
            fake_loss = criterion(discriminator(gen_imgs.detach()), fake_labels)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()
```

## Comparison with Other Generative Models

| Aspect | GANs | VAEs | Flows | Diffusion |
|--------|------|------|-------|-----------|
| **Training** | Adversarial game | ELBO maximization | MLE | Denoising |
| **Likelihood** | Implicit | Lower bound | Exact | Lower bound |
| **Sample Quality** | Sharp | Blurry | Good | Excellent |
| **Training Stability** | Challenging | Stable | Stable | Stable |
| **Sampling Speed** | Fast | Fast | Fast | Slow |

## Summary

| Component | Generator | Discriminator |
|-----------|-----------|---------------|
| **Input** | Noise $z$ | Data $x$ |
| **Output** | Fake data $G(z)$ | Probability $D(x)$ |
| **Architecture** | Upsampling (ConvT) | Downsampling (Conv) |
| **Hidden Activation** | ReLU | LeakyReLU |
| **Output Activation** | Tanh | Sigmoid |
| **BatchNorm** | All except output | All except input |
| **Goal** | Fool D | Classify correctly |

Understanding the generator and discriminator architectures, their roles in the adversarial game, and how they interact during training is fundamental to successfully implementing and debugging GANs.
