# Conditional VAE (CVAE)

Controlled generation through label conditioning.

---

## Learning Objectives

By the end of this section, you will be able to:

- Explain the motivation for conditional generation
- Derive the Conditional VAE objective
- Implement CVAE in PyTorch
- Generate samples conditioned on specific attributes

---

## Motivation: Why Conditional Generation?

### Limitations of Standard VAE

In a standard VAE, generation is **uncontrolled**:

```python
# Standard VAE generation
z = torch.randn(batch_size, latent_dim)  # Random sample
x_generated = decoder(z)  # What digit? Unknown!
```

We sample from the prior and hope for the best. There's no way to specify *what* we want to generate.

### The Conditional Generation Goal

With CVAE, we can **control** generation:

```python
# CVAE generation - specify what we want
y = 7  # I want a "7"
z = torch.randn(batch_size, latent_dim)
x_generated = decoder(z, y)  # Generate digit "7"
```

---

## Conditional VAE Formulation

### Generative Model

In CVAE, both the prior and likelihood are conditioned on label $y$:

$$p_\theta(x, z | y) = p_\theta(x | z, y) \cdot p(z | y)$$

**Common simplification:** Assume $p(z|y) = p(z) = \mathcal{N}(0, I)$ (prior independent of label).

### Inference Model

The encoder also conditions on $y$:

$$q_\phi(z | x, y) = \mathcal{N}(\mu_\phi(x, y), \sigma^2_\phi(x, y))$$

### CVAE ELBO

The Conditional ELBO is:

$$\mathcal{L}(\theta, \phi; x, y) = \mathbb{E}_{q_\phi(z|x,y)}[\log p_\theta(x | z, y)] - D_{KL}(q_\phi(z | x, y) \| p(z))$$

**Comparison with standard VAE:**

| Component | VAE | CVAE |
|-----------|-----|------|
| **Encoder** | $q_\phi(z\|x)$ | $q_\phi(z\|x, y)$ |
| **Decoder** | $p_\theta(x\|z)$ | $p_\theta(x\|z, y)$ |
| **Prior** | $p(z)$ | $p(z)$ (typically unchanged) |

---

## Architecture

### Standard VAE vs. CVAE

```
Standard VAE:                       Conditional VAE:
                                    
Input x ──► Encoder ──► (μ, σ²)     Input x ─┬─► Encoder ──► (μ, σ²)
                 │                  Label y ─┘       │
                 ▼                                   ▼
           Reparameterize                     Reparameterize
                 │                                   │
                 ▼                                   ▼
           z ~ N(μ, σ²)                       z ~ N(μ, σ²)
                 │                                   │
                 ▼                            ┌──────┴──────┐
            Decoder                           ▼             ▼
                 │                          z            Label y
                 ▼                            └──────┬──────┘
            Output x̂                                 ▼
                                                 Decoder
                                                    │
                                                    ▼
                                                Output x̂
```

### How to Incorporate the Label

**One-hot encoding** is the standard approach:

```python
# For MNIST: 10 classes
y_onehot = F.one_hot(y, num_classes=10).float()  # [batch_size, 10]

# Concatenate with input for encoder
encoder_input = torch.cat([x, y_onehot], dim=1)  # [batch_size, 784 + 10]

# Concatenate with z for decoder
decoder_input = torch.cat([z, y_onehot], dim=1)  # [batch_size, latent_dim + 10]
```

---

## PyTorch Implementation

### Complete CVAE Model

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class CVAE(nn.Module):
    """
    Conditional Variational Autoencoder.
    
    Args:
        input_dim: Input dimension (784 for MNIST)
        hidden_dim: Hidden layer dimension
        latent_dim: Latent space dimension
        num_classes: Number of classes for conditioning
    """
    
    def __init__(self, input_dim=784, hidden_dim=256, 
                 latent_dim=20, num_classes=10):
        super(CVAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # ============== ENCODER ==============
        # Input: x concatenated with y_onehot
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # ============== DECODER ==============
        # Input: z concatenated with y_onehot
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x, y):
        """
        Encode input conditioned on label.
        
        Args:
            x: Input data [batch_size, input_dim]
            y: Labels [batch_size] (integer class indices)
            
        Returns:
            mu, logvar: Distribution parameters
        """
        # One-hot encode labels
        y_onehot = F.one_hot(y, self.num_classes).float()
        
        # Concatenate input with condition
        x_cond = torch.cat([x, y_onehot], dim=1)
        
        # Encode
        h = self.encoder(x_cond)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, y):
        """
        Decode latent vector conditioned on label.
        
        Args:
            z: Latent vector [batch_size, latent_dim]
            y: Labels [batch_size] (integer class indices)
            
        Returns:
            Reconstructed output
        """
        # One-hot encode labels
        y_onehot = F.one_hot(y, self.num_classes).float()
        
        # Concatenate z with condition
        z_cond = torch.cat([z, y_onehot], dim=1)
        
        # Decode
        return self.decoder(z_cond)
    
    def forward(self, x, y):
        """
        Full forward pass.
        
        Args:
            x: Input data [batch_size, input_dim]
            y: Labels [batch_size]
            
        Returns:
            recon_x, mu, logvar
        """
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, y)
        return recon_x, mu, logvar
    
    def generate(self, y, num_samples=1, device='cpu'):
        """
        Generate samples conditioned on label.
        
        Args:
            y: Label to condition on (integer or tensor)
            num_samples: Number of samples to generate
            device: Device for computation
            
        Returns:
            Generated samples [num_samples, input_dim]
        """
        if isinstance(y, int):
            y = torch.full((num_samples,), y, dtype=torch.long, device=device)
        
        # Sample from prior
        z = torch.randn(num_samples, self.latent_dim, device=device)
        
        # Decode with condition
        with torch.no_grad():
            samples = self.decode(z, y)
        
        return samples
```

### Loss Function

```python
def cvae_loss(recon_x, x, mu, logvar, reduction='sum'):
    """
    CVAE loss = Reconstruction + KL Divergence
    
    (Same as standard VAE loss, but computed on conditioned outputs)
    
    Args:
        recon_x: Reconstructed output
        x: Original input
        mu: Encoder mean
        logvar: Encoder log-variance
        reduction: 'sum' or 'mean'
        
    Returns:
        total_loss, recon_loss, kl_loss
    """
    # Reconstruction loss (BCE for binary data)
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction=reduction)
    
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    if reduction == 'mean':
        kl_loss = kl_loss / x.size(0)
    
    total_loss = recon_loss + kl_loss
    
    return total_loss, recon_loss, kl_loss
```

---

## Training CVAE

### Training Loop

```python
def train_cvae_epoch(model, dataloader, optimizer, device):
    """Train CVAE for one epoch."""
    model.train()
    
    total_loss = 0
    total_recon = 0
    total_kl = 0
    
    for batch_idx, (data, labels) in enumerate(dataloader):
        # Flatten images and move to device
        data = data.view(data.size(0), -1).to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass (includes label conditioning)
        recon_batch, mu, logvar = model(data, labels)
        
        # Compute loss
        loss, recon, kl = cvae_loss(recon_batch, data, mu, logvar)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_recon += recon.item()
        total_kl += kl.item()
    
    n = len(dataloader.dataset)
    return {
        'loss': total_loss / n,
        'recon_loss': total_recon / n,
        'kl_loss': total_kl / n
    }
```

### Complete Training Script

```python
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def train_cvae(epochs=20, batch_size=64, learning_rate=1e-3):
    """Complete CVAE training pipeline."""
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST('./data', train=True, download=True, 
                                   transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Model
    model = CVAE(input_dim=784, hidden_dim=256, 
                 latent_dim=20, num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training
    for epoch in range(1, epochs + 1):
        stats = train_cvae_epoch(model, train_loader, optimizer, device)
        
        print(f"Epoch {epoch:2d} | Loss: {stats['loss']:.4f} | "
              f"Recon: {stats['recon_loss']:.4f} | KL: {stats['kl_loss']:.4f}")
    
    return model


# Train
model = train_cvae()
```

---

## Conditional Generation

### Generate Specific Digits

```python
def visualize_conditional_generation(model, device, num_samples_per_class=10):
    """Generate samples for each class."""
    import matplotlib.pyplot as plt
    
    model.eval()
    
    fig, axes = plt.subplots(10, num_samples_per_class, figsize=(15, 15))
    
    for digit in range(10):
        # Generate samples for this digit
        samples = model.generate(y=digit, num_samples=num_samples_per_class, 
                                device=device)
        
        for i in range(num_samples_per_class):
            img = samples[i].cpu().view(28, 28).numpy()
            axes[digit, i].imshow(img, cmap='gray')
            axes[digit, i].axis('off')
        
        # Label first column
        axes[digit, 0].set_ylabel(f'{digit}', fontsize=14, rotation=0, 
                                   labelpad=20, va='center')
    
    plt.suptitle('Conditional Generation: Samples per Digit', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('cvae_conditional_generation.png', dpi=150, bbox_inches='tight')
    plt.show()


# Visualize
visualize_conditional_generation(model, device)
```

### Conditional Interpolation

```python
def conditional_interpolation(model, device, digit, num_steps=10):
    """Interpolate in latent space while keeping label fixed."""
    model.eval()
    
    # Sample two random latent vectors
    z1 = torch.randn(1, model.latent_dim, device=device)
    z2 = torch.randn(1, model.latent_dim, device=device)
    
    # Interpolate
    interpolations = []
    for t in torch.linspace(0, 1, num_steps):
        z_interp = (1 - t) * z1 + t * z2
        y = torch.tensor([digit], device=device)
        
        with torch.no_grad():
            sample = model.decode(z_interp, y)
        interpolations.append(sample.cpu())
    
    # Plot
    fig, axes = plt.subplots(1, num_steps, figsize=(15, 2))
    for i, sample in enumerate(interpolations):
        axes[i].imshow(sample.view(28, 28).numpy(), cmap='gray')
        axes[i].axis('off')
    
    plt.suptitle(f'Latent Interpolation for Digit {digit}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'cvae_interpolation_digit_{digit}.png', dpi=150)
    plt.show()


# Interpolate for different digits
for digit in [0, 3, 7]:
    conditional_interpolation(model, device, digit)
```

---

## CVAE vs. Standard VAE

### Comparison

| Aspect | VAE | CVAE |
|--------|-----|------|
| **Generation** | Uncontrolled | Controlled by label |
| **Latent space** | Encodes everything | Encodes variation *within* class |
| **Use case** | General generation | Targeted generation |
| **Architecture** | Simpler | Label concatenation |

### When to Use CVAE

✓ **Controlled generation:** Generate specific types of outputs  
✓ **Semi-supervised learning:** Leverage label information  
✓ **Data augmentation:** Generate more examples of rare classes  
✓ **Attribute manipulation:** Change specific attributes while preserving others

---

## Advanced: Continuous Conditioning

### Beyond Discrete Labels

CVAE can condition on continuous attributes too:

```python
class ContinuousCVAE(nn.Module):
    """CVAE with continuous conditioning variables."""
    
    def __init__(self, input_dim, latent_dim, condition_dim):
        super().__init__()
        
        # Condition embedding (optional: can use directly)
        self.condition_embed = nn.Linear(condition_dim, 64)
        
        # Encoder: input + embedded condition
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + 64, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        # ... rest similar to discrete CVAE
    
    def encode(self, x, c):
        """Encode with continuous condition c."""
        c_embed = F.relu(self.condition_embed(c))
        x_cond = torch.cat([x, c_embed], dim=1)
        h = self.encoder(x_cond)
        # ...
```

**Applications:**
- Condition on numerical attributes (age, price, etc.)
- Time-series conditioning
- Multi-attribute conditioning

---

## Summary

| Concept | Key Point |
|---------|-----------|
| **CVAE motivation** | Control what we generate |
| **Architecture** | Concatenate label with encoder input and decoder input |
| **Loss** | Same as VAE (conditioned reconstruction + KL) |
| **Generation** | Sample z, specify y, decode |
| **Advantage** | Targeted generation for specific classes/attributes |

---

## Exercises

### Exercise 1: Implement and Train

Complete the CVAE implementation and train on MNIST:
a) Verify reconstruction quality per digit
b) Generate 100 samples of each digit
c) Compute and compare reconstruction error across classes

### Exercise 2: Latent Space Analysis

For a trained CVAE:
a) Visualize the latent space colored by digit class
b) Compare to standard VAE latent space
c) What structure do you observe?

### Exercise 3: Style Transfer

Using CVAE:
a) Encode a "3" to get its latent code z
b) Decode z with label "8"
c) What happens? Explain in terms of what z represents

### Exercise 4: Fashion-MNIST CVAE

Train CVAE on Fashion-MNIST:
a) Generate samples for each clothing category
b) Perform interpolation within categories
c) Try "style transfer" between categories

---

## What's Next

The [PyTorch Implementation](pytorch_impl.md) section provides a complete, production-ready VAE training pipeline with all variants.
