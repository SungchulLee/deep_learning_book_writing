# β-VAE and Disentanglement

Learning interpretable, disentangled representations.

---

## Learning Objectives

By the end of this section, you will be able to:

- Explain the β hyperparameter and its effects
- Understand disentangled representations
- Measure disentanglement quality
- Visualize latent traversals

---

## From VAE to β-VAE

### The Standard VAE Objective

$$\mathcal{L}_{\text{VAE}} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{KL}(q(z|x) \| p(z))$$

Reconstruction and KL terms are weighted equally.

### The β-VAE Modification

β-VAE introduces a hyperparameter β to weight the KL term:

$$\mathcal{L}_{\beta\text{-VAE}} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - \beta \cdot D_{KL}(q(z|x) \| p(z))$$

| β Value | Effect |
|---------|--------|
| β = 1 | Standard VAE |
| β > 1 | Stronger regularization, promotes disentanglement |
| β < 1 | Weaker regularization, better reconstruction |

---

## What is Disentanglement?

### Intuition

A **disentangled representation** has latent dimensions that correspond to independent, interpretable factors of variation in the data.

For handwritten digits, disentangled factors might include:
- Stroke thickness
- Slant angle
- Size
- Digit identity

### Formal Definition

A representation $z = (z_1, ..., z_k)$ is disentangled if:

1. **Independence:** $p(z_i, z_j) = p(z_i) p(z_j)$ for $i \neq j$
2. **Interpretability:** Each $z_i$ controls a single factor of variation
3. **Controllability:** Changing $z_i$ changes only factor $i$ in $x$

### Why β > 1 Promotes Disentanglement

Higher β pushes q(z|x) closer to the factorial prior p(z) = N(0, I):

- **Factorial prior:** Each $z_i$ is independent
- **Strong KL penalty:** Encoder must use latent dimensions efficiently
- **Information bottleneck:** Each $z_i$ captures distinct information

---

## β-VAE Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


class BetaVAE(nn.Module):
    """
    β-VAE for learning disentangled representations.
    
    Args:
        input_dim: Input dimension
        hidden_dim: Hidden layer dimension
        latent_dim: Latent space dimension
        beta: Weight for KL divergence (β > 1 for disentanglement)
    """
    
    def __init__(self, input_dim=784, hidden_dim=256, latent_dim=10, beta=4.0):
        super(BetaVAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def loss_function(self, recon_x, x, mu, logvar):
        """
        β-VAE loss = Reconstruction + β * KL
        """
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.beta * kl_loss, recon_loss, kl_loss


def compare_beta_values(train_loader, test_loader, device, betas=[1, 4, 10, 20]):
    """
    Train β-VAEs with different β values and compare.
    """
    results = {}
    
    for beta in betas:
        print(f"\nTraining β-VAE with β = {beta}")
        
        model = BetaVAE(latent_dim=10, beta=beta).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        # Train for a few epochs
        for epoch in range(10):
            model.train()
            total_loss = 0
            total_recon = 0
            total_kl = 0
            
            for data, _ in train_loader:
                data = data.view(data.size(0), -1).to(device)
                
                optimizer.zero_grad()
                recon, mu, logvar = model(data)
                loss, recon_loss, kl_loss = model.loss_function(recon, data, mu, logvar)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_recon += recon_loss.item()
                total_kl += kl_loss.item()
            
            n = len(train_loader.dataset)
            print(f"  Epoch {epoch+1}: Loss={total_loss/n:.4f}, "
                  f"Recon={total_recon/n:.4f}, KL={total_kl/n:.4f}")
        
        results[beta] = {
            'model': model,
            'final_recon': total_recon / len(train_loader.dataset),
            'final_kl': total_kl / len(train_loader.dataset)
        }
    
    return results
```

---

## Latent Traversals

### What Are Latent Traversals?

A **latent traversal** visualizes what each latent dimension encodes by:

1. Fixing all latent dimensions except one
2. Varying that dimension across a range (e.g., -3 to +3)
3. Decoding each point to see how the output changes

### Implementation

```python
def latent_traversal(model, device, dim_idx, num_steps=10, 
                     range_limit=3.0, base_z=None):
    """
    Traverse a single latent dimension.
    
    Args:
        model: Trained β-VAE
        device: Device for computation
        dim_idx: Index of latent dimension to traverse
        num_steps: Number of steps in traversal
        range_limit: Range [-range_limit, +range_limit]
        base_z: Starting latent vector (zeros if None)
        
    Returns:
        Tensor of decoded images along traversal
    """
    model.eval()
    
    if base_z is None:
        base_z = torch.zeros(1, model.latent_dim).to(device)
    
    # Values to traverse
    values = torch.linspace(-range_limit, range_limit, num_steps).to(device)
    
    # Generate traversal
    traversal_images = []
    
    with torch.no_grad():
        for val in values:
            z = base_z.clone()
            z[0, dim_idx] = val
            recon = model.decode(z)
            traversal_images.append(recon.cpu())
    
    return torch.stack(traversal_images)


def visualize_all_traversals(model, device, num_dims=None, num_steps=10,
                              range_limit=3.0, figsize=(15, 10)):
    """
    Visualize traversals for all latent dimensions.
    """
    model.eval()
    
    if num_dims is None:
        num_dims = model.latent_dim
    
    fig, axes = plt.subplots(num_dims, num_steps, figsize=figsize)
    
    for dim in range(num_dims):
        traversal = latent_traversal(model, device, dim, num_steps, range_limit)
        
        for step in range(num_steps):
            ax = axes[dim, step] if num_dims > 1 else axes[step]
            img = traversal[step].view(28, 28).numpy()
            ax.imshow(img, cmap='gray')
            ax.axis('off')
            
            # Label first column with dimension index
            if step == 0:
                ax.set_ylabel(f'z_{dim}', fontsize=10)
    
    # Add column labels for traversal values
    values = np.linspace(-range_limit, range_limit, num_steps)
    for step in range(num_steps):
        ax = axes[0, step] if num_dims > 1 else axes[step]
        ax.set_title(f'{values[step]:.1f}', fontsize=8)
    
    plt.suptitle('Latent Dimension Traversals', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('latent_traversals.png', dpi=150, bbox_inches='tight')
    plt.show()


def visualize_traversal_for_sample(model, test_loader, device, sample_idx=0):
    """
    Visualize traversals starting from a specific input sample.
    """
    model.eval()
    
    # Get sample
    data, labels = next(iter(test_loader))
    sample = data[sample_idx:sample_idx+1].view(1, -1).to(device)
    
    # Encode
    with torch.no_grad():
        mu, logvar = model.encode(sample)
    
    # Visualize original
    fig, axes = plt.subplots(model.latent_dim + 1, 11, figsize=(15, model.latent_dim + 2))
    
    # First row: original and reconstruction
    original = sample.cpu().view(28, 28).numpy()
    axes[0, 5].imshow(original, cmap='gray')
    axes[0, 5].set_title('Original')
    for j in range(11):
        if j != 5:
            axes[0, j].axis('off')
    axes[0, 5].axis('off')
    
    # Remaining rows: traversals
    for dim in range(model.latent_dim):
        traversal = latent_traversal(model, device, dim, num_steps=11, 
                                     range_limit=3.0, base_z=mu)
        
        for step in range(11):
            img = traversal[step].view(28, 28).numpy()
            axes[dim + 1, step].imshow(img, cmap='gray')
            axes[dim + 1, step].axis('off')
        
        axes[dim + 1, 0].set_ylabel(f'z_{dim}', fontsize=10)
    
    plt.suptitle(f'Latent Traversals for Digit {labels[sample_idx].item()}', y=1.02)
    plt.tight_layout()
    plt.savefig('sample_traversals.png', dpi=150, bbox_inches='tight')
    plt.show()
```

---

## Measuring Disentanglement

### Qualitative Assessment

Look at latent traversals:
- **Disentangled:** Each row changes only one visual attribute
- **Entangled:** Multiple attributes change simultaneously

### Quantitative Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| **Factor VAE score** | Accuracy of predicting factor from z | [0, 1] |
| **DCI Disentanglement** | Mutual information based | [0, 1] |
| **MIG** | Mutual Information Gap | [0, 1] |

### Simple Disentanglement Analysis

```python
def analyze_latent_variance(model, test_loader, device):
    """
    Analyze variance of each latent dimension.
    
    Active dimensions have higher variance.
    """
    model.eval()
    
    all_mu = []
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.view(data.size(0), -1).to(device)
            mu, _ = model.encode(data)
            all_mu.append(mu.cpu())
    
    all_mu = torch.cat(all_mu, dim=0)
    
    # Compute variance per dimension
    variances = torch.var(all_mu, dim=0).numpy()
    
    # Plot
    plt.figure(figsize=(12, 4))
    plt.bar(range(len(variances)), variances)
    plt.xlabel('Latent Dimension')
    plt.ylabel('Variance')
    plt.title('Latent Dimension Activity (Variance)')
    plt.axhline(y=0.1, color='r', linestyle='--', label='Activity threshold')
    plt.legend()
    plt.savefig('latent_variance.png', dpi=150)
    plt.show()
    
    # Print statistics
    active_dims = (variances > 0.1).sum()
    print(f"Active dimensions (var > 0.1): {active_dims}/{len(variances)}")
    
    return variances


def analyze_kl_per_dimension(model, test_loader, device):
    """
    Analyze KL divergence contribution from each dimension.
    
    High KL dimensions are being used for encoding.
    """
    model.eval()
    
    all_kl = []
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.view(data.size(0), -1).to(device)
            mu, logvar = model.encode(data)
            
            # KL per dimension
            kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
            all_kl.append(kl_per_dim.cpu())
    
    all_kl = torch.cat(all_kl, dim=0)
    mean_kl = all_kl.mean(dim=0).numpy()
    
    # Plot
    plt.figure(figsize=(12, 4))
    plt.bar(range(len(mean_kl)), mean_kl)
    plt.xlabel('Latent Dimension')
    plt.ylabel('Mean KL Divergence')
    plt.title('KL Divergence per Latent Dimension')
    plt.savefig('kl_per_dimension.png', dpi=150)
    plt.show()
    
    return mean_kl
```

---

## The β Trade-off

### Reconstruction vs. Disentanglement

| β Value | Reconstruction | Disentanglement | Use Case |
|---------|----------------|-----------------|----------|
| β = 1 | Good | Limited | Standard generation |
| β = 4 | Moderate | Good | Balanced |
| β = 10 | Poor | Better | Interpretability focus |
| β = 20+ | Bad | Best | Pure disentanglement |

### Visualizing the Trade-off

```python
def plot_beta_tradeoff(results):
    """
    Plot reconstruction vs KL for different β values.
    """
    betas = list(results.keys())
    recons = [results[b]['final_recon'] for b in betas]
    kls = [results[b]['final_kl'] for b in betas]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(betas, recons, 'bo-', markersize=10)
    axes[0].set_xlabel('β')
    axes[0].set_ylabel('Reconstruction Loss')
    axes[0].set_title('Reconstruction Quality vs β')
    axes[0].set_xscale('log')
    
    axes[1].plot(betas, kls, 'ro-', markersize=10)
    axes[1].set_xlabel('β')
    axes[1].set_ylabel('KL Divergence')
    axes[1].set_title('KL Divergence vs β')
    axes[1].set_xscale('log')
    
    plt.tight_layout()
    plt.savefig('beta_tradeoff.png', dpi=150)
    plt.show()
```

---

## Exercises

### Exercise 1: β Exploration

Train β-VAE with β ∈ {1, 2, 4, 8, 16, 32}:
a) Plot reconstruction loss vs β
b) Plot number of active latent dimensions vs β
c) Visually compare latent traversals

### Exercise 2: Disentanglement Analysis

For a trained β-VAE:
a) Identify which latent dimensions encode stroke thickness
b) Find dimensions that encode digit identity vs style
c) Attempt to change only the slant of a digit

### Exercise 3: β-VAE for Other Data

Try β-VAE on Fashion-MNIST:
a) What factors of variation exist? (sleeve length, collar type, etc.)
b) What β value gives good disentanglement?
c) Compare traversals with MNIST results

---

## Summary

| Concept | Key Point |
|---------|-----------|
| **β-VAE** | VAE with weighted KL term: β > 1 |
| **Disentanglement** | Each latent dim controls one factor |
| **Latent traversals** | Vary one dim to see its effect |
| **Trade-off** | Higher β → better disentanglement, worse reconstruction |

---

## What's Next

The next section covers [VQ-VAE](vqvae.md), which uses discrete latent representations through vector quantization.
