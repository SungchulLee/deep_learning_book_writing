# Contractive Autoencoders

Learn robust representations by penalizing the sensitivity of the encoder.

---

## Overview

**Key Concepts:**

- Jacobian of the encoder
- Contractive penalty
- Robust feature learning
- Connection to denoising autoencoders

**Time:** ~40 minutes  
**Level:** Intermediate-Advanced

---

## Mathematical Foundation

### The Contractive Penalty

A **contractive autoencoder (CAE)** adds a penalty on the Frobenius norm of the encoder's Jacobian:

$$\mathcal{L} = \|x - g(f(x))\|^2 + \lambda \|J_f(x)\|_F^2$$

where:
- $f$: encoder function
- $g$: decoder function
- $J_f(x) = \frac{\partial f(x)}{\partial x} \in \mathbb{R}^{k \times d}$: Jacobian matrix
- $\|J_f\|_F^2 = \sum_{ij} J_{ij}^2$: Frobenius norm squared

### Intuition

| Component | Effect |
|-----------|--------|
| Reconstruction loss | Learn to reconstruct inputs |
| Jacobian penalty | Make encoder insensitive to input perturbations |

The Jacobian penalty encourages:
- **Local invariance:** Small changes in input → small changes in latent
- **Robust representations:** Ignore noise, capture essential structure
- **Flat manifolds:** Latent space locally constant along noise directions

---

## Connection to Denoising Autoencoders

### Theoretical Link

For small noise, denoising autoencoders approximately minimize:

$$\mathcal{L}_{DAE} \approx \|x - g(f(x))\|^2 + \sigma^2 \|J_f(x)\|_F^2$$

**Key insight:** Denoising with Gaussian noise implicitly applies a contractive penalty!

### Comparison

| Aspect | Denoising AE | Contractive AE |
|--------|--------------|----------------|
| Regularization | Via corrupted inputs | Via explicit Jacobian penalty |
| Computation | Forward pass with noise | Requires Jacobian computation |
| Flexibility | Different noise types | Direct control over contraction |
| Interpretation | Learn to denoise | Minimize encoder sensitivity |

---

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import grad

class ContractiveAutoencoder(nn.Module):
    """
    Contractive Autoencoder with Jacobian penalty.
    """
    
    def __init__(self, input_dim=784, latent_dim=64):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder with sigmoid for bounded outputs
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Sigmoid(),
            nn.Linear(256, 128),
            nn.Sigmoid(),
            nn.Linear(128, latent_dim),
            nn.Sigmoid()  # Bounded [0,1] for stable Jacobian
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
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


def compute_jacobian_penalty(model, x):
    """
    Compute the Frobenius norm squared of the encoder Jacobian.
    
    J_f(x)_ij = ∂z_i / ∂x_j
    ||J_f||_F^2 = Σ_ij (∂z_i / ∂x_j)^2
    """
    x = x.requires_grad_(True)
    z = model.encode(x)
    
    # Compute Jacobian column by column
    jacobian_norm_sq = 0.0
    
    for i in range(z.shape[1]):
        # Gradient of z_i with respect to x
        grad_outputs = torch.zeros_like(z)
        grad_outputs[:, i] = 1.0
        
        jacobian_col = grad(
            outputs=z,
            inputs=x,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Sum of squares of this column
        jacobian_norm_sq = jacobian_norm_sq + torch.sum(jacobian_col ** 2)
    
    return jacobian_norm_sq / x.shape[0]  # Average over batch


def compute_jacobian_penalty_efficient(model, x):
    """
    Efficient Jacobian computation for sigmoid encoder.
    
    For sigmoid activation h = σ(Wx + b):
    ∂h/∂x = W * h * (1-h)
    
    ||J||_F^2 = Σ (W_ij * h_j * (1-h_j))^2
    """
    # This requires access to intermediate activations
    # For general networks, use the full computation above
    
    x = x.requires_grad_(True)
    z = model.encode(x)
    
    # For sigmoid: h' = h(1-h), so J = W * diag(h(1-h))
    # ||J||_F^2 = ||W||_F^2 * Σ (h(1-h))^2
    
    # For last layer only (approximation):
    h = z
    deriv = h * (1 - h)  # Sigmoid derivative
    
    # Get last encoder layer weights
    last_linear = None
    for layer in reversed(list(model.encoder.children())):
        if isinstance(layer, nn.Linear):
            last_linear = layer
            break
    
    if last_linear is not None:
        W = last_linear.weight  # (latent_dim, hidden_dim)
        # Jacobian penalty approximation
        penalty = torch.sum(deriv ** 2, dim=1).mean() * torch.sum(W ** 2)
        return penalty
    
    # Fallback to full computation
    return compute_jacobian_penalty(model, x)


def train_contractive_autoencoder(
    model, train_loader, device, 
    lambda_contractive=0.1, num_epochs=15
):
    """
    Train contractive autoencoder.
    
    Loss = Reconstruction + λ * ||J_f||_F^2
    """
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    recon_criterion = nn.MSELoss()
    
    history = {'recon_loss': [], 'contractive_loss': [], 'total_loss': []}
    
    for epoch in range(num_epochs):
        model.train()
        
        epoch_recon = 0
        epoch_contractive = 0
        epoch_total = 0
        
        for images, _ in train_loader:
            images = images.view(images.size(0), -1).to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            recon, z = model(images)
            
            # Reconstruction loss
            recon_loss = recon_criterion(recon, images)
            
            # Contractive penalty
            contractive_loss = compute_jacobian_penalty(model, images)
            
            # Total loss
            total_loss = recon_loss + lambda_contractive * contractive_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            epoch_recon += recon_loss.item()
            epoch_contractive += contractive_loss.item()
            epoch_total += total_loss.item()
        
        n_batches = len(train_loader)
        history['recon_loss'].append(epoch_recon / n_batches)
        history['contractive_loss'].append(epoch_contractive / n_batches)
        history['total_loss'].append(epoch_total / n_batches)
        
        print(f"Epoch {epoch+1}: Recon={epoch_recon/n_batches:.6f}, "
              f"Contract={epoch_contractive/n_batches:.6f}")
    
    return history


def analyze_contraction(model, test_loader, device, noise_std=0.1):
    """
    Analyze how contractive the learned representation is.
    """
    model.eval()
    
    sensitivity_scores = []
    
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.view(images.size(0), -1).to(device)
            
            # Original encoding
            z_original = model.encode(images)
            
            # Encoding of noisy input
            noise = torch.randn_like(images) * noise_std
            z_noisy = model.encode(images + noise)
            
            # Measure change in latent space
            input_change = torch.norm(noise, dim=1)
            latent_change = torch.norm(z_noisy - z_original, dim=1)
            
            # Sensitivity = ||Δz|| / ||Δx||
            sensitivity = latent_change / (input_change + 1e-8)
            sensitivity_scores.extend(sensitivity.cpu().numpy())
            
            if len(sensitivity_scores) > 1000:
                break
    
    return np.array(sensitivity_scores)


def compare_with_standard_ae(train_loader, test_loader, device):
    """
    Compare contractive AE with standard AE.
    """
    # Standard autoencoder
    standard_ae = ContractiveAutoencoder().to(device)
    optimizer = optim.Adam(standard_ae.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    print("Training Standard AE...")
    for epoch in range(15):
        standard_ae.train()
        for images, _ in train_loader:
            images = images.view(images.size(0), -1).to(device)
            optimizer.zero_grad()
            recon, _ = standard_ae(images)
            loss = criterion(recon, images)
            loss.backward()
            optimizer.step()
    
    # Contractive autoencoder
    contractive_ae = ContractiveAutoencoder().to(device)
    
    print("\nTraining Contractive AE...")
    train_contractive_autoencoder(
        contractive_ae, train_loader, device, 
        lambda_contractive=0.1, num_epochs=15
    )
    
    # Compare sensitivity
    print("\nAnalyzing sensitivity to noise...")
    
    std_sensitivity = analyze_contraction(standard_ae, test_loader, device)
    cae_sensitivity = analyze_contraction(contractive_ae, test_loader, device)
    
    print(f"Standard AE sensitivity: {np.mean(std_sensitivity):.4f} ± {np.std(std_sensitivity):.4f}")
    print(f"Contractive AE sensitivity: {np.mean(cae_sensitivity):.4f} ± {np.std(cae_sensitivity):.4f}")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].hist(std_sensitivity, bins=50, alpha=0.7, label='Standard AE')
    axes[0].hist(cae_sensitivity, bins=50, alpha=0.7, label='Contractive AE')
    axes[0].set_xlabel('Sensitivity (||Δz|| / ||Δx||)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Encoder Sensitivity Distribution')
    axes[0].legend()
    
    axes[1].boxplot([std_sensitivity, cae_sensitivity], 
                    labels=['Standard', 'Contractive'])
    axes[1].set_ylabel('Sensitivity')
    axes[1].set_title('Sensitivity Comparison')
    
    plt.tight_layout()
    plt.savefig('contractive_comparison.png', dpi=150)
    plt.show()
    
    return standard_ae, contractive_ae
```

---

## Geometric Interpretation

### Manifold Learning View

The contractive penalty encourages:

1. **Flat latent manifold:** Encoder output varies slowly with input
2. **Noise directions contracted:** Non-manifold directions compressed
3. **Data manifold preserved:** Important variations retained

### Trade-off

$$\text{Low } \lambda \to \text{Better reconstruction, less robustness}$$
$$\text{High } \lambda \to \text{More robustness, worse reconstruction}$$

---

## Exercises

### Exercise 1: λ Tuning
Train contractive autoencoders with λ ∈ {0.001, 0.01, 0.1, 1.0}. Plot reconstruction error vs sensitivity.

### Exercise 2: Comparison with Denoising
Compare contractive AE (λ=0.1) with denoising AE (noise σ=0.3). Are the learned representations similar?

### Exercise 3: Jacobian Visualization
For a trained contractive AE, visualize the Jacobian for different input digits. Which directions are most contracted?

---

## Summary

| Aspect | Standard AE | Contractive AE |
|--------|-------------|----------------|
| Loss | Reconstruction only | Reconstruction + ||J||²_F |
| Sensitivity | High | Low (by design) |
| Robustness | Limited | Improved |
| Computation | Fast | Slower (Jacobian) |

---

## References

- Rifai, S., et al. (2011). "Contractive Auto-Encoders: Explicit Invariance During Feature Extraction." ICML.
- Alain, G., & Bengio, Y. (2014). "What Regularized Auto-Encoders Learn from the Data-Generating Distribution." JMLR.

---

## Next: Variational Autoencoders

Section 9.3 introduces VAEs, adding probabilistic structure to the latent space.
