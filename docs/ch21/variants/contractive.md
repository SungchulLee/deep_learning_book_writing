# Contractive Autoencoder

Learn robust representations by explicitly penalizing the sensitivity of the encoder to input perturbations.

---

## Overview

**What you'll learn:**

- The Jacobian of the encoder and its role in measuring sensitivity
- Contractive penalty: Frobenius norm of the encoder Jacobian
- Geometric interpretation as manifold learning
- Connection to denoising autoencoders
- Efficient and exact Jacobian computation in PyTorch

---

## Mathematical Foundation

### The Contractive Penalty

A **contractive autoencoder (CAE)** augments the standard reconstruction loss with a penalty on the Frobenius norm of the encoder's Jacobian:

$$\mathcal{L} = \underbrace{\|x - g(f(x))\|^2}_{\text{reconstruction}} + \underbrace{\lambda \|J_f(x)\|_F^2}_{\text{contractive penalty}}$$

where:

- $f$: encoder function mapping input to latent representation
- $g$: decoder function mapping latent back to input space
- $J_f(x) = \frac{\partial f(x)}{\partial x} \in \mathbb{R}^{k \times d}$: Jacobian matrix of the encoder
- $\|J_f\|_F^2 = \sum_{ij} J_{ij}^2$: Frobenius norm squared (sum of all squared partial derivatives)
- $\lambda$: regularization strength controlling the reconstruction-contraction trade-off

### Intuition

| Component | Effect |
|-----------|--------|
| Reconstruction loss | Learn to faithfully reconstruct inputs |
| Jacobian penalty | Make encoder insensitive to input perturbations |

The Jacobian penalty encourages:

- **Local invariance:** Small changes in input produce small changes in the latent code
- **Robust representations:** The encoder learns to ignore noise while capturing essential structure
- **Flat manifolds:** The latent space is locally constant along noise directions but varies along the data manifold

### Trade-off

$$\text{Low } \lambda \to \text{Better reconstruction, less robustness}$$
$$\text{High } \lambda \to \text{More robustness, worse reconstruction}$$

The reconstruction term wants $f$ to be sensitive to all input variations (to enable accurate reconstruction), while the contractive term wants $f$ to be insensitive. The balance forces the encoder to be sensitive only along directions that matter for reconstruction — i.e., the data manifold.

---

## Connection to Denoising Autoencoders

### Theoretical Link

For small Gaussian noise with variance $\sigma^2$, the denoising autoencoder objective approximately minimizes:

$$\mathcal{L}_{DAE} \approx \|x - g(f(x))\|^2 + \sigma^2 \|J_f(x)\|_F^2$$

**Key insight:** Denoising with Gaussian noise implicitly applies a contractive penalty, with the noise variance $\sigma^2$ playing the role of $\lambda$.

### Comparison

| Aspect | Denoising AE | Contractive AE |
|--------|--------------|----------------|
| Regularization | Via corrupted inputs | Via explicit Jacobian penalty |
| Computation | Standard forward pass with noise | Requires Jacobian computation |
| Flexibility | Different noise types available | Direct control over contraction strength |
| Interpretation | Learn to denoise | Minimize encoder sensitivity |
| Training cost | Standard | Higher (Jacobian is expensive) |

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
    
    Uses sigmoid activations in the encoder for bounded outputs,
    which keeps the Jacobian well-behaved.
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
    ||J_f||²_F = Σ_ij (∂z_i / ∂x_j)²
    
    Computed column by column using autograd.
    """
    x = x.requires_grad_(True)
    z = model.encode(x)
    
    jacobian_norm_sq = 0.0
    
    for i in range(z.shape[1]):
        grad_outputs = torch.zeros_like(z)
        grad_outputs[:, i] = 1.0
        
        jacobian_col = grad(
            outputs=z,
            inputs=x,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True
        )[0]
        
        jacobian_norm_sq = jacobian_norm_sq + torch.sum(jacobian_col ** 2)
    
    return jacobian_norm_sq / x.shape[0]  # Average over batch
```

---

## Training

```python
def train_contractive_autoencoder(
    model, train_loader, device, 
    lambda_contractive=0.1, num_epochs=15
):
    """
    Train contractive autoencoder.
    
    Loss = Reconstruction + λ × ||J_f||²_F
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
```

---

## Sensitivity Analysis

```python
def analyze_contraction(model, test_loader, device, noise_std=0.1):
    """
    Analyze how contractive the learned representation is by
    measuring the ratio of latent change to input change.
    
    A contractive encoder has sensitivity ratio << 1.
    """
    model.eval()
    
    sensitivity_scores = []
    
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.view(images.size(0), -1).to(device)
            
            z_original = model.encode(images)
            
            noise = torch.randn_like(images) * noise_std
            z_noisy = model.encode(images + noise)
            
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
    Compare contractive AE with standard AE on encoder sensitivity.
    """
    # Standard autoencoder (same architecture, no contractive penalty)
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
    
    print(f"Standard AE sensitivity: {np.mean(std_sensitivity):.4f} "
          f"± {np.std(std_sensitivity):.4f}")
    print(f"Contractive AE sensitivity: {np.mean(cae_sensitivity):.4f} "
          f"± {np.std(cae_sensitivity):.4f}")
    
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

The contractive penalty encourages the encoder to learn a mapping where:

1. **Data manifold directions preserved:** The encoder varies along directions where the data actually lives
2. **Noise directions contracted:** Non-manifold directions (noise) are mapped to near-zero changes in latent space
3. **Flat latent manifold:** The latent representation is locally constant along noise directions

This is precisely the behavior desired for learning the intrinsic geometry of a data manifold embedded in high-dimensional space.

---

## Quantitative Finance Application

Contractive autoencoders are valuable in finance for learning **stable factor representations**:

- **Robust risk factors:** The contractive penalty ensures that small perturbations in market data (bid-ask bounce, microstructure noise) do not change the extracted factors, producing more stable risk decompositions
- **Regime-invariant features:** By penalizing sensitivity, the learned features are less susceptible to transient market dislocations
- **Regularized covariance estimation:** The contractive encoder implicitly regularizes the learned covariance structure, reducing estimation error in high-dimensional settings

---

## Exercises

### Exercise 1: λ Tuning
Train contractive autoencoders with $\lambda \in \{0.001, 0.01, 0.1, 1.0\}$. Plot reconstruction error vs sensitivity to characterize the trade-off frontier.

### Exercise 2: Comparison with Denoising
Compare contractive AE ($\lambda = 0.1$) with denoising AE (noise $\sigma = 0.3$). Are the learned representations similar? Measure using latent space distance correlation.

### Exercise 3: Jacobian Visualization
For a trained contractive AE, visualize the Jacobian matrix for different input digits. Which input directions are most contracted?

---

## Summary

| Aspect | Standard AE | Contractive AE |
|--------|-------------|----------------|
| Loss | Reconstruction only | Reconstruction + $\|J\|_F^2$ |
| Sensitivity | High (unconstrained) | Low (by design) |
| Robustness | Limited | Improved |
| Computation | Fast | Slower (Jacobian computation) |
| Manifold learning | Implicit | Explicit via penalty |

**Key Insight:** The contractive autoencoder provides a principled approach to learning representations that are robust to input perturbations by directly penalizing the encoder's sensitivity. The theoretical equivalence to denoising autoencoders (for small Gaussian noise) unifies two seemingly different regularization strategies under a common framework.

---

## References

- Rifai, S., et al. (2011). "Contractive Auto-Encoders: Explicit Invariance During Feature Extraction." *ICML*.
- Alain, G., & Bengio, Y. (2014). "What Regularized Auto-Encoders Learn from the Data-Generating Distribution." *JMLR*.
