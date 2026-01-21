# Amortized Variational Inference

## Learning Objectives

By the end of this section, you will be able to:

1. Understand the concept of amortization in variational inference
2. Implement inference networks for posterior approximation
3. Build and train Variational Autoencoders (VAEs)
4. Analyze the amortization gap and its implications
5. Apply amortized VI to large-scale problems

## From Per-Instance to Amortized Inference

Traditional VI optimizes variational parameters **separately for each observation**:

$$
\phi_n^* = \arg\max_\phi \text{ELBO}(q_\phi(z_n | x_n))
$$

For $N$ observations, this requires $N$ separate optimization problems!

**Amortized VI** learns a **single function** that maps observations to variational parameters:

$$
\phi = f_\psi(x) \quad \text{(inference network)}
$$

where $\psi$ are the network parameters, shared across all observations.

### Benefits of Amortization

1. **Scalability**: Single network for all data points
2. **Generalization**: Can infer posteriors for new, unseen data
3. **Speed**: No optimization at test time (just forward pass)
4. **Integration**: Natural fit with deep generative models

### The Cost: Amortization Gap

The amortization gap is the sub-optimality from using a shared network:

$$
\text{Gap}(x) = \max_\phi \text{ELBO}(q_\phi(z|x)) - \text{ELBO}(q_{f_\psi(x)}(z|x))
$$

The inference network may not perfectly capture the optimal variational parameters for every data point.

## Variational Autoencoders (VAEs)

The **Variational Autoencoder** is the canonical example of amortized VI, combining:

1. **Encoder** (Inference Network): Maps observations to variational parameters
2. **Decoder** (Generative Model): Maps latent variables to observations
3. **ELBO Objective**: Joint training of encoder and decoder

### VAE Generative Model

$$
\begin{aligned}
\text{Prior: } & z \sim p(z) = \mathcal{N}(0, I) \\
\text{Likelihood: } & x | z \sim p_\theta(x|z)
\end{aligned}
$$

The decoder neural network $p_\theta(x|z)$ maps latent codes to data distributions.

### VAE Inference Model

The encoder approximates the intractable posterior $p(z|x)$:

$$
q_\phi(z|x) = \mathcal{N}(\mu_\phi(x), \text{diag}(\sigma_\phi^2(x)))
$$

where $\mu_\phi(x)$ and $\sigma_\phi(x)$ are outputs of the encoder network.

### VAE ELBO

The VAE objective is the ELBO averaged over the dataset:

$$
\mathcal{L}(\theta, \phi) = \frac{1}{N} \sum_{n=1}^N \left[\mathbb{E}_{q_\phi(z|x_n)}[\log p_\theta(x_n|z)] - \text{KL}(q_\phi(z|x_n) \| p(z))\right]
$$

**Reconstruction term**: How well can the decoder reconstruct $x$ from sampled $z$?

**KL term**: How close is the approximate posterior to the prior?

### Reparameterization for VAE

To backpropagate through the sampling operation:

$$
z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

This allows gradients to flow through $\mu_\phi$ and $\sigma_\phi$.

## PyTorch Implementation

### Complete VAE Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Dict
import matplotlib.pyplot as plt
import numpy as np

class Encoder(nn.Module):
    """
    VAE Encoder: Maps input x to variational parameters (μ, log σ²).
    
    Architecture: x -> [hidden layers] -> (μ, log σ²)
    """
    
    def __init__(self, input_dim: int, hidden_dims: list, latent_dim: int):
        super().__init__()
        
        # Build encoder layers
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
            ])
            prev_dim = h_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Output layers for mean and log-variance
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode x to variational parameters.
        
        Returns:
            mu: Mean of q(z|x)
            logvar: Log-variance of q(z|x)
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    """
    VAE Decoder: Maps latent z to reconstruction parameters.
    
    Architecture: z -> [hidden layers] -> x_recon
    """
    
    def __init__(self, latent_dim: int, hidden_dims: list, output_dim: int):
        super().__init__()
        
        # Build decoder layers
        layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
            ])
            prev_dim = h_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode z to reconstruction.
        """
        return self.decoder(z)


class VAE(nn.Module):
    """
    Variational Autoencoder.
    
    Combines encoder (inference network) and decoder (generative model)
    trained jointly by maximizing ELBO.
    """
    
    def __init__(self, input_dim: int, hidden_dims: list, latent_dim: int):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.encoder = Encoder(input_dim, hidden_dims, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dims, input_dim)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to variational parameters."""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to reconstruction."""
        return self.decoder(z)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = μ + σ ⊙ ε
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass.
        
        Returns:
            x_recon: Reconstructed input
            mu: Variational mean
            logvar: Variational log-variance
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
    
    def loss(self, x: torch.Tensor, x_recon: torch.Tensor,
             mu: torch.Tensor, logvar: torch.Tensor,
             beta: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute VAE loss = -ELBO = Reconstruction Loss + β * KL Divergence
        
        Args:
            x: Input data
            x_recon: Reconstructed data
            mu: Variational mean
            logvar: Variational log-variance
            beta: Weight on KL term (β-VAE)
        
        Returns:
            loss: Total loss
            recon_loss: Reconstruction term
            kl_loss: KL divergence term
        """
        # Reconstruction loss (negative log-likelihood)
        # For continuous data, use MSE (Gaussian likelihood)
        recon_loss = F.mse_loss(x_recon, x, reduction='sum') / x.size(0)
        
        # KL divergence: KL(N(μ,σ²) || N(0,1))
        # = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        
        # Total loss
        loss = recon_loss + beta * kl_loss
        
        return loss, recon_loss, kl_loss
    
    def sample(self, n_samples: int, device: str = 'cpu') -> torch.Tensor:
        """
        Generate samples from the model.
        
        z ~ p(z) = N(0, I)
        x ~ p(x|z)
        """
        z = torch.randn(n_samples, self.latent_dim).to(device)
        samples = self.decode(z)
        return samples
    
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct input through encoder and decoder."""
        mu, _ = self.encode(x)  # Use mean (no sampling)
        return self.decode(mu)


def train_vae(model: VAE, train_loader: DataLoader,
              n_epochs: int = 100, lr: float = 1e-3,
              beta: float = 1.0, verbose: bool = True) -> Dict:
    """
    Train VAE by maximizing ELBO.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    history = {
        'loss': [],
        'recon_loss': [],
        'kl_loss': []
    }
    
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        epoch_recon = 0
        epoch_kl = 0
        
        for batch in train_loader:
            x = batch[0]
            
            optimizer.zero_grad()
            
            # Forward pass
            x_recon, mu, logvar = model(x)
            
            # Compute loss
            loss, recon_loss, kl_loss = model.loss(x, x_recon, mu, logvar, beta)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_kl += kl_loss.item()
        
        n_batches = len(train_loader)
        history['loss'].append(epoch_loss / n_batches)
        history['recon_loss'].append(epoch_recon / n_batches)
        history['kl_loss'].append(epoch_kl / n_batches)
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}: Loss = {history['loss'][-1]:.4f}, "
                  f"Recon = {history['recon_loss'][-1]:.4f}, "
                  f"KL = {history['kl_loss'][-1]:.4f}")
    
    return history


def measure_amortization_gap(model: VAE, x: torch.Tensor,
                              n_opt_steps: int = 1000,
                              lr: float = 0.01) -> Tuple[float, float]:
    """
    Measure amortization gap by comparing amortized vs per-instance optimization.
    
    Gap = ELBO(optimal per-instance) - ELBO(amortized)
    """
    # Amortized ELBO
    with torch.no_grad():
        x_recon, mu_amort, logvar_amort = model(x)
        _, recon_amort, kl_amort = model.loss(x, x_recon, mu_amort, logvar_amort)
        elbo_amortized = -(recon_amort + kl_amort).item()
    
    # Per-instance optimization
    mu_opt = mu_amort.clone().detach().requires_grad_(True)
    logvar_opt = logvar_amort.clone().detach().requires_grad_(True)
    
    optimizer = torch.optim.Adam([mu_opt, logvar_opt], lr=lr)
    
    for _ in range(n_opt_steps):
        optimizer.zero_grad()
        
        # Sample z using optimized parameters
        std = torch.exp(0.5 * logvar_opt)
        z = mu_opt + std * torch.randn_like(std)
        
        # Decode
        x_recon = model.decode(z)
        
        # Loss
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar_opt - mu_opt.pow(2) - logvar_opt.exp())
        loss = recon_loss + kl_loss
        
        loss.backward()
        optimizer.step()
    
    elbo_optimal = -loss.item() / x.size(0)
    gap = elbo_optimal - elbo_amortized
    
    return elbo_amortized, elbo_optimal, gap


def visualize_vae_results(model: VAE, history: Dict, test_data: torch.Tensor):
    """Comprehensive visualization of VAE results."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: Training curves
    ax = axes[0, 0]
    ax.plot(history['loss'], label='Total Loss', linewidth=2)
    ax.plot(history['recon_loss'], label='Reconstruction', linewidth=2)
    ax.plot(history['kl_loss'], label='KL Divergence', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('(a) Training Curves', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Original vs Reconstructed (first few samples)
    ax = axes[0, 1]
    model.eval()
    with torch.no_grad():
        recon = model.reconstruct(test_data[:10])
    
    n_show = min(5, len(test_data))
    for i in range(n_show):
        ax.plot(test_data[i].numpy(), 'b-', alpha=0.5)
        ax.plot(recon[i].numpy(), 'r--', alpha=0.5)
    ax.plot([], [], 'b-', label='Original')
    ax.plot([], [], 'r--', label='Reconstructed')
    ax.set_xlabel('Dimension', fontsize=11)
    ax.set_ylabel('Value', fontsize=11)
    ax.set_title('(b) Reconstruction Quality', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Latent space (if 2D)
    ax = axes[0, 2]
    with torch.no_grad():
        mu, _ = model.encode(test_data)
    
    if model.latent_dim == 2:
        ax.scatter(mu[:, 0].numpy(), mu[:, 1].numpy(), alpha=0.5, s=10)
        ax.set_xlabel('z₁', fontsize=11)
        ax.set_ylabel('z₂', fontsize=11)
    else:
        # Show first two dimensions
        ax.scatter(mu[:, 0].numpy(), mu[:, 1].numpy(), alpha=0.5, s=10)
        ax.set_xlabel('z₁', fontsize=11)
        ax.set_ylabel('z₂', fontsize=11)
    ax.set_title('(c) Latent Space (first 2 dims)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Generated samples
    ax = axes[1, 0]
    with torch.no_grad():
        samples = model.sample(20)
    
    for i in range(min(10, len(samples))):
        ax.plot(samples[i].numpy(), alpha=0.5)
    ax.set_xlabel('Dimension', fontsize=11)
    ax.set_ylabel('Value', fontsize=11)
    ax.set_title('(d) Generated Samples', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 5: KL per dimension
    ax = axes[1, 1]
    with torch.no_grad():
        mu, logvar = model.encode(test_data)
        # KL per latent dimension
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean(dim=0)
    
    ax.bar(range(len(kl_per_dim)), kl_per_dim.numpy())
    ax.set_xlabel('Latent Dimension', fontsize=11)
    ax.set_ylabel('KL Divergence', fontsize=11)
    ax.set_title('(e) KL per Latent Dimension', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Amortization gap analysis
    ax = axes[1, 2]
    gaps = []
    n_test = min(50, len(test_data))
    for i in range(n_test):
        _, _, gap = measure_amortization_gap(model, test_data[i:i+1], n_opt_steps=200)
        gaps.append(gap)
    
    ax.hist(gaps, bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(gaps), color='red', linestyle='--', 
               linewidth=2, label=f'Mean = {np.mean(gaps):.4f}')
    ax.set_xlabel('Amortization Gap', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('(f) Amortization Gap Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('vae_results.png', dpi=150, bbox_inches='tight')
    plt.show()


# Example usage
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate synthetic data (mixture of Gaussians in high-dim space)
    n_samples = 2000
    input_dim = 20
    
    # Generate from a simple model: z -> x = Wz + noise
    true_latent_dim = 3
    W = torch.randn(input_dim, true_latent_dim)
    z_true = torch.randn(n_samples, true_latent_dim)
    data = z_true @ W.T + 0.1 * torch.randn(n_samples, input_dim)
    
    # Split data
    train_data = data[:1600]
    test_data = data[1600:]
    
    # Create data loader
    train_dataset = TensorDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    print("=" * 60)
    print("Variational Autoencoder Training")
    print("=" * 60)
    print(f"\nData shape: {data.shape}")
    print(f"True latent dim: {true_latent_dim}")
    
    # Create and train VAE
    latent_dim = 5  # Slightly overestimate true latent dim
    model = VAE(
        input_dim=input_dim,
        hidden_dims=[64, 32],
        latent_dim=latent_dim
    )
    
    print(f"\nVAE architecture:")
    print(f"  Input dim: {input_dim}")
    print(f"  Hidden dims: [64, 32]")
    print(f"  Latent dim: {latent_dim}")
    
    history = train_vae(model, train_loader, n_epochs=100, lr=1e-3, verbose=True)
    
    # Visualize results
    visualize_vae_results(model, history, test_data)
    
    # Measure amortization gap
    print("\n--- Amortization Gap Analysis ---")
    elbo_amort, elbo_opt, gap = measure_amortization_gap(model, test_data[:1])
    print(f"Amortized ELBO: {elbo_amort:.4f}")
    print(f"Optimal ELBO: {elbo_opt:.4f}")
    print(f"Gap: {gap:.4f}")
```

## Conditional VAE (CVAE)

The **Conditional VAE** extends VAE to model $p(x|c)$ where $c$ is a conditioning variable:

$$
\begin{aligned}
\text{Prior: } & z | c \sim p(z|c) \\
\text{Likelihood: } & x | z, c \sim p_\theta(x|z, c) \\
\text{Inference: } & q_\phi(z|x, c)
\end{aligned}
$$

### CVAE ELBO

$$
\mathcal{L}(\theta, \phi; x, c) = \mathbb{E}_{q_\phi(z|x,c)}[\log p_\theta(x|z,c)] - \text{KL}(q_\phi(z|x,c) \| p(z|c))
$$

### Applications

- **Class-conditional generation**: Generate samples from specific classes
- **Image-to-image translation**: Generate target given source
- **Structured prediction**: Model conditional distributions

## Reducing the Amortization Gap

Several techniques can reduce the gap between amortized and per-instance optimization:

### 1. Semi-Amortized Inference

Start with amortized inference, then refine with a few optimization steps:

```python
def semi_amortized_inference(model, x, n_refine_steps=10):
    # Get amortized initial parameters
    mu, logvar = model.encode(x)
    
    # Refine with gradient descent
    mu = mu.clone().requires_grad_(True)
    logvar = logvar.clone().requires_grad_(True)
    
    optimizer = torch.optim.Adam([mu, logvar], lr=0.01)
    
    for _ in range(n_refine_steps):
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        x_recon = model.decode(z)
        loss = compute_loss(x, x_recon, mu, logvar)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return mu.detach(), logvar.detach()
```

### 2. Iterative Amortized Inference

Use multiple iterations of the encoder:

$$
(\mu^{(t+1)}, \sigma^{(t+1)}) = f_\psi(x, \mu^{(t)}, \sigma^{(t)})
$$

### 3. More Expressive Inference Networks

Use normalizing flows to increase the flexibility of $q_\phi(z|x)$.

## Summary

**Amortized VI** learns a shared inference network:

$$
\phi = f_\psi(x)
$$

**Key components of VAE:**

- **Encoder**: $q_\phi(z|x)$ - approximate posterior
- **Decoder**: $p_\theta(x|z)$ - generative model  
- **ELBO**: $\mathbb{E}_q[\log p(x|z)] - \text{KL}(q(z|x) \| p(z))$
- **Reparameterization**: $z = \mu + \sigma \odot \epsilon$

**Trade-offs:**

- Fast inference at test time
- Amortization gap reduces quality
- Can be mitigated with semi-amortized methods

## Exercises

### Exercise 1: Convolutional VAE

Implement a VAE with convolutional encoder/decoder for image data.

### Exercise 2: β-VAE

Implement β-VAE and study the effect of β on disentanglement.

### Exercise 3: CVAE for Conditional Generation

Implement a CVAE for class-conditional generation on MNIST.

## References

1. Kingma, D. P., & Welling, M. (2014). "Auto-Encoding Variational Bayes."

2. Rezende, D. J., Mohamed, S., & Wierstra, D. (2014). "Stochastic Backpropagation and Approximate Inference in Deep Generative Models."

3. Sohn, K., Lee, H., & Yan, X. (2015). "Learning Structured Output Representation using Deep Conditional Generative Models."

4. Cremer, C., Li, X., & Duvenaud, D. (2018). "Inference Suboptimality in Variational Autoencoders."

5. Higgins, I., et al. (2017). "β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework."

6. Kim, Y., et al. (2018). "Semi-Amortized Variational Autoencoders."
