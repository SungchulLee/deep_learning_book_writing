"""
FILE: 08_score_sde.py
DIFFICULTY: Advanced
ESTIMATED TIME: 4-5 hours
PREREQUISITES: 07_noise_conditional_scores.py, basic SDEs, calculus

LEARNING OBJECTIVES:
    1. Understand Score-based SDEs framework
    2. Implement Variance Exploding (VE) SDE
    3. Implement Variance Preserving (VP) SDE
    4. Understand probability flow ODE
    5. Implement reverse-time SDE sampling

MATHEMATICAL BACKGROUND:
    Score-based SDEs provide continuous-time formulation of diffusion.
    
    FORWARD SDE:
    dx = f(x, t)dt + g(t)dw
    
    where w is Brownian motion.
    
    REVERSE SDE:
    dx = [f(x, t) - g(t)²∇log p_t(x)]dt + g(t)dw̄
    
    The score ∇log p_t(x) is what we learn!
    
    VARIANCE EXPLODING (VE):
    f(x, t) = 0
    g(t) = σ(t)√(dσ(t)²/dt)
    
    VARIANCE PRESERVING (VP):
    f(x, t) = -β(t)x/2
    g(t) = √β(t)
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class ScoreSDE:
    """Base class for Score-based SDEs."""
    
    def __init__(self, beta_min=0.1, beta_max=20.0, T=1.0):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.T = T
    
    def f(self, x, t):
        """Drift coefficient."""
        raise NotImplementedError
    
    def g(self, t):
        """Diffusion coefficient."""
        raise NotImplementedError
    
    def marginal_prob(self, x0, t):
        """
        Compute mean and std of p_t(x|x₀).
        
        Returns:
            mean: E[x_t | x₀]
            std: √Var[x_t | x₀]
        """
        raise NotImplementedError


class VESDE(ScoreSDE):
    """
    Variance Exploding SDE.
    
    dx = σ(t)√(dσ(t)²/dt) dw
    
    Marginal: p_t(x|x₀) = N(x|x₀, σ(t)²I)
    """
    
    def __init__(self, sigma_min=0.01, sigma_max=50.0):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
    
    def f(self, x, t):
        return torch.zeros_like(x)
    
    def g(self, t):
        sigma_t = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        return sigma_t * np.sqrt(2 * np.log(self.sigma_max / self.sigma_min))
    
    def marginal_prob(self, x0, t):
        sigma_t = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        mean = x0
        std = sigma_t
        return mean, std


class VPSDE(ScoreSDE):
    """
    Variance Preserving SDE.
    
    dx = -β(t)x/2 dt + √β(t) dw
    
    Marginal: p_t(x|x₀) = N(x | α_t x₀, (1-α_t²)I)
    """
    
    def __init__(self, beta_min=0.1, beta_max=20.0):
        super().__init__(beta_min, beta_max)
    
    def beta(self, t):
        return self.beta_min + t * (self.beta_max - self.beta_min)
    
    def f(self, x, t):
        return -0.5 * self.beta(t) * x
    
    def g(self, t):
        return torch.sqrt(self.beta(t))
    
    def marginal_prob(self, x0, t):
        log_alpha_t = -0.25 * t ** 2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        alpha_t = torch.exp(log_alpha_t)
        mean = alpha_t * x0
        std = torch.sqrt(1 - alpha_t ** 2)
        return mean, std


def sde_loss(model, sde, x0):
    """
    Compute score matching loss for SDEs.
    
    L = E_t E_x₀ E_x_t[||s_θ(x_t, t) - ∇log p_t(x_t|x₀)||²]
    """
    # Sample random time
    batch_size = len(x0)
    t = torch.rand(batch_size, device=x0.device)
    
    # Sample from marginal distribution
    mean, std = sde.marginal_prob(x0, t.view(-1, 1))
    z = torch.randn_like(x0)
    x_t = mean + std * z
    
    # Predict score
    score_pred = model(x_t, t)
    
    # True score: ∇log p_t(x_t|x₀) = -(x_t - mean)/std²
    score_true = -z / std
    
    # MSE loss weighted by std²
    loss = torch.mean(std ** 2 * torch.sum((score_pred - score_true) ** 2, dim=1))
    
    return loss


def reverse_sde_sampling(model, sde, shape, n_steps=1000):
    """
    Sample using reverse-time SDE.
    
    Implements: dx = [f - g²∇log p]dt + g dw̄
    """
    x = torch.randn(shape)
    dt = 1.0 / n_steps
    
    with torch.no_grad():
        for i in range(n_steps):
            t = 1.0 - i * dt
            t_tensor = torch.full((shape[0],), t)
            
            # Compute coefficients
            f = sde.f(x, t)
            g = sde.g(t)
            
            # Compute score
            score = model(x, t_tensor)
            
            # Reverse SDE step
            drift = f - (g ** 2) * score
            diffusion = g
            
            x = x + drift * dt + diffusion * np.sqrt(dt) * torch.randn_like(x)
    
    return x


def demo_sde():
    """Demonstrate Score SDE on 2D Gaussian."""
    print("Score-based SDE Demo")
    print("=" * 80)
    
    # Simple 2D Gaussian data
    data = torch.randn(1000, 2)
    
    # Create SDE
    sde = VPSDE(beta_min=0.1, beta_max=20.0)
    
    # Simple score model (time-conditional)
    class SimpleScoreModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(3, 128),  # x (2D) + t (1D)
                nn.SiLU(),
                nn.Linear(128, 128),
                nn.SiLU(),
                nn.Linear(128, 2)
            )
        
        def forward(self, x, t):
            t = t.view(-1, 1)
            inp = torch.cat([x, t], dim=-1)
            return self.net(inp)
    
    model = SimpleScoreModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print("\nTraining score model...")
    for epoch in range(3000):
        optimizer.zero_grad()
        loss = sde_loss(model, sde, data)
        loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f}")
    
    # Sample
    print("\nGenerating samples via reverse SDE...")
    samples = reverse_sde_sampling(model, sde, (500, 2), n_steps=1000)
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].scatter(data[:, 0].numpy(), data[:, 1].numpy(), s=1, alpha=0.5)
    axes[0].set_title('Training Data')
    axes[0].set_aspect('equal')
    
    axes[1].scatter(samples[:, 0].numpy(), samples[:, 1].numpy(), s=1, alpha=0.5)
    axes[1].set_title('Generated Samples (Reverse SDE)')
    axes[1].set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('/home/claude/demo_score_sde.png', dpi=150, bbox_inches='tight')
    print("\nSaved demo_score_sde.png")
    print("\n✓ Score SDE successfully implemented!")


if __name__ == "__main__":
    demo_sde()
