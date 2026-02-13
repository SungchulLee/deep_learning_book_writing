"""
MODULE 08: Score-Based SDEs
===========================

DIFFICULTY: Advanced
TIME: 3-4 hours
PREREQUISITES: Modules 01-07, Basic SDE knowledge

LEARNING OBJECTIVES:
- Understand continuous-time diffusion framework
- Implement VE and VP SDEs
- Connect discrete DDPM to continuous SDEs

Key equation:
dx = f(x,t)dt + g(t)dw  (forward SDE)
dx = [f(x,t) - g(t)²∇log p_t(x)]dt + g(t)dw̄  (reverse SDE)

Author: Sungchul @ Yonsei University
"""

import torch
import numpy as np

print("MODULE 08: Score-Based SDE Framework")
print("="*80)

print("""
CONTINUOUS-TIME FORMULATION:
--------------------------
Instead of discrete steps t=0,1,2,...,T
Use continuous time t ∈ [0, T]

FORWARD SDE (add noise):
dx = f(x,t)dt + g(t)dw

where:
- f(x,t): drift coefficient
- g(t): diffusion coefficient  
- dw: Brownian motion

REVERSE SDE (remove noise):
dx = [f(x,t) - g(t)²∇log p_t(x)]dt + g(t)dw̄

Key: Score ∇log p_t(x) appears in reverse!
This is learned by neural network: s_θ(x,t) ≈ ∇log p_t(x)

TWO MAIN FORMULATIONS:
---------------------

1. VARIANCE EXPLODING (VE):
   Forward: dx = √(dσ²/dt) dw
   → Variance increases: σ_t² = σ_min² + t(σ_max² - σ_min²)/T
   
2. VARIANCE PRESERVING (VP):
   Forward: dx = -0.5β(t)x dt + √β(t) dw
   → Variance stays bounded
   → This is DDPM in continuous time!

β(t) controls noise schedule
""")

class VESDE:
    """Variance Exploding SDE"""
    def __init__(self, sigma_min=0.01, sigma_max=50.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
    
    def sigma_t(self, t):
        """Noise schedule"""
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t
    
    def forward_sde(self, x, t):
        """dx = √(dσ²/dt) dw"""
        sigma = self.sigma_t(t)
        drift = torch.zeros_like(x)
        
        # g(t) = σ(t)√(2 log(σ_max/σ_min))
        diffusion = sigma * np.sqrt(2 * np.log(self.sigma_max / self.sigma_min))
        return drift, diffusion
    
    def reverse_sde(self, score_fn, x, t):
        """
        dx = [-g²∇log p]dt + g dw̄
        
        For VE, f=0, so reverse drift is just -g²∇log p
        """
        sigma = self.sigma_t(t)
        score = score_fn(x, t)
        
        diffusion = sigma * np.sqrt(2 * np.log(self.sigma_max / self.sigma_min))
        
        # FIX: Reshape diffusion for proper broadcasting
        # diffusion has shape (batch,) but score has shape (batch, dim)
        # We need to make diffusion have shape (batch, 1) for broadcasting
        if isinstance(diffusion, torch.Tensor) and diffusion.dim() > 0:
            diffusion_sq = (diffusion ** 2).unsqueeze(-1)
        else:
            diffusion_sq = diffusion ** 2
        
        drift = -diffusion_sq * score
        
        return drift, diffusion

class VPSDE:
    """Variance Preserving SDE (equivalent to DDPM)"""
    def __init__(self, beta_min=0.1, beta_max=20.0):
        self.beta_min = beta_min
        self.beta_max = beta_max
    
    def beta_t(self, t):
        """Linear noise schedule"""
        return self.beta_min + t * (self.beta_max - self.beta_min)
    
    def forward_sde(self, x, t):
        """dx = -0.5β(t)x dt + √β(t) dw"""
        beta = self.beta_t(t)
        
        # FIX: Ensure proper broadcasting
        if isinstance(beta, torch.Tensor) and beta.dim() > 0:
            beta = beta.unsqueeze(-1)
        
        drift = -0.5 * beta * x
        diffusion = torch.sqrt(beta) if isinstance(beta, torch.Tensor) else np.sqrt(beta)
        return drift, diffusion
    
    def reverse_sde(self, score_fn, x, t):
        """
        dx = [-0.5β(t)x - β(t)∇log p]dt + √β(t) dw̄
        """
        beta = self.beta_t(t)
        score = score_fn(x, t)
        
        # FIX: Ensure proper broadcasting
        if isinstance(beta, torch.Tensor) and beta.dim() > 0:
            beta_reshaped = beta.unsqueeze(-1)
        else:
            beta_reshaped = beta
        
        drift = -0.5 * beta_reshaped * x - beta_reshaped * score
        diffusion = torch.sqrt(beta) if isinstance(beta, torch.Tensor) else np.sqrt(beta)
        
        return drift, diffusion

def euler_maruyama_sampler(sde, score_fn, shape, n_steps=1000):
    """
    Euler-Maruyama method for solving reverse SDE
    
    Discretization:
    x_{i-1} = x_i + drift * Δt + diffusion * √Δt * z
    where z ~ N(0, I)
    """
    # Start from prior
    x = torch.randn(shape)
    
    dt = 1.0 / n_steps
    trajectory = [x.clone()]
    
    for i in range(n_steps):
        t = 1.0 - i * dt  # Go backwards from 1 to 0
        t_tensor = torch.ones(shape[0]) * t
        
        # Reverse SDE step
        drift, diffusion = sde.reverse_sde(score_fn, x, t_tensor)
        
        # FIX: Handle diffusion broadcasting for noise term
        if isinstance(diffusion, torch.Tensor) and diffusion.dim() > 0:
            noise = torch.randn_like(x) * diffusion.unsqueeze(-1) * np.sqrt(dt)
        else:
            noise = torch.randn_like(x) * diffusion * np.sqrt(dt)
        
        # Euler-Maruyama update
        x = x + drift * dt + noise
        
        if i % 100 == 0:
            trajectory.append(x.clone())
    
    return x, trajectory

print("""
PROBABILITY FLOW ODE:
--------------------
Alternative to reverse SDE: Deterministic sampling!

dx = [f(x,t) - 0.5*g(t)²∇log p_t(x)]dt

Same marginals as SDE, but:
✓ Deterministic (no randomness)
✓ Invertible (can encode/decode)
✓ Faster (larger steps possible)
✗ May be less sample quality

This enables:
- Image interpolation
- Semantic editing
- Exact likelihood computation

CONNECTION TO DDPM:
------------------
DDPM discrete steps:
x_{t-1} = √(α_t) [x_t - (1-α_t)/√(1-ᾱ_t) * ε_θ(x_t,t)] + σ_t z

VP-SDE continuous limit:
dx = [-0.5β(t)x - β(t)ε_θ(x,t)]dt + √β(t) dw

They're equivalent! Score s(x,t) = -ε(x,t)/√(1-ᾱ_t)

KEY ADVANTAGES OF SDE VIEW:
--------------------------
✓ Unified framework (VE, VP, sub-VP, etc.)
✓ Flexible samplers (SDE, ODE, predictor-corrector)
✓ Theoretical analysis easier
✓ Novel architectures and schedules
✓ Connects to physics (Brownian motion, Langevin)
""")

# Simple demonstration
print("\nDemonstration: VE-SDE on 2D Gaussian")
print("-" * 80)

# True 2D Gaussian
mean = torch.zeros(2)
cov = torch.eye(2)

def gaussian_score(x, t):
    """Analytical score for Gaussian"""
    return -x  # For N(0, I), score is -x

# Sample with VE-SDE
vesde = VESDE(sigma_min=0.01, sigma_max=10.0)
samples, _ = euler_maruyama_sampler(vesde, gaussian_score, (500, 2), n_steps=500)

print(f"Generated samples shape: {samples.shape}")
print(f"Sample mean: {samples.mean(dim=0)}")
print(f"Sample std: {samples.std(dim=0)}")
print("✓ VE-SDE sampling successful!")

print("""
SAMPLING STRATEGIES:
-------------------

1. EULER-MARUYAMA (EM):
   Simple, first-order SDE solver
   x_{i-1} = x_i + drift*dt + diffusion*√dt*z

2. PREDICTOR-CORRECTOR:
   Predictor: EM step
   Corrector: Langevin MCMC steps
   Better quality, slower

3. ODE SOLVERS:
   Use probability flow ODE
   Adaptive step size (RK45, DPM-Solver)
   Faster, deterministic

4. DDIM-STYLE:
   Skip timesteps (non-Markovian)
   Much faster (10-50 steps)
   Quality trade-off

CHOOSING SDE TYPE:
-----------------
- VE: Better for unconditional generation
- VP: Better for conditional/guided generation
- Matches DDPM formulation
- Easier to condition

In practice, VP-SDE (DDPM) most common!
""")

print("\n✓ Module 08 complete!")
print("Next: Apply to real images (Module 09)!")