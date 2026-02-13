# ============================================================================
# diffusion_utils.py - Mathematical Foundations for Diffusion Models
# ============================================================================

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

"""
DIFFUSION MODEL UTILITIES - CORE CONCEPTS
==========================================

This module contains the mathematical engine for diffusion models.

KEY COMPONENTS:
---------------
1. NOISE SCHEDULES: Define how much noise to add at each timestep (β_t)
2. DIFFUSION PARAMETERS: Precomputed mathematical constants for efficiency
3. FORWARD DIFFUSION: Add noise using closed-form formula
   x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε
4. REVERSE DIFFUSION: Remove noise iteratively (generation)
5. TRAINING UTILITIES: Loss computation for learning
6. VISUALIZATION TOOLS: Understand diffusion visually

MATHEMATICAL FOUNDATION:
------------------------
Forward process (data → noise):
    q(x_t | x_{t-1}) = N(x_t; √(1-β_t)·x_{t-1}, β_t·I)

Closed-form solution:
    q(x_t | x_0) = N(x_t; √ᾱ_t·x_0, (1-ᾱ_t)·I)
    where ᾱ_t = ∏_{i=1}^t (1-β_i)

Reverse process (noise → data):
    p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t,t), Σ_θ(x_t,t))

The model learns μ_θ by predicting noise ε_θ(x_t, t).
"""

# ============================================================================
# NOISE SCHEDULES
# ============================================================================

def linear_beta_schedule(timesteps: int, beta_start: float = 0.0001, 
                        beta_end: float = 0.02) -> torch.Tensor:
    """
    Create a linear noise schedule.
    
    β_t increases linearly: [0.0001, 0.0002, ..., 0.02]
    
    PROS: Simple, intuitive, easy to understand
    CONS: Not optimal for images, adds noise too quickly at end
    
    Historical note: Used in original DDPM paper.
    Modern practice: Cosine schedule is better for images.
    
    Args:
        timesteps: Total diffusion steps (T)
        beta_start: Initial noise level (≈0.0001)
        beta_end: Final noise level (≈0.02)
    
    Returns:
        betas: Tensor of shape (timesteps,)
    """
    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Create a cosine noise schedule (improved over linear).
    
    WHY COSINE?
    -----------
    - Smoother progression throughout diffusion
    - Less aggressive at the end
    - Better preservation of signal-to-noise ratio
    - Higher quality image generation
    
    MATHEMATICAL DEFINITION:
    ------------------------
    f(t) = cos((t/T + s)/(1 + s) · π/2)²
    ᾱ_t = f(t)/f(0)
    β_t = 1 - (ᾱ_t/ᾱ_{t-1})
    
    The cosine creates a smooth S-curve for ᾱ_t decay.
    Offset 's' prevents numerical instabilities at boundaries.
    
    BENEFITS:
    ---------
    1. Smooth transitions: Each step removes similar noise amounts
    2. Better gradients: Uniform learning signal across timesteps
    3. Fewer artifacts: Less abrupt changes
    4. Empirical success: State-of-the-art models use this
    
    Reference: "Improved Denoising Diffusion Probabilistic Models"
               (Nichol & Dhariwal, 2021)
    
    Args:
        timesteps: Total diffusion steps
        s: Small offset (0.008 default, prevents β_t ≈ 0 at start)
    
    Returns:
        betas: Tensor of shape (timesteps,), smoothly increasing
    """
    # Create time indices (need T+1 points to compute differences)
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    
    # Compute ᾱ_t using cosine function
    t_normalized = (x / timesteps + s) / (1 + s)
    alphas_cumprod = torch.cos(t_normalized * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]  # Normalize so ᾱ_0=1
    
    # Compute β_t from ᾱ_t: β_t = 1 - (ᾱ_t/ᾱ_{t-1})
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    
    # Clip for numerical stability
    return torch.clip(betas, 0.0001, 0.9999)

# ============================================================================
# DIFFUSION PARAMETERS (PRECOMPUTATION FOR EFFICIENCY)
# ============================================================================

def get_diffusion_parameters(betas: torch.Tensor) -> dict:
    """
    Precompute all mathematical constants needed for diffusion.
    
    WHY PRECOMPUTE?
    ---------------
    Training requires these values repeatedly. Computing once is O(T),
    then O(1) lookups vs O(T) per iteration → massive speedup!
    
    For T=1000, 10k iterations: 1000 computations + 10k lookups vs 10M computations
    
    DERIVED QUANTITIES:
    -------------------
    α_t = 1 - β_t                    (signal retention)
    ᾱ_t = ∏_{i=1}^t α_i             (cumulative signal)
    √ᾱ_t, √(1-ᾱ_t)                  (forward diffusion coefficients)
    1/√α_t                           (reverse diffusion scaling)
    σ_t² = β_t(1-ᾱ_{t-1})/(1-ᾱ_t)  (posterior variance)
    
    Args:
        betas: β schedule, shape (T,)
    
    Returns:
        Dictionary with all precomputed parameters:
        - 'betas': β_t values
        - 'alphas': α_t = 1 - β_t
        - 'alphas_cumprod': ᾱ_t (cumulative product)
        - 'alphas_cumprod_prev': ᾱ_{t-1} (shifted)
        - 'sqrt_alphas_cumprod': √ᾱ_t (data coefficient)
        - 'sqrt_one_minus_alphas_cumprod': √(1-ᾱ_t) (noise coefficient)
        - 'sqrt_recip_alphas': 1/√α_t (reverse scaling)
        - 'posterior_variance': σ_t² (reverse noise)
    """
    # α_t = 1 - β_t (signal retention per step)
    alphas = 1.0 - betas
    
    # ᾱ_t = ∏ α_i (cumulative signal - KEY QUANTITY!)
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    # Properties: decreasing, starts ≈1, ends ≈0
    
    # ᾱ_{t-1} (prepend 1.0 because ᾱ_0 = 1)
    alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
    
    # Forward diffusion coefficients: x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    
    # Reverse diffusion scaling: μ = 1/√α_t·(...)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    
    # Posterior variance: how much noise to add during reverse
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    
    return {
        'betas': betas,
        'alphas': alphas,
        'alphas_cumprod': alphas_cumprod,
        'alphas_cumprod_prev': alphas_cumprod_prev,
        'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
        'sqrt_one_minus_alphas_cumprod': sqrt_one_minus_alphas_cumprod,
        'sqrt_recip_alphas': sqrt_recip_alphas,
        'posterior_variance': posterior_variance,
    }

# ============================================================================
# UTILITY: BATCHED INDEXING WITH BROADCASTING
# ============================================================================

def extract(tensor: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
    """
    Extract values from 1D tensor at indices t and reshape for broadcasting.
    
    THE PROBLEM:
    ------------
    Batch has different timesteps: t = [15, 73, 42, ...]
    Need different constants per sample: √ᾱ_15, √ᾱ_73, √ᾱ_42, ...
    
    SOLUTION:
    ---------
    1. Index: tensor[t] → get values for each sample
    2. Reshape: (batch,) → (batch, 1, 1, ...) for broadcasting
    
    BROADCASTING EXAMPLE:
    ---------------------
    x shape: (3, 2)          three 2D points
    values: (3, 1)           after reshape
    
    [[x1, y1],     [[v1, 1],      [[v1·x1, v1·y1],
     [x2, y2],  *   [v2, 1],  =    [v2·x2, v2·y2],
     [x3, y3]]      [v3, 1]]       [v3·x3, v3·y3]]
    
    Args:
        tensor: Source values, shape (T,)
        t: Timestep indices, shape (batch_size,)
        x_shape: Target data shape for broadcasting
    
    Returns:
        Extracted and reshaped values, shape (batch_size, 1, 1, ...)
    """
    batch_size = t.shape[0]
    out = tensor.gather(-1, t)  # Extract at indices
    
    # Reshape for broadcasting: (batch,) → (batch, 1, 1, ...)
    num_extra_dims = len(x_shape) - 1
    trailing_dims = (1,) * num_extra_dims
    return out.reshape(batch_size, *trailing_dims)

# ============================================================================
# FORWARD DIFFUSION
# ============================================================================

def forward_diffusion(x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor,
                     sqrt_alphas_cumprod: torch.Tensor,
                     sqrt_one_minus_alphas_cumprod: torch.Tensor) -> torch.Tensor:
    """
    Apply forward diffusion: add noise to clean data.
    
    FORMULA:
    --------
    x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε
    
    INTERPRETATION:
    ---------------
    Weighted sum of data and noise:
    - t=0:   ᾱ_0=1    → x_0 = x_0 (no noise)
    - t=50:  ᾱ_50≈0.6 → 60% data + 80% noise (magnitude)
    - t=100: ᾱ_100≈0.3 → 30% data + 95% noise
    - t→∞:   ᾱ_∞→0   → pure noise
    
    WHY SQUARE ROOTS?
    -----------------
    Variance preservation: √ᾱ_t² + √(1-ᾱ_t)² = 1
    Ensures x_t has same variance as x_0!
    
    Args:
        x_0: Clean data, shape (batch_size, *data_dims)
        t: Timesteps, shape (batch_size,)
        noise: Gaussian noise ε ~ N(0,I), same shape as x_0
        sqrt_alphas_cumprod: Precomputed √ᾱ_t, shape (T,)
        sqrt_one_minus_alphas_cumprod: Precomputed √(1-ᾱ_t), shape (T,)
    
    Returns:
        x_t: Noisy data at timestep t
    """
    # Extract coefficients for each sample's timestep and reshape for broadcasting
    sqrt_alpha_t = extract(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alpha_t = extract(sqrt_one_minus_alphas_cumprod, t, x_0.shape)
    
    # Apply forward diffusion formula
    return sqrt_alpha_t * x_0 + sqrt_one_minus_alpha_t * noise

# ============================================================================
# REVERSE DIFFUSION (GENERATION)
# ============================================================================

@torch.no_grad()
def p_sample(model: nn.Module, x_t: torch.Tensor, t: int, t_tensor: torch.Tensor,
            diffusion_params: dict, device: str = 'cpu') -> torch.Tensor:
    """
    Perform one reverse diffusion step: x_t → x_{t-1}.
    
    REVERSE FORMULA:
    ----------------
    μ_θ(x_t,t) = 1/√α_t · (x_t - β_t/√(1-ᾱ_t)·ε_θ(x_t,t))
    x_{t-1} = μ_θ + σ_t·z  (if t>0)
    x_0 = μ_θ              (if t=0)
    
    PROCESS:
    --------
    1. Model predicts noise: ε_θ(x_t,t)
    2. Compute denoised mean: μ_θ
    3. Add small noise σ_t·z for diversity (except at t=0)
    
    WHY ADD NOISE WHILE DENOISING?
    -------------------------------
    Without noise: deterministic → mode collapse, no diversity
    With noise: stochastic → rich samples, full distribution
    Key: noise added < noise removed → net progress!
    
    Args:
        model: Trained denoising model
        x_t: Noisy data at timestep t
        t: Current timestep (integer)
        t_tensor: Timestep as tensor, shape (batch_size,)
        diffusion_params: Precomputed constants
        device: 'cpu' or 'cuda'
    
    Returns:
        x_{t-1}: Less noisy data at previous timestep
    """
    # Predict noise using model
    predicted_noise = model(x_t, t_tensor)
    
    # Extract parameters for current timestep
    beta_t = extract(diffusion_params['betas'], t_tensor, x_t.shape)
    sqrt_recip_alpha_t = extract(diffusion_params['sqrt_recip_alphas'], t_tensor, x_t.shape)
    sqrt_one_minus_alpha_cumprod_t = extract(
        diffusion_params['sqrt_one_minus_alphas_cumprod'], t_tensor, x_t.shape
    )
    
    # Compute denoised mean: μ = 1/√α_t·(x_t - β_t/√(1-ᾱ_t)·ε_pred)
    noise_term = beta_t * predicted_noise / sqrt_one_minus_alpha_cumprod_t
    model_mean = sqrt_recip_alpha_t * (x_t - noise_term)
    
    if t == 0:
        # Final step: no noise added
        return model_mean
    else:
        # Add stochastic noise for diversity
        posterior_variance_t = extract(diffusion_params['posterior_variance'], t_tensor, x_t.shape)
        noise = torch.randn_like(x_t)
        posterior_std = torch.sqrt(posterior_variance_t)
        return model_mean + posterior_std * noise

@torch.no_grad()
def sample(model: nn.Module, shape: tuple, timesteps: int,
          diffusion_params: dict, device: str = 'cpu') -> torch.Tensor:
    """
    Generate samples via complete reverse diffusion.
    
    ALGORITHM:
    ----------
    x_T ~ N(0,I)           ← Start from pure noise
        ↓ denoise
    x_{T-1} = p_θ(· | x_T)
        ↓ ...
    x_0                    ← Final clean sample
    
    TRADE-OFFS:
    -----------
    More steps (large T): Higher quality, slower
    Fewer steps (small T): Faster, lower quality
    
    Common: T=100 (fast), T=1000 (standard), T=4000 (high quality)
    
    Args:
        model: Trained denoising model (in eval mode)
        shape: Shape of samples, e.g., (32, 2) or (16, 3, 256, 256)
        timesteps: Number of denoising steps (must match training!)
        diffusion_params: Precomputed constants
        device: 'cpu' or 'cuda'
    
    Returns:
        Generated samples matching shape
    """
    model.eval()
    
    # Start from pure noise
    x_t = torch.randn(shape, device=device)
    
    # Iterative denoising: T → T-1 → ... → 1 → 0
    for t in reversed(range(timesteps)):
        t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
        x_t = p_sample(model, x_t, t, t_tensor, diffusion_params, device)
    
    return x_t  # x_0: clean generated samples

# ============================================================================
# TRAINING LOSS
# ============================================================================

def get_loss(model: nn.Module, x_0: torch.Tensor, t: torch.Tensor,
            diffusion_params: dict, noise: torch.Tensor = None) -> torch.Tensor:
    """
    Compute training loss for diffusion model.
    
    DDPM TRAINING OBJECTIVE:
    ------------------------
    L_simple = E_{t,x_0,ε}[||ε - ε_θ(x_t,t)||²]
    
    We want model to predict EXACT noise added to create x_t.
    
    WHY PREDICT NOISE?
    ------------------
    - Noise is stationary: N(0,I) at all timesteps
    - Uniform learning objective across t
    - Highest quality samples (empirically)
    - Standard in modern diffusion models
    
    RANDOM TIMESTEP SAMPLING:
    -------------------------
    Key trick: Sample t uniformly from [0,T-1] per batch
    → Model learns to denoise at ALL noise levels
    → Prevents overfitting to specific timesteps
    
    Args:
        model: Denoising model to train
        x_0: Batch of clean data, shape (batch_size, *data_dims)
        t: Random timesteps, shape (batch_size,)
        diffusion_params: Precomputed constants
        noise: Optional pre-generated noise (if None, generate here)
    
    Returns:
        Scalar MSE loss for backpropagation
    """
    # Generate noise if not provided
    if noise is None:
        noise = torch.randn_like(x_0)
    
    # Apply forward diffusion: create x_t with known noise
    x_t = forward_diffusion(
        x_0, t, noise,
        diffusion_params['sqrt_alphas_cumprod'],
        diffusion_params['sqrt_one_minus_alphas_cumprod']
    )
    
    # Model predicts noise
    predicted_noise = model(x_t, t)
    
    # Compute MSE loss between predicted and actual noise
    return nn.functional.mse_loss(predicted_noise, noise)

# ============================================================================
# VISUALIZATION UTILITIES
# ============================================================================

def visualize_diffusion_process(x_0: torch.Tensor, timesteps: int,
                               diffusion_params: dict, num_images: int = 10):
    """
    Visualize forward diffusion on images (data → noise).
    
    Shows gradual corruption: clean → slightly noisy → very noisy → pure noise
    Helps build intuition for what model needs to reverse.
    """
    sqrt_alphas_cumprod = diffusion_params['sqrt_alphas_cumprod']
    sqrt_one_minus_alphas_cumprod = diffusion_params['sqrt_one_minus_alphas_cumprod']
    
    time_steps = np.linspace(0, timesteps - 1, num_images, dtype=int)
    fig, axes = plt.subplots(1, num_images, figsize=(15, 2))
    
    for idx, t in enumerate(time_steps):
        noise = torch.randn_like(x_0)
        t_tensor = torch.tensor([t])
        x_t = forward_diffusion(x_0, t_tensor, noise, 
                               sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
        
        # Display
        if x_0.shape[1] == 1:  # Grayscale
            img = x_t[0, 0].cpu().numpy()
            axes[idx].imshow(img, cmap='gray')
        else:  # RGB
            img = x_t[0].cpu().permute(1, 2, 0).numpy()
            img = (img - img.min()) / (img.max() - img.min())
            axes[idx].imshow(img)
        
        axes[idx].set_title(f't={t}')
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('diffusion_process.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved: diffusion_process.png")

def visualize_samples(samples: torch.Tensor, nrow: int = 8, 
                     filename: str = 'samples.png'):
    """Display generated samples in a grid."""
    from torchvision.utils import make_grid
    
    samples = (samples - samples.min()) / (samples.max() - samples.min())
    grid = make_grid(samples, nrow=nrow, padding=2)
    
    plt.figure(figsize=(12, 12))
    if samples.shape[1] == 1:
        plt.imshow(grid[0].cpu().numpy(), cmap='gray')
    else:
        plt.imshow(grid.cpu().permute(1, 2, 0).numpy())
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {filename}")

# ============================================================================
# TIME EMBEDDING (FOR ADVANCED MODELS)
# ============================================================================

class SinusoidalPositionEmbedding(nn.Module):
    """
    Sinusoidal time embedding for timestep conditioning.
    
    WHY SINUSOIDAL?
    ---------------
    Better than scalar t:
    - Continuous: Nearby timesteps have similar embeddings
    - Unique: Each timestep distinct
    - Smooth: Small Δt → small Δembedding
    - Periodic: Rich representations via sin/cos
    
    FORMULA (from Transformers):
    ----------------------------
    PE(t, 2i) = sin(t/10000^(2i/d))      even indices
    PE(t, 2i+1) = cos(t/10000^(2i/d))    odd indices
    
    Different dimensions oscillate at different frequencies.
    """
    
    def __init__(self, dim: int):
        """
        Args:
            dim: Embedding dimension (should be even)
                Common: 64, 128, 256
        """
        super().__init__()
        self.dim = dim
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Convert timesteps to sinusoidal embeddings.
        
        Args:
            time: Timesteps, shape (batch_size,)
        
        Returns:
            Embeddings, shape (batch_size, dim)
        """
        device = time.device
        half_dim = self.dim // 2
        
        # Compute frequency scaling (geometric progression)
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        
        # Compute time × frequency
        embeddings = time[:, None] * embeddings[None, :]
        
        # Apply sin and cos
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

# ============================================================================
# DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    """Compare noise schedules."""
    print("=" * 70)
    print("NOISE SCHEDULE COMPARISON")
    print("=" * 70)
    
    timesteps = 1000
    print(f"\nGenerating schedules for T={timesteps}...")
    
    betas_linear = linear_beta_schedule(timesteps)
    betas_cosine = cosine_beta_schedule(timesteps)
    
    print(f"\nLinear: [{betas_linear[0]:.6f}, {betas_linear[-1]:.6f}]")
    print(f"Cosine: [{betas_cosine[0]:.6f}, {betas_cosine[-1]:.6f}]")
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(betas_linear, label="Linear", alpha=0.7)
    axes[0].plot(betas_cosine, label="Cosine", alpha=0.7)
    axes[0].set_xlabel('Timestep t')
    axes[0].set_ylabel('β_t')
    axes[0].set_title('Full Schedule')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(betas_linear[-100:], label="Linear (end)", alpha=0.7)
    axes[1].plot(betas_cosine[-100:], label="Cosine (end)", alpha=0.7)
    axes[1].set_xlabel('Timestep t')
    axes[1].set_title('Last 100 Steps')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('noise_schedule_comparison.png', dpi=150)
    print("\n✓ Saved: noise_schedule_comparison.png")
    
    print("\nKEY DIFFERENCES:")
    print("  Linear: Uniform increase, aggressive at end")
    print("  Cosine: Smooth increase, better for images")
    print("  Recommendation: Use cosine for better quality")
    print("=" * 70)