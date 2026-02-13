# ============================================================================
# simple_diffusion.py - 2D Diffusion Model (Educational Implementation)
# ============================================================================

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from diffusion_utils import (
    cosine_beta_schedule,
    get_diffusion_parameters,
    forward_diffusion,
    get_loss
)
from tqdm import tqdm

# ============================================================================
# CORE CONCEPT: DIFFUSION MODELS
# ============================================================================

"""
WHAT IS A DIFFUSION MODEL?
===========================

Diffusion models learn to generate data by reversing a gradual noising process:

1. FORWARD PROCESS (Easy, no learning needed):
   Take real data → gradually add noise → pure random noise
   
2. REVERSE PROCESS (Hard, requires training):
   Take random noise → gradually remove noise → realistic data

KEY INSIGHT: If we learn to denoise at ANY noise level, we can start from 
pure noise and denoise our way back to realistic data!

WHY 2D TOY DATA?
================
Before tackling high-dimensional images (256×256×3 = 196,608 dims), we use
2D data (just x, y) for clear visualization and intuition building.
"""

# ============================================================================
# MATHEMATICAL FOUNDATION: CLOSED-FORM FORWARD DIFFUSION
# ============================================================================

"""
THE FORWARD DIFFUSION EQUATION (Full Derivation)
=================================================

ITERATIVE FORM (one step at a time):
    x_t = √α_t · x_{t-1} + √β_t · ε_t,  where ε_t ~ N(0, I)
    
    α_t = 1 - β_t (signal retention)
    β_t = noise variance at step t

THE MAGIC: CLOSED-FORM (jump to any step directly!)
====================================================

After t steps of recursion, all the noise terms combine into:

    ╔══════════════════════════════════════════════════════════╗
    ║  x_t = √ᾱ_t · x_0 + √(1 - ᾱ_t) · ε,  where ε ~ N(0, I)   ║
    ╚══════════════════════════════════════════════════════════╝

Where:
    ᾱ_t = ∏_{i=1}^t α_i  (cumulative product of all alphas)

INTUITION:
----------
    √ᾱ_t: How much original signal remains (decreases as t increases)
    √(1-ᾱ_t): How much noise is present (increases as t increases)
    
    At t=0:   ᾱ_0 = 1    → x_0 = 1·x_0 + 0·ε = x_0 (no noise)
    At t=100: ᾱ_100 ≈ 0.37 → x_100 = 0.6·x_0 + 0.8·ε (mostly noise)
    At t=∞:   ᾱ_∞ → 0    → x_∞ ≈ 0·x_0 + 1·ε = ε (pure noise)

WHY THIS IS AMAZING:
--------------------
1. EFFICIENCY: O(1) instead of O(t) - jump directly to any timestep
2. PARALLEL TRAINING: Sample different timesteps independently per batch
3. MATHEMATICAL ELEGANCE: Complex iteration reduces to simple weighted sum

QUICK REFERENCE:
----------------
Symbol  | Meaning                              | Typical Range
--------|--------------------------------------|---------------
β_t     | Noise added at step t                | 0.0001 to 0.02
α_t     | Signal retained (= 1 - β_t)          | 0.98 to 0.9999
ᾱ_t     | Cumulative signal (= ∏ α_i)          | 1.0 → 0 (decay)
√ᾱ_t    | Data coefficient in forward process  | 1.0 → 0
√(1-ᾱ_t)| Noise coefficient in forward process | 0 → 1.0
"""

# ============================================================================
# MATHEMATICAL DERIVATION: REVERSE DIFFUSION FORMULA
# ============================================================================

"""
PROOF: REVERSE DIFFUSION MEAN FORMULA
======================================

We want to prove:

    μ_θ(x_t, t) = 1/√α_t · (x_t - β_t/√(1-ᾱ_t) · ε_θ(x_t, t))

This formula tells us how to compute the mean for denoising: x_t → x_{t-1}

============================================================================
PART 1: THE TRUE POSTERIOR (If We Knew x_0)
============================================================================

By Bayes' rule, the true reverse distribution is:

    q(x_{t-1} | x_t, x_0) = q(x_t | x_{t-1}, x_0) · q(x_{t-1} | x_0) / q(x_t | x_0)

By Markov property: q(x_t | x_{t-1}, x_0) = q(x_t | x_{t-1})

So:
    q(x_{t-1} | x_t, x_0) ∝ q(x_t | x_{t-1}) · q(x_{t-1} | x_0)

This is a product of Gaussians → also Gaussian!

============================================================================
STEP 1: Recall Forward Diffusion Equations
============================================================================

From forward diffusion, we have:

(A) One-step forward:
    q(x_t | x_{t-1}) = N(x_t; √α_t · x_{t-1}, β_t · I)
    
    Density:
        p(x_t | x_{t-1}) ∝ exp(-1/(2β_t) ||x_t - √α_t · x_{t-1}||²)

(B) Multi-step to x_t from x_0:
    q(x_t | x_0) = N(x_t; √ᾱ_t · x_0, (1-ᾱ_t) · I)
    
    Density:
        p(x_t | x_0) ∝ exp(-1/(2(1-ᾱ_t)) ||x_t - √ᾱ_t · x_0||²)

(C) Multi-step to x_{t-1} from x_0:
    q(x_{t-1} | x_0) = N(x_{t-1}; √ᾱ_{t-1} · x_0, (1-ᾱ_{t-1}) · I)
    
    Density:
        p(x_{t-1} | x_0) ∝ exp(-1/(2(1-ᾱ_{t-1})) ||x_{t-1} - √ᾱ_{t-1} · x_0||²)

============================================================================
STEP 2: Compute the Posterior q(x_{t-1} | x_t, x_0)
============================================================================

Combine the densities:

    q(x_{t-1} | x_t, x_0) ∝ q(x_t | x_{t-1}) · q(x_{t-1} | x_0)

    ∝ exp(-1/(2β_t) ||x_t - √α_t·x_{t-1}||²) 
      · exp(-1/(2(1-ᾱ_{t-1})) ||x_{t-1} - √ᾱ_{t-1}·x_0||²)

Expand the quadratics:

    ||x_t - √α_t·x_{t-1}||² = ||x_t||² - 2√α_t·⟨x_t, x_{t-1}⟩ + α_t||x_{t-1}||²

    ||x_{t-1} - √ᾱ_{t-1}·x_0||² = ||x_{t-1}||² - 2√ᾱ_{t-1}·⟨x_{t-1}, x_0⟩ + ᾱ_{t-1}||x_0||²

The exponent is quadratic in x_{t-1}:

    E(x_{t-1}) = A||x_{t-1}||² - 2⟨B, x_{t-1}⟩ + C

Where:
    A = α_t/(2β_t) + 1/(2(1-ᾱ_{t-1}))
    B = √α_t·x_t/β_t + √ᾱ_{t-1}·x_0/(1-ᾱ_{t-1})
    C = (constant terms not involving x_{t-1})

Complete the square:
    E(x_{t-1}) = A||x_{t-1} - B/A||² + (constant)

So: q(x_{t-1} | x_t, x_0) = N(x_{t-1}; B/A, 1/(2A)·I)

The mean is: μ̃_t(x_t, x_0) = B/A

============================================================================
STEP 3: Simplify B/A to Get the Mean
============================================================================

COMPUTE A:
    A = α_t/(2β_t) + 1/(2(1-ᾱ_{t-1}))
    
    = [α_t(1-ᾱ_{t-1}) + β_t] / [2β_t(1-ᾱ_{t-1})]

Simplify numerator:
    α_t(1-ᾱ_{t-1}) + β_t = (1-β_t)(1-ᾱ_{t-1}) + β_t
                          = 1 - ᾱ_{t-1} - β_t + β_t·ᾱ_{t-1} + β_t
                          = 1 - ᾱ_{t-1} + β_t·ᾱ_{t-1}

Note: ᾱ_t = ᾱ_{t-1}·α_t = ᾱ_{t-1}·(1-β_t)

So: 1 - ᾱ_{t-1} + β_t·ᾱ_{t-1} = 1 - ᾱ_{t-1}(1-β_t)
                                = 1 - ᾱ_{t-1}·α_t
                                = 1 - ᾱ_t

Therefore:
    A = (1-ᾱ_t) / [2β_t(1-ᾱ_{t-1})]

COMPUTE B:
    B = √α_t·x_t/β_t + √ᾱ_{t-1}·x_0/(1-ᾱ_{t-1})

COMPUTE μ̃_t = B/A:
    μ̃_t(x_t, x_0) = [√α_t·x_t/β_t + √ᾱ_{t-1}·x_0/(1-ᾱ_{t-1})] / [(1-ᾱ_t)/(2β_t(1-ᾱ_{t-1}))]
    
    = [√α_t·x_t/β_t + √ᾱ_{t-1}·x_0/(1-ᾱ_{t-1})] · [2β_t(1-ᾱ_{t-1})/(1-ᾱ_t)]

Distribute:
    = 2β_t(1-ᾱ_{t-1})/(1-ᾱ_t) · √α_t·x_t/β_t 
      + 2β_t(1-ᾱ_{t-1})/(1-ᾱ_t) · √ᾱ_{t-1}·x_0/(1-ᾱ_{t-1})
    
    = 2(1-ᾱ_{t-1})/(1-ᾱ_t) · √α_t·x_t 
      + 2β_t/(1-ᾱ_t) · √ᾱ_{t-1}·x_0

Rearranging:
    μ̃_t(x_t, x_0) = √ᾱ_{t-1}·β_t/(1-ᾱ_t)·x_0 + √α_t·(1-ᾱ_{t-1})/(1-ᾱ_t)·x_t

This is the TRUE posterior mean (if we knew x_0).

============================================================================
STEP 4: Estimate x_0 from x_t (THE KEY TRICK!)
============================================================================

THE PROBLEM: During generation, we don't know x_0!

THE SOLUTION: Use forward diffusion to estimate x_0.

From: x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε

Solve for x_0:
    √ᾱ_t·x_0 = x_t - √(1-ᾱ_t)·ε
    
    x_0 = (x_t - √(1-ᾱ_t)·ε) / √ᾱ_t    ... (★)

Use model's prediction ε_θ(x_t, t):

    x̂_0 = (x_t - √(1-ᾱ_t)·ε_θ(x_t,t)) / √ᾱ_t    ... (★★)

============================================================================
STEP 5: Substitute x̂_0 into Posterior Mean
============================================================================

Replace x_0 with x̂_0 in:
    μ̃_t(x_t, x_0) = √ᾱ_{t-1}·β_t/(1-ᾱ_t)·x_0 + √α_t·(1-ᾱ_{t-1})/(1-ᾱ_t)·x_t

Becomes:
    μ_θ(x_t,t) = √ᾱ_{t-1}·β_t/(1-ᾱ_t)·x̂_0 + √α_t·(1-ᾱ_{t-1})/(1-ᾱ_t)·x_t

Plug in x̂_0:
    μ_θ = √ᾱ_{t-1}·β_t/(1-ᾱ_t) · [(x_t - √(1-ᾱ_t)·ε_θ)/√ᾱ_t] 
          + √α_t·(1-ᾱ_{t-1})/(1-ᾱ_t)·x_t

Expand first term:
    = [√ᾱ_{t-1}/(√ᾱ_t)] · [β_t/(1-ᾱ_t)] · [x_t - √(1-ᾱ_t)·ε_θ]
    + √α_t·(1-ᾱ_{t-1})/(1-ᾱ_t)·x_t

Note: ᾱ_t = ᾱ_{t-1}·α_t, so √ᾱ_{t-1}/√ᾱ_t = 1/√α_t

    = [1/√α_t] · [β_t/(1-ᾱ_t)] · [x_t - √(1-ᾱ_t)·ε_θ]
    + √α_t·(1-ᾱ_{t-1})/(1-ᾱ_t)·x_t

    = [β_t/((1-ᾱ_t)√α_t)]·x_t - [β_t/(√α_t·√(1-ᾱ_t))]·ε_θ
    + [√α_t·(1-ᾱ_{t-1})/(1-ᾱ_t)]·x_t

Combine x_t terms:
    x_t coefficient = β_t/((1-ᾱ_t)√α_t) + √α_t·(1-ᾱ_{t-1})/(1-ᾱ_t)
                    = [1/((1-ᾱ_t)√α_t)] · [β_t + α_t(1-ᾱ_{t-1})]

Key identity: 1 - ᾱ_t = 1 - α_t·ᾱ_{t-1} = (1-α_t) + α_t(1-ᾱ_{t-1})
                       = β_t + α_t(1-ᾱ_{t-1})

So: x_t coefficient = [1/((1-ᾱ_t)√α_t)] · (1-ᾱ_t) = 1/√α_t

Final result:
    μ_θ(x_t,t) = x_t/√α_t - [β_t/(√α_t·√(1-ᾱ_t))]·ε_θ(x_t,t)

Factor out 1/√α_t:

    ╔═══════════════════════════════════════════════════════════╗
    ║  μ_θ(x_t,t) = 1/√α_t · (x_t - β_t/√(1-ᾱ_t)·ε_θ(x_t,t))    ║
    ╚═══════════════════════════════════════════════════════════╝

✓✓✓ Q.E.D. ✓✓✓

This is the formula we use for reverse diffusion!

============================================================================
INTUITIVE INTERPRETATION
============================================================================

μ_θ(x_t,t) = 1/√α_t · (x_t - β_t/√(1-ᾱ_t)·ε_θ(x_t,t))
             └──┬──┘   └─────────┬─────────┘
              scale    remove predicted noise

1. Model predicts noise: ε_θ(x_t,t)
2. Scale the noise: β_t/√(1-ᾱ_t)·ε_θ
3. Remove from x_t: x_t - (scaled noise)
4. Rescale result: multiply by 1/√α_t

Result: Our best estimate of x_{t-1}!

============================================================================
"""

# ============================================================================
# MODEL: DENOISING NEURAL NETWORK
# ============================================================================

class Simple2DModel(nn.Module):
    """
    The core denoising network that predicts noise in data at any timestep.
    
    ARCHITECTURE:
    -------------
    Input: [noisy_data (2D), timestep_embedding (64D)]
    Output: predicted_noise (2D)
    
    The model learns: ε_θ(x_t, t) ≈ ε
    
    WHY PREDICT NOISE (not clean data)?
    -----------------------------------
    1. Noise is more uniform across data types
    2. Consistent objective at all timesteps
    3. Empirically better sample quality
    
    CONDITIONING ON TIME:
    ---------------------
    Different timesteps need different strategies:
    - Early (t small): Data barely noisy → small corrections
    - Late (t large): Data very noisy → aggressive denoising
    
    We embed timesteps into 64D vectors so the network can learn
    time-dependent denoising patterns.
    """
    
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        
        # Time embedding: scalar timestep → 64D representation
        self.time_embed = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),  # SiLU(x) = x * sigmoid(x)
            nn.Linear(64, 64),
        )
        
        # Main denoising network: [2D data + 64D time] → 2D noise
        self.network = nn.Sequential(
            nn.Linear(2 + 64, hidden_dim),  # 66 → 128
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),  # 128 → 2 (noise vector)
        )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predict noise in data at given timestep.
        
        Args:
            x: Noisy 2D points, shape (batch_size, 2)
            t: Timestep indices, shape (batch_size,)
        
        Returns:
            predicted_noise: shape (batch_size, 2)
        """
        # Normalize timesteps to [0, 1] range for better learning
        t_normalized = t.float().unsqueeze(-1) / 1000.0
        
        # Embed timesteps into rich representations
        t_emb = self.time_embed(t_normalized)  # (batch, 1) → (batch, 64)
        
        # Concatenate data with time embeddings
        h = torch.cat([x, t_emb], dim=-1)  # (batch, 66)
        
        # Predict noise
        return self.network(h)  # (batch, 2)

# ============================================================================
# TOY DATASETS
# ============================================================================

def generate_swiss_roll(n_samples: int = 1000) -> torch.Tensor:
    """
    Generate a 2D Swiss roll spiral dataset.
    
    A spiral-shaped point cloud perfect for testing:
    - Simple enough to visualize (2D)
    - Complex enough to be interesting (non-linear)
    - Tests if model can learn curved manifolds
    
    Math: x = θ·cos(θ), y = θ·sin(θ), θ ∈ [0, 4π]
    """
    t = torch.linspace(0, 4 * np.pi, n_samples)
    x = t * torch.cos(t)
    y = t * torch.sin(t)
    data = torch.stack([x, y], dim=1)
    return data + torch.randn_like(data) * 0.1  # Add small noise

def generate_moons(n_samples: int = 1000) -> torch.Tensor:
    """
    Generate two interleaving crescent moon shapes.
    Tests: multiple modes, curved shapes, close but separate structures.
    """
    from sklearn.datasets import make_moons
    data, _ = make_moons(n_samples=n_samples, noise=0.05)
    return torch.tensor(data, dtype=torch.float32)

# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_2d_diffusion(data, timesteps, diffusion_params, num_steps=10):
    """
    Visualize how data gradually becomes noise (forward diffusion).
    
    Shows 10 snapshots: clean data → partially noisy → pure noise
    Helps build intuition: this is what we need to learn to reverse!
    """
    sqrt_alphas_cumprod = diffusion_params['sqrt_alphas_cumprod']
    sqrt_one_minus_alphas_cumprod = diffusion_params['sqrt_one_minus_alphas_cumprod']
    
    time_steps = np.linspace(0, timesteps - 1, num_steps, dtype=int)
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()

    for idx, t in enumerate(time_steps):
        # Apply closed-form forward diffusion: x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε
        noise = torch.randn_like(data)
        t_tensor = torch.full((data.shape[0],), t)
        x_t = forward_diffusion(data, t_tensor, noise, 
                               sqrt_alphas_cumprod, 
                               sqrt_one_minus_alphas_cumprod)
        
        axes[idx].scatter(x_t[:, 0].numpy(), x_t[:, 1].numpy(), 
                         alpha=0.5, s=1)
        axes[idx].set_title(f't = {t}')
        axes[idx].set_xlim(-15, 15)
        axes[idx].set_ylim(-15, 15)
        axes[idx].set_aspect('equal')

    plt.tight_layout()
    plt.savefig('2d_forward_diffusion.png', dpi=150)
    plt.close()
    print("✓ Saved forward diffusion visualization")

# ============================================================================
# TRAINING
# ============================================================================

def train_2d_diffusion(data, timesteps=100, epochs=1000, batch_size=128):
    """
    Train the model to predict noise at any timestep.
    
    TRAINING OBJECTIVE:
    -------------------
    For each iteration:
    1. Sample batch of clean data x_0
    2. Sample random timesteps t
    3. Add noise: x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε
    4. Predict noise: ε_pred = model(x_t, t)
    5. Loss: MSE(ε_pred, ε)
    6. Backprop and update
    
    By learning to predict noise at ALL timesteps, the model implicitly
    learns how to denoise. During generation, we use this repeatedly to
    convert noise → data.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")

    # Setup diffusion parameters
    betas = cosine_beta_schedule(timesteps)
    diffusion_params = get_diffusion_parameters(betas)
    for key in diffusion_params:
        diffusion_params[key] = diffusion_params[key].to(device)
    data = data.to(device)

    # Initialize model and optimizer
    model = Simple2DModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    pbar = tqdm(range(epochs))

    for epoch in pbar:
        # Sample random batch
        indices = torch.randint(0, len(data), (batch_size,))
        x_0 = data[indices]
        
        # Sample random timesteps
        t = torch.randint(0, timesteps, (batch_size,), device=device)
        
        # Compute loss (handles forward diffusion + noise prediction)
        loss = get_loss(model, x_0, t, diffusion_params)
        
        # Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        pbar.set_description(f"Loss: {loss.item():.4f}")

    # Plot training progress
    plt.figure(figsize=(10, 4))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig('2d_training_loss.png', dpi=150)
    plt.close()
    print("✓ Saved training loss plot")

    return model, diffusion_params

# ============================================================================
# SAMPLING (REVERSE DIFFUSION)
# ============================================================================

@torch.no_grad()
def sample_2d(model, n_samples, timesteps, diffusion_params, device="cpu"):
    """
    Generate new samples via reverse diffusion (noise → data).
    
    ALGORITHM:
    ----------
    1. Start: x_T ~ N(0, I) [pure random noise]
    2. For t = T, T-1, ..., 1:
           ε_pred = model(x_t, t)                      # Predict noise
           μ_θ = 1/√α_t · (x_t - β_t/√(1-ᾱ_t)·ε_pred)  # Compute mean
           x_{t-1} = μ_θ + σ_t·z  (z ~ N(0,I))         # Add small noise
    3. End: x_0 [clean generated sample]
    
    THE REVERSE DIFFUSION FORMULA:
    -------------------------------
        x_{t-1} = 1/√α_t · (x_t - β_t/√(1-ᾱ_t)·ε_θ(x_t,t)) + σ_t·z
                  └─────────────── μ_θ(x_t,t) ───────────┘  └─noise─┘
    
    BREAKDOWN:
    ----------
    1. ε_θ(x_t,t): Model predicts noise in x_t
    2. β_t/√(1-ᾱ_t)·ε_θ: Scaled noise to remove
    3. x_t - [noise]: Remove estimated noise
    4. 1/√α_t · [...]: Rescale for proper variance
    5. σ_t·z: Add small controlled noise for diversity
    
    WHY ADD NOISE WHEN DENOISING?
    ------------------------------
    - Without it: deterministic path, no diversity, mode collapse
    - With it: stochastic paths, diverse samples, full distribution
    - Key: noise added < noise removed, so net progress toward clean data!
    
    SPECIAL CASE:
    -------------
    At t=0 (final step), NO noise added: x_0 = μ_θ(x_1, 0)
    This gives our final clean generated sample.
    
    EXAMPLE (one step):
    -------------------
    x_100 = [0.5, -0.3] (noisy)
    ε_pred = [0.4, -0.2] (model's prediction)
    μ_θ = [0.489, -0.295] (denoised mean)
    z = [0.01, 0.02] (random noise)
    x_99 = μ_θ + 0.05·z = [0.4905, -0.294] (slightly cleaner!)
    """
    model.eval()
    
    # Start from pure noise
    x_t = torch.randn(n_samples, 2, device=device)
    
    # Iteratively denoise: T → T-1 → ... → 1 → 0
    for t in reversed(range(timesteps)):
        t_tensor = torch.full((n_samples,), t, device=device, dtype=torch.long)
        
        # Predict noise
        predicted_noise = model(x_t, t_tensor)
        
        # Get parameters for this timestep
        beta_t = diffusion_params['betas'][t]
        sqrt_recip_alpha_t = diffusion_params['sqrt_recip_alphas'][t]
        sqrt_one_minus_alpha_cumprod_t = diffusion_params['sqrt_one_minus_alphas_cumprod'][t]
        
        # Compute denoised mean: μ_θ = 1/√α_t·(x_t - β_t/√(1-ᾱ_t)·ε_θ)
        mean = sqrt_recip_alpha_t * (
            x_t - (beta_t / sqrt_one_minus_alpha_cumprod_t) * predicted_noise
        )
        
        if t > 0:
            # Add controlled noise for diversity
            noise = torch.randn_like(x_t)
            sigma = torch.sqrt(diffusion_params['posterior_variance'][t])
            x_t = mean + sigma * noise
        else:
            # Final step: no noise added
            x_t = mean
    
    return x_t  # x_0: clean generated samples

def visualize_results(original_data, generated_data):
    """Compare original training data with generated samples."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].scatter(original_data[:, 0].cpu().numpy(), 
                   original_data[:, 1].cpu().numpy(), 
                   alpha=0.5, s=10)
    axes[0].set_title('Original Data')
    axes[0].set_aspect('equal')
    
    axes[1].scatter(generated_data[:, 0].cpu().numpy(), 
                   generated_data[:, 1].cpu().numpy(), 
                   alpha=0.5, s=10, color='red')
    axes[1].set_title('Generated Data')
    axes[1].set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('2d_comparison.png', dpi=150)
    plt.close()
    print("✓ Saved comparison plot")

# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main():
    """
    Complete 2D diffusion demonstration.
    
    PIPELINE:
    ---------
    1. Generate toy 2D dataset (Swiss roll)
    2. Visualize forward diffusion (data → noise)
    3. Train model to predict noise
    4. Generate new samples (noise → data)
    5. Compare original vs generated
    """
    print("=" * 50)
    print("Simple 2D Diffusion Model Demo")
    print("=" * 50)

    # 1. Generate data
    print("\n1. Generating Swiss roll data...")
    data = generate_swiss_roll(n_samples=2000)
    print(f"   ✓ Generated {len(data)} points")

    # 2. Visualize forward diffusion
    print("\n2. Visualizing forward diffusion...")
    timesteps = 100
    betas = cosine_beta_schedule(timesteps)
    diffusion_params = get_diffusion_parameters(betas)
    visualize_2d_diffusion(data, timesteps, diffusion_params)

    # 3. Train model
    print("\n3. Training diffusion model...")
    model, diffusion_params = train_2d_diffusion(
        data, timesteps=timesteps, epochs=2000, batch_size=256
    )
    print("   ✓ Training complete")

    # 4. Generate samples
    print("\n4. Generating samples...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generated = sample_2d(model, 2000, timesteps, diffusion_params, device)
    print(f"   ✓ Generated {len(generated)} samples")

    # 5. Compare results
    print("\n5. Comparing results...")
    visualize_results(data, generated)

    print("\n" + "=" * 50)
    print("✓ Demo complete! Check generated images:")
    print("  • 2d_forward_diffusion.png - Data dissolving into noise")
    print("  • 2d_training_loss.png - Training progress")
    print("  • 2d_comparison.png - Original vs Generated")
    print("=" * 50)
    
    print("\nKEY TAKEAWAYS:")
    print("  ✓ Forward: Add noise progressively (no learning)")
    print("  ✓ Reverse: Remove noise progressively (learned)")
    print("  ✓ Model predicts noise at any timestep")
    print("  ✓ Generation: Noise → iterative denoising → data")
    print("=" * 50)

if __name__ == "__main__":
    main()