"""
MODULE 09: Image Generation with U-Net
=====================================

DIFFICULTY: Advanced
TIME: 4-5 hours  
PREREQUISITES: Modules 01-08, CNN knowledge

LEARNING OBJECTIVES:
- Implement U-Net score architecture for images
- Train on MNIST digits
- Generate images via score-based sampling
- Understand computational considerations

Key: U-Net with time conditioning for image score modeling

Author: Sungchul @ Yonsei University
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

print("MODULE 09: Image Generation with U-Net")
print("="*80)

print("""
WHY U-NET FOR IMAGES?
--------------------
Score function s(x) must be same size as input x

For images:
- Input: [B, C, H, W]
- Output: [B, C, H, W] (score for each pixel/channel)

U-Net architecture:
1. Encoder: Downsample + extract features
2. Decoder: Upsample + generate output
3. Skip connections: Preserve spatial information
4. Time conditioning: Different scores at different noise levels

SIMPLIFIED U-NET FOR MNIST:
--------------------------
""")

class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings

class SimpleUNet(nn.Module):
    """
    Simplified U-Net for 28x28 MNIST images
    
    Architecture:
    - Encoder: 28 → 14 → 7
    - Decoder: 7 → 14 → 28
    - Time conditioning at each level
    """
    def __init__(self, channels=[1, 32, 64, 128], time_dim=128):
        super().__init__()
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            TimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU()
        )
        
        # Encoder (downsampling)
        self.enc1 = nn.Conv2d(channels[0], channels[1], 3, padding=1)
        self.enc2 = nn.Conv2d(channels[1], channels[2], 3, padding=1, stride=2)  # 28→14
        self.enc3 = nn.Conv2d(channels[2], channels[3], 3, padding=1, stride=2)  # 14→7
        
        # Middle
        self.mid = nn.Conv2d(channels[3], channels[3], 3, padding=1)
        
        # Decoder (upsampling)
        self.dec3 = nn.ConvTranspose2d(channels[3], channels[2], 4, stride=2, padding=1)  # 7→14
        self.dec2 = nn.ConvTranspose2d(channels[2]*2, channels[1], 4, stride=2, padding=1)  # 14→28
        self.dec1 = nn.Conv2d(channels[1]*2, channels[0], 3, padding=1)
        
        # Time projections
        self.time_proj1 = nn.Linear(time_dim, channels[1])
        self.time_proj2 = nn.Linear(time_dim, channels[2])
        self.time_proj3 = nn.Linear(time_dim, channels[3])
    
    def forward(self, x, t):
        """
        Args:
            x: Image [B, 1, 28, 28]
            t: Time [B]
        Returns:
            score: [B, 1, 28, 28]
        """
        # Time embedding
        t_emb = self.time_mlp(t)
        
        # Encoder
        h1 = F.silu(self.enc1(x) + self.time_proj1(t_emb)[:, :, None, None])
        h2 = F.silu(self.enc2(h1) + self.time_proj2(t_emb)[:, :, None, None])
        h3 = F.silu(self.enc3(h2) + self.time_proj3(t_emb)[:, :, None, None])
        
        # Middle
        h = F.silu(self.mid(h3))
        
        # Decoder with skip connections
        h = F.silu(self.dec3(h))
        h = torch.cat([h, h2], dim=1)  # Skip connection
        
        h = F.silu(self.dec2(h))
        h = torch.cat([h, h1], dim=1)  # Skip connection
        
        h = self.dec1(h)
        
        return h

print("U-Net architecture defined!")
print("""
KEY COMPONENTS:
--------------
1. Time embedding: Inform network of noise level
2. Skip connections: Preserve spatial details
3. Residual blocks: Easier optimization
4. Group normalization: Better than batch norm for generative models
5. Attention (optional): For high-res images

TRAINING STRATEGY:
-----------------
1. Sample image x ~ p_data
2. Sample noise level t ~ Uniform[0, T]
3. Add noise: x_t = √ᾱ_t x + √(1-ᾱ_t) ε
4. Predict noise: ε_θ(x_t, t)
5. Loss: ||ε - ε_θ(x_t, t)||²

This is DSM in disguise!
Score s(x_t, t) = -ε_θ(x_t, t) / √(1-ᾱ_t)
""")

# Simplified training loop (conceptual)
print("\nConceptual training code:")
print("-" * 80)

training_code = """
def train_score_model_mnist(model, dataloader, n_timesteps=1000, epochs=10):
    '''Train score model on MNIST'''
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Linear noise schedule
    betas = torch.linspace(0.0001, 0.02, n_timesteps)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    for epoch in range(epochs):
        for images, _ in dataloader:
            # Random timestep
            t = torch.randint(0, n_timesteps, (images.shape[0],))
            
            # Add noise
            noise = torch.randn_like(images)
            sqrt_alpha_bar = alphas_cumprod[t].sqrt()[:, None, None, None]
            sqrt_one_minus_alpha_bar = (1 - alphas_cumprod[t]).sqrt()[:, None, None, None]
            noisy_images = sqrt_alpha_bar * images + sqrt_one_minus_alpha_bar * noise
            
            # Predict noise (equivalent to predicting score)
            predicted_noise = model(noisy_images, t)
            
            # Loss
            loss = F.mse_loss(predicted_noise, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
    
    return model

# Sampling
@torch.no_grad()
def generate_images(model, n_samples=64, n_timesteps=1000):
    '''Generate images via reverse diffusion'''
    # Start from noise
    x = torch.randn(n_samples, 1, 28, 28)
    
    for t in reversed(range(n_timesteps)):
        t_tensor = torch.ones(n_samples, dtype=torch.long) * t
        
        # Predict noise
        predicted_noise = model(x, t_tensor)
        
        # Compute score
        alpha_t = alphas[t]
        alpha_bar_t = alphas_cumprod[t]
        
        # Denoise
        beta_t = betas[t]
        x = (1 / alpha_t.sqrt()) * (x - beta_t / (1 - alpha_bar_t).sqrt() * predicted_noise)
        
        # Add noise (except last step)
        if t > 0:
            x = x + beta_t.sqrt() * torch.randn_like(x)
    
    return x
"""

print(training_code)

print("""
PRACTICAL CONSIDERATIONS:
------------------------

COMPUTE REQUIREMENTS:
- MNIST: ~2-4 hours on GPU
- CIFAR-10: ~1-2 days on GPU  
- ImageNet: ~1 week on multiple GPUs

MEMORY OPTIMIZATION:
- Gradient checkpointing
- Mixed precision (FP16)
- Batch size tuning

SAMPLING SPEED:
- Standard: 1000 steps (~10s per image)
- DDIM: 50 steps (~0.5s per image)
- DPM-Solver: 20 steps (~0.2s per image)
- Consistency models: 1 step! (future topic)

QUALITY METRICS:
- FID (Fréchet Inception Distance)
- Inception Score
- Precision/Recall
- Human evaluation

TYPICAL RESULTS:
- MNIST: FID ~5-10 (excellent)
- CIFAR-10: FID ~3-10 (SOTA)
- ImageNet 256x256: FID ~2-5 (SOTA)

CONNECTION TO WHAT WE'VE LEARNED:
--------------------------------
✓ Score functions (Module 01) → Noise prediction
✓ DSM (Module 02) → Training objective  
✓ Langevin (Module 03) → Sampling procedure
✓ Multi-scale (Module 07) → Time conditioning
✓ SDE (Module 08) → Continuous formulation

Everything connects!
""")

print("\n✓ Module 09 complete!")
print("Final module: Complete unification with diffusion models!")
