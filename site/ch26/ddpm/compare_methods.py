# ==========================================
# Diffusion Methods Comparison Demo
# Compare DDPM vs DDIM sampling speed
# ==========================================
import torch
import torch.nn as nn
import torchvision
import time
from tqdm import tqdm

# Simple U-Net for comparison
class SimpleUNet(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(1, channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv3 = nn.Conv2d(channels, 1, 3, 1, 1)

    def forward(self, x, t):
        import torch.nn.functional as F
        t_embed = t[:, None, None, None].float() / 1000.0
        h = F.relu(self.conv1(x) + t_embed)
        h = F.relu(self.conv2(h))
        return self.conv3(h)

# Schedules
def get_cosine_schedule(T):
    import math
    steps = T + 1
    x = torch.linspace(0, T, steps)
    alphas_cumprod = torch.cos(((x / T) + 0.008) / (1 + 0.008) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 1e-8, 0.999)

# DDPM sampling
@torch.no_grad()
def ddpm_sample(model, T, betas, device, num_samples=1):
    model.eval()
    x = torch.randn(num_samples, 1, 28, 28).to(device)
    
    for t in tqdm(reversed(range(T)), desc="DDPM Sampling", total=T):
        z = torch.randn_like(x) if t > 0 else 0
        beta_t = betas[t]
        alpha_t = 1 - beta_t
        alpha_hat_t = torch.cumprod(1 - betas, dim=0)[t]
        
        eps_theta = model(x, torch.tensor([t]*x.size(0), device=device))
        
        x = (1 / torch.sqrt(alpha_t)) * (x - beta_t / torch.sqrt(1 - alpha_hat_t) * eps_theta) + torch.sqrt(beta_t) * z
    
    return x

# DDIM sampling
@torch.no_grad()
def ddim_sample(model, T, betas, device, ddim_steps=50, num_samples=1):
    model.eval()
    
    # Create subsequence
    c = T // ddim_steps
    ddim_timesteps = torch.arange(0, T, c).to(device)
    ddim_timesteps_prev = torch.cat([torch.tensor([0]).to(device), ddim_timesteps[:-1]])
    
    alphas_cumprod = torch.cumprod(1 - betas, dim=0)
    
    x = torch.randn(num_samples, 1, 28, 28).to(device)
    
    for i in tqdm(reversed(range(len(ddim_timesteps))), desc="DDIM Sampling", total=len(ddim_timesteps)):
        t = ddim_timesteps[i].item()
        t_prev = ddim_timesteps_prev[i].item()
        
        # Get alpha values
        alpha_t = alphas_cumprod[t]
        alpha_prev = alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0).to(device)
        
        # Predict noise
        eps = model(x, torch.tensor([t]*x.size(0), device=device))
        
        # Predict x0
        pred_x0 = (x - torch.sqrt(1 - alpha_t) * eps) / torch.sqrt(alpha_t)
        pred_x0 = pred_x0.clamp(-1, 1)
        
        # Direction
        dir_xt = torch.sqrt(1 - alpha_prev) * eps
        
        # Update
        x = torch.sqrt(alpha_prev) * pred_x0 + dir_xt
    
    return x

def main():
    print("=" * 70)
    print("🔬 DIFFUSION METHODS COMPARISON")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n📍 Using device: {device}")
    
    # Setup
    T = 1000
    DDIM_STEPS = 50
    model = SimpleUNet().to(device)
    betas = get_cosine_schedule(T).to(device)
    
    print(f"\n📊 Configuration:")
    print(f"   Total timesteps: {T}")
    print(f"   DDIM steps: {DDIM_STEPS}")
    print(f"   Speedup factor: {T / DDIM_STEPS}x")
    
    # Comparison
    print("\n" + "=" * 70)
    print("Method 1: DDPM (Original)")
    print("=" * 70)
    start = time.time()
    ddpm_samples = ddpm_sample(model, T, betas, device, num_samples=4)
    ddpm_time = time.time() - start
    print(f"⏱️  Time: {ddpm_time:.2f}s")
    
    print("\n" + "=" * 70)
    print("Method 2: DDIM (Fast Sampling)")
    print("=" * 70)
    start = time.time()
    ddim_samples = ddim_sample(model, T, betas, device, ddim_steps=DDIM_STEPS, num_samples=4)
    ddim_time = time.time() - start
    print(f"⏱️  Time: {ddim_time:.2f}s")
    
    # Results
    print("\n" + "=" * 70)
    print("📈 RESULTS")
    print("=" * 70)
    print(f"\n{'Method':<20} {'Time':<15} {'Steps':<15} {'Speed'}")
    print("-" * 70)
    print(f"{'DDPM':<20} {ddpm_time:>6.2f}s{'':<8} {T:<15} {'1.0x (baseline)'}")
    print(f"{'DDIM':<20} {ddim_time:>6.2f}s{'':<8} {DDIM_STEPS:<15} {f'{ddpm_time/ddim_time:.1f}x faster!'}")
    
    speedup = ddpm_time / ddim_time
    print(f"\n🚀 DDIM is {speedup:.1f}x faster than DDPM!")
    print(f"   DDPM: {T} denoising steps")
    print(f"   DDIM: {DDIM_STEPS} denoising steps")
    
    # Save comparison
    ddpm_grid = (ddpm_samples + 1) * 0.5
    ddim_grid = (ddim_samples + 1) * 0.5
    
    comparison = torch.cat([ddpm_grid, ddim_grid], dim=0)
    torchvision.utils.save_image(comparison, "comparison_ddpm_vs_ddim.png", nrow=4)
    
    print(f"\n💾 Saved comparison to: comparison_ddpm_vs_ddim.png")
    print(f"   Top row: DDPM samples")
    print(f"   Bottom row: DDIM samples")
    
    print("\n" + "=" * 70)
    print("💡 KEY TAKEAWAYS")
    print("=" * 70)
    print("""
    1. DDPM: Slow but original method
       - Uses all 1000 timesteps
       - Adds random noise at each step
       - High quality but time-consuming
    
    2. DDIM: Fast and deterministic
       - Skips most timesteps (uses only 50)
       - Deterministic (same seed = same image)
       - 20x faster with similar quality!
    
    3. When to use what:
       - DDPM: When you want stochastic sampling
       - DDIM: When you need speed or reproducibility
       - Both produce similar quality!
    """)

if __name__ == "__main__":
    main()
