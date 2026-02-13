# ==========================================
# Latent Diffusion Models (LDM)
# - Core idea behind Stable Diffusion
# - Diffusion in compressed latent space (not pixel space)
# - Much more efficient: faster training & sampling
# - Rombach et al., 2022
# ==========================================
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from tqdm import tqdm

# -------------------
# Config
# -------------------
DATASET = "CIFAR10"
IN_CHANNELS = 3 if DATASET == "CIFAR10" else 1
IMG_SIZE = 32 if DATASET == "CIFAR10" else 28
LATENT_DIM = 4          # compress to 4 channels
LATENT_SCALE = 4        # spatial downscaling factor
BATCH_SIZE = 128
LR = 2e-4
EPOCHS_VAE = 10         # train autoencoder first
EPOCHS_DIFF = 5         # then train diffusion
T = 1000
BASE_CH = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH = "latent_diffusion_samples.png"
SEED = 42

# ==========================================
# 1) Autoencoder (compress images to latents)
# ==========================================
class Encoder(nn.Module):
    """Compress images to latent space"""
    def __init__(self, in_ch=IN_CHANNELS, latent_dim=LATENT_DIM):
        super().__init__()
        # Simple encoder: 32x32 -> 8x8
        self.conv1 = nn.Conv2d(in_ch, 64, 3, stride=2, padding=1)      # /2
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)        # /2
        self.conv3 = nn.Conv2d(128, latent_dim * 2, 3, padding=1)      # mu and logvar
        self.act = nn.SiLU()

    def forward(self, x):
        h = self.act(self.conv1(x))
        h = self.act(self.conv2(h))
        h = self.conv3(h)
        mu, logvar = h.chunk(2, dim=1)
        return mu, logvar

class Decoder(nn.Module):
    """Reconstruct images from latent space"""
    def __init__(self, latent_dim=LATENT_DIM, out_ch=IN_CHANNELS):
        super().__init__()
        # Simple decoder: 8x8 -> 32x32
        self.conv1 = nn.Conv2d(latent_dim, 128, 3, padding=1)
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2 = nn.Conv2d(128, 64, 3, padding=1)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv3 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.act = nn.SiLU()

    def forward(self, z):
        h = self.act(self.conv1(z))
        h = self.up1(h)
        h = self.act(self.conv2(h))
        h = self.up2(h)
        return torch.tanh(self.conv3(h))  # [-1, 1]

class VAE(nn.Module):
    """Variational Autoencoder for learning compressed representation"""
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    @torch.no_grad()
    def encode(self, x):
        mu, logvar = self.encoder(x)
        return self.reparameterize(mu, logvar)

    @torch.no_grad()
    def decode(self, z):
        return self.decoder(z)

# ==========================================
# 2) Latent Diffusion U-Net
#    (operates on compressed latent space)
# ==========================================
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        t = t.float()[:, None]
        freqs = torch.exp(
            torch.arange(half, device=device).float() * -(math.log(10000) / (half - 1))
        )
        args = t * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch * 2)
        )
        
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        
        scale, shift = self.time_mlp(t_emb).chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = F.silu(h)
        h = self.conv2(h)
        
        return h + self.skip(x)

class LatentUNet(nn.Module):
    """
    Simplified U-Net for latent space.
    Input: compressed latent (e.g., 4 x 8 x 8)
    Output: predicted noise in latent space
    """
    def __init__(self, latent_dim=LATENT_DIM, base_ch=64, time_dim=128):
        super().__init__()
        
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim)
        )
        
        # Encoder
        self.in_conv = nn.Conv2d(latent_dim, base_ch, 3, padding=1)
        self.down1 = ResBlock(base_ch, base_ch * 2, time_dim)
        self.down2 = ResBlock(base_ch * 2, base_ch * 4, time_dim)
        
        # Middle
        self.mid1 = ResBlock(base_ch * 4, base_ch * 4, time_dim)
        self.mid2 = ResBlock(base_ch * 4, base_ch * 4, time_dim)
        
        # Decoder
        self.up1 = ResBlock(base_ch * 8, base_ch * 2, time_dim)  # concat from down2
        self.up2 = ResBlock(base_ch * 4, base_ch, time_dim)      # concat from down1
        
        self.out_norm = nn.GroupNorm(8, base_ch)
        self.out_conv = nn.Conv2d(base_ch, latent_dim, 3, padding=1)

    def forward(self, z, t):
        t_emb = self.time_emb(t)
        
        # Encoder
        h = self.in_conv(z)
        h1 = self.down1(h, t_emb)
        h2 = self.down2(h1, t_emb)
        
        # Middle
        h = self.mid1(h2, t_emb)
        h = self.mid2(h, t_emb)
        
        # Decoder with skip connections
        h = self.up1(torch.cat([h, h2], dim=1), t_emb)
        h = self.up2(torch.cat([h, h1], dim=1), t_emb)
        
        return self.out_conv(F.silu(self.out_norm(h)))

# ==========================================
# 3) Latent Diffusion Process
# ==========================================
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 1e-8, 0.999)

def extract(a, t, x_shape):
    out = a.gather(-1, t)
    return out.reshape(-1, *([1] * (len(x_shape) - 1)))

class LatentDiffusion(nn.Module):
    """Diffusion model in latent space"""
    def __init__(self, vae, unet, timesteps=T):
        super().__init__()
        self.vae = vae
        self.unet = unet
        self.timesteps = timesteps
        
        # Freeze VAE during diffusion training
        for param in self.vae.parameters():
            param.requires_grad = False
        
        betas = cosine_beta_schedule(timesteps).to(DEVICE)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def q_sample(self, z0, t, noise=None):
        """Add noise to latent z0"""
        if noise is None:
            noise = torch.randn_like(z0)
        sqrt_alpha = extract(self.sqrt_alphas_cumprod, t, z0.shape)
        sqrt_one_minus_alpha = extract(self.sqrt_one_minus_alphas_cumprod, t, z0.shape)
        return sqrt_alpha * z0 + sqrt_one_minus_alpha * noise, noise

    def loss(self, x0):
        """Training loss: encode to latent, add noise, predict noise"""
        # Encode to latent space
        with torch.no_grad():
            z0 = self.vae.encode(x0)
        
        # Sample random timestep
        b = z0.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=DEVICE).long()
        
        # Add noise and predict
        zt, noise = self.q_sample(z0, t)
        noise_pred = self.unet(zt, t)
        
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def sample(self, n, img_size=IMG_SIZE):
        """Generate samples: denoise in latent space, then decode"""
        self.unet.eval()
        
        # Start from random noise in latent space
        latent_size = img_size // LATENT_SCALE
        z = torch.randn(n, LATENT_DIM, latent_size, latent_size, device=DEVICE)
        
        # Denoise in latent space
        for step in reversed(range(self.timesteps)):
            t = torch.full((n,), step, device=DEVICE, dtype=torch.long)
            
            # Predict noise
            noise_pred = self.unet(z, t)
            
            # Denoise step
            alpha_t = extract(self.alphas_cumprod, t, z.shape)
            alpha_prev = extract(self.alphas_cumprod, t - 1, z.shape) if step > 0 else 1.0
            beta_t = extract(self.betas, t, z.shape)
            
            # DDPM formula
            z = (z - beta_t / torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(1 - beta_t)
            
            if step > 0:
                noise = torch.randn_like(z)
                z = z + torch.sqrt(beta_t) * noise
        
        # Decode latents to images
        samples = self.vae.decode(z)
        return samples

# ==========================================
# 4) Data
# ==========================================
def build_dataloader():
    torch.manual_seed(SEED)
    
    if DATASET == "CIFAR10":
        transform = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*IN_CHANNELS, [0.5]*IN_CHANNELS),
        ])
        trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    else:
        transform = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        trainset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    
    return DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

# ==========================================
# 5) Training
# ==========================================
def train_vae(vae, loader, epochs):
    """First train the autoencoder"""
    print("\n" + "="*60)
    print("STAGE 1: Training VAE (Autoencoder)")
    print("="*60)
    
    opt = torch.optim.AdamW(vae.parameters(), lr=LR)
    
    for epoch in range(1, epochs + 1):
        vae.train()
        running_recon = 0.0
        running_kl = 0.0
        
        pbar = tqdm(loader, desc=f"VAE Epoch {epoch}/{epochs}")
        for x, _ in pbar:
            x = x.to(DEVICE)
            
            recon, mu, logvar = vae(x)
            
            # Reconstruction loss
            recon_loss = F.mse_loss(recon, x, reduction='sum') / x.shape[0]
            
            # KL divergence
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0]
            
            # Total loss
            loss = recon_loss + 0.001 * kl_loss  # small KL weight
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            running_recon += recon_loss.item()
            running_kl += kl_loss.item()
            pbar.set_postfix(recon=f"{running_recon/(pbar.n or 1):.4f}", 
                           kl=f"{running_kl/(pbar.n or 1):.4f}")
        
        print(f"[VAE Epoch {epoch}] Recon: {running_recon/len(loader):.4f}, KL: {running_kl/len(loader):.4f}")

def train_latent_diffusion(ldm, loader, epochs):
    """Then train diffusion in latent space"""
    print("\n" + "="*60)
    print("STAGE 2: Training Latent Diffusion")
    print("="*60)
    
    opt = torch.optim.AdamW(ldm.unet.parameters(), lr=LR)
    
    for epoch in range(1, epochs + 1):
        ldm.unet.train()
        running = 0.0
        
        pbar = tqdm(loader, desc=f"Diffusion Epoch {epoch}/{epochs}")
        for x, _ in pbar:
            x = x.to(DEVICE)
            
            loss = ldm.loss(x)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            running += loss.item()
            pbar.set_postfix(loss=f"{running/(pbar.n or 1):.4f}")
        
        print(f"[Diffusion Epoch {epoch}] Loss: {running/len(loader):.4f}")

@torch.no_grad()
def to_image_range(x):
    return (x.clamp(-1, 1) + 1) * 0.5

def main():
    loader = build_dataloader()
    
    # Create models
    vae = VAE().to(DEVICE)
    unet = LatentUNet(latent_dim=LATENT_DIM, base_ch=BASE_CH).to(DEVICE)
    
    # Train autoencoder first
    train_vae(vae, loader, EPOCHS_VAE)
    
    # Create latent diffusion model
    ldm = LatentDiffusion(vae, unet, timesteps=T).to(DEVICE)
    
    # Train diffusion in latent space
    train_latent_diffusion(ldm, loader, EPOCHS_DIFF)
    
    # Generate samples
    print("\n" + "="*60)
    print("Generating samples from latent diffusion...")
    print("="*60)
    
    samples = ldm.sample(n=16, img_size=IMG_SIZE)
    samples = to_image_range(samples)
    
    torchvision.utils.save_image(samples, SAVE_PATH, nrow=4)
    print(f"✅ Saved samples to {SAVE_PATH}")
    print(f"\n💡 Key advantages of Latent Diffusion:")
    print(f"   - Trains on {LATENT_DIM}x{IMG_SIZE//LATENT_SCALE}x{IMG_SIZE//LATENT_SCALE} latents instead of {IN_CHANNELS}x{IMG_SIZE}x{IMG_SIZE} pixels")
    print(f"   - ~{(IMG_SIZE * IMG_SIZE * IN_CHANNELS) / (LATENT_DIM * (IMG_SIZE//LATENT_SCALE)**2):.1f}x less memory per sample")
    print(f"   - This is the core idea behind Stable Diffusion!")

if __name__ == "__main__":
    main()
