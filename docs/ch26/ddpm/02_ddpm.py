# ==========================================
# Improved DDPM (Ho et al., 2020) - PyTorch Implementation
# ==========================================
# Key improvements over baseline:
#   1. EMA (Exponential Moving Average) for stable sampling
#   2. Gradient clipping for training stability
#   3. Learning rate warmup + cosine decay schedule
#   4. Increased model capacity (BASE_CH=128)
#   5. Multi-resolution attention (8, 16)
#   6. More residual blocks per level (3 instead of 2)
#   7. Checkpoint saving for long training runs
#   8. Sample generation during training for monitoring
# ==========================================

import math
import copy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from tqdm import tqdm

# ==========================================
# Configuration (edit these parameters)
# ==========================================
class Config:
    # Dataset settings
    DATASET = "MNIST"           # "CIFAR10" or "MNIST"
    IMG_SIZE = 32               # Both datasets resized to 32x32
    
    # Training settings
    BATCH_SIZE = 128
    LR = 1e-4                   # Lower LR for stability
    EPOCHS = 100                # Increased from 10 - critical for quality
    WARMUP_STEPS = 1000         # LR warmup steps
    GRAD_CLIP = 1.0             # Gradient clipping threshold
    
    # Diffusion settings
    T = 1000                    # Number of diffusion timesteps
    
    # Model architecture
    BASE_CH = 128               # Increased from 64 for more capacity
    CH_MULTS = (1, 2, 4, 4)     # Channel multipliers at each level
    NUM_RES_BLOCKS = 3          # Residual blocks per level (was 2)
    ATTN_RES = {8, 16}          # Apply attention at these resolutions
    TIME_EMB_DIM = 256          # Timestep embedding dimension
    DROPOUT = 0.1               # Dropout for regularization
    
    # EMA settings
    EMA_DECAY = 0.9999          # EMA decay rate
    EMA_START = 2000            # Start EMA after this many steps
    
    # Logging and saving
    SAMPLE_INTERVAL = 10        # Generate samples every N epochs
    SAVE_INTERVAL = 20          # Save checkpoint every N epochs
    NUM_SAMPLES = 16            # Number of samples to generate
    
    # System settings
    NUM_WORKERS = 0             # DataLoader workers (0 for compatibility)
    SEED = 42
    
    @property
    def IN_CHANNELS(self):
        return 3 if self.DATASET == "CIFAR10" else 1
    
    @property
    def DEVICE(self):
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    @property
    def SAVE_DIR(self):
        return os.path.dirname(os.path.abspath(__file__))

config = Config()

# ==========================================
# 1) Noise Schedule Utilities
# ==========================================
@torch.no_grad()
def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule as proposed in "Improved DDPM" (Nichol & Dhariwal, 2021).
    Provides smoother noise levels compared to linear schedule.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 1e-8, 0.999)


def extract(a, t, x_shape):
    """
    Extract values from tensor 'a' at indices 't' and reshape for broadcasting.
    a: [T] tensor of values
    t: [B] tensor of indices
    x_shape: target shape for broadcasting
    """
    out = a.gather(-1, t).float()
    while out.ndim < len(x_shape):
        out = out[..., None]
    return out


# ==========================================
# 2) Sinusoidal Timestep Embedding
# ==========================================
class SinusoidalPosEmb(nn.Module):
    """
    Sinusoidal positional embeddings for timesteps.
    Maps scalar timestep to high-dimensional embedding using sine/cosine.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        t = t.float()[:, None]  # [B, 1]
        
        # Frequency bands (exponentially spaced)
        freqs = torch.exp(
            torch.arange(half, device=device).float() * -(math.log(10000) / (half - 1))
        )
        args = t * freqs[None, :]  # [B, half]
        
        # Concatenate sin and cos embeddings
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, dim]
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


# ==========================================
# 3) Building Blocks: ResBlock, Attention, Up/Down
# ==========================================
class ResidualBlock(nn.Module):
    """
    Residual block with time embedding conditioning.
    Uses GroupNorm, SiLU activation, and optional dropout.
    Time embedding modulates features via scale and shift (FiLM).
    """
    def __init__(self, in_ch, out_ch, time_emb_dim, dropout=0.1, groups=8):
        super().__init__()
        # First convolution path
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        # Time embedding projection (scale and shift)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch * 2)
        )

        # Second convolution path with dropout
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        # Skip connection (identity or 1x1 conv if channels differ)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        # First conv
        h = self.conv1(self.act1(self.norm1(x)))
        
        # Time conditioning via FiLM (Feature-wise Linear Modulation)
        scale, shift = self.time_mlp(t_emb).chunk(2, dim=1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]
        h = self.norm2(h)
        h = h * (1 + scale) + shift
        
        # Second conv with dropout
        h = self.conv2(self.dropout(self.act2(h)))
        
        return h + self.skip(x)


class SelfAttention2d(nn.Module):
    """
    Multi-head self-attention for 2D feature maps.
    Captures long-range spatial dependencies.
    """
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        self.norm = nn.GroupNorm(8, channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        
        # Scale factor for attention
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        b, c, h, w = x.shape
        
        # Normalize and compute Q, K, V
        x_norm = self.norm(x)
        q = self.q(x_norm).reshape(b, self.num_heads, self.head_dim, h * w)
        k = self.k(x_norm).reshape(b, self.num_heads, self.head_dim, h * w)
        v = self.v(x_norm).reshape(b, self.num_heads, self.head_dim, h * w)
        
        # Attention: softmax(Q @ K^T / sqrt(d)) @ V
        attn = torch.einsum("bhcn,bhcm->bhnm", q, k) * self.scale
        attn = attn.softmax(dim=-1)
        
        # Apply attention to values
        out = torch.einsum("bhnm,bhcm->bhcn", attn, v)
        out = out.reshape(b, c, h, w)
        out = self.proj(out)
        
        return out + x  # Residual connection


class Downsample(nn.Module):
    """Spatial downsampling by factor of 2 using strided convolution."""
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """Spatial upsampling by factor of 2 using nearest interpolation + conv."""
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


# ==========================================
# 4) U-Net Architecture
# ==========================================
class UNet(nn.Module):
    """
    U-Net backbone for noise prediction.
    
    Architecture:
    - Encoder: Downsampling path with residual blocks and attention
    - Bottleneck: Middle blocks with attention
    - Decoder: Upsampling path with skip connections
    
    Features:
    - Time embedding conditioning at every residual block
    - Self-attention at specified resolutions
    - Multiple residual blocks per resolution level
    """
    def __init__(
        self,
        in_ch,
        base_ch,
        ch_mults,
        num_res_blocks,
        attn_res,
        img_size,
        time_emb_dim,
        dropout,
    ):
        super().__init__()
        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        # Time embedding MLP
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        # ========== Encoder (Downsampling Path) ==========
        self.downs = nn.ModuleList()
        ch = base_ch
        res = img_size
        self.skip_channels = []  # Track channels for skip connections

        for i, mult in enumerate(ch_mults):
            out_ch = base_ch * mult
            level_blocks = nn.ModuleList()
            
            # Multiple residual blocks per level
            for j in range(num_res_blocks):
                block_in_ch = ch if j == 0 else out_ch
                level_blocks.append(ResidualBlock(block_in_ch, out_ch, time_emb_dim, dropout))
            
            # Attention at specified resolutions
            attn = SelfAttention2d(out_ch) if res in attn_res else nn.Identity()
            
            # Downsampling (except at last level)
            down = Downsample(out_ch) if i != len(ch_mults) - 1 else nn.Identity()

            self.downs.append(nn.ModuleDict({
                "blocks": level_blocks,
                "attn": attn,
                "down": down,
            }))
            
            self.skip_channels.append(out_ch)
            ch = out_ch
            if i != len(ch_mults) - 1:
                res //= 2

        # ========== Bottleneck (Middle) ==========
        self.mid = nn.ModuleDict({
            "block1": ResidualBlock(ch, ch, time_emb_dim, dropout),
            "attn": SelfAttention2d(ch),
            "block2": ResidualBlock(ch, ch, time_emb_dim, dropout),
        })

        # ========== Decoder (Upsampling Path) ==========
        self.ups = nn.ModuleList()
        
        for i, mult in reversed(list(enumerate(ch_mults))):
            out_ch = base_ch * mult
            skip_ch = self.skip_channels.pop()
            level_blocks = nn.ModuleList()
            
            # First block takes concatenated features (current + skip)
            for j in range(num_res_blocks):
                if j == 0:
                    block_in_ch = ch + skip_ch
                else:
                    block_in_ch = out_ch
                level_blocks.append(ResidualBlock(block_in_ch, out_ch, time_emb_dim, dropout))
            
            # Attention at specified resolutions
            attn = SelfAttention2d(out_ch) if res in attn_res else nn.Identity()
            
            # Upsampling (except at first level)
            up = Upsample(out_ch) if i != 0 else nn.Identity()

            self.ups.append(nn.ModuleDict({
                "blocks": level_blocks,
                "attn": attn,
                "up": up,
            }))

            ch = out_ch
            if i != 0:
                res *= 2

        # Output projection
        self.out_norm = nn.GroupNorm(8, ch)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(ch, in_ch, 3, padding=1)

    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_emb(t)

        # ========== Encoder ==========
        h = self.in_conv(x)
        skips = []
        
        for level in self.downs:
            for block in level["blocks"]:
                h = block(h, t_emb)
            h = level["attn"](h)
            skips.append(h)
            h = level["down"](h)

        # ========== Bottleneck ==========
        h = self.mid["block1"](h, t_emb)
        h = self.mid["attn"](h)
        h = self.mid["block2"](h, t_emb)

        # ========== Decoder ==========
        for level in self.ups:
            skip = skips.pop()
            
            # Handle potential size mismatch (center crop skip to match h)
            if skip.shape[2:] != h.shape[2:]:
                dh = skip.shape[2] - h.shape[2]
                dw = skip.shape[3] - h.shape[3]
                skip = skip[:, :, dh // 2 : dh // 2 + h.shape[2], 
                                 dw // 2 : dw // 2 + h.shape[3]]
            
            h = torch.cat([h, skip], dim=1)
            
            for block in level["blocks"]:
                h = block(h, t_emb)
            h = level["attn"](h)
            h = level["up"](h)

        # Output
        h = self.out_conv(self.out_act(self.out_norm(h)))
        return h


# ==========================================
# 5) DDPM Core: Forward/Reverse Process
# ==========================================
class DDPM(nn.Module):
    """
    Denoising Diffusion Probabilistic Model.
    
    Training: Add noise to data (forward process), learn to predict the noise.
    Sampling: Start from pure noise, iteratively denoise (reverse process).
    """
    def __init__(self, model, timesteps, device):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.device = device

        # Precompute diffusion parameters
        betas = cosine_beta_schedule(timesteps).to(device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1, device=device), alphas_cumprod[:-1]], dim=0
        )

        # Register as buffers (saved with model, moved with .to())
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer("posterior_variance",
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod).clamp_min(1e-20)
        )

    def q_sample(self, x0, t, noise=None):
        """
        Forward process: Add noise to clean data.
        q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
        """
        if noise is None:
            noise = torch.randn_like(x0)
        
        x_t = (
            extract(self.sqrt_alphas_cumprod, t, x0.shape) * x0 +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise
        )
        return x_t, noise

    @torch.no_grad()
    def p_sample(self, x_t, t):
        """
        Reverse process: One denoising step.
        p(x_{t-1} | x_t) using predicted noise.
        """
        betas_t = extract(self.betas, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_t.shape
        )
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x_t.shape)

        # Predict noise
        eps_theta = self.model(x_t, t)
        
        # Compute mean of p(x_{t-1} | x_t)
        model_mean = sqrt_recip_alphas_t * (
            x_t - betas_t * eps_theta / sqrt_one_minus_alphas_cumprod_t
        )

        # No noise at t=0
        if (t == 0).all():
            return model_mean

        # Add noise scaled by posterior variance
        posterior_var_t = extract(self.posterior_variance, t, x_t.shape)
        noise = torch.randn_like(x_t)
        return model_mean + torch.sqrt(posterior_var_t) * noise

    @torch.no_grad()
    def sample(self, n, img_channels, img_size, show_progress=True):
        """
        Generate samples by running the full reverse process.
        """
        self.model.eval()
        
        # Start from pure noise
        x_t = torch.randn(n, img_channels, img_size, img_size, device=self.device)
        
        # Iteratively denoise
        timesteps = reversed(range(self.timesteps))
        if show_progress:
            timesteps = tqdm(timesteps, desc="Sampling", total=self.timesteps)
        
        for step in timesteps:
            t = torch.full((n,), step, device=self.device, dtype=torch.long)
            x_t = self.p_sample(x_t, t)
        
        return x_t

    def loss(self, x0):
        """
        Training loss: MSE between predicted and actual noise.
        """
        b = x0.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=self.device).long()
        x_t, noise = self.q_sample(x0, t)
        noise_pred = self.model(x_t, t)
        return F.mse_loss(noise_pred, noise)


# ==========================================
# 6) EMA (Exponential Moving Average)
# ==========================================
class EMA:
    """
    Maintains exponential moving average of model parameters.
    Critical for stable, high-quality generation.
    """
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for param in self.shadow.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        """Update EMA parameters."""
        for ema_param, model_param in zip(self.shadow.parameters(), model.parameters()):
            ema_param.data.mul_(self.decay).add_(model_param.data, alpha=1 - self.decay)

    def forward(self, *args, **kwargs):
        """Use EMA model for inference."""
        return self.shadow(*args, **kwargs)


# ==========================================
# 7) Learning Rate Schedule
# ==========================================
class WarmupCosineSchedule:
    """
    Learning rate schedule with linear warmup followed by cosine decay.
    """
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]["lr"]
        self.current_step = 0

    def step(self):
        self.current_step += 1
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _get_lr(self):
        if self.current_step < self.warmup_steps:
            # Linear warmup
            return self.base_lr * self.current_step / self.warmup_steps
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))


# ==========================================
# 8) Dataset Loading
# ==========================================
def build_dataloader(config):
    """Build training dataloader for CIFAR10 or MNIST."""
    torch.manual_seed(config.SEED)

    if config.DATASET == "CIFAR10":
        transform = transforms.Compose([
            transforms.Resize(config.IMG_SIZE),
            transforms.RandomHorizontalFlip(),  # Data augmentation
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),  # -> [-1, 1]
        ])
        trainset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )
    elif config.DATASET == "MNIST":
        transform = transforms.Compose([
            transforms.Resize(config.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # -> [-1, 1]
        ])
        trainset = torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
    else:
        raise ValueError(f"Unsupported dataset: {config.DATASET}")

    loader = DataLoader(
        trainset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Avoid batch size issues
    )
    return loader


# ==========================================
# 9) Utility Functions
# ==========================================
@torch.no_grad()
def to_image_range(x):
    """Convert from [-1, 1] to [0, 1] for visualization."""
    return (x.clamp(-1, 1) + 1) * 0.5


def save_samples(ddpm, ema, config, epoch, use_ema=True):
    """Generate and save sample images."""
    model_to_use = ema.shadow if use_ema and ema is not None else ddpm.model
    
    # Temporarily swap model for sampling
    original_model = ddpm.model
    ddpm.model = model_to_use
    
    samples = ddpm.sample(
        n=config.NUM_SAMPLES,
        img_channels=config.IN_CHANNELS,
        img_size=config.IMG_SIZE,
        show_progress=False
    )
    
    # Restore original model
    ddpm.model = original_model
    
    samples = to_image_range(samples)
    save_path = os.path.join(config.SAVE_DIR, f"ddpm_samples_epoch_{epoch}.png")
    torchvision.utils.save_image(samples, save_path, nrow=4)
    print(f"  -> Saved samples to {save_path}")


def save_checkpoint(ddpm, ema, optimizer, scheduler, epoch, config):
    """Save training checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": ddpm.model.state_dict(),
        "ema_state_dict": ema.shadow.state_dict() if ema else None,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_step": scheduler.current_step,
        "config": {
            "DATASET": config.DATASET,
            "BASE_CH": config.BASE_CH,
            "EPOCHS": config.EPOCHS,
        }
    }
    save_path = os.path.join(config.SAVE_DIR, f"ddpm_checkpoint_epoch_{epoch}.pt")
    torch.save(checkpoint, save_path)
    print(f"  -> Saved checkpoint to {save_path}")


# ==========================================
# 10) Main Training Loop
# ==========================================
def main():
    print("=" * 60)
    print("Improved DDPM Training")
    print("=" * 60)
    print(f"Dataset: {config.DATASET}")
    print(f"Device: {config.DEVICE}")
    print(f"Epochs: {config.EPOCHS}")
    print(f"Base channels: {config.BASE_CH}")
    print(f"Attention resolutions: {config.ATTN_RES}")
    print(f"Residual blocks per level: {config.NUM_RES_BLOCKS}")
    print(f"EMA decay: {config.EMA_DECAY}")
    print("=" * 60)

    # Set random seed
    torch.manual_seed(config.SEED)

    # Build dataloader
    loader = build_dataloader(config)
    total_steps = config.EPOCHS * len(loader)

    # Build model
    unet = UNet(
        in_ch=config.IN_CHANNELS,
        base_ch=config.BASE_CH,
        ch_mults=config.CH_MULTS,
        num_res_blocks=config.NUM_RES_BLOCKS,
        attn_res=config.ATTN_RES,
        img_size=config.IMG_SIZE,
        time_emb_dim=config.TIME_EMB_DIM,
        dropout=config.DROPOUT,
    ).to(config.DEVICE)

    # Count parameters
    num_params = sum(p.numel() for p in unet.parameters())
    print(f"Model parameters: {num_params:,}")

    # Build DDPM wrapper
    ddpm = DDPM(unet, timesteps=config.T, device=config.DEVICE).to(config.DEVICE)

    # Optimizer
    optimizer = torch.optim.AdamW(ddpm.parameters(), lr=config.LR, weight_decay=0.01)

    # Learning rate scheduler
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=config.WARMUP_STEPS,
        total_steps=total_steps,
    )

    # EMA (initialized after some training steps)
    ema = None
    global_step = 0

    # Training loop
    for epoch in range(1, config.EPOCHS + 1):
        unet.train()
        running_loss = 0.0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{config.EPOCHS}")
        for x, _ in pbar:
            x = x.to(config.DEVICE, non_blocking=True)
            
            # Forward pass
            loss = ddpm.loss(x)
            
            # Backward pass
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(ddpm.parameters(), config.GRAD_CLIP)
            
            # Optimizer step
            optimizer.step()
            
            # LR scheduler step
            scheduler.step()
            
            # Initialize or update EMA
            global_step += 1
            if global_step == config.EMA_START:
                print(f"  -> Initializing EMA at step {global_step}")
                ema = EMA(unet, decay=config.EMA_DECAY)
            elif global_step > config.EMA_START and ema is not None:
                ema.update(unet)
            
            # Logging
            running_loss += loss.item()
            current_lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix(
                loss=f"{running_loss / (pbar.n or 1):.4f}",
                lr=f"{current_lr:.2e}"
            )

        avg_loss = running_loss / len(loader)
        print(f"[Epoch {epoch}] Mean loss: {avg_loss:.4f}")

        # Generate samples periodically
        if epoch % config.SAMPLE_INTERVAL == 0 or epoch == 1:
            save_samples(ddpm, ema, config, epoch, use_ema=(ema is not None))

        # Save checkpoint periodically
        if epoch % config.SAVE_INTERVAL == 0:
            save_checkpoint(ddpm, ema, optimizer, scheduler, epoch, config)

    # Final samples and checkpoint
    print("\nTraining complete! Generating final samples...")
    save_samples(ddpm, ema, config, config.EPOCHS, use_ema=(ema is not None))
    save_checkpoint(ddpm, ema, optimizer, scheduler, config.EPOCHS, config)

    # Also save a samples file with the original naming convention
    final_path = os.path.join(config.SAVE_DIR, f"ddpm_samples_{config.EPOCHS}.png")
    if ema is not None:
        ddpm.model = ema.shadow
    samples = ddpm.sample(
        n=config.NUM_SAMPLES,
        img_channels=config.IN_CHANNELS,
        img_size=config.IMG_SIZE,
    )
    samples = to_image_range(samples)
    torchvision.utils.save_image(samples, final_path, nrow=4)
    print(f"Saved final samples to {final_path}")


if __name__ == "__main__":
    main()
