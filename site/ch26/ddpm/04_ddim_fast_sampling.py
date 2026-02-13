# ==========================================
# DDIM (Denoising Diffusion Implicit Models)
# - Faster sampling: 50 steps instead of 1000
# - Deterministic sampling (no random noise)
# - Song et al., 2021
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
BATCH_SIZE = 128
LR = 2e-4
EPOCHS = 5
T = 1000              # training steps
DDIM_STEPS = 50       # sampling steps (much faster!)
DDIM_ETA = 0.0        # 0=deterministic, 1=DDPM-like stochastic
BASE_CH = 64
ATTN_RES = {16}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH = "ddim_samples.png"
SEED = 42

# ==========================================
# 1) Utilities
# ==========================================
@torch.no_grad()
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 1e-8, 0.999)

def extract(a, t, x_shape):
    out = a.gather(-1, t).float()
    while out.ndim < len(x_shape):
        out = out[..., None]
    return out

# ==========================================
# 2) Building blocks (same as DDPM)
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

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, groups=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.act1  = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch * 2)
        )

        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.act2  = nn.SiLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(self.act1(self.norm1(x)))
        scale, shift = self.time_mlp(t_emb).chunk(2, dim=1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]
        h = self.norm2(h)
        h = h * (1 + scale) + shift
        h = self.conv2(self.act2(h))
        return h + self.skip(x)

class SelfAttention2d(nn.Module):
    def __init__(self, channels, num_heads=1):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.num_heads = num_heads

    def forward(self, x):
        b, c, h, w = x.shape
        hds = self.num_heads
        x_norm = self.norm(x)
        q = self.q(x_norm).reshape(b, hds, c // hds, h * w)
        k = self.k(x_norm).reshape(b, hds, c // hds, h * w)
        v = self.v(x_norm).reshape(b, hds, c // hds, h * w)
        attn = torch.einsum("bhcn,bhcm->bhnm", q, k) * (1.0 / math.sqrt(c // hds))
        attn = attn.softmax(dim=-1)
        out = torch.einsum("bhnm,bhcm->bhcn", attn, v).reshape(b, c, h, w)
        out = self.proj(out)
        return out + x

class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 4, stride=2, padding=1)
    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)

# ==========================================
# 3) U-Net
# ==========================================
class UNet(nn.Module):
    def __init__(self, in_ch=IN_CHANNELS, base_ch=BASE_CH, ch_mults=(1, 2, 4, 4),
                 attn_res=ATTN_RES, img_size=IMG_SIZE, time_emb_dim=256):
        super().__init__()
        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )

        enc = []
        ch = base_ch
        hs = [ch]
        res = img_size
        for i, mult in enumerate(ch_mults):
            out_ch = base_ch * mult
            enc.append(ResidualBlock(ch, out_ch, time_emb_dim))
            ch = out_ch
            if res in attn_res:
                enc.append(SelfAttention2d(ch))
            enc.append(ResidualBlock(ch, ch, time_emb_dim))
            if res in attn_res:
                enc.append(SelfAttention2d(ch))
            hs.append(ch)
            if i != len(ch_mults) - 1:
                enc.append(Downsample(ch))
                res //= 2
                hs.append(ch)
        self.encoder = nn.ModuleList(enc)

        self.mid = nn.ModuleList([
            ResidualBlock(ch, ch, time_emb_dim),
            SelfAttention2d(ch),
            ResidualBlock(ch, ch, time_emb_dim)
        ])

        dec = []
        for i, mult in reversed(list(enumerate(ch_mults))):
            out_ch = base_ch * mult
            dec.append(ResidualBlock(ch + hs.pop(), out_ch, time_emb_dim))
            if res in attn_res:
                dec.append(SelfAttention2d(out_ch))
            dec.append(ResidualBlock(out_ch, out_ch, time_emb_dim))
            if res in attn_res:
                dec.append(SelfAttention2d(out_ch))
            if i != 0:
                dec.append(Upsample(out_ch))
                res *= 2
            ch = out_ch
        self.decoder = nn.ModuleList(dec)

        self.out_norm = nn.GroupNorm(8, ch)
        self.out_act  = nn.SiLU()
        self.out_conv = nn.Conv2d(ch, in_ch, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_emb(t)
        h = self.in_conv(x)
        skips = []
        for m in self.encoder:
            if isinstance(m, ResidualBlock):
                h = m(h, t_emb)
                skips.append(h)
            else:
                h = m(h)

        for m in self.mid:
            if isinstance(m, ResidualBlock):
                h = m(h, t_emb)
            else:
                h = m(h)

        for m in self.decoder:
            if isinstance(m, ResidualBlock):
                expected_in_ch = m.norm1.num_channels
                while h.shape[1] < expected_in_ch:
                    if not skips:
                        raise RuntimeError("Skip list exhausted")
                    skip = skips.pop()
                    if skip.shape[2:] != h.shape[2:]:
                        continue
                    h = torch.cat([h, skip], dim=1)
                if h.shape[1] != expected_in_ch:
                    raise RuntimeError(f"Channel mismatch: {h.shape[1]} vs {expected_in_ch}")
                h = m(h, t_emb)
            else:
                h = m(h)

        return self.out_conv(self.out_act(self.out_norm(h)))

# ==========================================
# 4) DDIM Module
# ==========================================
class DDIM(nn.Module):
    """
    DDIM: allows deterministic sampling with fewer steps.
    Key insight: we can skip timesteps during sampling!
    """
    def __init__(self, model, timesteps=T):
        super().__init__()
        self.model = model
        self.timesteps = timesteps

        betas = cosine_beta_schedule(timesteps).to(DEVICE)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, device=DEVICE), alphas_cumprod[:-1]], dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        return (
            extract(self.sqrt_alphas_cumprod, t, x0.shape) * x0
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise,
            noise,
        )

    def loss(self, x0):
        b = x0.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=DEVICE).long()
        x_t, noise = self.q_sample(x0, t)
        noise_pred = self.model(x_t, t)
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def ddim_sample(self, n, ddim_steps=DDIM_STEPS, eta=DDIM_ETA,
                    img_channels=IN_CHANNELS, img_size=IMG_SIZE):
        """
        DDIM sampling: use subset of timesteps for faster generation.
        
        eta: controls stochasticity
            - eta=0: deterministic (same seed → same image)
            - eta=1: equivalent to DDPM (stochastic)
        """
        self.model.eval()
        
        # Create subsequence of timesteps
        # Instead of all 1000 steps, we only use ddim_steps (e.g., 50)
        c = self.timesteps // ddim_steps
        ddim_timesteps = torch.arange(0, self.timesteps, c, device=DEVICE).long()
        ddim_timesteps_prev = torch.cat([torch.tensor([0], device=DEVICE), ddim_timesteps[:-1]])

        # Start from pure noise
        x = torch.randn(n, img_channels, img_size, img_size, device=DEVICE)

        # Reverse process
        for i in reversed(range(len(ddim_timesteps))):
            t = torch.full((n,), ddim_timesteps[i], device=DEVICE, dtype=torch.long)
            t_prev = torch.full((n,), ddim_timesteps_prev[i], device=DEVICE, dtype=torch.long)

            # Extract alpha values
            alpha_t = extract(self.alphas_cumprod, t, x.shape)
            alpha_t_prev = extract(self.alphas_cumprod, t_prev, x.shape)

            # Predict noise
            eps = self.model(x, t)

            # Predict x0 from noisy x_t
            pred_x0 = (x - torch.sqrt(1 - alpha_t) * eps) / torch.sqrt(alpha_t)
            pred_x0 = pred_x0.clamp(-1, 1)  # clip for stability

            # Direction pointing to x_t
            dir_xt = torch.sqrt(1 - alpha_t_prev - eta**2 * (1 - alpha_t) / (1 - alpha_t_prev) * (1 - alpha_t_prev)) * eps

            # Random noise (if eta > 0)
            noise = torch.randn_like(x) if eta > 0 else 0

            # DDIM update rule
            x = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt + eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t) * noise

        return x

# ==========================================
# 5) Data
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
    elif DATASET == "MNIST":
        transform = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        trainset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    else:
        raise ValueError("Unsupported dataset")

    loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,
                       num_workers=2, pin_memory=True)
    return loader

# ==========================================
# 6) Training
# ==========================================
@torch.no_grad()
def to_image_range(x):
    return (x.clamp(-1, 1) + 1) * 0.5

def main():
    loader = build_dataloader()

    unet = UNet(in_ch=IN_CHANNELS, base_ch=BASE_CH, ch_mults=(1, 2, 4, 4),
                attn_res=ATTN_RES, img_size=IMG_SIZE, time_emb_dim=256).to(DEVICE)
    ddim = DDIM(unet, timesteps=T).to(DEVICE)
    opt = torch.optim.AdamW(ddim.parameters(), lr=LR)

    print(f"Training DDIM with {T} steps, will sample with {DDIM_STEPS} steps (eta={DDIM_ETA})")
    
    for epoch in range(1, EPOCHS + 1):
        unet.train()
        running = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for x, _ in pbar:
            x = x.to(DEVICE, non_blocking=True)
            loss = ddim.loss(x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += loss.item()
            pbar.set_postfix(loss=f"{running / (pbar.n or 1):.4f}")
        print(f"[Epoch {epoch}] mean loss: {running / len(loader):.4f}")

    # Fast sampling with DDIM
    print(f"\n🚀 Sampling with DDIM ({DDIM_STEPS} steps instead of {T})")
    import time
    start = time.time()
    samples = ddim.ddim_sample(n=16, ddim_steps=DDIM_STEPS, eta=DDIM_ETA,
                               img_channels=IN_CHANNELS, img_size=IMG_SIZE)
    elapsed = time.time() - start
    
    samples = to_image_range(samples)
    torchvision.utils.save_image(samples, SAVE_PATH, nrow=4)
    print(f"✅ Generated 16 samples in {elapsed:.2f}s")
    print(f"   Saved to {SAVE_PATH}")
    print(f"   Speedup: ~{T/DDIM_STEPS:.1f}x faster than DDPM!")

if __name__ == "__main__":
    main()
