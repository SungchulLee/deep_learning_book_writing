# ============================================================
# Class-Conditional DDPM (CIFAR-10) Ñ PyTorch, well-commented
# - U-Net with residual blocks + (optional) attention
# - Sinusoidal timestep embeddings
# - Label conditioning (embedding added to time embedding)
# - Optional Classifier-Free Guidance via label dropout
# ============================================================
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from tqdm import tqdm

# -------------------
# Config (edit here)
# -------------------
DATASET = "CIFAR10"      # "CIFAR10" or "MNIST"
NUM_CLASSES = 10         # CIFAR-10 has 10 classes
IN_CHANNELS = 3 if DATASET == "CIFAR10" else 1
IMG_SIZE = 32 if DATASET == "CIFAR10" else 28
BATCH_SIZE = 128
LR = 2e-4
EPOCHS = 5
T = 1000                 # classic DDPM uses 1000 steps
BASE_CH = 64             # model width; increase for quality (e.g., 96/128)
ATTN_RES = {16}          # apply attention at this spatial size; set() to disable
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH = "ddpm_conditional_samples.png"

# ------- Classifier-Free Guidance (CFG) -------
USE_CFG = True           # set False to disable
CFG_DROPOUT_P = 0.1      # during training, drop labels with this prob
CFG_SCALE = 3.0          # during sampling, guidance strength (1.0 = off)
NULL_CLASS_ID = NUM_CLASSES  # reserved index for "null" condition

# ==========================================
# 1) Diffusion schedules & helpers
# ==========================================
@torch.no_grad()
def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine schedule (Nichol & Dhariwal 2021) for smoother training."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 1e-8, 0.999)

def extract(a, t, x_shape):
    """Gather a[t] and reshape for broadcasting to x_shape."""
    out = a.gather(-1, t).float()
    while out.ndim < len(x_shape):
        out = out[..., None]
    return out

# ==========================================
# 2) Sinusoidal timestep embedding
# ==========================================
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: [B] integer timesteps
        device = t.device
        half = self.dim // 2
        t = t.float()[:, None]              # [B,1]
        freqs = torch.exp(
            torch.arange(half, device=device).float() * -(math.log(10000) / (half - 1))
        )
        args = t * freqs[None, :]           # [B, half]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, dim or dim-1]
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0,1))
        return emb

# ==========================================
# 3) U-Net building blocks
# ==========================================
class ResidualBlock(nn.Module):
    """ResBlock with GroupNorm + SiLU + FiLM time conditioning."""
    def __init__(self, in_ch, out_ch, time_emb_dim, groups=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.act1  = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch * 2)  # scale & shift
        )

        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.act2  = nn.SiLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        # First conv
        h = self.conv1(self.act1(self.norm1(x)))

        # FiLM conditioning from time embedding
        scale, shift = self.time_mlp(t_emb).chunk(2, dim=1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]

        h = self.norm2(h)
        h = h * (1 + scale) + shift
        h = self.conv2(self.act2(h))
        return h + self.skip(x)

class SelfAttention2d(nn.Module):
    """Lightweight self-attention at a given resolution."""
    def __init__(self, ch, heads=1):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.q = nn.Conv2d(ch, ch, 1)
        self.k = nn.Conv2d(ch, ch, 1)
        self.v = nn.Conv2d(ch, ch, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        self.heads = heads

    def forward(self, x):
        b, c, h, w = x.shape
        hds = self.heads
        x_ = self.norm(x)
        q = self.q(x_).reshape(b, hds, c // hds, h * w)
        k = self.k(x_).reshape(b, hds, c // hds, h * w)
        v = self.v(x_).reshape(b, hds, c // hds, h * w)
        attn = torch.einsum('bhcn,bhcm->bhnm', q, k) * (1.0 / math.sqrt(c // hds))
        attn = attn.softmax(dim=-1)
        out = torch.einsum('bhnm,bhcm->bhcn', attn, v).reshape(b, c, h, w)
        return self.proj(out) + x

class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 4, stride=2, padding=1)
    def forward(self, x): return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)

# ==========================================
# 4) Conditional U-Net
# ==========================================
class UNetCond(nn.Module):
    """
    Conditional U-Net.
    Conditioning: class label -> embedding -> added to time embedding.
    With CFG, we reserve an extra embedding slot for the "null" label.
    """
    def __init__(self, in_ch=IN_CHANNELS, base_ch=BASE_CH, ch_mults=(1,2,4,4),
                 attn_res=ATTN_RES, img_size=IMG_SIZE, time_emb_dim=256,
                 num_classes=NUM_CLASSES, use_cfg=USE_CFG):
        super().__init__()
        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        # time embedding MLP
        self.time_sinus = SinusoidalPosEmb(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        # label embedding (note +1 for null token if CFG)
        label_vocab = num_classes + (1 if use_cfg else 0)
        self.label_emb = nn.Embedding(label_vocab, time_emb_dim)

        # Encoder
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

        # Middle
        self.mid = nn.ModuleList([
            ResidualBlock(ch, ch, time_emb_dim),
            SelfAttention2d(ch),
            ResidualBlock(ch, ch, time_emb_dim)
        ])

        # Decoder
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

    def forward(self, x, t, y):
        """
        x: [B, C, H, W] noisy images
        t: [B] integer timesteps
        y: [B] class labels (NUM_CLASSES or NULL_CLASS_ID if CFG null)
        """
        # Build time embedding and add label embedding
        t_emb = self.time_mlp(self.time_sinus(t))  # [B, D]
        y_emb = self.label_emb(y)                  # [B, D]
        t_emb = t_emb + y_emb                      # simple, effective conditioning

        # Encoder
        h = self.in_conv(x)
        skips = []
        for m in self.encoder:
            if isinstance(m, ResidualBlock):
                h = m(h, t_emb)
                skips.append(h)
            else:
                h = m(h)

        # Middle
        for m in self.mid:
            if isinstance(m, ResidualBlock):
                h = m(h, t_emb)
            else:
                h = m(h)

        # Decoder with skip connections
        for m in self.decoder:
            if isinstance(m, ResidualBlock):
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = m(h, t_emb)
            else:
                h = m(h)

        return self.out_conv(self.out_act(self.out_norm(h)))

# ==========================================
# 5) DDPM core with conditioning + CFG
# ==========================================
class DDPMCond(nn.Module):
    def __init__(self, model, timesteps=T):
        super().__init__()
        self.model = model
        self.timesteps = timesteps

        betas = cosine_beta_schedule(timesteps).to(DEVICE)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, device=DEVICE), alphas_cumprod[:-1]], dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        self.register_buffer('posterior_variance',
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod).clamp_min(1e-20))

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        return (
            extract(self.sqrt_alphas_cumprod, t, x0.shape) * x0 +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise,
            noise
        )

    def loss(self, x0, y):
        """
        Predict noise ? at a random timestep t, conditioned on label y.
        With CFG on, randomly replace y with NULL_CLASS_ID for a fraction of the batch.
        """
        b = x0.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=DEVICE).long()
        x_t, noise = self.q_sample(x0, t)

        if USE_CFG:
            # randomly drop labels -> null conditioning
            drop_mask = (torch.rand(b, device=DEVICE) < CFG_DROPOUT_P)
            y_train = torch.where(drop_mask, torch.full_like(y, NULL_CLASS_ID), y)
        else:
            y_train = y

        noise_pred = self.model(x_t, t, y_train)
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def p_mean_var(self, x_t, t, y, y_null=None, cfg_scale=1.0):
        """
        Compute the mean and variance for p(x_{t-1}|x_t).
        If cfg_scale>1, do 2 forward passes (cond & null) to get CFG.
        """
        betas_t = extract(self.betas, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x_t.shape)

        if cfg_scale == 1.0 or not USE_CFG:
            eps = self.model(x_t, t, y)
        else:
            # classifier-free guidance: eps = eps_null + s * (eps_cond - eps_null)
            eps_null = self.model(x_t, t, y_null)
            eps_cond = self.model(x_t, t, y)
            eps = eps_null + cfg_scale * (eps_cond - eps_null)

        model_mean = sqrt_recip_alphas_t * (x_t - betas_t * eps / sqrt_one_minus_alphas_cumprod_t)
        posterior_var_t = extract(self.posterior_variance, t, x_t.shape)
        return model_mean, posterior_var_t

    @torch.no_grad()
    def sample(self, n, y=None, cfg_scale=1.0, img_channels=IN_CHANNELS, img_size=IMG_SIZE):
        """
        Generate 'n' samples conditioned on labels y (shape [n]).
        If y is None, sample random labels. If CFG is used, also create null labels.
        """
        self.model.eval()
        x_t = torch.randn(n, img_channels, img_size, img_size, device=DEVICE)

        if y is None:
            y = torch.randint(0, NUM_CLASSES, (n,), device=DEVICE)

        y_null = None
        if USE_CFG and cfg_scale != 1.0:
            y_null = torch.full_like(y, NULL_CLASS_ID)

        for step in reversed(range(self.timesteps)):
            t = torch.full((n,), step, device=DEVICE, dtype=torch.long)
            mean, var = self.p_mean_var(x_t, t, y, y_null, cfg_scale)
            if step > 0:
                noise = torch.randn_like(x_t)
                x_t = mean + torch.sqrt(var) * noise
            else:
                x_t = mean
        return x_t

# ==========================================
# 6) Data
# ==========================================
if DATASET == "CIFAR10":
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2.0 - 1.0),  # [-1, 1]
    ])
    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
elif DATASET == "MNIST":
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2.0 - 1.0),
    ])
    trainset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
else:
    raise ValueError("Unsupported dataset")

loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

# ==========================================
# 7) Train
# ==========================================
unet = UNetCond(in_ch=IN_CHANNELS, base_ch=BASE_CH, ch_mults=(1,2,4,4),
                attn_res=ATTN_RES, img_size=IMG_SIZE, time_emb_dim=256,
                num_classes=NUM_CLASSES, use_cfg=USE_CFG).to(DEVICE)
ddpm = DDPMCond(unet, timesteps=T).to(DEVICE)
opt = torch.optim.AdamW(ddpm.parameters(), lr=LR)

for epoch in range(1, EPOCHS + 1):
    unet.train()
    running = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}")
    for x, y in pbar:
        x = x.to(DEVICE, non_blocking=True)
        # CIFAR-10 returns labels directly; MNIST too.
        y = y.to(DEVICE, non_blocking=True).long()

        loss = ddpm.loss(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        running += loss.item()
        pbar.set_postfix(loss=f"{running / (pbar.n or 1):.4f}")
    print(f"[Epoch {epoch}] mean loss: {running / len(loader):.4f}")

# ==========================================
# 8) Sample per-class grid & save
# ==========================================
@torch.no_grad()
def to_image_range(x):  # [-1,1] -> [0,1]
    return (x.clamp(-1, 1) + 1) * 0.5

# Make 10 classes, 4 images each (total 40)
n_per_class = 4
labels = torch.arange(NUM_CLASSES, device=DEVICE).repeat_interleave(n_per_class)
samples = ddpm.sample(n=len(labels), y=labels, cfg_scale=(CFG_SCALE if USE_CFG else 1.0),
                      img_channels=IN_CHANNELS, img_size=IMG_SIZE)
samples = to_image_range(samples)
torchvision.utils.save_image(samples, SAVE_PATH, nrow=n_per_class)
print(f"? Saved conditional samples to {SAVE_PATH}")
