# 숨은 퍼짐

이 단원은 요즘 만들어 내는 모델의 핵심 부품인 숨은 퍼짐을 짠다. 여기서 보이는 개념과 재주를 알면 퍼짐 모델과 점수 바탕 만들어 내는 방법을 다루는 데 꼭 필요한 앎을 얻는다. 이 짜기는 또렷함과 실제 쓸모의 균형을 맞추어 배우기에도 실험하기에도 알맞다.

## 1. 코드

```python
"""숨은 퍼짐."""
# ==========================================
# 숨은 퍼짐 모델(LDM)
# - Stable Diffusion의 핵심 생각
# - 누른 숨은 공간에서의 퍼짐(화소 공간이 아님)
# - 훨씬 효율이 좋다: 익히기와 뽑기가 빠르다
# - Rombach 외, 2022
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
# 짜임새
# -------------------
DATASET = "CIFAR10"
IN_CHANNELS = 3 if DATASET == "CIFAR10" else 1
IMG_SIZE = 32 if DATASET == "CIFAR10" else 28
LATENT_DIM = 4          # 채널 4개로 누른다
LATENT_SCALE = 4        # 공간 줄이기 갑절
BATCH_SIZE = 128
LR = 2e-4
EPOCHS_VAE = 10         # 먼저 자기 부호기를 익힌다
EPOCHS_DIFF = 5         # 그런 다음 퍼짐을 익힌다
T = 1000
BASE_CH = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH = "latent_diffusion_samples.png"
SEED = 42

# ==========================================
# 1) 자기 부호기(그림을 숨은 값으로 누른다)
# ==========================================
class Encoder(nn.Module):
    """그림을 숨은 공간으로 누른다"""
    def __init__(self, in_ch=IN_CHANNELS, latent_dim=LATENT_DIM):
        super().__init__()
        # 단순한 부호기: 32x32 -> 8x8
        self.conv1 = nn.Conv2d(in_ch, 64, 3, stride=2, padding=1)      # /2
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)        # /2
        self.conv3 = nn.Conv2d(128, latent_dim * 2, 3, padding=1)      # mu과 logvar
        self.act = nn.SiLU()

    def forward(self, x):
        h = self.act(self.conv1(x))
        h = self.act(self.conv2(h))
        h = self.conv3(h)
        mu, logvar = h.chunk(2, dim=1)
        return mu, logvar

class Decoder(nn.Module):
    """숨은 공간에서 그림을 되짓는다"""
    def __init__(self, latent_dim=LATENT_DIM, out_ch=IN_CHANNELS):
        super().__init__()
        # 단순한 풀개: 8x8 -> 32x32
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
    """누른 나타냄을 배우는 변분 자기 부호기"""
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
# 2) 숨은 퍼짐 U-Net
#    (누른 숨은 공간에서 돈다)
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
    숨은 공간을 위한 단순한 U-Net.
    들임: 눌러 담은 숨은 값(보기: 4 x 8 x 8)
    내놓기: 숨은 공간에서 헤아린 잡음
    """
    def __init__(self, latent_dim=LATENT_DIM, base_ch=64, time_dim=128):
        super().__init__()
        
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim)
        )
        
        # 부호기
        self.in_conv = nn.Conv2d(latent_dim, base_ch, 3, padding=1)
        self.down1 = ResBlock(base_ch, base_ch * 2, time_dim)
        self.down2 = ResBlock(base_ch * 2, base_ch * 4, time_dim)
        
        # 가운데
        self.mid1 = ResBlock(base_ch * 4, base_ch * 4, time_dim)
        self.mid2 = ResBlock(base_ch * 4, base_ch * 4, time_dim)
        
        # 복호기
        self.up1 = ResBlock(base_ch * 8, base_ch * 2, time_dim)  # down2에서 이어 붙인다
        self.up2 = ResBlock(base_ch * 4, base_ch, time_dim)      # down1에서 이어 붙인다
        
        self.out_norm = nn.GroupNorm(8, base_ch)
        self.out_conv = nn.Conv2d(base_ch, latent_dim, 3, padding=1)

    def forward(self, z, t):
        t_emb = self.time_emb(t)
        
        # 부호기
        h = self.in_conv(z)
        h1 = self.down1(h, t_emb)
        h2 = self.down2(h1, t_emb)
        
        # 가운데
        h = self.mid1(h2, t_emb)
        h = self.mid2(h, t_emb)
        
        # 건너뛰는 이음을 갖춘 풀개
        h = self.up1(torch.cat([h, h2], dim=1), t_emb)
        h = self.up2(torch.cat([h, h1], dim=1), t_emb)
        
        return self.out_conv(F.silu(self.out_norm(h)))

# ==========================================
# 3) 숨은 퍼짐 과정
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
    """숨은 공간의 퍼짐 모델"""
    def __init__(self, vae, unet, timesteps=T):
        super().__init__()
        self.vae = vae
        self.unet = unet
        self.timesteps = timesteps
        
        # 퍼짐 익히기 동안 변분 자기 부호기를 얼린다
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
        """숨은 값 z0에 잡음을 더한다"""
        if noise is None:
            noise = torch.randn_like(z0)
        sqrt_alpha = extract(self.sqrt_alphas_cumprod, t, z0.shape)
        sqrt_one_minus_alpha = extract(self.sqrt_one_minus_alphas_cumprod, t, z0.shape)
        return sqrt_alpha * z0 + sqrt_one_minus_alpha * noise, noise

    def loss(self, x0):
        """익히기 손실: 숨은 값으로 부호화하고 잡음을 더한 뒤 잡음을 헤아린다"""
        # 숨은 공간으로 부호화한다
        with torch.no_grad():
            z0 = self.vae.encode(x0)
        
        # 아무 때 걸음을 뽑는다
        b = z0.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=DEVICE).long()
        
        # 잡음을 더하고 헤아린다
        zt, noise = self.q_sample(z0, t)
        noise_pred = self.unet(zt, t)
        
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def sample(self, n, img_size=IMG_SIZE):
        """표본 만들기: 숨은 공간에서 잡음을 없앤 뒤 푼다"""
        self.unet.eval()
        
        # 숨은 공간의 아무 잡음에서 시작한다
        latent_size = img_size // LATENT_SCALE
        z = torch.randn(n, LATENT_DIM, latent_size, latent_size, device=DEVICE)
        
        # 숨은 공간에서 잡음을 없앤다
        for step in reversed(range(self.timesteps)):
            t = torch.full((n,), step, device=DEVICE, dtype=torch.long)
            
            # 잡음을 헤아린다
            noise_pred = self.unet(z, t)
            
            # 잡음 없애기 걸음
            alpha_t = extract(self.alphas_cumprod, t, z.shape)
            alpha_prev = extract(self.alphas_cumprod, t - 1, z.shape) if step > 0 else 1.0
            beta_t = extract(self.betas, t, z.shape)
            
            # DDPM 공식
            z = (z - beta_t / torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(1 - beta_t)
            
            if step > 0:
                noise = torch.randn_like(z)
                z = z + torch.sqrt(beta_t) * noise
        
        # 숨은 값을 그림으로 푼다
        samples = self.vae.decode(z)
        return samples

# ==========================================
# 4) 자료
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
# 5) 익히기
# ==========================================
def train_vae(vae, loader, epochs):
    """먼저 자기 부호기를 익힌다"""
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
            
            # 되살림 손실
            recon_loss = F.mse_loss(recon, x, reduction='sum') / x.shape[0]
            
            # KL 벌어짐
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0]
            
            # 전체 손실
            loss = recon_loss + 0.001 * kl_loss  # 작은 쿨백-라이블러 무게
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            running_recon += recon_loss.item()
            running_kl += kl_loss.item()
            pbar.set_postfix(recon=f"{running_recon/(pbar.n or 1):.4f}", 
                           kl=f"{running_kl/(pbar.n or 1):.4f}")
        
        print(f"[VAE Epoch {epoch}] Recon: {running_recon/len(loader):.4f}, KL: {running_kl/len(loader):.4f}")

def train_latent_diffusion(ldm, loader, epochs):
    """그런 다음 숨은 공간에서 퍼짐을 익힌다"""
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
    
    # 모델 만들기
    vae = VAE().to(DEVICE)
    unet = LatentUNet(latent_dim=LATENT_DIM, base_ch=BASE_CH).to(DEVICE)
    
    # 먼저 자기 부호기를 익힌다
    train_vae(vae, loader, EPOCHS_VAE)
    
    # 숨은 퍼짐 모델을 만든다
    ldm = LatentDiffusion(vae, unet, timesteps=T).to(DEVICE)
    
    # 숨은 공간에서 퍼짐을 익힌다
    train_latent_diffusion(ldm, loader, EPOCHS_DIFF)
    
    # 표본 만들기
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
    main()```

## 2. 논의

숨은 퍼짐의 짜기는 이 마당에 자리 잡은 방식을 따른다. 코드 짜임이 모델 뜻매김과 익히기 논리를 갈라 놓아 부품을 하나씩 고치기 쉽다. 얼개 고르기는 만들어 내는 모델 무리가 많은 실험에서 얻은 배움을 담고 있다.

이 짜기의 핵심에는 수치의 안정을 꼼꼼히 다루기, 고르게 맞추기 재주를 제대로 쓰기, 효율 좋은 셈 결이 든다. 익히기 절차에는 잡음 차례표, 기울기 다루기, 이따금의 따지기가 들며 모두 품질 높은 결과를 내는 데 결정적이다.

이 단원은 이론의 개념이 실제 짜기로 어떻게 옮겨지는지 보이며 만들어 내는 모델의 더 넓은 틀과 이어진다. 여기서 보이는 재주는 만들어 내는 모델이 이룰 수 있는 것의 가장자리를 넓히는 더 앞선 변형과 넓힘을 이해하는 바탕이 된다.

## 연습문제

**연습문제 1.**
구체적인 자료 묶음으로 이 단원의 으뜸 셈을 좇아라. 큰 걸음마다 텐서 꼴을 적고 모든 차원이 서로 맞는지 확인하라.

??? success "연습문제 1 풀이"
    모델에 알맞은 꼴의 들임 묶음에서 시작한다. 층이나 함수 부르기마다 셈을 따라가며 바뀜 뒤 텐서 꼴을 적는다. 겹말기 층에서는 내놓기 차원 공식을 쓴다. 눈길 얼개에서는 물음, 열쇠, 값의 차원이 맞는지 확인한다. 마지막 내놓기 꼴이 바라던 목표 차원과 맞는지 굳힌다. 이 익힘은 자료가 얼개를 어떻게 흐르는지에 대한 직관을 쌓아 준다.

---

**연습문제 2.**
이 단원에 쓰인 손실 함수를 가려내고 모델 매개변수에 대한 기울기를 이끌어 내라. 왜 이 손실 함수가 이 일에 알맞은지 설명하라.

??? success "연습문제 2 풀이"
    손실 함수는 모델이 헤아린 값과 목표 사이의 어긋남을 잰다. 잡음 헤아리기에서는 평균 제곱 어긋남 손실 $\|\epsilon - \epsilon_\theta(x_t, t)\|^2$을 쓰는데, 이것이 로그 가능도의 변분 아래 한계에 맞물리기 때문이다. 매개변수 $\theta$에 대한 기울기는 $-2(\epsilon - \epsilon_\theta) \nabla_\theta \epsilon_\theta$이며 헤아림 어긋남을 줄이는 방향을 가리킨다. 이 손실을 가장 작게 하는 것이 퍼짐 모델에서 자료 로그 가능도의 아래 한계를 가장 크게 하는 것과 같으므로 알맞다.

---

**연습문제 3.**
다른 잡음 차례표를 받쳐 주도록 이 짜기를 고쳐라(예컨대 선형에서 코사인으로, 또는 그 반대로). 두 차례표의 익히기 움직임과 표본 품질을 견주어라.

??? success "연습문제 3 풀이"
    두 차례표를 모두 짜고 각각으로 모델을 익힌다. $\bar{\alpha}_t = \cos^2\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)$으로 뜻매김한 코사인 차례표는 선형 차례표 $\beta_t = \beta_{\min} + t(\beta_{\max} - \beta_{\min})/T$에 견주어 잡음이 더 매끄럽게 늘어난다. 손실 곡선을 좇고 일정한 사이마다 표본을 만든다. 코사인 차례표는 신호 대 잡음비가 더 완만하게 줄어 때 걸음에 걸쳐 배움 신호가 더 고르므로 흔히 더 좋은 결과를 낸다.

## 정리하며

**다룬 것** — 숨은 퍼짐

숨은 퍼짐의 짜기는 이 마당에 자리 잡은 방식을 따른다.

고갱이 갈래는 `Encoder`, `Decoder`, `VAE`, `SinusoidalPosEmb`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
