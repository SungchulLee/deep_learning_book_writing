# DDIM 빠른 뽑기

이 단원은 요즘 만들어 내는 모델의 핵심 부품인 ddim 빠른 뽑기을 짠다. 여기서 보이는 개념과 재주를 알면 퍼짐 모델과 점수 바탕 만들어 내는 방법을 다루는 데 꼭 필요한 앎을 얻는다. 이 짜기는 또렷함과 실제 쓸모의 균형을 맞추어 배우기에도 실험하기에도 알맞다.

## 코드

```python
"""DDIM 빠른 뽑기."""
# ==========================================
# DDIM(잡음 없애는 은근한 퍼짐 모델)
# - 더 빠른 뽑기: 1000 대신 50걸음
# - 정해진 뽑기(아무 잡음 없음)
# - Song 외, 2021
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
BATCH_SIZE = 128
LR = 2e-4
EPOCHS = 5
T = 1000              # 익히기 걸음
DDIM_STEPS = 50       # 뽑기 걸음(훨씬 빠르다!)
DDIM_ETA = 0.0        # 0=정해짐, 1=DDPM 같은 확률
BASE_CH = 64
ATTN_RES = {16}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH = "ddim_samples.png"
SEED = 42

# ==========================================
# 1) 도구
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
# 2) 벽돌(DDPM과 같다)
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
# 4) DDIM 단원
# ==========================================
class DDIM(nn.Module):
    """
    DDIM: 걸음을 줄여 정해진 뽑기를 할 수 있게 한다.
    핵심 통찰: 뽑는 동안 때 걸음을 건너뛸 수 있다!
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
        DDIM 뽑기: 더 빨리 만들려 때 걸음의 일부만 쓴다.
        
        eta: 확률성을 다스린다
            - eta=0: deterministic (same seed → same image)
            - eta=1: equivalent to DDPM (stochastic)
        """
        self.model.eval()
        
        # 때 걸음의 부분 차례를 만든다
        # 1000걸음을 다 쓰지 않고 ddim_steps(예컨대 50)만 쓴다
        c = self.timesteps // ddim_steps
        ddim_timesteps = torch.arange(0, self.timesteps, c, device=DEVICE).long()
        ddim_timesteps_prev = torch.cat([torch.tensor([0], device=DEVICE), ddim_timesteps[:-1]])

        # 순수 잡음에서 시작한다
        x = torch.randn(n, img_channels, img_size, img_size, device=DEVICE)

        # 뒤 과정
        for i in reversed(range(len(ddim_timesteps))):
            t = torch.full((n,), ddim_timesteps[i], device=DEVICE, dtype=torch.long)
            t_prev = torch.full((n,), ddim_timesteps_prev[i], device=DEVICE, dtype=torch.long)

            # 알파 값을 뽑는다
            alpha_t = extract(self.alphas_cumprod, t, x.shape)
            alpha_t_prev = extract(self.alphas_cumprod, t_prev, x.shape)

            # 잡음을 헤아린다
            eps = self.model(x, t)

            # 잡음 섞인 x_t에서 x0을 헤아린다
            pred_x0 = (x - torch.sqrt(1 - alpha_t) * eps) / torch.sqrt(alpha_t)
            pred_x0 = pred_x0.clamp(-1, 1)  # 안정을 위해 자른다

            # x_t을 가리키는 방향
            dir_xt = torch.sqrt(1 - alpha_t_prev - eta**2 * (1 - alpha_t) / (1 - alpha_t_prev) * (1 - alpha_t_prev)) * eps

            # 아무 잡음(eta > 0이면)
            noise = torch.randn_like(x) if eta > 0 else 0

            # DDIM 고침 규칙
            x = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt + eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t) * noise

        return x

# ==========================================
# 5) 자료
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
# 6) 익히기
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

    # DDIM으로 빠르게 뽑기
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
    main()```

## 논의

ddim 빠른 뽑기의 짜기는 이 마당에 자리 잡은 방식을 따른다. 코드 짜임이 모델 뜻매김과 익히기 논리를 갈라 놓아 부품을 하나씩 고치기 쉽다. 얼개 고르기는 만들어 내는 모델 무리가 많은 실험에서 얻은 배움을 담고 있다.

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
