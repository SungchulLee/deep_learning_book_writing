# 조건 DDPM

이 단원은 요즘 만들어 내는 모델의 핵심 부품인 조건 ddpm을 짠다. 여기서 보이는 개념과 재주를 알면 퍼짐 모델과 점수 바탕 만들어 내는 방법을 다루는 데 꼭 필요한 앎을 얻는다. 이 짜기는 또렷함과 실제 쓸모의 균형을 맞추어 배우기에도 실험하기에도 알맞다.

## 코드

```python
"""조건 DDPM."""
# ============================================================
# 갈래 조건 DDPM(CIFAR-10) — PyTorch, 주석을 잘 단 코드
# - 남은 덩이와 (쓸 수 있는) 눈길을 갖춘 U-Net
# - 사인 꼴 때 걸음 박아 넣기
# - 이름표 조건 주기(때 박아 넣기에 박아 넣기를 더함)
# - 이름표 떨구기로 가름개 없는 이끌기를 쓸 수 있음
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
# 짜임새(여기서 고친다)
# -------------------
DATASET = "CIFAR10"      # "CIFAR10"이나 "MNIST"
NUM_CLASSES = 10         # CIFAR-10에는 갈래가 10개 있다
IN_CHANNELS = 3 if DATASET == "CIFAR10" else 1
IMG_SIZE = 32 if DATASET == "CIFAR10" else 28
BATCH_SIZE = 128
LR = 2e-4
EPOCHS = 5
T = 1000                 # 옛부터의 DDPM은 1000걸음을 쓴다
BASE_CH = 64             # 모델 너비. 품질을 위해 늘린다(예컨대 96/128)
ATTN_RES = {16}          # 이 공간 크기에서 눈길을 쓴다. 끄려면 set()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH = "ddpm_conditional_samples.png"

# ------- 가름개 없는 이끌기(CFG) -------
USE_CFG = True           # 끄려면 False으로 둔다
CFG_DROPOUT_P = 0.1      # 익힐 때 이 확률로 이름표를 떨군다
CFG_SCALE = 3.0          # 뽑을 때의 이끌기 세기(1.0 = 끔)
NULL_CLASS_ID = NUM_CLASSES  # "빈" 조건을 위해 남겨 둔 어깨수

# ==========================================
# 1) 퍼짐 차례표와 도우미
# ==========================================
@torch.no_grad()
def cosine_beta_schedule(timesteps, s=0.008):
    """더 매끄러운 익히기를 위한 코사인 차례표(Nichol와 Dhariwal 2021)."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 1e-8, 0.999)

def extract(a, t, x_shape):
    """a[t]을 모아 x_shape에 퍼뜨릴 수 있게 꼴을 바꾼다."""
    out = a.gather(-1, t).float()
    while out.ndim < len(x_shape):
        out = out[..., None]
    return out

# ==========================================
# 2) 사인 꼴 때 걸음 박아 넣기
# ==========================================
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: [B] 정수 때 걸음
        device = t.device
        half = self.dim // 2
        t = t.float()[:, None]              # [B,1]
        freqs = torch.exp(
            torch.arange(half, device=device).float() * -(math.log(10000) / (half - 1))
        )
        args = t * freqs[None, :]           # [B, half]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, dim이나 dim-1]
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0,1))
        return emb

# ==========================================
# 3) U-Net 벽돌
# ==========================================
class ResidualBlock(nn.Module):
    """GroupNorm + SiLU + FiLM 때 조건 주기를 갖춘 남은 덩이."""
    def __init__(self, in_ch, out_ch, time_emb_dim, groups=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.act1  = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch * 2)  # 잣수와 옮김
        )

        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.act2  = nn.SiLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        # 첫 겹말기
        h = self.conv1(self.act1(self.norm1(x)))

        # 때 박아 넣기에서 오는 FiLM 조건 주기
        scale, shift = self.time_mlp(t_emb).chunk(2, dim=1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]

        h = self.norm2(h)
        h = h * (1 + scale) + shift
        h = self.conv2(self.act2(h))
        return h + self.skip(x)

class SelfAttention2d(nn.Module):
    """주어진 해상도에서의 가벼운 스스로 눈길."""
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
# 4) 조건 U-Net
# ==========================================
class UNetCond(nn.Module):
    """
    조건 U-Net.
    조건 주기: 갈래 이름표 -> 박아 넣기 -> 때 박아 넣기에 더함.
    가름개 없는 이끌기에서는 "빈" 이름표를 위해 박아 넣기 자리를 하나 더 둔다.
    """
    def __init__(self, in_ch=IN_CHANNELS, base_ch=BASE_CH, ch_mults=(1,2,4,4),
                 attn_res=ATTN_RES, img_size=IMG_SIZE, time_emb_dim=256,
                 num_classes=NUM_CLASSES, use_cfg=USE_CFG):
        super().__init__()
        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        # 때 박아 넣기 여러 층 신경망
        self.time_sinus = SinusoidalPosEmb(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        # 이름표 박아 넣기(CFG이면 빈 토큰을 위해 +1)
        label_vocab = num_classes + (1 if use_cfg else 0)
        self.label_emb = nn.Embedding(label_vocab, time_emb_dim)

        # 부호기
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

        # 가운데
        self.mid = nn.ModuleList([
            ResidualBlock(ch, ch, time_emb_dim),
            SelfAttention2d(ch),
            ResidualBlock(ch, ch, time_emb_dim)
        ])

        # 복호기
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
        y: [B] 갈래 이름표(CFG 빈 조건이면 NUM_CLASSES 또는 NULL_CLASS_ID)
        """
        # 때 박아 넣기를 세우고 이름표 박아 넣기를 더한다
        t_emb = self.time_mlp(self.time_sinus(t))  # [B, D]
        y_emb = self.label_emb(y)                  # [B, D]
        t_emb = t_emb + y_emb                      # 단순하고 잘 듣는 조건 주기

        # 부호기
        h = self.in_conv(x)
        skips = []
        for m in self.encoder:
            if isinstance(m, ResidualBlock):
                h = m(h, t_emb)
                skips.append(h)
            else:
                h = m(h)

        # 가운데
        for m in self.mid:
            if isinstance(m, ResidualBlock):
                h = m(h, t_emb)
            else:
                h = m(h)

        # 건너뛰는 이음을 갖춘 풀개
        for m in self.decoder:
            if isinstance(m, ResidualBlock):
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = m(h, t_emb)
            else:
                h = m(h)

        return self.out_conv(self.out_act(self.out_norm(h)))

# ==========================================
# 5) 조건 주기 + CFG를 갖춘 DDPM 알맹이
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
        이름표 y을 조건으로 삼아 아무 때 걸음 t에서 잡음을 헤아린다.
        가름개 없는 이끌기를 켜면 묶음의 일부에서 y을 NULL_CLASS_ID로 아무렇게나 바꾼다.
        """
        b = x0.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=DEVICE).long()
        x_t, noise = self.q_sample(x0, t)

        if USE_CFG:
            # 이름표를 마구잡이로 떨군다 -> 빈 조건
            drop_mask = (torch.rand(b, device=DEVICE) < CFG_DROPOUT_P)
            y_train = torch.where(drop_mask, torch.full_like(y, NULL_CLASS_ID), y)
        else:
            y_train = y

        noise_pred = self.model(x_t, t, y_train)
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def p_mean_var(self, x_t, t, y, y_null=None, cfg_scale=1.0):
        """
        p(x_{t-1}|x_t)의 평균과 흩어짐을 셈한다.
        cfg_scale>1이면 CFG를 얻으려고 앞으로 걸음을 두 번 한다(조건 있음과 빈 조건).
        """
        betas_t = extract(self.betas, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x_t.shape)

        if cfg_scale == 1.0 or not USE_CFG:
            eps = self.model(x_t, t, y)
        else:
            # 가름개 없는 이끌기: eps = eps_null + s * (eps_cond - eps_null)
            eps_null = self.model(x_t, t, y_null)
            eps_cond = self.model(x_t, t, y)
            eps = eps_null + cfg_scale * (eps_cond - eps_null)

        model_mean = sqrt_recip_alphas_t * (x_t - betas_t * eps / sqrt_one_minus_alphas_cumprod_t)
        posterior_var_t = extract(self.posterior_variance, t, x_t.shape)
        return model_mean, posterior_var_t

    @torch.no_grad()
    def sample(self, n, y=None, cfg_scale=1.0, img_channels=IN_CHANNELS, img_size=IMG_SIZE):
        """
        이름표 y(꼴 [n])를 조건으로 표본 'n'개를 만든다.
        y이 None이면 아무 이름표를 뽑는다. 가름개 없는 이끌기를 쓰면 빈 이름표도 만든다.
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
# 6) 자료
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
# 7) 익히기
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
        # CIFAR-10은 이름표를 곧바로 돌려준다. MNIST도 그렇다.
        y = y.to(DEVICE, non_blocking=True).long()

        loss = ddpm.loss(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        running += loss.item()
        pbar.set_postfix(loss=f"{running / (pbar.n or 1):.4f}")
    print(f"[Epoch {epoch}] mean loss: {running / len(loader):.4f}")

# ==========================================
# 8) 갈래마다 격자로 뽑아 갈무리하기
# ==========================================
@torch.no_grad()
def to_image_range(x):  # [-1,1] -> [0,1]
    return (x.clamp(-1, 1) + 1) * 0.5

# 갈래 10개를 만들고 갈래마다 그림 4장(모두 40장)
n_per_class = 4
labels = torch.arange(NUM_CLASSES, device=DEVICE).repeat_interleave(n_per_class)
samples = ddpm.sample(n=len(labels), y=labels, cfg_scale=(CFG_SCALE if USE_CFG else 1.0),
                      img_channels=IN_CHANNELS, img_size=IMG_SIZE)
samples = to_image_range(samples)
torchvision.utils.save_image(samples, SAVE_PATH, nrow=n_per_class)
print(f"? Saved conditional samples to {SAVE_PATH}")


if __name__ == "__main__":
    pass
```

## 논의

조건 ddpm의 짜기는 이 마당에 자리 잡은 방식을 따른다. 코드 짜임이 모델 뜻매김과 익히기 논리를 갈라 놓아 부품을 하나씩 고치기 쉽다. 얼개 고르기는 만들어 내는 모델 무리가 많은 실험에서 얻은 배움을 담고 있다.

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
