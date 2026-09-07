# DDPM

이 단원은 깊은 만들어 내는 모델의 중요한 부품인 ddpm을 짠다. 이 짜기를 이해하면 요즘 만들어 내는 모델에 쓰이는 얼개의 결과 익히기 절차를 꿰뚫어 볼 수 있다. 이 코드는 연구와 실제 얼개에서 널리 쓰이는 쓸모 있는 재주를 보인다.

## 코드

```python
"""DDPM."""
# ==========================================
# 나아진 DDPM(Ho 외, 2020) - PyTorch 짜기
# ==========================================
# 바탕에 견준 핵심 개선:
#   1. 안정된 뽑기를 위한 지수 이동 평균(EMA)
#   2. 익히기의 안정을 위한 기울기 자르기
#   3. 배움 빠르기 몸 풀기 + 코사인 줄이기 차례표
#   4. 늘린 모델 담이(BASE_CH=128)
#   5. 여러 해상도 눈길(8, 16)
#   6. 층마다 남은 덩이를 더 둠(2 대신 3)
#   7. 오래 익힐 때를 위한 되짚을 자리 갈무리
#   8. 지켜보려 익히는 동안 표본 만들기
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
# 짜임새(이 매개변수를 고친다)
# ==========================================
class Config:
    # 자료 묶음 자리매김
    DATASET = "MNIST"           # "CIFAR10"이나 "MNIST"
    IMG_SIZE = 32               # 두 자료 묶음 모두 32x32으로 크기를 바꾼다
    
    # 익히기 자리매김
    BATCH_SIZE = 128
    LR = 1e-4                   # 안정을 위해 낮춘 배움 빠르기
    EPOCHS = 100                # 10에서 늘렸다 - 품질에 결정적이다
    WARMUP_STEPS = 1000         # 배움 빠르기 몸 풀기 걸음
    GRAD_CLIP = 1.0             # 기울기 자르기 문턱값
    
    # 퍼짐 자리매김
    T = 1000                    # 퍼짐 때 걸음 수
    
    # 모델 얼개
    BASE_CH = 128               # 담이를 늘리려 64에서 늘렸다
    CH_MULTS = (1, 2, 4, 4)     # 층마다 채널 갑절
    NUM_RES_BLOCKS = 3          # 층마다 남은 덩이(예전에는 2)
    ATTN_RES = {8, 16}          # 이 해상도에서 눈길을 쓴다
    TIME_EMB_DIM = 256          # 때 걸음 박아 넣기 차원
    DROPOUT = 0.1               # 정칙화를 위한 드롭아웃
    
    # 지수 이동 평균 자리매김
    EMA_DECAY = 0.9999          # 지수 이동 평균 줄임 비율
    EMA_START = 2000            # 이만큼 걸은 뒤 지수 이동 평균을 시작한다
    
    # 기록하고 갈무리하기
    SAMPLE_INTERVAL = 10        # N바퀴마다 표본을 만든다
    SAVE_INTERVAL = 20          # N바퀴마다 되짚을 자리를 갈무리한다
    NUM_SAMPLES = 16            # 만들 표본의 수
    
    # 얼개 자리매김
    NUM_WORKERS = 0             # DataLoader 일꾼(맞물림을 위해 0)
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
# 1) 잡음 차례표 도구
# ==========================================
@torch.no_grad()
def cosine_beta_schedule(timesteps, s=0.008):
    """
    "나아진 DDPM"(Nichol와 Dhariwal, 2021)이 내놓은 코사인 차례표.
    선형 차례표에 견주어 더 매끄러운 잡음 수준을 준다.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 1e-8, 0.999)


def extract(a, t, x_shape):
    """
    텐서 'a'의 어깨수 't'에서 값을 뽑고 퍼뜨리기에 맞게 꼴을 바꾼다.
    a: [T] tensor of values
    t: [B] tensor of indices
    x_shape: 퍼뜨리기의 목표 꼴
    """
    out = a.gather(-1, t).float()
    while out.ndim < len(x_shape):
        out = out[..., None]
    return out


# ==========================================
# 2) 사인 꼴 때 걸음 박아 넣기
# ==========================================
class SinusoidalPosEmb(nn.Module):
    """
    때 걸음의 사인 꼴 자리 박아 넣기.
    사인과 코사인으로 낱값 때 걸음을 차원 높은 박아 넣기로 옮긴다.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        t = t.float()[:, None]  # [B, 1]
        
        # 잦기 띠(지수로 벌린 것)
        freqs = torch.exp(
            torch.arange(half, device=device).float() * -(math.log(10000) / (half - 1))
        )
        args = t * freqs[None, :]  # [B, half]
        
        # 사인과 코사인 박아 넣기를 이어 붙인다
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, dim]
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


# ==========================================
# 3) 벽돌: 남은 덩이, 눈길, 키우기/줄이기
# ==========================================
class ResidualBlock(nn.Module):
    """
    때 박아 넣기로 조건을 준 남은 덩이.
    GroupNorm, SiLU 깨움, 그리고 쓸 수 있는 떨구기를 쓴다.
    때 박아 넣기가 잣수와 옮김(FiLM)으로 특징을 조절한다.
    """
    def __init__(self, in_ch, out_ch, time_emb_dim, dropout=0.1, groups=8):
        super().__init__()
        # 첫 겹말기 길
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        # 때 박아 넣기 쏘기(잣수와 옮김)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch * 2)
        )

        # 떨구기를 곁들인 둘째 겹말기 길
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        # 건너뛰기 이음(채널이 다르면 항등이나 1x1 겹말기)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        # 첫 겹말기
        h = self.conv1(self.act1(self.norm1(x)))
        
        # FiLM(특징마다 선형 조절)으로 때 조건 주기
        scale, shift = self.time_mlp(t_emb).chunk(2, dim=1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]
        h = self.norm2(h)
        h = h * (1 + scale) + shift
        
        # 떨구기를 곁들인 둘째 겹말기
        h = self.conv2(self.dropout(self.act2(h)))
        
        return h + self.skip(x)


class SelfAttention2d(nn.Module):
    """
    2차원 특징 지도의 여러 머리 스스로 눈길.
    멀리 떨어진 공간의 매임을 담는다.
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
        
        # 눈길의 잣수 갑절
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        b, c, h, w = x.shape
        
        # 고르게 맞추고 Q, K, V을 셈한다
        x_norm = self.norm(x)
        q = self.q(x_norm).reshape(b, self.num_heads, self.head_dim, h * w)
        k = self.k(x_norm).reshape(b, self.num_heads, self.head_dim, h * w)
        v = self.v(x_norm).reshape(b, self.num_heads, self.head_dim, h * w)
        
        # 눈길: softmax(Q @ K^T / sqrt(d)) @ V
        attn = torch.einsum("bhcn,bhcm->bhnm", q, k) * self.scale
        attn = attn.softmax(dim=-1)
        
        # 값에 어텐션 적용
        out = torch.einsum("bhnm,bhcm->bhcn", attn, v)
        out = out.reshape(b, c, h, w)
        out = self.proj(out)
        
        return out + x  # 잔차 연결


class Downsample(nn.Module):
    """성큼 겹말기로 공간을 2배 줄인다."""
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """가장 가까운 값 사이 메우기 + 겹말기로 공간을 2배 키운다."""
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


# ==========================================
# 4) U-Net 얼개
# ==========================================
class UNet(nn.Module):
    """
    잡음 헤아리기를 위한 U-Net 등뼈.
    
    구조:
    - 부호기: 남은 덩이와 눈길을 갖춘 줄이는 길
    - 병목: 눈길을 갖춘 가운데 덩이
    - 풀개: 건너뛰기 이음을 갖춘 키우는 길
    
    특징:
    - 남은 덩이마다 때 박아 넣기로 조건 주기
    - 정한 해상도에서의 스스로 눈길
    - 해상도 층마다 남은 덩이 여럿
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

        # 때 박아 넣기 여러 층 신경망
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        # ========== 부호기(줄이는 길) ==========
        self.downs = nn.ModuleList()
        ch = base_ch
        res = img_size
        self.skip_channels = []  # 건너뛰기 이음을 위해 채널을 좇는다

        for i, mult in enumerate(ch_mults):
            out_ch = base_ch * mult
            level_blocks = nn.ModuleList()
            
            # 층마다 남은 덩이 여럿
            for j in range(num_res_blocks):
                block_in_ch = ch if j == 0 else out_ch
                level_blocks.append(ResidualBlock(block_in_ch, out_ch, time_emb_dim, dropout))
            
            # 정한 해상도에서의 눈길
            attn = SelfAttention2d(out_ch) if res in attn_res else nn.Identity()
            
            # 줄이기(마지막 층만 빼고)
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

        # ========== 병목(가운데) ==========
        self.mid = nn.ModuleDict({
            "block1": ResidualBlock(ch, ch, time_emb_dim, dropout),
            "attn": SelfAttention2d(ch),
            "block2": ResidualBlock(ch, ch, time_emb_dim, dropout),
        })

        # ========== 풀개(키우는 길) ==========
        self.ups = nn.ModuleList()
        
        for i, mult in reversed(list(enumerate(ch_mults))):
            out_ch = base_ch * mult
            skip_ch = self.skip_channels.pop()
            level_blocks = nn.ModuleList()
            
            # 첫 덩이가 이어 붙인 특징(지금 것 + 건너뛴 것)을 받는다
            for j in range(num_res_blocks):
                if j == 0:
                    block_in_ch = ch + skip_ch
                else:
                    block_in_ch = out_ch
                level_blocks.append(ResidualBlock(block_in_ch, out_ch, time_emb_dim, dropout))
            
            # 정한 해상도에서의 눈길
            attn = SelfAttention2d(out_ch) if res in attn_res else nn.Identity()
            
            # 키우기(첫 층만 빼고)
            up = Upsample(out_ch) if i != 0 else nn.Identity()

            self.ups.append(nn.ModuleDict({
                "blocks": level_blocks,
                "attn": attn,
                "up": up,
            }))

            ch = out_ch
            if i != 0:
                res *= 2

        # 출력 사영
        self.out_norm = nn.GroupNorm(8, ch)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(ch, in_ch, 3, padding=1)

    def forward(self, x, t):
        # 때 박아 넣기
        t_emb = self.time_emb(t)

        # ========== 부호기 ==========
        h = self.in_conv(x)
        skips = []
        
        for level in self.downs:
            for block in level["blocks"]:
                h = block(h, t_emb)
            h = level["attn"](h)
            skips.append(h)
            h = level["down"](h)

        # ========== 병목 ==========
        h = self.mid["block1"](h, t_emb)
        h = self.mid["attn"](h)
        h = self.mid["block2"](h, t_emb)

        # ========== 풀개 ==========
        for level in self.ups:
            skip = skips.pop()
            
            # 크기가 어긋날 수 있으니 다룬다(h에 맞게 건너뛴 것을 가운데로 자른다)
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

        # 내놓기
        h = self.out_conv(self.out_act(self.out_norm(h)))
        return h


# ==========================================
# 5) DDPM 알맹이: 앞/뒤 과정
# ==========================================
class DDPM(nn.Module):
    """
    잡음 없애는 퍼짐 확률 모델.
    
    익히기: 자료에 잡음을 더하고(앞 과정) 그 잡음을 헤아리는 법을 배운다.
    뽑기: 순수 잡음에서 시작해 거듭 잡음을 없앤다(뒤 과정).
    """
    def __init__(self, model, timesteps, device):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.device = device

        # 퍼짐 매개변수를 미리 셈한다
        betas = cosine_beta_schedule(timesteps).to(device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1, device=device), alphas_cumprod[:-1]], dim=0
        )

        # 버퍼로 등록한다(모델과 함께 갈무리되고 .to()으로 함께 옮겨진다)
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
        앞 과정: 깨끗한 자료에 잡음을 더한다.
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
        뒤 과정: 잡음 없애기 걸음 하나.
        예측한 잡음으로 셈한 p(x_{t-1} | x_t).
        """
        betas_t = extract(self.betas, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_t.shape
        )
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x_t.shape)

        # 잡음을 헤아린다
        eps_theta = self.model(x_t, t)
        
        # p(x_{t-1} | x_t)의 평균을 셈한다
        model_mean = sqrt_recip_alphas_t * (
            x_t - betas_t * eps_theta / sqrt_one_minus_alphas_cumprod_t
        )

        # t=0에서는 잡음이 없다
        if (t == 0).all():
            return model_mean

        # 사후 흩어짐으로 잣수를 맞춘 잡음을 더한다
        posterior_var_t = extract(self.posterior_variance, t, x_t.shape)
        noise = torch.randn_like(x_t)
        return model_mean + torch.sqrt(posterior_var_t) * noise

    @torch.no_grad()
    def sample(self, n, img_channels, img_size, show_progress=True):
        """
        온전한 뒤 과정을 돌려 표본을 만든다.
        """
        self.model.eval()
        
        # 순수 잡음에서 시작한다
        x_t = torch.randn(n, img_channels, img_size, img_size, device=self.device)
        
        # 거듭 잡음을 없앤다
        timesteps = reversed(range(self.timesteps))
        if show_progress:
            timesteps = tqdm(timesteps, desc="Sampling", total=self.timesteps)
        
        for step in timesteps:
            t = torch.full((n,), step, device=self.device, dtype=torch.long)
            x_t = self.p_sample(x_t, t)
        
        return x_t

    def loss(self, x0):
        """
        익히기 손실: 헤아린 잡음과 실제 잡음 사이의 평균 제곱 어긋남.
        """
        b = x0.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=self.device).long()
        x_t, noise = self.q_sample(x0, t)
        noise_pred = self.model(x_t, t)
        return F.mse_loss(noise_pred, noise)


# ==========================================
# 6) 지수 이동 평균(EMA)
# ==========================================
class EMA:
    """
    모델 매개변수의 지수 이동 평균을 지킨다.
    안정되고 품질 높은 만들어 내기에 결정적이다.
    """
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for param in self.shadow.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        """지수 이동 평균 매개변수를 고친다."""
        for ema_param, model_param in zip(self.shadow.parameters(), model.parameters()):
            ema_param.data.mul_(self.decay).add_(model_param.data, alpha=1 - self.decay)

    def forward(self, *args, **kwargs):
        """추론에 지수 이동 평균 모델을 쓴다."""
        return self.shadow(*args, **kwargs)


# ==========================================
# 7) 배움 빠르기 차례표
# ==========================================
class WarmupCosineSchedule:
    """
    선형 몸 풀기 뒤 코사인 줄이기가 오는 배움 빠르기 차례표.
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
            # 선형 워밍업
            return self.base_lr * self.current_step / self.warmup_steps
        else:
            # 코사인 감쇠
            progress = (self.current_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))


# ==========================================
# 8) 자료 묶음 불러오기
# ==========================================
def build_dataloader(config):
    """CIFAR10이나 MNIST의 익히기 자료 불러개를 세운다."""
    torch.manual_seed(config.SEED)

    if config.DATASET == "CIFAR10":
        transform = transforms.Compose([
            transforms.Resize(config.IMG_SIZE),
            transforms.RandomHorizontalFlip(),  # 자료 불리기
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
        drop_last=True,  # 묶음 크기 문제를 피한다
    )
    return loader


# ==========================================
# 9) 도구 함수
# ==========================================
@torch.no_grad()
def to_image_range(x):
    """그림으로 보이려고 [-1, 1]을 [0, 1]로 바꾼다."""
    return (x.clamp(-1, 1) + 1) * 0.5


def save_samples(ddpm, ema, config, epoch, use_ema=True):
    """보기 그림을 만들어 갈무리한다."""
    model_to_use = ema.shadow if use_ema and ema is not None else ddpm.model
    
    # 뽑기를 위해 잠시 모델을 바꾼다
    original_model = ddpm.model
    ddpm.model = model_to_use
    
    samples = ddpm.sample(
        n=config.NUM_SAMPLES,
        img_channels=config.IN_CHANNELS,
        img_size=config.IMG_SIZE,
        show_progress=False
    )
    
    # 원래 모델 복원
    ddpm.model = original_model
    
    samples = to_image_range(samples)
    save_path = os.path.join(config.SAVE_DIR, f"ddpm_samples_epoch_{epoch}.png")
    torchvision.utils.save_image(samples, save_path, nrow=4)
    print(f"  -> Saved samples to {save_path}")


def save_checkpoint(ddpm, ema, optimizer, scheduler, epoch, config):
    """익히기 되짚을 자리를 갈무리한다."""
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
# 10) 으뜸 익히기 되풀이
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

    # 난수 씨앗 고정
    torch.manual_seed(config.SEED)

    # 자료 불러개를 세운다
    loader = build_dataloader(config)
    total_steps = config.EPOCHS * len(loader)

    # 모형을 세운다
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

    # 매개변수 개수 세기
    num_params = sum(p.numel() for p in unet.parameters())
    print(f"Model parameters: {num_params:,}")

    # DDPM 감싸개를 세운다
    ddpm = DDPM(unet, timesteps=config.T, device=config.DEVICE).to(config.DEVICE)

    # 최적화기
    optimizer = torch.optim.AdamW(ddpm.parameters(), lr=config.LR, weight_decay=0.01)

    # 학습률 스케줄러
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=config.WARMUP_STEPS,
        total_steps=total_steps,
    )

    # 지수 이동 평균(익히기 걸음을 얼마간 지난 뒤 첫자리매김)
    ema = None
    global_step = 0

    # 학습 루프
    for epoch in range(1, config.EPOCHS + 1):
        unet.train()
        running_loss = 0.0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{config.EPOCHS}")
        for x, _ in pbar:
            x = x.to(config.DEVICE, non_blocking=True)
            
            # 순전파
            loss = ddpm.loss(x)
            
            # 역전파
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            
            # 기울기 자르기
            torch.nn.utils.clip_grad_norm_(ddpm.parameters(), config.GRAD_CLIP)
            
            # 가장 좋게 하개 걸음
            optimizer.step()
            
            # 배움 빠르기 차례표 걸음
            scheduler.step()
            
            # 지수 이동 평균을 첫자리매김하거나 고친다
            global_step += 1
            if global_step == config.EMA_START:
                print(f"  -> Initializing EMA at step {global_step}")
                ema = EMA(unet, decay=config.EMA_DECAY)
            elif global_step > config.EMA_START and ema is not None:
                ema.update(unet)
            
            # 기록
            running_loss += loss.item()
            current_lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix(
                loss=f"{running_loss / (pbar.n or 1):.4f}",
                lr=f"{current_lr:.2e}"
            )

        avg_loss = running_loss / len(loader)
        print(f"[Epoch {epoch}] Mean loss: {avg_loss:.4f}")

        # 이따금 표본을 만든다
        if epoch % config.SAMPLE_INTERVAL == 0 or epoch == 1:
            save_samples(ddpm, ema, config, epoch, use_ema=(ema is not None))

        # 이따금 되짚을 자리를 갈무리한다
        if epoch % config.SAVE_INTERVAL == 0:
            save_checkpoint(ddpm, ema, optimizer, scheduler, epoch, config)

    # 마지막 표본과 되짚을 자리
    print("\nTraining complete! Generating final samples...")
    save_samples(ddpm, ema, config, config.EPOCHS, use_ema=(ema is not None))
    save_checkpoint(ddpm, ema, optimizer, scheduler, config.EPOCHS, config)

    # 본디 이름 약속으로 표본 파일도 갈무리한다
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
```

## 논의

이 짜기는 ddpm에 대해 자리 잡은 가장 좋은 방식을 따른다. 코드는 모델 뜻매김, 익히기 논리, 도구 함수를 또렷이 갈라 짜였다. 핵심 설계 결정에는 깨움 함수 고르기, 고르게 맞추기 방책, 가장 좋게 하기 웃매개변수가 들며 모두 익히기의 안정과 내놓기 품질에 크게 영향을 준다.

이 얼개는 깊은 만들어 내는 모델에 흔한 중요한 결 여럿을 보인다. 곧 여러 신경망 층을 지나며 특징을 차츰 다루기, 모델이 곁 앎을 받아들이게 하는 조건 주기 얼개, 익히는 동안 기울기가 안정되게 흐르도록 하는 꼼꼼한 첫자리매김이다.

새 자료 묶음이나 문제 마당에서는 웃매개변수 고르기와 익히기 절차를 꼼꼼히 맞추어야 할 때가 많으므로 다루는 이들은 이에 마음을 써야 한다. 코드가 조각으로 나뉘어 있어 다른 얼개, 손실 함수, 익히기 방책을 실험하기 쉽다.

## 연습문제

**연습문제 1.**
구체적인 들임 텐서로 이 단원의 으뜸 모델의 앞먹임을 좇아라. 층마다 꼴이 어떻게 바뀌는지 적고 내놓기 차원이 바라던 것과 맞는지 확인하라.

??? success "연습문제 1 풀이"
    들임 텐서에서 시작해 층마다 바뀜을 따라가라. 겹말기 층에서는 공간 차원에 공식 $H_{out} = \lfloor(H_{in} + 2p - k) / s\rfloor + 1$을 쓴다. 선형 층에서는 특징 차원의 바뀜을 좇는다. 중간 꼴을 하나씩 적고 마지막 내놓기가 그 일(그림 만들어 내기, 가르기 등)에 바라던 목표 차원과 맞는지 확인하라.

---

**연습문제 2.**
이 짜기의 핵심 웃매개변수(배움 빠르기, 묶음 크기, 얼개 고르기)를 가려내라. 다른 것을 붙박아 두고 하나씩 바꾸어 웃매개변수마다 익히기가 얼마나 민감한지 재는 실험을 짜라.

??? success "연습문제 2 풀이"
    핵심 웃매개변수에는 배움 빠르기(흔히 $10^{-4}$에서 $10^{-3}$), 묶음 크기(64-256), 층과 채널의 수, 깨움 함수가 든다. 웃매개변수마다 값을 3~5가지로 바꾸어 모델을 익히고 알맞은 잣대(손실, 표본 품질, 모이는 빠르기)를 좇아라. 결과를 그려 어느 웃매개변수가 가장 큰 영향을 주는지 가려내라. 흔히 배움 빠르기와 얼개 깊이가 가장 세게 영향을 주고, 묶음 크기는 알맞은 범위 안에서는 웬만큼 영향을 준다.

---

**연습문제 3.**
이 짜기에 새 기능을 더해 넓혀라. 곧 기울기 자르기, 배움 빠르기 차례표, 다른 손실 함수를 더하라. 고치기 앞뒤의 익히기 움직임을 견주어라.

??? success "연습문제 3 풀이"
    기울기 자르기는 `optimizer.step()` 앞에 `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`을 더한다. 배움 빠르기 차례표는 `torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)`을 쓰고 바퀴마다 `scheduler.step()`을 부른다. 익히기 손실 곡선, 모이는 빠르기, 마지막 모델 품질을 견주어라. 기울기 자르기는 흔히 익히기가 치솟는 것을 막고, 코사인 식히기는 뒤 바퀴에서 더 곱게 가장 좋게 하여 마지막 솜씨를 높일 수 있다.
