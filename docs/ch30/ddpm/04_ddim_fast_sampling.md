# DDIM으로 생성 단계 줄이기

DDIM은 DDPM과 같은 노이즈 예측 학습을 사용하면서 다른 생성 전이로 일부 시간 단계만 따라갈 수 있는 방법이다. η=0이면 초기 잠재값이 주어진 뒤의 경로가 결정적이고, 다른 설정에서는 잡음을 추가할 수 있다. 이 페이지는 생성 단계 선택과 전이식을 살펴보며 현재 구현의 종단 인덱스와 η>0 계수 문제를 구분한다.

## 1. 코드

```python
"""DDIM 빠른 뽑기."""
# ==========================================
# DDIM(잡음 없애는 은근한 확산 모델)
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
            - eta=0: 정해진 대로(같은 씨앗 → 같은 그림)
            - eta=1: DDPM과 같다(마구잡이)
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
    main()
```

## 2. 학습은 DDPM의 노이즈 회귀다

코드는 기본 CIFAR-10을 32×32 RGB로 읽고 −1~1로 정규화한다. 1,000개 코사인 β 스케줄로 노이즈 관측을 만들어 실제 ε와 예측의 원소 평균 MSE를 학습한다. DDIM이라는 클래스 이름이 별도의 50단계 학습 목적을 뜻하지는 않는다. 학습에서 t는 0~999 전체를 사용한다.

배치 128, 5에포크, AdamW 학습률 0.0002이고 마지막 80개를 포함해 에포크당 391번, 총 1,955번 갱신하도록 구성되어 있다. EMA·검증·체크포인트는 없다. 학습 손실 로그도 배치별 손실의 단순 평균이므로 마지막 작은 배치에 같은 가중치를 준다.

## 3. U-Net의 skip 선택 규칙

기본 인코더 크기는 32→16→8→4이고 채널은 64,128,256,256이다. 시간 조건 잔차 블록, 16×16 어텐션과 middle 어텐션을 사용한다. forward는 각 잔차 출력 뒤에 skip을 저장하지만, 생성자의 입력 폭 목록은 초기 특징과 downsample을 포함하는 다른 규칙이다.

이 구현은 디코더가 기대하는 채널 수보다 h가 작으면 skip을 계속 꺼내고, 공간 크기가 다르면 그 skip을 **버린다**. 기본 32×32에서는 아래처럼 채널을 맞출 수 있다.

- 4×4에서는 마지막 256채널 skip 하나를 붙인다.
- 8×8에서는 남은 4×4 skip을 버리고 256채널 8×8 skip을 붙인다.
- 16×16에서는 남은 8×8 skip을 버리고 128채널 skip 두 개를 붙인다.
- 32×32에서는 64채널 skip 두 개를 붙인다.

따라서 단순히 '대칭적인 U-Net의 skip 연결'이라고 설명하면 실제 동작을 놓친다. 잘린 공간 특징이나 여러 skip의 결합이 의도한 설계인지 확인해야 하며 임의 채널·해상도에서도 자동으로 맞는 구조는 아니다.

MNIST 옵션은 28→14→7→3으로 축소되고 첫 확대가 6이어서 7×7 skip과 맞지 않는다. 현재 while은 크기가 다른 skip을 계속 버려 목록이 소진될 수 있다. 기본 설정에서 모양이 맞는다는 사실을 모든 데이터 옵션의 유효성으로 일반화하지 않는다.

## 4. DDIM의 전이식

선택한 현재 단계 t와 이전 단계 s<t에 대해 누적 계수를 at=ᾱt, as=ᾱs라 쓰자. 노이즈 예측으로 원본 추정값을 구한다.

$$\widehat x_0=\frac{x_t-\sqrt{1-a_t}\epsilon_\theta(x_t,t)}{\sqrt{a_t}}$$

통상적인 DDIM 전이는 다음과 같다.

$$\sigma_{t\to s}^2=\eta^2\frac{1-a_s}{1-a_t}\left(1-\frac{a_t}{a_s}\right)$$

$$x_s=\sqrt{a_s}\widehat x_0+
\sqrt{1-a_s-\sigma_{t\to s}^2}\epsilon_\theta(x_t,t)+\sigma_{t\to s}z$$

[DDIM 논문](https://arxiv.org/abs/2010.02502)의 생성 가족과 연결되는 식이다. η=0은 주어진 초기값에서 결정적이지만 초기 x 자체는 무작위다. η를 바꾸거나 단계를 줄이면 유한 모델·수치 전이의 출력 분포도 달라질 수 있다.

## 5. η>0 계수의 불일치

코드의 dir_xt 루트 안은 대수적으로 **1−as−η²(1−at)**로 단순해지고, 추가 잡음 계수는 **η√(1−as)**다. 위의 σ²에 필요한 1−at/as 인자가 빠져 있어 일반적인 DDIM 확률 전이와 다르다.

η=1이고 실제로 시간을 거슬러 이동해 as>at이면 방향 계수의 루트 안은 at−as<0이 되어 NaN이 생긴다. 따라서 기본 η=0 경로를 보여 주는 코드라도 η를 1로 바꾸어 DDPM에 해당하는 유효한 전이를 얻는다고 설명할 수 없다. 일반 η를 지원하려면 두 계수를 같은 전이식에 맞춰야 한다. 코드와 출력은 그대로 보존한다.

## 6. 시간 목록과 마지막 단계

기본 설정은 c=1000//50=20이고 선택 시간은 0,20,…,980이다. 따라서 생성 첫 모델 호출은 999가 아니라 **980**이다. 초기 상태는 N(0,I)이므로 그 선택 단계의 실제 주변 분포와의 근사 차이도 고려해야 한다.

이전 시간 목록의 첫 원소를 0으로 놓아 마지막에는 t=0, t_prev=0을 사용한다. 그러나 코드 인덱스 0의 누적곱은 첫 노이즈 단계를 뜻하며 깨끗한 시점의 1이 아니다. η=0에서 clipping이 없다면 이 같은 단계 전이는 관측을 그대로 되돌리므로 **마지막 원본 복원 전이가 빠진다**. 코드의 pred_x0 clipping이 값을 바꿀 수는 있어도 올바른 종단 처리를 대신하지 않는다.

DDIM_STEPS가 T의 약수가 아니면 실제 선택 개수도 인수와 다를 수 있다. 예를 들어 T=1000, 요청 60이면 간격은 16이고 arange는 63개를 만든다. 요청이 0이면 정수 나눗셈이 실패하고 T보다 크면 간격 0이 된다. 유효 범위와 실제 목록·모델 호출 수를 기록해야 한다.

## 7. 속도와 품질 보고

기본 50개 시간에서는 모델 호출이 50번으로 1,000번보다 적다. 그러나 출력의 '20배 빠름'은 **호출 수 비율**이며 같은 환경의 DDPM 실행시간을 측정한 값이 아니다. CUDA는 비동기 실행이므로 time.time만 둘러싼 구간은 장치 동기화를 포함한 측정과 다를 수 있다. 배치 16의 시간과 이미지 한 장의 시간도 구분한다.

샘플러는 eval·no_grad를 사용하고 원본 추정값과 최종 표시값을 clipping한다. 속도뿐 아니라 같은 체크포인트·초기화·평가 데이터에서 단계 수와 η에 따른 품질·다양성을 비교해야 한다. 이 페이지에는 그 평가나 생성 출력이 없으므로 가속·품질 개선을 실측 결과로 단정하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
기본 T=1000, 요청 단계 50에서 선택된 시간과 호출 수를 설명하라.

</div>

??? success "연습문제 1 풀이"

    정수 간격은 20이고 목록은 0,20,…,980의 50개다. 역순 첫 호출은 980이며 999를 포함하지 않는다. 요청이 60이면 간격 16으로 실제 63개가 되어 요청 개수와 달라진다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
η=1에서 코드의 방향 계수에 어떤 문제가 생기는가?

</div>

??? success "연습문제 2 풀이"

    루트 안은 1−as−(1−at)=at−as다. 역진하는 실제 단계에서 as>at이므로 음수여서 NaN이 된다. 올바른 확률 전이는 σ²=η²(1−as)/(1−at)·(1−at/as)를 방향·잡음 항에 함께 사용해야 한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
마지막 이전 누적곱과 속도 보고에서 각각 어떤 관례를 구분해야 하는가?

</div>

??? success "연습문제 3 풀이"

    깨끗한 마지막 시점은 누적곱 1로 다뤄야 하지만 현재 코드는 이전 인덱스도 0으로 두어 첫 노이즈 단계의 계수를 사용한다. 또한 1000/50=20은 모델 호출 수 비율이지 측정한 실행시간 비율이 아니다. 올바른 종단·장치 동기화·동일 체크포인트 조건에서 품질과 속도를 별도로 검증해야 한다.



## 정리하며

DDIM은 학습과 다른 시간 목록·생성 전이를 사용할 수 있지만 계수와 마지막 원본 시점을 일치시켜야 한다. 이 구현은 η>0 전이식과 마지막 인덱스 처리가 맞지 않는다. 적은 모델 호출과 성공적인 고품질 가속의 실측 증거를 구분한다.
