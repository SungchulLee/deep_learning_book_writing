# 잔차 U-Net을 사용하는 DDPM

이 예제는 시간 조건 잔차 블록, 공간 어텐션과 다중 해상도 U-Net으로 노이즈를 예측하는 DDPM 학습 구성을 다룬다. 코사인 노이즈 스케줄, 러닝레이트 워밍업·코사인 감소, EMA와 체크포인트를 함께 사용한다. 이런 선택을 구현했다는 사실과 실제 생성 품질을 검증했다는 사실은 구분해야 한다.

## 1. 코드

??? note "코드 (795줄)"

    ```python
    """DDPM."""
    # ==========================================
    # 나아진 DDPM(Ho 외, 2020) - PyTorch 짜기
    # ==========================================
    # 바탕에 견준 핵심 개선:
    #   1. 안정된 뽑기를 위한 지수 이동 평균(EMA)
    #   2. 익히기의 안정을 위한 그래디언트 자르기
    #   3. 러닝 빠르기 몸 풀기 + 코사인 줄이기 차례표
    #   4. 늘린 모델 담이(BASE_CH=128)
    #   5. 여러 해상도 어텐션(8, 16)
    #   6. 레이어마다 남은 덩이를 더 둠(2 대신 3)
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
    # 짜임새(이 파라메터를 고친다)
    # ==========================================
    class Config:
        # 자료 묶음 자리매김
        DATASET = "MNIST"           # "CIFAR10"이나 "MNIST"
        IMG_SIZE = 32               # 두 자료 묶음 모두 32x32으로 크기를 바꾼다

        # 익히기 자리매김
        BATCH_SIZE = 128
        LR = 1e-4                   # 안정을 위해 낮춘 러닝 빠르기
        EPOCHS = 100                # 10에서 늘렸다 - 품질에 결정적이다
        WARMUP_STEPS = 1000         # 러닝 빠르기 몸 풀기 스텝
        GRAD_CLIP = 1.0             # 그래디언트 자르기 임계값

        # 확산 자리매김
        T = 1000                    # 확산 때 스텝 수

        # 모델 구조
        BASE_CH = 128               # 담이를 늘리려 64에서 늘렸다
        CH_MULTS = (1, 2, 4, 4)     # 레이어마다 채널 갑절
        NUM_RES_BLOCKS = 3          # 레이어마다 남은 덩이(예전에는 2)
        ATTN_RES = {8, 16}          # 이 해상도에서 어텐션을 쓴다
        TIME_EMB_DIM = 256          # 때 스텝 임베딩 차원
        DROPOUT = 0.1               # 정칙화를 위한 드롭아웃

        # 지수 이동 평균 자리매김
        EMA_DECAY = 0.9999          # 지수 이동 평균 줄임 비율
        EMA_START = 2000            # 이만큼 걸은 뒤 지수 이동 평균을 시작한다

        # 기록하고 갈무리하기
        SAMPLE_INTERVAL = 10        # N바퀴마다 표본을 만든다
        SAVE_INTERVAL = 20          # N바퀴마다 되짚을 자리를 갈무리한다
        NUM_SAMPLES = 16            # 만들 표본의 수

        # 구조 자리매김
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
    # 2) 사인 꼴 때 스텝 임베딩
    # ==========================================
    class SinusoidalPosEmb(nn.Module):
        """
        때 스텝의 사인 꼴 자리 임베딩.
        사인과 코사인으로 낱값 때 스텝을 차원 높은 임베딩으로 옮긴다.
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

            # 사인과 코사인 임베딩을 이어 붙인다
            emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, dim]
            if self.dim % 2 == 1:
                emb = F.pad(emb, (0, 1))
            return emb


    # ==========================================
    # 3) 벽돌: 남은 덩이, 어텐션, 키우기/줄이기
    # ==========================================
    class ResidualBlock(nn.Module):
        """
        때 임베딩으로 조건을 준 남은 덩이.
        GroupNorm, SiLU 깨움, 그리고 쓸 수 있는 떨구기를 쓴다.
        때 임베딩이 잣수와 옮김(FiLM)으로 피처를 조절한다.
        """
        def __init__(self, in_ch, out_ch, time_emb_dim, dropout=0.1, groups=8):
            super().__init__()
            # 첫 겹말기 길
            self.norm1 = nn.GroupNorm(groups, in_ch)
            self.act1 = nn.SiLU()
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

            # 때 임베딩 쏘기(잣수와 옮김)
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

            # FiLM(피처마다 선형 조절)으로 때 조건 주기
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
        2차원 피처 지도의 여러 머리 셀프 어텐션.
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

            # 어텐션의 잣수 갑절
            self.scale = self.head_dim ** -0.5

        def forward(self, x):
            b, c, h, w = x.shape

            # 고르게 맞추고 Q, K, V을 셈한다
            x_norm = self.norm(x)
            q = self.q(x_norm).reshape(b, self.num_heads, self.head_dim, h * w)
            k = self.k(x_norm).reshape(b, self.num_heads, self.head_dim, h * w)
            v = self.v(x_norm).reshape(b, self.num_heads, self.head_dim, h * w)

            # 어텐션: softmax(Q @ K^T / sqrt(d)) @ V
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
    # 4) U-Net 구조
    # ==========================================
    class UNet(nn.Module):
        """
        잡음 헤아리기를 위한 U-Net 등뼈.

        구조:
        - 인코더: 남은 덩이와 어텐션을 갖춘 줄이는 길
        - 병목: 어텐션을 갖춘 가운데 덩이
        - 디코더: 건너뛰기 이음을 갖춘 키우는 길

        피처:
        - 남은 덩이마다 때 임베딩으로 조건 주기
        - 정한 해상도에서의 셀프 어텐션
        - 해상도 레이어마다 남은 덩이 여럿
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

            # 때 임베딩 여러 레이어 뉴럴 넷트웍
            self.time_emb = nn.Sequential(
                SinusoidalPosEmb(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim * 4),
                nn.SiLU(),
                nn.Linear(time_emb_dim * 4, time_emb_dim),
            )

            # ========== 인코더(줄이는 길) ==========
            self.downs = nn.ModuleList()
            ch = base_ch
            res = img_size
            self.skip_channels = []  # 건너뛰기 이음을 위해 채널을 좇는다

            for i, mult in enumerate(ch_mults):
                out_ch = base_ch * mult
                level_blocks = nn.ModuleList()

                # 레이어마다 남은 덩이 여럿
                for j in range(num_res_blocks):
                    block_in_ch = ch if j == 0 else out_ch
                    level_blocks.append(ResidualBlock(block_in_ch, out_ch, time_emb_dim, dropout))

                # 정한 해상도에서의 어텐션
                attn = SelfAttention2d(out_ch) if res in attn_res else nn.Identity()

                # 줄이기(마지막 레이어만 빼고)
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

            # ========== 디코더(키우는 길) ==========
            self.ups = nn.ModuleList()

            for i, mult in reversed(list(enumerate(ch_mults))):
                out_ch = base_ch * mult
                skip_ch = self.skip_channels.pop()
                level_blocks = nn.ModuleList()

                # 첫 덩이가 이어 붙인 피처(지금 것 + 건너뛴 것)을 받는다
                for j in range(num_res_blocks):
                    if j == 0:
                        block_in_ch = ch + skip_ch
                    else:
                        block_in_ch = out_ch
                    level_blocks.append(ResidualBlock(block_in_ch, out_ch, time_emb_dim, dropout))

                # 정한 해상도에서의 어텐션
                attn = SelfAttention2d(out_ch) if res in attn_res else nn.Identity()

                # 키우기(첫 레이어만 빼고)
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
            # 때 임베딩
            t_emb = self.time_emb(t)

            # ========== 인코더 ==========
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

            # ========== 디코더 ==========
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
    # 5) DDPM 커널: 앞/뒤 과정
    # ==========================================
    class DDPM(nn.Module):
        """
        잡음 없애는 확산 확률 모델.

        익히기: 자료에 잡음을 더하고(앞 과정) 그 잡음을 헤아리는 법을 배운다.
        뽑기: 순수 잡음에서 시작해 거듭 잡음을 없앤다(뒤 과정).
        """
        def __init__(self, model, timesteps, device):
            super().__init__()
            self.model = model
            self.timesteps = timesteps
            self.device = device

            # 확산 파라메터를 미리 셈한다
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
            뒤 과정: 잡음 없애기 스텝 하나.
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
            익히기 로스: 헤아린 잡음과 실제 잡음 사이의 평균 제곱 어긋남.
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
        모델 파라메터의 지수 이동 평균을 지킨다.
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
            """지수 이동 평균 파라메터를 고친다."""
            for ema_param, model_param in zip(self.shadow.parameters(), model.parameters()):
                ema_param.data.mul_(self.decay).add_(model_param.data, alpha=1 - self.decay)

        def forward(self, *args, **kwargs):
            """추론에 지수 이동 평균 모델을 쓴다."""
            return self.shadow(*args, **kwargs)


    # ==========================================
    # 7) 러닝 빠르기 차례표
    # ==========================================
    class WarmupCosineSchedule:
        """
        선형 몸 풀기 뒤 코사인 줄이기가 오는 러닝 빠르기 차례표.
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
            drop_last=True,  # 배치 크기 문제를 피한다
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

        # 파라메터 개수 세기
        num_params = sum(p.numel() for p in unet.parameters())
        print(f"Model parameters: {num_params:,}")

        # DDPM 감싸개를 세운다
        ddpm = DDPM(unet, timesteps=config.T, device=config.DEVICE).to(config.DEVICE)

        # 최적화기
        optimizer = torch.optim.AdamW(ddpm.parameters(), lr=config.LR, weight_decay=0.01)

        # 러닝레이트 스케줄러
        scheduler = WarmupCosineSchedule(
            optimizer,
            warmup_steps=config.WARMUP_STEPS,
            total_steps=total_steps,
        )

        # 지수 이동 평균(익히기 스텝을 얼마간 지난 뒤 첫자리매김)
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

                # 그래디언트 자르기
                torch.nn.utils.clip_grad_norm_(ddpm.parameters(), config.GRAD_CLIP)

                # 가장 좋게 하개 스텝
                optimizer.step()

                # 러닝 빠르기 차례표 스텝
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

## 2. 데이터와 기본 설정

기본 데이터는 MNIST이며 28×28 이미지를 **32×32로 resize**하고 0~1을 −1~1로 정규화한다. CIFAR10을 선택하면 RGB와 무작위 좌우 반전을 사용한다. 레이블은 학습에서 버리므로 현재 모델은 클래스 조건 생성 모델이 아니다.

배치 크기는 128, 에포크 수는 100이고 drop_last=True다. MNIST는 에포크당 468배치, 전체 46,800번 갱신하며 매 에포크 나머지 96개를 생략한다. 데이터는 섞으므로 항상 같은 96개가 빠지는 것은 아니다. CIFAR10은 390배치이며 설정에 따라 학습량이 달라진다. 검증·테스트 집합과 생성 품질 지표는 이 루프에 없다.

SEED=42를 torch에 설정하지만 환경·라이브러리·장치가 달라도 완전히 같은 결과가 난다는 보장은 아니다. 샘플링도 전역 난수 상태를 소비하므로 평가 시점과 샘플 수를 바꾸면 이후의 난수 흐름이 달라질 수 있다.

## 3. 시간 조건 잔차 블록

시간 인덱스를 사인·코사인 256차원 임베딩과 MLP로 바꾼다. 각 잔차 블록은 GroupNorm→SiLU→컨볼루션 뒤에 시간별 scale·shift를 적용하고, 다시 SiLU·Dropout·컨볼루션을 통과시킨다. 입력은 Identity나 1×1 투영으로 출력에 더한다.

시간 피처는 두 번째 GroupNorm 출력의 h(1+scale)+shift에 반영된다. 시간 t가 같아도 입력 데이터에 따라 출력은 다르고, 같은 데이터라도 학습·평가 모드에서 Dropout의 동작이 다르다. GroupNorm은 배치 통계를 누적하지 않는다.

시간 임베딩은 홀수 폭에 0 padding을 추가하지만 half−1을 분모로 사용하므로 매우 작은 폭은 유효하지 않다. GroupNorm은 채널 수가 8의 배수여야 하고, 어텐션은 채널 수가 헤드 수 4로 나누어져야 한다. 기본 채널 구성은 이 조건을 만족한다.

## 4. 다중 해상도 경로와 어텐션

기본 base_ch=128, CH_MULTS=(1,2,4,4)이므로 인코더의 각 수준은 다음과 같다.

| 수준 | 피처 모양 | 다음 처리 |
|---|---|---|
| 32×32 | B×128×32×32 | 16×16으로 축소 |
| 16×16 | B×256×16×16 | 어텐션 후 8×8로 축소 |
| 8×8 | B×512×8×8 | 어텐션 후 4×4로 축소 |
| 4×4 | B×512×4×4 | 축소 없이 middle |

수준마다 잔차 블록 3개를 지나고 **그 수준의 마지막 피처 하나만** skip에 저장한다. middle은 잔차 블록 2개와 어텐션을 사용한다. 디코더는 저장 피처와 채널 방향으로 연결하고 수준별 잔차 블록 3개를 거쳐 위 해상도로 복원한다. 전체 잔차 블록은 12+2+12=26개다. 마지막 출력은 입력 채널과 같은 32×32 노이즈 예측이다.

SelfAttention2d는 N=H×W 위치 사이의 헤드별 N×N 행렬을 만든다. 어텐션 메모리는 해상도에 민감하며 16×16과 8×8을 선택했다고 모든 배치 크기에서 가벼운 것은 아니다. middle의 4×4 어텐션도 별도로 있다.

현재 32×32는 세 번의 축소·확대와 맞는다. 다른 크기에서는 skip을 중앙 crop하는 경로가 있지만 임의 크기의 출력이 원본과 일치하거나 skip이 작은 경우까지 처리한다고 보장할 수 없다. 로스는 예측과 노이즈 크기가 같아야 한다.

## 5. 코사인 노이즈 스케줄과 단순 로스

코사인 누적 계수는 다음 비율로 구성하고 이웃 비에서 β를 얻는다.

$$f(t)=\cos^2\!\left(\frac{t/T+s}{1+s}\frac\pi2\right),\qquad
\bar\alpha_t=\frac{f(t)}{f(0)},\qquad s=0.008$$

코드는 β를 [10⁻⁸,0.999]로 잘라 유효한 범위에 놓는다. 저장한 스케줄은 register_buffer로 등록되어 모델 상태·장치 이동에 포함되지만, self.device 문자열은 이후 .to 호출과 자동 동기화되는 속성이 아니다. 초기 설정과 이동을 일치시켜야 한다.

학습은 표본마다 t를 선택하고 가우시안 관측을 만든 뒤 ε와 예측의 모든 원소 MSE를 최소화한다. 정확한 전체 변분 우도 목적이나 학습 분산을 포함한 목적은 아니다. 이 예제는 [Improved DDPM](https://arxiv.org/abs/2102.09672)의 일부 아이디어를 사용하지만 논문의 모든 구조·목적·실험을 재현하는 구현으로 볼 수 없다.

## 6. 역분산과 생성 모드

역평균은 노이즈 예측 파라미터화를 사용하고 추가 잡음에는 **사후분산 β̃t**를 사용한다. βt를 바로 쓰는 앞의 간단한 예제와 다르다. 첫 코드 인덱스의 사후분산은 0이므로 혼합 시간 배치를 전달하더라도 해당 원소의 잡음은 0이 된다. 모두 t=0이면 평균을 곧바로 반환한다.

sample은 eval과 no_grad를 사용해 1,000단계를 수행한다. 이미지로 저장하기 전에는 −1~1로 clipping하고 (x+1)/2로 변환한다. 이 후처리는 표시 범위를 맞추지만 범위 밖 생성값을 숨길 수 있으므로 생성 분포의 원시 값과 구분한다.

## 7. 러닝레이트 스케줄과 EMA

AdamW의 초기 러닝레이트는 10⁻⁴, weight decay는 0.01이고 그래디언트 노름을 1로 클리핑한다. 러닝레이트 스케줄은 1,000단계 워밍업 뒤 코사인으로 10⁻⁶까지 감소한다. 이것은 노이즈 β 스케줄과 별개다.

스케줄러는 **optimizer.step 뒤** 호출하고 초기 러닝레이트를 0으로 바꾸지 않는다. 따라서 첫 업데이트는 기본 러닝레이트 10⁻⁴를 사용한 뒤 다음 업데이트의 러닝레이트가 10⁻⁷로 내려간다. 처음부터 작은 값에서 매끄럽게 증가하는 워밍업과 같지 않다. 로그의 lr은 다음 업데이트용 값이다.

EMA는 2,000번째 업데이트 뒤 현재 U-Net을 복사해 시작하고 이후 파라미터를 0.9999·이전값+0.0001·현재값으로 갱신한다. 시작 전 기록까지 평균한 모델은 아니다. 이 구현은 파라미터만 평균하므로 향후 BatchNorm 등의 buffer를 추가하면 별도 처리가 필요하다. 현재 U-Net의 GroupNorm에는 그 이동 통계가 없다. EMA가 어떤 데이터에서도 품질을 개선한다고 보장할 수는 없다.

## 8. 저장과 재현의 범위

샘플 저장 함수는 ddpm.model을 EMA shadow로 잠시 바꾼다. 예외가 발생하면 원래 모델을 복원하는 finally 처리는 없으며, 원본 모델을 직접 샘플링하면 eval 상태가 남는다. main은 다음 에포크 시작에 unet.train을 호출해 학습 모드를 다시 설정한다.

체크포인트에는 예측기·EMA·optimizer 상태와 스케줄 단계, 일부 설정만 있다. 스케줄 buffer 전체나 U-Net의 모든 구조 설정, RNG 상태·데이터로더 진행 위치를 저장하지 않아 이 파일만으로 임의 설정을 자동 복원하거나 중단 직후와 완전히 같은 학습을 재개할 수는 없다. 재개용 로더도 이 페이지에 없다.

에포크 끝 mean loss는 동일 크기 배치의 평균이지만, 진행 막대의 중간 평균은 pbar.n 갱신 시점에 따라 정확한 배치 수와 어긋날 수 있다. 학습 로스·샘플 파일·체크포인트의 존재와 독립적인 생성 품질 평가는 구분한다. 이 페이지에는 실측 FID나 학습 시간 비교가 없다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
기본 U-Net의 해상도와 skip 저장 규칙을 설명하라.

</div>

??? success "연습문제 1 풀이"

    32→16→8→4로 줄이고 디코더에서 4→8→16→32로 되돌린다. 인코더 수준마다 잔차 블록 3개를 지난 마지막 피처 하나를 저장하고 해당 디코더 수준에서 채널 방향으로 연결한다. 각 잔차 블록마다 skip을 하나씩 저장하는 구현은 아니다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
코사인 노이즈 스케줄과 워밍업·코사인 러닝레이트 스케줄은 무엇을 바꾸는가?

</div>

??? success "연습문제 2 풀이"

    노이즈 스케줄은 정방향 관측과 역전이의 β·누적 신호를 정한다. 러닝레이트 스케줄은 optimizer의 파라미터 업데이트 크기를 조절한다. 이 구현의 워밍업은 첫 optimizer 갱신 뒤 시작해 첫 업데이트는 기본 러닝레이트를 사용한다. 두 스케줄이 코사인이라는 이유로 같은 대상이라고 볼 수 없다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
저장된 체크포인트만으로 완전히 같은 학습을 자동 재개할 수 있는가?

</div>

??? success "연습문제 3 풀이"

    없다. 예측기·EMA·optimizer와 스케줄 단계는 있지만 전체 구조·노이즈 설정, RNG와 데이터로더 상태 및 재개 로더가 없다. 같은 설정을 재구성하고 필요한 상태를 복원해야 한다. 생성 품질의 비교도 같은 검증·샘플링 조건을 별도로 정해야 한다.



## 정리하며

이 DDPM은 잔차·시간 변조·공간 어텐션 U-Net에 코사인 노이즈와 EMA를 결합한다. 실제 모양과 업데이트 수, 워밍업 첫 단계 및 체크포인트의 저장 범위를 확인한다. 안정화 장치나 저장 성공만으로 생성 성능을 보장하지 않는다.
