# 클래스 조건 DDPM과 분류기 없는 가이던스

클래스 조건 확산 모델은 노이즈 관측과 시간에 더해 생성할 클래스의 레이블을 받는다. 분류기 없는 가이던스(classifier-free guidance, CFG)는 조건을 준 예측과 조건을 제거한 예측을 결합해 생성 방향을 조절한다. 이 페이지는 그 학습·샘플링 인터페이스를 살펴보되, 현재 U-Net의 skip 처리 불일치 때문에 완성된 실행 예제로 볼 수 없음을 먼저 명시한다.

## 1. 코드

??? note "코드 (436줄)"

    ```python
    """조건 DDPM."""
    # ============================================================
    # 클래스 조건 DDPM(CIFAR-10) — PyTorch, 주석을 잘 단 코드
    # - 남은 덩이와 (쓸 수 있는) 어텐션을 갖춘 U-Net
    # - 사인 꼴 때 스텝 임베딩
    # - 이름표 조건 주기(때 임베딩에 임베딩을 더함)
    # - 이름표 떨구기로 분류기 없는 이끌기를 쓸 수 있음
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
    NUM_CLASSES = 10         # CIFAR-10에는 클래스가 10개 있다
    IN_CHANNELS = 3 if DATASET == "CIFAR10" else 1
    IMG_SIZE = 32 if DATASET == "CIFAR10" else 28
    BATCH_SIZE = 128
    LR = 2e-4
    EPOCHS = 5
    T = 1000                 # 옛부터의 DDPM은 1000스텝을 쓴다
    BASE_CH = 64             # 모델 너비. 품질을 위해 늘린다(예컨대 96/128)
    ATTN_RES = {16}          # 이 공간 크기에서 어텐션을 쓴다. 끄려면 set()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SAVE_PATH = "ddpm_conditional_samples.png"

    # ------- 분류기 없는 이끌기(CFG) -------
    USE_CFG = True           # 끄려면 False으로 둔다
    CFG_DROPOUT_P = 0.1      # 익힐 때 이 확률로 이름표를 떨군다
    CFG_SCALE = 3.0          # 뽑을 때의 이끌기 세기(1.0 = 끔)
    NULL_CLASS_ID = NUM_CLASSES  # "빈" 조건을 위해 남겨 둔 어깨수

    # ==========================================
    # 1) 확산 차례표와 도우미
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
    # 2) 사인 꼴 때 스텝 임베딩
    # ==========================================
    class SinusoidalPosEmb(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim

        def forward(self, t):
            # t: [B] 정수 때 스텝
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

            # 때 임베딩에서 오는 FiLM 조건 주기
            scale, shift = self.time_mlp(t_emb).chunk(2, dim=1)
            scale = scale[:, :, None, None]
            shift = shift[:, :, None, None]

            h = self.norm2(h)
            h = h * (1 + scale) + shift
            h = self.conv2(self.act2(h))
            return h + self.skip(x)

    class SelfAttention2d(nn.Module):
        """주어진 해상도에서의 가벼운 셀프 어텐션."""
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
        조건 주기: 클래스 이름표 -> 임베딩 -> 때 임베딩에 더함.
        분류기 없는 이끌기에서는 "빈" 이름표를 위해 임베딩 자리를 하나 더 둔다.
        """
        def __init__(self, in_ch=IN_CHANNELS, base_ch=BASE_CH, ch_mults=(1,2,4,4),
                     attn_res=ATTN_RES, img_size=IMG_SIZE, time_emb_dim=256,
                     num_classes=NUM_CLASSES, use_cfg=USE_CFG):
            super().__init__()
            self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

            # 때 임베딩 여러 레이어 뉴럴 넷트웍
            self.time_sinus = SinusoidalPosEmb(time_emb_dim)
            self.time_mlp = nn.Sequential(
                nn.Linear(time_emb_dim, time_emb_dim * 4),
                nn.SiLU(),
                nn.Linear(time_emb_dim * 4, time_emb_dim),
            )

            # 이름표 임베딩(CFG이면 빈 토큰을 위해 +1)
            label_vocab = num_classes + (1 if use_cfg else 0)
            self.label_emb = nn.Embedding(label_vocab, time_emb_dim)

            # 인코더
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

            # 디코더
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
            y: [B] 클래스 이름표(CFG 빈 조건이면 NUM_CLASSES 또는 NULL_CLASS_ID)
            """
            # 때 임베딩을 세우고 이름표 임베딩을 더한다
            t_emb = self.time_mlp(self.time_sinus(t))  # [B, D]
            y_emb = self.label_emb(y)                  # [B, D]
            t_emb = t_emb + y_emb                      # 단순하고 잘 듣는 조건 주기

            # 인코더
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

            # 건너뛰는 이음을 갖춘 디코더
            for m in self.decoder:
                if isinstance(m, ResidualBlock):
                    skip = skips.pop()
                    h = torch.cat([h, skip], dim=1)
                    h = m(h, t_emb)
                else:
                    h = m(h)

            return self.out_conv(self.out_act(self.out_norm(h)))

    # ==========================================
    # 5) 조건 주기 + CFG를 갖춘 DDPM 커널
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
            이름표 y을 조건으로 삼아 아무 때 스텝 t에서 잡음을 헤아린다.
            분류기 없는 이끌기를 켜면 배치의 일부에서 y을 NULL_CLASS_ID로 아무렇게나 바꾼다.
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
            cfg_scale>1이면 CFG를 얻으려고 앞으로 스텝을 두 번 한다(조건 있음과 빈 조건).
            """
            betas_t = extract(self.betas, t, x_t.shape)
            sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
            sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x_t.shape)

            if cfg_scale == 1.0 or not USE_CFG:
                eps = self.model(x_t, t, y)
            else:
                # 분류기 없는 이끌기: eps = eps_null + s * (eps_cond - eps_null)
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
            y이 None이면 아무 이름표를 뽑는다. 분류기 없는 이끌기를 쓰면 빈 이름표도 만든다.
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
    # 8) 클래스마다 격자로 뽑아 갈무리하기
    # ==========================================
    @torch.no_grad()
    def to_image_range(x):  # [-1,1] -> [0,1]
        return (x.clamp(-1, 1) + 1) * 0.5

    # 클래스 10개를 만들고 클래스마다 그림 4장(모두 40장)
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

## 2. 데이터와 두 조건의 결합

기본 데이터는 CIFAR-10의 32×32 RGB이며 0~1을 −1~1로 바꾼다. 시간 인덱스는 사인·코사인 임베딩과 MLP로 256차원으로 바꾸고, 클래스 임베딩과 **더한 뒤** 모든 잔차 블록의 scale·shift에 전달한다. 이미지·시간·레이블의 배치 크기와 장치가 맞아야 한다.

클래스는 0~9이고 CFG용 null 인덱스는 10이다. use_cfg=True이면 임베딩 어휘를 11개로 만든다. null은 실제 열한 번째 이미지 클래스가 아니라 조건을 주지 않는 학습 상태다. 모델의 use_cfg 인수와 학습·샘플링의 전역 USE_CFG를 따로 관리하므로 서로 다르게 설정하면 null 임베딩 범위와 호출이 맞지 않을 수 있다.

## 3. 조건 제거와 노이즈 로스

loss는 표본마다 t를 고르고 코사인 스케줄로 노이즈 관측을 만든다. 조건 레이블은 확률 0.1로 null로 바꾸고 실제 표준 정규 노이즈와 예측의 원소 평균 MSE를 계산한다. 따라서 하나의 네트워크가 조건 예측과 null 예측을 학습하도록 설계되어 있다.

조건 제거는 Dropout 계층과 다른 기법이다. 이 ResidualBlock에는 앞 무조건 DDPM 예제의 Dropout 계층도 없다. 단순 노이즈 MSE와 전체 변분 우도 목적을 구분하고 조건 제거 확률은 데이터·모델에 맞춰 검증해야 한다.

## 4. CFG의 배율과 생성 조건

샘플링 예측은 다음과 같다.

$$\widehat\epsilon=\epsilon_{null}+w(\epsilon_{cond}-\epsilon_{null})$$

w=1이면 조건 예측, w=0이면 null 예측이고 w>1이면 두 예측의 차이를 외삽한다. 기본 w=3이다. [CFG 논문](https://arxiv.org/abs/2207.12598)의 개념과 연결되지만, 가이던스 증가가 모든 모델에서 더 정확한 조건·더 좋은 품질을 보장하는 것은 아니다. 다양성과 이미지 값 범위도 달라질 수 있다.

코드는 w=1이면 조건 예측 한 번만, 그 외 CFG에서는 null·조건 예측을 각각 계산한다. 1,000단계 생성에서 모델 호출은 각각 1,000번과 2,000번이다. null 분기에는 유효한 정수 레이블 텐서가 필요하며 p_mean_var를 직접 호출해 y_null=None으로 두면 실패할 수 있다.

sample에서 y=None이면 실제 클래스 0~9를 균등하게 **뽑아 조건 생성**한다. null 예측만 사용하는 무조건 생성과 다르며, 실제 데이터의 클래스 분포가 균등하지 않을 때는 그 주변 혼합분포도 달라질 수 있다.

## 5. 디코더의 skip 계약 오류

기본 인코더의 공간 크기는 32→16→8→4이며 수준마다 잔차 블록 두 개를 둔다. forward는 **모든 잔차 블록 뒤**에 피처를 저장한다. 반면 생성자에서 decoder의 첫 잔차 블록만 skip 채널을 더해 입력 폭을 정하고 두 번째 블록은 결합 없이 out_ch만 받도록 만든다.

첫 디코더 수준에서 h와 마지막 skip은 각각 256채널 4×4라 연결 후 512채널이 되고 첫 블록의 선언과 맞는다. 그러나 두 번째 블록에서도 forward가 또 skip을 꺼내 연결해 **512채널**을 만들지만 그 블록의 norm1·conv1은 **256채널**을 기대한다. 따라서 기본 CIFAR-10 설정도 이 두 번째 디코더 블록에서 채널이 맞지 않는다.

또한 생성자의 hs는 초기 컨볼루션과 downsample 출력을 포함하는 반면 런타임 skips는 잔차 출력만 저장한다. 두 목록의 길이가 같아 보여도 저장 위치와 의미가 다르다. 한 채널 값만 바꾸는 것으로 전체 skip 설계가 검증되는 것은 아니므로 생성·저장·소비 규칙을 일치시켜야 한다.

MNIST 옵션은 크기를 28로 바꿔 세 번 축소하면 28→14→7→3이 된다. 이를 두 배 확대하면 6으로 돌아오므로 7×7 skip과 공간 크기도 맞지 않는다. 현재 forward에는 crop·padding 보정이 없다. 코드 자체는 보존하되 위 모양 문제 때문에 성공적인 학습·샘플링을 했다고 설명하지 않는다.

## 6. 역전이와 학습 구성의 범위

정방향 스케줄은 buffer로 등록하고 역평균에는 가이던스 예측, 추가 잡음에는 사후분산 β̃t를 사용한다. 생성은 no_grad와 eval이며 마지막에는 잡음을 생략한다. 표시용으로 −1~1 clipping 후 0~1로 바꾼다. clipping이 범위 밖 값을 숨길 수 있으므로 원시 생성값의 검사와 구분한다.

기본 학습 계획은 배치 128, AdamW 러닝레이트 0.0002, 5에포크다. CIFAR-10 전체에서 에포크당 391배치, 마지막 80개를 포함해 총 1,955번 갱신하도록 구성되어 있다. 다만 현재 구조 오류로 실제 완료를 보장하지 않는다. EMA, 그래디언트 클리핑, 러닝레이트 스케줄, 검증·체크포인트는 없다. 에포크 로그는 서로 다른 크기 배치의 단순 평균이다.

device는 여러 함수의 전역 DEVICE를 참조하므로 모델을 나중에 다른 장치로 옮겼다고 모든 생성 텐서가 자동으로 맞춰지지는 않는다. 데이터 준비·훈련이 최상위에 있어 파일 import만으로도 실행되며 마지막 main의 pass가 이를 막지 않는다. num_workers=2인 데이터로더도 실행 환경의 프로세스 시작 방식에 맞는 진입점 관리가 필요하다.

## 7. 클래스별 격자 읽기

최종 레이블은 클래스마다 네 개씩 이어 붙여 총 40개이고 한 행에 네 장을 저장한다. 의도한 격자는 행별로 클래스가 같지만 레이블 순서만으로 실제 생성 이미지가 그 클래스를 충실히 표현한다고 볼 수 없다. 독립 분류 평가·표본 검사와 클래스 내 다양성을 함께 확인해야 한다. 이 페이지에는 그 품질 평가나 실행 출력이 없다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
첫 번째 디코더 수준의 두 잔차 블록에서 입력 채널이 왜 맞지 않는가?

</div>

??? success "연습문제 1 풀이"

    첫 블록은 h 256채널과 skip 256채널을 연결한 512채널을 받도록 선언되어 있다. 둘째 블록은 256채널로 선언되어 있으나 forward가 다시 skip을 연결해 512채널을 전달한다. 선언된 입력 폭과 런타임 skip 소비 규칙이 다르다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
CFG의 w=0,1,3과 y=None의 의미를 구분하라.

</div>

??? success "연습문제 2 풀이"

    유효한 null 조건이 있을 때 w=0은 null 예측, w=1은 조건 예측, w=3은 null에서 조건 방향으로 차이를 세 배 외삽한 예측이다. sample의 y=None은 실제 클래스를 균등 선택하는 동작으로 null 예측만의 무조건 생성과 다르다. 가이던스는 조건 제거를 학습한 모델을 전제로 한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
MNIST로 데이터 이름만 바꾸면 동일한 U-Net이 동작하는가?

</div>

??? success "연습문제 3 풀이"

    아니다. 크기가 28이 되어 28→14→7→3의 축소 경로와 3→6 확대 경로가 skip 7과 맞지 않는다. 기존 채널 불일치도 남는다. 입력 크기·skip 계약을 검토하고 학습·샘플링·평가를 같은 설정에 맞춰야 한다.



## 정리하며

조건 DDPM은 시간·클래스 임베딩을 결합하고 null 조건 학습으로 CFG를 구성한다. 현재 코드에는 디코더 채널·skip 저장 규칙과 MNIST 공간 크기 문제가 있다. 가이던스와 클래스별 저장 격자를 실제 학습 성공이나 조건 충실성의 증거로 해석하지 않는다.
