# 잠재 공간에서 학습하는 확산 모델

잠재 확산은 이미지를 인코더로 압축한 표현에서 확산을 학습하고, 생성한 잠재값을 디코더로 이미지에 되돌리는 접근이다. 픽셀 대신 작은 공간 표현을 사용해 계산을 줄일 수 있지만 인코더·디코더의 정보 로스와 비용도 포함해야 한다. 이 예제는 작은 VAE를 먼저 학습한 뒤 고정하고 잠재 노이즈 예측기를 학습하는 두 단계 구성을 살펴본다.

## 1. 코드

??? note "코드 (421줄)"

    ```python
    """숨은 확산."""
    # ==========================================
    # 숨은 확산 모델(LDM)
    # - Stable Diffusion의 핵심 생각
    # - 누른 숨은 공간에서의 확산(화소 공간이 아님)
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
    EPOCHS_VAE = 10         # 먼저 오토인코더를 익힌다
    EPOCHS_DIFF = 5         # 그런 다음 확산을 익힌다
    T = 1000
    BASE_CH = 64
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SAVE_PATH = "latent_diffusion_samples.png"
    SEED = 42

    # ==========================================
    # 1) 오토인코더(그림을 숨은 값으로 누른다)
    # ==========================================
    class Encoder(nn.Module):
        """그림을 숨은 공간으로 누른다"""
        def __init__(self, in_ch=IN_CHANNELS, latent_dim=LATENT_DIM):
            super().__init__()
            # 단순한 인코더: 32x32 -> 8x8
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
            # 단순한 디코더: 8x8 -> 32x32
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
        """누른 나타냄을 배우는 변분 오토인코더"""
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
    # 2) 숨은 확산 U-Net
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

            # 인코더
            self.in_conv = nn.Conv2d(latent_dim, base_ch, 3, padding=1)
            self.down1 = ResBlock(base_ch, base_ch * 2, time_dim)
            self.down2 = ResBlock(base_ch * 2, base_ch * 4, time_dim)

            # 가운데
            self.mid1 = ResBlock(base_ch * 4, base_ch * 4, time_dim)
            self.mid2 = ResBlock(base_ch * 4, base_ch * 4, time_dim)

            # 디코더
            self.up1 = ResBlock(base_ch * 8, base_ch * 2, time_dim)  # down2에서 이어 붙인다
            self.up2 = ResBlock(base_ch * 4, base_ch, time_dim)      # down1에서 이어 붙인다

            self.out_norm = nn.GroupNorm(8, base_ch)
            self.out_conv = nn.Conv2d(base_ch, latent_dim, 3, padding=1)

        def forward(self, z, t):
            t_emb = self.time_emb(t)

            # 인코더
            h = self.in_conv(z)
            h1 = self.down1(h, t_emb)
            h2 = self.down2(h1, t_emb)

            # 가운데
            h = self.mid1(h2, t_emb)
            h = self.mid2(h, t_emb)

            # 건너뛰는 이음을 갖춘 디코더
            h = self.up1(torch.cat([h, h2], dim=1), t_emb)
            h = self.up2(torch.cat([h, h1], dim=1), t_emb)

            return self.out_conv(F.silu(self.out_norm(h)))

    # ==========================================
    # 3) 숨은 확산 과정
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
        """숨은 공간의 확산 모델"""
        def __init__(self, vae, unet, timesteps=T):
            super().__init__()
            self.vae = vae
            self.unet = unet
            self.timesteps = timesteps

            # 확산 익히기 동안 변분 오토인코더를 얼린다
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
            """익히기 로스: 숨은 값으로 부호화하고 잡음을 더한 뒤 잡음을 헤아린다"""
            # 숨은 공간으로 부호화한다
            with torch.no_grad():
                z0 = self.vae.encode(x0)

            # 아무 때 스텝을 뽑는다
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

                # 잡음 없애기 스텝
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
        """먼저 오토인코더를 익힌다"""
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

                # 되살림 로스
                recon_loss = F.mse_loss(recon, x, reduction='sum') / x.shape[0]

                # KL 벌어짐
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0]

                # 전체 로스
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
        """그런 다음 숨은 공간에서 확산을 익힌다"""
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

        # 먼저 오토인코더를 익힌다
        train_vae(vae, loader, EPOCHS_VAE)

        # 숨은 확산 모델을 만든다
        ldm = LatentDiffusion(vae, unet, timesteps=T).to(DEVICE)

        # 숨은 공간에서 확산을 익힌다
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
        main()
    ```

## 2. VAE의 공간 표현

기본 CIFAR-10 입력은 B×3×32×32다. 인코더는 stride 2 컨볼루션 두 개로 32→16→8을 만들고 마지막 컨볼루션의 8채널을 μ와 logvar로 나눈다. 각 통계량은 **B×4×8×8**이며 잠재 표본도 같은 모양이다. LATENT_DIM=4는 전체 잠재 차원이 아니라 **잠재 채널 수**이고 한 이미지의 잠재 스칼라 수는 256이다.

디코더는 4→128채널 컨볼루션 뒤 공간을 두 배 확대하고, 64채널 컨볼루션 뒤 다시 확대해 32×32로 돌아온다. 마지막 tanh는 −1~1 이미지 범위와 맞춘다. LATENT_SCALE=4는 이 **공간 축소 배수**이며 잠재값의 진폭을 4배 조정하는 인수가 아니다.

MNIST의 28×28에서는 잠재 크기가 7×7이고 복원도 28×28이다. 일반 크기에서는 인코더의 올림 축소와 sample의 img_size//4가 일치하지 않을 수 있으므로 4의 배수 크기를 전제로 확인해야 한다. 이 구현은 정사각형 생성 크기를 받는다.

## 3. 첫 단계의 목적함수

재구성 오차는 픽셀 제곱오차를 이미지 안에서 합산하고 배치로 나눈 값이다. KL은 모든 잠재 좌표에 걸쳐 표준 정규 사전과의 차이를 합산하고 배치로 나눈다.

$$R=\frac1B\sum_i\|\widehat x_i-x_i\|^2,\qquad
K=\frac1{2B}\sum_{i,j}\left(\mu_{ij}^2+e^{\ell_{ij}}-1-\ell_{ij}\right)$$

여기서 ℓ은 logvar이며 최종 로스는 R+0.001K다. 재구성 MSE와 KL의 상대 웨이트는 명시적으로 선택한 값이다. 관측 분산·상수·전체 배율을 정하지 않고 이 숫자를 그대로 표준 가우시안 VAE의 정확한 음의 ELBO라고 부르면 안 된다.

`encode`는 μ만 반환하지 않고 리파라메리제이션으로 **확률적인 잠재 표본**을 만든다. 같은 이미지를 다시 인코딩해도 값이 달라질 수 있다. 두 번째 단계는 이 집계 인코딩 분포를 학습하는 것이며 이미지 분포를 로스 없이 그대로 옮긴 것은 아니다. 작은 KL 웨이트나 4채널이라는 조건만으로 잠재의 분산이 정확히 1이라고 보장할 수도 없다. 코드에는 잠재 표준편차 추정이나 별도 진폭 정규화가 없다.

## 4. VAE를 고정한 잠재 노이즈 학습

LatentDiffusion을 만들 때 VAE 파라미터의 requires_grad를 False로 바꾸고 loss에서도 no_grad로 인코딩한다. optimizer는 잠재 예측기의 파라미터만 받는다. 이는 VAE 파라미터 업데이트를 막지만 eval 전환과는 다르다. 현재 VAE에는 BatchNorm·Dropout이 없어 모드 차이가 없지만 다른 오토인코더를 넣을 때는 고정·추론 모드를 함께 관리해야 한다.

잠재 관측은 다음과 같다.

$$z_t=\sqrt{\bar\alpha_t}z_0+\sqrt{1-\bar\alpha_t}\epsilon$$

1,000개 코사인 β의 시간 인덱스를 표본별로 고르고 ε와 예측의 원소 평균 MSE를 계산한다. 이 목적은 잠재에서의 단순 노이즈 회귀이며 원래 픽셀 이미지의 정확한 로그 우도를 직접 계산하지 않는다.

## 5. LatentUNet은 공간 해상도를 유지한다

기본 4×8×8 잠재는 64채널 입력 투영, down1의 128채널, down2의 256채널, 256채널 middle 두 개를 거친다. 그러나 이름과 달리 down1·down2에는 **공간 다운샘플링이 없다**. up1·up2도 크기를 키우지 않는다. 모든 공간 해상도는 8×8로 유지된다.

middle과 h2를 연결한 512채널을 up1이 128채널로 바꾸고, h1의 128채널을 연결한 256채널을 up2가 64채널로 바꾼다. 출력은 다시 4×8×8이다. 잔차·시간별 scale·shift와 GroupNorm은 있지만 공간 다중 해상도 U-Net이나 어텐션·텍스트 조건은 없다. GroupNorm의 채널 나눗셈 조건도 맞춰야 한다.

## 6. 잠재 생성과 이미지 복원

sample은 N×4×8×8 표준 정규 노이즈에서 시작해 1,000번 역전이를 수행한다. 역평균은 일반 노이즈 예측 DDPM 식이고 잡음 분산은 **βt**다. 계산한 alpha_prev 변수는 사용하지 않으며 β̃t 사후분산을 쓰는 구현이 아니다. 마지막에는 잡음을 생략하고 VAE 디코더로 이미지를 만든다.

잠재는 이미지처럼 −1~1로 clipping하지 않는다. 최종 디코더의 tanh와 표시용 0~1 변환을 잠재 분포의 처리와 구분한다. 스케줄은 buffer로 저장하지만 초기 상태와 시간은 전역 DEVICE·LATENT_DIM·LATENT_SCALE을 참조하므로 인수와 모델 구성을 바꿀 때 모두 일치해야 한다.

## 7. 학습량과 비용의 해석

기본 CIFAR-10 학습은 VAE 10에포크와 잠재 확산 5에포크다. 배치 128에서 각 에포크 391배치이므로 각각 3,910번과 1,955번 갱신하도록 구성되어 있다. 마지막 배치 80개도 포함하며 로그는 배치 평균의 동일 가중 평균이다. 검증·체크포인트·EMA·품질 지표는 없다.

원본 텐서는 3×32×32=3,072개, 잠재 텐서는 4×8×8=256개 스칼라이므로 **텐서 원소 수는 12배 작다**. 출력의 '샘플당 메모리가 약 12배 감소'는 전체 학습·추론 메모리를 측정한 결과가 아니다. VAE·잠재 예측기의 파라미터와 엑티베이션, optimizer·그래디언트와 인코딩·디코딩 비용도 필요하다. 채널 수가 늘어난 잠재 네트워크의 비용이 원소 수 비율과 같지는 않다.

[잠재 확산 논문](https://arxiv.org/abs/2112.10752)의 핵심 아이디어와 연결되지만, 이 작은 학습 VAE·잔차 CNN은 텍스트 조건 Stable Diffusion 전체 구현이나 그 품질 재현이 아니다. 실제 생성 품질은 오토인코더 재구성, 잠재 적합도와 최종 이미지 평가를 구분해 확인해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
CIFAR-10 이미지와 잠재의 모양 및 원소 수 비율을 설명하라.

</div>

??? success "풀이"

    이미지는 B×3×32×32, μ·logvar·잠재 표본은 각각 B×4×8×8이다. 한 이미지의 3072개 스칼라가 잠재 256개로 줄어 원소 수 비율은 12다. 전체 모델·엑티베이션·optimizer를 포함한 메모리가 반드시 12배 줄었다는 뜻은 아니다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
이 코드의 encode는 결정적인가? 두 번째 단계는 VAE 파라미터를 업데이트하는가?

</div>

??? success "풀이"

    encode는 μ와 logvar로 새 정규 잡음을 샘플링하므로 결정적이지 않다. 두 번째 단계에서는 VAE 파라미터 그래디언트를 끄고 no_grad 인코딩을 사용해 업데이트하지 않는다. 다만 파라미터 고정과 eval은 다른 제어이며 임의의 다른 VAE 구조에 대해서는 둘 다 확인해야 한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
LatentUNet의 down·up 이름과 실제 공간 연산, 역분산 선택을 설명하라.

</div>

??? success "풀이"

    down1·down2는 채널을 늘리는 잔차 블록이고 공간 크기는 유지한다. up1·up2도 채널 결합 후 폭을 줄일 뿐 해상도를 키우지 않는다. 샘플러는 βt 분산을 사용하며 alpha_prev는 계산만 하고 쓰지 않는다. 따라서 공간 다중 해상도 U-Net이나 사후분산 사용이라고 설명하면 실제 코드와 다르다.



## 정리하며

잠재 확산은 먼저 학습한 확률적 이미지 표현을 고정하고 그 잠재에서 노이즈를 예측한다. 잠재 채널·공간 축소와 진폭 정규화, 텐서 원소 수와 전체 비용을 구분한다. 이 예제는 같은 공간 해상도의 잔차 예측기와 β 분산 샘플러를 사용하는 작은 구성이다.
