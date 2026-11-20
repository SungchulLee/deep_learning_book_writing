# VAE 학습 절차

이 학습 스크립트는 MNIST에서 완전 연결 VAE 또는 컨볼루션 VAE를 선택하고 재구성·KL 로스를 함께 최적화한다. 핵심은 모델별 입력 모양, 로스의 합산 기준과 평가·체크포인트 선택 규칙을 일치시키는 것이다.

## 1. 학습과 평가 구현

```python
"""
표준 변분 오토인코더 학습 스크립트
"""

import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ========================================================================
# 모델
# ========================================================================

class VAE(nn.Module):
    """
    가우시안 잠재 공간을 가진 표준 변분 오토인코더.
    
    인수:
        input_dim (int): 입력 차원(예컨대 MNIST는 784)
        hidden_dim (int): 은닉 레이어 차원
        latent_dim (int): 잠재 공간 차원
    """
    
    def __init__(self, input_dim=784, hidden_dim=256, latent_dim=32):
        super(VAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # 인코더: 잠재 분포의 파라메터를 출력한다
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 잠재 분포의 평균과 로그 분산
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # 디코더
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        """
        입력을 잠재 분포 파라메터로 인코딩한다.
        
        인수:
            x: 입력 텐서
            
        반환값:
            mu: 잠재 분포의 평균
            logvar: 잠재 분포의 로그 분산
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        리파라메리제이션 기법: z = mu + sigma * epsilon
        여기서 epsilon ~ N(0, 1)
        
        그러면 역전파를 위해 샘플링을 미분할 수 있다.
        
        인수:
            mu: 잠재 분포의 평균
            logvar: 잠재 분포의 로그 분산
            
        반환값:
            z: 샘플링한 잠재 벡터
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        """
        잠재 표현을 출력으로 디코딩한다.
        
        인수:
            z: 잠재 벡터
            
        반환값:
            reconstruction: 복원한 출력
        """
        return self.decoder(z)
    
    def forward(self, x):
        """
        전체 순전파: 인코딩 -> 리파라메리제이션 -> 디코딩
        
        인수:
            x: 입력 텐서
            
        반환값:
            reconstruction: 복원한 출력
            mu: 잠재 분포의 평균
            logvar: 잠재 분포의 로그 분산
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar
    
    def loss_function(self, reconstruction, x, mu, logvar, beta=1.0):
        """
        변분 오토인코더 로스 = 복원 로스 + β * KL 발산
        
        인수:
            reconstruction: 복원한 출력
            x: 원래 입력
            mu: 잠재 분포의 평균
            logvar: 잠재 분포의 로그 분산
            beta: KL 발산 항의 웨이트
            
        반환값:
            loss: 전체 변분 오토인코더 로스
            bce: 복원 로스
            kld: KL 발산
        """
        # 복원 로스(이진 교차 엔트로피)
        BCE = F.binary_cross_entropy(reconstruction, x, reduction='sum')
        
        # KL 발산: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return BCE + beta * KLD, BCE, KLD
    
    def sample(self, num_samples, device='cpu'):
        """
        잠재 공간에서 표본을 생성한다.
        
        인수:
            num_samples: 생성할 표본의 개수
            device: 표본을 생성할 장치
            
        반환값:
            samples: 생성한 표본
        """
        z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decode(z)
        return samples


class ConvVAE(nn.Module):
    """
    이미지 데이터를 위한 컨볼루션 변분 오토인코더.
    
    인수:
        latent_dim (int): 잠재 공간 차원
        img_channels (int): 입력 이미지 채널 수(회색조 1, RGB 3)
        img_size (int): 입력 이미지 크기(정사각형 이미지로 가정한다)
    """
    
    def __init__(self, latent_dim=128, img_channels=1, img_size=28):
        super(ConvVAE, self).__init__()
        
        self.latent_dim = latent_dim
        self.img_channels = img_channels
        self.img_size = img_size
        
        # 인코더
        self.encoder = nn.Sequential(
            # 28x28 -> 14x14
            nn.Conv2d(img_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # 14x14 -> 7x7
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # 7x7 -> 4x4
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Flatten()
        )
        
        # 펼친 크기를 계산한다
        self.flatten_size = 128 * 4 * 4
        
        # 잠재 공간 파라메터
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)
        
        # 디코더 입력
        self.decoder_input = nn.Linear(latent_dim, self.flatten_size)
        
        # 디코더
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 4, 4)),
            
            # 4x4 -> 7x7
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # 7x7 -> 14x14
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # 14x14 -> 28x28
            nn.ConvTranspose2d(32, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        """
        입력 이미지를 잠재 분포 파라메터로 인코딩한다.
        
        인수:
            x: 입력 이미지 텐서 [배치 크기, 채널, 높이, 너비]
            
        반환값:
            mu: 잠재 분포의 평균
            logvar: 잠재 분포의 로그 분산
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        샘플링을 위한 리파라메리제이션 기법.
        
        인수:
            mu: 잠재 분포의 평균
            logvar: 잠재 분포의 로그 분산
            
        반환값:
            z: 샘플링한 잠재 벡터
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """
        잠재 표현을 이미지로 디코딩한다.
        
        인수:
            z: 잠재 벡터
            
        반환값:
            reconstruction: 복원한 이미지
        """
        h = self.decoder_input(z)
        reconstruction = self.decoder(h)
        return reconstruction
    
    def forward(self, x):
        """
        전체 순전파.
        
        인수:
            x: 입력 이미지 텐서
            
        반환값:
            reconstruction: 복원한 이미지
            mu: 잠재 분포의 평균
            logvar: 잠재 분포의 로그 분산
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar
    
    def loss_function(self, reconstruction, x, mu, logvar, beta=1.0):
        """
        변분 오토인코더 로스 함수.
        
        인수:
            reconstruction: 복원한 출력
            x: 원래 입력
            mu: 잠재 분포의 평균
            logvar: 잠재 분포의 로그 분산
            beta: KL 발산 항의 웨이트
            
        반환값:
            loss: 전체 변분 오토인코더 로스
            bce: 복원 로스
            kld: KL 발산
        """
        # 복원 로스
        BCE = F.binary_cross_entropy(reconstruction, x, reduction='sum')
        
        # KL 발산
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return BCE + beta * KLD, BCE, KLD
    
    def sample(self, num_samples, device='cpu'):
        """
        잠재 공간에서 표본을 생성한다.
        
        인수:
            num_samples: 생성할 표본의 개수
            device: 표본을 생성할 장치
            
        반환값:
            samples: 생성한 이미지 표본
        """
        z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decode(z)
        return samples


# ========================================================================
# 로스
# ========================================================================

def vae_loss(reconstruction, target, mu, logvar, beta=1.0, reduction="sum"):
    """VAE의 로스를 계산한다.

    인수:
        reconstruction: 디코더가 내놓은 값. [0,1]이어야 한다(시그모이드를 거친 것)
        target: 원래 입력. 마찬가지로 [0,1]이어야 한다
        mu, logvar: 인코더가 내놓은 잠재 분포의 평균과 로그 분산
        beta: KL 항의 가중치. 1이면 보통의 VAE
        reduction: 'sum'이면 배치 전체의 합, 'mean'이면 표본마다의 평균

    반환값:
        (전체 로스, 복원 로스, KL 로스)
    """
    # 목표가 [0,1] 밖이면 BCE가 거부한다. 미리 일러 주는 편이 낫다.
    if target.min() < 0 or target.max() > 1:
        raise ValueError(
            "target은 [0,1] 안에 있어야 한다. 픽셀을 정규화할 때 "
            "Normalize로 음수를 만들지 않았는지 살펴보라."
        )

    recon_loss = F.binary_cross_entropy(
        reconstruction.reshape(target.shape[0], -1),
        target.reshape(target.shape[0], -1),
        reduction="sum",
    )

    # KL(N(mu, sigma^2) || N(0, 1)) 의 닫힌 형태
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    if reduction == "mean":
        n = target.shape[0]
        recon_loss, kl_loss = recon_loss / n, kl_loss / n

    return recon_loss + beta * kl_loss, recon_loss, kl_loss


# ========================================================================
# 시각화 도우미
# ========================================================================

def _to_image(t):
    """텐서를 matplotlib이 받는 형태로 바꾼다.

    완전 연결 오토인코더는 펼친 벡터를 내놓으므로, 1차원으로 들어오면
    정사각형 이미지로 되접어 준다. 그러지 않으면 imshow가
    "Invalid shape (784,) for image data" 로 거부한다.
    """
    import math

    t = t.detach().cpu()
    if t.dim() == 1:                          # 펼친 벡터 -> 정사각형
        side = int(math.isqrt(t.numel()))
        if side * side == t.numel():
            t = t.view(side, side)
        return t, "gray"
    if t.dim() == 3 and t.shape[0] == 1:      # 회색조
        return t.squeeze(0), "gray"
    if t.dim() == 3:                          # 컬러: (C,H,W) -> (H,W,C)
        return t.permute(1, 2, 0), None
    return t, "gray"


def visualize_reconstruction(model, data_loader, num_images=10, device="cpu",
                             conditional=False, save_path=None):
    """원본 이미지와 복원 이미지를 위아래로 나란히 놓는다.

    인수:
        model: forward가 (복원 값, mu, logvar)를 돌려주는 오토인코더
        data_loader: 볼 데이터를 주는 로더
        num_images: 보여 줄 이미지 수
        device: 모델이 올라가 있는 장치
        conditional: 조건부 모델이면 True. 라벨도 함께 넣어 준다
        save_path: 주면 그림을 그 경로로 저장한다
    """
    model.eval()
    images, labels = next(iter(data_loader))
    images = images[:num_images].to(device)
    labels = labels[:num_images].to(device)

    with torch.no_grad():
        # 완전 연결 오토인코더는 펼친 입력을 받고, 컨볼루션 쪽은 이미지
        # 형태 그대로를 받는다. 어느 쪽인지 모르므로 이미지 형태로 먼저
        # 넣어 보고, 거부하면 펼쳐서 다시 넣는다.
        try:
            out = model(images, labels) if conditional else model(images)
        except RuntimeError:
            flat = images.view(images.size(0), -1)
            out = model(flat, labels) if conditional else model(flat)
    # forward가 (recon, mu, logvar)를 돌려주기도 하고 recon만 돌려주기도 한다
    recon = out[0] if isinstance(out, (tuple, list)) else out
    recon = recon.view_as(images)

    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 1.2, 3))
    for i in range(num_images):
        img, cmap = _to_image(images[i])
        axes[0, i].imshow(img, cmap=cmap)
        axes[0, i].axis("off")
        rec, cmap = _to_image(recon[i])
        axes[1, i].imshow(rec, cmap=cmap)
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Original")
    axes[1, 0].set_ylabel("Reconstruction")
    fig.suptitle("Top: original    Bottom: reconstruction")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig

def visualize_samples(model, latent_dim, num_samples=10, device="cpu",
                      class_label=None, save_path=None):
    """잠재 공간에서 무작위로 뽑아 디코더에 넣어 본다.

    복원과 달리 이쪽은 모델이 정말로 생성할 줄 아는지를 본다.
    사전분포 N(0, I)에서 뽑으므로, 학습이 잘되었다면 그럴듯한 이미지가 나와야 한다.
    """
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim, device=device)
        if class_label is not None:
            labels = torch.full((num_samples,), int(class_label),
                                dtype=torch.long, device=device)
            samples = model.decode(z, labels)
        else:
            samples = model.decode(z)

    fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 1.2, 1.6))
    if num_samples == 1:
        axes = [axes]
    for i in range(num_samples):
        img, cmap = _to_image(samples[i])
        axes[i].imshow(img, cmap=cmap)
        axes[i].axis("off")
    title = "Samples from the latent space"
    if class_label is not None:
        title += f" (class {class_label})"
    fig.suptitle(title)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ========================================================================
# 메인
# ========================================================================

def train_epoch(model, train_loader, optimizer, device, beta=1.0):
    """한 에포크 동안 학습한다"""
    model.train()
    train_loss = 0
    train_recon = 0
    train_kl = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for data, _ in pbar:
        data = data.to(device)
        
        # 완전 연결 변분 오토인코더를 위해 데이터를 펼친다
        if isinstance(model, VAE):
            data_input = data.view(data.size(0), -1)
        else:
            data_input = data
        
        # 순전파
        reconstruction, mu, logvar = model(data_input)
        
        # 로스를 계산한다
        if isinstance(model, VAE):
            target = data.view(data.size(0), -1)
        else:
            target = data
        
        loss, recon_loss, kl_loss = vae_loss(reconstruction, target, mu, logvar, beta)
        
        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 지표를 추적한다
        train_loss += loss.item()
        train_recon += recon_loss.item()
        train_kl += kl_loss.item()
        
        # 진행 막대를 갱신한다
        pbar.set_postfix({
            'loss': loss.item() / data.size(0),
            'recon': recon_loss.item() / data.size(0),
            'kl': kl_loss.item() / data.size(0)
        })
    
    num_samples = len(train_loader.dataset)
    return train_loss / num_samples, train_recon / num_samples, train_kl / num_samples


def test_epoch(model, test_loader, device, beta=1.0):
    """시험 배치로 평가한다"""
    model.eval()
    test_loss = 0
    test_recon = 0
    test_kl = 0
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            
            # 완전 연결 변분 오토인코더를 위해 데이터를 펼친다
            if isinstance(model, VAE):
                data_input = data.view(data.size(0), -1)
            else:
                data_input = data
            
            # 순전파
            reconstruction, mu, logvar = model(data_input)
            
            # 로스를 계산한다
            if isinstance(model, VAE):
                target = data.view(data.size(0), -1)
            else:
                target = data
            
            loss, recon_loss, kl_loss = vae_loss(reconstruction, target, mu, logvar, beta)
            
            test_loss += loss.item()
            test_recon += recon_loss.item()
            test_kl += kl_loss.item()
    
    num_samples = len(test_loader.dataset)
    return test_loss / num_samples, test_recon / num_samples, test_kl / num_samples


def main(args):
    # 장치 지정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # MNIST 데이터셋 불러오기
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # 모델 생성
    if args.model_type == 'fc':
        model = VAE(input_dim=784, hidden_dim=args.hidden_dim, latent_dim=args.latent_dim)
    elif args.model_type == 'conv':
        model = ConvVAE(latent_dim=args.latent_dim, img_channels=1, img_size=28)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    model = model.to(device)
    print(f"Model: {model.__class__.__name__}")
    print(f"Latent dimension: {args.latent_dim}")
    print(f"Beta: {args.beta}")
    
    # 옵티마이저
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 학습 루프
    best_test_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # 학습
        train_loss, train_recon, train_kl = train_epoch(model, train_loader, optimizer, device, args.beta)
        
        # 시험
        test_loss, test_recon, test_kl = test_epoch(model, test_loader, device, args.beta)
        
        # 통계를 출력한다
        print(f"Train - Loss: {train_loss:.4f}, Recon: {train_recon:.4f}, KL: {train_kl:.4f}")
        print(f"Test  - Loss: {test_loss:.4f}, Recon: {test_recon:.4f}, KL: {test_kl:.4f}")
        
        # 최고 성능 모델 저장
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': test_loss,
            }, args.checkpoint_path)
            print(f"Saved checkpoint to {args.checkpoint_path}")
    
    # 가장 좋은 모델을 불러와 시각화한다
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("\nGenerating visualizations...")
    visualize_reconstruction(model, test_loader, num_images=10, device=device, conditional=False)
    visualize_samples(model, args.latent_dim, num_samples=10, device=device)
    
    print("\nTraining complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train VAE on MNIST')
    
    # 모델 인자
    parser.add_argument('--model-type', type=str, default='fc', choices=['fc', 'conv'],
                        help='Type of VAE (fc or conv)')
    parser.add_argument('--latent-dim', type=int, default=32,
                        help='Latent dimension')
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='Hidden dimension (for FC VAE)')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='Beta parameter for KL weight')
    
    # 학습 인자
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    
    # 체크포인트
    parser.add_argument('--checkpoint-path', type=str, default='checkpoints/vae_model.pt',
                        help='Path to save checkpoint')
    
    args = parser.parse_args()
    
    # 체크포인트 디렉터리를 만든다
    import os
    os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)
    
    main(args)
```

모델(`VAE`·`ConvVAE`)과 로스(`vae_loss`)·시각화 도우미는 모두 같은 코드 블록 안에 정의되어 있다. 데이터 경로 `./data`는 실행 작업 디렉터리에 따라 달라진다. 기본 장치는 CUDA 사용 가능 시 CUDA, 아니면 CPU다. 학습 데이터는 다운로드를 요청하지만 시험 데이터 생성에는 download=True를 지정하지 않으므로 필요한 파일이 준비되어 있어야 한다.

기본 설정은 잠재 32, 완전 연결 은닉 256, 배치 128, Adam 러닝레이트 0.001, β=1, 20에포크다. ToTensor로 입력을 0~1 범위로 바꾸고 별도 표준화나 이진화는 하지 않는다. 완전 연결 VAE만 `(B,784)`로 펼치며 컨볼루션 모델은 `(B,1,28,28)`을 받는다. isinstance 분기는 이 두 구체적 클래스에 맞춘 방식이지 모든 VAE 구조에 대한 일반적인 입력 판별 방법은 아니다.

## 2. 로스 기록의 단위

위 vae_loss는 기본적으로 픽셀·잠재 차원·배치에 대해 합산한 BCE와 KL을 반환한다. 학습 루프는 배치 로스 합으로 역전파하고, 기록에서는 합을 누적한 뒤 데이터셋 표본 수로 나눈다. 따라서 출력은 표본당 픽셀 BCE 합과 표본당 잠재 KL 합이다. 픽셀별 평균 BCE와 같은 수치가 아니다.

현재 로더는 전체 데이터를 한 번씩 사용하므로 len(dataset)이 분모와 맞는다. 샘플러·drop_last 등을 바꾸면 실제 처리한 표본 수를 세어야 한다. 배치 합으로 역전파하므로 배치 크기를 바꾸면 그래디언트 스케일도 달라진다. β가 1이 아닌 기록의 Loss는 표준 음의 ELBO가 아니며 비교용 표준 ELBO를 따로 기록할 수 있다.

## 3. 평가와 체크포인트의 해석

test_epoch는 eval과 no_grad를 적용한다. BatchNorm 통계 사용과 그래디언트 기록은 바뀌지만 두 모델의 리파라메리제이션(reparameterization)은 계속 표본을 뽑으므로 평가 재구성에는 몬테카를로 변동이 남는다. train 지표는 학습 도중 서로 다른 웨이트에서 누적한 값이므로 에포크 종료 모델의 train 평가와도 다르다.

코드는 시험 로스로 최적 에포크를 선택한다. 독립적인 최종 시험 성능을 보고하려면 학습 데이터에서 검증셋을 따로 만들고 선택에 사용해야 한다. 현재 저장에는 웨이트·옵티마이저 상태·에포크·로스만 포함된다. 다른 환경에서 복원하려면 구조 설정, 전처리, β와 합산 기준 등도 별도로 기록해야 한다.

저장된 최적 웨이트를 불러온 뒤 시각화 보조 함수를 호출한다. 그림을 반환하는 코드이며 호출부에는 저장 경로나 명시적인 표시 호출이 없다. 생성 그림의 파일 저장까지 자동으로 완료된다고 설명할 수 없다. 이 절은 코드의 동작을 설명하며 새 학습 결과를 제시하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
zero_grad를 제거하면 어떤 일이 일어나는가? 의도한 그래디언트 누적과 구분하라.

</div>

??? success "풀이"

    backward는 기존 grad에 새 그래디언트를 더한다. 매 배치 step을 수행하면서 초기화하지 않으면 이전 파라메터 위치에서 계산한 그래디언트까지 재사용하므로 원래 최적화와 달라진다. 이것이 단순히 러닝레이트에 단계 수를 곱한 것과 같거나 반드시 발산하는 것은 아니다. 의도한 누적은 여러 미니배치의 로스를 적절히 스케일해 backward한 뒤 한 번 step하고 초기화하는 방식이다. 누적 구간의 마지막 불완전 배치도 처리해야 한다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
본문 옵티마이저를 확인하고 대안과 비교하는 절차를 설계하라.

</div>

??? success "풀이"

    본문은 이미 Adam과 기본 러닝레이트 0.001을 사용한다. 따라서 “Adam으로 변경”은 새로운 실험이 아니다. SGD 등과 비교하려면 같은 초기 웨이트·데이터 분할·평가 지표를 사용하고 각 옵티마이저에 같은 튜닝 예산을 준다. 데이터 순서와 잠재 표본의 변동도 관리하며 여러 초기화의 검증 성능·비용을 비교한다. 같은 러닝레이트 하나만으로 옵티마이저의 일반적인 우열을 정하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
사후 붕괴와 유한하지 않은 로스(NaN·Inf)를 구분해 진단하라.

</div>

??? success "풀이"

    붕괴 가능성은 입력별·차원별 KL, 평균의 변동과 디코더의 잠재 표현 의존성으로 확인한다. 낮은 KL 차원이 있다는 사실만으로 학습 실패인 것은 아니다. NaN·Inf는 입력 범위, logvar.exp의 크기, 출력·그래디언트와 러닝레이트 등을 점검한다. 클리핑·초기화·정밀도 조정이나 KL 일정은 원인에 맞춰 선택해야 하며 모든 문제에 공통으로 효과적인 처방은 아니다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 조기 종료 예시에서 최적 웨이트 보존이 불완전한 이유를 설명하라.

</div>

??? success "풀이"

    state_dict().copy()는 딕셔너리의 얕은 복사라 텐서 저장소가 학습 중 값과 공유될 수 있다. 독립 복제나 파일 저장이 필요하다. 예시는 evaluate와 val_loader가 정의되어 있어야 하고 실제 학습 단계도 생략되어 있다. 조기 종료 조건이 발생할 때만 복원하므로 정상 종료 때도 최적 상태를 복원하도록 설계해야 한다. 체크포인트 선택에는 검증 데이터를 사용하고 최종 시험 데이터는 분리한다.

    ```python
    best_loss = float('inf')
    patience_counter = 0
    best_state = None
    for epoch in range(num_epochs):
        # ... 학습 단계 ...
        val_loss = evaluate(model, val_loader)
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
        if patience_counter >= 10:
            print(f'Early stopping at epoch {epoch}')
            model.load_state_dict(best_state)
            break
    ```


---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff easy" title="쉬움"></span>
재구성·KL·β를 별도로 기록해야 하는 이유는 무엇인가?

</div>

??? success "풀이"

    전체 로스가 내려가도 어느 항이 개선되었는지 알 수 없고 β 일정이 바뀌면 같은 모델에서도 숫자가 달라진다. 아래 기록은 rec_sum·kl_sum이 합이고 n이 처리 표본 수라는 전제에서 표본당 항을 표시한다. KL 감소는 정보 비용 감소나 집계 사후분포 일치 등 여러 원인이 있고 증가도 반드시 더 유용한 정보 획득을 뜻하지 않는다. 차원별 KL과 잠재 표현 개입·재구성·생성 평가를 함께 사용한다.

    ```python
    print(f"epoch {ep}  rec {rec_sum/n:.2f}  kl {kl_sum/n:.2f}")
    ```


---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
재구성 그림과 사전 생성 그림의 차이를 설명하라.

</div>

??? success "풀이"

    재구성은 관측에서 추론한 q(z|x)를 거쳐 출력을 만들고 사전 생성은 p(z)에서 직접 뽑는다. 두 경로가 방문하는 잠재 영역과 평가하는 능력이 달라 좋은 재구성이 좋은 생성의 충분조건은 아니다. 차이는 사전분포 불일치뿐 아니라 디코더·관측 모델·학습 부족 등에도 영향을 받는다. 픽셀 확률 이미지를 보는 것과 관측 픽셀을 샘플링하는 것도 구분한다. 특정 확신도 임계 비율을 실패 확률로 해석하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
언제 멈출지 결정하는 지표를 목적별로 설명하라.

</div>

??? success "풀이"

    가능도(likelihood, 우도라고도 한다) 목적에는 같은 관측 모델의 검증 표준 ELBO나 적절한 중요도 가중 평가를, 생성에는 검증용 품질·다양성 지표를, 표현 재사용에는 별도 검증 작업의 성능을 사용할 수 있다. 각 지표에는 변동과 편향이 있으므로 정한 평가 예산·인내 횟수로 선택한다. 병목이 작아도 과적합이 불가능하지 않고 낮은 KL 차원도 반드시 최적화 실패는 아니다. 최종 시험셋을 선택 과정에 반복 사용하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
KL warm-up을 본문 루프에 연결할 때 주의할 점은 무엇인가?

</div>

??? success "풀이"

    에포크나 스텝별 현재 β를 계산해 train_epoch에 전달하고 일정·목표값을 기록한다. 아래는 개념적 골격이며 loader의 `(x,)`는 MNIST의 `(data,label)` 반환과 다르다. rec·kl과 optimizer 단계도 정의·연결해야 한다. warmup_epochs는 양수여야 하고 0에서 시작하는 ep의 종료 범위에 따라 마지막 값이 목표에 도달하는지 확인한다. β가 변하는 학습 로스 대신 고정 β=1의 검증 지표를 별도로 기록하면 시점 간 비교가 쉽다. warm-up의 필요성과 효과는 진단해야 하며 일정한 권장 비율을 보장할 수 없다.

    ```python
    for ep in range(epochs):
        beta = min(1.0, ep / warmup_epochs)        # 0 -> 1
        for x, in loader:
            out, mu, logvar = model(x)
            loss = rec(out, x) + beta * kl(mu, logvar)
    ```


---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
배치 크기·러닝레이트 선택에서 본문 구현의 어떤 특성을 고려해야 하는가?

</div>

??? success "풀이"

    기본값은 배치 128과 Adam 러닝레이트 0.001이며 보편적인 최적값은 아니다. 배치 합 로스, 마지막 배치 크기, 컨볼루션 모델의 BatchNorm과 잠재 샘플링 변동이 학습에 영향을 준다. 배치를 바꾸면 정규화와 스텝 수·시간 예산도 함께 고려한다. Adam이 임의의 로스 스케일 변화나 β 변경을 자동으로 상쇄하는 것은 아니다. 검증 기준과 계산비용으로 후보를 비교한다.



---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff hard" title="어려움"></span>
재구성 곡선만으로 알 수 없는 실패와 추가 진단을 설명하라.

</div>

??? success "풀이"

    사전 생성의 다양성 부족, 잠재 미사용, 집계 사후분포 불일치와 수치적으로 극단적인 분산은 재구성 곡선만으로 구분하기 어렵다. 차원별 KL·평균·로그분산 분포, 사전분포 표본과 잠재 표현 개입 결과를 함께 본다. σ→0이면 표준정규 사전분포에 대한 KL은 일반적으로 발산하지만 재구성이 반드시 단조 개선되는 것은 아니다. 생성 품질 저하만으로 prior hole의 크기를 직접 측정할 수도 없다. 지표를 진단 근거로 사용하되 단일 패턴에 유일한 원인을 붙이지 않는다.



## 정리하며

VAE 학습에서는 입력 모양과 배치 합 로스를 확인하고 재구성·KL·β를 함께 기록한다. 검증셋으로 모델을 선택하며 사전 생성도 별도로 평가한다. 본문의 시험셋 기반 선택과 최소 체크포인트는 그 한계를 이해하고 사용해야 한다.
