# β-VAE의 학습과 비교

β-VAE 실험에서는 β마다 다른 학습 로스를 최소화하므로 전체 로스만으로 설정을 비교하면 안 된다. 재구성·표준 KL·표준 ELBO를 같은 단위로 기록하고 생성이나 표현 분리 등 목적별 지표를 추가한다. 이 절은 제공된 스크립트의 동작과 한계를 정리한다.

## 1. 학습 스크립트

```python
"""
β-변분 오토인코더(베타 VAE) 학습 스크립트
β 파라메터로 분리된 표현(disentangled representation)을 학습한다
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

class BetaVAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, latent_dim=32, beta=4.0):
        super().__init__()
        self.input_dim, self.latent_dim, self.beta = input_dim, latent_dim, beta
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim), nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def loss_function(self, reconstruction, x, mu, logvar):
        BCE = F.binary_cross_entropy(reconstruction, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + self.beta * KLD, BCE, KLD

    def traverse_latent_dimension(self, dim_idx, num_steps=10, range_limit=3.0, device='cpu'):
        z = torch.zeros(num_steps, self.latent_dim).to(device)
        z[:, dim_idx] = torch.linspace(-range_limit, range_limit, num_steps).to(device)
        return self.decoder(z)


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


class ConvBetaVAE(ConvVAE):
    """컨볼루션 beta-VAE.

    구조는 ConvVAE와 같고, 로스에서 KL 항에 주는 가중치 beta만 다르다.
    beta를 키우면 잠재 축이 서로 분리되는 쪽으로 더 세게 밀리는 대신
    복원이 흐려진다.
    """

    def __init__(self, latent_dim=128, img_channels=1, beta=4.0):
        super().__init__(latent_dim=latent_dim, img_channels=img_channels)
        self.beta = beta


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

def beta_vae_loss(reconstruction, target, mu, logvar, beta=4.0, reduction="sum"):
    """beta-VAE의 로스. vae_loss에 beta를 크게 준 것과 같다.

    beta를 키우면 잠재 축이 서로 분리되는 쪽으로 더 세게 밀리지만, 그만큼
    복원이 흐려진다. 이 트레이드오프가 beta-VAE의 요점이다.
    """
    return vae_loss(reconstruction, target, mu, logvar, beta=beta, reduction=reduction)


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

def _decode(model, z):
    """잠재 벡터 z를 이미지로 디코딩한다.

    VAE·ConvVAE·ConvBetaVAE는 decode 메서드가 있고, 완전 연결 BetaVAE는
    decoder 모듈만 있다. 둘 다 받아 준다.
    """
    return model.decode(z) if hasattr(model, "decode") else model.decoder(z)



def visualize_latent_traversal(model, dim_idx, num_steps=10, range_limit=3.0,
                               device="cpu", save_path=None):
    """잠재 축 하나만 -range_limit에서 +range_limit까지 움직이고 나머지는 0에 둔다.

    축 하나가 한 가지 변이 요인(굵기·기울기 따위)만 맡는다면, 그 축을 따라
    이미지가 그 요인만 바뀌며 이어져야 한다. 이것이 표현 분리를 눈으로 보는 길이다.
    """
    model.eval()
    with torch.no_grad():
        if hasattr(model, "traverse_latent_dimension"):
            images = model.traverse_latent_dimension(dim_idx, num_steps, range_limit, device)
        else:
            z = torch.zeros(num_steps, model.latent_dim, device=device)
            z[:, dim_idx] = torch.linspace(-range_limit, range_limit, num_steps, device=device)
            images = _decode(model, z)

    fig, axes = plt.subplots(1, num_steps, figsize=(num_steps * 1.2, 1.6))
    for i in range(num_steps):
        img, cmap = _to_image(images[i])
        axes[i].imshow(img, cmap=cmap)
        axes[i].axis("off")
    fig.suptitle(f"Latent dimension {dim_idx}: {-range_limit:+.1f} to {range_limit:+.1f}")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ========================================================================
# 메인
# ========================================================================

def train_epoch(model, train_loader, optimizer, device):
    """한 에포크 동안 학습한다"""
    model.train()
    train_loss = 0
    train_recon = 0
    train_kl = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for data, _ in pbar:
        data = data.to(device)
        
        # 완전 연결 β-변분 오토인코더를 위해 데이터를 펼친다
        if isinstance(model, BetaVAE):
            data_input = data.view(data.size(0), -1)
        else:
            data_input = data
        
        # 순전파
        reconstruction, mu, logvar = model(data_input)
        
        # 로스를 계산한다
        if isinstance(model, BetaVAE):
            target = data.view(data.size(0), -1)
        else:
            target = data
        
        loss, recon_loss, kl_loss = model.loss_function(reconstruction, target, mu, logvar)
        
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


def test_epoch(model, test_loader, device):
    """시험 배치로 평가한다"""
    model.eval()
    test_loss = 0
    test_recon = 0
    test_kl = 0
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            
            # 완전 연결 β-변분 오토인코더를 위해 데이터를 펼친다
            if isinstance(model, BetaVAE):
                data_input = data.view(data.size(0), -1)
            else:
                data_input = data
            
            # 순전파
            reconstruction, mu, logvar = model(data_input)
            
            # 로스를 계산한다
            if isinstance(model, BetaVAE):
                target = data.view(data.size(0), -1)
            else:
                target = data
            
            loss, recon_loss, kl_loss = model.loss_function(reconstruction, target, mu, logvar)
            
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
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # 모델 생성
    if args.model_type == 'fc':
        model = BetaVAE(
            input_dim=784,
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim,
            beta=args.beta
        )
    elif args.model_type == 'conv':
        model = ConvBetaVAE(
            latent_dim=args.latent_dim,
            beta=args.beta,
            img_channels=1
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    model = model.to(device)
    print(f"Model: {model.__class__.__name__}")
    print(f"Latent dimension: {args.latent_dim}")
    print(f"Beta (disentanglement weight): {args.beta}")
    print(f"  β=1: Standard VAE")
    print(f"  β>1: More disentangled (β=4-10 typical)")
    
    # 옵티마이저
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 학습 루프
    best_test_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # 학습
        train_loss, train_recon, train_kl = train_epoch(model, train_loader, optimizer, device)
        
        # 시험
        test_loss, test_recon, test_kl = test_epoch(model, test_loader, device)
        
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
                'beta': args.beta,
            }, args.checkpoint_path)
            print(f"Saved checkpoint to {args.checkpoint_path}")
    
    # 가장 좋은 모델을 불러와 시각화한다
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("\nGenerating visualizations...")
    
    # 기본 시각화
    visualize_reconstruction(model, test_loader, num_images=10, device=device, conditional=False)
    visualize_samples(model, args.latent_dim, num_samples=10, device=device)
    
    # β-변분 오토인코더 전용: 표현 분리를 보이는 잠재 축 탐색
    print("\nGenerating latent dimension traversals (disentanglement visualization)...")
    num_dims_to_traverse = min(10, args.latent_dim)
    for dim_idx in range(num_dims_to_traverse):
        print(f"  Traversing dimension {dim_idx}...")
        visualize_latent_traversal(model, dim_idx=dim_idx, num_steps=10, range_limit=3.0, device=device)
    
    print("\nTraining complete!")
    print(f"\nInterpretation tip:")
    print(f"  Look at the latent traversal images to see what each dimension encodes.")
    print(f"  With β={args.beta}, dimensions should encode independent factors of variation")
    print(f"  (e.g., one dimension for rotation, another for thickness, etc.)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train β-VAE on MNIST for disentangled representations')
    
    # 모델 인자
    parser.add_argument('--model-type', type=str, default='fc', choices=['fc', 'conv'],
                        help='Type of β-VAE (fc or conv)')
    parser.add_argument('--latent-dim', type=int, default=10,
                        help='Latent dimension (smaller for better disentanglement visualization)')
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='Hidden dimension (for FC β-VAE)')
    parser.add_argument('--beta', type=float, default=4.0,
                        help='Beta parameter (1.0=standard VAE, 4-10=disentangled)')
    
    # 학습 인자
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    
    # 체크포인트
    parser.add_argument('--checkpoint-path', type=str, default='checkpoints/beta_vae_model.pt',
                        help='Path to save checkpoint')
    
    args = parser.parse_args()
    
    # 체크포인트 디렉터리를 만든다
    import os
    os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)
    
    main(args)
```

기본값은 완전 연결 모델, 잠재 10, 은닉 256, β=4, 배치 128, Adam 러닝레이트 0.001, 20에포크다. 입력에는 ToTensor를 적용하며 완전 연결 모델은 이미지를 펼친다. 로스는 모델의 loss_function을 호출하고 에포크 기록은 배치 합을 누적해 데이터셋 크기로 나눈 표본 평균이다. beta_vae_loss도 정의해 두지만 실제 루프에서는 사용하지 않는다.

## 2. 현재 코드의 연결 상태

잠재 축 탐색은 `visualize_latent_traversal`이 맡는다. 축 하나만 $-3$에서 $+3$까지 움직이고 나머지 축은 0에 둔 잠재 벡터를 디코딩한다. 완전 연결 BetaVAE는 자기 `traverse_latent_dimension`을 쓰고, `decode` 메서드만 있는 ConvBetaVAE는 같은 잠재 벡터를 직접 만들어 디코딩한다. 그림은 축마다 하나씩 열 장이 나온다. 축이 정말 한 요인만 맡는지는 그림을 보고 판단할 일이고, 함수가 그것을 보장하지는 않는다.

위 ConvBetaVAE는 ConvVAE를 상속하고 self.beta만 저장한다. 그런데 상속한 loss_function의 기본 인수는 beta=1이고 이 학습 루프는 β를 전달하지 않는다. 따라서 이 컨볼루션 경로에서는 명령행 β가 저장·출력되어도 실제 로스는 기본 β=1로 계산된다. 완전 연결 BetaVAE는 self.beta를 로스에 사용한다. 두 경로가 같은 방식으로 β를 적용한다고 해석하면 안 된다.

## 3. 평가·선택·재현 조건

완전 연결 경로에서 출력하는 Loss는 R+βK이며 β가 다른 실험의 값은 직접적인 우열 기준이 아니다. 공통 지표로 R+K를 기록하고 관측 모델과 합산 축을 맞춘다. eval·no_grad는 잠재 표본 추출을 중단하지 않아 평가 변동이 남는다. 재구성·생성 그림과 잠재 축 탐색은 서로 다른 정보를 준다.

코드는 시험 로스로 최적 에포크를 고른다. 독립적인 최종 성능을 보고하려면 별도 검증셋으로 선택해야 한다. 체크포인트에는 β를 저장하지만 모델 구조·전처리 전체는 포함하지 않는다. state_dict 복원은 일반 속성 self.beta를 자동 복원하지 않으므로 실제 설정도 다시 적용해야 한다.

출력 안내문의 “β>1이면 더 분리된 표현”이나 “각 차원은 독립 요인을 담아야 한다”는 문장은 이론적 보장이 아니다. [β-VAE 모듈](../architecture/beta_vae.md)에서 설명한 것처럼 의미 요인 분리는 별도 가정과 검증을 요구한다. 이 페이지는 β 탐색 실험의 결과를 제시하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
매 배치 step을 유지하면서 zero_grad를 없애면 왜 의도와 달라지는가?

</div>

??? success "풀이"

    이전 배치의 그래디언트가 다음 backward에 더해지고 이전 파라메터 위치의 그래디언트까지 누적 업데이트에 사용된다. Adam의 모멘트에도 이 누적값이 들어가므로 단순한 러닝레이트 배수로 설명할 수 없다. 올바른 배치 학습에서는 각 step의 그래디언트를 초기화한다. 의도한 누적은 여러 배치의 적절히 정규화한 그래디언트로 한 번 step하는 별도 설계다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
본문에서 Adam을 대안 옵티마이저와 비교하려면 무엇을 바꿔야 하는가?

</div>

??? success "풀이"

    기본이 이미 Adam 0.001이므로 「Adam으로 바꾼다」는 것은 새 실험이 아니다. 대안의 종류·러닝레이트·튜닝 예산을 정하고 같은 구조·β·데이터 분할에서 비교한다. 동일 초기 상태와 여러 시드의 결과를 사용하고 반복 횟수뿐 아니라 시간·검증 목적도 확인한다. 구현 연결 오류를 먼저 해결해야 성능 비교에 의미가 있다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
로스가 예상과 다를 때 가장 먼저 확인할 설정은 무엇인가?

</div>

??? success "풀이"

    총로스가 실제 R+βK와 맞는지 확인해 β 전달 누락을 찾는다. 이 스크립트의 컨볼루션 경로가 구체적인 예다. 그다음 합산 기준·입력 전처리·출력 형태·유한하지 않은 값(NaN·Inf)을 확인한다. 낮은 차원별 KL은 붕괴 가능성을 시사하지만 항상 오류는 아니며 잠재 표현 의존성과 검증 목적을 함께 본다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
조기 종료 예시의 웨이트 복사와 종료 처리를 검토하라.

</div>

??? success "풀이"

    아래 state_dict().copy는 얕은 복사라 최적 시점 텐서가 고정되지 않을 수 있다. 독립 복제나 파일 저장이 필요하다. evaluate·val_loader·학습 단계는 생략되어 있으므로 별도로 연결해야 하고, 인내 횟수에 도달하지 않고 정상 종료할 때도 최적 상태를 복원하도록 한다. 선택 지표는 β별 목적과 공통 평가 목적을 구분하고 시험셋 대신 검증셋을 사용한다.

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
β 탐색의 로그에 무엇을 기록해야 하는가?

</div>

??? success "풀이"

    실제 적용한 β, 재구성·β를 곱하지 않은 KL·표준 ELBO, 차원별 KL과 활성 차원 판정 임계, 생성·표현 평가 지표, 학습 곡선과 설정·시드를 기록한다. 아래 alive는 KL>0.01이라는 임의 임계의 개수이며 내재 차원이나 의미 요인의 수가 아니다. rec·kl·kl_dim은 배치에 대해 동일한 정규화 기준으로 계산해야 한다. model 설정과 명령행 출력이 다른 경우도 로그가 감지해야 한다.

    ```python
    log = {'beta': beta, 'rec': rec.item(), 'kl': kl.item(),
           'kl_per_dim': kl_dim.tolist(), 'alive': int((kl_dim > 0.01).sum())}
    ```


---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
여러 β의 비교에서 어떤 조건을 통제할 수 있는가?

</div>

??? success "풀이"

    데이터 분할·전처리·구조·잠재 차원·로스 축·평가 표본 수·평가기를 맞춘다. 초기 웨이트·데이터 순서·생성용 잠재 표본을 공유하면 일부 변동을 줄일 수 있지만 같은 시드만으로 모든 환경의 결과가 같아지는 것은 아니다. 여러 초기화의 변동도 보고한다. 고정 학습 예산 비교와 동일 튜닝 예산 비교는 별도로 설계한다. 본문에는 재현 시드 설정이나 실제 β 탐색 결과가 없다.



---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
재구성·KL 산점도에서 파레토 경계를 해석할 때 주의할 점은 무엇인가?

</div>

??? success "풀이"

    각 설정의 검증 재구성과 KL을 점으로 표시하고 β·목적별 품질을 색이나 표식으로 덧붙일 수 있다. 하지만 유한 학습 결과의 모든 점이 파레토 최적이라고 할 수 없다. 두 비용 모두 다른 점보다 나쁜 점은 지배된 점일 수 있다. 또한 사용 차원 감소가 품질 저하의 유일한 원인이라고 단정하지 않는다. 학습 변동과 지표의 정의·단위를 함께 제시해야 한다.



---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
β를 바꾸면 러닝레이트를 정확히 반비례 조절해야 하는가?

</div>

??? success "풀이"

    아니다. 그래디언트는 ∇R+β∇K이므로 β 변화는 전체 그래디언트의 상수 배율이 아니라 방향·항별 균형까지 바꾼다. SGD에서도 β를 32배 바꿨다고 전체 스텝이 32배 되는 것은 아니다. Adam 역시 이런 변화를 자동으로 상쇄하지 않는다. 같은 러닝레이트가 충분할 수도 있지만 안정성·수렴과 검증 성능으로 확인하며 필요하면 동일 예산으로 조절한다.



---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff hard" title="어려움"></span>
다른 데이터셋에 β를 옮길 때 어떤 정규화 차이를 확인해야 하는가?

</div>

??? success "풀이"

    픽셀 수·관측 모델·전처리·잠재 차원·두 항의 합산 축을 확인한다. 더 많은 픽셀이 재구성 합을 키울 수 있어도 데이터 내용과 단위가 다르므로 MNIST에서 CIFAR로 가면 실효 β가 정확히 4분의 1이라고 말할 수 없다. KL도 잠재 차원에 정확히 비례하지 않는다. R를 D개 픽셀로, K를 d개 잠재 차원으로 나눈 목표 R/D+β_norm K/d는 원래 단위로 R+β_norm(D/d)K와 대응한다. 이 정의에서는 β_norm=1이 일반적으로 표준 ELBO가 아니다.



---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
매우 작은 β에서 예상 가능한 경향과 보장할 수 없는 결과를 구분하라.

</div>

??? success "풀이"

    KL 압력이 약해져 더 많은 정보와 작은 사후 분산을 사용하는 해를 선호할 수 있다. 그러나 유한 학습의 재구성이 언제나 개선되거나 분산이 반드시 0으로 가거나 특정 생성 실패율에 도달한다는 보장은 없다. 확률적 인코더·사전분포·디코더가 남으므로 자동으로 생성 모델이 아니게 되는 것도 아니다. 재구성만으로 β를 고르면 다른 목적을 놓칠 수 있어 생성·가능도(likelihood, 우도라고도 한다)·표현 지표를 목적에 맞게 별도 평가한다.



## 정리하며

β 탐색은 실제 로스에 적용된 웨이트부터 확인해야 한다. 이 스크립트에는 정의되지 않은 시각화 함수 호출과 컨볼루션 경로의 β 전달 문제가 있다. 코드의 연결 상태를 이해한 뒤 공통 ELBO·생성·표현 평가와 검증셋으로 설정을 비교한다.
