# 체크포인트 복원과 샘플 생성

VAE의 사전 생성은 잠재 분포에서 잠재 표현을 뽑아 디코더에 전달하는 과정이다. 재구성·잠재 표현 보간·잠재 축 탐색은 서로 다른 경로를 살펴보는 평가 도구다. 트레인 설정과 체크포인트 구조를 정확히 복원한 뒤 각 그림의 의미를 해석해야 한다.

## 1. 생성 스크립트

```python
"""
트레인된 변분 오토인코더 모델에서 표본을 생성하고 시각화하기
"""

import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
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


class ConditionalVAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, latent_dim=32, num_classes=10):
        super().__init__()
        self.input_dim, self.latent_dim, self.num_classes = input_dim, latent_dim, num_classes
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + num_classes, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_classes, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim), nn.Sigmoid()
        )

    def encode(self, x, c):
        return self.fc_mu(self.encoder(torch.cat([x, c], dim=1))), \
               self.fc_logvar(self.encoder(torch.cat([x, c], dim=1)))

    def reparameterize(self, mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def decode(self, z, c):
        return self.decoder(torch.cat([z, c], dim=1))

    def forward(self, x, labels):
        c = F.one_hot(labels, self.num_classes).float() if len(labels.shape) == 1 else labels
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar

    def sample(self, class_label, num_samples, device='cpu'):
        z = torch.randn(num_samples, self.latent_dim).to(device)
        labels = torch.tensor([class_label] * num_samples).to(device)
        c = F.one_hot(labels, self.num_classes).float()
        return self.decode(z, c)


class ConvConditionalVAE(nn.Module):
    """
    조건부 이미지 생성을 위한 컨볼루션 조건부 변분 오토인코더.
    
    인수:
        latent_dim (int): 잠재 공간 차원
        num_classes (int): 조건으로 줄 클래스의 수
        img_channels (int): 입력 이미지 채널 수
        img_size (int): 입력 이미지 크기(정사각형 이미지로 가정한다)
    """
    
    def __init__(self, latent_dim=128, num_classes=10, img_channels=1, img_size=28):
        super(ConvConditionalVAE, self).__init__()
        
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.img_channels = img_channels
        self.img_size = img_size
        
        # 공간 조건화를 위한 라벨 임베딩
        self.label_embedding = nn.Embedding(num_classes, img_size * img_size)
        
        # 인코더 - 이미지 + 임베딩한 라벨을 채널 하나로 더 받는다
        self.encoder = nn.Sequential(
            # 입력: img_channels + 1(임베딩한 라벨용)
            nn.Conv2d(img_channels + 1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Flatten()
        )
        
        # 펼친 크기를 계산한다
        self.flatten_size = 128 * 4 * 4
        
        # 잠재 분포 파라메터
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)
        
        # 디코더 입력: 잠재 표현 + 원핫 클래스
        self.decoder_input = nn.Linear(latent_dim + num_classes, self.flatten_size)
        
        # 디코더
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 4, 4)),
            
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.ConvTranspose2d(32, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def encode(self, x, labels):
        """
        클래스 라벨을 조건으로 입력 이미지를 인코딩한다.
        
        인수:
            x: 입력 이미지 텐서 [배치 크기, 채널, 높이, 너비]
            labels: 클래스 라벨 [배치 크기]
            
        반환값:
            mu: 잠재 분포의 평균
            logvar: 잠재 분포의 로그 분산
        """
        batch_size = x.size(0)
        
        # 라벨을 임베딩하고 공간 형태로 바꾼다
        c_embedded = self.label_embedding(labels)
        c_embedded = c_embedded.view(batch_size, 1, self.img_size, self.img_size)
        
        # 이미지와 임베딩한 라벨을 연결한다
        x_combined = torch.cat([x, c_embedded], dim=1)
        
        # 인코딩
        h = self.encoder(x_combined)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        리파라메리제이션 기법.
        
        인수:
            mu: 잠재 분포의 평균
            logvar: 잠재 분포의 로그 분산
            
        반환값:
            z: 샘플링한 잠재 벡터
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, labels):
        """
        클래스 라벨을 조건으로 잠재 표현을 디코딩한다.
        
        인수:
            z: 잠재 벡터 [배치 크기, 잠재 차원]
            labels: 클래스 라벨 [배치 크기]
            
        반환값:
            reconstruction: 복원한 이미지
        """
        # 라벨을 원핫 인코딩한다
        c_onehot = F.one_hot(labels, num_classes=self.num_classes).float()
        
        # 잠재 표현과 조건을 연결한다
        z_combined = torch.cat([z, c_onehot], dim=1)
        
        # 디코딩
        h = self.decoder_input(z_combined)
        reconstruction = self.decoder(h)
        return reconstruction
    
    def forward(self, x, labels):
        """
        조건을 포함한 전체 순전파.
        
        인수:
            x: 입력 이미지 텐서
            labels: 클래스 라벨
            
        반환값:
            reconstruction: 복원한 이미지
            mu: 잠재 분포의 평균
            logvar: 잠재 분포의 로그 분산
        """
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z, labels)
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
        BCE = F.binary_cross_entropy(reconstruction, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + beta * KLD, BCE, KLD
    
    def sample(self, class_label, num_samples, device='cpu'):
        """
        특정 클래스를 조건으로 표본을 생성한다.
        
        인수:
            class_label: 생성할 클래스(int)
            num_samples: 생성할 표본의 개수
            device: 표본을 생성할 장치
            
        반환값:
            samples: 생성한 이미지 표본
        """
        z = torch.randn(num_samples, self.latent_dim).to(device)
        labels = torch.tensor([class_label] * num_samples).to(device)
        samples = self.decode(z, labels)
        return samples
    
    def interpolate_classes(self, z, class1, class2, num_steps=10):
        """
        잠재 표현을 고정한 채 두 클래스 사이를 보간한다.
        
        인수:
            z: 고정한 잠재 표현 [1, 잠재 차원]
            class1: 시작 클래스
            class2: 끝 클래스
            num_steps: 보간 스텝 수
            
        반환값:
            interpolations: 보간한 표본
        """
        device = z.device
        interpolations = []
        
        for i in range(num_steps):
            alpha = i / (num_steps - 1)
            
            # 보간을 위한 소프트 라벨을 만든다
            c1_onehot = F.one_hot(torch.tensor([class1]), num_classes=self.num_classes).float().to(device)
            c2_onehot = F.one_hot(torch.tensor([class2]), num_classes=self.num_classes).float().to(device)
            c_interpolated = (1 - alpha) * c1_onehot + alpha * c2_onehot
            
            # 디코딩
            z_combined = torch.cat([z, c_interpolated], dim=1)
            h = self.decoder_input(z_combined)
            sample = self.decoder(h)
            interpolations.append(sample)
        
        return torch.cat(interpolations, dim=0)


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
    사전분포 N(0, I)에서 뽑으므로, 트레인이 잘되었다면 그럴듯한 이미지가 나와야 한다.
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


def _encode_mu(model, images):
    """이미지의 잠재 평균 mu를 구한다. 완전 연결 모델이면 펼쳐서 넣는다."""
    try:
        mu, _ = model.encode(images)
    except RuntimeError:
        mu, _ = model.encode(images.view(images.size(0), -1))
    return mu


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


def visualize_interpolation(model, data_loader, device="cpu", num_steps=10,
                            save_path=None):
    """테스트 이미지 둘의 잠재 평균 사이를 직선으로 잇고 그 위의 점들을 디코딩한다.

    잠재 공간이 매끄럽다면 한 숫자에서 다른 숫자로 중간 모양을 거쳐 넘어간다.
    """
    model.eval()
    images, _ = next(iter(data_loader))
    images = images[:2].to(device)
    with torch.no_grad():
        mu = _encode_mu(model, images)
        t = torch.linspace(0, 1, num_steps, device=device).unsqueeze(1)
        z = (1 - t) * mu[0] + t * mu[1]
        decoded = _decode(model, z)

    fig, axes = plt.subplots(1, num_steps, figsize=(num_steps * 1.2, 1.6))
    for i in range(num_steps):
        img, cmap = _to_image(decoded[i])
        axes[i].imshow(img, cmap=cmap)
        axes[i].axis("off")
    fig.suptitle("Linear interpolation between two test images")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_latent_space(model, data_loader, device="cpu", num_batches=20,
                      save_path=None):
    """잠재 차원이 2일 때 테스트 이미지의 mu를 평면에 찍고 라벨로 색을 칠한다."""
    model.eval()
    mus, labels = [], []
    with torch.no_grad():
        for b, (images, y) in enumerate(data_loader):
            if b >= num_batches:
                break
            mus.append(_encode_mu(model, images.to(device)).cpu())
            labels.append(y)
    mus = torch.cat(mus)
    labels = torch.cat(labels)

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(mus[:, 0], mus[:, 1], c=labels, cmap="tab10", s=4)
    fig.colorbar(sc, ax=ax, label="label")
    ax.set_xlabel("z[0]")
    ax.set_ylabel("z[1]")
    ax.set_title("Latent means of test images")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ========================================================================
# 메인
# ========================================================================

def load_model(model_type, checkpoint_path, device, **model_kwargs):
    """체크포인트에서 트레인된 모델을 불러온다"""
    # 모델 생성
    if model_type == 'vae':
        model = VAE(**model_kwargs)
    elif model_type == 'conv_vae':
        model = ConvVAE(**model_kwargs)
    elif model_type == 'cvae':
        model = ConditionalVAE(**model_kwargs)
    elif model_type == 'conv_cvae':
        model = ConvConditionalVAE(**model_kwargs)
    elif model_type == 'beta_vae':
        model = BetaVAE(**model_kwargs)
    elif model_type == 'conv_beta_vae':
        model = ConvBetaVAE(**model_kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # 웨이트를 불러온다
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded {model_type} from {checkpoint_path}")
    print(f"Training epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Training loss: {checkpoint.get('loss', 'unknown'):.4f}")
    
    return model


def main(args):
    # 장치 지정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 시험 데이터셋 불러오기
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=4)
    
    # 모델 인자
    model_kwargs = {
        'latent_dim': args.latent_dim,
    }
    
    if args.model_type in ['vae', 'beta_vae']:
        model_kwargs['input_dim'] = 784
        model_kwargs['hidden_dim'] = 256
    elif args.model_type in ['cvae']:
        model_kwargs['input_dim'] = 784
        model_kwargs['hidden_dim'] = 256
        model_kwargs['num_classes'] = 10
    elif args.model_type in ['conv_vae', 'conv_beta_vae']:
        model_kwargs['img_channels'] = 1
        model_kwargs['img_size'] = 28
    elif args.model_type in ['conv_cvae']:
        model_kwargs['img_channels'] = 1
        model_kwargs['img_size'] = 28
        model_kwargs['num_classes'] = 10
    
    if args.model_type in ['beta_vae', 'conv_beta_vae']:
        model_kwargs['beta'] = 4.0
    
    # 모델을 불러온다
    model = load_model(args.model_type, args.checkpoint_path, device, **model_kwargs)
    
    # 조건부인지 정한다
    is_conditional = 'cvae' in args.model_type
    
    # 시각화를 만든다
    print("\n=== Generating Visualizations ===")
    
    if args.reconstruction:
        print("\n1. Reconstruction visualization...")
        visualize_reconstruction(
            model, test_loader,
            num_images=args.num_samples,
            device=device,
            conditional=is_conditional
        )
    
    if args.samples:
        print("\n2. Random sample generation...")
        if is_conditional:
            # 클래스마다 표본을 생성한다
            for class_label in range(10):
                print(f"   Generating samples for class {class_label}...")
                visualize_samples(
                    model,
                    args.latent_dim,
                    num_samples=args.num_samples,
                    device=device,
                    class_label=class_label
                )
        else:
            visualize_samples(
                model,
                args.latent_dim,
                num_samples=args.num_samples,
                device=device
            )
    
    if args.interpolation and not is_conditional:
        print("\n3. Latent space interpolation...")
        visualize_interpolation(
            model,
            test_loader,
            device=device,
            num_steps=args.num_steps
        )
    
    if args.traversal and hasattr(model, 'traverse_latent_dimension'):
        print("\n4. Latent dimension traversals...")
        num_dims = min(args.num_traversals, args.latent_dim)
        for dim_idx in range(num_dims):
            print(f"   Traversing dimension {dim_idx}...")
            visualize_latent_traversal(
                model,
                dim_idx=dim_idx,
                num_steps=args.num_steps,
                range_limit=3.0,
                device=device
            )
    
    if args.latent_space and args.latent_dim == 2:
        print("\n5. Latent space visualization (2D only)...")
        plot_latent_space(
            model,
            test_loader,
            device=device,
            num_batches=20
        )
    
    print("\n=== All visualizations complete! ===")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate samples from trained VAE')
    
    # 모델 인자
    parser.add_argument('--model-type', type=str, required=True,
                        choices=['vae', 'conv_vae', 'cvae', 'conv_cvae', 'beta_vae', 'conv_beta_vae'],
                        help='Type of VAE model')
    parser.add_argument('--checkpoint-path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--latent-dim', type=int, default=32,
                        help='Latent dimension (must match training)')
    
    # 생성 인자
    parser.add_argument('--num-samples', type=int, default=10,
                        help='Number of samples to generate')
    parser.add_argument('--num-steps', type=int, default=10,
                        help='Number of steps for interpolation/traversal')
    parser.add_argument('--num-traversals', type=int, default=5,
                        help='Number of latent dimensions to traverse')
    
    # 무엇을 만들지
    parser.add_argument('--reconstruction', action='store_true',
                        help='Visualize reconstructions')
    parser.add_argument('--samples', action='store_true',
                        help='Generate random samples')
    parser.add_argument('--interpolation', action='store_true',
                        help='Visualize latent space interpolation')
    parser.add_argument('--traversal', action='store_true',
                        help='Visualize latent dimension traversals')
    parser.add_argument('--latent-space', action='store_true',
                        help='Visualize 2D latent space (only for 2D latent dim)')
    parser.add_argument('--all', action='store_true',
                        help='Generate all visualizations')
    
    args = parser.parse_args()
    
    # --all을 주면 모든 시각화를 켠다
    if args.all:
        args.reconstruction = True
        args.samples = True
        args.interpolation = True
        args.traversal = True
        args.latent_space = True
    
    # 시각화를 정하지 않으면 재구성과 표본 생성을 기본값으로 한다
    if not any([args.reconstruction, args.samples, args.interpolation, 
                args.traversal, args.latent_space]):
        args.reconstruction = True
        args.samples = True
    
    main(args)
```

모델 종류와 체크포인트 경로를 지정하고 생성·재구성 등의 플래그를 선택하는 구조다. 아무 개별 플래그도 없으면 재구성과 사전 생성을 선택한다. all을 지정해도 조건부 보간은 건너뛰고, 축 탐색은 해당 메서드가 있을 때만, 잠재 공간 표시는 차원이 2일 때만 시도한다. 따라서 모든 모델에 모든 시각화를 제공하는 것은 아니다.

## 2. 제공된 구현의 연결 상태

시각화 도우미는 다섯이다. 재구성·생성 외에 `visualize_interpolation`은 테스트 이미지 둘의 잠재 평균 $\mu$ 사이를 직선으로 이어 디코딩한다. `visualize_latent_traversal`은 축 하나만 움직인다. `plot_latent_space`는 잠재 차원이 2일 때 $\mu$를 평면에 찍고 라벨로 색을 칠한다. 완전 연결 BetaVAE에는 `decode` 메서드가 없고 `decoder` 모듈만 있어서, 디코딩은 둘을 다 받는 `_decode`를 거친다. 이 쪽은 트레인된 체크포인트를 읽어야 돌아가므로 출력 블록을 싣지 않았다.

conv_beta_vae 분기는 img_size=28을 생성자에 전달하지만 위 ConvBetaVAE 생성자는 이 인수를 받지 않는다. 완전 연결 cvae의 decode는 원핫 조건을 받는데 현재 visualize_samples는 정수 라벨을 전달하므로 이 경로도 그대로 호환되지 않는다. 보조 함수나 모델 이름이 비슷해도 인수 계약까지 같은 것은 아니다.

## 3. 체크포인트와 평가 설정

load_model은 먼저 model_kwargs로 구조를 만들고 model_state_dict를 불러온다. 잠재 차원뿐 아니라 은닉 폭·채널·클래스 매핑도 트레인 때와 맞아야 한다. 이 스크립트는 은닉 256·클래스 10 등을 고정하고 β 모델에는 4를 지정한다. 체크포인트의 실제 설정을 자동으로 읽어 복원하지 않는다. β는 일반 속성이라 웨이트 로드만으로 덮어써지지 않는다.

map_location은 체크포인트 텐서를 읽을 장치를 정하고 이후 모델도 그 장치로 옮긴다. eval은 평가 모드이며 그래디언트 기록을 중단하는 기능은 아니다. 현재 시각화 보조 함수의 재구성·생성 경로는 no_grad를 별도로 사용한다. 저장된 loss가 없으면 기본 문자열 unknown을 소수 형식으로 출력하려 해 오류가 나므로 체크포인트 메타데이터의 계약도 확인해야 한다.

순수한 사전 생성에는 MNIST가 필요하지 않지만 이 main은 모든 경우에 시험셋을 다운로드·생성한다. 현재 보조 함수 호출은 저장 경로나 명시적인 표시를 지정하지 않으므로 출력 메시지만으로 그림 파일 저장을 확인할 수는 없다.

## 4. 생성 결과를 읽는 기준

디코더의 Sigmoid 출력은 베르누이 확률 이미지다. 관측 모델 자체의 표본을 원하면 픽셀도 해당 분포에서 뽑아야 한다. 회색조 BCE 관례의 해석은 [VAE 모듈](../architecture/vae.md)을 따른다. 좋은 재구성·높은 분류기 확신도·부드러운 보간이 각각 전체 생성 분포의 정확성을 보장하지는 않는다.

잠재 온도나 truncation을 사용하면 잠재 샘플링 분포가 바뀐다. 해당 정책과 원래 사전분포에서의 결과를 구분하고 품질·다양성·조건 일치율을 함께 보고해야 한다. 이 페이지는 새 생성 실험 수치를 제시하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
이 스크립트의 실제 설계 결정 세 가지를 설명하라.

</div>

??? success "풀이"

    모델 종류에 따라 구조를 먼저 생성하고 state_dict를 로드하는 점, 재구성·사전 생성·보간·축 탐색을 플래그로 선택하는 점, 조건부 생성에서 클래스별로 평가하는 점이다. 첫 결정은 구조 설정의 정확한 복원을 요구하고 둘째는 평가 목적을 구분하며 셋째는 클래스별 품질 차이를 드러낼 수 있다. 잔차 연결 등 이 스크립트에 없는 일반 구조 설명으로 대신하지 않는다. 실제 제공된 보조 함수·생성자 계약과 맞는지도 확인해야 한다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
생성 전에 확인할 입력·설정 계약을 설명하라.

</div>

??? success "풀이"

    체크포인트 키와 구조·잠재 차원·전처리·클래스 매핑, 장치·자료형을 확인한다. 표본 수·보간 단계 수와 조건 범위도 유효해야 한다. 재구성 입력은 모델에 맞춰 펼치거나 이미지 형태로 유지한다. 생성 스크립트에는 모든 모델에 공통인 forward 입력 하나가 있는 것이 아니므로 임의의 expected_dims 검사로 충분하지 않다. float32만 강제하기보다 모델과 평가 정책의 호환성을 확인한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
코드의 연결 실패와 생성 품질 실패를 구분하라.

</div>

??? success "풀이"

    정의되지 않은 보조 함수 호출, 지원하지 않는 생성자 인수, 다른 잠재 차원의 state_dict, decode 조건 모양 불일치는 실행 계약 문제다. 반면 계약을 지켜도 흐릿함·모드 누락·조건 제어 실패가 나타날 수 있으며 트레인·모델·평가 요인을 따로 진단해야 한다. 스크립트가 찍는 진행 문구는 각 경로의 성공이나 품질을 증명하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
풀이에 실린 테스트 골격의 문제점과 적절한 검증 대상을 설명하라.

</div>

??? success "풀이"

    아래의 test_generate samples와 Generate Samples는 공백 때문에 유효한 Python 이름이 아니며 실제 생성 스크립트에도 그런 클래스가 없다. 생략 기호·미정 변수도 남아 있어 실행 가능한 테스트가 아니다. 적절한 검증은 모델 복원 계약, 출력 모양·범위·유한값, 조건별 인수와 평가 모드 등 실제 경로를 대상으로 한다. 사전 생성은 추론 작업이므로 모든 파라메터에 그래디언트가 생겨야 한다는 검사는 필수 조건이 아니다. 빈 배치·극단 잠재 입력에 대한 정책도 명확히 정의해야 한다.

    ```python
    def test_generate samples():
        model = Generate Samples(...)
        # 보통의 입력
        assert model(normal_input).shape == expected_shape
        # 원소가 하나인 배치
        assert model(single_input).shape == (1, ...)
        # 큰 값 (오버플로를 확인한다)
        out = model(torch.ones(...) * 1000)
        assert torch.isfinite(out).all()
        # 그래디언트의 흐름
        out = model(normal_input)
        out.sum().backward()
        for p in model.parameters():
            assert p.grad is not None
    ```


---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff easy" title="쉬움"></span>
사전 생성에서 인코더의 역할은 무엇인가?

</div>

??? success "풀이"

    일반적인 사전 생성에는 인코더가 필요 없고 p(z)와 디코더의 p(x|z)가 필요하다. 아래는 무조건부 확률 이미지 생성의 골격이다. 조건부 모델에는 맞는 표현의 조건을 더 전달해야 한다. 인코더는 트레인 후에도 재구성·사후 추론·피처 추출과 데이터 기반 보간에 사용된다. 메서드를 분리하면 이 경로를 명확히 호출할 수 있지만 메서드 이름을 분리하지 않았다고 생성 자체가 불가능한 것은 아니다. 결정적 AE도 잠재 분포를 별도로 정해 생성할 수 있으나 그 분포의 적합성은 따로 검증해야 한다.

    ```python
    z = torch.randn(n, latent_dim)
    imgs = model.decode(z)
    ```


---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
잠재 표준정규 표본에 T를 곱하는 것의 의미를 설명하라.

</div>

??? success "풀이"

    T>0이면 생성 잠재 분포가 N(0,T²I)로 바뀐다. E||z||²=T²d이며 평균 노름은 일반적으로 T√d와 정확히 같지는 않다. 이 변경이 출력 대비나 분류기 확신도를 어떻게 바꾸는지는 디코더·트레인 상태에 달려 있다. 확신도가 높아져도 품질·다양성·가능도(likelihood, 우도라고도 한다)가 함께 개선되는 것은 아니다. 원래 사전분포와 같은 지지집합을 가져도 확률 질량 배치는 다를 수 있다. 기본 사전분포에서의 결과와 온도 변경 정책의 결과를 따로 보고한다.



---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
잠재 truncation을 평가할 때 먼저 정의할 것은 무엇인가?

</div>

??? success "풀이"

    좌표별 재추출로 절단정규를 쓰는지, 범위를 벗어난 좌표를 경계로 clipping하는지, 노름으로 거르는지 명시해야 한다. 이들은 서로 다른 분포다. truncation은 다양성·품질을 바꿀 수 있지만 GAN이나 VAE에서 보편적으로 유리하거나 불리하지 않다. 집계 사후분포의 좌표별 표준편차가 1에 가깝다고 표준정규와 일치하는 것도 아니다. 평균·고차 구조·좌표 의존성과 생성 결과를 함께 진단한다.



---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
선형 잠재 보간이 의미적으로 좋은 경로를 보장하는가?

</div>

??? success "풀이"

    연속 디코더에 연속 경로를 넣으면 출력도 연속이지만 이것이 의미적 자연스러움이나 높은 데이터 밀도를 보장하지는 않는다. 데이터에서 얻은 두 잠재 표현 사이도 집계 사후분포에서 드문 영역을 지날 수 있다. 독립 표준정규 끝점의 선형 보간은 중간점에서 분산이 줄어드는 성질도 있다. 분류기 확신도 변화는 보간의 한 측면이며 VAE와 AE의 전반적 우열을 결정하지 못한다. 사용한 끝점·평균 또는 표본·경로 정의를 함께 기록한다.



---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff hard" title="어려움"></span>
생성 클래스 엔트로피와 FID가 각각 놓칠 수 있는 것은 무엇인가?

</div>

??? success "풀이"

    생성 이미지를 분류한 클래스 비율의 엔트로피는 클래스 간 분포만 요약한다. 10클래스 균등일 때 최댓값은 ln 10≈2.303 nat이지만 참 데이터가 불균형이면 균등이 정답은 아니다. 클래스마다 같은 이미지 하나만 생성해도 이 값은 높을 수 있다. 클래스 안의 다양성을 재는 쌍별 거리 역시 노이즈·배경·표현 선택에 민감하다. FID는 선택한 피처의 평균·공분산을 비교하는 척도이며 분포 전체의 모든 차이를 식별하지 못한다. 평가 피처가 MNIST에 적합한지, 표본 수와 편향이 어떤지도 확인하고 여러 지표를 함께 사용한다.



---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
생성 격자를 보고할 때 어떤 조건을 밝혀야 하는가?

</div>

??? success "풀이"

    표본 선택·제외 기준, 시드·잠재 샘플링 정책·조건 분포, 모델 설정·체크포인트와 확률 이미지인지 관측 표본인지 밝힌다. 아래 코드는 CPU 잠재 64개를 고정 시드로 만드는 골격이며 모델 장치와 맞춰야 한다. 처음 표본을 그대로 표시하면 선택 편향을 줄일 수 있지만 단일 격자가 전체 분포를 대표하지는 않는다. 동일 표시 범위·전처리의 실제 데이터와 비교하고 여러 배치·정량 진단을 함께 살핀다.

    ```python
    torch.manual_seed(0)                 # 고정하고
    z = torch.randn(64, latent_dim)      # 처음 64개를 그대로
    ```


## 정리하며

사전 생성은 잠재 분포와 디코더를 사용하며 복원·보간·축 탐색과 구분한다. 체크포인트 구조와 조건 인수 계약을 먼저 확인하고 현재 스크립트의 빠진 보조 함수·인수 불일치를 이해해야 한다. 온도·truncation·표본 선택을 밝히고 품질과 다양성을 따로 평가한다.
