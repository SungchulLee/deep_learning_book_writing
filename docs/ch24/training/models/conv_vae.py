"""conv_vae — ch24/architecture/conv_vae.md 의 모델을 모듈로 쓸 수 있게
옮겨 놓은 것이다. 고칠 일이 있으면 그 페이지를 고치는 편이 낫다.
"""

"""
누비기 변분 자기 부호기(ConvVAE)
공간 특징을 더 잘 뽑으려 누비기 층을 쓴다
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# 메인
# ========================================================================


class ConvVAE(nn.Module):
    """
    그림 자료를 위한 누비기 변분 자기 부호기.
    
    인수:
        latent_dim (int): 숨은 공간 차원
        img_channels (int): 들임 그림 채널 수(회색조 1, RGB 3)
        img_size (int): 들임 그림 크기(네모 그림이라 여긴다)
    """
    
    def __init__(self, latent_dim=128, img_channels=1, img_size=28):
        super(ConvVAE, self).__init__()
        
        self.latent_dim = latent_dim
        self.img_channels = img_channels
        self.img_size = img_size
        
        # 부호기
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
        
        # 펼친 크기를 셈한다
        self.flatten_size = 128 * 4 * 4
        
        # 숨은 공간 매개변수
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)
        
        # 풀개 들임
        self.decoder_input = nn.Linear(latent_dim, self.flatten_size)
        
        # 복호기
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
        들임 그림을 숨은 분포 매개변수로 부호화한다.
        
        인수:
            x: 들임 그림 텐서 [묶음 크기, 채널, 높이, 너비]
            
        반환값:
            mu: 숨은 분포의 평균
            logvar: 숨은 분포의 로그 흩어짐
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        뽑기를 위한 다시 매개변수화 재주.
        
        인수:
            mu: 숨은 분포의 평균
            logvar: 숨은 분포의 로그 흩어짐
            
        반환값:
            z: 뽑은 숨은 벡터
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """
        숨은 나타냄을 그림으로 푼다.
        
        인수:
            z: 숨은 벡터
            
        반환값:
            reconstruction: 다시 세운 그림
        """
        h = self.decoder_input(z)
        reconstruction = self.decoder(h)
        return reconstruction
    
    def forward(self, x):
        """
        온전한 앞먹임.
        
        인수:
            x: 들임 그림 텐서
            
        반환값:
            reconstruction: 다시 세운 그림
            mu: 숨은 분포의 평균
            logvar: 숨은 분포의 로그 흩어짐
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar
    
    def loss_function(self, reconstruction, x, mu, logvar, beta=1.0):
        """
        변분 자기 부호기 손실 함수.
        
        인수:
            reconstruction: 다시 세운 내놓기
            x: 본디 들임
            mu: 숨은 분포의 평균
            logvar: 숨은 분포의 로그 흩어짐
            beta: KL 벌어짐 항의 무게
            
        반환값:
            loss: 전체 변분 자기 부호기 손실
            bce: 다시 세우기 손실
            kld: KL 벌어짐
        """
        # 되살림 손실
        BCE = F.binary_cross_entropy(reconstruction, x, reduction='sum')
        
        # KL 발산
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return BCE + beta * KLD, BCE, KLD
    
    def sample(self, num_samples, device='cpu'):
        """
        숨은 공간에서 표본을 만든다.
        
        인수:
            num_samples: 만들 표본의 개수
            device: 표본을 만들 기기
            
        반환값:
            samples: 만든 그림 표본
        """
        z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decode(z)
        return samples


if __name__ == '__main__':
    # 모델을 시험한다
    model = ConvVAE(latent_dim=128, img_channels=1, img_size=28)
    # BCE 손실은 목표가 [0,1]이어야 한다
    x = torch.rand(32, 1, 28, 28)  # 회색조 28x28 그림 32개 묶음
    
    reconstruction, mu, logvar = model(x)
    loss, bce, kld = model.loss_function(reconstruction, x, mu, logvar)
    
    print(f"Input shape: {x.shape}")
    print(f"Reconstruction shape: {reconstruction.shape}")
    print(f"Latent mu shape: {mu.shape}")
    print(f"Total Loss: {loss.item():.4f}")
    print(f"Reconstruction Loss: {bce.item():.4f}")
    print(f"KL Divergence: {kld.item():.4f}")
    
    # 뽑기를 시험한다
    samples = model.sample(num_samples=10)
    print(f"Generated samples shape: {samples.shape}")
