"""conv_cvae — ch24/architecture/conv_cvae.md 의 모델을 모듈로 쓸 수 있게
옮겨 놓은 것이다. 고칠 일이 있으면 그 페이지를 고치는 편이 낫다.
"""

"""
누비기 조건부 변분 자기 부호기(ConvCVAE)
누비기 얼개와 조건부 만들어 내기를 아우른다
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# 메인
# ========================================================================


class ConvConditionalVAE(nn.Module):
    """
    조건부 그림 만들어 내기를 위한 누비기 조건부 변분 자기 부호기.
    
    인수:
        latent_dim (int): 숨은 공간 차원
        num_classes (int): 조건 지을 갈래의 수
        img_channels (int): 들임 그림 채널 수
        img_size (int): 들임 그림 크기(네모 그림이라 여긴다)
    """
    
    def __init__(self, latent_dim=128, num_classes=10, img_channels=1, img_size=28):
        super(ConvConditionalVAE, self).__init__()
        
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.img_channels = img_channels
        self.img_size = img_size
        
        # 공간 조건 짓기를 위한 이름표 묻힘
        self.label_embedding = nn.Embedding(num_classes, img_size * img_size)
        
        # 부호기 - 그림 + 묻은 이름표를 채널 하나로 더 받는다
        self.encoder = nn.Sequential(
            # 들임: img_channels + 1(묻은 이름표용)
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
        
        # 펼친 크기를 셈한다
        self.flatten_size = 128 * 4 * 4
        
        # 숨은 분포 매개변수
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)
        
        # 풀개 들임: 숨은 것 + 하나만 뜨거운 갈래
        self.decoder_input = nn.Linear(latent_dim + num_classes, self.flatten_size)
        
        # 복호기
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
        갈래 이름표에 조건 지어 들임 그림을 부호화한다.
        
        인수:
            x: 들임 그림 텐서 [묶음 크기, 채널, 높이, 너비]
            labels: 갈래 이름표 [묶음 크기]
            
        반환값:
            mu: 숨은 분포의 평균
            logvar: 숨은 분포의 로그 흩어짐
        """
        batch_size = x.size(0)
        
        # 이름표를 묻고 공간 꼴로 바꾼다
        c_embedded = self.label_embedding(labels)
        c_embedded = c_embedded.view(batch_size, 1, self.img_size, self.img_size)
        
        # 그림과 묻은 이름표를 잇는다
        x_combined = torch.cat([x, c_embedded], dim=1)
        
        # 부호화
        h = self.encoder(x_combined)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        다시 매개변수화 재주.
        
        인수:
            mu: 숨은 분포의 평균
            logvar: 숨은 분포의 로그 흩어짐
            
        반환값:
            z: 뽑은 숨은 벡터
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, labels):
        """
        갈래 이름표에 조건 지어 숨은 나타냄을 푼다.
        
        인수:
            z: 숨은 벡터 [묶음 크기, 숨은 차원]
            labels: 갈래 이름표 [묶음 크기]
            
        반환값:
            reconstruction: 다시 세운 그림
        """
        # 이름표를 하나만 뜨겁게 부호화한다
        c_onehot = F.one_hot(labels, num_classes=self.num_classes).float()
        
        # 숨은 부호와 조건을 잇는다
        z_combined = torch.cat([z, c_onehot], dim=1)
        
        # 디코딩
        h = self.decoder_input(z_combined)
        reconstruction = self.decoder(h)
        return reconstruction
    
    def forward(self, x, labels):
        """
        조건을 곁들인 온전한 앞먹임.
        
        인수:
            x: 들임 그림 텐서
            labels: 갈래 이름표
            
        반환값:
            reconstruction: 다시 세운 그림
            mu: 숨은 분포의 평균
            logvar: 숨은 분포의 로그 흩어짐
        """
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z, labels)
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
        BCE = F.binary_cross_entropy(reconstruction, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + beta * KLD, BCE, KLD
    
    def sample(self, class_label, num_samples, device='cpu'):
        """
        특정 갈래에 조건 지어 표본을 만든다.
        
        인수:
            class_label: 만들 갈래(int)
            num_samples: 만들 표본의 개수
            device: 표본을 만들 기기
            
        반환값:
            samples: 만든 그림 표본
        """
        z = torch.randn(num_samples, self.latent_dim).to(device)
        labels = torch.tensor([class_label] * num_samples).to(device)
        samples = self.decode(z, labels)
        return samples
    
    def interpolate_classes(self, z, class1, class2, num_steps=10):
        """
        숨은 부호를 고정한 채 두 갈래 사이를 사이 끼움한다.
        
        인수:
            z: 고정한 숨은 부호 [1, 숨은 차원]
            class1: 시작 갈래
            class2: 끝 갈래
            num_steps: 사이 끼움 걸음 수
            
        반환값:
            interpolations: 사이 끼움한 표본
        """
        device = z.device
        interpolations = []
        
        for i in range(num_steps):
            alpha = i / (num_steps - 1)
            
            # 사이 끼움을 위한 부드러운 이름표를 만든다
            c1_onehot = F.one_hot(torch.tensor([class1]), num_classes=self.num_classes).float().to(device)
            c2_onehot = F.one_hot(torch.tensor([class2]), num_classes=self.num_classes).float().to(device)
            c_interpolated = (1 - alpha) * c1_onehot + alpha * c2_onehot
            
            # 디코딩
            z_combined = torch.cat([z, c_interpolated], dim=1)
            h = self.decoder_input(z_combined)
            sample = self.decoder(h)
            interpolations.append(sample)
        
        return torch.cat(interpolations, dim=0)


if __name__ == '__main__':
    # 모델을 시험한다
    model = ConvConditionalVAE(latent_dim=128, num_classes=10, img_channels=1, img_size=28)
    # BCE 손실은 목표가 [0,1]이어야 한다
    x = torch.rand(32, 1, 28, 28)
    labels = torch.randint(0, 10, (32,))
    
    reconstruction, mu, logvar = model(x, labels)
    loss, bce, kld = model.loss_function(reconstruction, x, mu, logvar)
    
    print(f"Input shape: {x.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Reconstruction shape: {reconstruction.shape}")
    print(f"Total Loss: {loss.item():.4f}")
    print(f"Reconstruction Loss: {bce.item():.4f}")
    print(f"KL Divergence: {kld.item():.4f}")
    
    # 조건부 뽑기를 시험한다
    samples = model.sample(class_label=7, num_samples=10)
    print(f"Generated samples (class 7) shape: {samples.shape}")
