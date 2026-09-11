"""vae — ch24/architecture/vae.md 의 모델을 모듈로 쓸 수 있게
옮겨 놓은 것이다. 고칠 일이 있으면 그 페이지를 고치는 편이 낫다.
"""

"""
변분 자기 부호기(VAE)
다시 매개변수화 재주로 확률 숨은 공간을 짠다
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# 메인
# ========================================================================


class VAE(nn.Module):
    """
    정규 숨은 공간을 가진 여느 변분 자기 부호기.
    
    인수:
        input_dim (int): 들임 차원(예컨대 MNIST는 784)
        hidden_dim (int): 숨은 층 차원
        latent_dim (int): 숨은 공간 차원
    """
    
    def __init__(self, input_dim=784, hidden_dim=256, latent_dim=32):
        super(VAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # 부호기: 숨은 분포의 매개변수를 내놓는다
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 숨은 분포의 평균과 로그 흩어짐
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # 복호기
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
        들임을 숨은 분포 매개변수로 부호화한다.
        
        인수:
            x: 입력 텐서
            
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
        다시 매개변수화 재주: z = mu + sigma * epsilon
        여기서 epsilon ~ N(0, 1)
        
        그러면 뒤먹임 퍼뜨리기를 위해 뽑기를 미분할 수 있다.
        
        인수:
            mu: 숨은 분포의 평균
            logvar: 숨은 분포의 로그 흩어짐
            
        반환값:
            z: 뽑은 숨은 벡터
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        """
        숨은 나타냄을 내놓기로 푼다.
        
        인수:
            z: 숨은 벡터
            
        반환값:
            reconstruction: 다시 세운 내놓기
        """
        return self.decoder(z)
    
    def forward(self, x):
        """
        온전한 앞먹임: 부호화 -> 다시 매개변수화 -> 풀기
        
        인수:
            x: 입력 텐서
            
        반환값:
            reconstruction: 다시 세운 내놓기
            mu: 숨은 분포의 평균
            logvar: 숨은 분포의 로그 흩어짐
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar
    
    def loss_function(self, reconstruction, x, mu, logvar, beta=1.0):
        """
        변분 자기 부호기 손실 = 다시 세우기 손실 + β * KL 벌어짐
        
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
        # 다시 세우기 손실(두 값 엇갈린 엔트로피)
        BCE = F.binary_cross_entropy(reconstruction, x, reduction='sum')
        
        # KL 벌어짐: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return BCE + beta * KLD, BCE, KLD
    
    def sample(self, num_samples, device='cpu'):
        """
        숨은 공간에서 표본을 만든다.
        
        인수:
            num_samples: 만들 표본의 개수
            device: 표본을 만들 기기
            
        반환값:
            samples: 만든 표본
        """
        z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decode(z)
        return samples


if __name__ == '__main__':
    # 모델을 시험한다
    model = VAE(input_dim=784, latent_dim=32)
    # BCE 손실은 목표가 [0,1]이어야 한다. randn은 음수를 내므로 rand를 쓴다
    x = torch.rand(32, 784)
    
    reconstruction, mu, logvar = model(x)
    loss, bce, kld = model.loss_function(reconstruction, x, mu, logvar)
    
    print(f"Input shape: {x.shape}")
    print(f"Reconstruction shape: {reconstruction.shape}")
    print(f"Latent mu shape: {mu.shape}")
    print(f"Latent logvar shape: {logvar.shape}")
    print(f"Total Loss: {loss.item():.4f}")
    print(f"Reconstruction Loss: {bce.item():.4f}")
    print(f"KL Divergence: {kld.item():.4f}")
    
    # 뽑기를 시험한다
    samples = model.sample(num_samples=10)
    print(f"Generated samples shape: {samples.shape}")
