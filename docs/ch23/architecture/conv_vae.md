# 누비기 변분 자기 부호기

누비기 변분 자기 부호기(ConvVAE) 공간 특징을 더 잘 뽑으려 누비기 층을 쓴다

자기 부호기와 변분 자기 부호기는 눌러 담은 나타냄을 배우고 새 자료를 만들어 내는 힘 있는 연장이다. 이 짜기는 고갱이 얼개와 익히기 절차를 보이며 수학 얼거리를 도는 PyTorch 부호에 잇는다.

## 1. 코드

```python
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
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
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
    x = torch.randn(32, 1, 28, 28)  # 회색조 28x28 그림 32개 묶음
    
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
```

## 2. 논의

`ConvVAE` 갈래는 PyTorch의 `nn.Module` 겉면으로 모델 얼개를 감싼다. `forward` 방법이 셈 그래프를 정하며, 그래서 PyTorch의 저절로 미분하기가 익히는 동안 기울기 셈하기를 알아서 다룬다. 이 모듈 설계 덕분에 낱낱의 조각을 고치거나 모델을 더 큰 물길에 넣기가 쉽다.

손실 계산은 모델의 출력을 최적화 목표와 이어 준다. 알맞은 손실 함수를 고르는 일은 결정적으로 중요하다. 손실 함수가 모델이 무엇을 최적화하도록 배울지를 정하며, 학습된 표현과 결정 경계를 직접 빚어내기 때문이다.

여기서 보인 결은 더 복잡한 경우로 자연스레 넓어진다. 웃매개변수, 얼개 변형, 여러 자료 묶음을 시험해 보면 이해가 깊어지고 나타냄 배우기 일에 대한 실전 직관이 선다.

## 연습문제

**연습문제 1.**
`ConvVAE`의 앞먹임을 따라가며 텐서 꼴을 좇아라. 붙박이 매개변수로 들임 표본 4개짜리 묶음에 대해 주요 연산(누비기, 모으기, 선형 층)마다 그 뒤의 꼴을 적어라.

??? success "연습문제 1 풀이"
    입력 모양에서 출발하여 각 층을 차례로 적용한다. `Conv2d(in_c, out_c, k)`마다 공간 차원은 (덧대기가 없으면) $H_{\text{out}} = H_{\text{in}} - k + 1$으로 바뀌거나 (`padding=k//2`이면) 그대로 유지된다. 커널이 2인 풀링은 공간 차원을 절반으로 만든다. 선형 층은 마지막 차원을 바꾼다. 배치 차원은 내내 그대로임에 유의하며 추적한다. 중간 모양을 합성곱 층에서는 $(B, C, H, W)$로, 평탄화 후에는 $(B, F)$로 적는다.

---

**연습문제 2.**
$64 \times 64$ 크기의 RGB 이미지(입력 모양 $3 \times 64 \times 64$)를 받도록 구조를 수정하라. 모든 층의 차원을 그에 맞게 고치고 모델이 오류 없이 실행되는지 확인하라.

??? success "연습문제 2 풀이"
    첫 누비기 층의 `in_channels`을 지금 값에서 3으로 바꿔라. 식 $H_{\text{out}} = \lfloor(H_{\text{in}} + 2p - k) / s\rfloor + 1$으로 누비기와 모으기 층마다 뒤의 공간 차원을 다시 셈하라. 마지막 누비기/모으기 층의 펼친 내놓기에 맞도록 첫 선형 층의 `in_features`을 고쳐라. 다음으로 확인하라: `model = ConvVAE(...); x = torch.randn(1, 3, 64, 64); print(model(x).shape)`.

---

**연습문제 3.**
같은 입출력 차원에서 표준 합성곱과 깊이별 분리 합성곱의 매개변수 개수와 FLOPs를 비교하라. 계산 절감이 가장 큰 것은 언제인가?

??? success "연습문제 3 풀이"
    표준 `Conv2d(C_in, C_out, k)`은 $C_{{\text{{in}}}} \times C_{{\text{{out}}}} \times k^2$개의 매개변수를 갖는다. 깊이별 분리 합성곱은 이를 둘로 나눈다. (1) 깊이별: $C_{{\text{{in}}}} \times k^2$개(입력 채널마다 필터 하나), (2) 점별: $C_{{\text{{in}}}} \times C_{{\text{{out}}}}$개(1x1 합성곱)이다. 매개변수의 비는 대략 $1/C_{{\text{{out}}}} + 1/k^2$이다. $k=3$이고 $C_{{\text{{out}}}}=256$이면 매개변수가 약 $8{-}9\times$ 적어진다. 절감은 $C_{{\text{{out}}}}$과 $k$가 모두 클 때 가장 크다.

---

**연습문제 4.**
층이나 덩이의 수를 자리매김할 수 있도록 `ConvVAE`을 넓혀라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`으로 깊이를 바꿀 수 있는 얼개를 만들어라. 층 2, 4, 8개로 시험하라.

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 방법에서 `for layer in self.layers: x = layer(x)`으로 되풀이한다. (수수한 파이썬 목록이 아니라) `nn.ModuleList`을 써야 PyTorch가 모든 매개변수를 가장 좋게 하기에 올린다. 다음으로 시험하라: `for n in [2, 4, 8]: model = ConvVAE(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.

## 정리하며

**다룬 것** — 누비기 변분 자기 부호기

`ConvVAE` 갈래는 PyTorch의 `nn.Module` 겉면으로 모델 얼개를 감싼다.

고갱이 갈래는 `ConvVAE`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
