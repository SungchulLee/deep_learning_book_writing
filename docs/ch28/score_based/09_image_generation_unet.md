# 그림 만들어 내기 U-Net

이 단원은 요즘 만들어 내는 모델의 핵심 부품인 그림 만들어 내기 U-Net을 짠다. 여기서 보이는 개념과 재주를 알면 퍼짐 모델과 점수 바탕 만들어 내는 방법을 다루는 데 꼭 필요한 앎을 얻는다. 이 짜기는 또렷함과 실제 쓸모의 균형을 맞추어 배우기에도 실험하기에도 알맞다.

## 1. 코드

```python
"""
단원 09: U-Net으로 그림 만들어 내기
=====================================

어려움: 나아간 단계
시간: 4~5시간
미리 알 것: 단원 01-08, 겹말기 신경망 앎

학습 목표:
- 그림을 위한 U-Net 점수 얼개를 짠다
- MNIST 숫자로 익힌다
- 점수 바탕 뽑기로 그림을 만든다
- 셈에서 살필 것을 이해한다

핵심: 그림 점수 나타내기를 위한 때 조건 U-Net

지은이: 이성철 @ 연세대학교
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ========================================================================
# 메인
# ========================================================================

print("MODULE 09: Image Generation with U-Net")
print("="*80)

print("""
왜 그림에 U-Net인가?
--------------------
점수 함수 s(x)은 들임 x과 크기가 같아야 한다

그림에서는:
- Input: [B, C, H, W]
- 내놓음: [B, C, H, W](화소와 통로마다의 점수)

U-Net 얼개:
1. 부호기: 줄이기 + 특징 뽑기
2. 풀개: 키우기 + 내놓기 만들기
3. 건너뛰기 이음: 공간의 앎을 지킨다
4. 때 조건 주기: 잡음 수준마다 다른 점수

MNIST를 위한 단출한 U-Net:
--------------------------
""")

class TimeEmbedding(nn.Module):
    """사인 꼴 때 박아 넣기"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings

class SimpleUNet(nn.Module):
    """
    28x28 MNIST 그림을 위한 단순한 U-Net
    
    구조:
    - 부호기: 28 → 14 → 7
    - 풀개: 7 → 14 → 28
    - 층마다 때 조건 주기
    """
    def __init__(self, channels=[1, 32, 64, 128], time_dim=128):
        super().__init__()
        
        # 때 박아 넣기
        self.time_mlp = nn.Sequential(
            TimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU()
        )
        
        # 부호기(줄이기)
        self.enc1 = nn.Conv2d(channels[0], channels[1], 3, padding=1)
        self.enc2 = nn.Conv2d(channels[1], channels[2], 3, padding=1, stride=2)  # 28→14
        self.enc3 = nn.Conv2d(channels[2], channels[3], 3, padding=1, stride=2)  # 14→7
        
        # 가운데
        self.mid = nn.Conv2d(channels[3], channels[3], 3, padding=1)
        
        # 풀개(키우기)
        self.dec3 = nn.ConvTranspose2d(channels[3], channels[2], 4, stride=2, padding=1)  # 7→14
        self.dec2 = nn.ConvTranspose2d(channels[2]*2, channels[1], 4, stride=2, padding=1)  # 14→28
        self.dec1 = nn.Conv2d(channels[1]*2, channels[0], 3, padding=1)
        
        # 때 쏘기
        self.time_proj1 = nn.Linear(time_dim, channels[1])
        self.time_proj2 = nn.Linear(time_dim, channels[2])
        self.time_proj3 = nn.Linear(time_dim, channels[3])
    
    def forward(self, x, t):
        """
        인수:
            x: Image [B, 1, 28, 28]
            t: Time [B]
        반환값:
            score: [B, 1, 28, 28]
        """
        # 때 박아 넣기
        t_emb = self.time_mlp(t)
        
        # 부호기
        h1 = F.silu(self.enc1(x) + self.time_proj1(t_emb)[:, :, None, None])
        h2 = F.silu(self.enc2(h1) + self.time_proj2(t_emb)[:, :, None, None])
        h3 = F.silu(self.enc3(h2) + self.time_proj3(t_emb)[:, :, None, None])
        
        # 가운데
        h = F.silu(self.mid(h3))
        
        # 건너뛰는 이음을 갖춘 풀개
        h = F.silu(self.dec3(h))
        h = torch.cat([h, h2], dim=1)  # 건너뛰는 이음
        
        h = F.silu(self.dec2(h))
        h = torch.cat([h, h1], dim=1)  # 건너뛰는 이음
        
        h = self.dec1(h)
        
        return h

print("U-Net architecture defined!")
print("""
고갱이 조각:
--------------
1. 때 박아 넣기: 신경망에 잡음 수준을 알려 준다
2. 건너뛰기 이음: 공간의 세부를 지킨다
3. 남은 덩이: 가장 좋게 하기가 쉬워진다
4. 무리 고르게 맞추기: 만들어 내는 모델에서 묶음 고르게 맞추기보다 낫다
5. 눈길(쓸 수 있음): 해상도 높은 그림용

익힘 꾀:
-----------------
1. 그림 x ~ p_data을 뽑는다
2. 잡음 층 t ~ Uniform[0, T]을 뽑는다
3. Add noise: x_t = √ᾱ_t x + √(1-ᾱ_t) ε
4. Predict noise: ε_θ(x_t, t)
5. Loss: ||ε - ε_θ(x_t, t)||²

이는 모습만 다른 잡음 없애는 점수 맞추기이다!
Score s(x_t, t) = -ε_θ(x_t, t) / √(1-ᾱ_t)
""")

# 단순하게 만든 익히기 되풀이(개념)
print("\nConceptual training code:")
print("-" * 80)

training_code = """
def train_score_model_mnist(model, dataloader, n_timesteps=1000, epochs=10):
    '''MNIST에서 점수 모델을 익힌다'''
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # 선형 잡음 차례표
    betas = torch.linspace(0.0001, 0.02, n_timesteps)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    for epoch in range(epochs):
        for images, _ in dataloader:
            # 아무 때 걸음
            t = torch.randint(0, n_timesteps, (images.shape[0],))
            
            # 잡음 더하기
            noise = torch.randn_like(images)
            sqrt_alpha_bar = alphas_cumprod[t].sqrt()[:, None, None, None]
            sqrt_one_minus_alpha_bar = (1 - alphas_cumprod[t]).sqrt()[:, None, None, None]
            noisy_images = sqrt_alpha_bar * images + sqrt_one_minus_alpha_bar * noise
            
            # 잡음을 헤아린다(점수를 헤아리는 것과 같다)
            predicted_noise = model(noisy_images, t)
            
            # 손실
            loss = F.mse_loss(predicted_noise, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
    
    return model

# 표집
@torch.no_grad()
def generate_images(model, n_samples=64, n_timesteps=1000):
    '''거꾸로 퍼짐으로 그림을 만든다'''
    # 잡음에서 시작한다
    x = torch.randn(n_samples, 1, 28, 28)
    
    for t in reversed(range(n_timesteps)):
        t_tensor = torch.ones(n_samples, dtype=torch.long) * t
        
        # 잡음을 헤아린다
        predicted_noise = model(x, t_tensor)
        
        # 점수 셈하기
        alpha_t = alphas[t]
        alpha_bar_t = alphas_cumprod[t]
        
        # 잡음을 없앤다
        beta_t = betas[t]
        x = (1 / alpha_t.sqrt()) * (x - beta_t / (1 - alpha_bar_t).sqrt() * predicted_noise)
        
        # 잡음을 더한다(마지막 걸음만 빼고)
        if t > 0:
            x = x + beta_t.sqrt() * torch.randn_like(x)
    
    return x
"""

print(training_code)

print("""
실제로 살필 점:
------------------------

셈 자원 요구:
- MNIST: GPU에서 약 2~4시간
- CIFAR-10: GPU에서 약 1~2일
- ImageNet: 여러 GPU에서 약 1주

기억 자리 아끼기:
- 기울기 되짚을 자리 두기
- Mixed precision (FP16)
- 묶음 크기 맞추기

뽑기 빠르기:
- 표준: 걸음 1000개(그림 한 장에 10초쯤)
- DDIM: 걸음 50개(그림 한 장에 0.5초쯤)
- DPM-Solver: 걸음 20개(그림 한 장에 0.2초쯤)
- 한결같음 모델: 걸음 한 개!(앞으로 다룰 이야기)

품질 자:
- FID(프레셰 인셉션 거리)
- 인셉션 점수
- 정밀도와 재현율
- 사람이 따지기

흔한 결과:
- MNIST: FID 5~10쯤(아주 좋음)
- CIFAR-10: FID 3~10쯤(최고 수준)
- ImageNet 256x256: FID 2~5쯤(최고 수준)

여태 배운 것과의 이음:
--------------------------------
✓ 점수 함수(단원 01) → 잡음 헤아리기
✓ 잡음 없애는 점수 맞추기(단원 02) → 익히기 목표
✓ 랑주뱅(단원 03) → 뽑기 절차
✓ 여러 잣수(단원 07) → 때 조건 주기
✓ 확률 미분 방정식(단원 08) → 이어진 적기

모든 것이 이어진다!
""")

print("\n✓ Module 09 complete!")
print("Final module: Complete unification with diffusion models!")


if __name__ == "__main__":
    pass
```

## 2. 논의

그림 만들어 내기 U-Net의 짜기는 이 마당에 자리 잡은 방식을 따른다. 코드 짜임이 모델 뜻매김과 익히기 논리를 갈라 놓아 부품을 하나씩 고치기 쉽다. 얼개 고르기는 만들어 내는 모델 무리가 많은 실험에서 얻은 배움을 담고 있다.

이 짜기의 핵심에는 수치의 안정을 꼼꼼히 다루기, 고르게 맞추기 재주를 제대로 쓰기, 효율 좋은 셈 결이 든다. 익히기 절차에는 잡음 차례표, 기울기 다루기, 이따금의 따지기가 들며 모두 품질 높은 결과를 내는 데 결정적이다.

이 단원은 이론의 개념이 실제 짜기로 어떻게 옮겨지는지 보이며 만들어 내는 모델의 더 넓은 틀과 이어진다. 여기서 보이는 재주는 만들어 내는 모델이 이룰 수 있는 것의 가장자리를 넓히는 더 앞선 변형과 넓힘을 이해하는 바탕이 된다.

## 연습문제

**연습문제 1.**
구체적인 자료 묶음으로 이 단원의 으뜸 셈을 좇아라. 큰 걸음마다 텐서 꼴을 적고 모든 차원이 서로 맞는지 확인하라.

??? success "연습문제 1 풀이"
    모델에 알맞은 꼴의 들임 묶음에서 시작한다. 층이나 함수 부르기마다 셈을 따라가며 바뀜 뒤 텐서 꼴을 적는다. 겹말기 층에서는 내놓기 차원 공식을 쓴다. 눈길 얼개에서는 물음, 열쇠, 값의 차원이 맞는지 확인한다. 마지막 내놓기 꼴이 바라던 목표 차원과 맞는지 굳힌다. 이 익힘은 자료가 얼개를 어떻게 흐르는지에 대한 직관을 쌓아 준다.

---

**연습문제 2.**
이 단원에 쓰인 손실 함수를 가려내고 모델 매개변수에 대한 기울기를 이끌어 내라. 왜 이 손실 함수가 이 일에 알맞은지 설명하라.

??? success "연습문제 2 풀이"
    손실 함수는 모델이 헤아린 값과 목표 사이의 어긋남을 잰다. 잡음 헤아리기에서는 평균 제곱 어긋남 손실 $\|\epsilon - \epsilon_\theta(x_t, t)\|^2$을 쓰는데, 이것이 로그 가능도의 변분 아래 한계에 맞물리기 때문이다. 매개변수 $\theta$에 대한 기울기는 $-2(\epsilon - \epsilon_\theta) \nabla_\theta \epsilon_\theta$이며 헤아림 어긋남을 줄이는 방향을 가리킨다. 이 손실을 가장 작게 하는 것이 퍼짐 모델에서 자료 로그 가능도의 아래 한계를 가장 크게 하는 것과 같으므로 알맞다.

---

**연습문제 3.**
다른 잡음 차례표를 받쳐 주도록 이 짜기를 고쳐라(예컨대 선형에서 코사인으로, 또는 그 반대로). 두 차례표의 익히기 움직임과 표본 품질을 견주어라.

??? success "연습문제 3 풀이"
    두 차례표를 모두 짜고 각각으로 모델을 익힌다. $\bar{\alpha}_t = \cos^2\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)$으로 뜻매김한 코사인 차례표는 선형 차례표 $\beta_t = \beta_{\min} + t(\beta_{\max} - \beta_{\min})/T$에 견주어 잡음이 더 매끄럽게 늘어난다. 손실 곡선을 좇고 일정한 사이마다 표본을 만든다. 코사인 차례표는 신호 대 잡음비가 더 완만하게 줄어 때 걸음에 걸쳐 배움 신호가 더 고르므로 흔히 더 좋은 결과를 낸다.

## 정리하며

**다룬 것** — 그림 만들어 내기 U-Net

그림 만들어 내기 U-Net의 짜기는 이 마당에 자리 잡은 방식을 따른다.

고갱이 갈래는 `TimeEmbedding`, `SimpleUNet`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
