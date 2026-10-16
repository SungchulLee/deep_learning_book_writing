# 시간 조건 U-Net과 노이즈 예측

이미지 확산 모델의 네트워크는 노이즈 관측과 시간을 받아 원본 이미지와 같은 모양의 예측을 출력할 수 있다. 그 출력이 스코어인지 노이즈인지는 구조가 아니라 학습 목표와 파라미터화가 정한다. 이 페이지는 시간 조건 U-Net을 정의하고 학습·생성 흐름을 **문자열로 출력**하는 예제다.

## 1. 코드와 기존 출력

```python
"""
U-Net으로 그림 만들어 내기
=====================================

어려움: 나아간 단계
시간: 4~5시간
미리 알 것: 단원 01-08, 겹말기 뉴럴 넷트웍 앎

학습 목표:
- 그림을 위한 U-Net 점수 구조를 짠다
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
# 무작위로 뽑는 값이 아래에 나온다. 씨앗을 고정해야 이 쪽에 실린
# 수가 다시 나온다 — 고정하지 않으면 돌릴 때마다 다른 수가 찍힌다
torch.manual_seed(0)

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

U-Net 구조:
1. 인코더: 줄이기 + 피처 뽑기
2. 디코더: 키우기 + 내놓기 만들기
3. 건너뛰기 이음: 공간의 앎을 지킨다
4. 때 조건 주기: 잡음 수준마다 다른 점수

MNIST를 위한 단출한 U-Net:
--------------------------
""")

class TimeEmbedding(nn.Module):
    """사인 꼴 때 임베딩"""
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
    - 인코더: 28 → 14 → 7
    - 디코더: 7 → 14 → 28
    - 레이어마다 때 조건 주기
    """
    def __init__(self, channels=[1, 32, 64, 128], time_dim=128):
        super().__init__()
        
        # 때 임베딩
        self.time_mlp = nn.Sequential(
            TimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU()
        )
        
        # 인코더(줄이기)
        self.enc1 = nn.Conv2d(channels[0], channels[1], 3, padding=1)
        self.enc2 = nn.Conv2d(channels[1], channels[2], 3, padding=1, stride=2)  # 28→14
        self.enc3 = nn.Conv2d(channels[2], channels[3], 3, padding=1, stride=2)  # 14→7
        
        # 가운데
        self.mid = nn.Conv2d(channels[3], channels[3], 3, padding=1)
        
        # 디코더(키우기)
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
        # 때 임베딩
        t_emb = self.time_mlp(t)
        
        # 인코더
        h1 = F.silu(self.enc1(x) + self.time_proj1(t_emb)[:, :, None, None])
        h2 = F.silu(self.enc2(h1) + self.time_proj2(t_emb)[:, :, None, None])
        h3 = F.silu(self.enc3(h2) + self.time_proj3(t_emb)[:, :, None, None])
        
        # 가운데
        h = F.silu(self.mid(h3))
        
        # 건너뛰는 이음을 갖춘 디코더
        h = F.silu(self.dec3(h))
        h = torch.cat([h, h2], dim=1)  # 건너뛰는 이음
        
        h = F.silu(self.dec2(h))
        h = torch.cat([h, h1], dim=1)  # 건너뛰는 이음
        
        h = self.dec1(h)
        
        return h

print("U-Net architecture defined!")
print("""
핵심 조각:
--------------
1. 때 임베딩: 뉴럴 넷트웍에 잡음 수준을 알려 준다
2. 건너뛰기 이음: 공간의 세부를 지킨다
3. 남은 덩이: 가장 좋게 하기가 쉬워진다
4. 무리 고르게 맞추기: 만들어 내는 모델에서 배치 고르게 맞추기보다 낫다
5. 어텐션(쓸 수 있음): 해상도 높은 그림용

익힘 꾀:
-----------------
1. 그림 x ~ p_data을 뽑는다
2. 잡음 레이어 t ~ Uniform[0, T]을 뽑는다
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
            # 아무 때 스텝
            t = torch.randint(0, n_timesteps, (images.shape[0],))
            
            # 잡음 더하기
            noise = torch.randn_like(images)
            sqrt_alpha_bar = alphas_cumprod[t].sqrt()[:, None, None, None]
            sqrt_one_minus_alpha_bar = (1 - alphas_cumprod[t]).sqrt()[:, None, None, None]
            noisy_images = sqrt_alpha_bar * images + sqrt_one_minus_alpha_bar * noise
            
            # 잡음을 헤아린다(점수를 헤아리는 것과 같다)
            predicted_noise = model(noisy_images, t)
            
            # 로스
            loss = F.mse_loss(predicted_noise, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
    
    return model

# 표집
@torch.no_grad()
def generate_images(model, n_samples=64, n_timesteps=1000):
    '''거꾸로 확산으로 그림을 만든다'''
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
        
        # 잡음을 더한다(마지막 스텝만 빼고)
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
- 그래디언트 되짚을 자리 두기
- Mixed precision (FP16)
- 배치 크기 맞추기

뽑기 빠르기:
- 표준: 스텝 1000개(그림 한 장에 10초쯤)
- DDIM: 스텝 50개(그림 한 장에 0.5초쯤)
- DPM-Solver: 스텝 20개(그림 한 장에 0.2초쯤)
- 한결같음 모델: 스텝 한 개!(앞으로 다룰 이야기)

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

??? note "전체 출력 (150줄)"

    ```
    MODULE 09: Image Generation with U-Net
    ================================================================================

    왜 그림에 U-Net인가?
    --------------------
    점수 함수 s(x)은 들임 x과 크기가 같아야 한다

    그림에서는:
    - Input: [B, C, H, W]
    - 내놓음: [B, C, H, W](화소와 통로마다의 점수)

    U-Net 구조:
    1. 인코더: 줄이기 + 피처 뽑기
    2. 디코더: 키우기 + 내놓기 만들기
    3. 건너뛰기 이음: 공간의 앎을 지킨다
    4. 때 조건 주기: 잡음 수준마다 다른 점수

    MNIST를 위한 단출한 U-Net:
    --------------------------

    U-Net architecture defined!

    핵심 조각:
    --------------
    1. 때 임베딩: 뉴럴 넷트웍에 잡음 수준을 알려 준다
    2. 건너뛰기 이음: 공간의 세부를 지킨다
    3. 남은 덩이: 가장 좋게 하기가 쉬워진다
    4. 무리 고르게 맞추기: 만들어 내는 모델에서 배치 고르게 맞추기보다 낫다
    5. 어텐션(쓸 수 있음): 해상도 높은 그림용

    익힘 꾀:
    -----------------
    1. 그림 x ~ p_data을 뽑는다
    2. 잡음 레이어 t ~ Uniform[0, T]을 뽑는다
    3. Add noise: x_t = √ᾱ_t x + √(1-ᾱ_t) ε
    4. Predict noise: ε_θ(x_t, t)
    5. Loss: ||ε - ε_θ(x_t, t)||²

    이는 모습만 다른 잡음 없애는 점수 맞추기이다!
    Score s(x_t, t) = -ε_θ(x_t, t) / √(1-ᾱ_t)


    Conceptual training code:
    --------------------------------------------------------------------------------

    def train_score_model_mnist(model, dataloader, n_timesteps=1000, epochs=10):
        '''MNIST에서 점수 모델을 익힌다'''
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # 선형 잡음 차례표
        betas = torch.linspace(0.0001, 0.02, n_timesteps)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        for epoch in range(epochs):
            for images, _ in dataloader:
                # 아무 때 스텝
                t = torch.randint(0, n_timesteps, (images.shape[0],))
                
                # 잡음 더하기
                noise = torch.randn_like(images)
                sqrt_alpha_bar = alphas_cumprod[t].sqrt()[:, None, None, None]
                sqrt_one_minus_alpha_bar = (1 - alphas_cumprod[t]).sqrt()[:, None, None, None]
                noisy_images = sqrt_alpha_bar * images + sqrt_one_minus_alpha_bar * noise
                
                # 잡음을 헤아린다(점수를 헤아리는 것과 같다)
                predicted_noise = model(noisy_images, t)
                
                # 로스
                loss = F.mse_loss(predicted_noise, noise)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
        
        return model

    # 표집
    @torch.no_grad()
    def generate_images(model, n_samples=64, n_timesteps=1000):
        '''거꾸로 확산으로 그림을 만든다'''
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
            
            # 잡음을 더한다(마지막 스텝만 빼고)
            if t > 0:
                x = x + beta_t.sqrt() * torch.randn_like(x)
        
        return x


    실제로 살필 점:
    ------------------------

    셈 자원 요구:
    - MNIST: GPU에서 약 2~4시간
    - CIFAR-10: GPU에서 약 1~2일
    - ImageNet: 여러 GPU에서 약 1주

    기억 자리 아끼기:
    - 그래디언트 되짚을 자리 두기
    - Mixed precision (FP16)
    - 배치 크기 맞추기

    뽑기 빠르기:
    - 표준: 스텝 1000개(그림 한 장에 10초쯤)
    - DDIM: 스텝 50개(그림 한 장에 0.5초쯤)
    - DPM-Solver: 스텝 20개(그림 한 장에 0.2초쯤)
    - 한결같음 모델: 스텝 한 개!(앞으로 다룰 이야기)

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


    ✓ Module 09 complete!
    Final module: Complete unification with diffusion models!
    ```

## 2. 공간 크기와 시간 조건

기본 채널 [1,32,64,128]과 28×28 입력에서는 인코더 모양이 B×32×28×28, B×64×14×14, B×128×7×7로 변한다. middle은 7×7을 유지하고 dec3는 B×64×14×14를 만든다. 이를 h2와 채널 방향으로 연결해 B×128×14×14를 만든 뒤 dec2로 B×32×28×28을 얻는다. h1과 연결하면 B×64×28×28이며 최종 출력은 B×1×28×28이다.

시간 임베딩은 B×128이고 세 투영이 각각 채널 32·64·128의 벡터를 만든다. 공간 축에 브로드캐스팅해 각 인코더 컨볼루션 결과에 더한다. 앞의 간단한 이미지 예제와 달리 **시간 조건이 실제 출력 경로에 반영된다**.

두 번의 stride 2 축소와 두 배 확대가 건너뛰기 피처와 맞아야 한다. 모든 이미지 크기를 지원하는 구현은 아니며 28처럼 4의 배수인 크기는 맞는다. TimeEmbedding의 폭도 기본 128처럼 4 이상인 짝수여야 한다. 홀수 폭은 만들어지는 성분 수가 지정값보다 작고 폭 2는 주파수 분모가 0이 된다.

출력 설명에는 잔차 블록·GroupNorm·어텐션을 언급하지만 **이 SimpleUNet에는 그 계층이 없다**. 인코더와 디코더의 건너뛰기 연결은 있지만 잔차 블록과 동일한 구현은 아니다. 다른 구조의 설계 예와 현재 코드를 구분한다.

## 3. 학습 코드는 문자열이다

`training_code`는 학습 함수와 생성 함수를 담은 Python **문자열**이다. print만 하므로 해당 함수들이 정의되거나 호출되지 않는다. 이 스크립트는 MNIST를 다운로드하거나 모델을 학습하고 이미지를 생성하지 않는다. 네트워크의 실제 순방향 호출도 main에서 수행하지 않는다.

문자열 안에서는 1,000개 선형 β를 만들고 αt=1−βt, 누적곱 ᾱt를 사용한다. 원본 x₀와 표준 정규 ε에서 다음과 같이 관측을 만든다.

$$x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\,\epsilon$$

모델 출력은 ε와의 원소 평균 MSE를 최소화하므로 **노이즈 예측**이다. 같은 관측의 주변 스코어에 대응시키려면 다음 배율을 사용한다.

$$s_\theta(x_t,t)=-\frac{\epsilon_\theta(x_t,t)}{\sqrt{1-\bar\alpha_t}}$$

DSM과 연결되지만 단순한 무가중 노이즈 MSE를 모든 시간별 항이 포함된 정확한 변분 우도 목적과 같다고 부르면 안 된다. 웨이트·관측 모형·분산 선택에 따라 목적이 달라진다.

## 4. 출력된 생성 조각의 의존성

문자열의 학습 함수 안에서 betas, alphas, alphas_cumprod는 지역 변수다. 그런데 생성 함수는 같은 이름을 외부에서 참조한다. 따라서 문자열을 별도로 실행해 함수를 만들더라도 이 스케줄을 생성 함수가 사용할 수 있도록 전달·보관하는 인터페이스가 추가로 필요하다. 데이터로더·모델의 장치와 이미지 범위도 정의되어 있지 않다.

생성 평균 식은 통상적인 노이즈 예측 DDPM 형태이며, 추가 잡음 분산으로 βt를 선택하고 마지막 단계에서는 잡음을 생략한다. 이는 전방향 사후분산 β̃t를 사용하는 선택과 같지 않다. 스케줄과 분산의 관례를 보고해야 한다. 모든 텐서를 CPU에 만들므로 임의의 CUDA 모델에 그대로 쓰는 함수도 아니다.

샘플러에 필요한 단계 수만 줄이고 원래 시간별 갱신을 그대로 적용하는 것으로 올바른 DDIM·가속 솔버를 구현할 수는 없다. 선택한 솔버의 전이식과 시간 인덱스를 따로 맞춰야 한다.

## 5. 결과를 해석하는 범위

출력에 나열된 학습 시간, 이미지당 생성 시간과 데이터셋별 FID 범위는 이 코드에서 측정한 결과가 아니다. 하드웨어·모델·배치·정밀도·학습량·평가 구현이 없으면 보편적인 성능 기준으로 사용할 수 없다. U-Net 정의와 출력 문자열만으로 학습 성공이나 생성 품질을 주장하지 않는다.

실제 평가에서는 독립 데이터, 고정 전처리와 피처 모델, 생성 조건·표본 수를 명시하고 FID 등과 표본 검사를 함께 사용한다. 모델 출력이 이미지와 같은 크기라는 것은 필요한 인터페이스 조건일 뿐이다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
28×28 입력에서 dec3·dec2 앞뒤의 채널 수와 건너뛰기 연결을 설명하라.

</div>

??? success "연습문제 1 풀이"

    dec3는 128채널 7×7에서 64채널 14×14를 만든다. h2의 64채널과 연결해 128채널을 만든 뒤 dec2로 32채널 28×28을 얻는다. h1의 32채널과 연결한 64채널을 dec1에 넣어 1채널 출력을 만든다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
노이즈 예측 MSE로 학습한 출력에 시그모이드를 붙여야 하는가?

</div>

??? success "연습문제 2 풀이"

    아니다. 목표는 표준 정규 노이즈이므로 음수와 양수의 일반 실수값을 예측해야 한다. 출력은 표시용 이미지가 아니며 스코어로 사용할 때도 시간별 −1/√(1−ᾱt) 배율이 필요하다. 구조 이름만으로 이미지·스코어·노이즈 출력의 의미가 정해지지 않는다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
문자열의 학습·생성 함수를 그대로 옮기면 스케줄이 공유되는가?

</div>

??? success "연습문제 3 풀이"

    아니다. 학습 함수 안의 지역 스케줄은 생성 함수의 외부 이름 참조와 연결되지 않는다. 학습·체크포인트·생성에서 스케줄과 분산 선택을 전달·보관해야 한다. 원래 스크립트는 문자열을 출력할 뿐 함수를 실행하지 않으므로 실제 학습·시간·FID 결과도 제시하지 않는다.



## 정리하며

시간 조건 U-Net은 공간 크기를 복원하면서 인코더에 시간 피처를 주입한다. 이 페이지의 학습·생성 코드는 실행되지 않는 문자열이며 스케줄 공유도 별도 설계가 필요하다. 노이즈 출력과 스코어의 배율, 로스 웨이트와 평가 조건을 구분한다.
