# 노이즈 조건 스코어 네트워크 구조

노이즈가 다른 관측의 스코어는 서로 다르다. 하나의 네트워크로 여러 수준을 학습하려면 입력 데이터와 함께 해당 노이즈 수준을 식별하는 조건을 전달해야 한다. 이 페이지는 사인·코사인 임베딩을 사용하는 MLP로 다섯 노이즈 수준을 학습하는 구조를 살펴본다.

## 1. 코드와 기존 출력

```python
"""
점수 신경망 구조
=====================================

어려움: 중간
시간: 2~3시간
미리 알 것: 단원 01-05

학습 목표:
- 자료 갈래마다 잘 듣는 점수 신경망을 짠다
- 잡음과 때 조건 주기를 이해한다
- 요즘 구조를 짠다

지은이: 이성철 @ 연세대학교
"""

import torch
import torch.nn as nn
import numpy as np

# ========================================================================
# 메인
# ========================================================================

print("MODULE 06: Score Network Architectures")
print("="*80)

class TimeConditionalScoreNetwork(nn.Module):
    """
    때와 잡음 조건을 갖춘 점수 신경망
    
    확산 모델에서 점수는 잡음 층 t에 달렸다.
    s_θ(x, t) = ∇_x log p_t(x)
    """
    def __init__(self, data_dim=2, hidden_dim=128, time_embed_dim=32):
        super().__init__()
        
        # 때 임베딩(변환기처럼 사인 꼴)
        self.time_embed_dim = time_embed_dim
        
        # 때 임베딩을 위한 여러 층 신경망
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 으뜸 신경망
        self.net = nn.Sequential(
            nn.Linear(data_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, data_dim)
        )
    
    def get_timestep_embedding(self, timesteps, max_period=10000):
        """사인 꼴 때 묻힘(변환기에서 가져옴)"""
        half_dim = self.time_embed_dim // 2
        freqs = torch.exp(
            -np.log(max_period) * torch.arange(half_dim, dtype=torch.float32) / half_dim
        ).to(timesteps.device)
        args = timesteps[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return embedding
    
    def forward(self, x, t):
        """
        인수:
            x: 자료 [batch_size, data_dim]
            t: 때 또는 잡음 층 [batch_size]
        
        반환값:
            score: ∇_x log p_t(x) [batch_size, data_dim]
        """
        # 때를 박아 넣는다
        t_embed = self.get_timestep_embedding(t)
        t_embed = self.time_mlp(t_embed)
        
        # 자료와 이어 붙인다
        x_with_time = torch.cat([x, t_embed], dim=-1)
        
        # 점수 미리보기
        return self.net(x_with_time)

# 보기: 여러 잡음 수준으로 장난감 자료에서 익히기
from sklearn.datasets import make_swiss_roll
X, _ = make_swiss_roll(n_samples=2000, noise=0.1)
X = X[:, [0, 2]] / 10.0  # 정규화

model = TimeConditionalScoreNetwork(data_dim=2)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# 여러 잡음 수준
sigmas = [0.01, 0.05, 0.1, 0.5, 1.0]

print("Training time-conditional score network...")
X_tensor = torch.FloatTensor(X)

for epoch in range(1000):
    # 배치마다 아무 잡음 수준
    sigma_idx = np.random.randint(len(sigmas))
    sigma = sigmas[sigma_idx]
    t = torch.ones(len(X)) * sigma_idx  # 때 어깨수
    
    # 이 잡음 수준에서의 잡음 없애는 점수 맞추기 손실
    noise = torch.randn_like(X_tensor)
    X_noisy = X_tensor + sigma * noise
    
    pred_score = model(X_noisy, t)
    target_score = -noise / sigma
    
    loss = torch.mean((pred_score - target_score) ** 2)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 200 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.6f}, σ = {sigma:.3f}")

print("""
핵심 구조 고르기:

1. 때 조건 주기:
   - 확산 모델에 꼭 필요하다
   - 사인 꼴 묻힘(변환기 방식)
   - 또는 배울 수 있는 임베딩

2. 2차원과 표 자료에서:
   - 층 고르게 맞추기를 갖춘 여러 층 신경망
   - SiLU와 GELU 깨움
   - 남은 이음이 도움이 된다

3. 그림에서는(9단원에서 다룬다):
   - U-Net 구조
   - 어텐션 구조
   - 무리 고르게 맞추기

4. 이음 자료에서:
   - 변환기 덩이
   - 필요하면 인과 가림막

설계 원칙:
✓ 점수는 벡터장이다(들임과 차원이 같다)
✓ 여러 잡음과 때 수준을 다루어야 한다
✓ 매끄럽고 얌전해야 한다
✓ 담이는 자료의 복잡함에 따라 커진다
""")

print("\n✓ Module 06 complete!")


if __name__ == "__main__":
    pass
```

```
MODULE 06: Score Network Architectures
================================================================================
Training time-conditional score network...
Epoch 0: Loss = 410.753662, σ = 0.050
Epoch 200: Loss = 0.426716, σ = 1.000
Epoch 400: Loss = 390.588104, σ = 0.050
Epoch 600: Loss = 0.411998, σ = 1.000
Epoch 800: Loss = 2.645384, σ = 0.500

핵심 구조 고르기:

1. 때 조건 주기:
   - 확산 모델에 꼭 필요하다
   - 사인 꼴 묻힘(변환기 방식)
   - 또는 배울 수 있는 임베딩

2. 2차원과 표 자료에서:
   - 층 고르게 맞추기를 갖춘 여러 층 신경망
   - SiLU와 GELU 깨움
   - 남은 이음이 도움이 된다

3. 그림에서는(9단원에서 다룬다):
   - U-Net 구조
   - 어텐션 구조
   - 무리 고르게 맞추기

4. 이음 자료에서:
   - 변환기 덩이
   - 필요하면 인과 가림막

설계 원칙:
✓ 점수는 벡터장이다(들임과 차원이 같다)
✓ 여러 잡음과 때 수준을 다루어야 한다
✓ 매끄럽고 얌전해야 한다
✓ 담이는 자료의 복잡함에 따라 커진다


✓ Module 06 complete!
```

## 2. 조건 입력은 노이즈 인덱스다

노이즈 표준편차는 [0.01,0.05,0.1,0.5,1.0]이며 코드의 t는 **목록 인덱스 0~4**다. 실제 확산 시간이나 σ 자체를 입력하는 것은 아니다. 추론에서도 학습과 같은 인덱스·σ 대응을 사용해야 한다. 새로운 목록으로 바꾸고 같은 t를 주면 조건의 의미가 달라진다.

에포크마다 인덱스 하나를 균등하게 골라 2,000개 전체 표본에 같은 노이즈 수준을 적용한다. 표본마다 다른 수준을 선택하는 구현과 다르다. 스코어 네트워크는 t의 수치를 연속 입력으로 처리할 수 있어도, 학습하지 않은 중간 인덱스가 올바른 중간 σ에 대응한다고 보장하지 않는다.

## 3. 임베딩과 데이터 경로

기본 time_embed_dim=32에서는 주파수 16개를 만들고 cos·sin을 연결해 B×32 임베딩을 얻는다. 주파수는 $\omega_k=\exp[-\log(10000)k/16]$이고 각도는 tωk다. 이후 32→128→128 MLP를 통과해 B×128 조건 벡터가 된다.

데이터 x가 B×2이면 조건과 연결해 B×130을 만들고 130→128→128→2 경로로 스코어를 출력한다. 은닉층에는 LayerNorm과 SiLU를 쓰며, 코드에 잔차 연결이나 어텐션은 없다. 기본 전체 파라미터는 54,786개다.

x와 t는 같은 배치 크기·장치에 있어야 하고 t는 길이 B의 벡터여야 한다. `time_embed_dim`이 홀수면 함수가 2×floor(dim/2)개만 반환해 첫 Linear의 입력 폭과 맞지 않는다. 따라서 이 구현은 양의 짝수 임베딩 폭을 전제로 한다. 주파수와 t를 float32로 만들기 때문에 모델 전체를 다른 dtype으로 바꿀 때도 주의한다.

## 4. 다중 노이즈 DSM 목적

표준 정규 노이즈 ξ를 뽑아 $\widetilde x=x_0+\sigma_i\xi$로 만들고 조건 i와 함께 입력한다. 회귀 목표는 −ξ/σi다.

$$L_i(\theta)=\mathbb E\left\|s_\theta(x_0+\sigma_i\xi,i)+\frac\xi{\sigma_i}\right\|^2$$

실제 코드는 표본·좌표 전체 평균이라 2차원에서 위 제곱 노름 평균의 1/2배다. 각 i의 모집단 최적 출력은 해당 σi로 평활화한 주변 밀도의 스코어다. 공유 네트워크의 표현력과 유한 학습 오차 때문에 모든 수준을 동시에 정확히 맞춘다고 보장하지 않는다.

이 구현은 노이즈 수준별 가중치를 추가하지 않는다. 작은 σ에서 목표 크기가 커지므로 각 수준의 손실 크기가 다를 수 있다. 출력 로그는 에포크마다 선택된 σ가 다를 수 있어 손실의 오르내림을 학습 개선·악화만으로 해석하면 안 된다. 여러 수준을 균형 있게 학습하는 가중치 설계는 별도의 선택이며 여기에는 구현되어 있지 않다.

## 5. 실제 학습과 검증 범위

데이터는 3차원 Swiss roll의 두 좌표를 선택해 10으로 나눈 2,000×2 배열이다. AdamW 학습률은 0.001이고 weight_decay를 생략해 라이브러리 기본값을 사용한다. 매 에포크 한 번씩 총 1,000번 전체 배치를 갱신한다. 시드는 고정하지 않고 검증·체크포인트·샘플링도 없다.

출력에 나열된 U-Net, GroupNorm, 어텐션, Transformer는 다른 데이터 구조에서 고려할 수 있는 설계 예다. 이 코드가 구현하거나 비교한 결과는 아니다. 출력 차원이 입력과 같다는 조건은 벡터 스코어에 필요하지만, 그 자체로 올바른 확률 분포의 스코어나 매끄럽고 안정적인 샘플링을 보장하지는 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
B×2 데이터와 길이 B의 조건이 기본 모델을 통과할 때 텐서 모양을 설명하라.

</div>

??? success "연습문제 1 풀이"

    조건은 B×32 사인·코사인 임베딩에서 B×128로 바뀐다. 데이터와 연결해 B×130을 만들고 은닉 B×128 두 개를 거쳐 B×2를 출력한다. 데이터와 조건의 배치 크기·장치가 같아야 한다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
time_embed_dim=33으로 설정하면 어떤 문제가 생기는가?

</div>

??? success "연습문제 2 풀이"

    half_dim은 16이고 cos·sin을 합쳐 32개만 반환한다. 첫 Linear는 33개 입력을 기대하므로 차원 불일치가 생긴다. 이 구현은 양의 짝수 폭을 사용해야 하며 홀수 폭을 지원하려면 임베딩 정의도 바꾸어야 한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
에포크별 로그에서 σ=0.01과 σ=1의 손실을 그대로 비교하면 안 되는 이유는 무엇인가?

</div>

??? success "연습문제 3 풀이"

    목표 −ξ/σ의 크기와 조건부 분산, 학습 난이도가 서로 다르다. 코드에는 수준별 가중치가 없고 매 에포크 선택 수준도 바뀐다. 고정된 검증 조건에서 수준별 오차를 나누어 보아야 한다. 여기의 σ 목록은 확산의 선형·코사인 β 스케줄과 같은 대상이 아니다.



## 정리하며

노이즈 조건 모델은 데이터와 조건 인덱스를 함께 받아 같은 차원의 스코어를 출력한다. 인덱스·σ 대응, 짝수 임베딩 폭과 dtype을 확인하고 각 노이즈 수준의 손실을 구분해 읽는다. 구조 시연만으로 샘플 품질이 검증되는 것은 아니다.
