# 점수 신경망 얼개

이 단원은 요즘 만들어 내는 모델의 핵심 부품인 점수 신경망 얼개을 짠다. 여기서 보이는 개념과 재주를 알면 퍼짐 모델과 점수 바탕 만들어 내는 방법을 다루는 데 꼭 필요한 앎을 얻는다. 이 짜기는 또렷함과 실제 쓸모의 균형을 맞추어 배우기에도 실험하기에도 알맞다.

## 1. 코드

```python
"""
단원 06: 점수 신경망 얼개
=====================================

어려움: 중간
시간: 2~3시간
미리 알 것: 단원 01-05

학습 목표:
- 자료 갈래마다 잘 듣는 점수 신경망을 짠다
- 잡음과 때 조건 주기를 이해한다
- 요즘 얼개를 짠다

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
    
    퍼짐 모델에서 점수는 잡음 층 t에 달렸다.
    s_θ(x, t) = ∇_x log p_t(x)
    """
    def __init__(self, data_dim=2, hidden_dim=128, time_embed_dim=32):
        super().__init__()
        
        # 때 박아 넣기(변환기처럼 사인 꼴)
        self.time_embed_dim = time_embed_dim
        
        # 때 박아 넣기를 위한 여러 층 신경망
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
    # 묶음마다 아무 잡음 수준
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
고갱이 얼개 고르기:

1. 때 조건 주기:
   - 퍼짐 모델에 꼭 필요하다
   - 사인 꼴 묻힘(변환기 방식)
   - 또는 배울 수 있는 박아 넣기

2. 2차원과 표 자료에서:
   - 층 고르게 맞추기를 갖춘 여러 층 신경망
   - SiLU와 GELU 깨움
   - 남은 이음이 도움이 된다

3. 그림에서는(9단원에서 다룬다):
   - U-Net 얼개
   - 눈길 얼개
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

## 2. 논의

점수 신경망 얼개의 짜기는 이 마당에 자리 잡은 방식을 따른다. 코드 짜임이 모델 뜻매김과 익히기 논리를 갈라 놓아 부품을 하나씩 고치기 쉽다. 얼개 고르기는 만들어 내는 모델 무리가 많은 실험에서 얻은 배움을 담고 있다.

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

**다룬 것** — 점수 신경망 얼개

점수 신경망 얼개의 짜기는 이 마당에 자리 잡은 방식을 따른다.

고갱이 갈래는 `TimeConditionalScoreNetwork`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
