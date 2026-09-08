# 저민 점수 맞추기

이 단원은 요즘 만들어 내는 모델의 핵심 부품인 저민 점수 맞추기을 짠다. 여기서 보이는 개념과 재주를 알면 퍼짐 모델과 점수 바탕 만들어 내는 방법을 다루는 데 꼭 필요한 앎을 얻는다. 이 짜기는 또렷함과 실제 쓸모의 균형을 맞추어 배우기에도 실험하기에도 알맞다.

## 1. 코드

```python
"""
단원 05: 저민 점수 맞추기
================================

어려움: 중간
시간: 2시간
미리 알 것: 단원 04

학습 목표:
- 효율을 위한 저민 점수 맞추기를 이해한다
- 아무 쏘기 재주를 짠다
- 저민 점수 맞추기와 잡음 없애는 점수 맞추기를 견준다

핵심 생각: 셈을 줄이려 점수를 아무 방향에 쏜다

L_SSM(θ) = 𝔼_x 𝔼_v [v^T ∇s_θ(x) + 0.5||s_θ(x)||² v^T s_θ(x)]

여기서 v ~ Uniform(S^{d-1})은 마구잡이 단위 벡터다

지은이: 이성철 @ 연세대학교
"""

import torch
import torch.nn as nn
import numpy as np

# ========================================================================
# 메인
# ========================================================================
print("MODULE 05: Sliced Score Matching")
print("="*80)

def sliced_score_matching_loss(score_fn, x, n_projections=1):
    """
    저민 점수 맞추기 손실을 셈한다
    
    인수:
        score_fn: 점수 신경망
        x: Data batch [B, D]
        n_projections: 아무 쏘기 횟수
    
    반환값:
        loss: 저민 점수 맞추기 손실 값
    """
    x.requires_grad_(True)
    score = score_fn(x)
    
    loss = 0
    for _ in range(n_projections):
        # 아무 단위 벡터
        v = torch.randn_like(x)
        v = v / (torch.norm(v, dim=-1, keepdim=True) + 1e-8)
        
        # v^T * 점수
        v_score = torch.sum(v * score, dim=-1, keepdim=True)
        
        # x에 대한 ∇(v^T 점수)
        sv_x = torch.autograd.grad(v_score.sum(), x, create_graph=True)[0]
        
        # v^T * ∇점수
        v_grad_score = torch.sum(v * sv_x, dim=-1)
        
        # 저민 점수 맞추기 손실의 몫
        loss = loss + v_grad_score + 0.5 * v_score.squeeze() ** 2
    
    return loss.mean() / n_projections

# 사용 예
from sklearn.datasets import make_swiss_roll
X, _ = make_swiss_roll(n_samples=2000, noise=0.1)
X = X[:, [0, 2]]  # 2차원 조각을 취한다

class SimpleScoreNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.net(x)

model = SimpleScoreNet()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print("Training with Sliced Score Matching...")
X_tensor = torch.FloatTensor(X)
for epoch in range(1000):
    optimizer.zero_grad()
    loss = sliced_score_matching_loss(model, X_tensor, n_projections=2)
    loss.backward()
    optimizer.step()
    
    if epoch % 200 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

print("""
SSM의 좋은 점:
✓ 드러난 점수 맞추기보다 잘 든다(헤세가 없다)
✓ 피셔 벌어짐의 치우치지 않은 어림개
✓ 차원이 높아도 잘 듣는다
✓ 여러 번 쏘면 흩어짐이 줄어든다

COMPARISON:
- 잡음 없애는 점수 맞추기: 표본마다 O(d), 잡음이 필요하다
- SSM: 표본마다 O(d*p)이며 잡음이 필요 없다(p는 쏘아 내림 수)
- 드러난 점수 맞추기: 표본마다 O(d²), 쓸 수 없다

언제 SSM을 쓰는가:
- 잡음을 더하기 어렵거나 부자연스러울 때
- 정해진 대로 익히고 싶을 때(잡음을 뽑지 않는다)
- 어떤 이론상 보장이 필요할 때
""")

print("\n✓ Module 05 complete!")


if __name__ == "__main__":
    pass```

## 2. 논의

저민 점수 맞추기의 짜기는 이 마당에 자리 잡은 방식을 따른다. 코드 짜임이 모델 뜻매김과 익히기 논리를 갈라 놓아 부품을 하나씩 고치기 쉽다. 얼개 고르기는 만들어 내는 모델 무리가 많은 실험에서 얻은 배움을 담고 있다.

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

**다룬 것** — 저민 점수 맞추기

저민 점수 맞추기의 짜기는 이 마당에 자리 잡은 방식을 따른다.

고갱이 갈래는 `SimpleScoreNet`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
