# 슬라이스드 스코어 매칭

슬라이스드 스코어 매칭(sliced score matching, SSM)은 스코어 차이를 임의 방향에 투영해 학습하는 방법이다. 모든 입력 좌표의 야코비안을 명시적으로 만들지 않고 방향 미분을 사용할 수 있다. [SSM 논문](https://proceedings.mlr.press/v115/song20a.html)의 접근과 연결되며, 이 페이지는 2차원 나선 데이터에 대한 간단한 구현을 살펴본다.

## 1. 코드와 기존 출력

```python
"""
저민 점수 맞추기
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
    저민 점수 맞추기 로스를 셈한다
    
    인수:
        score_fn: 점수 뉴럴 넷트웍
        x: Data batch [B, D]
        n_projections: 아무 쏘기 횟수
    
    반환값:
        loss: 저민 점수 맞추기 로스 값
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
        
        # 저민 점수 맞추기 로스의 몫
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
    pass
```

```
MODULE 05: Sliced Score Matching
================================================================================
Training with Sliced Score Matching...
Epoch 0: Loss = 0.056761
Epoch 200: Loss = -0.061687
Epoch 400: Loss = -0.369804
Epoch 600: Loss = -1.321302
Epoch 800: Loss = -2.953426

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


✓ Module 05 complete!
```

## 2. 투영 목적함수

모델 스코어를 sθ, 입력 야코비안을 $J_xs_\theta$라 하자. 데이터와 독립적인 투영 벡터 v를 쓰고 경계항이 사라지는 조건에서, θ에 무관한 상수를 제외한 목적함수는 다음과 같다.

$$J_{SSM}(\theta)=\mathbb E_{x,v}\left[
 v^T(J_xs_\theta(x))v+\frac12(v^Ts_\theta(x))^2\right]$$

이는 참 데이터 스코어와의 **투영된** 제곱오차를 부분적분한 형태다. 관측에 가우시안 노이즈를 추가해 목표 −ξ/σ를 회귀하는 DSM과는 다르다. 다만 투영 방향을 새로 뽑기 때문에 학습 자체가 결정적인 것은 아니다.

코드는 정규 난수 v를 각 행의 길이로 나눈다. 분모의 작은 ε를 무시하면 방향은 단위 구면에서 균등하고 $\mathbb E[vv^T]=I/d$다. 그러므로 투영 평균은 전체 trace 목적함수의 **1/d배**다. 같은 최적점을 가지는 상수 배율이지만, 코드의 출력처럼 피셔 발산 값 자체의 비바이어스 추정량이라고 부르면 빠진 상수와 배율을 혼동하게 된다.

## 3. 자동미분이 계산하는 것

입력 x, score와 v의 모양은 모두 N×d다. `v_score`는 행별 내적이므로 N×1이고, 그 합을 x로 미분하면 표본별 $J_xs_\theta(x)^Tv$가 나온다. 여기에 v를 내적하면 필요한 스칼라 $v^TJs_\theta v$를 얻는다. 따라서 코드의 역방향 자동미분은 **벡터–야코비안 곱**을 사용하며 최종 이차형에서는 전치가 같은 값을 준다.

배치마다 독립적으로 작동하는 네트워크를 가정한다. 배치 정규화처럼 다른 표본에 의존하면 합을 미분한 값에 교차 표본 항이 섞일 수 있다. `create_graph=True`는 입력 미분 결과를 파라미터에 대해 다시 미분하기 위해 필요하다. 전체 야코비안·헤시안을 저장하지 않는다는 것과 입력 미분이나 고차 혼합 미분이 없다는 것은 다르다.

동일한 x에서 여러 독립 v를 평균 내면 유한 분산 조건에서 **투영에 따른** 몬테카를로 변동을 줄일 수 있다. 데이터 표본 오차·모델 오차까지 같은 비율로 사라지는 것은 아니다. n_projections는 양의 정수여야 하며 0이면 나눗셈이 유효하지 않다.

## 4. 실제 예제의 범위

main은 3차원 Swiss roll에서 두 좌표를 골라 2,000×2 입력을 만든다. 2→64→64→2의 tanh MLP는 파라미터 4,482개이며, Adam 러닝레이트 0.001로 전체 데이터를 1,000번 갱신한다. 갱신마다 두 개의 투영 방향을 사용한다. 코드에는 노이즈 스케줄, σ 조건 입력이나 샘플링이 없다.

목적값은 상수를 제거한 형태라 음수일 수 있다. 학습 로스의 크기나 감소만으로 정확한 밀도·스코어·샘플 품질을 입증하지 않는다. 임의의 벡터 MLP가 언제나 정규화 가능한 밀도의 스코어를 정의하는 것도 아니다. 검증 데이터와 해석적으로 알려진 대상 등을 이용한 별도 확인이 필요하다.

출력의 O(d·p), O(d²) 비교는 뉴럴 넷트웍 평가·자동미분 비용을 생략한 단순화다. 투영 수에 따라 방향 미분 비용이 늘지만 실제 시간과 메모리는 구조·차원·미분 구현에 달려 있다. 데이터에 노이즈를 더하지 않는다는 이유로 모든 높은 차원·비정칙 분포에서 학습이 잘된다고 보장할 수는 없다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
투영 방향 v와 score가 N×d일 때 v_score, 입력 미분과 최종 이차형의 모양을 설명하라.

</div>

??? success "풀이"

    v_score는 N×1, 그 합의 입력 미분은 N×d, v와 내적한 값은 N이다. 각 방향의 결과를 합산하고 표본 평균과 투영 수로 나누어 스칼라 로스를 만든다. 표본별 독립 모델에서 입력 미분은 Jᵀv에 해당한다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
단위 구면 투영에서 왜 trace 목적함수의 1/d배를 얻는가?

</div>

??? success "풀이"

    균등 단위 방향은 E[vvᵀ]=I/d이므로 E[vᵀJv]=tr(J)/d이고 E[(vᵀs)²]=‖s‖²/d다. 따라서 θ 의존 목적은 일반 스코어 매칭 목적의 1/d배다. 같은 최적점을 갖지만 참 스코어 제곱항의 상수가 빠져 있어 출력값이 피셔 발산 자체는 아니다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
n_projections를 늘리는 것과 노이즈 스케줄을 바꾸는 것의 차이는 무엇인가?

</div>

??? success "풀이"

    투영 수는 같은 입력에서 방향 평균을 얼마나 근사하는지 바꾼다. 이 페이지는 데이터 노이즈 수준이나 확산 시간을 조건으로 학습하지 않는다. 독립 투영의 평균은 조건부 투영 변동을 줄일 수 있지만 계산량이 늘며 데이터·모델 오차를 제거하지는 않는다.



## 정리하며

SSM은 임의 방향의 입력 미분으로 스코어 매칭을 근사한다. 단위 방향의 배율과 빠진 상수, 역방향 자동미분의 Jᵀv를 구분한다. 노이즈를 관측에 더하지 않더라도 투영은 확률적이며 학습·샘플 정확성은 별도로 평가해야 한다.
