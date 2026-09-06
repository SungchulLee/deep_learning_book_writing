# 저민 점수 맞추기 2

이 단원은 요즘 만들어 내는 모델의 핵심 부품인 저민 점수 맞추기 2을 짠다. 여기서 보이는 개념과 재주를 알면 퍼짐 모델과 점수 바탕 만들어 내는 방법을 다루는 데 꼭 필요한 앎을 얻는다. 이 짜기는 또렷함과 실제 쓸모의 균형을 맞추어 배우기에도 실험하기에도 알맞다.

## 코드

```python
"""
FILE: 05_sliced_score_matching.py
어려움: 중간
걸리는 시간: 2~3시간
PREREQUISITES: 02_score_matching_theory.py, 04_denoising_score_matching.py

학습 목표:
    1. Understand Sliced Score Matching (SSM)
    2. 아무 쏘기 재주를 짠다
    3. 저민 점수 맞추기와 잡음 없애는 점수 맞추기의 효율을 견준다
    4. 방법 사이의 맞바꿈을 살핀다

MATHEMATICAL BACKGROUND:
    저민 점수 맞추기는 온전한 야코비 셈을 피하려 아무 쏘기를 쓴다.
    
    L_SSM = E_x E_v[v^T∇s_θ(x)v + 1/2||v^Ts_θ(x)||²]
    
    여기서 v ~ N(0, I)은 아무 쏘기 방향이다.
    
    핵심 장점: 값싼 야코비-벡터 곱(JVP)만 필요하다!
"""

import torch
import torch.nn as nn
import numpy as np

# ========================================================================
# 메인
# ========================================================================


def sliced_score_matching_loss(model, x, n_projections=1):
    """
    아무 쏘기로 저민 점수 맞추기 손실을 셈한다.
    
    The key insight is that we only need directional derivatives:
    v^T∇s_θ(x)v은 야코비-벡터 곱으로 효율 좋게 셈할 수 있다.
    
    인수:
        model: 점수 신경망
        x: Data samples, shape (N, D)
        n_projections: 표본마다 아무 쏘기 횟수
    
    반환값:
        loss: 저민 점수 맞추기 손실 값
    """
    x = x.requires_grad_(True)
    N, D = x.shape
    
    loss = 0.0
    
    for _ in range(n_projections):
        # 아무 쏘기 방향을 뽑는다
        v = torch.randn_like(x)  # v ~ N(0, I)
        
        # 점수 셈하기
        score = model(x)
        
        # v^T s_θ(x)을 셈한다(안쪽 곱)
        score_v = torch.sum(score * v, dim=1, keepdim=True)
        
        # 자동 미분으로 ∇(v^T s_θ(x)) · v을 셈한다
        # 이는 v^T ∇s_θ(x) v과 같다
        grad_score_v = torch.autograd.grad(
            outputs=score_v,
            inputs=x,
            grad_outputs=torch.ones_like(score_v),
            create_graph=True
        )[0]
        
        trace_term = torch.sum(grad_score_v * v, dim=1)
        norm_term = 0.5 * score_v.squeeze() ** 2
        
        loss = loss + torch.mean(trace_term + norm_term)
    
    return loss / n_projections


def demo_ssm():
    """단순한 2차원 정규 분포에서 저민 점수 맞추기를 보여 준다."""
    print("Sliced Score Matching Demo")
    print("=" * 80)
    
    # 데이터를 생성한다
    data = torch.randn(500, 2)
    
    # 단순한 모델
    model = nn.Sequential(
        nn.Linear(2, 64),
        nn.Softplus(),
        nn.Linear(64, 64),
        nn.Softplus(),
        nn.Linear(64, 2)
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print("\nTraining with SSM...")
    for epoch in range(1000):
        optimizer.zero_grad()
        loss = sliced_score_matching_loss(model, data, n_projections=1)
        loss.backward()
        optimizer.step()
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f}")
    
    print("\n✓ SSM training complete!")
    print("\nKey observation: SSM avoids full Jacobian computation")
    print("  - Only needs JVPs (cheap!)")
    print("  - Scales well to high dimensions")
    print("  - Unbiased estimator of ESM objective")


if __name__ == "__main__":
    demo_ssm()```

## 논의

저민 점수 맞추기 2의 짜기는 이 마당에 자리 잡은 방식을 따른다. 코드 짜임이 모델 뜻매김과 익히기 논리를 갈라 놓아 부품을 하나씩 고치기 쉽다. 얼개 고르기는 만들어 내는 모델 무리가 많은 실험에서 얻은 배움을 담고 있다.

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
