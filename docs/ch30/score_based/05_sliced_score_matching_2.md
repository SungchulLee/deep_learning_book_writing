# 가우시안 투영을 사용하는 스코어 매칭

이 예제는 길이를 정규화하지 않은 표준 가우시안 방향을 사용해 SSM 목적함수를 계산한다. 앞 페이지의 단위 구면 방향과 달리 투영의 공분산이 I이므로, 투영 기대값의 배율이 일반 스코어 매칭 목적과 맞는다. 두 방향 분포는 같은 원리로 사용되지만 유한 투영의 크기와 변동은 다르다.

## 1. 코드와 기존 출력

```python
"""
FILE: 05_sliced_score_matching.py
어려움: 중간
걸리는 시간: 2~3시간
PREREQUISITES: 02_score_matching_theory.py, 04_denoising_score_matching.py

학습 목표:
    1. 저민 점수 맞추기(SSM)를 이해한다
    2. 아무 쏘기 재주를 짠다
    3. 저민 점수 맞추기와 잡음 없애는 점수 맞추기의 효율을 견준다
    4. 방법 사이의 맞바꿈을 살핀다

수학 바탕:
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
    아무 쏘기로 저민 점수 맞추기 로스를 셈한다.
    
    핵심 눈썰미는 방향 미분만 있으면 된다는 것이다.
    v^T∇s_θ(x)v는 야코비-벡터 곱으로 효율 좋게 셈할 수 있다.
    
    인수:
        model: 점수 뉴럴 넷트웍
        x: 자료 표본, 꼴 (N, D)
        n_projections: 표본마다 아무 쏘기 횟수
    
    반환값:
        loss: 저민 점수 맞추기 로스 값
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
        
        # 자동 미분으로 ∇(v^T s_θ(x)) · v를 셈한다
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
    demo_ssm()
```

```
Sliced Score Matching Demo
================================================================================

Training with SSM...
Epoch    0 | Loss: -0.020015
Epoch  200 | Loss: -1.292030
Epoch  400 | Loss: -1.061390
Epoch  600 | Loss: -1.039186
Epoch  800 | Loss: -0.922932

✓ SSM training complete!

Key observation: SSM avoids full Jacobian computation
  - Only needs JVPs (cheap!)
  - Scales well to high dimensions
  - Unbiased estimator of ESM objective
```

## 2. 가우시안 방향의 기대값

모델의 입력 야코비안을 $J_xs_\theta$라 하면 코드의 표본별 한 방향 목적은 다음과 같다.

$$\ell(x,v)=v^T(J_xs_\theta(x))v+\frac12(v^Ts_\theta(x))^2$$

v~N(0,I)는 입력과 독립이고 E[vvᵀ]=I이므로 다음 관계를 얻는다.

$$\mathbb E_v\ell(x,v)=\operatorname{Tr}(J_xs_\theta(x))+\frac12\|s_\theta(x)\|^2$$

따라서 고정 입력에 대한 투영 평균은 **상수를 제거한 trace 목적**의 비바이어스 추정이다. 데이터의 참 스코어 항까지 포함한 피셔 발산 값 자체를 계산하는 것은 아니다. 데이터 표본의 평균이 모집단 목적을 근사하려면 표본 조건과 적분·미분, 경계 조건도 필요하다.

이 페이지는 노이즈를 관측 x에 더하지 않지만 v는 매번 무작위로 생성한다. 투영 수를 늘리면 독립 방향에 따른 변동을 줄일 수 있어도 실제 학습의 최적성이나 모집단 일치를 보장하지는 않는다. n_projections는 양의 정수여야 한다.

## 3. 미분과 배치의 계약

x, score, v는 N×D이고 score_v는 N×1이다. `autograd.grad`는 score_v의 합을 x로 미분해 N×D의 $J_xs_\theta^Tv$를 반환한다. 다시 v와 내적하면 N개 이차형이 된다. 전체 야코비안 없이 계산하지만 출력의 'JVP만 필요'는 연산 방향을 단순화한 표현이며 실제 호출은 역방향 VJP다.

현재 MLP처럼 표본별로 독립적인 함수여야 이 해석이 성립한다. 다른 표본과 상호작용하는 모델은 교차 표본 미분을 별도로 고려해야 한다. `create_graph=True`는 로스를 파라미터로 역전파하기 위해 필요하고, no_grad 안에서는 입력 미분을 구할 수 없다.

모델의 순방향 계산은 투영 반복문 안에 있어 각 방향마다 다시 수행한다. 현재의 결정적인 Softplus MLP에서는 공유해 계산할 여지가 있지만 드롭아웃 등이 있다면 반복 평가 자체도 달라질 수 있다. 입력을 복제·분리하지 않고 requires_grad를 설정하므로 외부 그래프나 leaf 조건도 호출자가 확인해야 한다.

## 4. 실제 학습 예제와 한계

데모는 표준 정규분포에서 500×2 데이터를 한 번 뽑고, 2→64→64→2 Softplus MLP를 Adam 러닝레이트 0.001로 1,000번 갱신한다. 파라미터는 4,482개이고 갱신마다 투영은 한 개다. 난수 시드 고정, 검증, 노이즈 스케줄, 샘플 생성이나 스코어 오차 평가는 없다.

표준 정규의 참 스코어는 −x다. 이를 기준으로 별도 검증할 수 있지만 현재 출력은 학습 로스만 보여 주므로 성공 메시지를 정확한 스코어 추정의 증거로 읽지 않는다. 목적값은 음수가 될 수 있으며 더 작아진 모든 경험 로스가 일반화 개선을 뜻하지도 않는다.

SSM은 명시적인 전체 야코비안 저장을 피할 수 있는 방법이다. 실제 비용은 방향 수와 모델·미분 구조에 의존하며, 모든 고차원 문제에서 '저렴하다'거나 DSM보다 우수하다는 보장은 없다. [SSM 논문](https://proceedings.mlr.press/v115/song20a.html)의 방법과 조건을 함께 참고한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
500×2 입력의 한 투영에서 각 미분 결과의 모양을 설명하라.

</div>

??? success "연습문제 1 풀이"

    score와 v는 500×2, 내적 score_v는 500×1, 입력 미분은 500×2다. trace_term과 norm_term은 500개 값이고 이를 평균 내어 스칼라를 얻는다. 투영이 여러 개면 스칼라를 평균 낸다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
정규화하지 않은 가우시안 방향과 단위 구면 방향의 목적 배율은 어떻게 다른가?

</div>

??? success "연습문제 2 풀이"

    가우시안은 E[vvᵀ]=I라 trace 목적과 같은 배율이다. 균등 단위 구면은 I/D라 1/D배다. 이 예제 D=2에서는 단위 방향의 기대 목적이 절반이다. 두 값 모두 참 데이터 스코어의 상수가 빠진 형태이므로 피셔 발산 그 자체와 구분한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
참 스코어가 −x인 표준 정규 데이터에서 학습을 확인하려면 무엇을 비교할 수 있는가?

</div>

??? success "연습문제 3 풀이"

    독립 평가 점에서 모델 출력과 −x의 제곱오차를 비교하고 위치별 오차를 살펴볼 수 있다. 학습 로스는 투영 잡음과 경험 분포에 의존하므로 그것만으로 정확도를 판정하지 않는다. 이 페이지에는 그 검증이나 선형·코사인 노이즈 스케줄 비교가 구현되어 있지 않다.



## 정리하며

표준 가우시안 투영은 일반 trace 스코어 매칭 목적의 배율을 보존한다. 비바이어스라는 주장은 고정된 목적의 투영 추정에 관한 것이며 실제 스코어 정확도나 생성 품질의 보장은 아니다. 입력 미분, 배치 독립성, 투영 수와 검증 절차를 명시한다.
