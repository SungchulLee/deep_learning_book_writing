# 스코어 기반 확률미분방정식

스코어 기반 SDE는 데이터에 노이즈를 더하는 연속시간 과정과 이를 거꾸로 따라가는 생성 과정을 연결한다. 시간별 주변 밀도의 스코어를 알면 역과정의 드리프트를 표현할 수 있다. 이 페이지는 VE·VP 계수와 학습 예제를 살펴보며, 보존된 코드의 시간 부호와 로스 축 불일치를 구분한다.

## 1. 코드와 기존 출력

```python
"""
FILE: 08_score_sde.py
어려움: 나아간 단계
걸리는 시간: 4~5시간
미리 알 것: 07_noise_conditional_scores.py, 기본 확률 미분 방정식, 미적분

학습 목표:
    1. 점수 바탕 확률 미분 방정식 틀을 이해한다
    2. 흩어짐 터짐(VE) 확률 미분 방정식을 짠다
    3. 흩어짐 지키기(VP) 확률 미분 방정식을 짠다
    4. 확률 흐름 상미분 방정식을 이해한다
    5. 거꾸로 된 때의 확률 미분 방정식 뽑기를 짠다

수학 바탕:
    점수 바탕 확률 미분 방정식은 확산을 이어진 때로 적어 준다.
    
    앞으로 가는 SDE:
    dx = f(x, t)dt + g(t)dw
    
    여기서 w은 브라운 액션이다.
    
    거꾸로 가는 SDE:
    dx = [f(x, t) - g(t)²∇log p_t(x)]dt + g(t)dw̄
    
    우리가 배우는 것은 점수 ∇log p_t(x)이다!
    
    VARIANCE EXPLODING (VE):
    f(x, t) = 0
    g(t) = σ(t)√(dσ(t)²/dt)
    
    VARIANCE PRESERVING (VP):
    f(x, t) = -β(t)x/2
    g(t) = √β(t)
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ========================================================================
# 메인
# ========================================================================


class ScoreSDE:
    """점수 바탕 확률 미분 방정식의 바탕 클래스."""
    
    def __init__(self, beta_min=0.1, beta_max=20.0, T=1.0):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.T = T
    
    def f(self, x, t):
        """떠돎 계수."""
        raise NotImplementedError
    
    def g(self, t):
        """확산 계수."""
        raise NotImplementedError
    
    def marginal_prob(self, x0, t):
        """
        p_t(x|x₀)의 평균과 표준편차를 셈한다.
        
        반환값:
            mean: E[x_t | x₀]
            std: √Var[x_t | x₀]
        """
        raise NotImplementedError


class VESDE(ScoreSDE):
    """
    흩어짐 터짐 확률 미분 방정식.
    
    dx = σ(t)√(dσ(t)²/dt) dw
    
    Marginal: p_t(x|x₀) = N(x|x₀, σ(t)²I)
    """
    
    def __init__(self, sigma_min=0.01, sigma_max=50.0):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
    
    def f(self, x, t):
        return torch.zeros_like(x)
    
    def g(self, t):
        sigma_t = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        return sigma_t * np.sqrt(2 * np.log(self.sigma_max / self.sigma_min))
    
    def marginal_prob(self, x0, t):
        sigma_t = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        mean = x0
        std = sigma_t
        return mean, std


class VPSDE(ScoreSDE):
    """
    흩어짐 지키기 확률 미분 방정식.
    
    dx = -β(t)x/2 dt + √β(t) dw
    
    Marginal: p_t(x|x₀) = N(x | α_t x₀, (1-α_t²)I)
    """
    
    def __init__(self, beta_min=0.1, beta_max=20.0):
        super().__init__(beta_min, beta_max)
    
    def beta(self, t):
        return self.beta_min + t * (self.beta_max - self.beta_min)
    
    def f(self, x, t):
        return -0.5 * self.beta(t) * x
    
    def g(self, t):
        # t가 파이썬 실수로 들어오면 beta(t)도 실수가 되어 torch.sqrt가 거부한다
        return torch.sqrt(torch.as_tensor(self.beta(t), dtype=torch.float32))
    
    def marginal_prob(self, x0, t):
        log_alpha_t = -0.25 * t ** 2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        alpha_t = torch.exp(log_alpha_t)
        mean = alpha_t * x0
        std = torch.sqrt(1 - alpha_t ** 2)
        return mean, std


def sde_loss(model, sde, x0):
    """
    확률 미분 방정식의 점수 맞추기 로스를 셈한다.
    
    L = E_t E_x₀ E_x_t[||s_θ(x_t, t) - ∇log p_t(x_t|x₀)||²]
    """
    # 아무 때를 뽑는다
    batch_size = len(x0)
    t = torch.rand(batch_size, device=x0.device)
    
    # 가장자리 분포에서 뽑는다
    mean, std = sde.marginal_prob(x0, t.view(-1, 1))
    z = torch.randn_like(x0)
    x_t = mean + std * z
    
    # 점수 미리보기
    score_pred = model(x_t, t)
    
    # 참 점수: ∇log p_t(x_t|x₀) = -(x_t - mean)/std²
    score_true = -z / std
    
    # 표준 편차 제곱으로 무게를 준 평균 제곱 어긋남 로스
    loss = torch.mean(std ** 2 * torch.sum((score_pred - score_true) ** 2, dim=1))
    
    return loss


def reverse_sde_sampling(model, sde, shape, n_steps=1000):
    """
    거꾸로 된 때의 확률 미분 방정식으로 뽑는다.
    
    Implements: dx = [f - g²∇log p]dt + g dw̄
    """
    x = torch.randn(shape)
    dt = 1.0 / n_steps
    
    with torch.no_grad():
        for i in range(n_steps):
            t = 1.0 - i * dt
            t_tensor = torch.full((shape[0],), t)
            
            # 계수를 셈한다
            f = sde.f(x, t)
            g = sde.g(t)
            
            # 점수 셈하기
            score = model(x, t_tensor)
            
            # 뒤 확률 미분 방정식 걸음
            drift = f - (g ** 2) * score
            diffusion = g
            
            x = x + drift * dt + diffusion * np.sqrt(dt) * torch.randn_like(x)
    
    return x


def demo_sde():
    """2차원 정규 분포에서 점수 확률 미분 방정식을 보여 준다."""
    print("Score-based SDE Demo")
    print("=" * 80)
    
    # 단순한 2차원 정규 분포 자료
    data = torch.randn(1000, 2)
    
    # 확률 미분 방정식을 만든다
    sde = VPSDE(beta_min=0.1, beta_max=20.0)
    
    # 단순한 점수 모델(때 조건)
    class SimpleScoreModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(3, 128),  # x(2차원) + t(1차원)
                nn.SiLU(),
                nn.Linear(128, 128),
                nn.SiLU(),
                nn.Linear(128, 2)
            )
        
        def forward(self, x, t):
            t = t.view(-1, 1)
            inp = torch.cat([x, t], dim=-1)
            return self.net(inp)
    
    model = SimpleScoreModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print("\nTraining score model...")
    for epoch in range(3000):
        optimizer.zero_grad()
        loss = sde_loss(model, sde, data)
        loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f}")
    
    # 뽑기
    print("\nGenerating samples via reverse SDE...")
    samples = reverse_sde_sampling(model, sde, (500, 2), n_steps=1000)
    
    # 시각화한다
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].scatter(data[:, 0].numpy(), data[:, 1].numpy(), s=1, alpha=0.5)
    axes[0].set_title('Training Data')
    axes[0].set_aspect('equal')
    
    axes[1].scatter(samples[:, 0].numpy(), samples[:, 1].numpy(), s=1, alpha=0.5)
    axes[1].set_title('Generated Samples (Reverse SDE)')
    axes[1].set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('demo_score_sde.png', dpi=150, bbox_inches='tight')
    print("\nSaved demo_score_sde.png")
    print("\n✓ Score SDE successfully implemented!")


if __name__ == "__main__":
    demo_sde()
```

```
Score-based SDE Demo
================================================================================

Training score model...
Epoch    0 | Loss: 1246.737671
Epoch  500 | Loss: 19.561932
Epoch 1000 | Loss: 165.794067
Epoch 1500 | Loss: nan
Epoch 2000 | Loss: nan
Epoch 2500 | Loss: nan

Generating samples via reverse SDE...

Saved demo_score_sde.png

✓ Score SDE successfully implemented!
```

## 2. 정방향과 역시간의 부호

스칼라 확산 계수 g(t)가 상태에 의존하지 않는 정방향 과정을 생각한다.

$$dX_t=f(X_t,t)dt+g(t)dW_t$$

적절한 조건에서 주변 스코어 $s(x,t)=\nabla_x\log p_t(x)$를 사용한 역시간 식은 다음과 같다. 여기서 시간 t는 T에서 0으로 **감소한다**.

$$dX_t=[f(X_t,t)-g(t)^2s(X_t,t)]dt+g(t)d\overline W_t$$

양의 h로 t에서 t−h로 이동하는 오일러 단계는 드리프트에 **−h**를 곱하고 잡음 크기에 √h를 쓴다. 하지만 `reverse_sde_sampling`은 t를 줄이면서도 코드의 drift에 **+dt**를 곱한다. 따라서 표시한 역시간 SDE를 올바른 방향으로 적분하는 구현이 아니다. [스코어 SDE 논문](https://arxiv.org/abs/2011.13456)의 연속시간 식과 수치 구현의 시간 관례를 함께 확인해야 한다.

표준 정규분포가 모든 시간에서 유지되는 VP 예로 부호를 확인할 수 있다. s=−x, f=−βx/2, g²=β이면 역시간 드리프트는 +βx/2다. 올바른 뒤로 가는 단계의 결정적 항은 −hβx/2로 수축하지만 현재 코드는 +hβx/2로 팽창시킨다.

## 3. VP 과정의 조건부 분포

β(t)=βmin+t(βmax−βmin)인 VP 과정은 다음과 같다.

$$f(x,t)=-\frac12\beta(t)x,\qquad g(t)=\sqrt{\beta(t)}$$

$B(t)=\int_0^t\beta(u)du$라 두면 α(t)=exp(−B(t)/2)이며, 조건부 분포의 평균과 표준편차는 다음과 같다.

$$X_t\mid X_0=x_0\sim N(\alpha(t)x_0,[1-\alpha(t)^2]I)$$

코드의 alpha_t는 **신호 진폭**이다. DDPM의 누적 분산 계수 ᾱt와 비교하면 α(t)²에 대응하므로 이름만 보고 같은 수라고 해석하지 않는다. 이 과정은 초기 공분산이 I이면 이를 유지하지만 모든 초기 데이터의 분산을 그대로 보존한다는 뜻은 아니다.

`marginal_prob`는 학습에서 N×1 시간 텐서를 받아 N×d 평균과 N×1 표준편차를 만든다. `f`에서 표본별 t를 길이 N으로 직접 전달하면 N×d 입력과의 브로드캐스팅이 맞지 않을 수 있다. 현재 샘플러는 스칼라 t를 사용한다. T 속성은 저장하지만 학습·샘플링 시간 구간은 0~1로 고정되어 있어 T를 바꿔도 자동으로 일치하지 않는다.

## 4. VE 과정의 초기 분산 관례

VE 코드는 $\sigma(t)=\sigma_{min}(\sigma_{max}/\sigma_{min})^t$와 f=0, $g(t)^2=d\sigma(t)^2/dt$를 사용한다. 그런데 `marginal_prob`는 t=0에서도 표준편차 σmin을 반환한다. 깨끗한 X₀에서 이 SDE를 시작한다면 조건부 분산은 **σ(t)²−σmin²**다. σ(t)²를 사용하려면 초기 데이터가 이미 σmin 노이즈로 평활화된다는 관례를 명시해야 한다.

샘플러는 SDE 종류와 무관하게 N(0,I)에서 시작한다. VP에서는 충분한 노이즈 시간 뒤의 근사 사전분포로 사용할 수 있지만 일반 VE의 σmax 스케일과는 맞지 않는다. 클래스 인터페이스만 같다고 같은 초기 분포를 사용할 수 있는 것은 아니다.

## 5. DSM 로스와 축 문제

정방향 조건부 평균 m과 표준편차 std를 이용해 xₜ=m+std·z를 만들면 계산 가능한 조건부 스코어 목표는 −z/std다. 코드의 `score_true`는 이 조건부 목표이며 미지의 주변 스코어를 직접 계산한 값은 아니다. 회귀의 모집단 최적점이 주변 스코어가 된다.

그러나 std²는 N×1이고 좌표별 제곱오차의 합은 길이 N이다. 둘을 그대로 곱하면 **N×N** 외적이 된다. 실제 로스는 표본별 가중 평균이 아니라 평균 std²×평균 제곱오차이며, 의도한 시간별 가중 DSM과 다르다. 1,000개 데이터에서 불필요한 백만 원소의 중간 결과도 만든다.

VP의 std는 t→0에서 0으로 가므로 목표의 크기가 커지고 정확히 0이면 나눗셈이 유효하지 않다. 이 코드에는 시간 하한이나 안정화 검사가 없다. std²를 나중에 곱한다고 앞선 0 나눗셈이 안전해지는 것도 아니다.

## 6. 실제 데모의 범위

데모는 1,000×2 표준 정규 데이터와 t를 연결하는 3→128→128→2 SiLU MLP를 사용한다. Adam, 러닝레이트 0.001로 3,000번 전체 배치를 갱신하며 VP β 범위는 0.1~20이다. 검증·시드 고정·스코어 오차 비교는 없다. 생성은 CPU의 500×2 상태에서 1,000단계를 수행한다.

학습·생성 그림과 성공 메시지는 위 로스 축과 역시간 부호 문제를 해결하지 않는다. 원래 데이터가 표준 정규분포이므로 시간별 참 주변 스코어 −x를 이용해 정방향 계수와 역단계의 부호를 해석적으로 확인할 수 있다. 코드와 출력은 보존하되 이를 정확한 스코어 SDE 학습·생성의 실증으로 설명하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
시간이 t에서 t−h로 감소할 때 역 SDE의 오일러 단계에서 어떤 부호를 사용하는가?

</div>

??? success "연습문제 1 풀이"

    역 드리프트 b=f−g²s에 대해 상태 이동은 −hb이고 잡음 표준편차는 g√h다. 드리프트에 +h를 곱하려면 감소 시간에 맞춰 드리프트 자체를 반대로 정의해야 한다. 현재 함수는 그 반전을 하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
N×1 std²와 길이 N 오차 벡터의 곱은 의도한 가중 DSM인가?

</div>

??? success "연습문제 2 풀이"

    아니다. N×N 외적이 되어 전체 평균은 평균 웨이트×평균 오차다. 표본별 가중 평균에는 같은 표본 축의 원소끼리 곱해야 한다. 노이즈 목표의 조건부 배율과 t≈0에서의 0 나눗셈도 별도로 확인한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
VE의 σ(t)와 VP의 α(t)·std(t)는 같은 노이즈 스케줄을 표현하는가?

</div>

??? success "연습문제 3 풀이"

    서로 다른 정방향 과정을 정의한다. VE는 드리프트 없이 분산을 늘리고 VP는 신호를 수축시키며 노이즈를 추가한다. 초기 관례와 종단 사전분포도 맞춰야 한다. 선형·코사인이라는 이름만 바꾸는 것으로 서로 전환할 수 없으며 현재 시간 구간도 0~1로 고정되어 있다.



## 정리하며

역시간 SDE에서는 시간 감소와 드리프트 부호를 일치시켜야 한다. VE·VP의 조건부 분포와 종단 초기화를 구분하고, DSM 웨이트를 표본별 오차와 같은 축에 맞춘다. 이 코드의 부호·브로드캐스팅 문제는 성공 메시지와 별도로 해석해야 한다.
