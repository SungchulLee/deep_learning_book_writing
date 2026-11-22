# 정규분포 MLE

독립 정규표본의 평균과 표준편차를 해석적으로 추정하고 그래디언트 기반 결과와 비교한다. 양수 파라메터의 표현과 분산 MLE의 편향을 구분한다.

## 1. 코드

```python
#!/usr/bin/env python3
"""
================================================================================
정규 분포 최대가능도 — 평균과 분산 추정
================================================================================

난이도: ⭐⭐ 보통(2단계)

가우시안 분포의 평균(μ)과 분산(σ²)을 최대가능도로 한꺼번에
추정하는 법을 배운다.

최대가능도 해:
μ̂ = (1/N) Σ xᵢ  (표본 평균)
σ̂² = (1/N) Σ (xᵢ - μ̂)²  (표본 분산)

다변량 최대가능도를 이해하는 기본 예제다!
================================================================================
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

# ========================================================================
# 메인
# ========================================================================


def generate_normal_data(n_samples: int, true_mu: float, true_sigma: float, seed: int = 42):
    """정규분포에서 데이터를 생성한다"""
    torch.manual_seed(seed)
    data = torch.randn(n_samples) * true_sigma + true_mu
    return data


def compute_log_likelihood(data: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor):
    """
    정규 분포의 로그 가능도를 계산한다.
    
    ℓ(μ, σ²) = -N/2 * log(2πσ²) - (1/2σ²) * Σ(xᵢ - μ)²
    """
    n = len(data)
    sigma = torch.clamp(sigma, min=1e-6)  # Avoid division by zero
    
    # 로그가능도 공식
    log_lik = (-n/2) * torch.log(2 * np.pi * sigma**2) - torch.sum((data - mu)**2) / (2 * sigma**2)
    return log_lik


def analytical_mle(data: torch.Tensor):
    """MLE를 해석적으로 계산한다"""
    mu_mle = torch.mean(data)
    sigma_mle = torch.std(data, unbiased=False)  # MLE uses biased estimator
    return mu_mle.item(), sigma_mle.item()


def gradient_based_mle(data: torch.Tensor, n_iterations: int = 1000):
    """
    로그 가능도에 경사 상승법을 써서 파라메터를 추정한다.
    
    시그마가 양수가 되도록 로그로 매개변수화한다.
    σ = exp(log_sigma)
    """
    # 파라메터를 초기화한다
    mu = torch.tensor(0.0, requires_grad=True)
    log_sigma = torch.tensor(0.0, requires_grad=True)  # σ = exp(log_sigma) > 0
    
    optimizer = torch.optim.Adam([mu, log_sigma], lr=0.1)
    history = []
    
    for i in range(n_iterations):
        sigma = torch.exp(log_sigma)  # Transform to ensure σ > 0
        
        # 로그가능도를 계산한다
        log_lik = compute_log_likelihood(data, mu, sigma)
        
        # 로스는 음의 로그가능도이다
        loss = -log_lik
        
        history.append((mu.item(), sigma.item(), loss.item()))
        
        # 최적화 단계
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i + 1) % 200 == 0:
            print(f"   Iter {i+1}: μ={mu.item():.4f}, σ={sigma.item():.4f}, LL={log_lik.item():.2f}")
    
    return mu.item(), torch.exp(log_sigma).item(), history


def visualize_results(data, true_mu, true_sigma, analytical_mu, analytical_sigma,
                     gradient_mu, gradient_sigma, history):
    """시각화를 만든다"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # 그림 1: 적합된 분포를 함께 그린 데이터 히스토그램
    ax = axes[0, 0]
    ax.hist(data.numpy(), bins=30, density=True, alpha=0.6, edgecolor='black', label='Data')
    
    x_range = np.linspace(data.min().item(), data.max().item(), 100)
    
    # 참 분포
    from scipy.stats import norm
    ax.plot(x_range, norm.pdf(x_range, true_mu, true_sigma), 
           'g-', linewidth=2, label=f'True N({true_mu:.1f}, {true_sigma:.1f}²)')
    
    # MLE 분포
    ax.plot(x_range, norm.pdf(x_range, analytical_mu, analytical_sigma),
           'r--', linewidth=2, label=f'MLE N({analytical_mu:.2f}, {analytical_sigma:.2f}²)')
    
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.set_title('Data Distribution', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 그림 2: 파라메터의 수렴 (μ)
    ax = axes[0, 1]
    mus = [h[0] for h in history]
    ax.plot(mus, 'b-', linewidth=2, label='μ estimate')
    ax.axhline(true_mu, color='g', linestyle='--', label=f'True μ={true_mu}')
    ax.axhline(analytical_mu, color='r', linestyle='--', label=f'Analytical MLE')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('μ')
    ax.set_title('Mean Parameter Convergence', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 그림 3: 파라메터의 수렴 (σ)
    ax = axes[0, 2]
    sigmas = [h[1] for h in history]
    ax.plot(sigmas, 'b-', linewidth=2, label='σ estimate')
    ax.axhline(true_sigma, color='g', linestyle='--', label=f'True σ={true_sigma}')
    ax.axhline(analytical_sigma, color='r', linestyle='--', label=f'Analytical MLE')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('σ')
    ax.set_title('Std Dev Parameter Convergence', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 그림 4: 로그가능도 곡면
    ax = axes[1, 0]
    mu_range = np.linspace(true_mu - 2, true_mu + 2, 50)
    sigma_range = np.linspace(max(0.1, true_sigma - 1), true_sigma + 1, 50)
    MU, SIGMA = np.meshgrid(mu_range, sigma_range)
    
    LL = np.zeros_like(MU)
    for i in range(len(mu_range)):
        for j in range(len(sigma_range)):
            LL[j, i] = compute_log_likelihood(
                data, torch.tensor(MU[j, i]), torch.tensor(SIGMA[j, i])
            ).item()
    
    contour = ax.contour(MU, SIGMA, LL, levels=20, cmap='viridis')
    ax.clabel(contour, inline=True, fontsize=8)
    ax.plot(analytical_mu, analytical_sigma, 'r*', markersize=20, label='MLE')
    ax.plot(true_mu, true_sigma, 'go', markersize=12, label='True')
    ax.set_xlabel('μ')
    ax.set_ylabel('σ')
    ax.set_title('Log-Likelihood Surface', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 그림 5: Q-Q 그림
    ax = axes[1, 1]
    from scipy import stats
    stats.probplot((data.numpy() - analytical_mu) / analytical_sigma, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot (Normality Check)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 그림 6: 비교 표
    ax = axes[1, 2]
    ax.axis('off')
    
    table_data = [
        ['Method', 'μ', 'σ'],
        ['True', f'{true_mu:.4f}', f'{true_sigma:.4f}'],
        ['Analytical', f'{analytical_mu:.4f}', f'{analytical_sigma:.4f}'],
        ['Gradient', f'{gradient_mu:.4f}', f'{gradient_sigma:.4f}'],
    ]
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 3)
    
    for i in range(3):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax.set_title('Results Comparison', fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('normal_distribution_mle_results.png', dpi=150, bbox_inches='tight')
    print("\n📊 Figure saved as 'normal_distribution_mle_results.png'")
    plt.show()


def main():
    print("=" * 80)
    print("NORMAL DISTRIBUTION MLE - Parameter Estimation")
    print("=" * 80)
    
    # 준비
    N_SAMPLES = 500
    TRUE_MU = 5.0
    TRUE_SIGMA = 2.0
    
    print(f"\n📋 Setup: N={N_SAMPLES}, True μ={TRUE_MU}, True σ={TRUE_SIGMA}")
    
    # 데이터를 생성한다
    print("\n🎲 Generating data...")
    data = generate_normal_data(N_SAMPLES, TRUE_MU, TRUE_SIGMA)
    print(f"   Sample mean: {data.mean():.4f}")
    print(f"   Sample std:  {data.std():.4f}")
    
    # 해석적 MLE
    print("\n📐 Analytical MLE:")
    analytical_mu, analytical_sigma = analytical_mle(data)
    print(f"   μ̂ = {analytical_mu:.4f}")
    print(f"   σ̂ = {analytical_sigma:.4f}")
    
    # 그래디언트 기반 MLE
    print("\n🔄 Gradient-Based MLE:")
    gradient_mu, gradient_sigma, history = gradient_based_mle(data, n_iterations=1000)
    print(f"   Final μ̂ = {gradient_mu:.4f}")
    print(f"   Final σ̂ = {gradient_sigma:.4f}")
    
    # 시각화한다
    print("\n📊 Creating visualizations...")
    visualize_results(data, TRUE_MU, TRUE_SIGMA, analytical_mu, analytical_sigma,
                     gradient_mu, gradient_sigma, history)
    
    print("\n✅ Complete!")
    print("💡 Key takeaway: MLE for normal distribution gives sample mean and variance!")


if __name__ == "__main__":
    main()
```

**출력:**

```
================================================================================
NORMAL DISTRIBUTION MLE - Parameter Estimation
================================================================================

📋 Setup: N=500, True μ=5.0, True σ=2.0

🎲 Generating data...
   Sample mean: 5.0872
   Sample std:  1.9461

📐 Analytical MLE:
   μ̂ = 5.0872
   σ̂ = 1.9441

🔄 Gradient-Based MLE:
   Iter 200: μ=4.9860, σ=1.9662, LL=-1042.71
   Iter 400: μ=5.0872, σ=1.9441, LL=-1041.87
   Iter 600: μ=5.0872, σ=1.9441, LL=-1041.87
   Iter 800: μ=5.0872, σ=1.9441, LL=-1041.87
   Iter 1000: μ=5.0872, σ=1.9441, LL=-1041.87
   Final μ̂ = 5.0872
   Final σ̂ = 1.9441

📊 Creating visualizations...

📊 Figure saved as 'normal_distribution_mle_results.png'

✅ Complete!
💡 Key takeaway: MLE for normal distribution gives sample mean and variance!
```

## 2. 논의

### 해석적 추정값

독립 표본 $x_1,\dots,x_n$이 $\mathcal N(\mu,\sigma^2)$에서 나왔을 때 로그가능도(log-likelihood; 가능도는 우도라고도 한다)는 밀도의 로그를 모두 더한 것이다.

$$
\ell(\mu,\sigma^2) = -\frac n2\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n (x_i-\mu)^2
$$

두 파라메터로 각각 미분해 0으로 놓는다.

$$
\begin{aligned}
\frac{\partial \ell}{\partial \mu} &= \frac{1}{\sigma^2}\sum_i (x_i-\mu) = 0
&&\Longrightarrow\quad \hat\mu = \bar x = \frac1n\sum_i x_i, \\
\frac{\partial \ell}{\partial \sigma^2} &= -\frac{n}{2\sigma^2} + \frac{1}{2\sigma^4}\sum_i (x_i-\mu)^2 = 0
&&\Longrightarrow\quad \hat\sigma^2 = \frac1n\sum_i (x_i-\bar x)^2 .
\end{aligned}
$$

곧 평균 MLE는 표본평균이고, 분산 MLE는 분모가 $n$인 표본분산이다. 코드는 `torch.std(..., unbiased=False)`로 그 제곱근인 표준편차를 구한다.

이 분산 MLE는 **편향**되어 있다 — 추정량의 기댓값이 참값과 어긋난다는 뜻이다. $\mathbb E[\hat\sigma^2] = \frac{n-1}{n}\sigma^2$이어서 참 분산보다 조금 작게 나온다. 분모를 $n-1$로 바꾼 것이 기댓값이 참값과 같은 **불편** 표본분산이다. 위 출력의 `Sample std: 1.9461`은 분모 $n-1$로 구한 값이고(`Tensor.std()`의 기본값), MLE인 $\hat\sigma = 1.9441$은 거기에 $\sqrt{499/500}$을 곱한 값이다. 분산이 불편이어도 그 제곱근인 표준편차까지 불편인 것은 아니므로, 표준편차의 불편성은 따로 판단해야 한다.

### 양수 조건과 최적화

그래디언트 기반 방식은 `log_sigma`를 트레인하고 지수함수로 $\sigma$를 만든다. 이는 표준편차가 양수라는 조건을 만족시키지만 극단적인 값에서는 수치 문제가 생길 수 있다. 로그가능도 계산의 하한 클램프(`torch.clamp(sigma, min=1e-6)`)는 수치적 장치이며 분포의 원래 정의와 구분한다.

유한한 반복의 결과가 해석적 값에 가까운지 확인하면 최적화 상태를 평가할 수 있다. 위 실행에서는 400번째 반복에 이미 $\hat\mu=5.0872$, $\hat\sigma=1.9441$로 해석적 MLE와 소수 넷째 자리까지 일치하며, 그 뒤로 로그가능도도 $-1041.87$에서 변하지 않는다. 반면 생성 규칙의 참값과 추정값의 차이에는 표본 변동이 포함된다. 표본이 하나이거나 모든 관측값이 같으면 분산이 0으로 접근하면서 양의 분산 공간의 가능도가 무한히 커질 수 있으므로 일반적인 내부해 유도가 적용되지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
해석적 추정에서 `unbiased=False`를 사용하고 수치 최적화에서 `log_sigma`를 트레인하는 이유를 설명하라.

</div>

??? success "풀이"
    분산 MLE는 분모가 $n$이므로 `unbiased=False`의 표준편차가 그 제곱근을 준다. 로그 표준편차는 지수변환으로 양수 제약을 만족시킨다. 이는 불편 추정이나 최적화 수렴을 자동 보장하는 설정은 아니다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
정규분포 MLE에 입력할 데이터에서 어떤 조건과 경계 사례를 확인해야 하는가?

</div>

??? success "풀이"
    비어 있지 않은 유한한 실수 관측을 요구한다. 관측이 하나이거나 모든 값이 같으면 분산이 0으로 접근하는 퇴화가 생긴다. 결측값과 무한대, 상수 표본을 구분해 처리하고 단순히 `float32` 여부만 검사하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
참 파라메터와 수치 추정값의 차이를 표본 변동과 최적화 오차로 나누어 설명하라.

</div>

??? success "풀이"
    해석적 MLE와 수치 결과의 차이는 최적화·수치 오차를 살피는 기준이다. 해석적 MLE와 생성 규칙의 참값의 차이는 유한 표본의 변동을 포함한다. 표본을 다시 뽑지 않고 반복 횟수만 늘린다고 표본 변동까지 제거되지는 않는다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
`analytical_mle`와 `gradient_based_mle`를 검증하는 시험 함수를 작성하라. 보통의 입력에서 두 방법이 일치하는지 확인하고, 원소가 하나뿐인 입력, 상수 표본, 빈 입력, 평균이 아주 큰 입력 같은 경계 사례에서 `analytical_mle`가 무엇을 돌려주는지 시험하라. 각 결과가 뜻하는 바도 설명하라.

</div>

??? success "풀이"
    이 쪽의 함수 셋(`generate_normal_data`, `analytical_mle`, `gradient_based_mle`)을 그대로 불러 시험한다.
    ```python
    import math

    def test_normal_mle():
        data = generate_normal_data(500, 5.0, 2.0)
        a_mu, a_sigma = analytical_mle(data)

        # 1) 보통의 입력: 표본평균, 그리고 분모 n-1 표준편차에 sqrt(499/500)을 곱한 값
        assert abs(a_mu - data.mean().item()) < 1e-6
        assert abs(a_sigma - data.std().item() * math.sqrt(499 / 500)) < 1e-5

        # 2) 두 방법의 일치: 경사 상승법이 해석적 값에 닿는다
        g_mu, g_sigma, _ = gradient_based_mle(data, n_iterations=1000)
        assert abs(g_mu - a_mu) < 1e-4 and abs(g_sigma - a_sigma) < 1e-4

        # 3) 원소 하나 · 상수 표본: 표준편차가 0 으로 퇴화한다
        assert analytical_mle(torch.tensor([3.0])) == (3.0, 0.0)
        assert analytical_mle(torch.full((100,), 7.0)) == (7.0, 0.0)

        # 4) 빈 입력: 오류 없이 nan 이 나온다
        mu, sigma = analytical_mle(torch.tensor([]))
        assert math.isnan(mu) and math.isnan(sigma)

        # 5) 평균이 아주 큰 입력: float32 는 표준편차를 잃고 float64 는 지킨다
        _, s32 = analytical_mle(data + 1e6)
        _, s64 = analytical_mle(data.double() + 1e6)
        assert abs(s64 - a_sigma) < 1e-6
        assert abs(s32 - a_sigma) > 1e-3

        print("모든 시험 통과")

    test_normal_mle()
    ```
    돌리면 다섯 시험이 모두 통과한다. 통과했다는 것과 결과가 쓸 만하다는 것은 다르므로 하나씩 읽는다.

    - **1·2번**은 이 쪽의 주장 그대로다. 분산 MLE는 분모가 $n$이고, 경사 상승법은 같은 값에 닿는다.
    - **3번**의 $\hat\sigma=0$은 답이 아니라 **퇴화의 표시**다. 표준편차가 0인 정규분포는 없고, 이때 가능도는 $\sigma\to0$에서 한없이 커져 최댓값을 주는 $\sigma$가 없다.
    - **4번**은 함수가 빈 입력을 막지 않는다는 뜻이다. `nan`은 오류 없이 뒤의 계산으로 흘러가므로, 부르는 쪽에서 표본 수를 먼저 확인해야 한다.
    - **5번**은 자료형의 문제다. 같은 데이터에 $10^6$을 더하기만 했으니 표준편차는 $1.9441$ 그대로여야 하는데 float32로는 $1.9463$이 나온다. float32는 $10^6$ 근처에서 이웃한 두 수의 간격이 $0.0625$라서, 더하는 순간 소수 부분이 그 눈금으로 뭉개진다. float64로 바꾸면 $1.9441$이 지켜진다.

## 정리하며

정규분포의 해석적 MLE는 수치 최적화의 비교 기준이 된다. 표준편차와 분산, MLE와 불편 추정량, 최적화 오차와 표본 변동을 각각 구분한다.
