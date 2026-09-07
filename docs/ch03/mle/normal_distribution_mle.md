# 정규분포 MLE

정규분포 MLE - 평균과 분산 추정하기. 정규분포의 평균(μ)과 분산(σ²)을 함께 추정하는 법을 배운다

이 튜토리얼은 PyTorch에서 최대가능도 추정에 대한 기초적인 이해를 쌓는다. 코드를 따라가 보면 모델을 세우고, 손실 함수를 정의하고, 경사 하강법으로 학습시키고, 분류 과제에서 성능을 평가하는 법을 알 수 있다.

## 코드

```python
#!/usr/bin/env python3
"""
================================================================================
정규 분포 최대가능도 — 평균과 분산 어림
================================================================================

어려움: ⭐⭐ 보통(2단계)

가우스 분포의 평균(μ)과 분산(σ²)을 최대가능도로 한꺼번에
어림하는 법을 배운다.

최대가능도 풀이:
μ̂ = (1/N) Σ xᵢ  (표본 평균)
σ̂² = (1/N) Σ (xᵢ - μ̂)²  (표본 분산)

다변량 최대가능도를 이해하는 밑바탕 보기다!
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
    로그 가능도에 경사 상승법을 써서 매개변수를 어림한다.
    
    시그마가 양수가 되도록 로그로 매긴다.
    σ = exp(log_sigma)
    """
    # 매개변수를 초기화한다
    mu = torch.tensor(0.0, requires_grad=True)
    log_sigma = torch.tensor(0.0, requires_grad=True)  # σ = exp(log_sigma) > 0
    
    optimizer = torch.optim.Adam([mu, log_sigma], lr=0.01)
    history = []
    
    for i in range(n_iterations):
        sigma = torch.exp(log_sigma)  # Transform to ensure σ > 0
        
        # 로그가능도를 계산한다
        log_lik = compute_log_likelihood(data, mu, sigma)
        
        # 손실은 음의 로그가능도이다
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
    
    # 그림 2: 매개변수의 수렴 (μ)
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
    
    # 그림 3: 매개변수의 수렴 (σ)
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
    
    # 경사 기반 MLE
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
    main()```

## 논의

손실 계산은 모델의 출력을 최적화 목표와 이어 준다. 알맞은 손실 함수를 고르는 일은 결정적으로 중요하다. 손실 함수가 모델이 무엇을 최적화하도록 배울지를 정하며, 학습된 표현과 결정 경계를 직접 빚어내기 때문이다.

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다. 그림을 그리는 코드는 학습된 표현, 수렴의 움직임, 평가 지표에 대한 통찰을 주어 추상적인 계산을 손에 잡히게 만든다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 확장된다. 초매개변수, 구조의 변형, 여러 데이터셋을 두루 실험해 보면 이해가 깊어지고 통계적 추론 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
코드를 끝까지 읽고 핵심적인 설계 결정을 찾아내라. 구체적인 구현 선택 세 가지를 나열하고, 각각이 최대가능도 추정에 왜 적절한지 설명하라.

??? success "연습문제 1 풀이"
    설계 결정은 구현마다 다르지만 흔히 다음이 포함된다. (1) 활성화 함수의 선택 — ReLU 계열은 포화되지 않는 경사를 주어 학습을 빠르게 한다. (2) 정규화 전략 — 배치 정규화는 내부 공변량 이동을 줄여 학습을 안정시킨다. (3) 잔차 연결 — 있을 경우 건너뛰는 경로를 제공하여 깊은 신경망에서도 경사가 흐르게 한다. 각 선택은 표현력, 계산 비용, 학습 안정성 사이의 절충을 반영한다.

---

**연습문제 2.**
입력이 기대하는 모양과 데이터형을 갖는지 확인하도록 주 함수나 클래스에 입력 검증을 추가하라. 잘못된 입력에는 유익한 오류 메시지를 내라.

??? success "연습문제 2 풀이"
    `forward` 메서드(또는 해당 함수)의 첫머리에 다음과 같은 검사를 추가한다. `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`와 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'`. 모양을 검증할 때는 중요한 차원을 확인한다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 유익한 오류 메시지는 디버깅 속도를 크게 높이고 코드를 재사용하기에도 더 견고하게 만든다.

---

**연습문제 3.**
이 구현에서 생길 수 있는 실패 양상 두 가지를 서술하고, 각각을 어떻게 진단하고 고칠지 설명하라.

??? success "연습문제 3 풀이"
    흔한 실패 양상은 다음과 같다. (1) **경사 소실/폭발** — 경사의 노름을 지켜보아 진단한다(`torch.nn.utils.clip_grad_norm_`을 쓰거나 층마다 `param.grad.norm()`을 기록한다). 경사 자르기, 더 나은 초기화(Xavier/Kaiming), 또는 구조 변경(잔차 연결, 정규화)으로 고친다. (2) **과적합** — 학습 손실은 줄어드는데 검증 손실이 늘어나면 진단된다. 정칙화(드롭아웃, 가중치 감쇠, 데이터 증강)나 모델 용량 축소로 고친다. 이런 문제를 일찍 잡아내려면 언제나 학습 지표와 검증 지표를 함께 살펴라.

---

**연습문제 4.**
정규분포 MLE 구현을 검증하는 종합적인 시험 함수를 작성하라. 빈 입력, 원소가 하나뿐인 입력, 아주 큰 입력, 그리고 극단적인 값(0, 아주 큰 수)을 가진 입력 등 경계 사례를 시험하라.

??? success "연습문제 4 풀이"
    경계 조건을 두루 시험하는 함수를 만든다.
    ```python
    def test_normal distribution mle():
        model = Normal Distribution MLE(...)
        # 보통의 입력
        assert model(normal_input).shape == expected_shape
        # 원소가 하나인 배치
        assert model(single_input).shape == (1, ...)
        # 큰 값 (넘침을 확인한다)
        out = model(torch.ones(...) * 1000)
        assert torch.isfinite(out).all()
        # 경사의 흐름
        out = model(normal_input)
        out.sum().backward()
        for p in model.parameters():
            assert p.grad is not None
    ```
    경사의 흐름을 시험하는 것은 그 구조가 처음부터 끝까지 이어지는 학습을 지원하는지 확인하는 데 특히 중요하다.
