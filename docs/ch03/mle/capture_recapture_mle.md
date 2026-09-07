# 표지-재포획 MLE

표지-재포획 MLE - 야생동물 개체수 추정. 문제: 다음을 이용하여 서식지 내 동물의 전체 개체수를 추정한다

이 튜토리얼은 PyTorch에서 최대가능도 추정에 대한 기초적인 이해를 쌓는다. 코드를 따라가 보면 모델을 세우고, 손실 함수를 정의하고, 경사 하강법으로 학습시키고, 분류 과제에서 성능을 평가하는 법을 알 수 있다.

## 코드

```python
#!/usr/bin/env python3
"""
================================================================================
표지-재포획 최대가능도 — 야생 개체 수 어림
================================================================================

어려움: ⭐⭐ 보통(2단계)

문제: 표지-재포획 방법으로 어떤 서식지의 온 동물 수를
어림하여라.

METHODOLOGY:
1. 동물 C마리를 잡아 표시하고 놓아준다
2. 나중에 동물 R마리를 잡는다
3. 다시 잡은 것 가운데 표시된 T마리를 본다

물음: 온 개체 수 N은 얼마인가?

INTUITION: T/R ≈ C/N  =>  N ≈ (C × R) / T

이것이 링컨-피터슨 어림값이며 곧 최대가능도 어림값이다!

수학 모형:
다시 잡은 것 가운데 표시된 수는 초기하 분포를 따른다.
P(T | N) = C(C, T) × C(N-C, R-T) / C(N, R)

여기서 C(n, k)은 이항 계수 "n에서 k 고르기"다

최대가능도: P(T | N)을 가장 크게 하는 N을 찾는다

참 세상에서의 쓰임:
- 야생 개체 수 조사
- 역학(병이 얼마나 퍼졌는지 어림하기)
- 프로그램 시험(벌레 수 어림하기)
- 인구 조사 바로잡기(덜 센 수 어림하기)

지은이: PyTorch 최대가능도 익힘
DATE: 2025
================================================================================
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from typing import Tuple

# ========================================================================
# 메인
# ========================================================================


def compute_hypergeometric_pmf(N: int, C: int, R: int, T: int) -> float:
    """
    초기하 분포로 확률 P(T | N, C, R)을 셈한다.
    
    P(T) = C(C, T) × C(N-C, R-T) / C(N, R)
    
    Parameters:
    -----------
    N : 온 개체 수
    C : 처음에 잡아 표시한 수
    R : Number in recapture sample
    T : Number of marked animals in recapture
    
    Returns:
    --------
    probability : Probability of observing T marked animals
    """
    # 유효성을 확인한다
    if T > C or T > R or R - T > N - C or N < C or N < R:
        return 0.0
    
    try:
        # 초기하 확률질량함수
        numerator = comb(C, T, exact=True) * comb(N - C, R - T, exact=True)
        denominator = comb(N, R, exact=True)
        prob = numerator / denominator
        return prob
    except:
        return 0.0


def compute_log_likelihood(N: int, C: int, R: int, T: int) -> float:
    """개체수 N에 대한 로그가능도를 계산한다"""
    prob = compute_hypergeometric_pmf(N, C, R, T)
    if prob > 0:
        return np.log(prob)
    else:
        return -np.inf


def lincoln_petersen_estimator(C: int, R: int, T: int) -> float:
    """
    Compute the Lincoln-Petersen estimator (MLE approximation).
    
    N̂ = (C × R) / T
    
    This is the MLE for large populations and is very intuitive!
    
    Intuition: If T/R = C/N (proportion marked in sample = proportion in population)
    Then: N = (C × R) / T
    """
    if T == 0:
        return float('inf')  # Can't estimate if no recaptures
    return (C * R) / T


def find_mle_exact(C: int, R: int, T: int, max_N: int = 10000) -> Tuple[int, np.ndarray]:
    """
    Find MLE by computing likelihood for all possible N values.
    
    Returns:
    --------
    N_mle : Most likely population size
    likelihoods : Array of likelihoods for each N
    """
    # 가능한 최소 개체수
    min_N = max(C, R)
    
    # 가능한 N마다 가능도를 계산한다
    N_values = np.arange(min_N, max_N)
    log_likelihoods = np.array([compute_log_likelihood(N, C, R, T) for N in N_values])
    
    # 최댓값을 찾는다
    valid_mask = np.isfinite(log_likelihoods)
    if not np.any(valid_mask):
        return min_N, log_likelihoods
    
    max_idx = np.argmax(log_likelihoods[valid_mask])
    N_mle = N_values[valid_mask][max_idx]
    
    return N_mle, log_likelihoods


def visualize_results(C: int, R: int, T: int, N_true: int, 
                     N_mle: int, N_lp: float, log_likelihoods: np.ndarray):
    """종합적인 시각화를 만든다"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # ========================================================================
    # 그림 1: 가능도 함수
    # ========================================================================
    ax = axes[0, 0]
    
    min_N = max(C, R)
    N_values = np.arange(min_N, min(min_N + len(log_likelihoods), N_true * 3))
    
    # 그림을 그리기 위해 보통의 가능도로 바꾼다
    # 최댓값을 빼서 정규화한다 (수치적 안정성을 위해)
    log_lik_plot = log_likelihoods[:len(N_values) - min_N]
    max_log_lik = np.max(log_lik_plot[np.isfinite(log_lik_plot)])
    likelihood = np.exp(log_lik_plot - max_log_lik)
    
    ax.plot(N_values, likelihood, 'b-', linewidth=2, label='Likelihood')
    ax.axvline(N_mle, color='r', linestyle='-', linewidth=2, label=f'MLE = {N_mle}')
    ax.axvline(N_lp, color='orange', linestyle='--', linewidth=2, 
              label=f'Lincoln-Petersen = {N_lp:.1f}')
    ax.axvline(N_true, color='g', linestyle='--', linewidth=2, label=f'True N = {N_true}')
    
    ax.set_xlabel('Population Size (N)', fontsize=12)
    ax.set_ylabel('Likelihood (normalized)', fontsize=12)
    ax.set_title('Likelihood Function', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ========================================================================
    # 그림 2: 표지-재포획 시각화
    # ========================================================================
    ax = axes[0, 1]
    ax.axis('off')
    
    # 눈으로 볼 수 있는 표현을 만든다
    from matplotlib.patches import Circle, FancyBboxPatch
    
    # 개체군을 그린다
    ax.text(0.5, 0.95, 'Capture-Recapture Process', 
           ha='center', va='top', fontsize=14, fontweight='bold',
           transform=ax.transAxes)
    
    # 1단계: 최초 포획
    box1 = FancyBboxPatch((0.1, 0.65), 0.35, 0.20, 
                          boxstyle="round,pad=0.01", 
                          edgecolor='blue', facecolor='lightblue', linewidth=2)
    ax.add_patch(box1)
    ax.text(0.275, 0.75, f'Step 1: Capture & Mark\n{C} animals marked',
           ha='center', va='center', fontsize=10, transform=ax.transAxes)
    
    # 2단계: 방사
    ax.annotate('', xy=(0.5, 0.75), xytext=(0.46, 0.75),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'),
               transform=ax.transAxes)
    ax.text(0.48, 0.78, 'Release', ha='center', fontsize=9, transform=ax.transAxes)
    
    # 3단계: 재포획
    box2 = FancyBboxPatch((0.55, 0.65), 0.35, 0.20,
                          boxstyle="round,pad=0.01",
                          edgecolor='red', facecolor='lightcoral', linewidth=2)
    ax.add_patch(box2)
    ax.text(0.725, 0.75, f'Step 2: Recapture\n{R} animals caught\n{T} are marked',
           ha='center', va='center', fontsize=10, transform=ax.transAxes)
    
    # 결과
    box3 = FancyBboxPatch((0.2, 0.30), 0.6, 0.25,
                          boxstyle="round,pad=0.02",
                          edgecolor='green', facecolor='lightgreen', linewidth=2)
    ax.add_patch(box3)
    
    results_text = f"""
Observations:
• Initially marked: C = {C}
• Recaptured: R = {R}  
• Marked in recapture: T = {T}

Estimates:
• True population: N = {N_true}
• MLE estimate: N̂ = {N_mle}
• L-P estimate: N̂ = {N_lp:.1f}
• Error: {abs(N_mle - N_true)} animals
"""
    ax.text(0.5, 0.425, results_text, ha='center', va='center',
           fontsize=9, family='monospace', transform=ax.transAxes)
    
    # ========================================================================
    # 그림 3: 오차 분석
    # ========================================================================
    ax = axes[1, 0]
    
    methods = ['True N', 'MLE', 'Lincoln-Petersen']
    values = [N_true, N_mle, N_lp]
    colors = ['green', 'red', 'orange']
    
    bars = ax.barh(methods, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    # 값 이름표를 추가한다
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(val, i, f'  {val:.1f}', va='center', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Population Size', fontsize=12)
    ax.set_title('Method Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # ========================================================================
    # 그림 4: 표본분포 모의실험
    # ========================================================================
    ax = axes[1, 1]
    
    # 표본 변동성을 보이기 위해 여러 번의 조사를 흉내 낸다
    n_simulations = 1000
    estimates = []
    
    for _ in range(n_simulations):
        # T가 초기하분포를 따르는 재포획을 흉내 낸다
        possible_T = np.arange(max(0, R - (N_true - C)), min(R, C) + 1)
        probs = [compute_hypergeometric_pmf(N_true, C, R, t) for t in possible_T]
        probs = np.array(probs)
        probs = probs / probs.sum()  # Normalize
        
        simulated_T = np.random.choice(possible_T, p=probs)
        if simulated_T > 0:
            N_est = lincoln_petersen_estimator(C, R, simulated_T)
            if N_est < 10000:  # Reasonable bound
                estimates.append(N_est)
    
    ax.hist(estimates, bins=50, density=True, alpha=0.7, edgecolor='black',
           label='Sampling distribution')
    ax.axvline(N_true, color='g', linestyle='--', linewidth=2, label=f'True N = {N_true}')
    ax.axvline(N_lp, color='r', linestyle='-', linewidth=2, label=f'Our estimate = {N_lp:.1f}')
    ax.axvline(np.mean(estimates), color='orange', linestyle=':', linewidth=2,
              label=f'Mean of estimates = {np.mean(estimates):.1f}')
    
    ax.set_xlabel('Estimated Population Size', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Sampling Variability (1000 simulations)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('capture_recapture_mle_results.png', dpi=150, bbox_inches='tight')
    print("\n📊 Figure saved as 'capture_recapture_mle_results.png'")
    plt.show()


def main():
    print("=" * 80)
    print("CAPTURE-RECAPTURE MLE - Wildlife Population Estimation")
    print("=" * 80)
    
    # ========================================================================
    # 예: 사슴 개체수 추정
    # ========================================================================
    print("\n🦌 SCENARIO: Estimating Deer Population")
    print("-" * 80)
    
    # 참 개체수 (실제 응용에서는 우리가 모른다)
    N_TRUE = 150
    
    # 조사 매개변수
    C = 30  # Captured and marked in first session
    R = 40  # Captured in second session (recapture)
    T = 8   # Number of marked animals in recapture
    
    print(f"   • Step 1: Captured and marked C = {C} deer")
    print(f"   • Step 2: Recaptured R = {R} deer")
    print(f"   • Observed: T = {T} of them were marked")
    print(f"   • True population: N = {N_TRUE} (unknown in practice)")
    
    # ========================================================================
    # 방법 1: 링컨-피터슨 추정량
    # ========================================================================
    print("\n📐 Method 1: Lincoln-Petersen Estimator")
    print("-" * 80)
    
    N_lp = lincoln_petersen_estimator(C, R, T)
    print(f"   N̂ = (C × R) / T = ({C} × {R}) / {T} = {N_lp:.1f}")
    print(f"   Error: {abs(N_lp - N_TRUE):.1f} animals ({abs(N_lp - N_TRUE)/N_TRUE*100:.1f}%)")
    
    # ========================================================================
    # 방법 2: 정확한 MLE
    # ========================================================================
    print("\n🎯 Method 2: Exact MLE (Hypergeometric)")
    print("-" * 80)
    print("   Computing likelihood for all possible population sizes...")
    
    N_mle, log_likelihoods = find_mle_exact(C, R, T, max_N=500)
    
    print(f"   MLE estimate: N̂ = {N_mle}")
    print(f"   Error: {abs(N_mle - N_TRUE)} animals ({abs(N_mle - N_TRUE)/N_TRUE*100:.1f}%)")
    
    # ========================================================================
    # 비교
    # ========================================================================
    print("\n📊 COMPARISON")
    print("-" * 80)
    print(f"   True population:     N = {N_TRUE}")
    print(f"   Lincoln-Petersen:    N̂ = {N_lp:.1f}")
    print(f"   Exact MLE:           N̂ = {N_mle}")
    print(f"   Difference (L-P vs MLE): {abs(N_lp - N_mle):.1f}")
    
    # ========================================================================
    # 시각화
    # ========================================================================
    print("\n📊 Creating visualizations...")
    visualize_results(C, R, T, N_TRUE, N_mle, N_lp, log_likelihoods)
    
    # ========================================================================
    # 요약
    # ========================================================================
    print("\n" + "=" * 80)
    print("✅ SUMMARY")
    print("=" * 80)
    print("   The capture-recapture method works!")
    print(f"   • We estimated {N_mle} animals")
    print(f"   • True population is {N_TRUE} animals")
    print(f"   • Estimation error: {abs(N_mle - N_TRUE)/N_TRUE*100:.1f}%")
    print("=" * 80)
    
    print("\n💡 KEY INSIGHTS:")
    print("   1. MLE provides population estimates from limited samples")
    print("   2. Lincoln-Petersen ≈ Exact MLE for large populations")
    print("   3. More captures → better estimates")
    print("   4. Assumes: closed population, equal catchability, marks don't fade")
    print("   5. Widely used in ecology, epidemiology, and software testing!")
    print("\n" + "=" * 80)


"""
🎓 EXERCISES:

1. EASY: Try different values of C, R, T
   - What happens if T = 0 (no recaptures)?
   - How does increasing C and R improve accuracy?

2. MEDIUM: Add confidence intervals
   - Use likelihood-based confidence intervals
   - Find N values where likelihood drops by factor of exp(-1.92)

3. MEDIUM: Multiple recapture sessions
   - Extend to 3+ capture sessions
   - Schnabel method for multiple recaptures

4. CHALLENGING: Violations of assumptions
   - Simulate unequal catchability (trap-happy/trap-shy)
   - Population not closed (births, deaths, migration)
   - How robust is MLE to assumption violations?

5. CHALLENGING: Bayesian version
   - Add prior on N (e.g., Uniform or Geometric)
   - Compute posterior distribution
   - Compare Bayesian credible interval to MLE confidence interval
"""


if __name__ == "__main__":
    main()```

## 논의

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다. 그림을 그리는 코드는 학습된 표현, 수렴의 움직임, 평가 지표에 대한 통찰을 주어 추상적인 계산을 손에 잡히게 만든다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 확장된다. 초매개변수, 구조의 변형, 여러 데이터셋을 두루 실험해 보면 이해가 깊어지고 통계적 추론 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
코드를 끝까지 읽고 핵심적인 설계 결정을 찾아내라. 구체적인 구현 선택 세 가지를 나열하고, 각각이 최대가능도 추정에 왜 적절한지 설명하라.

??? success "연습문제 1 풀이"
    설계 결정은 구현마다 다르지만 흔히 다음이 포함된다. (1) 활성화 함수의 선택 — ReLU 계열은 포화되지 않는 경사를 주어 학습을 빠르게 한다. (2) 정규화 전략 — 배치 정규화는 내부 공변량 이동을 줄여 학습을 안정시킨다. (3) 잔차 연결 — 있을 경우 건너뛰는 경로를 제공하여 깊은 신경망에서도 경사가 흐르게 한다. 각 선택은 표현력, 계산 비용, 학습 안정성 사이의 절충을 반영한다.

---

**연습문제 2.**
입력이 기대하는 모양과 자료형을 갖는지 확인하도록 주 함수나 클래스에 입력 검증을 추가하라. 잘못된 입력에는 유익한 오류 메시지를 내라.

??? success "연습문제 2 풀이"
    `forward` 메서드(또는 해당 함수)의 첫머리에 다음과 같은 검사를 추가한다. `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`와 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'`. 모양을 검증할 때는 중요한 차원을 확인한다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 유익한 오류 메시지는 디버깅 속도를 크게 높이고 코드를 재사용하기에도 더 견고하게 만든다.

---

**연습문제 3.**
이 구현에서 생길 수 있는 실패 양상 두 가지를 서술하고, 각각을 어떻게 진단하고 고칠지 설명하라.

??? success "연습문제 3 풀이"
    흔한 실패 양상은 다음과 같다. (1) **경사 소실/폭발** — 경사의 노름을 지켜보아 진단한다(`torch.nn.utils.clip_grad_norm_`을 쓰거나 층마다 `param.grad.norm()`을 기록한다). 경사 자르기, 더 나은 초기화(Xavier/Kaiming), 또는 구조 변경(잔차 연결, 정규화)으로 고친다. (2) **과적합** — 학습 손실은 줄어드는데 검증 손실이 늘어나면 진단된다. 정칙화(드롭아웃, 가중치 감쇠, 데이터 증강)나 모델 용량 축소로 고친다. 이런 문제를 일찍 잡아내려면 언제나 학습 지표와 검증 지표를 함께 살펴라.

---

**연습문제 4.**
표지-재포획 MLE 구현을 검증하는 종합적인 시험 함수를 작성하라. 빈 입력, 원소가 하나뿐인 입력, 아주 큰 입력, 그리고 극단적인 값(0, 아주 큰 수)을 가진 입력 등 경계 사례를 시험하라.

??? success "연습문제 4 풀이"
    경계 조건을 두루 시험하는 함수를 만든다.
    ```python
    def test_capture recapture mle():
        model = Capture Recapture MLE(...)
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
