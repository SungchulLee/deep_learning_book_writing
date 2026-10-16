# 표지-재포획 MLE

두 번의 포획 자료로 전체 개체수 $N$을 추정한다. 정수 파라미터의 우도를 직접 비교하며 MLE가 유일하지 않을 수 있다는 점을 살펴본다.

## 1. 손으로 푸는 예제

최대가능도 추정을 뉴럴 넷트웍에 쓰기 전에, 손으로 끝까지 풀 수 있는 가장 작은 예제로 원리를 먼저 보자. 이 예제가 좋은 이유는 파라메터가 하나뿐인 데다 정수여서, 미분 대신 이웃한 값끼리 비교하는 것만으로 최댓값을 찾을 수 있기 때문이다.

호수에 물고기가 $N$마리 있다. $N$을 알고 싶지만 전부 잡아 셀 수는 없다.

1. **1단계(포획)** — 3마리를 잡아 표지를 붙이고 놓아준다.
2. **2단계(재포획)** — 얼마 뒤 5마리를 잡았더니 그중 1마리에 표지가 있었다.

두 번째 표본에서 표지된 물고기가 정확히 1마리일 확률은 초기하분포로 주어진다.

$$
P_N = \frac{\binom{3}{1}\binom{N-3}{4}}{\binom{N}{5}}
$$

여기서 $N$은 데이터가 아니라 **파라메터**이고, $P_N$은 그 파라메터의 함수인 **우도**다. 최대가능도 추정은 관찰한 결과(표지 1마리)를 가장 그럴듯하게 만드는 $N$을 고르는 것이다.

### 이웃과 비교하여 최댓값 찾기

$N$이 정수이므로 미분할 수 없다. 대신 $P_N$이 커지다 작아지는 지점, 곧 $P_{N-1} < P_N$이면서 $P_{N+1} \le P_N$인 $N$을 찾는다.

먼저 $P_{N-1} < P_N$을 정리한다. 공통 인수 $\binom{3}{1}$은 양변에서 지워진다.

$$
\frac{\binom{N-4}{4}}{\binom{N-1}{5}} < \frac{\binom{N-3}{4}}{\binom{N}{5}}
\;\Longrightarrow\;
N(N-7) < (N-5)(N-3)
$$

$$
N^2 - 7N < N^2 - 8N + 15 \;\Longrightarrow\; N < 15
$$

다음으로 $P_{N+1} \le P_N$을 정리한다.

$$
\frac{\binom{N-2}{4}}{\binom{N+1}{5}} \le \frac{\binom{N-3}{4}}{\binom{N}{5}}
\;\Longrightarrow\;
(N+1)(N-6) \ge (N-4)(N-2)
$$

$$
N^2 - 5N - 6 \ge N^2 - 6N + 8 \;\Longrightarrow\; N \ge 14
$$

두 조건을 함께 만족하는 정수는 $N = 14$뿐이지만, 부등식의 등호 자리를 따져 보면 $N = 15$에서도 우도가 같은 값을 갖는다. 실제로 계산하면 다음과 같다.

$$
P_{14} = P_{15} = \frac{45}{91} \approx 0.4945
$$

$$
\therefore \; \widehat{N}_{\text{MLE}} \in \{14,\ 15\}
$$

곧 최대가능도 추정값이 하나로 정해지지 않고 두 값에서 비긴다. $\square$

이 비김은 이 예제의 결함이 아니라 최대가능도 추정의 성질을 보여 준다. 우도 함수는 봉우리 근처에서 평평할 수 있고, 그때는 데이터가 두 값을 갈라놓을 만큼의 정보를 담고 있지 않다는 뜻이다. 추가 표본은 후보를 구분하는 정보를 늘릴 수 있지만, 동률이 반드시 사라진다는 보장은 없다.

### 뉴럴 넷트웍과 어떻게 이어지는가

같은 논리가 그대로 뉴럴 넷트웍으로 넘어간다. 다른 것은 규모와 푸는 방법뿐이다.

| | 표지-재포획 | MNIST 선형 분류기 |
|---|---|---|
| 파라메터 | $N$ 하나 (정수) | $(A, b)$, $784 \times 10 + 10 = 7850$개 (실수) |
| 우도 | $P_N$ | $L(A, b)$ |
| 푸는 법 | 이웃과 비교 | 로그를 취해 경사 하강법 |

고차원 연속 파라미터에서는 후보를 일일이 비교하기 어렵다. 미분 가능한 목적의 경사를 이용하는 방법을 사용할 수 있다. 우도에서 로그우도를 거쳐 로스와 경사 하강법으로 이어지는 사슬 전체는 [3.2 선형 모델과 소프트맥스](../../ch03/linear_softmax/index.md)에서 MNIST를 예로 다룬다.

---

## 2. 코드

??? note "코드 (402줄)"

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

    수학 모델:
    다시 잡은 것 가운데 표시된 수는 초기하 분포를 따른다.
    P(T | N) = C(C, T) × C(N-C, R-T) / C(N, R)

    여기서 C(n, k)은 이항 계수 "n에서 k 고르기"다

    최대가능도: P(T | N)을 가장 크게 하는 N을 찾는다

    참 세상에서의 쓰임:
    - 야생 개체 수 조사
    - 역학(병이 얼마나 퍼졌는지 어림하기)
    - 프로그램 시험(벌레 수 어림하기)
    - 인구 조사 바로잡기(덜 센 수 어림하기)

    지은이: PyTorch 최대가능도 학습
    DATE: 2025
    ================================================================================
    """

    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.special import comb
    from typing import Tuple
    # 무작위로 뽑는 값이 아래에 나온다. 씨앗을 고정해야 이 쪽에 실린
    # 수가 다시 나온다 — 고정하지 않으면 돌릴 때마다 다른 수가 찍힌다
    torch.manual_seed(0)
    np.random.seed(0)

    # ========================================================================
    # 메인
    # ========================================================================


    def compute_hypergeometric_pmf(N: int, C: int, R: int, T: int) -> float:
        """
        초기하 분포로 확률 P(T | N, C, R)을 계산한다.

        P(T) = C(C, T) × C(N-C, R-T) / C(N, R)

        Parameters:
        -----------
        N : 온 개체 수
        C : 처음에 잡아 표시한 수
        R : 다시 잡은 표본의 수
        T : 다시 잡은 것 가운데 표시된 수

        Returns:
        --------
        probability : 표시된 T마리를 볼 확률
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
        링컨-피터슨 어림값(최대가능도 어림)을 계산한다.

        N̂ = (C × R) / T

        개체 수가 많을 때의 최대가능도이며 느낌으로도 잘 잡힌다!

        느낌: T/R = C/N이면(표본에서 표시된 몫 = 온 무리에서 표시된 몫)
        Then: N = (C × R) / T
        """
        if T == 0:
            return float('inf')  # Can't estimate if no recaptures
        return (C * R) / T


    def find_mle_exact(C: int, R: int, T: int, max_N: int = 10000) -> Tuple[int, np.ndarray]:
        """
        나올 수 있는 N마다 가능도를 셈해 최대가능도를 찾는다.

        Returns:
        --------
        N_mle : 가장 그럴듯한 개체 수
        likelihoods : N마다의 가능도 배열
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
        # N_values는 min_N에서 시작하는 arange이므로 그 길이만큼 자르면 된다.
        # 길이에서 min_N을 빼면 색인과 길이를 뒤섞는 셈이 되어 짝이 어긋난다.
        log_lik_plot = log_likelihoods[:len(N_values)]
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
    • 처음에 표시한 수: C = {C}
    • Recaptured: R = {R}  
    • 다시 잡은 것 가운데 표시된 수: T = {T}

    Estimates:
    • 참 개체 수: N = {N_true}
    • 최대가능도 어림: N̂ = {N_mle}
    • 링컨-피터슨 어림: N̂ = {N_lp:.1f}
    • 오차: {abs(N_mle - N_true)}마리
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

        # 값 레이블을 추가한다
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

        # 조사 파라메터
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

    1. 쉬움: C, R, T 값을 바꾸어 보아라
       - T = 0이면(다시 잡힌 표시가 없으면) 어떻게 되는가?
       - C과 R을 키우면 정확도이 어떻게 나아지는가?

    2. 보통: 믿음 구간을 더하여라
       - 가능도 바탕 믿음 구간을 쓴다
       - 가능도가 exp(-1.92)배로 떨어지는 N 값을 찾는다

    3. 보통: 여러 번 다시 잡기
       - 잡기를 3번 넘게로 넓힌다
       - 여러 번 다시 잡을 때의 슈나벨 방법

    4. 어려움: 가정이 깨질 때
       - 잡히는 정도가 다른 경우를 흉내낸다(덫을 좋아하거나 꺼리는 개체)
       - 무리가 닫혀 있지 않을 때(태어남, 죽음, 옮겨감)
       - 가정이 깨질 때 최대가능도는 얼마나 든든한가?

    5. 어려움: 베이즈 클래스
       - N에 앞확률을 둔다(보기: 고른 분포나 기하 분포)
       - 뒤확률 분포를 계산한다
       - 베이즈 믿음 구간과 최대가능도 믿음 구간을 견준다
    """


    if __name__ == "__main__":
        main()
    ```


**출력:**

```
================================================================================
CAPTURE-RECAPTURE MLE - Wildlife Population Estimation
================================================================================

🦌 SCENARIO: Estimating Deer Population
--------------------------------------------------------------------------------
   • Step 1: Captured and marked C = 30 deer
   • Step 2: Recaptured R = 40 deer
   • Observed: T = 8 of them were marked
   • True population: N = 150 (unknown in practice)

📐 Method 1: Lincoln-Petersen Estimator
--------------------------------------------------------------------------------
   N̂ = (C × R) / T = (30 × 40) / 8 = 150.0
   Error: 0.0 animals (0.0%)

🎯 Method 2: Exact MLE (Hypergeometric)
--------------------------------------------------------------------------------
   Computing likelihood for all possible population sizes...
   MLE estimate: N̂ = 149
   Error: 1 animals (0.7%)

📊 COMPARISON
--------------------------------------------------------------------------------
   True population:     N = 150
   Lincoln-Petersen:    N̂ = 150.0
   Exact MLE:           N̂ = 149
   Difference (L-P vs MLE): 1.0

📊 Creating visualizations...

📊 Figure saved as 'capture_recapture_mle_results.png'

================================================================================
✅ SUMMARY
================================================================================
   The capture-recapture method works!
   • We estimated 149 animals
   • True population is 150 animals
   • Estimation error: 0.7%
================================================================================

💡 KEY INSIGHTS:
   1. MLE provides population estimates from limited samples
   2. Lincoln-Petersen ≈ Exact MLE for large populations
   3. More captures → better estimates
   4. Assumes: closed population, equal catchability, marks don't fade
   5. Widely used in ecology, epidemiology, and software testing!

================================================================================
```

## 3. 논의

### 초기하분포가 성립하는 조건

첫 포획 수를 $C$, 재포획 수를 $R$, 재포획 중 표지 개체 수를 $T$라 하면 우도는 $\binom CT\binom{N-C}{R-T}/\binom NR$이다. 표본 안에서는 비복원 추출을 가정하며, 두 조사 사이 개체군의 유입·유출이 없고 표지가 사라지지 않으며 각 개체의 포획 가능성이 같아야 한다. 두 조사 사이에는 표지 개체가 충분히 섞였다고 가정한다.

### 검색 범위와 비교 기준

가능한 관측에는 정수 $C,R,T$가 필요하며 $0\le T\le\min(C,R)$, $N\ge C+R-T$를 만족해야 한다. 코드는 유한한 후보 범위에서 로그우도를 비교하므로, 상한이 너무 작으면 전체 공간의 MLE가 아닌 범위 내 최댓값을 반환할 수 있다. $T=0$이면 우도가 $N$ 증가에 따라 상한으로 접근해 유한한 MLE가 없을 수 있다.

Lincoln–Petersen 값 $CR/T$는 표본 비율을 맞춘 추정값이다. 정수 MLE와 개념을 구분하고, 우도가 같은 두 후보가 있는지도 확인한다. `argmax`는 동률 후보 중 하나만 반환할 수 있다.

### 그림과 모의실험 읽기

우도 곡선은 어느 개체수가 관측을 더 잘 설명하는지 보여준다. 시뮬레이션은 가정한 모집단과 표본 크기에서 추정량의 변동을 살펴보는 자료이며, 하나의 포획 결과만으로 추정의 정확성을 보장하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
표지-재포획의 초기하분포 우도를 정당화하는 가정 세 가지를 설명하라.

</div>

??? success "연습문제 1 풀이"
    조사 사이 모집단이 닫혀 있고, 표지가 사라지지 않으며, 재포획에서 개체들의 포획 가능성이 같다고 가정한다. 표지 후 충분한 혼합도 필요하다. 위반되면 우도를 잘 최대화해도 추정이 바이어스될 수 있다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
관측 $C,R,T$와 후보 개체수 $N$에 어떤 입력 조건이 필요한가?

</div>

??? success "연습문제 2 풀이"
    모두 정수이고 포획 수는 음수가 아니어야 한다. $0\le T\le\min(C,R)$, $N\ge C+R-T$가 필요하다. 표본 수가 0인 경우에는 정보가 사라질 수 있고 $T=0$에서는 유한한 MLE가 없을 수 있으므로 별도로 처리한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
유한 범위 검색과 `argmax`가 MLE 해석에 만드는 주의점 두 가지를 설명하라.

</div>

??? success "연습문제 3 풀이"
    검색 상한이 작으면 참 최대점을 제외할 수 있다. 또한 동률이 있으면 `argmax`가 첫 후보만 반환하므로 최대 우도에 가까운 모든 후보를 확인해야 한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
표지-재포획 MLE 구현을 검증하는 종합적인 시험 함수를 작성하라. 빈 입력, 원소가 하나뿐인 입력, 아주 큰 입력, 그리고 극단적인 값(0, 아주 큰 수)을 가진 입력 등 경계 사례를 시험하라.

</div>

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

## 정리하며

정수 우도의 최대화에는 후보 범위와 동률 처리가 중요하다. 개체군과 포획 확률에 대한 가정이 맞는지도 추정식과 함께 확인한다.
