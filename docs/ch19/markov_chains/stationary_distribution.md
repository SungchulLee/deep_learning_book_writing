# 정상분포

정상분포는 한 번 전이해도 변하지 않는 확률분포다. 고유벡터·선형방정식·행렬 거듭제곱·장기 점유율로 계산하는 방법을 비교하되, 정상분포의 존재·유일성·시점별 수렴은 서로 다른 성질임을 구분한다.

## 1. 코드

??? note "코드 (490줄)"

    ```python
    """
    stationary_distribution.py (단원 04)

    멈춘 분포 살피기
    ==================================

    Location: 06_markov_chain/02_analysis_methods/
    난이도: ⭐⭐⭐ 중급
    걸리는 시간: 4-5시간

    학습 목표:
    - 멈춘 분포 이해하기
    - 여러 방법으로 멈춘 분포 셈하기
    - 있음과 하나뿐임의 조건 살피기
    - 모이는 빠르기 살피기

    수학적 바탕:
    멈춘 분포 π은 다음을 만족하는 확률 분포이다:
    π = π × P

    성질:
    - π은 고윳값이 1인 P의 왼쪽 고유벡터이다
    - 줄일 수 없고 주기가 없는 사슬에서는 멈춘 분포가 오직 하나 있다
    - lim_{n→∞} P^n은 π으로 채운 행으로 모인다
    - 물리로 풀이하면: 오래 보았을 때 상태마다 머문 시간의 비율
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.linalg import eig

    # 무작위로 뽑는 값이 아래에 나온다. 씨앗을 고정해야 이 쪽에 실린
    # 수가 다시 나온다 — 고정하지 않으면 돌릴 때마다 다른 수가 찍힌다
    np.random.seed(0)

    # ========================================================================
    # 메인
    # ========================================================================


    class StationaryDistributionAnalyzer:
        """
        멈춘 분포를 셈하고 살피는 도구.
        """

        def __init__(self, transition_matrix, state_names=None):
            """
            옮김 행렬로 첫값 잡기.

            매개변수:
                transition_matrix (np.ndarray): 옮김 확률 행렬 P
                state_names (list): 상태 이름(없어도 된다)
            """
            self.P = np.array(transition_matrix, dtype=float)
            self.n_states = self.P.shape[0]

            if state_names is None:
                self.state_names = [f"State {i}" for i in range(self.n_states)]
            else:
                self.state_names = state_names

        def compute_via_eigenvector(self):
            """
            고유벡터 방법으로 멈춘 분포 셈하기.

            반환값:
                np.ndarray: 멈춘 분포 π

            수학 방법:
            고윳값 λ = 1인 P의 왼쪽 고유벡터 v 찾기
            π × P = π이므로 π^T = P^T × π^T이다
            따라서 π^T은 고윳값이 1인 P^T의 오른쪽 고유벡터이다
            """
            # P^T의 고윳값과 고유벡터 셈하기
            eigenvalues, eigenvectors = eig(self.P.T)

            # 1에 가장 가까운 고윳값의 번호 찾기
            idx = np.argmin(np.abs(eigenvalues - 1.0))

            # 그에 딸린 고유벡터 뽑아내기
            stationary = np.real(eigenvectors[:, idx])

            # 합이 1이 되도록 고르게 하기(확률 분포)
            stationary = stationary / np.sum(stationary)

            # 성분이 모두 양수가 되도록 하기
            stationary = np.abs(stationary)
            stationary = stationary / np.sum(stationary)

            return stationary

        def compute_via_power_iteration(self, max_iter=1000, tol=1e-10):
            """
            행렬 거듭제곱으로 셈하기: lim_{n→∞} P^n

            반환값:
                tuple: (멈춘 분포, 되풀이 횟수)

            수학의 바탕:
            에르고드 사슬에서 P^n은 모든 행이 같은 행렬로 모이며,
            멈춘 분포 π과 같다
            """
            P_n = self.P.copy()

            for n in range(1, max_iter):
                P_next = P_n @ self.P

                # 모임 살피기
                if np.max(np.abs(P_next - P_n)) < tol:
                    # 멈춘 분포 뽑아내기(아무 행)
                    return P_next[0, :], n

                P_n = P_next

            # 모이지 않았으면 가장 좋은 어림값 돌려주기
            return P_n[0, :], max_iter

        def compute_via_linear_system(self):
            """
            연립 일차 방정식을 풀어 셈하기: π(P - I) = 0, Σπ_i = 1

            반환값:
                np.ndarray: 멈춘 분포

            수학의 차림:
            다음을 풀어야 한다:
            1. π × P = π  ⟹  π × (P - I) = 0
            2. Σ π_i = 1(고르게 하기)

            이는 다음을 푸는 것과 같다:
            π^T × (P^T - I) = 0
            Σ π_i = 1
            """
            # 연립 방정식 세우기: 제약 Σπ_i = 1 아래에서 (P^T - I) × π^T = 0
            A = (self.P.T - np.eye(self.n_states))

            # 마지막 식을 고르게 하기 제약으로 바꾸기
            A[-1, :] = np.ones(self.n_states)
            b = np.zeros(self.n_states)
            b[-1] = 1.0

            # 연립 일차 방정식 풀기
            stationary = np.linalg.solve(A, b)

            return stationary

        def compute_via_simulation(self, n_steps=100000, initial_state=0):
            """
            오래 돌리는 흉내내기로 어림하기.

            매개변수:
                n_steps (int): 흉내내기 걸음 수
                initial_state (int): 시작 상태

            반환값:
                np.ndarray: 어림한 멈춘 분포

            수학으로 뒷받침하기:
            에르고드 정리에 따라 시간 평균은 앙상블 평균과 같다:
            lim_{T→∞} (1/T) Σ I{X_t = j} = π_j
            """
            state_counts = np.zeros(self.n_states)
            current_state = initial_state

            for _ in range(n_steps):
                state_counts[current_state] += 1

                # 다음 상태로 옮기기
                current_state = np.random.choice(
                    self.n_states,
                    p=self.P[current_state, :]
                )

            # 확률을 얻으려고 고르게 하기
            return state_counts / n_steps

        def check_ergodicity(self):
            """
            사슬이 에르고드적인지 살피기(줄일 수 없고 주기가 없음).

            반환값:
                dict: 에르고드성 살피기의 결과

            수학의 조건:
            1. 줄일 수 없음: 어느 상태에서든 어느 상태로도 닿을 수 있다
            2. 주기 없음: 어느 상태로든 돌아오는 시간의 최대공약수가 1이다

            넉넉한 조건: 어떤 P^k의 성분이 모두 양수이다
            """
            results = {
                'is_ergodic': False,
                'is_aperiodic': False,
                'is_irreducible': False
            }

            # P의 어떤 거듭제곱이 모두 양수 성분인지 살피기
            # 이는 줄일 수 없음과 주기 없음을 모두 보장한다
            P_power = self.P.copy()

            for k in range(1, self.n_states + 1):
                if np.all(P_power > 0):
                    results['is_ergodic'] = True
                    results['is_aperiodic'] = True
                    results['is_irreducible'] = True
                    results['power_with_positive_entries'] = k
                    break
                P_power = P_power @ self.P

            return results


    def example_computing_methods():
        """
        보기 1: 멈춘 분포를 셈하는 여러 방법 견주기.
        """
        print("=" * 70)
        print("Example 1: Computing Stationary Distribution - Method Comparison")
        print("=" * 70)

        # 세 상태 사슬
        states = ['A', 'B', 'C']
        P = np.array([
            [0.5, 0.3, 0.2],
            [0.2, 0.6, 0.2],
            [0.3, 0.3, 0.4]
        ])

        print("\nTransition Matrix P:")
        print(P)

        analyzer = StationaryDistributionAnalyzer(P, states)

        # 방법 1: 고유벡터
        print("\n" + "-" * 70)
        print("Method 1: Eigenvector Approach")
        π_eig = analyzer.compute_via_eigenvector()
        print("Stationary distribution:")
        for i, state in enumerate(states):
            print(f"  π({state}) = {π_eig[i]:.8f}")

        # 확인: π × P은 π과 같아야 한다
        verification = π_eig @ P
        print("\nVerification (π × P should equal π):")
        print(f"  Max difference: {np.max(np.abs(verification - π_eig)):.2e}")

        # 방법 2: 거듭제곱 되풀이
        print("\n" + "-" * 70)
        print("Method 2: Matrix Power Iteration")
        π_power, iterations = analyzer.compute_via_power_iteration()
        print(f"Converged in {iterations} iterations")
        print("Stationary distribution:")
        for i, state in enumerate(states):
            print(f"  π({state}) = {π_power[i]:.8f}")

        # 방법 3: 연립 일차 방정식
        print("\n" + "-" * 70)
        print("Method 3: Linear System Solution")
        π_linear = analyzer.compute_via_linear_system()
        print("Stationary distribution:")
        for i, state in enumerate(states):
            print(f"  π({state}) = {π_linear[i]:.8f}")

        # 방법 4: 흉내내기
        print("\n" + "-" * 70)
        print("Method 4: Long-run Simulation (1,000,000 steps)")
        π_sim = analyzer.compute_via_simulation(n_steps=1000000)
        print("Stationary distribution:")
        for i, state in enumerate(states):
            print(f"  π({state}) = {π_sim[i]:.8f}")

        # 모든 방법 견주기
        print("\n" + "-" * 70)
        print("Comparison of All Methods:")
        print(f"{'State':<8} {'Eigenvec':<12} {'Power':<12} {'Linear':<12} {'Simulation':<12}")
        for i, state in enumerate(states):
            print(f"{state:<8} {π_eig[i]:<12.8f} {π_power[i]:<12.8f} "
                  f"{π_linear[i]:<12.8f} {π_sim[i]:<12.8f}")


    def example_interpretation():
        """
        보기 2: 멈춘 분포의 물리 풀이.
        """
        print("\n" + "=" * 70)
        print("Example 2: Physical Interpretation")
        print("=" * 70)

        # 줄 체계: {빔, 손님 1명, 손님 2명}
        states = ['Empty', '1 Customer', '2 Customers']
        P = np.array([
            [0.5, 0.4, 0.1],    # 빔에서: 손님이 올 가능성이 높음
            [0.3, 0.5, 0.2],    # 1에서: 균형
            [0.4, 0.4, 0.2]     # 2에서: 줄어드는 쪽
        ])

        print("\nQueue System Transition Matrix:")
        print(f"{'':15s} {'Empty':>12s} {'1 Customer':>12s} {'2 Customers':>12s}")
        for i, state in enumerate(states):
            row = " ".join(f"{P[i,j]:12.4f}" for j in range(len(states)))
            print(f"{state:15s} {row}")

        analyzer = StationaryDistributionAnalyzer(P, states)
        π = analyzer.compute_via_eigenvector()

        print("\nStationary Distribution (Long-run Proportions):")
        for i, state in enumerate(states):
            print(f"  {state:15s}: π = {π[i]:.6f} ({π[i]*100:.2f}%)")

        print("\nInterpretation:")
        print(f"  In the long run:")
        print(f"  - Queue is empty {π[0]*100:.1f}% of the time")
        print(f"  - Queue has 1 customer {π[1]*100:.1f}% of the time")
        print(f"  - Queue has 2 customers {π[2]*100:.1f}% of the time")

        # 손님 수의 기댓값
        expected_customers = 0*π[0] + 1*π[1] + 2*π[2]
        print(f"\n  Average number of customers in system: {expected_customers:.4f}")


    def example_ergodicity():
        """
        보기 3: 에르고드 조건 살피기.
        """
        print("\n" + "=" * 70)
        print("Example 3: Ergodicity Analysis")
        print("=" * 70)

        # 에르고드 사슬
        print("\nCase 1: Ergodic Chain")
        P_ergodic = np.array([
            [0.5, 0.3, 0.2],
            [0.2, 0.6, 0.2],
            [0.3, 0.3, 0.4]
        ])

        analyzer1 = StationaryDistributionAnalyzer(P_ergodic)
        results1 = analyzer1.check_ergodicity()

        print(f"  Is ergodic: {results1['is_ergodic']}")
        if results1['is_ergodic']:
            print(f"  P^{results1['power_with_positive_entries']} has all positive entries")
            print("  ⟹ Stationary distribution exists and is unique")
            π = analyzer1.compute_via_eigenvector()
            print(f"  Stationary: {π}")

        # 주기 사슬
        print("\nCase 2: Periodic Chain")
        P_periodic = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0]
        ])

        analyzer2 = StationaryDistributionAnalyzer(P_periodic)
        results2 = analyzer2.check_ergodicity()

        print(f"  Is ergodic: {results2['is_ergodic']}")
        print("  This chain cycles: A → B → C → A")
        print("  Stationary distribution exists but convergence doesn't occur")
        π2 = analyzer2.compute_via_eigenvector()
        print(f"  Stationary: {π2}")

        # 줄일 수 있는 사슬
        print("\nCase 3: Reducible Chain (Two Components)")
        P_reducible = np.array([
            [0.5, 0.5, 0, 0],
            [0.5, 0.5, 0, 0],
            [0, 0, 0.7, 0.3],
            [0, 0, 0.3, 0.7]
        ])

        analyzer3 = StationaryDistributionAnalyzer(P_reducible)
        results3 = analyzer3.check_ergodicity()

        print(f"  Is ergodic: {results3['is_ergodic']}")
        print("  Two separate components: {0,1} and {2,3}")
        print("  Stationary distribution depends on initial state")


    def visualize_convergence():
        """
        멈춘 분포로 모이는 것 그려 보기.
        """
        print("\n" + "=" * 70)
        print("Creating Convergence Visualization")
        print("=" * 70)

        states = ['A', 'B', 'C']
        P = np.array([
            [0.5, 0.3, 0.2],
            [0.2, 0.6, 0.2],
            [0.3, 0.3, 0.4]
        ])

        analyzer = StationaryDistributionAnalyzer(P, states)
        π_stationary = analyzer.compute_via_eigenvector()

        # 서로 다른 첫 분포에서의 모임 기록하기
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 서로 다른 첫 분포
        initial_dists = [
            np.array([1.0, 0.0, 0.0]),  # A에서 시작
            np.array([0.0, 1.0, 0.0]),  # B에서 시작
            np.array([0.0, 0.0, 1.0]),  # C에서 시작
            np.array([1/3, 1/3, 1/3])   # 고름
        ]

        labels = ['Start at A', 'Start at B', 'Start at C', 'Uniform']
        colors = ['red', 'blue', 'green', 'purple']

        # 그림 1: 멈춘 분포까지의 거리
        ax = axes[0]

        for init_dist, label, color in zip(initial_dists, labels, colors):
            distances = []
            dist = init_dist.copy()

            for n in range(50):
                # 멈춘 분포까지의 거리 셈하기
                distance = np.linalg.norm(dist - π_stationary)
                distances.append(distance)

                # 분포 새로 고치기
                dist = dist @ P

            ax.semilogy(distances, label=label, color=color, linewidth=2)

        ax.set_xlabel('Step n', fontsize=12)
        ax.set_ylabel('||π_n - π*|| (log scale)', fontsize=12)
        ax.set_title('Convergence to Stationary Distribution', fontsize=13)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 그림 2: 상태 확률의 흘러감
        ax = axes[1]

        init_dist = np.array([1.0, 0.0, 0.0])  # A에서 시작
        steps = range(51)
        state_probs = {state: [] for state in states}

        dist = init_dist.copy()
        for n in steps:
            for i, state in enumerate(states):
                state_probs[state].append(dist[i])
            dist = dist @ P

        for i, state in enumerate(states):
            ax.plot(steps, state_probs[state], marker='o', markersize=3,
                   label=state, linewidth=2)
            # 멈춘 분포 선 더하기
            ax.axhline(y=π_stationary[i], linestyle='--', color=f'C{i}', alpha=0.5)

        ax.set_xlabel('Step n', fontsize=12)
        ax.set_ylabel('Probability', fontsize=12)
        ax.set_title('State Probabilities Over Time (Starting from A)', fontsize=13)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('stationary_convergence.png', 
                    dpi=150, bbox_inches='tight')
        plt.close()
        print("Convergence visualization saved")


    def main():
        """
        보기 모두 돌리기.
        """
        print("STATIONARY DISTRIBUTION ANALYSIS")
        print("=================================\n")

        example_computing_methods()
        example_interpretation()
        example_ergodicity()
        visualize_convergence()

        print("\n" + "=" * 70)
        print("Key Theoretical Results:")
        print("=" * 70)
        print("1. Stationary distribution satisfies: π = π × P")
        print("2. For ergodic chains: unique stationary distribution exists")
        print("3. Ergodic = irreducible + aperiodic")
        print("4. P^n converges to π for ergodic chains")
        print("5. Long-run proportion in state j equals π_j")


    if __name__ == "__main__":
        main()
    ```


??? note "전체 출력 (108줄)"

    ```
    STATIONARY DISTRIBUTION ANALYSIS
    =================================

    ======================================================================
    Example 1: Computing Stationary Distribution - Method Comparison
    ======================================================================

    Transition Matrix P:
    [[0.5 0.3 0.2]
     [0.2 0.6 0.2]
     [0.3 0.3 0.4]]

    ----------------------------------------------------------------------
    Method 1: Eigenvector Approach
    Stationary distribution:
      π(A) = 0.32142857
      π(B) = 0.42857143
      π(C) = 0.25000000

    Verification (π × P should equal π):
      Max difference: 5.55e-17

    ----------------------------------------------------------------------
    Method 2: Matrix Power Iteration
    Converged in 19 iterations
    Stationary distribution:
      π(A) = 0.32142857
      π(B) = 0.42857143
      π(C) = 0.25000000

    ----------------------------------------------------------------------
    Method 3: Linear System Solution
    Stationary distribution:
      π(A) = 0.32142857
      π(B) = 0.42857143
      π(C) = 0.25000000

    ----------------------------------------------------------------------
    Method 4: Long-run Simulation (1,000,000 steps)
    Stationary distribution:
      π(A) = 0.32097100
      π(B) = 0.42841900
      π(C) = 0.25061000

    ----------------------------------------------------------------------
    Comparison of All Methods:
    State    Eigenvec     Power        Linear       Simulation  
    A        0.32142857   0.32142857   0.32142857   0.32097100  
    B        0.42857143   0.42857143   0.42857143   0.42841900  
    C        0.25000000   0.25000000   0.25000000   0.25061000  

    ======================================================================
    Example 2: Physical Interpretation
    ======================================================================

    Queue System Transition Matrix:
                           Empty   1 Customer  2 Customers
    Empty                 0.5000       0.4000       0.1000
    1 Customer            0.3000       0.5000       0.2000
    2 Customers           0.4000       0.4000       0.2000

    Stationary Distribution (Long-run Proportions):
      Empty          : π = 0.395062 (39.51%)
      1 Customer     : π = 0.444444 (44.44%)
      2 Customers    : π = 0.160494 (16.05%)

    Interpretation:
      In the long run:
      - Queue is empty 39.5% of the time
      - Queue has 1 customer 44.4% of the time
      - Queue has 2 customers 16.0% of the time

      Average number of customers in system: 0.7654

    ======================================================================
    Example 3: Ergodicity Analysis
    ======================================================================

    Case 1: Ergodic Chain
      Is ergodic: True
      P^1 has all positive entries
      ⟹ Stationary distribution exists and is unique
      Stationary: [0.32142857 0.42857143 0.25      ]

    Case 2: Periodic Chain
      Is ergodic: False
      This chain cycles: A → B → C → A
      Stationary distribution exists but convergence doesn't occur
      Stationary: [0.33333333 0.33333333 0.33333333]

    Case 3: Reducible Chain (Two Components)
      Is ergodic: False
      Two separate components: {0,1} and {2,3}
      Stationary distribution depends on initial state

    ======================================================================
    Creating Convergence Visualization
    ======================================================================
    Convergence visualization saved

    ======================================================================
    Key Theoretical Results:
    ======================================================================
    1. Stationary distribution satisfies: π = π × P
    2. For ergodic chains: unique stationary distribution exists
    3. Ergodic = irreducible + aperiodic
    4. P^n converges to π for ergodic chains
    5. Long-run proportion in state j equals π_j
    ```


## 2. 논의

### 정의와 존재 조건

행벡터 π가 다음을 만족하면 정상분포다.

$$\pi P=\pi,\qquad\pi_i\geq0,\quad\sum_i\pi_i=1$$

모든 유한 확률행렬에는 적어도 하나가 존재한다. 기약이면 유일하며, 기약성에 비주기성까지 있으면 모든 초기분포의 p₀Pⁿ이 π로 수렴한다. 기약성은 유일성의 충분조건이지 필요조건은 아니다. 과도상태와 닫힌 클래스 하나만 있는 가약 사슬도 유일한 정상분포를 가질 수 있다.

### 계산 방법의 차이

Pᵀ의 고유값 1인 고유벡터를 정규화하거나, (Pᵀ−I)πᵀ=0과 합 1 조건을 풀 수 있다. 유효한 유한 확률행렬의 정확한 불변 벡터 v라면 |v|도 불변이다. 비음성에 따른 |v|P≥|vP|와 합 보존에서 등호가 나오기 때문이다. 따라서 절댓값 정규화 자체가 항상 잘못된 것은 아니다. 다만 근사 고유벡터나 잘못된 행렬의 오류를 자동 복구하지 않으며, 코드처럼 먼저 성분 합으로 나누면 합이 0인 부호 혼합 벡터에서 실패할 수 있다. 정상분포가 여러 개이면 선택된 해가 유일한 장기분포라는 뜻도 아니다. 선형계가 특이한지와 최종 비음성·합·정상성 잔차를 확인한다.

행렬 거듭제곱 방법은 수렴 조건이 필요하다. 주기 사슬은 정상분포가 있어도 거듭제곱이 진동할 수 있으며, 코드가 최대 반복까지 갔을 때도 첫 행을 반환하므로 반환값이 수렴한 정상분포라고 단정하지 않는다. 인접 거듭제곱 차이가 작다는 것만으로 극한 오차를 보장할 수도 없다.

장기 시뮬레이션은 방문 빈도를 계산한다. 유한 기약 사슬의 시간 평균은 주기성이 있어도 π와 연결되지만, 가약 사슬의 한 경로는 도달한 닫힌 클래스의 빈도를 보인다. 유한 표본은 상관되므로 독립 표본의 표준오차를 그대로 적용하지 않는다.

### 진단 함수의 한계

check_ergodicity는 P¹부터 P^N까지만 원소가 모두 양수인 거듭제곱을 찾는다. 하나를 찾으면 유한 사슬의 기약·비주기성의 충분한 증거지만, 못 찾았다고 두 성질이 거짓인 것은 아니다. 필요한 양의 거듭제곱 지수는 상태 수보다 클 수 있다. 강연결성과 주기를 별도로 검사하는 정확한 구조 진단과 구분한다. 또한 생성자에는 전이행렬 유효성 검사가 없으므로 입력 확률 조건을 먼저 확인해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
유한 사슬의 정상분포 존재·유일성·시점별 수렴 조건을 구분하라.

</div>

??? success "연습문제 1 풀이"

    유한 확률행렬이면 정상분포가 존재한다. 기약성은 유일성을 보장하고 기약·비주기성은 모든 초기분포의 수렴을 보장한다. 주기적인 두 상태 교대 사슬은 정상분포가 유일하지만 한 상태에서 시작한 시점분포는 진동한다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
정확한 불변 벡터의 절댓값 정규화와 근사 계산의 검증을 구분하라.

</div>

??? success "연습문제 2 풀이"
    유효한 유한 확률행렬에서 vP=v인 0이 아닌 실수 벡터라면 |v|도 불변이므로 정규화해 정상분포를 얻을 수 있다. 하지만 근사 벡터의 절댓값이 정상성 오차를 없애지는 않는다. 특히 성분 합으로 먼저 나누는 구현은 합이 0일 때 실패할 수 있고, 여러 해 중 하나를 얻었다고 유일성이나 모든 초기분포의 수렴을 증명한 것은 아니다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
P¹,…,P^N에 모두 양수인 행렬이 없으면 비주기성이 부정되는가?

</div>

??? success "연습문제 3 풀이"

    아니다. 유한 기약·비주기 행렬은 어떤 유한 거듭제곱에서 모두 양수가 되지만 그 지수가 상태 수 N 이하라는 보장은 없다. 코드의 미발견 결과를 확정적인 구조 판정으로 읽어서는 안 된다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
아래 신경망 시험 조각을 정상분포 분석에 적용할 수 있는지 평가하라.

</div>

??? success "연습문제 4 풀이"

    이 분석기는 학습 가능한 신경망 인터페이스를 갖지 않으므로 아래 backward·parameters 검사는 해당하지 않는다. 정상성 잔차·비음성·합 1, 알려진 주기 사슬과 여러 닫힌 클래스, 실패·미수렴 표시를 점검하는 것이 적합하다. 유한 시뮬레이션 빈도와 해석적 값의 차이는 표본 상관도 고려해야 한다.

    ```python
    def test_stationarydistributionanalyzer():
        model = StationaryDistributionAnalyzer(...)
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


## 정리하며

정상분포의 정의는 πP=π와 확률 제약이다. 고유벡터·선형계 계산과 분포 수렴·시간 평균은 요구 조건이 다르다. 수치 결과의 잔차와 구조를 함께 확인하고 양의 거듭제곱을 짧게 탐색한 결과나 미수렴 반환값을 확정적 진단으로 해석하지 않는다.
