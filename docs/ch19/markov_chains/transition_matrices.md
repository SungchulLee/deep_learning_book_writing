# 전이 행렬

전이행렬의 거듭제곱은 여러 단계 뒤의 전이확률을 나타낸다. 행벡터 분포를 사용하는 규약에서 확률 질량의 변화를 계산하고, Chapman–Kolmogorov 관계와 행렬 수렴 진단의 의미를 살펴본다. 단순한 인접 거듭제곱 차이만으로 정상분포의 유일성이나 혼합 시간을 보장할 수는 없다.

## 1. 코드

??? note "코드 (480줄)"

    ```python
    """
    transition_matrices.py (단원 02)

    옮김 행렬과 상태 확률 다루기
    =========================================================

    Location: 06_markov_chain/01_fundamentals/
    난이도: ⭐⭐ 기초
    걸리는 시간: 3-4시간

    학습 목표:
    - 옮김 행렬을 수학으로 이해하기
    - 여러 걸음 옮김 확률 셈하기
    - 행렬 거듭제곱 P^n 살피기
    - 시간 n에서의 상태 확률 셈하기

    수학적 바탕:
    - 옮김 행렬 P: P[i][j] = P(X_{n+1} = j | X_n = i)
    - 채프먼-콜모고로프 방정식: P^(n) = P^n(행렬 거듭제곱)
    - n걸음 옮김 확률: P^(n)[i][j] = P(X_n = j | X_0 = i)
    - 상태 분포의 흘러감: π_n = π_0 × P^n
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from numpy.linalg import matrix_power   # scipy.linalg에는 이 함수가 없다

    # ========================================================================
    # 메인
    # ========================================================================


    class TransitionMatrixAnalyzer:
        """
        옮김 행렬을 살피고 셈하는 도구.

        수학의 성질:
        1. 확률 행렬: 행의 합이 1
        2. 채프먼-콜모고로프: P^(m+n) = P^m × P^n
        3. 거듭제곱의 모임: lim_{n→∞} P^n이 있을 수 있다
        """

        def __init__(self, transition_matrix, state_names=None):
            """
            옮김 행렬로 분석기 첫값 잡기.

            파라메터:
                transition_matrix (np.ndarray or list): 옮김 확률 행렬
                state_names (list): 상태의 이름(없어도 된다)
            """
            self.P = np.array(transition_matrix, dtype=float)
            self.n_states = self.P.shape[0]

            if state_names is None:
                self.state_names = [f"State {i}" for i in range(self.n_states)]
            else:
                self.state_names = state_names

            # 검증
            self._validate_matrix()

        def _validate_matrix(self):
            """
            그 행렬이 제대로 된 확률 행렬인지 확인하기.

            필요한 것:
            1. 정사각 행렬
            2. 성분이 모두 [0, 1] 안에 있다
            3. 행마다 합이 1이다
            """
            # 정사각인지 살피기
            if self.P.shape[0] != self.P.shape[1]:
                raise ValueError("Transition matrix must be square")

            # 음이 아니고 1 이하인지 살피기
            if np.any(self.P < 0) or np.any(self.P > 1):
                raise ValueError("All probabilities must be in [0, 1]")

            # 행의 합 살피기
            row_sums = np.sum(self.P, axis=1)
            if not np.allclose(row_sums, 1.0):
                raise ValueError(f"Row sums must equal 1. Got: {row_sums}")

        def n_step_transition_matrix(self, n):
            """
            n걸음 옮김 행렬 P^n 셈하기.

            파라메터:
                n (int): 걸음 수

            반환값:
                np.ndarray: P^n[i][j] = P(X_n = j | X_0 = i)인 P^n

            수학적 바탕:
            채프먼-콜모고로프 방정식에 따라:
            P^(n)[i][j] = Σ_k P^(m)[i][k] × P^(n-m)[k][j]

            이는 행렬 곱과 같다: P^n = P × P × ... × P(n번)
            """
            if n < 0:
                raise ValueError("n must be non-negative")
            if n == 0:
                return np.eye(self.n_states)  # 항등 행렬

            # scipy의 최적화된 행렬 거듭제곱 쓰기
            return matrix_power(self.P, n)

        def probability_after_n_steps(self, initial_state, target_state, n):
            """
            initial_state에서 n걸음 뒤에 target_state에 있을 확률 셈하기.

            파라메터:
                initial_state (int or str): 시작 상태
                target_state (int or str): 타깃 상태
                n (int): 걸음 수

            반환값:
                float: P(X_n = target | X_0 = initial)

            수학 공식:
            P(X_n = j | X_0 = i) = [P^n]_{i,j}
            """
            # 필요하면 상태 이름을 번호로 바꾸기
            if isinstance(initial_state, str):
                i = self.state_names.index(initial_state)
            else:
                i = initial_state

            if isinstance(target_state, str):
                j = self.state_names.index(target_state)
            else:
                j = target_state

            # P^n을 셈하고 확률 뽑아내기
            P_n = self.n_step_transition_matrix(n)
            return P_n[i, j]

        def state_distribution_after_n_steps(self, initial_distribution, n):
            """
            n걸음 뒤의 상태 확률 분포 셈하기.

            파라메터:
                initial_distribution (np.ndarray): 첫 확률 분포 π_0
                n (int): 걸음 수

            반환값:
                np.ndarray: π_n = π_0 × P^n인 분포 π_n

            수학적 바탕:
            π_0이 첫 상태 확률을 나타내는 행벡터이면,
            n걸음 뒤의 분포는 다음과 같다:
            π_n = π_0 × P^n

            성분별로: π_n[j] = Σ_i π_0[i] × P^n[i][j]
            """
            initial_distribution = np.array(initial_distribution)

            # 첫 분포 확인하기
            if not np.isclose(np.sum(initial_distribution), 1.0):
                raise ValueError("Initial distribution must sum to 1")

            # P^n 셈하기
            P_n = self.n_step_transition_matrix(n)

            # 행렬 곱: π_n = π_0 × P^n
            return initial_distribution @ P_n

        def analyze_convergence(self, max_steps=100, tolerance=1e-6):
            """
            n → ∞일 때 P^n이 모이는지 살피기.

            파라메터:
                max_steps (int): 살필 최대 걸음 수
                tolerance (float): 모임 너그러움

            반환값:
                dict: 모임 상태와 끝값을 담은 분석 결과

            수학 메모:
            규칙 마르코프 사슬에서는(P의 어떤 거듭제곱의 성분이 모두 양수이면)
            P^n은 모든 행이 같은 행렬로 모이며,
            멈춘 분포를 나타낸다.
            """
            results = {
                'converged': False,
                'convergence_step': None,
                'limit_matrix': None,
                'differences': []
            }

            P_prev = self.P.copy()

            for step in range(1, max_steps + 1):
                P_current = self.P @ P_prev  # P^(n+1) = P × P^n

                # 잇따른 거듭제곱 사이의 최대 차이 셈하기
                diff = np.max(np.abs(P_current - P_prev))
                results['differences'].append(diff)

                # 모임 살피기
                if diff < tolerance:
                    results['converged'] = True
                    results['convergence_step'] = step
                    results['limit_matrix'] = P_current
                    break

                P_prev = P_current

            return results

        def visualize_n_step_probabilities(self, initial_state, max_steps=50):
            """
            n걸음 동안 확률이 어떻게 흘러가는지 그려 보기.

            파라메터:
                initial_state (int or str): 시작 상태
                max_steps (int): 그려 볼 최대 걸음 수
            """
            if isinstance(initial_state, str):
                i = self.state_names.index(initial_state)
            else:
                i = initial_state

            # 걸음마다 확률 셈하기
            probabilities = np.zeros((max_steps + 1, self.n_states))
            probabilities[0, i] = 1.0  # 확률 1로 initial_state에서 시작

            for n in range(1, max_steps + 1):
                P_n = self.n_step_transition_matrix(n)
                probabilities[n, :] = P_n[i, :]

            # 시각화 만들기
            plt.figure(figsize=(12, 6))

            for j in range(self.n_states):
                plt.plot(range(max_steps + 1), probabilities[:, j], 
                        marker='o', markersize=4, label=self.state_names[j],
                        linewidth=2, alpha=0.7)

            plt.xlabel('Number of Steps (n)', fontsize=12)
            plt.ylabel('Probability', fontsize=12)
            plt.title(f'State Probabilities Over Time (Starting from {self.state_names[i]})',
                     fontsize=14)
            plt.legend(loc='best', fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.ylim(-0.05, 1.05)
            plt.tight_layout()
            plt.savefig('transition_probabilities.png', 
                       dpi=150, bbox_inches='tight')
            plt.close()


    def example_two_step_computation():
        """
        보기 1: 2걸음 옮김 확률을 손수 그리고 행렬 거듭제곱으로 셈하기.

        채프먼-콜모고로프 방정식을 보인다.
        """
        print("=" * 70)
        print("Example 1: Two-Step Transition Probability Computation")
        print("=" * 70)

        # 단순한 두 상태 사슬
        P = np.array([
            [0.7, 0.3],
            [0.4, 0.6]
        ])

        print("\nTransition Matrix P:")
        print(P)

        # 채프먼-콜모고로프로 P^2 손수 셈하기
        print("\nComputing P^2 manually using Chapman-Kolmogorov:")
        print("P^2[0][0] = P[0][0]*P[0][0] + P[0][1]*P[1][0]")

        P_2_manual = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                # 채프먼-콜모고로프: P^2[i][j] = Σ_k P[i][k] * P[k][j]
                value = sum(P[i][k] * P[k][j] for k in range(2))
                P_2_manual[i][j] = value
                print(f"P^2[{i}][{j}] = {value:.4f}")

        # 행렬 곱으로 P^2 셈하기
        P_2_matrix = P @ P

        print("\nP^2 via matrix multiplication:")
        print(P_2_matrix)

        print("\nVerification (difference should be ~0):")
        print(np.abs(P_2_manual - P_2_matrix))

        # 해석
        print("\nInterpretation:")
        print(f"Starting from state 0, probability of being in state 0 after 2 steps: {P_2_matrix[0,0]:.4f}")
        print(f"Starting from state 0, probability of being in state 1 after 2 steps: {P_2_matrix[0,1]:.4f}")


    def example_state_distribution_evolution():
        """
        보기 2: 시간에 따른 상태 분포의 흘러감.

        첫 분포가 π_n = π_0 × P^n을 따라 어떻게 흘러가는지 보인다
        """
        print("\n" + "=" * 70)
        print("Example 2: State Distribution Evolution")
        print("=" * 70)

        # 세 상태 날씨 모형
        states = ['Sunny', 'Cloudy', 'Rainy']
        P = np.array([
            [0.7, 0.25, 0.05],
            [0.3, 0.4, 0.3],
            [0.1, 0.4, 0.5]
        ])

        analyzer = TransitionMatrixAnalyzer(P, states)

        # 고른 분포로 시작(상태마다 확률이 같음)
        print("\nInitial distribution (uniform):")
        π_0 = np.array([1/3, 1/3, 1/3])
        print(f"π_0 = {π_0}")

        # 여러 걸음에 대한 분포 셈하기
        steps_to_show = [1, 2, 5, 10, 20, 50]

        print("\nDistribution evolution:")
        print(f"{'Step':<8} {'Sunny':<12} {'Cloudy':<12} {'Rainy':<12}")
        print(f"{'0':<8} {π_0[0]:<12.6f} {π_0[1]:<12.6f} {π_0[2]:<12.6f}")

        for n in steps_to_show:
            π_n = analyzer.state_distribution_after_n_steps(π_0, n)
            print(f"{n:<8} {π_n[0]:<12.6f} {π_n[1]:<12.6f} {π_n[2]:<12.6f}")

        # 서로 다른 첫 분포 시도하기
        print("\n" + "-" * 70)
        print("Starting from definitely Sunny (π_0 = [1, 0, 0]):")
        print(f"{'Step':<8} {'Sunny':<12} {'Cloudy':<12} {'Rainy':<12}")

        π_0_sunny = np.array([1.0, 0.0, 0.0])
        print(f"{'0':<8} {π_0_sunny[0]:<12.6f} {π_0_sunny[1]:<12.6f} {π_0_sunny[2]:<12.6f}")

        for n in steps_to_show:
            π_n = analyzer.state_distribution_after_n_steps(π_0_sunny, n)
            print(f"{n:<8} {π_n[0]:<12.6f} {π_n[1]:<12.6f} {π_n[2]:<12.6f}")


    def example_convergence_analysis():
        """
        보기 3: n → ∞일 때 P^n의 모임 살피기.

        규칙 사슬에서 P^n은 끝 행렬로 모인다.
        """
        print("\n" + "=" * 70)
        print("Example 3: Convergence Analysis")
        print("=" * 70)

        # 규칙 마르코프 사슬 만들기(어떤 P^k의 성분이 모두 양수)
        states = ['A', 'B', 'C']
        P = np.array([
            [0.5, 0.3, 0.2],
            [0.2, 0.6, 0.2],
            [0.3, 0.3, 0.4]
        ])

        analyzer = TransitionMatrixAnalyzer(P, states)

        print("\nTransition Matrix P:")
        print(P)

        # 모임 살피기
        results = analyzer.analyze_convergence(max_steps=100, tolerance=1e-8)

        if results['converged']:
            print(f"\nConvergence achieved at step {results['convergence_step']}")
            print("\nLimiting matrix (all rows identical = stationary distribution):")
            print(results['limit_matrix'])

            # 멈춘 분포 뽑아내기(끝 행렬의 아무 행)
            stationary = results['limit_matrix'][0, :]
            print(f"\nStationary distribution:")
            for i, state in enumerate(states):
                print(f"  π({state}) = {stationary[i]:.6f}")
        else:
            print("\nDid not converge within 100 steps")

        # 모임 그리기
        plt.figure(figsize=(10, 5))
        plt.semilogy(range(1, len(results['differences']) + 1), results['differences'],
                    'b-', linewidth=2)
        plt.xlabel('Step n', fontsize=12)
        plt.ylabel('||P^(n+1) - P^n|| (log scale)', fontsize=12)
        plt.title('Convergence of Transition Matrix Powers', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('convergence_plot.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("\nConvergence plot saved to convergence_plot.png")


    def example_specific_probabilities():
        """
        보기 4: 정해진 n걸음 옮김 확률 셈하기.

        다음과 같은 물음에 답한다: "상태 A에서 시작해 10걸음 뒤에
        상태 B에 있을 확률은 얼마인가?"
        """
        print("\n" + "=" * 70)
        print("Example 4: Specific n-Step Probabilities")
        print("=" * 70)

        states = ['Healthy', 'Sick', 'Recovered']
        P = np.array([
            [0.8, 0.2, 0.0],    # 건강: 80% 건강 그대로, 20% 앓음
            [0.0, 0.5, 0.5],    # 앓음: 50% 앓은 채, 50% 나음
            [0.9, 0.0, 0.1]     # 회복: 90% 건강해짐, 10% 회복 그대로
        ])

        analyzer = TransitionMatrixAnalyzer(P, states)

        print("\nTransition Matrix (Health States):")
        print("             Healthy  Sick  Recovered")
        for i, state in enumerate(states):
            print(f"{state:12s} {P[i]}")

        # 정해진 물음에 답하기
        questions = [
            ("Healthy", "Sick", 1),
            ("Healthy", "Sick", 5),
            ("Healthy", "Recovered", 10),
            ("Sick", "Healthy", 3),
        ]

        print("\nSpecific probability queries:")
        for initial, target, steps in questions:
            prob = analyzer.probability_after_n_steps(initial, target, steps)
            print(f"P({target} after {steps} steps | start from {initial}) = {prob:.6f}")


    def main():
        """
        옮김 행렬 연산을 보이는 보기 모두 돌리기.
        """
        print("TRANSITION MATRIX ANALYSIS")
        print("==========================\n")

        # 예제 실행
        example_two_step_computation()
        example_state_distribution_evolution()
        example_convergence_analysis()
        example_specific_probabilities()

        # 시각화 만들기
        print("\n" + "=" * 70)
        print("Creating Probability Evolution Visualization")
        print("=" * 70)

        states = ['State A', 'State B', 'State C']
        P = np.array([
            [0.5, 0.3, 0.2],
            [0.2, 0.6, 0.2],
            [0.3, 0.3, 0.4]
        ])

        analyzer = TransitionMatrixAnalyzer(P, states)
        analyzer.visualize_n_step_probabilities('State A', max_steps=50)
        print("Visualization saved to transition_probabilities.png")

        print("\n" + "=" * 70)
        print("Key Takeaways:")
        print("=" * 70)
        print("1. P^n[i][j] gives the probability of transitioning from i to j in n steps")
        print("2. Chapman-Kolmogorov: P^(m+n) = P^m × P^n")
        print("3. Distribution evolution: π_n = π_0 × P^n")
        print("4. For regular chains, P^n converges to a limit matrix")
        print("5. The limit matrix has all rows equal to the stationary distribution")


    if __name__ == "__main__":
        main()
    ```


??? note "전체 출력 (110줄)"

    ```
    TRANSITION MATRIX ANALYSIS
    ==========================

    ======================================================================
    Example 1: Two-Step Transition Probability Computation
    ======================================================================

    Transition Matrix P:
    [[0.7 0.3]
     [0.4 0.6]]

    Computing P^2 manually using Chapman-Kolmogorov:
    P^2[0][0] = P[0][0]*P[0][0] + P[0][1]*P[1][0]
    P^2[0][0] = 0.6100
    P^2[0][1] = 0.3900
    P^2[1][0] = 0.5200
    P^2[1][1] = 0.4800

    P^2 via matrix multiplication:
    [[0.61 0.39]
     [0.52 0.48]]

    Verification (difference should be ~0):
    [[1.11022302e-16 5.55111512e-17]
     [0.00000000e+00 0.00000000e+00]]

    Interpretation:
    Starting from state 0, probability of being in state 0 after 2 steps: 0.6100
    Starting from state 0, probability of being in state 1 after 2 steps: 0.3900

    ======================================================================
    Example 2: State Distribution Evolution
    ======================================================================

    Initial distribution (uniform):
    π_0 = [0.33333333 0.33333333 0.33333333]

    Distribution evolution:
    Step     Sunny        Cloudy       Rainy       
    0        0.333333     0.333333     0.333333    
    1        0.366667     0.350000     0.283333    
    2        0.390000     0.345000     0.265000    
    5        0.413970     0.338485     0.247545    
    10       0.418382     0.337271     0.244347    
    20       0.418604     0.337209     0.244186    
    50       0.418605     0.337209     0.244186    

    ----------------------------------------------------------------------
    Starting from definitely Sunny (π_0 = [1, 0, 0]):
    Step     Sunny        Cloudy       Rainy       
    0        1.000000     0.000000     0.000000    
    1        0.700000     0.250000     0.050000    
    2        0.570000     0.295000     0.135000    
    5        0.443070     0.330475     0.226455    
    10       0.419780     0.336886     0.243334    
    20       0.418607     0.337209     0.244184    
    50       0.418605     0.337209     0.244186    

    ======================================================================
    Example 3: Convergence Analysis
    ======================================================================

    Transition Matrix P:
    [[0.5 0.3 0.2]
     [0.2 0.6 0.2]
     [0.3 0.3 0.4]]

    Convergence achieved at step 15

    Limiting matrix (all rows identical = stationary distribution):
    [[0.32142857 0.42857143 0.25      ]
     [0.32142857 0.42857143 0.25      ]
     [0.32142857 0.42857143 0.25      ]]

    Stationary distribution:
      π(A) = 0.321429
      π(B) = 0.428571
      π(C) = 0.250000

    Convergence plot saved to convergence_plot.png

    ======================================================================
    Example 4: Specific n-Step Probabilities
    ======================================================================

    Transition Matrix (Health States):
                 Healthy  Sick  Recovered
    Healthy      [0.8 0.2 0. ]
    Sick         [0.  0.5 0.5]
    Recovered    [0.9 0.  0.1]

    Specific probability queries:
    P(Sick after 1 steps | start from Healthy) = 0.200000
    P(Sick after 5 steps | start from Healthy) = 0.246220
    P(Recovered after 10 steps | start from Healthy) = 0.136996
    P(Healthy after 3 steps | start from Sick) = 0.630000

    ======================================================================
    Creating Probability Evolution Visualization
    ======================================================================
    Visualization saved to transition_probabilities.png

    ======================================================================
    Key Takeaways:
    ======================================================================
    1. P^n[i][j] gives the probability of transitioning from i to j in n steps
    2. Chapman-Kolmogorov: P^(m+n) = P^m × P^n
    3. Distribution evolution: π_n = π_0 × P^n
    4. For regular chains, P^n converges to a limit matrix
    5. The limit matrix has all rows equal to the stationary distribution
    ```


## 2. 논의

### 여러 단계의 전이

시간 동질적인 유한 사슬에서 중간 상태를 합산하면 Chapman–Kolmogorov 관계를 얻는다.

$$[P^{m+n}]_{ij}=\sum_k[P^m]_{ik}[P^n]_{kj},\qquad p_n=p_0P^n$$

P⁰는 항등행렬이다. 코드의 행은 출발 상태, 열은 도착 상태이며 초기분포는 행벡터다. 2상태 예제 P=[[0.7,0.3],[0.4,0.6]]에서 P²의 첫 행은 (0.61,0.39)다.

### 수렴과 정상성

정상분포 π는 πP=π를 만족한다. 유한 기약·비주기 사슬이면 Pⁿ의 모든 행이 π로 수렴하지만, 주기 사슬에서는 진동할 수 있다. 가약 사슬에서도 Pⁿ 자체는 수렴할 수 있으나 행마다 극한이 다를 수 있다.

analyze_convergence는 Pⁿ과 Pⁿ⁺¹의 원소별 최대 차이가 작으면 종료한다. 이는 수치적 정지 규칙이지 참극한과의 오차 상한이 아니다. 거의 항등인 P에서는 차이가 작아도 초기조건을 오래 기억할 수 있다. step=1에서 P²와 P를 비교하므로 convergence_step과 실제 행렬 지수를 구분한다.

### 입력 검증

행렬은 비어 있지 않은 유한 정방행렬이며 각 행이 확률분포여야 한다. 초기분포는 합 1뿐 아니라 길이·음이 아닌 값·유한성도 확인한다. 단계 수는 음이 아닌 정수이고 상태명 목록과 인덱스 범위가 맞아야 한다. 최종 행 하나를 정상분포라 보고하려면 확률 제약과 πP−π 잔차, 유일성·수렴 조건을 함께 점검한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
P²의 (i,j) 원소는 무엇이며 중간 상태 합이 필요한 이유는?

</div>

??? success "연습문제 1 풀이"

    두 번 전이해 i에서 j로 갈 확률이다. 첫 전이의 중간 상태 k가 여러 가지이므로 서로 배타적인 경로 확률 P_ik P_kj를 k에 대해 합한다. 행렬 곱은 이 합산을 수행한다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
state_distribution_after_n_steps가 초기분포의 합만 검사하면 충분한가?

</div>

??? success "연습문제 2 풀이"

    아니다. 상태 수와 같은 길이, 유한한 원소와 음이 아닌 값이 필요하다. 합이 1인 음수 벡터도 존재하므로 합 검사만으로 확률분포가 되지는 않는다. n도 음이 아닌 정수여야 한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
Pⁿ⁺¹−Pⁿ이 작으면 정상분포에 가깝다는 결론이 항상 성립하는가?

</div>

??? success "연습문제 3 풀이"

    아니다. P가 항등행렬에 매우 가까우면 한 단계 변화는 작지만 고유한 정상분포까지의 거리는 클 수 있다. 가약 행렬은 여러 정상분포를 가질 수도 있다. 종료 차이와 정상성 잔차, 구조 및 혼합 정보를 구분해야 한다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
아래 보존된 시험 조각이 전이행렬 분석기의 유효한 시험인지 설명하라.

</div>

??? success "연습문제 4 풀이"

    아래 조각은 뉴럴 넷트웍 호출·경사·parameters를 가정하므로 이 클래스에 맞지 않는다. 적절한 정적 점검 사례는 P⁰=I, P¹=P, 작은 행렬의 P²와 수계산 일치, 유효 초기분포의 합·비음성 보존 및 주기 행렬의 진동이다. 여기서는 조각을 수정하거나 실행하지 않는다.

    ```python
    def test_transitionmatrixanalyzer():
        model = TransitionMatrixAnalyzer(...)
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

Pⁿ은 여러 단계 전이확률이고 초기 행분포는 p₀Pⁿ으로 변한다. 정상성·유일성·모든 초기조건의 수렴과 수치 종료 규칙은 구분한다. 입력 확률 조건과 코드가 반환하는 단계 번호의 의미를 확인한다.
