# 흡수 사슬

흡수 상태는 한 번 들어가면 떠나지 않는 상태다. 유한 흡수 사슬에서는 모든 비흡수 상태에서 어떤 흡수 상태로 도달할 수 있어야 한다. 이 조건 아래 과도 상태의 방문 횟수, 흡수까지 걸리는 시간과 최종 흡수 확률을 기본행렬로 계산한다.

## 1. 코드

??? note "코드 (498줄)"

    ```python
    """
    absorbing_chains.py (단원 05)

    흡수 마르코프 사슬
    ========================

    Location: 06_markov_chain/02_analysis_methods/
    난이도: ⭐⭐⭐ 중급
    걸리는 시간: 3-4시간

    학습 목표:
    - 흡수 상태와 흡수 사슬 이해하기
    - 흡수 확률 셈하기
    - 흡수까지의 기댓값 시간 셈하기
    - 바탕 행렬 살피기

    수학적 바탕:
    흡수 상태는 한 번 들어가면 떠날 수 없는 상태이다.
    P[i][i] = 1이면 상태 i은 흡수 상태이다.

    흡수 사슬은 다음을 갖는다:
    1. 흡수 상태가 적어도 하나 있다
    2. 모든 상태에서 흡수 상태에 닿을 수 있다

    P의 정준 꼴:
        ┌       ┐
    P = │ Q  R  │  여기서:
        │ 0  I  │
        └       ┘
    - Q: 지나가는 상태 사이의 옮김
    - R: 지나가는 상태에서 흡수 상태로의 옮김
    - I: 항등 행렬(흡수 상태)
    - 0: 영행렬(흡수 상태를 떠날 수 없다)

    핵심 양:
    - 바탕 행렬: N = (I - Q)^{-1}
    - N[i][j] = i에서 시작해 지나가는 상태 j에 들르는 기댓값 횟수
    - 흡수까지의 기댓값 스텝 수: t = N × 1(1로 채운 열벡터)
    - 흡수 확률: B = N × R
    """

    import numpy as np
    import matplotlib.pyplot as plt

    # 무작위로 뽑는 값이 아래에 나온다. 씨앗을 고정해야 이 쪽에 실린
    # 수가 다시 나온다 — 고정하지 않으면 돌릴 때마다 다른 수가 찍힌다
    np.random.seed(0)

    # ========================================================================
    # 메인
    # ========================================================================


    class AbsorbingMarkovChain:
        """
        흡수 마르코프 사슬을 살피는 도구.
        """

        def __init__(self, transition_matrix, state_names=None):
            """
            흡수 사슬 첫값 잡기.

            파라메터:
                transition_matrix (np.ndarray): 옮김 행렬
                state_names (list): 상태 이름(없어도 된다)
            """
            self.P = np.array(transition_matrix, dtype=float)
            self.n_states = self.P.shape[0]

            if state_names is None:
                self.state_names = [f"State {i}" for i in range(self.n_states)]
            else:
                self.state_names = state_names

            # 흡수 상태와 지나가는 상태 가려내기
            self._identify_states()

            # 필요하면 차례 바꾸기
            self._reorder_canonical()

        def _identify_states(self):
            """
            어느 상태가 흡수 상태인지 가려내기.

            P[i][i] = 1이고 나머지 P[i][j] = 0이면 상태 i은 흡수 상태이다
            """
            self.absorbing_indices = []
            self.transient_indices = []

            for i in range(self.n_states):
                if np.isclose(self.P[i, i], 1.0) and np.allclose(self.P[i, :i], 0.0) and np.allclose(self.P[i, i+1:], 0.0):
                    self.absorbing_indices.append(i)
                else:
                    self.transient_indices.append(i)

            self.n_transient = len(self.transient_indices)
            self.n_absorbing = len(self.absorbing_indices)

        def _reorder_canonical(self):
            """
            상태를 정준 꼴로 다시 늘어놓기: 지나가는 상태 먼저, 그다음 흡수 상태.

            Q, R 행렬을 만들고 바탕 행렬 N 셈하기.
            """
            if self.n_absorbing == 0:
                raise ValueError("No absorbing states found")

            # 상태 차례 바꾸기
            reordered_indices = self.transient_indices + self.absorbing_indices

            # 옮김 행렬의 차례 바꾸기
            P_canonical = self.P[np.ix_(reordered_indices, reordered_indices)]

            # Q과 R 뽑아내기
            self.Q = P_canonical[:self.n_transient, :self.n_transient]
            self.R = P_canonical[:self.n_transient, self.n_transient:]

            # 차례 바꾼 이름 저장
            self.transient_names = [self.state_names[i] for i in self.transient_indices]
            self.absorbing_names = [self.state_names[i] for i in self.absorbing_indices]

        def fundamental_matrix(self):
            """
            바탕 행렬 N = (I - Q)^{-1} 셈하기.

            반환값:
                np.ndarray: 바탕 행렬 N

            수학으로 풀이하기:
            N[i][j] = 지나가는 상태 j에 머무는 기댓값 횟수,
                      지나가는 상태 i에서 시작해 흡수되기 전까지

            이끌어 내기:
            M[i][j] = E[i에서 시작해 j에 들르는 횟수]이라 하자
            M[i][j] = δ_{ij} + Σ_k P[i][k] × M[k][j]
            행렬로 쓰면: M = I + Q × M
            풀면: M = (I - Q)^{-1} = N
            """
            I = np.eye(self.n_transient)
            self.N = np.linalg.inv(I - self.Q)
            return self.N

        def expected_steps_to_absorption(self):
            """
            상태마다 흡수까지의 기댓값 스텝 수 셈하기.

            반환값:
                dict: 지나가는 상태마다의 기댓값 스텝 수

            수학 공식:
            t = N × 1(여기서 1은 1로 채운 열벡터)

            해석:
            t[i] = 지나가는 상태 i에서 시작해 흡수까지의 기댓값 스텝 수
            """
            if not hasattr(self, 'N'):
                self.fundamental_matrix()

            # N에 1로 채운 열벡터 곱하기
            ones = np.ones((self.n_transient, 1))
            t = self.N @ ones

            # 사전으로 돌려주기
            result = {}
            for i, name in enumerate(self.transient_names):
                result[name] = t[i, 0]

            return result

        def absorption_probabilities(self):
            """
            흡수 상태마다 흡수될 확률 셈하기.

            반환값:
                dict: 지나가는 상태마다 흡수 상태별로 흡수될 확률

            수학 공식:
            B = N × R

            해석:
            B[i][j] = 흡수 상태 j으로 흡수될 확률,
                      지나가는 상태 i에서 시작해
            """
            if not hasattr(self, 'N'):
                self.fundamental_matrix()

            self.B = self.N @ self.R

            # 겹친 사전으로 돌려주기
            result = {}
            for i, trans_name in enumerate(self.transient_names):
                result[trans_name] = {}
                for j, abs_name in enumerate(self.absorbing_names):
                    result[trans_name][abs_name] = self.B[i, j]

            return result

        def variance_steps_to_absorption(self):
            """
            흡수까지 스텝 수의 흩어짐 셈하기.

            반환값:
                dict: 지나가는 상태마다의 흩어짐

            수학 공식:
            Var[T_i] = (2N - I) × t - t²
            여기서 t은 기댓값 스텝 수 벡터이다
            """
            if not hasattr(self, 'N'):
                self.fundamental_matrix()

            ones = np.ones((self.n_transient, 1))
            t = self.N @ ones

            I = np.eye(self.n_transient)
            variance_vec = (2 * self.N - I) @ t - t**2

            result = {}
            for i, name in enumerate(self.transient_names):
                result[name] = variance_vec[i, 0]

            return result


    def example_simple_gambler():
        """
        보기 1: 노름꾼의 파산 문제.

        노름꾼이 $2으로 시작한다. 판마다 $1을 따거나(p=0.5) $1을 잃는다(1-p=0.5).
        $0(파산)이나 $4(목표)에 이르면 노름이 끝난다.
        """
        print("=" * 70)
        print("Example 1: Gambler's Ruin")
        print("=" * 70)

        # 상태: $0, $1, $2, $3, $4
        # 흡수: $0(파산), $4(승리)
        # 지나감: $1, $2, $3

        states = ['$0 (Broke)', '$1', '$2', '$3', '$4 (Win)']

        # 옮김 행렬(공정한 노름이면 p = 0.5)
        P = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0],  # $0: 파산 그대로
            [0.5, 0.0, 0.5, 0.0, 0.0],  # $1: $0이나 $2으로
            [0.0, 0.5, 0.0, 0.5, 0.0],  # $2: $1이나 $3으로
            [0.0, 0.0, 0.5, 0.0, 0.5],  # $3: $2이나 $4으로
            [0.0, 0.0, 0.0, 0.0, 1.0]   # $4: 승리 그대로
        ])

        print("\nTransition Matrix:")
        print(P)

        chain = AbsorbingMarkovChain(P, states)

        print(f"\nAbsorbing states: {chain.absorbing_names}")
        print(f"Transient states: {chain.transient_names}")

        # 바탕 행렬
        print("\n" + "-" * 70)
        print("Fundamental Matrix N (expected visits):")
        N = chain.fundamental_matrix()
        print(f"{'':8s} " + " ".join(f"{s:8s}" for s in chain.transient_names))
        for i, name in enumerate(chain.transient_names):
            row = " ".join(f"{N[i,j]:8.4f}" for j in range(len(chain.transient_names)))
            print(f"{name:8s} {row}")

        # 흡수까지의 기댓값 스텝 수
        print("\n" + "-" * 70)
        print("Expected Steps to Absorption:")
        expected_steps = chain.expected_steps_to_absorption()
        for state, steps in expected_steps.items():
            print(f"  Starting from {state}: {steps:.4f} steps")

        # 흡수 확률
        print("\n" + "-" * 70)
        print("Absorption Probabilities:")
        absorption_probs = chain.absorption_probabilities()
        for trans_state in chain.transient_names:
            print(f"\n  Starting from {trans_state}:")
            for abs_state in chain.absorbing_names:
                prob = absorption_probs[trans_state][abs_state]
                print(f"    P(absorb at {abs_state}) = {prob:.6f}")

        # 흩어짐
        print("\n" + "-" * 70)
        print("Variance of Steps to Absorption:")
        variances = chain.variance_steps_to_absorption()
        for state, var in variances.items():
            print(f"  Starting from {state}: {var:.4f} (std = {np.sqrt(var):.4f})")


    def example_disease_model():
        """
        보기 2: 병의 진행 모형.

        상태: 건강, 감염, 회복, 죽음
        흡수: 회복, 죽음
        """
        print("\n" + "=" * 70)
        print("Example 2: Disease Progression Model")
        print("=" * 70)

        states = ['Healthy', 'Infected', 'Recovered', 'Dead']

        P = np.array([
            [0.7, 0.3, 0.0, 0.0],    # 건강: 옮을 수 있음
            [0.0, 0.4, 0.5, 0.1],    # 감염: 낫거나, 앓은 채이거나, 죽음
            [0.0, 0.0, 1.0, 0.0],    # 회복: 흡수
            [0.0, 0.0, 0.0, 1.0]     # 죽음: 흡수
        ])

        print("\nTransition Matrix:")
        print(f"{'':12s} {'Healthy':>10s} {'Infected':>10s} {'Recovered':>10s} {'Dead':>10s}")
        for i, state in enumerate(states):
            row = " ".join(f"{P[i,j]:10.4f}" for j in range(len(states)))
            print(f"{state:12s} {row}")

        chain = AbsorbingMarkovChain(P, states)

        print(f"\nAbsorbing states: {chain.absorbing_names}")
        print(f"Transient states: {chain.transient_names}")

        # 흡수까지의 기댓값 시간
        expected_steps = chain.expected_steps_to_absorption()
        print("\nExpected time until recovery or death:")
        for state, steps in expected_steps.items():
            print(f"  From {state}: {steps:.4f} days")

        # 흡수 확률
        absorption_probs = chain.absorption_probabilities()
        print("\nFinal outcome probabilities:")
        for trans_state in chain.transient_names:
            print(f"\n  Starting from {trans_state}:")
            for abs_state in chain.absorbing_names:
                prob = absorption_probs[trans_state][abs_state]
                print(f"    {abs_state}: {prob:.4f} ({prob*100:.2f}%)")


    def visualize_absorption():
        """
        흉내내기로 흡수 과정 그려 보기.
        """
        print("\n" + "=" * 70)
        print("Creating Absorption Visualization")
        print("=" * 70)

        # 노름꾼의 파산
        states_idx = {'$0': 0, '$1': 1, '$2': 2, '$3': 3, '$4': 4}
        P = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5, 0.0, 0.5],
            [0.0, 0.0, 0.0, 0.0, 1.0]
        ])

        # 여러 경로 흉내내기
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 그림 1: 표본 경로
        ax = axes[0, 0]

        for _ in range(20):
            path = [2]  # $2에서 시작
            current = 2

            while current != 0 and current != 4 and len(path) < 100:
                probs = P[current, :]
                current = np.random.choice(5, p=probs)
                path.append(current)

            ax.plot(path, alpha=0.6, linewidth=1.5)

        ax.set_xlabel('Time Step', fontsize=11)
        ax.set_ylabel('Money ($)', fontsize=11)
        ax.set_title('Sample Paths in Gambler\'s Ruin', fontsize=12)
        ax.set_yticks([0, 1, 2, 3, 4])
        ax.set_yticklabels(['$0', '$1', '$2', '$3', '$4'])
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Broke')
        ax.axhline(y=4, color='green', linestyle='--', alpha=0.5, label='Win')
        ax.legend()

        # 그림 2: 흡수 시간의 분포
        ax = axes[0, 1]

        absorption_times = []
        for _ in range(10000):
            steps = 0
            current = 2

            while current != 0 and current != 4 and steps < 1000:
                probs = P[current, :]
                current = np.random.choice(5, p=probs)
                steps += 1

            absorption_times.append(steps)

        ax.hist(absorption_times, bins=50, density=True, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Steps to Absorption', fontsize=11)
        ax.set_ylabel('Probability Density', fontsize=11)
        ax.set_title('Distribution of Time to Absorption', fontsize=12)
        ax.grid(True, alpha=0.3)

        # 이론의 평균 더하기
        chain = AbsorbingMarkovChain(P, ['$0', '$1', '$2', '$3', '$4'])
        expected = chain.expected_steps_to_absorption()
        theoretical_mean = expected['$2']
        ax.axvline(x=theoretical_mean, color='red', linestyle='--', linewidth=2,
                  label=f'Theoretical Mean: {theoretical_mean:.2f}')
        ax.axvline(x=np.mean(absorption_times), color='blue', linestyle='--', linewidth=2,
                  label=f'Empirical Mean: {np.mean(absorption_times):.2f}')
        ax.legend()

        # 그림 3: 흡수 확률
        ax = axes[1, 0]

        outcomes = {'Win': 0, 'Broke': 0}
        for _ in range(10000):
            current = 2

            while current != 0 and current != 4:
                probs = P[current, :]
                current = np.random.choice(5, p=probs)

            if current == 4:
                outcomes['Win'] += 1
            else:
                outcomes['Broke'] += 1

        labels = list(outcomes.keys())
        values = [outcomes[k] / 10000 for k in labels]
        colors = ['green', 'red']

        bars = ax.bar(labels, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax.set_ylabel('Probability', fontsize=11)
        ax.set_title('Absorption Outcomes (Starting from $2)', fontsize=12)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

        # 그림 4: 바탕 행렬 그림
        ax = axes[1, 1]

        chain = AbsorbingMarkovChain(P, ['$0', '$1', '$2', '$3', '$4'])
        N = chain.fundamental_matrix()

        im = ax.imshow(N, cmap='YlOrRd', aspect='auto')
        ax.set_xticks(range(len(chain.transient_names)))
        ax.set_yticks(range(len(chain.transient_names)))
        ax.set_xticklabels(chain.transient_names)
        ax.set_yticklabels(chain.transient_names)
        ax.set_xlabel('To State', fontsize=11)
        ax.set_ylabel('From State', fontsize=11)
        ax.set_title('Fundamental Matrix N (Expected Visits)', fontsize=12)

        for i in range(len(chain.transient_names)):
            for j in range(len(chain.transient_names)):
                text = ax.text(j, i, f'{N[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=10)

        plt.colorbar(im, ax=ax)

        plt.tight_layout()
        plt.savefig('absorbing_chains.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Absorption visualization saved")


    def main():
        """
        보기 모두 돌리기.
        """
        print("ABSORBING MARKOV CHAINS")
        print("=======================\n")

        example_simple_gambler()
        example_disease_model()
        visualize_absorption()

        print("\n" + "=" * 70)
        print("Key Concepts:")
        print("=" * 70)
        print("1. Absorbing state: P[i][i] = 1")
        print("2. Fundamental matrix: N = (I - Q)^{-1}")
        print("3. Expected steps to absorption: t = N × 1")
        print("4. Absorption probabilities: B = N × R")
        print("5. N[i][j] = expected visits to state j from state i")


    if __name__ == "__main__":
        main()
    ```


??? note "전체 출력 (92줄)"

    ```
    ABSORBING MARKOV CHAINS
    =======================

    ======================================================================
    Example 1: Gambler's Ruin
    ======================================================================

    Transition Matrix:
    [[1.  0.  0.  0.  0. ]
     [0.5 0.  0.5 0.  0. ]
     [0.  0.5 0.  0.5 0. ]
     [0.  0.  0.5 0.  0.5]
     [0.  0.  0.  0.  1. ]]

    Absorbing states: ['$0 (Broke)', '$4 (Win)']
    Transient states: ['$1', '$2', '$3']

    ----------------------------------------------------------------------
    Fundamental Matrix N (expected visits):
             $1       $2       $3      
    $1         1.5000   1.0000   0.5000
    $2         1.0000   2.0000   1.0000
    $3         0.5000   1.0000   1.5000

    ----------------------------------------------------------------------
    Expected Steps to Absorption:
      Starting from $1: 3.0000 steps
      Starting from $2: 4.0000 steps
      Starting from $3: 3.0000 steps

    ----------------------------------------------------------------------
    Absorption Probabilities:

      Starting from $1:
        P(absorb at $0 (Broke)) = 0.750000
        P(absorb at $4 (Win)) = 0.250000

      Starting from $2:
        P(absorb at $0 (Broke)) = 0.500000
        P(absorb at $4 (Win)) = 0.500000

      Starting from $3:
        P(absorb at $0 (Broke)) = 0.250000
        P(absorb at $4 (Win)) = 0.750000

    ----------------------------------------------------------------------
    Variance of Steps to Absorption:
      Starting from $1: 8.0000 (std = 2.8284)
      Starting from $2: 8.0000 (std = 2.8284)
      Starting from $3: 8.0000 (std = 2.8284)

    ======================================================================
    Example 2: Disease Progression Model
    ======================================================================

    Transition Matrix:
                    Healthy   Infected  Recovered       Dead
    Healthy          0.7000     0.3000     0.0000     0.0000
    Infected         0.0000     0.4000     0.5000     0.1000
    Recovered        0.0000     0.0000     1.0000     0.0000
    Dead             0.0000     0.0000     0.0000     1.0000

    Absorbing states: ['Recovered', 'Dead']
    Transient states: ['Healthy', 'Infected']

    Expected time until recovery or death:
      From Healthy: 5.0000 days
      From Infected: 1.6667 days

    Final outcome probabilities:

      Starting from Healthy:
        Recovered: 0.8333 (83.33%)
        Dead: 0.1667 (16.67%)

      Starting from Infected:
        Recovered: 0.8333 (83.33%)
        Dead: 0.1667 (16.67%)

    ======================================================================
    Creating Absorption Visualization
    ======================================================================
    Absorption visualization saved

    ======================================================================
    Key Concepts:
    ======================================================================
    1. Absorbing state: P[i][i] = 1
    2. Fundamental matrix: N = (I - Q)^{-1}
    3. Expected steps to absorption: t = N × 1
    4. Absorption probabilities: B = N × R
    5. N[i][j] = expected visits to state j from state i
    ```


## 2. 논의

### 표준형과 기본행렬

과도 상태를 먼저, 흡수 상태를 뒤에 정렬하면 다음 블록 행렬을 얻는다.

$$P=\begin{pmatrix}Q&R\\0&I\end{pmatrix}$$

모든 비흡수 상태가 결국 흡수되는 유한 모형이면 Q의 스펙트럼 반지름이 1보다 작다. 따라서 기본행렬은 다음과 같다.

$$N=(I-Q)^{-1}=\sum_{k=0}^\infty Q^k$$

N_ij는 과도 상태 i에서 시작해 흡수 전 j에 방문하는 기대 횟수이며 초기시점 방문도 포함한다. 기대 흡수 단계 벡터는 t=N1, 흡수 확률 행렬은 B=NR이다. i번째 행의 B는 가능한 흡수 상태들에 걸쳐 합이 1이다.

$$\operatorname{Var}(T)=(2N-I)t-t\odot t$$

마지막 제곱은 원소별 제곱이다. 이미 흡수 상태에서 시작했다면 흡수 시간은 0이며 코드의 과도 상태 결과 사전에는 포함되지 않는다.

### 구현과 조건

코드는 대각선이 1인 행을 흡수 상태로 찾고 나머지를 모두 과도 상태로 분류한다. 흡수 상태가 하나 있다는 사실만으로 나머지가 과도인 것은 아니다. 흡수 상태에 닿지 않는 닫힌 클래스가 별도로 있으면 I−Q가 특이하고 위 공식을 적용할 수 없다. 모든 비흡수 상태의 흡수 상태 도달 가능성, 행렬의 확률 조건과 수치 조건수를 확인한다.

도박 파산 예제는 0·4의 경계를 흡수 상태로 둔다. 대칭인 경우 자금 i에서 4에 도달할 확률은 i/4, 기대 종료 단계는 i(4−i)다. 무한 격자에서 단일 목표의 최초 도달 시간과는 다른 유한 경계 문제다. 질병 이름을 사용한 예제의 전이율·흡수 가정도 교육용 숫자이며 실제 예후를 추정한 결과는 아니다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
기본행렬 N의 대각선 원소와 t=N1의 의미를 설명하라.

</div>

??? success "연습문제 1 풀이"

    N_ii는 i에서 시작했을 때 초기 방문을 포함해 흡수 전에 i에 머문 기대 횟수다. 행의 합은 모든 과도 상태의 기대 방문 수이며 초기 과도시점부터 흡수 직전까지를 세므로 흡수까지의 기대 전이 수와 같다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
흡수 상태가 하나라도 있으면 I−Q가 항상 가역인가?

</div>

??? success "연습문제 2 풀이"

    아니다. 흡수 상태로 나갈 수 없는 비흡수 닫힌 클래스가 있으면 그 블록은 사라지지 않고 Q의 고유값 1이 생겨 I−Q가 특이해진다. 유한 흡수 사슬의 공식을 쓰려면 모든 비흡수 상태가 흡수 상태에 도달 가능해야 한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
0·4를 흡수 경계로 둔 대칭 워크에서 자금 2의 종료 확률과 기대 단계를 구하라.

</div>

??? success "연습문제 3 풀이"

    상·하 경계의 흡수 확률은 각각 1/2이며 기대 종료 단계는 2(4−2)=4다. 이는 양쪽 경계에서 중단하는 문제이므로 무한 격자의 단일 목표 도달 시간의 무한 기댓값과 모순되지 않는다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
아래 뉴럴 넷트웍 형태 시험 조각을 흡수 사슬 분석에 적용할 수 있는가?

</div>

??? success "연습문제 4 풀이"

    해당 클래스는 호출·backward·parameters 인터페이스가 없으므로 맞지 않는다. 유효한 확률행렬, 흡수 상태의 유지, 모든 과도 상태의 흡수 도달 가능성, (I−Q)N≈I, B의 비음성·행합 1과 알려진 도박 파산 공식을 검증해야 한다. 이미 흡수된 초기조건과 특이 I−Q도 별도 처리한다.

    ```python
    def test_absorbingmarkovchain():
        model = AbsorbingMarkovChain(...)
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

기본행렬은 유한 사슬의 모든 비흡수 상태가 과도라는 조건에서 방문 횟수·흡수 시간·최종 흡수 확률을 연결한다. 흡수 상태의 존재만으로 충분하지 않으며 닫힌 클래스와 역행렬의 안정성을 확인해야 한다.
