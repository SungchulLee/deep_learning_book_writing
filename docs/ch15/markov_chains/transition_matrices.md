# 옮김 행렬

transition_matrices.py (모듈 02) 옮김 행렬과 상태 확률 다루기

확률 과정과 표집 방법은 확률적 기계 학습의 바탕 도구이다. 이 모듈은 마르코프 사슬의 개념을 보이며 수학 이론과 셈 구현 사이의 틈을 잇는다.

## 코드

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
from scipy.linalg import matrix_power

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
        
        매개변수:
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
        
        매개변수:
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
        
        매개변수:
            initial_state (int or str): 시작 상태
            target_state (int or str): 과녁 상태
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
        
        매개변수:
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
        
        매개변수:
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
        
        매개변수:
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
        plt.savefig('/mnt/user-data/outputs/transition_probabilities.png', 
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
    plt.savefig('/mnt/user-data/outputs/convergence_plot.png', dpi=150, bbox_inches='tight')
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
    main()```

## 논의

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다. 그림을 그리는 코드는 학습된 표현, 수렴의 움직임, 평가 지표에 대한 통찰을 주어 추상적인 계산을 손에 잡히게 만든다.

여기서 보인 무늬는 더 복잡한 상황으로 자연스럽게 넓어진다. 웃매개변수, 구조의 변형, 서로 다른 자료 묶음을 이리저리 시험해 보면 이해가 깊어지고 확률 과정 일감에 대한 실전 직관이 쌓인다.

## 연습문제

**연습문제 1.**
코드를 죽 읽고 핵심 설계 결정을 가려내어라. 구체적인 구현 고름 셋을 적고 저마다 왜 마르코프 사슬에 알맞은지 설명하여라.

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
옮김 행렬 구현을 확인하는 두루 갖춘 시험 함수를 적어라. 빈 입력, 원소 하나짜리 입력, 아주 큰 입력, 그리고 극단적인 값(0, 아주 큰 수)이 든 입력 같은 모서리 경우를 시험하여라.

??? success "연습문제 4 풀이"
    경계 조건을 두루 시험하는 함수를 만든다.
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
    경사의 흐름을 시험하는 것은 그 구조가 처음부터 끝까지 이어지는 학습을 지원하는지 확인하는 데 특히 중요하다.
