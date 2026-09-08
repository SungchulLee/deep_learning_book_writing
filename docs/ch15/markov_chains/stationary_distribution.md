# 멈춘 분포

stationary_distribution.py (모듈 04) 멈춘 분포 분석

확률 과정과 표집 방법은 확률적 기계 학습의 바탕 도구이다. 이 모듈은 마르코프 사슬의 개념을 보이며 수학 이론과 셈 구현 사이의 틈을 잇는다.

## 1. 코드

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
    plt.savefig('/mnt/user-data/outputs/stationary_convergence.png', 
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
    main()```

## 2. 논의

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
멈춘 분포 구현을 확인하는 두루 갖춘 시험 함수를 적어라. 빈 입력, 원소 하나짜리 입력, 아주 큰 입력, 그리고 극단적인 값(0, 아주 큰 수)이 든 입력 같은 모서리 경우를 시험하여라.

??? success "연습문제 4 풀이"
    경계 조건을 두루 시험하는 함수를 만든다.
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
    경사의 흐름을 시험하는 것은 그 구조가 처음부터 끝까지 이어지는 학습을 지원하는지 확인하는 데 특히 중요하다.

## 정리하며

**다룬 것** — 멈춘 분포

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다.

고갱이 갈래는 `StationaryDistributionAnalyzer`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
