# 에르고드성

에르고드성은 마르코프 사슬의 장기 거동과 MCMC의 정당성을 설명한다. 이 페이지에서는 유한·가산 상태 사슬의 기약성·주기·재귀성과 정상분포 수렴을 구분한다. 유한 사슬의 기하 수렴, 가역 사슬의 스펙트럼 상계, 일반 MCMC의 진단은 서로 다른 조건을 요구한다.

## 1. 도달 가능성과 기약성

### 도달 가능성과 소통류

어떤 정수 n≥0에서 [Pⁿ]_ij>0이면 i에서 j로 도달 가능하다. 서로 도달 가능하면 소통한다고 하며 이 관계는 반사성·대칭성·추이성을 만족하는 동치관계다. 상태 공간을 소통류로 나누고, 외부로 나가는 전이가 없는 소통류를 닫힌 클래스라 한다.

전체 상태가 한 소통류이면 사슬은 **기약**이다. 유한 사슬은 양의 확률 간선을 가진 그래프의 강연결성으로 검사할 수 있다. 서로 다른 i·j의 도달 가능성은 길이 N−1 이하의 경로로 판별할 수 있지만, 자기 자신은 길이 0의 경로도 포함한다. 따라서 I+P+…+P^(N−1)의 모든 항목이 양수인 조건을 사용할 수 있다. I를 빼면 대각선 귀환 경로가 빠져 잘못된 판정이 가능하다.

연속 상태 공간에는 개별 점 간 양의 전이확률 대신 측도에 따른 기약성 등 일반 상태 공간의 개념이 필요하다.

## 2. 주기성

상태 i의 양의 확률 귀환 시간들의 최대공약수를 주기라 한다.

$$d(i)=\gcd\{n\geq1:[P^n]_{ii}>0\}$$

기약 사슬에서는 모든 상태의 주기가 같다. 주기 1이면 **비주기적**이다. 귀환이 불가능한 과도상태에는 이 정의에서 양의 주기가 없을 수 있으며 닫힌 소통류의 주기와 구분한다.

P_ii>0이면 그 상태의 주기는 1이다. 사슬이 기약이라는 조건까지 있으면 모든 상태가 비주기적이다. 가약 사슬에서 한 상태의 자기 루프만으로 전체 상태의 비주기성을 결론 낼 수는 없다.

두 상태가 반드시 번갈아 바뀌는 P=[[0,1],[1,0]]은 주기 2다. P=[[0.5,0.5],[1,0]]은 기약이며 상태 0의 자기 루프 때문에 비주기다. 비주기성은 매 시점 상태를 독립적으로 생성한다는 뜻이 아니다.

## 3. 재귀성과 과도성

X₀=i에서 첫 귀환 시간을 $T_i=\inf\{n\geq1:X_n=i\}$라 하자. 돌아오지 않으면 T_i=∞다.

- **재귀 상태**: P_i(T_i<∞)=1이다.
- **과도 상태**: P_i(T_i<∞)<1이다.
- **양의 재귀**: 재귀 상태이면서 E_i[T_i]<∞다.
- **영 재귀**: 재귀하지만 E_i[T_i]=∞다.

가산 상태에서 재귀성은 ∑_{n≥0}[Pⁿ]_ii=∞와 동치다. 유한 상태 사슬의 재귀 상태는 양의 재귀이며 기약 유한 사슬의 모든 상태가 이에 해당한다. 가산 무한 상태에서는 정수 격자의 대칭 랜덤 워크처럼 기약이어도 영 재귀여서 정상 **확률**분포를 갖지 않을 수 있다.

## 4. 에르고드성과 수렴

이 페이지에서 가산 상태의 **에르고드 사슬**은 기약·비주기·양의 재귀인 사슬을 뜻한다. 유한 기약 사슬은 이미 양의 재귀이므로 기약성과 비주기성만 확인하면 된다. 문헌마다 에르고드라는 용어가 시간 평균에 초점을 두어 쓰이기도 하므로 사용한 의미를 명시한다.

이 조건에서 유일한 정상분포 π가 존재하고 모든 초기 상태 i에 대해 [Pⁿ]_ij→π_j다. π_i=1/E_i[T_i]라는 평균 귀환 시간 관계도 성립한다. 시간 평균의 법칙에는 비주기성이 필수가 아니므로 시점별 분포 수렴과 구분한다.

유한 기약·비주기 사슬은 어떤 C<∞, ρ<1에서 기하 수렴을 갖는다. 그러나 가산 무한 사슬의 위 조건만으로 기하 속도나 유한한 최악 초기상태 혼합 시간이 보장되지는 않는다. 실제 유한 실행의 정확도는 별도의 정량 분석과 진단이 필요하다.

## 5. 스펙트럼 갭과 수렴 속도

유한 기약·비주기 **가역** 사슬에서 1을 제외한 고유값의 최대 절댓값을 ρ라 하고 절대 스펙트럼 갭 γ*=1−ρ를 정의한다. 이때 다음과 같은 상계를 얻는다.

$$\|P^n(i,\cdot)-\pi\|_{TV}\leq\frac12\sqrt{\pi_i^{-1}-1}\,\rho^n$$

고유값이 음수일 수 있으므로 대수적으로 두 번째 큰 고유값을 사용하는 갭과 절대 갭을 구분한다. 비가역 행렬의 비대각화 가능한 부분에는 다항 인자가 생길 수 있어 똑같은 Cρⁿ 공식을 자동 적용하지 않는다.

약한 연결로 나뉜 지역이나 거의 주기적인 움직임은 느린 혼합을 만들 수 있다. 그러나 큰 차원·상관·작은 갭만으로 잘못 설계한 제안이라고 단정할 수는 없다. 목표분포의 구조와 커널을 함께 분석해야 한다.

## 6. 혼합 시간

### 총변동 거리와 혼합 시간

유한 사슬에서 최악 초기상태의 총변동 거리를 사용한다.

$$d(n)=\max_i\frac12\sum_j|[P^n]_{ij}-\pi_j|$$

$$t_{mix}(\epsilon)=\min\{n\geq0:d(n)\leq\epsilon\}$$

일반적으로 ε=1/4를 기준으로 쓰지만 정확도 목표에 맞춰 명시한다. 가산 무한 사슬은 각 시작점에서 수렴하더라도 최악 초기상태를 취한 혼합 시간이 무한할 수 있다.

### 가역 유한 사슬의 상계

앞 절의 조건과 0<ε<1/2, π_min=min_iπ_i>0에서 다음 n은 d(n)≤ε를 보장하는 충분한 선택이다.

$$n\geq\frac{\tfrac12\log(1/\pi_{min})+\log(1/(2\epsilon))}{\gamma_*}$$

정수 단계는 위로 반올림한다. 이 결과와 조건은 [Levin–Peres의 교재](https://pages.uoregon.edu/dlevin/MARKOV/mcmt2e.pdf)의 스펙트럼 분석을 참고한다. 단순히 t_mix=1/γ*라고 둘 수 없고 ε·작은 정상 질량·초기조건의 영향이 남는다. 일반 비가역 사슬에 같은 식을 무조건 적용하지 않는다.

워밍업은 초기화 영향을 줄이기 위한 계산이며 고정 비율을 버리는 것으로 수렴이 보장되지 않는다. 표본을 k개마다 하나만 남기는 thinning도 혼합 시간과 같은 간격으로 무조건 수행할 필요는 없다. 저장 비용 같은 이유가 없다면 대체로 전체 표본으로 자기상관과 몬테카를로 오차를 평가한다.

## 7. PyTorch 구현

StateClassifier는 양의 행렬 거듭제곱으로 도달 가능성을 계산하지만 길이 0을 제외하므로 자기 자신에 대한 도달 가능성 정의와 다를 수 있다. compute_period는 2N단계와 1e−10 임계값만 검사하므로 매우 작은 양의 귀환확률을 놓칠 수 있다. is_aperiodic은 한 상태의 자기 루프를 전체 비주기성으로 처리하므로 그 해석에는 기약성이 필요하다. 생성자에는 확률행렬 검증이 없다.

MixingTimeAnalyzer는 Pᵀ가 아니라 P의 **오른쪽 고유벡터**로 pi를 만든다. 행 확률행렬 P의 고유값 1 오른쪽 고유벡터는 상수 벡터이므로 일반적인 정상분포를 얻는 방법이 아니다. 대칭·이중 확률행렬 예제에서 우연히 균등 정상분포와 맞을 수 있지만, 일반 행렬에서는 TV 거리와 혼합 시간 결과도 잘못될 수 있다.

고유값 정렬은 복소수의 절댓값이 아닌 실수부 절댓값을 사용하므로 비가역 행렬의 복소 고유값 순서도 잘못될 수 있다. 1상태 행렬에는 두 번째 고유값이 없다. CPU 배열·NumPy 변환과 자료형 조건을 확인하고 반환한 pi의 비음성·합·pi@P−pi 잔차를 독립적으로 점검해야 한다. 아래 코드와 출력은 이런 제한이 있는 기존 예제로 보존했다.

### 상태 분류기

```python
import torch
from typing import Dict, List, Set, Tuple
from math import gcd

class StateClassifier:
    """
    마르코프 사슬의 상태 가르기: 서로 통하는 무리,
    줄일 수 없음, 주기, 에르고드성.
    """

    def __init__(self, transition_matrix: torch.Tensor):
        self.P = transition_matrix.clone()
        self.n_states = self.P.shape[0]

    def is_accessible(
        self, i: int, j: int, max_steps: int = None
    ) -> bool:
        """상태 i에서 상태 j에 닿을 수 있는지 살피기."""
        if max_steps is None:
            max_steps = self.n_states

        P_sum = torch.zeros_like(self.P)
        P_k = self.P.clone()
        for _ in range(max_steps):
            P_sum += P_k
            P_k = P_k @ self.P

        return P_sum[i, j].item() > 0

    def communicates(self, i: int, j: int) -> bool:
        """상태 i과 j이 서로 통하는지 살피기(i ↔ j)."""
        return self.is_accessible(i, j) and self.is_accessible(j, i)

    def find_communicating_classes(self) -> List[Set[int]]:
        """상태 공간을 서로 통하는 무리로 나누기."""
        visited = [False] * self.n_states
        classes = []

        for start in range(self.n_states):
            if visited[start]:
                continue
            current_class = {start}
            visited[start] = True

            for other in range(self.n_states):
                if not visited[other] and self.communicates(start, other):
                    current_class.add(other)
                    visited[other] = True

            classes.append(current_class)

        return classes

    def is_irreducible(self) -> bool:
        """사슬에 서로 통하는 무리가 하나뿐인지 살피기."""
        return len(self.find_communicating_classes()) == 1

    def compute_period(self, state: int) -> int:
        """
        주기 d(i) = gcd{n ≥ 1 : P^{(n)}_{ii} > 0} 셈하기.
        """
        return_times = []
        P_n = self.P.clone()

        for n in range(1, 2 * self.n_states + 1):
            if P_n[state, state].item() > 1e-10:
                return_times.append(n)
            P_n = P_n @ self.P

        if not return_times:
            return 0

        period = return_times[0]
        for t in return_times[1:]:
            period = gcd(period, t)
            if period == 1:
                break

        return period

    def is_aperiodic(self) -> bool:
        """주기 없음 살피기(넉넉한 조건: 자기 고리가 하나라도 있음)."""
        if torch.any(torch.diag(self.P) > 0):
            return True
        return self.compute_period(0) == 1

    def is_ergodic(self) -> Dict[str, bool]:
        """
        에르고드성 살피기 = 줄일 수 없음 + 주기 없음.
        끝이 있는 사슬에서는 이것이 양의 되돌아옴을 뜻한다.
        """
        is_irr = self.is_irreducible()
        is_aper = self.is_aperiodic()
        return {
            'irreducible': is_irr,
            'aperiodic': is_aper,
            'ergodic': is_irr and is_aper
        }

    def classify_all_states(self) -> Dict[int, Dict]:
        """모든 상태의 온전한 가르기."""
        classes = self.find_communicating_classes()
        state_to_class = {}
        for idx, cls in enumerate(classes):
            for state in cls:
                state_to_class[state] = idx

        results = {}
        for state in range(self.n_states):
            results[state] = {
                'communicating_class': state_to_class[state],
                'period': self.compute_period(state),
                'has_self_loop': self.P[state, state].item() > 0
            }

        return results
```

### 혼합 시간 분석기

```python
class MixingTimeAnalyzer:
    """
    모임과 섞임의 성질 살피기: 스펙트럼 틈,
    섞임 시간, 총변동 거리의 흘러감.
    """

    def __init__(
        self,
        transition_matrix: torch.Tensor,
        state_names: list = None
    ):
        self.P = transition_matrix.clone().double()
        self.n_states = self.P.shape[0]
        self.state_names = state_names or [
            f"S{i}" for i in range(self.n_states)
        ]
        self._compute_spectrum()
        self._compute_stationary()

    def _compute_spectrum(self):
        """절댓값으로 정렬한 고윳값 셈하기."""
        eigenvalues, eigenvectors = torch.linalg.eig(self.P)
        abs_vals = torch.abs(eigenvalues.real)
        sorted_idx = torch.argsort(abs_vals, descending=True)
        self.eigenvalues = eigenvalues[sorted_idx]
        self.eigenvectors = eigenvectors[:, sorted_idx]

    def _compute_stationary(self):
        """고유벡터에서 멈춘 분포 뽑아내기."""
        idx = torch.argmin(torch.abs(self.eigenvalues.real - 1.0))
        pi = self.eigenvectors[:, idx].real
        self.pi = torch.abs(pi) / torch.abs(pi).sum()

    def spectral_gap(self) -> float:
        """스펙트럼 틈 γ = 1 - |λ₂|."""
        lambda_2 = self.eigenvalues[1]
        return (1 - torch.abs(lambda_2)).item()

    def total_variation_distance(
        self, dist1: torch.Tensor, dist2: torch.Tensor
    ) -> float:
        """TV(μ, ν) = (1/2) Σ |μ(x) - ν(x)|."""
        return 0.5 * torch.sum(torch.abs(dist1 - dist2)).item()

    def mixing_time(
        self,
        epsilon: float = 0.25,
        max_steps: int = 10000
    ) -> Dict:
        """
        τ_mix(ε) 셈하기: max_x TV(P^n(x,·), π) ≤ ε이 되는 첫 n.
        """
        results = {
            'epsilon': epsilon,
            'mixing_time': None,
            'max_tv_over_time': []
        }

        for step in range(max_steps):
            P_n = torch.linalg.matrix_power(self.P, step)

            max_tv = 0
            for i in range(self.n_states):
                tv = self.total_variation_distance(P_n[i], self.pi)
                max_tv = max(max_tv, tv)

            results['max_tv_over_time'].append(max_tv)

            if max_tv <= epsilon and results['mixing_time'] is None:
                results['mixing_time'] = step

        return results

    def convergence_rate(self, max_steps: int = 100) -> Dict:
        """시작 상태마다의 TV 거리 흘러감."""
        results = {
            'spectral_gap': self.spectral_gap(),
            'second_eigenvalue': self.eigenvalues[1].item(),
            'theoretical_rate': torch.abs(self.eigenvalues[1]).item(),
            'distances': {}
        }

        for start_idx in range(self.n_states):
            distances = []
            dist = torch.zeros(self.n_states, dtype=self.P.dtype)
            dist[start_idx] = 1.0

            for step in range(max_steps):
                tv = self.total_variation_distance(
                    dist.float(), self.pi.float()
                )
                distances.append(tv)
                dist = dist @ self.P

            results['distances'][self.state_names[start_idx]] = distances

        return results
```

### 빠른 혼합과 느린 혼합 예제

```python
def demonstrate_mixing_analysis():
    """잘 이어진 사슬과 병목이 있는 사슬의 섞임 굶 견주기."""
    print("Mixing Time Analysis")
    print("=" * 70)

    # 빠른 섞임: 잘 이어짐
    print("\n1. Fast Mixing Chain (Well-Connected)")
    print("-" * 50)

    P_fast = torch.tensor([
        [0.4, 0.3, 0.3],
        [0.3, 0.4, 0.3],
        [0.3, 0.3, 0.4]
    ])

    analyzer_fast = MixingTimeAnalyzer(P_fast, ['A', 'B', 'C'])
    print(f"Spectral gap: {analyzer_fast.spectral_gap():.6f}")
    print(f"|λ₂|: {torch.abs(analyzer_fast.eigenvalues[1]).item():.6f}")

    mixing_fast = analyzer_fast.mixing_time(epsilon=0.01)
    print(f"Mixing time (ε=0.01): {mixing_fast['mixing_time']} steps")

    # 느린 섞임: 병목
    print("\n2. Slow Mixing Chain (Bottleneck)")
    print("-" * 50)

    P_slow = torch.tensor([
        [0.45, 0.45, 0.05, 0.05],
        [0.45, 0.45, 0.05, 0.05],
        [0.05, 0.05, 0.45, 0.45],
        [0.05, 0.05, 0.45, 0.45]
    ])

    analyzer_slow = MixingTimeAnalyzer(P_slow, ['A1', 'A2', 'B1', 'B2'])
    print(f"Spectral gap: {analyzer_slow.spectral_gap():.6f}")
    print(f"|λ₂|: {torch.abs(analyzer_slow.eigenvalues[1]).item():.6f}")

    mixing_slow = analyzer_slow.mixing_time(epsilon=0.01, max_steps=500)
    print(f"Mixing time (ε=0.01): {mixing_slow['mixing_time']} steps")

    # 비교
    print(f"\nSpectral gap ratio: "
          f"{analyzer_fast.spectral_gap() / analyzer_slow.spectral_gap():.1f}x")

demonstrate_mixing_analysis()
```

**출력:**

```
Mixing Time Analysis
======================================================================

1. Fast Mixing Chain (Well-Connected)
--------------------------------------------------
Spectral gap: 0.900000
|λ₂|: 0.100000
Mixing time (ε=0.01): 2 steps

2. Slow Mixing Chain (Bottleneck)
--------------------------------------------------
Spectral gap: 0.200000
|λ₂|: 0.800000
Mixing time (ε=0.01): 18 steps

Spectral gap ratio: 4.5x
```

---

## 8. 시각화

### 사슬 구조 그래프

```python
import matplotlib.pyplot as plt
import networkx as nx

def visualize_chain_structure(
    P: torch.Tensor,
    state_names: List[str] = None,
    title: str = "Markov Chain Structure"
):
    """서로 통하는 무리로 색칠한 방향 그래프로 사슬 그려 보기."""
    n_states = P.shape[0]
    if state_names is None:
        state_names = [f"S{i}" for i in range(n_states)]

    classifier = StateClassifier(P)
    classes = classifier.find_communicating_classes()
    state_info = classifier.classify_all_states()

    G = nx.DiGraph()
    for i in range(n_states):
        G.add_node(i, label=state_names[i])
    for i in range(n_states):
        for j in range(n_states):
            if P[i, j].item() > 0.01:
                G.add_edge(i, j, weight=P[i, j].item())

    colors = plt.cm.Set3(range(len(classes)))
    node_colors = [colors[state_info[s]['communicating_class']]
                   for s in range(n_states)]

    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42, k=2)

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=2000,
                          node_color=node_colors, alpha=0.8)

    edges = G.edges(data=True)
    edge_weights = [e[2]['weight'] for e in edges]
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True,
                          arrowsize=20, edge_color='gray',
                          width=[w * 3 for w in edge_weights],
                          alpha=0.6, connectionstyle="arc3,rad=0.1")

    labels = {i: state_names[i] for i in range(n_states)}
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=12,
                           font_weight='bold')

    edge_labels = {(i, j): f"{P[i,j]:.2f}"
                   for i, j in G.edges() if P[i, j].item() > 0.01}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax, font_size=8)

    ax.set_title(title, fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    return fig

def visualize_convergence(
    P: torch.Tensor,
    state_names: list = None,
    max_steps: int = 50
):
    """네 칸짜리 모임 그림."""
    analyzer = MixingTimeAnalyzer(P, state_names)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (0,0) 시간에 따른 TV 거리
    ax = axes[0, 0]
    conv_data = analyzer.convergence_rate(max_steps)
    for state, distances in conv_data['distances'].items():
        ax.semilogy(distances, marker='o', markersize=3,
                   label=f'Start: {state}', linewidth=2, alpha=0.7)
    rate = conv_data['theoretical_rate']
    theoretical = [rate ** n for n in range(max_steps)]
    ax.semilogy(theoretical, 'k--', linewidth=2, alpha=0.5,
               label=f'|λ₂|^n = {rate:.3f}^n')
    ax.set_xlabel('Time Step n')
    ax.set_ylabel('Total Variation Distance')
    ax.set_title('Convergence Rate (Log Scale)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (0,1) 고윳값 스펙트럼
    ax = axes[0, 1]
    eigenvalues = analyzer.eigenvalues
    ax.scatter(eigenvalues.real.numpy(), eigenvalues.imag.numpy(),
              s=200, c='blue', alpha=0.7, edgecolors='black', linewidth=2)
    ax.scatter([1], [0], s=300, c='red', marker='*', label='λ₁ = 1')
    ax.scatter([eigenvalues[1].real.item()], [eigenvalues[1].imag.item()],
              s=300, c='green', marker='*',
              label=f'λ₂ = {eigenvalues[1].real:.3f}')
    theta = torch.linspace(0, 2 * 3.14159, 100)
    ax.plot(torch.cos(theta).numpy(), torch.sin(theta).numpy(),
           'k--', alpha=0.3)
    ax.set_xlabel('Real')
    ax.set_ylabel('Imaginary')
    ax.set_title(f'Eigenvalue Spectrum (γ = {analyzer.spectral_gap():.4f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    # (1,0) 상태 분포의 흘러감
    ax = axes[1, 0]
    n_states = P.shape[0]
    sn = state_names or [f'S{i}' for i in range(n_states)]

    dist = torch.zeros(n_states, dtype=P.dtype)
    dist[0] = 1.0
    evolution = [dist.clone()]
    for _ in range(max_steps):
        dist = dist @ P
        evolution.append(dist.clone())
    evolution = torch.stack(evolution)

    for i in range(n_states):
        ax.plot(evolution[:, i].numpy(), marker='o', markersize=3,
               label=sn[i], linewidth=2, alpha=0.7)
        ax.axhline(y=analyzer.pi[i].item(), linestyle='--',
                  color=f'C{i}', alpha=0.4)
    ax.set_xlabel('Time Step n')
    ax.set_ylabel('Probability')
    ax.set_title(f'Distribution Evolution (Start: {sn[0]})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (1,1) 섞임 시간
    ax = axes[1, 1]
    mixing_data = analyzer.mixing_time(epsilon=0.25, max_steps=max_steps)
    ax.semilogy(mixing_data['max_tv_over_time'], 'b-', linewidth=2,
               label='Max TV distance')
    ax.axhline(y=0.25, color='red', linestyle='--', linewidth=2,
              label='ε = 0.25')
    if mixing_data['mixing_time'] is not None:
        ax.axvline(x=mixing_data['mixing_time'], color='green',
                  linestyle='--', linewidth=2,
                  label=f'τ_mix = {mixing_data["mixing_time"]}')
    ax.set_xlabel('Time Step n')
    ax.set_ylabel('Max TV Distance')
    ax.set_title('Mixing Time Analysis')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
```

---

## 9. 에르고드·주기·가약 사슬 예제

```python
# 보기 1: 에르고드 사슬
print("Example 1: Ergodic Chain")
print("=" * 50)

P_ergodic = torch.tensor([
    [0.5, 0.3, 0.2],
    [0.2, 0.6, 0.2],
    [0.3, 0.3, 0.4]
])

classifier = StateClassifier(P_ergodic)
result = classifier.is_ergodic()
print(f"Irreducible: {result['irreducible']}")
print(f"Aperiodic: {result['aperiodic']}")
print(f"Ergodic: {result['ergodic']}")
# → 에르고드적: 참. π이 오직 하나 있고 P^n이 모인다.

# 보기 2: 주기 사슬(주기 3)
print("\nExample 2: Periodic Chain")
print("=" * 50)

P_periodic = torch.tensor([
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0]
])

classifier_periodic = StateClassifier(P_periodic)
print(f"This chain cycles: 0 → 1 → 2 → 0 → ...")
print(f"Period of state 0: {classifier_periodic.compute_period(0)}")
result = classifier_periodic.is_ergodic()
print(f"Ergodic: {result['ergodic']} (irreducible but periodic)")
# → 에르고드적이 아니다. (줄일 수 없어) π이 오직 하나 있지만 P^n이 흔들린다.

# 보기 3: 줄일 수 있는 사슬
print("\nExample 3: Reducible Chain")
print("=" * 50)

P_reducible = torch.tensor([
    [0.5, 0.5, 0.0, 0.0],
    [0.5, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.7, 0.3],
    [0.0, 0.0, 0.4, 0.6]
])

classifier_reducible = StateClassifier(P_reducible)
classes = classifier_reducible.find_communicating_classes()
print(f"Communicating classes: {classes}")
print(f"Irreducible: {classifier_reducible.is_irreducible()}")
# → 줄일 수 있다. 멈춘 분포가 여럿 있다.
```

**출력:**

```
Example 1: Ergodic Chain
==================================================
Irreducible: True
Aperiodic: True
Ergodic: True

Example 2: Periodic Chain
==================================================
This chain cycles: 0 → 1 → 2 → 0 → ...
Period of state 0: 3
Ergodic: False (irreducible but periodic)

Example 3: Reducible Chain
==================================================
Communicating classes: [{0, 1}, {2, 3}]
Irreducible: False
```

---

## 10. 핵심 정리 간추림

| 조건·개념 | 보장과 한계 |
|---|---|
| 유한 확률행렬 | 정상분포가 적어도 하나 존재 |
| 유한 기약성 | 유일한 정상분포와 양의 재귀성 |
| 추가 비주기성 | 모든 초기분포의 같은 정상분포 수렴 |
| 기약·양의 재귀 | 시간 평균과 정상 기대값의 관계, 비주기성은 필수 아님 |
| 유한 기약·비주기 | 어떤 기하 수렴률이 존재 |
| 가역성·절대 갭 | 특정 스펙트럼 상계에 사용 |

정상성만으로 유일성·수렴·빠른 혼합까지 결론 내리지 않는다.

## 11. 차원과 계산 효율

차원·상관·다봉성·곡률은 커널의 효율에 영향을 줄 수 있지만 방법 이름만으로 보편적인 혼합 시간의 차원 지수를 정할 수는 없다. 특정 곱분포·확산 극한에서의 이동 스케일과 수락률, 한 단계의 경사 계산 비용, 전체 유효 표본당 비용은 서로 다른 분석량이다.

랜덤 워크 MH·MALA·HMC의 단일 고정 지수를 일반 목표분포의 혼합 시간으로 제시하지 않는다. 같은 관심 함수와 정확도, 경사 평가 비용을 맞추고 ESS/시간 등 실제 계산 효율을 비교한다.

## 12. 몬테카를로 오차

정상분포에서 시작하고 f의 분산 σ_f²가 유한하면 N개 평균의 정확한 분산은 다음과 같다.

$$\operatorname{Var}(\bar f_N)=\frac{\sigma_f^2}{N}\left[1+2\sum_{k=1}^{N-1}\left(1-\frac kN\right)\rho_k\right]$$

자기상관이 충분히 합산 가능하고 중심극한정리 조건을 만족하면 큰 N에서 괄호를 $\tau_f=1+2\sum_{k\geq1}\rho_k$로 근사한다. 이때 ESS≈N/τ_f, MCSE≈σ_f/√ESS다. 무한 합 공식을 유한 N의 정확한 분산식으로 쓰지 않는다.

ESS는 관심 함수마다 다르며 음의 자기상관으로 N보다 클 수도 있다. 분포 수렴이나 작은 R-hat만으로 모든 함수의 모멘트와 중심극한정리 조건이 보장되지는 않는다. 초기화 오차와 정상 상태에서도 남는 몬테카를로 오차를 나누어 평가한다.

## 13. 실무 지침

여러 체인을 충분히 다른 유효 초기값에서 시작하고 워밍업·적응 구간과 추론 표본을 구분한다. 순위 정규화·분할 R-hat, bulk·tail ESS, 궤적과 해당 알고리즘의 진단을 함께 확인한다. [Stan의 사후 분석 문서](https://mc-stan.org/docs/reference-manual/analysis.html)는 R-hat 1.01 등의 실용적 지침을 설명하지만 하나의 임계값은 수렴 증명이 아니다.

필요한 표본 수는 추정할 평균·꼬리·분위수와 허용 MCSE로 정한다. 50%를 버리거나 ESS가 특정 숫자를 넘었다는 이유만으로 멈추지 않는다. 알려지지 않은 모드를 모든 체인이 함께 놓칠 수도 있다. 정규화하지 않은 목표밀도를 계산할 수 있다는 사실만으로 임의의 복잡한 분포에서 효율적이고 정확한 샘플링이 보장되지는 않는다.

## 연습문제

1. 기약 사슬의 한 상태에 양의 자기 루프가 있으면 전체 사슬이 비주기인 이유를 설명하라. 가약 사슬의 반례를 구성하라.
2. 유한 기약 사슬의 양의 재귀성을 보여라. 기약성과 유한성으로부터 어떤 고정 단계 수 이내에 특정 상태를 방문할 확률의 양의 하한을 얻고 귀환 시간의 꼬리를 묶는 방법을 사용하라.
3. 두 그룹 사이의 전이를 약하게 만든 유효 확률행렬에서 연결 강도에 따른 혼합 시간을 분석하라. 모든 행의 합과 정상성 잔차를 확인하고 본문 분석기의 오른쪽 고유벡터 오류를 피할 계산을 설명하라.
4. 가역인 3×3 전이행렬을 선택해 절대 스펙트럼 갭과 TV 상계를 비교하라. 일반 비가역 행렬에 같은 정확한 상계를 적용할 수 없는 이유를 설명하라.
5. 목표분포 π=(0.2,0.3,0.5)를 갖는 상세 균형 커널을 구성하라. 예를 들어 각 행을 π로 둔 P를 사용하면 한 단계 후 목표분포가 된다. 이 경우의 기약성·비주기성·혼합 시간과 독립 표본 생성을 설명하라.

## 정리하며

기약성·주기·재귀성은 서로 다른 장기 성질을 설명한다. 유한 상태의 수렴 결과를 무한·연속 공간에 자동 적용하지 않고, 스펙트럼 상계에는 가역성 등의 조건을 명시한다. 실제 MCMC에서는 목표 불변성, 초기화 영향과 관심 함수의 몬테카를로 오차를 모두 확인한다.

**참고 문헌**

1. Levin, D.A., Peres, Y., & Wilmer, E.L. *Markov Chains and Mixing Times* (2nd ed.). AMS, 2017.
2. Norris, J.R. *Markov Chains*, 1-2장. Cambridge University Press, 1997.
3. Montenegro, R. & Tetali, P. "Mathematical Aspects of Mixing Times in Markov Chains." *Foundations and Trends in TCS*, 2006.
4. Diaconis, P. & Stroock, D. "Geometric Bounds for Eigenvalues of Markov Chains." *Annals of Applied Probability*, 1991.
5. Durrett, R. *Essentials of Stochastic Processes*, 1장. Springer, 2016.

---
