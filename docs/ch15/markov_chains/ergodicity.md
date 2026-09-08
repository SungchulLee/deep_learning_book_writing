# 에르고드성

에르고드성은 마르코프 사슬이 멈춘 분포로 모임을 보장하는 성질이며, MCMC가 맞기 위한 가장 중요한 조건 하나이다. 이 마당은 사슬이 에르고드인지 가리는 갈래 나누기 장치(통하는 갈래, 쪼갤 수 없음, 주기 없음, 되돌아옴)를 세우고, 이어서 모임이 *얼마나 빠른지*를 재는 모임 정리와 섞임 시간 분석을 다룬다.

에르고드성을 이해하는 일은 MCMC를 쓰는 사람에게 꼭 필요하다. 곧 표집기가 언제 올바른지(맞음), 태우기 기간을 얼마나 돌려야 하는지(모임 속도), 섞임이 나쁜 사슬을 어떻게 진단하는지(짜임의 병목)를 알려 준다.

---

## 1. 상태 갈래 나누기

### 닿음과 통함

$n \geq 0$인 어떤 $n$에 대해 다음이 성립하면 상태 $i$에서 상태 $j$에 **닿을 수 있다**($i \to j$으로 쓴다):

$$P^{(n)}_{ij} > 0$$

$i \to j$이고 $j \to i$이면 상태 $i$과 $j$이 **통한다**($i \leftrightarrow j$으로 쓴다).

**정리.** 통함은 같음 관계이다:

1. **되돌이성**: $i \leftrightarrow i$($P^{(0)}_{ii} = 1$이므로)
2. **대칭성**: $i \leftrightarrow j$이면 $j \leftrightarrow i$이다(정의에 따라)
3. **이행성**: $i \leftrightarrow j$이고 $j \leftrightarrow k$이면 $i \leftrightarrow k$이다(길을 이어 붙인다)

### 통하는 갈래

통함이라는 같음 관계는 상태 공간 $S$을 **통하는 갈래**로 나눈다. 같은 갈래의 상태끼리는 서로 닿을 수 있다. 다른 갈래의 상태 사이에는 한쪽으로만 닿거나 아예 이음이 없을 수 있다.

### 쪼갤 수 없음

모든 상태가 통하면 마르코프 사슬을 **쪼갤 수 없다**고 한다:

$$i \leftrightarrow j \quad \text{for all } i, j \in S$$

마찬가지로 상태 공간 전체가 통하는 갈래 하나를 이룬다.

**실전 검정.** 상태가 $N$개인 끝이 있는 사슬을 쪼갤 수 없을 때 그리고 그때만 $\sum_{k=1}^{N-1} P^k$의 항목이 모두 양이다.

**MCMC에서 뜻하는 바.** MCMC 사슬을 쪼갤 수 없으면 표집기가 어느 시작점에서든 언젠가 상태 공간의 모든 구역에 닿을 수 있다. 쪼갤 수 있는 사슬은 어떤 상태를 영영 들르지 않은 채 남긴다.

---

## 2. 주기성

### 정의

상태 $i$의 **주기**는 다음과 같다:

$$d(i) = \gcd\{n \geq 1 : P^{(n)}_{ii} > 0\}$$

$d(i) = 1$이면 그 상태는 **주기가 없고** $d(i) > 1$이면 **주기가 있다**.

### 해석

주기가 $d$이라는 것은 사슬이 $d$의 배수인 때에만 상태 $i$으로 돌아올 수 있다는 뜻이다. 주기가 없는 상태는 어떤 붙박이 고리에도 매이지 않고 "들쭉날쭉한" 사이를 두고 돌아올 수 있다.

**핵심 결과.** 쪼갤 수 없는 사슬에서는 모든 상태의 주기가 같다. 그래서 사슬 자체의 주기를 말할 수 있다.

### 주기 없음의 충분조건

어떤 상태 $i$의 제 고리 확률이 양이면($P_{ii} > 0$) $d(i) = 1$이고, 사슬을 쪼갤 수 없으므로 모든 상태의 주기가 없다. 실전에서 주기 없음이 생기는 가장 흔한 길이며, 많은 MCMC 알고리즘이 "머무름" 확률을 두는 까닭이기도 하다.

### 보기

**주기 있음(주기 2):**

$$P = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} \quad \Rightarrow \quad 0 \to 1 \to 0 \to 1 \to \cdots$$

짝수인 때에만 상태 0으로 돌아온다. 주기 $= \gcd\{2, 4, 6, \ldots\} = 2$이다.

**주기 없음(제 고리):**

$$P = \begin{pmatrix} 0.5 & 0.5 \\ 1.0 & 0.0 \end{pmatrix}$$

상태 0은 때 1(제 고리)이나 때 2에 돌아올 수 있다. 주기 $= \gcd\{1, 2, 3, \ldots\} = 1$이다.

---

## 3. 되돌아옴과 스쳐 지나감

### 첫 돌아옴 때

상태 $i$으로의 **첫 돌아옴 때**는 다음과 같다:

$$T_i = \min\{n \geq 1 : X_n = i \mid X_0 = i\}$$

### 갈래 나누기

상태 $i$은 다음과 같다:

- $P(T_i < \infty \mid X_0 = i) = 1$이면 **되돌아옴**이다(반드시 돌아온다)
- $P(T_i < \infty \mid X_0 = i) < 1$이면 **스쳐 지나감**이다(영영 돌아오지 않을 수 있다)

### 같은 뜻의 성격 밝힘

상태 $i$이 되돌아옴일 때 그리고 그때만 $\sum_{n=0}^{\infty} P^{(n)}_{ii} = \infty$이고(기대 들름 횟수가 끝없고), 이 합이 끝이 있을 때 그리고 그때만 스쳐 지나감이다.

### 양의 되돌아옴과 영의 되돌아옴

되돌아옴 상태는 다시 다음으로 갈린다:

- **양의 되돌아옴**: $\mathbb{E}[T_i \mid X_0 = i] < \infty$(평균 돌아옴 때가 끝이 있다)
- **영의 되돌아옴**: $\mathbb{E}[T_i \mid X_0 = i] = \infty$(평균 돌아옴 때가 끝없다)

**끝이 있는** 상태 공간에서는 되돌아옴 상태가 모두 양의 되돌아옴이다. 영의 되돌아옴은 셀 수 있게 끝없는 사슬(이를테면 $\mathbb{Z}$ 위의 대칭 무작위 걸음)에서만 나타난다.

---

## 4. 에르고드성: 온전한 그림

### 정의

마르코프 사슬이 다음을 만족하면 **에르고드**이다:

1. **쪼갤 수 없음**: 모든 상태가 통한다
2. **주기 없음**: 주기가 1이다
3. **양의 되돌아옴**: 모든 상태의 기대 돌아옴 때가 끝이 있다

끝이 있는 상태 공간에서는 조건 1과 2가 조건 3을 뜻한다.

### 근본 모임 정리

**정리(에르고드 정리).** 옮김 행렬 $P$과 멈춘 분포 $\pi$을 갖는 에르고드 마르코프 사슬에 대해:

1. **하나뿐인** 멈춘 분포 $\pi$이 있다
2. 어떤 첫 분포에서든 모든 $i, j$에 대해 $\displaystyle\lim_{n \to \infty} P^n_{ij} = \pi_j$이다
3. $\pi_i = 1/\mathbb{E}[T_i]$이며, 여기서 $T_i$은 상태 $i$으로의 첫 돌아옴 때이다

게다가 모임은 **지수**이다:

$$|P^n_{ij} - \pi_j| \leq C \cdot \rho^n$$

여기서 $\rho = |\lambda_2| < 1$은 둘째로 큰 고유값의 크기이다.

### MCMC에서 뜻하는 바

이 정리가 MCMC의 수학 바탕이다:

| 성질 | 보장 |
|----------|-----------|
| 쪼갤 수 없음 | 표집기가 상태 공간 전체를 살펴본다 |
| 주기 없음 | 표집기가 정해진 대로 맴돌지 않는다 |
| 양의 되돌아옴 | 표집기가 모든 구역을 끝없이 다시 들른다 |
| 모임 정리 | 태우기를 넉넉히 하면 표본이 $\pi$을 어림한다 |
| 지수 속도 | $\gamma$이 스펙트럼 틈일 때 $O(1/\gamma)$ 걸음에 모인다 |

---

## 5. 스펙트럼 틈과 모임 속도

### 스펙트럼 틈

옮김 행렬 $P$의 **스펙트럼 틈**은 다음과 같다:

$$\gamma = 1 - |\lambda_2|$$

여기서 $\lambda_2$은 절댓값으로 둘째로 큰 고유값이다(가장 큰 것은 늘 $\lambda_1 = 1$이다).

스펙트럼 틈이 모임 속도를 곧바로 다스린다:

- **큰 $\gamma$**(1에 가까움): 빠른 모임, 좋은 섞임
- **작은 $\gamma$**(0에 가까움): 느린 모임, 나쁜 섞임
- $\gamma = 0$: 사슬이 모이지 않는다

### 짜임으로 풀이하기

| 사슬의 짜임 | 스펙트럼 틈 | 섞임 |
|----------------|:---:|--------|
| 잘 이어짐(모든 상태에 쉽게 닿음) | 큼 | 빠름 |
| 병목(뭉치 사이의 이음이 약함) | 작음 | 느림 |
| 거의 주기적임 | 아주 작음 | 아주 느림 |

MCMC에서 스펙트럼 틈이 작다는 것은 제안 분포를 잘못 짰다는 신호이다. 곧 사슬이 국소 구역에 "갇혀" 상태 공간 전체를 훑는 데 오래 걸린다.

---

## 6. 섞임 시간

### 정의

**섞임 시간**은 사슬이 멈춘 분포에 "가까워지기"까지 몇 걸음이 드는지를 잰다.

**총 변동 거리:**

$$\|P^n(x, \cdot) - \pi\|_{TV} = \frac{1}{2} \sum_{y \in S} |P^n_{xy} - \pi_y|$$

**$\epsilon$-섞임 시간:**

$$\tau_{\text{mix}}(\epsilon) = \min\{n : \max_x \|P^n(x, \cdot) - \pi\|_{TV} \leq \epsilon\}$$

표준으로는 $\epsilon = 1/4$을 고르며 그냥 $\tau_{\text{mix}}$으로 쓴다.

### 섞임 시간의 한계

섞임 시간은 스펙트럼 틈으로 묶인다:

$$\frac{1}{\gamma} \leq \tau_{\text{mix}} \leq \frac{\log(1/\epsilon \cdot \pi_{\min})}{\gamma}$$

여기서 $\pi_{\min} = \min_i \pi_i$이다.

### MCMC에서 섞임 시간이 왜 중요한가

MCMC 실전에서는:

- **태우기 기간**: 모으기 전에 처음 $\sim \tau_{\text{mix}}$개 표본을 버린다
- **솎아내기 사이**: 자기상관을 줄이려고 $k \sim \tau_{\text{mix}}$일 때 $k$번째마다 표본을 남긴다
- **효율**: 섞임이 빠를수록 셈 한 단위마다 실효 독립 표본이 많아진다

---

## 7. PyTorch 구현

### 상태 갈래 나누개

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

### 섞임 시간 분석기

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

### 보여 주기: 빠른 섞임과 느린 섞임

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

---

## 8. 시각화

### 사슬 짜임 그래프

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

## 9. 보기: 에르고드 사슬, 주기 사슬, 쪼갤 수 있는 사슬

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

---

## 10. 핵심 정리 간추림

| 정리 | 진술 | MCMC과의 관련 |
|---------|-----------|----------------|
| **있음** | 끝이 있는 사슬마다 멈춘 분포가 $\geq 1$개 있다 | 기본 보장 |
| **하나뿐임** | 쪼갤 수 없음 $\Rightarrow$ $\pi$이 꼭 하나 | MCMC가 알맞은 과녁으로 모인다 |
| **모임** | 에르고드 $\Rightarrow$ 모든 $i,j$에 대해 $P^n_{ij} \to \pi_j$ | 태우기 뒤 표본이 $\pi$을 어림한다 |
| **지수 속도** | $\|P^n(x,\cdot) - \pi\|_{TV} \leq C \cdot |\lambda_2|^n$ | 태우기 길이를 재어 준다 |
| **섞임 시간 한계** | $\tau_{\text{mix}} \sim 1/\gamma$ | 스펙트럼 틈이 효율을 정한다 |

---

## 11. 섞임 시간

**정의**: 사슬이 멈춘 분포에 "가까워지는" 데 걸리는 시간.

**엄밀하게**: 총 변동 섞임 시간

$$
\tau_{\text{mix}}(\epsilon) = \min\{t : \sup_x \|P^t(x, \cdot) - \pi\|_{\text{TV}} \leq \epsilon\}
$$

### 무엇이 섞임 시간에 영향을 주나?

1. **차원**: 차원이 높을수록 → 섞임이 길어진다
2. **상관**: 상관이 셀수록 → 섞임이 느려진다
3. **봉우리 여럿**: 봉우리가 갈라져 있을수록 → 섞임이 훨씬 길어진다
4. **굽음**: 평평한 구역과 가파른 구역 → 살펴보기에 영향을 준다

### 흔한 커짐새

| 방법 | 섞임 시간 |
|--------|-------------|
| 무작위 걸음 MH | $\sim d^2$ |
| 랑주뱅(MALA) | $\sim d^{5/3}$ |
| HMC | $\sim d^{5/4}$ 또는 더 좋음 |

이 커짐새 때문에 높은 차원에서 HMC가 앞선다.

---

## 12. 몬테카를로 오차

완벽히 모여도 몬테카를로에는 **통계 오차**가 있다.

### 어림자의 흩어짐

$$
\text{Var}\left[\frac{1}{N}\sum_{t=1}^N f(X^{(t)})\right] = \frac{\sigma_f^2}{N_{\text{eff}}}
$$

여기서 $\sigma_f^2 = \text{Var}_\pi[f]$이고 $N_{\text{eff}}$은 **실효 표본 크기**이다:

$$
N_{\text{eff}} = \frac{N}{1 + 2\sum_{k=1}^\infty \rho_k}
$$

여기서 $\rho_k$은 뒤짐 $k$에서의 자기상관이다.

---

## 13. 실무 지침

### 사슬 시작하기

- **무작위 첫걸음**: 흔히 잘 되지만 태우기가 길어질 수 있다
- **아는 것을 쓴 첫걸음**: MAP 어림값, 앞확률 최빈값, 또는 앞서 맞춘 값을 쓴다
- **여러 사슬**: 서로 다른 곳에서 시작해 모임을 살핀다

### 사슬 돌리기

- **태우기**: 처음 50%을 버리거나(깐깐하게) 진단을 쓴다
- **솎아내기**: $k$번째마다 표본을 남긴다(말이 갈리며 대개 필요 없다)
- **지켜보기**: $\hat{R}$, ESS, 자취 그림을 쓴다

### 사슬 멈추기

- **최소**: $\hat{R} < 1.01$이고 ESS $> 100$이 될 때까지
- **바람직함**: 미더운 추론에는 ESS $> 1000$
- **아주 중요함**: 꼬리와 분위수에는 표본이 더 필요하다

### MCMC의 기예

MCMC를 잘 쓰려면 과녁 분포를 이해하고, 알맞은 알고리즘을 고르고, 모임을 꼼꼼히 지켜보고, 결과를 알맞게 의심하며 풀이해야 한다. MCMC의 아름다움은 (고르게 하지 않은) $\tilde{\pi}(x)$의 값을 매길 수 있는 힘과 영리한 옮김 알맹이, 그리고 끈기만으로 아무리 복잡한 분포에서도 표집할 수 있다는 데 있다.

---

## 연습문제

1. **주기 알아내기.** 쪼갤 수 없는 사슬에서 어떤 상태의 제 고리 확률이 양이면($P_{ii} > 0$) 모든 상태의 주기가 없음을 보여라.

2. **끝이 있는 되돌아옴.** 끝이 있고 쪼갤 수 없는 사슬에서 모든 상태가 양의 되돌아옴임을 증명하여라. (*힌트*: 비둘기집 원리를 써라.)

3. **병목 효과.** 상태 3개씩의 뭉치 둘을 갖는 사슬을 지어라. 뭉치 사이 옮김 확률 $\alpha \in \{0.01, 0.05, 0.1, 0.3\}$을 바꿔 가며 섞임 시간을 $\alpha$의 함수로 그려라.

4. **스펙트럼 틈 셈하기.** 마음대로 고른 3×3 옮김 행렬에서 스펙트럼 틈을 해석으로 셈하고, 겪어 본 모임 속도가 $|\lambda_2|^n$과 맞는지 확인하여라.

5. **MCMC 맛보기.** 과녁 분포 $\pi = (0.2, 0.3, 0.5)$이 주어졌을 때, $\pi$을 멈춘 분포로 갖고 자세한 균형을 만족하는 옮김 행렬 $P$을 지어라. 에르고드성을 확인하고 섞임 시간을 셈하여라.

## 정리하며

| 성질 | 정의 | 뜻하는 바 |
|----------|------------|-------------|
| **닿음** | $i \to j$: $\exists n,\, P^{(n)}_{ij} > 0$ | $i$에서 $j$에 닿을 수 있다 |
| **통함** | $i \leftrightarrow j$ | 서로 닿을 수 있다 |
| **쪼갤 수 없음** | 통하는 갈래가 하나 | 상태 공간 전체를 살펴본다 |
| **주기 없음** | 주기 $= 1$ | 정해진 고리가 없다 |
| **되돌아옴** | $P(\text{돌아옴}) = 1$ | 반드시 다시 들른다 |
| **스쳐 지나감** | $P(\text{돌아옴}) < 1$ | 영영 다시 들르지 않을 수 있다 |
| **에르고드** | 쪼갤 수 없음 + 주기 없음 | 하나뿐인 $\pi$, 모임 보장 |
| **스펙트럼 틈** | $\gamma = 1 - |\lambda_2|$ | 모임 속도를 다스린다 |
| **섞임 시간** | 총 변동이 $\leq \epsilon$이 될 때까지의 걸음 | 실전 태우기 잣대 |

**참고 문헌**

1. Levin, D.A., Peres, Y., & Wilmer, E.L. *Markov Chains and Mixing Times* (2nd ed.). AMS, 2017.
2. Norris, J.R. *Markov Chains*, 1-2장. Cambridge University Press, 1997.
3. Montenegro, R. & Tetali, P. "Mathematical Aspects of Mixing Times in Markov Chains." *Foundations and Trends in TCS*, 2006.
4. Diaconis, P. & Stroock, D. "Geometric Bounds for Eigenvalues of Markov Chains." *Annals of Applied Probability*, 1991.
5. Durrett, R. *Essentials of Stochastic Processes*, 1장. Springer, 2016.

---
