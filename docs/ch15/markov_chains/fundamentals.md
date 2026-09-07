# 마르코프 사슬의 바탕
## 들어가며

마르코프 사슬은 모든 마르코프 사슬 몬테카를로(MCMC) 방법의 이론적 등뼈이다. 과녁 분포로 모이는 표집기를 짓기에 앞서, 마르코프 사슬이 어떻게 흘러가는지, 옮김 확률이 여러 걸음에 걸쳐 어떻게 이어지는지, 그리고 그 동역학이 행렬 하나에 어떻게 온전히 담기는지를 이해해야 한다. 이 마당은 18장의 나머지를 떠받치는 핵심 정의와 셈 도구를 세운다.

## 마르코프 성질

### 형식적 정의

상태 공간이 $S$인 이산 시간 확률 과정 $\{X_n\}_{n \geq 0}$이 모든 $n \geq 0$과 모든 상태 $i_0, i_1, \ldots, i_{n-1}, i, j \in S$에 대해 다음을 만족하면 **마르코프 사슬**이다:

$$P(X_{n+1} = j \mid X_n = i, X_{n-1} = i_{n-1}, \ldots, X_0 = i_0) = P(X_{n+1} = j \mid X_n = i)$$

이것이 **마르코프 성질**(기억 없음)이다:

> *지금 상태가 주어지면 앞날은 지난날과 조건부 독립이다.*

이 성질은 지금 상태 $X_n = i$이 $X_{n+1}$을 미리 알아보는 데 필요한 정보를 모두 담고 있다고 말한다. 자취 전체 $(X_0, X_1, \ldots, X_{n-1})$은 $X_n$이 이미 주는 것 말고 더 보태는 예측력이 없다.

### 시간에 한결같음

옮김 확률이 시간 첨자에 기대지 않으면 마르코프 사슬이 **시간에 한결같다**고 한다:

$$P(X_{n+1} = j \mid X_n = i) = P(X_1 = j \mid X_0 = i) \quad \text{for all } n \geq 0$$

이 장에서는 따로 말하지 않는 한 시간에 한결같음을 놓고 다음과 같이 쓴다:

$$P_{ij} = P(X_{n+1} = j \mid X_n = i)$$

### 상태 공간

**상태 공간** $S$은 사슬이 가질 수 있는 모든 값의 묶음이다:

| 갈래 | 상태 공간 | 보기 |
|------|------------|---------|
| 끝이 있음 | $S = \{0, 1, 2, \ldots, N-1\}$ | 신용 등급, 날씨 상태 |
| 셀 수 있게 끝없음 | $S = \mathbb{Z}^+ = \{0, 1, 2, \ldots\}$ | 줄의 길이, 무작위 걸음 |
| 이어짐 | $S = \mathbb{R}^d$ | 이어진 매개변수 공간에서의 MCMC |

18.1-18.2절은 주로 끝이 있는 상태 공간에 초점을 맞추고, 18.3절(MCMC)은 이어진 공간으로 넓힌다.

## 옮김 확률과 옮김 행렬

### 한 걸음 옮김 확률

상태 $i$에서 상태 $j$으로의 **한 걸음 옮김 확률**은 다음과 같다:

$$P_{ij} = P(X_{n+1} = j \mid X_n = i)$$

이는 두 제약을 만족해야 한다:

1. **음이 아님**: 모든 $i, j \in S$에 대해 $P_{ij} \geq 0$
2. **고르게 하기**: 모든 $i \in S$에 대해 $\sum_{j \in S} P_{ij} = 1$

옮김 확률의 행마다 다음 상태에 걸친 올바른 확률 분포를 이룬다.

### 옮김 행렬

상태가 $N$개인 끝이 있는 상태 공간에서는 옮김 확률을 $N \times N$ **옮김 행렬**(확률 행렬)로 늘어놓는다:

$$P = \begin{pmatrix}
P_{00} & P_{01} & \cdots & P_{0,N-1} \\
P_{10} & P_{11} & \cdots & P_{1,N-1} \\
\vdots & \vdots & \ddots & \vdots \\
P_{N-1,0} & P_{N-1,1} & \cdots & P_{N-1,N-1}
\end{pmatrix}$$

옮김 행렬은 **행 확률 행렬**이다. 곧 항목이 모두 음이 아니고 행마다 합이 1이다. 자리 $(i, j)$의 항목은 상태 $i$에서 상태 $j$으로 옮길 확률을 주고, 대각선 항목 $P_{ii}$은 상태 $i$에 머무를 확률을 준다.

### 확률 행렬의 성질

행렬 $P$이 (행) 확률 행렬일 때 그리고 그때만 모든 $i,j$에 대해 $P_{ij} \geq 0$이고 $P \mathbf{1} = \mathbf{1}$이다. 여기서 $\mathbf{1}$은 모두 1인 열 벡터이다. 핵심 결과는 다음과 같다:

- 확률 행렬 둘의 곱도 확률 행렬이다(곱하기에 닫혀 있다).
- $P$의 고유값은 모두 $|\lambda| \leq 1$을 만족한다.
- $\lambda_1 = 1$은 늘 고유값이며 오른쪽 고유벡터가 $\mathbf{1}$이다.

## n 걸음 옮김 확률

### 정의

**$n$ 걸음 옮김 확률**은 상태 $i$에서 꼭 $n$ 걸음에 상태 $j$에 닿을 확률이다:

$$P^{(n)}_{ij} = P(X_{n+m} = j \mid X_m = i)$$

시간에 한결같은 사슬에서는 이것이 $m$에 기대지 않는다.

### 행렬의 거듭제곱

**정리.** $n$ 걸음 옮김 확률은 옮김 행렬의 $n$제곱으로 주어진다:

$$P^{(n)}_{ij} = (P^n)_{ij}$$

여기서 $P^n = \underbrace{P \cdot P \cdots P}_{n \text{ times}}$은 보통의 행렬 곱하기이다.

*증명 얼개.* 전체 확률의 법칙에 따라:

$$P^{(n)}_{ij} = \sum_{k \in S} P^{(n-1)}_{ik} P_{kj}$$

이것이 바로 $P^{(n-1)}$과 $P$에 쓴 행렬 곱하기의 정의이다. 귀납으로 $P^{(n)} = P^n$이다. $\square$

이 결과는 몹시 쓸모 있다. 곧 여러 걸음 동역학이 모두 $P$의 거듭제곱에 담겨 있다.

## 채프먼-콜모고로프 방정식

### 진술

음이 아닌 아무 정수 $m, n$에 대해:

$$P^{(m+n)}_{ij} = \sum_{k \in S} P^{(m)}_{ik} P^{(n)}_{kj}$$

행렬 꼴로 쓰면:

$$P^{m+n} = P^m \cdot P^n$$

### 해석

상태 $i$에서 $m + n$ 걸음에 상태 $j$으로 가려면 사슬이 때 $m$에 어떤 가운데 상태 $k$을 지나야 한다. 채프먼-콜모고로프 방정식은 $(m+n)$ 걸음 옮김을 다음과 같이 쪼갠다:

$$\underbrace{i \xrightarrow{m \text{ steps}} k}_{\text{probability } P^{(m)}_{ik}} \xrightarrow{n \text{ steps}} \underbrace{j}_{\text{probability } P^{(n)}_{kj}}$$

있을 수 있는 모든 가운데 상태 $k$에 걸쳐 합한다.

## 분포의 흐름

### 첫 분포

**첫 분포** $\pi^{(0)}$은 상태마다 거기서 시작할 확률을 못 박는다:

$$\pi^{(0)}_i = P(X_0 = i)$$

### 퍼뜨리기

(행 벡터로 쓴) 첫 분포 $\pi^{(0)}$이 주어지면 때 $n$의 분포는 다음과 같다:

$$\pi^{(n)} = \pi^{(0)} P^n$$

성분별로 쓰면:

$$\pi^{(n)}_j = P(X_n = j) = \sum_{i \in S} \pi^{(0)}_i P^{(n)}_{ij}$$

이것이 옮김 행렬과 분포의 흐름을 잇는 근본 방정식이다. MCMC에서 결정적인 물음은 이것이다. *첫 분포 $\pi^{(0)}$이 무엇이든 $n \to \infty$일 때 $\pi^{(n)}$이 과녁 분포 $\pi$으로 모이는가?*

## PyTorch 구현

### 마르코프 사슬 클래스

```python
import torch
import torch.linalg as LA
from typing import List, Optional, Union, Dict, Tuple

class MarkovChain:
    """
    PyTorch로 구현한 띄엄띄엄한 시간 마르코프 사슬.

    마르코프 성질은 다음을 말한다:
    P(X_{n+1} = j | X_n = i, X_{n-1}, ..., X_0) = P(X_{n+1} = j | X_n = i)

    속성:
        P: 옮김 확률 행렬(행 확률 행렬)
        n_states: 상태의 개수
        state_names: 상태의 이름(없어도 된다)
    """

    def __init__(
        self,
        transition_matrix: torch.Tensor,
        state_names: Optional[List[str]] = None,
        validate: bool = True
    ):
        """
        마르코프 사슬 첫값 잡기.

        인수:
            transition_matrix: N×N 옮김 확률 행렬
                P[i,j] = P(X_{n+1} = j | X_n = i)
            state_names: 상태 이름의 목록(없어도 된다)
            validate: 옮김 행렬을 확인할지 여부
        """
        self.P = transition_matrix.clone()
        self.n_states = self.P.shape[0]

        if state_names is None:
            self.state_names = [f"State_{i}" for i in range(self.n_states)]
        else:
            self.state_names = state_names

        if validate:
            self._validate_transition_matrix()

    def _validate_transition_matrix(self):
        """
        P이 제대로 된 확률 행렬인지 확인하기.

        필요한 것:
        1. 정사각 행렬
        2. 성분이 모두 [0, 1] 안에 있다
        3. 행마다 합이 1이다
        """
        if self.P.shape[0] != self.P.shape[1]:
            raise ValueError(
                f"Transition matrix must be square, got shape {self.P.shape}"
            )
        if torch.any(self.P < 0):
            raise ValueError("All transition probabilities must be non-negative")
        if torch.any(self.P > 1):
            raise ValueError("All transition probabilities must be ≤ 1")

        row_sums = self.P.sum(dim=1)
        if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6):
            raise ValueError(
                f"Each row must sum to 1. Got row sums: {row_sums.tolist()}"
            )

    def step(self, current_state: int) -> int:
        """
        한 걸음 밟기: X_n = current_state일 때 X_{n+1} 표집.

        인수:
            current_state: 지금 상태의 번호

        반환값:
            다음 상태의 번호
        """
        probs = self.P[current_state]
        return torch.multinomial(probs, num_samples=1).item()

    def simulate(
        self,
        n_steps: int,
        initial_state: Optional[int] = None,
        initial_distribution: Optional[torch.Tensor] = None
    ) -> List[int]:
        """
        마르코프 사슬을 n걸음 흉내내기.

        인수:
            n_steps: 옮김의 횟수
            initial_state: 시작 상태(정했을 때)
            initial_distribution: 첫 상태를 표집할 분포

        반환값:
            들른 상태의 목록(길이 n_steps + 1)
        """
        if initial_state is not None:
            state = initial_state
        elif initial_distribution is not None:
            state = torch.multinomial(initial_distribution, num_samples=1).item()
        else:
            state = torch.randint(0, self.n_states, (1,)).item()

        trajectory = [state]
        for _ in range(n_steps):
            state = self.step(state)
            trajectory.append(state)

        return trajectory

    def get_transition_probability(
        self,
        from_state: Union[int, str],
        to_state: Union[int, str]
    ) -> float:
        """P(from_state → to_state) 얻기."""
        if isinstance(from_state, str):
            from_state = self.state_names.index(from_state)
        if isinstance(to_state, str):
            to_state = self.state_names.index(to_state)
        return self.P[from_state, to_state].item()


def create_stochastic_matrix(matrix: torch.Tensor) -> torch.Tensor:
    """
    음이 아닌 아무 행렬이나 행 확률 행렬로 바꾼다
    행마다 합이 1이 되도록 고르게 하여.
    """
    matrix = torch.relu(matrix)
    row_sums = matrix.sum(dim=1, keepdim=True)

    # 0인 행 다루기: 고른 분포 주기
    zero_rows = (row_sums == 0).squeeze()
    if zero_rows.any():
        n = matrix.shape[1]
        matrix[zero_rows] = 1.0 / n
        row_sums[zero_rows] = 1.0

    return matrix / row_sums
```

### 옮김 행렬 분석기

```python
class TransitionMatrixAnalyzer:
    """
    옮김 행렬을 살피고 여러 걸음 옮김 확률을
    셈하는 도구.
    """

    def __init__(
        self,
        transition_matrix: torch.Tensor,
        state_names: Optional[List[str]] = None
    ):
        self.P = transition_matrix.clone()
        self.n_states = self.P.shape[0]
        self.state_names = state_names or [
            f"State_{i}" for i in range(self.n_states)
        ]
        self._validate()

    def _validate(self):
        """확률 행렬의 성질 확인하기."""
        assert self.P.shape[0] == self.P.shape[1], "Matrix must be square"
        assert torch.all(self.P >= 0), "All entries must be non-negative"
        row_sums = self.P.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6), \
            "All rows must sum to 1"

    def n_step_matrix(self, n: int) -> torch.Tensor:
        """
        n걸음 옮김 행렬 P^n 셈하기.

        P^n[i,j] = P(X_n = j | X_0 = i)
        """
        if n < 0:
            raise ValueError("n must be non-negative")
        if n == 0:
            return torch.eye(self.n_states, dtype=self.P.dtype)
        return torch.linalg.matrix_power(self.P, n)

    def n_step_probability(
        self, from_state: int, to_state: int, n: int
    ) -> float:
        """P^{(n)}_{ij} 셈하기."""
        P_n = self.n_step_matrix(n)
        return P_n[from_state, to_state].item()

    def distribution_evolution(
        self,
        initial_dist: torch.Tensor,
        n_steps: int
    ) -> torch.Tensor:
        """
        π^{(k)} = π^{(0)} P^k일 때 π^{(0)}, π^{(1)}, ..., π^{(n)} 셈하기.

        반환값:
            꼴이 (n_steps+1, n_states)인 텐서
        """
        distributions = torch.zeros(n_steps + 1, self.n_states)
        distributions[0] = initial_dist

        current_dist = initial_dist.clone()
        for k in range(1, n_steps + 1):
            current_dist = current_dist @ self.P
            distributions[k] = current_dist

        return distributions

    def chapman_kolmogorov_verify(
        self, m: int, n: int, tol: float = 1e-6
    ) -> bool:
        """채프먼-콜모고로프 확인하기: P^{m+n} = P^m × P^n."""
        P_m = self.n_step_matrix(m)
        P_n = self.n_step_matrix(n)
        P_mn = self.n_step_matrix(m + n)
        return torch.allclose(P_mn, P_m @ P_n, atol=tol)
```

### 모임 분석

```python
def analyze_convergence(
    P: torch.Tensor,
    max_steps: int = 100,
    tol: float = 1e-8
) -> Dict:
    """
    n → ∞일 때 P^n의 모임 살피기.

    에르고드 사슬에서 P^n의 모든 행은
    멈춘 분포 π.
    """
    results = {
        'converged': False,
        'convergence_step': None,
        'limit_matrix': None,
        'differences': []
    }

    P_prev = P.clone()
    for step in range(1, max_steps + 1):
        P_current = P_prev @ P
        diff = torch.max(torch.abs(P_current - P_prev)).item()
        results['differences'].append(diff)

        if diff < tol:
            results['converged'] = True
            results['convergence_step'] = step
            results['limit_matrix'] = P_current
            results['stationary_distribution'] = P_current[0].clone()
            break

        P_prev = P_current

    return results
```

## 보기: 날씨 모형

상태 셋짜리 날씨 모형이 핵심 개념을 보여 준다:

```python
# 상태: 맑음, 흐림, 비
states = ["Sunny", "Cloudy", "Rainy"]

P = torch.tensor([
    [0.70, 0.25, 0.05],  # 맑음에서
    [0.30, 0.40, 0.30],  # 흐림에서
    [0.10, 0.40, 0.50]   # 비에서
])

analyzer = TransitionMatrixAnalyzer(P, state_names=states)

# 여러 걸음 옮김 확률
print("Weather Model: Multi-Step Transition Probabilities")
print("=" * 60)

for n in [1, 2, 5, 10, 50]:
    P_n = analyzer.n_step_matrix(n)
    print(f"\n{n}-Step Transition Matrix P^{n}:")
    print("-" * 40)

    header = "         " + "  ".join(f"{s:>8}" for s in states)
    print(header)

    for i, state_i in enumerate(states):
        row = f"{state_i:8s} " + "  ".join(
            f"{P_n[i,j]:.6f}" for j in range(3)
        )
        print(row)

# 정해진 시작점에서의 분포 흘러감
pi_0 = torch.tensor([1.0, 0.0, 0.0])  # 맑음에서 시작

print("\nDistribution evolution starting from Sunny:")
for n in [0, 1, 2, 5, 10, 50]:
    P_n = analyzer.n_step_matrix(n)
    pi_n = pi_0 @ P_n
    print(f"  n={n:2d}: π = [{pi_n[0]:.6f}, {pi_n[1]:.6f}, {pi_n[2]:.6f}]")

# 채프먼-콜모고로프 확인하기
for m, n in [(2, 3), (5, 5), (10, 10)]:
    holds = analyzer.chapman_kolmogorov_verify(m, n)
    print(f"Chapman-Kolmogorov P^{m+n} = P^{m} · P^{n}: {holds}")
```

$n$이 커지면 $P^n$의 모든 행이 같은 벡터로 모인다. 이것이 다음 마당에서 다룰 **멈춘 분포**이다.

## 시각화

### 옮김 행렬 열지도

```python
import matplotlib.pyplot as plt

def plot_transition_matrix(
    P: torch.Tensor,
    state_names: List[str] = None,
    title: str = "Transition Matrix"
):
    """옮김 행렬의 열지도 그림 만들기."""
    n_states = P.shape[0]
    if state_names is None:
        state_names = [f"S{i}" for i in range(n_states)]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(P.numpy(), cmap='YlOrRd', vmin=0, vmax=1)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Probability', fontsize=12)

    ax.set_xticks(range(n_states))
    ax.set_yticks(range(n_states))
    ax.set_xticklabels(state_names, fontsize=11)
    ax.set_yticklabels(state_names, fontsize=11)
    ax.set_xlabel('To State', fontsize=12)
    ax.set_ylabel('From State', fontsize=12)
    ax.set_title(title, fontsize=14)

    for i in range(n_states):
        for j in range(n_states):
            value = P[i, j].item()
            color = 'white' if value > 0.5 else 'black'
            ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                   color=color, fontsize=10, fontweight='bold')

    plt.tight_layout()
    return fig


def plot_distribution_evolution(
    distributions: torch.Tensor,
    state_names: List[str] = None,
    title: str = "Distribution Evolution"
):
    """시간에 따른 상태 분포의 흘러감 그리기."""
    n_steps, n_states = distributions.shape
    if state_names is None:
        state_names = [f"S{i}" for i in range(n_states)]

    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(n_states):
        ax.plot(range(n_steps), distributions[:, i].numpy(),
               marker='o', markersize=4, linewidth=2,
               label=state_names[i], alpha=0.8)

    ax.set_xlabel('Time Step n', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    return fig
```

## 이것이 MCMC에 왜 중요한가

여기서 세운 옮김 행렬 얼개가 MCMC를 곧바로 가능하게 한다:

| 마르코프 사슬 개념 | MCMC에서의 쓰임 |
|---------------------|------------------|
| 옮김 행렬 $P$ | MCMC 알맹이(제안 + 받아들임/물리침) |
| $n$ 걸음 분포 $\pi^{(n)}$ | $n$번째 MCMC 표본의 분포 |
| 모임 $\pi^{(n)} \to \pi$ | MCMC 표본이 과녁 분포를 어림한다 |
| 채프먼-콜모고로프 | 사슬을 여러 걸음 돌리는 것을 뒷받침한다 |

남은 핵심 물음, 곧 $\pi^{(n)}$이 *언제* 모이는지, *무엇으로* 모이는지, *얼마나 빨리* 모이는지는 다음 마당의 멈춘 분포 이론과 에르고드성 결과가 답한다.

## 요약

| 개념 | 수학 꼴 | 설명 |
|---------|------------------|-------------|
| **마르코프 성질** | $P(X_{n+1}=j \mid X_n=i, \ldots) = P(X_{n+1}=j \mid X_n=i)$ | 앞날은 지금에만 기댄다 |
| **옮김 행렬** | $P_{ij} \geq 0$, $\sum_j P_{ij} = 1$인 $P$ | 한 걸음 동역학을 모두 담은 행 확률 행렬 |
| **$n$ 걸음 확률** | $P^{(n)}_{ij} = (P^n)_{ij}$ | 행렬 거듭제곱으로 얻는 여러 걸음 옮김 |
| **채프먼-콜모고로프** | $P^{m+n} = P^m \cdot P^n$ | 가운데 상태에 걸친 쪼개기 |
| **분포의 흐름** | $\pi^{(n)} = \pi^{(0)} P^n$ | 상태 확률이 시간에 따라 흘러가는 모습 |

## 참고 문헌

1. Lawler, G.F. *Introduction to Stochastic Processes*, 1장. Chapman & Hall/CRC, 2006.
2. Norris, J.R. *Markov Chains*, 1장. Cambridge University Press, 1997.
3. Kemeny, J.G. & Snell, J.L. *Finite Markov Chains*, 3장. Springer-Verlag, 1976.
4. Horn, R.A. & Johnson, C.R. *Matrix Analysis*, 8장. Cambridge University Press, 2012.
5. Durrett, R. *Essentials of Stochastic Processes*, 1장. Springer, 2016.

## 연습문제

1. **확률 행렬의 닫힘.** $P$과 $Q$이 크기가 맞는 행 확률 행렬이면 $PQ$도 행 확률 행렬임을 증명하여라.

2. **고유값 한계.** 행 확률 행렬의 모든 고유값 $\lambda$이 $|\lambda| \leq 1$을 만족함을 보여라. (*힌트*: 게르슈고린 원 정리를 써라.)

3. **판 놀이.** 자리 6개의 둥근 판에서 말이 같은 확률로 1-3칸 앞으로 나아가는 단순한 판 놀이의 마르코프 사슬을 만들어라. 10 걸음 옮김 행렬을 셈하고 행마다 고른 분포에 가까워지는지 확인하여라.

4. **채프먼-콜모고로프 확인.** 위 날씨 모형에서 $P^{(3)}_{0,2}$(맑음에서 시작해 사흘 뒤 비 올 확률)을 $P^3$으로 곧바로, 그리고 $m=1, n=2$인 채프먼-콜모고로프 방정식으로 손수 셈하여라. 둘이 맞는지 확인하여라.

5. **금융에서의 쓰임새.** 날마다의 주가 움직임을 상태 $\{$내림, 그대로, 오름$\}$의 마르코프 사슬로 본떠라. 지난 자료에서 옮김 확률을 어림하고 거래일 20일 뒤 그 주식 상태의 분포를 셈하여라.
