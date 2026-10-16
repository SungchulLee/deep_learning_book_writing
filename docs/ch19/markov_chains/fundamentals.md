# 마르코프 사슬의 기초

마르코프 사슬은 현재 상태를 바탕으로 다음 상태를 생성하는 확률 과정이다. MCMC에서는 목표분포를 불변으로 유지하는 전이를 구성하고 적절한 수렴 조건을 확인한다. 먼저 유한 상태의 시간 동질적인 사슬에서 전이행렬·여러 단계 확률·초기분포의 변화를 살펴본다.

## 1. 마르코프 성질

### 형식적 정의

유한 또는 가산 상태 공간이 $S$인 이산 시간 확률 과정 $\{X_n\}_{n \geq 0}$이 모든 $n \geq 0$과 조건부 확률이 정의되는 상태 이력 $i_0,i_1,\ldots,i_{n-1},i,j\in S$에 대해 다음을 만족하면 **마르코프 사슬**이다:

$$P(X_{n+1} = j \mid X_n = i, X_{n-1} = i_{n-1}, \ldots, X_0 = i_0) = P(X_{n+1} = j \mid X_n = i)$$

이것이 **마르코프 성질**(기억 없음)이다:

> *지금 상태가 주어지면 앞날은 지난날과 조건부 독립이다.*

이 성질은 지금 상태 $X_n = i$가 $X_{n+1}$을 미리 알아보는 데 필요한 정보를 모두 담고 있다고 말한다. 자취 전체 $(X_0, X_1, \ldots, X_{n-1})$은 $X_n$이 이미 주는 것 말고 더 보태는 예측력이 없다.

### 시간 동질성

전이확률이 시간 첨자에 기대지 않으면 마르코프 사슬이 **시간 동질적이다**고 한다:

$$P(X_{n+1} = j \mid X_n = i) = P(X_1 = j \mid X_0 = i) \quad \text{for all } n \geq 0$$

이 장에서는 따로 말하지 않는 한 시간 동질성을 놓고 다음과 같이 쓴다:

$$P_{ij} = P(X_{n+1} = j \mid X_n = i)$$

### 상태 공간

**상태 공간** $S$는 사슬이 가질 수 있는 모든 값의 배치다:

| 클래스 | 상태 공간 | 보기 |
|------|------------|---------|
| 유한 | $S = \{0, 1, 2, \ldots, N-1\}$ | 신용 등급, 날씨 상태 |
| 가산 무한 | $S = \mathbb{Z}^+ = \{0, 1, 2, \ldots\}$ | 줄의 길이, 랜덤 워크 |
| 연속 | $S = \mathbb{R}^d$ | 연속 파라메터 공간에서의 MCMC |

이 페이지의 상태별 등식과 행렬 표기는 유한 또는 가산 상태에 해당한다. 연속 상태 공간에서는 한 점으로 이동할 확률 대신 집합 A에 대한 전이 커널 K(x,A)를 사용한다. MCMC의 연속 상태 이론에는 일반 상태 공간에 맞는 조건이 필요하다.

---

## 2. 전이확률과 전이 행렬

### 한 스텝 전이확률

상태 $i$에서 상태 $j$로의 **한 스텝 전이확률**은 다음과 같다:

$$P_{ij} = P(X_{n+1} = j \mid X_n = i)$$

이는 두 제약을 만족해야 한다:

1. **음이 아님**: 모든 $i, j \in S$에 대해 $P_{ij} \geq 0$
2. **정규화**: 모든 $i \in S$에 대해 $\sum_{j \in S} P_{ij} = 1$

전이확률의 행마다 다음 상태에 걸친 올바른 확률 분포를 이룬다.

### 전이 행렬

상태가 $N$개인 유한 상태 공간에서는 전이확률을 $N \times N$ **전이 행렬**(확률 행렬)로 늘어놓는다:

$$P = \begin{pmatrix}
P_{00} & P_{01} & \cdots & P_{0,N-1} \\
P_{10} & P_{11} & \cdots & P_{1,N-1} \\
\vdots & \vdots & \ddots & \vdots \\
P_{N-1,0} & P_{N-1,1} & \cdots & P_{N-1,N-1}
\end{pmatrix}$$

전이 행렬은 **행 확률 행렬**이다. 곧 항목이 모두 음이 아니고 행마다 합이 1이다. 자리 $(i, j)$의 항목은 상태 $i$에서 상태 $j$로 옮길 확률을 주고, 대각선 항목 $P_{ii}$은 상태 $i$에 머무를 확률을 준다.

### 확률 행렬의 성질

행렬 $P$가 (행) 확률 행렬일 때 그리고 그때만 모든 $i,j$에 대해 $P_{ij} \geq 0$이고 $P \mathbf{1} = \mathbf{1}$이다. 여기서 $\mathbf{1}$은 모두 1인 열 벡터이다. 핵심 결과는 다음과 같다:

- 확률 행렬 둘의 곱도 확률 행렬이다(곱하기에 닫혀 있다).
- $P$의 고윳값은 모두 $|\lambda| \leq 1$을 만족한다.
- $\lambda_1 = 1$은 늘 고윳값이며 오른쪽 고유벡터가 $\mathbf{1}$이다.

---

## 3. n 스텝 전이확률

### 정의

**$n$ 스텝 전이확률**은 상태 $i$에서 꼭 $n$ 스텝에 상태 $j$에 닿을 확률이다:

$$P^{(n)}_{ij} = P(X_{n+m} = j \mid X_m = i)$$

시간 동질적인 사슬에서는 이것이 $m$에 기대지 않는다.

### 행렬의 거듭제곱

**정리.** $n$ 스텝 전이확률은 전이 행렬의 $n$제곱으로 주어진다:

$$P^{(n)}_{ij} = (P^n)_{ij}$$

여기서 $P^n = \underbrace{P \cdot P \cdots P}_{n \text{ times}}$은 보통의 행렬 곱하기이다.

*증명 구조.* 전체 확률의 법칙에 따라:

$$P^{(n)}_{ij} = \sum_{k \in S} P^{(n-1)}_{ik} P_{kj}$$

이것이 바로 $P^{(n-1)}$과 $P$에 쓴 행렬 곱하기의 정의이다. 귀납으로 $P^{(n)} = P^n$이다. $\square$

이 결과는 몹시 쓸모 있다. 곧 여러 스텝 동역학이 모두 $P$의 거듭제곱에 담겨 있다.

---

## 4. Chapman-콜모고로프 방정식

### 진술

음이 아닌 아무 정수 $m, n$에 대해:

$$P^{(m+n)}_{ij} = \sum_{k \in S} P^{(m)}_{ik} P^{(n)}_{kj}$$

행렬 꼴로 쓰면:

$$P^{m+n} = P^m \cdot P^n$$

### 해석

상태 $i$에서 $m + n$ 스텝에 상태 $j$로 가려면 사슬이 때 $m$에 어떤 가운데 상태 $k$을 지나야 한다. Chapman-콜모고로프 방정식은 $(m+n)$ 스텝 전이을 다음과 같이 쪼갠다:

$$\underbrace{i \xrightarrow{m \text{ steps}} k}_{\text{probability } P^{(m)}_{ik}} \xrightarrow{n \text{ steps}} \underbrace{j}_{\text{probability } P^{(n)}_{kj}}$$

있을 수 있는 모든 가운데 상태 $k$에 걸쳐 합한다.

---

## 5. 분포의 변화

### 초기분포

**초기분포** $\pi^{(0)}$은 상태마다 거기서 시작할 확률을 정의한다:

$$\pi^{(0)}_i = P(X_0 = i)$$

### 분포의 전파

(행 벡터로 쓴) 초기분포 $\pi^{(0)}$이 주어지면 때 $n$의 분포는 다음과 같다:

$$\pi^{(n)} = \pi^{(0)} P^n$$

성분별로 쓰면:

$$\pi^{(n)}_j = P(X_n = j) = \sum_{i \in S} \pi^{(0)}_i P^{(n)}_{ij}$$

이것이 전이 행렬과 분포의 변화을 잇는 근본 방정식이다. MCMC에서 결정적인 물음은 이것이다. *초기분포 $\pi^{(0)}$이 무엇이든 $n \to \infty$일 때 $\pi^{(n)}$이 목표분포 $\pi$으로 모이는가?*

---

## 6. PyTorch 구현

### 마르코프 사슬 클래스

전이행렬은 비어 있지 않은 유한 실수 정방행렬이고 행마다 확률분포여야 한다. 상태 수·이름과 초기 상태·분포의 차원을 맞춘다. simulate는 초기 상태를 포함한 n_steps+1개 상태를 반환하며 initial_state와 initial_distribution을 모두 주면 전자가 우선한다. create_stochastic_matrix는 음수를 0으로 자르고 0합 행을 균등하게 만드는 임의 변환이므로 원래 추정 문제의 올바른 확률행렬을 자동으로 찾아주는 함수는 아니다.

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
        한 스텝 밟기: X_n = current_state일 때 X_{n+1} 표집.

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
        마르코프 사슬을 n스텝 흉내내기.

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

### 전이 행렬 분석기

CPU에서 새 배열·항등행렬을 생성하는 부분이 있으므로 장치와 자료형을 확인한다. 분포 변화 함수는 초기 벡터를 확률분포로 검증하지 않는다.

```python
class TransitionMatrixAnalyzer:
    """
    옮김 행렬을 살피고 여러 스텝 옮김 확률을
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
        n스텝 옮김 행렬 P^n 셈하기.

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

### 수렴 분석

다음 함수는 인접 거듭제곱의 차이로 종료한다. 참극한과의 오차 상한이나 정상분포 유일성 검사가 아니며 step=1에서 P²를 계산한다. 거의 항등인 전이행렬이나 가약 사슬에서는 차이가 작다는 이유만으로 모든 초기분포가 같은 정상분포에 가까워졌다고 할 수 없다.

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

---

## 7. 보기: 날씨 모델

상태 셋짜리 날씨 모델이 핵심 개념을 보여 준다:

```python
# 상태: 맑음, 흐림, 비
states = ["Sunny", "Cloudy", "Rainy"]

P = torch.tensor([
    [0.70, 0.25, 0.05],  # 맑음에서
    [0.30, 0.40, 0.30],  # 흐림에서
    [0.10, 0.40, 0.50]   # 비에서
])

analyzer = TransitionMatrixAnalyzer(P, state_names=states)

# 여러 스텝 옮김 확률
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

**출력:**

```
Weather Model: Multi-Step Transition Probabilities
============================================================

1-Step Transition Matrix P^1:
----------------------------------------
            Sunny    Cloudy     Rainy
Sunny    0.700000  0.250000  0.050000
Cloudy   0.300000  0.400000  0.300000
Rainy    0.100000  0.400000  0.500000

2-Step Transition Matrix P^2:
----------------------------------------
            Sunny    Cloudy     Rainy
Sunny    0.570000  0.295000  0.135000
Cloudy   0.360000  0.355000  0.285000
Rainy    0.240000  0.385000  0.375000

5-Step Transition Matrix P^5:
----------------------------------------
            Sunny    Cloudy     Rainy
Sunny    0.443070  0.330475  0.226455
Cloudy   0.409230  0.339790  0.250980
Rainy    0.389610  0.345190  0.265200

10-Step Transition Matrix P^10:
----------------------------------------
            Sunny    Cloudy     Rainy
Sunny    0.419780  0.336886  0.243334
Cloudy   0.418154  0.337333  0.244513
Rainy    0.417211  0.337593  0.245196

50-Step Transition Matrix P^50:
----------------------------------------
            Sunny    Cloudy     Rainy
Sunny    0.418605  0.337210  0.244186
Cloudy   0.418605  0.337210  0.244186
Rainy    0.418605  0.337210  0.244186

Distribution evolution starting from Sunny:
  n= 0: π = [1.000000, 0.000000, 0.000000]
  n= 1: π = [0.700000, 0.250000, 0.050000]
  n= 2: π = [0.570000, 0.295000, 0.135000]
  n= 5: π = [0.443070, 0.330475, 0.226455]
  n=10: π = [0.419780, 0.336886, 0.243334]
  n=50: π = [0.418605, 0.337210, 0.244186]
Chapman-Kolmogorov P^5 = P^2 · P^3: True
Chapman-Kolmogorov P^10 = P^5 · P^5: True
Chapman-Kolmogorov P^20 = P^10 · P^10: True
```

이 날씨 예제는 모든 전이확률이 양수이므로 기약·비주기다. 따라서 n이 커지면 Pⁿ의 모든 행이 같은 정상분포로 수렴한다. 일반적인 전이행렬에서 항상 성립하는 결론은 아니다.

---

## 8. 시각화

### 전이 행렬 열지도

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

---

## 9. 이것이 MCMC에 왜 중요한가

MCMC는 목표분포를 불변으로 유지하는 전이 커널을 구성한다. 유한 상태에서는 확률행렬로 표현하지만 연속 상태에서는 일반 커널이 필요하다. Metropolis–Hastings는 제안·수락·기각으로 이를 구성하는 한 방법이며 모든 MCMC가 같은 형식일 필요는 없다.

목표분포의 불변성만으로 실제 실행의 수렴·효율이 보장되지는 않는다. 도달 가능성·재귀성·주기와 초기조건의 영향을 확인하고, 분포의 수렴과 표본 평균의 몬테카를로 오차도 구분한다.

## 연습문제

1. **확률 행렬의 닫힘.** $P$와 $Q$가 크기가 맞는 행 확률 행렬이면 $PQ$도 행 확률 행렬임을 증명하여라.

2. **고윳값 한계.** 행 확률 행렬의 모든 고윳값 $\lambda$가 $|\lambda| \leq 1$을 만족함을 보여라. (*힌트*: 게르슈고린 원 정리를 써라.)

3. **판 놀이.** 자리 6개의 둥근 판에서 말이 같은 확률로 1~3칸 앞으로 나아가는 단순한 판 놀이의 마르코프 사슬을 만들어라. 10 스텝 전이 행렬을 계산하고 행마다 고른 분포에 가까워지는지 확인하여라.

4. **Chapman-콜모고로프 확인.** 위 날씨 모델에서 $P^{(3)}_{0,2}$(맑음에서 시작해 사흘 뒤 비 올 확률)을 $P^3$으로 곧바로, 그리고 $m=1, n=2$인 Chapman-콜모고로프 방정식으로 손수 계산하여라. 둘이 맞는지 확인하여라.

5. **자료 기반 전이 추정.** 일별 수익률을 내림·보합·오름으로 구분하는 기준을 먼저 정하고 학습 기간에서 전이 도수를 추정하라. 자료가 없는 출발 상태 행의 처리도 정의하라. 시간 동질성·마르코프 가정을 점검하고 이후 기간에서 20단계 상태 예측을 평가하라. 이는 20일 누적 수익률 분포나 투자 수익의 보장과 다르다.

## 정리하며

시간 동질적인 유한 마르코프 사슬은 행 확률행렬 P와 초기분포로 정의된다. 여러 단계 전이는 Pⁿ, 초기 행분포의 변화는 p₀Pⁿ이다. 마르코프 성질은 시간 독립성과 다르며, 불변분포 존재·분포 수렴·시간 평균은 다음 절의 조건을 구분해 적용한다.

**참고 문헌**

1. Lawler, G.F. *Introduction to Stochastic Processes*, 1장. Chapman & Hall/CRC, 2006.
2. Norris, J.R. *Markov Chains*, 1장. Cambridge University Press, 1997.
3. Kemeny, J.G. & Snell, J.L. *Finite Markov Chains*, 3장. Springer-Verlag, 1976.
4. Horn, R.A. & Johnson, C.R. *Matrix Analysis*, 8장. Cambridge University Press, 2012.
5. Durrett, R. *Essentials of Stochastic Processes*, 1장. Springer, 2016.
