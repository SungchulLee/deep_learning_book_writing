# 멈춘 분포
## 들어가며

멈춘 분포는 마르코프 사슬 이론과 MCMC 표집을 잇는 중심 대상이다. 이는 사슬의 긴 눈으로 본 평형, 곧 시작 상태와 상관없이 사슬이 모여 가는 분포를 나타낸다. MCMC 얼개에서 하려는 일은 그 멈춘 분포가 주어진 과녁 $\pi$과 같은 사슬을 짓는 것이며, 그러면 사슬을 넉넉히 오래 돌려 $\pi$의 어림 표본을 얻는다.

## 정의와 성질

### 수학적 정의

확률 분포 $\pi = (\pi_0, \pi_1, \ldots, \pi_{N-1})$이 다음을 만족하면 옮김 행렬이 $P$인 마르코프 사슬의 **멈춘 분포**(불변 분포, 평형 분포)이다:

$$\pi = \pi P$$

성분별로 쓰면:

$$\pi_j = \sum_{i \in S} \pi_i P_{ij} \quad \text{for all } j \in S$$

### 왜 "멈춘"인가

사슬이 분포 $\pi$으로 시작하면, 곧 $P(X_0 = i) = \pi_i$이면 그 뒤 어느 때에도 다음이 성립한다:

$$P(X_n = j) = \sum_{i \in S} \pi_i P^{(n)}_{ij} = (\pi P^n)_j = \pi_j$$

동역학이 분포를 지킨다. 곧 한번 평형에 들면 사슬은 평형에 머문다. 고정점 방정식 $\pi = \pi P$이 이 불변함을 담는다.

### 물리적인 풀이

에르고드 사슬에서 $\pi_i$은 같은 뜻의 풀이 셋을 갖는다:

1. **긴 눈으로 본 몫**: $n \to \infty$일 때 상태 $i$에서 보내는 시간의 비율
2. **극한 확률**: 시작 상태와 상관없는 $\lim_{n \to \infty} P(X_n = i)$
3. **평균 돌아옴 때의 역수**: $T_i = \min\{n \geq 1 : X_n = i \mid X_0 = i\}$일 때 $\pi_i = 1/\mathbb{E}[T_i]$

셋째 풀이가 MCMC에 특히 쓸모 있다. 곧 멈춘 확률이 높은 상태를 더 자주 들르며(기대 돌아옴 때가 짧으며), 이것이 바로 $\pi$에서 표집할 때 우리가 바라는 바이다.

## 있음과 하나뿐임

### 있음

**정리.** 끝이 있는 마르코프 사슬마다 멈춘 분포가 적어도 하나 있다.

*증명 얼개.* $S$ 위 확률 분포의 묶음은 옹골지고($\mathbb{R}^N$에서 닫혀 있고 유계이고) 사상 $\mu \mapsto \mu P$은 이어져 있다. 브라우어 고정점 정리에 따라 고정점이 있다.

### 하나뿐임

**정리.** **쪼갤 수 없는** 마르코프 사슬에는 멈춘 분포가 꼭 하나 있다.

쪼갤 수 없음(모든 상태가 통함)은 하나뿐임에는 넉넉하지만 모임에는 그렇지 않다. 모임을 얻으려면 주기 없음이 더 필요하며, 이것이 다음 마당의 에르고드성 정리가 다루는 내용이다.

### 조건 간추림

| 조건 | $\pi$이 하나뿐인가? | $P^n \to \mathbf{1}\pi$으로 모이는가? |
|-----------|:---:|:---:|
| 쪼갤 수 없음만 | 예 | 보장되지 않음(흔들릴 수 있음) |
| 주기 없음만 | 여럿일 수 있음 | 경우에 따라 다름 |
| **에르고드**(쪼갤 수 없음 + 주기 없음) | **예** | **예** |
| 쪼갤 수 있음 | 여럿일 수 있음 | 아니오(시작에 따라 다름) |

## $\pi$을 셈하는 네 가지 방법

### 방법 1: 고유벡터 방법

$\pi P = \pi$이므로 옮겨 놓으면 $P^T \pi^T = \pi^T$이다. 곧 $\pi^T$은 고유값 $\lambda = 1$에 대한 $P^T$의 오른쪽 고유벡터이다.

**알고리즘:**

1. $P^T$의 고유값과 고유벡터를 셈한다
2. 고유값 1에 해당하는 고유벡터를 찾는다
3. 합이 1이 되도록 고르게 한다

### 방법 2: 선형 얼개

방정식 $\pi(P - I) = \mathbf{0}$은 동차 선형 얼개를 이룬다. 식 하나를 고르게 하기 제약 $\sum_i \pi_i = 1$으로 바꾸면 하나뿐인 풀이를 얻는다.

**알고리즘:**

1. 행렬 $A = P^T - I$을 만든다
2. $A$의 마지막 행을 $[1, 1, \ldots, 1]$으로 바꾼다
3. 오른쪽 변을 $b = [0, 0, \ldots, 0, 1]^T$으로 놓는다
4. $A \pi^T = b$을 푼다

### 방법 3: 거듭제곱 되풀이

에르고드 사슬에서는 아무 첫 분포에 $P$을 되풀이해 곱하면 $\pi$으로 모인다.

**알고리즘:**

1. 아무 분포 $\pi^{(0)}$으로 시작한다
2. 되풀이한다: $\pi^{(k+1)} = \pi^{(k)} P$
3. $\|\pi^{(k+1)} - \pi^{(k)}\| < \epsilon$이면 멈춘다

개념으로는 가장 단순한 방법이며 MCMC가 하는 일을 곧바로 비춘다. 곧 사슬을 돌리고 모이기를 기다리는 것이다.

### 방법 4: 흉내내기(에르고드 정리)

에르고드 사슬에서는 지시 함수의 시간 평균이 공간 평균으로 모인다.

**알고리즘:**

1. 사슬을 $T$ 걸음 흉내 낸다(태우기를 두고)
2. 상태마다 들른 횟수를 센다: $N_i = \sum_{t=0}^{T} \mathbf{1}\{X_t = i\}$
3. 어림한다: $\hat{\pi}_i = N_i / T$

이것이 실전에서 MCMC가 하는 일과 다름없다. 이 방법은 모든 MCMC 어림자를 뒷받침하는 에르고드 정리를 미리 보여 준다.

## PyTorch 구현

```python
import torch
import torch.linalg as LA
from typing import Dict, Tuple, Optional

class StationaryDistributionAnalyzer:
    """
    여러 방법으로 멈춘 분포 셈하기.

    멈춘 분포 π은 π = πP을 만족한다
    같은 말로 π^T은 고윳값이 1인 P^T의 오른쪽 고유벡터이다.
    """

    def __init__(
        self,
        transition_matrix: torch.Tensor,
        state_names: Optional[list] = None
    ):
        self.P = transition_matrix.clone().double()
        self.n_states = self.P.shape[0]
        self.state_names = state_names or [
            f"State_{i}" for i in range(self.n_states)
        ]

    def via_eigenvector(self) -> torch.Tensor:
        """
        방법 1: 고윳값이 1인 P^T의 고유벡터.

        πP = π  ⟹  P^T π^T = π^T
        """
        eigenvalues, eigenvectors = LA.eig(self.P.T)
        idx = torch.argmin(torch.abs(eigenvalues.real - 1.0))
        pi = eigenvectors[:, idx].real
        pi = torch.abs(pi)
        pi = pi / pi.sum()
        return pi.float()

    def via_linear_system(self) -> torch.Tensor:
        """
        방법 2: 고르게 하기 제약 아래에서 (P^T - I)π^T = 0 풀기.

        마지막 식을 Σπ_i = 1으로 바꾸기.
        """
        A = self.P.T - torch.eye(self.n_states, dtype=self.P.dtype)
        A[-1, :] = torch.ones(self.n_states, dtype=self.P.dtype)
        b = torch.zeros(self.n_states, dtype=self.P.dtype)
        b[-1] = 1.0
        pi = LA.solve(A, b)
        return pi.float()

    def via_power_iteration(
        self,
        max_iter: int = 1000,
        tol: float = 1e-10
    ) -> Tuple[torch.Tensor, int]:
        """
        방법 3: 모일 때까지 π^{(k+1)} = π^{(k)} P 되풀이하기.
        """
        P_n = self.P.clone()
        for n in range(1, max_iter + 1):
            P_next = P_n @ self.P
            max_diff = torch.max(torch.abs(P_next - P_n))
            if max_diff < tol:
                return P_next[0].float(), n
            P_n = P_next
        return P_n[0].float(), max_iter

    def via_simulation(
        self,
        n_steps: int = 100000,
        initial_state: int = 0,
        burn_in: int = 1000
    ) -> torch.Tensor:
        """
        방법 4: 오래 돌리는 흉내내기로 π 어림하기(에르고드 정리).

        lim_{T→∞} (1/T) Σ_{t=0}^{T} 1{X_t = j} = π_j
        """
        state_counts = torch.zeros(self.n_states)
        current_state = initial_state

        for step in range(n_steps + burn_in):
            if step >= burn_in:
                state_counts[current_state] += 1
            probs = self.P[current_state].float()
            current_state = torch.multinomial(probs, num_samples=1).item()

        return state_counts / state_counts.sum()

    def compare_all_methods(
        self,
        n_simulation_steps: int = 100000
    ) -> Dict[str, torch.Tensor]:
        """견주려고 네 방법 모두로 π 셈하기."""
        results = {}
        results['eigenvector'] = self.via_eigenvector()
        results['linear_system'] = self.via_linear_system()
        pi_power, iterations = self.via_power_iteration()
        results['power_iteration'] = pi_power
        results['power_iterations_count'] = iterations
        results['simulation'] = self.via_simulation(n_steps=n_simulation_steps)
        return results

    def verify_stationary(
        self,
        pi: torch.Tensor,
        tol: float = 1e-6
    ) -> Dict:
        """
        π이 올바른 멈춘 분포인지 확인하기.

        살피기: (1) 올바른 확률 분포인가, (2) πP = π인가.
        """
        results = {}
        results['is_non_negative'] = torch.all(pi >= -tol).item()
        results['sums_to_one'] = torch.abs(pi.sum() - 1.0).item() < tol
        pi_P = pi.double() @ self.P
        fixed_point_error = torch.max(torch.abs(pi_P - pi.double())).item()
        results['fixed_point_error'] = fixed_point_error
        results['is_stationary'] = fixed_point_error < tol
        return results
```

### 보여 주기: 방법 견주기

```python
def demonstrate_stationary_distribution():
    """세 상태 날씨 모형에서 네 방법 모두 견주기."""
    states = ['Sunny', 'Cloudy', 'Rainy']
    P = torch.tensor([
        [0.7, 0.25, 0.05],
        [0.3, 0.40, 0.30],
        [0.1, 0.40, 0.50]
    ])

    analyzer = StationaryDistributionAnalyzer(P, state_names=states)
    results = analyzer.compare_all_methods(n_simulation_steps=100000)

    print("Stationary Distribution: Method Comparison")
    print("=" * 70)

    header = f"{'State':<12}" + "".join(
        f"{m:<15}" for m in ['Eigenvector', 'Linear Sys',
                              'Power Iter', 'Simulation']
    )
    print(header)

    for i, state in enumerate(states):
        row = f"{state:<12}"
        row += f"{results['eigenvector'][i]:<15.8f}"
        row += f"{results['linear_system'][i]:<15.8f}"
        row += f"{results['power_iteration'][i]:<15.8f}"
        row += f"{results['simulation'][i]:<15.8f}"
        print(row)

    print(f"\nPower iteration converged in "
          f"{results['power_iterations_count']} iterations")

    # 확인
    verification = analyzer.verify_stationary(results['eigenvector'])
    print(f"\nVerification: ||πP - π|| = "
          f"{verification['fixed_point_error']:.2e}")

    # 해석
    pi = results['eigenvector']
    print("\nLong-run interpretation:")
    for i, state in enumerate(states):
        pct = pi[i].item() * 100
        mean_return = 1.0 / pi[i].item()
        print(f"  {state}: {pct:.2f}% of the time "
              f"(mean return time: {mean_return:.2f} days)")


demonstrate_stationary_distribution()
```

## 멈춘 분포로 모이기

### 모임 속도

에르고드 사슬에서 모임 속도는 **스펙트럼 틈**이 다스린다:

$$\gamma = 1 - |\lambda_2|$$

여기서 $\lambda_2$은 절댓값으로 둘째로 큰 $P$의 고유값이다. 때 $n$의 분포와 멈춘 분포 사이의 총 변동 거리는 지수로 사그라든다:

$$\|P^n_{i,\cdot} - \pi\|_{TV} \leq C \cdot |\lambda_2|^n$$

스펙트럼 틈이 클수록 빨리 모인다. 곧 사슬이 시작 상태를 더 빨리 "잊는다".

### 눈으로 보기

```python
import matplotlib.pyplot as plt

def visualize_convergence_to_stationary(
    P: torch.Tensor,
    state_names: list = None,
    max_steps: int = 50
):
    """분포가 멈춘 분포로 어떻게 모이는지 그려 보기."""
    n_states = P.shape[0]
    if state_names is None:
        state_names = [f"S{i}" for i in range(n_states)]

    analyzer = StationaryDistributionAnalyzer(P, state_names)
    pi_stationary = analyzer.via_eigenvector()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 왼쪽: 멈춘 분포까지의 TV 거리(로그 눈금)
    ax1 = axes[0]
    for start_state in range(n_states):
        dist = torch.zeros(n_states)
        dist[start_state] = 1.0

        distances = []
        for step in range(max_steps):
            tv_dist = 0.5 * torch.sum(
                torch.abs(dist - pi_stationary)
            ).item()
            distances.append(tv_dist)
            dist = dist @ P

        ax1.semilogy(distances, marker='o', markersize=3,
                    label=f'Start: {state_names[start_state]}',
                    linewidth=2, alpha=0.7)

    ax1.set_xlabel('Time Step n', fontsize=12)
    ax1.set_ylabel('TV Distance (log scale)', fontsize=12)
    ax1.set_title('Convergence to Stationary Distribution', fontsize=13)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 오른쪽: 상태 0에서의 성분별 흘러감
    ax2 = axes[1]
    dist = torch.zeros(n_states)
    dist[0] = 1.0

    distributions = [dist.clone()]
    for step in range(max_steps):
        dist = dist @ P
        distributions.append(dist.clone())

    distributions = torch.stack(distributions)
    for i in range(n_states):
        ax2.plot(distributions[:, i].numpy(), marker='o', markersize=3,
                label=state_names[i], linewidth=2, alpha=0.7)
        ax2.axhline(y=pi_stationary[i].item(), linestyle='--',
                   color=f'C{i}', alpha=0.4)

    ax2.set_xlabel('Time Step n', fontsize=12)
    ax2.set_ylabel('Probability', fontsize=12)
    ax2.set_title(f'Component Evolution (Start: {state_names[0]})',
                  fontsize=13)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
```

## 자세한 균형과 뒤집힘

마르코프 사슬이 분포 $\pi$에 대해 다음을 만족하면 **자세한 균형**을 만족한다:

$$\pi_i P_{ij} = \pi_j P_{ji} \quad \text{for all } i, j$$

이는 평형에서 $i$에서 $j$으로 흐르는 확률과 $j$에서 $i$으로 흐르는 확률이 같다는 말이다. 자세한 균형을 만족하는 사슬을 **뒤집을 수 있다**고 한다.

**명제.** $\pi$이 $P$과 자세한 균형을 만족하면 $\pi$은 $P$의 멈춘 분포이다.

*증명.* 자세한 균형 식을 $i$에 걸쳐 합하면:

$$\sum_i \pi_i P_{ij} = \sum_i \pi_j P_{ji} = \pi_j \sum_i P_{ji} = \pi_j$$

이것이 바로 $\pi = \pi P$이다. $\square$

자세한 균형은 멈춤에 *넉넉하지만* *꼭 있어야 하는 것*은 아니다. 그러나 MCMC 알고리즘을 짜는 으뜸 도구이다. 메트로폴리스-헤이스팅스 알고리즘(18.3절)은 과녁 분포에 대해 자세한 균형을 만족하는 옮김 알맹이를 짓는다.

## MCMC 설계와의 이음

멈춘 분포 이론이 MCMC의 밑그림을 준다:

| 이론 | MCMC에서의 쓰임 |
|--------|-----------------|
| $\pi = \pi P$(고정점) | $\pi$이 멈춘 분포가 되도록 $P$을 짠다 |
| 하나뿐임(쪼갤 수 없음) | MCMC 사슬이 모든 상태에 닿을 수 있게 한다 |
| 모임(에르고드성) | 표본이 언젠가 $\pi$을 어림함을 보장한다 |
| 자세한 균형 | 올바른 MCMC 알맹이를 짓는 으뜸 도구 |
| 스펙트럼 틈 | 사슬을 얼마나 돌릴지(태우기)를 정한다 |

## 요약

| 방법 | 셈하기 | 좋은 점 | 나쁜 점 |
|--------|-------------|------|------|
| **고유벡터** | $P^T v = v$ 뒤 고르게 하기 | 정확하고 작은 행렬에 빠름 | $N$이 크면 수치 문제 |
| **선형 얼개** | $(P^T - I)\pi^T = 0$ + 고르게 하기 | 정확하고 수치로 안정 | 행렬 분해가 필요 |
| **거듭제곱 되풀이** | $\pi^{(k+1)} = \pi^{(k)}P$ | 단순하고 알아보기 쉬움 | 스펙트럼 틈이 작으면 느림 |
| **흉내내기** | 상태 들름 횟수 세기 | 큰 상태 공간까지 감당 | 어림이고 오래 돌려야 함 |

## 참고 문헌

1. Levin, D.A., Peres, Y., & Wilmer, E.L. *Markov Chains and Mixing Times*, 1-4장. AMS, 2017.
2. Norris, J.R. *Markov Chains*, 1장. Cambridge University Press, 1997.
3. Kemeny, J.G. & Snell, J.L. *Finite Markov Chains*, 4장. Springer-Verlag, 1976.
4. Robert, C.P. & Casella, G. *Monte Carlo Statistical Methods*, 6장. Springer, 2004.

## 연습문제

1. **멈춘 분포가 여럿.** 서로 다른 멈춘 분포 둘을 갖는, 쪼갤 수 있는 마르코프 사슬을 지어라. 둘 다 $\pi = \pi P$을 만족하는지 확인하여라.

2. **자세한 균형.** 옮김 행렬 $P = \begin{pmatrix} 0.7 & 0.3 \\ 0.4 & 0.6 \end{pmatrix}$의 멈춘 분포 $\pi$을 찾고 자세한 균형이 성립하는지 확인하여라.

3. **모임 속도.** 날씨 모형의 스펙트럼 틈을 셈하여라. 총 변동 거리가 $|\lambda_2|^n$의 속도로 사그라드는지 겪어 보고 확인하여라.

4. **손님의 충실함.** 손님의 굴러감을 상태 $\{$충실, 그저 그럼, 떠남$\}$으로 본떠라. 상태마다 긴 눈으로 본 손님의 몫과, 그저 그런 손님이 충실해지기까지의 기대 시간을 셈하여라.

5. **수치 안정성.** 어떤 옮김 확률이 0에 가까운, 거의 쪼갤 수 있는 사슬에서 네 방법을 견주어라. 어느 방법이 가장 튼튼한가?
