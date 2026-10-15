# 정상분포

정상분포는 전이를 한 번 적용해도 변하지 않는 확률분포다. 시작 상태와 무관한 극한분포라는 해석에는 추가 조건이 필요하다. MCMC는 목표분포를 정상분포로 갖는 커널을 구성한 뒤 수렴 조건과 표본 평균의 오차를 확인한다.

## 1. 정의와 성질

### 수학적 정의

확률 분포 $\pi = (\pi_0, \pi_1, \ldots, \pi_{N-1})$이 다음을 만족하면 전이 행렬이 $P$인 마르코프 사슬의 **정상분포**(불변 분포, 평형 분포)이다:

$$\pi = \pi P$$

성분별로 쓰면:

$$\pi_j = \sum_{i \in S} \pi_i P_{ij} \quad \text{for all } j \in S$$

### 분포가 변하지 않는다는 의미

사슬이 분포 $\pi$으로 시작하면, 곧 $P(X_0 = i) = \pi_i$이면 그 뒤 어느 때에도 다음이 성립한다:

$$P(X_n = j) = \sum_{i \in S} \pi_i P^{(n)}_{ij} = (\pi P^n)_j = \pi_j$$

동역학이 분포를 지킨다. 곧 한번 평형에 들면 사슬은 평형에 머문다. 고정점 방정식 $\pi = \pi P$가 이 불변함을 담는다.

### 물리적인 풀이

에르고드 사슬에서 $\pi_i$은 같은 뜻의 풀이 셋을 갖는다:

1. **장기 몫**: $n \to \infty$일 때 상태 $i$에서 보내는 시간의 비율
2. **극한 확률**: 시작 상태와 상관없는 $\lim_{n \to \infty} P(X_n = i)$
3. **평균 귀환 시간의 역수**: $T_i = \min\{n \geq 1 : X_n = i \mid X_0 = i\}$일 때 $\pi_i = 1/\mathbb{E}[T_i]$

셋째 풀이가 MCMC에 특히 쓸모 있다. 곧 정상 확률이 높은 상태를 더 자주 들르며(기대 귀환 시간가 짧으며), 이것이 바로 $\pi$에서 표집할 때 우리가 바라는 바이다.

---

## 2. 존재와 유일성

### 존재

**정리.** 유한한 마르코프 사슬마다 정상분포가 적어도 하나 있다.

*증명 구조.* $S$ 위 확률 분포의 배치는 콤팩트하고($\mathbb{R}^N$에서 닫혀 있고 유계이고) 사상 $\mu \mapsto \mu P$은 연속이다. 브라우어 고정점 정리에 따라 고정점이 있다.

### 유일성

**정리.** 유한 기약 마르코프 사슬에는 정상분포가 유일하게 존재한다. 가산 무한 상태에서는 기약성에 양의 재귀성이 추가로 필요하다. 기약성은 유일성의 충분조건이지 필요조건은 아니다.

기약성(모든 상태가 통함)은 유일성에는 넉넉하지만 수렴에는 그렇지 않다. 수렴을 얻으려면 비주기성이 더 필요하며, 이것이 다음 절의 에르고드성 정리가 다루는 내용이다.

### 조건 간추림

| 유한 사슬의 조건 | 정상분포 유일성 | 모든 초기분포의 같은 극한 |
|---|---|---|
| 기약 | 유일 | 비주기성이 없으면 보장되지 않음 |
| 기약·비주기 | 유일 | 보장됨 |
| 가약 | 하나 또는 여러 개 | 구조에 따라 다름 |

과도상태와 비주기 닫힌 클래스 하나만 있는 가약 사슬도 모든 초기분포에서 유일한 정상분포로 수렴할 수 있다.

---

## 3. 정상분포를 계산하는 네 가지 방법

### 방법 1: 고유벡터 방법

$\pi P = \pi$가므로 옮겨 놓으면 $P^T \pi^T = \pi^T$이다. 곧 $\pi^T$은 고윳값 $\lambda = 1$에 대한 $P^T$의 오른쪽 고유벡터이다.

**알고리즘:**

1. $P^T$의 고윳값과 고유벡터를 계산한다
2. 고윳값 1에 해당하는 고유벡터를 찾는다
3. 합이 1이 되도록 정규화한다

### 방법 2: 선형방정식

방정식 $\pi(P-I)=0$에 합 1과 비음성 제약을 더한다. 정상분포가 유일한 경우 독립 방정식 하나를 정규화 조건으로 바꿔 풀 수 있다. 여러 닫힌 클래스가 있으면 선형계가 특이할 수 있어 정상분포들의 집합을 다뤄야 한다.

**알고리즘:**

1. 행렬 $A = P^T - I$을 만든다
2. $A$의 마지막 행을 $[1, 1, \ldots, 1]$으로 바꾼다
3. 오른쪽 변을 $b = [0, 0, \ldots, 0, 1]^T$으로 놓는다
4. $A \pi^T = b$을 푼다

### 방법 3: 거듭제곱 반복

에르고드 사슬에서는 아무 초기분포에 $P$를 되풀이해 곱하면 $\pi$으로 모인다.

**알고리즘:**

1. 아무 분포 $\pi^{(0)}$으로 시작한다
2. 되풀이한다: $\pi^{(k+1)} = \pi^{(k)} P$
3. $\|\pi^{(k+1)} - \pi^{(k)}\| < \epsilon$이면 멈춘다

개념으로는 가장 단순한 방법이며 MCMC가 하는 일을 곧바로 비춘다. 곧 사슬을 돌리고 모이기를 기다리는 것이다.

### 방법 4: 시간 평균 시뮬레이션

에르고드 사슬에서는 지시 함수의 시간 평균이 공간 평균으로 모인다.

**알고리즘:**

1. 사슬을 $T$ 걸음 흉내 낸다(워밍업를 두고)
2. 유지한 T개 상태의 방문 횟수를 센다: $N_i=\sum_{t=1}^{T}\mathbf1\{X_t=i\}$
3. 근사한다: $\hat{\pi}_i = N_i / T$

이것이 실전에서 MCMC가 하는 일과 다름없다. 이 방법은 모든 MCMC 추정량를 뒷받침하는 에르고드 정리를 미리 보여 준다.

---

## 4. PyTorch 구현

행렬 유효성과 정상분포의 유일성을 먼저 확인한다. 유효한 유한 확률행렬에서 정확한 고유값 1의 실수 불변 벡터라면 그 절댓값도 불변이므로 정규화해 정상분포를 얻을 수 있다. 그러나 via_eigenvector의 절댓값 처리가 근사 고유벡터나 입력 오류를 자동 복구하지는 않는다. via_linear_system은 비유일한 경우 특이할 수 있으며 via_power_iteration은 최대 반복 후에도 수렴 여부와 무관하게 첫 행을 반환한다.

verify_stationary의 is_stationary는 고정점 잔차만 확인하므로 비음성·합 1 결과도 함께 봐야 한다. CPU 배열과 float32 반환을 사용하므로 장치·자료형·허용오차를 맞춘다. 시뮬레이션은 워밍업 이후 n_steps개를 세며 n_steps>0이어야 한다.

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

### 보여 주기: 방법 비교

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

**출력:**

```
Stationary Distribution: Method Comparison
======================================================================
State       Eigenvector    Linear Sys     Power Iter     Simulation     
Sunny       0.41860464     0.41860464     0.41860765     0.41903999     
Cloudy      0.33720931     0.33720931     0.33721173     0.33811000     
Rainy       0.24418606     0.24418604     0.24418780     0.24285001     

Power iteration converged in 1000 iterations

Verification: ||πP - π|| = 6.84e-09

Long-run interpretation:
  Sunny: 41.86% of the time (mean return time: 2.39 days)
  Cloudy: 33.72% of the time (mean return time: 2.97 days)
  Rainy: 24.42% of the time (mean return time: 4.10 days)
```

---

## 5. 정상분포로 모이기

### 수렴 속도와 가역성

유한 기약·비주기 **가역** 사슬에서 $\rho=\max_{j\geq2}|\lambda_j|<1$, 절대 스펙트럼 갭 γ*=1−ρ를 사용하면 다음과 같은 총변동 상계를 얻는다.

$$\|P^n(i,\cdot)-\pi\|_{TV}\leq\tfrac12\sqrt{\pi_i^{-1}-1}\,\rho^n$$

일반적인 비가역 사슬은 비대각화 가능한 부분의 다항 인자 등이 있을 수 있어 정확히 Cρⁿ 꼴을 그대로 적용하지 않는다. 유한 기약·비주기 사슬은 어떤 ρ′<1에 대한 기하 수렴을 갖지만 상수와 속도는 구조에 의존한다. 가산 무한 사슬의 에르고드성만으로 기하 수렴이 보장되지는 않는다.

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

---

## 6. 상세 균형과 가역성

마르코프 사슬이 분포 $\pi$에 대해 다음을 만족하면 **상세 균형**을 만족한다:

$$\pi_i P_{ij} = \pi_j P_{ji} \quad \text{for all } i, j$$

이는 평형에서 $i$에서 $j$으로 흐르는 확률과 $j$에서 $i$으로 흐르는 확률이 같다는 말이다. 상세 균형을 만족하는 사슬을 **가역 사슬**이라 한다.

**명제.** $\pi$가 $P$와 상세 균형을 만족하면 $\pi$는 $P$의 정상분포다.

*증명.* 상세 균형 식을 $i$에 걸쳐 합하면:

$$\sum_i \pi_i P_{ij} = \sum_i \pi_j P_{ji} = \pi_j \sum_i P_{ji} = \pi_j$$

이것이 바로 $\pi = \pi P$가다. $\square$

상세 균형은 정상성의 충분조건이지 필요조건은 아니다. π가 양수인 상태에서 역시간 전이가 원래 전이와 같다는 가역성 해석을 갖는다. Metropolis–Hastings는 이를 이용해 목표분포에 대한 커널을 구성하지만 불변성을 보였다는 사실만으로 기약성·수렴·효율까지 보장하지 않는다.

---

## 7. MCMC 설계와의 이음

목표분포 π가 불변이 되도록 커널을 구성하고 해당 상태 공간에 맞는 도달 가능성·재귀성·주기 조건을 확인한다. 상세 균형은 유용한 충분조건이지만 비가역 MCMC도 가능하다.

분포 수렴은 초기화 오차를, 시간 평균과 자기상관은 몬테카를로 오차를 다룬다. 스펙트럼 갭을 안다고 무조건 고정된 워밍업 길이를 정할 수는 없으며 오차 목표와 초기분포·상태별 질량도 고려한다.

## 연습문제

1. **정상분포가 여럿.** 서로 다른 정상분포 둘을 갖는, 가약인 마르코프 사슬을 지어라. 둘 다 $\pi = \pi P$를 만족하는지 확인하여라.

2. **상세 균형.** 전이 행렬 $P = \begin{pmatrix} 0.7 & 0.3 \\ 0.4 & 0.6 \end{pmatrix}$의 정상분포 $\pi$를 찾고 상세 균형이 성립하는지 확인하여라.

3. **수렴 속도.** 날씨 행렬의 가역성을 먼저 확인하고 고유값과 실제 총변동 거리의 감소를 비교하라. 비가역 행렬에서 가역 사슬의 정확한 상계를 자동 적용하지 말고 차이와 근사 조건을 설명하라.

4. **손님의 충실함.** 손님의 굴러감을 상태 $\{$충실, 그저 그럼, 떠남$\}$으로 본떠라. 상태마다 장기 손님의 몫과, 그저 그런 손님이 충실해지기까지의 기대 시간을 계산하여라.

5. **수치 안정성.** 어떤 옮김 확률이 0에 가까운, 거의 가약인 사슬에서 네 방법을 비교하여라. 어느 방법이 가장 튼튼한가?

## 정리하며

정상분포는 πP=π와 확률 제약을 만족한다. 유한 기약성은 유일성을, 추가 비주기성은 모든 초기분포의 수렴을 보장한다. 가약 사슬에도 유일한 정상분포가 있을 수 있으며 시간 평균은 시점별 수렴과 다른 결과다. 계산 방법의 잔차·비음성·정규화와 미수렴 여부를 함께 확인한다.

**참고 문헌**

1. Levin, D.A., Peres, Y., & Wilmer, E.L. *Markov Chains and Mixing Times*, 1-4장. AMS, 2017.
2. Norris, J.R. *Markov Chains*, 1장. Cambridge University Press, 1997.
3. Kemeny, J.G. & Snell, J.L. *Finite Markov Chains*, 4장. Springer-Verlag, 1976.
4. Robert, C.P. & Casella, G. *Monte Carlo Statistical Methods*, 6장. Springer, 2004.
