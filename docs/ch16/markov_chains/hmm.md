# 숨은 마르코프 모형

지금까지 살펴본 마르코프 사슬에서는 상태 $X_n$을 곧바로 관측한다. 그러나 실제 세상의 여러 얼개에서는 바탕 상태가 **숨어** 있고 우리는 그것에 기댄 시끄러운 신호만 본다. **숨은 마르코프 모형(HMM)**은 숨은 마르코프 사슬과 관측 모형을 묶어 이를 엄밀하게 담는다.

HMM은 마르코프 사슬 이론과 통계 추론을 잇는다. 곧 숨은 사슬이 시간의 짜임을 주고, 관측 모형은 관측 자료에서 숨은 상태를 되찾는 추론 문제를 만든다. 그래서 HMM은 더 일반적인 숨은 변수 모형의 추론을 다루는 MCMC 방법(18.3절)으로 가는 자연스러운 디딤돌이 된다.

---

## 1. 수학적 틀

### 모형의 정의

HMM은 세 부분으로 이루어진다:

1. **숨은 상태 과정** $\{Z_t\}_{t=1}^T$ — 옮김 행렬이 $A$인 마르코프 사슬:

$$A_{ij} = P(Z_t = j \mid Z_{t-1} = i)$$

2. **관측 과정** $\{X_t\}_{t=1}^T$ — 숨은 상태가 주어지면 조건부 독립:

$$P(X_t = x \mid Z_t = k) = B_k(x)$$

여기서 $B_k$은 상태 $k$의 **방출 분포**이다.

3. **첫 분포** $\boldsymbol{\pi}$:

$$\pi_k = P(Z_1 = k)$$

### 조건부 독립의 짜임

숨은 상태 $\mathbf{z} = (z_1, \ldots, z_T)$과 관측 $\mathbf{x} = (x_1, \ldots, x_T)$의 결합 확률은 다음처럼 쪼개진다:

$$P(\mathbf{z}, \mathbf{x}) = \pi_{z_1} B_{z_1}(x_1) \prod_{t=2}^{T} A_{z_{t-1}, z_t} B_{z_t}(x_t)$$

### 근본이 되는 세 문제

| 문제 | 물음 | 알고리즘 |
|---------|----------|-----------|
| **값 매기기** | $P(\mathbf{x} \mid \theta)$ — 관측의 가능도는? | 앞 알고리즘 |
| **풀어내기** | $\arg\max_{\mathbf{z}} P(\mathbf{z} \mid \mathbf{x}, \theta)$ — 가장 그럴듯한 숨은 차례는? | 비터비 알고리즘 |
| **배우기** | $\arg\max_\theta P(\mathbf{x} \mid \theta)$ — 가장 좋은 모형 매개변수는? | 바움-웰치(EM) |

---

## 2. 앞뒤 알고리즘

### 앞 변수

**앞 변수** $\alpha_t(j) = P(X_1 = x_1, \ldots, X_t = x_t, Z_t = j)$은 처음 $t$개의 방출을 관측하고 때 $t$에 숨은 상태 $j$에 있을 결합 확률이다.

**되돌이:**

$$\alpha_1(j) = \pi_j B_j(x_1)$$

$$\alpha_t(j) = \left[\sum_{i=1}^{K} \alpha_{t-1}(i) A_{ij}\right] B_j(x_t), \quad t = 2, \ldots, T$$

관측 전체의 가능도는 $P(\mathbf{x}) = \sum_{j=1}^{K} \alpha_T(j)$이다.

**복잡도:** 있을 수 있는 상태 차례를 마구잡이로 다 세면 $O(K^T)$인데 견주어 $O(K^2 T)$이다.

### 뒤 변수

**뒤 변수** $\beta_t(i) = P(X_{t+1}, \ldots, X_T \mid Z_t = i)$은 다음을 만족한다:

$$\beta_T(i) = 1, \qquad \beta_t(i) = \sum_{j=1}^{K} A_{ij} B_j(x_{t+1}) \beta_{t+1}(j)$$

### 뒤확률 상태 확률

앞 변수와 뒤 변수를 합치면:

$$\gamma_t(j) = P(Z_t = j \mid \mathbf{x}) = \frac{\alpha_t(j) \beta_t(j)}{P(\mathbf{x})}$$

$$\xi_t(i, j) = P(Z_t = i, Z_{t+1} = j \mid \mathbf{x}) = \frac{\alpha_t(i) A_{ij} B_j(x_{t+1}) \beta_{t+1}(j)}{P(\mathbf{x})}$$

---

## 3. 비터비 알고리즘

비터비 알고리즘은 로그 공간에서 동적 계획법으로 가장 그럴듯한 숨은 상태 차례를 찾는다.

$\delta_t(j) = \max_{z_1, \ldots, z_{t-1}} P(z_1, \ldots, z_{t-1}, Z_t = j, x_1, \ldots, x_t)$으로 정한다.

**되돌이:**

$$\delta_1(j) = \pi_j B_j(x_1), \qquad \delta_t(j) = \max_{i} [\delta_{t-1}(i) A_{ij}] \cdot B_j(x_t)$$

$z_T^* = \arg\max_j \delta_T(j)$에서 **거슬러 가면** 가장 좋은 길이 되살아난다.

---

## 4. 바움-웰치 알고리즘

바움-웰치는 HMM을 위한 EM이다. 되풀이마다:

**E-걸음:** 앞뒤 알고리즘으로 $\gamma_t(j)$과 $\xi_t(i,j)$을 셈한다.

**M-걸음:** 매개변수를 새로 고친다:

$$\hat{\pi}_j = \gamma_1(j), \qquad \hat{A}_{ij} = \frac{\sum_{t=1}^{T-1} \xi_t(i, j)}{\sum_{t=1}^{T-1} \gamma_t(i)}, \qquad \hat{B}_j(v) = \frac{\sum_{t : x_t = v} \gamma_t(j)}{\sum_{t=1}^{T} \gamma_t(j)}$$

---

## 5. PyTorch 구현

```python
import torch
from typing import Dict, List, Tuple, Optional

class HiddenMarkovModel:
    """
    띄엄띄엄한 내보냄을 갖는 숨은 마르코프 모형.

    성분:
    - A: 숨은 상태의 K×K 옮김 행렬
    - B: K×V 내보냄 행렬(B[k,v] = P(obs=v | state=k))
    - pi: K차원 첫 상태 분포
    """

    def __init__(
        self,
        transition_matrix: torch.Tensor,
        emission_matrix: torch.Tensor,
        initial_distribution: torch.Tensor,
        state_names: Optional[List[str]] = None,
        obs_names: Optional[List[str]] = None
    ):
        self.A = transition_matrix.clone().double()
        self.B = emission_matrix.clone().double()
        self.pi = initial_distribution.clone().double()
        self.K = self.A.shape[0]
        self.V = self.B.shape[1]
        self.state_names = state_names or [f"S{i}" for i in range(self.K)]
        self.obs_names = obs_names or [f"O{i}" for i in range(self.V)]

    def forward_algorithm(
        self, observations: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """
        앞쪽 변수 α_t(j)과 log P(x) 셈하기.

        반환값:
            (alpha, log_likelihood), 여기서 alpha은 (T, K)
        """
        T = len(observations)
        alpha = torch.zeros(T, self.K, dtype=torch.float64)

        alpha[0] = self.pi * self.B[:, observations[0]]
        for t in range(1, T):
            alpha[t] = (alpha[t-1] @ self.A) * self.B[:, observations[t]]

        log_likelihood = torch.log(alpha[-1].sum()).item()
        return alpha, log_likelihood

    def backward_algorithm(
        self, observations: torch.Tensor
    ) -> torch.Tensor:
        """뒤쪽 변수 β_t(i) 셈하기. (T, K) 텐서를 돌려준다."""
        T = len(observations)
        beta = torch.zeros(T, self.K, dtype=torch.float64)
        beta[-1] = 1.0

        for t in range(T - 2, -1, -1):
            beta[t] = self.A @ (self.B[:, observations[t+1]] * beta[t+1])

        return beta

    def posterior_states(
        self, observations: torch.Tensor
    ) -> torch.Tensor:
        """γ_t(j) = P(Z_t = j | x) 셈하기. (T, K) 텐서를 돌려준다."""
        alpha, _ = self.forward_algorithm(observations)
        beta = self.backward_algorithm(observations)
        gamma = alpha * beta
        return gamma / gamma.sum(dim=1, keepdim=True)

    def viterbi(
        self, observations: torch.Tensor
    ) -> Tuple[List[int], float]:
        """
        가장 그럴듯한 숨은 상태 늘어놓음 찾기(로그 공간).

        반환값:
            (best_path, log_probability)
        """
        T = len(observations)
        log_A = torch.log(self.A + 1e-300)
        log_B = torch.log(self.B + 1e-300)
        log_pi = torch.log(self.pi + 1e-300)

        delta = torch.zeros(T, self.K, dtype=torch.float64)
        psi = torch.zeros(T, self.K, dtype=torch.long)

        delta[0] = log_pi + log_B[:, observations[0]]

        for t in range(1, T):
            for j in range(self.K):
                scores = delta[t-1] + log_A[:, j]
                psi[t, j] = scores.argmax()
                delta[t, j] = scores.max() + log_B[j, observations[t]]

        path = [0] * T
        path[-1] = delta[-1].argmax().item()
        log_prob = delta[-1].max().item()

        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]].item()

        return path, log_prob

    def baum_welch(
        self,
        observations: torch.Tensor,
        max_iter: int = 100,
        tol: float = 1e-6
    ) -> Dict:
        """매개변수 어림을 위한 바움-웰치(EM)."""
        T = len(observations)
        log_likelihoods = []

        for iteration in range(max_iter):
            # E 걸음
            alpha, ll = self.forward_algorithm(observations)
            beta = self.backward_algorithm(observations)
            log_likelihoods.append(ll)

            if iteration > 0 and abs(ll - log_likelihoods[-2]) < tol:
                break

            gamma = alpha * beta
            gamma = gamma / gamma.sum(dim=1, keepdim=True)

            xi = torch.zeros(T - 1, self.K, self.K, dtype=torch.float64)
            for t in range(T - 1):
                numerator = (
                    alpha[t].unsqueeze(1) * self.A
                    * self.B[:, observations[t+1]].unsqueeze(0)
                    * beta[t+1].unsqueeze(0)
                )
                xi[t] = numerator / numerator.sum()

            # M 걸음
            self.pi = gamma[0]
            self.A = xi.sum(dim=0) / gamma[:-1].sum(dim=0).unsqueeze(1)

            for v in range(self.V):
                mask = (observations == v).double()
                self.B[:, v] = (gamma * mask.unsqueeze(1)).sum(dim=0)
            self.B = self.B / self.B.sum(dim=1, keepdim=True)

        return {
            'A': self.A, 'B': self.B, 'pi': self.pi,
            'log_likelihoods': log_likelihoods,
            'iterations': len(log_likelihoods)
        }

    def simulate(self, n_steps: int) -> Tuple[List[int], List[int]]:
        """숨은 상태와 관측 흉내내기."""
        states, observations = [], []
        state = torch.multinomial(self.pi.float(), 1).item()

        for t in range(n_steps):
            states.append(state)
            obs = torch.multinomial(self.B[state].float(), 1).item()
            observations.append(obs)
            state = torch.multinomial(self.A[state].float(), 1).item()

        return states, observations
```

---

## 6. 쓰임새: 시장 국면 찾기

```python
def demonstrate_hmm_regime_detection():
    """
    관측한 값 움직임으로 강세장과 약세장 국면 알아내기.
    """
    print("HMM: Market Regime Detection")
    print("=" * 70)

    state_names = ['Bull', 'Bear']
    obs_names = ['Up', 'Flat', 'Down']

    A = torch.tensor([
        [0.95, 0.05],   # 강세장: 95% 이어짐
        [0.10, 0.90]    # 약세장: 90% 이어짐
    ])

    B = torch.tensor([
        [0.60, 0.30, 0.10],  # 강세장: 대체로 오름
        [0.15, 0.25, 0.60]   # 약세장: 대체로 내림
    ])

    pi = torch.tensor([0.7, 0.3])
    hmm = HiddenMarkovModel(A, B, pi, state_names, obs_names)

    # 흉내내고 풀어내기
    true_states, observations = hmm.simulate(200)
    obs_tensor = torch.tensor(observations)

    decoded_states, log_prob = hmm.viterbi(obs_tensor)
    accuracy = sum(t == d for t, d in zip(true_states, decoded_states))
    accuracy /= len(true_states)

    print(f"Viterbi decoding accuracy: {accuracy:.1%}")
    print(f"Log-likelihood: {log_prob:.2f}")

    # 뒤확률 상태 확률
    gamma = hmm.posterior_states(obs_tensor)

    # 국면이 이어지는 기댓값 길이
    bull_duration = 1.0 / (1.0 - A[0, 0].item())
    bear_duration = 1.0 / (1.0 - A[1, 1].item())
    print(f"\nExpected regime durations:")
    print(f"  Bull: {bull_duration:.1f} periods")
    print(f"  Bear: {bear_duration:.1f} periods")

demonstrate_hmm_regime_detection()
```

---

## 7. 흡수 HMM과 신용 위험

숨은 마르코프 사슬에 **흡수 상태**가 있으면 그 모형은 언젠가 끝 상태에 자리 잡는 얼개를 담는다. 신용 등급의 옮겨 감이 금융의 대표적인 보기이다. 등급은 시간에 따라 확률적으로 옮겨 가며 부도가 흡수 상태이다.

### 흡수 사슬 분석

```python
class AbsorbingMarkovChain:
    """
    흡수 마르코프 사슬 살피기.

    정준 꼴: P = [[Q, R], [0, I]]
    핵심 결과:
    - 바탕 행렬: N = (I - Q)^{-1}
    - 흡수까지의 기댓값 시간: t = N·1
    - 흡수 확률: B = N·R
    """

    def __init__(
        self,
        transition_matrix: torch.Tensor,
        state_names: Optional[List[str]] = None
    ):
        self.P = transition_matrix.clone().double()
        self.n_states = self.P.shape[0]
        self.state_names = state_names or [
            f"State_{i}" for i in range(self.n_states)
        ]
        self._classify_states()
        self._build_canonical_form()

    def _classify_states(self):
        """흡수 상태(P[i,i]=1)와 지나가는 상태 가려내기."""
        self.absorbing_indices = []
        self.transient_indices = []
        for i in range(self.n_states):
            if torch.isclose(self.P[i, i],
                           torch.tensor(1.0, dtype=self.P.dtype)):
                other = self.P[i, :i].sum() + self.P[i, i+1:].sum()
                if torch.isclose(other,
                               torch.tensor(0.0, dtype=self.P.dtype)):
                    self.absorbing_indices.append(i)
                    continue
            self.transient_indices.append(i)

        self.n_transient = len(self.transient_indices)
        self.n_absorbing = len(self.absorbing_indices)
        self.transient_names = [self.state_names[i]
                                for i in self.transient_indices]
        self.absorbing_names = [self.state_names[i]
                                for i in self.absorbing_indices]

    def _build_canonical_form(self):
        """Q과 R 부분 행렬 뽑아내기."""
        reordered = self.transient_indices + self.absorbing_indices
        P_c = self.P[torch.tensor(reordered)][:, torch.tensor(reordered)]
        t = self.n_transient
        self.Q = P_c[:t, :t]
        self.R = P_c[:t, t:]

    def fundamental_matrix(self) -> torch.Tensor:
        """N = (I - Q)^{-1}."""
        I = torch.eye(self.n_transient, dtype=self.Q.dtype)
        self.N = torch.linalg.inv(I - self.Q)
        return self.N

    def expected_absorption_time(self) -> Dict[str, float]:
        """t = N·1."""
        if not hasattr(self, 'N'):
            self.fundamental_matrix()
        ones = torch.ones(self.n_transient, 1, dtype=self.N.dtype)
        t = self.N @ ones
        return {n: t[i, 0].item()
                for i, n in enumerate(self.transient_names)}

    def absorption_probabilities(self) -> Dict[str, Dict[str, float]]:
        """B = N·R."""
        if not hasattr(self, 'N'):
            self.fundamental_matrix()
        B = self.N @ self.R
        result = {}
        for i, tn in enumerate(self.transient_names):
            result[tn] = {an: B[i, j].item()
                          for j, an in enumerate(self.absorbing_names)}
        return result

    def variance_absorption_time(self) -> Dict[str, float]:
        """Var[T_i] = [(2N - I)·t]_i - t_i^2."""
        if not hasattr(self, 'N'):
            self.fundamental_matrix()
        I = torch.eye(self.n_transient, dtype=self.N.dtype)
        ones = torch.ones(self.n_transient, 1, dtype=self.N.dtype)
        t = self.N @ ones
        var = (2 * self.N - I) @ t - t ** 2
        return {n: var[i, 0].item()
                for i, n in enumerate(self.transient_names)}
```

### 신용 등급의 옮겨 감

```python
class CreditRatingModel:
    """
    흡수 마르코프 사슬로 본 신용 등급 옮김.
    부도(D)가 흡수 상태이다.
    """

    def __init__(self, transition_matrix: torch.Tensor, ratings: List[str]):
        self.P = transition_matrix.clone().double()
        self.ratings = ratings
        self.n_ratings = len(ratings)
        self.default_idx = self.n_ratings - 1

    def cumulative_default_prob(
        self, initial_rating: str, max_horizon: int = 10
    ) -> torch.Tensor:
        """P(시간 t까지 부도 | Rating_0 = initial_rating)."""
        idx = self.ratings.index(initial_rating)
        cum_probs = torch.zeros(max_horizon)
        for t in range(1, max_horizon + 1):
            P_t = torch.linalg.matrix_power(self.P, t)
            cum_probs[t-1] = P_t[idx, self.default_idx]
        return cum_probs

    def credit_var(
        self, portfolio: Dict[str, float], horizon: int,
        lgd: float = 0.6, n_simulations: int = 10000
    ) -> Dict:
        """몬테카를로로 구하는 신용 위험 가치."""
        losses = []
        for _ in range(n_simulations):
            total_loss = 0
            for rating, exposure in portfolio.items():
                current = self.ratings.index(rating)
                for t in range(horizon):
                    probs = self.P[current].float()
                    current = torch.multinomial(probs, 1).item()
                    if current == self.default_idx:
                        total_loss += exposure * lgd
                        break
            losses.append(total_loss)
        losses = torch.tensor(losses)
        return {
            'mean_loss': losses.mean().item(),
            'var_95': torch.quantile(losses, 0.95).item(),
            'var_99': torch.quantile(losses, 0.99).item(),
            'cvar_95': losses[
                losses >= torch.quantile(losses, 0.95)
            ].mean().item()
        }

def demonstrate_credit_transitions():
    """부도 확률과 VaR을 갖는 신용 등급 모형."""
    print("\nCredit Rating Transition Model")
    print("=" * 70)

    ratings = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'D']
    P = torch.tensor([
        [0.91, 0.08, 0.01, 0.00, 0.00, 0.00, 0.00, 0.00],  # AAA
        [0.01, 0.90, 0.08, 0.01, 0.00, 0.00, 0.00, 0.00],  # AA
        [0.00, 0.02, 0.91, 0.05, 0.01, 0.01, 0.00, 0.00],  # A
        [0.00, 0.00, 0.04, 0.89, 0.05, 0.01, 0.01, 0.00],  # BBB
        [0.00, 0.00, 0.00, 0.06, 0.83, 0.08, 0.02, 0.01],  # BB
        [0.00, 0.00, 0.00, 0.00, 0.06, 0.82, 0.08, 0.04],  # B
        [0.00, 0.00, 0.00, 0.00, 0.01, 0.06, 0.65, 0.28],  # CCC
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00],  # D
    ])

    model = CreditRatingModel(P, ratings)

    # 쌓인 부도 확률
    print("\nCumulative Default Probabilities:")
    print("-" * 50)
    header = "Rating  " + "  ".join(f"Year {t}" for t in range(1, 6))
    print(header)
    for rating in ['AAA', 'BBB', 'B', 'CCC']:
        cum_pds = model.cumulative_default_prob(rating, 5)
        row = f"{rating:6}  " + "  ".join(f"{pd:6.2%}" for pd in cum_pds)
        print(row)

    # 흡수 사슬 살피기
    chain = AbsorbingMarkovChain(P, state_names=ratings)
    times = chain.expected_absorption_time()
    print("\nExpected Years to Default (from transient states):")
    for state, t in times.items():
        print(f"  {state}: {t:.1f} years")

    # 포트폴리오 VaR
    portfolio = {
        'AAA': 10e6, 'AA': 25e6, 'A': 35e6,
        'BBB': 20e6, 'BB': 8e6, 'B': 2e6
    }
    var_results = model.credit_var(portfolio, horizon=1)
    print(f"\nPortfolio Credit VaR (1-year, LGD=60%):")
    print(f"  Expected Loss: ${var_results['mean_loss']:,.0f}")
    print(f"  VaR (95%):     ${var_results['var_95']:,.0f}")
    print(f"  VaR (99%):     ${var_results['var_99']:,.0f}")
    print(f"  CVaR (95%):    ${var_results['cvar_95']:,.0f}")

demonstrate_credit_transitions()
```

### 노름꾼의 파산 보기

고전적인 노름꾼의 파산 문제가 흡수 사슬 분석을 보여 준다:

```python
def demonstrate_gamblers_ruin():
    """흡수 마르코프 사슬로 본 노름꾼의 파산."""
    print("\nGambler's Ruin (Target = \$4)")
    print("=" * 70)

    states = ['$0 (Broke)', '\$1', '\$2', '\$3', '\$4 (Win)']
    P = torch.tensor([
        [1.0, 0.0, 0.0, 0.0, 0.0],  # \$0: 흡수
        [0.5, 0.0, 0.5, 0.0, 0.0],  # \$1
        [0.0, 0.5, 0.0, 0.5, 0.0],  # \$2
        [0.0, 0.0, 0.5, 0.0, 0.5],  # \$3
        [0.0, 0.0, 0.0, 0.0, 1.0]   # \$4: 흡수
    ])

    chain = AbsorbingMarkovChain(P, state_names=states)

    times = chain.expected_absorption_time()
    probs = chain.absorption_probabilities()
    variances = chain.variance_absorption_time()

    for state in chain.transient_names:
        print(f"\nStarting from {state}:")
        print(f"  E[steps to end]: {times[state]:.2f}")
        print(f"  Std[steps]:      {variances[state]**0.5:.2f}")
        for abs_state in chain.absorbing_names:
            print(f"  P(end at {abs_state}): {probs[state][abs_state]:.4f}")

demonstrate_gamblers_ruin()
```

---

## 8. 국면 전환 수익률 모형

HMM에 이어진 방출을 합치면 계량 금융에서 널리 쓰는 **국면 전환 모형**이 된다:

```python
import numpy as np

class RegimeSwitchingModel:
    """
    국면 바뀜 모형: 숨은 마르코프 사슬이
    국면마다 다른 수익 분포를 이끈다.

    r_t | S_t = k ~ N(μ_k, σ_k²)
    """

    def __init__(
        self,
        transition_matrix: torch.Tensor,
        regime_means: torch.Tensor,
        regime_stds: torch.Tensor,
        regime_names: List[str] = None
    ):
        self.P = transition_matrix.clone()
        self.n_regimes = self.P.shape[0]
        self.means = regime_means.clone()
        self.stds = regime_stds.clone()
        self.regime_names = regime_names or [
            f"Regime_{i}" for i in range(self.n_regimes)
        ]
        self._compute_stationary()

    def _compute_stationary(self):
        eigenvalues, eigenvectors = torch.linalg.eig(self.P.T)
        idx = torch.argmin(torch.abs(eigenvalues.real - 1.0))
        pi = eigenvectors[:, idx].real
        self.stationary = torch.abs(pi) / torch.abs(pi).sum()

    def simulate(
        self, n_periods: int, initial_regime: int = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """수익과 국면 이름표 흉내내기."""
        if initial_regime is None:
            initial_regime = torch.multinomial(
                self.stationary.float(), 1
            ).item()

        regimes = torch.zeros(n_periods, dtype=torch.long)
        returns = torch.zeros(n_periods)
        current = initial_regime

        for t in range(n_periods):
            regimes[t] = current
            returns[t] = torch.normal(self.means[current],
                                       self.stds[current])
            current = torch.multinomial(
                self.P[current].float(), 1
            ).item()

        return returns, regimes

    def unconditional_moments(self) -> Dict:
        """E[r] = Σ π_k μ_k, Var[r] = Σ π_k(σ_k² + μ_k²) - E[r]²."""
        mean = (self.stationary * self.means).sum()
        second = (self.stationary * (self.stds**2 + self.means**2)).sum()
        var = second - mean**2
        return {'mean': mean.item(), 'std': var.sqrt().item()}

    def regime_duration(self) -> Dict[str, float]:
        """E[이어지는 길이] = 1/(1 - P[k,k])."""
        return {self.regime_names[k]: 1 / (1 - self.P[k, k].item())
                for k in range(self.n_regimes)}

def demonstrate_regime_switching():
    """날마다의 수익을 다루는 두 국면 강세장/약세장 모형."""
    print("\nRegime-Switching Return Model")
    print("=" * 70)

    P = torch.tensor([[0.95, 0.05], [0.10, 0.90]])
    means = torch.tensor([0.0005, -0.0003])    # 날마다
    stds = torch.tensor([0.01, 0.025])

    model = RegimeSwitchingModel(P, means, stds, ['Bull', 'Bear'])

    print(f"Stationary: P(Bull)={model.stationary[0]:.3f}, "
          f"P(Bear)={model.stationary[1]:.3f}")

    durations = model.regime_duration()
    for regime, d in durations.items():
        print(f"E[{regime} duration]: {d:.1f} days")

    moments = model.unconditional_moments()
    print(f"Unconditional: E[r]={moments['mean']*252*100:.2f}% ann, "
          f"σ={moments['std']*np.sqrt(252)*100:.1f}% ann")

demonstrate_regime_switching()
```

---

## 9. MCMC과의 이음

HMM은 정확한 추론에서 MCMC로 넘어가는 까닭을 준다:

| HMM 추론 | 한계 | MCMC의 풀이 |
|--------------|------------|---------------|
| 앞뒤 알고리즘 | 숨은 상태가 이산이고 끝이 있어야 한다 | MCMC는 이어진 숨은 변수를 다룬다 |
| 비터비 | MAP만 주고 불확실함을 재지 않는다 | MCMC는 온전한 뒤확률 표본을 준다 |
| 바움-웰치 | 국소 최적점, 점 어림값 | MCMC는 매개변수 뒤확률 전체를 살펴본다 |
| 정확한 셈하기 | 차례마다 $O(K^2 T)$ | MCMC는 차원 높은 숨은 변수까지 감당한다 |

숨은 공간이 이어져 있거나 차원이 높아지면 HMM의 정확한 동적 계획법 알고리즘을 더는 쓸 수 없고, 18.3절에서 다루는 MCMC 표집으로 돌아서야 한다.

---

## 연습문제

1. **날씨 HMM.** 숨은 상태가 $\{$고기압, 저기압$\}$이고 관측이 $\{$맑음, 흐림, 비$\}$인 HMM을 지어라. 자료를 흉내 내어 만든 뒤 비터비 풀어내기로 숨은 상태를 되찾아라.

2. **바움-웰치의 모임.** 무작위 매개변수에서 시작해 흉내 낸 HMM 자료에 바움-웰치를 돌려라. 되풀이에 따른 로그 가능도를 그리고 한결같이 커지는지 확인하여라.

3. **신용 옮겨 감 HMM.** 등급이 숨어 있고 관측 신호가 (이산으로 나눈) 재무 비율이 되도록 신용 등급 모형을 넓혀라. 앞 알고리즘으로 관측한 비율 차례의 가능도를 셈하여라.

4. **흡수 분석.** 상태가 $\{$건강, 가벼움, 심함, 회복, 사망$\}$인(뒤 둘이 흡수 상태) 질병 진행 모형에서, 스쳐 지나감 상태마다 회복할 확률과 사망할 확률을 셈하여라.

5. **실제 자료에서 국면 찾기.** S&P 500의 날마다 수익률(오름/그대로/내림으로 이산화)에 두 국면 HMM을 맞춰라. 찾아낸 국면을 알려진 시장 사건과 견주어라.

## 정리하며

| 개념 | 핵심 식 | 복잡도 |
|---------|-------------|-----------|
| **앞 알고리즘** | $\alpha_t(j) = [\sum_i \alpha_{t-1}(i) A_{ij}] B_j(x_t)$ | $O(K^2 T)$ |
| **뒤 알고리즘** | $\beta_t(i) = \sum_j A_{ij} B_j(x_{t+1}) \beta_{t+1}(j)$ | $O(K^2 T)$ |
| **비터비** | $\delta_t(j) = \max_i [\delta_{t-1}(i) A_{ij}] B_j(x_t)$ | $O(K^2 T)$ |
| **바움-웰치** | 앞뒤 알고리즘에서 얻은 $\gamma_t, \xi_t$을 쓴 EM | 되풀이마다 $O(K^2 T)$ |
| **근본 행렬** | $N = (I - Q)^{-1}$ | $O(K^3)$ |
| **흡수 확률** | $B = NR$ | $O(K^2 r)$ |

**참고 문헌**

1. Rabiner, L.R. "A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition." *Proceedings of the IEEE*, 77(2), 1989.
2. Bishop, C.M. *Pattern Recognition and Machine Learning*, 13장. Springer, 2006.
3. Hamilton, J.D. "A New Approach to the Economic Analysis of Nonstationary Time Series." *Econometrica*, 57(2), 1989.
4. Kemeny, J.G. & Snell, J.L. *Finite Markov Chains*, 3장. Springer-Verlag, 1976.
5. Lando, D. *Credit Risk Modeling*. Princeton University Press, 2004.
6. Jarrow, R.A., Lando, D., & Turnbull, S.M. "A Markov Model for the Term Structure of Credit Risk Spreads." *Review of Financial Studies*, 10(2), 1997.
