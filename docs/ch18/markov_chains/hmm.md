# Hidden Markov Models

## Introduction

In the Markov chains studied so far, the state $X_n$ is directly observed. In many real-world systems, however, the underlying state is **hidden** (or latent) and we observe only a noisy signal that depends on it. **Hidden Markov Models (HMMs)** formalize this by coupling a latent Markov chain with an observation model.

HMMs bridge Markov chain theory and statistical inference: the latent chain provides temporal structure, while the observation model creates an inference problem—recovering the hidden states from observed data. This makes HMMs a natural stepping stone toward MCMC methods (Section 18.3), which tackle inference in more general latent variable models.

## Mathematical Framework

### Model Definition

An HMM consists of three components:

1. **Hidden state process** $\{Z_t\}_{t=1}^T$ — a Markov chain with transition matrix $A$:

$$A_{ij} = P(Z_t = j \mid Z_{t-1} = i)$$

2. **Observation process** $\{X_t\}_{t=1}^T$ — conditionally independent given the hidden states:

$$P(X_t = x \mid Z_t = k) = B_k(x)$$

where $B_k$ is the **emission distribution** for state $k$.

3. **Initial distribution** $\boldsymbol{\pi}$:

$$\pi_k = P(Z_1 = k)$$

### Conditional Independence Structure

The joint probability of hidden states $\mathbf{z} = (z_1, \ldots, z_T)$ and observations $\mathbf{x} = (x_1, \ldots, x_T)$ factorizes as:

$$P(\mathbf{z}, \mathbf{x}) = \pi_{z_1} B_{z_1}(x_1) \prod_{t=2}^{T} A_{z_{t-1}, z_t} B_{z_t}(x_t)$$

### The Three Fundamental Problems

| Problem | Question | Algorithm |
|---------|----------|-----------|
| **Evaluation** | $P(\mathbf{x} \mid \theta)$ — likelihood of observations? | Forward algorithm |
| **Decoding** | $\arg\max_{\mathbf{z}} P(\mathbf{z} \mid \mathbf{x}, \theta)$ — most likely hidden sequence? | Viterbi algorithm |
| **Learning** | $\arg\max_\theta P(\mathbf{x} \mid \theta)$ — best model parameters? | Baum-Welch (EM) |

## The Forward-Backward Algorithm

### Forward Variables

The **forward variable** $\alpha_t(j) = P(X_1 = x_1, \ldots, X_t = x_t, Z_t = j)$ is the joint probability of observing the first $t$ emissions and being in hidden state $j$ at time $t$.

**Recursion:**

$$\alpha_1(j) = \pi_j B_j(x_1)$$

$$\alpha_t(j) = \left[\sum_{i=1}^{K} \alpha_{t-1}(i) A_{ij}\right] B_j(x_t), \quad t = 2, \ldots, T$$

The total observation likelihood is $P(\mathbf{x}) = \sum_{j=1}^{K} \alpha_T(j)$.

**Complexity:** $O(K^2 T)$ versus $O(K^T)$ for brute-force enumeration over all possible state sequences.

### Backward Variables

The **backward variable** $\beta_t(i) = P(X_{t+1}, \ldots, X_T \mid Z_t = i)$ satisfies:

$$\beta_T(i) = 1, \qquad \beta_t(i) = \sum_{j=1}^{K} A_{ij} B_j(x_{t+1}) \beta_{t+1}(j)$$

### Posterior State Probabilities

Combining forward and backward variables:

$$\gamma_t(j) = P(Z_t = j \mid \mathbf{x}) = \frac{\alpha_t(j) \beta_t(j)}{P(\mathbf{x})}$$

$$\xi_t(i, j) = P(Z_t = i, Z_{t+1} = j \mid \mathbf{x}) = \frac{\alpha_t(i) A_{ij} B_j(x_{t+1}) \beta_{t+1}(j)}{P(\mathbf{x})}$$

## The Viterbi Algorithm

The Viterbi algorithm finds the most likely hidden state sequence via dynamic programming in log-space.

Define $\delta_t(j) = \max_{z_1, \ldots, z_{t-1}} P(z_1, \ldots, z_{t-1}, Z_t = j, x_1, \ldots, x_t)$.

**Recursion:**

$$\delta_1(j) = \pi_j B_j(x_1), \qquad \delta_t(j) = \max_{i} [\delta_{t-1}(i) A_{ij}] \cdot B_j(x_t)$$

**Backtracking** from $z_T^* = \arg\max_j \delta_T(j)$ recovers the optimal path.

## The Baum-Welch Algorithm

Baum-Welch is EM for HMMs. Each iteration:

**E-step:** Compute $\gamma_t(j)$ and $\xi_t(i,j)$ via forward-backward.

**M-step:** Update parameters:

$$\hat{\pi}_j = \gamma_1(j), \qquad \hat{A}_{ij} = \frac{\sum_{t=1}^{T-1} \xi_t(i, j)}{\sum_{t=1}^{T-1} \gamma_t(i)}, \qquad \hat{B}_j(v) = \frac{\sum_{t : x_t = v} \gamma_t(j)}{\sum_{t=1}^{T} \gamma_t(j)}$$

## PyTorch Implementation

```python
import torch
from typing import Dict, List, Tuple, Optional

class HiddenMarkovModel:
    """
    Hidden Markov Model with discrete emissions.

    Components:
    - A: K×K transition matrix for hidden states
    - B: K×V emission matrix (B[k,v] = P(obs=v | state=k))
    - pi: K-dim initial state distribution
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
        Compute forward variables α_t(j) and log P(x).

        Returns:
            (alpha, log_likelihood) where alpha is (T, K)
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
        """Compute backward variables β_t(i). Returns (T, K) tensor."""
        T = len(observations)
        beta = torch.zeros(T, self.K, dtype=torch.float64)
        beta[-1] = 1.0

        for t in range(T - 2, -1, -1):
            beta[t] = self.A @ (self.B[:, observations[t+1]] * beta[t+1])

        return beta

    def posterior_states(
        self, observations: torch.Tensor
    ) -> torch.Tensor:
        """Compute γ_t(j) = P(Z_t = j | x). Returns (T, K) tensor."""
        alpha, _ = self.forward_algorithm(observations)
        beta = self.backward_algorithm(observations)
        gamma = alpha * beta
        return gamma / gamma.sum(dim=1, keepdim=True)

    def viterbi(
        self, observations: torch.Tensor
    ) -> Tuple[List[int], float]:
        """
        Find most likely hidden state sequence (log-space).

        Returns:
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
        """Baum-Welch (EM) for parameter estimation."""
        T = len(observations)
        log_likelihoods = []

        for iteration in range(max_iter):
            # E-step
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

            # M-step
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
        """Simulate hidden states and observations."""
        states, observations = [], []
        state = torch.multinomial(self.pi.float(), 1).item()

        for t in range(n_steps):
            states.append(state)
            obs = torch.multinomial(self.B[state].float(), 1).item()
            observations.append(obs)
            state = torch.multinomial(self.A[state].float(), 1).item()

        return states, observations
```

## Application: Market Regime Detection

```python
def demonstrate_hmm_regime_detection():
    """
    Detect bull/bear market regimes from observed price movements.
    """
    print("HMM: Market Regime Detection")
    print("=" * 70)

    state_names = ['Bull', 'Bear']
    obs_names = ['Up', 'Flat', 'Down']

    A = torch.tensor([
        [0.95, 0.05],   # Bull: 95% persist
        [0.10, 0.90]    # Bear: 90% persist
    ])

    B = torch.tensor([
        [0.60, 0.30, 0.10],  # Bull: mostly Up
        [0.15, 0.25, 0.60]   # Bear: mostly Down
    ])

    pi = torch.tensor([0.7, 0.3])
    hmm = HiddenMarkovModel(A, B, pi, state_names, obs_names)

    # Simulate and decode
    true_states, observations = hmm.simulate(200)
    obs_tensor = torch.tensor(observations)

    decoded_states, log_prob = hmm.viterbi(obs_tensor)
    accuracy = sum(t == d for t, d in zip(true_states, decoded_states))
    accuracy /= len(true_states)

    print(f"Viterbi decoding accuracy: {accuracy:.1%}")
    print(f"Log-likelihood: {log_prob:.2f}")

    # Posterior state probabilities
    gamma = hmm.posterior_states(obs_tensor)

    # Expected regime durations
    bull_duration = 1.0 / (1.0 - A[0, 0].item())
    bear_duration = 1.0 / (1.0 - A[1, 1].item())
    print(f"\nExpected regime durations:")
    print(f"  Bull: {bull_duration:.1f} periods")
    print(f"  Bear: {bear_duration:.1f} periods")


demonstrate_hmm_regime_detection()
```

## Absorbing HMMs and Credit Risk

When the hidden Markov chain includes **absorbing states**, the model captures systems that eventually settle into terminal conditions. Credit rating migration is the canonical financial example: ratings transition stochastically over time, with Default as an absorbing state.

### Absorbing Chain Analysis

```python
class AbsorbingMarkovChain:
    """
    Analysis of absorbing Markov chains.

    Canonical form: P = [[Q, R], [0, I]]
    Key results:
    - Fundamental matrix: N = (I - Q)^{-1}
    - Expected absorption time: t = N·1
    - Absorption probabilities: B = N·R
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
        """Identify absorbing (P[i,i]=1) and transient states."""
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
        """Extract Q and R sub-matrices."""
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

### Credit Rating Transitions

```python
class CreditRatingModel:
    """
    Credit rating migration as an absorbing Markov chain.
    Default (D) is the absorbing state.
    """

    def __init__(self, transition_matrix: torch.Tensor, ratings: List[str]):
        self.P = transition_matrix.clone().double()
        self.ratings = ratings
        self.n_ratings = len(ratings)
        self.default_idx = self.n_ratings - 1

    def cumulative_default_prob(
        self, initial_rating: str, max_horizon: int = 10
    ) -> torch.Tensor:
        """P(Default by time t | Rating_0 = initial_rating)."""
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
        """Credit Value at Risk via Monte Carlo."""
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
    """Credit rating model with default probabilities and VaR."""
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

    # Cumulative default probabilities
    print("\nCumulative Default Probabilities:")
    print("-" * 50)
    header = "Rating  " + "  ".join(f"Year {t}" for t in range(1, 6))
    print(header)
    for rating in ['AAA', 'BBB', 'B', 'CCC']:
        cum_pds = model.cumulative_default_prob(rating, 5)
        row = f"{rating:6}  " + "  ".join(f"{pd:6.2%}" for pd in cum_pds)
        print(row)

    # Absorbing chain analysis
    chain = AbsorbingMarkovChain(P, state_names=ratings)
    times = chain.expected_absorption_time()
    print("\nExpected Years to Default (from transient states):")
    for state, t in times.items():
        print(f"  {state}: {t:.1f} years")

    # Portfolio VaR
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

### Gambler's Ruin Example

The classic gambler's ruin problem illustrates absorbing chain analysis:

```python
def demonstrate_gamblers_ruin():
    """Gambler's ruin as an absorbing Markov chain."""
    print("\nGambler's Ruin (Target = $4)")
    print("=" * 70)

    states = ['$0 (Broke)', '$1', '$2', '$3', '$4 (Win)']
    P = torch.tensor([
        [1.0, 0.0, 0.0, 0.0, 0.0],  # $0: absorbing
        [0.5, 0.0, 0.5, 0.0, 0.0],  # $1
        [0.0, 0.5, 0.0, 0.5, 0.0],  # $2
        [0.0, 0.0, 0.5, 0.0, 0.5],  # $3
        [0.0, 0.0, 0.0, 0.0, 1.0]   # $4: absorbing
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

## Regime-Switching Return Models

Combining HMMs with continuous emissions gives **regime-switching models** widely used in quantitative finance:

```python
import numpy as np

class RegimeSwitchingModel:
    """
    Regime-switching model: hidden Markov chain drives
    regime-specific return distributions.

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
        """Simulate returns and regime labels."""
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
        """E[duration] = 1/(1 - P[k,k])."""
        return {self.regime_names[k]: 1 / (1 - self.P[k, k].item())
                for k in range(self.n_regimes)}


def demonstrate_regime_switching():
    """Two-regime bull/bear model for daily returns."""
    print("\nRegime-Switching Return Model")
    print("=" * 70)

    P = torch.tensor([[0.95, 0.05], [0.10, 0.90]])
    means = torch.tensor([0.0005, -0.0003])    # Daily
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

## Connection to MCMC

HMMs motivate the transition from exact inference to MCMC:

| HMM Inference | Limitation | MCMC Solution |
|--------------|------------|---------------|
| Forward-backward | Requires discrete, finite hidden states | MCMC handles continuous latent variables |
| Viterbi | MAP only, no uncertainty quantification | MCMC provides full posterior samples |
| Baum-Welch | Local optima, point estimates | MCMC explores full parameter posterior |
| Exact computation | $O(K^2 T)$ per sequence | MCMC scales to high-dimensional latents |

When the latent space becomes continuous or high-dimensional, the exact dynamic programming algorithms of HMMs no longer apply, and we must turn to MCMC sampling—the subject of Section 18.3.

## Summary

| Concept | Key Equation | Complexity |
|---------|-------------|-----------|
| **Forward algorithm** | $\alpha_t(j) = [\sum_i \alpha_{t-1}(i) A_{ij}] B_j(x_t)$ | $O(K^2 T)$ |
| **Backward algorithm** | $\beta_t(i) = \sum_j A_{ij} B_j(x_{t+1}) \beta_{t+1}(j)$ | $O(K^2 T)$ |
| **Viterbi** | $\delta_t(j) = \max_i [\delta_{t-1}(i) A_{ij}] B_j(x_t)$ | $O(K^2 T)$ |
| **Baum-Welch** | EM with $\gamma_t, \xi_t$ from forward-backward | $O(K^2 T)$ per iter |
| **Fundamental matrix** | $N = (I - Q)^{-1}$ | $O(K^3)$ |
| **Absorption probabilities** | $B = NR$ | $O(K^2 r)$ |

## Exercises

1. **Weather HMM.** Construct an HMM where hidden states are $\{$High Pressure, Low Pressure$\}$ and observations are $\{$Sunny, Cloudy, Rainy$\}$. Simulate data, then recover the hidden states using Viterbi decoding.

2. **Baum-Welch Convergence.** Starting from random parameters, run Baum-Welch on simulated HMM data. Plot the log-likelihood over iterations and verify monotonic increase.

3. **Credit Migration HMM.** Extend the credit rating model so that ratings are hidden and observed signals are financial ratios (discretized). Use the forward algorithm to compute the likelihood of an observed sequence of ratios.

4. **Absorption Analysis.** For a disease progression model with states $\{$Healthy, Mild, Severe, Recovered, Deceased$\}$ (last two absorbing), compute the probability of recovery vs. death starting from each transient state.

5. **Regime Detection on Real Data.** Fit a two-regime HMM to S&P 500 daily returns (discretized as Up/Flat/Down). Compare the detected regimes with known market events.

## References

1. Rabiner, L.R. "A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition." *Proceedings of the IEEE*, 77(2), 1989.
2. Bishop, C.M. *Pattern Recognition and Machine Learning*, Chapter 13. Springer, 2006.
3. Hamilton, J.D. "A New Approach to the Economic Analysis of Nonstationary Time Series." *Econometrica*, 57(2), 1989.
4. Kemeny, J.G. & Snell, J.L. *Finite Markov Chains*, Chapter 3. Springer-Verlag, 1976.
5. Lando, D. *Credit Risk Modeling*. Princeton University Press, 2004.
6. Jarrow, R.A., Lando, D., & Turnbull, S.M. "A Markov Model for the Term Structure of Credit Risk Spreads." *Review of Financial Studies*, 10(2), 1997.
