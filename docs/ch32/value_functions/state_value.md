# 32.3.1 State Value Function

## Definition

The **state value function** $V_\pi(s)$ gives the expected return when starting in state $s$ and following policy $\pi$ thereafter:

$$V_\pi(s) = \mathbb{E}_\pi[G_t \mid S_t = s] = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \mid S_t = s\right]$$

This function answers: "How good is it to be in state $s$ under policy $\pi$?"

## Properties

1. **Policy-dependent**: Different policies yield different value functions
2. **Bounded**: For $\gamma < 1$ and bounded rewards: $|V_\pi(s)| \leq \frac{R_{\max}}{1-\gamma}$
3. **Unique**: For a given policy $\pi$ and MDP, $V_\pi$ is uniquely defined
4. **Recursive**: Satisfies the Bellman equation (see Section 32.3.3)

## Computing $V_\pi$ for Finite MDPs

### Direct Computation (Matrix Form)

For finite MDPs, $V_\pi$ can be computed by solving a system of linear equations:

$$\mathbf{v}_\pi = \mathbf{r}_\pi + \gamma \mathbf{P}_\pi \mathbf{v}_\pi$$

Rearranging:

$$\mathbf{v}_\pi = (\mathbf{I} - \gamma \mathbf{P}_\pi)^{-1} \mathbf{r}_\pi$$

where:
- $\mathbf{P}_\pi \in \mathbb{R}^{|\mathcal{S}| \times |\mathcal{S}|}$ with $[\mathbf{P}_\pi]_{ss'} = \sum_a \pi(a|s) P(s'|s,a)$
- $\mathbf{r}_\pi \in \mathbb{R}^{|\mathcal{S}|}$ with $[\mathbf{r}_\pi]_s = \sum_a \pi(a|s) R(s,a)$

This requires $O(|\mathcal{S}|^3)$ computation for the matrix inverse.

### Iterative Computation

For large state spaces, iterative methods are preferred:

$$V_{k+1}(s) = \sum_a \pi(a|s) \left[R(s,a) + \gamma \sum_{s'} P(s'|s,a) V_k(s')\right]$$

This converges to $V_\pi$ as $k \to \infty$ (guaranteed by contraction mapping).

### Monte Carlo Estimation

Estimate $V_\pi(s)$ by averaging returns from many episodes:

$$V_\pi(s) \approx \frac{1}{N(s)} \sum_{i=1}^{N(s)} G_t^{(i)}$$

where $G_t^{(i)}$ is the return from the $i$-th visit to state $s$.

## Optimal State Value Function

The **optimal state value function** $V^*(s)$ is the maximum over all policies:

$$V^*(s) = \max_\pi V_\pi(s) \quad \text{for all } s \in \mathcal{S}$$

An optimal policy $\pi^*$ achieves $V_{\pi^*}(s) = V^*(s)$ for all states simultaneously.

## Partial Ordering of Policies

Policy $\pi$ is **better than or equal to** policy $\pi'$ (written $\pi \geq \pi'$) if:

$$V_\pi(s) \geq V_{\pi'}(s) \quad \text{for all } s \in \mathcal{S}$$

There always exists at least one optimal policy $\pi^*$ that is better than or equal to all other policies.

## Interpretation in Finance

In portfolio management:
- $V_\pi(s)$ represents the expected future risk-adjusted return from market state $s$ under strategy $\pi$
- Comparing $V_\pi(s)$ across states reveals which market conditions are more favorable
- $V^*(s)$ represents the best possible expected performance achievable from any state

## Summary

The state value function is the central quantity in RL, encoding the long-term desirability of each state under a given policy. It can be computed exactly for small MDPs (matrix inversion) or estimated via iterative/sampling methods for larger problems. The optimal value function $V^*$ characterizes the best achievable performance.
