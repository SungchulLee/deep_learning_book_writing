# 32.8.3 Convergence Issues in Function Approximation

## The Deadly Triad

Instability and divergence can occur when three elements are combined:

1. **Function approximation** (generalizing from a subset of states)
2. **Bootstrapping** (updating estimates based on other estimates)
3. **Off-policy training** (learning about a policy different from the one generating data)

Any two of these three are safe. All three together can cause divergence.

| Elements Present | Example | Stable? |
|-----------------|---------|---------|
| FA + Bootstrap | On-policy linear TD | Yes (converges) |
| FA + Off-policy | MC with FA | Yes (converges) |
| Bootstrap + Off-policy | Tabular Q-learning | Yes (converges) |
| All three | Off-policy linear TD | **Can diverge** |

## Baird's Counterexample

Baird (1995) demonstrated a simple 7-state MDP where off-policy semi-gradient TD(0) with linear function approximation diverges — the weights grow without bound despite the problem being simple.

This counterexample was crucial in establishing that naive combination of function approximation with bootstrapping and off-policy learning is fundamentally problematic.

## Why Divergence Occurs

### Distribution Mismatch

Off-policy learning updates states according to the behavior policy distribution $d_b$, not the target policy distribution $d_\pi$. The TD fixed point may not exist or may be unstable under $d_b$.

### Projection Error Amplification

With function approximation, updates project the Bellman backup onto the function class. Off-policy + bootstrapping can create cycles where projection errors are amplified rather than reduced.

## Solutions and Mitigations

### Gradient TD Methods (GTD, GTD2, TDC)

Use true stochastic gradient descent on a proper objective:

$$\text{MSPBE}(\mathbf{w}) = \|\hat{V}_\mathbf{w} - \Pi \mathcal{T}_\pi \hat{V}_\mathbf{w}\|_{d}^2$$

where $\Pi$ is the projection onto the function class. GTD methods converge under off-policy training.

### Emphatic TD

Reweights updates using the **emphasis** — a product of importance sampling ratios and interest:

$$M_t = \gamma \rho_{t-1} M_{t-1} + I_t$$

This corrects for the distribution mismatch and guarantees convergence.

### Experience Replay + Target Networks (DQN approach)

- **Experience replay**: Break temporal correlations by sampling random mini-batches
- **Target network**: Use a slowly-updated copy for the bootstrap target, reducing instability
- Not guaranteed to converge but works well in practice

### Residual Gradient Methods

Minimize the mean squared Bellman error directly:

$$\text{MSBE}(\mathbf{w}) = \mathbb{E}[(\hat{V}(S; \mathbf{w}) - R - \gamma \hat{V}(S'; \mathbf{w}))^2]$$

Requires double sampling (two independent next-state samples) for unbiased gradients.

## Practical Recommendations

1. **Prefer on-policy** when possible (avoids the deadliest part of the triad)
2. **Use experience replay** for off-policy methods
3. **Target networks** stabilize bootstrapping
4. **Gradient clipping** prevents exploding updates
5. **Learning rate scheduling** (decreasing α)
6. **Monitor for divergence** (track value function magnitude)

## Financial Implications

In financial RL:
- Off-policy learning from historical data is extremely valuable but carries divergence risk
- Using complex neural network value functions + bootstrapping + historical data = deadly triad
- Practical solutions: careful regularization, target networks, conservative learning rates
- On-policy methods (PPO, A2C) avoid the worst convergence issues

## Summary

Function approximation introduces convergence challenges, especially when combined with bootstrapping and off-policy learning. The deadly triad explains why naive approaches can diverge. Solutions include gradient TD methods, emphatic TD, experience replay, and target networks. Understanding these issues is essential for building reliable RL systems, particularly in high-stakes financial applications.
