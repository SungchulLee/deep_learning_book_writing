# 32.2.5 Discount Factor

## Definition

The **discount factor** $\gamma \in [0, 1]$ determines how much the agent values future rewards relative to immediate ones. The discounted return is:

$$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots$$

## Interpretation

The discount factor has multiple interpretations:

### Mathematical: Convergence Guarantee

For bounded rewards $|R| \leq R_{\max}$ and $\gamma < 1$:

$$|G_t| \leq \sum_{k=0}^{\infty} \gamma^k R_{\max} = \frac{R_{\max}}{1 - \gamma}$$

Without discounting ($\gamma = 1$), the return may diverge for continuing tasks.

### Economic: Time Value

A dollar today is worth more than a dollar tomorrow. The discount factor encodes this time preference, analogous to the **discount rate** in finance:

$$\gamma = \frac{1}{1 + r}$$

where $r$ is the per-period interest rate.

### Probabilistic: Survival Probability

$\gamma$ can be interpreted as the probability that the process continues at each step. With probability $1 - \gamma$, the episode terminates, making the effective horizon geometric with mean $\frac{1}{1-\gamma}$.

### Computational: Effective Horizon

The **effective horizon** is the number of future steps that significantly influence the return:

$$\text{Effective Horizon} \approx \frac{1}{1 - \gamma}$$

| $\gamma$ | Effective Horizon | Interpretation |
|-----------|------------------|----------------|
| 0.0 | 1 step | Completely myopic |
| 0.9 | 10 steps | Short-term planning |
| 0.95 | 20 steps | Medium-term |
| 0.99 | 100 steps | Long-term planning |
| 0.999 | 1000 steps | Very long horizon |
| 1.0 | $\infty$ | Undiscounted (episodic only) |

## Effect on Optimal Policy

The discount factor directly affects which policy is optimal:

- **Low $\gamma$ (myopic)**: The agent prioritizes immediate rewards, ignoring long-term consequences. May miss strategies that require short-term sacrifice for long-term gain.
- **High $\gamma$ (far-sighted)**: The agent considers long-term consequences but learning becomes harder due to high variance in return estimates and slower value propagation.

### Discount Factor and Bias-Variance Trade-off

| Low $\gamma$ | High $\gamma$ |
|--------------|---------------|
| Low variance in return estimates | High variance |
| High bias (ignores future) | Low bias |
| Faster convergence | Slower convergence |
| May miss long-term strategies | Can learn complex strategies |

## Special Cases

### $\gamma = 0$: Myopic Agent

$$G_t = R_{t+1}$$

The agent is a **greedy** or **myopic** agent, caring only about the immediate reward. The optimal policy maximizes $R(s,a)$ at each step.

### $\gamma = 1$: Undiscounted Return

$$G_t = \sum_{k=0}^{T-t-1} R_{t+k+1}$$

Valid only for episodic tasks where $T$ is finite. The agent treats all future rewards equally. Used in many game-playing applications.

### Average Reward Setting

For continuing tasks with $\gamma \to 1$, an alternative is the **average reward** criterion:

$$r(\pi) = \lim_{T \to \infty} \frac{1}{T} \sum_{t=1}^{T} \mathbb{E}_\pi[R_t]$$

This avoids divergence while treating all time steps equally.

## Choosing $\gamma$ in Practice

### Guidelines

1. **Episodic tasks**: $\gamma = 1.0$ or $\gamma = 0.99$ is common
2. **Continuing tasks**: Must have $\gamma < 1$; typically $0.95 \leq \gamma \leq 0.999$
3. **Problem-dependent**: Match $\frac{1}{1-\gamma}$ to the natural planning horizon
4. **Tuning**: Often treated as a hyperparameter to be tuned

### Interaction with Other Hyperparameters

The discount factor interacts with:

- **Learning rate**: Higher $\gamma$ may require lower learning rates for stability
- **Episode length**: $\gamma$ should be high enough that the agent can "see" the end of the episode
- **Reward scale**: Effective value range is $\left[-\frac{R_{\max}}{1-\gamma}, \frac{R_{\max}}{1-\gamma}\right]$

## Financial Applications

### Investment Horizon Mapping

Map $\gamma$ to the investor's time horizon:

| Investor Type | Horizon | Suggested $\gamma$ |
|--------------|---------|-------------------|
| Day trader | Hours to days | 0.9 - 0.95 |
| Swing trader | Days to weeks | 0.95 - 0.99 |
| Position trader | Weeks to months | 0.99 - 0.995 |
| Long-term investor | Months to years | 0.995 - 0.999 |

### Risk-Free Rate Connection

If the risk-free rate is $r_f$ per period:

$$\gamma_{\text{financial}} = \frac{1}{1 + r_f}$$

For daily data with annual risk-free rate of 5%: $\gamma \approx \frac{1}{1 + 0.05/252} \approx 0.9998$

### Practical Considerations

- Transaction costs create a natural preference for patience (higher $\gamma$)
- Market microstructure effects are more relevant for low $\gamma$ (short horizon)
- Regime changes may warrant adaptive $\gamma$ or state-dependent discounting

## Summary

The discount factor is a fundamental parameter that controls the agent's planning horizon and the trade-off between immediate and future rewards. Its choice significantly affects the optimal policy, convergence properties, and practical performance. In financial applications, $\gamma$ should be aligned with the investment horizon and risk preferences of the strategy being developed.
