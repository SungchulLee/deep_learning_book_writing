# 32.9.2 Upper Confidence Bound (UCB)

## Overview

**UCB** selects actions based on both their estimated value and the uncertainty in that estimate. It implements the principle of "optimism in the face of uncertainty."

## UCB1 Formula

$$A_t = \arg\max_a \left[Q(s, a) + c \sqrt{\frac{\ln t}{N(s, a)}}\right]$$

where:
- $Q(s, a)$: Estimated action value
- $N(s, a)$: Number of times action $a$ taken in state $s$
- $c > 0$: Exploration parameter (typically $c = \sqrt{2}$)
- $t$: Total number of steps

The second term is the **exploration bonus** — it is large for rarely-tried actions and shrinks as they are explored.

## Intuition

UCB maintains an optimistic estimate of each action's value:

$$\text{UCB}(s, a) = \underbrace{Q(s, a)}_{\text{exploitation}} + \underbrace{c\sqrt{\frac{\ln t}{N(s, a)}}}_{\text{exploration bonus}}$$

- Actions tried many times: bonus shrinks → exploits known good actions
- Actions rarely tried: bonus is large → explores uncertain actions
- Over time: all actions explored, converges to greedy

## Theoretical Guarantee

UCB1 achieves **logarithmic regret** for the multi-armed bandit problem:

$$\text{Regret}(T) \leq \sum_{a: \mu_a < \mu^*} \frac{8 \ln T}{\Delta_a} + (1 + \frac{\pi^2}{3}) \sum_a \Delta_a$$

where $\Delta_a = \mu^* - \mu_a$ is the suboptimality gap.

## UCB vs. ε-Greedy

| Feature | ε-Greedy | UCB |
|---------|---------|-----|
| Exploration type | Random | Directed (uncertainty-based) |
| Regret | Linear ($O(\epsilon T)$) | Logarithmic ($O(\ln T)$) |
| Parameter sensitivity | Sensitive to ε | Robust (c ≈ √2) |
| Action selection | Equal exploration prob | Favors uncertain actions |
| Computational cost | O(1) | O(|A|) |
| Applicability | General (MDPs) | Best for bandits |

## Challenges for MDPs

UCB was designed for multi-armed bandits (stationary, single-state). In MDPs:
- State-dependent action values complicate count maintenance
- Non-stationarity from policy changes
- Function approximation makes counts ill-defined

Extensions like **UCB-VI** (Upper Confidence Bound Value Iteration) adapt UCB to MDPs.

## Financial Application

UCB in portfolio selection: maintain confidence intervals for each asset/strategy's expected return. Allocate more to strategies with high upper confidence bounds — either because they have high estimated returns (exploit) or because they're poorly understood (explore). This resembles Bayesian portfolio optimization with parameter uncertainty.
