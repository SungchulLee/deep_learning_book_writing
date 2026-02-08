# 32.6.5 Expected SARSA

## Overview

**Expected SARSA** improves upon SARSA by taking the **expectation** over next actions instead of sampling a single next action. This reduces variance while maintaining the on-policy character.

## Algorithm

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[R_{t+1} + \gamma \sum_{a'} \pi(a'|S_{t+1}) Q(S_{t+1}, a') - Q(S_t, A_t)\right]$$

### Comparison of TD Control Updates

| Algorithm | Update Target |
|-----------|--------------|
| SARSA | $R + \gamma Q(S', A')$ where $A' \sim \pi$ |
| Expected SARSA | $R + \gamma \sum_{a'} \pi(a'\|S') Q(S', a')$ |
| Q-Learning | $R + \gamma \max_{a'} Q(S', a')$ |

## Key Properties

1. **Lower variance than SARSA**: Eliminates randomness from next-action selection
2. **Same expected update as SARSA**: Both have the same expected TD target
3. **Generalizes both SARSA and Q-Learning**: With ε-greedy, Expected SARSA interpolates between SARSA (uses sampled $A'$) and Q-Learning (uses $\max$)
4. **Can be off-policy**: Works with any target policy in the expectation

## Expected SARSA as a Generalization

When the target policy $\pi$ is:
- **ε-greedy**: Expected SARSA (standard form)
- **Greedy** ($\epsilon = 0$): Reduces to **Q-Learning** ($\sum_a \pi(a|s) Q(s,a) = \max_a Q(s,a)$)
- **Uniform random**: Reduces to computing expected Q under random policy

## Computational Cost

Expected SARSA requires summing over all actions at each step:
- **SARSA**: $O(1)$ per update (single action lookup)
- **Expected SARSA**: $O(|\mathcal{A}|)$ per update (sum over actions)
- **Q-Learning**: $O(|\mathcal{A}|)$ per update (max over actions)

For small action spaces, the extra cost is negligible and the variance reduction is worthwhile.

## Performance

On benchmark problems, Expected SARSA typically:
- Outperforms SARSA (lower variance → faster convergence)
- Performs comparably to Q-Learning for small $\epsilon$
- Is more stable than Q-Learning (no maximization bias)

## Financial Application

Expected SARSA is well-suited for portfolio allocation where:
- Actions are portfolio weights across a small number of assets
- The ε-greedy exploration means occasionally trying random allocations
- Expected SARSA accounts for the full distribution of exploration actions
- More stable than Q-learning in volatile market environments

## Summary

Expected SARSA eliminates the sampling variance of SARSA by computing the expectation over next actions analytically. It subsumes both SARSA and Q-Learning as special cases and typically provides better or comparable performance. Its main cost is $O(|\mathcal{A}|)$ computation per step, which is negligible for small action spaces.
