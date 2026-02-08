# 32.9.4 Exploration Bonuses (Intrinsic Motivation)

## Overview

**Exploration bonuses** augment the environment reward with an intrinsic reward that encourages visiting novel or uncertain states. The total reward becomes:

$$R_t^{\text{total}} = R_t^{\text{extrinsic}} + \beta \cdot R_t^{\text{intrinsic}}$$

where $\beta$ controls the exploration-exploitation balance.

## Count-Based Exploration

Add a bonus inversely proportional to visit counts:

$$R_t^{\text{intrinsic}} = \frac{1}{\sqrt{N(s)}}$$

For large/continuous state spaces, approximate counts using:
- **Hash-based counts**: Hash states to a fixed-size table
- **Density models**: Use density estimation; bonus = $1/\sqrt{\hat{\rho}(s)}$
- **Pseudo-counts**: Derive counts from a density model's learning progress

## Prediction Error Bonus (Curiosity)

Use the agent's prediction error as intrinsic reward:

$$R_t^{\text{intrinsic}} = \|f(s_{t+1}; \theta) - \text{target}\|^2$$

where $f$ is a learned forward model or random network. States that are poorly predicted are "interesting" and receive higher bonuses.

### Random Network Distillation (RND)

- Fixed random network $f_{\text{target}}(s)$
- Trained predictor $f_{\text{pred}}(s; \theta)$
- Bonus = prediction error: $\|f_{\text{pred}}(s) - f_{\text{target}}(s)\|^2$
- Novel states have high prediction error (haven't been trained on)

## Information-Theoretic Bonuses

### VIME (Variational Information Maximizing Exploration)

Bonus based on information gain about the environment dynamics:

$$R_t^{\text{intrinsic}} = D_{KL}[\hat{P}(\theta | h_{t+1}) \| \hat{P}(\theta | h_t)]$$

The agent is rewarded for transitions that change its model the most.

### Maximum Entropy

Encourage diverse behavior by adding entropy bonus:

$$R_t^{\text{intrinsic}} = \mathcal{H}[\pi(\cdot | s_t)]$$

This prevents premature convergence to deterministic policies (used in SAC).

## Comparison of Methods

| Method | Mechanism | Requires Model | Scalability |
|--------|-----------|---------------|-------------|
| ε-greedy | Random action | No | High |
| UCB | Visit counts | No | Medium |
| Count-based bonus | State novelty | No | Medium |
| Curiosity/RND | Prediction error | Partial | High |
| VIME | Information gain | Yes | Low |
| Entropy bonus | Policy diversity | No | High |

## Challenges

1. **Bonus scaling**: $\beta$ must balance intrinsic and extrinsic rewards
2. **Noisy TV problem**: Stochastic environments generate perpetual prediction error
3. **Bonus decay**: Intrinsic rewards should decrease as exploration progresses
4. **Detachment**: The agent may explore irrelevant parts of the state space

## Financial Application

Exploration bonuses in financial RL:
- **Count-based**: Encourage trading in market regimes rarely encountered
- **Curiosity**: Explore strategies that produce unexpected P&L patterns
- **Entropy**: Maintain diverse portfolio allocations, prevent over-concentration
- **Information gain**: Prioritize trades that reveal the most about market dynamics

## Summary

Exploration bonuses transform the exploration problem from action selection (ε-greedy, UCB, Boltzmann) to reward augmentation. By incentivizing visits to novel or uncertain states, they can dramatically improve exploration in sparse-reward or large state-space problems. The choice of bonus type depends on the problem structure, computational budget, and the nature of the exploration challenge.
