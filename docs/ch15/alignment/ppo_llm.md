# PPO for LLM Alignment

## Learning Objectives

- Understand PPO adapted for language model training
- Implement the per-token KL penalty
- Navigate practical training challenges

## PPO Objective for LLMs

The standard PPO clipped objective, adapted for language models:

$$\mathcal{L}_{\text{PPO}} = \mathbb{E}_t \left[\min\left(\rho_t A_t, \; \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon) A_t\right)\right]$$

where $\rho_t = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{old}}(a_t \mid s_t)}$ is the importance ratio and $A_t$ is the advantage.

## LLM-Specific Adaptations

### Per-Token KL Penalty

Instead of a global KL constraint, apply a per-token penalty:

$$r_{\text{total}}(x, y) = r_\phi(x, y) - \beta \sum_{t=1}^{|y|} \log \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\text{ref}}(y_t \mid x, y_{<t})}$$

This prevents the policy from deviating from the reference (SFT) model at any individual token position.

### Value Function

A separate value head estimates the expected return:

$$V_\psi(x, y_{\leq t}) \approx \mathbb{E}\left[\sum_{t'=t}^{|y|} r_{t'} \mid x, y_{\leq t}\right]$$

## Training Architecture

```
┌─────────────┐  ┌──────────────┐
│  Actor       │  │  Critic      │
│  (Policy π)  │  │  (Value V)   │
└──────┬──────┘  └──────┬───────┘
       │                │
       ▼                ▼
  Generate y         Estimate V(s)
       │                │
       ▼                ▼
┌──────────────┐  ┌──────────────┐
│ Reward Model │  │  Reference   │
│  r_φ(x, y)  │  │  Model π_ref │
└──────────────┘  └──────────────┘
```

Four models in memory simultaneously—a key practical challenge.

## Practical Challenges

| Challenge | Mitigation |
|-----------|-----------|
| Memory (4 models) | Model parallelism, offloading, share base weights |
| Training instability | Low learning rate (1e-6), gradient clipping, warm-up |
| Reward hacking | KL penalty, reward model ensembles |
| Hyperparameter sensitivity | $\beta$, $\epsilon$, learning rate require careful tuning |

## References

1. Schulman, J., et al. (2017). "Proximal Policy Optimization Algorithms." *arXiv*.
2. Ziegler, D., et al. (2019). "Fine-Tuning Language Models from Human Preferences." *arXiv*.
