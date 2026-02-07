# Direct Preference Optimization (DPO)

## Learning Objectives

- Understand the key insight connecting rewards to policies
- Derive the DPO loss function
- Compare DPO with RLHF in practice

## Key Insight

Rafailov et al. (2023) showed that the optimal policy under the KL-constrained reward maximization objective has a **closed-form solution**:

$$\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\left(\frac{1}{\beta} r(x, y)\right)$$

Rearranging, the reward can be expressed in terms of the policy:

$$r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$$

## DPO Loss

Substituting this reward expression into the Bradley-Terry preference model, the partition function $Z(x)$ cancels:

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l)} \left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)\right]$$

This directly optimizes the policy on preference data—**no reward model, no RL**.

## Implementation

```python
import torch
import torch.nn.functional as F


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = 0.1,
):
    """Compute DPO loss.

    All inputs are log-probabilities of the full response sequences.
    """
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps)

    loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()

    # Metrics for monitoring
    reward_margin = (chosen_rewards - rejected_rewards).mean().item()
    accuracy = (chosen_rewards > rejected_rewards).float().mean().item()

    return loss, {"reward_margin": reward_margin, "accuracy": accuracy}
```

## DPO vs. RLHF

| Aspect | RLHF | DPO |
|--------|------|-----|
| Components | Policy + RM + Value + Reference | Policy + Reference |
| Training stability | Low (RL instability) | High (supervised loss) |
| Memory | 4 models | 2 models |
| Hyperparameters | Many (PPO + KL) | Few ($\beta$, lr) |
| Sample efficiency | Lower (online generation) | Higher (offline data) |
| Quality ceiling | Potentially higher | Comparable in practice |
| Reward hacking risk | Higher | Lower |

## Variants

| Method | Year | Key Change |
|--------|------|-----------|
| DPO | 2023 | Direct preference optimization |
| IPO | 2023 | Identity preference optimization (bounded loss) |
| KTO | 2024 | Only needs good/bad labels, not pairs |
| ORPO | 2024 | Combines SFT and alignment in one stage |
| SimPO | 2024 | Reference-free, uses length-normalized rewards |

## References

1. Rafailov, R., et al. (2023). "Direct Preference Optimization: Your Language Model is Secretly a Reward Model." *NeurIPS*.
2. Azar, M., et al. (2023). "A General Theoretical Paradigm to Understand Learning from Human Feedback." *arXiv*.
