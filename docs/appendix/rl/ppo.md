# PPO

PPO was introduced in the 2017 paper "Proximal Policy Optimization Algorithms." - Policy gradient with a *clipped surrogate objective* to prevent large updates   - Often uses GAE advantages and minibatch epochs.

This implementation provides a concise, educational reference for PPO. The code focuses on the core architecture and forward pass, making it straightforward to study the key design patterns and adapt them for experimentation.

## Code

```python
#!/usr/bin/env python3
"""
PPO - Proximal Policy Optimization
Paper: "Proximal Policy Optimization Algorithms" (2017)
Authors: John Schulman et al.
Key idea:
  - Policy gradient with a *clipped surrogate objective* to prevent large updates
  - Often uses GAE advantages and minibatch epochs

Clipped objective:
  r_t = pi(a|s) / pi_old(a|s)
  L_clip = E[ min( r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t ) ]

File: appendix/rl/ppo.py
Note: Educational implementation of PPO loss (discrete actions).
"""

import torch
import torch.nn.functional as F

# ========================================================================
# Main
# ========================================================================


def ppo_loss(
    logits_new,        # (B, A) current policy logits
    logits_old,        # (B, A) behavior/old policy logits (frozen)
    actions,           # (B,)
    advantages,        # (B,)
    returns,           # (B,)
    values,            # (B,) critic values from current network
    clip_eps=0.2,
    value_coef=0.5,
    entropy_coef=0.01,
):
    """
    Compute PPO losses:
      - clipped policy loss
      - value loss
      - entropy bonus
    """
    # Compute log probs under new and old policies
    logp_new = F.log_softmax(logits_new, dim=1)
    logp_old = F.log_softmax(logits_old, dim=1)

    logp_new_a = logp_new.gather(1, actions.long().unsqueeze(1)).squeeze(1)
    logp_old_a = logp_old.gather(1, actions.long().unsqueeze(1)).squeeze(1)

    # Importance ratio r_t = exp(log pi_new - log pi_old)
    ratio = torch.exp(logp_new_a - logp_old_a)

    # Advantages are typically standardized; detach so actor doesn't backprop into advantage
    A = advantages.detach()

    # Clipped surrogate objective
    unclipped = ratio * A
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * A
    policy_loss = -torch.min(unclipped, clipped).mean()

    # Value function loss
    value_loss = F.mse_loss(values, returns)

    # Entropy bonus (encourage exploration)
    p_new = torch.softmax(logits_new, dim=1)
    entropy = -(p_new * logp_new).sum(dim=1).mean()

    total = policy_loss + value_coef * value_loss - entropy_coef * entropy
    return total, {"policy_loss": policy_loss, "value_loss": value_loss, "entropy": entropy}


if __name__ == "__main__":
    pass```

## Discussion

This implementation demonstrates key concepts in reinforcement learning using clean, readable PyTorch code. The modular structure makes it easy to study individual components and adapt them for different tasks or datasets.

As a reference implementation, this code prioritizes clarity over optimization. Production systems would typically add mixed-precision training, distributed data parallelism, and more sophisticated data augmentation. Nevertheless, the core architectural ideas shown here remain the same regardless of scale.

## Exercises

**Exercise 1.**
Read through the code and identify the key design decisions. List three specific implementation choices and explain why each is appropriate for reinforcement learning.

??? success "Solution to Exercise 1"
    Design decisions vary by implementation but commonly include: (1) choice of activation functions -- ReLU variants provide non-saturating gradients for faster training; (2) normalization strategy -- batch normalization stabilizes training by reducing internal covariate shift; (3) residual connections -- when present, they enable gradient flow in deep networks by providing skip paths. Each choice reflects a trade-off between expressiveness, computational cost, and training stability.

---

**Exercise 2.**
Add input validation to the main function or class to check that inputs have the expected shape and dtype. Raise informative error messages for invalid inputs.

??? success "Solution to Exercise 2"
    At the start of the `forward` method (or relevant function), add checks like: `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'` and `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'`. For shape validation, check critical dimensions: `B, C, H, W = x.shape; assert C == self.expected_channels`. Informative error messages significantly speed up debugging and make the code more robust for reuse.

---

**Exercise 3.**
Describe two potential failure modes of this implementation and explain how you would diagnose and fix each one.

??? success "Solution to Exercise 3"
    Common failure modes include: (1) **Vanishing/exploding gradients** -- diagnosed by monitoring gradient norms (`torch.nn.utils.clip_grad_norm_` or logging `param.grad.norm()` per layer). Fix with gradient clipping, better initialization (Xavier/Kaiming), or architectural changes (residual connections, normalization). (2) **Overfitting** -- diagnosed when training loss decreases but validation loss increases. Fix with regularization (dropout, weight decay, data augmentation) or reducing model capacity. Always monitor both training and validation metrics to catch these issues early.

---

**Exercise 4.**
Write a comprehensive test function that validates the PPO implementation. Test edge cases including empty inputs, single-element inputs, very large inputs, and inputs with extreme values (zeros, very large numbers).

??? success "Solution to Exercise 4"
    Create a test function that exercises boundary conditions:
    ```python
    def test_ppo():
        model = PPO(...)
        # Normal input
        assert model(normal_input).shape == expected_shape
        # Single element batch
        assert model(single_input).shape == (1, ...)
        # Large values (check for overflow)
        out = model(torch.ones(...) * 1000)
        assert torch.isfinite(out).all()
        # Gradient flow
        out = model(normal_input)
        out.sum().backward()
        for p in model.parameters():
            assert p.grad is not None
    ```
    Testing gradient flow is especially important to ensure the architecture supports end-to-end training.
