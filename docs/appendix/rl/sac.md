# SAC

SAC was introduced in the 2018 paper "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning." - Actor maximizes expected return + entropy (exploration)   - Two Q networks (double Q) to reduce positive bias   - Target Q networks for stability.

This implementation provides a concise, educational reference for SAC. The code focuses on the core architecture and forward pass, making it straightforward to study the key design patterns and adapt them for experimentation.

## Code

```python
#!/usr/bin/env python3
"""
SAC - Soft Actor-Critic (continuous control)
Paper: "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning" (2018)
Authors: Tuomas Haarnoja et al.
Key idea:
  - Actor maximizes expected return + entropy (exploration)
  - Two Q networks (double Q) to reduce positive bias
  - Target Q networks for stability

File: appendix/rl/sac.py
Note: Educational reference (networks + core target computations).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# Main
# ========================================================================


class GaussianPolicy(nn.Module):
    """
    Stochastic actor for continuous actions:
      a ~ N(mu(s), sigma(s)), then typically tanh-squashed to bounds [-1,1].
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)

    def forward(self, obs):
        h = self.net(obs)
        mu = self.mu(h)
        log_std = self.log_std(h).clamp(-20, 2)  # numerical stability
        std = torch.exp(log_std)
        return mu, std

    def sample(self, obs):
        """
        Reparameterized sampling:
          a = tanh(mu + std * eps)
        Also returns log_prob(a) adjusted for tanh squashing (omitted for brevity).
        """
        mu, std = self.forward(obs)
        eps = torch.randn_like(std)
        pre_tanh = mu + std * eps
        a = torch.tanh(pre_tanh)
        # For a real SAC implementation, compute log_prob with tanh correction.
        log_prob = None
        return a, log_prob


class QNetwork(nn.Module):
    """Critic Q(s,a) for continuous actions."""
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=1)
        return self.net(x).squeeze(1)  # (B,)


def sac_target(q1_tgt, q2_tgt, policy, s2, r, done, gamma=0.99, alpha=0.2):
    """
    Compute SAC target:
      a' ~ pi(s')
      y = r + gamma*(1-done) * ( min(Q1_tgt(s',a'), Q2_tgt(s',a')) - alpha * log pi(a'|s') )

    Here, log pi term is omitted for brevity; include it in a full implementation.
    """
    with torch.no_grad():
        a2, logp2 = policy.sample(s2)
        q1v = q1_tgt(s2, a2)
        q2v = q2_tgt(s2, a2)
        qmin = torch.min(q1v, q2v)

        # If logp2 is None (as in this educational code), ignore entropy term.
        if logp2 is None:
            backup = qmin
        else:
            backup = qmin - alpha * logp2

        y = r + gamma * (1.0 - done) * backup
    return y


if __name__ == "__main__":
    pass```

## Discussion

The implementation defines 2 classes (`GaussianPolicy`, `QNetwork`) that work together to form the complete reinforcement learning architecture. Each class encapsulates a distinct component, making the code modular and easy to extend. The `forward` methods define the computational graph that PyTorch uses for automatic differentiation.

As a reference implementation, this code prioritizes clarity over optimization. Production systems would typically add mixed-precision training, distributed data parallelism, and more sophisticated data augmentation. Nevertheless, the core architectural ideas shown here remain the same regardless of scale.

## Exercises

**Exercise 1.**
Calculate the total number of learnable parameters in `GaussianPolicy` with the default initialization. Break down the count by layer, including both weights and biases.

??? success "Solution to Exercise 1"
    For each `nn.Linear(in_features, out_features)`, there are `in_features * out_features` weight parameters plus `out_features` bias parameters (unless `bias=False`). For `nn.Conv2d(in_c, out_c, k)`, there are `in_c * out_c * k * k` weight parameters plus `out_c` bias parameters. For `nn.Embedding(num, dim)`, there are `num * dim` parameters. Sum across all layers. You can verify with `sum(p.numel() for p in model.parameters())`.

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
Extend `GaussianPolicy` to support a configurable number of layers or blocks. Add a `num_layers` parameter to `__init__` and use `nn.ModuleList` to create a variable-depth architecture. Test with 2, 4, and 8 layers.

??? success "Solution to Exercise 4"
    Replace the hardcoded layers with:
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    In the `forward` method, iterate: `for layer in self.layers: x = layer(x)`. Using `nn.ModuleList` (not a plain Python list) ensures PyTorch registers all parameters for optimization. Test with: `for n in [2, 4, 8]: model = GaussianPolicy(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.
