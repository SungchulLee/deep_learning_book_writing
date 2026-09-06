# Debugging Reinforcement Learning

Debugging Reinforcement Learning is an important concept in practical RL techniques. This implementation provides a hands-on demonstration of the key algorithms and data structures involved, illustrating both the theoretical foundations and practical considerations for real-world deployment.

## Code

```python
"""
Chapter 34.6.4: Debugging Reinforcement Learning
==================================================
Diagnostic tools, health checks, and monitoring utilities.
"""

import torch
import torch.nn as nn
import numpy as np
from collections import deque
from typing import Dict, List, Optional

# ========================================================================
# Main
# ========================================================================


class RLDiagnostics:
    """
    Comprehensive RL training diagnostics.
    Tracks and reports key health metrics.
    """
    
    def __init__(self, window=100):
        self.window = window
        self.metrics = {
            "episode_rewards": deque(maxlen=window),
            "episode_lengths": deque(maxlen=window),
            "policy_loss": deque(maxlen=window),
            "value_loss": deque(maxlen=window),
            "entropy": deque(maxlen=window),
            "kl_divergence": deque(maxlen=window),
            "clip_fraction": deque(maxlen=window),
            "grad_norm": deque(maxlen=window),
            "value_predictions": deque(maxlen=window),
            "actual_returns": deque(maxlen=window),
        }
        self.alerts = []
    
    def log(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.metrics:
                self.metrics[key].append(value)
    
    def compute_explained_variance(self):
        """EV = 1 - Var(returns - predictions) / Var(returns)."""
        if len(self.metrics["value_predictions"]) < 10:
            return 0.0
        preds = np.array(list(self.metrics["value_predictions"]))
        actuals = np.array(list(self.metrics["actual_returns"]))
        var_actual = np.var(actuals)
        if var_actual < 1e-8:
            return 0.0
        return 1.0 - np.var(actuals - preds) / var_actual
    
    def health_check(self) -> List[str]:
        """Run diagnostic checks and return warnings."""
        alerts = []
        
        # Entropy check
        if len(self.metrics["entropy"]) >= 10:
            recent_entropy = list(self.metrics["entropy"])[-10:]
            if all(e < 0.01 for e in recent_entropy):
                alerts.append("⚠️ ENTROPY COLLAPSE: Policy may be stuck (entropy near 0)")
        
        # Gradient norm check
        if len(self.metrics["grad_norm"]) >= 10:
            recent_grads = list(self.metrics["grad_norm"])[-10:]
            if any(np.isnan(g) or np.isinf(g) for g in recent_grads):
                alerts.append("🔥 NaN/Inf GRADIENTS detected!")
            elif np.mean(recent_grads) > 100:
                alerts.append("⚠️ LARGE GRADIENTS: Consider reducing learning rate")
        
        # KL divergence check
        if len(self.metrics["kl_divergence"]) >= 10:
            recent_kl = list(self.metrics["kl_divergence"])[-10:]
            if np.mean(recent_kl) > 0.1:
                alerts.append("⚠️ HIGH KL: Policy changing too fast")
        
        # Clip fraction check
        if len(self.metrics["clip_fraction"]) >= 10:
            recent_cf = list(self.metrics["clip_fraction"])[-10:]
            avg_cf = np.mean(recent_cf)
            if avg_cf > 0.5:
                alerts.append("⚠️ HIGH CLIP FRACTION: Consider reducing learning rate or clip range")
            elif avg_cf < 0.01:
                alerts.append("⚠️ LOW CLIP FRACTION: Updates may be too conservative")
        
        # Value function check
        ev = self.compute_explained_variance()
        if ev < 0:
            alerts.append("⚠️ NEGATIVE EXPLAINED VARIANCE: Value function worse than mean prediction")
        
        # Learning progress
        if len(self.metrics["episode_rewards"]) >= self.window:
            rewards = list(self.metrics["episode_rewards"])
            first_half = np.mean(rewards[:len(rewards)//2])
            second_half = np.mean(rewards[len(rewards)//2:])
            if second_half < first_half * 0.9:
                alerts.append("⚠️ PERFORMANCE DEGRADATION: Reward declining")
        
        return alerts
    
    def report(self) -> str:
        """Generate a training status report."""
        lines = ["=" * 50, "RL Training Diagnostics Report", "=" * 50]
        
        for key in ["episode_rewards", "policy_loss", "value_loss", "entropy",
                     "kl_divergence", "clip_fraction", "grad_norm"]:
            vals = list(self.metrics[key])
            if vals:
                lines.append(
                    f"{key:<20}: mean={np.mean(vals):>8.4f}  "
                    f"std={np.std(vals):>8.4f}  "
                    f"last={vals[-1]:>8.4f}"
                )
        
        ev = self.compute_explained_variance()
        lines.append(f"{'explained_variance':<20}: {ev:.4f}")
        
        alerts = self.health_check()
        if alerts:
            lines.append("\n" + "-" * 50)
            lines.append("ALERTS:")
            for alert in alerts:
                lines.append(f"  {alert}")
        else:
            lines.append("\n✅ All diagnostics healthy")
        
        return "\n".join(lines)


def check_gradient_flow(model: nn.Module) -> Dict[str, float]:
    """Check gradient flow through all layers."""
    grad_info = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_info[name] = {
                "mean": param.grad.abs().mean().item(),
                "max": param.grad.abs().max().item(),
                "has_nan": torch.isnan(param.grad).any().item(),
                "has_inf": torch.isinf(param.grad).any().item(),
            }
    return grad_info


def verify_loss_sign():
    """Verify that loss is constructed correctly for policy gradient."""
    print("=" * 60)
    print("Loss Sign Verification")
    print("=" * 60)
    
    # Policy gradient: we MAXIMIZE E[log_prob * advantage]
    # As a loss to MINIMIZE: L = -E[log_prob * advantage]
    
    log_prob = torch.tensor(-1.5, requires_grad=True)
    advantage = torch.tensor(2.0)  # Positive: good action
    
    # Correct: negative sign (we minimize the negative)
    correct_loss = -(log_prob * advantage)
    correct_loss.backward()
    print(f"Positive advantage (A={advantage.item()}):")
    print(f"  Loss = {correct_loss.item():.4f}")
    print(f"  Gradient = {log_prob.grad.item():.4f}")
    print(f"  Direction: {'Increase' if log_prob.grad.item() < 0 else 'Decrease'} log_prob ✓")
    
    log_prob2 = torch.tensor(-1.5, requires_grad=True)
    advantage2 = torch.tensor(-1.0)  # Negative: bad action
    
    correct_loss2 = -(log_prob2 * advantage2)
    correct_loss2.backward()
    print(f"\nNegative advantage (A={advantage2.item()}):")
    print(f"  Loss = {correct_loss2.item():.4f}")
    print(f"  Gradient = {log_prob2.grad.item():.4f}")
    print(f"  Direction: {'Decrease' if log_prob2.grad.item() > 0 else 'Increase'} log_prob ✓")


def demo_diagnostics():
    """Simulate training and show diagnostic output."""
    print("\n" + "=" * 60)
    print("Training Diagnostics Simulation")
    print("=" * 60)
    
    diag = RLDiagnostics(window=50)
    
    # Simulate healthy training
    for step in range(100):
        reward = 100 + step * 2 + np.random.randn() * 20
        diag.log(
            episode_rewards=reward,
            policy_loss=0.5 - step * 0.003 + np.random.randn() * 0.1,
            value_loss=1.0 - step * 0.005 + np.random.randn() * 0.2,
            entropy=0.7 - step * 0.005,
            kl_divergence=0.015 + np.random.randn() * 0.005,
            clip_fraction=0.15 + np.random.randn() * 0.05,
            grad_norm=0.3 + np.random.randn() * 0.1,
            value_predictions=reward + np.random.randn() * 10,
            actual_returns=reward,
        )
    
    print(diag.report())
    
    # Simulate problematic training
    print("\n\n" + "=" * 50)
    print("Simulating Problematic Training...")
    print("=" * 50)
    
    diag2 = RLDiagnostics(window=20)
    for step in range(50):
        diag2.log(
            episode_rewards=50 - step * 0.5,  # Degrading
            entropy=0.001,  # Collapsed
            kl_divergence=0.2,  # Too high
            clip_fraction=0.7,  # Too high
            grad_norm=200 + step * 10,  # Exploding
            value_predictions=np.random.randn() * 100,
            actual_returns=50.0,
        )
    
    print(diag2.report())


if __name__ == "__main__":
    verify_loss_sign()
    demo_diagnostics()```

## Discussion

The implementation centers on the `RLDiagnostics` class, which encapsulates the core logic of debugging reinforcement learning. The code follows a modular design that separates the algorithmic components from the demonstration and evaluation logic.

The demonstration function shows the practical application of these components on synthetic data that highlights the key behaviors. By examining the output, one can observe how the algorithm's performance varies with different hyperparameter choices and problem configurations.

From a practical standpoint, this implementation prioritizes clarity over raw performance. Production systems would typically incorporate additional optimizations such as batched computation, GPU acceleration, and more sophisticated hyperparameter tuning. Nevertheless, the core algorithmic ideas demonstrated here transfer directly to large-scale applications.

## Exercises

**Exercise 1.**
Run the demonstration code and record the key output metrics. Modify one hyperparameter (such as the learning rate, hidden dimension, or number of layers) and describe how the results change.

??? success "Solution to Exercise 1"
    After running the demo, systematically vary the chosen hyperparameter while keeping others fixed. For example, doubling the hidden dimension typically improves representational capacity but increases computation time. The learning rate has a non-monotonic effect: too small leads to slow convergence, while too large causes instability. Document the specific numbers for at least three different values of the chosen hyperparameter.

---

**Exercise 2.**
Explain the role of the key architectural choices in the implementation. Why are specific activation functions, normalization strategies, or loss functions used? What would happen if you substituted alternatives?

??? success "Solution to Exercise 2"
    The architectural choices reflect established best practices in practical RL techniques. For instance, ReLU activations provide non-linearity while avoiding vanishing gradients for positive inputs. The loss function is chosen to match the task type (cross-entropy for classification, MSE for regression). Substituting alternatives (e.g., sigmoid activations, L1 loss) would change the optimization landscape and potentially degrade performance, though some substitutions may be beneficial in specific scenarios.

---

**Exercise 3.**
Extend the implementation to handle a more challenging scenario: either a larger dataset, a different problem variant, or an additional feature. Describe your modification and evaluate its impact on performance.

??? success "Solution to Exercise 3"
    One natural extension is to add regularization (dropout, weight decay) or a more sophisticated architecture (additional layers, skip connections). Implement the chosen extension, train on the same data, and compare metrics before and after. The extension should demonstrate understanding of both the original algorithm and the modification's theoretical motivation.
