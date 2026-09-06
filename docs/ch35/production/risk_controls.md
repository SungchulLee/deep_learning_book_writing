# Risk Controls

Chapter 35.7.4: Risk Controls Production risk controls with kill switches and position limits.

Deploying deep learning in quantitative finance requires robust production infrastructure. This module covers production system design patterns including monitoring, risk controls, and deployment strategies for financial applications.

## Code

```python
"""
Chapter 35.7.4: Risk Controls
================================
Production risk controls with kill switches and position limits.
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

# ========================================================================
# Main
# ========================================================================


class RiskAction(Enum):
    PASS = "pass"
    SCALE_DOWN = "scale_down"
    FLATTEN = "flatten"
    HALT = "halt"


@dataclass
class RiskControlConfig:
    max_position_per_asset: float = 0.25
    max_leverage: float = 1.5
    max_daily_loss: float = 0.02
    max_drawdown: float = 0.10
    max_daily_turnover: float = 2.0
    max_order_size: float = 0.10
    vol_scaling_threshold: float = 0.02
    vol_target: float = 0.10


class PreTradeRiskCheck:
    """Pre-trade risk checks before order submission."""

    def __init__(self, config: RiskControlConfig):
        self.config = config

    def check(self, target_weights: np.ndarray, current_weights: np.ndarray,
              portfolio_value: float) -> Dict:
        issues = []

        # Position limits
        max_pos = np.max(np.abs(target_weights))
        if max_pos > self.config.max_position_per_asset:
            issues.append(f"Position limit: max={max_pos:.3f}")

        # Leverage
        leverage = np.sum(np.abs(target_weights))
        if leverage > self.config.max_leverage:
            issues.append(f"Leverage limit: {leverage:.3f}")

        # Order size (fat-finger)
        delta = np.abs(target_weights - current_weights)
        max_order = np.max(delta)
        if max_order > self.config.max_order_size:
            issues.append(f"Order size: max={max_order:.3f}")

        passed = len(issues) == 0
        return {"passed": passed, "issues": issues}

    def enforce(self, target_weights: np.ndarray) -> np.ndarray:
        """Clip weights to satisfy constraints."""
        w = np.clip(target_weights, -self.config.max_position_per_asset,
                     self.config.max_position_per_asset)
        leverage = np.sum(np.abs(w))
        if leverage > self.config.max_leverage:
            w *= self.config.max_leverage / leverage
        return w


class KillSwitch:
    """Emergency kill switch for trading system."""

    def __init__(self, config: RiskControlConfig):
        self.config = config
        self.triggered = False
        self.trigger_reason = ""
        self.daily_pnl = 0.0
        self.peak_value = 1.0
        self.current_value = 1.0
        self.daily_turnover = 0.0

    def reset_daily(self):
        self.daily_pnl = 0.0
        self.daily_turnover = 0.0

    def update(self, portfolio_return: float, turnover: float) -> RiskAction:
        if self.triggered:
            return RiskAction.HALT

        self.daily_pnl += portfolio_return
        self.daily_turnover += turnover
        self.current_value *= (1 + portfolio_return)
        self.peak_value = max(self.peak_value, self.current_value)

        drawdown = (self.peak_value - self.current_value) / (self.peak_value + 1e-8)

        # Check triggers
        if self.daily_pnl < -self.config.max_daily_loss:
            self.triggered = True
            self.trigger_reason = f"Daily loss limit: {self.daily_pnl*100:.1f}%"
            return RiskAction.FLATTEN

        if drawdown > self.config.max_drawdown:
            self.triggered = True
            self.trigger_reason = f"Drawdown limit: {drawdown*100:.1f}%"
            return RiskAction.FLATTEN

        if self.daily_turnover > self.config.max_daily_turnover:
            self.trigger_reason = f"Turnover limit: {self.daily_turnover:.1f}x"
            return RiskAction.HALT

        # Scale down when approaching limits
        if drawdown > self.config.max_drawdown * 0.7:
            return RiskAction.SCALE_DOWN
        if self.daily_pnl < -self.config.max_daily_loss * 0.7:
            return RiskAction.SCALE_DOWN

        return RiskAction.PASS

    def manual_trigger(self, reason: str = "Manual override"):
        self.triggered = True
        self.trigger_reason = reason

    def manual_reset(self):
        self.triggered = False
        self.trigger_reason = ""


class VolatilityScaler:
    """Scale positions based on current volatility."""

    def __init__(self, config: RiskControlConfig, lookback: int = 20):
        self.config = config
        self.lookback = lookback
        self.return_buffer: List[float] = []

    def update(self, portfolio_return: float):
        self.return_buffer.append(portfolio_return)
        if len(self.return_buffer) > self.lookback * 2:
            self.return_buffer = self.return_buffer[-self.lookback * 2:]

    def get_scale(self) -> float:
        if len(self.return_buffer) < self.lookback:
            return 1.0
        recent_vol = np.std(self.return_buffer[-self.lookback:]) * np.sqrt(252)
        if recent_vol < 1e-8:
            return 1.0
        scale = self.config.vol_target / recent_vol
        return np.clip(scale, 0.1, 2.0)


class ProductionRiskManager:
    """Complete production risk management system."""

    def __init__(self, config: RiskControlConfig, num_assets: int):
        self.config = config
        self.pre_trade = PreTradeRiskCheck(config)
        self.kill_switch = KillSwitch(config)
        self.vol_scaler = VolatilityScaler(config)
        self.num_assets = num_assets

    def process_action(self, target_weights: np.ndarray,
                       current_weights: np.ndarray,
                       portfolio_return: float = 0.0,
                       turnover: float = 0.0) -> Dict:
        # Update risk state
        self.vol_scaler.update(portfolio_return)
        action = self.kill_switch.update(portfolio_return, turnover)

        if action == RiskAction.HALT:
            return {"weights": current_weights, "action": action,
                    "reason": self.kill_switch.trigger_reason}

        if action == RiskAction.FLATTEN:
            return {"weights": np.zeros(self.num_assets), "action": action,
                    "reason": self.kill_switch.trigger_reason}

        # Vol scaling
        vol_scale = self.vol_scaler.get_scale()
        scaled_weights = target_weights * vol_scale

        # Scale down if approaching limits
        if action == RiskAction.SCALE_DOWN:
            scaled_weights *= 0.5

        # Pre-trade checks and enforcement
        check = self.pre_trade.check(scaled_weights, current_weights, 1.0)
        final_weights = self.pre_trade.enforce(scaled_weights)

        return {
            "weights": final_weights,
            "action": action,
            "vol_scale": vol_scale,
            "pre_trade_passed": check["passed"],
            "issues": check["issues"],
        }


def demo_risk_controls():
    """Demonstrate production risk controls."""
    print("=" * 70)
    print("Production Risk Controls Demonstration")
    print("=" * 70)

    config = RiskControlConfig(
        max_position_per_asset=0.25, max_leverage=1.5,
        max_daily_loss=0.02, max_drawdown=0.10,
    )
    N = 5
    rm = ProductionRiskManager(config, N)

    np.random.seed(42)
    weights = np.ones(N) / N

    print("\n--- Simulation with Risk Controls ---")
    for step in range(50):
        target = np.random.dirichlet(np.ones(N)) * 1.2  # Slightly aggressive
        ret = np.random.randn() * 0.01 + 0.0002
        if 20 <= step <= 30:
            ret -= 0.012  # Drawdown

        turnover = np.sum(np.abs(target - weights))
        result = rm.process_action(target, weights, ret, turnover)
        weights = result["weights"]

        action = result["action"]
        if action != RiskAction.PASS:
            print(f"  Step {step}: [{action.value}] "
                  f"{result.get('reason', '')} "
                  f"vol_scale={result.get('vol_scale', 'N/A')}")

    print(f"\nKill switch triggered: {rm.kill_switch.triggered}")
    if rm.kill_switch.triggered:
        print(f"Reason: {rm.kill_switch.trigger_reason}")

    # Pre-trade check examples
    print("\n--- Pre-Trade Risk Checks ---")
    tests = [
        ("Normal", np.array([0.2, 0.2, 0.2, 0.2, 0.2])),
        ("Concentrated", np.array([0.5, 0.3, 0.1, 0.05, 0.05])),
        ("High leverage", np.array([0.5, 0.4, 0.3, 0.2, 0.1])),
    ]
    for name, w in tests:
        check = rm.pre_trade.check(w, np.ones(N) / N, 1e6)
        enforced = rm.pre_trade.enforce(w)
        print(f"  {name:<15}: passed={check['passed']}, "
              f"issues={check['issues'] or 'none'}")
        print(f"  {'':15}  enforced={np.round(enforced, 3)}")


if __name__ == "__main__":
    demo_risk_controls()```

## Discussion

This implementation demonstrates key concepts in production system using clean, readable PyTorch code. The modular structure makes it easy to study individual components and adapt them for different tasks or datasets.

The patterns demonstrated here extend naturally to more complex scenarios. Experimenting with hyperparameters, architectural variations, and different datasets deepens understanding and builds practical intuition for deployment tasks.

## Exercises

**Exercise 1.**
Read through the code and identify the key design decisions. List three specific implementation choices and explain why each is appropriate for production system.

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
Write a comprehensive test function that validates the Risk Controls implementation. Test edge cases including empty inputs, single-element inputs, very large inputs, and inputs with extreme values (zeros, very large numbers).

??? success "Solution to Exercise 4"
    Create a test function that exercises boundary conditions:
    ```python
    def test_riskaction():
        model = RiskAction(...)
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
