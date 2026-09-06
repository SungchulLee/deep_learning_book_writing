# Paper Trading

Chapter 35.7.2: Paper Trading Paper trading validation system.

Deploying deep learning in quantitative finance requires robust production infrastructure. This module covers production system design patterns including monitoring, risk controls, and deployment strategies for financial applications.

## Code

```python
"""
Chapter 35.7.2: Paper Trading
================================
Paper trading validation system.
"""

import numpy as np
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from collections import deque

# ========================================================================
# Main
# ========================================================================


@dataclass
class PaperTradingConfig:
    initial_capital: float = 1_000_000.0
    transaction_cost: float = 0.001
    slippage_model: str = "gaussian"
    slippage_std: float = 0.0005
    max_position: float = 0.25


class PaperTradingEngine:
    """Paper trading engine with realistic execution simulation."""

    def __init__(self, config: PaperTradingConfig, num_assets: int):
        self.config = config
        self.num_assets = num_assets
        self.portfolio_value = config.initial_capital
        self.peak_value = config.initial_capital
        self.weights = np.zeros(num_assets)
        self.step = 0

        self.value_history: List[float] = [config.initial_capital]
        self.return_history: List[float] = []
        self.trade_log: List[Dict] = []
        self.daily_pnl: List[float] = []

    def execute(self, target_weights: np.ndarray, prices: np.ndarray,
                next_prices: np.ndarray) -> Dict:
        target_weights = np.clip(target_weights, -self.config.max_position, self.config.max_position)

        turnover = np.sum(np.abs(target_weights - self.weights))
        tc = self.config.transaction_cost * turnover

        # Slippage
        slippage = self.config.slippage_std * np.sqrt(turnover) * abs(np.random.randn())

        returns = (next_prices - prices) / (prices + 1e-8)
        port_return = float(np.dot(target_weights, returns)) - tc - slippage

        self.portfolio_value *= (1 + port_return)
        self.peak_value = max(self.peak_value, self.portfolio_value)

        self.weights = target_weights.copy()
        self.value_history.append(self.portfolio_value)
        self.return_history.append(port_return)
        self.step += 1

        self.trade_log.append({
            "step": self.step,
            "turnover": turnover,
            "tc": tc,
            "slippage": slippage,
            "return": port_return,
        })

        drawdown = (self.peak_value - self.portfolio_value) / (self.peak_value + 1e-8)

        return {
            "portfolio_value": self.portfolio_value,
            "return": port_return,
            "drawdown": drawdown,
            "turnover": turnover,
        }

    def get_performance_summary(self) -> Dict:
        returns = np.array(self.return_history)
        if len(returns) == 0:
            return {"no_data": True}

        total_return = (self.portfolio_value / self.config.initial_capital - 1)
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        max_dd = (self.peak_value - min(self.value_history)) / (self.peak_value + 1e-8)
        avg_turnover = np.mean([t["turnover"] for t in self.trade_log])
        total_tc = sum(t["tc"] + t["slippage"] for t in self.trade_log)

        return {
            "total_return": float(total_return),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_dd),
            "avg_turnover": float(avg_turnover),
            "total_costs": float(total_tc * self.config.initial_capital),
            "num_steps": self.step,
        }


class BacktestPaperComparator:
    """Compare backtest results with paper trading results."""

    @staticmethod
    def compare(backtest_returns: np.ndarray, paper_returns: np.ndarray) -> Dict:
        bt_sharpe = np.mean(backtest_returns) / (np.std(backtest_returns) + 1e-8) * np.sqrt(252)
        pt_sharpe = np.mean(paper_returns) / (np.std(paper_returns) + 1e-8) * np.sqrt(252)

        bt_vol = np.std(backtest_returns) * np.sqrt(252)
        pt_vol = np.std(paper_returns) * np.sqrt(252)

        return {
            "backtest_sharpe": float(bt_sharpe),
            "paper_sharpe": float(pt_sharpe),
            "sharpe_diff": float(bt_sharpe - pt_sharpe),
            "backtest_vol": float(bt_vol),
            "paper_vol": float(pt_vol),
            "correlation": float(np.corrcoef(
                backtest_returns[:min(len(backtest_returns), len(paper_returns))],
                paper_returns[:min(len(backtest_returns), len(paper_returns))]
            )[0, 1]) if len(backtest_returns) > 1 and len(paper_returns) > 1 else 0,
        }


def demo_paper_trading():
    """Demonstrate paper trading."""
    print("=" * 70)
    print("Paper Trading Demonstration")
    print("=" * 70)

    np.random.seed(42)
    N, T = 5, 200
    returns_data = np.random.randn(T, N) * 0.015 + 0.0003
    prices = 100 * np.exp(np.cumsum(returns_data, axis=0))

    config = PaperTradingConfig(transaction_cost=0.001)
    engine = PaperTradingEngine(config, N)

    for t in range(T - 1):
        weights = np.random.dirichlet(np.ones(N))
        result = engine.execute(weights, prices[t], prices[t + 1])

    summary = engine.get_performance_summary()
    print(f"\n--- Paper Trading Summary ---")
    for k, v in summary.items():
        if "return" in k or "drawdown" in k:
            print(f"  {k:<20}: {v*100:.2f}%")
        elif "ratio" in k:
            print(f"  {k:<20}: {v:.4f}")
        else:
            print(f"  {k:<20}: {v}")

    # Compare with backtest
    print("\n--- Backtest vs Paper Comparison ---")
    bt_returns = np.random.randn(200) * 0.012 + 0.0004
    pt_returns = np.array(engine.return_history)
    comp = BacktestPaperComparator.compare(bt_returns, pt_returns)
    for k, v in comp.items():
        print(f"  {k:<20}: {v:.4f}")


if __name__ == "__main__":
    demo_paper_trading()```

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
Write a comprehensive test function that validates the Paper Trading implementation. Test edge cases including empty inputs, single-element inputs, very large inputs, and inputs with extreme values (zeros, very large numbers).

??? success "Solution to Exercise 4"
    Create a test function that exercises boundary conditions:
    ```python
    def test_papertradingconfig():
        model = PaperTradingConfig(...)
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
