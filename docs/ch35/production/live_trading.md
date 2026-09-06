# Live Trading

Chapter 35.7.1: Live Trading Systems Production-grade live trading system components.

Deploying deep learning in quantitative finance requires robust production infrastructure. This module covers production system design patterns including monitoring, risk controls, and deployment strategies for financial applications.

## Code

```python
"""
Chapter 35.7.1: Live Trading Systems
=======================================
Production-grade live trading system components.
"""

import numpy as np
import time
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import logging

# ========================================================================
# Main
# ========================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LiveTrading")


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    symbol: str
    side: OrderSide
    quantity: float
    price: Optional[float] = None
    order_type: str = "market"
    status: OrderStatus = OrderStatus.PENDING
    fill_price: float = 0.0
    fill_quantity: float = 0.0
    timestamp: float = 0.0
    order_id: str = ""


@dataclass
class Position:
    symbol: str
    quantity: float = 0.0
    avg_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0


class FeatureStore:
    """Point-in-time feature computation and storage."""

    def __init__(self, lookback: int = 60):
        self.lookback = lookback
        self.price_buffer: Dict[str, deque] = {}
        self.feature_cache: Dict[str, np.ndarray] = {}

    def update(self, symbol: str, price: float, volume: float = 0.0):
        if symbol not in self.price_buffer:
            self.price_buffer[symbol] = deque(maxlen=self.lookback + 10)
        self.price_buffer[symbol].append(price)
        self._compute_features(symbol)

    def _compute_features(self, symbol: str):
        prices = np.array(self.price_buffer[symbol])
        if len(prices) < 2:
            return
        log_returns = np.diff(np.log(prices + 1e-8))
        features = {
            "last_price": prices[-1],
            "return_1d": log_returns[-1] if len(log_returns) > 0 else 0,
            "volatility": np.std(log_returns[-20:]) if len(log_returns) >= 20 else np.std(log_returns),
            "momentum_5d": np.sum(log_returns[-5:]) if len(log_returns) >= 5 else 0,
            "momentum_20d": np.sum(log_returns[-20:]) if len(log_returns) >= 20 else 0,
        }
        self.feature_cache[symbol] = features

    def get_features(self, symbols: List[str]) -> Optional[np.ndarray]:
        features = []
        for sym in symbols:
            if sym in self.feature_cache:
                f = self.feature_cache[sym]
                features.append([f["return_1d"], f["volatility"],
                                f["momentum_5d"], f["momentum_20d"]])
            else:
                features.append([0, 0, 0, 0])
        return np.array(features, dtype=np.float32)


class OrderManager:
    """Manages order lifecycle."""

    def __init__(self):
        self.orders: List[Order] = []
        self.positions: Dict[str, Position] = {}
        self.order_counter = 0

    def create_order(self, symbol: str, side: OrderSide,
                     quantity: float, price: Optional[float] = None) -> Order:
        self.order_counter += 1
        order = Order(
            symbol=symbol, side=side, quantity=quantity, price=price,
            order_id=f"ORD-{self.order_counter:06d}", timestamp=time.time(),
        )
        self.orders.append(order)
        return order

    def simulate_fill(self, order: Order, market_price: float,
                      spread: float = 0.01, slippage_std: float = 0.005):
        slip = np.random.normal(0, slippage_std)
        if order.side == OrderSide.BUY:
            fill_price = market_price + spread / 2 + slip
        else:
            fill_price = market_price - spread / 2 + slip

        order.fill_price = fill_price
        order.fill_quantity = order.quantity
        order.status = OrderStatus.FILLED

        sym = order.symbol
        if sym not in self.positions:
            self.positions[sym] = Position(symbol=sym)
        pos = self.positions[sym]

        if order.side == OrderSide.BUY:
            total_cost = pos.avg_price * pos.quantity + fill_price * order.quantity
            pos.quantity += order.quantity
            pos.avg_price = total_cost / (pos.quantity + 1e-8) if pos.quantity > 0 else 0
        else:
            pos.quantity -= order.quantity
            if pos.quantity <= 0:
                pos.avg_price = 0
                pos.quantity = max(0, pos.quantity)

    def get_target_orders(self, current_weights: np.ndarray,
                          target_weights: np.ndarray,
                          symbols: List[str],
                          portfolio_value: float,
                          prices: np.ndarray) -> List[Order]:
        orders = []
        delta_weights = target_weights - current_weights
        for i, sym in enumerate(symbols):
            if abs(delta_weights[i]) < 0.001:
                continue
            dollar_amount = abs(delta_weights[i]) * portfolio_value
            quantity = dollar_amount / (prices[i] + 1e-8)
            side = OrderSide.BUY if delta_weights[i] > 0 else OrderSide.SELL
            orders.append(self.create_order(sym, side, quantity))
        return orders


class LiveTradingSystem:
    """Complete live trading system."""

    def __init__(self, symbols: List[str], initial_capital: float = 1_000_000.0):
        self.symbols = symbols
        self.capital = initial_capital
        self.feature_store = FeatureStore()
        self.order_manager = OrderManager()
        self.portfolio_value = initial_capital
        self.current_weights = np.zeros(len(symbols))
        self.step_count = 0

    def on_market_data(self, prices: Dict[str, float]):
        for sym, price in prices.items():
            self.feature_store.update(sym, price)

    def generate_signals(self, model=None) -> np.ndarray:
        features = self.feature_store.get_features(self.symbols)
        if model is not None:
            return model(features)
        # Default: equal weight
        return np.ones(len(self.symbols)) / len(self.symbols)

    def execute_trades(self, target_weights: np.ndarray, prices: np.ndarray):
        orders = self.order_manager.get_target_orders(
            self.current_weights, target_weights,
            self.symbols, self.portfolio_value, prices,
        )
        for order in orders:
            idx = self.symbols.index(order.symbol)
            self.order_manager.simulate_fill(order, prices[idx])
        self.current_weights = target_weights

    def run_step(self, prices: Dict[str, float], model=None) -> Dict:
        self.on_market_data(prices)
        price_array = np.array([prices[s] for s in self.symbols])
        target = self.generate_signals(model)
        self.execute_trades(target, price_array)
        self.step_count += 1
        return {
            "step": self.step_count,
            "weights": self.current_weights.copy(),
            "portfolio_value": self.portfolio_value,
        }


def demo_live_trading():
    """Demonstrate live trading system."""
    print("=" * 70)
    print("Live Trading System Demonstration")
    print("=" * 70)

    symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "META"]
    system = LiveTradingSystem(symbols, initial_capital=1_000_000.0)

    np.random.seed(42)
    prices_base = np.array([150.0, 140.0, 380.0, 170.0, 350.0])

    print(f"\nSymbols: {symbols}")
    print(f"Initial capital: ${system.capital:,.0f}")

    for step in range(10):
        noise = 1 + np.random.randn(5) * 0.01
        current_prices = prices_base * noise * (1 + 0.001 * step)
        prices_dict = dict(zip(symbols, current_prices))

        result = system.run_step(prices_dict)
        if step % 3 == 0:
            print(f"\nStep {step}: weights={np.round(result['weights'], 3)}")

    print(f"\nFinal orders: {len(system.order_manager.orders)}")
    print(f"Feature store symbols: {list(system.feature_store.feature_cache.keys())}")


if __name__ == "__main__":
    demo_live_trading()```

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
Write a comprehensive test function that validates the Live Trading implementation. Test edge cases including empty inputs, single-element inputs, very large inputs, and inputs with extreme values (zeros, very large numbers).

??? success "Solution to Exercise 4"
    Create a test function that exercises boundary conditions:
    ```python
    def test_orderside():
        model = OrderSide(...)
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
