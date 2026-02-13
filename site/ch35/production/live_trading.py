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
    demo_live_trading()
