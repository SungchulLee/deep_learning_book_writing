# 살아 있는 거래

35.7.1장: 살아 있는 거래 얼개. 서비스 품질의 살아 있는 거래 얼개 조각들.

계량 금융에 깊은 배움을 올리려면 든든한 서비스 바탕이 있어야 한다. 이 꾸러미는 지켜보기, 무릅씀 다스리기, 금융 쓰임을 서비스에 올리는 꾀를 아우르는 서비스 얼개 설계 결을 다룬다.

## 1. 코드

```python
"""
35.7.1장: 살아 있는 거래 얼개
=======================================
서비스 품질의 살아 있는 거래 얼개 조각들.
"""

import numpy as np
import time
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import logging

# ========================================================================
# 메인
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
    """그때 그 자리의 결을 셈하고 갈무리한다."""

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
    """주문이 나고 스러지는 삶을 다룬다."""

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
    """온전히 갖춘 살아 있는 거래 얼개."""

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
        # 맡긴 값: 고른 몫
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
    """살아 있는 거래 얼개를 보인다."""
    print("=" * 70)
    print("살아 있는 거래 얼개 보이기")
    print("=" * 70)

    symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "META"]
    system = LiveTradingSystem(symbols, initial_capital=1_000_000.0)

    np.random.seed(42)
    prices_base = np.array([150.0, 140.0, 380.0, 170.0, 350.0])

    print(f"\n종목: {symbols}")
    print(f"첫 밑천: ${system.capital:,.0f}")

    for step in range(10):
        noise = 1 + np.random.randn(5) * 0.01
        current_prices = prices_base * noise * (1 + 0.001 * step)
        prices_dict = dict(zip(symbols, current_prices))

        result = system.run_step(prices_dict)
        if step % 3 == 0:
            print(f"\n걸음 {step}: 몫={np.round(result['weights'], 3)}")

    print(f"\n마지막 주문 수: {len(system.order_manager.orders)}")
    print(f"결 곳간의 종목: {list(system.feature_store.feature_cache.keys())}")


if __name__ == "__main__":
    demo_live_trading()
```

## 2. 논의

이 짜보기는 깔끔하고 읽기 쉬운 PyTorch 코드로 서비스 얼개의 고갱이가 되는 생각을 보여 준다. 조각으로 나눈 얼개 덕에 부분마다 따로 살피고 다른 일이나 자료에 맞추어 고치기 쉽다.

여기서 보인 결은 더 까다로운 자리로도 자연스레 넓혀진다. 하이퍼파라미터, 얼개의 갈래, 여러 자료를 바꿔 가며 해 보면 이해가 깊어지고 서비스에 올리는 일에 대한 감이 몸에 붙는다.

## 연습문제

**연습문제 1.**
코드를 읽고 고갱이가 되는 설계 판단을 짚어라. 짜기에서 고른 것 셋을 들고, 저마다 왜 서비스 얼개에 알맞은지 밝혀라.

??? success "연습문제 1 풀이"
    설계 판단은 짜보기마다 다르나 흔히 이런 것이 있다. (1) 살림 함수 고르기 -- ReLU 갈래는 기울기가 잦아들지 않아 익히기가 빠르다. (2) 고르게 하는 꾀 -- 묶음 고르게 하기가 안쪽 함께 바뀌는 옮겨감을 줄여 익힘을 든든하게 한다. (3) 나머지 이음 -- 있으면 건너뛰는 길을 주어 깊은 그물에서 기울기가 흐르게 한다. 고른 것마다 나타내는 힘, 셈 값, 익힘의 든든함 사이의 맞바꿈을 드러낸다.

---

**연습문제 2.**
들임의 꼴과 자료 갈래가 바라는 대로인지 살피는 들임 살피기를 으뜸 함수나 클래스에 더하여라. 올바르지 않은 들임에는 알아듣기 쉬운 어긋남 알림을 띄워라.

??? success "연습문제 2 풀이"
    `forward` 방법(또는 알맞은 함수)의 첫머리에 `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`이나 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'` 같은 살핌을 더한다. 꼴을 살피려면 종요로운 차원을 본다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 알아듣기 쉬운 어긋남 알림은 벌레잡기를 크게 앞당기고 코드를 되쓰기 든든하게 한다.

---

**연습문제 3.**
이 짜보기가 무너질 만한 결 둘을 밝히고, 저마다 어떻게 짚어내고 고칠지 밝혀라.

??? success "연습문제 3 풀이"
    흔히 무너지는 결은 이렇다. (1) **기울기가 사라지거나 터짐** -- 기울기 크기를 지켜보아 짚어낸다(`torch.nn.utils.clip_grad_norm_`이나 켜마다 `param.grad.norm()` 적기). 기울기 자르기, 더 나은 첫값 잡기(Xavier/Kaiming), 얼개 고치기(나머지 이음, 고르게 하기)로 고친다. (2) **지나치게 맞추기** -- 익힘 잃음은 줄어드는데 살핌 잃음이 오르면 짚어낸다. 정칙화(드롭아웃, 짐 줄이기, 자료 늘리기)나 모형 크기 줄이기로 고친다. 익힘과 살핌 자를 늘 함께 지켜보아 이를 일찍 잡아야 한다.

---

**연습문제 4.**
살아 있는 거래 짜보기를 살피는 두루 갖춘 시험 함수를 써라. 빈 들임, 원소 하나짜리 들임, 아주 큰 들임, 그리고 끝자락 값(0, 아주 큰 수)이 든 들임 같은 가장자리 자리를 시험하여라.

??? success "연습문제 4 풀이"
    금 언저리 조건을 두루 건드리는 시험 함수를 짓는다.
    ```python
    def test_orderside():
        model = OrderSide(...)
        # 여느 들임
        assert model(normal_input).shape == expected_shape
        # 원소 하나짜리 묶음
        assert model(single_input).shape == (1, ...)
        # 큰 값(넘침을 살핀다)
        out = model(torch.ones(...) * 1000)
        assert torch.isfinite(out).all()
        # 기울기 흐름
        out = model(normal_input)
        out.sum().backward()
        for p in model.parameters():
            assert p.grad is not None
    ```
    얼개가 끝에서 끝까지 익히기를 받치는지 알려면 기울기 흐름을 시험하는 것이 특히 중요하다.

## 정리하며

**다룬 것** — 살아 있는 거래

이 짜보기는 깔끔하고 읽기 쉬운 PyTorch 코드로 서비스 얼개의 고갱이가 되는 생각을 보여 준다.

고갱이 갈래는 `OrderSide`, `OrderStatus`, `Order`, `Position`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
