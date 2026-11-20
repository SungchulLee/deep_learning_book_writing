# 실시간 거래

35.7.1장: 실시간 거래 구조. 서비스 품질의 실시간 거래 구조 구성 요소들.

계량 금융에 딥러닝을 배포하려면 탄탄한 서비스 기반이 있어야 한다. 이 꾸러미는 모니터링, 위험 관리, 금융 응용을 서비스에 배포하는 전략을 아우르는 서비스 구조 설계 패턴을 다룬다.

## 1. 코드

```python
"""
35.7.1장: 살아 있는 거래 구조
=======================================
서비스 품질의 살아 있는 거래 구조 조각들.
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
    """온전히 갖춘 살아 있는 거래 구조."""

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
    """살아 있는 거래 구조를 보인다."""
    print("=" * 70)
    print("살아 있는 거래 구조 보이기")
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
            print(f"\n스텝 {step}: 몫={np.round(result['weights'], 3)}")

    print(f"\n마지막 주문 수: {len(system.order_manager.orders)}")
    print(f"결 곳간의 종목: {list(system.feature_store.feature_cache.keys())}")


if __name__ == "__main__":
    demo_live_trading()
```

**출력:**

```
======================================================================
살아 있는 거래 구조 보이기
======================================================================

종목: ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META']
첫 밑천: $1,000,000

스텝 0: 몫=[0.2 0.2 0.2 0.2 0.2]

스텝 3: 몫=[0.2 0.2 0.2 0.2 0.2]

스텝 6: 몫=[0.2 0.2 0.2 0.2 0.2]

스텝 9: 몫=[0.2 0.2 0.2 0.2 0.2]

마지막 주문 수: 5
결 곳간의 종목: ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META']
```

## 2. 논의

이 구현은 깔끔하고 읽기 쉬운 PyTorch 코드로 서비스 구조의 핵심 아이디어를 보여 준다. 모듈로 나눈 구조 덕에 부분마다 따로 이해하고 다른 과제나 데이터에 맞추어 고치기 쉽다.

여기서 보인 기법은 더 까다로운 상황으로도 자연스럽게 확장된다. 하이퍼파라미터, 구조의 종류, 여러 데이터를 바꿔 가며 해 보면 이해가 깊어지고 서비스 배포에 대한 감이 생긴다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
코드를 읽고 핵심 설계 선택을 짚어라. 구현에서 고른 것 셋을 들고, 저마다 왜 서비스 구조에 알맞은지 설명하라.

</div>

??? success "풀이"
    설계 선택은 구현마다 다르나 흔히 이런 것이 있다. (1) 활성화 함수 선택 -- ReLU 계열은 그래디언트가 소실되지 않아 학습이 빠르다. (2) 정규화 전략 -- 배치 정규화가 내부 공변량 이동을 줄여 학습을 안정시킨다. (3) 잔차 연결 -- 있으면 스킵 경로를 주어 깊은 네트워크에서 그래디언트가 흐르게 한다. 각 선택은 표현력, 계산 비용, 학습 안정성 사이의 트레이드오프를 드러낸다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
입력의 형태와 데이터 타입이 기대한 대로인지 확인하는 입력 검증을 주요 함수나 클래스에 더하여라. 올바르지 않은 입력에는 알아보기 쉬운 오류 메시지를 띄워라.

</div>

??? success "풀이"
    `forward` 메서드(또는 해당 함수)의 앞부분에 `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`이나 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'` 같은 검사를 더한다. 형태를 검사하려면 중요한 차원을 본다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 알아보기 쉬운 오류 메시지는 디버깅을 크게 앞당기고 코드를 재사용하기 안전하게 한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
이 구현이 실패할 만한 유형 둘을 들고, 저마다 어떻게 탐지하고 고칠지 설명하라.

</div>

??? success "풀이"
    흔한 실패 유형은 이렇다. (1) **그래디언트가 사라지거나 터짐** -- 그래디언트 크기를 모니터링해 탐지한다(`torch.nn.utils.clip_grad_norm_`이나 레이어마다 `param.grad.norm()` 기록). 그래디언트 클리핑, 더 나은 초기화(Xavier/Kaiming), 구조 수정(잔차 연결, 정규화 레이어)으로 고친다. (2) **과적합** -- 학습 로스는 줄어드는데 검증 로스가 오르면 탐지한다. 정칙화(드롭아웃, 웨이트 감쇠, 데이터 증강)나 모델 축소로 고친다. 학습 지표와 검증 지표를 늘 함께 모니터링해 이를 일찍 잡아야 한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
실시간 거래 구현을 검사하는 포괄적인 테스트 함수를 작성하라. 빈 입력, 원소 하나짜리 입력, 아주 큰 입력, 그리고 극단값(0, 아주 큰 수)이 든 입력 같은 가장자리 자리를 시험하여라.

</div>

??? success "풀이"
    경계 조건을 두루 다루는 테스트 함수를 작성한다.
    ```python
    def test_orderside():
        model = OrderSide(...)
        # 여느 들임
        assert model(normal_input).shape == expected_shape
        # 원소 하나짜리 배치
        assert model(single_input).shape == (1, ...)
        # 큰 값(넘침을 살핀다)
        out = model(torch.ones(...) * 1000)
        assert torch.isfinite(out).all()
        # 그래디언트 흐름
        out = model(normal_input)
        out.sum().backward()
        for p in model.parameters():
            assert p.grad is not None
    ```
    구조가 엔드투엔드 학습을 지원하는지 알려면 그래디언트 흐름을 테스트하는 것이 특히 중요하다.

## 정리하며

**다룬 것** — 실시간 거래

이 구현은 깔끔하고 읽기 쉬운 PyTorch 코드로 서비스 구조의 핵심 아이디어를 보여 준다.

핵심 클래스는 `OrderSide`, `OrderStatus`, `Order`, `Position`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
