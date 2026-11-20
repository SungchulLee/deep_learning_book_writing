# 실시간 거래

이 쪽은 실제 운용 수준의 실시간 거래 시스템을 이루는 구성 요소들을 다룬다.

계량 금융에 딥러닝을 배포하려면 탄탄한 운용 기반이 있어야 한다. 이 절은 모니터링, 위험 관리, 금융 응용을 실제 운용 환경에 배포하는 전략을 아우르는 시스템 설계 패턴을 다룬다.

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

이 구현은 NumPy만으로 실시간 거래 시스템의 뼈대를 보여 준다. `FeatureStore`는 종목마다 최근 가격을 버퍼에 담아 1일 수익률, 20일 변동성, 5일·20일 모멘텀 같은 피처를 새 가격이 들어올 때마다 계산한다. `OrderManager`는 목표 비중과 현재 비중의 차이로 주문을 만들고, 스프레드와 슬리피지를 넣어 체결을 시뮬레이션하며 포지션을 갱신한다. `LiveTradingSystem`은 시장 데이터 수신 → 신호 생성 → 주문 실행을 한 스텝으로 묶는다. 모델을 넘기지 않으면 동일 비중(종목마다 0.2)을 목표로 삼는다. 모듈로 나눈 구조 덕에 부분마다 따로 이해하고, 예를 들어 `simulate_fill`을 실제 브로커 API 호출로 바꾸기 쉽다.

출력의 주문 수가 5인 까닭도 여기서 나온다. 첫 스텝에서 비중을 0에서 0.2로 옮기며 종목마다 주문이 하나씩 다섯 개 나가고, 그 뒤로는 목표 비중이 바뀌지 않아 비중 차이가 0.001보다 작으므로 주문이 나가지 않는다. 다만 이 뼈대는 가격이 움직여도 비중 드리프트를 계산하지 않고 `current_weights`를 목표 값으로 덮어쓰며, `portfolio_value`도 갱신하지 않는다. 실제 시스템이라면 체결 결과로 포지션 가치와 비중을 다시 계산해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
코드를 읽고 핵심 설계 선택을 짚어라. 구현에서 고른 것 셋을 들고, 저마다 왜 실시간 거래 시스템에 알맞은지 설명하라.

</div>

??? success "풀이"
    (1) 길이가 정해진 `deque` 버퍼 -- `FeatureStore`는 종목마다 최근 `lookback + 10`개 가격만 지니므로 시스템이 오래 돌아도 메모리가 늘지 않고, 새 가격이 들어올 때마다 피처를 증분으로 다시 계산할 수 있다. (2) 신호 생성과 주문 실행의 분리 -- `generate_signals`는 목표 비중만 내놓고 주문은 `OrderManager`가 만든다. 모델을 바꾸거나 시뮬레이션 체결을 실제 브로커 연결로 바꿀 때 다른 부분을 건드리지 않아도 되고, 각 부분을 따로 시험할 수 있다. (3) 최소 주문 임계값(비중 차이 0.001) -- 아주 작은 비중 차이로 주문을 내면 거래 비용만 치르므로 이를 건너뛴다. 덧붙여 `Enum`으로 주문 상태를 못박아 두면 주문의 생애 주기(대기 → 체결/취소/거부)가 코드에 드러난다.

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
    (1) **데이터 누락이나 지연** -- `get_features`는 피처가 없는 종목에 0을 채워 넣으므로, 데이터 피드가 끊겨도 시스템은 아무 경고 없이 0 피처로 신호를 만든다. 종목마다 마지막 가격의 수신 시각을 기록해 일정 시간 넘게 갱신이 없으면 탐지하고, 그 종목의 거래를 멈추거나 시스템 전체를 안전 모드로 돌려 고친다. (2) **내부 상태와 실제 포지션의 어긋남** -- `execute_trades`는 체결 결과와 상관없이 `current_weights`를 목표 비중으로 덮어쓰고, `portfolio_value`는 한 번도 갱신하지 않는다. 일부만 체결되거나 가격이 움직이면 시스템이 믿는 비중과 실제 비중이 갈라진다. 주기적으로 브로커의 실제 포지션과 대조(reconciliation)해 탐지하고, 비중을 체결된 포지션과 현재 가격에서 다시 계산하도록 고친다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
실시간 거래 구현을 검사하는 포괄적인 테스트 함수를 작성하라. 빈 입력, 원소 하나짜리 입력, 아주 큰 입력, 그리고 극단값(0, 아주 큰 수)이 든 입력 같은 경계 조건을 시험하여라.

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

이 구현은 NumPy만으로 피처 계산, 주문 관리, 신호 생성과 실행을 잇는 실시간 거래 시스템의 뼈대를 보여 준다.

핵심 클래스는 `FeatureStore`, `OrderManager`, `LiveTradingSystem`이고 주문과 포지션은 `Order`, `Position`, `OrderSide`, `OrderStatus`로 나타내며, 앞의 연습문제 4개로 스스로 확인해 볼 수 있다.
