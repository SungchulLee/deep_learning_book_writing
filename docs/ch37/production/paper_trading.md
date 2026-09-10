# 종이 거래

35.7.2장: 종이 거래. 종이 거래로 살펴보는 얼개.

계량 금융에 깊은 배움을 올리려면 든든한 서비스 바탕이 있어야 한다. 이 꾸러미는 지켜보기, 무릅씀 다스리기, 금융 쓰임을 서비스에 올리는 꾀를 아우르는 서비스 얼개 설계 결을 다룬다.

## 1. 코드

```python
"""
35.7.2장: 종이 거래
================================
종이 거래로 살펴보는 얼개.
"""

import numpy as np
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from collections import deque

# ========================================================================
# 메인
# ========================================================================


@dataclass
class PaperTradingConfig:
    initial_capital: float = 1_000_000.0
    transaction_cost: float = 0.001
    slippage_model: str = "gaussian"
    slippage_std: float = 0.0005
    max_position: float = 0.25


class PaperTradingEngine:
    """참에 가깝게 벌이는 모습을 흉내내는 종이 거래 엔진."""

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

        # 미끄러짐
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
    """되짚어 시험한 열매와 종이 거래 열매를 견준다."""

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
    """종이 거래를 보인다."""
    print("=" * 70)
    print("종이 거래 보이기")
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
    print(f"\n--- 종이 거래 간추림 ---")
    for k, v in summary.items():
        if "return" in k or "drawdown" in k:
            print(f"  {k:<20}: {v*100:.2f}%")
        elif "ratio" in k:
            print(f"  {k:<20}: {v:.4f}")
        else:
            print(f"  {k:<20}: {v}")

    # 되짚어 시험한 것과 견주기
    print("\n--- 되짚어 시험 대 종이 거래 견주기 ---")
    bt_returns = np.random.randn(200) * 0.012 + 0.0004
    pt_returns = np.array(engine.return_history)
    comp = BacktestPaperComparator.compare(bt_returns, pt_returns)
    for k, v in comp.items():
        print(f"  {k:<20}: {v:.4f}")


if __name__ == "__main__":
    demo_paper_trading()
```

**출력:**

```
======================================================================
종이 거래 보이기
======================================================================

--- 종이 거래 간추림 ---
  total_return        : -1.01%
  sharpe_ratio        : -0.0817
  max_drawdown        : 9.51%
  avg_turnover        : 0.49560774157267456
  total_costs         : 154022.73065975023
  num_steps           : 199

--- 되짚어 시험 대 종이 거래 견주기 ---
  backtest_sharpe     : -1.1643
  paper_sharpe        : -0.0817
  sharpe_diff         : -1.0826
  backtest_vol        : 0.1795
  paper_vol           : 0.0984
  correlation         : 0.0443
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
종이 거래 짜보기를 살피는 두루 갖춘 시험 함수를 써라. 빈 들임, 원소 하나짜리 들임, 아주 큰 들임, 그리고 끝자락 값(0, 아주 큰 수)이 든 들임 같은 가장자리 자리를 시험하여라.

??? success "연습문제 4 풀이"
    금 언저리 조건을 두루 건드리는 시험 함수를 짓는다.
    ```python
    def test_papertradingconfig():
        model = PaperTradingConfig(...)
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

**다룬 것** — 종이 거래

이 짜보기는 깔끔하고 읽기 쉬운 PyTorch 코드로 서비스 얼개의 고갱이가 되는 생각을 보여 준다.

고갱이 갈래는 `PaperTradingConfig`, `PaperTradingEngine`, `BacktestPaperComparator`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
