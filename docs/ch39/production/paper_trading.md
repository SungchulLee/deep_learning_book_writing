# 모의 거래

이 쪽은 실제 돈을 걸기 전에 모의 거래(paper trading)로 전략을 검증하는 시스템을 다룬다. 모의 거래는 실시간 시장 데이터로 주문과 체결을 흉내 내되 실제 주문은 내지 않는 단계다.

계량 금융에 딥러닝을 배포하려면 탄탄한 운용 기반이 있어야 한다. 이 절은 모니터링, 위험 관리, 금융 응용을 실제 운용 환경에 배포하는 전략을 아우르는 시스템 설계 패턴을 다룬다.

## 1. 코드

```python
"""
35.7.2장: 종이 거래
================================
종이 거래로 살펴보는 구조.
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

이 구현은 NumPy만으로 모의 거래 엔진의 뼈대를 보여 준다. `PaperTradingEngine`은 목표 비중을 종목당 한도(`max_position` = 0.25)로 자른 뒤, 회전율에 비례하는 거래 비용과 회전율의 제곱근에 비례하는 무작위 슬리피지를 수익률에서 빼고, 스텝마다 거래 기록을 남긴다. `BacktestPaperComparator`는 백테스트 수익률과 모의 거래 수익률의 샤프 비율, 변동성, 상관을 견주어 둘이 어긋나는지 본다. 실전에서는 같은 전략의 두 결과 사이에 샤프 비율 차이가 크거나 상관이 낮으면, 백테스트가 체결 비용을 과소평가했거나 미래 정보를 썼다는 신호로 읽는다.

출력을 읽을 때는 데모의 한계를 알아 둔다. 데모 전략은 스텝마다 디리클레 분포에서 비중을 새로 뽑으므로 평균 회전율이 약 0.50으로 아주 높고, 그만큼 비용이 쌓인다. `total_costs`는 스텝마다의 비용 비율을 더해 초기 자본을 곱한 근사값이다. 또 비교에 쓴 백테스트 수익률은 이 전략과 상관없이 따로 뽑은 난수이므로 상관이 0.0443으로 거의 0이고, 샤프 비율 차이 -1.0826도 아무 뜻이 없다. 비교기의 사용법을 보여 줄 뿐이다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
코드를 읽고 핵심 설계 선택을 짚어라. 구현에서 고른 것 셋을 들고, 저마다 왜 모의 거래 시스템에 알맞은지 설명하라.

</div>

??? success "풀이"
    (1) 종목당 비중 한도(`np.clip`) -- 전략이 내놓는 비중을 실행 직전에 자르므로, 모델이 이상한 출력을 내도 한 종목에 자본이 쏠리지 않는다. (2) 회전율 기반 비용과 슬리피지 모델 -- 백테스트와 같은 비용 가정을 모의 거래에도 넣어야 두 결과를 공정하게 견줄 수 있다. (3) 스텝마다의 거래 기록(`trade_log`) -- 회전율, 비용, 슬리피지를 따로 남겨 두면 모의 거래 성과가 백테스트와 어긋날 때 어느 항목에서 차이가 났는지 추적할 수 있다. `BacktestPaperComparator`를 따로 둔 것도 같은 까닭이다. 모의 거래 단계의 목적이 백테스트 가정을 검증하는 데 있기 때문이다.

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
    (1) **비중 합이 1에서 벗어남** -- 비중을 종목마다 0.25로 자르면 합이 1보다 작아지고 남은 몫은 아무 수익도 내지 않는 현금처럼 남는다. 반대로 음수 비중을 허용하면 합이 1을 넘는 레버리지가 생길 수 있다. 스텝마다 비중의 합과 절댓값의 합을 기록해 탐지하고, 자른 뒤 다시 정규화하거나 현금 비중을 명시적으로 두어 고친다. (2) **최대 낙폭 계산 오류** -- `get_performance_summary`는 최대 낙폭을 (최고점 − 이력 전체의 최저점)/최고점으로 계산하므로, 최저점이 최고점보다 먼저 나온 경우에도 그 차이를 낙폭으로 센다. 손으로 만든 짧은 가치 이력(예를 들어 먼저 떨어졌다가 크게 오르는 경우)으로 단위 테스트를 돌려 탐지하고, 각 시점의 누적 최고점에서 낙폭을 계산해 그 최댓값을 취하도록 고친다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
모의 거래 구현을 검사하는 포괄적인 테스트 함수를 작성하라. 빈 입력, 원소 하나짜리 입력, 아주 큰 입력, 그리고 극단값(0, 아주 큰 수)이 든 입력 같은 경계 조건을 시험하여라.

</div>

??? success "풀이"
    경계 조건을 두루 다루는 테스트 함수를 작성한다.
    ```python
    def test_papertradingconfig():
        model = PaperTradingConfig(...)
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

**다룬 것** — 모의 거래

이 구현은 NumPy만으로 비용과 슬리피지를 넣은 모의 거래 엔진과 백테스트 비교기의 뼈대를 보여 준다.

핵심 클래스는 `PaperTradingConfig`, `PaperTradingEngine`, `BacktestPaperComparator`이며 앞의 연습문제 4개로 스스로 확인해 볼 수 있다.
