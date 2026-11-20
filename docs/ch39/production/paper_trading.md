# 모의 거래

35.7.2장: 모의 거래. 모의 거래로 검증하는 구조.

계량 금융에 딥러닝을 배포하려면 탄탄한 서비스 기반이 있어야 한다. 이 꾸러미는 모니터링, 위험 관리, 금융 응용을 서비스에 배포하는 전략을 아우르는 서비스 구조 설계 패턴을 다룬다.

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
모의 거래 구현을 검사하는 포괄적인 테스트 함수를 작성하라. 빈 입력, 원소 하나짜리 입력, 아주 큰 입력, 그리고 극단값(0, 아주 큰 수)이 든 입력 같은 가장자리 자리를 시험하여라.

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

이 구현은 깔끔하고 읽기 쉬운 PyTorch 코드로 서비스 구조의 핵심 아이디어를 보여 준다.

핵심 클래스는 `PaperTradingConfig`, `PaperTradingEngine`, `BacktestPaperComparator`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
