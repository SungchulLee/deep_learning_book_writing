# 백테스팅 프레임워크

거래 전략을 실전에 배포하기 전에 평가하려면 백테스팅 프레임워크가 있어야 한다. 이 틀은 거래 비용, 슬리피지, 비중 드리프트(가격이 움직여 포트폴리오 비중이 목표에서 벗어나는 것)처럼 현실적인 시장 상황을 계산에 넣으면서 과거 데이터 위에서 전략이 도는 모습을 시뮬레이션한다. 이렇게 체계적으로 시험하면 진짜 전략 알파를 과적합의 흔적이나 비현실적인 가정에서 갈라낼 수 있다.

## 1. 코드

```python
"""
35.6.1장: 되짚어 시험하기 틀
=======================================
힘 북돋우는 러닝 거래 꾀를 위한 두루 갖춘 되짚어 시험하기 엔진.
"""

import numpy as np
from typing import Dict, Optional, Tuple, List, Callable
from dataclasses import dataclass, field

# ========================================================================
# 메인
# ========================================================================


@dataclass
class BacktestConfig:
    initial_capital: float = 1_000_000.0
    transaction_cost: float = 0.001
    slippage_std: float = 0.0005
    max_leverage: float = 1.0
    risk_free_rate: float = 0.02 / 252


class PortfolioTracker:
    """되짚어 시험하는 동안 밑천 꾸러미의 상태와 자취를 좇는다."""

    def __init__(self, initial_capital: float, num_assets: int):
        self.initial_capital = initial_capital
        self.num_assets = num_assets
        self.cash = initial_capital
        self.positions = np.zeros(num_assets)
        self.weights = np.zeros(num_assets)
        self.portfolio_value = initial_capital
        self.peak_value = initial_capital

        self.value_history: List[float] = [initial_capital]
        self.return_history: List[float] = []
        self.weight_history: List[np.ndarray] = []
        self.turnover_history: List[float] = []
        self.cost_history: List[float] = []

    def update(self, new_weights: np.ndarray, prices: np.ndarray,
               next_prices: np.ndarray, tc_rate: float,
               slippage_std: float = 0.0) -> Dict:
        old_weights = self.weights.copy()
        turnover = np.sum(np.abs(new_weights - old_weights))
        tc = tc_rate * turnover * self.portfolio_value
        slippage = slippage_std * np.sqrt(turnover) * self.portfolio_value * np.abs(np.random.randn())

        returns = (next_prices - prices) / (prices + 1e-8)
        port_return = float(np.dot(new_weights, returns))
        net_return = port_return - (tc + slippage) / (self.portfolio_value + 1e-8)

        self.portfolio_value *= (1 + net_return)
        self.peak_value = max(self.peak_value, self.portfolio_value)

        # 몫이 흘러간다
        drifted = new_weights * (1 + returns)
        self.weights = drifted / (np.sum(drifted) + 1e-8) if np.sum(drifted) > 0 else np.zeros(self.num_assets)

        self.value_history.append(self.portfolio_value)
        self.return_history.append(net_return)
        self.weight_history.append(new_weights.copy())
        self.turnover_history.append(turnover)
        self.cost_history.append(tc + slippage)

        return {
            "portfolio_value": self.portfolio_value,
            "return": net_return,
            "turnover": turnover,
            "cost": tc + slippage,
            "drawdown": (self.peak_value - self.portfolio_value) / (self.peak_value + 1e-8),
        }


class BacktestEngine:
    """으뜸 되짚어 시험하기 엔진."""

    def __init__(self, prices: np.ndarray, config: BacktestConfig):
        self.prices = prices
        self.config = config
        self.num_steps = len(prices) - 1
        self.num_assets = prices.shape[1]

    def run(self, strategy: Callable) -> Dict:
        """
        꾀 함수를 받아 되짚어 시험한다.

        Args:
            strategy: function(prices_history, current_weights, step) -> new_weights
        """
        tracker = PortfolioTracker(self.config.initial_capital, self.num_assets)

        for t in range(self.num_steps):
            prices_so_far = self.prices[:t + 1]
            new_weights = strategy(prices_so_far, tracker.weights, t)

            # 몫이 올바른지 살핀다
            if np.any(np.isnan(new_weights)):
                new_weights = tracker.weights.copy()

            tracker.update(
                new_weights, self.prices[t], self.prices[t + 1],
                self.config.transaction_cost, self.config.slippage_std,
            )

        return {
            "value_history": np.array(tracker.value_history),
            "return_history": np.array(tracker.return_history),
            "weight_history": np.array(tracker.weight_history),
            "turnover_history": np.array(tracker.turnover_history),
            "cost_history": np.array(tracker.cost_history),
            "final_value": tracker.portfolio_value,
            "total_return": (tracker.portfolio_value / self.config.initial_capital - 1),
        }


# 베이스라인 꾀
def equal_weight_strategy(prices, weights, step):
    N = prices.shape[1]
    return np.ones(N) / N

def buy_and_hold_strategy(prices, weights, step):
    if step == 0:
        return np.ones(prices.shape[1]) / prices.shape[1]
    return weights  # 다시 맞추지 않는다


def demo_backtesting():
    """되짚어 시험하기 틀을 보인다."""
    print("=" * 70)
    print("되짚어 시험하기 틀 보이기")
    print("=" * 70)

    np.random.seed(42)
    N, T = 5, 500
    returns = np.random.randn(T, N) * 0.015 + 0.0003
    prices = 100 * np.exp(np.cumsum(returns, axis=0))

    config = BacktestConfig(transaction_cost=0.001, slippage_std=0.0003)
    engine = BacktestEngine(prices, config)

    # 고른 몫
    result_ew = engine.run(equal_weight_strategy)
    print(f"\n고른 몫: 돌아옴={result_ew['total_return']*100:.2f}%, "
          f"고른 갈아치움={np.mean(result_ew['turnover_history']):.4f}")

    # 사서 지니기
    result_bh = engine.run(buy_and_hold_strategy)
    print(f"사서 지니기: 돌아옴={result_bh['total_return']*100:.2f}%, "
          f"고른 갈아치움={np.mean(result_bh['turnover_history']):.4f}")

    # 밀림 꾀
    def momentum_strat(prices_hist, weights, step):
        if len(prices_hist) < 21:
            return np.ones(N) / N
        ret = (prices_hist[-1] / prices_hist[-21]) - 1
        w = np.maximum(ret, 0)
        return w / (np.sum(w) + 1e-8) if np.sum(w) > 0 else np.ones(N) / N

    result_mom = engine.run(momentum_strat)
    print(f"밀림: 돌아옴={result_mom['total_return']*100:.2f}%, "
          f"고른 갈아치움={np.mean(result_mom['turnover_history']):.4f}")

    total_costs = np.sum(result_mom['cost_history'])
    print(f"\n밀림 꾀의 온 거래 비용: ${total_costs:,.0f}")


if __name__ == "__main__":
    demo_backtesting()
```

**출력:**

```
======================================================================
되짚어 시험하기 틀 보이기
======================================================================

고른 몫: 돌아옴=52.31%, 고른 갈아치움=0.0125
사서 지니기: 돌아옴=53.31%, 고른 갈아치움=0.0020
밀림: 돌아옴=27.26%, 고른 갈아치움=0.2947

밀림 꾀의 온 거래 비용: $223,067
```

## 2. 논의

이 틀의 핵심인 백테스팅 엔진은 과거 가격 데이터를 차례대로 다루며, 포지션과 비중과 누적 수익률을 기록하는 포트폴리오 추적기를 지닌다. 타임스텝마다 엔진이 전략 함수에 새 목표 비중을 묻고, 회전율에 비례하는 거래 비용과 무작위로 생기는 슬리피지를 포함해 체결 과정을 시뮬레이션한다. 이렇게 현실적인 체결 모델이 중요하다. 비용 없이 보면 수익이 나는 듯한 전략이 현실적인 마찰을 넣으면 손실을 내는 일이 잦다.

포트폴리오 추적기는 리밸런싱 사이에 비중이 드리프트하는 것을 다룬다. 수익률이 실현되고 나면 자산마다 오르내린 폭이 달라 포트폴리오 비중이 목표 값에서 벗어난다. 추적기는 드리프트한 비중을 계산하고 가치, 수익률, 비중, 비용의 전체 이력을 기록해 나중에 분석할 데이터를 넉넉히 남긴다.

베이스라인 전략 셋이 이 틀을 잘 드러낸다. 동일 비중 리밸런싱, 매수 후 보유, 모멘텀이다. 동일 비중은 모든 자산에 같은 비중을 유지하고, 매수 후 보유는 처음 배분한 뒤로 리밸런싱하지 않으며, 모멘텀은 최근 오른 자산에 비중을 싣는다. 강화학습으로 트레인된 전략을 이 베이스라인과 비교하는 것이 중요하다. 트레인한 폴리시가 단순한 휴리스틱 전략을 넘어 실제 가치를 더하는지 가려 주기 때문이다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
백테스팅 프레임워크 안에 평균-분산 최적화 전략을 짜 넣어라. 리밸런싱하는 스텝마다 60일치 수익률을 롤링 윈도로 삼고 위험 회피 계수 $\gamma = 1$로 최적 비중을 계산하여라.

</div>

??? success "풀이"
    평균-분산 전략은 비중을 $w^* = \frac{1}{\gamma} \Sigma^{-1} \mu$로 계산한다. $\mu$는 롤링 평균 수익률 벡터이고 $\Sigma$는 롤링 공분산 행렬이다. 공분산 행렬이 특이 행렬에 가까워 역행렬이 불안정한 경우를 다루려고 $\Sigma$에 $\lambda I$(예를 들어 $\lambda = 0.01$)를 더해 정칙화한다. 아래 코드는 음수 비중을 0으로 잘라 합이 1이 되게 다시 맞추므로, 공매도를 허용하지 않는 근사다.

    ```python
    def mean_variance_strategy(prices, weights, step, window=60, gamma=1.0):
        if len(prices) < window + 1:
            return np.ones(prices.shape[1]) / prices.shape[1]
        rets = np.diff(np.log(prices[-window-1:]), axis=0)
        mu = np.mean(rets, axis=0)
        cov = np.cov(rets.T) + 0.01 * np.eye(rets.shape[1])
        w = np.linalg.solve(gamma * cov, mu)
        w = np.maximum(w, 0)
        return w / (np.sum(w) + 1e-8) if np.sum(w) > 0 else np.ones(len(mu)) / len(mu)
    ```

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
자주 리밸런싱하는 전략에서 슬리피지를 모델링하는 일이 왜 중요한지 밝혀라. 슬리피지 모델에 든 회전율의 제곱근은 시장 미시구조와 어떻게 이어지는가?

</div>

??? success "풀이"
    슬리피지는 거래 체결이 시장 가격을 거래자에게 불리하게 움직이기 때문에 생긴다. 제곱근 관계 $\text{미끄러짐} \propto \sigma \sqrt{\text{갈아치움}}$(슬리피지 ∝ σ√회전율)는 시장 충격이 거래 크기에 대해 준선형으로 자란다는 시장 미시구조의 경험적 결과를 반영한다. 큰 주문일수록 잘게 쪼개어 내보내야 하고 쪼갠 주문마다 가격을 조금씩 밀기 때문이다. 회전율이 큰 고빈도 전략에서는 슬리피지가 거래 비용을 넘어서서 수익을 갉아먹을 수 있다. 하루 회전율이 200%인 전략은 100%인 전략보다 스텝마다 치르는 슬리피지가 대략 $\sqrt{2} \approx 1.41$배다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
포트폴리오의 낙폭이 5%를 넘으면 포지션 크기를 자동으로 반으로 줄이고 10%를 넘으면 모든 포지션을 청산하는, 최대 낙폭 제약을 백테스팅 엔진에 더하여라.

</div>

??? success "풀이"
    ```python
    class DrawdownAwareEngine(BacktestEngine):
        def run(self, strategy):
            tracker = PortfolioTracker(self.config.initial_capital, self.num_assets)
            for t in range(self.num_steps):
                prices_so_far = self.prices[:t + 1]
                new_weights = strategy(prices_so_far, tracker.weights, t)

                # 내림폭 다스림을 건다
                dd = (tracker.peak_value - tracker.portfolio_value) / (tracker.peak_value + 1e-8)
                if dd > 0.10:
                    new_weights = np.zeros(self.num_assets)  # 자리를 접는다
                elif dd > 0.05:
                    new_weights = new_weights * 0.5  # 반으로 줄인다

                tracker.update(new_weights, self.prices[t], self.prices[t + 1],
                             self.config.transaction_cost, self.config.slippage_std)
            return tracker
    ```
    손실이 쌓일수록 익스포저를 줄이는 단순한 위험 관리 레이어다. 5~10% 구간에서는 비중을 반만 줄이므로 낙폭이 잦아들면 원래 크기로 돌아온다. 그러나 10%를 넘어 전부 청산하면 포트폴리오 가치가 더는 움직이지 않아 낙폭도 줄지 않으므로, 이 코드로는 다시 진입하지 못한다. 실전에서는 일정 기간 뒤 재진입하는 규칙을 따로 둔다.

## 정리하며

**다룬 것** — 백테스팅 프레임워크

이 틀의 핵심인 백테스팅 엔진은 과거 가격 데이터를 차례대로 다루며, 포지션과 비중과 누적 수익률을 기록하는 포트폴리오 추적기를 지닌다.

핵심 클래스는 `BacktestConfig`, `PortfolioTracker`, `BacktestEngine`이고, 연습문제 3의 `DrawdownAwareEngine`이 여기에 낙폭 제약을 더한다. 앞의 연습문제 세 개로 스스로 점검해 볼 수 있다.
