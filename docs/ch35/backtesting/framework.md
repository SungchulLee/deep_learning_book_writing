# 되짚어 시험하기 틀

거래 꾀를 내놓기에 앞서 따져 보려면 되짚어 시험하기 틀이 있어야 한다. 이 틀은 거래 비용, 미끄러짐, 몫 흘러감처럼 참에 가까운 저자 형편을 셈에 넣으면서 지난 자료 위에서 꾀가 도는 모습을 흉내낸다. 이렇게 짜임새 있게 하면 참된 꾀의 알파를 지나치게 맞춘 자국이나 참되지 않은 여김에서 갈라낼 수 있다.

## 코드

```python
"""
35.6.1장: 되짚어 시험하기 틀
=======================================
힘 북돋우는 배움 거래 꾀를 위한 두루 갖춘 되짚어 시험하기 엔진.
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


# 밑금 꾀
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

## 논의

이 틀의 고갱이인 되짚어 시험하기 엔진은 지난 값 자료를 차례대로 다루며, 자리와 몫과 쌓인 돌아옴을 적는 밑천 꾸러미 좇개를 지닌다. 때 걸음마다 엔진이 꾀 함수에 새 겨눔 몫을 묻고, 갈아치움에 비례하는 거래 비용과 아무렇게나 생기는 미끄러짐을 아울러 벌이는 모습을 흉내낸다. 이렇게 참에 가까운 벌임 모형이 종요롭다. 비용 없이 보면 남는 듯한 꾀가 참에 가까운 마찰을 넣으면 밑지는 일이 잦다.

밑천 꾸러미 좇개는 다시 맞추는 사이에 몫이 흘러가는 것을 다룬다. 돌아옴이 실현되고 나면 자산마다 오르내리는 결이 달라 밑천 꾸러미의 몫이 겨눔 값에서 벗어난다. 좇개는 흘러간 몫을 셈하고 값, 돌아옴, 몫, 비용의 온 자취를 적어 뒤에 살필 자료를 넉넉히 남긴다.

밑금 꾀 셋이 이 틀을 잘 드러낸다. 고른 몫 다시 맞추기, 사서 지니기, 밀림이다. 고른 몫은 자산마다 한결같은 나눔을 지키고, 사서 지니기는 처음 나눈 뒤로 다시 맞추지 않으며, 밀림은 요즘 오른 것으로 기운다. 힘 북돋우는 배움으로 익힌 꾀를 이 밑금과 견주는 것이 종요롭다. 배운 방침이 단순한 어림 꾀를 넘어 참된 값어치를 더하는지 가려 주기 때문이다.

## 익힘 문제

**익힘 1.**
되짚어 시험하기 틀 안에 평균-분산 가장 좋게 하기 꾀를 짜 넣어라. 다시 맞추는 걸음마다 60일치 돌아옴을 굴러가는 창으로 삼고 무릅씀 꺼림 값 $\gamma = 1$으로 가장 좋은 몫을 셈하여라.

??? success "익힘 1 풀이"
    평균-분산 꾀는 몫을 $w^* = \frac{1}{\gamma} \Sigma^{-1} \mu$으로 셈한다. $\mu$은 굴러가는 평균 돌아옴 벡터이고 $\Sigma$은 굴러가는 함께 바뀜 행렬이다. 거의 뒤집을 수 없는 자리를 다루려고 $\Sigma$에 $\lambda I$(보기로 $\lambda = 0.01$)을 더해 정칙화한다.

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


**익힘 2.**
잦게 다시 맞추는 꾀에서 미끄러짐을 본뜨는 일이 왜 중요한지 밝혀라. 미끄러짐 모형에 든 갈아치움의 제곱근은 저자의 잔 얼개와 어떻게 이어지는가?

??? success "익힘 2 풀이"
    미끄러짐은 거래를 벌이는 일이 저자 값을 거래하는 이에게 불리하게 움직이기 때문에 생긴다. 제곱근 관계 $\text{미끄러짐} \propto \sigma \sqrt{\text{갈아치움}}$은 저자에 미치는 힘이 거래 크기에 대해 밑선형으로 자란다는 저자 잔 얼개의 겪음에서 온 열매를 드러낸다. 큰 주문일수록 잘게 쪼개어 내보내야 하고 쪼갠 것마다 값을 조금씩 밀기 때문이다. 갈아치움이 큰 잦은 거래 꾀에서는 미끄러짐이 거래 비용을 넘어서서 벌이를 갉아먹을 수 있다. 날마다 200%을 갈아치우는 꾀는 100%을 갈아치우는 꾀에 견주어 거래 낱마다 미끄러짐을 대략 $\sqrt{2} \approx 1.41$곱절 치른다.

---


**익힘 3.**
밑천 꾸러미의 내림폭이 5%을 넘으면 자리 크기를 저절로 반으로 줄이고 10%을 넘으면 모든 자리를 접는, 가장 큰 내림폭 매임을 되짚어 시험하기 엔진에 더하여라.

??? success "익힘 3 풀이"
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
    잃음이 쌓일수록 내놓음을 줄이는 단순한 무릅씀 다루기 덧켜다. 큰 내림폭을 막으면서도 내림폭이 잦아들면 되살아날 길을 열어 둔다.

