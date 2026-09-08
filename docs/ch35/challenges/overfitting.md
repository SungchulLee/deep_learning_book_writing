# 지나치게 맞추기 막기

지나치게 맞추기는 계량 거래에서 가장 무서운 함정이다. 지난 자료에서는 눈부시게 돌지만 살아 있는 저자에서는 어그러지는 꾀가 그것이다. 앞으로 걸어가며 살피기, 바람 뺀 샤프 비, 되짚어 시험이 지나치게 맞을 낌새는 참 밑천이 걸리기 앞서 이 어그러짐을 알아내고 막는 엄밀한 도구를 준다.

## 1. 코드

```python
"""
35.5.4장: 지나치게 맞추기 막기
========================================
금융 힘 북돋우는 배움에서 지나치게 맞추기를 막는 앞으로 걸어가며
살피기, 정칙화, 통계 검정.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# ========================================================================
# 메인
# ========================================================================


@dataclass
class OverfittingConfig:
    train_window: int = 252
    test_window: int = 63
    gap: int = 5
    n_splits: int = 5
    significance_level: float = 0.05


class WalkForwardValidator:
    """때 열을 위한 앞으로 걸어가며 엇갈려 살피기."""

    def __init__(self, config: OverfittingConfig):
        self.config = config

    def generate_splits(self, T: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        splits = []
        train_w = self.config.train_window
        test_w = self.config.test_window
        gap = self.config.gap

        start = 0
        while start + train_w + gap + test_w <= T:
            train_idx = np.arange(start, start + train_w)
            test_idx = np.arange(start + train_w + gap, start + train_w + gap + test_w)
            splits.append((train_idx, test_idx))
            start += test_w
        return splits

    def evaluate(self, returns: np.ndarray, strategy_returns_func) -> Dict:
        T = len(returns)
        splits = self.generate_splits(T)

        is_sharpes = []
        oos_sharpes = []

        for train_idx, test_idx in splits:
            train_r = strategy_returns_func(returns, train_idx)
            test_r = strategy_returns_func(returns, test_idx)

            is_sharpe = np.mean(train_r) / (np.std(train_r) + 1e-8) * np.sqrt(252)
            oos_sharpe = np.mean(test_r) / (np.std(test_r) + 1e-8) * np.sqrt(252)

            is_sharpes.append(is_sharpe)
            oos_sharpes.append(oos_sharpe)

        return {
            "is_sharpes": is_sharpes,
            "oos_sharpes": oos_sharpes,
            "mean_is": float(np.mean(is_sharpes)),
            "mean_oos": float(np.mean(oos_sharpes)),
            "overfit_ratio": float(np.mean(is_sharpes)) / (float(np.mean(oos_sharpes)) + 1e-8),
            "num_splits": len(splits),
        }


class DeflatedSharpeRatio:
    """
    바람 뺀 샤프 비(베일리와 로페스 데 프라도, 2014).
    여러 번 검정한 것을 셈에 넣어 맞춘다.
    """

    @staticmethod
    def compute(observed_sharpe: float, num_trials: int,
                sharpe_std: float = 1.0, skewness: float = 0.0,
                kurtosis: float = 3.0, T: int = 252) -> Dict:
        from scipy.stats import norm

        # 영가설 아래 어림 최대 샤프
        e_max = sharpe_std * ((1 - 0.5772) / (np.log(num_trials) + 1e-8) +
                              0.5772 / (np.sqrt(2 * np.log(num_trials)) + 1e-8))

        # 샤프 비의 표준 잘못
        se = np.sqrt((1 + 0.5 * observed_sharpe**2 -
                       skewness * observed_sharpe +
                       (kurtosis - 3) / 4 * observed_sharpe**2) / T)

        # 바람 뺀 검정 통계
        dsr_stat = (observed_sharpe - e_max) / (se + 1e-8)
        p_value = 1 - norm.cdf(dsr_stat)

        return {
            "dsr_statistic": float(dsr_stat),
            "p_value": float(p_value),
            "expected_max_sharpe": float(e_max),
            "significant": p_value < 0.05,
        }


class ProbabilityOfOverfitting:
    """되짚어 시험이 지나치게 맞을 낌새(PBO)를 어림한다."""

    @staticmethod
    def compute(is_returns: List[np.ndarray], oos_returns: List[np.ndarray]) -> Dict:
        n = len(is_returns)
        is_sharpes = [np.mean(r) / (np.std(r) + 1e-8) for r in is_returns]
        oos_sharpes = [np.mean(r) / (np.std(r) + 1e-8) for r in oos_returns]

        best_is_idx = np.argmax(is_sharpes)
        best_is_oos = oos_sharpes[best_is_idx]

        # PBO: 뽑기 안 으뜸이 뽑기 밖 가운뎃값에 못 미치는 몫
        median_oos = np.median(oos_sharpes)
        pbo = float(best_is_oos < median_oos)

        return {
            "pbo": pbo,
            "best_is_sharpe": float(is_sharpes[best_is_idx]),
            "best_is_oos_sharpe": float(best_is_oos),
            "median_oos_sharpe": float(median_oos),
        }


def demo_overfitting():
    """지나치게 맞추기를 알아내고 막는 것을 보인다."""
    print("=" * 70)
    print("Overfitting Prevention Demonstration")
    print("=" * 70)

    np.random.seed(42)
    T = 1000
    returns = np.random.randn(T) * 0.015 + 0.0002

    # 앞으로 걸어가며 살피기
    print("\n--- Walk-Forward Validation ---")
    config = OverfittingConfig(train_window=252, test_window=63, gap=5)
    validator = WalkForwardValidator(config)

    def momentum_strategy(returns, idx):
        r = returns[idx]
        signal = np.sign(np.cumsum(r)[-1] if len(r) > 0 else 0)
        return r * signal

    result = validator.evaluate(returns, momentum_strategy)
    print(f"Splits: {result['num_splits']}")
    print(f"Mean IS Sharpe:  {result['mean_is']:.4f}")
    print(f"Mean OOS Sharpe: {result['mean_oos']:.4f}")
    print(f"Overfit ratio:   {result['overfit_ratio']:.4f}")

    # 바람 뺀 샤프 비
    print("\n--- Deflated Sharpe Ratio ---")
    for n_trials in [1, 10, 50, 100, 500]:
        try:
            dsr = DeflatedSharpeRatio.compute(
                observed_sharpe=1.5, num_trials=n_trials, T=252)
            print(f"  Trials={n_trials:>4}: DSR stat={dsr['dsr_statistic']:.3f}, "
                  f"p={dsr['p_value']:.4f}, significant={dsr['significant']}")
        except ImportError:
            print(f"  (scipy required for DSR computation)")
            break

    # PBO
    print("\n--- Probability of Backtest Overfitting ---")
    is_rets = [np.random.randn(63) * 0.015 + 0.001 * (i + 1) for i in range(10)]
    oos_rets = [np.random.randn(63) * 0.015 + 0.0001 for _ in range(10)]
    pbo = ProbabilityOfOverfitting.compute(is_rets, oos_rets)
    print(f"PBO: {pbo['pbo']}")
    print(f"Best IS Sharpe: {pbo['best_is_sharpe']:.4f}")
    print(f"Its OOS Sharpe: {pbo['best_is_oos_sharpe']:.4f}")


if __name__ == "__main__":
    demo_overfitting()
```

## 2. 논의

때 열을 위한 앞으로 걸어가며 살피기는 온 익힘 자료가 시험 자료보다 앞서게 하여 뽑기 밖 됨됨이를 정직하게 따진다. 지나치게 맞춘 비, 곧 뽑기 안 샤프와 뽑기 밖 샤프의 비는 꾀가 신호가 아니라 잡음을 얼마나 파고들었는지를 곧바로 잰다. 비가 1.0에 가까우면 두루 잘 맞음을 뜻하고, 2~3을 넘으면 심하게 지나치게 맞춘 것이다.

바람 뺀 샤프 비(DSR)는 만드는 동안 꾀를 몇 벌이나 꾀해 보았는지를 셈에 넣어 본 샤프 비를 맞춘다. 살피는 이가 꾀 100벌을 시험하면 참된 앞섬이 없어도 그 가운데 가장 좋은 것이 우연히 돈을 벌어 보인다. DSR은 영가설 아래 어림 최대 샤프를 셈하고 본 샤프가 그것을 뜻있게 넘는지 검정한다. 500벌을 꾀했다면 한 해에 걸친 샤프 1.5는 흔히 통계로 뜻있지 않다.

되짚어 시험이 지나치게 맞을 낌새(PBO)는 뽑기 안에서 가장 좋았던 꾀가 뽑기 밖에서 못 미칠 낌새를 어림한다. 어우름 엇갈려 살피기로 익힘-시험 쪼갬을 많이 지어내고, 뽑기 안 됨됨이를 가장 좋게 하는 꾀가 뽑기 밖에서도 한결같이 잘 도는지 살핀다. PBO가 1.0에 가까우면 뽑기 안 가장 좋게 하기가 사실상 아무렇게나 고르는 것과 다름없다는 뜻이다.

## 연습문제

**연습문제 1.**
살피는 이가 꾀 50벌을 시험해 252 거래일에 걸쳐 샤프 비 2.0인 것을 골랐다. 바람 뺀 샤프 비 틀로 이 열매가 통계로 뜻있는지 따져라.

??? success "연습문제 1 풀이"
    서로 매이지 않은 50벌을 꾀했을 때 영가설(솜씨 없음) 아래 어림 최대 샤프는 대략 다음과 같다.

    $E[\max SR] \approx \sigma_{SR} \left(\frac{1 - 0.5772}{\log(50)} + \frac{0.5772}{\sqrt{2\log(50)}}\right)$

    $\sigma_{SR} \approx 1$이고 $\log(50) \approx 3.91$이면 $E[\max SR] \approx 0.108 + 0.206 \approx 0.99$이다.

    252일에 걸친 샤프 비의 표준 잘못은 $SE \approx \sqrt{(1 + 0.5 \cdot 2^2)/252} \approx 0.109$이다.

    DSR 통계는 $(2.0 - 0.99) / 0.109 \approx 9.27$으로 매우 뜻있다(p < 0.001). 다만 50벌이 서로 얽혀 있으면(비슷한 꾀라면) 실제 꾀함 횟수가 더 적으므로 열매가 한층 더 뜻있어진다.

---


**연습문제 2.**
뽑기 안에서 지나치게 맞추기와 뽑기 밖에서 나빠짐의 다름을 풀어라. 나빠짐이 적으면서도 지나치게 맞은 꾀가 있을 수 있는가?

??? success "연습문제 2 풀이"
    뽑기 안 지나치게 맞추기는 꾀가 익힘 자료에만 있는 잡음 결을 잡을 때 생긴다. 뽑기 밖 나빠짐은 익힘에서 시험으로 옮길 때 눈에 보이는 됨됨이 떨어짐이다. 나빠짐이 크면 지나치게 맞춘 것이지만, 나빠짐이 적다고 굳셈이 보장되지는 않는다.

    나빠짐이 적으면서도 지나치게 맞을 수 있는 때는 이렇다. (1) 시험 구간이 마침 익힘 구간과 비슷하다(운 좋은 쪼갬). (2) 앞으로 걸어가며 얻은 열매를 보고 여러 벌 가운데 그 꾀를 골랐다. 고르기 치우침이 한 켜 더 들어온 것이다. (3) 시험 구간이 너무 짧아 솜씨와 운을 가릴 수 없다. 그래서 앞으로 걸어가며 여러 번 쪼개고 통계로 뜻있음을 검정해야 한다. 익힘-시험을 한 번만 쪼개는 것으로는 모자란다.

---


**연습문제 3.**
지난 돌아옴을 아무렇게나 10번 익힘-시험으로 쪼개어, 뽑기 안 으뜸 꾀가 뽑기 밖 가운뎃값에 못 미칠 낌새를 알려 주는 쉬운 PBO 어림개를 만들어라.

??? success "연습문제 3 풀이"
    ```python
    def estimate_pbo(returns, n_strategies=10, n_splits=16):
        T = len(returns)
        half = T // 2
        underperform_count = 0

        for _ in range(n_splits):
            # 아무렇게나 두 쪽으로 가른다
            idx = np.random.permutation(T)
            is_idx = idx[:half]
            oos_idx = idx[half:]

            # 꾀마다 돌아옴을 짓는다(보기로 되돌아보는 길이를 달리한다)
            is_sharpes = []
            oos_sharpes = []
            for lookback in np.linspace(5, 60, n_strategies).astype(int):
                signal_is = np.sign(np.convolve(returns[is_idx], np.ones(lookback)/lookback, 'same'))
                signal_oos = np.sign(np.convolve(returns[oos_idx], np.ones(lookback)/lookback, 'same'))
                is_sharpes.append(np.mean(signal_is * returns[is_idx]))
                oos_sharpes.append(np.mean(signal_oos * returns[oos_idx]))

            best_is = np.argmax(is_sharpes)
            if oos_sharpes[best_is] < np.median(oos_sharpes):
                underperform_count += 1

        return underperform_count / n_splits
    ```
    PBO가 0.5을 넘으면 뽑기 안 으뜸 꾀를 고르는 일이 뽑기 밖에서는 아무렇게나 고르는 것보다 나을 것이 없다는 뜻이며, 지나치게 맞추었음을 강하게 가리킨다.

## 정리하며

**다룬 것** — 지나치게 맞추기 막기

때 열을 위한 앞으로 걸어가며 살피기는 온 익힘 자료가 시험 자료보다 앞서게 하여 뽑기 밖 됨됨이를 정직하게 따진다.

고갱이 갈래는 `OverfittingConfig`, `WalkForwardValidator`, `DeflatedSharpeRatio`, `ProbabilityOfOverfitting`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
