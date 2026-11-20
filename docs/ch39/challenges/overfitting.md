# 과적합 방지

과적합은 계량 거래에서 가장 무서운 함정이다. 과거 데이터에서는 눈부신 성과를 내지만 실제 시장에서는 무너지는 전략이 그것이다. 워크포워드 검증, 디플레이티드 샤프 비율(deflated Sharpe ratio, DSR), 백테스트 과적합 확률(probability of backtest overfitting, PBO)은 실제 자본이 걸리기 전에 이런 실패를 탐지하고 막는 엄밀한 도구다.

## 1. 코드

```python
"""
35.5.4장: 지나치게 맞추기 막기
========================================
금융 힘 북돋우는 러닝에서 지나치게 맞추기를 막는 앞으로 걸어가며
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
    """되짚어 시험이 지나치게 맞을 확률(PBO)를 어림한다."""

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

**출력:**

```
======================================================================
Overfitting Prevention Demonstration
======================================================================

--- Walk-Forward Validation ---
Splits: 11
Mean IS Sharpe:  0.7068
Mean OOS Sharpe: 2.0064
Overfit ratio:   0.3523

--- Deflated Sharpe Ratio ---
  Trials=   1: DSR stat=-1088982120.645, p=1.0000, significant=False
  Trials=  10: DSR stat=11.406, p=0.0000, significant=True
  Trials=  50: DSR stat=12.911, p=0.0000, significant=True
  Trials= 100: DSR stat=13.264, p=0.0000, significant=True
  Trials= 500: DSR stat=13.811, p=0.0000, significant=True

--- Probability of Backtest Overfitting ---
PBO: 1.0
Best IS Sharpe: 0.7644
Its OOS Sharpe: -0.0552
```

## 2. 논의

시계열을 위한 워크포워드 검증은 모든 학습 데이터가 시험 데이터보다 앞서게 하여 표본 외 성능을 정직하게 평가한다. 과적합 비율, 곧 표본 내 샤프 비율을 표본 외 샤프 비율로 나눈 값은 전략이 신호가 아니라 잡음을 얼마나 파고들었는지를 바로 잰다. 비율이 1.0에 가까우면 일반화가 잘 된 것이고, 2~3을 넘으면 심하게 과적합된 것이다. 위 출력의 비율은 0.3523으로 오히려 표본 외가 더 좋은데, 이 보기의 `momentum_strategy`가 신호의 부호를 그 구간 자신의 누적 수익률에서 정하기 때문이다. 표본 내와 표본 외 모두 미래 정보를 쓰므로 두 샤프 비율은 각 구간 샤프 비율의 절댓값이 되고, 63일짜리 짧은 시험 구간에서는 그 절댓값이 크게 나오기 쉽다.

디플레이티드 샤프 비율(DSR)은 개발 과정에서 전략을 몇 개나 시도했는지를 계산에 넣어 관측한 샤프 비율을 보정한다. 연구자가 전략 100개를 시험하면 진짜 우위가 없어도 그 가운데 가장 좋은 것은 우연히 돈을 버는 것처럼 보인다. DSR은 귀무가설 아래에서 기대되는 최대 샤프 비율을 계산하고, 관측한 샤프 비율이 그것을 유의하게 넘는지 검정한다. 전략 500개를 시도했다면 한 해 동안의 샤프 비율 1.5는 대개 통계적으로 유의하지 않다. 다만 위 코드의 `e_max` 식은 Bailey와 López de Prado의 식에 있는 정규 분위수 $\Phi^{-1}$ 대신 로그를 쓴 근사라 값이 지나치게 작게 나온다(500개에서 약 0.23). 그래서 출력에서는 500개에서도 유의하다고 나오며, 시도가 1개일 때는 $\log 1 = 0$으로 나누어 통계량이 터진다.

백테스트 과적합 확률(PBO)은 표본 내에서 가장 좋았던 전략이 표본 외에서 중앙값에 못 미칠 확률을 추정한다. 조합 대칭 교차 검증(combinatorially symmetric cross-validation)으로 학습-시험 분할을 많이 만들고, 표본 내 성능이 가장 좋은 전략이 표본 외에서도 꾸준히 좋은지 살핀다. PBO가 0.5면 표본 내 최적화가 무작위로 고르는 것과 다름없고, 1.0에 가까우면 그보다도 나쁘다. 위 코드는 분할 하나만 보므로 PBO가 0 아니면 1로만 나오며, 출력의 1.0은 그 한 번에 표본 내 최고 전략이 표본 외 중앙값에 못 미쳤다는 뜻이다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
연구자가 전략 50개를 시험해 252 거래일 동안 샤프 비율이 2.0인 것을 골랐다. 디플레이티드 샤프 비율로 이 결과가 통계적으로 유의한지 가려라.

</div>

??? success "풀이"
    서로 독립인 전략 50개를 시도했을 때 귀무가설(실력 없음) 아래에서 기대되는 최대 샤프 비율은 대략 다음과 같다.

    $E[\max SR] \approx \sigma_{SR} \left(\frac{1 - 0.5772}{\log(50)} + \frac{0.5772}{\sqrt{2\log(50)}}\right)$

    $\sigma_{SR} \approx 1$이고 $\log(50) \approx 3.91$이면 $E[\max SR] \approx 0.108 + 0.206 \approx 0.99$이다.

    252일 동안의 샤프 비율의 표준 오차는 $SE \approx \sqrt{(1 + 0.5 \cdot 2^2)/252} \approx 0.109$이다.

    DSR 통계는 $(2.0 - 0.99) / 0.109 \approx 9.27$으로 매우 유의하다(p < 0.001). 다만 50개 전략이 서로 상관되어 있으면(비슷한 전략이라면) 실질적인 시도 횟수가 더 적으므로 결과가 한층 더 유의해진다.

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
표본 내 과적합과 표본 외 성능 저하의 차이를 설명하여라. 저하가 작으면서도 과적합된 전략이 있을 수 있는가?

</div>

??? success "풀이"
    표본 내 과적합은 전략이 학습 데이터에만 있는 잡음 패턴을 잡을 때 생긴다. 표본 외 성능 저하는 학습 구간에서 시험 구간으로 옮길 때 눈에 보이는 성능 하락이다. 저하가 크면 과적합된 것이지만, 저하가 작다고 강건성이 보장되지는 않는다.

    저하가 작으면서도 과적합될 수 있는 경우는 이렇다. (1) 시험 구간이 마침 학습 구간과 비슷하다(운 좋은 분할). (2) 워크포워드 검증의 결과를 보고 여러 후보 가운데 그 전략을 골랐다. 선택 바이어스가 한 겹 더 들어온 것이다. (3) 시험 구간이 너무 짧아 실력과 운을 가릴 수 없다. 그래서 워크포워드 방식으로 여러 번 분할하고 통계적 유의성을 검정해야 한다. 학습-시험을 한 번만 나누는 것으로는 모자란다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
과거 수익률을 무작위로 여러 번 학습-시험으로 나누어, 표본 내 최고 전략이 표본 외 중앙값에 못 미칠 확률을 알려 주는 간단한 PBO 추정량을 만들어라.

</div>

??? success "풀이"
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
    PBO가 0.5를 넘으면 표본 내 최고 전략을 고르는 일이 표본 외에서는 무작위로 고르는 것보다 나을 것이 없다는 뜻이며, 과적합을 강하게 가리킨다. 덧붙여 이 풀이는 기본값으로 분할을 16번 하고, 시계열을 무작위로 섞어 나누므로 시간 순서와 이동 평균 신호의 의미가 깨진다는 한계가 있다.

## 정리하며

**다룬 것** — 과적합 방지

시계열을 위한 워크포워드 검증은 모든 학습 데이터가 시험 데이터보다 앞서게 하여 표본 외 성능을 정직하게 평가한다.

핵심 클래스는 `OverfittingConfig`, `WalkForwardValidator`, `DeflatedSharpeRatio`, `ProbabilityOfOverfitting`이며 앞의 연습문제 3개로 스스로 확인해 볼 수 있다.
