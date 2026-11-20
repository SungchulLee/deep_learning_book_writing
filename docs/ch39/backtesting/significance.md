# 통계적 유의성

통계적 유의성 검정은 거래 전략의 성능이 진짜인지 그저 우연이 낳은 것인지 가린다. 부트스트랩 검정, 순열 검정, 그리고 다중 검정 보정이 실력을 운에서 갈라내는 데 도움을 준다. 데이터 마이닝 바이어스와 과적합이라는 위험이 널려 있는 계량 금융에서 이 도구가 더욱 중요하다.

## 1. 코드

```python
"""
35.6.4장: 통계로 뜻있음
==========================================
부트스트랩, 뒤섞기 검정, 여러 번 검정 바로잡기.
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

# ========================================================================
# 메인
# ========================================================================


class BootstrapTest:
    """꾀의 됨됨이에 대한 부트스트랩 가설 검정."""

    def __init__(self, num_bootstrap: int = 10000, block_size: int = 20):
        self.num_bootstrap = num_bootstrap
        self.block_size = block_size

    def test_sharpe(self, returns: np.ndarray, null_mean: float = 0.0) -> Dict:
        T = len(returns)
        observed_sharpe = (np.mean(returns) - null_mean) / (np.std(returns) + 1e-8) * np.sqrt(252)

        # 덩이 부트스트랩
        centered = returns - np.mean(returns) + null_mean
        n_blocks = T // self.block_size + 1

        boot_sharpes = []
        for _ in range(self.num_bootstrap):
            indices = np.random.randint(0, T - self.block_size, n_blocks)
            boot_sample = np.concatenate([centered[i:i+self.block_size] for i in indices])[:T]
            sr = np.mean(boot_sample) / (np.std(boot_sample) + 1e-8) * np.sqrt(252)
            boot_sharpes.append(sr)

        boot_sharpes = np.array(boot_sharpes)
        p_value = np.mean(boot_sharpes >= observed_sharpe)

        # 믿음 구간
        ci_lower = np.percentile(boot_sharpes, 2.5)
        ci_upper = np.percentile(boot_sharpes, 97.5)

        return {
            "observed_sharpe": float(observed_sharpe),
            "p_value": float(p_value),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "significant_5pct": p_value < 0.05,
        }


class PermutationTest:
    """때 맞추는 솜씨에 대한 뒤섞기 검정."""

    def __init__(self, num_permutations: int = 10000):
        self.num_permutations = num_permutations

    def test_timing(self, positions: np.ndarray, returns: np.ndarray) -> Dict:
        observed = np.sum(positions * returns)

        perm_results = []
        for _ in range(self.num_permutations):
            shuffled_pos = np.random.permutation(positions)
            perm_results.append(np.sum(shuffled_pos * returns))
        perm_results = np.array(perm_results)

        p_value = np.mean(perm_results >= observed)

        return {
            "observed_pnl": float(observed),
            "mean_perm_pnl": float(np.mean(perm_results)),
            "p_value": float(p_value),
            "significant_5pct": p_value < 0.05,
        }


class MultipleTestingCorrection:
    """여러 번 하는 가설 검정 바로잡기."""

    @staticmethod
    def bonferroni(p_values: np.ndarray, alpha: float = 0.05) -> Dict:
        adjusted = np.minimum(p_values * len(p_values), 1.0)
        return {
            "adjusted_p_values": adjusted.tolist(),
            "significant": (adjusted < alpha).tolist(),
            "n_significant": int(np.sum(adjusted < alpha)),
        }

    @staticmethod
    def holm_bonferroni(p_values: np.ndarray, alpha: float = 0.05) -> Dict:
        n = len(p_values)
        sorted_idx = np.argsort(p_values)
        sorted_p = p_values[sorted_idx]

        adjusted = np.zeros(n)
        for i in range(n):
            adjusted[sorted_idx[i]] = min(sorted_p[i] * (n - i), 1.0)

        # 한 방향으로만 오르게 한다
        for i in range(1, n):
            adjusted[sorted_idx[i]] = max(adjusted[sorted_idx[i]], adjusted[sorted_idx[i-1]])

        return {
            "adjusted_p_values": adjusted.tolist(),
            "significant": (adjusted < alpha).tolist(),
            "n_significant": int(np.sum(adjusted < alpha)),
        }


class MinimumBacktestLength:
    """뜻있음에 있어야 할 가장 짧은 되짚어 시험 길이를 셈한다."""

    @staticmethod
    def compute(target_sharpe: float, confidence: float = 0.95) -> Dict:
        from scipy.stats import norm
        z = norm.ppf(confidence)
        T_min_days = (z / target_sharpe) ** 2
        return {
            "min_days": int(np.ceil(T_min_days)),
            "min_years": float(T_min_days / 252),
            "target_sharpe": target_sharpe,
            "confidence": confidence,
        }


def demo_significance():
    """통계로 뜻있음을 따지는 검정을 보인다."""
    print("=" * 70)
    print("통계로 뜻있음 검정")
    print("=" * 70)

    np.random.seed(42)
    T = 504
    returns = np.random.randn(T) * 0.015 + 0.0004

    # 부트스트랩 검정
    print("\n--- 부트스트랩 샤프 검정 ---")
    boot = BootstrapTest(num_bootstrap=5000)
    result = boot.test_sharpe(returns)
    print(f"본 샤프: {result['observed_sharpe']:.4f}")
    print(f"p 값: {result['p_value']:.4f}")
    print(f"95% 믿음 구간: [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
    print(f"뜻있음: {result['significant_5pct']}")

    # 뒤섞기 검정
    print("\n--- 뒤섞기 검정(때 맞추는 솜씨) ---")
    positions = np.sign(np.random.randn(T))
    perm = PermutationTest(num_permutations=5000)
    result = perm.test_timing(positions, returns)
    print(f"본 손익: {result['observed_pnl']:.4f}")
    print(f"고른 뒤섞기 손익: {result['mean_perm_pnl']:.4f}")
    print(f"p 값: {result['p_value']:.4f}")

    # 여러 번 검정
    print("\n--- 여러 번 검정 바로잡기 ---")
    p_values = np.array([0.001, 0.01, 0.03, 0.05, 0.10, 0.20, 0.50])
    bonf = MultipleTestingCorrection.bonferroni(p_values)
    holm = MultipleTestingCorrection.holm_bonferroni(p_values)
    print(f"{'날 p':>8} {'본페로니':>12} {'홀름':>12}")
    print("-" * 34)
    for i in range(len(p_values)):
        print(f"{p_values[i]:>8.3f} {bonf['adjusted_p_values'][i]:>11.3f} "
              f"{holm['adjusted_p_values'][i]:>11.3f}")

    # 가장 짧은 되짚어 시험 길이
    print("\n--- 가장 짧은 되짚어 시험 길이 ---")
    try:
        for sr in [0.25, 0.5, 1.0, 1.5, 2.0]:
            mbl = MinimumBacktestLength.compute(sr)
            print(f"  SR={sr:.2f}: {mbl['min_years']:.1f}해 ({mbl['min_days']}일)")
    except ImportError:
        print("  (scipy이 있어야 한다)")


if __name__ == "__main__":
    demo_significance()
```

**출력:**

```
======================================================================
통계로 뜻있음 검정
======================================================================

--- 부트스트랩 샤프 검정 ---
본 샤프: 0.6042
p 값: 0.1512
95% 믿음 구간: [-1.4394, 1.2132]
뜻있음: False

--- 뒤섞기 검정(때 맞추는 솜씨) ---
본 손익: 0.1292
고른 뒤섞기 손익: 0.0021
p 값: 0.3452

--- 여러 번 검정 바로잡기 ---
     날 p         본페로니           홀름
----------------------------------
   0.001       0.007       0.007
   0.010       0.070       0.060
   0.030       0.210       0.150
   0.050       0.350       0.200
   0.100       0.700       0.300
   0.200       1.000       0.400
   0.500       1.000       0.500

--- 가장 짧은 되짚어 시험 길이 ---
  SR=0.25: 0.2해 (44일)
  SR=0.50: 0.0해 (11일)
  SR=1.00: 0.0해 (3일)
  SR=1.50: 0.0해 (2일)
  SR=2.00: 0.0해 (1일)
```

## 2. 논의

샤프 비율에 대한 부트스트랩 검정은 블록 부트스트랩을 써서 금융 수익률의 자기상관 구조를 지킨다. 귀무가설(평균이 0) 아래에서 수익률 블록을 다시 뽑아, 우연히 나올 만한 샤프 비율의 분포를 만든다. 그러고 나서 관측한 샤프 비율을 이 귀무 분포와 비교하여 p 값을 얻는다. 블록 크기는 흔히 거래일 20~30일로 잡아 월 단위 상관 구조를 담는다. 위 출력에서 관측 샤프 0.6042의 p 값은 0.1512로, 2년치 데이터로는 5% 수준에서 유의하지 않다.

순열 검정은 마켓 타이밍 능력을 평가하는 비모수적 방법이다. 포지션과 수익률의 짝을 무작위로 뒤섞으면 진짜 타이밍 신호는 부서지면서도 각각의 분포는 그대로 남는다. 뒤섞은 손익이 관측 손익 이상인 비율이 곧 타이밍 능력이 없다는 귀무가설에 대한 p 값이다. 분포를 가정하지 않으므로 이 검정은 특히 강건하다.

여러 전략 변형을 평가할 때는 다중 검정 보정이 꼭 있어야 한다. 본페로니 보정은 유의 수준을 검정 횟수로 나누어 족별 오류율(FWER, 거짓 발견이 하나라도 나올 확률)을 통제한다. 홀름-본페로니 방법은 덜 보수적인 단계적 방법을 준다. 최소 백테스트 길이 공식은 목표 샤프 비율과 필요한 데이터 길이를 이어 준다. 연습문제 1의 식대로 연율 샤프 0.5를 일간 값 $0.5/\sqrt{252}$로 바꾸어 넣으면, 95% 신뢰로 이를 확인하려면 일간 데이터가 약 2,728 거래일, 곧 11년쯤 있어야 한다. 금융에서 통계적 유의성을 확립하는 일이 얼마나 어려운지 잘 드러낸다. 다만 위 코드의 `MinimumBacktestLength`는 연율 샤프를 일간 값으로 바꾸지 않고 그대로 넣으므로, 출력 표의 기간(SR=0.50에 11일)은 252분의 1로 줄어든 값이다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
어떤 전략이 일간 데이터 2년에서 샤프 비율 1.2를 기록했다. 최소 백테스트 길이 공식으로 이 결과가 95% 신뢰 수준에서 통계적으로 유의한지 가려라.

</div>

??? success "풀이"
    신뢰 수준 $c$에서 유의하려면 필요한 최소 백테스트 길이가 거래일로 $T_{\min} = (z / SR)^2$이고 $z = \Phi^{-1}(c)$이다. 여기서 $SR$은 일간 샤프 비율이다.

    95% 신뢰 수준이면 $z = 1.645$이다. 일간 $SR = 1.2/\sqrt{252} \approx 0.0756$이므로

    $T_{\min} = (1.645 / 0.0756)^2 \approx 473$ 거래일 $\approx 1.9$년이다.

    우리에게는 2년 $\approx 504$ 거래일이 있어 473을 넘는다. 95% 신뢰 수준에서 가까스로 유의하다. 다만 이 공식은 수익률이 독립이고 같은 분포를 따른다(i.i.d.)고 가정한 것이라, 자기상관이나 변동성 군집이 있으면 더 긴 백테스트가 있어야 한다.

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
서로 상관된 전략 변형들을 평가할 때 본페로니 보정이 왜 지나치게 보수적인지 밝혀라. 어떤 다른 방법이 더 알맞은가?

</div>

??? success "풀이"
    본페로니 보정은 최악의 경우를 대비해 $\alpha$를 전체 검정 횟수 $m$으로 나눈다. 전략들이 서로 상관되어 있으면(예를 들어 룩백 길이만 다른 모멘텀 전략들) 실질적으로 독립인 검정의 수가 $m$보다 훨씬 적으므로 본페로니가 지나치게 보수적이 된다.

    홀름-본페로니 방법은 p 값이 작은 차례대로 가설을 기각하며 스텝마다 임계값을 고치는 개선된 방법이다. 서로 상관된 검정이 많을 때는 거짓 발견률(FDR)을 통제하는 방법(벤야미니-호크버그)이 더 알맞다. 거짓 발견이 하나라도 날 확률이 아니라 발견 가운데 거짓 발견이 차지하는 비율의 기댓값을 통제하므로, 검정이 많을 때 검정력이 훨씬 크다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
모멘텀 신호가 다음 기간 수익률을 예측하는 힘이 통계적으로 유의한지 가리는 순열 검정을 짜라.

</div>

??? success "풀이"
    ```python
    def permutation_test_momentum(returns, lookback=20, n_perms=10000):
        T = len(returns)
        # 밀림 신호를 셈한다
        signal = np.array([np.mean(returns[max(0,t-lookback):t])
                          for t in range(T)])
        positions = np.sign(signal)

        # 본 손익
        observed_pnl = np.sum(positions[:-1] * returns[1:])

        # 뒤섞기 분포
        perm_pnls = []
        for _ in range(n_perms):
            shuffled_returns = np.random.permutation(returns[1:])
            perm_pnls.append(np.sum(positions[:-1] * shuffled_returns))

        p_value = np.mean(np.array(perm_pnls) >= observed_pnl)
        return {'observed_pnl': observed_pnl, 'p_value': p_value,
                'significant': p_value < 0.05}
    ```
    이 검정은 신호와 미래 수익률의 시간 대응을 부수면서도 각각의 분포는 그대로 두므로, 신호에 진짜 예측력이 있는지를 깔끔하게 평가한다.

## 정리하며

**다룬 것** — 통계적 유의성

샤프 비율에 대한 부트스트랩 검정은 블록 부트스트랩을 써서 금융 수익률의 자기상관 구조를 지킨다.

핵심 클래스는 `BootstrapTest`, `PermutationTest`, `MultipleTestingCorrection`, `MinimumBacktestLength`이며 앞의 연습문제 3개로 스스로 확인해 볼 수 있다.
