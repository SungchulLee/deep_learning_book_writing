# 통계로 뜻있음

통계로 뜻있음을 따지는 일은 거래 꾀의 됨됨이가 참된 것인지 그저 우연이 낳은 것인지 가린다. 부트스트랩 검정, 뒤섞기 검정, 그리고 여러 번 검정에 대한 바로잡기가 솜씨를 운에서 갈라내는 데 도움을 준다. 자료를 캐다 생기는 치우침과 지나치게 맞추기라는 무릅씀이 널려 있는 계량 금융에서 이 연장이 더욱 종요롭다.

## 코드

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

## 논의

샤프 비에 대한 부트스트랩 검정은 덩이 부트스트랩을 써서 금융 돌아옴의 앞뒤 얽힘 얼개를 지킨다. 영 가설(평균이 0) 아래에서 돌아옴 덩이를 다시 뽑아, 우연히 나올 만한 샤프 비의 분포를 세운다. 그러고 나서 본 샤프 비를 이 영 분포에 견주어 p 값을 얻는다. 덩이 크기는 흔히 거래일 20~30일로 잡아 달 단위 얽힘 결을 담는다.

뒤섞기 검정은 때 맞추는 솜씨를 따지는 매개변수 없는 길이다. 자리와 돌아옴의 짝을 아무렇게나 뒤섞으면 참된 때 맞춤 신호가 부서지면서도 저마다의 분포는 그대로 남는다. 뒤섞은 손익이 본 손익을 넘는 비율이 곧 때 맞추는 힘이 없다는 영 가설에 대한 p 값이다. 분포를 여기지 않으므로 이 검정이 특히 힘이 세다.

여러 꾀 갈래를 따질 때는 여러 번 검정 바로잡기가 꼭 있어야 한다. 본페로니 바로잡기는 뜻있음 문턱을 검정 횟수로 나누어 온 집안 어긋남률을 다스린다. 홀름-본페로니 방법은 덜 빡빡한 차례대로의 길을 준다. 가장 짧은 되짚어 시험 길이 꼴은 겨눈 샤프 비와 있어야 할 자료를 이어 준다. 95% 믿음으로 샤프 0.5을 세우려면 날마다의 자료가 64해쯤 있어야 하는데, 금융에서 통계로 뜻있음을 세우는 일이 얼마나 어려운지 잘 드러낸다.

## 익힘 문제

**익힘 1.**
어떤 꾀가 날마다의 자료 2해에서 샤프 비 1.2을 이루었다. 가장 짧은 되짚어 시험 길이 꼴로 이 열매가 95% 믿음에서 통계로 뜻있는지 가려라.

??? success "익힘 1 풀이"
    믿음 $c$에서 뜻있으려면 가장 짧은 되짚어 시험 길이가 거래일로 $T_{\min} = (z / SR)^2$이고 $z = \Phi^{-1}(c)$이다.

    95% 믿음이면 $z = 1.645$이다. 날마다의 $SR = 1.2/\sqrt{252} \approx 0.0756$이므로

    $T_{\min} = (1.645 / 0.0756)^2 \approx 473$ 거래일 $\approx 1.9$해다.

    우리에게는 2해 $\approx 504$ 거래일이 있어 473을 넘는다. 95% 믿음에서 가까스로 뜻있다. 다만 이 꼴은 돌아옴이 서로 매이지 않고 같은 분포를 따른다고 여긴 것이라, 앞뒤 얽힘이 있거나 흔들림 없지 않으면 더 긴 되짚어 시험이 있어야 한다.

---


**익힘 2.**
서로 얽힌 꾀 갈래를 따질 때 본페로니 바로잡기가 왜 지나치게 빡빡한지 밝혀라. 어떤 다른 길이 더 알맞은가?

??? success "익힘 2 풀이"
    본페로니 바로잡기는 검정이 모두 서로 남남이라고 여겨 $\alpha$을 온 검정 횟수 $m$으로 나눈다. 꾀들이 서로 얽혀 있으면(보기로 되짚어 보는 길이만 다른 밀림 꾀들) 참으로 남남인 검정의 수가 $m$보다 훨씬 적으므로 본페로니가 지나치게 빡빡해진다.

    홀름-본페로니 방법은 뜻있음 차례대로 가설을 물리치며 걸음마다 문턱을 고치는 나아진 길이다. 서로 얽힌 검정에는 거짓 찾음률(FDR)을 다스리는 길(베냐미니-호흐베르크)이 더 알맞다. 거짓 찾음이 하나라도 날 낌새가 아니라 거짓 찾음이 차지할 몫의 기댓값을 다스리므로, 얽힌 검정이 많을 때 통계의 힘이 훨씬 세다.

---


**익힘 3.**
밀림 신호가 다음 날 돌아옴을 미리 알리는 힘이 통계로 뜻있는지 가리는 뒤섞기 검정을 짜라.

??? success "익힘 3 풀이"
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
    이 검정은 신호와 앞날 돌아옴의 때 맞춤을 부수면서도 저마다의 분포는 그대로 두므로, 신호에 참으로 미리 알리는 힘이 있는지를 깔끔하게 따진다.

