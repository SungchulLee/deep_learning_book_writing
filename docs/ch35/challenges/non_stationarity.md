# 흐름 바뀜

금융 저자는 본디부터 흐름이 바뀐다. 저자 판이 옮겨 가면서 돌아옴의 통계 성질이 때에 따라 달라진다. 힘 북돋우는 배움 거래 시스템에는 알아내기와 맞춰 가기 재주가 꼭 있어야 한다. 배운 방침이 지금 저자 자리와 더는 들어맞지 않음을 알아채고 그에 맞추어 고쳐야 하기 때문이다.

## 코드

```python
"""
35.5.1장: 흐름 바뀜
===================================
흐름이 바뀌는 금융 저자를 알아내고 맞춰 가는 재주.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque

# ========================================================================
# 메인
# ========================================================================


@dataclass
class NonStationarityConfig:
    window_size: int = 60
    detection_threshold: float = 2.0
    ema_alpha: float = 0.05
    cusum_threshold: float = 5.0


class CUSUMDetector:
    """쌓인 합으로 바뀜 점을 알아내는 개."""

    def __init__(self, threshold: float = 5.0, drift: float = 0.5):
        self.threshold = threshold
        self.drift = drift
        self.reset()

    def reset(self):
        self.s_pos = 0.0
        self.s_neg = 0.0
        self.mean = 0.0
        self.count = 0

    def update(self, x: float) -> Dict[str, float]:
        self.count += 1
        if self.count < 10:
            self.mean += (x - self.mean) / self.count
            return {"change_detected": False, "s_pos": 0, "s_neg": 0}

        self.s_pos = max(0, self.s_pos + x - self.mean - self.drift)
        self.s_neg = min(0, self.s_neg + x - self.mean + self.drift)

        detected = self.s_pos > self.threshold or abs(self.s_neg) > self.threshold
        if detected:
            self.s_pos = 0
            self.s_neg = 0
            self.mean = x  # 잣대를 되돌린다

        return {
            "change_detected": detected,
            "s_pos": self.s_pos,
            "s_neg": self.s_neg,
        }


class DistributionShiftDetector:
    """굴러가는 창에 KS 비슷한 검정을 매겨 분포 옮겨감을 알아낸다."""

    def __init__(self, reference_window: int = 120, test_window: int = 30,
                 threshold: float = 0.2):
        self.ref_window = reference_window
        self.test_window = test_window
        self.threshold = threshold
        self.buffer = deque(maxlen=reference_window + test_window)

    def update(self, x: float) -> Dict:
        self.buffer.append(x)
        if len(self.buffer) < self.ref_window + self.test_window:
            return {"shift_detected": False, "distance": 0.0}

        data = np.array(self.buffer)
        ref = data[:self.ref_window]
        test = data[self.ref_window:]

        # 바서슈타인 비슷한 쉬운 거리
        ref_sorted = np.sort(ref)
        test_sorted = np.sort(test)
        # 같은 크기로 사이 끼우기
        ref_quantiles = np.quantile(ref_sorted, np.linspace(0, 1, 50))
        test_quantiles = np.quantile(test_sorted, np.linspace(0, 1, 50))
        distance = np.mean(np.abs(ref_quantiles - test_quantiles))

        return {
            "shift_detected": distance > self.threshold,
            "distance": float(distance),
            "ref_mean": float(np.mean(ref)),
            "test_mean": float(np.mean(test)),
        }


class AdaptivePolicy:
    """지수 무게 주기로 흐름 바뀜에 맞춰 가는 방침."""

    def __init__(self, num_assets: int, ema_alpha: float = 0.05):
        self.num_assets = num_assets
        self.ema_alpha = ema_alpha
        self.ema_returns = np.zeros(num_assets)
        self.ema_var = np.ones(num_assets) * 0.01

    def update(self, returns: np.ndarray):
        self.ema_returns = (1 - self.ema_alpha) * self.ema_returns + self.ema_alpha * returns
        self.ema_var = (1 - self.ema_alpha) * self.ema_var + self.ema_alpha * (returns - self.ema_returns) ** 2

    def get_weights(self) -> np.ndarray:
        # 지수로 맞춰 가는 흔들림 거꾸로 무게 주기
        inv_vol = 1.0 / (np.sqrt(self.ema_var) + 1e-8)
        weights = inv_vol / np.sum(inv_vol)
        return weights


def demo_non_stationarity():
    """흐름 바뀜 알아내기와 맞춰 가기를 보인다."""
    print("=" * 70)
    print("Non-Stationarity Detection & Adaptation")
    print("=" * 70)

    np.random.seed(42)
    # t=200에서 판이 바뀌는 자료를 짓는다
    T = 400
    returns = np.concatenate([
        np.random.randn(200) * 0.01 + 0.001,   # 판 1: 낮은 흔들림, 0보다 큼
        np.random.randn(200) * 0.025 - 0.002,   # 판 2: 높은 흔들림, 0보다 작음
    ])

    # CUSUM
    print("\n--- CUSUM Change Detection ---")
    cusum = CUSUMDetector(threshold=3.0)
    changes = []
    for t in range(T):
        result = cusum.update(returns[t])
        if result["change_detected"]:
            changes.append(t)
    print(f"Changes detected at: {changes}")
    print(f"True change at: 200")

    # 분포 옮겨감
    print("\n--- Distribution Shift Detection ---")
    shift_det = DistributionShiftDetector(reference_window=100, test_window=30, threshold=0.005)
    shift_times = []
    for t in range(T):
        result = shift_det.update(returns[t])
        if result["shift_detected"] and (not shift_times or t - shift_times[-1] > 20):
            shift_times.append(t)
            print(f"  Shift at t={t}: distance={result['distance']:.6f}")

    # 맞춰 가는 방침
    print("\n--- Adaptive Policy ---")
    N = 5
    multi_returns = np.random.randn(T, N) * 0.01
    # 자산 서로 얽힘에 판 바뀜이 온다
    multi_returns[200:] = multi_returns[200:] * 2.5  # 흔들림이 곱절

    adaptive = AdaptivePolicy(N, ema_alpha=0.05)
    print(f"{'Step':>5} {'Weights':>50}")
    for t in range(T):
        adaptive.update(multi_returns[t])
        if t in [0, 100, 199, 200, 250, 350]:
            w = adaptive.get_weights()
            print(f"{t:>5}  {np.array2string(w, precision=3)}")


if __name__ == "__main__":
    demo_non_stationarity()
```

## 논의

CUSUM(쌓인 합) 알아내개는 잣대 평균에서 벗어난 만큼을 쌓아 평균이 옮겨 가는지 지켜본다. 쌓인 합이 문턱을 넘으면 바뀜 점이 있다고 알린다. 이 알고리즘은 0보다 큰 쪽과 작은 쪽의 벗어남을 좇는 통계 둘을 지녀 위로 옮겨 감과 아래로 옮겨 감을 모두 알아낸다. 흘러감 매개변수가 예민함을 다스린다. 흘러감이 작으면 더 작은 바뀜도 잡지만 헛된 알림이 는다.

분포 옮겨감 알아내기는 미끄러지는 창을 써서 최근 눈금의 분포를 잣대 창과 견준다. 바서슈타인 거리(또는 비슷한 재기)가 두 겪음 분포의 다름을 잰다. 이 거리가 문턱을 넘으면 판이 바뀌었다고 알린다. 이 길은 평균이 옮겨 감뿐 아니라 흔들림, 치우침, 꼬리 거동의 바뀜도 잡는다.

맞춰 가는 방침은 지수 이동 평균으로 매개변수를 끊임없이 고쳐, 알아낸 흐름 바뀜에 답한다. 자산마다 최근 흔들림의 거꾸로에 견주어 나누는 흔들림 거꾸로 무게 주기는 흔들림이 치솟은 자산에 대한 노출을 절로 줄인다. 지수 이동 평균의 삭임 매개변수가 방침이 옛 자료를 얼마나 빨리 잊을지를 다스린다. 빨리 삭이면 더 빨리 맞춰 가지만 잡음에도 더 예민해진다.

## 연습문제

**연습문제 1.**
굴러가는 통계에 바탕해 저자 자리를 "낮은 흔들림 흐름 타기", "높은 흔들림 흐름 타기", "평균으로 돌아옴"으로 가르는 판 알아내기 알고리즘을 만들어라. 흔들림에는 60일 창을, 흐름 알아내기에는 20일 창을 쓰라.

??? success "연습문제 1 풀이"
    ```python
    def detect_regime(returns, vol_window=60, trend_window=20):
        vol = np.std(returns[-vol_window:])
        trend = np.mean(returns[-trend_window:])
        vol_threshold = np.median(np.std(returns[i:i+vol_window])
                                  for i in range(len(returns)-vol_window))

        if abs(trend) > 2 * np.std(returns[-trend_window:]) / np.sqrt(trend_window):
            if vol > vol_threshold:
                return "high_vol_trending"
            return "low_vol_trending"
        return "mean_reverting"
    ```
    흐름이 뜻있는지는 평균의 표준 잘못에 견주어 살핀다. 높은 흔들림은 지난 흔들림의 가운뎃값에 견주어 정한다. 이 쉬운 가름개로 판에 따라 꾀를 고를 수 있다.

---


**연습문제 2.**
흐름이 바뀌는 금융 자료에서 지난 자료에 지수 무게를 주는 것이 붙박인 굴러가는 창보다 나은 까닭을 풀어라. 삭임 매개변수가 $\alpha$인 지수 이동 평균의 실제 뽑기 크기는 얼마인가?

??? success "연습문제 2 풀이"
    붙박인 굴러가는 창은 창 안의 온 눈금에 같은 무게를, 밖에는 0을 매기므로 옛 자료가 떨어져 나갈 때 끊김이 생긴다. 지수 무게 주기는 매끄럽게 삭이는 무게 $w_t = \alpha(1-\alpha)^t$을 매겨 튐 없이 천천히 넘어간다.

    지수 이동 평균의 실제 뽑기 크기는 대략 $N_{\text{실제}} = 2/\alpha - 1$이다. $\alpha = 0.05$이면 $N_{\text{실제}} \approx 39$개다. 곧 지수 이동 평균이 대략 39일 굴러가는 창처럼 움직이되 넘어감이 더 매끄럽다. $\alpha$이 작으면 더 매끄럽고(실제 창이 넓고) 크면 바뀜에 더 빨리 답한다.

---


**연습문제 3.**
알아낸 저자 판에 따라 밀기 꾀와 평균으로 돌아옴 꾀 사이를 오가는 맞춰 가는 거래 방침을 설계하여라. 오가는 논리를 만들고 빠질 수 있는 함정을 따져라.

??? success "연습문제 3 풀이"
    ```python
    class AdaptiveStrategyPolicy:
        def __init__(self, num_assets, switch_threshold=0.005):
            self.cusum = CUSUMDetector(threshold=3.0)
            self.regime = "trending"
            self.switch_threshold = switch_threshold

        def get_weights(self, returns_history):
            recent = returns_history[-60:]
            autocorr = np.corrcoef(recent[:-1], recent[1:])[0,1]

            if autocorr > self.switch_threshold:
                # 스스로 얽힘이 0보다 큼: 밀기
                signal = np.mean(returns_history[-20:], axis=0)
                w = np.maximum(signal, 0)
            else:
                # 스스로 얽힘이 0보다 작음: 평균으로 돌아옴
                signal = -np.mean(returns_history[-5:], axis=0)
                w = np.maximum(signal, 0)

            return w / (np.sum(w) + 1e-8)
    ```
    종요로운 함정: (1) 판 알아내기가 늦어 이미 판이 바뀐 뒤에야 방침이 옮겨 간다. (2) 꾀를 자주 오가면 거래 비용이 크게 든다. (3) 창이 짧으면 스스로 얽힘 어림에 잡음이 많다. 누그러뜨리는 길로는 오가는 규칙에 되돌이 문턱을 두기, 거래 비용 벌 주기, 오가는 대신 섞는 무리 짓기가 있다.
