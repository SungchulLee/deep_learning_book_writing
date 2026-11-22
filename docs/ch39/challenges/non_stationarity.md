# 비정상성

금융 시장은 본질적으로 비정상적(non-stationary)이다. 곧 시장 레짐이 바뀌면서 수익률의 통계적 성질이 시간에 따라 달라진다. 그래서 강화학습 거래 시스템에는 변화 탐지와 적응 기법이 꼭 있어야 한다. 트레인한 폴리시가 지금 시장 상태와 더는 들어맞지 않음을 알아채고 그에 맞추어 고쳐야 하기 때문이다.

## 1. 코드

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
    """지수 무게 주기로 흐름 바뀜에 맞춰 가는 폴리시."""

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
    """흐름 바뀜 탐지와 맞춰 가기를 보인다."""
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

    # 맞춰 가는 폴리시
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

**출력:**

```
======================================================================
Non-Stationarity Detection & Adaptation
======================================================================

--- CUSUM Change Detection ---
Changes detected at: []
True change at: 200

--- Distribution Shift Detection ---
  Shift at t=211: distance=0.005675
  Shift at t=232: distance=0.013139
  Shift at t=253: distance=0.010335
  Shift at t=274: distance=0.013261
  Shift at t=295: distance=0.005204
  Shift at t=316: distance=0.011279
  Shift at t=337: distance=0.007573
  Shift at t=358: distance=0.008869
  Shift at t=379: distance=0.008012

--- Adaptive Policy ---
 Step                                            Weights
    0  [0.2 0.2 0.2 0.2 0.2]
  100  [0.189 0.213 0.193 0.204 0.201]
  199  [0.211 0.169 0.196 0.209 0.214]
  200  [0.205 0.183 0.188 0.214 0.21 ]
  250  [0.219 0.175 0.215 0.211 0.18 ]
  350  [0.196 0.192 0.217 0.206 0.189]
```

## 2. 논의

CUSUM(누적합) 탐지기는 기준 평균에서 벗어난 양을 누적해 평균이 이동하는지 지켜본다. 누적합이 임계값을 넘으면 변화점이 있다고 알린다. 이 알고리즘은 양의 방향과 음의 방향 이탈을 따로 누적하는 통계량 둘을 지녀 위쪽 이동과 아래쪽 이동을 모두 탐지한다. 드리프트 파라미터가 민감도를 조절한다. 드리프트가 작으면 더 작은 변화도 잡지만 거짓 경보가 는다. 다만 위 출력에서 CUSUM은 변화점을 하나도 찾지 못했다(`[]`). 드리프트 0.5와 임계값 3.0이 하루 수익률(크기 0.01~0.025)과 단위가 맞지 않아 누적합이 임계값에 닿을 수 없기 때문이다. 실전에서는 수익률을 표준화하거나 드리프트와 임계값을 표준편차 단위로 잡아야 한다.

분포 이동 탐지는 슬라이딩 윈도를 써서 최근 관측값의 분포를 기준 윈도와 견준다. 바서슈타인 거리(또는 비슷한 거리)가 두 경험적 분포의 차이를 잰다. 이 거리가 임계값을 넘으면 레짐이 바뀌었다고 알린다. 이 방법은 평균의 이동뿐 아니라 변동성, 비대칭, 꼬리 거동의 변화도 잡는다. 위 출력에서는 실제 변화점(t=200) 뒤 11스텝 만인 t=211에 처음 탐지했고, 그 뒤로는 20스텝 간격 제한 때문에 21스텝마다 다시 알렸다.

적응형 폴리시는 지수 이동 평균으로 파라미터를 끊임없이 갱신해 비정상성에 대응한다. 자산마다 최근 변동성의 역수에 비례해 비중을 나누는 역변동성 가중은 변동성이 치솟은 자산에 대한 노출을 자동으로 줄인다. 지수 이동 평균의 감쇠 파라미터가 폴리시가 옛 데이터를 얼마나 빨리 잊을지를 정한다. 빨리 감쇠하면 더 빨리 적응하지만 잡음에도 더 예민해진다. 데모에서는 t=200 이후 다섯 자산의 변동성이 모두 똑같이 2.5배가 되므로 상대 변동성이 그대로라, 비중이 0.2 언저리에서 크게 움직이지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
롤링 통계에 바탕해 시장 상태를 "낮은 변동성 추세", "높은 변동성 추세", "평균 회귀"로 가르는 레짐 탐지 알고리즘을 만들어라. 변동성에는 60일 윈도를, 추세 탐지에는 20일 윈도를 쓰라.

</div>

??? success "풀이"
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
    추세가 유의한지는 평균을 그 표준 오차와 견주어 판단한다. 높은 변동성은 과거 롤링 변동성의 중앙값과 견주어 정한다. 이 단순한 분류기로 레짐에 따라 전략을 고를 수 있다.

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
비정상적인 금융 데이터에서 과거 데이터에 지수 가중치를 주는 것이 고정된 롤링 윈도보다 나은 까닭을 설명하여라. 감쇠 파라미터가 $\alpha$인 지수 이동 평균의 유효 표본 크기는 얼마인가?

</div>

??? success "풀이"
    고정된 롤링 윈도는 윈도 안의 모든 관측값에 같은 가중치를, 밖에는 0을 매기므로 옛 데이터가 빠져나갈 때 추정값이 불연속적으로 튄다. 지수 가중은 매끄럽게 감쇠하는 가중치 $w_t = \alpha(1-\alpha)^t$를 매겨 튐 없이 서서히 넘어간다.

    지수 이동 평균의 유효 표본 크기는 대략 $N_{\text{실제}} = 2/\alpha - 1$이다. $\alpha = 0.05$이면 $N_{\text{실제}} \approx 39$개다. 곧 지수 이동 평균이 대략 39일 롤링 윈도처럼 움직이되 넘어감이 더 매끄럽다. $\alpha$가 작으면 더 매끄럽고(유효 윈도가 넓고) 크면 변화에 더 빨리 반응한다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
탐지한 시장 레짐에 따라 모멘텀 전략과 평균 회귀 전략 사이를 오가는 적응형 거래 폴리시를 설계하여라. 전환 논리를 만들고 빠질 수 있는 함정을 논하여라.

</div>

??? success "풀이"
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
    중요한 함정: (1) 레짐 탐지가 늦어 이미 레짐이 바뀐 뒤에야 폴리시가 옮겨 간다. (2) 전략을 자주 오가면 거래 비용이 크게 든다. (3) 창이 짧으면 자기상관 추정에 잡음이 많다. 완화하는 방법으로는 전환 규칙에 히스테리시스 임계값(들어갈 때와 나올 때의 임계값을 다르게 두기)을 두기, 거래 비용 벌점 주기, 전략을 딱 잘라 오가는 대신 레짐 확률로 섞기(블렌딩)가 있다. 덧붙여 위 코드는 `self.cusum`과 `self.regime`을 만들어 놓고 쓰지 않으며, 자기상관만으로 전환한다.

## 정리하며

**다룬 것** — 비정상성

CUSUM(누적합) 탐지기는 기준 평균에서 벗어난 양을 누적해 평균이 이동하는지 지켜본다.

핵심 클래스는 `NonStationarityConfig`, `CUSUMDetector`, `DistributionShiftDetector`, `AdaptivePolicy`이며 앞의 연습문제 3개로 스스로 확인해 볼 수 있다.
