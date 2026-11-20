# 워크포워드 검증

워크포워드 검증은 시계열 전략을 평가하는 표준 방법이다. 일반 교차 검증과 달리 과거 데이터로 학습하고 미래 데이터로 시험하되 윈도를 굴리거나 넓혀 가므로 데이터의 시간 순서를 지킨다. 이렇게 하면 표본 외 성능을 현실에 가깝게 추정할 수 있고, 시장 레짐이 바뀔 때 전략이 얼마나 잘 버티는지도 드러난다.

## 1. 코드

```python
"""
35.6.2장: 앞으로 걸어가며 살피기
========================================
힘 북돋우는 러닝 꾀를 따지기 위한 앞으로 걸어가며 살피기.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass

# ========================================================================
# 메인
# ========================================================================


@dataclass
class WalkForwardConfig:
    train_window: int = 252
    test_window: int = 63
    gap: int = 5
    expanding: bool = False
    min_train: int = 126


class WalkForwardAnalyzer:
    """앞으로 걸어가며 살피기 엔진."""

    def __init__(self, config: WalkForwardConfig):
        self.config = config

    def generate_splits(self, T: int) -> List[Dict]:
        splits = []
        cfg = self.config

        if cfg.expanding:
            test_start = cfg.min_train + cfg.gap
            while test_start + cfg.test_window <= T:
                splits.append({
                    "train_start": 0,
                    "train_end": test_start - cfg.gap,
                    "test_start": test_start,
                    "test_end": min(test_start + cfg.test_window, T),
                })
                test_start += cfg.test_window
        else:
            start = 0
            while start + cfg.train_window + cfg.gap + cfg.test_window <= T:
                splits.append({
                    "train_start": start,
                    "train_end": start + cfg.train_window,
                    "test_start": start + cfg.train_window + cfg.gap,
                    "test_end": start + cfg.train_window + cfg.gap + cfg.test_window,
                })
                start += cfg.test_window

        return splits

    def run(self, returns: np.ndarray,
            train_fn: Callable, eval_fn: Callable) -> Dict:
        """
        앞으로 걸어가며 살피기를 돌린다.

        Args:
            returns: (T, N)이나 (T,) 형태 돌아옴 열
            train_fn: function(train_returns) -> 모형/몫
            eval_fn: function(model, test_returns) -> 꾀의 돌아옴
        """
        T = len(returns)
        splits = self.generate_splits(T)
        results = []

        for i, split in enumerate(splits):
            train_r = returns[split["train_start"]:split["train_end"]]
            test_r = returns[split["test_start"]:split["test_end"]]

            model = train_fn(train_r)
            strat_returns = eval_fn(model, test_r)

            is_sharpe = np.mean(train_r if train_r.ndim == 1 else train_r.mean(1)) / (
                np.std(train_r if train_r.ndim == 1 else train_r.mean(1)) + 1e-8) * np.sqrt(252)
            oos_sharpe = np.mean(strat_returns) / (np.std(strat_returns) + 1e-8) * np.sqrt(252)

            results.append({
                "split": i,
                "train_size": split["train_end"] - split["train_start"],
                "test_size": split["test_end"] - split["test_start"],
                "is_sharpe": float(is_sharpe),
                "oos_sharpe": float(oos_sharpe),
                "oos_return": float(np.sum(strat_returns)),
                "oos_returns": strat_returns,
            })

        # 한데 모으기
        oos_sharpes = [r["oos_sharpe"] for r in results]
        is_sharpes = [r["is_sharpe"] for r in results]
        all_oos = np.concatenate([r["oos_returns"] for r in results])

        return {
            "splits": results,
            "mean_oos_sharpe": float(np.mean(oos_sharpes)),
            "std_oos_sharpe": float(np.std(oos_sharpes)),
            "mean_is_sharpe": float(np.mean(is_sharpes)),
            "degradation": float(np.mean(is_sharpes) - np.mean(oos_sharpes)),
            "aggregate_sharpe": float(np.mean(all_oos) / (np.std(all_oos) + 1e-8) * np.sqrt(252)),
            "num_splits": len(results),
        }


def demo_walk_forward():
    """앞으로 걸어가며 살피기를 보인다."""
    print("=" * 70)
    print("앞으로 걸어가며 살피기 보이기")
    print("=" * 70)

    np.random.seed(42)
    T = 1000
    returns = np.random.randn(T) * 0.015 + 0.0002

    # 단순한 밀림 꾀
    def train_fn(train_r):
        return {"signal": np.sign(np.mean(train_r))}

    def eval_fn(model, test_r):
        return test_r * model["signal"]

    for expanding in [False, True]:
        name = "넓혀 가는 창" if expanding else "굴러가는 창"
        config = WalkForwardConfig(
            train_window=252, test_window=63, gap=5, expanding=expanding
        )
        analyzer = WalkForwardAnalyzer(config)
        result = analyzer.run(returns, train_fn, eval_fn)

        print(f"\n--- {name}으로 앞으로 걸어가기 ---")
        print(f"쪼갬: {result['num_splits']}")
        print(f"고른 익힌 안 샤프:  {result['mean_is_sharpe']:.4f}")
        print(f"고른 익힌 밖 샤프: {result['mean_oos_sharpe']:.4f}")
        print(f"떨어짐:     {result['degradation']:.4f}")
        print(f"한데 모은 익힌 밖:   {result['aggregate_sharpe']:.4f}")


if __name__ == "__main__":
    demo_walk_forward()
```

**출력:**

```
======================================================================
앞으로 걸어가며 살피기 보이기
======================================================================

--- 굴러가는 창으로 앞으로 걸어가기 ---
쪼갬: 11
고른 익힌 안 샤프:  0.2371
고른 익힌 밖 샤프: 0.0727
떨어짐:     0.1644
한데 모은 익힌 밖:   0.1037

--- 넓혀 가는 창으로 앞으로 걸어가기 ---
쪼갬: 13
고른 익힌 안 샤프:  0.1508
고른 익힌 밖 샤프: -0.2783
떨어짐:     0.4290
한데 모은 익힌 밖:   -0.2733
```

## 2. 논의

워크포워드 분석기는 시간 순서를 지키면서 시험 구간이 서로 겹치지 않는 학습-시험 분할을 만든다. 분할마다 학습 윈도(여기서 전략을 최적화한다), 선택 사항인 갭(데이터 누수로 생기는 미래 참조 바이어스를 막는다), 시험 윈도(여기서 성능을 평가한다)로 이루어진다. 이 구조가 시간을 따라 앞으로 미끄러지며 표본 외 성능 추정을 여럿 내놓고, 이를 한데 모아 전체 판단을 내린다.

윈도를 잡는 방식은 흔히 둘, 곧 롤링 윈도와 확장 윈도이다. 롤링 윈도는 크기가 고정된 학습 구간을 앞으로 옮기며, 윈도 안에서는 가까운 과거와 먼 과거에 같은 가중치를 준다. 확장 윈도는 학습 데이터를 시간이 갈수록 늘려 가용한 과거 데이터를 모두 담는다. 시장의 성질이 시간에 따라 바뀌는 비정상적 환경에서는 롤링 윈도가 낫고, 데이터가 많을수록 모델이 꾸준히 좋아질 때는 확장 윈도가 낫다.

성능 저하 지표, 곧 평균 표본 내 샤프 비율과 평균 표본 외 샤프 비율의 차이는 과적합을 직접 재는 지표다. 저하가 크면 전략이 신호가 아니라 잡음을 붙든 것이다. 잘 설계한 전략이라면 분할마다 표본 외 성능이 일관되고 저하가 0에 가까워야 한다. 위 출력에서 저하는 롤링 윈도가 0.1644, 확장 윈도가 0.4290이다. 다만 위 코드의 `is_sharpe`는 전략 수익률이 아니라 학습 구간 원래 수익률의 샤프 비율이다. 이 보기의 전략은 학습 구간 평균의 부호를 따르므로 전략의 표본 내 샤프는 그 절댓값이 되고, 학습 구간 평균이 음수였던 분할에서는 출력의 저하가 실제보다 작게 잡힌다. 분할마다의 성능을 지켜보면 레짐에 의존하는 성질도 드러난다. 어떤 시기에는 통하고 어떤 시기에는 무너지는 전략은 오래가는 우위가 아니라 잠깐 스쳐 가는 시장 상황을 이용하고 있을 수 있다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
일간 수익률 1000개짜리 데이터에서 학습=252, 시험=63, 갭=5인 롤링 윈도로 워크포워드 검증을 할 때 분할이 몇 개 나오는지 계산하여라. 표본 외 데이터 점은 몇 개 생기는가?

</div>

??? success "풀이"
    분할마다 $252 + 5 + 63 = 320$일을 쓴다. 스텝 크기는 시험 윈도(63일)와 같다. 0일에서 시작하면

    분할 1: 학습 [0, 252), 시험 [257, 320)
    분할 2: 학습 [63, 315), 시험 [320, 383)
    ...

    분할의 수 $= \lfloor (1000 - 320) / 63 \rfloor + 1 = \lfloor 680/63 \rfloor + 1 = 10 + 1 = 11$이다. 위 출력의 롤링 윈도 분할 수 11과 같다.

    전체 표본 외 날수 $= 11 \times 63 = 693$일이다(첫 학습 윈도와 갭 뒤의 데이터를 거의 다 덮는다).

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
금융 데이터로 워크포워드 검증을 할 때 학습 윈도와 시험 윈도 사이에 갭을 두는 일이 왜 중요한지 밝혀라. 이 갭이 막아 주는 미래 참조 바이어스의 보기를 들어라.

</div>

??? success "풀이"
    갭은 학습-시험 경계를 넘나드는 피처에서 정보가 새는 것을 막는다. 예를 들어 전략이 20일 이동 평균을 피처로 쓴다면, 시험 구간 첫날의 이동 평균에는 그 앞 20일, 곧 학습 구간 끝부분의 정보가 들어 있다. 갭이 없으면 시험의 첫 며칠치 피처가 학습 데이터와 겹쳐 성능이 부풀어 보인다.

    5일 갭을 두면 룩백 길이가 짧은 피처는 학습 정보가 시험 구간으로 새지 않는다. 룩백 길이가 긴 피처(예를 들어 60일 변동성)에는 그만큼 긴 갭이 있어야 할 수 있다. 이 갭은 실제 거래 구조에서 모델을 만드는 시점(학습)과 배포하는 시점(시험) 사이에 놓이는 지연도 반영한다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
확장 윈도를 쓰고, 분할마다의 표본 외 샤프 비율의 변이 계수(표준편차를 평균으로 나눈 것)로 안정성 지표를 계산하는 워크포워드 검증을 구현하여라.

</div>

??? success "풀이"
    ```python
    def expanding_walk_forward(returns, train_fn, eval_fn,
                               min_train=252, test_window=63, gap=5):
        T = len(returns)
        splits = []
        test_start = min_train + gap

        while test_start + test_window <= T:
            train_r = returns[:test_start - gap]
            test_r = returns[test_start:test_start + test_window]

            model = train_fn(train_r)
            strat_returns = eval_fn(model, test_r)
            sharpe = np.mean(strat_returns) / (np.std(strat_returns) + 1e-8) * np.sqrt(252)
            splits.append(sharpe)
            test_start += test_window

        splits = np.array(splits)
        stability = np.std(splits) / (np.abs(np.mean(splits)) + 1e-8)

        return {'mean_oos_sharpe': np.mean(splits),
                'stability_cv': stability,
                'consistent': stability < 1.0}
    ```
    평균 표본 외 샤프 비율이 양수이고 변이 계수가 1.0 아래면, 분할 사이의 흔들림이 평균보다 작아 전략이 시기마다 꾸준히 양수 샤프 비율을 낸다는 뜻이다. 변이 계수는 평균의 절댓값으로 나누므로 평균이 음수여도 작게 나올 수 있으니 부호를 함께 본다. 2.0을 넘으면 성과가 레짐에 크게 좌우되어 오래가지 않을 가능성이 크다.

## 정리하며

**다룬 것** — 워크포워드 검증

워크포워드 분석기는 시간 순서를 지키면서 시험 구간이 서로 겹치지 않는 학습-시험 분할을 만든다.

핵심 클래스는 `WalkForwardConfig`, `WalkForwardAnalyzer`이며 앞의 연습문제 3개로 스스로 확인해 볼 수 있다.
