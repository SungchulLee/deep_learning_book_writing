# 앞으로 걸어가며 살피기

앞으로 걸어가며 살피기는 때 열 꾀를 따지는 으뜸 잣대다. 여느 엇갈려 살피기와 달리 지난 자료로 익히고 앞날 자료로 시험하되 창을 굴리거나 넓혀 가므로 자료의 때 차례를 지킨다. 이렇게 하면 익힌 밖 됨됨이를 참에 가깝게 어림할 수 있고, 저자 판이 바뀔 때 꾀가 얼마나 잘 듣는지도 드러난다.

## 코드

```python
"""
35.6.2장: 앞으로 걸어가며 살피기
========================================
힘 북돋우는 배움 꾀를 따지기 위한 앞으로 걸어가며 살피기.
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
            returns: (T, N)이나 (T,) 꼴 돌아옴 열
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

## 논의

앞으로 걸어가며 살피는 개는 때 차례를 지키면서 겹치지 않는 익힘-시험 쪼갬을 만든다. 쪼갬마다 익힘 창(여기서 꾀를 가장 좋게 한다), 있어도 되고 없어도 되는 틈(자료가 새어 앞을 미리 보는 치우침을 막는다), 시험 창(여기서 됨됨이를 따진다)으로 이루어진다. 이 얼개가 때를 따라 앞으로 미끄러지며 익힌 밖 됨됨이 어림을 여럿 내놓고, 이를 한데 모아 온 판단을 내린다.

창을 잡는 결은 흔히 둘, 곧 굴러가는 창과 넓혀 가는 창이다. 굴러가는 창은 크기가 붙박인 익힘 때를 앞으로 옮기며, 창 안에서는 가까운 지난날과 먼 지난날에 같은 무게를 준다. 넓혀 가는 창은 익힘 자료를 때가 갈수록 늘려 있는 지난 자료를 모두 담는다. 저자의 움직임이 바뀔 때(흔들림 없지 않을 때)는 굴러가는 창이 낫고, 자료가 많을수록 모형이 한결같이 좋아질 때는 넓혀 가는 창이 낫다.

떨어짐 자, 곧 고른 익힌 안 샤프 비와 고른 익힌 밖 샤프 비의 차이는 지나치게 맞추기를 곧바로 재는 자다. 떨어짐이 크면 꾀가 신호가 아니라 잡음을 붙든 것이다. 잘 꾸민 꾀라면 쪼갬마다 익힌 밖 됨됨이가 한결같고 떨어짐이 0에 가까워야 한다. 쪼갬마다의 됨됨이를 지켜보면 판에 매인 결도 드러난다. 어떤 때에는 되고 어떤 때에는 무너지는 꾀는 오래가는 잇속이 아니라 잠깐 스쳐 가는 저자 형편을 파먹고 있을 수 있다.

## 익힘 문제

**익힘 1.**
날마다의 돌아옴 1000개짜리 자료에서 익힘=252, 시험=63, 틈=5인 굴러가는 창으로 앞으로 걸어가며 살필 때 쪼갬이 몇 개 나오는지 셈하여라. 익힌 밖 자료 점은 몇 개 생기는가?

??? success "익힘 1 풀이"
    쪼갬마다 $252 + 5 + 63 = 320$일을 쓴다. 걸음 크기는 시험 창(63일)과 같다. 0일에서 시작하면

    쪼갬 1: 익힘 [0, 252), 시험 [257, 320)
    쪼갬 2: 익힘 [63, 315), 시험 [320, 383)
    ...

    쪼갬의 수 $= \lfloor (1000 - 320) / 63 \rfloor + 1 = \lfloor 680/63 \rfloor + 1 = 10 + 1 = 11$(10~11개 남짓).

    온 익힌 밖 날수 $= 11 \times 63 = 693$일이다(첫 익힘 창 뒤의 자료를 거의 다 덮는다).

---


**익힘 2.**
금융 자료를 앞으로 걸어가며 살필 때 익힘 창과 시험 창 사이에 틈을 두는 일이 왜 중요한지 밝혀라. 이 틈이 막아 주는 앞을 미리 보는 치우침의 보기를 들어라.

??? success "익힘 2 풀이"
    틈은 익힘-시험 금을 넘나드는 결에서 소식이 새는 것을 막는다. 보기로 꾀가 20일 옮김 평균을 결로 쓴다면, 익힘의 마지막 날에 셈한 옮김 평균에는 그 앞 20일의 소식이 들어 있다. 틈이 없으면 시험의 첫 며칠치 결이 익힘 자료와 얼마쯤 겹쳐 됨됨이가 부풀어 보인다.

    5일 틈을 두면 되짚어 보는 길이가 짧은 결이라도 익힘 소식이 시험 때로 새지 않는다. 되짚어 보는 길이가 긴 결(보기로 60일 출렁임)에는 그만큼 긴 틈이 있어야 할 수 있다. 이 틈은 참 거래 얼개에서 모형을 짓는 때(익힘)와 내놓는 때(시험) 사이에 놓이는 늦음도 담아낸다.

---


**익힘 3.**
넓혀 가는 창을 쓰고, 쪼갬마다의 익힌 밖 샤프 비의 변이 계수(잣대 벗어남을 평균으로 나눈 것)로 한결같음 자를 셈하는 앞으로 걸어가며 살피기를 꾸며라.

??? success "익힘 3 풀이"
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
    한결같음 변이 계수가 1.0 아래면 꾀가 때마다 한결같이 양수 샤프 비를 낸다는 뜻이다. 2.0을 넘으면 판에 크게 매인 됨됨이여서 오래가지 않을 낌새가 크다.

