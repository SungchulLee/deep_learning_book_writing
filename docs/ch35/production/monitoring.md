# 지켜보기

35.7.3장: 지켜보기. 살아 있는 힘 북돋우는 배움 거래 얼개를 실시간으로 지켜보기.

계량 금융에 깊은 배움을 올리려면 든든한 서비스 바탕이 있어야 한다. 이 꾸러미는 지켜보기, 무릅씀 다스리기, 금융 쓰임을 서비스에 올리는 꾀를 아우르는 서비스 얼개 설계 결을 다룬다.

## 1. 코드

```python
"""
35.7.3장: 지켜보기
=============================
살아 있는 힘 북돋우는 배움 거래 얼개를 실시간으로 지켜보기.
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

# ========================================================================
# 메인
# ========================================================================


class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class Alert:
    level: AlertLevel
    metric: str
    message: str
    value: float
    threshold: float
    timestamp: int = 0


class MetricTracker:
    """지켜보려고 굴러가는 자를 좇는다."""

    def __init__(self, window: int = 60):
        self.window = window
        self.values: deque = deque(maxlen=window)

    def update(self, value: float):
        self.values.append(value)

    def mean(self) -> float:
        return float(np.mean(self.values)) if self.values else 0.0

    def std(self) -> float:
        return float(np.std(self.values)) if len(self.values) > 1 else 0.0

    def last(self) -> float:
        return float(self.values[-1]) if self.values else 0.0

    def z_score(self) -> float:
        m, s = self.mean(), self.std()
        return (self.last() - m) / (s + 1e-8) if s > 1e-8 else 0.0


class TradingMonitor:
    """두루 갖춘 거래 얼개 지켜보개."""

    def __init__(self, config: Dict = None):
        self.config = config or {
            "max_drawdown": 0.10,
            "daily_loss_limit": 0.02,
            "max_position": 0.30,
            "max_leverage": 1.5,
            "sharpe_warning": 0.0,
            "vol_spike_threshold": 2.0,
        }

        self.return_tracker = MetricTracker(60)
        self.vol_tracker = MetricTracker(60)
        self.turnover_tracker = MetricTracker(20)
        self.alerts: List[Alert] = []
        self.step = 0

        self.portfolio_value = 1.0
        self.peak_value = 1.0
        self.daily_pnl = 0.0

    def update(self, metrics: Dict) -> List[Alert]:
        self.step += 1
        new_alerts = []

        # 좇개를 고쳐 쓴다
        ret = metrics.get("return", 0.0)
        self.return_tracker.update(ret)
        self.portfolio_value *= (1 + ret)
        self.peak_value = max(self.peak_value, self.portfolio_value)
        self.daily_pnl += ret

        vol = metrics.get("volatility", abs(ret))
        self.vol_tracker.update(vol)
        self.turnover_tracker.update(metrics.get("turnover", 0.0))

        # 내림폭 살피기
        dd = (self.peak_value - self.portfolio_value) / (self.peak_value + 1e-8)
        if dd > self.config["max_drawdown"]:
            new_alerts.append(Alert(
                AlertLevel.CRITICAL, "drawdown",
                f"내림폭 {dd*100:.1f}%이 위끝을 넘음", dd, self.config["max_drawdown"], self.step))
        elif dd > self.config["max_drawdown"] * 0.7:
            new_alerts.append(Alert(
                AlertLevel.WARNING, "drawdown",
                f"내림폭이 위끝에 다가감: {dd*100:.1f}%", dd, self.config["max_drawdown"], self.step))

        # 날마다의 잃음
        if self.daily_pnl < -self.config["daily_loss_limit"]:
            new_alerts.append(Alert(
                AlertLevel.EMERGENCY, "daily_loss",
                f"오늘 잃음 {self.daily_pnl*100:.1f}%이 위끝을 넘음",
                self.daily_pnl, -self.config["daily_loss_limit"], self.step))

        # 출렁임이 치솟음
        vol_z = self.vol_tracker.z_score()
        if abs(vol_z) > self.config["vol_spike_threshold"]:
            new_alerts.append(Alert(
                AlertLevel.WARNING, "vol_spike",
                f"출렁임이 치솟음: z 점수={vol_z:.2f}", vol_z,
                self.config["vol_spike_threshold"], self.step))

        # 자리가 쏠림
        weights = metrics.get("weights", np.array([]))
        if len(weights) > 0 and np.max(np.abs(weights)) > self.config["max_position"]:
            new_alerts.append(Alert(
                AlertLevel.WARNING, "concentration",
                f"자리가 쏠림: 가장 큼={np.max(np.abs(weights)):.2f}",
                float(np.max(np.abs(weights))), self.config["max_position"], self.step))

        self.alerts.extend(new_alerts)
        return new_alerts

    def reset_daily(self):
        self.daily_pnl = 0.0

    def get_dashboard(self) -> Dict:
        returns = np.array(self.return_tracker.values) if self.return_tracker.values else np.array([0])
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        dd = (self.peak_value - self.portfolio_value) / (self.peak_value + 1e-8)

        return {
            "portfolio_value": self.portfolio_value,
            "drawdown": dd,
            "rolling_sharpe": sharpe,
            "rolling_vol": self.vol_tracker.mean() * np.sqrt(252),
            "avg_turnover": self.turnover_tracker.mean(),
            "total_alerts": len(self.alerts),
            "critical_alerts": sum(1 for a in self.alerts if a.level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY]),
        }


def demo_monitoring():
    """거래 지켜보기를 보인다."""
    print("=" * 70)
    print("거래 지켜보기 보이기")
    print("=" * 70)

    monitor = TradingMonitor()
    np.random.seed(42)

    for step in range(100):
        ret = np.random.randn() * 0.01 + 0.0002
        if 40 <= step <= 50:
            ret -= 0.015  # 내림폭이 이어진 때

        weights = np.random.dirichlet(np.ones(5))
        alerts = monitor.update({
            "return": ret,
            "volatility": abs(ret),
            "turnover": np.random.uniform(0, 0.1),
            "weights": weights,
        })
        if alerts:
            for a in alerts:
                print(f"  [{a.level.value:>9}] 걸음 {step}: {a.message}")

    print(f"\n--- 계기판 ---")
    dash = monitor.get_dashboard()
    for k, v in dash.items():
        if isinstance(v, float):
            print(f"  {k:<20}: {v:.4f}")
        else:
            print(f"  {k:<20}: {v}")


if __name__ == "__main__":
    demo_monitoring()```

## 2. 논의

이 짜보기는 깔끔하고 읽기 쉬운 PyTorch 코드로 서비스 얼개의 고갱이가 되는 생각을 보여 준다. 조각으로 나눈 얼개 덕에 부분마다 따로 살피고 다른 일이나 자료에 맞추어 고치기 쉽다.

여기서 보인 결은 더 까다로운 자리로도 자연스레 넓혀진다. 하이퍼파라미터, 얼개의 갈래, 여러 자료를 바꿔 가며 해 보면 이해가 깊어지고 서비스에 올리는 일에 대한 감이 몸에 붙는다.

## 연습문제

**연습문제 1.**
코드를 읽고 고갱이가 되는 설계 판단을 짚어라. 짜기에서 고른 것 셋을 들고, 저마다 왜 서비스 얼개에 알맞은지 밝혀라.

??? success "연습문제 1 풀이"
    설계 판단은 짜보기마다 다르나 흔히 이런 것이 있다. (1) 살림 함수 고르기 -- ReLU 갈래는 기울기가 잦아들지 않아 익히기가 빠르다. (2) 고르게 하는 꾀 -- 묶음 고르게 하기가 안쪽 함께 바뀌는 옮겨감을 줄여 익힘을 든든하게 한다. (3) 나머지 이음 -- 있으면 건너뛰는 길을 주어 깊은 그물에서 기울기가 흐르게 한다. 고른 것마다 나타내는 힘, 셈 값, 익힘의 든든함 사이의 맞바꿈을 드러낸다.

---

**연습문제 2.**
들임의 꼴과 자료 갈래가 바라는 대로인지 살피는 들임 살피기를 으뜸 함수나 클래스에 더하여라. 올바르지 않은 들임에는 알아듣기 쉬운 어긋남 알림을 띄워라.

??? success "연습문제 2 풀이"
    `forward` 방법(또는 알맞은 함수)의 첫머리에 `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`이나 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'` 같은 살핌을 더한다. 꼴을 살피려면 종요로운 차원을 본다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 알아듣기 쉬운 어긋남 알림은 벌레잡기를 크게 앞당기고 코드를 되쓰기 든든하게 한다.

---

**연습문제 3.**
이 짜보기가 무너질 만한 결 둘을 밝히고, 저마다 어떻게 짚어내고 고칠지 밝혀라.

??? success "연습문제 3 풀이"
    흔히 무너지는 결은 이렇다. (1) **기울기가 사라지거나 터짐** -- 기울기 크기를 지켜보아 짚어낸다(`torch.nn.utils.clip_grad_norm_`이나 켜마다 `param.grad.norm()` 적기). 기울기 자르기, 더 나은 첫값 잡기(Xavier/Kaiming), 얼개 고치기(나머지 이음, 고르게 하기)로 고친다. (2) **지나치게 맞추기** -- 익힘 잃음은 줄어드는데 살핌 잃음이 오르면 짚어낸다. 정칙화(드롭아웃, 짐 줄이기, 자료 늘리기)나 모형 크기 줄이기로 고친다. 익힘과 살핌 자를 늘 함께 지켜보아 이를 일찍 잡아야 한다.

---

**연습문제 4.**
지켜보기 짜보기를 살피는 두루 갖춘 시험 함수를 써라. 빈 들임, 원소 하나짜리 들임, 아주 큰 들임, 그리고 끝자락 값(0, 아주 큰 수)이 든 들임 같은 가장자리 자리를 시험하여라.

??? success "연습문제 4 풀이"
    금 언저리 조건을 두루 건드리는 시험 함수를 짓는다.
    ```python
    def test_alertlevel():
        model = AlertLevel(...)
        # 여느 들임
        assert model(normal_input).shape == expected_shape
        # 원소 하나짜리 묶음
        assert model(single_input).shape == (1, ...)
        # 큰 값(넘침을 살핀다)
        out = model(torch.ones(...) * 1000)
        assert torch.isfinite(out).all()
        # 기울기 흐름
        out = model(normal_input)
        out.sum().backward()
        for p in model.parameters():
            assert p.grad is not None
    ```
    얼개가 끝에서 끝까지 익히기를 받치는지 알려면 기울기 흐름을 시험하는 것이 특히 중요하다.

## 정리하며

**다룬 것** — 지켜보기

이 짜보기는 깔끔하고 읽기 쉬운 PyTorch 코드로 서비스 얼개의 고갱이가 되는 생각을 보여 준다.

고갱이 갈래는 `AlertLevel`, `Alert`, `MetricTracker`, `TradingMonitor`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
