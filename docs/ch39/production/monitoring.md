# 모니터링

35.7.3장: 지켜보기. 살아 있는 강화학습 거래 구조를 실시간으로 지켜보기.

계량 금융에 딥러닝을 올리려면 든든한 서비스 기초가 있어야 한다. 이 꾸러미는 지켜보기, 위험 다스리기, 금융 쓰임을 서비스에 올리는 꾀를 아우르는 서비스 구조 설계 결을 다룬다.

## 1. 코드

```python
"""
35.7.3장: 지켜보기
=============================
살아 있는 힘 북돋우는 러닝 거래 구조를 실시간으로 지켜보기.
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
    """두루 갖춘 거래 구조 지켜보개."""

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
                print(f"  [{a.level.value:>9}] 스텝 {step}: {a.message}")

    print(f"\n--- 계기판 ---")
    dash = monitor.get_dashboard()
    for k, v in dash.items():
        if isinstance(v, float):
            print(f"  {k:<20}: {v:.4f}")
        else:
            print(f"  {k:<20}: {v}")


if __name__ == "__main__":
    demo_monitoring()
```

??? note "전체 출력 (230줄)"

    ```
    ======================================================================
    거래 지켜보기 보이기
    ======================================================================
      [  warning] 스텝 0: 자리가 쏠림: 가장 큼=0.50
      [  warning] 스텝 1: 자리가 쏠림: 가장 큼=0.47
      [  warning] 스텝 2: 자리가 쏠림: 가장 큼=0.32
      [  warning] 스텝 3: 자리가 쏠림: 가장 큼=0.48
      [  warning] 스텝 4: 자리가 쏠림: 가장 큼=0.45
      [  warning] 스텝 5: 자리가 쏠림: 가장 큼=0.49
      [  warning] 스텝 6: 자리가 쏠림: 가장 큼=0.62
      [  warning] 스텝 7: 출렁임이 치솟음: z 점수=2.02
      [  warning] 스텝 7: 자리가 쏠림: 가장 큼=0.33
      [  warning] 스텝 8: 자리가 쏠림: 가장 큼=0.53
      [  warning] 스텝 9: 자리가 쏠림: 가장 큼=0.57
      [  warning] 스텝 10: 자리가 쏠림: 가장 큼=0.48
      [  warning] 스텝 11: 자리가 쏠림: 가장 큼=0.51
      [  warning] 스텝 13: 자리가 쏠림: 가장 큼=0.50
      [  warning] 스텝 14: 자리가 쏠림: 가장 큼=0.49
      [  warning] 스텝 15: 출렁임이 치솟음: z 점수=2.65
      [  warning] 스텝 15: 자리가 쏠림: 가장 큼=0.46
      [  warning] 스텝 16: 자리가 쏠림: 가장 큼=0.31
      [  warning] 스텝 17: 자리가 쏠림: 가장 큼=0.44
      [  warning] 스텝 18: 자리가 쏠림: 가장 큼=0.63
      [emergency] 스텝 19: 오늘 잃음 -2.3%이 위끝을 넘음
      [  warning] 스텝 19: 자리가 쏠림: 가장 큼=0.43
      [emergency] 스텝 20: 오늘 잃음 -2.2%이 위끝을 넘음
      [  warning] 스텝 20: 자리가 쏠림: 가장 큼=0.75
      [  warning] 스텝 21: 출렁임이 치솟음: z 점수=2.88
      [  warning] 스텝 21: 자리가 쏠림: 가장 큼=0.58
      [  warning] 스텝 22: 자리가 쏠림: 가장 큼=0.44
      [  warning] 스텝 23: 자리가 쏠림: 가장 큼=0.40
      [  warning] 스텝 24: 자리가 쏠림: 가장 큼=0.55
      [  warning] 스텝 25: 자리가 쏠림: 가장 큼=0.33
      [  warning] 스텝 26: 자리가 쏠림: 가장 큼=0.38
      [  warning] 스텝 28: 자리가 쏠림: 가장 큼=0.37
      [  warning] 스텝 29: 자리가 쏠림: 가장 큼=0.32
      [  warning] 스텝 30: 자리가 쏠림: 가장 큼=0.76
      [  warning] 스텝 31: 자리가 쏠림: 가장 큼=0.35
      [  warning] 스텝 32: 자리가 쏠림: 가장 큼=0.64
      [  warning] 스텝 33: 자리가 쏠림: 가장 큼=0.43
      [  warning] 스텝 34: 자리가 쏠림: 가장 큼=0.46
      [  warning] 스텝 35: 자리가 쏠림: 가장 큼=0.69
      [  warning] 스텝 36: 자리가 쏠림: 가장 큼=0.34
      [  warning] 스텝 37: 자리가 쏠림: 가장 큼=0.33
      [  warning] 스텝 38: 자리가 쏠림: 가장 큼=0.54
      [  warning] 스텝 39: 자리가 쏠림: 가장 큼=0.33
      [  warning] 스텝 40: 자리가 쏠림: 가장 큼=0.41
      [  warning] 스텝 41: 자리가 쏠림: 가장 큼=0.33
      [emergency] 스텝 42: 오늘 잃음 -5.0%이 위끝을 넘음
      [  warning] 스텝 42: 출렁임이 치솟음: z 점수=3.73
      [  warning] 스텝 42: 자리가 쏠림: 가장 큼=0.61
      [  warning] 스텝 43: 내림폭이 위끝에 다가감: 7.8%
      [emergency] 스텝 43: 오늘 잃음 -6.3%이 위끝을 넘음
      [  warning] 스텝 43: 자리가 쏠림: 가장 큼=0.47
      [  warning] 스텝 44: 내림폭이 위끝에 다가감: 7.6%
      [emergency] 스텝 44: 오늘 잃음 -6.0%이 위끝을 넘음
      [  warning] 스텝 44: 자리가 쏠림: 가장 큼=0.47
      [  warning] 스텝 45: 내림폭이 위끝에 다가감: 8.6%
      [emergency] 스텝 45: 오늘 잃음 -7.1%이 위끝을 넘음
      [  warning] 스텝 45: 자리가 쏠림: 가장 큼=0.52
      [  warning] 스텝 46: 내림폭이 위끝에 다가감: 9.5%
      [emergency] 스텝 46: 오늘 잃음 -8.1%이 위끝을 넘음
      [  warning] 스텝 46: 자리가 쏠림: 가장 큼=0.30
      [ critical] 스텝 47: 내림폭 10.2%이 위끝을 넘음
      [emergency] 스텝 47: 오늘 잃음 -8.8%이 위끝을 넘음
      [  warning] 스텝 47: 자리가 쏠림: 가장 큼=0.34
      [ critical] 스텝 48: 내림폭 10.4%이 위끝을 넘음
      [emergency] 스텝 48: 오늘 잃음 -9.1%이 위끝을 넘음
      [  warning] 스텝 48: 자리가 쏠림: 가장 큼=0.45
      [ critical] 스텝 49: 내림폭 11.8%이 위끝을 넘음
      [emergency] 스텝 49: 오늘 잃음 -10.6%이 위끝을 넘음
      [  warning] 스텝 49: 자리가 쏠림: 가장 큼=0.48
      [ critical] 스텝 50: 내림폭 13.2%이 위끝을 넘음
      [emergency] 스텝 50: 오늘 잃음 -12.2%이 위끝을 넘음
      [  warning] 스텝 50: 자리가 쏠림: 가장 큼=0.54
      [ critical] 스텝 51: 내림폭 12.7%이 위끝을 넘음
      [emergency] 스텝 51: 오늘 잃음 -11.6%이 위끝을 넘음
      [  warning] 스텝 51: 자리가 쏠림: 가장 큼=0.52
      [ critical] 스텝 52: 내림폭 13.2%이 위끝을 넘음
      [emergency] 스텝 52: 오늘 잃음 -12.2%이 위끝을 넘음
      [  warning] 스텝 52: 자리가 쏠림: 가장 큼=0.37
      [ critical] 스텝 53: 내림폭 12.4%이 위끝을 넘음
      [emergency] 스텝 53: 오늘 잃음 -11.3%이 위끝을 넘음
      [  warning] 스텝 53: 자리가 쏠림: 가장 큼=0.57
      [ critical] 스텝 54: 내림폭 11.3%이 위끝을 넘음
      [emergency] 스텝 54: 오늘 잃음 -10.0%이 위끝을 넘음
      [  warning] 스텝 54: 자리가 쏠림: 가장 큼=0.38
      [ critical] 스텝 55: 내림폭 11.8%이 위끝을 넘음
      [emergency] 스텝 55: 오늘 잃음 -10.6%이 위끝을 넘음
      [  warning] 스텝 55: 자리가 쏠림: 가장 큼=0.44
      [ critical] 스텝 56: 내림폭 11.5%이 위끝을 넘음
      [emergency] 스텝 56: 오늘 잃음 -10.2%이 위끝을 넘음
      [  warning] 스텝 56: 자리가 쏠림: 가장 큼=0.52
      [ critical] 스텝 57: 내림폭 11.6%이 위끝을 넘음
      [emergency] 스텝 57: 오늘 잃음 -10.4%이 위끝을 넘음
      [ critical] 스텝 58: 내림폭 11.5%이 위끝을 넘음
      [emergency] 스텝 58: 오늘 잃음 -10.3%이 위끝을 넘음
      [  warning] 스텝 58: 자리가 쏠림: 가장 큼=0.63
      [ critical] 스텝 59: 내림폭 12.1%이 위끝을 넘음
      [emergency] 스텝 59: 오늘 잃음 -10.9%이 위끝을 넘음
      [  warning] 스텝 59: 자리가 쏠림: 가장 큼=0.49
      [ critical] 스텝 60: 내림폭 12.5%이 위끝을 넘음
      [emergency] 스텝 60: 오늘 잃음 -11.4%이 위끝을 넘음
      [  warning] 스텝 60: 자리가 쏠림: 가장 큼=0.62
      [ critical] 스텝 61: 내림폭 12.2%이 위끝을 넘음
      [emergency] 스텝 61: 오늘 잃음 -11.1%이 위끝을 넘음
      [  warning] 스텝 61: 자리가 쏠림: 가장 큼=0.55
      [ critical] 스텝 62: 내림폭 11.5%이 위끝을 넘음
      [emergency] 스텝 62: 오늘 잃음 -10.2%이 위끝을 넘음
      [  warning] 스텝 62: 자리가 쏠림: 가장 큼=0.45
      [ critical] 스텝 63: 내림폭 11.6%이 위끝을 넘음
      [emergency] 스텝 63: 오늘 잃음 -10.4%이 위끝을 넘음
      [  warning] 스텝 63: 자리가 쏠림: 가장 큼=0.49
      [ critical] 스텝 64: 내림폭 11.0%이 위끝을 넘음
      [emergency] 스텝 64: 오늘 잃음 -9.7%이 위끝을 넘음
      [  warning] 스텝 64: 자리가 쏠림: 가장 큼=0.49
      [ critical] 스텝 65: 내림폭 11.3%이 위끝을 넘음
      [emergency] 스텝 65: 오늘 잃음 -10.0%이 위끝을 넘음
      [  warning] 스텝 65: 자리가 쏠림: 가장 큼=0.49
      [ critical] 스텝 66: 내림폭 10.0%이 위끝을 넘음
      [emergency] 스텝 66: 오늘 잃음 -8.6%이 위끝을 넘음
      [  warning] 스텝 66: 자리가 쏠림: 가장 큼=0.48
      [  warning] 스텝 67: 내림폭이 위끝에 다가감: 9.1%
      [emergency] 스텝 67: 오늘 잃음 -7.6%이 위끝을 넘음
      [  warning] 스텝 67: 자리가 쏠림: 가장 큼=0.42
      [ critical] 스텝 68: 내림폭 11.0%이 위끝을 넘음
      [emergency] 스텝 68: 오늘 잃음 -9.7%이 위끝을 넘음
      [  warning] 스텝 68: 자리가 쏠림: 가장 큼=0.42
      [ critical] 스텝 69: 내림폭 11.5%이 위끝을 넘음
      [emergency] 스텝 69: 오늘 잃음 -10.2%이 위끝을 넘음
      [  warning] 스텝 69: 자리가 쏠림: 가장 큼=0.42
      [ critical] 스텝 70: 내림폭 12.6%이 위끝을 넘음
      [emergency] 스텝 70: 오늘 잃음 -11.5%이 위끝을 넘음
      [  warning] 스텝 70: 자리가 쏠림: 가장 큼=0.51
      [ critical] 스텝 71: 내림폭 11.0%이 위끝을 넘음
      [emergency] 스텝 71: 오늘 잃음 -9.7%이 위끝을 넘음
      [  warning] 스텝 71: 자리가 쏠림: 가장 큼=0.46
      [ critical] 스텝 72: 내림폭 11.1%이 위끝을 넘음
      [emergency] 스텝 72: 오늘 잃음 -9.8%이 위끝을 넘음
      [  warning] 스텝 72: 자리가 쏠림: 가장 큼=0.44
      [  warning] 스텝 73: 내림폭이 위끝에 다가감: 10.0%
      [emergency] 스텝 73: 오늘 잃음 -8.5%이 위끝을 넘음
      [  warning] 스텝 73: 자리가 쏠림: 가장 큼=0.45
      [ critical] 스텝 74: 내림폭 10.9%이 위끝을 넘음
      [emergency] 스텝 74: 오늘 잃음 -9.5%이 위끝을 넘음
      [  warning] 스텝 74: 자리가 쏠림: 가장 큼=0.49
      [ critical] 스텝 75: 내림폭 11.0%이 위끝을 넘음
      [emergency] 스텝 75: 오늘 잃음 -9.7%이 위끝을 넘음
      [  warning] 스텝 75: 자리가 쏠림: 가장 큼=0.35
      [  warning] 스텝 76: 내림폭이 위끝에 다가감: 9.2%
      [emergency] 스텝 76: 오늘 잃음 -7.6%이 위끝을 넘음
      [  warning] 스텝 76: 자리가 쏠림: 가장 큼=0.50
      [  warning] 스텝 77: 내림폭이 위끝에 다가감: 7.6%
      [emergency] 스텝 77: 오늘 잃음 -5.8%이 위끝을 넘음
      [  warning] 스텝 77: 자리가 쏠림: 가장 큼=0.59
      [  warning] 스텝 78: 내림폭이 위끝에 다가감: 8.7%
      [emergency] 스텝 78: 오늘 잃음 -7.0%이 위끝을 넘음
      [  warning] 스텝 78: 자리가 쏠림: 가장 큼=0.42
      [ critical] 스텝 79: 내림폭 10.5%이 위끝을 넘음
      [emergency] 스텝 79: 오늘 잃음 -9.0%이 위끝을 넘음
      [  warning] 스텝 79: 자리가 쏠림: 가장 큼=0.35
      [ critical] 스텝 80: 내림폭 10.1%이 위끝을 넘음
      [emergency] 스텝 80: 오늘 잃음 -8.6%이 위끝을 넘음
      [  warning] 스텝 80: 자리가 쏠림: 가장 큼=0.41
      [ critical] 스텝 81: 내림폭 10.1%이 위끝을 넘음
      [emergency] 스텝 81: 오늘 잃음 -8.6%이 위끝을 넘음
      [  warning] 스텝 81: 자리가 쏠림: 가장 큼=0.61
      [ critical] 스텝 82: 내림폭 10.2%이 위끝을 넘음
      [emergency] 스텝 82: 오늘 잃음 -8.6%이 위끝을 넘음
      [  warning] 스텝 82: 자리가 쏠림: 가장 큼=0.64
      [  warning] 스텝 83: 내림폭이 위끝에 다가감: 9.3%
      [emergency] 스텝 83: 오늘 잃음 -7.7%이 위끝을 넘음
      [  warning] 스텝 83: 자리가 쏠림: 가장 큼=0.49
      [  warning] 스텝 84: 내림폭이 위끝에 다가감: 7.5%
      [emergency] 스텝 84: 오늘 잃음 -5.7%이 위끝을 넘음
      [  warning] 스텝 84: 자리가 쏠림: 가장 큼=0.43
      [  warning] 스텝 85: 내림폭이 위끝에 다가감: 7.5%
      [emergency] 스텝 85: 오늘 잃음 -5.6%이 위끝을 넘음
      [  warning] 스텝 85: 자리가 쏠림: 가장 큼=0.45
      [  warning] 스텝 86: 내림폭이 위끝에 다가감: 7.7%
      [emergency] 스텝 86: 오늘 잃음 -5.9%이 위끝을 넘음
      [  warning] 스텝 86: 자리가 쏠림: 가장 큼=0.40
      [  warning] 스텝 87: 내림폭이 위끝에 다가감: 9.8%
      [emergency] 스텝 87: 오늘 잃음 -8.2%이 위끝을 넘음
      [  warning] 스텝 87: 자리가 쏠림: 가장 큼=0.44
      [ critical] 스텝 88: 내림폭 11.2%이 위끝을 넘음
      [emergency] 스텝 88: 오늘 잃음 -9.7%이 위끝을 넘음
      [  warning] 스텝 88: 자리가 쏠림: 가장 큼=0.33
      [ critical] 스텝 89: 내림폭 11.0%이 위끝을 넘음
      [emergency] 스텝 89: 오늘 잃음 -9.5%이 위끝을 넘음
      [  warning] 스텝 89: 자리가 쏠림: 가장 큼=0.37
      [ critical] 스텝 90: 내림폭 10.9%이 위끝을 넘음
      [emergency] 스텝 90: 오늘 잃음 -9.4%이 위끝을 넘음
      [  warning] 스텝 90: 자리가 쏠림: 가장 큼=0.42
      [ critical] 스텝 91: 내림폭 11.7%이 위끝을 넘음
      [emergency] 스텝 91: 오늘 잃음 -10.3%이 위끝을 넘음
      [  warning] 스텝 91: 자리가 쏠림: 가장 큼=0.57
      [ critical] 스텝 92: 내림폭 12.3%이 위끝을 넘음
      [emergency] 스텝 92: 오늘 잃음 -10.9%이 위끝을 넘음
      [  warning] 스텝 92: 자리가 쏠림: 가장 큼=0.74
      [ critical] 스텝 93: 내림폭 12.7%이 위끝을 넘음
      [emergency] 스텝 93: 오늘 잃음 -11.4%이 위끝을 넘음
      [  warning] 스텝 93: 자리가 쏠림: 가장 큼=0.47
      [ critical] 스텝 94: 내림폭 13.1%이 위끝을 넘음
      [emergency] 스텝 94: 오늘 잃음 -11.9%이 위끝을 넘음
      [  warning] 스텝 94: 자리가 쏠림: 가장 큼=0.47
      [ critical] 스텝 95: 내림폭 12.5%이 위끝을 넘음
      [emergency] 스텝 95: 오늘 잃음 -11.1%이 위끝을 넘음
      [  warning] 스텝 95: 자리가 쏠림: 가장 큼=0.44
      [ critical] 스텝 96: 내림폭 12.2%이 위끝을 넘음
      [emergency] 스텝 96: 오늘 잃음 -10.9%이 위끝을 넘음
      [  warning] 스텝 96: 자리가 쏠림: 가장 큼=0.31
      [ critical] 스텝 97: 내림폭 13.0%이 위끝을 넘음
      [emergency] 스텝 97: 오늘 잃음 -11.8%이 위끝을 넘음
      [  warning] 스텝 97: 자리가 쏠림: 가장 큼=0.82
      [ critical] 스텝 98: 내림폭 12.3%이 위끝을 넘음
      [emergency] 스텝 98: 오늘 잃음 -10.9%이 위끝을 넘음
      [  warning] 스텝 98: 자리가 쏠림: 가장 큼=0.36
      [ critical] 스텝 99: 내림폭 13.2%이 위끝을 넘음
      [emergency] 스텝 99: 오늘 잃음 -12.0%이 위끝을 넘음
      [  warning] 스텝 99: 자리가 쏠림: 가장 큼=0.36

    --- 계기판 ---
      portfolio_value     : 0.8817
      drawdown            : 0.1323
      rolling_sharpe      : -3.0441
      rolling_vol         : 0.1421
      avg_turnover        : 0.0413
      total_alerts        : 218
      critical_alerts     : 103
    ```


## 2. 논의

이 구현은 깔끔하고 읽기 쉬운 PyTorch 코드로 서비스 구조의 핵심이 되는 생각을 보여 준다. 조각으로 나눈 구조 덕에 부분마다 따로 살피고 다른 일이나 데이터에 맞추어 고치기 쉽다.

여기서 보인 결은 더 까다로운 자리로도 자연스레 넓혀진다. 하이퍼파라미터, 구조의 클래스, 여러 데이터를 바꿔 가며 해 보면 이해가 깊어지고 서비스에 올리는 일에 대한 감이 몸에 붙는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
코드를 읽고 핵심이 되는 설계 판단을 짚어라. 짜기에서 고른 것 셋을 들고, 저마다 왜 서비스 구조에 알맞은지 밝혀라.

</div>

??? success "연습문제 1 풀이"
    설계 판단은 구현마다 다르나 흔히 이런 것이 있다. (1) 엑티베이션 함수 고르기 -- ReLU 클래스는 그래디언트가 잦아들지 않아 학습이 빠르다. (2) 고르게 하는 꾀 -- 배치 고르게 하기가 안쪽 함께 바뀌는 옮겨감을 줄여 학습을 든든하게 한다. (3) 나머지 이음 -- 있으면 건너뛰는 길을 주어 깊은 네트워크에서 그래디언트가 흐르게 한다. 고른 것마다 나타내는 힘, 계산 비용, 학습의 든든함 사이의 트레이드오프를 드러낸다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
입력의 꼴과 데이터 클래스가 바라는 대로인지 살피는 입력 살피기를 으뜸 함수나 클래스에 더하여라. 올바르지 않은 입력에는 알아듣기 쉬운 어긋남 알림을 띄워라.

</div>

??? success "연습문제 2 풀이"
    `forward` 방법(또는 알맞은 함수)의 소개에 `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`이나 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'` 같은 살핌을 더한다. 꼴을 살피려면 중요한 차원을 본다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 알아듣기 쉬운 어긋남 알림은 디버깅을 크게 앞당기고 코드를 되쓰기 든든하게 한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
이 구현이 무너질 만한 결 둘을 밝히고, 저마다 어떻게 짚어내고 고칠지 밝혀라.

</div>

??? success "연습문제 3 풀이"
    흔히 무너지는 결은 이렇다. (1) **그래디언트가 사라지거나 터짐** -- 그래디언트 크기를 지켜보아 짚어낸다(`torch.nn.utils.clip_grad_norm_`이나 켜마다 `param.grad.norm()` 적기). 그래디언트 자르기, 더 나은 첫값 잡기(Xavier/Kaiming), 구조 고치기(나머지 이음, 고르게 하기)로 고친다. (2) **지나치게 맞추기** -- 학습 로스는 줄어드는데 살핌 로스가 오르면 짚어낸다. 정규화(드롭아웃, 짐 줄이기, 데이터 늘리기)나 모델 크기 줄이기로 고친다. 학습과 살핌 자를 늘 함께 지켜보아 이를 일찍 잡아야 한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
지켜보기 구현을 살피는 두루 갖춘 시험 함수를 써라. 빈 입력, 원소 하나짜리 입력, 아주 큰 입력, 그리고 끝자락 값(0, 아주 큰 수)이 든 입력 같은 가장자리 자리를 시험하여라.

</div>

??? success "연습문제 4 풀이"
    금 언저리 조건을 두루 건드리는 시험 함수를 짓는다.
    ```python
    def test_alertlevel():
        model = AlertLevel(...)
        # 여느 들임
        assert model(normal_input).shape == expected_shape
        # 원소 하나짜리 배치
        assert model(single_input).shape == (1, ...)
        # 큰 값(넘침을 살핀다)
        out = model(torch.ones(...) * 1000)
        assert torch.isfinite(out).all()
        # 그래디언트 흐름
        out = model(normal_input)
        out.sum().backward()
        for p in model.parameters():
            assert p.grad is not None
    ```
    구조가 끝에서 끝까지 학습을 받치는지 알려면 그래디언트 흐름을 시험하는 것이 특히 중요하다.

## 정리하며

**다룬 것** — 지켜보기

이 구현은 깔끔하고 읽기 쉬운 PyTorch 코드로 서비스 구조의 핵심이 되는 생각을 보여 준다.

핵심 클래스는 `AlertLevel`, `Alert`, `MetricTracker`, `TradingMonitor`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
