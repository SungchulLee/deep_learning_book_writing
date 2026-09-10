# 됨됨이 재기

두루 갖춘 됨됨이 자는 거래 꾀의 됨됨이를 나누는 말이다. 단순한 돌아옴을 넘어 샤프 비, 소르티노 비, 칼마 비 같은 무릅씀을 맞춘 자가 갚음과 무릅씀 사이의 맞바꿈을 담아낸다. 무릅씀 값(VaR)이나 매인 무릅씀 값(CVaR) 같은 꼬리 무릅씀 자는 끝자락 잃음에 얼마나 드러나 있는지를 재고, 이김률이나 벌이 인자 같은 거래 자는 꾀가 어떻게 움직이는지를 드러낸다.

## 1. 코드

```python
"""
35.6.3장: 됨됨이 재기
=====================================
거래 꾀를 위한 두루 갖춘 됨됨이 자.
"""

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass

# ========================================================================
# 메인
# ========================================================================


class PerformanceMetrics:
    """두루 갖춘 됨됨이 자를 셈한다."""

    def __init__(self, risk_free_rate: float = 0.02 / 252):
        self.rf = risk_free_rate

    def compute_all(self, returns: np.ndarray, benchmark_returns: Optional[np.ndarray] = None) -> Dict:
        metrics = {}

        # 돌아옴 자
        metrics["total_return"] = float(np.prod(1 + returns) - 1)
        metrics["cagr"] = float((1 + metrics["total_return"]) ** (252 / max(len(returns), 1)) - 1)
        metrics["daily_mean"] = float(np.mean(returns))

        # 무릅씀 자
        metrics["volatility"] = float(np.std(returns) * np.sqrt(252))
        metrics["downside_vol"] = float(np.std(returns[returns < 0]) * np.sqrt(252)) if np.any(returns < 0) else 0.0

        # 내림폭
        cum = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cum)
        dd = (peak - cum) / (peak + 1e-8)
        metrics["max_drawdown"] = float(np.max(dd))

        # 가장 긴 내림폭이 이어진 때를 찾는다
        underwater = dd > 0
        durations = []
        current = 0
        for u in underwater:
            if u:
                current += 1
            else:
                if current > 0:
                    durations.append(current)
                current = 0
        if current > 0:
            durations.append(current)
        metrics["max_dd_duration"] = max(durations) if durations else 0

        # VaR과 CVaR
        sorted_r = np.sort(returns)
        n5 = max(1, int(len(sorted_r) * 0.05))
        metrics["var_95"] = float(-sorted_r[n5])
        metrics["cvar_95"] = float(-np.mean(sorted_r[:n5]))

        # 무릅씀을 맞춘 자
        excess = returns - self.rf
        metrics["sharpe_ratio"] = float(np.mean(excess) / (np.std(excess) + 1e-8) * np.sqrt(252))

        downside = returns[returns < self.rf]
        ds_std = np.std(downside) * np.sqrt(252) if len(downside) > 1 else 1e-8
        metrics["sortino_ratio"] = float((metrics["cagr"] - 0.02) / (ds_std + 1e-8))

        metrics["calmar_ratio"] = float(metrics["cagr"] / (metrics["max_drawdown"] + 1e-8))

        # 거래 자
        metrics["win_rate"] = float(np.mean(returns > 0))
        gains = returns[returns > 0]
        losses = returns[returns < 0]
        metrics["profit_factor"] = float(np.sum(gains) / (np.abs(np.sum(losses)) + 1e-8))
        metrics["avg_win"] = float(np.mean(gains)) if len(gains) > 0 else 0.0
        metrics["avg_loss"] = float(np.mean(losses)) if len(losses) > 0 else 0.0

        # 꼬리 비
        p95 = np.percentile(returns, 95)
        p5 = np.percentile(returns, 5)
        metrics["tail_ratio"] = float(abs(p95) / (abs(p5) + 1e-8))

        # 밑금과 견주기
        if benchmark_returns is not None:
            te = returns - benchmark_returns
            metrics["tracking_error"] = float(np.std(te) * np.sqrt(252))
            metrics["information_ratio"] = float(np.mean(te) / (np.std(te) + 1e-8) * np.sqrt(252))
            metrics["beta"] = float(np.cov(returns, benchmark_returns)[0, 1] / (np.var(benchmark_returns) + 1e-8))
            metrics["alpha"] = float((metrics["cagr"] - 0.02) - metrics["beta"] * (np.mean(benchmark_returns) * 252 - 0.02))

        return metrics

    def format_report(self, metrics: Dict) -> str:
        lines = ["=" * 50, "됨됨이 알림", "=" * 50]
        sections = {
            "돌아옴": ["total_return", "cagr", "daily_mean"],
            "무릅씀": ["volatility", "max_drawdown", "max_dd_duration", "var_95", "cvar_95"],
            "무릅씀 맞춤": ["sharpe_ratio", "sortino_ratio", "calmar_ratio"],
            "거래": ["win_rate", "profit_factor", "avg_win", "avg_loss", "tail_ratio"],
        }
        for section, keys in sections.items():
            lines.append(f"\n--- {section} ---")
            for k in keys:
                if k in metrics:
                    v = metrics[k]
                    if "return" in k or "cagr" in k or "rate" in k or "alpha" in k:
                        lines.append(f"  {k:<22}: {v*100:>10.2f}%")
                    elif "ratio" in k or "factor" in k or "beta" in k:
                        lines.append(f"  {k:<22}: {v:>10.4f}")
                    elif "duration" in k:
                        lines.append(f"  {k:<22}: {v:>10.0f}일")
                    else:
                        lines.append(f"  {k:<22}: {v:>10.6f}")
        return "\n".join(lines)


def demo_metrics():
    """됨됨이 자를 보인다."""
    print("=" * 70)
    print("됨됨이 재기 보이기")
    print("=" * 70)

    np.random.seed(42)
    T = 504  # 2년
    strategy_returns = np.random.randn(T) * 0.012 + 0.0004
    strategy_returns[100:120] -= 0.02  # 내림폭

    benchmark_returns = np.random.randn(T) * 0.01 + 0.0003

    pm = PerformanceMetrics()
    metrics = pm.compute_all(strategy_returns, benchmark_returns)
    print(pm.format_report(metrics))


if __name__ == "__main__":
    demo_metrics()
```

**출력:**

```
======================================================================
됨됨이 재기 보이기
======================================================================
==================================================
됨됨이 알림
==================================================

--- 돌아옴 ---
  total_return          :     -15.82%
  cagr                  :      -8.25%
  daily_mean            :  -0.000265

--- 무릅씀 ---
  volatility            :   0.196875
  max_drawdown          :   0.434106
  max_dd_duration       :   494.0000
  var_95                :   0.020041
  cvar_95               :   0.025786

--- 무릅씀 맞춤 ---
  sharpe_ratio          :    -0.4403
  sortino_ratio         :    -0.8636
  calmar_ratio          :    -0.1900

--- 거래 ---
  win_rate              :      50.99%
  profit_factor         :     0.9478
  avg_win               :   0.009429
  avg_loss              :  -0.010350
  tail_ratio            :     0.9692
```

## 2. 논의

샤프 비는 해로 환산한 넘치는 돌아옴을 해로 환산한 출렁임으로 나눈 것으로, 무릅씀을 맞춘 됨됨이 자 가운데 가장 널리 인용된다. 다만 위로 튀는 출렁임과 아래로 처지는 출렁임을 똑같이 다루므로 돌아옴이 오른쪽으로 치우친 꾀에 벌을 준다. 소르티노 비는 아래로 처지는 벗어남만 아랫자리에 두어 이를 바로잡으므로, 이따금 큰 벌이를 내는 꾀에 더 알맞다.

내림폭 살피기는 꾀가 가장 나쁠 때 어떻게 움직이는지를 드러낸다. 가장 큰 내림폭은 마루에서 골까지 가장 크게 떨어진 만큼을 재고, 가장 긴 내림폭 이어짐은 꾀가 얼마나 오래 물속에 잠겨 있었는지를 담는다. 칼마 비(해로 환산한 돌아옴을 가장 큰 내림폭으로 나눈 것)는 가장 나쁜 잃음에 견준 돌아옴을 숫자 하나로 간추린다. 내림폭을 마음으로 견뎌야 하는 살아 있는 거래에서 이 자들이 더욱 종요롭다.

밑금이 있으면 견주는 됨됨이 자가 꼭 있어야 한다. 좇음 어긋남은 꾀와 밑금의 돌아옴 차이가 얼마나 출렁이는지를 잰다. 소식 비(넘치는 돌아옴을 좇음 어긋남으로 나눈 것)는 스스로 건 내기가 갚음을 받는지를 잰다. CAPM 틀의 알파와 베타는 돌아옴을 저자가 이끈 몫과 솜씨가 이끈 몫으로 가르므로, 꾀가 참으로 알파를 내는지 아니면 그저 얼개에 매인 무릅씀을 짊어질 뿐인지 가리는 데 도움이 된다.

## 연습문제

**연습문제 1.**
날마다 고르게 0.05% 돌아오고 날마다 잣대 벗어남이 1.5%이며, 스무 날 동안 날마다 고르게 -0.8% 돌아온 때가 있는 돌아옴 열에 대해 샤프 비, 소르티노 비, 가장 큰 내림폭을 셈하여라.

??? success "연습문제 1 풀이"
    해로 환산한 고른 넘치는 돌아옴 $= (0.0005 - 0.02/252) \times 252 \approx 0.106$(10.6%).

    해로 환산한 출렁임 $= 0.015 \times \sqrt{252} \approx 0.238$(23.8%).

    샤프 비 $= 0.106 / 0.238 \approx 0.445$.

    소르티노 비에서는 아래로 처지는 벗어남에 무릅씀 없는 빠르기보다 낮은 돌아옴만 쓴다. 날의 절반쯤이 음수라고 여기면 해로 환산한 아래 벗어남이 $\approx 0.015 \times \sqrt{252/2} \approx 0.168$이다. 소르티노 $\approx 0.106 / 0.168 \approx 0.631$.

    내림폭이 이어진 동안 쌓인 잃음은 $\approx 20 \times (-0.008) = -0.16$, 곧 16%이다. 이것이 마루에서 시작했다면 가장 큰 내림폭이 $\approx 16\%$이다.

---


**연습문제 2.**
거래 잦기가 크게 다른 꾀들 사이에서 벌이 인자(벌이의 합을 잃음의 합으로 나눈 것)가 왜 잘못 읽힐 수 있는지 밝혀라. 고르게 맞춘 다른 자를 내놓아라.

??? success "연습문제 2 풀이"
    해에 한 번 거래해 큰 벌이 하나와 작은 잃음 하나를 낸 꾀는 한결같은 솜씨를 보이지 않고도 벌이 인자가 아주 높을 수 있다. 거꾸로 작은 벌이 수천 번과 그보다 조금 적은 작은 잃음을 내는 잦은 거래 꾀는 아주 미덥더라도 벌이 인자가 어중간할 수 있다.

    더 나은 자는 거래마다의 벌이 인자나 벌이 대 아픔 비다. $\text{GtP} = \sum r_i / \sum |r_i^-|$이고 $r_i^-$은 음수 돌아옴이다. 다른 길로는 굴러가는 창마다 벌이 인자를 셈하고 그 한결같음(굴러가는 벌이 인자의 잣대 벗어남)을 함께 알리면 크기와 한결같음을 모두 담을 수 있다.

---


**연습문제 3.**
꼬리 비(95번째 백분위 벌이를 5번째 백분위 잃음의 절댓값으로 나눈 것)를 셈하는 함수를 짜고, 꾀를 따질 때 이를 어떻게 읽는지 밝혀라.

??? success "연습문제 3 풀이"
    ```python
    def tail_ratio(returns):
        p95 = np.percentile(returns, 95)
        p5 = np.percentile(returns, 5)
        return abs(p95) / (abs(p5) + 1e-8)
    ```

    꼬리 비가 1보다 크면 오른 꼬리(큰 벌이)가 왼 꼬리(큰 잃음)보다 두껍다는 뜻이니 반길 일이다. 1보다 작으면 왼 꼬리가 더 두꺼워 큰 잃음이 큰 벌이보다 끝자락에 있다는 뜻이다. 흐름을 좇는 꾀는 꼬리 비가 흔히 1을 넘고(작은 잃음이 많고 큰 벌이가 드물다), 평균으로 되돌아가는 꾀는 흔히 1보다 낮다(작은 벌이가 많고 이따금 큰 잃음이 난다).

## 정리하며

**다룬 것** — 됨됨이 재기

샤프 비는 해로 환산한 넘치는 돌아옴을 해로 환산한 출렁임으로 나눈 것으로, 무릅씀을 맞춘 됨됨이 자 가운데 가장 널리 인용된다.

고갱이 갈래는 `PerformanceMetrics`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
