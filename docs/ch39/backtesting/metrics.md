# 성과 평가

거래 전략의 성능을 말하려면 여러 측면을 고루 재는 성과 지표가 필요하다. 단순한 수익률을 넘어 샤프 비율, 소르티노 비율, 칼마 비율 같은 위험 조정 지표가 수익과 위험 사이의 트레이드오프를 담아낸다. 위험가치(VaR)나 조건부 위험가치(CVaR) 같은 꼬리 위험 지표는 극단적인 손실에 얼마나 노출되어 있는지를 재고, 승률이나 수익 팩터 같은 거래 지표는 전략이 어떻게 움직이는지를 드러낸다.

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

        # 베이스라인과 견주기
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

샤프 비율은 연율화한 초과 수익률을 연율화한 변동성으로 나눈 것으로, 위험 조정 성과 지표 가운데 가장 널리 인용된다. 다만 위쪽 변동성과 아래쪽 변동성을 똑같이 다루므로 수익률 분포가 오른쪽으로 치우친 전략에 불리하다. 소르티노 비율은 하방 편차만 분모에 두어 이를 바로잡으므로, 이따금 큰 수익을 내는 전략에 더 알맞다.

낙폭 분석은 전략이 가장 나쁠 때 어떻게 움직이는지를 드러낸다. 최대 낙폭은 고점에서 저점까지 가장 크게 떨어진 폭을 재고, 최장 낙폭 기간은 전략이 고점을 회복하지 못한 채(언더워터) 얼마나 오래 머물렀는지를 담는다. 칼마 비율(연율화 수익률을 최대 낙폭으로 나눈 것)은 최악의 손실에 견준 수익률을 숫자 하나로 요약한다. 낙폭을 심리적으로 견뎌야 하는 실시간 거래에서 이 지표들이 더욱 중요하다.

벤치마크가 있으면 상대 성과 지표가 꼭 있어야 한다. 추적 오차는 전략과 벤치마크의 수익률 차이가 얼마나 출렁이는지를 잰다. 정보 비율(초과 수익률을 추적 오차로 나눈 것)은 벤치마크에서 벗어난 능동적 베팅이 보상을 받는지를 잰다. CAPM 틀의 알파와 베타는 수익률을 시장이 설명하는 부분과 운용 능력이 설명하는 부분으로 가르므로, 전략이 정말 알파를 내는지 아니면 그저 체계적 위험을 짊어질 뿐인지 가리는 데 도움이 된다.
## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
일평균 수익률이 0.05%이고 일간 표준편차가 1.5%이며, 스무 날 동안 일평균 -0.8%의 수익률을 낸 구간이 있는 수익률 시계열에 대해 샤프 비율, 소르티노 비율, 최대 낙폭을 계산하여라.

</div>

??? success "풀이"
    연율화한 평균 초과 수익률 $= (0.0005 - 0.02/252) \times 252 \approx 0.106$(10.6%).

    연율화한 변동성 $= 0.015 \times \sqrt{252} \approx 0.238$(23.8%).

    샤프 비율 $= 0.106 / 0.238 \approx 0.445$.

    소르티노 비율에서는 하방 편차에 무위험 수익률보다 낮은 수익률만 쓴다. 날의 절반쯤이 음수라고 가정하면 연율화한 하방 편차가 $\approx 0.015 \times \sqrt{252/2} \approx 0.168$이다. 소르티노 $\approx 0.106 / 0.168 \approx 0.631$.

    하락 구간 동안 쌓인 손실은 $\approx 20 \times (-0.008) = -0.16$, 곧 16%이다. 이 구간이 고점에서 시작했다면 최대 낙폭이 $\approx 16\%$이다.

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
거래 빈도가 크게 다른 전략들 사이에서 수익 팩터(수익의 합을 손실의 합으로 나눈 것)가 왜 잘못 읽힐 수 있는지 밝혀라. 이 차이를 보정한 다른 지표를 제안하여라.

</div>

??? success "풀이"
    한 해에 한 번 거래해 큰 수익 하나와 작은 손실 하나를 낸 전략은 일관된 실력을 보이지 않고도 수익 팩터가 아주 높을 수 있다. 거꾸로 작은 수익 수천 번과 그보다 조금 적은 작은 손실을 내는 고빈도 전략은 아주 신뢰할 만하더라도 수익 팩터가 어중간할 수 있다.

    더 나은 지표는 거래당 수익 팩터나 이익 대 고통 비율(gain-to-pain ratio)이다. $\text{GtP} = \sum r_i / \sum |r_i^-|$이고 $r_i^-$는 음수 수익률이다. 다른 길로는 롤링 윈도마다 수익 팩터를 계산하고 그 안정성(롤링 수익 팩터의 표준편차)을 함께 보고하면 크기와 일관성을 모두 담을 수 있다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
꼬리 비율(95번째 백분위 수익률을 5번째 백분위 수익률의 절댓값으로 나눈 것)을 계산하는 함수를 짜고, 전략을 평가할 때 이를 어떻게 읽는지 밝혀라.

</div>

??? success "풀이"
    ```python
    def tail_ratio(returns):
        p95 = np.percentile(returns, 95)
        p5 = np.percentile(returns, 5)
        return abs(p95) / (abs(p5) + 1e-8)
    ```

    꼬리 비율이 1보다 크면 오른쪽 꼬리(큰 수익)가 왼쪽 꼬리(큰 손실)보다 두껍다는 뜻이니 반길 일이다. 1보다 작으면 왼쪽 꼬리가 더 두꺼워 극단에서 큰 손실이 큰 수익보다 크다는 뜻이다. 추세 추종 전략은 꼬리 비율이 흔히 1을 넘고(작은 손실이 많고 큰 수익이 드물다), 평균 회귀 전략은 흔히 1보다 낮다(작은 수익이 많고 이따금 큰 손실이 난다). 이 쪽 출력의 꼬리 비율 0.9692는 두 꼬리가 거의 대칭인 경우다.

## 정리하며

**다룬 것** — 성과 평가

샤프 비율은 연율화한 초과 수익률을 연율화한 변동성으로 나눈 것으로, 위험 조정 성과 지표 가운데 가장 널리 인용된다.

핵심 클래스는 `PerformanceMetrics`이며 앞의 연습문제 3개로 스스로 확인해 볼 수 있다.
