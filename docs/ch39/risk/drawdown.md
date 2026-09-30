# 내림폭

35.4.2장: 내림폭 다스리기. 자리 크기 잣대기와 회로 차단기를 곁들인, 내림폭을 살피는 힘 북돋우는 배움.

계량 금융에 깊은 배움을 올리려면 든든한 서비스 바탕이 있어야 한다. 이 꾸러미는 지켜보기, 무릅씀 다스리기, 서비스에 올리는 꾀를 아우르는 무릅씀 다루기 설계 결을 다룬다.

## 1. 코드

```python
"""
35.4.2장: 내림폭 다스리기
==================================
자리 크기 잣대기, 회로 차단기, 매인 방침 가장 좋게 하기를
곁들인, 내림폭을 살피는 힘 북돋우는 배움.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
# 무작위로 뽑는 값이 아래에 나온다. 씨앗을 고정해야 이 쪽에 실린
# 수가 다시 나온다 — 고정하지 않으면 돌릴 때마다 다른 수가 찍힌다
torch.manual_seed(0)

# ========================================================================
# 메인
# ========================================================================


@dataclass
class DrawdownConfig:
    max_drawdown: float = 0.10
    warning_threshold: float = 0.05
    circuit_breaker: float = 0.15
    recovery_threshold: float = 0.03
    drawdown_penalty: float = 2.0
    position_scaling: bool = True


class DrawdownTracker:
    """내림폭 재기를 실시간으로 좇는다."""

    def __init__(self):
        self.peak_value = 1.0
        self.current_value = 1.0
        self.max_drawdown = 0.0
        self.current_dd_duration = 0
        self.max_dd_duration = 0
        self.step = 0
        self.history: List[float] = []

    def reset(self, initial_value: float = 1.0):
        self.peak_value = initial_value
        self.current_value = initial_value
        self.max_drawdown = 0.0
        self.current_dd_duration = 0
        self.max_dd_duration = 0
        self.step = 0
        self.history = []

    def update(self, portfolio_value: float) -> Dict[str, float]:
        self.current_value = portfolio_value
        self.step += 1

        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value
            self.current_dd_duration = 0
        else:
            self.current_dd_duration += 1

        drawdown = (self.peak_value - portfolio_value) / (self.peak_value + 1e-8)
        self.max_drawdown = max(self.max_drawdown, drawdown)
        self.max_dd_duration = max(self.max_dd_duration, self.current_dd_duration)
        self.history.append(drawdown)

        return {
            "drawdown": float(drawdown),
            "max_drawdown": float(self.max_drawdown),
            "dd_duration": self.current_dd_duration,
            "max_dd_duration": self.max_dd_duration,
            "recovery_ratio": float(portfolio_value / (self.peak_value + 1e-8)),
        }


class DrawdownPositionScaler:
    """지금 내림폭에 따라 자리 크기를 잣댄다."""

    def __init__(self, config: DrawdownConfig):
        self.config = config

    def compute_scale(self, drawdown: float) -> float:
        if drawdown <= self.config.warning_threshold:
            return 1.0
        elif drawdown >= self.config.max_drawdown:
            return 0.0
        else:
            range_ = self.config.max_drawdown - self.config.warning_threshold
            excess = drawdown - self.config.warning_threshold
            return max(0.0, 1.0 - excess / (range_ + 1e-8))

    def scale_weights(self, weights: np.ndarray, drawdown: float) -> np.ndarray:
        return weights * self.compute_scale(drawdown)


class CircuitBreaker:
    """내림폭이 위끝을 넘으면 굳게 멈춘다."""

    def __init__(self, config: DrawdownConfig):
        self.config = config
        self.triggered = False

    def reset(self):
        self.triggered = False

    def check(self, drawdown: float) -> bool:
        if drawdown >= self.config.circuit_breaker:
            self.triggered = True
        if self.triggered and drawdown <= self.config.recovery_threshold:
            self.triggered = False
        return self.triggered


class DrawdownConstrainedPolicy(nn.Module):
    """내림폭 상태를 덧붙인 방침 그물."""

    def __init__(self, base_state_dim: int, num_assets: int, hidden_dim: int = 128):
        super().__init__()
        # 덧붙인 상태: 바탕 + (내림폭, 내림폭 이어진 때, 되찾음 비)
        augmented_dim = base_state_dim + 3

        self.encoder = nn.Sequential(
            nn.Linear(augmented_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, num_assets)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor, dd_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        augmented = torch.cat([state, dd_state], dim=-1)
        features = self.encoder(augmented)
        weights = F.softmax(self.policy_head(features), dim=-1)
        value = self.value_head(features).squeeze(-1)
        return {"weights": weights, "value": value}


class DrawdownRewardWrapper:
    """바탕 보상에 내림폭 벌을 더하는 감싸개."""

    def __init__(self, config: DrawdownConfig):
        self.config = config
        self.tracker = DrawdownTracker()
        self.scaler = DrawdownPositionScaler(config)
        self.breaker = CircuitBreaker(config)

    def reset(self, initial_value: float = 1.0):
        self.tracker.reset(initial_value)
        self.breaker.reset()

    def compute(self, base_reward: float, portfolio_value: float) -> Tuple[float, Dict]:
        dd_info = self.tracker.update(portfolio_value)
        dd = dd_info["drawdown"]

        # 문턱을 넘어선 몫에 제곱 벌
        penalty = 0.0
        if dd > self.config.warning_threshold:
            excess = dd - self.config.warning_threshold
            penalty = self.config.drawdown_penalty * excess ** 2

        adjusted = base_reward - penalty
        circuit = self.breaker.check(dd)

        info = {
            **dd_info,
            "penalty": penalty,
            "circuit_breaker": circuit,
            "position_scale": self.scaler.compute_scale(dd),
        }
        return float(adjusted), info


def demo_drawdown_control():
    """내림폭 다스리기 장치를 보인다."""
    print("=" * 70)
    print("Drawdown Control Demonstration")
    print("=" * 70)

    config = DrawdownConfig(
        max_drawdown=0.10, warning_threshold=0.05,
        circuit_breaker=0.15, drawdown_penalty=2.0,
    )

    # 내림폭 사건이 있는 밑천을 흉내 낸다
    np.random.seed(42)
    T = 200
    returns = np.random.randn(T) * 0.01 + 0.0003
    returns[70:90] = np.random.randn(20) * 0.015 - 0.008  # 내림폭
    returns[140:155] = np.random.randn(15) * 0.02 - 0.012  # 심한 내림폭

    wrapper = DrawdownRewardWrapper(config)
    wrapper.reset(1.0)

    portfolio_value = 1.0
    print(f"\n{'Step':>5} {'Value':>10} {'DD%':>8} {'Scale':>8} {'CB':>4} {'Penalty':>10}")
    print("-" * 50)

    for t in range(T):
        portfolio_value *= (1 + returns[t])
        adj_reward, info = wrapper.compute(returns[t], portfolio_value)

        if t % 20 == 0 or info["circuit_breaker"] or info["drawdown"] > 0.05:
            print(f"{t:>5} {portfolio_value:>9.4f} "
                  f"{info['drawdown']*100:>7.2f}% "
                  f"{info['position_scale']:>7.3f} "
                  f"{'Y' if info['circuit_breaker'] else 'N':>3} "
                  f"{info['penalty']:>9.6f}")

    print(f"\nMax drawdown: {wrapper.tracker.max_drawdown*100:.2f}%")
    print(f"Max DD duration: {wrapper.tracker.max_dd_duration} steps")

    # 자리 크기 잣대기 보여 주기
    print("\n--- Position Scaling ---")
    scaler = DrawdownPositionScaler(config)
    for dd in [0.0, 0.03, 0.05, 0.07, 0.08, 0.10, 0.12, 0.15]:
        scale = scaler.compute_scale(dd)
        print(f"  DD={dd*100:5.1f}% -> scale={scale:.3f}")

    # 방침 그물
    print("\n--- Drawdown-Constrained Policy ---")
    policy = DrawdownConstrainedPolicy(base_state_dim=20, num_assets=5)
    params = sum(p.numel() for p in policy.parameters())
    print(f"Parameters: {params:,}")

    state = torch.randn(1, 20)
    dd_state = torch.FloatTensor([[0.03, 5.0, 0.97]])
    with torch.no_grad():
        out = policy(state, dd_state)
    print(f"Weights: {out['weights'][0].numpy()}")
    print(f"Value: {out['value'].item():.4f}")


if __name__ == "__main__":
    demo_drawdown_control()
```

??? note "전체 출력 (209줄)"

    ```
    ======================================================================
    Drawdown Control Demonstration
    ======================================================================

     Step      Value      DD%    Scale   CB    Penalty
    --------------------------------------------------
        0    1.0053    0.00%   1.000   N  0.000000
       16    0.9902    5.57%   0.886   N  0.000065
       17    0.9936    5.25%   0.951   N  0.000012
       18    0.9849    6.08%   0.785   N  0.000232
       19    0.9713    7.38%   0.525   N  0.001129
       20    0.9858    5.99%   0.802   N  0.000196
       21    0.9839    6.17%   0.765   N  0.000276
       22    0.9848    6.08%   0.783   N  0.000234
       23    0.9711    7.39%   0.521   N  0.001145
       24    0.9661    7.87%   0.426   N  0.001646
       25    0.9675    7.74%   0.452   N  0.001501
       26    0.9566    8.77%   0.245   N  0.002848
       27    0.9605    8.40%   0.319   N  0.002317
       28    0.9550    8.93%   0.215   N  0.003083
       29    0.9525    9.16%   0.167   N  0.003468
       30    0.9471    9.68%   0.063   N  0.004387
       31    0.9649    7.98%   0.403   N  0.001780
       32    0.9650    7.97%   0.406   N  0.001762
       33    0.9551    8.91%   0.217   N  0.003064
       34    0.9633    8.14%   0.372   N  0.001969
       35    0.9518    9.23%   0.154   N  0.003581
       36    0.9541    9.01%   0.197   N  0.003224
       37    0.9357   10.77%   0.000   N  0.006660
       38    0.9235   11.93%   0.000   N  0.009602
       39    0.9256   11.73%   0.000   N  0.009056
       40    0.9327   11.05%   0.000   N  0.007323
       41    0.9346   10.87%   0.000   N  0.006895
       42    0.9338   10.95%   0.000   N  0.007076
       43    0.9313   11.19%   0.000   N  0.007662
       44    0.9178   12.48%   0.000   N  0.011178
       45    0.9114   13.08%   0.000   N  0.013056
       46    0.9075   13.45%   0.000   N  0.014294
       47    0.9174   12.51%   0.000   N  0.011289
       48    0.9208   12.19%   0.000   N  0.010329
       49    0.9049   13.71%   0.000   N  0.015166
       50    0.9081   13.40%   0.000   N  0.014121
       51    0.9048   13.71%   0.000   N  0.015173
       52    0.8990   14.27%   0.000   N  0.017180
       53    0.9048   13.72%   0.000   N  0.015201
       54    0.9144   12.80%   0.000   N  0.012177
       55    0.9231   11.96%   0.000   N  0.009701
       56    0.9157   12.68%   0.000   N  0.011787
       57    0.9131   12.92%   0.000   N  0.012548
       58    0.9164   12.61%   0.000   N  0.011571
       59    0.9256   11.73%   0.000   N  0.009051
       60    0.9215   12.12%   0.000   N  0.010150
       61    0.9200   12.26%   0.000   N  0.010543
       62    0.9101   13.21%   0.000   N  0.013464
       63    0.8995   14.22%   0.000   N  0.016991
       64    0.9071   13.49%   0.000   N  0.014431
       65    0.9197   12.30%   0.000   N  0.010644
       66    0.9193   12.33%   0.000   N  0.010752
       67    0.9288   11.43%   0.000   N  0.008259
       68    0.9324   11.08%   0.000   N  0.007391
       69    0.9267   11.63%   0.000   N  0.008781
       70    0.9243   11.86%   0.000   N  0.009409
       71    0.9246   11.82%   0.000   N  0.009310
       72    0.9323   11.10%   0.000   N  0.007431
       73    0.9395   10.40%   0.000   N  0.005835
       74    0.9126   12.97%   0.000   N  0.012703
       75    0.8925   14.89%   0.000   N  0.019564
       76    0.8922   14.91%   0.000   N  0.019656
       77    0.8920   14.94%   0.000   N  0.019755
       78    0.8917   14.96%   0.000   N  0.019848
       79    0.9361   10.73%   0.000   N  0.006562
       80    0.9366   10.68%   0.000   N  0.006447
       81    0.9451    9.87%   0.026   N  0.004745
       82    0.9511    9.30%   0.140   N  0.003701
       83    0.9527    9.14%   0.172   N  0.003430
       84    0.9406   10.30%   0.000   N  0.005613
       85    0.9438    9.99%   0.001   N  0.004988
       86    0.9253   11.76%   0.000   N  0.009133
       87    0.9146   12.78%   0.000   N  0.012096
       88    0.9006   14.11%   0.000   N  0.016598
       89    0.8945   14.69%   0.000   N  0.018785
       90    0.8957   14.58%   0.000   N  0.018367
       91    0.9046   13.73%   0.000   N  0.015243
       92    0.8986   14.31%   0.000   N  0.017335
       93    0.8959   14.56%   0.000   N  0.018298
       94    0.8926   14.87%   0.000   N  0.019500
       95    0.8798   16.09%   0.000   Y  0.024618
       96    0.8827   15.82%   0.000   Y  0.023419
       97    0.8853   15.58%   0.000   Y  0.022370
       98    0.8856   15.55%   0.000   Y  0.022245
       99    0.8838   15.72%   0.000   Y  0.022980
      100    0.8715   16.89%   0.000   Y  0.028259
      101    0.8681   17.21%   0.000   Y  0.029823
      102    0.8654   17.47%   0.000   Y  0.031101
      103    0.8587   18.11%   0.000   Y  0.034362
      104    0.8576   18.22%   0.000   Y  0.034928
      105    0.8613   17.86%   0.000   Y  0.033077
      106    0.8778   16.29%   0.000   Y  0.025476
      107    0.8796   16.11%   0.000   Y  0.024708
      108    0.8822   15.87%   0.000   Y  0.023647
      109    0.8818   15.91%   0.000   Y  0.023810
      110    0.8651   17.50%   0.000   Y  0.031247
      111    0.8651   17.50%   0.000   Y  0.031232
      112    0.8659   17.42%   0.000   Y  0.030861
      113    0.8875   15.36%   0.000   Y  0.021479
      114    0.8861   15.50%   0.000   Y  0.022052
      115    0.8890   15.22%   0.000   Y  0.020891
      116    0.8890   15.22%   0.000   Y  0.020908
      117    0.8788   16.19%   0.000   Y  0.025042
      118    0.8891   15.21%   0.000   Y  0.020836
      119    0.8961   14.54%   0.000   Y  0.018217
      120    0.9035   13.84%   0.000   Y  0.015637
      121    0.8955   14.60%   0.000   Y  0.018431
      122    0.9083   13.38%   0.000   Y  0.014032
      123    0.8959   14.56%   0.000   Y  0.018296
      124    0.9014   14.04%   0.000   Y  0.016335
      125    0.9214   12.13%   0.000   Y  0.010164
      126    0.9126   12.97%   0.000   Y  0.012713
      127    0.9077   13.44%   0.000   Y  0.014245
      128    0.9089   13.33%   0.000   Y  0.013869
      129    0.9046   13.74%   0.000   Y  0.015269
      130    0.8908   15.05%   0.000   Y  0.020198
      131    0.8917   14.97%   0.000   Y  0.019863
      132    0.8825   15.84%   0.000   Y  0.023516
      133    0.8869   15.42%   0.000   Y  0.021714
      134    0.8790   16.17%   0.000   Y  0.024963
      135    0.8929   14.85%   0.000   Y  0.019395
      136    0.8862   15.49%   0.000   Y  0.022004
      137    0.8836   15.74%   0.000   Y  0.023051
      138    0.8911   15.03%   0.000   Y  0.020100
      139    0.8804   16.05%   0.000   Y  0.024400
      140    0.9105   13.17%   0.000   Y  0.013338
      141    0.8656   17.45%   0.000   Y  0.031007
      142    0.8671   17.31%   0.000   Y  0.030301
      143    0.8287   20.97%   0.000   Y  0.050997
      144    0.8110   22.66%   0.000   Y  0.062393
      145    0.8189   21.91%   0.000   Y  0.057165
      146    0.8101   22.74%   0.000   Y  0.062963
      147    0.7829   25.34%   0.000   Y  0.082705
      148    0.7623   27.30%   0.000   Y  0.099454
      149    0.7636   27.18%   0.000   Y  0.098424
      150    0.7432   29.12%   0.000   Y  0.116367
      151    0.7375   29.66%   0.000   Y  0.121672
      152    0.7294   30.44%   0.000   Y  0.129488
      153    0.7111   32.19%   0.000   Y  0.147815
      154    0.7331   30.09%   0.000   Y  0.125921
      155    0.7280   30.57%   0.000   Y  0.130768
      156    0.7418   29.25%   0.000   Y  0.117652
      157    0.7456   28.90%   0.000   Y  0.114220
      158    0.7369   29.72%   0.000   Y  0.122249
      159    0.7420   29.24%   0.000   Y  0.117524
      160    0.7350   29.91%   0.000   Y  0.124095
      161    0.7410   29.34%   0.000   Y  0.118454
      162    0.7498   28.50%   0.000   Y  0.110419
      163    0.7439   29.06%   0.000   Y  0.115797
      164    0.7512   28.36%   0.000   Y  0.109114
      165    0.7546   28.04%   0.000   Y  0.106170
      166    0.7610   27.43%   0.000   Y  0.100594
      167    0.7757   26.03%   0.000   Y  0.088441
      168    0.7740   26.19%   0.000   Y  0.089787
      169    0.7684   26.72%   0.000   Y  0.094371
      170    0.7618   27.35%   0.000   Y  0.099923
      171    0.7558   27.92%   0.000   Y  0.105092
      172    0.7554   27.96%   0.000   Y  0.105404
      173    0.7583   27.69%   0.000   Y  0.102963
      174    0.7606   27.47%   0.000   Y  0.100960
      175    0.7671   26.85%   0.000   Y  0.095450
      176    0.7674   26.81%   0.000   Y  0.095175
      177    0.7788   25.73%   0.000   Y  0.085937
      178    0.7770   25.90%   0.000   Y  0.087388
      179    0.7984   23.87%   0.000   Y  0.071180
      180    0.8036   23.37%   0.000   Y  0.067463
      181    0.7969   24.00%   0.000   Y  0.072200
      182    0.7886   24.79%   0.000   Y  0.078337
      183    0.7927   24.41%   0.000   Y  0.075316
      184    0.7912   24.55%   0.000   Y  0.076455
      185    0.7970   23.99%   0.000   Y  0.072128
      186    0.8010   23.61%   0.000   Y  0.069252
      187    0.8007   23.64%   0.000   Y  0.069496
      188    0.7942   24.26%   0.000   Y  0.074224
      189    0.7824   25.39%   0.000   Y  0.083143
      190    0.7791   25.70%   0.000   Y  0.085696
      191    0.7860   25.04%   0.000   Y  0.080330
      192    0.7879   24.86%   0.000   Y  0.078870
      193    0.7784   25.77%   0.000   Y  0.086293
      194    0.7799   25.62%   0.000   Y  0.085045
      195    0.7832   25.31%   0.000   Y  0.082516
      196    0.7765   25.95%   0.000   Y  0.087779
      197    0.7779   25.81%   0.000   Y  0.086642
      198    0.7786   25.75%   0.000   Y  0.086098
      199    0.7699   26.57%   0.000   Y  0.093093

    Max drawdown: 32.19%
    Max DD duration: 190 steps

    --- Position Scaling ---
      DD=  0.0% -> scale=1.000
      DD=  3.0% -> scale=1.000
      DD=  5.0% -> scale=1.000
      DD=  7.0% -> scale=0.600
      DD=  8.0% -> scale=0.400
      DD= 10.0% -> scale=0.000
      DD= 12.0% -> scale=0.000
      DD= 15.0% -> scale=0.000

    --- Drawdown-Constrained Policy ---
    Parameters: 20,614
    Weights: [0.1679834  0.21150012 0.2279685  0.20135412 0.19119391]
    Value: -0.0866
    ```


## 2. 논의

이 짜기는 갈래 여섯(`DrawdownConfig`, `DrawdownTracker`, `DrawdownPositionScaler`, `CircuitBreaker`과 둘 더)을 두어 온전한 무릅씀 다루기 구조를 함께 이룬다. 갈래마다 남다른 조각을 감싸므로 코드가 조각조각 나뉘어 늘리기 쉽다. `forward` 방법이 파이토치가 저절로 미분할 때 쓰는 셈 그래프를 정한다.

여기서 보인 결은 더 얽힌 자리로 자연스럽게 넓어진다. 매개변수와 구조 갈래와 자료 뭉치를 바꿔 가며 해 보면 이해가 깊어지고 계량 금융 일감에 대한 실제 감이 쌓인다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
기본 첫 값으로 만든 `DrawdownConfig`에서 배우는 매개변수의 온 개수를 셈하여라. 무게와 치우침을 모두 넣어 켜마다 나누어 적어라.

</div>

??? success "연습문제 1 풀이"
    `nn.Linear(in_features, out_features)`마다 무게 매개변수가 `in_features * out_features`개, 치우침 매개변수가 `out_features`개다(`bias=False`가 아니면). `nn.Conv2d(in_c, out_c, k)`마다 무게가 `in_c * out_c * k * k`개, 치우침이 `out_c`개다. `nn.Embedding(num, dim)`은 `num * dim`개다. 온 켜에 걸쳐 더한다. `sum(p.numel() for p in model.parameters())`으로 살펴볼 수 있다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
으뜸 함수나 갈래에 들임이 바라는 꼴과 자료형인지 살피는 검사를 더하라. 옳지 않은 들임에는 알기 쉬운 잘못 알림을 내어라.

</div>

??? success "연습문제 2 풀이"
    `forward` 방법(또는 걸맞은 함수)의 첫머리에 `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`이나 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'` 같은 살피기를 더한다. 꼴을 살필 때에는 종요로운 차원을 짚는다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 알기 쉬운 잘못 알림은 벌레잡기를 크게 빠르게 하고 코드를 되쓰기 좋게 만든다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
이 짜기가 어그러질 수 있는 결 둘을 밝히고, 저마다 어떻게 짚어 내고 고칠지 풀어라.

</div>

??? success "연습문제 3 풀이"
    흔한 어그러짐은 이렇다. (1) **기울기가 사라지거나 터짐** -- 기울기 노름을 지켜보아(`torch.nn.utils.clip_grad_norm_`이나 켜마다 `param.grad.norm()` 적기) 짚어 낸다. 기울기 자르기, 더 나은 첫 값 매기기(자비에/카이밍), 구조 바꾸기(남는 이음, 고르게 하기)로 고친다. (2) **지나치게 맞추기** -- 익힘 손실은 줄어드는데 살피기 손실이 늘면 짚어 낸다. 정칙화(드롭아웃, 무게 삭임, 자료 늘리기)나 모형 그릇 줄이기로 고친다. 익힘 재기와 살피기 재기를 늘 함께 지켜보아 이런 걸림돌을 일찍 잡아야 한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
`DrawdownConfig`을 켜나 덩이의 개수를 마음대로 잡을 수 있게 넓혀라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`으로 깊이가 들쭉날쭉한 구조를 만들어라. 켜 2, 4, 8개로 시험해 보라.

</div>

??? success "연습문제 4 풀이"
    못 박은 켜를 다음으로 갈음한다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 방법에서 되돌이한다. `for layer in self.layers: x = layer(x)`. (그냥 파이썬 목록이 아니라) `nn.ModuleList`을 쓰면 파이토치가 온 매개변수를 가장 좋게 하기에 등록한다. 시험: `for n in [2, 4, 8]: model = DrawdownConfig(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.

## 정리하며

**다룬 것** — 내림폭

이 짜기는 갈래 여섯(`DrawdownConfig`, `DrawdownTracker`, `DrawdownPositionScaler`, `CircuitBreaker`과 둘 더)을 두어 온전한 무릅씀 다루기 구조를 함께 이룬다.

핵심 갈래는 `DrawdownConfig`, `DrawdownTracker`, `DrawdownPositionScaler`, `CircuitBreaker`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
