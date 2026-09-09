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

## 2. 논의

이 짜기는 갈래 여섯(`DrawdownConfig`, `DrawdownTracker`, `DrawdownPositionScaler`, `CircuitBreaker`과 둘 더)을 두어 온전한 무릅씀 다루기 얼개를 함께 이룬다. 갈래마다 남다른 조각을 감싸므로 코드가 조각조각 나뉘어 늘리기 쉽다. `forward` 방법이 파이토치가 저절로 미분할 때 쓰는 셈 그래프를 정한다.

여기서 보인 결은 더 얽힌 자리로 자연스럽게 넓어진다. 매개변수와 얼개 갈래와 자료 뭉치를 바꿔 가며 해 보면 이해가 깊어지고 계량 금융 일감에 대한 실제 감이 쌓인다.

## 연습문제

**연습문제 1.**
기본 첫 값으로 만든 `DrawdownConfig`에서 배우는 매개변수의 온 개수를 셈하여라. 무게와 치우침을 모두 넣어 켜마다 나누어 적어라.

??? success "연습문제 1 풀이"
    `nn.Linear(in_features, out_features)`마다 무게 매개변수가 `in_features * out_features`개, 치우침 매개변수가 `out_features`개다(`bias=False`가 아니면). `nn.Conv2d(in_c, out_c, k)`마다 무게가 `in_c * out_c * k * k`개, 치우침이 `out_c`개다. `nn.Embedding(num, dim)`은 `num * dim`개다. 온 켜에 걸쳐 더한다. `sum(p.numel() for p in model.parameters())`으로 살펴볼 수 있다.

---

**연습문제 2.**
으뜸 함수나 갈래에 들임이 바라는 꼴과 자료형인지 살피는 검사를 더하라. 옳지 않은 들임에는 알기 쉬운 잘못 알림을 내어라.

??? success "연습문제 2 풀이"
    `forward` 방법(또는 걸맞은 함수)의 첫머리에 `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`이나 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'` 같은 살피기를 더한다. 꼴을 살필 때에는 종요로운 차원을 짚는다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 알기 쉬운 잘못 알림은 벌레잡기를 크게 빠르게 하고 코드를 되쓰기 좋게 만든다.

---

**연습문제 3.**
이 짜기가 어그러질 수 있는 결 둘을 밝히고, 저마다 어떻게 짚어 내고 고칠지 풀어라.

??? success "연습문제 3 풀이"
    흔한 어그러짐은 이렇다. (1) **기울기가 사라지거나 터짐** -- 기울기 노름을 지켜보아(`torch.nn.utils.clip_grad_norm_`이나 켜마다 `param.grad.norm()` 적기) 짚어 낸다. 기울기 자르기, 더 나은 첫 값 매기기(자비에/카이밍), 얼개 바꾸기(남는 이음, 고르게 하기)로 고친다. (2) **지나치게 맞추기** -- 익힘 손실은 줄어드는데 살피기 손실이 늘면 짚어 낸다. 정칙화(드롭아웃, 무게 삭임, 자료 늘리기)나 모형 그릇 줄이기로 고친다. 익힘 재기와 살피기 재기를 늘 함께 지켜보아 이런 걸림돌을 일찍 잡아야 한다.

---

**연습문제 4.**
`DrawdownConfig`을 켜나 덩이의 개수를 마음대로 잡을 수 있게 넓혀라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`으로 깊이가 들쭉날쭉한 얼개를 만들어라. 켜 2, 4, 8개로 시험해 보라.

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

이 짜기는 갈래 여섯(`DrawdownConfig`, `DrawdownTracker`, `DrawdownPositionScaler`, `CircuitBreaker`과 둘 더)을 두어 온전한 무릅씀 다루기 얼개를 함께 이룬다.

고갱이 갈래는 `DrawdownConfig`, `DrawdownTracker`, `DrawdownPositionScaler`, `CircuitBreaker`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
