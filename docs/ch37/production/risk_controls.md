# 무릅씀 다스리개

35.7.4장: 무릅씀 다스리개. 죽임 스위치와 자리 위끝을 갖춘 서비스 무릅씀 다스리기.

계량 금융에 깊은 배움을 올리려면 든든한 서비스 바탕이 있어야 한다. 이 꾸러미는 지켜보기, 무릅씀 다스리기, 금융 쓰임을 서비스에 올리는 꾀를 아우르는 서비스 얼개 설계 결을 다룬다.

## 1. 코드

```python
"""
35.7.4장: 무릅씀 다스리개
================================
죽임 스위치와 자리 위끝을 갖춘 서비스 무릅씀 다스리기.
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

# ========================================================================
# 메인
# ========================================================================


class RiskAction(Enum):
    PASS = "pass"
    SCALE_DOWN = "scale_down"
    FLATTEN = "flatten"
    HALT = "halt"


@dataclass
class RiskControlConfig:
    max_position_per_asset: float = 0.25
    max_leverage: float = 1.5
    max_daily_loss: float = 0.02
    max_drawdown: float = 0.10
    max_daily_turnover: float = 2.0
    max_order_size: float = 0.10
    vol_scaling_threshold: float = 0.02
    vol_target: float = 0.10


class PreTradeRiskCheck:
    """주문을 내기에 앞서 하는 거래 앞 무릅씀 살피기."""

    def __init__(self, config: RiskControlConfig):
        self.config = config

    def check(self, target_weights: np.ndarray, current_weights: np.ndarray,
              portfolio_value: float) -> Dict:
        issues = []

        # 자리 위끝
        max_pos = np.max(np.abs(target_weights))
        if max_pos > self.config.max_position_per_asset:
            issues.append(f"자리 위끝: 가장 큼={max_pos:.3f}")

        # 지렛대
        leverage = np.sum(np.abs(target_weights))
        if leverage > self.config.max_leverage:
            issues.append(f"지렛대 위끝: {leverage:.3f}")

        # 주문 크기(손가락 헛짚기)
        delta = np.abs(target_weights - current_weights)
        max_order = np.max(delta)
        if max_order > self.config.max_order_size:
            issues.append(f"주문 크기: 가장 큼={max_order:.3f}")

        passed = len(issues) == 0
        return {"passed": passed, "issues": issues}

    def enforce(self, target_weights: np.ndarray) -> np.ndarray:
        """매임을 채우도록 몫을 잘라 낸다."""
        w = np.clip(target_weights, -self.config.max_position_per_asset,
                     self.config.max_position_per_asset)
        leverage = np.sum(np.abs(w))
        if leverage > self.config.max_leverage:
            w *= self.config.max_leverage / leverage
        return w


class KillSwitch:
    """거래 얼개를 위한 다급할 때의 죽임 스위치."""

    def __init__(self, config: RiskControlConfig):
        self.config = config
        self.triggered = False
        self.trigger_reason = ""
        self.daily_pnl = 0.0
        self.peak_value = 1.0
        self.current_value = 1.0
        self.daily_turnover = 0.0

    def reset_daily(self):
        self.daily_pnl = 0.0
        self.daily_turnover = 0.0

    def update(self, portfolio_return: float, turnover: float) -> RiskAction:
        if self.triggered:
            return RiskAction.HALT

        self.daily_pnl += portfolio_return
        self.daily_turnover += turnover
        self.current_value *= (1 + portfolio_return)
        self.peak_value = max(self.peak_value, self.current_value)

        drawdown = (self.peak_value - self.current_value) / (self.peak_value + 1e-8)

        # 당김쇠를 살핀다
        if self.daily_pnl < -self.config.max_daily_loss:
            self.triggered = True
            self.trigger_reason = f"오늘 잃음 위끝: {self.daily_pnl*100:.1f}%"
            return RiskAction.FLATTEN

        if drawdown > self.config.max_drawdown:
            self.triggered = True
            self.trigger_reason = f"내림폭 위끝: {drawdown*100:.1f}%"
            return RiskAction.FLATTEN

        if self.daily_turnover > self.config.max_daily_turnover:
            self.trigger_reason = f"갈아치움 위끝: {self.daily_turnover:.1f}곱절"
            return RiskAction.HALT

        # 위끝에 다가가면 줄인다
        if drawdown > self.config.max_drawdown * 0.7:
            return RiskAction.SCALE_DOWN
        if self.daily_pnl < -self.config.max_daily_loss * 0.7:
            return RiskAction.SCALE_DOWN

        return RiskAction.PASS

    def manual_trigger(self, reason: str = "손으로 젖힘"):
        self.triggered = True
        self.trigger_reason = reason

    def manual_reset(self):
        self.triggered = False
        self.trigger_reason = ""


class VolatilityScaler:
    """지금 출렁임에 맞추어 자리를 키우거나 줄인다."""

    def __init__(self, config: RiskControlConfig, lookback: int = 20):
        self.config = config
        self.lookback = lookback
        self.return_buffer: List[float] = []

    def update(self, portfolio_return: float):
        self.return_buffer.append(portfolio_return)
        if len(self.return_buffer) > self.lookback * 2:
            self.return_buffer = self.return_buffer[-self.lookback * 2:]

    def get_scale(self) -> float:
        if len(self.return_buffer) < self.lookback:
            return 1.0
        recent_vol = np.std(self.return_buffer[-self.lookback:]) * np.sqrt(252)
        if recent_vol < 1e-8:
            return 1.0
        scale = self.config.vol_target / recent_vol
        return np.clip(scale, 0.1, 2.0)


class ProductionRiskManager:
    """온전히 갖춘 서비스 무릅씀 다루기 얼개."""

    def __init__(self, config: RiskControlConfig, num_assets: int):
        self.config = config
        self.pre_trade = PreTradeRiskCheck(config)
        self.kill_switch = KillSwitch(config)
        self.vol_scaler = VolatilityScaler(config)
        self.num_assets = num_assets

    def process_action(self, target_weights: np.ndarray,
                       current_weights: np.ndarray,
                       portfolio_return: float = 0.0,
                       turnover: float = 0.0) -> Dict:
        # 무릅씀 상태를 고쳐 쓴다
        self.vol_scaler.update(portfolio_return)
        action = self.kill_switch.update(portfolio_return, turnover)

        if action == RiskAction.HALT:
            return {"weights": current_weights, "action": action,
                    "reason": self.kill_switch.trigger_reason}

        if action == RiskAction.FLATTEN:
            return {"weights": np.zeros(self.num_assets), "action": action,
                    "reason": self.kill_switch.trigger_reason}

        # 출렁임에 맞추어 크기 잡기
        vol_scale = self.vol_scaler.get_scale()
        scaled_weights = target_weights * vol_scale

        # 위끝에 다가가면 줄인다
        if action == RiskAction.SCALE_DOWN:
            scaled_weights *= 0.5

        # 거래 앞 살피기와 매임 걸기
        check = self.pre_trade.check(scaled_weights, current_weights, 1.0)
        final_weights = self.pre_trade.enforce(scaled_weights)

        return {
            "weights": final_weights,
            "action": action,
            "vol_scale": vol_scale,
            "pre_trade_passed": check["passed"],
            "issues": check["issues"],
        }


def demo_risk_controls():
    """서비스 무릅씀 다스리개를 보인다."""
    print("=" * 70)
    print("서비스 무릅씀 다스리개 보이기")
    print("=" * 70)

    config = RiskControlConfig(
        max_position_per_asset=0.25, max_leverage=1.5,
        max_daily_loss=0.02, max_drawdown=0.10,
    )
    N = 5
    rm = ProductionRiskManager(config, N)

    np.random.seed(42)
    weights = np.ones(N) / N

    print("\n--- 무릅씀 다스리개를 얹은 흉내내기 ---")
    for step in range(50):
        target = np.random.dirichlet(np.ones(N)) * 1.2  # 조금 세게 잡는다
        ret = np.random.randn() * 0.01 + 0.0002
        if 20 <= step <= 30:
            ret -= 0.012  # 내림폭

        turnover = np.sum(np.abs(target - weights))
        result = rm.process_action(target, weights, ret, turnover)
        weights = result["weights"]

        action = result["action"]
        if action != RiskAction.PASS:
            print(f"  걸음 {step}: [{action.value}] "
                  f"{result.get('reason', '')} "
                  f"vol_scale={result.get('vol_scale', 'N/A')}")

    print(f"\n죽임 스위치가 당겨졌는가: {rm.kill_switch.triggered}")
    if rm.kill_switch.triggered:
        print(f"까닭: {rm.kill_switch.trigger_reason}")

    # 거래 앞 살피기 보기
    print("\n--- 거래 앞 무릅씀 살피기 ---")
    tests = [
        ("여느 것", np.array([0.2, 0.2, 0.2, 0.2, 0.2])),
        ("쏠린 것", np.array([0.5, 0.3, 0.1, 0.05, 0.05])),
        ("지렛대 큰 것", np.array([0.5, 0.4, 0.3, 0.2, 0.1])),
    ]
    for name, w in tests:
        check = rm.pre_trade.check(w, np.ones(N) / N, 1e6)
        enforced = rm.pre_trade.enforce(w)
        print(f"  {name:<15}: 지나감={check['passed']}, "
              f"걸린 것={check['issues'] or '없음'}")
        print(f"  {'':15}  매임 건 뒤={np.round(enforced, 3)}")


if __name__ == "__main__":
    demo_risk_controls()
```

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
무릅씀 다스리개 짜보기를 살피는 두루 갖춘 시험 함수를 써라. 빈 들임, 원소 하나짜리 들임, 아주 큰 들임, 그리고 끝자락 값(0, 아주 큰 수)이 든 들임 같은 가장자리 자리를 시험하여라.

??? success "연습문제 4 풀이"
    금 언저리 조건을 두루 건드리는 시험 함수를 짓는다.
    ```python
    def test_riskaction():
        model = RiskAction(...)
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

**다룬 것** — 무릅씀 다스리개

이 짜보기는 깔끔하고 읽기 쉬운 PyTorch 코드로 서비스 얼개의 고갱이가 되는 생각을 보여 준다.

고갱이 갈래는 `RiskAction`, `RiskControlConfig`, `PreTradeRiskCheck`, `KillSwitch`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
