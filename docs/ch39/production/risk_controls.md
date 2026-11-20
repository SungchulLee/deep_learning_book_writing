# 위험 통제

35.7.4장: 위험 관리자. 킬 스위치와 포지션 상한을 갖춘 서비스 위험 관리.

계량 금융에 딥러닝을 배포하려면 탄탄한 서비스 기반이 있어야 한다. 이 꾸러미는 모니터링, 위험 관리, 금융 응용을 서비스에 배포하는 전략을 아우르는 서비스 구조 설계 패턴을 다룬다.

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
    """거래 구조를 위한 다급할 때의 죽임 스위치."""

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
    """온전히 갖춘 서비스 무릅씀 다루기 구조."""

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
            print(f"  스텝 {step}: [{action.value}] "
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

**출력:**

```
======================================================================
서비스 무릅씀 다스리개 보이기
======================================================================

--- 무릅씀 다스리개를 얹은 흉내내기 ---
  스텝 2: [halt] 갈아치움 위끝: 2.4곱절 vol_scale=N/A
  스텝 3: [halt] 갈아치움 위끝: 3.3곱절 vol_scale=N/A
  스텝 4: [halt] 갈아치움 위끝: 4.4곱절 vol_scale=N/A
  스텝 5: [halt] 갈아치움 위끝: 5.6곱절 vol_scale=N/A
  스텝 6: [halt] 갈아치움 위끝: 6.8곱절 vol_scale=N/A
  스텝 7: [halt] 갈아치움 위끝: 7.8곱절 vol_scale=N/A
  스텝 8: [halt] 갈아치움 위끝: 9.0곱절 vol_scale=N/A
  스텝 9: [halt] 갈아치움 위끝: 10.2곱절 vol_scale=N/A
  스텝 10: [halt] 갈아치움 위끝: 11.5곱절 vol_scale=N/A
  스텝 11: [halt] 갈아치움 위끝: 12.6곱절 vol_scale=N/A
  스텝 12: [halt] 갈아치움 위끝: 13.7곱절 vol_scale=N/A
  스텝 13: [halt] 갈아치움 위끝: 14.8곱절 vol_scale=N/A
  스텝 14: [halt] 갈아치움 위끝: 15.4곱절 vol_scale=N/A
  스텝 15: [halt] 갈아치움 위끝: 16.3곱절 vol_scale=N/A
  스텝 16: [halt] 갈아치움 위끝: 16.9곱절 vol_scale=N/A
  스텝 17: [halt] 갈아치움 위끝: 17.9곱절 vol_scale=N/A
  스텝 18: [halt] 갈아치움 위끝: 18.6곱절 vol_scale=N/A
  스텝 19: [halt] 갈아치움 위끝: 19.6곱절 vol_scale=N/A
  스텝 20: [halt] 갈아치움 위끝: 20.6곱절 vol_scale=N/A
  스텝 21: [halt] 갈아치움 위끝: 21.7곱절 vol_scale=N/A
  스텝 22: [halt] 갈아치움 위끝: 22.6곱절 vol_scale=N/A
  스텝 23: [halt] 갈아치움 위끝: 23.4곱절 vol_scale=N/A
  스텝 24: [flatten] 오늘 잃음 위끝: -3.0% vol_scale=N/A
  스텝 25: [halt] 오늘 잃음 위끝: -3.0% vol_scale=N/A
  스텝 26: [halt] 오늘 잃음 위끝: -3.0% vol_scale=N/A
  스텝 27: [halt] 오늘 잃음 위끝: -3.0% vol_scale=N/A
  스텝 28: [halt] 오늘 잃음 위끝: -3.0% vol_scale=N/A
  스텝 29: [halt] 오늘 잃음 위끝: -3.0% vol_scale=N/A
  스텝 30: [halt] 오늘 잃음 위끝: -3.0% vol_scale=N/A
  스텝 31: [halt] 오늘 잃음 위끝: -3.0% vol_scale=N/A
  스텝 32: [halt] 오늘 잃음 위끝: -3.0% vol_scale=N/A
  스텝 33: [halt] 오늘 잃음 위끝: -3.0% vol_scale=N/A
  스텝 34: [halt] 오늘 잃음 위끝: -3.0% vol_scale=N/A
  스텝 35: [halt] 오늘 잃음 위끝: -3.0% vol_scale=N/A
  스텝 36: [halt] 오늘 잃음 위끝: -3.0% vol_scale=N/A
  스텝 37: [halt] 오늘 잃음 위끝: -3.0% vol_scale=N/A
  스텝 38: [halt] 오늘 잃음 위끝: -3.0% vol_scale=N/A
  스텝 39: [halt] 오늘 잃음 위끝: -3.0% vol_scale=N/A
  스텝 40: [halt] 오늘 잃음 위끝: -3.0% vol_scale=N/A
  스텝 41: [halt] 오늘 잃음 위끝: -3.0% vol_scale=N/A
  스텝 42: [halt] 오늘 잃음 위끝: -3.0% vol_scale=N/A
  스텝 43: [halt] 오늘 잃음 위끝: -3.0% vol_scale=N/A
  스텝 44: [halt] 오늘 잃음 위끝: -3.0% vol_scale=N/A
  스텝 45: [halt] 오늘 잃음 위끝: -3.0% vol_scale=N/A
  스텝 46: [halt] 오늘 잃음 위끝: -3.0% vol_scale=N/A
  스텝 47: [halt] 오늘 잃음 위끝: -3.0% vol_scale=N/A
  스텝 48: [halt] 오늘 잃음 위끝: -3.0% vol_scale=N/A
  스텝 49: [halt] 오늘 잃음 위끝: -3.0% vol_scale=N/A

죽임 스위치가 당겨졌는가: True
까닭: 오늘 잃음 위끝: -3.0%

--- 거래 앞 무릅씀 살피기 ---
  여느 것           : 지나감=True, 걸린 것=없음
                   매임 건 뒤=[0.2 0.2 0.2 0.2 0.2]
  쏠린 것           : 지나감=False, 걸린 것=['자리 위끝: 가장 큼=0.500', '주문 크기: 가장 큼=0.300']
                   매임 건 뒤=[0.25 0.25 0.1  0.05 0.05]
  지렛대 큰 것        : 지나감=False, 걸린 것=['자리 위끝: 가장 큼=0.500', '주문 크기: 가장 큼=0.300']
                   매임 건 뒤=[0.25 0.25 0.25 0.2  0.1 ]
```

## 2. 논의

이 구현은 깔끔하고 읽기 쉬운 PyTorch 코드로 서비스 구조의 핵심 아이디어를 보여 준다. 모듈로 나눈 구조 덕에 부분마다 따로 이해하고 다른 과제나 데이터에 맞추어 고치기 쉽다.

여기서 보인 기법은 더 까다로운 상황으로도 자연스럽게 확장된다. 하이퍼파라미터, 구조의 종류, 여러 데이터를 바꿔 가며 해 보면 이해가 깊어지고 서비스 배포에 대한 감이 생긴다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
코드를 읽고 핵심 설계 선택을 짚어라. 구현에서 고른 것 셋을 들고, 저마다 왜 서비스 구조에 알맞은지 설명하라.

</div>

??? success "풀이"
    설계 선택은 구현마다 다르나 흔히 이런 것이 있다. (1) 활성화 함수 선택 -- ReLU 계열은 그래디언트가 소실되지 않아 학습이 빠르다. (2) 정규화 전략 -- 배치 정규화가 내부 공변량 이동을 줄여 학습을 안정시킨다. (3) 잔차 연결 -- 있으면 스킵 경로를 주어 깊은 네트워크에서 그래디언트가 흐르게 한다. 각 선택은 표현력, 계산 비용, 학습 안정성 사이의 트레이드오프를 드러낸다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
입력의 형태와 데이터 타입이 기대한 대로인지 확인하는 입력 검증을 주요 함수나 클래스에 더하여라. 올바르지 않은 입력에는 알아보기 쉬운 오류 메시지를 띄워라.

</div>

??? success "풀이"
    `forward` 메서드(또는 해당 함수)의 앞부분에 `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`이나 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'` 같은 검사를 더한다. 형태를 검사하려면 중요한 차원을 본다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 알아보기 쉬운 오류 메시지는 디버깅을 크게 앞당기고 코드를 재사용하기 안전하게 한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
이 구현이 실패할 만한 유형 둘을 들고, 저마다 어떻게 탐지하고 고칠지 설명하라.

</div>

??? success "풀이"
    흔한 실패 유형은 이렇다. (1) **그래디언트가 사라지거나 터짐** -- 그래디언트 크기를 모니터링해 탐지한다(`torch.nn.utils.clip_grad_norm_`이나 레이어마다 `param.grad.norm()` 기록). 그래디언트 클리핑, 더 나은 초기화(Xavier/Kaiming), 구조 수정(잔차 연결, 정규화 레이어)으로 고친다. (2) **과적합** -- 학습 로스는 줄어드는데 검증 로스가 오르면 탐지한다. 정칙화(드롭아웃, 웨이트 감쇠, 데이터 증강)나 모델 축소로 고친다. 학습 지표와 검증 지표를 늘 함께 모니터링해 이를 일찍 잡아야 한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
위험 관리자 구현을 검사하는 포괄적인 테스트 함수를 작성하라. 빈 입력, 원소 하나짜리 입력, 아주 큰 입력, 그리고 극단값(0, 아주 큰 수)이 든 입력 같은 가장자리 자리를 시험하여라.

</div>

??? success "풀이"
    경계 조건을 두루 다루는 테스트 함수를 작성한다.
    ```python
    def test_riskaction():
        model = RiskAction(...)
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
    구조가 엔드투엔드 학습을 지원하는지 알려면 그래디언트 흐름을 테스트하는 것이 특히 중요하다.

## 정리하며

**다룬 것** — 위험 관리자

이 구현은 깔끔하고 읽기 쉬운 PyTorch 코드로 서비스 구조의 핵심 아이디어를 보여 준다.

핵심 클래스는 `RiskAction`, `RiskControlConfig`, `PreTradeRiskCheck`, `KillSwitch`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
