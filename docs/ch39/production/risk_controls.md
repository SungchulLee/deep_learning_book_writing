# 위험 통제

이 쪽은 킬 스위치와 포지션 상한을 갖춘, 실제 운용 수준의 위험 관리자를 다룬다. 킬 스위치는 손실이 한도를 넘으면 거래를 자동으로 멈추는 비상 장치다.

계량 금융에 딥러닝을 배포하려면 탄탄한 운용 기반이 있어야 한다. 이 절은 모니터링, 위험 관리, 금융 응용을 실제 운용 환경에 배포하는 전략을 아우르는 시스템 설계 패턴을 다룬다.

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

이 구현은 NumPy만으로 위험 관리자를 세 겹으로 짠다. `PreTradeRiskCheck`는 주문을 내기 전에 종목당 비중 상한, 레버리지 상한, 한 번의 주문 크기 상한(잘못 입력한 큰 주문, 이른바 팻 핑거를 막는다)을 검사하고, `enforce`로 비중을 잘라 상한 안에 넣는다. `KillSwitch`는 하루 손실·최대 낙폭·하루 회전율을 누적해 한도를 넘으면 포지션을 모두 청산(FLATTEN)하거나 거래를 멈추고(HALT), 한도의 70%에 이르면 포지션을 줄이게(SCALE_DOWN) 한다. `VolatilityScaler`는 최근 20스텝의 실현 변동성이 목표 변동성(연 10%)에 맞도록 포지션 크기를 0.1~2배 사이에서 조절한다. `ProductionRiskManager`가 이 셋을 킬 스위치 → 변동성 조절 → 거래 전 검사 차례로 잇는다.

출력에서는 두 가지를 읽을 수 있다. 첫째, 데모는 `reset_daily`를 부르지 않으므로 하루 회전율이 스텝을 넘어 쌓여 스텝 2부터 회전율 한도(2.0배)를 넘고, 그 뒤로 내내 HALT가 나온다. 회전율 HALT는 `triggered`를 켜지 않으므로 손익은 계속 쌓이고, 하락 구간에서 누적 손실이 -3.0%가 된 스텝 24에 하루 손실 한도로 킬 스위치가 당겨져 전부 청산된다. 그 뒤로는 손으로 재설정하기 전까지 HALT만 돌려준다. 둘째, 거래 전 검사의 세 번째 보기는 비중 합이 정확히 1.5라서 레버리지 검사(`> 1.5`)에 걸리지 않고, 비중 상한과 주문 크기 상한에만 걸린다. `enforce`가 종목당 0.25로 자른 뒤의 합은 1.05다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
코드를 읽고 핵심 설계 선택을 짚어라. 구현에서 고른 것 셋을 들고, 저마다 왜 실제 운용 위험 관리에 알맞은지 설명하라.

</div>

??? success "풀이"
    (1) 한 번 당겨지면 손으로만 풀리는 킬 스위치 -- `triggered`가 켜지면 `manual_reset` 전까지 HALT만 돌려준다. 시장이 잠깐 회복했다고 모델이 저절로 다시 거래를 시작하지 않게 하여, 사람이 원인을 확인한 뒤에만 재개하도록 한다. (2) 단계별 대응 -- 한도의 70%에서는 포지션을 반으로 줄이고, 한도를 넘으면 청산한다. 갑작스러운 전량 청산보다 먼저 위험을 덜어 내므로 시장 충격과 불필요한 청산을 줄인다. (3) 검사와 강제의 분리 -- `check`는 무엇이 걸렸는지 기록하고 `enforce`는 비중을 상한 안으로 잘라 낸다. 거래는 계속하되 위반 기록은 남기므로, 사후에 모델이 얼마나 자주 한도를 넘보는지 분석할 수 있다.

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
    (1) **하루 누적값을 초기화하지 않음** -- 위 출력처럼 `reset_daily`를 부르지 않으면 하루 회전율과 하루 손익이 날을 넘어 쌓여, 스텝 2부터 회전율 HALT가 끝없이 이어진다. HALT가 하루 넘게 이어지는 것으로 탐지하고, 거래일 경계에서 `reset_daily`를 자동으로 부르도록 고친다. (2) **경계값에서 검사가 빠짐** -- 레버리지 검사가 `>`로 비교하므로 비중 합이 정확히 1.5인 포트폴리오는 걸리지 않는다(출력의 세 번째 보기). 또 회전율 HALT 중에도 `turnover`가 계속 더해져 값이 23배까지 불어난다. 경계값을 넣은 단위 테스트로 탐지하고, 한도의 의미(이하/미만)를 문서로 정한 뒤 비교 연산자를 맞추고, 거래하지 않은 스텝에는 회전율을 0으로 넘기도록 고친다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
위험 관리자 구현을 검사하는 포괄적인 테스트 함수를 작성하라. 빈 입력, 원소 하나짜리 입력, 아주 큰 입력, 그리고 극단값(0, 아주 큰 수)이 든 입력 같은 경계 조건을 시험하여라.

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

이 구현은 NumPy만으로 거래 전 검사, 킬 스위치, 변동성 조절을 겹쳐 쌓은 위험 관리자의 뼈대를 보여 준다.

핵심 클래스는 `PreTradeRiskCheck`, `KillSwitch`, `VolatilityScaler`, `ProductionRiskManager`이고 설정과 대응은 `RiskControlConfig`, `RiskAction`으로 나타내며, 앞의 연습문제 4개로 스스로 확인해 볼 수 있다.
