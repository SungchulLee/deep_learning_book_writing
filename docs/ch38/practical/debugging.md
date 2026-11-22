# 강화학습 디버깅

강화학습은 버그가 있어도 오류 없이 돌아가고 리턴이 조금 덜 오를 뿐이라, 무엇이 잘못됐는지 알기 어렵다. 그래서 트레인 도중 엔트로피, 쿨백-라이블러 발산, 클리핑 비율, 그래디언트 노름, 설명 분산 같은 진단 지표를 함께 기록하고 이상 신호를 자동으로 알리는 도구가 필요하다. 이 쪽은 폴리시 경사 로스의 부호를 확인하는 함수와, 그런 진단을 모아 보고하는 `RLDiagnostics` 를 구현해 건강한 트레인과 문제 있는 트레인을 흉내 낸 기록에 적용한다.

## 1. 코드

```python
"""
34.6.4장: 힘 북돋우는 러닝 벌레잡기
==================================================
짚어 내는 도구, 건강 살피기, 지켜보기 도구.
"""

import torch
import torch.nn as nn
import numpy as np
from collections import deque
from typing import Dict, List, Optional
# 무작위로 뽑는 값이 아래에 나온다. 씨앗을 고정해야 이 쪽에 실린
# 수가 다시 나온다 — 고정하지 않으면 돌릴 때마다 다른 수가 찍힌다.
# 꾸며 낸 러닝 자취는 numpy로 뽑으므로 numpy 씨앗도 함께 고정한다
torch.manual_seed(0)
np.random.seed(0)

# ========================================================================
# 메인
# ========================================================================


class RLDiagnostics:
    """
    힘 북돋우는 러닝 익힘을 두루 짚어 내는 것.
    종요로운 건강 재기를 좇고 알린다.
    """
    
    def __init__(self, window=100):
        self.window = window
        self.metrics = {
            "episode_rewards": deque(maxlen=window),
            "episode_lengths": deque(maxlen=window),
            "policy_loss": deque(maxlen=window),
            "value_loss": deque(maxlen=window),
            "entropy": deque(maxlen=window),
            "kl_divergence": deque(maxlen=window),
            "clip_fraction": deque(maxlen=window),
            "grad_norm": deque(maxlen=window),
            "value_predictions": deque(maxlen=window),
            "actual_returns": deque(maxlen=window),
        }
        self.alerts = []
    
    def log(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.metrics:
                self.metrics[key].append(value)
    
    def compute_explained_variance(self):
        """EV = 1 - Var(돌아옴 - 미리 봄) / Var(돌아옴)."""
        if len(self.metrics["value_predictions"]) < 10:
            return 0.0
        preds = np.array(list(self.metrics["value_predictions"]))
        actuals = np.array(list(self.metrics["actual_returns"]))
        var_actual = np.var(actuals)
        if var_actual < 1e-8:
            return 0.0
        return 1.0 - np.var(actuals - preds) / var_actual
    
    def health_check(self) -> List[str]:
        """짚어 내기를 벌이고 미리 알림을 돌려준다."""
        alerts = []
        
        # 엔트로피 살피기
        if len(self.metrics["entropy"]) >= 10:
            recent_entropy = list(self.metrics["entropy"])[-10:]
            if all(e < 0.01 for e in recent_entropy):
                alerts.append("⚠️ ENTROPY COLLAPSE: Policy may be stuck (entropy near 0)")
        
        # 그래디언트 노름 살피기
        if len(self.metrics["grad_norm"]) >= 10:
            recent_grads = list(self.metrics["grad_norm"])[-10:]
            if any(np.isnan(g) or np.isinf(g) for g in recent_grads):
                alerts.append("🔥 NaN/Inf GRADIENTS detected!")
            elif np.mean(recent_grads) > 100:
                alerts.append("⚠️ LARGE GRADIENTS: Consider reducing learning rate")
        
        # 쿨백-라이블러 어긋남 살피기
        if len(self.metrics["kl_divergence"]) >= 10:
            recent_kl = list(self.metrics["kl_divergence"])[-10:]
            if np.mean(recent_kl) > 0.1:
                alerts.append("⚠️ HIGH KL: Policy changing too fast")
        
        # 잘라 낸 조각 살피기
        if len(self.metrics["clip_fraction"]) >= 10:
            recent_cf = list(self.metrics["clip_fraction"])[-10:]
            avg_cf = np.mean(recent_cf)
            if avg_cf > 0.5:
                alerts.append("⚠️ HIGH CLIP FRACTION: Consider reducing learning rate or clip range")
            elif avg_cf < 0.01:
                alerts.append("⚠️ LOW CLIP FRACTION: Updates may be too conservative")
        
        # 값 함수 살피기
        ev = self.compute_explained_variance()
        if ev < 0:
            alerts.append("⚠️ NEGATIVE EXPLAINED VARIANCE: Value function worse than mean prediction")
        
        # 러닝이 나아가는지
        if len(self.metrics["episode_rewards"]) >= self.window:
            rewards = list(self.metrics["episode_rewards"])
            first_half = np.mean(rewards[:len(rewards)//2])
            second_half = np.mean(rewards[len(rewards)//2:])
            if second_half < first_half * 0.9:
                alerts.append("⚠️ PERFORMANCE DEGRADATION: Reward declining")
        
        return alerts
    
    def report(self) -> str:
        """익힘 상태 알림글을 짓는다."""
        lines = ["=" * 50, "RL Training Diagnostics Report", "=" * 50]
        
        for key in ["episode_rewards", "policy_loss", "value_loss", "entropy",
                     "kl_divergence", "clip_fraction", "grad_norm"]:
            vals = list(self.metrics[key])
            if vals:
                lines.append(
                    f"{key:<20}: mean={np.mean(vals):>8.4f}  "
                    f"std={np.std(vals):>8.4f}  "
                    f"last={vals[-1]:>8.4f}"
                )
        
        ev = self.compute_explained_variance()
        lines.append(f"{'explained_variance':<20}: {ev:.4f}")
        
        alerts = self.health_check()
        if alerts:
            lines.append("\n" + "-" * 50)
            lines.append("ALERTS:")
            for alert in alerts:
                lines.append(f"  {alert}")
        else:
            lines.append("\n✅ All diagnostics healthy")
        
        return "\n".join(lines)


def check_gradient_flow(model: nn.Module) -> Dict[str, float]:
    """온 켜를 지나는 그래디언트 흐름을 살핀다."""
    grad_info = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_info[name] = {
                "mean": param.grad.abs().mean().item(),
                "max": param.grad.abs().max().item(),
                "has_nan": torch.isnan(param.grad).any().item(),
                "has_inf": torch.isinf(param.grad).any().item(),
            }
    return grad_info


def verify_loss_sign():
    """폴리시 그래디언트를 위해 로스를 옳게 지었는지 살핀다."""
    print("=" * 60)
    print("Loss Sign Verification")
    print("=" * 60)
    
    # 폴리시 그래디언트: E[log_prob * advantage]을 가장 크게 한다
    # 가장 작게 할 로스로는: L = -E[log_prob * advantage]
    
    log_prob = torch.tensor(-1.5, requires_grad=True)
    advantage = torch.tensor(2.0)  # 0보다 큼: 좋은 액션
    
    # 옳음: 음의 부호(음수를 가장 작게 한다)
    correct_loss = -(log_prob * advantage)
    correct_loss.backward()
    print(f"Positive advantage (A={advantage.item()}):")
    print(f"  Loss = {correct_loss.item():.4f}")
    print(f"  Gradient = {log_prob.grad.item():.4f}")
    print(f"  Direction: {'Increase' if log_prob.grad.item() < 0 else 'Decrease'} log_prob ✓")
    
    log_prob2 = torch.tensor(-1.5, requires_grad=True)
    advantage2 = torch.tensor(-1.0)  # 0보다 작음: 나쁜 액션
    
    correct_loss2 = -(log_prob2 * advantage2)
    correct_loss2.backward()
    print(f"\nNegative advantage (A={advantage2.item()}):")
    print(f"  Loss = {correct_loss2.item():.4f}")
    print(f"  Gradient = {log_prob2.grad.item():.4f}")
    print(f"  Direction: {'Decrease' if log_prob2.grad.item() > 0 else 'Increase'} log_prob ✓")


def demo_diagnostics():
    """익힘을 흉내 내고 짚어 낸 결과를 보인다."""
    print("\n" + "=" * 60)
    print("Training Diagnostics Simulation")
    print("=" * 60)
    
    diag = RLDiagnostics(window=50)
    
    # 건강한 익힘을 흉내 낸다
    for step in range(100):
        reward = 100 + step * 2 + np.random.randn() * 20
        diag.log(
            episode_rewards=reward,
            policy_loss=0.5 - step * 0.003 + np.random.randn() * 0.1,
            value_loss=1.0 - step * 0.005 + np.random.randn() * 0.2,
            entropy=0.7 - step * 0.005,
            kl_divergence=0.015 + np.random.randn() * 0.005,
            clip_fraction=0.15 + np.random.randn() * 0.05,
            grad_norm=0.3 + np.random.randn() * 0.1,
            value_predictions=reward + np.random.randn() * 10,
            actual_returns=reward,
        )
    
    print(diag.report())
    
    # 골칫거리가 있는 익힘을 흉내 낸다
    print("\n\n" + "=" * 50)
    print("Simulating Problematic Training...")
    print("=" * 50)
    
    diag2 = RLDiagnostics(window=20)
    for step in range(50):
        diag2.log(
            episode_rewards=50 - step * 0.5,  # 나빠짐
            entropy=0.001,  # 무너짐
            kl_divergence=0.2,  # 너무 높음
            clip_fraction=0.7,  # 너무 높음
            grad_norm=200 + step * 10,  # 터짐
            value_predictions=np.random.randn() * 100,
            actual_returns=50.0,
        )
    
    print(diag2.report())


if __name__ == "__main__":
    verify_loss_sign()
    demo_diagnostics()
```

**출력:**

```
============================================================
Loss Sign Verification
============================================================
Positive advantage (A=2.0):
  Loss = 3.0000
  Gradient = -2.0000
  Direction: Increase log_prob ✓

Negative advantage (A=-1.0):
  Loss = -1.5000
  Gradient = 1.0000
  Direction: Decrease log_prob ✓

============================================================
Training Diagnostics Simulation
============================================================
==================================================
RL Training Diagnostics Report
==================================================
episode_rewards     : mean=245.2857  std= 35.5063  last=262.4467
policy_loss         : mean=  0.2411  std=  0.1106  last=  0.1497
value_loss          : mean=  0.6024  std=  0.2143  last=  0.7231
entropy             : mean=  0.3275  std=  0.0722  last=  0.2050
kl_divergence       : mean=  0.0144  std=  0.0052  last=  0.0133
clip_fraction       : mean=  0.1495  std=  0.0460  last=  0.1103
grad_norm           : mean=  0.3050  std=  0.0924  last=  0.3198
explained_variance  : 0.9192

✅ All diagnostics healthy


==================================================
Simulating Problematic Training...
==================================================
==================================================
RL Training Diagnostics Report
==================================================
episode_rewards     : mean= 30.2500  std=  2.8831  last= 25.5000
entropy             : mean=  0.0010  std=  0.0000  last=  0.0010
kl_divergence       : mean=  0.2000  std=  0.0000  last=  0.2000
clip_fraction       : mean=  0.7000  std=  0.0000  last=  0.7000
grad_norm           : mean=595.0000  std= 57.6628  last=690.0000
explained_variance  : 0.0000

--------------------------------------------------
ALERTS:
  ⚠️ ENTROPY COLLAPSE: Policy may be stuck (entropy near 0)
  ⚠️ LARGE GRADIENTS: Consider reducing learning rate
  ⚠️ HIGH KL: Policy changing too fast
  ⚠️ HIGH CLIP FRACTION: Consider reducing learning rate or clip range
  ⚠️ PERFORMANCE DEGRADATION: Reward declining
```

## 2. 논의

이 구현은 강화학습 디버깅의 핵심 논리를 담은 `RLDiagnostics` 클래스를 축으로 삼는다. 코드는 진단 도구를 데모 논리와 분리하는 모듈식 설계를 따른다.

로스 부호 확인에서는 어드밴티지가 양수(2.0)일 때 그래디언트가 -2.0000이라 경사 하강이 로그 확률을 올리고, 음수(-1.0)일 때 그래디언트가 1.0000이라 로그 확률을 내린다. 로스에 음의 부호를 붙여야 하는 까닭이 이것이다.

진단 데모는 실제 트레인이 아니라 지표를 흉내 낸 기록이다. 건강한 기록에서는 설명 분산이 0.9192이고 경고가 하나도 없다. 문제 있는 기록에서는 경고가 다섯 개 뜬다. 다만 이 기록에서 설명 분산이 0.0000으로 나오는 것은 밸류 함수가 좋아서가 아니다. 실제 리턴을 상수 50.0으로 두었으므로 분산이 0이라 `compute_explained_variance` 가 계산을 건너뛰고 0을 돌려준 것이고, 그래서 예측이 엉망인데도 음의 설명 분산 경고가 뜨지 않았다. 진단 도구도 그 계산이 정의되지 않는 자리에서는 침묵한다는 것을 기억해 둘 만하다.

실용적 관점에서 이 구현은 순수한 성능보다 명료함을 앞세운다. 프로덕션 시스템은 배치 계산, GPU 가속, 더 정교한 하이퍼파라미터 튜닝 같은 최적화를 더 넣는 것이 보통이다. 그렇더라도 여기서 보인 핵심 알고리즘 아이디어는 대규모 응용에 그대로 옮겨 간다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
데모 코드를 돌리고 중요한 출력 지표를 기록하라. 하이퍼파라미터 하나(진단 창 크기, 경고 문턱값 따위)를 바꿔 결과가 어떻게 달라지는지 설명하라.

</div>

??? success "풀이"
    데모를 돌린 뒤 다른 것은 고정하고 고른 하이퍼파라미터만 체계적으로 바꾼다. 창 크기를 줄이면 경고가 빨리 뜨지만 잡음에도 쉽게 뜬다. 성능 저하 검사는 창이 가득 찬 뒤에야 돌므로, 창을 기록 길이보다 크게 잡으면 아예 돌지 않는다. 쿨백-라이블러 문턱 0.1을 0.01로 낮추면 건강한 기록(평균 0.0144)에서도 경고가 뜬다. 고른 하이퍼파라미터의 서로 다른 값 적어도 셋에 대해 구체적인 수를 기록해 두라.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
이 구현에서 중요한 구조 선택이 맡은 역할을 설명하라. 왜 이런 진단 지표(엔트로피, 쿨백-라이블러 발산, 클리핑 비율, 그래디언트 노름, 설명 분산)를 고르는가? 각 지표가 어떤 고장을 잡는가?

</div>

??? success "풀이"
    엔트로피가 0 근처로 붕괴하면 폴리시가 한 액션에 굳어 탐험을 멈춘 것이다. 쿨백-라이블러 발산이 크면 한 번의 갱신으로 폴리시가 너무 많이 바뀐 것이고, 클리핑 비율이 크면 PPO의 클리핑이 갱신 대부분을 잘라 내고 있다는 뜻이라 둘 다 러닝레이트가 크다는 신호다. 그래디언트 노름이 크거나 NaN이면 수치 폭발이다. 설명 분산은 밸류 함수가 리턴 분산을 얼마나 설명하는지 재며, 음수이면 상수(평균) 예측보다도 못하다. 다만 본문에서 보았듯 리턴의 분산이 0이면 이 지표는 정의되지 않아 0을 돌려준다. 성능 저하 검사는 리워드가 양수라고 가정하므로(`second_half < first_half * 0.9`) 리워드가 음수인 과제에서는 거꾸로 동작한다. $\square$

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
이 구현을 더 어려운 상황으로 확장하라. 실제 트레인, 다른 문제 유형, 새 기능 가운데 하나를 고르라. 수정 사항을 설명하고 성능에 미친 영향을 분석하라.

</div>

??? success "풀이"
    자연스러운 확장 하나는 `RLDiagnostics` 를 실제 PPO 트레인 루프에 붙이고, 일부러 버그(로스 부호 뒤집기, 어드밴티지 정규화 빼기, 러닝레이트 열 배)를 넣어 어느 경고가 뜨는지 보는 것이다. 리턴이 상수일 때 설명 분산이 0으로 침묵하는 문제를 고치는 것도 좋은 확장이다. 고른 확장을 구현하고 같은 둘레와 시드로 트레인한 뒤 전후의 지표를 비교하여라. 이 확장은 원래 알고리즘과 수정 사항의 이론적 근거를 모두 이해했음을 보여야 한다.

## 정리하며

**다룬 것** — 강화학습 디버깅

이 구현은 강화학습 디버깅의 핵심 논리를 담은 `RLDiagnostics` 클래스를 축으로 삼는다.

로스 부호를 확인하는 `verify_loss_sign` 과 함께, 앞의 연습문제 세 개로 이 도구들을 직접 바꿔 보며 확인할 수 있다.
