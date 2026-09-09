# 힘 북돋우는 배움 벌레잡기

힘 북돋우는 배움 벌레잡기는 실제 힘 북돋우는 배움 재주에서 종요로운 생각이다. 이 구현은 여기에 걸린 종요로운 알고리즘과 자료 얼개를 손으로 만져 보이며, 이치 바탕과 실제로 서비스에 올릴 때 살필 것을 함께 보여 준다.

## 1. 코드

```python
"""
34.6.4장: 힘 북돋우는 배움 벌레잡기
==================================================
짚어 내는 도구, 건강 살피기, 지켜보기 도구.
"""

import torch
import torch.nn as nn
import numpy as np
from collections import deque
from typing import Dict, List, Optional

# ========================================================================
# 메인
# ========================================================================


class RLDiagnostics:
    """
    힘 북돋우는 배움 익힘을 두루 짚어 내는 것.
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
        
        # 기울기 노름 살피기
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
        
        # 배움이 나아가는지
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
    """온 켜를 지나는 기울기 흐름을 살핀다."""
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
    """방침 기울기를 위해 손실을 옳게 지었는지 살핀다."""
    print("=" * 60)
    print("Loss Sign Verification")
    print("=" * 60)
    
    # 방침 기울기: E[log_prob * advantage]을 가장 크게 한다
    # 가장 작게 할 손실로는: L = -E[log_prob * advantage]
    
    log_prob = torch.tensor(-1.5, requires_grad=True)
    advantage = torch.tensor(2.0)  # 0보다 큼: 좋은 움직임
    
    # 옳음: 음의 부호(음수를 가장 작게 한다)
    correct_loss = -(log_prob * advantage)
    correct_loss.backward()
    print(f"Positive advantage (A={advantage.item()}):")
    print(f"  Loss = {correct_loss.item():.4f}")
    print(f"  Gradient = {log_prob.grad.item():.4f}")
    print(f"  Direction: {'Increase' if log_prob.grad.item() < 0 else 'Decrease'} log_prob ✓")
    
    log_prob2 = torch.tensor(-1.5, requires_grad=True)
    advantage2 = torch.tensor(-1.0)  # 0보다 작음: 나쁜 움직임
    
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

## 2. 논의

이 구현은 힘 북돋우는 배움 벌레잡기의 한가운데 논리를 담은 `RLDiagnostics` 클래스를 축으로 삼는다. 코드는 알고리즘 조각을 보여 주기와 따지기 논리에서 갈라놓는 조각 설계를 따른다.

보여 주기 함수는 이 조각들을 종요로운 거동이 드러나는 지어낸 자료에 실제로 써 보인다. 그 출력을 살피면 매개변수 고름과 문제 얼개에 따라 알고리즘의 됨됨이가 어떻게 달라지는지 볼 수 있다.

쓰임의 눈으로 보면 이 구현은 날 성능보다 또렷함을 앞세운다. 서비스 시스템은 묶음 셈하기, GPU 빠르게 하기, 더 야무진 매개변수 벼리기 같은 다듬기를 더 넣는 것이 보통이다. 그렇더라도 여기서 보인 한가운데 알고리즘 생각은 큰 잣대의 쓰임새에 그대로 옮겨 간다.

## 연습문제

**연습문제 1.**
보여 주기 코드를 돌리고 종요로운 출력 재기를 적어라. 매개변수 하나(배움률, 숨은 차원, 켜 개수 따위)를 고쳐 열매가 어떻게 달라지는지 밝혀라.

??? success "연습문제 1 풀이"
    보여 주기를 돌린 뒤 다른 것을 붙박아 두고 고른 매개변수만 짜임 있게 바꾼다. 보기로 숨은 차원을 곱절로 늘리면 나타내는 그릇이 커지지만 셈하는 때가 는다. 배움률은 한결같지 않은 결과를 낳는다. 너무 작으면 더디게 모이고 너무 크면 들쭉날쭉해진다. 고른 매개변수의 서로 다른 값 적어도 셋에 대해 또렷한 수를 적어 두라.

---

**연습문제 2.**
이 구현에서 종요로운 얼개 고름이 맡은 몫을 풀어라. 왜 그런 활성 함수, 고르게 하기 꾀, 손실 함수를 쓰는가? 다른 것으로 바꾸면 무슨 일이 생기는가?

??? success "연습문제 2 풀이"
    이 얼개 고름은 실제 힘 북돋우는 배움 재주에서 자리 잡은 좋은 버릇을 비춘다. 보기로 ReLU 활성은 곧지 않음을 주면서 0보다 큰 들임에서 기울기가 사라지는 것을 막는다. 손실 함수는 일감 갈래에 맞추어 고른다(갈래 나누기에는 사귐 엔트로피, 되돌이에는 평균 제곱 잘못). 다른 것으로 바꾸면(보기로 시그모이드 활성, L1 손실) 가장 좋게 하기 지형이 바뀌어 됨됨이가 나빠질 수 있으나, 어떤 자리에서는 바꾸는 것이 이로울 수도 있다.

---

**연습문제 3.**
이 구현을 더 만만치 않은 자리로 넓혀라. 더 큰 자료 뭉치, 다른 문제 갈래, 덧붙인 기능 가운데 하나를 고르라. 고친 바를 밝히고 됨됨이에 미친 바를 따져라.

??? success "연습문제 3 풀이"
    절로 떠오르는 넓히기 하나는 정칙화(드롭아웃, 무게 삭임)나 더 야무진 얼개(켜 더하기, 건너뛰는 이음)를 더하는 것이다. 고른 넓히기를 만들고 같은 자료로 익힌 뒤 앞뒤의 재기를 견주어라. 이 넓히기는 처음 알고리즘과 고친 바의 이치 밑뜻을 모두 아는 것을 보여야 한다.

## 정리하며

**다룬 것** — 힘 북돋우는 배움 벌레잡기

이 구현은 힘 북돋우는 배움 벌레잡기의 한가운데 논리를 담은 `RLDiagnostics` 클래스를 축으로 삼는다.

고갱이 갈래는 `RLDiagnostics`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
