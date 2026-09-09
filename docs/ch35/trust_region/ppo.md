# 가까운 곳 방침 가장 좋게 하기(PPO)

가까운 곳 방침 가장 좋게 하기(PPO)는 요즘 힘 북돋우는 배움에서 가장 널리 쓰이는 방침 기울기 알고리즘으로, 믿음 구역 방법의 든든함과 일차 가장 좋게 하기의 쉬움을 함께 얻는다. PPO는 새 방침과 옛 방침의 낌새 비를 잘라 방침 고침을 매어 두어, 쿨백-라이블러 어긋남 매임을 드러내 셈하는 덧듦 없이 무너뜨릴 만큼 큰 걸음을 막는다. 이 구현은 자르기 장치를 보이고, 자르는 문턱이 익힘에 어떻게 미치는지 살피며, 잘라 낸 대리 목표와 자르지 않은 것을 견준다.

## 1. 코드

```python
"""
34.3.3장: PPO -- 잘라 낸 목표 보여 주기
======================================================
PPO의 자르기 장치를 보이고 잘라 낸 대리 목표와 자르지 않은
것을 견준다.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib

# ========================================================================
# 메인
# ========================================================================
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def visualize_ppo_clipping():
    """이점이 0보다 클 때와 작을 때의 PPO 잘라 낸 목표를 그려 본다."""
    epsilon = 0.2
    ratios = torch.linspace(0.5, 2.0, 300)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for idx, (A, title) in enumerate([(1.0, "Positive Advantage (A=1)"),
                                       (-1.0, "Negative Advantage (A=-1)")]):
        ax = axes[idx]
        
        # 자르지 않은 대리 목표
        unclipped = ratios * A
        
        # 잘라 낸 대리 목표
        clipped_ratios = torch.clamp(ratios, 1 - epsilon, 1 + epsilon)
        clipped = clipped_ratios * A
        
        # PPO 목표: min(자르지 않음, 잘라 냄)
        ppo_obj = torch.min(unclipped, clipped)
        
        ax.plot(ratios.numpy(), unclipped.numpy(), "b--", label="Unclipped", alpha=0.7)
        ax.plot(ratios.numpy(), clipped.numpy(), "r--", label="Clipped", alpha=0.7)
        ax.plot(ratios.numpy(), ppo_obj.numpy(), "g-", label="PPO (min)", linewidth=2)
        
        ax.axvline(x=1.0, color="gray", linestyle=":", alpha=0.5)
        ax.axvline(x=1 - epsilon, color="orange", linestyle=":", alpha=0.5, label=f"1-ε={1-epsilon}")
        ax.axvline(x=1 + epsilon, color="orange", linestyle=":", alpha=0.5, label=f"1+ε={1+epsilon}")
        
        ax.set_xlabel("Probability Ratio r(θ)")
        ax.set_ylabel("Objective")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("/home/claude/ppo_clipping.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved PPO clipping visualization to ppo_clipping.png")


def demonstrate_clip_behavior():
    """자르기가 기울기 흐름에 어떻게 미치는지 보인다."""
    print("=" * 60)
    print("PPO Clipping Behavior Analysis")
    print("=" * 60)
    
    epsilon = 0.2
    
    # 서로 다른 비/이점 어우름을 흉내 낸다
    scenarios = [
        (1.5, 1.0, "Large ratio, positive advantage"),
        (0.7, 1.0, "Small ratio, positive advantage"),
        (1.5, -1.0, "Large ratio, negative advantage"),
        (0.7, -1.0, "Small ratio, negative advantage"),
        (1.1, 1.0, "Moderate ratio, positive advantage"),
        (0.9, -1.0, "Moderate ratio, negative advantage"),
    ]
    
    print(f"\nε = {epsilon}")
    print(f"{'Scenario':<45} {'Ratio':>6} {'A':>5} {'Unclip':>8} {'Clip':>8} {'PPO':>8} {'Clipped?':>8}")
    print("-" * 95)
    
    for ratio_val, adv_val, desc in scenarios:
        ratio = torch.tensor(ratio_val)
        adv = torch.tensor(adv_val)
        
        unclipped = ratio * adv
        clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
        clipped = clipped_ratio * adv
        ppo = torch.min(unclipped, clipped)
        is_clipped = (ppo != unclipped).item()
        
        print(
            f"{desc:<45} {ratio_val:>6.2f} {adv_val:>5.1f} "
            f"{unclipped.item():>8.3f} {clipped.item():>8.3f} "
            f"{ppo.item():>8.3f} {'Yes' if is_clipped else 'No':>8}"
        )


def ppo_loss_computation_example():
    """PPO 손실을 걸음마다 셈해 본다."""
    print("\n" + "=" * 60)
    print("PPO Loss Computation Example")
    print("=" * 60)
    
    batch_size = 8
    epsilon = 0.2
    value_coef = 0.5
    entropy_coef = 0.01
    
    torch.manual_seed(42)
    
    # 흉내 낸 자료
    old_log_probs = torch.randn(batch_size) - 1.0  # 흔한 로그 낌새 값
    new_log_probs = old_log_probs + torch.randn(batch_size) * 0.1  # 조금 다르다
    advantages = torch.randn(batch_size)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    values = torch.randn(batch_size, requires_grad=True)
    returns = values.detach() + torch.randn(batch_size) * 0.5
    entropy = torch.tensor(0.5)
    
    # 비를 셈한다
    ratio = torch.exp(new_log_probs - old_log_probs)
    
    # 잘라 낸 대리 목표
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    
    # 값 손실
    value_loss = nn.functional.mse_loss(values, returns)
    
    # 온 손실
    total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
    
    print(f"\nBatch size: {batch_size}")
    print(f"Clip epsilon: {epsilon}")
    print(f"\nRatios: {ratio.detach().numpy().round(3)}")
    print(f"Advantages: {advantages.numpy().round(3)}")
    print(f"\nClipped count: {(torch.min(surr1, surr2) != surr1).sum().item()}/{batch_size}")
    print(f"Policy loss: {policy_loss.item():.4f}")
    print(f"Value loss: {value_loss.item():.4f}")
    print(f"Entropy bonus: {entropy.item():.4f}")
    print(f"Total loss: {total_loss.item():.4f}")


def compare_clip_values():
    """자르는 값을 달리하여 PPO를 견준다."""
    print("\n" + "=" * 60)
    print("Effect of Clipping Parameter ε")
    print("=" * 60)
    
    torch.manual_seed(42)
    batch_size = 1000
    old_log_probs = torch.randn(batch_size) - 1.0
    new_log_probs = old_log_probs + torch.randn(batch_size) * 0.3
    advantages = torch.randn(batch_size)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    ratio = torch.exp(new_log_probs - old_log_probs)
    
    print(f"\nRatio stats: mean={ratio.mean():.3f}, std={ratio.std():.3f}, "
          f"min={ratio.min():.3f}, max={ratio.max():.3f}")
    
    epsilons = [0.05, 0.1, 0.2, 0.3, 0.5]
    print(f"\n{'ε':>6} {'Loss':>10} {'% Clipped':>12} {'Effective KL':>14}")
    print("-" * 44)
    
    for eps in epsilons:
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - eps, 1 + eps) * advantages
        ppo_obj = torch.min(surr1, surr2)
        loss = -ppo_obj.mean()
        pct_clipped = (ppo_obj != surr1).float().mean() * 100
        
        approx_kl = ((ratio - 1) - torch.log(ratio)).mean()
        
        print(f"{eps:>6.2f} {loss.item():>10.4f} {pct_clipped.item():>11.1f}% {approx_kl.item():>14.5f}")


if __name__ == "__main__":
    demonstrate_clip_behavior()
    ppo_loss_computation_example()
    compare_clip_values()
    visualize_ppo_clipping()
```

## 2. 논의

이 구현은 PPO의 종요로운 연산을 만든 도구 함수 여럿을 축으로 삼는다. 코드는 알고리즘 조각을 보여 주기와 따지기 논리에서 갈라놓는 조각 설계를 따른다.

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
    이 얼개 고름은 믿음 구역 방법에서 자리 잡은 좋은 버릇을 비춘다. 보기로 ReLU 활성은 곧지 않음을 주면서 0보다 큰 들임에서 기울기가 사라지는 것을 막는다. 손실 함수는 일감 갈래에 맞추어 고른다(갈래 나누기에는 사귐 엔트로피, 되돌이에는 평균 제곱 잘못). 다른 것으로 바꾸면(보기로 시그모이드 활성, L1 손실) 가장 좋게 하기 지형이 바뀌어 됨됨이가 나빠질 수 있으나, 어떤 자리에서는 바꾸는 것이 이로울 수도 있다.

---

**연습문제 3.**
이 구현을 더 만만치 않은 자리로 넓혀라. 더 큰 자료 뭉치, 다른 문제 갈래, 덧붙인 기능 가운데 하나를 고르라. 고친 바를 밝히고 됨됨이에 미친 바를 따져라.

??? success "연습문제 3 풀이"
    절로 떠오르는 넓히기 하나는 정칙화(드롭아웃, 무게 삭임)나 더 야무진 얼개(켜 더하기, 건너뛰는 이음)를 더하는 것이다. 고른 넓히기를 만들고 같은 자료로 익힌 뒤 앞뒤의 재기를 견주어라. 이 넓히기는 처음 알고리즘과 고친 바의 이치 밑뜻을 모두 아는 것을 보여야 한다.

## 정리하며

**다룬 것** — 가까운 곳 방침 가장 좋게 하기(PPO)

이 구현은 PPO의 종요로운 연산을 만든 도구 함수 여럿을 축으로 삼는다.

앞의 연습문제 3개로 스스로 따져 볼 수 있다.
