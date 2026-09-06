# V-Trace

V-Trace은 IMPALA 나눠 하는 힘 북돋우는 배움 얼개를 위해 설계한 벗어난 방침 바로잡기 알고리즘이다. 그 얼개에서 배우들은 묵었을 수도 있는 방침으로 겪음을 모으고 가운데 배움이가 잡을 고친다. V-Trace은 값 바로잡기의 $\bar{\rho}$과 자취 퍼뜨리기의 $\bar{c}$이라는 잘라내기 켜 둘로 움직임 방침과 과녁 방침이 갈릴 때 치우침과 흩어짐의 균형을 잡는다. 이 설계가 모임 보장을 지키면서 커질 수 있는 나눠 익히기를 가능하게 한다.

## 코드

```python
"""
33.3.3 V자취
===============

흩뿌린 힘 북돋우는 배움 환경을 위한 V자취 방침 밖 바로잡기.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

# ========================================================================
# 메인
# ========================================================================


def compute_vtrace(
    values: torch.Tensor,          # V(s_t), 꼴 (T+1,) — 부트스트랩을 넣음
    rewards: torch.Tensor,         # r_t, 꼴 (T,)
    dones: torch.Tensor,           # done_t, 꼴 (T,)
    target_log_probs: torch.Tensor,   # log π(a_t|s_t), 꼴 (T,)
    behavior_log_probs: torch.Tensor, # log μ(a_t|s_t), 꼴 (T,)
    gamma: float = 0.99,
    rho_bar: float = 1.0,
    c_bar: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """V자취 과녁과 이점을 셈한다.
    
    인수:
        values: V(s_t) for t=0..T (T+1 values, last is bootstrap)
        rewards: rewards r_t for t=0..T-1
        dones: done flags for t=0..T-1
        target_log_probs: log π(a_t|s_t) under current policy
        behavior_log_probs: log μ(a_t|s_t) under behavior policy
        gamma: 깎기 인수
        rho_bar: 값 바로잡기의 자르기
        c_bar: 자취 퍼뜨리기의 자르기
        
    반환값:
        (vs, advantages): V-trace values and policy gradient advantages
    """
    T = len(rewards)
    
    # 중요도 뽑기 비
    log_ratios = target_log_probs - behavior_log_probs
    ratios = torch.exp(log_ratios)
    
    # 잘린 중요도 뽑기 비
    rho = torch.clamp(ratios, max=rho_bar)
    c = torch.clamp(ratios, max=c_bar)
    
    # rho 바로잡기를 한 때 차이 어긋남
    not_done = 1.0 - dones
    delta_v = rho * (rewards + not_done * gamma * values[1:] - values[:T])
    
    # V자취 과녁을 거꾸로 셈
    vs_minus_v = torch.zeros(T + 1)
    for t in reversed(range(T)):
        vs_minus_v[t] = delta_v[t] + not_done[t] * gamma * c[t] * vs_minus_v[t + 1]
    
    vs = vs_minus_v[:T] + values[:T]
    
    # 방침 기울기를 위한 이점(필요하면)
    advantages = rho * (rewards + not_done * gamma * vs[1:].detach() 
                        if T > 1 else rewards - values[:T])
    # 값 바탕 방법을 위한 간단한 이점
    advantages = vs - values[:T]
    
    return vs, advantages


def compute_vtrace_batch(
    values_batch: torch.Tensor,          # (B, T+1)
    rewards_batch: torch.Tensor,         # (B, T)
    dones_batch: torch.Tensor,           # (B, T)
    target_log_probs_batch: torch.Tensor,  # (B, T)
    behavior_log_probs_batch: torch.Tensor,  # (B, T)
    gamma: float = 0.99,
    rho_bar: float = 1.0,
    c_bar: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """묶음 V자취 셈."""
    B, T = rewards_batch.shape
    
    log_ratios = target_log_probs_batch - behavior_log_probs_batch
    ratios = torch.exp(log_ratios)
    rho = torch.clamp(ratios, max=rho_bar)
    c = torch.clamp(ratios, max=c_bar)
    
    not_done = 1.0 - dones_batch
    delta_v = rho * (rewards_batch + not_done * gamma * values_batch[:, 1:] 
                     - values_batch[:, :T])
    
    vs_minus_v = torch.zeros(B, T + 1)
    for t in reversed(range(T)):
        vs_minus_v[:, t] = delta_v[:, t] + not_done[:, t] * gamma * c[:, t] * vs_minus_v[:, t + 1]
    
    vs = vs_minus_v[:, :T] + values_batch[:, :T]
    advantages = vs - values_batch[:, :T]
    
    return vs, advantages


# ---------------------------------------------------------------------------
# V자취 살피기 연장
# ---------------------------------------------------------------------------

def analyze_is_ratios(target_probs: torch.Tensor, behavior_probs: torch.Tensor,
                      rho_bar: float = 1.0, c_bar: float = 1.0):
    """중요도 뽑기 비의 셈밝힘을 살핀다."""
    ratios = target_probs / (behavior_probs + 1e-8)
    rho = torch.clamp(ratios, max=rho_bar)
    c = torch.clamp(ratios, max=c_bar)
    
    return {
        'raw_ratio_mean': ratios.mean().item(),
        'raw_ratio_std': ratios.std().item(),
        'raw_ratio_max': ratios.max().item(),
        'rho_mean': rho.mean().item(),
        'c_mean': c.mean().item(),
        'fraction_clipped_rho': (ratios > rho_bar).float().mean().item(),
        'fraction_clipped_c': (ratios > c_bar).float().mean().item(),
    }


# ---------------------------------------------------------------------------
# 시연
# ---------------------------------------------------------------------------

def demo_vtrace():
    print("=" * 60)
    print("V-Trace Demo")
    print("=" * 60)

    # --- 기본 V자취 셈 ---
    print("\n--- V-Trace Target Computation ---")
    T = 10
    values = torch.randn(T + 1) * 2 + 5  # t=0..T의 V(s_t)
    rewards = torch.ones(T)
    dones = torch.zeros(T)
    
    # 방침 뒤처짐 흉내: 움직임 방침이 더 오래되고 조금 다름
    target_log_probs = torch.zeros(T)  # 욕심쟁이는 log(1) = 0
    behavior_log_probs = torch.zeros(T) - 0.3  # 조금 다름
    
    vs, adv = compute_vtrace(values, rewards, dones,
                              target_log_probs, behavior_log_probs)
    print(f"  Values: {values[:5].numpy().round(3)}...")
    print(f"  V-trace targets: {vs[:5].detach().numpy().round(3)}...")
    print(f"  Advantages: {adv[:5].detach().numpy().round(3)}...")

    # --- 자르는 문턱의 효과 ---
    print("\n--- Effect of Truncation Thresholds ---")
    for rho_bar in [0.5, 1.0, 5.0, float('inf')]:
        for c_bar in [0.5, 1.0, 5.0]:
            vs, _ = compute_vtrace(values, rewards, dones,
                                    target_log_probs, behavior_log_probs,
                                    rho_bar=rho_bar, c_bar=c_bar)
            print(f"  ρ̄={rho_bar:>5}, c̄={c_bar}: "
                  f"target mean={vs.mean():.3f}, std={vs.std():.3f}")

    # --- 중요도 뽑기 비 살피기 ---
    print("\n--- IS Ratio Analysis ---")
    # 방침 뒤처짐 정도를 달리해 흉내
    for lag_desc, beh_offset in [("Small lag", -0.1), ("Medium lag", -0.5),
                                   ("Large lag", -1.0)]:
        tp = torch.zeros(100)
        bp = torch.zeros(100) + beh_offset
        stats = analyze_is_ratios(tp.exp(), bp.exp())
        print(f"  {lag_desc}:")
        print(f"    Raw ratio: {stats['raw_ratio_mean']:.3f} ± {stats['raw_ratio_std']:.3f}")
        print(f"    Fraction clipped (ρ): {stats['fraction_clipped_rho']:.1%}")

    # --- 묶음 셈 ---
    print("\n--- Batch V-Trace ---")
    B, T = 8, 20
    values_b = torch.randn(B, T + 1)
    rewards_b = torch.randn(B, T)
    dones_b = torch.zeros(B, T)
    tp_b = torch.zeros(B, T)
    bp_b = torch.zeros(B, T) - 0.2
    
    vs_b, adv_b = compute_vtrace_batch(values_b, rewards_b, dones_b, tp_b, bp_b)
    print(f"  Input shapes: values={values_b.shape}, rewards={rewards_b.shape}")
    print(f"  Output shapes: targets={vs_b.shape}, advantages={adv_b.shape}")
    print(f"  Target mean: {vs_b.mean():.3f}")

    # --- 견주기: V자취와 바로잡지 않은 것 ---
    print("\n--- V-trace vs Uncorrected N-step ---")
    T = 15
    values = torch.ones(T + 1) * 5.0
    rewards = torch.ones(T)
    dones = torch.zeros(T)
    
    # 방침 안(바로잡기 필요 없음)
    tp_on = torch.zeros(T)
    bp_on = torch.zeros(T)
    vs_on, _ = compute_vtrace(values, rewards, dones, tp_on, bp_on)
    
    # 방침 밖(바로잡기 필요)
    tp_off = torch.zeros(T)
    bp_off = torch.zeros(T) - 0.5
    vs_off, _ = compute_vtrace(values, rewards, dones, tp_off, bp_off)
    
    print(f"  On-policy targets:  {vs_on[:5].detach().numpy().round(3)}")
    print(f"  Off-policy targets: {vs_off[:5].detach().numpy().round(3)}")
    print(f"  Correction effect: {(vs_on - vs_off).abs().mean():.4f}")

    print("\nV-trace demo complete!")


if __name__ == "__main__":
    demo_vtrace()```

## 논의

이 짜기는 V-Trace의 핵심 연산을 짜는 여러 연장 함수를 한가운데 둔다. 코드는 알고리즘 조각을 보여 주기와 따지기 논리에서 떼어 놓는 조각 짜기를 따른다.

보여 주기 함수는 핵심 움직임을 도드라지게 하는 만든 자료에서 이 조각들의 실제 쓰임을 보인다. 내놓기를 살펴보면 윗매개변수를 어떻게 고르고 문제를 어떻게 차리느냐에 따라 알고리즘의 성능이 어떻게 달라지는지 볼 수 있다.

실제 관점에서 이 짜기는 순수한 성능보다 또렷함을 앞세운다. 실제로 쓰는 얼개는 보통 묶음 셈, GPU 빠르게 하기, 더 정교한 윗매개변수 맞추기 같은 개선을 더한다. 그럼에도 여기 보인 핵심 알고리즘 생각은 큰 규모의 쓰임새로 곧바로 옮겨 간다.

## 연습문제

**연습문제 1.**
보여 주기 코드를 돌려 핵심 내놓기 잣대를 적어라. 윗매개변수 하나(배움 빠르기, 숨은 차원, 층 개수 같은 것)를 고치고 결과가 어떻게 바뀌는지 적어라.

??? success "연습문제 1 풀이"
    보여 주기를 돌린 뒤 나머지를 붙박아 두고 고른 윗매개변수를 차근히 바꾼다. 보기로 숨은 차원을 두 배로 하면 보통 나타냄 담이가 늘지만 셈 시간이 커진다. 배움 빠르기는 단조롭지 않은 영향을 준다. 너무 작으면 느리게 모이고 너무 크면 흔들린다. 고른 윗매개변수의 서로 다른 값을 적어도 셋 잡아 구체적인 수를 적어 두라.

---

**연습문제 2.**
이 짜기에서 핵심 얼개 고르기의 몫을 밝혀라. 왜 그 깨움 함수, 고르게 맞추기 셈속, 손실 함수를 쓰는가? 다른 것으로 바꾸면 어떻게 되는가?

??? success "연습문제 2 풀이"
    이 얼개 고르기는 여러 걸음 배움에서 자리 잡은 가장 좋은 방식을 따른다. 보기로 ReLU 깨움은 비선형을 주면서 양의 들임에서 기울기가 사라지는 것을 피한다. 손실 함수는 일의 갈래에 맞게 고른다(가름에는 교차 엔트로피, 되돌이 맞춤에는 평균 제곱 어긋남). 다른 것으로 바꾸면(보기로 시그모이드 깨움, L1 손실) 가장 좋게 하기의 풍경이 바뀌어 성능이 나빠질 수 있지만 어떤 상황에서는 바꿈이 이로울 수도 있다.

---

**연습문제 3.**
더 어려운 상황을 다루도록 짜기를 넓혀라. 더 큰 자료 묶음, 다른 문제 변형, 특징 하나 더하기 가운데 하나이다. 고침을 적고 성능에 미친 영향을 따져라.

??? success "연습문제 3 풀이"
    자연스러운 넓힘 하나는 규칙 세우기(떨구기, 무게 줄임)나 더 정교한 얼개(층 더하기, 건너뛰는 이음)를 더하는 것이다. 고른 넓힘을 짜고 같은 자료로 익힌 뒤 앞뒤의 잣대를 견주어라. 이 넓힘은 본디 알고리즘과 고침의 이론 까닭을 모두 이해했음을 보여야 한다.
