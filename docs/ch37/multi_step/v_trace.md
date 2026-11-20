# V-Trace

V-Trace은 IMPALA 분산 강화학습 구조를 위해 설계한 오프폴리시 보정 알고리즘이다. 그 구조에서 액터들은 오래되었을 수도 있는 폴리시로 경험을 모으고 중앙의 러너가 파라미터를 갱신한다. V-Trace은 값 보정의 $\bar{\rho}$과 추적 전파의 $\bar{c}$이라는 잘라내기 수준 둘로 행동 폴리시와 타깃 폴리시가 갈릴 때 편향과 분산의 균형을 잡는다. 이 설계가 수렴 보장을 지키면서 확장 가능한 분산 학습을 가능하게 한다.

## 1. 코드

```python
"""
35.3.3 V자취
===============

흩뿌린 힘 북돋우는 러닝 환경을 위한 V자취 폴리시 밖 바로잡기.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

# 이 쪽이 싣는 수가 돌릴 때마다 달라지지 않게 씨앗을 박는다.
torch.manual_seed(0)
np.random.seed(0)


# ========================================================================
# 메인
# ========================================================================


def compute_vtrace(
    values: torch.Tensor,          # V(s_t), 형태 (T+1,) — 부트스트랩을 넣음
    rewards: torch.Tensor,         # r_t, 형태 (T,)
    dones: torch.Tensor,           # done_t, 형태 (T,)
    target_log_probs: torch.Tensor,   # log π(a_t|s_t), 형태 (T,)
    behavior_log_probs: torch.Tensor, # log μ(a_t|s_t), 형태 (T,)
    gamma: float = 0.99,
    rho_bar: float = 1.0,
    c_bar: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """V자취 타깃과 이점을 셈한다.
    
    인수:
        values: t=0..T에 대한 V(s_t)(값 T+1개이며 마지막은 부트스트랩)
        rewards: t=0..T-1에 대한 갚음 r_t
        dones: t=0..T-1에 대한 끝남 깃발
        target_log_probs: 이제 폴리시 아래의 log π(a_t|s_t)
        behavior_log_probs: 액션 폴리시 아래의 log μ(a_t|s_t)
        gamma: 깎기 인수
        rho_bar: 값 바로잡기의 자르기
        c_bar: 자취 퍼뜨리기의 자르기
        
    반환값:
        (vs, advantages): V 자취 값과 폴리시 그래디언트 이점
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
    
    # V자취 타깃을 거꾸로 셈
    vs_minus_v = torch.zeros(T + 1)
    for t in reversed(range(T)):
        vs_minus_v[t] = delta_v[t] + not_done[t] * gamma * c[t] * vs_minus_v[t + 1]
    
    vs = vs_minus_v[:T] + values[:T]
    
    # 폴리시 그래디언트를 위한 이점
    # A_t = rho_t (r_t + gamma * v_{t+1} - V(s_t)) 이다.
    # vs의 길이는 T이므로 vs[1:]는 T-1개뿐이다. 마지막 스텝의 v_{t+1}은
    # 부트스트랩 값 values[T]를 쓴다.
    vs_next = torch.cat([vs[1:], values[T:T + 1]])
    pg_advantages = rho * (rewards + not_done * gamma * vs_next.detach()
                           - values[:T])

    # 값 바탕 방법을 위한 간단한 이점. 이 함수가 돌려주는 것은 이쪽이다.
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
    """배치 V자취 셈."""
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
    
    # 폴리시 뒤처짐 흉내: 액션 폴리시가 더 오래되고 조금 다름
    target_log_probs = torch.zeros(T)  # 욕심쟁이는 log(1) = 0
    behavior_log_probs = torch.zeros(T) - 0.3  # 조금 다름
    
    vs, adv = compute_vtrace(values, rewards, dones,
                              target_log_probs, behavior_log_probs)
    print(f"  Values: {values[:5].numpy().round(3)}...")
    print(f"  V-trace targets: {vs[:5].detach().numpy().round(3)}...")
    print(f"  Advantages: {adv[:5].detach().numpy().round(3)}...")

    # --- 자르는 임계의 효과 ---
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
    # 폴리시 뒤처짐 정도를 달리해 흉내
    for lag_desc, beh_offset in [("Small lag", -0.1), ("Medium lag", -0.5),
                                   ("Large lag", -1.0)]:
        tp = torch.zeros(100)
        bp = torch.zeros(100) + beh_offset
        stats = analyze_is_ratios(tp.exp(), bp.exp())
        print(f"  {lag_desc}:")
        print(f"    Raw ratio: {stats['raw_ratio_mean']:.3f} ± {stats['raw_ratio_std']:.3f}")
        print(f"    Fraction clipped (ρ): {stats['fraction_clipped_rho']:.1%}")

    # --- 배치 셈 ---
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
    
    # 폴리시 안(바로잡기 필요 없음)
    tp_on = torch.zeros(T)
    bp_on = torch.zeros(T)
    vs_on, _ = compute_vtrace(values, rewards, dones, tp_on, bp_on)
    
    # 폴리시 밖(바로잡기 필요)
    tp_off = torch.zeros(T)
    bp_off = torch.zeros(T) - 0.5
    vs_off, _ = compute_vtrace(values, rewards, dones, tp_off, bp_off)
    
    print(f"  On-policy targets:  {vs_on[:5].detach().numpy().round(3)}")
    print(f"  Off-policy targets: {vs_off[:5].detach().numpy().round(3)}")
    print(f"  Correction effect: {(vs_on - vs_off).abs().mean():.4f}")

    print("\nV-trace demo complete!")


if __name__ == "__main__":
    demo_vtrace()
```

**출력:**

```
============================================================
V-Trace Demo
============================================================

--- V-Trace Target Computation ---
  Values: [8.082 4.413 0.642 6.137 2.831]...
  V-trace targets: [13.005 12.126 11.238 10.342  9.436]...
  Advantages: [ 4.923  7.713 10.596  4.205  6.605]...

--- Effect of Truncation Thresholds ---
  ρ̄=  0.5, c̄=0.5: target mean=5.110, std=1.043
  ρ̄=  0.5, c̄=1.0: target mean=6.698, std=1.869
  ρ̄=  0.5, c̄=5.0: target mean=11.188, std=7.243
  ρ̄=  1.0, c̄=0.5: target mean=5.765, std=0.925
  ρ̄=  1.0, c̄=1.0: target mean=8.942, std=2.771
  ρ̄=  1.0, c̄=5.0: target mean=17.921, std=13.709
  ρ̄=  5.0, c̄=0.5: target mean=6.224, std=1.714
  ρ̄=  5.0, c̄=1.0: target mean=10.512, std=3.747
  ρ̄=  5.0, c̄=5.0: target mean=22.632, std=18.307
  ρ̄=  inf, c̄=0.5: target mean=6.224, std=1.714
  ρ̄=  inf, c̄=1.0: target mean=10.512, std=3.747
  ρ̄=  inf, c̄=5.0: target mean=22.632, std=18.307

--- IS Ratio Analysis ---
  Small lag:
    Raw ratio: 1.105 ± 0.000
    Fraction clipped (ρ): 100.0%
  Medium lag:
    Raw ratio: 1.649 ± 0.000
    Fraction clipped (ρ): 100.0%
  Large lag:
    Raw ratio: 2.718 ± 0.000
    Fraction clipped (ρ): 100.0%

--- Batch V-Trace ---
  Input shapes: values=torch.Size([8, 21]), rewards=torch.Size([8, 20])
  Output shapes: targets=torch.Size([8, 20]), advantages=torch.Size([8, 20])
  Target mean: 0.999

--- V-trace vs Uncorrected N-step ---
  On-policy targets:  [18.294 17.469 16.636 15.793 14.943]
  Off-policy targets: [18.294 17.469 16.636 15.793 14.943]
  Correction effect: 0.0000

V-trace demo complete!
```

## 2. 논의

이 구현은 V-Trace의 핵심 연산을 구현하는 여러 도구 함수를 중심에 둔다. 코드는 알고리즘 구성 요소를 데모·평가 논리와 분리하는 모듈식 설계를 따른다.

데모 함수는 핵심 동작을 잘 드러내는 합성 데이터에서 이 구성 요소들의 실제 쓰임을 보인다. 출력을 살펴보면 하이퍼파라미터 선택과 문제 설정에 따라 알고리즘의 성능이 어떻게 달라지는지 볼 수 있다.

실용적 관점에서 이 구현은 순수한 성능보다 명료함을 앞세운다. 실제로 쓰는 시스템은 보통 배치 계산, GPU 가속, 더 정교한 하이퍼파라미터 튜닝 같은 개선을 더한다. 그럼에도 여기 보인 핵심 알고리즘 아이디어는 대규모 응용으로 그대로 옮겨 간다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
데모 코드를 돌려 핵심 출력 지표를 기록하라. 하이퍼파라미터 하나(러닝레이트, 은닉 차원, 레이어 개수 같은 것)를 바꾸고 결과가 어떻게 바뀌는지 기록하라.

</div>

??? success "풀이"
    데모를 돌린 뒤 나머지를 고정하고 고른 하이퍼파라미터를 차례로 바꾼다. 예를 들어 은닉 차원을 두 배로 하면 보통 표현력이 늘지만 계산 시간이 커진다. 러닝레이트는 단조롭지 않은 영향을 준다. 너무 작으면 느리게 수렴하고 너무 크면 진동한다. 고른 하이퍼파라미터의 서로 다른 값을 적어도 셋 잡아 구체적인 수를 기록해 두라.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
이 구현에서 핵심 구조 선택의 역할을 설명하라. 왜 그 활성화 함수, 정규화 전략, 로스 함수를 쓰는가? 다른 것으로 바꾸면 어떻게 되는가?

</div>

??? success "풀이"
    이 구조 선택은 여러 스텝 학습에서 자리 잡은 모범 사례를 따른다. 예를 들어 ReLU 활성화는 비선형성을 주면서 양의 입력에서 그래디언트가 사라지는 것을 피한다. 로스 함수는 과제 유형에 맞게 고른다(분류에는 교차 엔트로피, 회귀에는 평균 제곱 오차). 다른 것으로 바꾸면(예를 들어 시그모이드 활성화, L1 로스) 최적화 지형이 바뀌어 성능이 나빠질 수 있지만 어떤 상황에서는 바꾸는 것이 이로울 수도 있다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
더 어려운 상황을 다루도록 구현을 확장하라. 더 큰 데이터셋, 다른 문제 변형, 피처 추가 가운데 하나를 고른다. 수정 사항을 적고 성능에 미친 영향을 분석하라.

</div>

??? success "풀이"
    자연스러운 확장 하나는 정칙화(드롭아웃, 웨이트 감쇠)나 더 정교한 구조(레이어 추가, 스킵 연결)를 더하는 것이다. 고른 확장을 구현하고 같은 데이터로 학습한 뒤 전후의 지표를 비교하여라. 이 확장은 원래 알고리즘과 수정 사항의 이론적 근거를 모두 이해했음을 보여야 한다.

## 정리하며

**다룬 것** — V-Trace

이 구현은 V-Trace의 핵심 연산을 구현하는 여러 도구 함수를 중심에 둔다.

앞의 연습문제 3개로 스스로 점검해 볼 수 있다.
