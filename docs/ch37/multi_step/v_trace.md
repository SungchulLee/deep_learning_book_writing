# V-Trace

V-Trace는 IMPALA 분산 강화학습 구조를 위해 설계한 오프폴리시 보정 알고리즘이다. 그 구조에서 액터들은 오래되었을 수도 있는 폴리시로 경험을 모으고 중앙의 러너가 파라미터를 갱신한다. V-Trace는 가치 보정의 $\bar{\rho}$ 와 트레이스 전파의 $\bar{c}$ 라는 잘라내기 수준 둘로 행동 폴리시와 타깃 폴리시가 갈릴 때 편향과 분산의 균형을 잡는다. 이 설계가 수렴 보장을 지키면서 확장 가능한 분산 학습을 가능하게 한다.

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

이 구현의 중심은 `compute_vtrace` 다. 중요도 비 $\pi(a_t|s_t)/\mu(a_t|s_t)$ 를 두 번 자른다. $\rho_t = \min(\bar\rho, \cdot)$ 는 TD 오차 $\delta_t = \rho_t\big(r_t + \gamma V(s_{t+1}) - V(s_t)\big)$ 에 곱해 **어느 폴리시의 가치로 수렴하는지**를 정하고, $c_t = \min(\bar c, \cdot)$ 는 뒤에서 앞으로 $v_t - V(s_t) = \delta_t + \gamma c_t\big(v_{t+1} - V(s_{t+1})\big)$ 를 쌓을 때 곱해 **얼마나 멀리 이어 붙이는지**, 곧 분산을 정한다. 함수가 돌려주는 어드밴티지는 $v_t - V(s_t)$ 다. 폴리시 그래디언트용 어드밴티지 $\rho_t\big(r_t + \gamma v_{t+1} - V(s_t)\big)$ 도 `pg_advantages` 로 계산하지만 돌려주지는 않는다. `compute_vtrace_batch` 는 같은 계산을 배치 축으로 벡터화한 것이다.

데모 함수는 다섯 부분이며 모두 합성 데이터로 계산만 한다. (1) 길이 10 의 궤적에서 타깃과 어드밴티지를 찍는다. 타깃 폴리시의 로그 확률이 0, 행동 폴리시가 −0.3 이라 비가 $e^{0.3} \approx 1.35$ 이고, 기본값 $\bar\rho = \bar c = 1$ 에서 둘 다 1 로 잘린다. (2) $\bar\rho$ 와 $\bar c$ 를 바꿔 가며 타깃의 평균과 표준편차를 찍는다. (3) 「뒤처짐」 세 가지에서 비와 잘린 비율을 본다. 세 경우 모두 행동 폴리시의 확률을 낮춰 비가 1 보다 크게 만들었으므로 잘린 비율이 늘 100% 다. (4) 배치 함수의 입출력 형태를 확인한다. (5) 온폴리시와 오프폴리시 궤적의 타깃을 견주는데, 차이가 0 으로 나온다(연습문제 3). 이 부분의 제목은 「Uncorrected N-step」이지만 실제로는 둘 다 V-Trace 로 계산한 것이다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
모든 $t$ 에서 $\rho_t = c_t = 1$ 이고 에피소드가 끝나지 않으면, V-Trace 타깃이 $v_t = r_t + \gamma v_{t+1}$ 을 만족하며, 결국 $v_t = \sum_{k=t}^{T-1}\gamma^{k-t}r_k + \gamma^{T-t}V(s_T)$ 로 중간 상태의 $V$ 에 의존하지 않음을 보여라. 데모 첫 부분의 출력(`13.005`, `12.126`, `11.238`, …)으로 이 관계를 확인하라.

</div>

??? success "풀이"
    $c_t = 1$ 이면 점화식이 $v_t - V(s_t) = r_t + \gamma V(s_{t+1}) - V(s_t) + \gamma\big(v_{t+1} - V(s_{t+1})\big)$ 이고, $V(s_t)$ 와 $\gamma V(s_{t+1})$ 이 지워져 $v_t = r_t + \gamma v_{t+1}$ 이 남는다. 끝에서는 `vs_minus_v[T] = 0` 이라 $v_T = V(s_T)$ 로 부트스트랩하므로, 펼치면 $(T-t)$-스텝 리턴이 된다. 데모 첫 부분은 비가 1.35 라 기본값 $\bar\rho = \bar c = 1$ 에서 둘 다 1 로 잘리므로 이 경우에 해당한다. 리워드가 1 이니 $1 + 0.99 \times 12.126 = 13.005$, $1 + 0.99 \times 11.238 = 12.126$ 으로 출력과 맞는다. 반면 `Values` 는 8.082, 4.413, 0.642 처럼 들쭉날쭉한데 타깃은 매끄럽게 줄어든다. 중간 상태의 $V$ 가 지워졌기 때문이다. 어드밴티지 $v_t - V(s_t)$ 는 $13.005 - 8.082 = 4.923$ 처럼 그 들쭉날쭉함을 그대로 물려받는다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
데모의 「Effect of Truncation Thresholds」 출력에서 세 가지를 설명하라. (가) $\bar\rho = 5$ 와 $\bar\rho = \infty$ 의 줄이 똑같은 까닭, (나) $\bar c = 5$ 일 때 표준편차가 13~18 로 커지는 까닭, (다) $\bar c$ 를 0.5 로 줄이면 평균이 5~6 으로 내려가는 까닭.

</div>

??? success "풀이"
    (가) 비가 $e^{0.3} \approx 1.35$ 라 $\bar\rho$ 가 1.35 보다 크면 자르기가 일어나지 않는다. 5 와 $\infty$ 는 둘 다 $\rho = 1.35$ 를 주므로 결과가 같다. (나) $\bar c = 5$ 이면 $c = 1.35$ 이고, 뒤에서 앞으로 쌓을 때마다 $\gamma c = 0.99 \times 1.35 \approx 1.34$ 배가 곱해진다. 1 보다 크므로 앞쪽 스텝으로 갈수록 뒤의 TD 오차가 기하급수적으로 불어나 타깃이 폭주한다. $\bar c \le 1$ 로 두어야 하는 까닭, 곧 V-Trace 가 $c$ 를 자르는 까닭이 이것이다. (다) $\bar c$ 가 작으면 $\gamma c$ 가 작아 트레이스가 빨리 끊기고, 타깃이 한 스텝 타깃 $V(s_t) + \delta_t$ 쪽으로 가까워진다. 이 궤적의 $V$ 는 평균 5 근처에서 뽑았으므로 타깃의 평균도 그 근처(5.1~6.2)에 머문다. $\bar c$ 는 분산과 전파 거리를 정할 뿐, 수렴하는 가치는 $\bar\rho$ 가 정한다는 점도 기억해 두자. Espeholt 등은 $\bar\rho$ 가 유한하면 V-Trace 가 $\pi$ 와 $\mu$ 사이의 폴리시 $\pi_{\bar\rho} \propto \min(\bar\rho\,\mu, \pi)$ 의 가치로 수렴함을 보였다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
데모 마지막 부분은 「Correction effect: 0.0000」을 찍는다. 오프폴리시 궤적인데도 보정 효과가 0 인 까닭을 설명하라. 보정이 실제로 일을 하도록 행동 폴리시와 타깃 폴리시의 로그 확률을 바꾸고, 그때 스텝 0 의 오프폴리시 타깃이 대략 얼마가 되는지 계산하라.

</div>

??? success "풀이"
    오프폴리시 쪽은 타깃 로그 확률 0, 행동 로그 확률 −0.5 라 비가 $e^{0.5} \approx 1.65$ 다. 기본값 $\bar\rho = \bar c = 1$ 에서 이 비는 1 로 잘리므로 온폴리시와 똑같은 계산이 되어 차이가 0 이다. 행동 폴리시가 고른 행동을 타깃 폴리시가 **더** 좋아하는 경우는 잘림 때문에 보정이 일어나지 않는다. 보정이 일을 하는 것은 비가 1 보다 작을 때, 곧 행동 폴리시는 자주 골랐지만 타깃 폴리시는 덜 고를 행동일 때다. 예를 들어 타깃 −0.5, 행동 0 으로 바꾸면 $\rho = c = e^{-0.5} \approx 0.607$ 이다. 그러면 TD 오차가 $0.607 \times 0.95 \approx 0.576$ 으로 줄고 스텝마다 $\gamma c \approx 0.600$ 배씩 줄어들므로, 스텝 0 의 타깃은 $5 + 0.576 \times (1 - 0.600^{15})/(1 - 0.600) \approx 6.44$ 다. 온폴리시 타깃 18.294 보다 훨씬 작다. 행동 폴리시가 받은 리워드를 타깃 폴리시의 것으로 그대로 믿지 않고, 비만큼 깎아 반영한 결과다.

## 정리하며

**다룬 것** — V-Trace

이 구현은 두 가지로 자른 중요도 비 $\rho_t$, $c_t$ 로 오프폴리시 궤적의 가치 타깃을 계산하는 `compute_vtrace` 와, 그 배치판 `compute_vtrace_batch`, 비의 통계를 보는 `analyze_is_ratios` 를 담는다.

핵심 함수는 `compute_vtrace` 이며, 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
