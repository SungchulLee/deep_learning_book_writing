# Retrace

Retrace는 여러 스텝 시간차 학습의 오프폴리시 보정 방법으로, 다른 폴리시가 모은 데이터에서 안전하고 분산이 작은 리턴 추정을 준다. 잘라 낸 중요도 샘플링 비 $c_i = \lambda \min(1, \pi/\mu)$을 써서 행동 폴리시가 타깃 폴리시에서 크게 벗어나면 트레이스(trace, 앞선 스텝의 TD 오차를 이어 붙이는 가중치)를 자동으로 끊는다. 이로써 행동 폴리시와 상관없이 올바른 밸류 함수로 수렴이 보장되어 경험 리플레이와 분산 학습 설정에 특히 쓸모 있다.

## 1. 코드

```python
"""
35.3.2 Retrace(λ)
==================

리트레이스(λ)로 폴리시 밖을 바로잡은 여러 스텝 돌아옴.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple

# 이 쪽이 싣는 수가 돌릴 때마다 달라지지 않게 씨앗을 박는다.
torch.manual_seed(0)
np.random.seed(0)


# ========================================================================
# 메인
# ========================================================================


def compute_retrace_targets(
    q_values: torch.Tensor,        # 자취의 t마다 Q(s_t, a_t), 형태 (T,)
    next_q_max: torch.Tensor,      # t마다 max_a Q(s_{t+1}, a), 형태 (T,)
    rewards: torch.Tensor,         # r_t, 형태 (T,)
    dones: torch.Tensor,           # done_t, 형태 (T,)
    target_policy_probs: torch.Tensor,  # π(a_t|s_t), 형태 (T,)
    behavior_policy_probs: torch.Tensor,  # μ(a_t|s_t), 형태 (T,)
    gamma: float = 0.99,
    lambda_: float = 0.95,
) -> torch.Tensor:
    """자취 하나의 리트레이스(λ) 타깃을 셈한다.
    
    인수:
        q_values: 때 스텝마다 (s_t, a_t)에서의 Q 값
        next_q_max: 때 스텝마다의 max_a' Q(s_{t+1}, a')
        rewards: 때 스텝마다의 리워드
        dones: 끝남 깃발(1.0이면 끝났다)
        target_policy_probs: 타깃(이제) 폴리시 아래의 π(a_t | s_t)
        behavior_policy_probs: 액션(자료) 폴리시 아래의 μ(a_t | s_t)
        gamma: 깎기 인수
        lambda_: 리트레이스 람다 값
        
    반환값:
        리트레이스 타깃, 형태 (T,)
    """
    T = len(rewards)
    
    # 자취 계수: c_i = λ * min(1, π/μ)
    is_ratios = target_policy_probs / (behavior_policy_probs + 1e-8)
    c = lambda_ * torch.clamp(is_ratios, max=1.0)
    
    # 때 차이 어긋남: δ_t = r_t + γ * max_a Q(s_{t+1}, a) - Q(s_t, a_t)
    td_errors = rewards + (1 - dones) * gamma * next_q_max - q_values
    
    # 리트레이스 타깃을 거꾸로 셈
    targets = torch.zeros(T)
    # Q^ret(s_t, a_t) = Q(s_t, a_t) + Σ_{k=t}^{T-1} γ^{k-t} (Π c_i) δ_k
    
    for t in range(T):
        target = q_values[t].item()
        trace_product = 1.0
        for k in range(t, T):
            if k > t:
                trace_product *= c[k].item()
                if dones[k - 1].item() > 0.5:
                    break
            target += (gamma ** (k - t)) * trace_product * td_errors[k].item()
        targets[t] = target
    
    return targets


def compute_retrace_batch(
    q_online: nn.Module,
    q_target: nn.Module,
    trajectories: List[dict],
    gamma: float = 0.99,
    lambda_: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """자취 배치의 리트레이스 타깃을 셈한다.
    
    인수:
        q_online: 온라인 Q 네트워크
        q_target: 타깃 Q 네트워크
        trajectories: states, actions, rewards, next_states, dones,
                      behavior_probs 열쇠를 지닌 사전의 목록
        gamma: 깎기
        lambda_: 리트레이스 람다
        
    반환값:
        자취를 가로질러 이어 붙인 (all_q_values, all_targets)
    """
    all_q = []
    all_targets = []
    
    for traj in trajectories:
        states = torch.FloatTensor(np.array(traj['states']))
        actions = torch.LongTensor(np.array(traj['actions']))
        rewards = torch.FloatTensor(np.array(traj['rewards']))
        next_states = torch.FloatTensor(np.array(traj['next_states']))
        dones = torch.FloatTensor(np.array(traj['dones']))
        behavior_probs = torch.FloatTensor(np.array(traj['behavior_probs']))
        
        with torch.no_grad():
            q_vals = q_target(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            next_q_max = q_target(next_states).max(dim=1)[0]
            
            # 타깃 폴리시: 온라인 네트워크에 대해 욕심쟁이
            online_actions = q_online(states).argmax(dim=1)
            target_probs = (online_actions == actions).float()
        
        targets = compute_retrace_targets(
            q_vals, next_q_max, rewards, dones,
            target_probs, behavior_probs, gamma, lambda_
        )
        
        # 로스 셈을 위해 온라인 Q 값을 얻음
        online_q = q_online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        all_q.append(online_q)
        all_targets.append(targets)
    
    return torch.cat(all_q), torch.cat(all_targets)


# ---------------------------------------------------------------------------
# 리트레이스를 위한 자취 버퍼
# ---------------------------------------------------------------------------

class TrajectoryBuffer:
    """리트레이스 클래스 알고리즘을 위해 온전한 자취를 갈무리한다."""
    
    def __init__(self, max_trajectories: int = 1000):
        self.max_size = max_trajectories
        self.trajectories: List[dict] = []
        self.ptr = 0
    
    def push_trajectory(self, states, actions, rewards, next_states, dones, behavior_probs):
        traj = {
            'states': states, 'actions': actions, 'rewards': rewards,
            'next_states': next_states, 'dones': dones,
            'behavior_probs': behavior_probs,
        }
        if len(self.trajectories) < self.max_size:
            self.trajectories.append(traj)
        else:
            self.trajectories[self.ptr] = traj
        self.ptr = (self.ptr + 1) % self.max_size
    
    def sample(self, n: int) -> List[dict]:
        indices = np.random.choice(len(self.trajectories), min(n, len(self.trajectories)),
                                   replace=False)
        return [self.trajectories[i] for i in indices]
    
    def __len__(self):
        return len(self.trajectories)


# ---------------------------------------------------------------------------
# 시연
# ---------------------------------------------------------------------------

def demo_retrace():
    print("=" * 60)
    print("Retrace(λ) Demo")
    print("=" * 60)

    # --- 리트레이스 타깃 셈 ---
    print("\n--- Retrace Target Computation ---")
    T = 8
    q_vals = torch.randn(T) * 2 + 5
    next_q_max = torch.randn(T) * 2 + 5
    rewards = torch.ones(T)
    dones = torch.zeros(T)
    dones[-1] = 1.0
    target_probs = torch.ones(T) * 0.8
    behavior_probs = torch.ones(T) * 0.5
    
    for lam in [0.0, 0.5, 0.95, 1.0]:
        targets = compute_retrace_targets(
            q_vals, next_q_max, rewards, dones,
            target_probs, behavior_probs, gamma=0.99, lambda_=lam)
        print(f"  λ={lam:.2f}: targets = {targets[:4].numpy().round(3)}...")
    
    # --- 자취 계수 살피기 ---
    print("\n--- Trace Coefficients ---")
    print("  For greedy target policy with ε-greedy behavior (ε=0.1, |A|=4):")
    eps = 0.1
    n_actions = 4
    
    # 욕심쟁이 액션
    mu_greedy = 1 - eps + eps / n_actions
    c_greedy = 0.95 * min(1.0, 1.0 / mu_greedy)
    print(f"    Greedy action: μ={mu_greedy:.3f}, π=1.0, c=λ·min(1,π/μ)={c_greedy:.3f}")
    
    # 욕심쟁이 아닌 액션
    mu_random = eps / n_actions
    c_random = 0.95 * min(1.0, 0.0 / mu_random)
    print(f"    Non-greedy:    μ={mu_random:.3f}, π=0.0, c=λ·min(1,π/μ)={c_random:.3f}")
    print("    → Trace is cut when behavior took non-greedy action")
    
    # --- 자취 길이 살피기 ---
    print("\n--- Effective Trace Length ---")
    for lam in [0.5, 0.8, 0.95, 1.0]:
        for greedy_frac in [0.5, 0.8, 0.95]:
            # 잘리기 전 어림 자취 길이
            p_continue = lam * greedy_frac
            expected_len = 1.0 / (1.0 - p_continue) if p_continue < 1 else float('inf')
            print(f"    λ={lam}, P(greedy)={greedy_frac}: "
                  f"expected trace = {expected_len:.1f} steps")
    
    # --- n스텝과 견주기 ---
    print("\n--- Retrace vs N-step (with off-policy data) ---")
    T = 10
    q_vals = torch.ones(T) * 5.0
    next_q_max = torch.ones(T) * 5.0
    rewards = torch.ones(T)
    dones = torch.zeros(T)
    
    # 폴리시 밖 흉내: 어떤 액션은 욕심쟁이가 아니었음
    target_probs = torch.tensor([1, 1, 0, 1, 1, 0, 0, 1, 1, 1], dtype=torch.float32)
    behavior_probs = torch.ones(T) * 0.3
    
    retrace_targets = compute_retrace_targets(
        q_vals, next_q_max, rewards, dones, target_probs, behavior_probs)
    
    # n스텝(바로잡기 없음)
    nstep_targets = compute_retrace_targets(
        q_vals, next_q_max, rewards, dones,
        torch.ones(T), torch.ones(T))  # c_i = λ 늘
    
    print(f"  Retrace targets: {retrace_targets[:5].numpy().round(3)}")
    print(f"  N-step targets:  {nstep_targets[:5].numpy().round(3)}")
    print(f"  Difference: {(retrace_targets - nstep_targets).abs().mean():.4f}")
    
    print("\nRetrace demo complete!")


if __name__ == "__main__":
    demo_retrace()
```

**출력:**

```
============================================================
Retrace(λ) Demo
============================================================

--- Retrace Target Computation ---
  λ=0.00: targets = [4.526 5.151 4.769 6.31 ]...
  λ=0.50: targets = [6.115 7.623 5.635 7.886]...
  λ=0.95: targets = [9.425 9.623 5.397 6.804]...
  λ=1.00: targets = [9.32  9.255 4.788 6.156]...

--- Trace Coefficients ---
  For greedy target policy with ε-greedy behavior (ε=0.1, |A|=4):
    Greedy action: μ=0.925, π=1.0, c=λ·min(1,π/μ)=0.950
    Non-greedy:    μ=0.025, π=0.0, c=λ·min(1,π/μ)=0.000
    → Trace is cut when behavior took non-greedy action

--- Effective Trace Length ---
    λ=0.5, P(greedy)=0.5: expected trace = 1.3 steps
    λ=0.5, P(greedy)=0.8: expected trace = 1.7 steps
    λ=0.5, P(greedy)=0.95: expected trace = 1.9 steps
    λ=0.8, P(greedy)=0.5: expected trace = 1.7 steps
    λ=0.8, P(greedy)=0.8: expected trace = 2.8 steps
    λ=0.8, P(greedy)=0.95: expected trace = 4.2 steps
    λ=0.95, P(greedy)=0.5: expected trace = 1.9 steps
    λ=0.95, P(greedy)=0.8: expected trace = 4.2 steps
    λ=0.95, P(greedy)=0.95: expected trace = 10.3 steps
    λ=1.0, P(greedy)=0.5: expected trace = 2.0 steps
    λ=1.0, P(greedy)=0.8: expected trace = 5.0 steps
    λ=1.0, P(greedy)=0.95: expected trace = 20.0 steps

--- Retrace vs N-step (with off-policy data) ---
  Retrace targets: [6.843 5.95  7.684 6.843 5.95 ]
  N-step targets:  [12.321 11.774 11.192 10.574  9.916]
  Difference: 2.5774

Retrace demo complete!
```

## 2. 논의

이 구현의 중심은 `compute_retrace_targets` 다. 궤적 하나에서 TD 오차 $\delta_k = r_k + \gamma(1-d_k)\max_{a'}Q(s_{k+1}, a') - Q(s_k, a_k)$ 를 구하고, 스텝 $t$ 의 타깃을 $Q^{\text{ret}}(s_t, a_t) = Q(s_t, a_t) + \sum_{k \ge t}\gamma^{k-t}\big(\prod_{i=t+1}^{k} c_i\big)\delta_k$ 로 더한다. 트레이스 계수 $c_i = \lambda\min(1, \pi(a_i|s_i)/\mu(a_i|s_i))$ 는 중요도 비를 1 에서 잘라 분산이 커지지 않게 하고, 타깃 폴리시가 고르지 않았을 행동($\pi = 0$)을 만나면 0 이 되어 그 뒤의 TD 오차를 끊는다. 에피소드가 끝난 스텝을 넘어서도 끊는다. `compute_retrace_batch` 와 `TrajectoryBuffer` 는 이 함수를 학습 루프에 붙이기 위한 도구로, 온라인 네트워크의 그리디 행동을 타깃 폴리시로 삼아 $\pi(a_t|s_t)$ 를 0 또는 1 로 만든다. 데모에서는 이 둘을 쓰지 않는다.

데모 함수는 네 부분이며 모두 합성 데이터로 계산만 한다(학습은 하지 않는다). (1) 무작위 Q 값으로 만든 길이 8 의 궤적에서 $\lambda$ 를 0, 0.5, 0.95, 1 로 바꿔 타깃을 찍는다. 이 궤적은 $\pi = 0.8$, $\mu = 0.5$ 라 잘린 비가 늘 1 이고 $c_i = \lambda$ 다. $\lambda = 0$ 이면 한 스텝 타깃 $r_t + \gamma\max_{a'}Q(s_{t+1}, a')$ 과 같다. (2) $\varepsilon = 0.1$, 행동 4 개인 $\varepsilon$-그리디 행동 폴리시에서 그리디 행동은 $c = 0.95$, 그 밖의 행동은 $c = 0$ 임을 보인다. (3) $\lambda$ 와 그리디 행동 비율에 따른 트레이스의 기대 길이 $1/(1 - \lambda p)$ 를 표로 찍는다. (4) 일부 행동이 그리디가 아니었던 궤적에서 Retrace 타깃과, 보정 없이 모든 $c_i = \lambda$ 로 둔 타깃을 견준다. 출력의 「N-step targets」는 이름과 달리 $n$-step 리턴이 아니라 보정하지 않은 $\lambda$-리턴이다(연습문제 3).

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
데모의 「Retrace vs N-step」 부분은 $Q = 5$, 다음 상태의 $\max Q = 5$, 리워드 1, $\gamma = 0.99$, $\lambda = 0.95$ 이고, 타깃 폴리시 확률이 `[1, 1, 0, 1, 1, 0, 0, 1, 1, 1]`, 행동 폴리시 확률이 모두 0.3 이다. 스텝 0 과 스텝 2 의 Retrace 타깃을 손으로 계산해 출력과 맞춰 보라. 출력에서 스텝 0 과 3, 스텝 1 과 4 의 타깃이 같은 까닭은 무엇인가?

</div>

??? success "풀이"
    모든 스텝에서 $\delta = 1 + 0.99 \times 5 - 5 = 0.95$ 이다. 잘린 비는 $\pi = 1$ 이면 $\min(1, 1/0.3) = 1$ 이라 $c = 0.95$, $\pi = 0$ 이면 $c = 0$ 이다. 스텝 0 은 $5 + 0.95 + 0.99 \times 0.95 \times 0.95 = 6.843$ 이고, 스텝 2 에서 $c_2 = 0$ 이라 거기서 끊긴다. 스텝 2 는 자기 자신의 $\delta_2$ 는 곱 없이 들어가므로 $5 + 0.95 + 0.99 \times 0.95 \times 0.95 + 0.99^2 \times 0.95^2 \times 0.95 = 5.95 + 0.8935 + 0.8403 = 7.684$ 이고, $c_5 = 0$ 에서 끊긴다. 출력의 `6.843`, `7.684` 와 맞다. 스텝 0 과 3 은 둘 다 「자기 스텝 + 한 스텝 더」 뒤에 $c = 0$ 을 만나고, 스텝 1 과 4 는 바로 다음 스텝에서 끊겨 자기 $\delta$ 하나만 더한다($5.95$). 값이 모두 같은 합성 궤적이라 끊기는 자리까지의 길이만이 타깃을 정한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
데모의 「Effective Trace Length」는 기대 길이를 $1/(1 - \lambda p)$ 로 계산한다($p$ 는 행동 폴리시가 그리디 행동을 고를 확률). 타깃 폴리시가 그리디이고 $\mu(\text{그리디}) < 1$ 일 때 이 식이 $\mathbb{E}\big[\sum_{k \ge 0}\prod_{i=1}^{k} c_i\big]$ 와 같음을 보여라. 「Trace Coefficients」 부분의 행동 폴리시($\varepsilon = 0.1$, 행동 4 개)와 $\lambda = 0.95$ 이면 기대 길이는 얼마인가?

</div>

??? success "풀이"
    그리디 행동이면 $\pi/\mu = 1/\mu > 1$ 이라 잘려서 $c = \lambda$, 그 밖의 행동이면 $\pi = 0$ 이라 $c = 0$ 이다. 행동이 스텝마다 독립으로 뽑힌다고 보면 $\prod_{i=1}^{k} c_i$ 는 확률 $p^k$ 로 $\lambda^k$, 그 밖에는 0 이므로 기댓값이 $(\lambda p)^k$ 이다. 이것을 $k = 0$ 부터 더하면 $\sum_k(\lambda p)^k = 1/(1 - \lambda p)$ 다. 엄밀히는 「끊기기까지의 스텝 수」라기보다 트레이스 가중치의 기대 합이다. $\varepsilon$-그리디에서 그리디 행동의 확률은 $p = 1 - 0.1 + 0.1/4 = 0.925$ 이므로 $1/(1 - 0.95 \times 0.925) \approx 8.2$ 스텝이다. 표의 $\lambda = 0.95$, $p = 0.95$ 줄(10.3 스텝)보다 조금 짧다. $\varepsilon$ 이 클수록 트레이스가 빨리 끊겨 Retrace 는 한 스텝 Q 러닝에 가까워진다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
데모가 「N-step targets」로 찍는 값은 `target_probs` 와 `behavior_probs` 를 둘 다 1 로 넘겨 얻은 것이다. 이때 $c_i$ 는 얼마이고, 이 타깃은 실제로 무엇인가? 같은 궤적에서 이 값(스텝 0 에서 12.321)이 Retrace 타깃(6.843)보다 훨씬 큰 까닭을 설명하고, 어느 쪽이 그리디 폴리시의 $Q$ 를 배우는 데 맞는 타깃인지 논하라.

</div>

??? success "풀이"
    비가 1 이므로 $c_i = \lambda = 0.95$ 로 늘 같고, 타깃은 $Q + \sum_{k=0}^{9}(\gamma\lambda)^k\delta_k$, 곧 행동 폴리시가 무엇이었는지 보지 않는 $\lambda$-리턴(Peng 의 $Q(\lambda)$ 에 가까운 것)이다. 진짜 $n$-step 리턴이라면 처음 $n$ 개 리워드를 가중치 없이 더하고 $\gamma^n\max Q$ 로 부트스트랩해야 한다. 스텝 0 에서 $5 + 0.95 \times (1 - 0.9405^{10})/(1 - 0.9405) \approx 5 + 7.32 = 12.32$ 로 출력과 맞는다. 이 합성 궤적은 $\delta = 0.95 > 0$ 이라 더하는 항이 많을수록 타깃이 커진다. Retrace 는 그리디가 아니었던 스텝 2, 5, 6 에서 트레이스를 끊어 항을 두세 개만 더하고, 보정 없는 쪽은 열 개를 다 더한다. 그리디 폴리시의 $Q$ 를 배우려면 Retrace 쪽이 맞다. 스텝 2 의 행동은 그리디 폴리시라면 고르지 않았을 것이므로, 그 뒤의 리워드는 행동 폴리시의 리턴이지 그리디 폴리시의 리턴이 아니다. 보정 없는 타깃은 분산이 작고 리워드를 멀리 전파하지만 행동 폴리시 쪽으로 치우친다. 「n-step 리턴」 쪽의 「5. 오프폴리시 보정」이 말하는 문제가 이것이다.

## 정리하며

**다룬 것** — Retrace

이 구현은 잘린 중요도 비 $c_i = \lambda\min(1, \pi/\mu)$ 로 오프폴리시 데이터의 여러 스텝 타깃을 계산하는 `compute_retrace_targets` 와, 그것을 학습에 붙이는 `compute_retrace_batch`·`TrajectoryBuffer` 를 담는다.

핵심 함수는 `compute_retrace_targets` 이고 핵심 클래스는 `TrajectoryBuffer` 이며, 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
