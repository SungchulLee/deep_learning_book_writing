# Retrace

Retrace은 여러 걸음 때 차이 배우기의 벗어난 방침 바로잡기 방법으로, 다른 방침이 모은 자료에서 안전하고 흩어짐이 작은 돌아옴 어림을 준다. 잘라 낸 중요도 뽑기 비 $c_i = \lambda \min(1, \pi/\mu)$을 써서 움직임 방침이 과녁 방침에서 크게 벗어나면 자취를 저절로 끊는다. 이로써 움직임 방침과 상관없이 올바른 값 함수로 모임이 보장되어 겪음 되돌려 보기와 나눠 하는 배움 자리에 특히 쓸모 있다.

## 1. 코드

```python
"""
33.3.2 Retrace(λ)
==================

리트레이스(λ)로 방침 밖을 바로잡은 여러 걸음 돌아옴.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple

# ========================================================================
# 메인
# ========================================================================


def compute_retrace_targets(
    q_values: torch.Tensor,        # 자취의 t마다 Q(s_t, a_t), 꼴 (T,)
    next_q_max: torch.Tensor,      # t마다 max_a Q(s_{t+1}, a), 꼴 (T,)
    rewards: torch.Tensor,         # r_t, 꼴 (T,)
    dones: torch.Tensor,           # done_t, 꼴 (T,)
    target_policy_probs: torch.Tensor,  # π(a_t|s_t), 꼴 (T,)
    behavior_policy_probs: torch.Tensor,  # μ(a_t|s_t), 꼴 (T,)
    gamma: float = 0.99,
    lambda_: float = 0.95,
) -> torch.Tensor:
    """자취 하나의 리트레이스(λ) 과녁을 셈한다.
    
    인수:
        q_values: 때 걸음마다 (s_t, a_t)에서의 Q 값
        next_q_max: 때 걸음마다의 max_a' Q(s_{t+1}, a')
        rewards: 때 걸음마다의 보상
        dones: 끝남 깃발(1.0이면 끝났다)
        target_policy_probs: 과녁(이제) 방침 아래의 π(a_t | s_t)
        behavior_policy_probs: 행동(자료) 방침 아래의 μ(a_t | s_t)
        gamma: 깎기 인수
        lambda_: 리트레이스 람다 값
        
    반환값:
        리트레이스 과녁, 꼴 (T,)
    """
    T = len(rewards)
    
    # 자취 계수: c_i = λ * min(1, π/μ)
    is_ratios = target_policy_probs / (behavior_policy_probs + 1e-8)
    c = lambda_ * torch.clamp(is_ratios, max=1.0)
    
    # 때 차이 어긋남: δ_t = r_t + γ * max_a Q(s_{t+1}, a) - Q(s_t, a_t)
    td_errors = rewards + (1 - dones) * gamma * next_q_max - q_values
    
    # 리트레이스 과녁을 거꾸로 셈
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
    """자취 묶음의 리트레이스 과녁을 셈한다.
    
    인수:
        q_online: 온라인 Q 그물
        q_target: 과녁 Q 그물
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
            
            # 과녁 방침: 온라인 그물에 대해 욕심쟁이
            online_actions = q_online(states).argmax(dim=1)
            target_probs = (online_actions == actions).float()
        
        targets = compute_retrace_targets(
            q_vals, next_q_max, rewards, dones,
            target_probs, behavior_probs, gamma, lambda_
        )
        
        # 손실 셈을 위해 온라인 Q 값을 얻음
        online_q = q_online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        all_q.append(online_q)
        all_targets.append(targets)
    
    return torch.cat(all_q), torch.cat(all_targets)


# ---------------------------------------------------------------------------
# 리트레이스를 위한 자취 담개
# ---------------------------------------------------------------------------

class TrajectoryBuffer:
    """리트레이스 갈래 알고리즘을 위해 온전한 자취를 갈무리한다."""
    
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

    # --- 리트레이스 과녁 셈 ---
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
    
    # 욕심쟁이 움직임
    mu_greedy = 1 - eps + eps / n_actions
    c_greedy = 0.95 * min(1.0, 1.0 / mu_greedy)
    print(f"    Greedy action: μ={mu_greedy:.3f}, π=1.0, c=λ·min(1,π/μ)={c_greedy:.3f}")
    
    # 욕심쟁이 아닌 움직임
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
    
    # --- n걸음과 견주기 ---
    print("\n--- Retrace vs N-step (with off-policy data) ---")
    T = 10
    q_vals = torch.ones(T) * 5.0
    next_q_max = torch.ones(T) * 5.0
    rewards = torch.ones(T)
    dones = torch.zeros(T)
    
    # 방침 밖 흉내: 어떤 움직임은 욕심쟁이가 아니었음
    target_probs = torch.tensor([1, 1, 0, 1, 1, 0, 0, 1, 1, 1], dtype=torch.float32)
    behavior_probs = torch.ones(T) * 0.3
    
    retrace_targets = compute_retrace_targets(
        q_vals, next_q_max, rewards, dones, target_probs, behavior_probs)
    
    # n걸음(바로잡기 없음)
    nstep_targets = compute_retrace_targets(
        q_vals, next_q_max, rewards, dones,
        torch.ones(T), torch.ones(T))  # c_i = λ 늘
    
    print(f"  Retrace targets: {retrace_targets[:5].numpy().round(3)}")
    print(f"  N-step targets:  {nstep_targets[:5].numpy().round(3)}")
    print(f"  Difference: {(retrace_targets - nstep_targets).abs().mean():.4f}")
    
    print("\nRetrace demo complete!")


if __name__ == "__main__":
    demo_retrace()```

## 2. 논의

이 짜기는 Retrace의 핵심 논리를 감싼 `TrajectoryBuffer` 갈래를 한가운데 둔다. 코드는 알고리즘 조각을 보여 주기와 따지기 논리에서 떼어 놓는 조각 짜기를 따른다.

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

## 정리하며

**다룬 것** — Retrace

이 짜기는 Retrace의 핵심 논리를 감싼 `TrajectoryBuffer` 갈래를 한가운데 둔다.

고갱이 갈래는 `TrajectoryBuffer`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
