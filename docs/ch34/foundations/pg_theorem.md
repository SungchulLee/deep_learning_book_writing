# 방침 기울기 정리

방침 기울기 정리는 방침 기울기 바탕에서 종요로운 생각이다. 점수 함수 어림개와 여러 갈래 기울기 어림 꾀를 다룬다. 이 구현은 여기에 걸린 종요로운 알고리즘과 자료 얼개를 손으로 만져 보이며, 이치 바탕과 실제로 서비스에 올릴 때 살필 것을 함께 보여 준다.

## 코드

```python
"""
34.1.2장: 방침 기울기 정리
========================================
방침 기울기 정리, 점수 함수 어림개, 여러 갈래 기울기 어림 꾀를
보여 준다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import numpy as np
import gymnasium as gym
from typing import List, Tuple

# ========================================================================
# 메인
# ========================================================================


# ---------------------------------------------------------------------------
# 점수 함수 어림개 보여 주기
# ---------------------------------------------------------------------------

class SimplePolicy(nn.Module):
    """방침 기울기 정리를 보이기 위한 쉬운 소프트맥스 방침."""
    
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
        )
    
    def forward(self, obs: torch.Tensor) -> Categorical:
        logits = self.net(obs)
        return Categorical(logits=logits)


def collect_trajectory(
    env: gym.Env,
    policy: SimplePolicy,
    max_steps: int = 200,
) -> Tuple[List, List, List]:
    """
    주어진 방침으로 자취 하나를 모은다.
    
    돌려주는 값
    -------
    log_probs : Tensor의 목록
        벌인 움직임의 로그 낌새.
    rewards : float의 목록
        걸음마다 받은 보상.
    states : ndarray의 목록
        들른 상태.
    """
    obs, _ = env.reset()
    log_probs, rewards, states = [], [], []
    
    for _ in range(max_steps):
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        dist = policy(obs_t)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        next_obs, reward, terminated, truncated, _ = env.step(action.item())
        
        log_probs.append(log_prob)
        rewards.append(reward)
        states.append(obs)
        
        obs = next_obs
        if terminated or truncated:
            break
    
    return log_probs, rewards, states


def compute_returns(rewards: List[float], gamma: float = 0.99) -> List[float]:
    """
    깎은 앞으로의 돌아옴 G_t = sum_{k=t}^T gamma^{k-t} r_k을 셈한다.
    
    이는 까닭 매김을 다듬은 것이다. G_t마다 때 t 뒤의 보상만 담는다.
    """
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns


# ---------------------------------------------------------------------------
# 방침 기울기 어림개
# ---------------------------------------------------------------------------

def pg_total_reward(
    log_probs: List[torch.Tensor],
    rewards: List[float],
    gamma: float = 0.99,
) -> torch.Tensor:
    """
    온 자취 보상을 쓰는 REINFORCE(까닭 매김 없음).
    
    기울기: sum_t [log pi(a_t|s_t) * R(tau)]
    
    치우치지 않지만 온 움직임에 온 자취 보상으로 무게를 주므로
    흩어짐이 아주 크다.
    """
    # 자취의 온 깎은 돌아옴
    R = sum(gamma**t * r for t, r in enumerate(rewards))
    
    # 방침 기울기 손실
    loss = -sum(lp * R for lp in log_probs)
    return loss


def pg_reward_to_go(
    log_probs: List[torch.Tensor],
    rewards: List[float],
    gamma: float = 0.99,
) -> torch.Tensor:
    """
    앞으로의 보상을 쓰는 REINFORCE(까닭 매김을 매김).
    
    기울기: sum_t [log pi(a_t|s_t) * G_t]
    
    움직임마다 앞날의 보상으로만 무게를 주므로 온 보상보다
    흩어짐이 작다.
    """
    returns = compute_returns(rewards, gamma)
    
    loss = -sum(lp * G for lp, G in zip(log_probs, returns))
    return loss


def pg_with_baseline(
    log_probs: List[torch.Tensor],
    rewards: List[float],
    baseline: float,
    gamma: float = 0.99,
) -> torch.Tensor:
    """
    상수 밑금을 빼는 REINFORCE.
    
    기울기: sum_t [log pi(a_t|s_t) * (G_t - b)]
    
    밑금 b는 기댓값을 바꾸지 않지만(치우치지 않음) b ≈ E[G_t]일 때
    흩어짐을 줄인다.
    """
    returns = compute_returns(rewards, gamma)
    
    loss = -sum(lp * (G - baseline) for lp, G in zip(log_probs, returns))
    return loss


# ---------------------------------------------------------------------------
# 로그 미분 재주 보여 주기
# ---------------------------------------------------------------------------

def demonstrate_log_derivative_trick():
    """
    풀어낸 기울기와 점수 함수 기울기를 견주어
    ∇_θ E_π[f(x)] = E_π[∇_θ log π(x) · f(x)]임을 보인다.
    """
    print("=" * 60)
    print("Log-Derivative Trick Verification")
    print("=" * 60)
    
    # 쉬운 1차원 경우: π_θ(x) = Categorical([θ, 1-θ])
    # f(x) = [3.0, 1.0] (움직임마다의 보상)
    
    theta = torch.tensor([0.6], requires_grad=True)
    f_values = torch.tensor([3.0, 1.0])
    
    # E[f(x)] = θ·3 + (1-θ)·1 = 2θ + 1의 풀어낸 기울기
    # ∇_θ E[f(x)] = 2.0
    analytical_grad = 2.0
    
    # 점수 함수 어림(몬테카를로)
    n_samples = 100000
    torch.manual_seed(42)
    
    probs = torch.cat([theta, 1 - theta])
    dist = Categorical(probs=probs)
    
    samples = dist.sample((n_samples,))
    log_probs = dist.log_prob(samples)
    rewards = f_values[samples]
    
    # ∇_θ E[f] ≈ (1/N) Σ ∇_θ log π(x_i) · f(x_i)
    surrogate_loss = -(log_probs * rewards).mean()
    surrogate_loss.backward()
    
    score_function_grad = -theta.grad.item()  # 가장 작게 했으므로 부호를 뒤집는다
    
    print(f"Analytical gradient:        {analytical_grad:.4f}")
    print(f"Score function estimate:    {score_function_grad:.4f}")
    print(f"Error:                      {abs(analytical_grad - score_function_grad):.4f}")


# ---------------------------------------------------------------------------
# 어림개별 흩어짐 견주기
# ---------------------------------------------------------------------------

def compare_estimator_variance():
    """
    기울기 어림을 여러 번 벌여 방침 기울기 어림개들의 흩어짐을
    견준다.
    """
    print("\n" + "=" * 60)
    print("Policy Gradient Estimator Variance Comparison")
    print("=" * 60)
    
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    torch.manual_seed(42)
    policy = SimplePolicy(obs_dim, act_dim)
    
    n_estimates = 50
    n_trajectories = 5
    gamma = 0.99
    
    grads_total = []
    grads_rtg = []
    grads_baseline = []
    
    for i in range(n_estimates):
        # 자취 묶음을 모은다
        batch_log_probs = []
        batch_rewards = []
        all_returns = []
        
        for _ in range(n_trajectories):
            lps, rews, _ = collect_trajectory(env, policy, max_steps=200)
            batch_log_probs.append(lps)
            batch_rewards.append(rews)
            all_returns.extend(compute_returns(rews, gamma))
        
        baseline = np.mean(all_returns)
        
        # 어림개마다 기울기를 셈한다
        for est_name, est_func, grads_list in [
            ("total", pg_total_reward, grads_total),
            ("rtg", pg_reward_to_go, grads_rtg),
            ("baseline", lambda lp, r: pg_with_baseline(lp, r, baseline), grads_baseline),
        ]:
            policy.zero_grad()
            total_loss = sum(
                est_func(lps, rews)
                for lps, rews in zip(batch_log_probs, batch_rewards)
            ) / n_trajectories
            total_loss.backward()
            
            # 기울기 노름을 모은다
            grad_norm = sum(
                p.grad.norm().item() ** 2 
                for p in policy.parameters() 
                if p.grad is not None
            ) ** 0.5
            grads_list.append(grad_norm)
    
    print(f"\nGradient norm statistics over {n_estimates} estimates:")
    print(f"{'Estimator':<15} {'Mean':>10} {'Std':>10} {'CV':>10}")
    print("-" * 45)
    for name, grads in [
        ("Total reward", grads_total),
        ("Reward-to-go", grads_rtg),
        ("With baseline", grads_baseline),
    ]:
        mean_g = np.mean(grads)
        std_g = np.std(grads)
        cv = std_g / (mean_g + 1e-8)
        print(f"{name:<15} {mean_g:>10.4f} {std_g:>10.4f} {cv:>10.4f}")
    
    env.close()


# ---------------------------------------------------------------------------
# 서로 다른 돌아옴 어림개로 기울기 어림하기
# ---------------------------------------------------------------------------

def demonstrate_gradient_computation():
    """방침 기울기를 걸음마다 셈해 보인다."""
    print("\n" + "=" * 60)
    print("Step-by-Step Policy Gradient Computation")
    print("=" * 60)
    
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    torch.manual_seed(0)
    policy = SimplePolicy(obs_dim, act_dim)
    
    # 자취 하나를 모은다
    log_probs, rewards, states = collect_trajectory(env, policy, max_steps=50)
    returns = compute_returns(rewards, gamma=0.99)
    
    T = len(rewards)
    print(f"\nTrajectory length: {T}")
    print(f"Total reward: {sum(rewards):.1f}")
    print(f"Discounted return G_0: {returns[0]:.4f}")
    
    # 앞쪽 몇 걸음을 보인다
    print(f"\n{'Step':>4} {'Reward':>8} {'G_t':>10} {'log π':>10}")
    print("-" * 36)
    for t in range(min(10, T)):
        print(f"{t:>4} {rewards[t]:>8.2f} {returns[t]:>10.4f} {log_probs[t].item():>10.4f}")
    
    if T > 10:
        print(f"  ... ({T - 10} more steps)")
    
    # 앞으로의 보상으로 방침 기울기를 셈한다
    policy.zero_grad()
    loss = pg_reward_to_go(log_probs, rewards, gamma=0.99)
    loss.backward()
    
    print(f"\nPolicy gradient (reward-to-go):")
    for name, param in policy.named_parameters():
        if param.grad is not None:
            print(f"  {name}: grad norm = {param.grad.norm().item():.6f}")
    
    env.close()


# ---------------------------------------------------------------------------
# 대리 손실 짓기
# ---------------------------------------------------------------------------

def demonstrate_surrogate_loss():
    """
    저절로 미분하도록 대리 손실을 어떻게 짓는지 보인다.
    
    종요로운 눈길: 돌아옴이나 이점을 거쳐 미분하지 않는다.
    L(θ) = -E[log π_θ(a|s) · Â]을 짓고 여느 가장 좋게 하는 개를 쓴다.
    """
    print("\n" + "=" * 60)
    print("Surrogate Loss Construction")
    print("=" * 60)
    
    # 흉내 낸 넘어감 묶음
    batch_size = 32
    obs_dim, act_dim = 4, 2
    
    torch.manual_seed(42)
    policy = SimplePolicy(obs_dim, act_dim)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    
    # 흉내 낸 자료(실제로는 둘레에서 모은다)
    obs = torch.randn(batch_size, obs_dim)
    actions = torch.randint(0, act_dim, (batch_size,))
    advantages = torch.randn(batch_size)  # 고르게 한 이점
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # 대리 손실을 짓는다
    dist = policy(obs)
    log_probs = dist.log_prob(actions)
    
    # 종요로움: 이점은 상수로 다룬다(떼어 놓는다)
    # 오직 log_probs만 θ을 거쳐 기울기를 나른다
    surrogate_loss = -(log_probs * advantages.detach()).mean()
    
    print(f"Batch size: {batch_size}")
    print(f"Surrogate loss: {surrogate_loss.item():.6f}")
    
    # 여느 기울기 내림 걸음
    optimizer.zero_grad()
    surrogate_loss.backward()
    
    grad_norm = sum(
        p.grad.norm().item() ** 2 
        for p in policy.parameters() 
        if p.grad is not None
    ) ** 0.5
    print(f"Gradient norm: {grad_norm:.6f}")
    
    optimizer.step()
    print("Parameter update applied via Adam optimizer.")


if __name__ == "__main__":
    demonstrate_log_derivative_trick()
    compare_estimator_variance()
    demonstrate_gradient_computation()
    demonstrate_surrogate_loss()```

## 논의

이 구현은 방침 기울기 정리의 한가운데 논리를 담은 `SimplePolicy` 클래스를 축으로 삼는다. 코드는 알고리즘 조각을 보여 주기와 따지기 논리에서 갈라놓는 조각 설계를 따른다.

보여 주기 함수는 이 조각들을 여느 힘 북돋우는 배움 잣대에 실제로 써 보인다. 그 출력을 살피면 매개변수 고름과 문제 얼개에 따라 알고리즘의 됨됨이가 어떻게 달라지는지 볼 수 있다.

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
    이 얼개 고름은 방침 기울기 바탕에서 자리 잡은 좋은 버릇을 비춘다. 보기로 ReLU 활성은 곧지 않음을 주면서 0보다 큰 들임에서 기울기가 사라지는 것을 막는다. 손실 함수는 일감 갈래에 맞추어 고른다(갈래 나누기에는 사귐 엔트로피, 되돌이에는 평균 제곱 잘못). 다른 것으로 바꾸면(보기로 시그모이드 활성, L1 손실) 가장 좋게 하기 지형이 바뀌어 됨됨이가 나빠질 수 있으나, 어떤 자리에서는 바꾸는 것이 이로울 수도 있다.

---

**연습문제 3.**
이 구현을 더 만만치 않은 자리로 넓혀라. 더 큰 자료 뭉치, 다른 문제 갈래, 덧붙인 기능 가운데 하나를 고르라. 고친 바를 밝히고 됨됨이에 미친 바를 따져라.

??? success "연습문제 3 풀이"
    절로 떠오르는 넓히기 하나는 정칙화(드롭아웃, 무게 삭임)나 더 야무진 얼개(켜 더하기, 건너뛰는 이음)를 더하는 것이다. 고른 넓히기를 만들고 같은 자료로 익힌 뒤 앞뒤의 재기를 견주어라. 이 넓히기는 처음 알고리즘과 고친 바의 이치 밑뜻을 모두 아는 것을 보여야 한다.
