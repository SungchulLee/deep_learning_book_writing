# 자연 방침 기울기

자연 방침 기울기는 믿음 구역 방법에서 종요로운 생각이다. 행렬 셈하기와 맹탕 방침 기울기와의 견줌을 다룬다. 이 구현은 여기에 걸린 종요로운 알고리즘과 자료 얼개를 손으로 만져 보이며, 이치 바탕과 실제로 서비스에 올릴 때 살필 것을 함께 보여 준다.

## 코드

```python
"""
34.3.2장: 자연 방침 기울기
=========================================
피셔 정보 행렬 셈하기를 곁들인 자연 방침 기울기 구현과 맹탕
방침 기울기와의 견줌.
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import gymnasium as gym
from typing import List
from collections import deque

# ========================================================================
# 메인
# ========================================================================


# ---------------------------------------------------------------------------
# 그물
# ---------------------------------------------------------------------------

class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, act_dim),
        )
    
    def forward(self, obs):
        return Categorical(logits=self.net(obs))


class ValueNet(nn.Module):
    def __init__(self, obs_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, obs):
        return self.net(obs).squeeze(-1)


# ---------------------------------------------------------------------------
# 피셔 정보 도구
# ---------------------------------------------------------------------------

def compute_fisher_vector_product(policy, obs, v, damping=0.1):
    """
    거꾸로 퍼뜨리기를 두 번 하여 피셔-벡터 곱 Fv를 셈한다.
    
    F = E[∇log π ∇log π^T]
    Fv = ∇(∇KL · v)
    """
    dist = policy(obs)
    # 헤세 행렬을 셈하려고 제 자신과의 쿨백-라이블러 어긋남을 쓴다
    log_probs = dist.logits - dist.logits.logsumexp(dim=-1, keepdim=True)
    probs = dist.probs
    kl = (probs * log_probs).sum(-1).mean()
    
    params = list(policy.parameters())
    kl_grad = torch.autograd.grad(kl, params, create_graph=True)
    kl_grad_flat = torch.cat([g.reshape(-1) for g in kl_grad])
    
    kl_v = kl_grad_flat.dot(v)
    fvp_grads = torch.autograd.grad(kl_v, params)
    fvp = torch.cat([g.reshape(-1) for g in fvp_grads])
    
    return fvp + damping * v


def conjugate_gradient(fvp_fn, b, n_steps=10, residual_tol=1e-10):
    """켤레 기울기로 Fx = b를 푼다."""
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdotr = r.dot(r)
    
    for _ in range(n_steps):
        Fp = fvp_fn(p)
        alpha = rdotr / (p.dot(Fp) + 1e-8)
        x += alpha * p
        r -= alpha * Fp
        new_rdotr = r.dot(r)
        if new_rdotr < residual_tol:
            break
        p = r + (new_rdotr / (rdotr + 1e-8)) * p
        rdotr = new_rdotr
    
    return x


def compute_empirical_fisher(policy, obs, actions, n_samples=None):
    """
    겪음에서 얻은 피셔 행렬을 셈한다(작은 그물에서).
    
    F = (1/N) Σ ∇log π(a|s) ∇log π(a|s)^T
    
    매개변수가 적을 때에만 다룰 수 있다.
    """
    if n_samples is None:
        n_samples = len(obs)
    
    indices = np.random.choice(len(obs), min(n_samples, len(obs)), replace=False)
    
    grads = []
    for i in indices:
        policy.zero_grad()
        dist = policy(obs[i:i+1])
        log_prob = dist.log_prob(actions[i:i+1])
        log_prob.backward()
        
        grad = torch.cat([p.grad.reshape(-1) for p in policy.parameters()])
        grads.append(grad)
    
    grads = torch.stack(grads)
    fisher = grads.T @ grads / len(grads)
    
    return fisher


# ---------------------------------------------------------------------------
# 자연 방침 기울기 부림꾼
# ---------------------------------------------------------------------------

class NaturalPolicyGradient:
    """
    자연 방침 기울기 부림꾼.
    
    피셔 정보 행렬로 자연 기울기 쪽을 셈하여, 매개변수 나타내기에
    흔들리지 않는 고침을 준다.
    """
    
    def __init__(
        self,
        env: gym.Env,
        step_size: float = 0.01,
        gamma: float = 0.99,
        lam: float = 0.97,
        hidden_dim: int = 64,
        cg_iters: int = 10,
        cg_damping: float = 0.1,
        value_lr: float = 1e-3,
        value_epochs: int = 5,
    ):
        self.env = env
        self.step_size = step_size
        self.gamma = gamma
        self.lam = lam
        self.cg_iters = cg_iters
        self.cg_damping = cg_damping
        self.value_epochs = value_epochs
        
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n
        
        self.policy = PolicyNet(obs_dim, act_dim, hidden_dim)
        self.value_fn = ValueNet(obs_dim, hidden_dim)
        self.value_opt = torch.optim.Adam(self.value_fn.parameters(), lr=value_lr)
    
    def _flat_params(self):
        return torch.cat([p.data.reshape(-1) for p in self.policy.parameters()])
    
    def _set_flat_params(self, flat):
        idx = 0
        for p in self.policy.parameters():
            n = p.numel()
            p.data.copy_(flat[idx:idx + n].reshape(p.shape))
            idx += n
    
    def collect_data(self, n_steps=2048):
        states, actions, rewards, dones = [], [], [], []
        obs, _ = self.env.reset()
        ep_rewards = []
        ep_r = 0.0
        
        for _ in range(n_steps):
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                dist = self.policy(obs_t)
                action = dist.sample()
            
            next_obs, reward, term, trunc, _ = self.env.step(action.item())
            done = term or trunc
            
            states.append(obs)
            actions.append(action.item())
            rewards.append(reward)
            dones.append(float(done))
            ep_r += reward
            
            obs = next_obs
            if done:
                ep_rewards.append(ep_r)
                ep_r = 0.0
                obs, _ = self.env.reset()
        
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            ep_rewards,
        )
    
    def compute_gae(self, rewards, dones, values, last_value):
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            nv = last_value if t == T - 1 else values[t + 1]
            nt = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * nv * nt - values[t]
            advantages[t] = gae = delta + self.gamma * self.lam * nt * gae
        return torch.FloatTensor(advantages), torch.FloatTensor(advantages + values)
    
    def update_policy(self, obs, actions, advantages):
        """자연 방침 기울기 고침."""
        # 방침 기울기를 셈한다
        dist = self.policy(obs)
        log_probs = dist.log_prob(actions)
        surrogate = (log_probs * advantages).mean()
        
        grads = torch.autograd.grad(surrogate, self.policy.parameters())
        pg = torch.cat([g.reshape(-1) for g in grads])
        
        if pg.norm() < 1e-8:
            return 0.0
        
        # 켤레 기울기로 자연 기울기를 셈한다
        fvp_fn = lambda v: compute_fisher_vector_product(
            self.policy, obs, v, self.cg_damping
        )
        natural_grad = conjugate_gradient(fvp_fn, pg, self.cg_iters)
        
        # 쿨백-라이블러 매임에 바탕을 둔 걸음 크기: α = sqrt(2δ / g^T F^{-1} g)
        sFs = pg.dot(natural_grad)
        if sFs <= 0:
            return 0.0
        
        alpha = torch.sqrt(2 * self.step_size / (sFs + 1e-8))
        
        # 매개변수를 고친다
        old_params = self._flat_params()
        new_params = old_params + alpha * natural_grad
        self._set_flat_params(new_params)
        
        return (alpha * natural_grad).norm().item()
    
    def update_value(self, obs, returns):
        for _ in range(self.value_epochs):
            pred = self.value_fn(obs)
            loss = nn.functional.mse_loss(pred, returns)
            self.value_opt.zero_grad()
            loss.backward()
            self.value_opt.step()
    
    def train(self, n_iters=100, steps_per_iter=2048, print_interval=10):
        all_rewards = []
        recent = deque(maxlen=100)
        
        for it in range(1, n_iters + 1):
            obs, actions, rewards, dones, ep_rewards = self.collect_data(steps_per_iter)
            
            with torch.no_grad():
                values = self.value_fn(obs).numpy()
                last_val = self.value_fn(obs[-1:]).item()
            
            advantages, returns = self.compute_gae(rewards, dones, values, last_val)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            step_norm = self.update_policy(obs, actions, advantages)
            self.update_value(obs, returns)
            
            for r in ep_rewards:
                all_rewards.append(r)
                recent.append(r)
            
            if it % print_interval == 0 and len(recent) > 0:
                print(
                    f"Iter {it:>4d} | Avg(100): {np.mean(recent):>7.1f} | "
                    f"Step: {step_norm:>8.5f}"
                )
        
        return all_rewards


# ---------------------------------------------------------------------------
# 견줌: 여느 방침 기울기와 자연 방침 기울기
# ---------------------------------------------------------------------------

class VanillaPG(NaturalPolicyGradient):
    """견주기 위한 여느 방침 기울기."""
    
    def __init__(self, env, lr=1e-3, **kwargs):
        super().__init__(env, **kwargs)
        self.pg_lr = lr
    
    def update_policy(self, obs, actions, advantages):
        dist = self.policy(obs)
        log_probs = dist.log_prob(actions)
        loss = -(log_probs * advantages).mean()
        
        # 여느 기울기 고침
        grads = torch.autograd.grad(loss, self.policy.parameters())
        with torch.no_grad():
            for p, g in zip(self.policy.parameters(), grads):
                p.data -= self.pg_lr * g
        
        return sum(g.norm().item() for g in grads)


def demo_npg():
    print("=" * 60)
    print("Natural Policy Gradient on CartPole-v1")
    print("=" * 60)
    
    env = gym.make("CartPole-v1")
    agent = NaturalPolicyGradient(env=env, step_size=0.01, gamma=0.99, lam=0.97)
    rewards = agent.train(n_iters=80, steps_per_iter=2048, print_interval=10)
    env.close()
    
    if len(rewards) >= 50:
        print(f"\nFinal avg (last 50): {np.mean(rewards[-50:]):.1f}")


def demo_comparison():
    print("\n" + "=" * 60)
    print("Standard PG vs Natural PG Comparison")
    print("=" * 60)
    
    n_iters = 50
    n_trials = 3
    
    for name, AgentClass, kwargs in [
        ("Vanilla PG", VanillaPG, {"lr": 1e-3}),
        ("Natural PG", NaturalPolicyGradient, {"step_size": 0.01}),
    ]:
        trial_results = []
        for trial in range(n_trials):
            torch.manual_seed(trial)
            env = gym.make("CartPole-v1")
            agent = AgentClass(env=env, gamma=0.99, lam=0.97, **kwargs)
            rewards = agent.train(n_iters=n_iters, steps_per_iter=2048, print_interval=n_iters + 1)
            env.close()
            trial_results.append(np.mean(rewards[-30:]) if len(rewards) >= 30 else 0)
        
        print(f"{name:<15}: {np.mean(trial_results):.1f} ± {np.std(trial_results):.1f}")


if __name__ == "__main__":
    demo_npg()
    demo_comparison()```

## 논의

이 구현은 자연 방침 기울기의 한가운데 논리를 담은 `PolicyNet`, `ValueNet`, `NaturalPolicyGradient` 클래스를 축으로 삼는다. 코드는 알고리즘 조각을 보여 주기와 따지기 논리에서 갈라놓는 조각 설계를 따른다.

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
    이 얼개 고름은 믿음 구역 방법에서 자리 잡은 좋은 버릇을 비춘다. 보기로 ReLU 활성은 곧지 않음을 주면서 0보다 큰 들임에서 기울기가 사라지는 것을 막는다. 손실 함수는 일감 갈래에 맞추어 고른다(갈래 나누기에는 사귐 엔트로피, 되돌이에는 평균 제곱 잘못). 다른 것으로 바꾸면(보기로 시그모이드 활성, L1 손실) 가장 좋게 하기 지형이 바뀌어 됨됨이가 나빠질 수 있으나, 어떤 자리에서는 바꾸는 것이 이로울 수도 있다.

---

**연습문제 3.**
이 구현을 더 만만치 않은 자리로 넓혀라. 더 큰 자료 뭉치, 다른 문제 갈래, 덧붙인 기능 가운데 하나를 고르라. 고친 바를 밝히고 됨됨이에 미친 바를 따져라.

??? success "연습문제 3 풀이"
    절로 떠오르는 넓히기 하나는 정칙화(드롭아웃, 무게 삭임)나 더 야무진 얼개(켜 더하기, 건너뛰는 이음)를 더하는 것이다. 고른 넓히기를 만들고 같은 자료로 익힌 뒤 앞뒤의 재기를 견주어라. 이 넓히기는 처음 알고리즘과 고친 바의 이치 밑뜻을 모두 아는 것을 보여야 한다.
