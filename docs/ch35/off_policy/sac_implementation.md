# SAC 구현

부드러운 행위자-비평가(SAC)는 이어진 다스리기에서 앞서가는 벗어난 방침 알고리즘으로, 가장 큰 엔트로피 힘 북돋우는 배움과 행위자-비평가 얼개를 엮는다. SAC은 어림 돌아옴과 방침 엔트로피를 함께 가장 크게 하여, 쌍둥이 Q 그물과 다시 매개변수 매기기로 뽑는 눌러 담은 가우스 방침, 저절로 벼리는 온도로 든든한 익힘을 지키면서 살펴보기를 북돋운다. 서비스 품질의 이 구현은 SAC의 한가운데 조각을 모두 Pendulum 둘레에서 보이고, 저절로 벼리는 엔트로피 온도와 붙박인 온도를 견준다.

## 1. 코드

```python
"""
34.4.4장: SAC 온전한 구현
=============================================
저절로 벼리는 온도, 눌러 담은 가우스 방침, 쌍둥이 비평가를
갖춘 서비스 품질의 SAC.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import gymnasium as gym
from collections import deque

# ========================================================================
# 메인
# ========================================================================


LOG_STD_MIN, LOG_STD_MAX = -20, 2
EPS = 1e-6


class ReplayBuffer:
    def __init__(self, capacity, obs_dim, act_dim):
        self.cap = capacity
        self.idx = self.size = 0
        self.s = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.a = np.zeros((capacity, act_dim), dtype=np.float32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.s2 = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.d = np.zeros(capacity, dtype=np.float32)
    
    def add(self, s, a, r, s2, d):
        self.s[self.idx], self.a[self.idx] = s, a
        self.r[self.idx], self.s2[self.idx], self.d[self.idx] = r, s2, float(d)
        self.idx = (self.idx + 1) % self.cap
        self.size = min(self.size + 1, self.cap)
    
    def sample(self, n):
        i = np.random.randint(0, self.size, n)
        return (torch.FloatTensor(self.s[i]), torch.FloatTensor(self.a[i]),
                torch.FloatTensor(self.r[i]), torch.FloatTensor(self.s2[i]),
                torch.FloatTensor(self.d[i]))


# ---------------------------------------------------------------------------
# 그물
# ---------------------------------------------------------------------------

class SquashedGaussianActor(nn.Module):
    """눌러 담은 가우스 방침을 쓰는 SAC 확률 행위자."""
    
    def __init__(self, obs_dim, act_dim, hidden=256, max_action=1.0):
        super().__init__()
        self.max_action = max_action
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden, act_dim)
        self.log_std_head = nn.Linear(hidden, act_dim)
    
    def forward(self, obs):
        h = self.net(obs)
        mu = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std
    
    def sample(self, obs):
        """
        다시 매개변수 매기기로 움직임을 뽑고 로그 낌새를 셈한다.
        
        돌려주는 값: 움직임, 로그 낌새, 평균 움직임
        """
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mu, std)
        
        # 다시 매개변수 매긴 뽑기
        u = dist.rsample()
        action = torch.tanh(u) * self.max_action
        
        # tanh 바로잡기를 곁들인 로그 낌새
        # log π(a|s) = log N(u; μ, σ) - Σ log(1 - tanh²(u))
        log_prob = dist.log_prob(u)
        log_prob -= torch.log(1 - (action / self.max_action).pow(2) + EPS)
        log_prob = log_prob.sum(dim=-1)
        
        mean_action = torch.tanh(mu) * self.max_action
        
        return action, log_prob, mean_action


class TwinQCritic(nn.Module):
    """SAC를 위한 쌍둥이 Q 그물."""
    
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    
    def forward(self, obs, action):
        sa = torch.cat([obs, action], -1)
        return self.q1(sa).squeeze(-1), self.q2(sa).squeeze(-1)


# ---------------------------------------------------------------------------
# SAC 부림꾼
# ---------------------------------------------------------------------------

class SAC:
    """
    저절로 벼리는 엔트로피를 갖춘 부드러운 행위자-비평가.
    
    매개변수
    ----------
    env : gym.Env
        이어진 움직임 둘레.
    lr : float
        온 그물의 배움률.
    gamma : float
        깎기 인자.
    tau : float
        부드러운 과녁 고침 계수.
    alpha_lr : float
        온도의 배움률.
    init_alpha : float
        온도의 첫 값.
    buffer_size : int
        되돌려 보기 버퍼 크기.
    batch_size : int
        익힘 작은 묶음 크기.
    warmup_steps : int
        익히기 앞서 벌이는 아무 살펴보기 걸음.
    auto_alpha : bool
        온도를 저절로 벼릴지 여부.
    """
    
    def __init__(
        self,
        env: gym.Env,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha_lr=3e-4,
        init_alpha=0.2,
        hidden_dim=256,
        buffer_size=1000000,
        batch_size=256,
        warmup_steps=5000,
        auto_alpha=True,
    ):
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.auto_alpha = auto_alpha
        
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])
        
        # 행위자
        self.actor = SquashedGaussianActor(obs_dim, act_dim, hidden_dim, max_action)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        
        # 쌍둥이 비평가
        self.critic = TwinQCritic(obs_dim, act_dim, hidden_dim)
        self.critic_target = TwinQCritic(obs_dim, act_dim, hidden_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # 온도(alpha)
        if auto_alpha:
            self.target_entropy = -act_dim  # 어림 규칙: -dim(A)
            self.log_alpha = torch.tensor(np.log(init_alpha), requires_grad=True)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = init_alpha
        
        self.buffer = ReplayBuffer(buffer_size, obs_dim, act_dim)
    
    def select_action(self, obs, deterministic=False):
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            action, _, mean_action = self.actor.sample(obs_t)
        if deterministic:
            return mean_action.numpy().flatten()
        return action.numpy().flatten()
    
    def _soft_update(self):
        for tp, sp in zip(self.critic_target.parameters(), self.critic.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)
    
    def update(self):
        if self.buffer.size < self.batch_size:
            return {}
        
        s, a, r, s2, d = self.buffer.sample(self.batch_size)
        
        # === 비평가 고침 ===
        with torch.no_grad():
            next_a, next_log_prob, _ = self.actor.sample(s2)
            target_q1, target_q2 = self.critic_target(s2, next_a)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target = r + self.gamma * (1 - d) * target_q
        
        q1, q2 = self.critic(s, a)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # === 행위자 고침 ===
        new_a, log_prob, _ = self.actor.sample(s)
        q1_new, q2_new = self.critic(s, new_a)
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (self.alpha * log_prob - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # === 온도 고침 ===
        alpha_loss = 0.0
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha.exp() * (log_prob.detach() + self.target_entropy)).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp().item()
            alpha_loss = alpha_loss.item()
        
        # 과녁을 부드럽게 고친다
        self._soft_update()
        
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss,
            "alpha": self.alpha,
            "entropy": -log_prob.mean().item(),
        }
    
    def train(self, total_steps=200000, print_interval=10000):
        obs, _ = self.env.reset()
        ep_rewards, recent = [], deque(maxlen=100)
        ep_r = 0.0
        
        for step in range(1, total_steps + 1):
            if step < self.warmup_steps:
                action = self.env.action_space.sample()
            else:
                action = self.select_action(obs)
            
            next_obs, reward, term, trunc, _ = self.env.step(action)
            done = term or trunc
            self.buffer.add(obs, action, reward, next_obs, done)
            ep_r += reward
            
            if step >= self.warmup_steps:
                metrics = self.update()
            
            obs = next_obs
            if done:
                ep_rewards.append(ep_r)
                recent.append(ep_r)
                ep_r = 0.0
                obs, _ = self.env.reset()
            
            if step % print_interval == 0 and recent:
                m = metrics if step >= self.warmup_steps else {}
                print(
                    f"Step {step:>8d} | "
                    f"Avg(100): {np.mean(recent):>8.1f} | "
                    f"α: {self.alpha:>6.4f} | "
                    f"H: {m.get('entropy', 0):>6.3f}"
                )
        
        return ep_rewards
    
    def evaluate(self, n_episodes=10):
        env = gym.make(self.env.spec.id)
        rewards = []
        for _ in range(n_episodes):
            obs, _ = env.reset()
            total_r, done = 0.0, False
            while not done:
                action = self.select_action(obs, deterministic=True)
                obs, r, term, trunc, _ = env.step(action)
                total_r += r
                done = term or trunc
            rewards.append(total_r)
        env.close()
        return np.mean(rewards), np.std(rewards)


# ---------------------------------------------------------------------------
# 보여 주기
# ---------------------------------------------------------------------------

def demo_sac():
    print("=" * 60)
    print("SAC on Pendulum-v1")
    print("=" * 60)
    
    env = gym.make("Pendulum-v1")
    agent = SAC(
        env=env, lr=3e-4, gamma=0.99, tau=0.005,
        auto_alpha=True, warmup_steps=5000,
        batch_size=256, hidden_dim=256,
    )
    
    rewards = agent.train(total_steps=100000, print_interval=10000)
    
    mean_r, std_r = agent.evaluate(n_episodes=20)
    print(f"\nEvaluation: {mean_r:.1f} ± {std_r:.1f}")
    
    env.close()
    return rewards


def demo_sac_alpha_comparison():
    """저절로 벼리는 온도와 붙박인 온도를 견준다."""
    print("\n" + "=" * 60)
    print("SAC: Auto vs Fixed Temperature")
    print("=" * 60)
    
    for auto, alpha in [(True, 0.2), (False, 0.2), (False, 1.0)]:
        env = gym.make("Pendulum-v1")
        agent = SAC(env=env, auto_alpha=auto, init_alpha=alpha, warmup_steps=5000)
        rewards = agent.train(total_steps=50000, print_interval=100000)
        env.close()
        
        label = f"auto (init={alpha})" if auto else f"fixed={alpha}"
        final = np.mean(rewards[-30:]) if len(rewards) >= 30 else 0
        print(f"  α={label:<20}: final avg = {final:.1f}")


if __name__ == "__main__":
    demo_sac()
```

## 2. 논의

이 구현은 SAC 온전한 구현의 한가운데 논리를 담은 `ReplayBuffer`, `SquashedGaussianActor`, `TwinQCritic` 클래스를 축으로 삼는다. 코드는 알고리즘 조각을 보여 주기와 따지기 논리에서 갈라놓는 조각 설계를 따른다.

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
    이 얼개 고름은 벗어난 방침 방법에서 자리 잡은 좋은 버릇을 비춘다. 보기로 ReLU 활성은 곧지 않음을 주면서 0보다 큰 들임에서 기울기가 사라지는 것을 막는다. 손실 함수는 일감 갈래에 맞추어 고른다(갈래 나누기에는 사귐 엔트로피, 되돌이에는 평균 제곱 잘못). 다른 것으로 바꾸면(보기로 시그모이드 활성, L1 손실) 가장 좋게 하기 지형이 바뀌어 됨됨이가 나빠질 수 있으나, 어떤 자리에서는 바꾸는 것이 이로울 수도 있다.

---

**연습문제 3.**
이 구현을 더 만만치 않은 자리로 넓혀라. 더 큰 자료 뭉치, 다른 문제 갈래, 덧붙인 기능 가운데 하나를 고르라. 고친 바를 밝히고 됨됨이에 미친 바를 따져라.

??? success "연습문제 3 풀이"
    절로 떠오르는 넓히기 하나는 정칙화(드롭아웃, 무게 삭임)나 더 야무진 얼개(켜 더하기, 건너뛰는 이음)를 더하는 것이다. 고른 넓히기를 만들고 같은 자료로 익힌 뒤 앞뒤의 재기를 견주어라. 이 넓히기는 처음 알고리즘과 고친 바의 이치 밑뜻을 모두 아는 것을 보여야 한다.

## 정리하며

**다룬 것** — SAC 구현

이 구현은 SAC 온전한 구현의 한가운데 논리를 담은 `ReplayBuffer`, `SquashedGaussianActor`, `TwinQCritic` 클래스를 축으로 삼는다.

고갱이 갈래는 `ReplayBuffer`, `SquashedGaussianActor`, `TwinQCritic`, `SAC`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
