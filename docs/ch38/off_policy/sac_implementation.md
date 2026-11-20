# SAC 구현

Soft Actor-Critic(SAC)은 연속 제어에서 최고 수준의 오프폴리시 알고리즘으로, 최대 엔트로피 강화학습과 Actor-Critic 구조를 엮는다. SAC은 기대 리턴과 폴리시 엔트로피를 함께 최대화한다. 트윈 Q 네트워크, 재파라미터화로 샘플링하는 tanh 압축 가우스 폴리시, 자동으로 조정되는 온도 덕분에 학습이 강건하면서도 탐험이 살아 있다. 이 구현은 SAC의 핵심 구성 요소를 모두 갖추고 Pendulum 둘레에서 돌린다. 자동으로 조정되는 온도와 고정된 온도를 견주는 함수(`demo_sac_alpha_comparison`)도 담았지만, `__main__` 은 `demo_sac` 만 돌린다.

## 1. 코드

```python
"""
34.4.4장: SAC 온전한 구현
=============================================
저절로 벼리는 온도, 눌러 담은 가우스 폴리시, 쌍둥이 크리틱을
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
# 네트워크
# ---------------------------------------------------------------------------

class SquashedGaussianActor(nn.Module):
    """눌러 담은 가우스 폴리시를 쓰는 SAC 확률 액터."""
    
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
        다시 파라메터 매기기로 액션을 뽑고 로그 확률을 셈한다.
        
        돌려주는 값: 액션, 로그 확률, 평균 액션
        """
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mu, std)
        
        # 다시 파라메터 매긴 뽑기
        u = dist.rsample()
        action = torch.tanh(u) * self.max_action
        
        # tanh 바로잡기를 곁들인 로그 확률
        # log π(a|s) = log N(u; μ, σ) - Σ log(1 - tanh²(u))
        log_prob = dist.log_prob(u)
        log_prob -= torch.log(1 - (action / self.max_action).pow(2) + EPS)
        log_prob = log_prob.sum(dim=-1)
        
        mean_action = torch.tanh(mu) * self.max_action
        
        return action, log_prob, mean_action


class TwinQCritic(nn.Module):
    """SAC를 위한 쌍둥이 Q 네트워크."""
    
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
# SAC 에이전트
# ---------------------------------------------------------------------------

class SAC:
    """
    저절로 벼리는 엔트로피를 갖춘 부드러운 액터-크리틱.
    
    파라메터
    ----------
    env : gym.Env
        이어진 액션 둘레.
    lr : float
        온 네트워크의 러닝레이트.
    gamma : float
        깎기 인자.
    tau : float
        부드러운 타깃 고침 계수.
    alpha_lr : float
        온도의 러닝레이트.
    init_alpha : float
        온도의 첫 값.
    buffer_size : int
        되돌려 보기 버퍼 크기.
    batch_size : int
        익힘 작은 배치 크기.
    warmup_steps : int
        익히기 앞서 벌이는 아무 살펴보기 스텝.
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
        
        # 액터
        self.actor = SquashedGaussianActor(obs_dim, act_dim, hidden_dim, max_action)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        
        # 쌍둥이 크리틱
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
        
        # === 크리틱 고침 ===
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
        
        # === 액터 고침 ===
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
        
        # 타깃을 부드럽게 고친다
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

이 구현은 SAC의 핵심 논리를 담은 `ReplayBuffer`, `SquashedGaussianActor`, `TwinQCritic`, `SAC` 클래스를 축으로 삼는다. 코드는 알고리즘 구성 요소를 데모·평가 논리와 분리하는 모듈식 설계를 따른다.

데모 함수는 이 구성 요소들을 전형적인 강화학습 벤치마크인 Pendulum-v1에 적용한다. `demo_sac` 은 1만 스텝마다 최근 100 에피소드 평균 리턴, 온도 α, 엔트로피를 찍고 끝에 결정론적 폴리시로 20 에피소드를 평가한다. `demo_sac_alpha_comparison` 을 따로 부르면 온도 설정에 따라 성능이 어떻게 달라지는지 볼 수 있다.

실용적 관점에서 이 구현은 순수한 성능보다 명료함을 앞세운다. 프로덕션 시스템은 배치 계산, GPU 가속, 더 정교한 하이퍼파라미터 튜닝 같은 최적화를 더 넣는 것이 보통이다. 그렇더라도 여기서 보인 핵심 알고리즘 아이디어는 대규모 응용에 그대로 옮겨 간다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
데모 코드를 돌리고 중요한 출력 지표를 기록하라. 하이퍼파라미터 하나(러닝레이트, 은닉 차원, 레이어 개수 따위)를 바꿔 결과가 어떻게 달라지는지 설명하라.

</div>

??? success "풀이"
    데모를 돌린 뒤 다른 것은 고정하고 고른 하이퍼파라미터만 체계적으로 바꾼다. 예를 들어 은닉 차원을 두 배로 늘리면 표현력이 커지지만 계산 시간이 는다. 러닝레이트는 단조롭지 않은 결과를 낳는다. 너무 작으면 더디게 수렴하고 너무 크면 진동한다. 고른 하이퍼파라미터의 서로 다른 값 적어도 셋에 대해 구체적인 수를 기록해 두라.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
이 구현에서 중요한 구조 선택이 맡은 역할을 설명하라. 왜 그런 활성화 함수, 로그 표준편차 범위 제한, 로스 함수를 쓰는가? 다른 것으로 바꾸면 무슨 일이 생기는가?

</div>

??? success "풀이"
    이 구조 선택은 오프폴리시 방법에서 자리 잡은 모범 사례를 따른다. ReLU 활성화는 비선형성을 주면서 양의 입력에서 그래디언트가 사라지는 것을 막는다. 로그 표준편차를 [-20, 2]로 자르면 표준편차가 0에 붙거나 터지지 않아 로그 확률이 수치적으로 안정된다. 크리틱 로스는 소프트 벨만 타깃에 대한 평균 제곱 오차(TD 회귀)이고, 액터 로스는 α log π 에서 두 Q의 최솟값을 뺀 값의 평균이다. 다른 것으로 바꾸면(예를 들어 시그모이드 활성화, 크리틱에 후버 로스) 최적화 지형이 바뀌어 성능이 나빠질 수 있으나, 리워드 규모가 큰 과제에서는 후버 로스가 오히려 안정적일 수도 있다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
이 구현을 더 어려운 상황으로 확장하라. 더 어려운 환경(예를 들어 HalfCheetah), 다른 문제 유형, 새 기능 가운데 하나를 고르라. 수정 사항을 설명하고 성능에 미친 영향을 분석하라.

</div>

??? success "풀이"
    자연스러운 확장 하나는 정칙화(드롭아웃, 웨이트 감쇠)나 더 정교한 구조(레이어 추가, 스킵 연결)를 더하는 것이다. 고른 확장을 구현하고 같은 환경과 시드로 학습한 뒤 전후의 지표를 비교하여라. 이 확장은 원래 알고리즘과 수정 사항의 이론적 근거를 모두 이해했음을 보여야 한다.

## 정리하며

**다룬 것** — SAC 구현

이 구현은 SAC의 핵심 논리를 담은 `ReplayBuffer`, `SquashedGaussianActor`, `TwinQCritic` 클래스를 `SAC` 에이전트로 묶는다.

앞의 연습문제 세 개로 이 구성 요소들을 직접 바꿔 보며 확인할 수 있다.
