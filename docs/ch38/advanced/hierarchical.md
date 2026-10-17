# 계층적 강화학습

켜 있는 강화학습은 높은 그래프 뉴럴 넷트웍 재주에서 중요한 생각이다. 목표에 매인 폴리시를 다룬다. 이 구현은 여기에 걸린 중요한 알고리즘과 자료 구조를 손으로 만져 보이며, 이치 기초와 실제로 서비스에 올릴 때 살필 것을 함께 보여 준다.

## 1. 코드

```python
"""
34.5.2장: 켜 있는 힘 북돋우는 러닝
=====================================================
고르기 틀과 목표에 매인 폴리시를 쓰는 쉬운 켜 있는 힘 북돋우는
러닝.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import gymnasium as gym
from collections import deque

# ========================================================================
# 메인
# ========================================================================


class OptionPolicy(nn.Module):
    """고르기 하나에 딸린 고르기 안 폴리시."""
    def __init__(self, obs_dim, act_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )
    def forward(self, obs):
        return Categorical(logits=self.net(obs))


class TerminationFunction(nn.Module):
    """고르기를 끝낼 확률 β(s)."""
    def __init__(self, obs_dim, n_options, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, n_options),
        )
    def forward(self, obs):
        return torch.sigmoid(self.net(obs))


class OptionCritic(nn.Module):
    """
    고르기-크리틱 구조(베이컨 외, 2017).
    
    고르기(고르기 안 폴리시 + 끝내기 함수)와 고르기에 대한 폴리시를
    한꺼번에 배운다.
    """
    def __init__(self, obs_dim, act_dim, n_options=4, hidden=64):
        super().__init__()
        self.n_options = n_options
        
        # 고르기에 대한 폴리시
        self.option_policy = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, n_options),
        )
        
        # 고르기 안 폴리시
        self.options = nn.ModuleList([
            OptionPolicy(obs_dim, act_dim, hidden) for _ in range(n_options)
        ])
        
        # 끝내기 함수
        self.termination = TerminationFunction(obs_dim, n_options, hidden)
        
        # 고르기마다의 값 함수 Q(s, ω)
        self.q_options = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, n_options),
        )
    
    def get_option(self, obs):
        """Q(s, ω)에 대한 엡실론-욕심쟁이로 고르기를 고른다."""
        q = self.q_options(obs)
        return q.argmax(dim=-1)
    
    def get_action(self, obs, option_idx):
        """고르기 안 폴리시에서 액션을 얻는다."""
        dist = self.options[option_idx](obs)
        action = dist.sample()
        return action, dist.log_prob(action)
    
    def should_terminate(self, obs, option_idx):
        """고르기를 끝내야 하는지 살핀다."""
        beta = self.termination(obs)
        terminate_prob = beta[:, option_idx]
        return torch.bernoulli(terminate_prob).bool()


class OptionCriticAgent:
    """고르기-크리틱 익힘 에이전트."""
    
    def __init__(self, env, n_options=4, lr=1e-3, gamma=0.99, hidden=64):
        self.env = env
        self.gamma = gamma
        self.n_options = n_options
        
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n
        
        self.model = OptionCritic(obs_dim, act_dim, n_options, hidden)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.epsilon = 0.1
    
    def train(self, n_episodes=1000, print_interval=100):
        rewards_history = []
        recent = deque(maxlen=100)
        option_usage = np.zeros(self.n_options)
        
        for ep in range(1, n_episodes + 1):
            obs, _ = self.env.reset()
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            
            # 첫 고르기를 고른다
            if np.random.random() < self.epsilon:
                current_option = np.random.randint(self.n_options)
            else:
                with torch.no_grad():
                    current_option = self.model.get_option(obs_t).item()
            
            total_reward = 0.0
            done = False
            transitions = []
            
            while not done:
                # 지금 고르기에서 액션을 얻는다
                action, log_prob = self.model.get_action(obs_t, current_option)
                
                next_obs, reward, term, trunc, _ = self.env.step(action.item())
                done = term or trunc
                next_obs_t = torch.FloatTensor(next_obs).unsqueeze(0)
                
                transitions.append((obs_t, current_option, action, log_prob, reward, next_obs_t, done))
                option_usage[current_option] += 1
                total_reward += reward
                
                # 끝내기를 살핀다
                if not done:
                    with torch.no_grad():
                        should_term = self.model.should_terminate(next_obs_t, current_option).item()
                    
                    if should_term:
                        if np.random.random() < self.epsilon:
                            current_option = np.random.randint(self.n_options)
                        else:
                            with torch.no_grad():
                                current_option = self.model.get_option(next_obs_t).item()
                
                obs_t = next_obs_t
            
            # 고치기
            self._update(transitions)
            
            rewards_history.append(total_reward)
            recent.append(total_reward)
            
            if ep % print_interval == 0:
                usage = option_usage / option_usage.sum() * 100
                print(
                    f"Episode {ep:>5d} | Avg(100): {np.mean(recent):>7.1f} | "
                    f"Options: {usage.round(1)}"
                )
        
        return rewards_history
    
    def _update(self, transitions):
        total_loss = torch.tensor(0.0)
        
        for obs, opt, act, lp, reward, next_obs, done in transitions:
            with torch.no_grad():
                q_next = self.model.q_options(next_obs).squeeze(0)
                beta_next = self.model.termination(next_obs).squeeze(0)[opt]
                
                # 고르기 값: (1-β)Q(s',ω) + β max_ω' Q(s',ω')
                if done:
                    target = reward
                else:
                    target = reward + self.gamma * (
                        (1 - beta_next) * q_next[opt] + beta_next * q_next.max()
                    )
            
            q_current = self.model.q_options(obs).squeeze(0)[opt]
            q_loss = (q_current - target).pow(2)
            
            # 폴리시 로스
            advantage = (target - q_current).detach()
            policy_loss = -lp * advantage
            
            # 끝내기 로스
            beta = self.model.termination(obs).squeeze(0)[opt]
            q_omega = self.model.q_options(obs).squeeze(0)
            term_advantage = q_omega[opt] - q_omega.max()
            term_loss = beta * term_advantage.detach()
            
            total_loss = total_loss + q_loss + policy_loss + 0.01 * term_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()


# ---------------------------------------------------------------------------
# 목표에 매인 폴리시
# ---------------------------------------------------------------------------

class GoalConditionedPolicy(nn.Module):
    """목표에 매인 폴리시: π(a|s, g)."""
    
    def __init__(self, obs_dim, goal_dim, act_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + goal_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )
    
    def forward(self, obs, goal):
        x = torch.cat([obs, goal], dim=-1)
        return Categorical(logits=self.net(x))


def demo_option_critic():
    print("=" * 60)
    print("Option-Critic on CartPole-v1")
    print("=" * 60)
    
    env = gym.make("CartPole-v1")
    agent = OptionCriticAgent(env, n_options=4, lr=1e-3, gamma=0.99)
    rewards = agent.train(n_episodes=500, print_interval=100)
    env.close()
    
    if len(rewards) >= 100:
        print(f"\nFinal avg (last 100): {np.mean(rewards[-100:]):.1f}")


if __name__ == "__main__":
    demo_option_critic()
```

## 2. 논의

이 구현은 켜 있는 강화학습의 한가운데 논리를 담은 `OptionPolicy`, `TerminationFunction`, `OptionCritic` 클래스를 축으로 삼는다. 코드는 알고리즘 조각을 보여 주기와 따지기 논리에서 갈라놓는 조각 설계를 따른다.

보여 주기 함수는 이 조각들을 여느 강화학습 잣대에 실제로 써 보인다. 그 출력을 살피면 파라메터 공정성과 문제 구조에 따라 알고리즘의 성능이 어떻게 달라지는지 볼 수 있다.

쓰임의 눈으로 보면 이 구현은 날 성능보다 또렷함을 앞세운다. 서비스 시스템은 배치 계산, GPU 빠르게 하기, 더 야무진 파라메터 벼리기 같은 다듬기를 더 넣는 것이 보통이다. 그렇더라도 여기서 보인 한가운데 알고리즘 생각은 큰 잣대의 쓰임새에 그대로 옮겨 간다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
보여 주기 코드를 돌리고 중요한 출력 재기를 적어라. 파라메터 하나(러닝레이트, 숨은 차원, 켜 개수 따위)를 고쳐 열매가 어떻게 달라지는지 밝혀라.

</div>

??? success "풀이"
    보여 주기를 돌린 뒤 다른 것을 붙박아 두고 고른 파라메터만 구조 있게 바꾼다. 보기로 숨은 차원을 곱절로 늘리면 나타내는 그릇이 커지지만 셈하는 때가 는다. 러닝레이트는 한결같지 않은 결과를 낳는다. 너무 작으면 더디게 모이고 너무 크면 들쭉날쭉해진다. 고른 파라메터의 서로 다른 값 적어도 셋에 대해 또렷한 수를 적어 두라.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
이 구현에서 중요한 구조 공정성이 맡은 몫을 풀어라. 왜 그런 엑티베이션 함수, 고르게 하기 꾀, 로스 함수를 쓰는가? 다른 것으로 바꾸면 무슨 일이 생기는가?

</div>

??? success "풀이"
    이 구조 공정성은 높은 그래프 뉴럴 넷트웍 재주에서 자리 잡은 좋은 버릇을 비춘다. 보기로 ReLU 활성은 곧지 않음을 주면서 0보다 큰 입력에서 그래디언트가 사라지는 것을 막는다. 로스 함수는 과제 클래스에 맞추어 고른다(분류에는 교차 엔트로피, 회귀에는 평균 제곱 오차). 다른 것으로 바꾸면(보기로 시그모이드 활성, L1 로스) 최적화 지형이 바뀌어 성능이 나빠질 수 있으나, 어떤 자리에서는 바꾸는 것이 이로울 수도 있다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
이 구현을 더 만만치 않은 자리로 넓혀라. 더 큰 데이터셋, 다른 문제 클래스, 덧붙인 기능 가운데 하나를 고르라. 고친 바를 밝히고 성능에 미친 바를 따져라.

</div>

??? success "풀이"
    절로 떠오르는 넓히기 하나는 정규화(드롭아웃, 웨이트 삭임)나 더 야무진 구조(켜 더하기, 건너뛰는 이음)를 더하는 것이다. 고른 넓히기를 만들고 같은 데이터로 학습된 뒤 앞뒤의 재기를 비교하여라. 이 넓히기는 처음 알고리즘과 고친 바의 이치 밑뜻을 모두 아는 것을 보여야 한다.

## 정리하며

**다룬 것** — 켜 있는 강화학습

이 구현은 켜 있는 강화학습의 한가운데 논리를 담은 `OptionPolicy`, `TerminationFunction`, `OptionCritic` 클래스를 축으로 삼는다.

핵심 클래스는 `OptionPolicy`, `TerminationFunction`, `OptionCritic`, `OptionCriticAgent`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
