# 계층적 강화학습

계층적 강화학습은 고급 강화학습 기법에서 중요한 아이디어다. 목표 조건부 폴리시를 다룬다. 이 구현은 관련된 중요한 알고리즘과 자료 구조를 직접 다뤄 보이며, 이론적 기초와 실제 배포 시 고려할 점을 함께 보여 준다.

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

이 구현은 계층적 강화학습의 핵심 논리를 담은 `OptionPolicy`, `TerminationFunction`, `OptionCritic` 클래스를 축으로 삼는다. 코드는 알고리즘 구성 요소를 데모·평가 논리와 분리하는 모듈식 설계를 따른다.

데모 함수는 이 구성 요소들을 전형적인 강화학습 벤치마크에 실제로 적용해 보인다. 그 출력을 살피면 하이퍼파라미터 선택과 문제 구조에 따라 알고리즘의 성능이 어떻게 달라지는지 볼 수 있다.

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
이 구현에서 중요한 구조 선택이 맡은 역할을 설명하라. 왜 그런 활성화 함수, 정규화 전략, 로스 함수를 쓰는가? 다른 것으로 바꾸면 무슨 일이 생기는가?

</div>

??? success "풀이"
    이 구조 선택은 고급 강화학습 기법에서 자리 잡은 모범 사례를 따른다. 예를 들어 ReLU 활성화는 비선형성을 주면서 양의 입력에서 그래디언트가 사라지는 것을 막는다. 로스 함수는 과제 유형에 맞추어 고른다(분류에는 교차 엔트로피, 회귀에는 평균 제곱 오차). 다른 것으로 바꾸면(예를 들어 시그모이드 활성화, L1 로스) 최적화 지형이 바뀌어 성능이 나빠질 수 있으나, 어떤 상황에서는 바꾸는 것이 이로울 수도 있다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
이 구현을 더 어려운 상황으로 확장하라. 더 큰 데이터셋, 다른 문제 유형, 새 기능 가운데 하나를 고르라. 수정 사항을 설명하고 성능에 미친 영향을 분석하라.

</div>

??? success "풀이"
    자연스러운 확장 하나는 정칙화(드롭아웃, 웨이트 감쇠)나 더 정교한 구조(레이어 추가, 스킵 연결)를 더하는 것이다. 고른 확장을 구현하고 같은 데이터로 학습한 뒤 전후의 지표를 비교하여라. 이 확장은 원래 알고리즘과 수정 사항의 이론적 근거를 모두 이해했음을 보여야 한다.

## 정리하며

**다룬 것** — 계층적 강화학습

이 구현은 계층적 강화학습의 핵심 논리를 담은 `OptionPolicy`, `TerminationFunction`, `OptionCritic` 클래스를 축으로 삼는다.

핵심 클래스는 `OptionPolicy`, `TerminationFunction`, `OptionCritic`, `OptionCriticAgent`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
