# 계층적 강화학습

계층적 강화학습은 긴 과제를 여러 시간 규모로 나누어 푸는 고급 강화학습 기법이다. 이 쪽의 중심은 옵션 프레임워크의 Option-Critic(베이컨 외, 2017)이다. 옵션은 「어떤 폴리시로 움직이다가 언제 끝낼지」를 한데 묶은, 시간적으로 확장된 액션이다. 코드는 옵션 안 폴리시, 종료 함수 β(s), 옵션 밸류 Q(s, ω)를 함께 배워 CartPole-v1을 풀고, 목표 조건부 폴리시 π(a|s, g)의 뼈대(`GoalConditionedPolicy`)도 함께 적는다. 다만 데모는 이 목표 조건부 폴리시를 쓰지 않는다.

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

데모 함수는 CartPole-v1에서 500 에피소드를 학습하며 100 에피소드마다 최근 평균 리턴과 옵션별 사용 비율(%)을 찍는다. 사용 비율이 한 옵션에 몰리는지 보면 옵션들이 실제로 나뉘어 쓰이는지 알 수 있다.

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
이 구현에서 중요한 구조 선택이 맡은 역할을 설명하라. 왜 그런 활성화 함수, 그래디언트 클리핑, 로스 함수를 쓰는가? 다른 것으로 바꾸면 무슨 일이 생기는가?

</div>

??? success "풀이"
    ReLU 활성화는 비선형성을 주면서 양의 입력에서 그래디언트가 사라지는 것을 막는다. 종료 함수만 시그모이드를 쓰는 것은 그 출력이 확률 β(s)여야 하기 때문이다. 로스는 셋을 더한 것이다. 옵션 밸류 Q(s, ω)는 옵션 밸류 타깃에 대한 제곱 오차로 배우고, 옵션 안 폴리시는 어드밴티지를 곱한 로그 확률로 배우며, 종료 로스는 지금 옵션이 가장 좋은 옵션보다 못할수록(Q(s, ω) − max Q < 0) β를 키워 그 옵션을 일찍 끝내게 한다. 에피소드 전체 로스를 한 번에 역전파하므로 그래디언트 노름을 1.0으로 잘라 큰 갱신을 막는다. 클리핑을 빼거나 종료 로스의 계수 0.01을 키우면 옵션이 너무 자주 바뀌어 사실상 단일 폴리시로 무너질 수 있다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
이 구현을 더 어려운 상황으로 확장하라. 더 어려운 환경, 다른 문제 유형, 새 기능 가운데 하나를 고르라. 수정 사항을 설명하고 성능에 미친 영향을 분석하라.

</div>

??? success "풀이"
    자연스러운 확장 하나는 코드에 정의만 해 둔 `GoalConditionedPolicy` 를 써서 목표 조건부 과제(예를 들어 격자 세계에서 목표 칸 찾기)를 푸는 것이다. 또는 정칙화(드롭아웃, 웨이트 감쇠)나 더 정교한 구조를 더할 수도 있다. 고른 확장을 구현하고 같은 환경과 시드로 학습한 뒤 전후의 지표를 비교하여라. 이 확장은 원래 알고리즘과 수정 사항의 이론적 근거를 모두 이해했음을 보여야 한다.

## 정리하며

**다룬 것** — 계층적 강화학습

이 구현은 계층적 강화학습의 핵심 논리를 담은 `OptionPolicy`, `TerminationFunction`, `OptionCritic` 클래스를 축으로 삼는다.

이 클래스들은 `OptionCriticAgent` 가 묶어 학습시키며, 앞의 연습문제 세 개로 직접 바꿔 보며 확인할 수 있다.
