# DQN 짜기

실제로 쓸 수 있는 온전한 DQN 짜기는 겪음 되돌려 보기, 과녁 그물, 기울기 자르기, 짜임 있는 따지기를 하나의 익히기 흐름으로 모은다. 이 짜기는 손실, Q 값, 기울기 크기의 두루 기록하기, 욕심쟁이 방침으로 하는 때때로의 따지기, 되풀이할 수 있도록 모델 되짚기 표시를 담는다. 후버 손실과 여느 웃잡을 모두 갖추고 CartPole-v1에서 온전한 익히기 고리를 보여 준다.

## 1. 코드

```python
"""
33.1.4 DQN 짜기
==========================

적기까지 갖춘, 참으로 굴릴 수 있는 온전한 DQN 짜기,
CartPole-v1에서 값 매기기와 중간 갈무리.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from collections import deque, namedtuple
from typing import Tuple, List, Dict, Optional
import random
import time
import json
import os

# ========================================================================
# 메인
# ========================================================================

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


# ---------------------------------------------------------------------------
# 되돌려 보기 담개
# ---------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int):
        self.capacity = capacity
        self.size = 0
        self.ptr = 0
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.states[idx]),
            torch.LongTensor(self.actions[idx]),
            torch.FloatTensor(self.rewards[idx]),
            torch.FloatTensor(self.next_states[idx]),
            torch.FloatTensor(self.dones[idx]),
        )

    def __len__(self):
        return self.size


# ---------------------------------------------------------------------------
# Q 그물
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: Tuple[int, ...] = (128, 128)):
        super().__init__()
        layers = []
        prev = state_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, action_dim))
        self.net = nn.Sequential(*layers)

        # 사비에르 초기화
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# 온전한 DQN 부림꾼
# ---------------------------------------------------------------------------

class DQNAgent:
    """겪음 되돌려 보기, 과녁 그물, 기록을 갖춘 온전한 DQN 부림꾼."""

    def __init__(self, state_dim: int, action_dim: int,
                 # 그물
                 hidden_dims: Tuple[int, ...] = (128, 128),
                 # 학습
                 lr: float = 1e-3,
                 gamma: float = 0.99,
                 batch_size: int = 64,
                 # 되돌려 보기 담개
                 buffer_capacity: int = 50000,
                 min_buffer_size: int = 1000,
                 # 과녁 그물
                 target_update_freq: int = 200,
                 # 살펴보기
                 eps_start: float = 1.0,
                 eps_end: float = 0.01,
                 eps_decay_steps: int = 10000,
                 # 그 밖
                 grad_clip: float = 10.0,
                 loss_fn: str = 'huber',  # 'mse' 또는 'huber'
                 device: str = 'cpu'):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.min_buffer_size = min_buffer_size
        self.target_update_freq = target_update_freq
        self.grad_clip = grad_clip
        self.device = device

        # 엡실론 차례표
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps

        # 그물
        self.online_net = QNetwork(state_dim, action_dim, hidden_dims).to(device)
        self.target_net = QNetwork(state_dim, action_dim, hidden_dims).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        # 다듬개와 손실
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        if loss_fn == 'huber':
            self.loss_fn = nn.SmoothL1Loss()
        else:
            self.loss_fn = nn.MSELoss()

        # 되돌려 보기 담개
        self.buffer = ReplayBuffer(buffer_capacity, state_dim)

        # 셈틀
        self.total_steps = 0
        self.update_count = 0

        # 기록
        self.log: Dict[str, List] = {
            'losses': [],
            'q_values': [],
            'grad_norms': [],
            'epsilons': [],
        }

    @property
    def epsilon(self) -> float:
        frac = min(1.0, self.total_steps / self.eps_decay_steps)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.online_net(state_t).argmax(dim=1).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)
        self.total_steps += 1

    def train_step(self) -> Optional[float]:
        """익히기 한 걸음을 한다. 손실을 돌려주되 담개가 너무 작으면 None을 돌려준다."""
        if len(self.buffer) < self.min_buffer_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # 고른 움직임의 지금 Q 값
        q_values = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # 과녁 Q 값
        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1)[0]
            targets = rewards + (1 - dones) * self.gamma * next_q

        # 손실
        loss = self.loss_fn(q_values, targets)

        # 최적화
        self.optimizer.zero_grad()
        loss.backward()

        # 기울기 자르기
        grad_norm = nn.utils.clip_grad_norm_(self.online_net.parameters(), self.grad_clip)

        self.optimizer.step()
        self.update_count += 1

        # 과녁 그물 고치기
        if self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        # 기록
        loss_val = loss.item()
        self.log['losses'].append(loss_val)
        self.log['q_values'].append(q_values.mean().item())
        self.log['grad_norms'].append(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
        self.log['epsilons'].append(self.epsilon)

        return loss_val

    def evaluate(self, env, n_episodes: int = 10) -> Dict[str, float]:
        """욕심쟁이 방침(ε=0)으로 따진다."""
        returns = []
        lengths = []
        for _ in range(n_episodes):
            state, _ = env.reset()
            total_r = 0.0
            steps = 0
            done = False
            while not done:
                action = self.select_action(state, training=False)
                state, r, term, trunc, _ = env.step(action)
                total_r += r
                steps += 1
                done = term or trunc
            returns.append(total_r)
            lengths.append(steps)
        return {
            'mean_return': np.mean(returns),
            'std_return': np.std(returns),
            'min_return': np.min(returns),
            'max_return': np.max(returns),
            'mean_length': np.mean(lengths),
        }

    def save(self, path: str):
        torch.save({
            'online_net': self.online_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
            'update_count': self.update_count,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt['online_net'])
        self.target_net.load_state_dict(ckpt['target_net'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.total_steps = ckpt['total_steps']
        self.update_count = ckpt['update_count']


# ---------------------------------------------------------------------------
# 학습 루프
# ---------------------------------------------------------------------------

def train_dqn(env_name: str = 'CartPole-v1',
              n_episodes: int = 500,
              eval_freq: int = 50,
              eval_episodes: int = 10,
              seed: int = 42,
              **agent_kwargs) -> Tuple[DQNAgent, Dict]:
    """값 매기기를 포함한 온전한 DQN 익히기 되돌이."""

    # 모든 씨앗 정하기
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make(env_name)
    eval_env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim, **agent_kwargs)

    # 학습 기록
    history = {
        'episode_rewards': [],
        'episode_lengths': [],
        'eval_returns': [],
        'eval_episodes': [],
        'wall_time': [],
    }

    start_time = time.time()

    for ep in range(1, n_episodes + 1):
        state, _ = env.reset()
        total_reward = 0
        ep_length = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store_transition(state, action, reward, next_state, float(done))
            agent.train_step()
            state = next_state
            total_reward += reward
            ep_length += 1

        history['episode_rewards'].append(total_reward)
        history['episode_lengths'].append(ep_length)
        history['wall_time'].append(time.time() - start_time)

        # 때맞춰 값 매기기
        if ep % eval_freq == 0:
            eval_result = agent.evaluate(eval_env, eval_episodes)
            history['eval_returns'].append(eval_result['mean_return'])
            history['eval_episodes'].append(ep)

            # 흐르는 평균
            recent = history['episode_rewards'][-50:]
            print(f"Episode {ep:>4d} | "
                  f"Avg50: {np.mean(recent):>7.1f} | "
                  f"Eval: {eval_result['mean_return']:>7.1f} ± {eval_result['std_return']:.1f} | "
                  f"ε: {agent.epsilon:.3f} | "
                  f"Steps: {agent.total_steps:,} | "
                  f"Loss: {np.mean(agent.log['losses'][-100:]):.4f} | "
                  f"Q̄: {np.mean(agent.log['q_values'][-100:]):.2f}")

    env.close()
    eval_env.close()
    return agent, history


# ---------------------------------------------------------------------------
# 시연
# ---------------------------------------------------------------------------

def demo_dqn_implementation():
    """CartPole에서 온전한 DQN 익히기 보이기."""
    print("=" * 70)
    print("DQN Implementation Demo — CartPole-v1")
    print("=" * 70)

    agent, history = train_dqn(
        env_name='CartPole-v1',
        n_episodes=300,
        eval_freq=50,
        lr=1e-3,
        gamma=0.99,
        batch_size=64,
        buffer_capacity=50000,
        min_buffer_size=1000,
        target_update_freq=200,
        eps_start=1.0,
        eps_end=0.01,
        eps_decay_steps=5000,
        loss_fn='huber',
    )

    # --- 마지막 값 매기기 ---
    print("\n--- Final Evaluation (20 episodes, greedy) ---")
    eval_env = gym.make('CartPole-v1')
    final_eval = agent.evaluate(eval_env, n_episodes=20)
    eval_env.close()
    for k, v in final_eval.items():
        print(f"  {k}: {v:.2f}")

    # --- 익히기 셈밝힘 ---
    print("\n--- Training Statistics ---")
    rewards = history['episode_rewards']
    print(f"  Total episodes: {len(rewards)}")
    print(f"  Total steps: {agent.total_steps:,}")
    print(f"  Total updates: {agent.update_count:,}")
    print(f"  Best episode reward: {max(rewards):.0f}")
    print(f"  Last 50 avg: {np.mean(rewards[-50:]):.1f}")

    if agent.log['losses']:
        print(f"  Final loss (avg 100): {np.mean(agent.log['losses'][-100:]):.4f}")
    if agent.log['q_values']:
        print(f"  Final Q-value (avg 100): {np.mean(agent.log['q_values'][-100:]):.2f}")

    # --- 중간 갈무리 ---
    save_path = 'dqn_cartpole.pt'
    agent.save(save_path)
    print(f"\n  Checkpoint saved to {save_path}")

    # --- 불러오기 확인 ---
    agent2 = DQNAgent(4, 2)
    agent2.load(save_path)
    eval_env = gym.make('CartPole-v1')
    loaded_eval = agent2.evaluate(eval_env, n_episodes=5)
    eval_env.close()
    print(f"  Loaded checkpoint eval: {loaded_eval['mean_return']:.1f}")

    print("\nDQN implementation demo complete!")


if __name__ == "__main__":
    demo_dqn_implementation()
```

## 2. 논의

이 짜기는 핵심 논리를 감싼 `ReplayBuffer`, `QNetwork`, `DQNAgent` 갈래를 한가운데 둔다. 코드는 알고리즘 조각을 보여 주기와 따지기 논리에서 떼어 놓는 조각 짜기를 따른다.

보여 주기 함수는 이 조각들을 여느 힘 북돋우는 배움 잣대에 실제로 쓰는 모습을 보인다. 내놓기를 살피면 웃잡 고름과 문제 짜임에 따라 알고리즘의 성능이 어떻게 달라지는지 볼 수 있다.

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
    이 얼개 고르기는 깊은 Q 그물에서 자리 잡은 가장 좋은 방식을 따른다. 보기로 ReLU 깨움은 비선형을 주면서 양의 들임에서 기울기가 사라지는 것을 피한다. 손실 함수는 일의 갈래에 맞게 고른다(가름에는 교차 엔트로피, 되돌이 맞춤에는 평균 제곱 어긋남). 다른 것으로 바꾸면(보기로 시그모이드 깨움, L1 손실) 가장 좋게 하기의 풍경이 바뀌어 성능이 나빠질 수 있지만 어떤 상황에서는 바꿈이 이로울 수도 있다.

---

**연습문제 3.**
더 어려운 상황을 다루도록 짜기를 넓혀라. 더 큰 자료 묶음, 다른 문제 변형, 특징 하나 더하기 가운데 하나이다. 고침을 적고 성능에 미친 영향을 따져라.

??? success "연습문제 3 풀이"
    자연스러운 넓힘 하나는 규칙 세우기(떨구기, 무게 줄임)나 더 정교한 얼개(층 더하기, 건너뛰는 이음)를 더하는 것이다. 고른 넓힘을 짜고 같은 자료로 익힌 뒤 앞뒤의 잣대를 견주어라. 이 넓힘은 본디 알고리즘과 고침의 이론 까닭을 모두 이해했음을 보여야 한다.

## 정리하며

**다룬 것** — DQN 짜기

이 짜기는 핵심 논리를 감싼 `ReplayBuffer`, `QNetwork`, `DQNAgent` 갈래를 한가운데 둔다.

고갱이 갈래는 `ReplayBuffer`, `QNetwork`, `DQNAgent`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
