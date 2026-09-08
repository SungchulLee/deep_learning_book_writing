# 과녁 그물

과녁 그물은 깊은 Q 그물의 중요한 개념이다. 굳은 고침과 부드러운 고침을 견주고 익히기 안정에 주는 영향을 보인다. 이 짜기는 얽힌 핵심 알고리즘과 자료 얼개를 손으로 만져 보게 하며 이론 바탕과 실제로 펼칠 때 살필 것을 함께 보여 준다.

## 1. 코드

```python
"""
33.1.3 과녁 그물
======================

딱딱한 과녁 그물 고치기와 부드러운 고치기, 그리고 그것이
익히기의 안정에 미치는 영향을 보인다.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from collections import deque, namedtuple
from typing import Tuple, Dict, List
import copy
import random

# ========================================================================
# 메인
# ========================================================================

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


# ---------------------------------------------------------------------------
# Q 그물
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# 되돌려 보기 담개
# ---------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        batch = Transition(*zip(*random.sample(self.buffer, batch_size)))
        return (
            torch.FloatTensor(np.array(batch.state)),
            torch.LongTensor(np.array(batch.action)),
            torch.FloatTensor(np.array(batch.reward)),
            torch.FloatTensor(np.array(batch.next_state)),
            torch.FloatTensor(np.array(batch.done, dtype=np.float32)),
        )

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# 과녁 그물 고치기 함수
# ---------------------------------------------------------------------------

def hard_update(target_net: nn.Module, online_net: nn.Module):
    """딱딱한 고치기: 온라인의 모든 값을 과녁으로 옮겨 적는다."""
    target_net.load_state_dict(online_net.state_dict())


def soft_update(target_net: nn.Module, online_net: nn.Module, tau: float = 0.005):
    """부드러운(폴랴크) 고치기: θ⁻ ← τθ + (1-τ)θ⁻"""
    for tp, op in zip(target_net.parameters(), online_net.parameters()):
        tp.data.copy_(tau * op.data + (1.0 - tau) * tp.data)


def compute_parameter_distance(net1: nn.Module, net2: nn.Module) -> float:
    """두 그물의 매개변수 사이 L2 거리."""
    dist = 0.0
    for p1, p2 in zip(net1.parameters(), net2.parameters()):
        dist += (p1 - p2).pow(2).sum().item()
    return np.sqrt(dist)


# ---------------------------------------------------------------------------
# 과녁 고치기 셈속을 고를 수 있는 DQN 부림꾼
# ---------------------------------------------------------------------------

class DQNAgent:
    """딱딱한 고치기와 부드러운 고치기를 모두 받치는 DQN 부림꾼."""

    def __init__(self, state_dim: int, action_dim: int,
                 lr: float = 1e-3, gamma: float = 0.99,
                 update_mode: str = 'hard',  # 'hard', 'soft', 또는 'none'
                 hard_update_freq: int = 100,
                 tau: float = 0.005,
                 buffer_capacity: int = 10000,
                 batch_size: int = 64,
                 eps_start: float = 1.0, eps_end: float = 0.01,
                 eps_decay: int = 5000):
        self.gamma = gamma
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.update_mode = update_mode
        self.hard_update_freq = hard_update_freq
        self.tau = tau

        # 그물
        self.online_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        hard_update(self.target_net, self.online_net)
        self.target_net.eval()  # 과녁 그물은 익히지 않음

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity)

        # 살펴보기
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.step_count = 0
        self.update_count = 0

        # 추적
        self.losses: List[float] = []
        self.target_distances: List[float] = []

    @property
    def epsilon(self) -> float:
        frac = min(1.0, self.step_count / self.eps_decay)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        if training:
            self.step_count += 1
            if random.random() < self.epsilon:
                return random.randrange(self.action_dim)
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            return self.online_net(state_t).argmax(dim=1).item()

    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def update(self) -> float:
        if len(self.buffer) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # 지금 Q 값
        q_values = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # 과녁 Q 값
        with torch.no_grad():
            if self.update_mode == 'none':
                # 과녁 그물 없음 — 과녁에 온라인 그물을 씀
                next_q = self.online_net(next_states).max(dim=1)[0]
            else:
                next_q = self.target_net(next_states).max(dim=1)[0]
            targets = rewards + (1 - dones) * self.gamma * next_q

        loss = nn.functional.mse_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        # 안정성을 위한 기울기 자르기
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()

        self.update_count += 1
        loss_val = loss.item()
        self.losses.append(loss_val)

        # 과녁 그물 고치기
        if self.update_mode == 'hard' and self.update_count % self.hard_update_freq == 0:
            hard_update(self.target_net, self.online_net)
        elif self.update_mode == 'soft':
            soft_update(self.target_net, self.online_net, self.tau)

        # 값 사이 거리 좇기
        dist = compute_parameter_distance(self.online_net, self.target_net)
        self.target_distances.append(dist)

        return loss_val


# ---------------------------------------------------------------------------
# 학습 루프
# ---------------------------------------------------------------------------

def train_agent(agent: DQNAgent, env_name: str = 'CartPole-v1',
                n_episodes: int = 300, min_buffer: int = 500) -> List[float]:
    """DQN 부림꾼을 익히고 마당 보상을 돌려준다."""
    env = gym.make(env_name)
    rewards_history = []

    for ep in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store(state, action, reward, next_state, float(done))
            if len(agent.buffer) >= min_buffer:
                agent.update()
            state = next_state
            total_reward += reward

        rewards_history.append(total_reward)

    env.close()
    return rewards_history


# ---------------------------------------------------------------------------
# 보이기: 과녁 고치기 셈속 견주기
# ---------------------------------------------------------------------------

def demo_target_networks():
    """서로 다른 과녁 그물 셈속으로 익히기를 견준다."""
    print("=" * 60)
    print("Target Networks Demo")
    print("=" * 60)

    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    env.close()

    configs = {
        'No Target Net': {'update_mode': 'none'},
        'Hard Update (C=100)': {'update_mode': 'hard', 'hard_update_freq': 100},
        'Hard Update (C=500)': {'update_mode': 'hard', 'hard_update_freq': 500},
        'Soft Update (τ=0.005)': {'update_mode': 'soft', 'tau': 0.005},
        'Soft Update (τ=0.05)': {'update_mode': 'soft', 'tau': 0.05},
    }

    n_episodes = 200
    results = {}

    for name, cfg in configs.items():
        print(f"\nTraining: {name}")
        # 고른 견주기를 위해 씨앗 고정
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

        agent = DQNAgent(state_dim, action_dim, lr=1e-3, **cfg)
        rewards = train_agent(agent, n_episodes=n_episodes)
        results[name] = {
            'rewards': rewards,
            'losses': agent.losses,
            'target_distances': agent.target_distances,
        }

        # 간추린 셈밝힘
        last_50 = rewards[-50:]
        print(f"  Last 50 episodes: mean={np.mean(last_50):.1f}, "
              f"std={np.std(last_50):.1f}, max={np.max(last_50):.0f}")
        if agent.losses:
            last_losses = agent.losses[-100:]
            print(f"  Recent loss: mean={np.mean(last_losses):.4f}")
        if agent.target_distances:
            print(f"  Final online-target distance: {agent.target_distances[-1]:.4f}")

    # --- 견주기 간추림 ---
    print("\n" + "=" * 60)
    print("Summary Comparison")
    print("=" * 60)
    print(f"{'Strategy':<25s} {'Mean(last50)':>12s} {'Std':>8s} {'Max':>6s}")
    print("-" * 55)
    for name, data in results.items():
        last_50 = data['rewards'][-50:]
        print(f"{name:<25s} {np.mean(last_50):>12.1f} {np.std(last_50):>8.1f} "
              f"{np.max(last_50):>6.0f}")

    # --- 값 좇기 보이기 ---
    print("\n--- Parameter Distance Evolution ---")
    for name, data in results.items():
        dists = data['target_distances']
        if dists:
            print(f"  {name}: start={dists[0]:.4f}, end={dists[-1]:.4f}, "
                  f"mean={np.mean(dists):.4f}")

    # --- 딱딱한 고치기와 부드러운 고치기의 얼개 ---
    print("\n--- Update Mechanics Illustration ---")
    net_a = QNetwork(4, 2)
    net_b = QNetwork(4, 2)
    hard_update(net_b, net_a)

    # net_a에 기울기 걸음 10번 흉내
    optimizer = optim.Adam(net_a.parameters(), lr=0.01)
    for _ in range(10):
        dummy_loss = net_a(torch.randn(16, 4)).sum()
        optimizer.zero_grad()
        dummy_loss.backward()
        optimizer.step()

    dist_before = compute_parameter_distance(net_a, net_b)
    print(f"  After 10 gradient steps, online-target distance: {dist_before:.4f}")

    # 부드러운 고치기
    net_soft = copy.deepcopy(net_b)
    for _ in range(100):
        soft_update(net_soft, net_a, tau=0.01)
    dist_soft = compute_parameter_distance(net_a, net_soft)
    print(f"  After 100 soft updates (τ=0.01): distance = {dist_soft:.4f}")

    # 딱딱한 고치기
    net_hard = copy.deepcopy(net_b)
    hard_update(net_hard, net_a)
    dist_hard = compute_parameter_distance(net_a, net_hard)
    print(f"  After hard update: distance = {dist_hard:.6f}")

    print("\nTarget networks demo complete!")


if __name__ == "__main__":
    demo_target_networks()
```

## 2. 논의

이 짜기는 과녁 그물의 핵심 논리를 감싼 `QNetwork`, `ReplayBuffer`, `DQNAgent` 갈래를 한가운데 둔다. 코드는 알고리즘 조각을 보여 주기와 따지기 논리에서 떼어 놓는 조각 짜기를 따른다.

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

**다룬 것** — 과녁 그물

이 짜기는 과녁 그물의 핵심 논리를 감싼 `QNetwork`, `ReplayBuffer`, `DQNAgent` 갈래를 한가운데 둔다.

고갱이 갈래는 `QNetwork`, `ReplayBuffer`, `DQNAgent`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
