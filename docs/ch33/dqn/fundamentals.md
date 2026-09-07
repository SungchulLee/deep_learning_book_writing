# DQN의 바탕

깊은 Q 그물은 Q 배우기와 신경망 함수 어림을 아울러 차원이 높은 상태 자리를 다룬다. 핵심 조각은 벡터와 픽셀 관찰을 위한 Q 그물 얼개, 식혀 가는 일정을 쓰는 엡실론 욕심쟁이 살펴보기, 때 차이 배우기의 고침 규칙이다. 이 짜기는 여러 층 신경망과 겹말기 Q 그물, 엡실론 욕심쟁이 움직임 고르기 셈속, 그리고 모든 DQN 변형이 딛고 선 근본 벽돌을 보여 주는 기본 익히기 고리를 다룬다.

## 코드

```python
"""
33.1.1 DQN의 바탕
========================

Core DQN components: Q-network architectures, action selection,
그리고 기본 DQN 익히기 되돌이.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from collections import deque, namedtuple
from typing import Tuple, Optional
import random

# ========================================================================
# 메인
# ========================================================================

# ---------------------------------------------------------------------------
# 옮김 꾸러미
# ---------------------------------------------------------------------------

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


# ---------------------------------------------------------------------------
# Q 그물 얼개
# ---------------------------------------------------------------------------

class MLPQNetwork(nn.Module):
    """벡터 살핌 자리를 위한 여러 켜 퍼셉트론 Q 그물."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: Tuple[int, ...] = (128, 128)):
        super().__init__()
        layers = []
        prev_dim = state_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h), nn.ReLU()])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """상태 묶음이 주어지면 모든 움직임의 Q 값을 돌려준다."""
        return self.net(x)


class ConvQNetwork(nn.Module):
    """아타리 갈래 그림점 살핌을 위한 감음 Q 그물.
    
    들임: (batch, 4, 84, 84) — 잿빛 틀 4개를 쌓음.
    내놓기: (batch, action_dim) — 움직임마다의 Q 값.
    """

    def __init__(self, action_dim: int, in_channels: int = 4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        # 편 크기 셈: 84x84 들임에서 64 * 7 * 7 = 3136
        self.fc = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 그림점 값을 [0, 1]로 고름
        x = x.float() / 255.0
        features = self.conv(x).view(x.size(0), -1)
        return self.fc(features)


# ---------------------------------------------------------------------------
# 엡실론 욕심쟁이 움직임 고르기
# ---------------------------------------------------------------------------

class EpsilonGreedy:
    """선형 식힘을 하는 엡실론 욕심쟁이 움직임 고르기."""

    def __init__(self, eps_start: float = 1.0, eps_end: float = 0.01,
                 eps_decay_steps: int = 10000):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.step_count = 0

    @property
    def epsilon(self) -> float:
        fraction = min(1.0, self.step_count / self.eps_decay_steps)
        return self.eps_start + fraction * (self.eps_end - self.eps_start)

    def select_action(self, q_values: torch.Tensor, training: bool = True) -> int:
        """엡실론 욕심쟁이 방침으로 움직임을 고른다.
        
        인수:
            q_values: Q-values for current state, shape (1, action_dim)
            training: 거짓이면 늘 욕심쟁이 움직임을 고른다
            
        반환값:
            고른 움직임 번호
        """
        if training:
            self.step_count += 1
            if random.random() < self.epsilon:
                return random.randrange(q_values.shape[1])
        return q_values.argmax(dim=1).item()


# ---------------------------------------------------------------------------
# 기본 DQN 부림꾼(간단히, 아직 되돌려 보기와 과녁 없음)
# ---------------------------------------------------------------------------

class BasicDQNAgent:
    """신경 그물로 핵심 Q 배움을 보이는 가장 작은 DQN 부림꾼.
    
    적바림: 일부러 간단히 했다. 겪음 되돌려 보기와 과녁 그물을 갖춘
    온전한 DQN은 implementation.py에 있다.
    """

    def __init__(self, state_dim: int, action_dim: int, lr: float = 1e-3,
                 gamma: float = 0.99, hidden_dims: Tuple[int, ...] = (128, 128),
                 device: str = 'cpu'):
        self.device = device
        self.gamma = gamma
        self.action_dim = action_dim

        self.q_network = MLPQNetwork(state_dim, action_dim, hidden_dims).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.explorer = EpsilonGreedy()

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_t)
        return self.explorer.select_action(q_values, training)

    def update(self, state, action, reward, next_state, done) -> float:
        """Single-sample online update (for illustration; not recommended)."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state_t = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)

        # 지금 Q 값
        q_value = self.q_network(state_t)[0, action]

        # 때 차이 과녁
        with torch.no_grad():
            next_q = self.q_network(next_state_t).max(dim=1)[0]
            target = reward + (1 - done) * self.gamma * next_q

        # 평균 제곱 어긋남 손실
        loss = nn.functional.mse_loss(q_value, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


# ---------------------------------------------------------------------------
# 때 차이 어긋남 셈
# ---------------------------------------------------------------------------

def compute_td_error(q_network: nn.Module, target_network: nn.Module,
                     states: torch.Tensor, actions: torch.Tensor,
                     rewards: torch.Tensor, next_states: torch.Tensor,
                     dones: torch.Tensor, gamma: float = 0.99) -> torch.Tensor:
    """옮김 묶음의 때 차이 어긋남을 셈한다.
    
    인수:
        q_network: 온라인 Q 그물
        target_network: 과녁 Q 그물
        states: (batch, state_dim)
        actions: (batch,)
        rewards: (batch,)
        next_states: (batch, state_dim)
        dones: (batch,) — 끝맺음이면 1.0
        gamma: 깎기 인수
        
    반환값:
        TD errors of shape (batch,)
    """
    # 고른 움직임의 Q(s, a)
    q_values = q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # 과녁: r + γ max_a' Q_target(s', a')
    with torch.no_grad():
        next_q_values = target_network(next_states).max(dim=1)[0]
        targets = rewards + (1 - dones) * gamma * next_q_values

    td_errors = targets - q_values
    return td_errors


# ---------------------------------------------------------------------------
# 시연
# ---------------------------------------------------------------------------

def demo_fundamentals():
    """CartPole에서 DQN의 바탕을 보인다."""
    print("=" * 60)
    print("DQN Fundamentals Demo")
    print("=" * 60)

    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    print(f"\nEnvironment: CartPole-v1")
    print(f"State dim: {state_dim}, Action dim: {action_dim}")

    # --- Q 그물 보이기 ---
    print("\n--- MLP Q-Network ---")
    q_net = MLPQNetwork(state_dim, action_dim)
    print(f"Architecture:\n{q_net}")
    total_params = sum(p.numel() for p in q_net.parameters())
    print(f"Total parameters: {total_params:,}")

    state, _ = env.reset()
    state_t = torch.FloatTensor(state).unsqueeze(0)
    q_values = q_net(state_t)
    print(f"\nQ-values for initial state: {q_values.detach().numpy()}")
    print(f"Greedy action: {q_values.argmax().item()}")

    # --- 엡실론 욕심쟁이 보이기 ---
    print("\n--- Epsilon-Greedy Schedule ---")
    explorer = EpsilonGreedy(eps_start=1.0, eps_end=0.01, eps_decay_steps=1000)
    for step in [0, 250, 500, 750, 1000, 2000]:
        explorer.step_count = step
        print(f"  Step {step:>5d}: ε = {explorer.epsilon:.4f}")

    # --- 기본 온라인 DQN(보임용으로 몇 마당만) ---
    print("\n--- Basic Online DQN (simplified, 50 episodes) ---")
    agent = BasicDQNAgent(state_dim, action_dim, lr=1e-3)
    episode_rewards = []

    for ep in range(50):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.update(state, action, reward, next_state, float(done))
            state = next_state
            total_reward += reward
        episode_rewards.append(total_reward)
        if (ep + 1) % 10 == 0:
            avg = np.mean(episode_rewards[-10:])
            print(f"  Episode {ep+1:>3d}: reward = {total_reward:.0f}, "
                  f"avg(10) = {avg:.1f}, ε = {agent.explorer.epsilon:.3f}")

    # --- 때 차이 어긋남 셈 보이기 ---
    print("\n--- TD Error Computation ---")
    target_net = MLPQNetwork(state_dim, action_dim)
    target_net.load_state_dict(q_net.state_dict())

    batch_states = torch.randn(8, state_dim)
    batch_actions = torch.randint(0, action_dim, (8,))
    batch_rewards = torch.randn(8)
    batch_next_states = torch.randn(8, state_dim)
    batch_dones = torch.zeros(8)

    td_errors = compute_td_error(q_net, target_net, batch_states, batch_actions,
                                  batch_rewards, batch_next_states, batch_dones)
    print(f"TD errors (batch of 8): {td_errors.detach().numpy().round(4)}")
    print(f"Mean absolute TD error: {td_errors.abs().mean().item():.4f}")

    # --- 감음 Q 그물 꼴 살피기 ---
    print("\n--- Conv Q-Network (Atari-style) ---")
    conv_net = ConvQNetwork(action_dim=4)
    dummy_frames = torch.randint(0, 256, (2, 4, 84, 84), dtype=torch.uint8)
    conv_q = conv_net(dummy_frames.float())
    print(f"Input shape: {dummy_frames.shape}")
    print(f"Output Q-values shape: {conv_q.shape}")
    conv_params = sum(p.numel() for p in conv_net.parameters())
    print(f"Total parameters: {conv_params:,}")

    env.close()
    print("\nFundamentals demo complete!")


if __name__ == "__main__":
    demo_fundamentals()```

## 논의

이 짜기는 Q 그물 얼개와 움직임 고르기의 핵심 논리를 감싼 `MLPQNetwork`, `ConvQNetwork`, `EpsilonGreedy` 갈래를 한가운데 둔다. 코드는 알고리즘 조각을 보여 주기와 따지기 논리에서 떼어 놓는 조각 짜기를 따른다.

보여 주기 함수는 이 조각들을 여느 힘 북돋우는 배움 잣대에 실제로 쓰는 모습을 보인다. 내놓기를 살피면 웃잡 고름과 문제 짜임에 따라 알고리즘의 성능이 어떻게 달라지는지 볼 수 있다.

실제 관점에서 이 짜기는 순수한 성능보다 또렷함을 앞세운다. 실제로 쓰는 얼개는 보통 묶음 셈, GPU 빠르게 하기, 더 정교한 윗매개변수 맞추기 같은 개선을 더한다. 그럼에도 여기 보인 핵심 알고리즘 생각은 큰 규모의 쓰임새로 곧바로 옮겨 간다.

### 그림점을 상태로 삼을 때: 켜 쌓기와 차원

`ConvQNetwork`이 그림을 여러 장 겹쳐 받는 까닭은 **찰나본 한 장으로는 공이 어느 쪽으로 가는지 알 수 없기** 때문이다. 자리는 보이지만 빠르기가 보이지 않는다. 그래서 잇단 그림 넉 장을 겹쳐 한 상태로 삼는다. 이렇게 하면 마르코프 성질이 되살아난다.

값으로 치르는 것은 차원이다. $100 \times 100$짜리 화면을 다섯 켜로 받으면 들임이 $100 \times 100 \times 5$, 곧 대략 $50{,}000$차원이다. 그런데 벽돌 깨기에서 실제로 필요한 것은 이만큼이다.

$$
(x,\, y,\, \dot{x},\, \dot{y},\, \ddot{x},\, \ddot{y}) \;+\; \text{남은 벽돌 수} \;+\; \text{막대 자리}
$$

이렇게 추리면 상태를 **예순여섯 개 남짓한 수**로 적을 수 있다. 그림점에서 곧바로 배우는 길과 이렇게 추린 특징으로 배우는 길의 다름이 여기서 드러난다. 앞의 길은 손으로 특징을 고르지 않아도 되지만 겹칩 그물과 훨씬 많은 뽑기를 치러야 하고, 뒤의 길은 잘 추리기만 하면 작은 여러 켜 퍼셉트론으로도 풀린다. 어느 쪽이든 **상태 나타냄을 줄이는 일이 모형 크기를 줄이는 일**이라는 점은 같다.

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
