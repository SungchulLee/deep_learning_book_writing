# DQN의 기초

DQN은 Q 학습과 뉴럴 네트워크 함수 근사를 아울러 차원이 높은 상태 공간을 다룬다. 핵심 구성 요소는 벡터와 픽셀 관찰을 위한 Q 네트워크 구조, 어닐링 스케줄을 쓰는 엡실론 그리디 탐험, 시간차 학습의 갱신 규칙이다. 이 구현은 다층 뉴럴 네트워크와 컨볼루션 Q 네트워크, 엡실론 그리디 액션 선택 전략, 그리고 모든 DQN 변형이 딛고 선 근본 벽돌을 보여 주는 기본 학습 루프를 다룬다.

## 1. 코드

```python
"""
35.1.1 DQN의 바탕
========================

DQN의 핵심 조각: Q 네트워크 구조, 액션 고르기,
그리고 기본 DQN 익히기 회귀.
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
# Q 네트워크 구조
# ---------------------------------------------------------------------------

class MLPQNetwork(nn.Module):
    """벡터 살핌 자리를 위한 여러 켜 퍼셉트론 Q 네트워크."""

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
        """상태 배치가 주어지면 모든 액션의 Q 값을 돌려준다."""
        return self.net(x)


class ConvQNetwork(nn.Module):
    """아타리 클래스 픽셀 살핌을 위한 감음 Q 네트워크.
    
    들임: (batch, 4, 84, 84) — 잿빛 틀 4개를 쌓음.
    내놓기: (batch, action_dim) — 액션마다의 Q 값.
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
        # 픽셀 값을 [0, 1]로 고름
        x = x.float() / 255.0
        features = self.conv(x).view(x.size(0), -1)
        return self.fc(features)


# ---------------------------------------------------------------------------
# 엡실론 욕심쟁이 액션 고르기
# ---------------------------------------------------------------------------

class EpsilonGreedy:
    """선형 식힘을 하는 엡실론 욕심쟁이 액션 고르기."""

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
        """엡실론 욕심쟁이 폴리시로 액션을 고른다.
        
        인수:
            q_values: 이제 상태의 Q 값, 형태 (1, action_dim)
            training: 거짓이면 늘 욕심쟁이 액션을 고른다
            
        반환값:
            고른 액션 번호
        """
        if training:
            self.step_count += 1
            if random.random() < self.epsilon:
                return random.randrange(q_values.shape[1])
        return q_values.argmax(dim=1).item()


# ---------------------------------------------------------------------------
# 기본 DQN 에이전트(간단히, 아직 되돌려 보기와 타깃 없음)
# ---------------------------------------------------------------------------

class BasicDQNAgent:
    """뉴럴 네트워크로 핵심 Q 러닝을 보이는 가장 작은 DQN 에이전트.
    
    적바림: 일부러 간단히 했다. 겪음 되돌려 보기와 타깃 네트워크를 갖춘
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
        """표본 하나로 하는 온라인 고침(보이기용이며 권하지 않는다)."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state_t = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)

        # 지금 Q 값
        q_value = self.q_network(state_t)[0, action]

        # 때 차이 타깃
        with torch.no_grad():
            next_q = self.q_network(next_state_t).max(dim=1)[0]
            target = reward + (1 - done) * self.gamma * next_q

        # 평균 제곱 어긋남 로스
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
    """옮김 배치의 때 차이 어긋남을 셈한다.
    
    인수:
        q_network: 온라인 Q 네트워크
        target_network: 타깃 Q 네트워크
        states: (batch, state_dim)
        actions: (batch,)
        rewards: (batch,)
        next_states: (batch, state_dim)
        dones: (batch,) — 끝맺음이면 1.0
        gamma: 깎기 인수
        
    반환값:
        형태가 (batch,)인 시간차 어긋남
    """
    # 고른 액션의 Q(s, a)
    q_values = q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # 타깃: r + γ max_a' Q_target(s', a')
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

    # --- Q 네트워크 보이기 ---
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

    # --- 감음 Q 네트워크 형태 살피기 ---
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
    demo_fundamentals()
```

## 2. 논의

이 구현은 Q 네트워크 구조와 액션 선택의 핵심 논리를 감싼 `MLPQNetwork`, `ConvQNetwork`, `EpsilonGreedy` 클래스를 중심에 둔다. 코드는 알고리즘 구성 요소를 데모·평가 논리와 분리하는 모듈식 설계를 따른다.

데모 함수는 이 구성 요소들을 전형적인 강화학습 벤치마크에 실제로 적용하는 모습을 보인다. 출력을 살피면 하이퍼파라미터 선택과 문제 구조에 따라 알고리즘의 성능이 어떻게 달라지는지 볼 수 있다.

실용적 관점에서 이 구현은 순수한 성능보다 명료함을 앞세운다. 실제로 쓰는 시스템은 보통 배치 계산, GPU 가속, 더 정교한 하이퍼파라미터 튜닝 같은 개선을 더한다. 그럼에도 여기 보인 핵심 알고리즘 아이디어는 대규모 응용으로 그대로 옮겨 간다.

### 픽셀을 상태로 삼을 때: 프레임 쌓기와 차원

`ConvQNetwork`이 그림을 여러 장 겹쳐 받는 까닭은 **스냅숏 한 장으로는 공이 어느 쪽으로 가는지 알 수 없기** 때문이다. 위치는 보이지만 속도는 보이지 않는다. 그래서 연속된 프레임 넉 장을 겹쳐 한 상태로 삼는다. 이렇게 하면 마르코프 성질이 되살아난다.

값으로 치르는 것은 차원이다. $100 \times 100$짜리 화면을 다섯 프레임으로 받으면 입력이 $100 \times 100 \times 5$, 곧 대략 $50{,}000$차원이다. 그런데 벽돌 깨기에서 실제로 필요한 것은 이만큼이다.

$$
(x,\, y,\, \dot{x},\, \dot{y},\, \ddot{x},\, \ddot{y}) \;+\; \text{남은 벽돌 수} \;+\; \text{막대 자리}
$$

이렇게 추리면 상태를 **예순여섯 개 남짓한 수**로 적을 수 있다. 픽셀에서 곧바로 배우는 길과 이렇게 추린 피처로 배우는 길의 다름이 여기서 드러난다. 앞의 길은 손으로 피처를 고르지 않아도 되지만 컨볼루션 네트워크와 훨씬 많은 샘플링을 치러야 하고, 뒤의 길은 잘 추리기만 하면 작은 다층 퍼셉트론으로도 풀린다. 어느 쪽이든 **상태 표현을 줄이는 일이 모델 크기를 줄이는 일**이라는 점은 같다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
데모 코드를 돌려 핵심 출력 지표를 기록하라. 하이퍼파라미터 하나(러닝레이트, 은닉 차원, 레이어 개수 같은 것)를 바꾸고 결과가 어떻게 바뀌는지 기록하라.

</div>

??? success "풀이"
    데모를 돌린 뒤 나머지를 고정하고 고른 하이퍼파라미터를 차례로 바꾼다. 예를 들어 은닉 차원을 두 배로 하면 보통 표현력이 늘지만 계산 시간이 커진다. 러닝레이트는 단조롭지 않은 영향을 준다. 너무 작으면 느리게 수렴하고 너무 크면 진동한다. 고른 하이퍼파라미터의 서로 다른 값을 적어도 셋 잡아 구체적인 수를 기록해 두라.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
이 구현에서 핵심 구조 선택의 역할을 설명하라. 왜 그 활성화 함수, 정규화 전략, 로스 함수를 쓰는가? 다른 것으로 바꾸면 어떻게 되는가?

</div>

??? success "풀이"
    이 구조 선택은 DQN 연구에서 자리 잡은 모범 사례를 따른다. 예를 들어 ReLU 활성화는 비선형성을 주면서 양의 입력에서 그래디언트가 사라지는 것을 피한다. 로스 함수는 과제 유형에 맞게 고른다(분류에는 교차 엔트로피, 회귀에는 평균 제곱 오차). 다른 것으로 바꾸면(예를 들어 시그모이드 활성화, L1 로스) 최적화 지형이 바뀌어 성능이 나빠질 수 있지만 어떤 상황에서는 바꾸는 것이 이로울 수도 있다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
더 어려운 상황을 다루도록 구현을 확장하라. 더 큰 데이터셋, 다른 문제 변형, 피처 추가 가운데 하나를 고른다. 수정 사항을 적고 성능에 미친 영향을 분석하라.

</div>

??? success "풀이"
    자연스러운 확장 하나는 정칙화(드롭아웃, 웨이트 감쇠)나 더 정교한 구조(레이어 추가, 스킵 연결)를 더하는 것이다. 고른 확장을 구현하고 같은 데이터로 학습한 뒤 전후의 지표를 비교하여라. 이 확장은 원래 알고리즘과 수정 사항의 이론적 근거를 모두 이해했음을 보여야 한다.

## 정리하며

**다룬 것** — DQN의 기초

이 구현은 Q 네트워크 구조와 액션 선택의 핵심 논리를 감싼 `MLPQNetwork`, `ConvQNetwork`, `EpsilonGreedy` 클래스를 중심에 둔다.

핵심 클래스는 `MLPQNetwork`, `ConvQNetwork`, `EpsilonGreedy`, `BasicDQNAgent`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
