# 타깃 네트워크

타깃 네트워크는 DQN의 중요한 개념이다. TD 타깃 $r + \gamma \max_{a'} Q(s', a')$ 을 지금 트레인 중인 온라인 네트워크로 계산하면, 업데이트할 때마다 타깃도 같이 움직여 트레인이 제 꼬리를 쫓게 된다. 그래서 온라인 네트워크의 복사본인 **타깃 네트워크** $\theta^-$ 를 따로 두고 타깃은 그것으로 계산한다. 복사본을 갱신하는 길은 둘이다. $C$ 스텝마다 통째로 덮어쓰는 **하드 업데이트**와, 매 스텝 $\theta^- \leftarrow \tau\theta + (1-\tau)\theta^-$ 로 조금씩 따라가게 하는 **소프트 업데이트**(폴랴크 평균)다. 이 쪽은 두 방식과 타깃 네트워크가 없는 경우를 견주고 트레인 안정성에 주는 영향을 본다.

## 1. 코드

```python
"""
35.1.3 타깃 네트워크
======================

딱딱한 타깃 네트워크 고치기와 부드러운 고치기, 그리고 그것이
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
# Q 네트워크
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
# 되돌려 보기 버퍼
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
# 타깃 네트워크 고치기 함수
# ---------------------------------------------------------------------------

def hard_update(target_net: nn.Module, online_net: nn.Module):
    """딱딱한 고치기: 온라인의 모든 값을 타깃으로 옮겨 적는다."""
    target_net.load_state_dict(online_net.state_dict())


def soft_update(target_net: nn.Module, online_net: nn.Module, tau: float = 0.005):
    """부드러운(폴랴크) 고치기: θ⁻ ← τθ + (1-τ)θ⁻"""
    for tp, op in zip(target_net.parameters(), online_net.parameters()):
        tp.data.copy_(tau * op.data + (1.0 - tau) * tp.data)


def compute_parameter_distance(net1: nn.Module, net2: nn.Module) -> float:
    """두 네트워크의 파라메터 사이 L2 거리."""
    dist = 0.0
    for p1, p2 in zip(net1.parameters(), net2.parameters()):
        dist += (p1 - p2).pow(2).sum().item()
    return np.sqrt(dist)


# ---------------------------------------------------------------------------
# 타깃 고치기 셈속을 고를 수 있는 DQN 에이전트
# ---------------------------------------------------------------------------

class DQNAgent:
    """딱딱한 고치기와 부드러운 고치기를 모두 받치는 DQN 에이전트."""

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

        # 네트워크
        self.online_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        hard_update(self.target_net, self.online_net)
        self.target_net.eval()  # 타깃 네트워크는 익히지 않음

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

        # 타깃 Q 값
        with torch.no_grad():
            if self.update_mode == 'none':
                # 타깃 네트워크 없음 — 타깃에 온라인 네트워크를 씀
                next_q = self.online_net(next_states).max(dim=1)[0]
            else:
                next_q = self.target_net(next_states).max(dim=1)[0]
            targets = rewards + (1 - dones) * self.gamma * next_q

        loss = nn.functional.mse_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        # 안정성을 위한 그래디언트 자르기
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()

        self.update_count += 1
        loss_val = loss.item()
        self.losses.append(loss_val)

        # 타깃 네트워크 고치기
        if self.update_mode == 'hard' and self.update_count % self.hard_update_freq == 0:
            hard_update(self.target_net, self.online_net)
        elif self.update_mode == 'soft':
            soft_update(self.target_net, self.online_net, self.tau)

        # 값 사이 거리 좇기
        dist = compute_parameter_distance(self.online_net, self.target_net)
        self.target_distances.append(dist)

        return loss_val


# ---------------------------------------------------------------------------
# 트레인 루프
# ---------------------------------------------------------------------------

def train_agent(agent: DQNAgent, env_name: str = 'CartPole-v1',
                n_episodes: int = 300, min_buffer: int = 500) -> List[float]:
    """DQN 에이전트를 익히고 마당 리워드를 돌려준다."""
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
# 보이기: 타깃 고치기 셈속 견주기
# ---------------------------------------------------------------------------

def demo_target_networks():
    """서로 다른 타깃 네트워크 셈속으로 익히기를 견준다."""
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

    # --- 딱딱한 고치기와 부드러운 고치기의 구조 ---
    print("\n--- Update Mechanics Illustration ---")
    net_a = QNetwork(4, 2)
    net_b = QNetwork(4, 2)
    hard_update(net_b, net_a)

    # net_a에 그래디언트 스텝 10번 흉내
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

이 구현은 `QNetwork`, `ReplayBuffer`, `DQNAgent` 세 클래스와 두 갱신 함수 `hard_update`, `soft_update` 를 중심에 둔다. `DQNAgent` 는 `update_mode` 하나로 타깃 네트워크 없음(`'none'`), 하드 업데이트(`'hard'`), 소프트 업데이트(`'soft'`)를 갈아 끼운다. `'none'` 이면 타깃도 온라인 네트워크로 계산한다.

데모 함수는 CartPole 에서 다섯 설정(타깃 없음, 하드 $C=100$·$500$, 소프트 $\tau=0.005$·$0.05$)을 200 에피소드씩 트레인하고, 마지막 50 에피소드의 리워드 평균·표준편차·최댓값과 온라인–타깃 파라메터 거리를 출력한다. 설정마다 같은 시드 42 로 **한 번씩만** 돌리므로, 설정 사이의 작은 차이는 시드를 바꾸면 뒤집힐 수 있다. 우열을 말하려면 시드를 여럿 돌려 흩어짐과 함께 보아야 한다.

마지막 부분은 트레인과 상관없이 두 갱신 방식의 성질만 보인다. 온라인 네트워크를 그래디언트 스텝 10번만큼 떼어 놓은 뒤, 하드 업데이트는 거리를 한 번에 0 으로 만들고, $\tau = 0.01$ 소프트 업데이트 100번은 거리를 $(1-\tau)^{100} \approx 0.37$ 배로만 줄인다. 소프트 업데이트에서 타깃이 온라인 네트워크를 따라가는 시간 규모가 대략 $1/\tau$ 스텝이라는 뜻이다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
데모의 다섯 설정을 시드 셋(예: 0, 1, 2)으로 각각 돌려 마지막 50 에피소드 평균 리워드의 평균과 표준편차를 표로 만들라. 시드 42 한 번으로 본 순위가 그대로 유지되는가?

</div>

??? success "풀이"
    `demo_target_networks` 의 시드 고정 세 줄을 바깥 루프의 시드 값으로 바꾸고, 설정마다 마지막 50 에피소드 평균을 모아 시드에 걸친 평균과 표준편차를 낸다. 두 설정의 평균 차이가 표준편차보다 작으면 그 순위는 이 실험으로 가릴 수 없다. CartPole 200 에피소드처럼 짧은 트레인에서는 시드에 따른 흩어짐이 커서, 한 번 돌려 본 순위가 뒤집히는 일이 흔하다. 어떤 결과가 나오든 수와 흩어짐을 함께 적어 두라.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
온라인 네트워크가 고정되어 있다고 할 때, 소프트 업데이트를 $k$ 번 하면 온라인–타깃 거리가 처음의 몇 배가 되는가? 하드 업데이트 주기 $C$ 와 비슷한 "따라잡는 시간"을 주는 $\tau$ 를 어림하라.

</div>

??? success "풀이"
    `soft_update` 는 $\theta^- \leftarrow \tau\theta + (1-\tau)\theta^-$ 이므로 $\theta^- - \theta \leftarrow (1-\tau)(\theta^- - \theta)$ 이다. $k$ 번 하면 거리는 정확히 $(1-\tau)^k$ 배가 된다. $\tau = 0.01$, $k = 100$ 이면 $0.99^{100} \approx 0.37$ 이고, 이 값이 데모의 「100 soft updates」 줄이 보여 주는 비율이다. 거리가 $1/e$ 로 줄어드는 데는 $k \approx 1/\tau$ 스텝이 걸리므로, 하드 업데이트 주기 $C$ 와 견줄 만한 값은 대략 $\tau \approx 1/C$ 다. $C = 100$ 이면 $\tau \approx 0.01$, $C = 500$ 이면 $\tau \approx 0.002$ 이다. 다만 하드 업데이트는 $C$ 스텝 동안 타깃이 완전히 멈춰 있다가 한 번에 뛰고, 소프트 업데이트는 매 스텝 조금씩 움직인다는 점이 다르다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
`update_mode='none'` 에서는 타깃 $r + \gamma \max_{a'} Q_\theta(s', a')$ 이 `torch.no_grad()` 안에서 계산되는데도 트레인이 불안정해지기 쉽다. 그래디언트를 막았는데 왜 문제가 남는가? 함수 근사에서 한 상태의 업데이트가 다른 상태의 값에 미치는 영향을 들어 설명하라.

</div>

??? success "풀이"
    `no_grad` 는 타깃 쪽으로 그래디언트가 흐르는 것만 막을 뿐, 타깃 값이 $\theta$ 에 의존한다는 사실은 그대로다. 뉴럴 네트워크는 파라메터를 공유하므로 $(s, a)$ 에서 $Q$ 를 올리는 업데이트는 비슷한 $s'$ 에서의 $Q$ 도 함께 올린다. 그러면 다음 업데이트의 타깃 $\max_{a'} Q(s', a')$ 이 커지고, 그것을 따라 $Q(s, a)$ 를 또 올리는 되먹임이 생긴다. 타깃 네트워크는 이 타깃을 $C$ 스텝(또는 약 $1/\tau$ 스텝) 동안 거의 고정해, 각 구간의 트레인을 고정된 회귀 문제에 가깝게 만든다. 데모의 `Final online-target distance` 는 이 고정의 정도를 보여 준다. 거리가 클수록 타깃이 온라인 네트워크보다 많이 뒤처져 있다는 뜻이다.

## 정리하며

**다룬 것** — 타깃 네트워크

이 구현은 하드 업데이트, 소프트 업데이트, 타깃 네트워크 없음을 `update_mode` 하나로 갈아 끼우는 `DQNAgent` 를 중심에 둔다.

핵심 클래스는 `QNetwork`, `ReplayBuffer`, `DQNAgent`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
