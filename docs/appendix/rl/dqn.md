# DQN

DQN was introduced in the 2015 paper "Human-level control through deep reinforcement learning." - Approximate Q(s,a) with a neural network   - Train with TD target using a *target network*   - Use experience replay to break correlation.

This implementation provides a concise, educational reference for DQN. The code focuses on the core architecture and forward pass, making it straightforward to study the key design patterns and adapt them for experimentation.

## 코드

```python
#!/usr/bin/env python3
"""
DQN - Deep Q-Network
Paper: "Human-level control through deep reinforcement learning" (2015)
Authors: Volodymyr Mnih et al.
Key idea:
  - Approximate Q(s,a) with a neural network
  - Train with TD target using a *target network*
  - Use experience replay to break correlation

File: appendix/rl/dqn.py
Note: Educational reference: model + replay + TD loss computation (no full env loop).
"""

from dataclasses import dataclass
import random
from collections import deque

# ========================================================================
# 메인
# ========================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Simple MLP Q(s,a) approximator for discrete actions."""
    def __init__(self, obs_dim: int, num_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (B, obs_dim)
        return self.net(obs)  # (B, num_actions)


@dataclass
class Transition:
    """One experience tuple stored in replay buffer."""
    s: torch.Tensor
    a: torch.Tensor
    r: torch.Tensor
    s2: torch.Tensor
    done: torch.Tensor


class ReplayBuffer:
    """Fixed-size FIFO replay buffer."""
    def __init__(self, capacity: int = 100_000):
        self.buf = deque(maxlen=capacity)

    def push(self, t: Transition):
        self.buf.append(t)

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        # Stack fields into batch tensors
        s = torch.stack([b.s for b in batch], dim=0)
        a = torch.stack([b.a for b in batch], dim=0)
        r = torch.stack([b.r for b in batch], dim=0)
        s2 = torch.stack([b.s2 for b in batch], dim=0)
        done = torch.stack([b.done for b in batch], dim=0)
        return s, a, r, s2, done

    def __len__(self):
        return len(self.buf)


def dqn_td_loss(q_net: nn.Module, target_net: nn.Module, batch, gamma: float = 0.99):
    """
    Compute DQN TD loss.

    For each transition (s,a,r,s',done):
      target = r + gamma * (1-done) * max_a' Q_target(s', a')
      loss = MSE( Q(s,a), target )

    Note:
      - a is discrete action index (shape: (B,))
      - done is 1 if terminal else 0
    """
    s, a, r, s2, done = batch

    # Current Q-values for all actions: (B, A)
    q_values = q_net(s)

    # Select Q(s,a) using gather:
    # a must be shape (B,1) for gather on dim=1
    q_sa = q_values.gather(1, a.long().unsqueeze(1)).squeeze(1)  # (B,)

    # Compute target using target network (no grad)
    with torch.no_grad():
        q_next = target_net(s2)                   # (B, A)
        max_q_next = q_next.max(dim=1).values     # (B,)
        target = r + gamma * (1.0 - done) * max_q_next

    loss = F.mse_loss(q_sa, target)
    return loss


if __name__ == "__main__":
    # Toy smoke test (no environment)
    obs_dim, num_actions = 8, 4
    q = QNetwork(obs_dim, num_actions)
    tgt = QNetwork(obs_dim, num_actions)

    # Fake batch
    B = 5
    s = torch.randn(B, obs_dim)
    a = torch.randint(0, num_actions, (B,))
    r = torch.randn(B)
    s2 = torch.randn(B, obs_dim)
    done = torch.randint(0, 2, (B,), dtype=torch.float32)

    loss = dqn_td_loss(q, tgt, (s, a, r, s2, done))
    print("loss:", float(loss))```

## 논의

이 짜보기는 갈래 3개(`QNetwork`, `Transition`, `ReplayBuffer`)를 매기고, 이들이 어울려 온전한 북돋움 배움 얼개를 이룬다. 갈래마다 남다른 몫을 담아 코드를 묶음으로 나누고 넓히기 쉽게 한다. `forward` 방법이 PyTorch이 절로 미분하는 데 쓰는 셈 그림을 매긴다.

여기 실린 코드는 본보기 짜보기라 다듬기보다 알아보기 쉬움을 앞세운다. 서비스 얼개라면 흔히 섞인 촘촘함 익히기, 흩은 자료 나란히, 더 정교한 자료 불리기를 더한다. 그래도 여기서 보인 얼개의 고갱이 깨침은 크기와 상관없이 그대로다.

## 익힘 문제

**익힘 1.**
기본 첫자리로 잡은 `QNetwork`의 배울 수 있는 매개변수를 모두 세어라. 짐과 치우침을 아울러 켜마다 나누어 적어라.

??? success "익힘 1 풀이"
    `nn.Linear(in_features, out_features)`마다 짐 매개변수가 `in_features * out_features`개이고 치우침이 `out_features`개다(`bias=False`가 아니면). `nn.Conv2d(in_c, out_c, k)`마다 짐이 `in_c * out_c * k * k`개, 치우침이 `out_c`개다. `nn.Embedding(num, dim)`은 매개변수가 `num * dim`개다. 켜를 모두 더한다. `sum(p.numel() for p in model.parameters())`으로 따져 볼 수 있다.

---

**익힘 2.**
들임의 꼴과 자료 갈래가 바라는 대로인지 살피는 들임 살피기를 으뜸 함수나 갈래에 더하여라. 올바르지 않은 들임에는 알아듣기 쉬운 어긋남 알림을 띄워라.

??? success "익힘 2 풀이"
    `forward` 방법(또는 알맞은 함수)의 첫머리에 `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`이나 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'` 같은 살핌을 더한다. 꼴을 살피려면 종요로운 차수를 본다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 알아듣기 쉬운 어긋남 알림은 벌레잡기를 크게 앞당기고 코드를 되쓰기 든든하게 한다.

---

**익힘 3.**
이 짜보기가 무너질 만한 결 둘을 밝히고, 저마다 어떻게 짚어내고 고칠지 밝혀라.

??? success "익힘 3 풀이"
    흔히 무너지는 결은 이렇다. (1) **기울기가 사라지거나 터짐** -- 기울기 크기를 지켜보아 짚어낸다(`torch.nn.utils.clip_grad_norm_`이나 켜마다 `param.grad.norm()` 적기). 기울기 자르기, 더 나은 첫자리 잡기(Xavier/Kaiming), 얼개 고치기(나머지 이음, 잣대 잡기)로 고친다. (2) **지나치게 맞추기** -- 익힘 잃음은 줄어드는데 따짐 잃음이 오르면 짚어낸다. 다독임(드롭아웃, 짐 줄이기, 자료 불리기)이나 모형 크기 줄이기로 고친다. 익힘과 따짐 자를 늘 함께 지켜보아 이를 일찍 잡아야 한다.

---

**익힘 4.**
`QNetwork`을 켜나 덩이의 수를 골라 잡을 수 있게 넓혀라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`으로 깊이를 바꿀 수 있는 얼개를 짜라. 켜 2개, 4개, 8개로 시험하여라.

??? success "익힘 4 풀이"
    박아 넣은 켜를 다음으로 갈음한다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 방법에서 `for layer in self.layers: x = layer(x)`으로 되돈다. 여느 파이썬 목록이 아니라 `nn.ModuleList`을 써야 PyTorch이 매개변수를 모두 다듬기에 올린다. `for n in [2, 4, 8]: model = QNetwork(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.
