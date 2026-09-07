# A2C / A3C

A2C / A3C - Advantage Actor-Critic Papers:

This implementation provides a concise, educational reference for A2C / A3C. The code focuses on the core architecture and forward pass, making it straightforward to study the key design patterns and adapt them for experimentation.

## 코드

```python
#!/usr/bin/env python3
"""
A2C / A3C - Advantage Actor-Critic
Papers:
  - A3C: "Asynchronous Methods for Deep Reinforcement Learning" (2016)
  - A2C: synchronous variant commonly used in practice
Key idea:
  - Actor outputs policy π(a|s)
  - Critic outputs value V(s)
  - Use advantage: A = R - V(s) (or GAE) to update actor

File: appendix/rl/a2c_a3c.py
Note: Educational implementation of the *losses* (policy loss + value loss + entropy).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# 메인
# ========================================================================


class ActorCritic(nn.Module):
    """
    Shared backbone with two heads:
      - policy logits over discrete actions
      - state value estimate
    """
    def __init__(self, obs_dim: int, num_actions: int, hidden: int = 128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )
        self.policy_head = nn.Linear(hidden, num_actions)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, obs):
        h = self.backbone(obs)
        logits = self.policy_head(h)            # (B, A)
        value = self.value_head(h).squeeze(1)   # (B,)
        return logits, value


def a2c_loss(logits, values, actions, returns, entropy_coef=0.01, value_coef=0.5):
    """
    Compute A2C/A3C losses.

    Inputs:
      logits:  (B, A) policy logits
      values:  (B,) critic V(s)
      actions: (B,) actions taken
      returns: (B,) empirical returns (e.g., n-step)

    Advantage:
      adv = returns - values

    Loss:
      policy_loss = - E[ log pi(a|s) * adv ]
      value_loss  = MSE(values, returns)
      entropy_bonus encourages exploration
    """
    # Log-probabilities of the chosen actions
    logp = F.log_softmax(logits, dim=1)  # (B, A)
    logp_a = logp.gather(1, actions.long().unsqueeze(1)).squeeze(1)  # (B,)

    # Advantage (stop gradient through advantage when updating actor)
    adv = (returns - values).detach()

    policy_loss = -(logp_a * adv).mean()

    value_loss = F.mse_loss(values, returns)

    # Entropy: -sum p log p (higher entropy = more exploration)
    p = F.softmax(logits, dim=1)
    entropy = -(p * logp).sum(dim=1).mean()

    total = policy_loss + value_coef * value_loss - entropy_coef * entropy
    return total, {"policy_loss": policy_loss, "value_loss": value_loss, "entropy": entropy}


if __name__ == "__main__":
    pass```

## 논의

`ActorCritic` 갈래는 PyTorch의 `nn.Module` 낯을 써서 모형 얼개를 담는다. `forward` 방법이 셈 그림을 매기므로 익히는 동안 PyTorch의 autograd이 기울기 셈을 절로 다룬다. 이렇게 묶음으로 나눈 꾸밈 덕에 몫 하나하나를 고치거나 더 큰 흐름에 넣기가 쉽다.

The loss computation connects the model's outputs to the optimization objective. Choosing the appropriate loss function is critical because it defines what the model learns to optimize, directly shaping the learned representations and decision boundaries.

여기 실린 코드는 본보기 짜보기라 다듬기보다 알아보기 쉬움을 앞세운다. 서비스 얼개라면 흔히 섞인 촘촘함 익히기, 흩은 자료 나란히, 더 정교한 자료 불리기를 더한다. 그래도 여기서 보인 얼개의 고갱이 깨침은 크기와 상관없이 그대로다.

## 익힘 문제

**익힘 1.**
기본 첫자리로 잡은 `ActorCritic`의 배울 수 있는 매개변수를 모두 세어라. 짐과 치우침을 아울러 켜마다 나누어 적어라.

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
`ActorCritic`을 켜나 덩이의 수를 골라 잡을 수 있게 넓혀라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`으로 깊이를 바꿀 수 있는 얼개를 짜라. 켜 2개, 4개, 8개로 시험하여라.

??? success "익힘 4 풀이"
    박아 넣은 켜를 다음으로 갈음한다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 방법에서 `for layer in self.layers: x = layer(x)`으로 되돈다. 여느 파이썬 목록이 아니라 `nn.ModuleList`을 써야 PyTorch이 매개변수를 모두 다듬기에 올린다. `for n in [2, 4, 8]: model = ActorCritic(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.
