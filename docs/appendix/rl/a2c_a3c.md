# A2C / A3C

A2C / A3C - 이득 움직이는 이-따지는 이 글:

여기 구현은 A2C / A3C을 짧고 학습하기 좋게 보인 본이다. 코드는 핵심 구조와 앞으로 스텝에 마음을 두어, 핵심 꾸밈새를 살펴보고 이리저리 바꾸어 보기 쉽다.

## 1. 코드

```python
#!/usr/bin/env python3
"""
A2C / A3C - 이득 움직이는 이-따지는 이
글:
  - A3C: "깊은 북돋움 러닝을 위한 발 맞추지 않는 방법" (2016)
  - A2C: 참으로 흔히 쓰는 발 맞추는 클래스
핵심 깨침:
  - 움직이는 이는 폴리시 π(a|s)을 낸다
  - 따지는 이는 값 V(s)을 낸다
  - 이득 A = R - V(s)(또는 GAE)으로 움직이는 이를 고친다

두루마리: appendix/rl/a2c_a3c.py
눈여겨볼 것: *잃음*(폴리시 잃음 + 값 잃음 + 엔트로피)을 배우기 위해 짜 본 것이다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# 메인
# ========================================================================


class ActorCritic(nn.Module):
    """
    머리 둘을 지닌, 나누어 쓰는 등뼈:
      - 따로 떨어진 액션에 대한 폴리시 로짓
      - 상태 값 어림
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
    A2C/A3C 잃음을 셈한다.

    들임:
      logits:  (B, A) 폴리시 로짓
      values:  (B,) 따지는 이의 V(s)
      actions: (B,) 한 액션
      returns: (B,) 겪어 본 되받음(n 스텝 따위)

    이득:
      adv = returns - values

    잃음:
      policy_loss = - E[ log pi(a|s) * adv ]
      value_loss  = MSE(values, returns)
      entropy_bonus은 둘러보기를 북돋운다
    """
    # 고른 액션의 로그 확률
    logp = F.log_softmax(logits, dim=1)  # (B, A)
    logp_a = logp.gather(1, actions.long().unsqueeze(1)).squeeze(1)  # (B,)

    # 이득(움직이는 이를 고칠 때 이득으로 흐르는 그래디언트는 끊는다)
    adv = (returns - values).detach()

    policy_loss = -(logp_a * adv).mean()

    value_loss = F.mse_loss(values, returns)

    # 엔트로피: -sum p log p (엔트로피가 클수록 더 둘러본다)
    p = F.softmax(logits, dim=1)
    entropy = -(p * logp).sum(dim=1).mean()

    total = policy_loss + value_coef * value_loss - entropy_coef * entropy
    return total, {"policy_loss": policy_loss, "value_loss": value_loss, "entropy": entropy}


if __name__ == "__main__":
    pass
```

## 2. 논의

`ActorCritic` 클래스는 PyTorch의 `nn.Module` 낯을 써서 모델 구조를 담는다. `forward` 방법이 계산 그림을 매기므로 익히는 동안 PyTorch의 autograd가 그래디언트 계산을 절로 다룬다. 이렇게 묶음으로 나눈 꾸밈 덕에 몫 하나하나를 고치거나 더 큰 흐름에 넣기가 쉽다.

로스 계산은 모델의 출력을 다듬기의 목표와 잇는다. 알맞은 로스 함수를 고르는 일은 몹시 중요하다. 모델이 무엇을 다듬도록 배울지를 매기고, 배운 리프리젠테이션과 분류 테두리를 곧바로 빚어내기 때문이다.

이 코드는 구조를 이해하기 위한 교육용 구현이다. 실제 학습에서는 데이터와 하드웨어에 맞춰 혼합 정밀도, 분산 학습, 데이터 어그멘테이션 등을 검토할 수 있다. 다만 이런 기법의 적용 조건과 효과는 모델별로 확인해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
기본 첫자리로 잡은 `ActorCritic`의 배울 수 있는 파라메터를 모두 세어라. 짐과 바이어스를 아울러 켜마다 나누어 적어라.

</div>

??? success "풀이"
    `nn.Linear(in_features, out_features)`마다 짐 파라메터가 `in_features * out_features`개이고 바이어스가 `out_features`개다(`bias=False`가 아니면). `nn.Conv2d(in_c, out_c, k)`마다 짐이 `in_c * out_c * k * k`개, 바이어스가 `out_c`개다. `nn.Embedding(num, dim)`은 파라메터가 `num * dim`개다. 켜를 모두 더한다. `sum(p.numel() for p in model.parameters())`으로 따져 볼 수 있다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
입력의 꼴과 데이터 클래스가 바라는 대로인지 살피는 입력 살피기를 으뜸 함수나 클래스에 더하여라. 올바르지 않은 입력에는 알아듣기 쉬운 어긋남 알림을 띄워라.

</div>

??? success "풀이"
    `forward` 방법(또는 알맞은 함수)의 소개에 `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`이나 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'` 같은 살핌을 더한다. 꼴을 살피려면 중요한 차수를 본다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 알아듣기 쉬운 어긋남 알림은 디버깅을 크게 앞당기고 코드를 되쓰기 든든하게 한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
이 구현이 무너질 만한 결 둘을 밝히고, 저마다 어떻게 짚어내고 고칠지 밝혀라.

</div>

??? success "풀이"
    흔히 무너지는 결은 이렇다. (1) **그래디언트가 사라지거나 터짐** -- 그래디언트 크기를 지켜보아 짚어낸다(`torch.nn.utils.clip_grad_norm_`이나 켜마다 `param.grad.norm()` 적기). 그래디언트 자르기, 더 나은 첫자리 잡기(Xavier/Kaiming), 구조 고치기(나머지 이음, 잣대 잡기)로 고친다. (2) **지나치게 맞추기** -- 학습 로스는 줄어드는데 따짐 로스가 오르면 짚어낸다. 다독임(드롭아웃, 짐 줄이기, 데이터 어그멘테이션)이나 모델 크기 줄이기로 고친다. 학습과 따짐 자를 늘 함께 지켜보아 이를 일찍 잡아야 한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
`ActorCritic`을 켜나 덩이의 수를 골라 잡을 수 있게 넓혀라. `__init__`에 `num_layers` 파라메터를 더하고 `nn.ModuleList`으로 깊이를 바꿀 수 있는 구조를 짜라. 켜 2개, 4개, 8개로 시험하여라.

</div>

??? success "풀이"
    박아 넣은 켜를 다음으로 갈음한다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 방법에서 `for layer in self.layers: x = layer(x)`으로 되돈다. 여느 파이썬 목록이 아니라 `nn.ModuleList`을 써야 PyTorch가 파라메터를 모두 다듬기에 올린다. `for n in [2, 4, 8]: model = ActorCritic(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.

## 정리하며

**다룬 것** — A2C / A3C

`ActorCritic` 클래스는 PyTorch의 `nn.Module` 낯을 써서 모델 구조를 담는다.

핵심 클래스는 `ActorCritic`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
