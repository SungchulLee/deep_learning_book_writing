# TD3

TD3은 2018년 글 "Addressing Function Approximation Error in Actor-Critic Methods"에서 나왔다. 1) 쌍둥이 Q 그물(따지는 이 둘 가운데 작은 값) 2) 과녁 방침 매끄럽게 하기(과녁 움직임에 시끄러움을 더한다) 3) 움직이는 이를 늦추어 고치기.

여기 짜보기는 TD3을 짧고 배우기 좋게 보인 본이다. 코드는 고갱이 얼개와 앞으로 걸음에 마음을 두어, 고갱이 꾸밈새를 살펴보고 이리저리 바꾸어 보기 쉽다.

## 1. 코드

```python
#!/usr/bin/env python3
"""
TD3 - 쌍둥이에 늦춘 DDPG
글: "움직이는 이-따지는 이 방법의 함수 어림 어긋남 다루기" (2018)
지은이: 스콧 후지모토, 헤르케 판 호프, 데이비드 메거
고갱이 깨침:
  1) 쌍둥이 Q 그물(따지는 이 둘 가운데 작은 값)
  2) 과녁 방침 매끄럽게 하기(과녁 움직임에 시끄러움을 더한다)
  3) 움직이는 이를 늦추어 고치기

두루마리: appendix/rl/td3.py
눈여겨볼 것: 배우기 위한 본이다(고갱이 과녁 셈 + 그물).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# 메인
# ========================================================================


class Actor(nn.Module):
    """딱 정해진 방침 a = pi(s)."""
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, act_dim),
            nn.Tanh(),  # assume actions scaled to [-1, 1]
        )

    def forward(self, obs):
        return self.net(obs)


class Critic(nn.Module):
    """Q(s,a) 따지는 이."""
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=1)
        return self.net(x).squeeze(1)


def td3_target(q1_tgt, q2_tgt, actor_tgt, s2, r, done, gamma=0.99, noise_std=0.2, noise_clip=0.5):
    """
    TD3 과녁:
      a' = actor_tgt(s') + clipped_noise
      y  = r + gamma*(1-done) * min(Q1_tgt(s',a'), Q2_tgt(s',a'))

    과녁 방침을 매끄럽게 하면 뾰족한 Q 봉우리에서 오는 지나친 어림이 줄어든다.
    """
    with torch.no_grad():
        a2 = actor_tgt(s2)

        # 잘라 낸 가우스 시끄러움을 더한다
        noise = torch.randn_like(a2) * noise_std
        noise = noise.clamp(-noise_clip, noise_clip)
        a2 = (a2 + noise).clamp(-1.0, 1.0)

        q1v = q1_tgt(s2, a2)
        q2v = q2_tgt(s2, a2)
        qmin = torch.min(q1v, q2v)

        y = r + gamma * (1.0 - done) * qmin
    return y


if __name__ == "__main__":
    pass```

## 2. 논의

이 짜보기는 갈래 2개(`Actor`, `Critic`)를 매기고, 이들이 어울려 온전한 북돋움 배움 얼개를 이룬다. 갈래마다 남다른 몫을 담아 코드를 묶음으로 나누고 넓히기 쉽게 한다. `forward` 방법이 PyTorch가 절로 미분하는 데 쓰는 셈 그림을 매긴다.

여기 실린 코드는 본보기 짜보기라 다듬기보다 알아보기 쉬움을 앞세운다. 서비스 얼개라면 흔히 섞인 촘촘함 익히기, 흩은 자료 나란히, 더 정교한 자료 불리기를 더한다. 그래도 여기서 보인 얼개의 고갱이 깨침은 크기와 상관없이 그대로다.

## 연습문제

**연습문제 1.**
기본 첫자리로 잡은 `Actor`의 배울 수 있는 매개변수를 모두 세어라. 짐과 치우침을 아울러 켜마다 나누어 적어라.

??? success "연습문제 1 풀이"
    `nn.Linear(in_features, out_features)`마다 짐 매개변수가 `in_features * out_features`개이고 치우침이 `out_features`개다(`bias=False`가 아니면). `nn.Conv2d(in_c, out_c, k)`마다 짐이 `in_c * out_c * k * k`개, 치우침이 `out_c`개다. `nn.Embedding(num, dim)`은 매개변수가 `num * dim`개다. 켜를 모두 더한다. `sum(p.numel() for p in model.parameters())`으로 따져 볼 수 있다.

---

**연습문제 2.**
들임의 꼴과 자료 갈래가 바라는 대로인지 살피는 들임 살피기를 으뜸 함수나 갈래에 더하여라. 올바르지 않은 들임에는 알아듣기 쉬운 어긋남 알림을 띄워라.

??? success "연습문제 2 풀이"
    `forward` 방법(또는 알맞은 함수)의 첫머리에 `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`이나 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'` 같은 살핌을 더한다. 꼴을 살피려면 종요로운 차수를 본다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 알아듣기 쉬운 어긋남 알림은 벌레잡기를 크게 앞당기고 코드를 되쓰기 든든하게 한다.

---

**연습문제 3.**
이 짜보기가 무너질 만한 결 둘을 밝히고, 저마다 어떻게 짚어내고 고칠지 밝혀라.

??? success "연습문제 3 풀이"
    흔히 무너지는 결은 이렇다. (1) **기울기가 사라지거나 터짐** -- 기울기 크기를 지켜보아 짚어낸다(`torch.nn.utils.clip_grad_norm_`이나 켜마다 `param.grad.norm()` 적기). 기울기 자르기, 더 나은 첫자리 잡기(Xavier/Kaiming), 얼개 고치기(나머지 이음, 잣대 잡기)로 고친다. (2) **지나치게 맞추기** -- 익힘 잃음은 줄어드는데 따짐 잃음이 오르면 짚어낸다. 다독임(드롭아웃, 짐 줄이기, 자료 불리기)이나 모형 크기 줄이기로 고친다. 익힘과 따짐 자를 늘 함께 지켜보아 이를 일찍 잡아야 한다.

---

**연습문제 4.**
`Actor`을 켜나 덩이의 수를 골라 잡을 수 있게 넓혀라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`으로 깊이를 바꿀 수 있는 얼개를 짜라. 켜 2개, 4개, 8개로 시험하여라.

??? success "연습문제 4 풀이"
    박아 넣은 켜를 다음으로 갈음한다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 방법에서 `for layer in self.layers: x = layer(x)`으로 되돈다. 여느 파이썬 목록이 아니라 `nn.ModuleList`을 써야 PyTorch가 매개변수를 모두 다듬기에 올린다. `for n in [2, 4, 8]: model = Actor(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.

## 정리하며

**다룬 것** — TD3

이 짜보기는 갈래 2개(`Actor`, `Critic`)를 매기고, 이들이 어울려 온전한 북돋움 배움 얼개를 이룬다.

고갱이 갈래는 `Actor`, `Critic`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
