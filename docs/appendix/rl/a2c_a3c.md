# A2C / A3C

A2C / A3C - 어드밴티지 액터-크리틱(Advantage Actor-Critic) 구현이다.

여기 구현은 A2C / A3C을 짧고 학습하기 좋게 보인 예제다. 코드는 핵심 구조와 순전파(forward pass)에 초점을 두어, 핵심 구성 요소를 살펴보고 이리저리 바꾸어 보기 쉽다.

## 1. 코드

```python
#!/usr/bin/env python3
"""
A2C / A3C - 어드밴티지 액터-크리틱
논문:
  - A3C: "심층 강화학습을 위한 비동기 방법" (2016)
  - A2C: 실제로 흔히 쓰는 동기 버전
핵심 아이디어:
  - 액터는 폴리시 π(a|s)을 낸다
  - 크리틱은 밸류 V(s)을 낸다
  - 어드밴티지 A = R - V(s)(또는 GAE)으로 액터를 갱신한다

파일: appendix/rl/a2c_a3c.py
참고: *로스*(폴리시 로스 + 밸류 로스 + 엔트로피)를 교육용으로 구현한 것이다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# 메인
# ========================================================================


class ActorCritic(nn.Module):
    """
    헤드 둘을 지닌 공유 백본:
      - 이산 액션에 대한 폴리시 로짓
      - 상태 밸류 추정
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
    A2C/A3C 로스를 계산한다.

    입력:
      logits:  (B, A) 폴리시 로짓
      values:  (B,) 크리틱의 V(s)
      actions: (B,) 수행한 액션
      returns: (B,) 경험적 리턴(n 스텝 등)

    어드밴티지:
      adv = returns - values

    로스:
      policy_loss = - E[ log pi(a|s) * adv ]
      value_loss  = MSE(values, returns)
      entropy_bonus은 탐색을 장려한다
    """
    # 고른 액션의 로그 확률
    logp = F.log_softmax(logits, dim=1)  # (B, A)
    logp_a = logp.gather(1, actions.long().unsqueeze(1)).squeeze(1)  # (B,)

    # 어드밴티지(액터를 갱신할 때 어드밴티지로 흐르는 그래디언트는 끊는다)
    adv = (returns - values).detach()

    policy_loss = -(logp_a * adv).mean()

    value_loss = F.mse_loss(values, returns)

    # 엔트로피: -sum p log p (엔트로피가 클수록 더 탐색한다)
    p = F.softmax(logits, dim=1)
    entropy = -(p * logp).sum(dim=1).mean()

    total = policy_loss + value_coef * value_loss - entropy_coef * entropy
    return total, {"policy_loss": policy_loss, "value_loss": value_loss, "entropy": entropy}


if __name__ == "__main__":
    pass
```

## 2. 논의

`ActorCritic` 클래스는 PyTorch의 `nn.Module` 인터페이스로 모델 구조를 캡슐화한다. `forward` 메서드가 계산 그래프를 정의하므로 학습하는 동안 PyTorch의 autograd가 그래디언트 계산을 자동으로 처리한다. 이런 모듈식 설계 덕분에 구성 요소 하나하나를 고치거나 더 큰 파이프라인에 넣기가 쉽다.

로스 계산은 모델의 출력을 최적화 목표와 잇는다. 알맞은 로스 함수를 고르는 일은 매우 중요하다. 모델이 무엇을 최적화하도록 학습할지를 정하고, 학습된 표현과 결정 경계를 직접 좌우하기 때문이다.

이 코드는 구조를 이해하기 위한 교육용 구현이다. 실제 학습에서는 데이터와 하드웨어에 맞춰 혼합 정밀도, 분산 학습, 데이터 어그멘테이션 등을 검토할 수 있다. 다만 이런 기법의 적용 조건과 효과는 모델별로 확인해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
기본 설정으로 초기화한 `ActorCritic`의 학습 가능한 파라메터를 모두 세어라. 웨이트와 바이어스를 포함해 레이어마다 나누어 적어라.

</div>

??? success "풀이"
    `nn.Linear(in_features, out_features)`마다 웨이트 파라메터가 `in_features * out_features`개이고 바이어스가 `out_features`개다(`bias=False`가 아니면). `nn.Conv2d(in_c, out_c, k)`마다 웨이트가 `in_c * out_c * k * k`개, 바이어스가 `out_c`개다. `nn.Embedding(num, dim)`은 파라메터가 `num * dim`개다. 모든 레이어의 값을 더한다. `sum(p.numel() for p in model.parameters())`으로 확인할 수 있다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
입력의 형태와 자료형이 기대한 대로인지 확인하는 입력 검증을 주요 함수나 클래스에 더하여라. 올바르지 않은 입력에는 이해하기 쉬운 오류 메시지를 띄워라.

</div>

??? success "풀이"
    `forward` 메서드(또는 알맞은 함수)의 첫머리에 `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`이나 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'` 같은 검사를 더한다. 형태를 검증하려면 핵심 차원을 확인한다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 이해하기 쉬운 오류 메시지는 디버깅을 크게 앞당기고 코드를 재사용하기에 더 강건하게 만든다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
이 구현에서 일어날 수 있는 실패 양상(failure mode) 둘을 들고, 각각을 어떻게 탐지하고 고칠지 설명하라.

</div>

??? success "풀이"
    흔한 실패 양상은 이렇다. (1) **그래디언트 소실·폭발** -- 그래디언트 노름을 모니터링해 탐지한다(`torch.nn.utils.clip_grad_norm_`이나 레이어마다 `param.grad.norm()` 기록하기). 그래디언트 클리핑, 더 나은 초기화(Xavier/Kaiming), 구조 변경(잔차 연결, 정규화)으로 고친다. (2) **과적합** -- 학습 로스는 줄어드는데 검증 로스가 오르면 탐지한다. 정칙화(드롭아웃, 웨이트 감쇠, 데이터 어그멘테이션)나 모델 크기 줄이기로 고친다. 학습 지표와 검증 지표를 늘 함께 모니터링해 이를 일찍 잡아야 한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
`ActorCritic`을 레이어나 블록의 수를 설정할 수 있게 확장하라. `__init__`에 `num_layers` 인자를 더하고 `nn.ModuleList`으로 깊이를 바꿀 수 있는 구조를 만들어라. 레이어 2개, 4개, 8개로 시험하여라.

</div>

??? success "풀이"
    하드코딩된 레이어를 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`으로 반복한다. 일반 파이썬 리스트가 아니라 `nn.ModuleList`을 써야 PyTorch가 모든 파라메터를 옵티마이저에 등록한다. `for n in [2, 4, 8]: model = ActorCritic(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.

## 정리하며

**다룬 것** — A2C / A3C

`ActorCritic` 클래스는 PyTorch의 `nn.Module` 인터페이스로 모델 구조를 캡슐화한다.

핵심 클래스는 `ActorCritic`이며 앞의 연습문제 4개로 스스로 확인해 볼 수 있다.
