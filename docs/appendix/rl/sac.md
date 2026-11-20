# SAC

SAC은 2018년 논문 "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning"에서 나왔다. 액터는 기대 리턴에 엔트로피를 더한 값을 최대화하고(엔트로피가 크다는 것은 폴리시가 여러 액션을 고루 시도한다는 뜻이라 탐색이 유지된다), Q 네트워크 둘(이중 Q) 가운데 작은 값을 써서 과대추정 편향을 줄이며, 학습을 안정시키려고 타깃 Q 네트워크를 쓴다.

여기 구현은 SAC을 짧고 공부하기 좋게 보인 예제다. 코드는 핵심 구조와 순전파(forward pass)에 초점을 두어, 핵심 구성 요소를 살펴보고 이리저리 바꾸어 보기 쉽다.

## 1. 코드

```python
#!/usr/bin/env python3
"""
SAC - 소프트 액터-크리틱(연속 제어)
논문: "소프트 액터-크리틱: 오프폴리시 최대 엔트로피 심층 강화학습" (2018)
지은이: 투오마스 하르노야 외
핵심 아이디어:
  - 액터가 기대 리턴 + 엔트로피(탐색)를 최대화한다
  - Q 네트워크 둘(이중 Q)로 과대추정 편향을 줄인다
  - 학습 안정성을 위해 타깃 Q 네트워크를 쓴다

파일: appendix/rl/sac.py
참고: 교육용 예제다(네트워크 + 핵심 타깃 계산).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# 메인
# ========================================================================


class GaussianPolicy(nn.Module):
    """
    연속 액션을 위한 확률적 액터:
      a ~ N(mu(s), sigma(s))을 뽑은 뒤 흔히 tanh으로 [-1,1] 안에 스쿼싱(squashing)한다.
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)

    def forward(self, obs):
        h = self.net(obs)
        mu = self.mu(h)
        log_std = self.log_std(h).clamp(-20, 2)  # numerical stability
        std = torch.exp(log_std)
        return mu, std

    def sample(self, obs):
        """
        리파라메리제이션(reparameterization) 샘플링:
          a = tanh(mu + std * eps)
        tanh 스쿼싱을 반영한 log_prob(a)도 반환한다(간결하게 하려고 여기서는 뺐다).
        """
        mu, std = self.forward(obs)
        eps = torch.randn_like(std)
        pre_tanh = mu + std * eps
        a = torch.tanh(pre_tanh)
        # 실제 SAC 구현에서는 tanh 보정을 넣어 log_prob을 계산한다.
        log_prob = None
        return a, log_prob


class QNetwork(nn.Module):
    """연속 액션을 위한 크리틱 Q(s,a)."""
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
        return self.net(x).squeeze(1)  # (B,)


def sac_target(q1_tgt, q2_tgt, policy, s2, r, done, gamma=0.99, alpha=0.2):
    """
    SAC 타깃을 계산한다:
      a' ~ pi(s')
      y = r + gamma*(1-done) * ( min(Q1_tgt(s',a'), Q2_tgt(s',a')) - alpha * log pi(a'|s') )

    여기서는 간결하게 하려고 log pi 항을 뺐다. 완전한 구현에서는 넣어야 한다.
    """
    with torch.no_grad():
        a2, logp2 = policy.sample(s2)
        q1v = q1_tgt(s2, a2)
        q2v = q2_tgt(s2, a2)
        qmin = torch.min(q1v, q2v)

        # logp2이 None이면(이 교육용 코드처럼) 엔트로피 항을 계산에서 뺀다.
        if logp2 is None:
            backup = qmin
        else:
            backup = qmin - alpha * logp2

        y = r + gamma * (1.0 - done) * backup
    return y


if __name__ == "__main__":
    pass
```

## 2. 논의

이 구현은 클래스 2개(`GaussianPolicy`, `QNetwork`)와 타깃 계산 함수 `sac_target`을 정의하고, 이들이 함께 SAC의 액터와 크리틱, 그리고 크리틱이 맞출 타깃을 이룬다. 클래스마다 고유한 역할을 맡아 코드를 모듈화하고 확장하기 쉽게 한다. `forward` 메서드가 PyTorch의 자동 미분에 쓰이는 계산 그래프를 정의한다.

다만 완전한 SAC은 아니다. `sample`이 `log_prob`으로 `None`을 돌려주므로 `sac_target`은 엔트로피 항 $-\alpha \log \pi(a'|s')$을 빼고 계산한다. 이대로는 최대 엔트로피 타깃이 아니라 두 Q 가운데 작은 값을 쓰는 보통의 타깃일 뿐이며, 액터·크리틱 갱신 루프와 타깃 네트워크의 소프트 갱신도 들어 있지 않다.

이 코드는 구조를 이해하기 위한 교육용 구현이다. 실제 학습에서는 데이터와 하드웨어에 맞춰 혼합 정밀도, 분산 학습, 데이터 어그멘테이션 등을 검토할 수 있다. 다만 이런 기법의 적용 조건과 효과는 모델별로 확인해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
기본 설정으로 초기화한 `GaussianPolicy`의 학습 가능한 파라메터를 모두 세어라. 웨이트와 바이어스를 포함해 레이어마다 나누어 적어라.

</div>

??? success "풀이"
    `nn.Linear(in_features, out_features)`마다 웨이트 파라메터가 `in_features * out_features`개이고 바이어스가 `out_features`개다(`bias=False`가 아니면). `nn.Conv2d(in_c, out_c, k)`마다 웨이트가 `in_c * out_c * k * k`개, 바이어스가 `out_c`개다. `nn.Embedding(num, dim)`은 파라메터가 `num * dim`개다. 모든 레이어의 값을 더한다. `sum(p.numel() for p in model.parameters())`로 확인할 수 있다. `GaussianPolicy`는 `obs_dim`과 `act_dim`에 기본값이 없으므로 두 값을 $o$, $a$라 하고 `hidden=256`으로 세면, 첫 `nn.Linear`가 $256o + 256$, 둘째가 $256 \cdot 256 + 256 = 65{,}792$, `mu`와 `log_std`가 저마다 $256a + a$개라 모두 $256o + 514a + 66{,}048$개다.

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
    흔한 실패 양상은 이렇다. (1) **그래디언트 소실·폭발** -- 그래디언트 노름을 모니터링해 탐지한다(`torch.nn.utils.clip_grad_norm_`이나 레이어마다 `param.grad.norm()` 기록하기). 그래디언트 클리핑, 더 나은 초기화(Xavier/Kaiming), 구조 변경(잔차 연결, 정규화)으로 고친다. (2) **과적합** -- 학습 로스는 줄어드는데 검증 로스가 오르면 탐지한다. 정칙화(드롭아웃, 웨이트 감쇠, 데이터 어그멘테이션)나 모델 크기 줄이기로 고친다. 학습 지표와 검증 지표를 늘 함께 모니터링해 이를 일찍 잡아야 한다. 다만 강화학습에는 고정된 검증 집합이 없으므로, SAC에서는 다음 둘이 더 흔하다. (3) **Q 값 발산** -- 크리틱이 내놓는 Q 값이 실제 리턴보다 계속 커지면 탐지한다. 타깃 네트워크의 소프트 갱신 계수를 줄이거나 보상 스케일을 낮춘다. (4) **엔트로피 항 누락이나 잘못된 $\alpha$** -- 이 쪽의 코드처럼 `log_prob`이 빠지면 폴리시가 일찍 결정론적으로 굳어 탐색이 멈춘다. 폴리시의 표준편차가 0 가까이 줄어드는 것으로 탐지하고, tanh 보정을 넣은 `log_prob`을 계산하거나 $\alpha$를 자동으로 조절해 고친다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
`GaussianPolicy`를 레이어나 블록의 수를 설정할 수 있게 확장하라. `__init__`에 `num_layers` 인자를 더하고 `nn.ModuleList`로 깊이를 바꿀 수 있는 구조를 만들어라. 레이어 2개, 4개, 8개로 시험하여라.

</div>

??? success "풀이"
    하드코딩된 레이어를 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`로 반복한다. 일반 파이썬 리스트가 아니라 `nn.ModuleList`를 써야 PyTorch가 모든 파라메터를 옵티마이저에 등록한다. `for n in [2, 4, 8]: model = GaussianPolicy(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.

## 정리하며

**다룬 것** — SAC

이 구현은 클래스 2개(`GaussianPolicy`, `QNetwork`)와 `sac_target`으로 SAC의 액터, 크리틱, 타깃 계산을 보인다. 엔트로피 항은 빠져 있다.

핵심 클래스는 `GaussianPolicy`, `QNetwork`이며 앞의 연습문제 4개로 스스로 확인해 볼 수 있다.
