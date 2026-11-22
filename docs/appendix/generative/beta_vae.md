# Beta-VAE

Beta-VAE는 2017년 논문 "β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework"에서 나왔다. VAE 로스의 KL 항(잠재 분포를 표준 정규분포 쪽으로 당기는 항)에 웨이트 β를 두며, β가 클수록 분리(disentanglement, 잠재 변수의 각 좌표가 데이터의 서로 다른 요인 하나씩을 맡게 되는 것)는 강해지고 복원은 나빠진다.

여기 구현은 Beta-VAE를 짧게 줄인 학습용 예제다. 모델은 `beta` 값을 저장만 하고, 그것을 쓰는 로스 함수와 실행 예는 코드에 들어 있지 않다(`__main__` 블록이 비어 있어 출력도 없다). 코드는 핵심 구조와 순전파(forward pass)에 초점을 두어, 핵심 구성 요소를 살펴보고 이리저리 바꾸어 보기 쉽다.

## 1. 코드

```python
#!/usr/bin/env python3
"""
Beta-VAE - 분리를 조절하는 변분 오토인코더
논문: "β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework" (2017)
핵심 아이디어:
  - KL 항에 웨이트 β를 둔다
  - β가 클수록 분리는 강해지고 복원은 나빠진다

파일: appendix/generative/beta_vae.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# 메인
# ========================================================================


class Encoder(nn.Module):
    """입력 x를 잠재 평균과 로그 분산으로 인코딩한다."""
    def __init__(self, latent_dim=16):
        super().__init__()
        self.fc = nn.Linear(784, 256)
        self.mu = nn.Linear(256, latent_dim)
        self.logvar = nn.Linear(256, latent_dim)

    def forward(self, x):
        h = F.relu(self.fc(x))
        return self.mu(h), self.logvar(h)


class Decoder(nn.Module):
    """잠재 변수 z를 입력 공간으로 디코딩한다."""
    def __init__(self, latent_dim=16):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256)
        self.out = nn.Linear(256, 784)

    def forward(self, z):
        h = F.relu(self.fc(z))
        return torch.sigmoid(self.out(h))


class BetaVAE(nn.Module):
    """β-VAE 모델."""
    def __init__(self, latent_dim=16, beta=4.0):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.beta = beta

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar


if __name__ == "__main__":
    pass
```

## 2. 논의

이 구현은 클래스 3개(`Encoder`, `Decoder`, `BetaVAE`)를 정의하고, 이들이 함께 완전한 생성 모델 구조를 이룬다. 클래스마다 고유한 역할을 맡아 코드를 모듈로 나누고 확장하기 쉽게 한다. `forward` 메서드가 PyTorch의 자동 미분이 쓰는 계산 그래프를 정의한다.

이 코드는 구조를 이해하기 위한 교육용 구현이다. 실제 트레인에서는 데이터와 하드웨어에 맞춰 혼합 정밀도, 분산 트레인, 데이터 어그멘테이션 등을 검토할 수 있다. 다만 이런 기법의 적용 조건과 효과는 모델별로 확인해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
기본 설정으로 초기화한 `Encoder`의 트레인 가능한 파라메터를 모두 세어라. 웨이트와 바이어스를 함께 레이어마다 나누어 적어라.

</div>

??? success "풀이"
    `nn.Linear(in_features, out_features)`마다 웨이트 파라메터가 `in_features * out_features`개이고 바이어스가 `out_features`개다(`bias=False`가 아니면). `nn.Conv2d(in_c, out_c, k)`마다 웨이트가 `in_c * out_c * k * k`개, 바이어스가 `out_c`개다. `nn.Embedding(num, dim)`은 파라메터가 `num * dim`개다. 레이어를 모두 더한다. `sum(p.numel() for p in model.parameters())`로 확인할 수 있다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
입력의 형태와 자료형이 기대한 대로인지 확인하는 입력 검증을 주 함수나 클래스에 추가하라. 올바르지 않은 입력에는 알아보기 쉬운 오류 메시지를 띄워라.

</div>

??? success "풀이"
    `forward` 메서드(또는 알맞은 함수)의 첫머리에 `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`이나 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'` 같은 검사를 추가한다. 형태를 검사하려면 중요한 차원을 본다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 알아보기 쉬운 오류 메시지는 디버깅을 크게 앞당기고 코드를 재사용하기에 강건하게 만든다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
이 구현에서 생길 수 있는 실패 양상 둘을 들고, 저마다 어떻게 진단하고 고칠지 설명하라.

</div>

??? success "풀이"
    흔한 실패 양상은 이렇다. (1) **그래디언트 소실 또는 폭발** -- 그래디언트 크기를 관찰해 진단한다(`torch.nn.utils.clip_grad_norm_`이나 레이어마다 `param.grad.norm()` 기록하기). 그래디언트 클리핑, 더 나은 초기화(Xavier/Kaiming), 구조 변경(잔차 연결, 정규화)으로 고친다. (2) **과적합** -- 트레인 로스는 줄어드는데 검증 로스가 오르면 과적합으로 진단한다. 정칙화(드롭아웃, 웨이트 감쇠, 데이터 어그멘테이션)나 모델 크기 줄이기로 고친다. 트레인 지표와 검증 지표를 늘 함께 관찰해 이를 일찍 잡아야 한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
`Encoder`를 레이어나 블록의 수를 설정할 수 있게 확장하라. `__init__`에 `num_layers` 파라메터를 추가하고 `nn.ModuleList`로 깊이를 바꿀 수 있는 구조를 구성하라. 레이어 2개, 4개, 8개로 시험하여라.

</div>

??? success "풀이"
    하드코딩된 레이어를 다음으로 대체한다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`로 반복한다. 일반 파이썬 리스트가 아니라 `nn.ModuleList`를 써야 PyTorch가 파라메터를 모두 최적화 대상으로 등록한다. `for n in [2, 4, 8]: model = Encoder(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`로 시험한다.

## 정리하며

**다룬 것** — Beta-VAE

이 구현은 클래스 3개(`Encoder`, `Decoder`, `BetaVAE`)를 정의하고, 이들이 함께 완전한 생성 모델 구조를 이룬다.

핵심 클래스는 `Encoder`, `Decoder`, `BetaVAE`이며 앞의 연습문제 4개로 스스로 점검해 볼 수 있다.
