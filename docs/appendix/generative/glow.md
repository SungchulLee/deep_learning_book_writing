# Glow

Glow는 2018년 논문 "Glow: Generative Flow with Invertible 1x1 Convolutions"에서 나왔다. 학습되는 가역(invertible) 1x1 컨볼루션으로 RealNVP를 개선한다.

여기 구현은 Glow를 짧게 줄인 학습용 예제다. 코드는 핵심 구조와 순전파(forward pass)에 초점을 두어, 핵심 구성 요소를 살펴보고 이리저리 바꾸어 보기 쉽다.

## 1. 코드

```python
#!/usr/bin/env python3
"""
Glow - 가역 1x1 컨볼루션을 쓰는 흐름 기반 생성 모델
논문: "Glow: Generative Flow with Invertible 1x1 Convolutions" (2018)
핵심 아이디어:
  - 학습되는 가역 1x1 컨볼루션으로 RealNVP를 개선한다

파일: appendix/generative/glow.py
"""

import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================


class Invertible1x1Conv(nn.Module):
    """가역 선형 변환."""
    def __init__(self, dim=784):
        super().__init__()
        W = torch.qr(torch.randn(dim, dim))[0]
        self.weight = nn.Parameter(W)

    def forward(self, x):
        return x @ self.weight


if __name__ == "__main__":
    pass
```

## 2. 논의

`Invertible1x1Conv` 클래스는 PyTorch의 `nn.Module` 인터페이스를 써서 모델 구조를 담는다. `forward` 메서드가 계산 그래프를 정의하므로 학습하는 동안 PyTorch의 autograd가 그래디언트 계산을 자동으로 처리한다. 이렇게 모듈로 나눈 설계 덕에 구성 요소 하나하나를 고치거나 더 큰 파이프라인에 넣기가 쉽다.

이 코드는 구조를 이해하기 위한 교육용 구현이다. 실제 학습에서는 데이터와 하드웨어에 맞춰 혼합 정밀도, 분산 학습, 데이터 어그멘테이션 등을 검토할 수 있다. 다만 이런 기법의 적용 조건과 효과는 모델별로 확인해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
코드를 읽고 핵심이 되는 설계 결정을 짚어라. 구조에서 선택한 것 셋을 들고, 저마다 왜 생성 모델에 알맞은지 설명하라.

</div>

??? success "풀이"
    설계 결정은 구현마다 다르나 흔히 이런 것이 있다. (1) 활성화 함수 선택 -- ReLU 계열은 그래디언트가 포화되지 않아 학습이 빠르다. (2) 정규화 전략 -- 배치 정규화가 내부 공변량 이동(internal covariate shift)을 줄여 학습을 안정시킨다. (3) 잔차 연결 -- 있으면 스킵 경로를 주어 깊은 네트워크에서 그래디언트가 흐르게 한다. 선택마다 표현력, 계산 비용, 학습 안정성 사이의 트레이드오프를 보여 준다.

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
    흔한 실패 양상은 이렇다. (1) **그래디언트 소실 또는 폭발** -- 그래디언트 크기를 관찰해 진단한다(`torch.nn.utils.clip_grad_norm_`이나 레이어마다 `param.grad.norm()` 기록하기). 그래디언트 클리핑, 더 나은 초기화(Xavier/Kaiming), 구조 변경(잔차 연결, 정규화)으로 고친다. (2) **과적합** -- 학습 로스는 줄어드는데 검증 로스가 오르면 과적합으로 진단한다. 정칙화(드롭아웃, 웨이트 감쇠, 데이터 어그멘테이션)나 모델 크기 줄이기로 고친다. 학습 지표와 검증 지표를 늘 함께 관찰해 이를 일찍 잡아야 한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
`Invertible1x1Conv`을 레이어나 블록의 수를 설정할 수 있게 확장하라. `__init__`에 `num_layers` 파라메터를 추가하고 `nn.ModuleList`으로 깊이를 바꿀 수 있는 구조를 구성하라. 레이어 2개, 4개, 8개로 시험하여라.

</div>

??? success "풀이"
    하드코딩된 레이어를 다음으로 대체한다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`으로 반복한다. 일반 파이썬 리스트가 아니라 `nn.ModuleList`을 써야 PyTorch가 파라메터를 모두 최적화 대상으로 등록한다. `for n in [2, 4, 8]: model = Invertible1x1Conv(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.

## 정리하며

**다룬 것** — Glow

`Invertible1x1Conv` 클래스는 PyTorch의 `nn.Module` 인터페이스를 써서 모델 구조를 담는다.

핵심 클래스는 `Invertible1x1Conv`이며 앞의 연습문제 4개로 스스로 점검해 볼 수 있다.
