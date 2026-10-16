# 정규화 레이어

잣대 잡기 켜 - 딥러닝에 쓰이는 흔한 클래스. 담긴 것:

여기 구현은 Normalization Layers을 짧고 학습하기 좋게 보인 본이다. 코드는 핵심 구조와 앞으로 걸음에 마음을 두어, 핵심 꾸밈새를 살펴보고 이리저리 바꾸어 보기 쉽다.

## 1. 코드

```python
#!/usr/bin/env python3
"""
잣대 잡기 켜 - 깊은 러닝에 쓰이는 흔한 클래스
담긴 것:
  - BatchNorm1d/2d(PyTorch 붙박이를 감싼 것)
  - LayerNorm(PyTorch 붙박이를 감싼 것)
  - GroupNorm(PyTorch 붙박이를 감싼 것)
  - RMSNorm(요즘 큰 말 모형에서 흔히 쓴다)

두루마리: appendix/utils/normalization.py
눈여겨볼 것: 저마다 언제 쓰는지 주석으로 밝힌 배우기용 구현이다.
"""

import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================


class RMSNorm(nn.Module):
    """
    RMSNorm(제곱 평균 제곱근 켜 잣대 잡기)

    켜 잣대 잡기와 달리 RMSNorm은:
      - 평균을 빼지 않는다
      - RMS(sqrt(mean(x^2)))으로 나누기만 한다

    LLaMA 꼴 모형에서 흔히 쓴다.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: (..., dim)
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


# 눈여겨볼 것:
# - BatchNorm, LayerNorm, GroupNorm은 PyTorch에 들어 있다.
# - 흔히 이렇게 들여와 쓴다:
#     nn.BatchNorm2d(C)
#     nn.LayerNorm(D)
#     nn.GroupNorm(num_groups, C)
#
# 이 두루마리는 RMSNorm과 짧은 쓰는 법 적바림을 더한다.


if __name__ == "__main__":
    pass
```

## 2. 논의

`RMSNorm` 클래스는 PyTorch의 `nn.Module` 낯을 써서 모델 구조를 담는다. `forward` 방법이 계산 그림을 매기므로 익히는 동안 PyTorch의 autograd가 그래디언트 계산을 절로 다룬다. 이렇게 묶음으로 나눈 꾸밈 덕에 몫 하나하나를 고치거나 더 큰 흐름에 넣기가 쉽다.

이 코드는 구조를 이해하기 위한 교육용 구현이다. 실제 학습에서는 데이터와 하드웨어에 맞춰 혼합 정밀도, 분산 학습, 데이터 증강 등을 검토할 수 있다. 다만 이런 기법의 적용 조건과 효과는 모델별로 확인해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
코드를 읽고 핵심이 되는 설계 판단을 짚어라. 짜기에서 고른 것 셋을 들고, 저마다 왜 잔손질 묶음에 알맞은지 밝혀라.

</div>

??? success "연습문제 1 풀이"
    설계 판단은 구현마다 다르나 흔히 이런 것이 있다. (1) 엑티베이션 함수 고르기 -- ReLU 클래스는 그래디언트가 잦아들지 않아 학습이 빠르다. (2) 잣대 잡는 꾀 -- 배치 잣대 잡기가 안쪽 함께 바뀌는 옮겨감을 줄여 학습을 든든하게 한다. (3) 나머지 이음 -- 있으면 건너뛰는 길을 주어 깊은 네트워크에서 그래디언트가 흐르게 한다. 고른 것마다 드러내는 힘, 계산 비용, 학습의 든든함 사이의 트레이드오프를 보여 준다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
입력의 꼴과 데이터 클래스가 바라는 대로인지 살피는 입력 살피기를 으뜸 함수나 클래스에 더하여라. 올바르지 않은 입력에는 알아듣기 쉬운 어긋남 알림을 띄워라.

</div>

??? success "연습문제 2 풀이"
    `forward` 방법(또는 알맞은 함수)의 소개에 `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`이나 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'` 같은 살핌을 더한다. 꼴을 살피려면 중요한 차수를 본다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 알아듣기 쉬운 어긋남 알림은 디버깅을 크게 앞당기고 코드를 되쓰기 든든하게 한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
이 구현이 무너질 만한 결 둘을 밝히고, 저마다 어떻게 짚어내고 고칠지 밝혀라.

</div>

??? success "연습문제 3 풀이"
    흔히 무너지는 결은 이렇다. (1) **그래디언트가 사라지거나 터짐** -- 그래디언트 크기를 지켜보아 짚어낸다(`torch.nn.utils.clip_grad_norm_`이나 켜마다 `param.grad.norm()` 적기). 그래디언트 자르기, 더 나은 첫자리 잡기(Xavier/Kaiming), 구조 고치기(나머지 이음, 잣대 잡기)로 고친다. (2) **지나치게 맞추기** -- 학습 로스는 줄어드는데 따짐 로스가 오르면 짚어낸다. 다독임(드롭아웃, 짐 줄이기, 데이터 증강)이나 모델 크기 줄이기로 고친다. 학습과 따짐 자를 늘 함께 지켜보아 이를 일찍 잡아야 한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
`RMSNorm`을 켜나 덩이의 수를 골라 잡을 수 있게 넓혀라. `__init__`에 `num_layers` 파라메터를 더하고 `nn.ModuleList`으로 깊이를 바꿀 수 있는 구조를 짜라. 켜 2개, 4개, 8개로 시험하여라.

</div>

??? success "연습문제 4 풀이"
    박아 넣은 켜를 다음으로 갈음한다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 방법에서 `for layer in self.layers: x = layer(x)`으로 되돈다. 여느 파이썬 목록이 아니라 `nn.ModuleList`을 써야 PyTorch가 파라메터를 모두 다듬기에 올린다. `for n in [2, 4, 8]: model = RMSNorm(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.

## 정리하며

**다룬 것** — 잣대 잡기 켜

`RMSNorm` 클래스는 PyTorch의 `nn.Module` 낯을 써서 모델 구조를 담는다.

핵심 클래스는 `RMSNorm`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
