# 잣대 잡기 켜

잣대 잡기 켜 - 깊은 배움에 쓰이는 흔한 갈래. 담긴 것:

여기 짜보기는 Normalization Layers을 짧고 배우기 좋게 보인 본이다. 코드는 고갱이 얼개와 앞으로 걸음에 마음을 두어, 고갱이 꾸밈새를 살펴보고 이리저리 바꾸어 보기 쉽다.

## 코드

```python
#!/usr/bin/env python3
"""
잣대 잡기 켜 - 깊은 배움에 쓰이는 흔한 갈래
담긴 것:
  - BatchNorm1d/2d(PyTorch 붙박이를 감싼 것)
  - LayerNorm(PyTorch 붙박이를 감싼 것)
  - GroupNorm(PyTorch 붙박이를 감싼 것)
  - RMSNorm(요즘 큰 말 모형에서 흔히 쓴다)

두루마리: appendix/utils/normalization.py
눈여겨볼 것: 저마다 언제 쓰는지 주석으로 밝힌 배우기용 짜보기다.
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
    pass```

## 논의

`RMSNorm` 갈래는 PyTorch의 `nn.Module` 낯을 써서 모형 얼개를 담는다. `forward` 방법이 셈 그림을 매기므로 익히는 동안 PyTorch의 autograd가 기울기 셈을 절로 다룬다. 이렇게 묶음으로 나눈 꾸밈 덕에 몫 하나하나를 고치거나 더 큰 흐름에 넣기가 쉽다.

여기 실린 코드는 본보기 짜보기라 다듬기보다 알아보기 쉬움을 앞세운다. 서비스 얼개라면 흔히 섞인 촘촘함 익히기, 흩은 자료 나란히, 더 정교한 자료 불리기를 더한다. 그래도 여기서 보인 얼개의 고갱이 깨침은 크기와 상관없이 그대로다.

## 익힘 문제

**익힘 1.**
코드를 읽고 고갱이가 되는 설계 판단을 짚어라. 짜기에서 고른 것 셋을 들고, 저마다 왜 잔손질 묶음에 알맞은지 밝혀라.

??? success "익힘 1 풀이"
    설계 판단은 짜보기마다 다르나 흔히 이런 것이 있다. (1) 살림 함수 고르기 -- ReLU 갈래는 기울기가 잦아들지 않아 익히기가 빠르다. (2) 잣대 잡는 꾀 -- 묶음 잣대 잡기가 안쪽 함께 바뀌는 옮겨감을 줄여 익힘을 든든하게 한다. (3) 나머지 이음 -- 있으면 건너뛰는 길을 주어 깊은 그물에서 기울기가 흐르게 한다. 고른 것마다 드러내는 힘, 셈 값, 익힘의 든든함 사이의 맞바꿈을 보여 준다.

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
`RMSNorm`을 켜나 덩이의 수를 골라 잡을 수 있게 넓혀라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`으로 깊이를 바꿀 수 있는 얼개를 짜라. 켜 2개, 4개, 8개로 시험하여라.

??? success "익힘 4 풀이"
    박아 넣은 켜를 다음으로 갈음한다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 방법에서 `for layer in self.layers: x = layer(x)`으로 되돈다. 여느 파이썬 목록이 아니라 `nn.ModuleList`을 써야 PyTorch가 매개변수를 모두 다듬기에 올린다. `for n in [2, 4, 8]: model = RMSNorm(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.
