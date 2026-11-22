# 활성화 함수

자주 쓰는 비선형 활성화 함수를 모았다. 포함한 항목은 코드 첫머리의 독스트링에 적혀 있다.

이 쪽의 구현은 활성화 함수의 핵심만 남긴 교육용 예시다. ReLU·LeakyReLU·GELU·SiLU는 PyTorch 내장 구현을 쓰라는 메모로 두고, 직접 구현한 것은 게이트 활성화 함수 GLU와 SwiGLU 둘이다. 코드는 핵심 구조와 순전파(forward pass, 입력에서 출력까지의 계산)에 집중하므로, 주요 구성 요소를 살펴보고 여러 가지로 바꾸어 보기 쉽다.

## 1. 코드

```python
#!/usr/bin/env python3
"""
활성화 함수 - 자주 쓰는 비선형 함수
포함한 것:
  - ReLU
  - LeakyReLU
  - GELU
  - SiLU (Swish)
  - GLU / SwiGLU(트랜스포머와 대형 언어 모델에서 쓰는 게이트 활성화 함수)

파일: appendix/utils/activations.py
참고: 교육용 예시 구현과 사용법 메모다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# 메인
# ========================================================================


class GLU(nn.Module):
    """
    게이트 선형 유닛(Gated Linear Unit):
      GLU(x) = (xW) * sigmoid(xV)
    """
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.fc = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        # 값과 게이트로 나눈다
        v, g = self.fc(x).chunk(2, dim=-1)
        return v * torch.sigmoid(g)


class SwiGLU(nn.Module):
    """
    SwiGLU(LLaMA나 PaLM 형태 피드포워드에서 쓴다):
      SwiGLU(x) = (SiLU(xW1) * (xW3)) W2
    """
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# 참고:
# 표준 활성화 함수는 PyTorch 내장 구현을 쓰는 편이 낫다:
#   F.relu(x), nn.ReLU()
#   F.gelu(x), nn.GELU()
#   F.silu(x), nn.SiLU()


if __name__ == "__main__":
    pass
```

## 2. 논의

이 구현은 클래스 2개(`GLU`, `SwiGLU`)를 정의하고, 이들이 함께 하나의 유틸리티 모듈을 이룬다. 클래스마다 고유한 역할을 맡아 코드가 모듈로 나뉘고 확장하기 쉽다. `forward` 메서드가 PyTorch의 자동 미분이 쓰는 계산 그래프를 정의한다.

이 코드는 구조를 이해하기 위한 교육용 구현이다. 실제 트레인에서는 데이터와 하드웨어에 맞춰 혼합 정밀도, 분산 트레인, 데이터 어그멘테이션 등을 검토할 수 있다. 다만 이런 기법의 적용 조건과 효과는 모델별로 확인해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
기본 설정으로 만든 `GLU`의 트레인 가능한 파라메터를 모두 세어라. 웨이트와 바이어스를 포함해 레이어마다 나누어 적어라.

</div>

??? success "풀이"
    `nn.Linear(in_features, out_features)`마다 웨이트 파라메터가 `in_features * out_features`개이고 바이어스가 `out_features`개다(`bias=False`가 아니면). `nn.Conv2d(in_c, out_c, k)`마다 웨이트가 `in_c * out_c * k * k`개, 바이어스가 `out_c`개다. `nn.Embedding(num, dim)`은 파라메터가 `num * dim`개다. 모든 레이어의 값을 더한다. `sum(p.numel() for p in model.parameters())`로 확인할 수 있다. `GLU`는 `dim_in`과 `dim_out`에 기본값이 없으므로 두 값을 $m$, $n$이라 하면, 레이어는 바이어스가 있는 `nn.Linear(dim_in, dim_out * 2)` 하나뿐이라 웨이트 $2mn$개와 바이어스 $2n$개, 모두 $2mn + 2n$개다. 출력을 값과 게이트로 반씩 나누어 쓰므로 같은 출력 크기의 보통 선형 레이어보다 파라메터가 두 배다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
입력의 형태와 자료형이 기대한 대로인지 확인하는 입력 검증을 주요 함수나 클래스에 추가하여라. 올바르지 않은 입력에는 이해하기 쉬운 오류 메시지를 띄워라.

</div>

??? success "풀이"
    `forward` 메서드(또는 알맞은 함수)의 첫머리에 `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`이나 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'` 같은 검사를 추가한다. 형태를 검사하려면 중요한 차원을 본다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 이해하기 쉬운 오류 메시지는 디버깅을 크게 앞당기고 코드를 재사용하기에 더 견고하게 만든다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
이 구현에서 일어날 만한 실패 유형 둘을 들고, 각각 어떻게 탐지하고 고칠지 설명하여라.

</div>

??? success "풀이"
    흔한 실패 유형은 이렇다. (1) **그래디언트 소실 또는 폭발** -- 그래디언트 크기를 관찰해 탐지한다(`torch.nn.utils.clip_grad_norm_`이나 레이어마다 `param.grad.norm()` 기록하기). 그래디언트 클리핑, 더 나은 초기화(Xavier/Kaiming), 구조 변경(잔차 연결, 정규화 레이어)으로 고친다. (2) **과적합** -- 트레인 로스는 줄어드는데 검증 로스가 오르면 탐지한다. 정칙화(드롭아웃, 웨이트 감쇠, 데이터 어그멘테이션)나 모델 크기 줄이기로 고친다. 트레인 지표와 검증 지표를 늘 함께 관찰해 이를 일찍 잡아야 한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
`GLU`를 레이어나 블록의 수를 설정할 수 있게 확장하여라. `__init__`에 `num_layers` 인자를 추가하고 `nn.ModuleList`로 깊이를 바꿀 수 있는 구조를 구현하여라. 레이어 2개, 4개, 8개로 시험하여라.

</div>

??? success "풀이"
    하드코딩한 레이어를 다음으로 대체한다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`로 순회한다. 일반 파이썬 리스트가 아니라 `nn.ModuleList`를 써야 PyTorch가 그 안의 파라메터를 모델에 등록하고, 옵티마이저가 그것들을 갱신할 수 있다. `for n in [2, 4, 8]: model = GLU(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.

## 정리하며

**다룬 것** — 활성화 함수

이 구현은 클래스 2개(`GLU`, `SwiGLU`)를 정의하고, 이들이 함께 하나의 유틸리티 모듈을 이룬다.

핵심 클래스는 `GLU`, `SwiGLU`이며 앞의 연습문제 4개로 스스로 확인할 수 있다.
