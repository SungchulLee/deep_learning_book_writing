# 정규화 레이어

딥러닝에서 자주 쓰는 정규화 레이어를 모았다. 포함한 항목은 코드 첫머리의 독스트링에 적혀 있다.

이 쪽의 구현은 정규화 레이어의 핵심만 남긴 교육용 예시다. BatchNorm·LayerNorm·GroupNorm은 PyTorch 내장 구현을 쓰라는 메모로 두고, 직접 구현한 것은 RMSNorm 하나다. 코드는 핵심 구조와 순전파(forward pass, 입력에서 출력까지의 계산)에 집중하므로, 주요 구성 요소를 살펴보고 여러 가지로 바꾸어 보기 쉽다.

## 1. 코드

```python
#!/usr/bin/env python3
"""
정규화 레이어 - 딥러닝에서 자주 쓰는 종류
포함한 것:
  - BatchNorm1d/2d(PyTorch 내장 구현을 감싼 것)
  - LayerNorm(PyTorch 내장 구현을 감싼 것)
  - GroupNorm(PyTorch 내장 구현을 감싼 것)
  - RMSNorm(최근 대형 언어 모델에서 흔히 쓴다)

파일: appendix/utils/normalization.py
참고: 각각 언제 쓰는지 주석으로 설명한 교육용 구현이다.
"""

import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================


class RMSNorm(nn.Module):
    """
    RMSNorm(제곱 평균 제곱근 레이어 정규화)

    레이어 정규화와 달리 RMSNorm은:
      - 평균을 빼지 않는다
      - RMS(sqrt(mean(x^2)))로 나누기만 한다

    LLaMA 형태 모델에서 흔히 쓴다.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: (..., dim)
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


# 참고:
# - BatchNorm, LayerNorm, GroupNorm은 PyTorch에 들어 있다.
# - 흔히 이렇게 임포트해 쓴다:
#     nn.BatchNorm2d(C)
#     nn.LayerNorm(D)
#     nn.GroupNorm(num_groups, C)
#
# 이 파일은 RMSNorm과 짧은 사용법 메모를 더한다.


if __name__ == "__main__":
    pass
```

## 2. 논의

`RMSNorm` 클래스는 PyTorch의 `nn.Module` 인터페이스를 써서 모델 구조를 담는다. `forward` 메서드가 계산 그래프를 정의하므로 트레인하는 동안 PyTorch의 autograd가 그래디언트 계산을 자동으로 처리한다. 이렇게 모듈로 나눈 설계 덕에 구성 요소 하나하나를 고치거나 더 큰 파이프라인에 넣기가 쉽다.

이 코드는 구조를 이해하기 위한 교육용 구현이다. 실제 트레인에서는 데이터와 하드웨어에 맞춰 혼합 정밀도, 분산 트레인, 데이터 어그멘테이션 등을 검토할 수 있다. 다만 이런 기법의 적용 조건과 효과는 모델별로 확인해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
코드를 읽고 핵심이 되는 설계 판단을 짚어라. 구현에서 선택한 것 셋을 들고, 각각 왜 유틸리티 모듈에 알맞은지 설명하여라.

</div>

??? success "풀이"
    `RMSNorm`에서는 이런 셋을 들 수 있다. (1) 평균을 빼지 않고 RMS로만 나눈다 -- 레이어 정규화에서 평균 계산과 바이어스(이동) 파라메터를 덜어 내 계산이 가벼워진다. 크기를 맞추는 효과(재스케일 불변성)만으로도 트레인이 충분히 안정된다는 것이 RMSNorm의 주장이다. (2) `eps`를 제곱근 안에 더한다 -- 한 벡터의 원소가 모두 0이어도 0으로 나누지 않는다. (3) 마지막 차원(`dim=-1`)을 따라 샘플마다 정규화하고, 트레인 가능한 `weight`를 1로 초기화한다 -- 배치 정규화와 달리 배치 통계를 쓰지 않으므로 배치 크기와 상관없고 트레인과 추론에서 동작이 같다. 처음에는 순수한 정규화로 시작하고, 차원마다 스케일은 트레인이 되찾는다. 셋 다 대형 언어 모델처럼 시퀀스 길이와 배치 크기가 들쭉날쭉한 곳에 쓰기 좋은 선택이다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
입력의 형태와 자료형이 기대한 대로인지 확인하는 입력 검증을 주요 함수나 클래스에 추가하여라. 올바르지 않은 입력에는 이해하기 쉬운 오류 메시지를 띄워라.

</div>

??? success "풀이"
    `forward` 메서드(또는 알맞은 함수)의 첫머리에 `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`이나 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'` 같은 검사를 추가한다. 형태를 검사하려면 중요한 차원을 본다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 이해하기 쉬운 오류 메시지는 디버깅을 크게 앞당기고 코드를 재사용하기에 더 견고하게 만든다. `RMSNorm`은 입력이 `(..., dim)` 형태라 차원 수는 자유롭고, 확인할 것은 `x.size(-1) == self.weight.numel()` 하나다.

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
`RMSNorm`을 레이어나 블록의 수를 설정할 수 있게 확장하여라. `__init__`에 `num_layers` 인자를 추가하고 `nn.ModuleList`로 깊이를 바꿀 수 있는 구조를 구현하여라. 레이어 2개, 4개, 8개로 시험하여라.

</div>

??? success "풀이"
    `RMSNorm`은 그 자체로 레이어 하나라 깊이를 가질 수 없다. 그래서 `RMSNorm`을 담은 블록(예: `RMSNorm` → `nn.Linear` → 엑티베이션 함수)을 아래의 `YourBlock` 자리에 넣어 쌓는다. 하드코딩한 레이어를 다음으로 대체한다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`로 순회한다. 일반 파이썬 리스트가 아니라 `nn.ModuleList`를 써야 PyTorch가 그 안의 파라메터를 모델에 등록하고, 옵티마이저가 그것들을 갱신할 수 있다. `for n in [2, 4, 8]: model = RMSNorm(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.

## 정리하며

**다룬 것** — 정규화 레이어

`RMSNorm` 클래스는 PyTorch의 `nn.Module` 인터페이스를 써서 모델 구조를 담는다.

핵심 클래스는 `RMSNorm`이며 앞의 연습문제 4개로 스스로 확인할 수 있다.
