# Vanilla RNN

바닐라 RNN - 순환 뉴럴 네트워크(엘만 RNN). 고전적 아이디어: 시간에 따라 차례로 갱신되는 은닉 상태를 지닌다.

여기 구현은 Vanilla RNN을 짧고 학습하기 좋게 보인 예제다. 코드는 핵심 구조와 순전파(forward pass)에 초점을 두어, 핵심 구성 요소를 살펴보고 이리저리 바꾸어 보기 쉽다.

## 1. 코드

```python
#!/usr/bin/env python3
"""
바닐라 RNN - 순환 뉴럴 네트워크(엘만 RNN)
고전적 아이디어: 시간에 따라 차례로 갱신되는 은닉 상태를 지닌다.

출처: "시간 속에서 구조 찾기(Finding Structure in Time)" (1990), 제프리 엘만(단순한 RNN을 널리 알렸다)
핵심: h_t = tanh(W_x x_t + W_h h_{t-1} + b)

파일: appendix/sequence/rnn.py
참고: 주석을 빠짐없이 단, 교육용 구현이다(레이어 하나, 배치를 앞에 둔다).
"""

import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================


class RNNCell(nn.Module):
    """
    바닐라 RNN 셀 하나(시간 스텝 하나).

    형태:
      x_t     : (B, input_size)
      h_prev  : (B, hidden_size)
      h_t     : (B, hidden_size)

    갱신 식:
      h_t = tanh( W_x * x_t + W_h * h_{t-1} + b )
    """
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 입력과 이전 은닉 상태를 변환하는 선형 레이어
        self.Wx = nn.Linear(input_size, hidden_size, bias=True)
        self.Wh = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        # 변환한 입력과 은닉 상태를 더한 뒤 tanh을 적용한다
        h_t = torch.tanh(self.Wx(x_t) + self.Wh(h_prev))
        return h_t


class RNN(nn.Module):
    """
    바닐라 RNN(시간 축으로 직접 펼침).

    입력:
      x : (B, T, input_size)  배치를 앞에 둔 시퀀스
    출력:
      y : (B, T, hidden_size) 모든 시간 스텝의 은닉 상태
      h_T : (B, hidden_size) 마지막 은닉 상태
    """
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.cell = RNNCell(input_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor, h0: torch.Tensor | None = None):
        # 배치 크기와 시퀀스 길이를 꺼낸다
        B, T, _ = x.shape
        device = x.device

        # 은닉 상태가 주어지지 않으면 0으로 초기화한다
        if h0 is None:
            h_t = torch.zeros(B, self.hidden_size, device=device)
        else:
            h_t = h0

        outputs = []
        for t in range(T):
            # 시간 스텝 t의 입력을 가져온다
            x_t = x[:, t, :]               # (B, input_size)

            # RNN 셀로 은닉 상태를 갱신한다
            h_t = self.cell(x_t, h_t)      # (B, hidden_size)

            # 이 시간 스텝의 은닉 상태를 저장한다
            outputs.append(h_t)

        # 모든 시간 스텝의 은닉 상태를 쌓아 텐서 하나로 만든다
        y = torch.stack(outputs, dim=1)     # (B, T, hidden_size)
        return y, h_t


if __name__ == "__main__":
    # 간단한 동작 확인: 순전파를 돌리고 형태를 출력한다
    model = RNN(input_size=8, hidden_size=16)
    x = torch.randn(2, 5, 8)     # (B=2, T=5, input=8)
    y, hT = model(x)

    print("y :", y.shape)        # expected (2, 5, 16)
    print("hT:", hT.shape)       # expected (2, 16)
```

**출력:**

```
y : torch.Size([2, 5, 16])
hT: torch.Size([2, 16])
```

## 2. 논의

이 구현은 클래스 2개(`RNNCell`, `RNN`)를 정의하고, 이들이 함께 완전한 시퀀스 모델 구조를 이룬다. 클래스마다 고유한 역할을 맡아 코드를 모듈화하고 확장하기 쉽게 한다. `forward` 메서드가 PyTorch의 자동 미분에 쓰이는 계산 그래프를 정의한다.

이 코드는 구조를 이해하기 위한 교육용 구현이다. 실제 학습에서는 데이터와 하드웨어에 맞춰 혼합 정밀도, 분산 학습, 데이터 어그멘테이션 등을 검토할 수 있다. 다만 이런 기법의 적용 조건과 효과는 모델별로 확인해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
기본 설정으로 초기화한 `RNNCell`의 학습 가능한 파라메터를 모두 세어라. 웨이트와 바이어스를 포함해 레이어마다 나누어 적어라.

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
`RNNCell`을 레이어나 블록의 수를 설정할 수 있게 확장하라. `__init__`에 `num_layers` 인자를 더하고 `nn.ModuleList`으로 깊이를 바꿀 수 있는 구조를 만들어라. 레이어 2개, 4개, 8개로 시험하여라.

</div>

??? success "풀이"
    하드코딩된 레이어를 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`으로 반복한다. 일반 파이썬 리스트가 아니라 `nn.ModuleList`을 써야 PyTorch가 모든 파라메터를 옵티마이저에 등록한다. `for n in [2, 4, 8]: model = RNNCell(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.

## 정리하며

**다룬 것** — Vanilla RNN

이 구현은 클래스 2개(`RNNCell`, `RNN`)를 정의하고, 이들이 함께 완전한 시퀀스 모델 구조를 이룬다.

핵심 클래스는 `RNNCell`, `RNN`이며 앞의 연습문제 4개로 스스로 확인해 볼 수 있다.
