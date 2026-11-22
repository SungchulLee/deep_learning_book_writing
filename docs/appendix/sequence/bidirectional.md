# 양방향 RNN / GRU / LSTM

양방향 RNN은 시퀀스를 양쪽 방향으로 읽는다. RNN 하나는 순방향($t=1,\dots,T$)으로, 다른 하나는 역방향($t=T,\dots,1$)으로 돌린 뒤 두 출력을 시간 스텝마다 결합하므로, 각 위치의 출력이 앞 문맥과 뒤 문맥을 함께 담는다. GRU나 LSTM 셀에도 똑같이 쓸 수 있다.

여기 구현은 양방향 RNN을 짧고 트레인하기 좋게 보인 예제다. 셀은 바닐라 RNN 셀만 쓴다. 코드는 핵심 구조와 순전파(forward pass)에 초점을 두어, 핵심 구성 요소를 살펴보고 이리저리 바꾸어 보기 쉽다.

## 1. 코드

```python
#!/usr/bin/env python3
"""
양방향 RNN / GRU / LSTM - 시퀀스를 양방향으로 처리하기
핵심 아이디어: RNN 하나는 순방향(t=1..T)으로, 다른 하나는 역방향(t=T..1)으로 돌린 뒤
그 출력을 결합한다(이어 붙이거나 더한다).

이 파일이 제공하는 것:
  - *셀 기반* RNN(바닐라), GRU, LSTM 방식 모듈을 양방향으로 감싸는 래퍼
  - 단순하고 이해하기 쉽도록 여기서는 양방향 바닐라 RNN을 구현한다.

파일: appendix/sequence/bidirectional.py
참고: 주석을 빠짐없이 단, 교육용 구현이다(배치를 앞에 둔다).
"""

import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================


class RNNCell(nn.Module):
    """
    단순한 바닐라 RNN 셀을 재사용한다.

    h_t = tanh(Wx x_t + Wh h_{t-1})
    """
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.Wx = nn.Linear(input_size, hidden_size, bias=True)
        self.Wh = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.Wx(x_t) + self.Wh(h_prev))


class BidirectionalRNN(nn.Module):
    """
    양방향 바닐라 RNN.

    구성:
      - 순방향 은닉 상태 h_f (왼쪽 -> 오른쪽)
      - 역방향 은닉 상태 h_b (오른쪽 -> 왼쪽)

    시간 스텝 t마다:
      순방향: h_f[t] = f(x[t], h_f[t-1])
      역방향: h_b[t] = f(x[t], h_b[t+1])   (시간을 거꾸로 돌며 계산한다)

    출력 결합(흔한 선택):
      - 이어 붙이기: y[t] = [h_f[t], h_b[t]]  -> 차원 2*hidden
      - 더하기:    y[t] = h_f[t] + h_b[t]  -> 차원 hidden

    여기서는 이어 붙이기를 구현한다(가장 흔하다).
    """
    def __init__(self, input_size: int, hidden_size: int, concat: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.concat = concat

        # 독립된 RNN 셀 둘: 하나는 순방향, 하나는 역방향
        self.cell_f = RNNCell(input_size, hidden_size)
        self.cell_b = RNNCell(input_size, hidden_size)

    def forward(self, x: torch.Tensor, h0_f: torch.Tensor | None = None, h0_b: torch.Tensor | None = None):
        """
        x: (B, T, input_size)

        반환:
          y: concat=True이면 (B, T, 2*hidden), 아니면 (B, T, hidden)
          (hT_f, hT_b): 마지막 순방향/역방향 은닉 상태 (B, hidden)
        """
        B, T, _ = x.shape
        device = x.device

        # 은닉 상태가 주어지지 않으면 초기화한다
        h_f = torch.zeros(B, self.hidden_size, device=device) if h0_f is None else h0_f
        h_b = torch.zeros(B, self.hidden_size, device=device) if h0_b is None else h0_b

        # ---- 순방향 패스 (t = 0..T-1) ----
        forward_states = []
        for t in range(T):
            x_t = x[:, t, :]               # (B, input)
            h_f = self.cell_f(x_t, h_f)    # update forward hidden
            forward_states.append(h_f)     # store h_f[t]

        # ---- 역방향 패스 (t = T-1..0) ----
        backward_states_reversed = []
        for t in reversed(range(T)):
            x_t = x[:, t, :]               # (B, input)
            h_b = self.cell_b(x_t, h_b)    # update backward hidden (moving right->left)
            backward_states_reversed.append(h_b)  # this is h_b[t], but collected reversed

        # 역방향 리스트를 다시 뒤집어 시간 순서 0..T-1에 맞춘다
        backward_states = list(reversed(backward_states_reversed))

        # ---- 시간 스텝마다 순방향/역방향 상태를 결합한다 ----
        y_steps = []
        for t in range(T):
            hf_t = forward_states[t]       # (B, hidden)
            hb_t = backward_states[t]      # (B, hidden)

            if self.concat:
                # 피처 차원을 따라 이어 붙인다 -> (B, 2*hidden)
                y_t = torch.cat([hf_t, hb_t], dim=1)
            else:
                # 더한다 -> (B, hidden)
                y_t = hf_t + hb_t

            y_steps.append(y_t)

        # 쌓아서 (B, T, feat_dim)으로 만든다
        y = torch.stack(y_steps, dim=1)

        # 마지막 은닉 상태(방향마다 마지막으로 갱신한 뒤)
        hT_f = forward_states[-1]          # (B, hidden)
        hT_b = backward_states[0]          # (B, hidden)  (backward final corresponds to t=0 in aligned order)

        return y, (hT_f, hT_b)


if __name__ == "__main__":
    # 간단한 동작 확인
    model = BidirectionalRNN(input_size=8, hidden_size=16, concat=True)
    x = torch.randn(2, 5, 8)

    y, (hF, hB) = model(x)
    print("y :", y.shape)   # expected (2, 5, 32) when concat=True
    print("hF:", hF.shape)  # expected (2, 16)
    print("hB:", hB.shape)  # expected (2, 16)
```

**출력:**

```
y : torch.Size([2, 5, 32])
hF: torch.Size([2, 16])
hB: torch.Size([2, 16])
```

## 2. 논의

이 구현은 클래스 2개(`RNNCell`, `BidirectionalRNN`)를 정의하고, 이들이 함께 완전한 시퀀스 모델 구조를 이룬다. 클래스마다 고유한 역할을 맡아 코드를 모듈화하고 확장하기 쉽게 한다. `forward` 메서드가 PyTorch의 자동 미분에 쓰이는 계산 그래프를 정의한다.

이 코드는 구조를 이해하기 위한 교육용 구현이다. 실제 트레인에서는 데이터와 하드웨어에 맞춰 혼합 정밀도, 분산 트레인, 데이터 어그멘테이션 등을 검토할 수 있다. 다만 이런 기법의 적용 조건과 효과는 모델별로 확인해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
기본 설정으로 초기화한 `RNNCell`의 트레인 가능한 파라메터를 모두 세어라. 웨이트와 바이어스를 포함해 레이어마다 나누어 적어라.

</div>

??? success "풀이"
    `nn.Linear(in_features, out_features)`마다 웨이트 파라메터가 `in_features * out_features`개이고 바이어스가 `out_features`개다(`bias=False`가 아니면). `nn.Conv2d(in_c, out_c, k)`마다 웨이트가 `in_c * out_c * k * k`개, 바이어스가 `out_c`개다. `nn.Embedding(num, dim)`은 파라메터가 `num * dim`개다. 모든 레이어의 값을 더한다. `sum(p.numel() for p in model.parameters())`로 확인할 수 있다.

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
입력 크기가 $x$, 은닉 크기가 $h$일 때 LSTM 셀과 GRU 셀의 파라메터 수를 비교하여라. 어느 쪽이 더 적고 왜 그런가?

</div>

??? success "풀이"
    LSTM에는 게이트가 넷(입력, 망각, 셀, 출력) 있고 각각 입력과 은닉 상태의 웨이트 행렬을 지니므로 파라메터는 $4 \times (x \cdot h + h \cdot h + h) = 4(xh + h^2 + h)$개다. GRU에는 게이트가 셋(리셋, 업데이트, 후보) 있어 $3 \times (x \cdot h + h \cdot h + h) = 3(xh + h^2 + h)$개다. GRU는 게이트가 넷이 아니라 셋이고 셀 상태와 은닉 상태를 하나로 합치므로 LSTM의 75%다. 실제로는 파라메터가 적어도 GRU가 LSTM과 비슷한 성능을 내는 일이 잦다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
`RNNCell`을 레이어나 블록의 수를 설정할 수 있게 확장하라. `__init__`에 `num_layers` 인자를 더하고 `nn.ModuleList`로 깊이를 바꿀 수 있는 구조를 만들어라. 레이어 2개, 4개, 8개로 시험하여라.

</div>

??? success "풀이"
    하드코딩된 레이어를 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`로 반복한다. 일반 파이썬 리스트가 아니라 `nn.ModuleList`를 써야 PyTorch가 모든 파라메터를 옵티마이저에 등록한다. `for n in [2, 4, 8]: model = RNNCell(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`로 시험한다.

## 정리하며

**다룬 것** — 양방향 RNN / GRU / LSTM

이 구현은 클래스 2개(`RNNCell`, `BidirectionalRNN`)를 정의하고, 이들이 함께 완전한 시퀀스 모델 구조를 이룬다.

핵심 클래스는 `RNNCell`, `BidirectionalRNN`이며 앞의 연습문제 4개로 스스로 확인해 볼 수 있다.
