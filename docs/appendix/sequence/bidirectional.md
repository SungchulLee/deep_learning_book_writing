# 두 방향 RNN / GRU / LSTM

두 방향 RNN / GRU / LSTM - 이음을 두 방향으로 다루기. 고갱이 깨침: RNN 하나는 앞으로(t=1..T), 다른 하나는 뒤로(t=T..1) 돌린다,

여기 짜보기는 Bidirectional RNN / GRU / LSTM을 짧고 배우기 좋게 보인 본이다. 코드는 고갱이 얼개와 앞으로 걸음에 마음을 두어, 고갱이 꾸밈새를 살펴보고 이리저리 바꾸어 보기 쉽다.

## 1. 코드

```python
#!/usr/bin/env python3
"""
두 방향 RNN / GRU / LSTM - 이음을 두 방향으로 다루기
고갱이 깨침: RNN 하나는 앞으로(t=1..T), 다른 하나는 뒤로(t=T..1) 돌린 뒤
그 날임을 아우른다(이어 붙이거나 더한다).

이 두루마리가 주는 것:
  - *칸 바탕* RNN(맨 것), GRU, LSTM 꼴 묶음을 두 방향으로 감싸는 것
  - 단순하고 알아보기 쉽도록 여기서는 두 방향 맨 RNN을 짠다.

두루마리: appendix/sequence/bidirectional.py
눈여겨볼 것: 주석을 빠짐없이 단, 배우기 위한 짜보기다(묶음을 앞에 둔다).
"""

import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================


class RNNCell(nn.Module):
    """
    단순한 맨 RNN 칸을 되쓴다.

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
    두 방향 맨 RNN.

    지니는 것:
      - 앞으로 가는 숨은 상태 h_f (왼쪽 -> 오른쪽)
      - 뒤로 가는 숨은 상태 h_b (오른쪽 -> 왼쪽)

    때 걸음 t마다:
      앞으로:  h_f[t] = f(x[t], h_f[t-1])
      뒤로: h_b[t] = f(x[t], h_b[t+1])   (때를 거꾸로 돌며 셈한다)

    날임 아우르기(흔히 고르는 것):
      - 이어 붙이기: y[t] = [h_f[t], h_b[t]]  -> 차수 2*hidden
      - 더하기:    y[t] = h_f[t] + h_b[t]  -> 차수 hidden

    여기서는 이어 붙이기를 짠다(가장 흔하다).
    """
    def __init__(self, input_size: int, hidden_size: int, concat: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.concat = concat

        # 따로 도는 RNN 칸 둘: 하나는 앞으로, 하나는 뒤로
        self.cell_f = RNNCell(input_size, hidden_size)
        self.cell_b = RNNCell(input_size, hidden_size)

    def forward(self, x: torch.Tensor, h0_f: torch.Tensor | None = None, h0_b: torch.Tensor | None = None):
        """
        x: (B, T, input_size)

        돌려주는 것:
          y: concat=True이면 (B, T, 2*hidden), 아니면 (B, T, hidden)
          (hT_f, hT_b): 마지막 앞으로/뒤로 숨은 상태 (B, hidden)
        """
        B, T, _ = x.shape
        device = x.device

        # 숨은 상태가 주어지지 않으면 첫자리를 잡는다
        h_f = torch.zeros(B, self.hidden_size, device=device) if h0_f is None else h0_f
        h_b = torch.zeros(B, self.hidden_size, device=device) if h0_b is None else h0_b

        # ---- 앞으로 걸음 (t = 0..T-1) ----
        forward_states = []
        for t in range(T):
            x_t = x[:, t, :]               # (B, input)
            h_f = self.cell_f(x_t, h_f)    # update forward hidden
            forward_states.append(h_f)     # store h_f[t]

        # ---- 뒤로 걸음 (t = T-1..0) ----
        backward_states_reversed = []
        for t in reversed(range(T)):
            x_t = x[:, t, :]               # (B, input)
            h_b = self.cell_b(x_t, h_b)    # update backward hidden (moving right->left)
            backward_states_reversed.append(h_b)  # this is h_b[t], but collected reversed

        # 뒤로 간 목록을 되뒤집어 때 차례 0..T-1에 맞춘다
        backward_states = list(reversed(backward_states_reversed))

        # ---- 때 걸음마다 앞으로/뒤로 상태를 아우른다 ----
        y_steps = []
        for t in range(T):
            hf_t = forward_states[t]       # (B, hidden)
            hb_t = backward_states[t]      # (B, hidden)

            if self.concat:
                # 결 차수를 따라 이어 붙인다 -> (B, 2*hidden)
                y_t = torch.cat([hf_t, hb_t], dim=1)
            else:
                # 더한다 -> (B, hidden)
                y_t = hf_t + hb_t

            y_steps.append(y_t)

        # 쌓아서 (B, T, feat_dim)으로 만든다
        y = torch.stack(y_steps, dim=1)

        # 마지막 숨은 상태(방향마다 마지막으로 고친 뒤)
        hT_f = forward_states[-1]          # (B, hidden)
        hT_b = backward_states[0]          # (B, hidden)  (backward final corresponds to t=0 in aligned order)

        return y, (hT_f, hT_b)


if __name__ == "__main__":
    # 얼른 해 보는 맛보기 살핌
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

이 짜보기는 갈래 2개(`RNNCell`, `BidirectionalRNN`)를 매기고, 이들이 어울려 온전한 이음 모형 얼개를 이룬다. 갈래마다 남다른 몫을 담아 코드를 묶음으로 나누고 넓히기 쉽게 한다. `forward` 방법이 PyTorch가 절로 미분하는 데 쓰는 셈 그림을 매긴다.

여기 실린 코드는 본보기 짜보기라 다듬기보다 알아보기 쉬움을 앞세운다. 서비스 얼개라면 흔히 섞인 촘촘함 익히기, 흩은 자료 나란히, 더 정교한 자료 불리기를 더한다. 그래도 여기서 보인 얼개의 고갱이 깨침은 크기와 상관없이 그대로다.

## 연습문제

**연습문제 1.**
기본 첫자리로 잡은 `RNNCell`의 배울 수 있는 매개변수를 모두 세어라. 짐과 치우침을 아울러 켜마다 나누어 적어라.

??? success "연습문제 1 풀이"
    `nn.Linear(in_features, out_features)`마다 짐 매개변수가 `in_features * out_features`개이고 치우침이 `out_features`개다(`bias=False`가 아니면). `nn.Conv2d(in_c, out_c, k)`마다 짐이 `in_c * out_c * k * k`개, 치우침이 `out_c`개다. `nn.Embedding(num, dim)`은 매개변수가 `num * dim`개다. 켜를 모두 더한다. `sum(p.numel() for p in model.parameters())`으로 따져 볼 수 있다.

---

**연습문제 2.**
들임의 꼴과 자료 갈래가 바라는 대로인지 살피는 들임 살피기를 으뜸 함수나 갈래에 더하여라. 올바르지 않은 들임에는 알아듣기 쉬운 어긋남 알림을 띄워라.

??? success "연습문제 2 풀이"
    `forward` 방법(또는 알맞은 함수)의 첫머리에 `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`이나 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'` 같은 살핌을 더한다. 꼴을 살피려면 종요로운 차수를 본다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 알아듣기 쉬운 어긋남 알림은 벌레잡기를 크게 앞당기고 코드를 되쓰기 든든하게 한다.

---

**연습문제 3.**
숨은 크기 $h$과 들임 크기 $x$이 같을 때 LSTM 칸과 GRU 칸의 매개변수 수를 견주어라. 어느 쪽이 더 적고 왜 그런가?

??? success "연습문제 3 풀이"
    LSTM에는 문이 넷(들임, 잊음, 칸, 날임) 있고 저마다 들임과 숨은 상태의 짐 행렬을 지니므로 매개변수는 $4 \times (x \cdot h + h \cdot h + h) = 4(xh + h^2 + h)$개다. GRU에는 문이 셋(되돌림, 고침, 새것) 있어 $3 \times (x \cdot h + h \cdot h + h) = 3(xh + h^2 + h)$개다. GRU은 문이 넷이 아니라 셋이고 칸 상태와 숨은 상태를 하나로 묶으므로 LSTM의 75%이다. 참으로는 매개변수가 적어도 GRU이 LSTM과 엇비슷하게 잘 듣는 일이 잦다.

---

**연습문제 4.**
`RNNCell`을 켜나 덩이의 수를 골라 잡을 수 있게 넓혀라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`으로 깊이를 바꿀 수 있는 얼개를 짜라. 켜 2개, 4개, 8개로 시험하여라.

??? success "연습문제 4 풀이"
    박아 넣은 켜를 다음으로 갈음한다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 방법에서 `for layer in self.layers: x = layer(x)`으로 되돈다. 여느 파이썬 목록이 아니라 `nn.ModuleList`을 써야 PyTorch가 매개변수를 모두 다듬기에 올린다. `for n in [2, 4, 8]: model = RNNCell(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.

## 정리하며

**다룬 것** — 두 방향 RNN / GRU / LSTM

이 짜보기는 갈래 2개(`RNNCell`, `BidirectionalRNN`)를 매기고, 이들이 어울려 온전한 이음 모형 얼개를 이룬다.

고갱이 갈래는 `RNNCell`, `BidirectionalRNN`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
