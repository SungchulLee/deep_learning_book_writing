# Bidirectional RNN / GRU / LSTM

Bidirectional RNN / GRU / LSTM - Processing sequences in both directions Key idea: run one RNN forward (t=1..T) and another backward (t=T..1),

This implementation provides a concise, educational reference for Bidirectional RNN / GRU / LSTM. The code focuses on the core architecture and forward pass, making it straightforward to study the key design patterns and adapt them for experimentation.

## 코드

```python
#!/usr/bin/env python3
"""
Bidirectional RNN / GRU / LSTM - Processing sequences in both directions
Key idea: run one RNN forward (t=1..T) and another backward (t=T..1),
then combine their outputs (concat or sum).

This file provides:
  - Bidirectional wrapper around a *cell-based* RNN (vanilla), GRU, or LSTM-like module
  - For simplicity and clarity, we implement a bidirectional vanilla RNN here.

File: appendix/sequence/bidirectional.py
Note: Educational, fully commented implementation (batch-first).
"""

import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================


class RNNCell(nn.Module):
    """
    Reuse a simple vanilla RNN cell.

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
    Bidirectional vanilla RNN.

    We maintain:
      - forward hidden state h_f (left -> right)
      - backward hidden state h_b (right -> left)

    For each time step t:
      forward:  h_f[t] = f(x[t], h_f[t-1])
      backward: h_b[t] = f(x[t], h_b[t+1])   (computed by iterating reversed time)

    Output combination (common choices):
      - concat: y[t] = [h_f[t], h_b[t]]  -> dimension 2*hidden
      - sum:    y[t] = h_f[t] + h_b[t]  -> dimension hidden

    Here we implement concatenation (most common).
    """
    def __init__(self, input_size: int, hidden_size: int, concat: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.concat = concat

        # Two independent RNN cells: one forward, one backward
        self.cell_f = RNNCell(input_size, hidden_size)
        self.cell_b = RNNCell(input_size, hidden_size)

    def forward(self, x: torch.Tensor, h0_f: torch.Tensor | None = None, h0_b: torch.Tensor | None = None):
        """
        x: (B, T, input_size)

        Returns:
          y: (B, T, 2*hidden) if concat=True else (B, T, hidden)
          (hT_f, hT_b): final forward/backward hidden states (B, hidden)
        """
        B, T, _ = x.shape
        device = x.device

        # Initialize hidden states if not provided
        h_f = torch.zeros(B, self.hidden_size, device=device) if h0_f is None else h0_f
        h_b = torch.zeros(B, self.hidden_size, device=device) if h0_b is None else h0_b

        # ---- Forward pass (t = 0..T-1) ----
        forward_states = []
        for t in range(T):
            x_t = x[:, t, :]               # (B, input)
            h_f = self.cell_f(x_t, h_f)    # update forward hidden
            forward_states.append(h_f)     # store h_f[t]

        # ---- Backward pass (t = T-1..0) ----
        backward_states_reversed = []
        for t in reversed(range(T)):
            x_t = x[:, t, :]               # (B, input)
            h_b = self.cell_b(x_t, h_b)    # update backward hidden (moving right->left)
            backward_states_reversed.append(h_b)  # this is h_b[t], but collected reversed

        # Reverse backward list so it aligns with time order 0..T-1
        backward_states = list(reversed(backward_states_reversed))

        # ---- Combine forward and backward states per time step ----
        y_steps = []
        for t in range(T):
            hf_t = forward_states[t]       # (B, hidden)
            hb_t = backward_states[t]      # (B, hidden)

            if self.concat:
                # Concatenate along feature dimension -> (B, 2*hidden)
                y_t = torch.cat([hf_t, hb_t], dim=1)
            else:
                # Sum -> (B, hidden)
                y_t = hf_t + hb_t

            y_steps.append(y_t)

        # Stack to (B, T, feat_dim)
        y = torch.stack(y_steps, dim=1)

        # Final hidden states (after last updates in each direction)
        hT_f = forward_states[-1]          # (B, hidden)
        hT_b = backward_states[0]          # (B, hidden)  (backward final corresponds to t=0 in aligned order)

        return y, (hT_f, hT_b)


if __name__ == "__main__":
    # Quick sanity check
    model = BidirectionalRNN(input_size=8, hidden_size=16, concat=True)
    x = torch.randn(2, 5, 8)

    y, (hF, hB) = model(x)
    print("y :", y.shape)   # expected (2, 5, 32) when concat=True
    print("hF:", hF.shape)  # expected (2, 16)
    print("hB:", hB.shape)  # expected (2, 16)```

## 논의

이 짜보기는 갈래 2개(`RNNCell`, `BidirectionalRNN`)를 매기고, 이들이 어울려 온전한 이음 모형 얼개를 이룬다. 갈래마다 남다른 몫을 담아 코드를 묶음으로 나누고 넓히기 쉽게 한다. `forward` 방법이 PyTorch이 절로 미분하는 데 쓰는 셈 그림을 매긴다.

여기 실린 코드는 본보기 짜보기라 다듬기보다 알아보기 쉬움을 앞세운다. 서비스 얼개라면 흔히 섞인 촘촘함 익히기, 흩은 자료 나란히, 더 정교한 자료 불리기를 더한다. 그래도 여기서 보인 얼개의 고갱이 깨침은 크기와 상관없이 그대로다.

## 익힘 문제

**익힘 1.**
기본 첫자리로 잡은 `RNNCell`의 배울 수 있는 매개변수를 모두 세어라. 짐과 치우침을 아울러 켜마다 나누어 적어라.

??? success "익힘 1 풀이"
    `nn.Linear(in_features, out_features)`마다 짐 매개변수가 `in_features * out_features`개이고 치우침이 `out_features`개다(`bias=False`가 아니면). `nn.Conv2d(in_c, out_c, k)`마다 짐이 `in_c * out_c * k * k`개, 치우침이 `out_c`개다. `nn.Embedding(num, dim)`은 매개변수가 `num * dim`개다. 켜를 모두 더한다. `sum(p.numel() for p in model.parameters())`으로 따져 볼 수 있다.

---

**익힘 2.**
들임의 꼴과 자료 갈래가 바라는 대로인지 살피는 들임 살피기를 으뜸 함수나 갈래에 더하여라. 올바르지 않은 들임에는 알아듣기 쉬운 어긋남 알림을 띄워라.

??? success "익힘 2 풀이"
    `forward` 방법(또는 알맞은 함수)의 첫머리에 `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`이나 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'` 같은 살핌을 더한다. 꼴을 살피려면 종요로운 차수를 본다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 알아듣기 쉬운 어긋남 알림은 벌레잡기를 크게 앞당기고 코드를 되쓰기 든든하게 한다.

---

**익힘 3.**
숨은 크기 $h$과 들임 크기 $x$이 같을 때 LSTM 칸과 GRU 칸의 매개변수 수를 견주어라. 어느 쪽이 더 적고 왜 그런가?

??? success "익힘 3 풀이"
    LSTM에는 문이 넷(들임, 잊음, 칸, 날임) 있고 저마다 들임과 숨은 상태의 짐 행렬을 지니므로 매개변수는 $4 \times (x \cdot h + h \cdot h + h) = 4(xh + h^2 + h)$개다. GRU에는 문이 셋(되돌림, 고침, 새것) 있어 $3 \times (x \cdot h + h \cdot h + h) = 3(xh + h^2 + h)$개다. GRU은 문이 넷이 아니라 셋이고 칸 상태와 숨은 상태를 하나로 묶으므로 LSTM의 75%이다. 참으로는 매개변수가 적어도 GRU이 LSTM과 엇비슷하게 잘 듣는 일이 잦다.

---

**익힘 4.**
`RNNCell`을 켜나 덩이의 수를 골라 잡을 수 있게 넓혀라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`으로 깊이를 바꿀 수 있는 얼개를 짜라. 켜 2개, 4개, 8개로 시험하여라.

??? success "익힘 4 풀이"
    박아 넣은 켜를 다음으로 갈음한다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 방법에서 `for layer in self.layers: x = layer(x)`으로 되돈다. 여느 파이썬 목록이 아니라 `nn.ModuleList`을 써야 PyTorch이 매개변수를 모두 다듬기에 올린다. `for n in [2, 4, 8]: model = RNNCell(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.
