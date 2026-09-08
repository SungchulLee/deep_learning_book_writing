# Vanilla RNN

맨 RNN - 되도는 신경 그물(엘만 RNN). 옛 깨침: 때를 따라 차례로 고쳐지는 숨은 상태를 지닌다.

여기 짜보기는 Vanilla RNN을 짧고 배우기 좋게 보인 본이다. 코드는 고갱이 얼개와 앞으로 걸음에 마음을 두어, 고갱이 꾸밈새를 살펴보고 이리저리 바꾸어 보기 쉽다.

## 1. 코드

```python
#!/usr/bin/env python3
"""
맨 RNN - 되도는 신경 그물(엘만 RNN)
옛 깨침: 때를 따라 차례로 고쳐지는 숨은 상태를 지닌다.

본: "때 속에서 얼개 찾기" (1990), 제프리 엘만(단순한 RNN을 널리 알렸다)
고갱이: h_t = tanh(W_x x_t + W_h h_{t-1} + b)

두루마리: appendix/sequence/rnn.py
눈여겨볼 것: 주석을 빠짐없이 단, 배우기 위한 짜보기다(한 켜, 묶음을 앞에 둔다).
"""

import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================


class RNNCell(nn.Module):
    """
    맨 RNN 칸 하나(때 걸음 하나).

    꼴:
      x_t     : (B, input_size)
      h_prev  : (B, hidden_size)
      h_t     : (B, hidden_size)

    고침:
      h_t = tanh( W_x * x_t + W_h * h_{t-1} + b )
    """
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 들임과 앞선 숨은 상태를 바꾸는 선형 켜
        self.Wx = nn.Linear(input_size, hidden_size, bias=True)
        self.Wh = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        # 바꾼 들임과 숨은 상태를 아우른 뒤 tanh을 건다
        h_t = torch.tanh(self.Wx(x_t) + self.Wh(h_prev))
        return h_t


class RNN(nn.Module):
    """
    맨 RNN(때를 따라 손으로 풀어 놓음).

    들임:
      x : (B, T, input_size)  묶음을 앞에 둔 이음
    날임:
      y : (B, T, hidden_size) 때에 걸친 모든 숨은 상태
      h_T : (B, hidden_size) 마지막 숨은 상태
    """
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.cell = RNNCell(input_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor, h0: torch.Tensor | None = None):
        # 묶음 크기와 때 길이를 뽑는다
        B, T, _ = x.shape
        device = x.device

        # 숨은 상태가 주어지지 않으면 0으로 첫자리를 잡는다
        if h0 is None:
            h_t = torch.zeros(B, self.hidden_size, device=device)
        else:
            h_t = h0

        outputs = []
        for t in range(T):
            # 때 걸음 t의 들임을 가져온다
            x_t = x[:, t, :]               # (B, input_size)

            # RNN 칸으로 숨은 상태를 고친다
            h_t = self.cell(x_t, h_t)      # (B, hidden_size)

            # 이 때 걸음의 숨은 상태를 담아 둔다
            outputs.append(h_t)

        # 때에 걸친 숨은 상태를 쌓아 텐서 하나로 만든다
        y = torch.stack(outputs, dim=1)     # (B, T, hidden_size)
        return y, h_t


if __name__ == "__main__":
    # 얼른 해 보는 맛보기 살핌: 앞으로 걸음을 돌리고 꼴을 찍는다
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

이 짜보기는 갈래 2개(`RNNCell`, `RNN`)를 매기고, 이들이 어울려 온전한 이음 모형 얼개를 이룬다. 갈래마다 남다른 몫을 담아 코드를 묶음으로 나누고 넓히기 쉽게 한다. `forward` 방법이 PyTorch가 절로 미분하는 데 쓰는 셈 그림을 매긴다.

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
이 짜보기가 무너질 만한 결 둘을 밝히고, 저마다 어떻게 짚어내고 고칠지 밝혀라.

??? success "연습문제 3 풀이"
    흔히 무너지는 결은 이렇다. (1) **기울기가 사라지거나 터짐** -- 기울기 크기를 지켜보아 짚어낸다(`torch.nn.utils.clip_grad_norm_`이나 켜마다 `param.grad.norm()` 적기). 기울기 자르기, 더 나은 첫자리 잡기(Xavier/Kaiming), 얼개 고치기(나머지 이음, 잣대 잡기)로 고친다. (2) **지나치게 맞추기** -- 익힘 잃음은 줄어드는데 따짐 잃음이 오르면 짚어낸다. 다독임(드롭아웃, 짐 줄이기, 자료 불리기)이나 모형 크기 줄이기로 고친다. 익힘과 따짐 자를 늘 함께 지켜보아 이를 일찍 잡아야 한다.

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

**다룬 것** — Vanilla RNN

이 짜보기는 갈래 2개(`RNNCell`, `RNN`)를 매기고, 이들이 어울려 온전한 이음 모형 얼개를 이룬다.

고갱이 갈래는 `RNNCell`, `RNN`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
