# GRU

GRU은 2014년 글 "Learning Phrase Representations using RNN Encoder-Decoder"에서 나왔다. LSTM보다 문 구조가 단순하고 파라메터가 적다.

여기 구현은 GRU을 짧고 학습하기 좋게 보인 본이다. 코드는 핵심 구조와 앞으로 스텝에 마음을 두어, 핵심 꾸밈새를 살펴보고 이리저리 바꾸어 보기 쉽다.

## 1. 코드

```python
#!/usr/bin/env python3
'''
GRU - 문 달린 되도는 알갱이
논문: "Learning Phrase Representations using RNN Encoder-Decoder" (2014)
핵심: LSTM보다 단출한 문 구조와 더 적은 파라메터
'''
import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

class GRUModel(nn.Module):
    def __init__(self, input_size=100, hidden_size=256, num_layers=2, num_classes=10):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU 켜
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        # 온통 이은 켜
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # 숨은 상태의 첫자리를 잡는다
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # GRU 앞으로 스텝
        out, _ = self.gru(x, h0)
        
        # 마지막 때 스텝의 숨은 상태를 푼다
        out = self.fc(out[:, -1, :])
        return out

if __name__ == "__main__":
    model = GRUModel()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    x = torch.randn(32, 10, 100)
    print(f"Input: {x.shape}, Output: {model(x).shape}")
```

**출력:**

```
Parameters: 672,266
Input: torch.Size([32, 10, 100]), Output: torch.Size([32, 10])
```

## 2. 논의

`GRUModel` 클래스는 PyTorch의 `nn.Module` 낯을 써서 모델 구조를 담는다. `forward` 방법이 계산 그림을 매기므로 익히는 동안 PyTorch의 autograd가 그래디언트 계산을 절로 다룬다. 이렇게 묶음으로 나눈 꾸밈 덕에 몫 하나하나를 고치거나 더 큰 흐름에 넣기가 쉽다.

이 코드는 구조를 이해하기 위한 교육용 구현이다. 실제 학습에서는 데이터와 하드웨어에 맞춰 혼합 정밀도, 분산 학습, 데이터 어그멘테이션 등을 검토할 수 있다. 다만 이런 기법의 적용 조건과 효과는 모델별로 확인해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
기본 첫자리로 잡은 `GRUModel`의 배울 수 있는 파라메터를 모두 세어라. 짐과 바이어스를 아울러 켜마다 나누어 적어라.

</div>

??? success "풀이"
    `nn.Linear(in_features, out_features)`마다 짐 파라메터가 `in_features * out_features`개이고 바이어스가 `out_features`개다(`bias=False`가 아니면). `nn.Conv2d(in_c, out_c, k)`마다 짐이 `in_c * out_c * k * k`개, 바이어스가 `out_c`개다. `nn.Embedding(num, dim)`은 파라메터가 `num * dim`개다. 켜를 모두 더한다. `sum(p.numel() for p in model.parameters())`으로 따져 볼 수 있다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
입력의 꼴과 데이터 클래스가 바라는 대로인지 살피는 입력 살피기를 으뜸 함수나 클래스에 더하여라. 올바르지 않은 입력에는 알아듣기 쉬운 어긋남 알림을 띄워라.

</div>

??? success "풀이"
    `forward` 방법(또는 알맞은 함수)의 소개에 `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`이나 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'` 같은 살핌을 더한다. 꼴을 살피려면 중요한 차수를 본다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 알아듣기 쉬운 어긋남 알림은 디버깅을 크게 앞당기고 코드를 되쓰기 든든하게 한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
숨은 크기 $h$과 입력 크기 $x$이 같을 때 LSTM 칸과 GRU 칸의 파라메터 수를 비교하여라. 어느 쪽이 더 적고 왜 그런가?

</div>

??? success "풀이"
    LSTM에는 문이 넷(입력, 잊음, 칸, 출력) 있고 저마다 입력과 은닉 상태의 짐 행렬을 지니므로 파라메터는 $4 \times (x \cdot h + h \cdot h + h) = 4(xh + h^2 + h)$개다. GRU에는 문이 셋(되돌림, 고침, 새것) 있어 $3 \times (x \cdot h + h \cdot h + h) = 3(xh + h^2 + h)$개다. GRU은 문이 넷이 아니라 셋이고 칸 상태와 은닉 상태를 하나로 묶으므로 LSTM의 75%이다. 참으로는 파라메터가 적어도 GRU이 LSTM과 엇비슷하게 잘 듣는 일이 잦다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
`GRUModel`을 켜나 덩이의 수를 골라 잡을 수 있게 넓혀라. `__init__`에 `num_layers` 파라메터를 더하고 `nn.ModuleList`으로 깊이를 바꿀 수 있는 구조를 짜라. 켜 2개, 4개, 8개로 시험하여라.

</div>

??? success "풀이"
    박아 넣은 켜를 다음으로 갈음한다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 방법에서 `for layer in self.layers: x = layer(x)`으로 되돈다. 여느 파이썬 목록이 아니라 `nn.ModuleList`을 써야 PyTorch가 파라메터를 모두 다듬기에 올린다. `for n in [2, 4, 8]: model = GRUModel(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.

## 정리하며

**다룬 것** — GRU

`GRUModel` 클래스는 PyTorch의 `nn.Module` 낯을 써서 모델 구조를 담는다.

핵심 클래스는 `GRUModel`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
