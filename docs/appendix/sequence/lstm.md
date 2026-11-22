# LSTM

LSTM은 1997년 논문 "Long Short-Term Memory"에서 나왔다. 시퀀스 안의 장거리 의존성(long-range dependency)을 담는 게이트 구조를 쓴다.

여기 구현은 LSTM을 짧고 트레인하기 좋게 보인 예제다. 코드는 핵심 구조와 순전파(forward pass)에 초점을 두어, 핵심 구성 요소를 살펴보고 이리저리 바꾸어 보기 쉽다.

## 1. 코드

```python
#!/usr/bin/env python3
'''
LSTM - 장단기 메모리(Long Short-Term Memory) 네트워크
논문: "Long Short-Term Memory" (1997)
핵심: 시퀀스의 장거리 의존성을 담아내는 게이트 구조
'''
import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

class LSTMModel(nn.Module):
    def __init__(self, input_size=100, hidden_size=256, num_layers=2, num_classes=10):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM 레이어
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        # 완전 연결 레이어
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # 은닉 상태와 셀 상태를 초기화한다
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM 순전파
        out, _ = self.lstm(x, (h0, c0))
        
        # 마지막 시간 스텝의 은닉 상태를 출력 레이어에 넣는다
        out = self.fc(out[:, -1, :])
        return out

if __name__ == "__main__":
    model = LSTMModel()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    x = torch.randn(32, 10, 100)  # batch, sequence, features
    print(f"Input: {x.shape}, Output: {model(x).shape}")
```

**출력:**

```
Parameters: 895,498
Input: torch.Size([32, 10, 100]), Output: torch.Size([32, 10])
```

## 2. 논의

`LSTMModel` 클래스는 PyTorch의 `nn.Module` 인터페이스로 모델 구조를 캡슐화한다. `forward` 메서드가 계산 그래프를 정의하므로 트레인하는 동안 PyTorch의 autograd가 그래디언트 계산을 자동으로 처리한다. 이런 모듈식 설계 덕분에 구성 요소 하나하나를 고치거나 더 큰 파이프라인에 넣기가 쉽다.

이 코드는 구조를 이해하기 위한 교육용 구현이다. 실제 트레인에서는 데이터와 하드웨어에 맞춰 혼합 정밀도, 분산 트레인, 데이터 어그멘테이션 등을 검토할 수 있다. 다만 이런 기법의 적용 조건과 효과는 모델별로 확인해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
기본 설정으로 초기화한 `LSTMModel`의 트레인 가능한 파라메터를 모두 세어라. 웨이트와 바이어스를 포함해 레이어마다 나누어 적어라.

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
`LSTMModel`을 레이어나 블록의 수를 설정할 수 있게 확장하라. `__init__`에 `num_layers` 인자를 더하고 `nn.ModuleList`로 깊이를 바꿀 수 있는 구조를 만들어라. 레이어 2개, 4개, 8개로 시험하여라.

</div>

??? success "풀이"
    하드코딩된 레이어를 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`로 반복한다. 일반 파이썬 리스트가 아니라 `nn.ModuleList`를 써야 PyTorch가 모든 파라메터를 옵티마이저에 등록한다. `for n in [2, 4, 8]: model = LSTMModel(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`로 시험한다.

## 정리하며

**다룬 것** — LSTM

`LSTMModel` 클래스는 PyTorch의 `nn.Module` 인터페이스로 모델 구조를 캡슐화한다.

핵심 클래스는 `LSTMModel`이며 앞의 연습문제 4개로 스스로 확인해 볼 수 있다.
