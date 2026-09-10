# LSTM 말 모델

긴 짧은 기억(LSTM) 그물은 그물 안의 앎 흐름을 다스리는 문 얼개를 들여와 맨 되돌이 그물의 기울기 사라짐 문제를 다룬다. 칸 상태가 때 걸음을 가로지르는 곧은 기울기 통로를 주어, LSTM은 맨 되돌이 그물이 담아내지 못하는 먼 거리 얽힘을 배운다. LSTM은 변환기가 나오기 전까지 차례 나타내기의 판을 잡은 얼개였다.

## 1. 코드

```python
"""
길잡이 06: LSTM 말 모델

LSTM 식:
f_t = sigma(W_f . [h_{t-1}, x_t] + b_f)  # 망각 문
i_t = sigma(W_i . [h_{t-1}, x_t] + b_i)  # 입력 문
C_t = f_t * C_{t-1} + i_t * tanh(W_C . [h_{t-1}, x_t] + b_C)
o_t = sigma(W_o . [h_{t-1}, x_t] + b_o)  # 출력 문
h_t = o_t * tanh(C_t)  # 새 숨은 상태
"""

import torch
import torch.nn as nn
import numpy as np

# ========================================================================
# 메인
# ========================================================================


class LSTMLanguageModel(nn.Module):
    """LSTM 바탕 말 모델."""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, 
                 num_layers=2, dropout=0.2):
        super(LSTMLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden=None):
        embeds = self.dropout(self.embedding(x))
        output, hidden = self.lstm(embeds, hidden)
        output = self.dropout(output)
        logits = self.fc(output)
        return logits, hidden


if __name__ == "__main__":
    model = LSTMLanguageModel(vocab_size=10000, embedding_dim=256, 
                               hidden_dim=512, num_layers=2)
    params = sum(p.numel() for p in model.parameters())
    print(f"LSTM LM parameters: {params:,}")
```

**출력:**

```
LSTM LM parameters: 11,368,208
```

## 2. 논의

LSTM은 문 셋(잊음, 들임, 날임)을 두어 칸 상태를 지나는 소식의 흐름을 다스린다. 잊음 문은 앞선 칸 상태에서 무엇을 버릴지 정한다. 들임 문과 후보 값은 어떤 새 소식을 담을지 정한다. 칸 상태 고침 $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$은 원소마다 곱하기를 쓰므로, 되도는 짐 행렬을 거듭 곱하지 않고도 기울기가 칸 상태를 따라 곧바로 흐른다.

칸 상태를 지나는 이 곧은 기울기 통로가 핵심 얼개 새로움이다. 잊음 문이 1에 가깝고 들임 문이 0에 가까우면 칸 상태가 거의 그대로 지켜지고 기울기가 거의 줄지 않고 뒤로 흐른다. 그물은 여러 때 걸음에 걸쳐 남아야 할 앎에 대해 잊음 문을 열어 두는 법을 배워 기울기 사라짐을 에도는 "기울기 고속도로"를 만든다. 잊음 문의 치우침을 양수(보기로 1.0)로 첫자리매김하면 모델이 기본으로 기억하도록 북돋운다.

LSTM은 말 나타내기 잣대에서 맨 되돌이 그물보다 헷갈림도가 크게 낮은 것이 보통이다. 숨은 낱 512개의 2층 LSTM은 Penn Treebank에서 헷갈림도 약 80~120을 얻는데, 견줄 만한 되돌이 그물은 120~180이다. 무게 묶기(묻힘 행렬과 내놓는 내리쬐기 행렬을 나눠 쓰기)는 매개변수를 줄이고 흔히 성능을 낫게 하는 널리 쓰이는 재주이다. 계단식 줄이기를 쓴 배움 비율 일정 짜기와 기울기 자르기도 여전히 중요한 익히기 버릇이다.

## 연습문제

**연습문제 1.**
들임 차원 $d = 256$, 숨은 차원 $h = 512$인 LSTM 층 하나의 매개변수 개수를 셈하여라. 차원이 같은 맨 되돌이 그물 층과 견주어라.

??? success "연습문제 1 풀이"
    LSTM에는 문이 4개 있고, 문마다 들임-숨은 층 무게와 숨은 층-숨은 층 무게에 치우침이 있다:
    
    - 모든 문의 들임에서 숨은 켜로: $4 \times d \times h = 4 \times 256 \times 512 = 524{,}288$
    - 모든 문의 숨은 켜에서 숨은 켜로: $4 \times h \times h = 4 \times 512 \times 512 = 1{,}048{,}576$
    - 모든 문의 치우침: $4 \times h = 2{,}048$
    
    LSTM 모두: 매개변수 $1{,}574{,}912$개.
    
    A vanilla RNN: $(d \times h) + (h \times h) + 2h = 131{,}072 + 262{,}144 + 1{,}024 = 394{,}240$.
    
    견줌은 대략 $4\times$이며 문이 넷인 얼개와 맞는다.

---

**연습문제 2.**
잊음 문 $f_t$이 늘 0이고 들임 문 $i_t$이 늘 1일 때 어떤 일이 일어나는지 설명하여라. 이는 맨 되돌이 그물과 어떻게 다른가?

??? success "연습문제 2 풀이"
    $f_t = 0$이고 $i_t = 1$이면 $C_t = 0 \cdot C_{t-1} + 1 \cdot \tilde{C}_t = \tilde{C}_t$이다. 칸 상태가 걸음마다 통째로 덮어써져 앞선 기억이 모두 사라진다. $\partial C_t / \partial C_{t-1} = 0$이므로 칸 상태가 주던 기울기 이점이 없어지고, 모델은 숨은 상태를 다스리는 날임 문이 하나 더 붙은 맨 되도는 신경망처럼 움직인다. 그러면 기울기가 사라지는 탈이 되살아난다.

---

**연습문제 3.**
묻힘 행렬과 내놓는 내리쬐기 행렬이 무게를 나눠 쓰도록 LSTM 말 모델에 무게 묶기를 짜라. 왜 `embedding_dim == hidden_dim`이어야 하는지 밝혀라.

??? success "연습문제 3 풀이"
    ```python
    class TiedLSTMLanguageModel(nn.Module):
        def __init__(self, vocab_size, dim, num_layers=2, dropout=0.2):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, dim)
            self.lstm = nn.LSTM(dim, dim, num_layers=num_layers,
                               dropout=dropout if num_layers > 1 else 0,
                               batch_first=True)
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(dim, vocab_size)
            self.fc.weight = self.embedding.weight  # 무게 묶기
        
        def forward(self, x, hidden=None):
            embeds = self.dropout(self.embedding(x))
            output, hidden = self.lstm(embeds, hidden)
            output = self.dropout(output)
            return self.fc(output), hidden
    ```
    
    짐 묶기는 `embedding_dim == hidden_dim`이어야 한다. 담기 행렬의 꼴이 $(V, d_e)$이고 날임 되비춤의 꼴이 $(V, d_h)$이기 때문이다. 나누어 쓰려면 $d_e = d_h$이어야 한다. 이러면 매개변수가 $V \times d$만큼 줄고 다독임 노릇도 한다.

## 정리하며

**다룬 것** — LSTM 말 모델

LSTM은 문 셋(잊음, 들임, 날임)을 두어 칸 상태를 지나는 소식의 흐름을 다스린다.

고갱이 갈래는 `LSTMLanguageModel`, `TiedLSTMLanguageModel`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
