# 되돌이 그물 말 모델

되돌이 신경망은 때 걸음을 가로질러 앎을 나르는 숨은 상태를 지녀 앞먹임 말 모델의 붙박이 맥락 한계를 넘어선다. 덕분에 매개변수 한 벌로 아무 길이의 차례도 다룰 수 있어, 되돌이 그물은 말의 먼 거리 얽힘을 나타낼 수 있는 첫 신경 얼개가 되었다.

## 코드

```python
"""
길잡이 05: 되돌이 그물 말 모델

수학적 바탕:
h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)
y_t = W_hy * h_t + b_y
P(w_t | w_1,...,w_{t-1}) = softmax(y_t)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple

# ========================================================================
# 메인
# ========================================================================


class RNNLanguageModel(nn.Module):
    """맥락 길이가 바뀌는 되돌이 그물 바탕 말 모델."""
    
    def __init__(self, vocab_size: int, embedding_dim: int, 
                 hidden_dim: int, num_layers: int = 1, dropout: float = 0.2):
        super(RNNLanguageModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(
            embedding_dim, hidden_dim, num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, hidden=None):
        embeds = self.dropout(self.embedding(x))
        output, hidden = self.rnn(embeds, hidden)
        output = self.dropout(output)
        logits = self.fc(output)
        return logits, hidden
    
    def init_hidden(self, batch_size: int):
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim)


if __name__ == "__main__":
    print("RNN Language Model: handles variable-length sequences")
    print("Challenge: vanishing/exploding gradients")
```

## 논의

The RNN language model processes sequences one token at a time, updating a hidden state vector at each step according to $h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$. This hidden state serves as a compressed representation of the entire sequence history, theoretically allowing the model to condition its predictions on arbitrarily long contexts. Unlike the feedforward model, which sees only a fixed window, the RNN can in principle learn that a word at position 100 depends on a word at position 1.

실전에서 맨 되돌이 그물은 기울기 사라짐 때문에 먼 거리 얽힘에 약하다. 때를 거슬러 뒤로 퍼뜨리는 동안 걸음마다 기울기에 되돌이 무게 행렬이 곱해진다. 이 행렬의 스펙트럼 노름이 1보다 작으면 기울기가 차례 길이에 따라 지수로 줄어들어 먼 거리 무늬를 배울 수 없게 된다. 기울기 자르기는 기울기 터짐을 다루지만, 기울기 사라짐은 LSTM이나 GRU 칸 같은 얼개의 바뀜이 있어야 한다.

되돌이 그물 얼개는 모든 때 자리에서 매개변수를 나눠 쓴다. 곧 걸음마다 같은 무게 행렬을 쓴다. 이러면 맥락 창이 큰 앞먹임 모델에 견주어 매개변수 수가 크게 줄고, 익히는 동안 본 것보다 긴 차례에도 두루 통하게 된다. 그러나 되돌이 셈이 차례차례라는 성질 탓에 때 걸음을 나란히 할 수 없어, 모든 자리를 한꺼번에 다룰 수 있는 변환기 같은 얼개보다 익히기가 느리다.

## 연습문제

**연습문제 1.**
숨은 차원 $h = 256$, 묻힘 차원 $d = 128$인 되돌이 그물에서 되돌이 층의 매개변수 개수를 셈하여라(묻힘과 내놓는 내리쬐기는 빼고 무게와 치우침만).

??? success "연습문제 1 풀이"
    되돌이 그물의 되돌이 층에는 다음이 있다:
    
    - Input-to-hidden weights $W_{xh}$: $d \times h = 128 \times 256 = 32{,}768$
    - Hidden-to-hidden weights $W_{hh}$: $h \times h = 256 \times 256 = 65{,}536$
    - 치우침: $h + h = 512$(무게 행렬마다 하나씩)
    
    모두: $32{,}768 + 65{,}536 + 512 = 98{,}816$개의 매개변수.

---

**연습문제 2.**
사슬 규칙으로 되돌이 그물의 기울기 사라짐 문제를 밝혀라. 때 $T$의 손실을 때 $t$의 숨은 상태로 미분한 기울기는 왜 $T - t$이 커질수록 작아지는가?

??? success "연습문제 2 풀이"
    연쇄 법칙에 의해 다음과 같다.
    
    $$
    \frac{\partial L_T}{\partial h_t} = \frac{\partial L_T}{\partial h_T} \prod_{k=t+1}^{T} \frac{\partial h_k}{\partial h_{k-1}}
    $$
    
    Since $h_k = \tanh(W_{hh} h_{k-1} + W_{xh} x_k + b)$, we have $\partial h_k / \partial h_{k-1} = \text{diag}(1 - \tanh^2(\cdot)) \cdot W_{hh}$. The product of $T - t$ such matrices involves repeated multiplication by $W_{hh}$ scaled by the tanh derivative (which is at most 1). If the spectral norm of $W_{hh}$ is less than 1, this product shrinks exponentially, making gradients vanish for large $T - t$.

---

**연습문제 3.**
`k` 걸음마다 숨은 상태를 떼어 내어 기울기가 `k` 걸음보다 더 뒤로 흐르지 못하게 함으로써 잘라 낸 때 거슬러 퍼뜨리기를 짜라.

??? success "연습문제 3 풀이"
    ```python
    def train_with_truncated_bptt(model, data_loader, optimizer, 
                                   criterion, k=35, epochs=10):
        for epoch in range(epochs):
            hidden = None
            total_loss = 0
            num_batches = 0
            for inputs, targets in data_loader:
                if hidden is not None:
                    hidden = hidden.detach()
                logits, hidden = model(inputs, hidden)
                loss = criterion(logits.view(-1, logits.size(-1)),
                                 targets.view(-1))
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1
            print(f"Epoch {epoch+1}: loss = {total_loss/num_batches:.4f}")
    ```
    
    핵심 줄은 `hidden = hidden.detach()`이다. 이는 값은 같지만 기울기 발자취가 없는 새 텐서를 만들어 때 거슬러 퍼뜨리기를 묶음 하나 길이로 제한한다.
