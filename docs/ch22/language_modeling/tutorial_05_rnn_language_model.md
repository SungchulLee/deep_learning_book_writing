# RNN 언어 모델

RNN 언어 모델은 단어를 순서대로 읽고 은닉 상태에 이전 문맥의 정보를 전달한다. 동일한 순환 파라메터를 각 시점에서 공유하므로 고정 창을 연결하는 모델과 달리 입력 길이에 맞추어 반복할 수 있다. 다만 길이를 처리할 수 있다는 사실과 먼 문맥을 정확히 기억한다는 성능 보장은 다르다.

## 1. 코드와 기존 출력

```python
"""
길잡이 05: 리커런트 넷트웍 말 모델

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
    """맥락 길이가 바뀌는 리커런트 넷트웍 바탕 말 모델."""
    
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

**기존 출력:**

```
RNN Language Model: handles variable-length sequences
Challenge: vanishing/exploding gradients
```

## 2. 은닉 상태와 다음 단어 예측

기본 tanh RNN은 h_t=tanh(W_ih x_t+W_hh h_(t−1)+b_ih+b_hh)로 상태를 갱신한다. PyTorch의 단방향 batch_first 입력은 B×T의 단어 ID, 임베딩은 B×T×d다. 출력 로짓은 B×T×V이고 마지막 은닉 상태는 레이어 수×B×h다. 위치 t의 목표가 다음 단어가 되도록 입력·목표를 한 칸 이동해 구성해야 한다.

명시적인 hidden을 전달하지 않으면 RNN이 초기 상태를 0으로 만든다. init_hidden은 CPU의 기본 부동소수점 텐서를 직접 생성하므로 다른 장치·dtype에서 사용할 때 모델과 맞춰야 한다. 가변 길이 배치를 위한 padding·길이 마스킹·로스 제외는 이 클래스에 포함되어 있지 않다.

## 3. 역전파와 그래디언트 문제

시간 역전파에서는 여러 시점의 상태 Jacobian이 곱해진다. tanh 도함수와 순환 웨이트의 반복 곱이 작아지면 먼 시점의 학습 신호가 약해지고, 커지면 그래디언트가 폭발할 수 있다. 순환 웨이트의 스펙트럴 노름이 1보다 작다는 조건은 tanh Jacobian 곱이 줄어드는 충분조건이다. 반대로 노름이 1보다 크다는 사실만으로 매번 그래디언트가 폭발한다고 결론 내릴 수는 없다.

그래디언트 클리핑은 큰 그래디언트를 제한하지만 사라진 신호를 복원하지는 않는다. LSTM·GRU의 게이트와 다른 초기화·학습 설계는 장기 의존성 학습을 도울 수 있으나 모든 거리의 기억을 보장하지 않는다. RNN의 시간 축 계산은 순차적이며, 실제 속도·메모리와 파라메터 수의 비교는 차원·레이어 수·입력 길이·실행 환경에 따라 달라진다.

## 4. 드롭아웃과 잘린 시간 역전파

코드는 임베딩과 RNN 출력에 별도 드롭아웃을 적용한다. nn.RNN의 dropout은 레이어가 두 개 이상일 때 레이어 사이에 적용되며 단일 레이어에서는 0으로 설정한다. 이것을 순환 상태 연결 자체에 적용하는 recurrent dropout과 혼동하지 않는다.

연습문제의 truncated BPTT는 상태의 값은 유지하고 배치 경계마다 detach하여 계산 그래프를 끊는다. 이를 사용하려면 각 배치 행이 이전 배치의 같은 시퀀스를 이어 받아야 한다. 독립 문장을 섞은 DataLoader에 상태를 그대로 넘기면 서로 다른 문맥이 섞인다. 마지막 배치 크기가 달라지는 경우의 상태 크기도 맞춰야 한다. 현재 main은 설명 문구만 출력하며 실제 학습을 하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
입력 차원 d=128, 은닉 차원 h=256인 단층 단방향 nn.RNN의 순환 레이어 파라메터 수를 계산하라.

</div>

??? success "풀이"

    입력 웨이트 hd=32,768개, 순환 웨이트 h²=65,536개, 두 바이어스 b_ih·b_hh가 각각 h개로 총 512개다. 합계 98,816개이며 임베딩과 출력 레이어는 제외한다. 수식에서 두 바이어스를 하나로 합쳐 표기하더라도 이 PyTorch 구현의 파라메터는 두 벌이다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
tanh RNN의 상태 Jacobian이 먼 시점의 그래디언트에 미치는 영향과 노름 조건의 한계를 설명하라.

</div>

??? success "풀이"

    J_k=diag(1−tanh²(a_k))W_hh이며 상태에 대한 그래디언트는 이 Jacobian들의 연쇄 곱을 포함한다. 각 도함수의 절댓값이 1 이하이므로 ||W_hh||₂<1이면 곱의 노름은 ||W_hh||₂^(T−t)로 상계되어 감소한다. 이 조건이 아니어도 tanh 포화나 방향별 수축으로 그래디언트가 작아질 수 있다. 클리핑은 폭발을 제한하는 방법이며 사라짐의 직접적인 해결책은 아니다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
아래 함수가 실제로 어디에서 시간 역전파를 자르는지, k와 배치 상태 전달에 어떤 조건이 필요한지 설명하라.

</div>

??? success "풀이"

    hidden.detach()는 이전 배치의 상태 값은 이어 쓰되 그 계산 그래프로는 역전파하지 않는다. 함수의 k 인자는 사용되지 않으므로 코드가 스스로 k 시점 단위로 입력을 나누는 것은 아니다. DataLoader가 연속된 길이 k의 청크를 같은 배치 행에 제공해야 의도한 학습이 된다. 독립 문장은 상태를 초기화하고, 장치·배치 크기를 맞추며 train 모드·padding 로스 제외를 별도로 설정해야 한다. 빈 DataLoader에서는 평균 로스의 0 나눗셈도 처리해야 한다.

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


## 정리하며

RNN은 공유 순환 계산으로 이전 문맥을 전달한다. 긴 입력의 처리와 장기 기억 성능을 구분하고, 다음 단어 목표·상태 초기화·연속 배치·detach 범위를 일관되게 구성해야 한다.
