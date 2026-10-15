# LSTM 언어 모델

LSTM은 셀 상태와 게이트로 정보를 유지·갱신하는 순환 신경망이다. 단순 tanh RNN보다 장기 의존성의 학습을 돕도록 설계되었지만 기울기 사라짐이나 기억 손실을 항상 없애 주지는 않는다. 이 예제는 단어 임베딩·다층 LSTM·출력 선형 층으로 다음 단어 로짓을 만드는 구조다.

## 1. 코드와 기존 출력

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

**기존 출력:**

```
LSTM LM parameters: 11,368,208
```

## 2. 게이트와 셀 상태

입력 게이트 i_t, 망각 게이트 f_t, 출력 게이트 o_t와 후보 셀 g_t를 계산한다. 셀은 c_t=f_t⊙c_(t−1)+i_t⊙g_t, 은닉 상태는 h_t=o_t⊙tanh(c_t)다. 후보 셀은 네 번째 게이트라기보다 새로 저장할 내용이다. 이 네 묶음의 선형 변환 때문에 같은 차원의 단순 RNN보다 순환 층의 매개변수가 네 배다.

게이트를 고정해서 본 셀의 직접 경로 미분은 f_t다. f_t가 1에 가까우면 기억과 기울기를 유지하는 데 도움이 되고 여러 시점에서 작으면 직접 경로도 줄어든다. 실제 전체 미분에는 게이트·후보가 이전 은닉 상태에 의존하는 간접 경로도 있다. 따라서 셀의 덧셈 구조가 모든 기울기의 보존을 보장한다고 설명하지 않는다. 양의 망각 편향은 가능한 초기화 선택이지만 현재 코드는 별도로 적용하지 않는다.

## 3. 텐서와 학습 계약

단어 ID 입력은 B×T, 로짓은 B×T×V다. hidden은 (h,c) 튜플이며 각 텐서는 층 수×B×h다. 이전 청크의 상태를 계속 쓰려면 시퀀스와 배치 행이 이어져야 하고 truncated BPTT에서는 두 텐서를 각각 detach해야 한다. 문장별 독립 처리·padding 길이·손실 마스킹은 호출부가 구성해야 한다.

드롭아웃은 임베딩과 최종 출력에 적용되고 다층 LSTM 내부에서는 층 사이에도 적용된다. 현재 main은 모델 생성과 매개변수 수 출력만 한다. 자료·손실·학습 루프·스케줄러·기울기 클리핑과 시험 퍼플렉서티는 구현되어 있지 않다. 특정 Penn Treebank 수치나 RNN 대비 향상을 이 코드의 결과로 제시할 수 없다.

## 4. 가중치 공유

출력층 가중치와 입력 임베딩이 같은 V×d 행렬을 공유하면 별도의 출력 가중치 Vd개를 줄인다. 직접 공유하려면 LSTM 출력 차원과 임베딩 차원이 같아야 한다. 차원이 다르면 중간 사영 등 다른 설계가 필요하다. 출력 편향은 별도 매개변수로 남는다.

공유는 매개변수를 줄이고 학습을 제약하는 선택이며 성능 향상을 보장하지 않는다. 연습문제의 TiedLSTMLanguageModel은 동일 차원을 사용하고 실제 Parameter를 공유하지만, 기본 LSTMLanguageModel에는 공유가 적용되어 있지 않다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
d=256, h=512인 단층 단방향 PyTorch LSTM의 매개변수 수를 구하고 같은 차원의 nn.RNN과 비교하라.

</div>

??? success "연습문제 1 풀이"

    입력 가중치 4hd=524,288개, 순환 가중치 4h²=1,048,576개, 두 편향 묶음 8h=4,096개로 총 1,576,960개다. 같은 차원의 RNN은 hd+h²+2h=394,240개이므로 정확히 네 배다. 임베딩·출력층과 추가 LSTM 층은 제외한다. 편향 한 벌만 세는 수식과 PyTorch의 두 편향 구현을 혼용하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
f_t=0, i_t=1이면 셀 상태와 직접 기억 경로에 어떤 일이 생기며 일반 RNN과 완전히 같아지는지 설명하라.

</div>

??? success "연습문제 2 풀이"

    c_t=g_t가 되어 이전 셀 상태의 직접적인 보존 항이 사라진다. 게이트를 고정한 직접 미분은 0이지만 g_t가 h_(t−1)에 의존하는 간접 경로는 남을 수 있다. 출력 게이트와 tanh의 조합도 남으므로 일반 tanh RNN과 동일한 식은 아니다. 셀의 직접 기억 경로가 없어진다는 결론과 모든 과거 정보·전체 미분이 0이라는 주장은 구분해야 한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
아래 가중치 공유 구현에서 필요한 차원 조건과 줄어드는 매개변수, 남는 매개변수를 설명하라.

</div>

??? success "연습문제 3 풀이"

    embedding.weight와 fc.weight는 모두 V×dim이며 같은 Parameter를 참조한다. LSTM 출력 차원도 dim으로 맞추므로 직접 공유가 가능하다. 비공유 모델의 출력 가중치 Vdim개가 줄지만 fc.bias의 V개와 LSTM 매개변수는 남는다. 입력 임베딩 차원과 출력 차원이 다르면 별도 사영 등 구조 변경 없이 직접 공유할 수 없다. 이 예제는 학습 성능을 비교하지 않는다.

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


## 정리하며

LSTM의 셀 상태는 게이트를 통한 직접 기억 경로를 제공한다. 상태 튜플·다층 편향 개수·padding·학습 범위를 정확히 구분하고, 가중치 공유는 차원 조건과 평가를 갖춘 별도 선택으로 다룬다.
