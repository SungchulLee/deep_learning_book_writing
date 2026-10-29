# 트랜스포머 언어 모델

자기회귀 언어 모델은 각 위치에서 이전 토큰만 보고 다음 토큰을 예측해야 한다. 트랜스포머는 위치 표현과 어텐션으로 이 문맥을 다룬다. 학습 때는 정답 입력이 주어진 여러 위치의 계산을 병렬화할 수 있지만, 자기회귀 생성은 새 토큰을 순서대로 만들어야 한다. 다음 예제는 위치 인코딩·디코더 레이어를 보여 주며 인과성에 중요한 구현 한계가 있다.

## 1. 코드와 기존 출력

```python
"""
길잡이 07: 변환기 말 모델(GPT 방식)
====================================================

변환기는 회귀 대신 셀프 어텐션 구조를 써서
나란히 익히기와 더 나은 먼 거리 나타내기를 가능하게 한다.

구조:
- 토막 묻힘 + 자리 묻힘
- 여러 머리 셀프 어텐션
- 피드포워드 넷트웍
- 레이어 고르게 맞추기
- 잔차 이음

핵심 개념:
- 어텐션: Q, K, V 행렬
- 잣수 맞춘 점곱 어텐션
- 인과(자기되돌리기) 가리기
- 차례 전체를 나란히 다루기

어텐션 식:
Attention(Q, K, V) = softmax(QK^T / √d_k) V
"""

import torch
import torch.nn as nn
import math

# ========================================================================
# 메인
# ========================================================================


class PositionalEncoding(nn.Module):
    """사인파 위치 인코딩."""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        # 위치 인코딩 행렬을 만든다
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerLanguageModel(nn.Module):
    """GPT 방식 변환기 말 모델."""
    
    def __init__(self, vocab_size, d_model=512, nhead=8, 
                 num_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # 변환기 디코더 레이어(인과)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)
        
        self.fc = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)
    
    def generate_square_subsequent_mask(self, sz):
        """앞으로 올 토막에 어텐션을 주지 못하게 하는 인과 마스크 만들기."""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask
    
    def forward(self, src):
        """
        인수:
            src: (batch, seq_len) 들임 토막
        반환값:
            (batch, seq_len, vocab_size) 로짓
        """
        seq_len = src.size(1)
        
        # 임베딩
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        src = self.dropout(src)
        
        # 인과 가림
        mask = self.generate_square_subsequent_mask(seq_len).to(src.device)
        
        # 변환기(셀프 어텐션을 쓰는 디코더로 사용)
        # 유의: nn.TransformerDecoder에 허수아비 기억을 쓴다
        output = self.transformer(src, src, tgt_mask=mask)
        
        # 출력 사영
        logits = self.fc(output)
        return logits


def train_transformer_lm(corpus, vocab, d_model=256, nhead=4, 
                        num_layers=4, epochs=10):
    """변환기 말 모델 익히기."""
    
import numpy as np
from typing import List, Tuple
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

# --- tutorial_05_rnn_language_model ---
"""
길잡이 05: 되돌이 그물 말 모델

수학적 바탕:
h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)
y_t = W_hy * h_t + b_y
P(w_t | w_1,...,w_{t-1}) = softmax(y_t)
"""


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


class RNNDataset(Dataset):
    """말 모델용 자료. 글월을 max_seq_len 길이의 토막으로 자른다.

    말 모델이 배우는 것은 "다음 낱말 맞히기"이므로, 들임과 정답이 한 칸씩
    어긋난 같은 토막이다.

        토막      the  cat  sat  on   the  mat
        들임 x    the  cat  sat  on   the
        정답 y         cat  sat  on   the  mat

    Args:
        corpus: 토큰의 열. 낱말 문자열이어도 되고 이미 번호로 바꾼 것이어도 된다.
        vocab:  낱말 -> 번호 사전. 문자열 corpus를 번호로 바꿀 때 쓴다.
        max_seq_len: 토막 하나의 길이.
    """

    def __init__(self, corpus, vocab, max_seq_len: int = 50):
        self.vocab = vocab
        self.max_seq_len = max_seq_len

        if len(corpus) and isinstance(corpus[0], str):
            unk = vocab.get("<unk>", 0) if hasattr(vocab, "get") else 0
            ids = [vocab.get(t, unk) if hasattr(vocab, "get") else vocab[t] for t in corpus]
        else:
            ids = list(corpus)

        # 정답이 한 칸 뒤이므로 토막마다 max_seq_len + 1개가 필요하다
        step = max_seq_len
        self.chunks = [ids[i:i + max_seq_len + 1]
                       for i in range(0, max(len(ids) - max_seq_len, 0), step)]
        self.chunks = [c for c in self.chunks if len(c) >= 2]

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def collate_fn(batch, pad_value: int = 0):
    """길이가 다른 토막들을 한 배치로 묶는다. 짧은 것은 뒤를 채운다."""
    xs, ys = zip(*batch)
    return (pad_sequence(xs, batch_first=True, padding_value=pad_value),
            pad_sequence(ys, batch_first=True, padding_value=pad_value))
    from torch.utils.data import DataLoader
    import numpy as np
    
    print("Training Transformer Language Model")
    print("=" * 60)
    
    split = int(0.9 * len(corpus))
    train_dataset = RNNDataset(corpus[:split], vocab, max_seq_len=50)
    val_dataset = RNNDataset(corpus[split:], vocab, max_seq_len=50)
    
    train_loader = DataLoader(train_dataset, batch_size=32, 
                             shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=collate_fn)
    
    model = TransformerLanguageModel(
        vocab_size=len(vocab),
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers
    )
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for inputs, targets in train_loader:
            logits = model(inputs)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                logits = model(inputs)
                loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                val_loss += loss.item()
        
        print(f"Epoch {epoch+1}: "
              f"Train PPL={np.exp(train_loss/len(train_loader)):.2f}, "
              f"Val PPL={np.exp(val_loss/len(val_loader)):.2f}")
    
    return model


if __name__ == "__main__":
    print("""
변환기 말 모델
===========================

이점:
1. 나란히 익히기(차례에 얽매이지 않음)
2. 어텐션으로 먼 거리 얽힘이 더 낫다
3. 자료와 셈이 늘어날수록 잘 된다
4. 요즘 큰 말 모델(GPT, BERT 등)의 바탕

구조의 조각:
- 셀프 어텐션: 모든 자리 사이의 관계를 나타낸다
- 여러 머리: 서로 다른 어텐션 무늬
- 자리 부호: 자리 앎을 준다
- 레이어 고르게 맞추기: 익히기를 든든하게 한다
- 잔차 이음: 깊은 넷트웍을 가능하게 한다

견줌:
- 리커런트 넷트웍/LSTM: 차례차례, 익히기가 느리다
- 변환기: 나란히, 더 빠르고, 규모 키우기에 낫다
- 맞바꿈: 기억 공간과 빠르기

요즘 큰 말 모델(GPT-3, GPT-4)은 이 구조에 다음을 곁들여 쓴다:
- 수십억 개의 파라메터
- 어마어마한 익힘 자료
- 앞선 가장 좋게 하기 재주

익힘 문제:
1. 어텐션 무게 그려 보기
2. 머리 수를 달리해 실험하기
3. 자리 부호를 달리해 보기(배운 것과 사인꼴)
4. 상대 자리 부호 짜기
5. 레이어 앞 고르게 맞추기 더하기
6. 같은 자료로 LSTM과 견주기
    """)
```

**기존 출력:**

```

변환기 말 모델
===========================

이점:
1. 나란히 익히기(차례에 얽매이지 않음)
2. 어텐션으로 먼 거리 얽힘이 더 낫다
3. 자료와 셈이 늘어날수록 잘 된다
4. 요즘 큰 말 모델(GPT, BERT 등)의 바탕

구조의 조각:
- 셀프 어텐션: 모든 자리 사이의 관계를 나타낸다
- 여러 머리: 서로 다른 어텐션 무늬
- 자리 부호: 자리 앎을 준다
- 레이어 고르게 맞추기: 익히기를 든든하게 한다
- 잔차 이음: 깊은 넷트웍을 가능하게 한다

견줌:
- 리커런트 넷트웍/LSTM: 차례차례, 익히기가 느리다
- 변환기: 나란히, 더 빠르고, 규모 키우기에 낫다
- 맞바꿈: 기억 공간과 빠르기

요즘 큰 말 모델(GPT-3, GPT-4)은 이 구조에 다음을 곁들여 쓴다:
- 수십억 개의 파라메터
- 어마어마한 익힘 자료
- 앞선 가장 좋게 하기 재주

익힘 문제:
1. 어텐션 무게 그려 보기
2. 머리 수를 달리해 실험하기
3. 자리 부호를 달리해 보기(배운 것과 사인꼴)
4. 상대 자리 부호 짜기
5. 레이어 앞 고르게 맞추기 더하기
6. 같은 자료로 LSTM과 견주기
```

## 2. 위치 표현과 텐서 모양

B×T 단어 ID를 B×T×d_model로 임베딩하고 √d_model 배율과 사인·코사인 위치 인코딩을 적용한다. 위치 표는 register_buffer로 저장되며 학습 가능한 Parameter가 아니다. 모델 장치 이동에는 따라가지만 옵티마이저가 갱신하지 않는다. 현재 구현은 짝수 d_model을 전제로 하고 입력 길이도 max_len=5000 이하여야 한다. head 수로 나누어지는 차원도 필요하다.

출력 선형 레이어는 B×T×V 로짓을 만든다. 다음 단어 목표는 입력에서 한 위치 이동해 구성해야 하며, 미래 토큰을 차단하는 인과 마스크와 PAD를 차단하는 padding mask는 서로 다른 역할이다.

## 3. 디코더의 미래 정보 누출

현재 호출은 transformer(src,src,tgt_mask=mask)다. [TransformerDecoder](https://docs.pytorch.org/docs/2.14/generated/torch.nn.TransformerDecoder.html)의 레이어에는 타깃 셀프 어텐션뿐 아니라 memory에 대한 교차 어텐션도 있다. tgt_mask는 전자만 제한한다. 같은 전체 src를 memory로 주면서 memory_mask를 제공하지 않으므로 교차 어텐션에서 미래 토큰을 볼 수 있다. 따라서 이 구조를 올바른 GPT식 인과 언어 모델이라고 해석할 수 없다.

이는 단순한 러닝레이트 문제가 아니다. 인과 셀프 어텐션만 사용하는 디코더 전용 설계를 쓰거나 모든 정보 경로의 마스크를 일관되게 구성해야 한다. PAD 마스크도 전달하지 않으므로 loss의 ignore_index만으로 padding이 어텐션에 미치는 영향을 없애지는 못한다. 코드와 출력은 요청에 따라 그대로 보존한다.

## 4. 학습 함수와 평가의 한계

train_transformer_lm에는 순전파·교차 엔트로피·역전파·클리핑·갱신과 평가 모드 전환이 있다. 그러나 가져오는 RNNDataset·collate_fn은 앞 RNN 튜토리얼에 정의되어 있지 않다. 제공된 코드만으로 학습 함수가 완성되어 있지 않으며 main도 학습을 호출하지 않고 설명만 출력한다.

ignore_index=0은 실제 PAD ID가 0일 때만 맞다. 출력 PPL은 배치별 평균 로스를 다시 단순 평균해 지수화한 값이다. 유효 토큰 수가 다른 배치에서는 전체 유효 토큰의 음의 로그우도 합을 토큰 수로 나눈 표준 PPL과 다르다. 이 문제와 미래 정보 누출을 해결하지 않은 수치로 일반화 성능을 판단하지 않는다.

## 5. 병렬 계산과 긴 문맥

일반적인 밀집 어텐션의 토큰 간 계산은 O(T²d), 명시적으로 저장한 head별 어텐션 행렬은 O(T²) 공간을 쓴다. Q·K·V 사영의 O(Td²) 등도 전체 레이어 비용에 포함된다. 메모리를 줄이는 구현과 희소·저차원 어텐션은 서로 다른 접근이며 효율·표현력은 실제 설정에서 비교해야 한다. 긴 문맥의 처리 가능성 자체가 장기 의존성의 정확한 학습을 보장하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
PositionalEncoding의 학습 가능한 파라메터 수와 저장된 버퍼 크기를 구하라.

</div>

??? success "풀이"

    학습 가능한 파라메터는 0개다. pe는 1×max_len×d_model의 버퍼이므로 max_len·d_model개의 값을 저장한다. d_model과 max_len을 생성자에 제공해야 하므로 인자 없는 기본 생성은 없다. 드롭아웃이나 Linear·Conv2d 파라메터를 이 위치 인코딩에 합산하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
현재 디코더 레이어의 어텐션 드롭아웃과 인과 마스크가 각각 무엇을 하는지 설명하라.

</div>

??? success "풀이"

    nn.TransformerDecoderLayer에 전달한 dropout은 내부 어텐션과 관련 드롭아웃 구성에 사용된다. 별도 self.attn_dropout을 바깥 클래스에 선언하는 것만으로 내부 웨이트에 연결되지는 않는다. 어텐션 확률에 드롭아웃을 적용하면 일부 값이 0이 되고 나머지는 재조정되므로 매 시행의 합이 반드시 1인 것은 아니다. 드롭아웃은 정규화 선택이며 미래 memory를 차단하는 인과 마스크의 대체가 아니다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
밀집 셀프 어텐션과 희소·저차원 어텐션의 비용을 비교할 때 어떤 항과 조건을 고려해야 하는가?

</div>

??? success "풀이"

    밀집 토큰 간 계산은 O(T²d)이고 사영에는 O(Td²)가 든다. 창 폭 w의 국소 어텐션은 해당 부분이 O(Twd)이며 전역 토큰을 추가하면 그 비용도 더한다. 길이 축을 k차원으로 줄이는 저차원 방식은 해당 상호작용을 O(Tkd)로 줄일 수 있다. 실제 메모리는 head 수·배치·역전파·중간 행렬을 저장하는 구현에 달려 있다. 효율 개선과 원래 연산의 근사·연결 구조 변경을 구분하고 품질을 평가해야 한다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 ModuleList 예가 위치 인코딩의 레이어 수를 늘리는 완성 구현인지 검토하고, 실제 깊이는 어디에서 설정하는지 설명하라.

</div>

??? success "풀이"

    현재 깊이는 TransformerLanguageModel의 num_layers로 TransformerDecoder에 전달한다. PositionalEncoding에는 반복할 블록이 없다. 아래 YourBlock과 dim은 정의되지 않은 자리표시자이므로 완성된 확장 코드가 아니다. 직접 블록을 쌓는 별도 구현에서는 ModuleList로 등록하고 forward에서 순서대로 호출해야 한다. 단순 Python 리스트에 담은 모듈은 자동 등록되지 않지만 ModuleList에 넣는 것만으로 forward 계산이 자동 구성되는 것도 아니다.

    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```


## 정리하며

트랜스포머 언어 모델의 핵심은 모든 경로의 인과성, 위치·padding 처리와 다음 단어 목표의 일치다. 현재 예제에는 교차 어텐션의 미래 누출과 누락된 데이터 도구가 있어 구조 시연과 올바른 학습·평가를 구분해야 한다.
