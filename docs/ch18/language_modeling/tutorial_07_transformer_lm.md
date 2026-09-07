# 실습 07

길잡이 07: 변환기 말 모델(GPT 방식). 변환기는 되돌이 대신 스스로 눈길 얼개를 쓴다.

자연어 다루기는 깊은 배움 방식으로 크게 달라졌다. 이 단원은 말 모델 재주를 보여 주며, 신경망이 글을 어떻게 다루고 만들어 내는지 밝히는 실전 짜기를 준다.

## 코드

```python
"""
길잡이 07: 변환기 말 모델(GPT 방식)
====================================================

변환기는 되돌이 대신 스스로 눈길 얼개를 써서
나란히 익히기와 더 나은 먼 거리 나타내기를 가능하게 한다.

구조:
- 토막 묻힘 + 자리 묻힘
- 여러 머리 스스로 눈길
- 앞먹임 그물
- 층 고르게 맞추기
- 잔차 이음

핵심 개념:
- 눈길: Q, K, V 행렬
- 잣수 맞춘 점곱 눈길
- 인과(자기되돌리기) 가리기
- 차례 전체를 나란히 다루기

눈길 식:
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
        
        # 변환기 풀개 층(인과)
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
        """앞으로 올 토막에 눈길을 주지 못하게 하는 인과 마스크 만들기."""
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
        
        # 변환기(스스로 눈길을 쓰는 풀개로 사용)
        # 유의: nn.TransformerDecoder에 허수아비 기억을 쓴다
        output = self.transformer(src, src, tgt_mask=mask)
        
        # 출력 사영
        logits = self.fc(output)
        return logits


def train_transformer_lm(corpus, vocab, d_model=256, nhead=4, 
                        num_layers=4, epochs=10):
    """변환기 말 모델 익히기."""
    
    from tutorial_05_rnn_language_model import RNNDataset, collate_fn
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
2. 눈길으로 먼 거리 얽힘이 더 낫다
3. 자료와 셈이 늘어날수록 잘 된다
4. 요즘 큰 말 모델(GPT, BERT 등)의 바탕

얼개의 조각:
- 스스로 눈길: 모든 자리 사이의 관계를 나타낸다
- 여러 머리: 서로 다른 눈길 무늬
- 자리 부호: 자리 앎을 준다
- 층 고르게 맞추기: 익히기를 든든하게 한다
- 잔차 이음: 깊은 그물을 가능하게 한다

견줌:
- 되돌이 그물/LSTM: 차례차례, 익히기가 느리다
- 변환기: 나란히, 더 빠르고, 규모 키우기에 낫다
- 맞바꿈: 기억 공간과 빠르기

요즘 큰 말 모델(GPT-3, GPT-4)은 이 얼개에 다음을 곁들여 쓴다:
- 수십억 개의 매개변수
- 어마어마한 익힘 자료
- 앞선 가장 좋게 하기 재주

익힘 문제:
1. 눈길 무게 그려 보기
2. 머리 수를 달리해 실험하기
3. 자리 부호를 달리해 보기(배운 것과 사인꼴)
4. 상대 자리 부호 짜기
5. 층 앞 고르게 맞추기 더하기
6. 같은 자료로 LSTM과 견주기
    """)```

## 논의

여기 짠 것은 함께 어울려 온전한 말 모델 얼개를 이루는 클래스 2개(`PositionalEncoding`, `TransformerLanguageModel`)를 정한다. 클래스마다 뚜렷한 조각 하나를 감싸므로 코드가 단원별로 나뉘고 넓히기 쉽다. `forward` 메서드가 PyTorch의 자동 미분에 쓰이는 셈 그래프를 정한다.

학습 루프는 표준적인 PyTorch 패턴을 따른다. 예측을 계산하는 순전파, 손실 계산, 경사를 구하는 역전파, 그리고 최적화기를 통한 매개변수 갱신이다. 에폭에 걸쳐 지표를 추적하면 수렴 양상이 드러나고 과소적합이나 과적합 같은 문제를 진단하는 데 도움이 된다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 넓혀진다. 초매개변수와 구조의 변형, 다른 데이터셋으로 실험해 보면 이해가 깊어지고 자연어 처리 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
기본 초기화 상태의 `PositionalEncoding`에 든 학습 가능한 매개변수의 총 개수를 셈하라. 가중치와 편향을 모두 넣어 층별로 나누어 보여라.

??? success "연습문제 1 풀이"
    `nn.Linear(in_features, out_features)` 각각에는 `in_features * out_features`개의 가중치 매개변수와 (`bias=False`가 아닌 한) `out_features`개의 편향 매개변수가 있다. `nn.Conv2d(in_c, out_c, k)`에는 `in_c * out_c * k * k`개의 가중치와 `out_c`개의 편향이 있다. `nn.Embedding(num, dim)`에는 `num * dim`개의 매개변수가 있다. 모든 층에 대해 더하면 된다. `sum(p.numel() for p in model.parameters())`로 확인할 수 있다.

---

**연습문제 2.**
어텐션 가중치 뒤에(값과 곱하기 전에) 드롭아웃 층을 추가하라. 학습 중에는 드롭아웃 비율 0.1을 쓴다. 어텐션 드롭아웃이 정칙화에 도움이 되는 이유를 설명하라.

??? success "연습문제 2 풀이"
    `__init__`에 `self.attn_dropout = nn.Dropout(0.1)`을 추가하고 소프트맥스 뒤에 적용한다. `attn_weights = self.attn_dropout(F.softmax(scores, dim=-1))`이다. 어텐션 드롭아웃은 학습 중에 일부 어텐션 가중치를 무작위로 0으로 만들어, 모델이 특정 토큰 사이의 관계에 지나치게 기대지 않게 한다. 이는 모델이 어텐션을 더 고르게 분산시키고 더 견고한 표현을 배우도록 북돋우며, 표준 드롭아웃이 뉴런의 공적응을 막는 것과 비슷하다.

---

**연습문제 3.**
자기 어텐션의 계산 복잡도를 열의 길이 $n$과 모델 차원 $d$의 함수로 설명하라. 이것이 왜 긴 열에 대해 Longformer나 Linformer 같은 구조의 동기가 되는가?

??? success "연습문제 3 풀이"
    표준 자기 어텐션은 $n \times n$ 어텐션 행렬을 계산하므로 시간 복잡도가 $O(n^2 d)$이고 어텐션 가중치에 $O(n^2)$의 메모리가 든다. 열이 길면(예: $n = 4096$) 감당하기 어려워진다. Longformer는 국소적인 미끄럼창 어텐션($w$이 창 크기일 때 $O(n \cdot w \cdot d)$)과 선택된 토큰에 대한 희소한 전역 어텐션을 결합한다. Linformer는 키와 값을 더 낮은 차원 $k \ll n$으로 사영하여 복잡도를 $O(n \cdot k \cdot d)$으로 줄인다. 둘 다 표현력을 조금 내주고 긴 입력에서의 실용적인 효율을 얻는다.

---

**연습문제 4.**
`PositionalEncoding`이 층이나 블록의 개수를 설정할 수 있도록 넓혀라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`를 써서 깊이를 바꿀 수 있는 구조를 만들어라. 층 2개, 4개, 8개로 시험하라.

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`처럼 되풀이하라. (그냥 파이썬 리스트가 아니라) `nn.ModuleList`를 써야 파이토치가 최적화 대상 매개변수를 모두 등록한다. `for n in [2, 4, 8]: model = PositionalEncoding(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`로 시험하라.
