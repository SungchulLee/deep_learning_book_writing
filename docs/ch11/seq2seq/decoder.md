# 디코더

이전 토큰과 순환 상태로 다음 토큰의 로짓을 계산하는 디코더를 비교한다. BasicDecoder는 상태만, AttentionDecoder는 인코더 위치별 가중합도 사용한다. 가산·내적 어텐션은 점수 함수의 대안이다.

## 1. 코드

```python
"""
Seq2Seq 모델을 위한 디코더 모듈
어텐션이 있는 것과 없는 것 등 여러 디코더 구조를 구현한다
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# 메인
# ========================================================================


class BasicDecoder(nn.Module):
    """
    Seq2Seq 모델을 위한 기본 RNN 디코더
    
    인수:
        output_size: 출력 어휘의 크기
        embedding_dim: 낱말 임베딩의 차원
        hidden_size: 숨은 상태의 크기
        num_layers: 순환 레이어의 수
        dropout: 드롭아웃 확률
        rnn_type: RNN의 종류 ('LSTM' 또는 'GRU')
    """
    
    def __init__(self, output_size, embedding_dim, hidden_size, 
                 num_layers=1, dropout=0.1, rnn_type='LSTM'):
        super(BasicDecoder, self).__init__()
        
        self.output_size = output_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        
        # 임베딩 레이어
        self.embedding = nn.Embedding(output_size, embedding_dim)
        
        # 드롭아웃
        self.dropout = nn.Dropout(dropout)
        
        # RNN 레이어
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                embedding_dim,
                hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(
                embedding_dim,
                hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        else:
            raise ValueError(f"Unknown RNN type: {rnn_type}")
        
        # 출력 레이어
        self.fc_out = nn.Linear(hidden_size, output_size)
        
    def forward(self, input_token, hidden, cell=None):
        """
        한 시각의 순전파
        
        인수:
            input_token: 입력 토큰 (배치 크기, 1)
            hidden: 앞 시각의 숨은 상태
            cell: 앞 시각의 세포 상태 (LSTM에만 있다)
            
        반환값:
            output: 출력 예측 (배치 크기, output_size)
            hidden: 갱신된 숨은 상태
            cell: 갱신된 세포 상태 (LSTM에만 있다)
        """
        # 입력 토큰 임베딩
        embedded = self.embedding(input_token)  # (배치 크기, 1, embedding_dim)
        embedded = self.dropout(embedded)
        
        # RNN 통과
        if self.rnn_type == 'LSTM':
            rnn_output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        else:  # GRU
            rnn_output, hidden = self.rnn(embedded, hidden)
            cell = None
        
        # 출력 예측 만들기
        output = self.fc_out(rnn_output.squeeze(1))  # (배치 크기, output_size)
        
        return output, hidden, cell


class AttentionDecoder(nn.Module):
    """
    바다나우(덧셈) 어텐션 장치가 있는 디코더
    
    인수:
        output_size: 출력 어휘의 크기
        embedding_dim: 낱말 임베딩의 차원
        hidden_size: 디코더 숨은 상태의 크기
        encoder_hidden_size: 인코더 숨은 상태의 크기
        num_layers: 순환 레이어의 수
        dropout: 드롭아웃 확률
        rnn_type: RNN의 종류 ('LSTM' 또는 'GRU')
    """
    
    def __init__(self, output_size, embedding_dim, hidden_size, 
                 encoder_hidden_size, num_layers=1, dropout=0.1, rnn_type='LSTM'):
        super(AttentionDecoder, self).__init__()
        
        self.output_size = output_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.encoder_hidden_size = encoder_hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        
        # 임베딩 레이어
        self.embedding = nn.Embedding(output_size, embedding_dim)
        
        # 드롭아웃
        self.dropout = nn.Dropout(dropout)
        
        # 어텐션 장치
        self.attention = BahdanauAttention(hidden_size, encoder_hidden_size)
        
        # RNN 레이어 (입력은 임베딩과 문맥 벡터)
        rnn_input_size = embedding_dim + encoder_hidden_size
        
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                rnn_input_size,
                hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(
                rnn_input_size,
                hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        else:
            raise ValueError(f"Unknown RNN type: {rnn_type}")
        
        # 출력 레이어
        self.fc_out = nn.Linear(hidden_size + encoder_hidden_size + embedding_dim, output_size)
        
    def forward(self, input_token, hidden, encoder_outputs, cell=None, mask=None):
        """
        어텐션을 쓰는 한 시각의 순전파
        
        인수:
            input_token: 입력 토큰 (배치 크기, 1)
            hidden: 앞 시각의 숨은 상태
            encoder_outputs: 인코더의 모든 출력 (배치 크기, src_len, encoder_hidden_size)
            cell: 앞 시각의 세포 상태 (LSTM에만 있다)
            mask: 덧댐을 가리는 가림막 (배치 크기, src_len)
            
        반환값:
            output: 출력 예측 (배치 크기, output_size)
            hidden: 갱신된 숨은 상태
            cell: 갱신된 세포 상태 (LSTM에만 있다)
            attention_weights: 어텐션 웨이트 (배치 크기, src_len)
        """
        # 입력 토큰 임베딩
        embedded = self.embedding(input_token)  # (배치 크기, 1, embedding_dim)
        embedded = self.dropout(embedded)
        
        # 어텐션 계산
        # 어텐션에 맨 위 레이어의 숨은 상태 쓰기
        query = hidden[-1].unsqueeze(1) if hidden.dim() == 3 else hidden.unsqueeze(1)
        context, attention_weights = self.attention(query, encoder_outputs, mask)
        
        # 임베딩한 입력과 문맥 벡터 이어 붙이기
        rnn_input = torch.cat([embedded, context], dim=2)
        
        # RNN 통과
        if self.rnn_type == 'LSTM':
            rnn_output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        else:  # GRU
            rnn_output, hidden = self.rnn(rnn_input, hidden)
            cell = None
        
        # 예측을 위해 RNN 출력과 문맥과 임베딩한 입력 이어 붙이기
        output_input = torch.cat([
            rnn_output.squeeze(1),
            context.squeeze(1),
            embedded.squeeze(1)
        ], dim=1)
        
        # 출력 예측 만들기
        output = self.fc_out(output_input)  # (배치 크기, output_size)
        
        return output, hidden, cell, attention_weights.squeeze(1)


class BahdanauAttention(nn.Module):
    """
    바다나우(덧셈) 어텐션 장치
    
    인수:
        decoder_hidden_size: 디코더 숨은 상태의 크기
        encoder_hidden_size: 인코더 숨은 상태의 크기
    """
    
    def __init__(self, decoder_hidden_size, encoder_hidden_size):
        super(BahdanauAttention, self).__init__()
        
        self.W_decoder = nn.Linear(decoder_hidden_size, decoder_hidden_size)
        self.W_encoder = nn.Linear(encoder_hidden_size, decoder_hidden_size)
        self.V = nn.Linear(decoder_hidden_size, 1)
        
    def forward(self, query, keys, mask=None):
        """
        어텐션 웨이트와 문맥 벡터를 계산한다
        
        인수:
            query: 디코더의 숨은 상태 (배치 크기, 1, decoder_hidden_size)
            keys: 인코더의 출력 (배치 크기, src_len, encoder_hidden_size)
            mask: 덧댐 가림막 (배치 크기, src_len)
            
        반환값:
            context: 문맥 벡터 (배치 크기, 1, encoder_hidden_size)
            attention_weights: 어텐션 웨이트 (배치 크기, 1, src_len)
        """
        # 어텐션 점수 계산
        # query: (배치, 1, dec_hidden)
        # keys: (배치, src_len, enc_hidden)
        
        query_transformed = self.W_decoder(query)  # (배치, 1, dec_hidden)
        keys_transformed = self.W_encoder(keys)    # (배치, src_len, dec_hidden)
        
        # 방송하여 더하기
        scores = self.V(torch.tanh(query_transformed + keys_transformed))  # (배치, src_len, 1)
        scores = scores.squeeze(2)  # (배치, src_len)
        
        # 가림막이 있으면 씌우기
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # 어텐션 웨이트 계산
        attention_weights = F.softmax(scores, dim=1).unsqueeze(1)  # (배치, 1, src_len)
        
        # 문맥 벡터 계산
        context = torch.bmm(attention_weights, keys)  # (배치, 1, encoder_hidden_size)
        
        return context, attention_weights


class LuongAttention(nn.Module):
    """
    루옹(곱셈) 어텐션 장치
    
    인수:
        decoder_hidden_size: 디코더 숨은 상태의 크기
        encoder_hidden_size: 인코더 숨은 상태의 크기
        attention_type: 점수 함수의 종류 ('dot', 'general', 'concat')
    """
    
    def __init__(self, decoder_hidden_size, encoder_hidden_size, attention_type='general'):
        super(LuongAttention, self).__init__()
        
        self.attention_type = attention_type
        
        if attention_type == 'general':
            self.W = nn.Linear(encoder_hidden_size, decoder_hidden_size, bias=False)
        elif attention_type == 'concat':
            self.W = nn.Linear(decoder_hidden_size + encoder_hidden_size, decoder_hidden_size)
            self.V = nn.Linear(decoder_hidden_size, 1, bias=False)
        
    def forward(self, query, keys, mask=None):
        """
        어텐션 웨이트와 문맥 벡터를 계산한다
        
        인수:
            query: 디코더의 숨은 상태 (배치 크기, 1, decoder_hidden_size)
            keys: 인코더의 출력 (배치 크기, src_len, encoder_hidden_size)
            mask: 덧댐 가림막 (배치 크기, src_len)
            
        반환값:
            context: 문맥 벡터 (배치 크기, 1, encoder_hidden_size)
            attention_weights: 어텐션 웨이트 (배치 크기, 1, src_len)
        """
        if self.attention_type == 'dot':
            # 단순 내적
            scores = torch.bmm(query, keys.transpose(1, 2))  # (배치, 1, src_len)
        elif self.attention_type == 'general':
            # 일반형: query * W * keys^T
            keys_transformed = self.W(keys)  # (배치, src_len, dec_hidden)
            scores = torch.bmm(query, keys_transformed.transpose(1, 2))  # (배치, 1, src_len)
        elif self.attention_type == 'concat':
            # 이어 붙이기: V * tanh(W * [query; keys])
            src_len = keys.size(1)
            query_expanded = query.expand(-1, src_len, -1)  # (배치, src_len, dec_hidden)
            concat = torch.cat([query_expanded, keys], dim=2)  # (배치, src_len, dec+enc_hidden)
            scores = self.V(torch.tanh(self.W(concat)))  # (배치, src_len, 1)
            scores = scores.transpose(1, 2)  # (배치, 1, src_len)
        
        scores = scores.squeeze(1)  # (배치, src_len)
        
        # 가림막이 있으면 씌우기
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # 어텐션 웨이트 계산
        attention_weights = F.softmax(scores, dim=1).unsqueeze(1)  # (배치, 1, src_len)
        
        # 문맥 벡터 계산
        context = torch.bmm(attention_weights, keys)  # (배치, 1, encoder_hidden_size)
        
        return context, attention_weights


if __name__ == "__main__":
    # 사용 예
    batch_size = 32
    src_len = 20
    vocab_size = 10000
    embedding_dim = 256
    hidden_size = 512
    encoder_hidden_size = 512
    
    # 어텐션 디코더 만들기
    decoder = AttentionDecoder(
        output_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        encoder_hidden_size=encoder_hidden_size,
        num_layers=2,
        dropout=0.1,
        rnn_type='LSTM'
    )
    
    # 예제 입력
    input_token = torch.randint(0, vocab_size, (batch_size, 1))
    hidden = torch.randn(2, batch_size, hidden_size)
    cell = torch.randn(2, batch_size, hidden_size)
    encoder_outputs = torch.randn(batch_size, src_len, encoder_hidden_size)
    
    # 순전파
    output, hidden, cell, attention_weights = decoder(
        input_token, hidden, encoder_outputs, cell
    )
    
    print(f"Input token shape: {input_token.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Hidden shape: {hidden.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
```

**출력:**

```
Input token shape: torch.Size([32, 1])
Output shape: torch.Size([32, 10000])
Hidden shape: torch.Size([2, 32, 512])
Attention weights shape: torch.Size([32, 20])
```

## 2. 논의

한 단계의 토큰 입력은 $(B,1)$의 long ID다. 기본 디코더는 임베딩과 순환 레이어를 거쳐 $(B,V)$의 로짓과 새 상태를 반환한다. LSTM에는 은닉·셀 상태가 모두 필요하고 GRU의 셀 반환값은 None이다. 여러 토큰을 넣으면 squeeze(1)이 시간 축을 제거하지 못하므로 문서화된 한 단계 호출과 구분해야 한다.

AttentionDecoder는 최상위 은닉 상태를 쿼리로 삼고 BahdanauAttention을 호출한다. 문맥은 $(B,1,H_e)$이고 이를 토큰 임베딩과 함께 순환 레이어에 넣는다. 출력 레이어는 새 순환 출력·문맥·임베딩을 이어 붙여 로짓을 만든다. LuongAttention 클래스가 존재해도 기본 AttentionDecoder에 자동으로 사용되는 것은 아니다.

마스크는 유효 위치가 1 또는 True라는 규약이다. 모든 입력 위치가 가려지면 -inf 점수의 소프트맥스에서 NaN이 생길 수 있으므로 빈 입력 처리가 필요하다. Luong dot 점수는 쿼리와 키 피처 차원이 같아야 하고 general은 투영으로 차원을 맞춘다. 무작위 입력의 출력 형태는 번역 품질을 검증하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
어휘 V, 임베딩 E, 은닉 H, 레이어수 L인 기본 LSTM 디코더의 파라메터를 계산하는 식을 구하라.

</div>

??? success "연습문제 1 풀이"
    임베딩 VE, 첫 순환 레이어 $4H(E+H+2)$, 추가 각 레이어 $4H(2H+2)$, 출력 레이어 VH+V를 합한다. 총 $VE+4H(E+H+2)+(L-1)4H(2H+2)+VH+V$다. 기본 두 바이어스를 포함한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
어텐션 웨이트에 드롭아웃을 적용할 때 마스크·정규화와 평가 모드를 설명하라.

</div>

??? success "연습문제 2 풀이"
    먼저 유효 위치만 소프트맥스로 정규화하고 그 뒤 드롭아웃을 적용할 수 있다. 학습 중 합이 매번 1인 것은 아니며 유지된 값은 재스케일된다. 평가에서는 드롭아웃을 끈다. 패딩 마스크를 대신하거나 균등한 주의를 보장하는 기법은 아니다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
크로스 어텐션의 입력 길이 S와 출력 길이 T를 셀프 어텐션의 길이와 구분하라.

</div>

??? success "연습문제 3 풀이"
    디코더의 각 위치는 S개의 인코더 위치를 참조하므로 웨이트 수는 전체 T단계에 대해 TS다. 입력에 대한 셀프 어텐션의 S²와 같지 않다. 시간 비용에는 점수 함수의 차원과 투영, 순환 레이어의 계산도 포함한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 풀이의 `ModuleList` 코드가 원래 모델에 바로 적용될 수 있는지 검토하라. 입력·출력 차원, 활성화, 생성자와 `forward`에서 필요한 수정을 설명하라.

</div>

??? success "연습문제 4 풀이"
    아래 코드는 깊이를 설정하는 아이디어를 위한 조각이다. 원래 모델의 생성자와 텐서 차원을 반영한 완성 구현은 아니다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `ModuleList`는 하위 모듈을 등록하지만 실행 순서나 활성화를 자동으로 정하지 않는다. 입력 레이어와 출력 레이어는 과제의 차원을 유지하고, 은닉 레이어 사이에 필요한 비선형성을 넣어야 한다. 선형 레이어만 반복하면 아핀 함수 하나로 합쳐진다. 원래 클래스에 없는 `num_layers` 인수를 그대로 넘길 수 없으므로 생성자와 `forward`를 함께 수정해야 한다.

## 정리하며

한 단계 디코더의 토큰·상태 규약을 맞추고 어텐션 마스크가 유효한지 확인한다. 점수 함수를 바꾸는 것과 디코더 전체를 바꾸는 것은 다른 작업이다.
