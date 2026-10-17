# 인코더

입력 토큰열을 위치별 표현과 최종 상태로 바꾸는 인코더 예제다. BasicEncoder는 LSTM·GRU 중 하나를 사용하고 ConvEncoder는 1차원 컨볼루션을 사용한다. 두 클래스는 대안이며 함께 연결되어 하나의 모델을 이루지는 않는다.

## 1. 코드

```python
"""
Seq2Seq 모델을 위한 인코더 모듈
LSTM과 GRU를 비롯한 여러 인코더 구조를 구현한다
"""

import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================


class BasicEncoder(nn.Module):
    """
    Seq2Seq 모델을 위한 기본 RNN 인코더
    
    인수:
        input_size: 입력 어휘의 크기
        embedding_dim: 낱말 임베딩의 차원
        hidden_size: 숨은 상태의 크기
        num_layers: 순환 레이어의 수
        dropout: 드롭아웃 확률
        bidirectional: 양방향 RNN을 쓸지 여부
        rnn_type: RNN의 종류 ('LSTM' 또는 'GRU')
    """
    
    def __init__(self, input_size, embedding_dim, hidden_size, 
                 num_layers=1, dropout=0.1, bidirectional=False, rnn_type='LSTM'):
        super(BasicEncoder, self).__init__()
        
        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type
        
        # 임베딩 레이어
        self.embedding = nn.Embedding(input_size, embedding_dim)
        
        # 드롭아웃 레이어
        self.dropout = nn.Dropout(dropout)
        
        # RNN 레이어
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                embedding_dim,
                hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True
            )
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(
                embedding_dim,
                hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True
            )
        else:
            raise ValueError(f"Unknown RNN type: {rnn_type}")
        
    def forward(self, input_seq, input_lengths=None):
        """
        인코더를 지나는 순전파
        
        인수:
            input_seq: 입력 순차열 텐서 (배치 크기, seq_len)
            input_lengths: 순차열의 실제 길이 (선택)
            
        반환값:
            outputs: 모든 숨은 상태 (배치 크기, seq_len, hidden_size * num_directions)
            hidden: 마지막 숨은 상태
            cell: 마지막 세포 상태 (LSTM에만 있다)
        """
        # 입력 임베딩
        embedded = self.embedding(input_seq)
        embedded = self.dropout(embedded)
        
        # 길이가 주어지면 덧댄 순차열을 꾸리기
        if input_lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, input_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        
        # RNN 통과
        if self.rnn_type == 'LSTM':
            outputs, (hidden, cell) = self.rnn(embedded)
        else:  # GRU
            outputs, hidden = self.rnn(embedded)
            cell = None
        
        # 꾸렸으면 풀기
        if input_lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        
        # 양방향이면 순방향과 역방향 숨은 상태를 합치기
        if self.bidirectional:
            # hidden: (num_layers * 2, 배치 크기, hidden_size)
            # (num_layers, 배치 크기, hidden_size * 2)로 모양 바꾸기
            hidden = self._combine_bidirectional(hidden)
            if cell is not None:
                cell = self._combine_bidirectional(cell)
        
        return outputs, hidden, cell
    
    def _combine_bidirectional(self, hidden):
        """순방향과 역방향의 숨은 상태를 합친다"""
        # hidden: (num_layers * 2, 배치 크기, hidden_size)
        # 출력: (num_layers, 배치 크기, hidden_size * 2)
        num_directions = 2
        batch_size = hidden.size(1)
        
        hidden = hidden.view(self.num_layers, num_directions, batch_size, self.hidden_size)
        hidden = torch.cat([hidden[:, 0], hidden[:, 1]], dim=2)
        
        return hidden


class ConvEncoder(nn.Module):
    """
    Seq2Seq 모델을 위한 컨볼루션 인코더
    순차열 부호화에 1차원 컨볼루션을 쓴다
    """
    
    def __init__(self, input_size, embedding_dim, hidden_size, 
                 num_layers=3, kernel_size=3, dropout=0.1):
        super(ConvEncoder, self).__init__()
        
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 컨볼루션 레이어
        conv_layers = []
        in_channels = embedding_dim
        
        for _ in range(num_layers):
            conv_layers.extend([
                nn.Conv1d(in_channels, hidden_size, kernel_size, padding=kernel_size//2),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_channels = hidden_size
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
    def forward(self, input_seq):
        """
        컨볼루션 인코더를 지나는 순전파
        
        인수:
            input_seq: 입력 순차열 (배치 크기, seq_len)
            
        반환값:
            outputs: 부호화된 표현 (배치 크기, seq_len, hidden_size)
        """
        # 임베딩하고 conv1d를 위해 전치
        embedded = self.embedding(input_seq)  # (배치, seq_len, embed_dim)
        embedded = self.dropout(embedded)
        embedded = embedded.transpose(1, 2)  # (배치, embed_dim, seq_len)
        
        # 컨볼루션 적용
        outputs = self.conv_layers(embedded)  # (배치, hidden_size, seq_len)
        outputs = outputs.transpose(1, 2)  # (배치, seq_len, hidden_size)
        
        return outputs, None, None


if __name__ == "__main__":
    # 사용 예
    batch_size = 32
    seq_len = 20
    vocab_size = 10000
    embedding_dim = 256
    hidden_size = 512
    
    # 인코더 만들기
    encoder = BasicEncoder(
        input_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=2,
        dropout=0.1,
        bidirectional=True,
        rnn_type='LSTM'
    )
    
    # 예제 입력
    input_seq = torch.randint(0, vocab_size, (batch_size, seq_len))
    input_lengths = torch.randint(10, seq_len+1, (batch_size,))
    
    # 순전파
    outputs, hidden, cell = encoder(input_seq, input_lengths)
    
    print(f"Input shape: {input_seq.shape}")
    print(f"Output shape: {outputs.shape}")
    print(f"Hidden shape: {hidden.shape}")
    if cell is not None:
        print(f"Cell shape: {cell.shape}")
```

**출력:**

```
Input shape: torch.Size([32, 20])
Output shape: torch.Size([32, 20, 1024])
Hidden shape: torch.Size([2, 32, 1024])
Cell shape: torch.Size([2, 32, 1024])
```

## 2. 논의

BasicEncoder는 $(B,S)$의 정수 ID를 임베딩한다. 실제 길이를 주면 pack_padded_sequence로 패딩을 제외해 순환 계산하고 다시 텐서로 푼다. 길이는 양수이고 입력 길이 이하여야 한다. 풀 때 total_length를 지정하지 않아 출력 길이는 배치의 실제 최대 길이가 되며 원래 패딩 폭과 다를 수 있다.

양방향 출력은 피처 축에 두 방향이 이미 이어져 있다. 최종 은닉·셀 상태도 레이어별로 두 방향을 이어 붙여 $(L,B,2H)$로 바꾼다. 기본 내장 모듈의 $(2L,B,H)$와 다른 반환 규약이므로 디코더에 전달할 때 맞춰야 한다. GRU에서는 셀 상태가 None이다.

ConvEncoder는 임베딩 축을 채널로 바꾸어 Conv1d를 적용한다. 기본 홀수 커널에서는 길이가 유지되지만 짝수 커널과 padding=k//2에서는 레이어마다 길이가 하나 늘 수 있다. 최종 상태를 반환하지 않아 순환 디코더의 초기 상태를 별도로 구성해야 한다. 길이 마스크도 받지 않으므로 PAD가 유효 위치의 컨볼루션 결과에 영향을 줄 수 있다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
B=4, 길이 S, 임베딩 E, 은닉 H인 BasicEncoder의 주요 형태를 추적하라.

</div>

??? success "풀이"
    ID는 $(4,S)$, 임베딩은 $(4,S,E)$다. 단방향 출력은 $(4,S_{out},H)$, 양방향 출력은 $(4,S_{out},2H)$이며 길이 입력이 있으면 S′는 실제 배치 최대 길이다. 반환 최종 상태는 각각 $(L,4,H)$와 $(L,4,2H)$이고 LSTM 셀 상태도 같다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
packed 출력과 원래 패딩 폭이 다르면 후속 어텐션에서 무엇을 맞춰야 하는가?

</div>

??? success "풀이"
    어텐션 마스크의 길이와 인코더 출력의 위치 수를 맞춰야 한다. 풀 때 원래 전체 길이를 지정하거나 실제 출력 폭에 맞춰 마스크를 구성한다. 양방향 상태의 피처 차원과 디코더 차원도 확인한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
ConvEncoder를 BasicEncoder 대신 사용할 때 필요한 연결 설계를 설명하라.

</div>

??? success "풀이"
    위치별 출력의 피처 차원을 디코더·어텐션에 맞춘다. 반환 은닉·셀 상태가 None이므로 풀링·투영이나 별도 초기 상태를 설계한다. 패딩 마스크와 컨볼루션의 가장자리·길이 변화를 함께 처리한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 풀이의 `ModuleList` 코드가 원래 모델에 바로 적용될 수 있는지 검토하라. 입력·출력 차원, 활성화, 생성자와 `forward`에서 필요한 수정을 설명하라.

</div>

??? success "풀이"
    아래 코드는 깊이를 설정하는 아이디어를 위한 조각이다. 원래 모델의 생성자와 텐서 차원을 반영한 완성 구현은 아니다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `ModuleList`는 하위 모듈을 등록하지만 실행 순서나 활성화를 자동으로 정하지 않는다. 입력 레이어와 출력 레이어는 과제의 차원을 유지하고, 은닉 레이어 사이에 필요한 비선형성을 넣어야 한다. 선형 레이어만 반복하면 아핀 함수 하나로 합쳐진다. 원래 클래스에 없는 `num_layers` 인수를 그대로 넘길 수 없으므로 생성자와 `forward`를 함께 수정해야 한다.

## 정리하며

인코더의 반환 규약은 순환 종류와 방향, 길이 처리에 따라 다르다. 출력 길이와 마스크, 최종 상태의 형태를 맞춰야 디코더에 연결할 수 있다.
