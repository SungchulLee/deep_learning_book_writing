# 추론

학습된 Seq2Seq 모델에 텍스트를 넣고 탐욕 선택이나 빔 탐색으로 출력 토큰을 얻는 도구다. 이 페이지의 실행부는 무작위 초기 모델을 사용하므로 출력 문자열은 학습된 번역 결과가 아니다.

## 1. 코드

```python
"""
Seq2Seq 모델을 위한 추론 스크립트
탐욕 디코딩, 빔 탐색, 번역 도구를 담고 있다
"""

import torch
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# --- encoder ---
"""
Seq2Seq 모델을 위한 인코더 모듈
LSTM과 GRU를 비롯한 여러 인코더 구조를 구현한다
"""


# ========================================================================
# 메인
# ========================================================================


class BasicEncoder(nn.Module):
    """
    Seq2Seq 모델을 위한 기본 RNN 인코더
    
    인수:
        input_size: 입력 어휘의 크기
        embedding_dim: 단어 임베딩의 차원
        hidden_size: 은닉 상태의 크기
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
            input_seq: 입력 시퀀스 텐서 (배치 크기, seq_len)
            input_lengths: 시퀀스의 실제 길이 (선택)
            
        반환값:
            outputs: 모든 은닉 상태 (배치 크기, seq_len, hidden_size * num_directions)
            hidden: 마지막 은닉 상태
            cell: 마지막 셀 상태 (LSTM에만 있다)
        """
        # 입력 임베딩
        embedded = self.embedding(input_seq)
        embedded = self.dropout(embedded)
        
        # 길이가 주어지면 패딩된 시퀀스를 패킹하기
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
        
        # 패킹했으면 풀기
        if input_lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        
        # 양방향이면 순방향과 역방향 은닉 상태를 합치기
        if self.bidirectional:
            # hidden: (num_layers * 2, 배치 크기, hidden_size)
            # (num_layers, 배치 크기, hidden_size * 2)로 형태 바꾸기
            hidden = self._combine_bidirectional(hidden)
            if cell is not None:
                cell = self._combine_bidirectional(cell)
        
        return outputs, hidden, cell
    
    def _combine_bidirectional(self, hidden):
        """순방향과 역방향의 은닉 상태를 합친다"""
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
    시퀀스 인코딩에 1차원 컨볼루션을 쓴다
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
            input_seq: 입력 시퀀스 (배치 크기, seq_len)
            
        반환값:
            outputs: 인코딩된 표현 (배치 크기, seq_len, hidden_size)
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


# --- decoder ---
"""
Seq2Seq 모델을 위한 디코더 모듈
어텐션이 있는 것과 없는 것 등 여러 디코더 구조를 구현한다
"""


# ========================================================================
# 메인
# ========================================================================


class BasicDecoder(nn.Module):
    """
    Seq2Seq 모델을 위한 기본 RNN 디코더
    
    인수:
        output_size: 출력 어휘의 크기
        embedding_dim: 단어 임베딩의 차원
        hidden_size: 은닉 상태의 크기
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
        한 타임스텝의 순전파
        
        인수:
            input_token: 입력 토큰 (배치 크기, 1)
            hidden: 이전 타임스텝의 은닉 상태
            cell: 이전 타임스텝의 셀 상태 (LSTM에만 있다)
            
        반환값:
            output: 출력 예측 (배치 크기, output_size)
            hidden: 갱신된 은닉 상태
            cell: 갱신된 셀 상태 (LSTM에만 있다)
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
    바다나우(덧셈) 어텐션 메커니즘이 있는 디코더
    
    인수:
        output_size: 출력 어휘의 크기
        embedding_dim: 단어 임베딩의 차원
        hidden_size: 디코더 은닉 상태의 크기
        encoder_hidden_size: 인코더 은닉 상태의 크기
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
        
        # 어텐션 메커니즘
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
        어텐션을 쓰는 한 타임스텝의 순전파
        
        인수:
            input_token: 입력 토큰 (배치 크기, 1)
            hidden: 이전 타임스텝의 은닉 상태
            encoder_outputs: 인코더의 모든 출력 (배치 크기, src_len, encoder_hidden_size)
            cell: 이전 타임스텝의 셀 상태 (LSTM에만 있다)
            mask: 패딩을 가리는 마스크 (배치 크기, src_len)
            
        반환값:
            output: 출력 예측 (배치 크기, output_size)
            hidden: 갱신된 은닉 상태
            cell: 갱신된 셀 상태 (LSTM에만 있다)
            attention_weights: 어텐션 웨이트 (배치 크기, src_len)
        """
        # 입력 토큰 임베딩
        embedded = self.embedding(input_token)  # (배치 크기, 1, embedding_dim)
        embedded = self.dropout(embedded)
        
        # 어텐션 계산
        # 어텐션에 맨 위 레이어의 은닉 상태 쓰기
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
    바다나우(덧셈) 어텐션 메커니즘
    
    인수:
        decoder_hidden_size: 디코더 은닉 상태의 크기
        encoder_hidden_size: 인코더 은닉 상태의 크기
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
            query: 디코더의 은닉 상태 (배치 크기, 1, decoder_hidden_size)
            keys: 인코더의 출력 (배치 크기, src_len, encoder_hidden_size)
            mask: 패딩 마스크 (배치 크기, src_len)
            
        반환값:
            context: 문맥 벡터 (배치 크기, 1, encoder_hidden_size)
            attention_weights: 어텐션 웨이트 (배치 크기, 1, src_len)
        """
        # 어텐션 점수 계산
        # query: (배치, 1, dec_hidden)
        # keys: (배치, src_len, enc_hidden)
        
        query_transformed = self.W_decoder(query)  # (배치, 1, dec_hidden)
        keys_transformed = self.W_encoder(keys)    # (배치, src_len, dec_hidden)
        
        # 브로드캐스팅하여 더하기
        scores = self.V(torch.tanh(query_transformed + keys_transformed))  # (배치, src_len, 1)
        scores = scores.squeeze(2)  # (배치, src_len)
        
        # 마스크가 있으면 씌우기
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # 어텐션 웨이트 계산
        attention_weights = F.softmax(scores, dim=1).unsqueeze(1)  # (배치, 1, src_len)
        
        # 문맥 벡터 계산
        context = torch.bmm(attention_weights, keys)  # (배치, 1, encoder_hidden_size)
        
        return context, attention_weights


class LuongAttention(nn.Module):
    """
    루옹(곱셈) 어텐션 메커니즘
    
    인수:
        decoder_hidden_size: 디코더 은닉 상태의 크기
        encoder_hidden_size: 인코더 은닉 상태의 크기
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
            query: 디코더의 은닉 상태 (배치 크기, 1, decoder_hidden_size)
            keys: 인코더의 출력 (배치 크기, src_len, encoder_hidden_size)
            mask: 패딩 마스크 (배치 크기, src_len)
            
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
        
        # 마스크가 있으면 씌우기
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


# --- seq2seq_model ---
"""
완전한 시퀀스-투-시퀀스 모델
인코더와 디코더를 엮어 온전한 구조를 만든다
"""


# ========================================================================
# 메인
# ========================================================================


class Seq2Seq(nn.Module):
    """
    어텐션이 없는 기본 시퀀스-투-시퀀스 모델
    
    인수:
        encoder: 인코더 모듈
        decoder: 디코더 모듈
        device: 돌릴 장치
    """
    
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5, src_lengths=None):
        """
        seq2seq 모델을 지나는 순전파
        
        인수:
            src: 원본 시퀀스 (배치 크기, src_len)
            trg: 타깃 시퀀스 (배치 크기, trg_len)
            teacher_forcing_ratio: 티처 포싱을 쓸 확률
            src_lengths: 원본 시퀀스의 실제 길이
            
        반환값:
            outputs: 타임스텝마다의 예측 (배치 크기, trg_len, output_size)
        """
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_size
        
        # 디코더의 출력을 담을 텐서
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        # 원본 시퀀스 인코딩
        encoder_outputs, hidden, cell = self.encoder(src, src_lengths)
        
        # 디코더의 첫 입력은 <sos> 토큰이다
        decoder_input = trg[:, 0].unsqueeze(1)
        
        # 한 번에 토큰 하나씩 디코딩
        for t in range(1, trg_len):
            # 디코더를 지나는 순전파
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            
            # 예측 담기
            outputs[:, t] = output
            
            # 티처 포싱을 쓸지 정하기
            teacher_force = random.random() < teacher_forcing_ratio
            
            # 예측한 토큰 얻기
            top1 = output.argmax(1)
            
            # 다음 입력은 정답이거나 예측한 토큰이다
            decoder_input = trg[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)
        
        return outputs
    
    def generate(self, src, max_len=50, sos_idx=1, eos_idx=2, src_lengths=None):
        """
        탐욕 디코딩으로 시퀀스를 만든다
        
        인수:
            src: 원본 시퀀스 (배치 크기, src_len)
            max_len: 만들 시퀀스의 최대 길이
            sos_idx: 시퀀스 시작 토큰의 인덱스
            eos_idx: 시퀀스 끝 토큰의 인덱스
            src_lengths: 원본 시퀀스의 실제 길이
            
        반환값:
            generated: 만들어진 시퀀스 (배치 크기, seq_len)
        """
        self.eval()
        
        with torch.no_grad():
            batch_size = src.shape[0]
            
            # 원본 인코딩
            encoder_outputs, hidden, cell = self.encoder(src, src_lengths)
            
            # <sos> 토큰으로 시작
            decoder_input = torch.full((batch_size, 1), sos_idx, dtype=torch.long).to(self.device)
            
            generated = [decoder_input]
            finished = torch.zeros(batch_size, dtype=torch.bool).to(self.device)
            
            for _ in range(max_len):
                # 순전파
                output, hidden, cell = self.decoder(decoder_input, hidden, cell)
                
                # 예측한 토큰 얻기
                predicted = output.argmax(1).unsqueeze(1)
                
                # 끝난 시퀀스 갱신
                finished |= (predicted.squeeze(1) == eos_idx)
                
                # 예측 담기
                generated.append(predicted)
                
                # 모든 시퀀스가 끝나면 멈추기
                if finished.all():
                    break
                
                # 다음 입력
                decoder_input = predicted
            
            # 모든 예측 이어 붙이기
            generated = torch.cat(generated, dim=1)
            
        return generated


class Seq2SeqAttention(nn.Module):
    """
    어텐션 메커니즘이 있는 시퀀스-투-시퀀스 모델
    
    인수:
        encoder: 인코더 모듈
        decoder: 어텐션이 있는 디코더 모듈
        device: 돌릴 장치
        pad_idx: 패딩 토큰의 인덱스
    """
    
    def __init__(self, encoder, decoder, device, pad_idx=0):
        super(Seq2SeqAttention, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.pad_idx = pad_idx
        
    def create_mask(self, src):
        """패딩 토큰의 마스크를 만든다"""
        mask = (src != self.pad_idx)
        return mask
    
    def forward(self, src, trg, teacher_forcing_ratio=0.5, src_lengths=None):
        """
        어텐션이 있는 seq2seq 모델을 지나는 순전파
        
        인수:
            src: 원본 시퀀스 (배치 크기, src_len)
            trg: 타깃 시퀀스 (배치 크기, trg_len)
            teacher_forcing_ratio: 티처 포싱을 쓸 확률
            src_lengths: 원본 시퀀스의 실제 길이
            
        반환값:
            outputs: 타임스텝마다의 예측 (배치 크기, trg_len, output_size)
            attentions: 타임스텝마다의 어텐션 웨이트 (배치 크기, trg_len, src_len)
        """
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_size
        
        # 출력과 어텐션 웨이트를 담을 텐서
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        attentions = torch.zeros(batch_size, trg_len, src.shape[1]).to(self.device)
        
        # 원본 시퀀스의 마스크 만들기
        mask = self.create_mask(src)
        
        # 원본 시퀀스 인코딩
        encoder_outputs, hidden, cell = self.encoder(src, src_lengths)
        
        # 디코더의 첫 입력은 <sos> 토큰이다
        decoder_input = trg[:, 0].unsqueeze(1)
        
        # 한 번에 토큰 하나씩 디코딩
        for t in range(1, trg_len):
            # 어텐션이 있는 디코더를 지나는 순전파
            output, hidden, cell, attention_weights = self.decoder(
                decoder_input, hidden, encoder_outputs, cell, mask
            )
            
            # 예측과 어텐션 웨이트 담기
            outputs[:, t] = output
            attentions[:, t] = attention_weights
            
            # 티처 포싱을 쓸지 정하기
            teacher_force = random.random() < teacher_forcing_ratio
            
            # 예측한 토큰 얻기
            top1 = output.argmax(1)
            
            # 다음 입력은 정답이거나 예측한 토큰이다
            decoder_input = trg[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)
        
        return outputs, attentions
    
    def generate(self, src, max_len=50, sos_idx=1, eos_idx=2, src_lengths=None):
        """
        어텐션과 함께 탐욕 디코딩으로 시퀀스를 만든다
        
        인수:
            src: 원본 시퀀스 (배치 크기, src_len)
            max_len: 만들 시퀀스의 최대 길이
            sos_idx: 시퀀스 시작 토큰의 인덱스
            eos_idx: 시퀀스 끝 토큰의 인덱스
            src_lengths: 원본 시퀀스의 실제 길이
            
        반환값:
            generated: 만들어진 시퀀스 (배치 크기, seq_len)
            all_attentions: 스텝마다의 어텐션 웨이트 (배치 크기, seq_len, src_len)
        """
        self.eval()
        
        with torch.no_grad():
            batch_size = src.shape[0]
            
            # 마스크 만들기
            mask = self.create_mask(src)
            
            # 원본 인코딩
            encoder_outputs, hidden, cell = self.encoder(src, src_lengths)
            
            # <sos> 토큰으로 시작
            decoder_input = torch.full((batch_size, 1), sos_idx, dtype=torch.long).to(self.device)
            
            generated = [decoder_input]
            all_attentions = []
            finished = torch.zeros(batch_size, dtype=torch.bool).to(self.device)
            
            for _ in range(max_len):
                # 순전파
                output, hidden, cell, attention_weights = self.decoder(
                    decoder_input, hidden, encoder_outputs, cell, mask
                )
                
                # 어텐션 웨이트 담기
                all_attentions.append(attention_weights.unsqueeze(1))
                
                # 예측한 토큰 얻기
                predicted = output.argmax(1).unsqueeze(1)
                
                # 끝난 시퀀스 갱신
                finished |= (predicted.squeeze(1) == eos_idx)
                
                # 예측 담기
                generated.append(predicted)
                
                # 모든 시퀀스가 끝나면 멈추기
                if finished.all():
                    break
                
                # 다음 입력
                decoder_input = predicted
            
            # 모든 예측과 어텐션 이어 붙이기
            generated = torch.cat(generated, dim=1)
            all_attentions = torch.cat(all_attentions, dim=1)
            
        return generated, all_attentions
    
    def beam_search(self, src, beam_width=5, max_len=50, sos_idx=1, eos_idx=2, 
                   src_lengths=None, length_penalty=0.6):
        """
        빔 탐색으로 시퀀스를 만든다
        
        인수:
            src: 원본 시퀀스 (1, src_len) — 시퀀스 하나만
            beam_width: 빔의 수
            max_len: 만들 시퀀스의 최대 길이
            sos_idx: 시퀀스 시작 토큰의 인덱스
            eos_idx: 시퀀스 끝 토큰의 인덱스
            src_lengths: 원본 시퀀스의 실제 길이
            length_penalty: 길이 정규화 패널티 (alpha)
            
        반환값:
            best_sequence: 가장 좋은 시퀀스
            best_score: 가장 좋은 시퀀스의 점수
        """
        self.eval()
        
        with torch.no_grad():
            # 원본 인코딩
            mask = self.create_mask(src)
            encoder_outputs, hidden, cell = self.encoder(src, src_lengths)
            
            # 빔 초기화
            # 빔마다: (시퀀스, 점수, 은닉 상태, 셀 상태)
            beams = [(torch.full((1, 1), sos_idx, dtype=torch.long).to(self.device), 
                     0.0, hidden, cell)]
            completed = []
            
            for step in range(max_len):
                candidates = []
                
                for sequence, score, h, c in beams:
                    # 시퀀스가 이미 끝났으면 건너뛰기
                    if sequence[0, -1].item() == eos_idx:
                        completed.append((sequence, score, h, c))
                        continue
                    
                    # 마지막 토큰 얻기
                    last_token = sequence[:, -1].unsqueeze(1)
                    
                    # 순전파
                    output, new_h, new_c, _ = self.decoder(
                        last_token, h, encoder_outputs, c, mask
                    )
                    
                    # 상위 k개 예측 얻기
                    log_probs = torch.log_softmax(output, dim=1)
                    top_probs, top_indices = log_probs.topk(beam_width)
                    
                    # 새 후보 만들기
                    for prob, idx in zip(top_probs[0], top_indices[0]):
                        new_sequence = torch.cat([sequence, idx.unsqueeze(0).unsqueeze(0)], dim=1)
                        new_score = score + prob.item()
                        candidates.append((new_sequence, new_score, new_h, new_c))
                
                # 상위 beam_width개 후보 고르기
                candidates.sort(key=lambda x: x[1] / (len(x[0][0]) ** length_penalty), reverse=True)
                beams = candidates[:beam_width]
                
                # 모든 빔이 끝나면 멈추기
                if len(beams) == 0:
                    break
            
            # 남은 빔을 끝난 목록에 더하기
            completed.extend(beams)
            
            # 가장 좋은 시퀀스 고르기
            if completed:
                best = max(completed, key=lambda x: x[1] / (len(x[0][0]) ** length_penalty))
                return best[0], best[1]
            else:
                return beams[0][0], beams[0][1]


if __name__ == "__main__":
    # 사용 예
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 하이퍼파라메터
    input_vocab_size = 10000
    output_vocab_size = 10000
    embedding_dim = 256
    hidden_size = 512
    num_layers = 2
    dropout = 0.1
    
    # 인코더와 디코더 만들기
    encoder = BasicEncoder(
        input_size=input_vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=True,
        rnn_type='LSTM'
    ).to(device)
    
    decoder = AttentionDecoder(
        output_size=output_vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size * 2,  # 인코더가 양방향이라 2를 곱한다
        encoder_hidden_size=hidden_size * 2,
        num_layers=num_layers,
        dropout=dropout,
        rnn_type='LSTM'
    ).to(device)
    
    # seq2seq 모델 만들기
    model = Seq2SeqAttention(encoder, decoder, device).to(device)
    
    # 예제 데이터
    batch_size = 32
    src_len = 20
    trg_len = 25
    
    src = torch.randint(3, input_vocab_size, (batch_size, src_len)).to(device)
    trg = torch.randint(3, output_vocab_size, (batch_size, trg_len)).to(device)
    
    # 순전파
    outputs, attentions = model(src, trg, teacher_forcing_ratio=0.5)
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Input shape: {src.shape}")
    print(f"Target shape: {trg.shape}")
    print(f"Output shape: {outputs.shape}")
    print(f"Attention shape: {attentions.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")


# ========================================================================
# 메인
# ========================================================================


class Seq2SeqInference:
    """
    Seq2Seq 모델을 위한 추론 도구
    
    인수:
        model: 학습된 Seq2Seq 모델
        src_vocab: 원본 어휘 (토큰을 인덱스로)
        trg_vocab: 타깃 어휘 (토큰을 인덱스로)
        device: 추론을 돌릴 장치
        sos_idx: 시퀀스 시작 토큰의 인덱스
        eos_idx: 시퀀스 끝 토큰의 인덱스
        pad_idx: 패딩 토큰의 인덱스
    """
    
    def __init__(self, model, src_vocab, trg_vocab, device, sos_idx=1, eos_idx=2, pad_idx=0):
        self.model = model
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.device = device
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        
        # 거꾸로 된 어휘 만들기 (인덱스에서 토큰으로)
        self.idx_to_src = {idx: token for token, idx in src_vocab.items()}
        self.idx_to_trg = {idx: token for token, idx in trg_vocab.items()}
    
    def tokenize_source(self, text):
        """
        원본 텍스트를 토큰으로 나눈다
        
        인수:
            text: 원본 텍스트 문자열이나 토큰의 목록
            
        반환값:
            tokens: 토큰 인덱스의 목록
        """
        if isinstance(text, str):
            tokens = text.lower().split()
        else:
            tokens = text
        
        # 인덱스로 바꾸기
        indices = [self.src_vocab.get(token, self.src_vocab.get('<unk>', 3)) 
                   for token in tokens]
        
        return indices
    
    def detokenize_target(self, indices):
        """
        타깃의 인덱스를 다시 텍스트로 바꾼다
        
        인수:
            indices: 토큰 인덱스의 목록이나 텐서
            
        반환값:
            text: 토큰을 되돌린 텍스트 문자열
        """
        if torch.is_tensor(indices):
            indices = indices.tolist()
        
        tokens = []
        for idx in indices:
            if idx == self.eos_idx:
                break
            if idx not in [self.pad_idx, self.sos_idx]:
                token = self.idx_to_trg.get(idx, '<unk>')
                tokens.append(token)
        
        return ' '.join(tokens)
    
    def greedy_decode(self, src_text, max_len=50):
        """
        탐욕 디코딩으로 번역을 만든다
        
        인수:
            src_text: 원본 텍스트 문자열이나 토큰의 목록
            max_len: 만들 시퀀스의 최대 길이
            
        반환값:
            translation: 번역된 텍스트
            attention_weights: 어텐션 웨이트 (있을 때)
        """
        self.model.eval()
        
        # 원본을 토큰으로 나누기
        src_indices = self.tokenize_source(src_text)
        src_tensor = torch.tensor([src_indices]).to(self.device)
        src_lengths = torch.tensor([len(src_indices)]).to(self.device)
        
        # 생성
        with torch.no_grad():
            if hasattr(self.model, 'generate'):
                if hasattr(self.model, 'encoder'):
                    # 어텐션이 있는 모델
                    output, attention = self.model.generate(
                        src_tensor, max_len, self.sos_idx, self.eos_idx, src_lengths
                    )
                    attention = attention[0].cpu()
                else:
                    output = self.model.generate(
                        src_tensor, max_len, self.sos_idx, self.eos_idx, src_lengths
                    )
                    attention = None
            else:
                raise ValueError("Model doesn't have generate method")
        
        # 토큰을 다시 글로
        translation = self.detokenize_target(output[0])
        
        return translation, attention
    
    def beam_search_decode(self, src_text, beam_width=5, max_len=50, length_penalty=0.6):
        """
        빔 탐색으로 번역을 만든다
        
        인수:
            src_text: 원본 텍스트 문자열이나 토큰의 목록
            beam_width: 빔의 수
            max_len: 만들 시퀀스의 최대 길이
            length_penalty: 길이 정규화 패널티
            
        반환값:
            translation: 번역된 텍스트
            score: 가장 좋은 번역의 점수
        """
        self.model.eval()
        
        # 원본을 토큰으로 나누기
        src_indices = self.tokenize_source(src_text)
        src_tensor = torch.tensor([src_indices]).to(self.device)
        src_lengths = torch.tensor([len(src_indices)]).to(self.device)
        
        # 생성
        with torch.no_grad():
            if hasattr(self.model, 'beam_search'):
                output, score = self.model.beam_search(
                    src_tensor, beam_width, max_len, 
                    self.sos_idx, self.eos_idx, src_lengths, length_penalty
                )
            else:
                raise ValueError("Model doesn't have beam_search method")
        
        # 토큰을 다시 글로
        translation = self.detokenize_target(output[0])
        
        return translation, score
    
    def translate_batch(self, src_texts, method='greedy', **kwargs):
        """
        텍스트 배치를 번역한다
        
        인수:
            src_texts: 원본 텍스트의 목록
            method: 디코딩 방법 ('greedy' 또는 'beam')
            **kwargs: 디코딩에 쓸 추가 인자
            
        반환값:
            translations: 번역된 텍스트의 목록
        """
        translations = []
        
        for src_text in src_texts:
            if method == 'greedy':
                translation, _ = self.greedy_decode(src_text, **kwargs)
            elif method == 'beam':
                translation, _ = self.beam_search_decode(src_text, **kwargs)
            else:
                raise ValueError(f"Unknown decoding method: {method}")
            
            translations.append(translation)
        
        return translations
    
    def interactive_translate(self):
        """대화식 번역 모드"""
        print("Interactive Translation Mode")
        print("Enter 'quit' to exit")
        print("-" * 50)
        
        while True:
            src_text = input("\nSource: ").strip()
            
            if src_text.lower() == 'quit':
                break
            
            if not src_text:
                continue
            
            # 탐욕 디코딩
            translation_greedy, attention = self.greedy_decode(src_text)
            print(f"Greedy: {translation_greedy}")
            
            # 빔 탐색 (쓸 수 있으면)
            try:
                translation_beam, score = self.beam_search_decode(src_text)
                print(f"Beam:   {translation_beam} (score: {score:.2f})")
            except:
                pass
    
    def visualize_attention(self, src_text, translation, attention_weights):
        """
        어텐션 웨이트 그려 보기
        
        인수:
            src_text: 원본 텍스트
            translation: 번역된 텍스트
            attention_weights: 어텐션 웨이트 텐서
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # 토큰으로 나누기
            src_tokens = src_text.lower().split()
            trg_tokens = translation.split()
            
            # 어텐션을 실제 길이에 맞게 잘라 내기
            attention = attention_weights[:len(trg_tokens), :len(src_tokens)].numpy()
            
            # 그림
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(attention, xticklabels=src_tokens, yticklabels=trg_tokens,
                       cmap='viridis', ax=ax, cbar=True)
            ax.set_xlabel('Source')
            ax.set_ylabel('Target')
            ax.set_title('Attention Weights')
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib and seaborn required for visualization")


class BLEU:
    """
    BLEU 점수 계산기
    """
    
    @staticmethod
    def compute_bleu(reference, hypothesis, max_n=4):
        """
        BLEU 점수를 계산한다
        
        인수:
            reference: 기준 번역 (문자열이나 토큰의 목록)
            hypothesis: 가설 번역 (문자열이나 토큰의 목록)
            max_n: n-그램의 최대 차수
            
        반환값:
            bleu_score: BLEU 점수
        """
        if isinstance(reference, str):
            reference = reference.split()
        if isinstance(hypothesis, str):
            hypothesis = hypothesis.split()
        
        # n-그램마다 정밀도 계산
        precisions = []
        for n in range(1, max_n + 1):
            ref_ngrams = BLEU._get_ngrams(reference, n)
            hyp_ngrams = BLEU._get_ngrams(hypothesis, n)
            
            if len(hyp_ngrams) == 0:
                precisions.append(0)
                continue
            
            # 맞은 개수 세기
            matches = sum(min(ref_ngrams.get(ng, 0), hyp_ngrams.get(ng, 0)) 
                         for ng in hyp_ngrams)
            
            precision = matches / len(hyp_ngrams)
            precisions.append(precision)
        
        # 간결성 패널티(brevity penalty)
        bp = BLEU._brevity_penalty(len(reference), len(hypothesis))
        
        # 정밀도의 기하 평균
        if min(precisions) > 0:
            log_precision_sum = sum(torch.log(torch.tensor(p)) for p in precisions)
            geo_mean = torch.exp(log_precision_sum / max_n)
            bleu_score = bp * geo_mean.item()
        else:
            bleu_score = 0
        
        return bleu_score
    
    @staticmethod
    def _get_ngrams(tokens, n):
        """토큰에서 n-그램을 뽑는다"""
        ngrams = {}
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams[ngram] = ngrams.get(ngram, 0) + 1
        return ngrams
    
    @staticmethod
    def _brevity_penalty(ref_len, hyp_len):
        """간결성 패널티(brevity penalty)를 계산한다"""
        if hyp_len > ref_len:
            return 1.0
        else:
            return torch.exp(torch.tensor(1 - ref_len / hyp_len)).item()


if __name__ == "__main__":
    # 사용 예
    
    # 장치 지정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 임시 어휘 만들기
    src_vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
    trg_vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
    
    # 임시 단어 몇 개 더하기
    words = ['hello', 'world', 'how', 'are', 'you', 'good', 'morning']
    for i, word in enumerate(words):
        src_vocab[word] = i + 4
        trg_vocab[word] = i + 4
    
    # 모델 생성
    INPUT_DIM = len(src_vocab)
    OUTPUT_DIM = len(trg_vocab)
    
    encoder = BasicEncoder(
        input_size=INPUT_DIM,
        embedding_dim=256,
        hidden_size=512,
        num_layers=2,
        bidirectional=True,
        rnn_type='LSTM'
    )
    
    decoder = AttentionDecoder(
        output_size=OUTPUT_DIM,
        embedding_dim=256,
        hidden_size=1024,
        encoder_hidden_size=1024,
        num_layers=2,
        rnn_type='LSTM'
    )
    
    model = Seq2SeqAttention(encoder, decoder, device).to(device)
    
    # 추론 객체 만들기
    inference = Seq2SeqInference(model, src_vocab, trg_vocab, device)
    
    # 번역 시험
    src_text = "hello world"
    translation, attention = inference.greedy_decode(src_text)
    
    print(f"Source: {src_text}")
    print(f"Translation: {translation}")
    
    # BLEU 점수 시험
    reference = "good morning world"
    hypothesis = "good morning"
    bleu_score = BLEU.compute_bleu(reference, hypothesis)
    print(f"\nBLEU score: {bleu_score:.4f}")
```

**출력:**

```
Source: hello world
Translation: 

BLEU score: 0.0000
```

## 2. 논의

추론 래퍼의 원문 토큰화는 소문자·공백 분리다. 전처리 페이지의 구두점 분리와 다르므로 학습에 사용한 규칙과 맞춰야 한다. 사전 객체가 아닌 token→index 딕셔너리를 받으며 특수 토큰 ID도 모델과 일치해야 한다. 빈 문자열은 길이 0 입력을 만들어 패킹에서 실패할 수 있다.

탐욕 디코딩은 model.generate에 위임한다. encoder 속성만으로 `(output,attention)` 반환을 가정하므로 기본 Seq2Seq처럼 텐서만 반환하는 모델에는 맞지 않는다. 빔 탐색은 해당 메서드가 있는 모델만 사용할 수 있다. translate_batch는 실제 배치 연산이 아니라 문장별 순차 호출이다.

타깃 복원은 첫 EOS에서 중단하고 PAD·SOS를 제외한다. 시각화는 이 복원된 토큰 수로 어텐션 행을 자르므로 생성 단계와의 대응, 학습·추론 토큰화가 같은지 확인해야 한다. 웨이트는 관련성의 계산 계수이며 인과적인 중요도 자체는 아니다.

BLEU 구현에는 중요한 한계가 있다. n-그램 정밀도의 분모에 전체 발생 수가 아니라 서로 다른 n-그램 종류 수를 사용하므로, 반복되는 일치가 있으면 정밀도가 1을 넘을 수도 있다. 표준 clipped precision의 분모는 후보 n-그램 발생 수의 합이어야 한다. 빈 가설은 brevity penalty에서 0 나눗셈이 생긴다. 짧은 문장은 높은 차수 n-그램이 없어 기본 4-그램 점수가 0이 될 수 있다. 이 함수의 수치를 표준 코퍼스 BLEU로 보고해서는 안 된다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
학습 전처리와 추론 토큰화가 다르면 어떤 문제가 생기는가?

</div>

??? success "풀이"
    같은 단어나 문장 부호가 다른 토큰으로 나뉘어 OOV와 입력 ID가 달라질 수 있다. 학습 때의 토큰화·정규화·사전과 특수 토큰 규약을 동일하게 사용해야 한다. 무작위 모델의 출력 문자열로 번역 품질을 평가할 수는 없다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
현재 BLEU 정밀도의 분모가 잘못된 이유를 간단한 반복 예로 설명하라.

</div>

??? success "풀이"
    후보와 정답이 모두 a a라면 유니그램 일치 발생 수는 2다. 현재 코드는 후보의 유니그램 종류 수 1로 나누어 2를 얻는다. 올바른 분모는 발생 수 합 2여서 정밀도는 1이다. n-그램 종류와 횟수를 구분해야 한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
greedy, beam과 translate_batch의 역할을 구분하라.

</div>

??? success "풀이"
    greedy는 단계별 최대 로짓 토큰을 선택하는 모델 메서드에 위임한다. beam은 제한된 여러 후보를 유지하는 근사 탐색이다. translate_batch는 문장을 하나씩 처리하므로 병렬 배치 추론과 다르다. 각 방식의 점수·종료 조건과 실제 시간을 맞춰 비교해야 한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
풀이 첫머리에 실린 테스트 코드 조각을 검토하라. 이 페이지의 클래스와 맞지 않는 호출, 쓸 수 없는 이름, 입력 형태 가정을 찾아 이 페이지의 실제 함수에 맞는 점검 항목을 제안하라.

</div>

??? success "풀이"
    아래 조각은 실제로 실행되는 테스트가 아니다. 이 페이지의 `Seq2SeqInference` 는 `nn.Module` 이 아니라 학습된 모델과 사전을 감싸는 래퍼다. `__call__` 이 없어 `model(...)` 처럼 부를 수 없고 `greedy_decode(src_text)` 나 `beam_search_decode(src_text)` 로 불러야 하며, 돌려주는 것도 텐서가 아니라 `(번역 문자열, 어텐션 또는 점수)` 쌍이다. 래퍼 자체에는 `parameters()` 가 없다.
    ```python
    def test_seq2seqinference():
        model = Seq2SeqInference(...)
        # 보통의 입력
        assert model(normal_input).shape == expected_shape
        # 원소가 하나인 배치
        assert model(single_input).shape == (1, ...)
        # 큰 값 (오버플로를 확인한다)
        out = model(torch.ones(...) * 1000)
        assert torch.isfinite(out).all()
        # 그래디언트의 흐름
        out = model(normal_input)
        out.sum().backward()
        for p in model.parameters():
            assert p.grad is not None
    ```
    추론은 `no_grad` 안에서 돌므로 그래디언트 항목은 해당하지 않는다. 대신 반환 문자열에 SOS·PAD 가 섞이지 않는지, 첫 EOS 에서 멈추는지, 사전에 없는 단어가 `<unk>` 로 가는지, 어텐션 행렬의 형태가 (생성 길이, 입력 길이) 인지를 검사한다. 빈 문자열은 길이 0 입력이 되어 패킹에서 실패할 수 있으므로, 실패하는 것이 맞는지 미리 걸러야 하는지를 정해 두고 검사한다.

## 정리하며

추론에서는 학습 때의 전처리와 사전, 모델 반환 규약을 유지한다. 검색 방식과 시각화의 의미를 구분하고, 여기의 간단한 BLEU 함수를 표준 평가기로 사용하지 않는다.
