# 학습

Seq2Seq의 배치 패딩, 로스 계산, 그래디언트 클리핑과 체크포인트 관리를 모은 예제다. 실행부는 무작위 정수 시퀀스를 사용하므로 실제 번역 학습의 품질을 보여 주지 않는다. 모델의 반환값과 평가 목적을 공통 학습 함수에 맞춰야 한다.

## 1. 코드

```python
"""
Seq2Seq 모델을 위한 학습 스크립트
학습 반복문, 평가, 검사점 저장을 담고 있다
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import time
import math
from pathlib import Path

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
        embedding_dim: 낱말 임베딩의 차원
        hidden_size: 숨은 상태의 크기
        num_layers: 순환 층의 수
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
        
        # 임베딩 층
        self.embedding = nn.Embedding(input_size, embedding_dim)
        
        # 드롭아웃 층
        self.dropout = nn.Dropout(dropout)
        
        # RNN 층
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
    Seq2Seq 모델을 위한 합성곱 인코더
    순차열 부호화에 1차원 합성곱을 쓴다
    """
    
    def __init__(self, input_size, embedding_dim, hidden_size, 
                 num_layers=3, kernel_size=3, dropout=0.1):
        super(ConvEncoder, self).__init__()
        
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 합성곱 층
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
        합성곱 인코더를 지나는 순전파
        
        인수:
            input_seq: 입력 순차열 (배치 크기, seq_len)
            
        반환값:
            outputs: 부호화된 표현 (배치 크기, seq_len, hidden_size)
        """
        # 임베딩하고 conv1d를 위해 전치
        embedded = self.embedding(input_seq)  # (배치, seq_len, embed_dim)
        embedded = self.dropout(embedded)
        embedded = embedded.transpose(1, 2)  # (배치, embed_dim, seq_len)
        
        # 합성곱 적용
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
        embedding_dim: 낱말 임베딩의 차원
        hidden_size: 숨은 상태의 크기
        num_layers: 순환 층의 수
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
        
        # 임베딩 층
        self.embedding = nn.Embedding(output_size, embedding_dim)
        
        # 드롭아웃
        self.dropout = nn.Dropout(dropout)
        
        # RNN 층
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
        
        # 출력층
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
        num_layers: 순환 층의 수
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
        
        # 임베딩 층
        self.embedding = nn.Embedding(output_size, embedding_dim)
        
        # 드롭아웃
        self.dropout = nn.Dropout(dropout)
        
        # 어텐션 장치
        self.attention = BahdanauAttention(hidden_size, encoder_hidden_size)
        
        # RNN 층 (입력은 임베딩과 문맥 벡터)
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
        
        # 출력층
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
            attention_weights: 어텐션 가중치 (배치 크기, src_len)
        """
        # 입력 토큰 임베딩
        embedded = self.embedding(input_token)  # (배치 크기, 1, embedding_dim)
        embedded = self.dropout(embedded)
        
        # 어텐션 계산
        # 어텐션에 맨 위 층의 숨은 상태 쓰기
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
        어텐션 가중치와 문맥 벡터를 계산한다
        
        인수:
            query: 디코더의 숨은 상태 (배치 크기, 1, decoder_hidden_size)
            keys: 인코더의 출력 (배치 크기, src_len, encoder_hidden_size)
            mask: 덧댐 가림막 (배치 크기, src_len)
            
        반환값:
            context: 문맥 벡터 (배치 크기, 1, encoder_hidden_size)
            attention_weights: 어텐션 가중치 (배치 크기, 1, src_len)
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
        
        # 어텐션 가중치 계산
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
        어텐션 가중치와 문맥 벡터를 계산한다
        
        인수:
            query: 디코더의 숨은 상태 (배치 크기, 1, decoder_hidden_size)
            keys: 인코더의 출력 (배치 크기, src_len, encoder_hidden_size)
            mask: 덧댐 가림막 (배치 크기, src_len)
            
        반환값:
            context: 문맥 벡터 (배치 크기, 1, encoder_hidden_size)
            attention_weights: 어텐션 가중치 (배치 크기, 1, src_len)
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
        
        # 어텐션 가중치 계산
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
완전한 순차열 대 순차열 모델
인코더와 디코더를 엮어 온전한 구조를 만든다
"""


# ========================================================================
# 메인
# ========================================================================


class Seq2Seq(nn.Module):
    """
    어텐션이 없는 기본 순차열 대 순차열 모델
    
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
            src: 원본 순차열 (배치 크기, src_len)
            trg: 표적 순차열 (배치 크기, trg_len)
            teacher_forcing_ratio: 교사 강요를 쓸 확률
            src_lengths: 원본 순차열의 실제 길이
            
        반환값:
            outputs: 시각마다의 예측 (배치 크기, trg_len, output_size)
        """
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_size
        
        # 디코더의 출력을 담을 텐서
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        # 원본 순차열 부호화
        encoder_outputs, hidden, cell = self.encoder(src, src_lengths)
        
        # 디코더의 첫 입력은 <sos> 토큰이다
        decoder_input = trg[:, 0].unsqueeze(1)
        
        # 한 번에 토큰 하나씩 복호
        for t in range(1, trg_len):
            # 디코더를 지나는 순전파
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            
            # 예측 담기
            outputs[:, t] = output
            
            # 교사 강요를 쓸지 정하기
            teacher_force = random.random() < teacher_forcing_ratio
            
            # 예측한 토큰 얻기
            top1 = output.argmax(1)
            
            # 다음 입력은 정답이거나 예측한 토큰이다
            decoder_input = trg[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)
        
        return outputs
    
    def generate(self, src, max_len=50, sos_idx=1, eos_idx=2, src_lengths=None):
        """
        탐욕적 복호로 순차열을 만든다
        
        인수:
            src: 원본 순차열 (배치 크기, src_len)
            max_len: 만들 순차열의 최대 길이
            sos_idx: 순차열 시작 토큰의 색인
            eos_idx: 순차열 끝 토큰의 색인
            src_lengths: 원본 순차열의 실제 길이
            
        반환값:
            generated: 만들어진 순차열 (배치 크기, seq_len)
        """
        self.eval()
        
        with torch.no_grad():
            batch_size = src.shape[0]
            
            # 원본 부호화
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
                
                # 끝난 순차열 갱신
                finished |= (predicted.squeeze(1) == eos_idx)
                
                # 예측 담기
                generated.append(predicted)
                
                # 모든 순차열이 끝나면 멈추기
                if finished.all():
                    break
                
                # 다음 입력
                decoder_input = predicted
            
            # 모든 예측 이어 붙이기
            generated = torch.cat(generated, dim=1)
            
        return generated


class Seq2SeqAttention(nn.Module):
    """
    어텐션 장치가 있는 순차열 대 순차열 모델
    
    인수:
        encoder: 인코더 모듈
        decoder: 어텐션이 있는 디코더 모듈
        device: 돌릴 장치
        pad_idx: 덧댐 토큰의 색인
    """
    
    def __init__(self, encoder, decoder, device, pad_idx=0):
        super(Seq2SeqAttention, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.pad_idx = pad_idx
        
    def create_mask(self, src):
        """덧댐 토큰의 가림막을 만든다"""
        mask = (src != self.pad_idx)
        return mask
    
    def forward(self, src, trg, teacher_forcing_ratio=0.5, src_lengths=None):
        """
        어텐션이 있는 seq2seq 모델을 지나는 순전파
        
        인수:
            src: 원본 순차열 (배치 크기, src_len)
            trg: 표적 순차열 (배치 크기, trg_len)
            teacher_forcing_ratio: 교사 강요를 쓸 확률
            src_lengths: 원본 순차열의 실제 길이
            
        반환값:
            outputs: 시각마다의 예측 (배치 크기, trg_len, output_size)
            attentions: 시각마다의 어텐션 가중치 (배치 크기, trg_len, src_len)
        """
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_size
        
        # 출력과 어텐션 가중치를 담을 텐서
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        attentions = torch.zeros(batch_size, trg_len, src.shape[1]).to(self.device)
        
        # 원본 순차열의 가림막 만들기
        mask = self.create_mask(src)
        
        # 원본 순차열 부호화
        encoder_outputs, hidden, cell = self.encoder(src, src_lengths)
        
        # 디코더의 첫 입력은 <sos> 토큰이다
        decoder_input = trg[:, 0].unsqueeze(1)
        
        # 한 번에 토큰 하나씩 복호
        for t in range(1, trg_len):
            # 어텐션이 있는 디코더를 지나는 순전파
            output, hidden, cell, attention_weights = self.decoder(
                decoder_input, hidden, encoder_outputs, cell, mask
            )
            
            # 예측과 어텐션 가중치 담기
            outputs[:, t] = output
            attentions[:, t] = attention_weights
            
            # 교사 강요를 쓸지 정하기
            teacher_force = random.random() < teacher_forcing_ratio
            
            # 예측한 토큰 얻기
            top1 = output.argmax(1)
            
            # 다음 입력은 정답이거나 예측한 토큰이다
            decoder_input = trg[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)
        
        return outputs, attentions
    
    def generate(self, src, max_len=50, sos_idx=1, eos_idx=2, src_lengths=None):
        """
        어텐션과 함께 탐욕적 복호로 순차열을 만든다
        
        인수:
            src: 원본 순차열 (배치 크기, src_len)
            max_len: 만들 순차열의 최대 길이
            sos_idx: 순차열 시작 토큰의 색인
            eos_idx: 순차열 끝 토큰의 색인
            src_lengths: 원본 순차열의 실제 길이
            
        반환값:
            generated: 만들어진 순차열 (배치 크기, seq_len)
            all_attentions: 걸음마다의 어텐션 가중치 (배치 크기, seq_len, src_len)
        """
        self.eval()
        
        with torch.no_grad():
            batch_size = src.shape[0]
            
            # 가림막 만들기
            mask = self.create_mask(src)
            
            # 원본 부호화
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
                
                # 어텐션 가중치 담기
                all_attentions.append(attention_weights.unsqueeze(1))
                
                # 예측한 토큰 얻기
                predicted = output.argmax(1).unsqueeze(1)
                
                # 끝난 순차열 갱신
                finished |= (predicted.squeeze(1) == eos_idx)
                
                # 예측 담기
                generated.append(predicted)
                
                # 모든 순차열이 끝나면 멈추기
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
        빔 탐색으로 순차열을 만든다
        
        인수:
            src: 원본 순차열 (1, src_len) — 순차열 하나만
            beam_width: 빔의 수
            max_len: 만들 순차열의 최대 길이
            sos_idx: 순차열 시작 토큰의 색인
            eos_idx: 순차열 끝 토큰의 색인
            src_lengths: 원본 순차열의 실제 길이
            length_penalty: 길이 정규화 벌점 (alpha)
            
        반환값:
            best_sequence: 가장 좋은 순차열
            best_score: 가장 좋은 순차열의 점수
        """
        self.eval()
        
        with torch.no_grad():
            # 원본 부호화
            mask = self.create_mask(src)
            encoder_outputs, hidden, cell = self.encoder(src, src_lengths)
            
            # 빔 초기화
            # 빔마다: (순차열, 점수, 숨은 상태, 세포 상태)
            beams = [(torch.full((1, 1), sos_idx, dtype=torch.long).to(self.device), 
                     0.0, hidden, cell)]
            completed = []
            
            for step in range(max_len):
                candidates = []
                
                for sequence, score, h, c in beams:
                    # 순차열이 이미 끝났으면 건너뛰기
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
            
            # 가장 좋은 순차열 고르기
            if completed:
                best = max(completed, key=lambda x: x[1] / (len(x[0][0]) ** length_penalty))
                return best[0], best[1]
            else:
                return beams[0][0], beams[0][1]


if __name__ == "__main__":
    # 사용 예
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 초매개변수
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


class Seq2SeqDataset(Dataset):
    """
    Seq2Seq 학습을 위한 사용자 정의 데이터셋
    
    인수:
        src_data: 원본 순차열의 목록 (토큰 색인)
        trg_data: 표적 순차열의 목록 (토큰 색인)
        src_vocab: 원본 어휘
        trg_vocab: 표적 어휘
    """
    
    def __init__(self, src_data, trg_data):
        self.src_data = src_data
        self.trg_data = trg_data
        
    def __len__(self):
        return len(self.src_data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.src_data[idx]), torch.tensor(self.trg_data[idx])


def collate_fn(batch, pad_idx=0):
    """
    길이가 다른 순차열을 배치로 묶는 함수
    
    인수:
        batch: (src, trg) 쌍의 목록
        pad_idx: 덧댐 토큰의 색인
        
    반환값:
        src_batch: 덧댄 원본 순차열
        trg_batch: 덧댄 표적 순차열
        src_lengths: 원본 순차열의 실제 길이
        trg_lengths: 표적 순차열의 실제 길이
    """
    src_batch, trg_batch = zip(*batch)
    
    # 길이 얻기
    src_lengths = torch.tensor([len(s) for s in src_batch])
    trg_lengths = torch.tensor([len(t) for t in trg_batch])
    
    # 순차열 덧대기
    src_batch = nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
    trg_batch = nn.utils.rnn.pad_sequence(trg_batch, batch_first=True, padding_value=pad_idx)
    
    return src_batch, trg_batch, src_lengths, trg_lengths


class Seq2SeqTrainer:
    """
    Seq2Seq 모델을 위한 학습기 클래스
    
    인수:
        model: Seq2Seq 모델
        optimizer: 최적화기
        criterion: 로스 함수
        device: 학습에 쓸 장치
        pad_idx: 덧댐 토큰의 색인
        clip: 그래디언트를 자를 값
    """
    
    def __init__(self, model, optimizer, criterion, device, pad_idx=0, clip=1.0):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.pad_idx = pad_idx
        self.clip = clip
        
    def train_epoch(self, dataloader, teacher_forcing_ratio=0.5):
        """
        한 세대 학습시킨다
        
        인수:
            dataloader: 학습 데이터로더
            teacher_forcing_ratio: 교사 강요를 쓸 확률
            
        반환값:
            epoch_loss: 그 세대의 평균 로스
        """
        self.model.train()
        epoch_loss = 0
        
        for batch_idx, (src, trg, src_lengths, trg_lengths) in enumerate(dataloader):
            src = src.to(self.device)
            trg = trg.to(self.device)
            src_lengths = src_lengths.to(self.device)
            
            self.optimizer.zero_grad()
            
            # 순전파
            if hasattr(self.model, 'encoder'):
                # Seq2SeqAttention 모델
                output, _ = self.model(src, trg, teacher_forcing_ratio, src_lengths)
            else:
                output = self.model(src, trg, teacher_forcing_ratio, src_lengths)
            
            # 로스 계산을 위해 출력과 표적의 모양 바꾸기
            # output: (배치 크기, trg_len, output_dim)
            # trg: (배치 크기, trg_len)
            output_dim = output.shape[-1]
            
            # 표적의 첫 토큰(<sos>) 건너뛰기
            output = output[:, 1:].contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)
            
            # 로스 계산
            loss = self.criterion(output, trg)
            
            # 역전파
            loss.backward()
            
            # 경사를 자른다
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            
            # 파라메터 갱신
            self.optimizer.step()
            
            epoch_loss += loss.item()
        
        return epoch_loss / len(dataloader)
    
    def evaluate(self, dataloader):
        """
        모델을 평가한다
        
        인수:
            dataloader: 검증 데이터로더
            
        반환값:
            epoch_loss: 그 세대의 평균 로스
        """
        self.model.eval()
        epoch_loss = 0
        
        with torch.no_grad():
            for batch_idx, (src, trg, src_lengths, trg_lengths) in enumerate(dataloader):
                src = src.to(self.device)
                trg = trg.to(self.device)
                src_lengths = src_lengths.to(self.device)
                
                # 순전파 (평가 중에는 교사 강요 없음)
                if hasattr(self.model, 'encoder'):
                    output, _ = self.model(src, trg, teacher_forcing_ratio=0, src_lengths=src_lengths)
                else:
                    output = self.model(src, trg, teacher_forcing_ratio=0, src_lengths=src_lengths)
                
                # 로스 계산을 위해 모양 바꾸기
                output_dim = output.shape[-1]
                output = output[:, 1:].contiguous().view(-1, output_dim)
                trg = trg[:, 1:].contiguous().view(-1)
                
                # 로스 계산
                loss = self.criterion(output, trg)
                epoch_loss += loss.item()
        
        return epoch_loss / len(dataloader)
    
    def train(self, train_loader, val_loader, num_epochs, checkpoint_dir='checkpoints', 
              teacher_forcing_ratio=0.5, save_every=1):
        """
        여러 세대에 걸쳐 모델을 학습시킨다
        
        인수:
            train_loader: 학습 데이터 로더
            val_loader: 검증 데이터 로더
            num_epochs: 학습할 세대 수
            checkpoint_dir: 검사점을 저장할 디렉터리
            teacher_forcing_ratio: 처음의 교사 강요 비율
            save_every: N세대마다 검사점 저장
            
        반환값:
            train_losses: 학습 로스의 목록
            val_losses: 검증 로스의 목록
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True)
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # 학습
            train_loss = self.train_epoch(train_loader, teacher_forcing_ratio)
            
            # 평가한다
            val_loss = self.evaluate(val_loader)
            
            # 로스 담기
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            end_time = time.time()
            epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
            
            # 진행 상황 출력
            print(f'Epoch: {epoch+1:02} | Time: {int(epoch_mins)}m {int(epoch_secs)}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            print(f'\t Val. Loss: {val_loss:.3f} |  Val. PPL: {math.exp(val_loss):7.3f}')
            
            # 검사점 저장
            if (epoch + 1) % save_every == 0:
                checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt'
                self.save_checkpoint(checkpoint_path, epoch, train_loss, val_loss)
            
            # 최고 성능 모델 저장
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_checkpoint_path = checkpoint_dir / 'best_model.pt'
                self.save_checkpoint(best_checkpoint_path, epoch, train_loss, val_loss)
                print(f'\t[Saved Best Model]')
            
            # 교사 강요 비율 줄이기 (선택)
            teacher_forcing_ratio = max(0.5 * teacher_forcing_ratio, 0.1)
        
        return train_losses, val_losses
    
    def save_checkpoint(self, path, epoch, train_loss, val_loss):
        """모델 검사점을 저장한다"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, path)
    
    def load_checkpoint(self, path):
        """모델 검사점을 불러온다"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['train_loss'], checkpoint['val_loss']


def initialize_weights(model):
    """모델의 웨이트를 초기화한다"""
    for name, param in model.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


def count_parameters(model):
    """학습 가능한 파라메터를 센다"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 사용 예
    
    # 장치 지정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 초파라메터
    INPUT_DIM = 10000
    OUTPUT_DIM = 10000
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5
    PAD_IDX = 0
    
    # 모델 생성
    encoder = BasicEncoder(
        input_size=INPUT_DIM,
        embedding_dim=ENC_EMB_DIM,
        hidden_size=HID_DIM,
        num_layers=N_LAYERS,
        dropout=ENC_DROPOUT,
        bidirectional=True,
        rnn_type='LSTM'
    )
    
    decoder = AttentionDecoder(
        output_size=OUTPUT_DIM,
        embedding_dim=DEC_EMB_DIM,
        hidden_size=HID_DIM * 2,
        encoder_hidden_size=HID_DIM * 2,
        num_layers=N_LAYERS,
        dropout=DEC_DROPOUT,
        rnn_type='LSTM'
    )
    
    model = Seq2SeqAttention(encoder, decoder, device, PAD_IDX).to(device)
    
    # 웨이트 초기화
    initialize_weights(model)
    
    print(f'The model has {count_parameters(model):,} trainable parameters')
    
    # 최적화기와 로스 함수 만들기
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    
    # 학습기 만들기
    trainer = Seq2SeqTrainer(model, optimizer, criterion, device, PAD_IDX, clip=1.0)
    
    # 임시 데이터 만들기
    print("\nCreating dummy dataset...")
    num_samples = 1000
    src_data = [np.random.randint(3, INPUT_DIM, size=np.random.randint(10, 30)).tolist() 
                for _ in range(num_samples)]
    trg_data = [np.random.randint(3, OUTPUT_DIM, size=np.random.randint(10, 30)).tolist() 
                for _ in range(num_samples)]
    
    train_dataset = Seq2SeqDataset(src_data[:800], trg_data[:800])
    val_dataset = Seq2SeqDataset(src_data[800:], trg_data[800:])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, PAD_IDX)
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=32, 
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, PAD_IDX)
    )
    
    # 모델을 학습시킨다
    print("\nStarting training...")
    train_losses, val_losses = trainer.train(
        train_loader, 
        val_loader, 
        num_epochs=5,
        checkpoint_dir='checkpoints',
        teacher_forcing_ratio=0.5
    )
    
    print("\nTraining completed!")
```

## 2. 논의

collate_fn은 원문·타깃을 각각 배치 최대 길이로 패딩하고 길이를 반환한다. 로스는 타깃 첫 위치를 제외한 출력과 정답을 평탄화해 계산한다. PAD 제외는 trainer의 pad_idx 자체가 아니라 전달된 criterion의 ignore_index가 수행하므로 두 설정을 맞춰야 한다.

모델에 encoder 속성이 있으면 반환값을 `(output, _)`로 분해하지만, 기본 Seq2Seq도 encoder를 가지면서 텐서만 반환한다. 따라서 이 분기는 두 모델을 올바르게 구별하지 못한다. 반환 계약을 명시적으로 맞추어야 하며 배치 크기에 따라 텐서가 잘못 분해되는 경우도 주의한다.

검증은 teacher_forcing_ratio=0으로 이전 예측을 입력하면서 정답 위치별 교차 엔트로피를 계산한다. 이는 정답 문맥에 조건을 두는 일반적인 언어 모델 음의 로그 우도와 다르다. 따라서 표시된 exp(val_loss)를 표준 퍼플렉시티로 해석하면 안 된다. 학습 역시 티처 포싱 비율을 바꾸므로 에포크 간 로스의 입력 조건이 달라진다.

기록 로스는 배치 로스의 단순 평균이며 배치 크기와 유효 토큰 수가 다르면 전체 토큰 평균이 아니다. 데이터는 원문·타깃이 독립적인 무작위 ID이고 SOS·EOS도 넣지 않으므로 형식과 동작을 살펴보는 시연으로 읽는다.

체크포인트에는 모델·옵티마이저와 에포크·로스를 저장하지만 난수 상태, 사전·전처리와 티처 포싱 비율은 없다. 최고 검증 모델을 저장해도 마지막에 자동 복원하지는 않는다. load_checkpoint는 저장된 위치를 반환할 뿐 나머지 학습 설정까지 복원하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
매 배치의 그래디언트 초기화를 제거하면 어떤 변화가 생기는가? 의도적인 그래디언트 누적과 구분하라.

</div>

??? success "풀이"
    backward는 기존 grad에 값을 더하므로 매번 optimizer.step을 하면서 지우지 않으면 서로 다른 웨이트에서 계산한 과거 그래디언트까지 계속 남는다. 단순히 러닝레이트가 단계 수만큼 커진 것과 같지는 않다. 의도적인 누적은 정해진 배치 구간 동안 웨이트를 고정하고 로스 배율을 맞춘 뒤 한 번 갱신하고 초기화한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
이 검증 함수의 exp(loss)를 일반적인 퍼플렉시티라고 부르기 어려운 이유를 설명하라.

</div>

??? success "풀이"
    검증 입력이 이전 정답이 아니라 모델 예측이어서 정답 문맥의 로그 우도와 다르다. 또한 배치 평균을 다시 단순 평균해 유효 토큰별 평균이 아니다. 표준 퍼플렉시티에는 같은 토큰화·마스크에서 정답 문맥의 토큰 음의 로그 우도를 합산하고 유효 토큰 수로 나눈 뒤 지수화해야 한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
최고 체크포인트를 저장한 것과 학습을 정확히 재개할 수 있는 것은 어떻게 다른가?

</div>

??? success "풀이"
    선택한 모델의 웨이트를 복원하는 것과 같은 데이터 순서·난수 상태로 이후 업데이트를 재현하는 것은 다르다. 정확한 재개에는 옵티마이저 외에도 난수·로더 상태와 스케줄, 티처 포싱 비율 등 현재 학습 상태가 필요하다. 사전과 구조 설정도 함께 관리해야 한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 조기 종료 코드 조각에서 현재 Seq2SeqTrainer에 연결하기 위해 필요한 수정과 최고 웨이트 복원 단계를 설명하라.

</div>

??? success "풀이"
    아래 조각은 조기 종료의 제어 흐름 예시다. 실제 trainer의 검증 호출과 체크포인트 형식에 맞춰 연결해야 한다.
    ```python
    best_loss = float('inf')
    patience_counter = 0
    best_state = None
    for epoch in range(num_epochs):
        # ... 학습 단계 ...
        val_loss = evaluate(model, val_loader)
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
        if patience_counter >= 10:
            print(f'Early stopping at epoch {epoch}')
            model.load_state_dict(best_state)
            break
    ```
    `train`의 기존 루프에 최저 검증 로스와 연속 미개선 횟수를 유지하고, 개선할 때 실제 모델 상태를 별도로 저장한다. 중단 후 그 웨이트를 명시적으로 불러와야 한다. 검증을 반복해 선택한 결과는 독립 테스트 평가와 구분한다. 유효하지 않은 로스나 저장 전에 중단되는 경우도 처리한다.

## 정리하며

Seq2Seq 학습에서는 반환 형식, 타깃 이동과 PAD 로스 제외를 맞춘다. 이전 예측으로 입력을 만드는 검증과 정답 문맥의 우도 평가를 구분하고, 배치 로스 평균과 체크포인트 복원 범위를 명시해야 한다.
