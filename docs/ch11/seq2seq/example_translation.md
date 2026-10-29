# 완전한 예제

영어·프랑스어의 작은 병렬 문장 목록을 사용해 전처리, 모델 구성과 학습·추론 도구를 연결하는 예제다. 이 페이지는 다른 모듈에 의존하므로 전체 파이프라인의 입력·반환 계약을 함께 확인해야 한다.

## 1. 코드

```python
"""
완전한 예제: 영어에서 프랑스어로 번역하기
데이터 준비부터 추론까지의 전 과정을 보인다
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# ========================================================================
# 메인
# ========================================================================

import re
from collections import Counter
import pickle
from pathlib import Path
import unicodedata
import torch.nn.functional as F
import random
from torch.utils.data import DataLoader, Dataset
import time
import math

# --- data_preprocessing ---
"""
Seq2Seq 모델을 위한 데이터 전처리 도구
토큰화, 어휘 만들기, 데이터 적재를 담고 있다
"""


# ========================================================================
# 메인
# ========================================================================


class Tokenizer:
    """
    텍스트 전처리를 위한 간단한 토큰 나누개
    """
    
    def __init__(self, lower=True, remove_punct=False):
        self.lower = lower
        self.remove_punct = remove_punct
    
    def tokenize(self, text):
        """텍스트를 토큰으로 나눈다"""
        if self.lower:
            text = text.lower()
        
        if self.remove_punct:
            # 문장 부호 없애기
            text = re.sub(r'[^\w\s]', '', text)
        else:
            # 문장 부호를 공백으로 떼기
            text = re.sub(r'([.,!?;:])', r' \1 ', text)
        
        # 공백으로 쪼개고 빈 문자열 걸러 내기
        tokens = text.split()
        tokens = [t for t in tokens if t]
        
        return tokens
    
    def detokenize(self, tokens):
        """토큰을 다시 텍스트로 바꾼다"""
        text = ' '.join(tokens)
        # 문장 부호 앞의 공백 없애기
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        return text


class Vocabulary:
    """
    토큰과 색인의 대응을 관리하는 어휘 클래스
    
    인수:
        max_size: 어휘의 최대 크기 (None이면 제한 없음)
        min_freq: 토큰이 들어가려면 넘어야 할 최소 빈도
        special_tokens: 특수 토큰의 목록
    """
    
    def __init__(self, max_size=None, min_freq=1, 
                 special_tokens=['<pad>', '<sos>', '<eos>', '<unk>']):
        self.max_size = max_size
        self.min_freq = min_freq
        self.special_tokens = special_tokens
        
        # 토큰을 색인으로 잇기
        self.token2idx = {}
        self.idx2token = {}
        
        # 특수 토큰 더하기
        for idx, token in enumerate(special_tokens):
            self.token2idx[token] = idx
            self.idx2token[idx] = token
        
        self.pad_idx = self.token2idx.get('<pad>', 0)
        self.sos_idx = self.token2idx.get('<sos>', 1)
        self.eos_idx = self.token2idx.get('<eos>', 2)
        self.unk_idx = self.token2idx.get('<unk>', 3)
    
    def build_vocab(self, texts, tokenizer=None):
        """
        텍스트에서 어휘를 만든다
        
        인수:
            texts: 텍스트 문자열의 목록 또는 토큰 목록의 목록
            tokenizer: 토큰 나누개 함수 (선택)
        """
        # 토큰의 빈도 세기
        counter = Counter()
        
        for text in texts:
            if tokenizer is not None:
                tokens = tokenizer(text)
            elif isinstance(text, str):
                tokens = text.split()
            else:
                tokens = text
            
            counter.update(tokens)
        
        # 빈도로 걸러 내기
        tokens = [token for token, freq in counter.items() if freq >= self.min_freq]
        
        # 빈도로 정렬 (가장 흔한 것 먼저)
        tokens = sorted(tokens, key=lambda t: counter[t], reverse=True)
        
        # 어휘 크기 제한
        if self.max_size is not None:
            tokens = tokens[:self.max_size - len(self.special_tokens)]
        
        # 어휘에 토큰 더하기
        for token in tokens:
            if token not in self.token2idx:
                idx = len(self.token2idx)
                self.token2idx[token] = idx
                self.idx2token[idx] = token
    
    def encode(self, tokens):
        """토큰을 색인으로 바꾼다"""
        if isinstance(tokens, str):
            tokens = tokens.split()
        return [self.token2idx.get(token, self.unk_idx) for token in tokens]
    
    def decode(self, indices, skip_special=True):
        """색인을 토큰으로 바꾼다"""
        tokens = []
        for idx in indices:
            if skip_special and idx in [self.pad_idx, self.sos_idx, self.eos_idx]:
                if idx == self.eos_idx:
                    break
                continue
            tokens.append(self.idx2token.get(idx, '<unk>'))
        return tokens
    
    def __len__(self):
        return len(self.token2idx)
    
    def save(self, path):
        """어휘를 파일에 저장한다"""
        with open(path, 'wb') as f:
            pickle.dump({
                'token2idx': self.token2idx,
                'idx2token': self.idx2token,
                'max_size': self.max_size,
                'min_freq': self.min_freq,
                'special_tokens': self.special_tokens
            }, f)
    
    @classmethod
    def load(cls, path):
        """파일에서 어휘를 불러온다"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        vocab = cls(
            max_size=data['max_size'],
            min_freq=data['min_freq'],
            special_tokens=data['special_tokens']
        )
        vocab.token2idx = data['token2idx']
        vocab.idx2token = data['idx2token']
        
        return vocab


class ParallelDataset:
    """
    순차열 대 순차열 과제를 위한 병렬 데이터셋
    
    인수:
        src_texts: 원본 텍스트의 목록
        trg_texts: 표적 텍스트의 목록
        src_vocab: 원본 어휘
        trg_vocab: 표적 어휘
        src_tokenizer: 원본 토큰 나누개
        trg_tokenizer: 표적 토큰 나누개
        max_len: 순차열의 최대 길이
    """
    
    def __init__(self, src_texts, trg_texts, src_vocab, trg_vocab,
                 src_tokenizer=None, trg_tokenizer=None, max_len=None):
        self.src_texts = src_texts
        self.trg_texts = trg_texts
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.src_tokenizer = src_tokenizer or Tokenizer()
        self.trg_tokenizer = trg_tokenizer or Tokenizer()
        self.max_len = max_len
    
    def process_pair(self, src_text, trg_text):
        """원본-표적 쌍을 처리한다"""
        # 토큰으로 나누기
        src_tokens = self.src_tokenizer.tokenize(src_text)
        trg_tokens = self.trg_tokenizer.tokenize(trg_text)
        
        # 필요하면 잘라 내기
        if self.max_len is not None:
            src_tokens = src_tokens[:self.max_len]
            trg_tokens = trg_tokens[:self.max_len]
        
        # 부호화
        src_indices = self.src_vocab.encode(src_tokens)
        trg_indices = [self.trg_vocab.sos_idx] + self.trg_vocab.encode(trg_tokens) + [self.trg_vocab.eos_idx]
        
        return src_indices, trg_indices
    
    def __len__(self):
        return len(self.src_texts)
    
    def __getitem__(self, idx):
        return self.process_pair(self.src_texts[idx], self.trg_texts[idx])


def load_parallel_data(src_path, trg_path, max_samples=None):
    """
    파일에서 병렬 데이터를 불러온다
    
    인수:
        src_path: 원본 파일의 경로
        trg_path: 표적 파일의 경로
        max_samples: 불러올 표본의 최대 수
        
    반환값:
        src_texts: 원본 텍스트의 목록
        trg_texts: 표적 텍스트의 목록
    """
    with open(src_path, 'r', encoding='utf-8') as f:
        src_texts = [line.strip() for line in f]
    
    with open(trg_path, 'r', encoding='utf-8') as f:
        trg_texts = [line.strip() for line in f]
    
    # 길이를 같게 맞추기
    assert len(src_texts) == len(trg_texts), "Source and target files must have same length"
    
    # 지정되었으면 표본 수 제한
    if max_samples is not None:
        src_texts = src_texts[:max_samples]
        trg_texts = trg_texts[:max_samples]
    
    return src_texts, trg_texts


def normalize_text(text):
    """
    텍스트를 정규화한다 (유니코드 정규화 등)
    
    인수:
        text: 입력 텍스트
        
    반환값:
        normalized_text: 정규화된 텍스트
    """
    # 유니코드 정규화
    text = unicodedata.normalize('NFD', text)
    
    # 강세 부호 없애기
    text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
    
    return text


def split_data(data, train_ratio=0.8, val_ratio=0.1):
    """
    데이터를 학습·검증·시험 집합으로 나눈다
    
    인수:
        data: 데이터의 목록이나 쌍
        train_ratio: 학습 데이터의 비율
        val_ratio: 검증 데이터의 비율
        
    반환값:
        train_data, val_data, test_data: 나뉜 데이터셋
    """
    if isinstance(data, tuple):
        # 여러 데이터셋 (원본과 표적 따위)
        total_len = len(data[0])
        train_len = int(total_len * train_ratio)
        val_len = int(total_len * val_ratio)
        
        train_data = tuple(d[:train_len] for d in data)
        val_data = tuple(d[train_len:train_len + val_len] for d in data)
        test_data = tuple(d[train_len + val_len:] for d in data)
    else:
        # 데이터셋 하나
        total_len = len(data)
        train_len = int(total_len * train_ratio)
        val_len = int(total_len * val_ratio)
        
        train_data = data[:train_len]
        val_data = data[train_len:train_len + val_len]
        test_data = data[train_len + val_len:]
    
    return train_data, val_data, test_data


if __name__ == "__main__":
    # 사용 예
    print("Data Preprocessing Example")
    print("-" * 50)
    
    # 예제 데이터
    src_texts = [
        "Hello, how are you?",
        "I am doing well, thank you.",
        "What is your name?",
        "My name is Claude.",
        "Nice to meet you!"
    ]
    
    trg_texts = [
        "Bonjour, comment allez-vous?",
        "Je vais bien, merci.",
        "Quel est votre nom?",
        "Je m'appelle Claude.",
        "Enchanté de vous rencontrer!"
    ]
    
    # 토큰 나누개 만들기
    tokenizer = Tokenizer(lower=True, remove_punct=False)
    
    # 어휘 만들기
    print("\nBuilding vocabularies...")
    src_vocab = Vocabulary(max_size=1000, min_freq=1)
    src_vocab.build_vocab(src_texts, tokenizer.tokenize)
    
    trg_vocab = Vocabulary(max_size=1000, min_freq=1)
    trg_vocab.build_vocab(trg_texts, tokenizer.tokenize)
    
    print(f"Source vocabulary size: {len(src_vocab)}")
    print(f"Target vocabulary size: {len(trg_vocab)}")
    
    # 부호화와 복호 시험
    print("\nTesting encoding/decoding...")
    test_text = "Hello, how are you?"
    tokens = tokenizer.tokenize(test_text)
    indices = src_vocab.encode(tokens)
    decoded = src_vocab.decode(indices, skip_special=False)
    
    print(f"Original: {test_text}")
    print(f"Tokens: {tokens}")
    print(f"Indices: {indices}")
    print(f"Decoded: {decoded}")
    
    # 데이터셋 생성
    print("\nCreating dataset...")
    dataset = ParallelDataset(
        src_texts, trg_texts, src_vocab, trg_vocab,
        src_tokenizer=tokenizer, trg_tokenizer=tokenizer
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # 시험 데이터셋
    src_indices, trg_indices = dataset[0]
    print(f"\nSample data:")
    print(f"Source: {src_texts[0]}")
    print(f"Source indices: {src_indices}")
    print(f"Target: {trg_texts[0]}")
    print(f"Target indices: {trg_indices}")
    
    # 어휘 저장
    print("\nSaving vocabularies...")
    src_vocab.save('src_vocab.pkl')
    trg_vocab.save('trg_vocab.pkl')
    print("Vocabularies saved!")
    
    # 어휘 불러오기
    print("\nLoading vocabularies...")
    loaded_src_vocab = Vocabulary.load('src_vocab.pkl')
    loaded_trg_vocab = Vocabulary.load('trg_vocab.pkl')
    print(f"Loaded source vocabulary size: {len(loaded_src_vocab)}")
    print(f"Loaded target vocabulary size: {len(loaded_trg_vocab)}")


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


# --- train ---
"""
Seq2Seq 모델을 위한 학습 스크립트
학습 반복문, 평가, 검사점 저장을 담고 있다
"""


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
        criterion: 손실 함수
        device: 학습에 쓸 장치
        pad_idx: 덧댐 토큰의 색인
        clip: 기울기를 자를 값
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
            epoch_loss: 그 세대의 평균 손실
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
            
            # 손실 계산을 위해 출력과 표적의 모양 바꾸기
            # output: (배치 크기, trg_len, output_dim)
            # trg: (배치 크기, trg_len)
            output_dim = output.shape[-1]
            
            # 표적의 첫 토큰(<sos>) 건너뛰기
            output = output[:, 1:].contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)
            
            # 손실 계산
            loss = self.criterion(output, trg)
            
            # 역전파
            loss.backward()
            
            # 경사를 자른다
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            
            # 매개변수 갱신
            self.optimizer.step()
            
            epoch_loss += loss.item()
        
        return epoch_loss / len(dataloader)
    
    def evaluate(self, dataloader):
        """
        모델을 평가한다
        
        인수:
            dataloader: 검증 데이터로더
            
        반환값:
            epoch_loss: 그 세대의 평균 손실
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
                
                # 손실 계산을 위해 모양 바꾸기
                output_dim = output.shape[-1]
                output = output[:, 1:].contiguous().view(-1, output_dim)
                trg = trg[:, 1:].contiguous().view(-1)
                
                # 손실 계산
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
            train_losses: 학습 손실의 목록
            val_losses: 검증 손실의 목록
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
            
            # 손실 담기
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
    """모델의 가중치를 초기화한다"""
    for name, param in model.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


def count_parameters(model):
    """학습 가능한 매개변수를 센다"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 사용 예
    
    # 장치 지정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 초매개변수
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
    
    # 가중치 초기화
    initialize_weights(model)
    
    print(f'The model has {count_parameters(model):,} trainable parameters')
    
    # 최적화기와 손실 함수 만들기
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


# --- inference ---
"""
Seq2Seq 모델을 위한 추론 스크립트
탐욕적 복호, 빔 탐색, 번역 도구를 담고 있다
"""


# ========================================================================
# 메인
# ========================================================================


class Seq2SeqInference:
    """
    Seq2Seq 모델을 위한 추론 도구
    
    인수:
        model: 학습된 Seq2Seq 모델
        src_vocab: 원본 어휘 (토큰을 색인으로)
        trg_vocab: 표적 어휘 (토큰을 색인으로)
        device: 추론을 돌릴 장치
        sos_idx: 순차열 시작 토큰의 색인
        eos_idx: 순차열 끝 토큰의 색인
        pad_idx: 덧댐 토큰의 색인
    """
    
    def __init__(self, model, src_vocab, trg_vocab, device, sos_idx=1, eos_idx=2, pad_idx=0):
        self.model = model
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.device = device
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        
        # 거꾸로 된 어휘 만들기 (색인에서 토큰으로)
        self.idx_to_src = {idx: token for token, idx in src_vocab.items()}
        self.idx_to_trg = {idx: token for token, idx in trg_vocab.items()}
    
    def tokenize_source(self, text):
        """
        원본 텍스트를 토큰으로 나눈다
        
        인수:
            text: 원본 텍스트 문자열이나 토큰의 목록
            
        반환값:
            tokens: 토큰 색인의 목록
        """
        if isinstance(text, str):
            tokens = text.lower().split()
        else:
            tokens = text
        
        # 색인으로 바꾸기
        indices = [self.src_vocab.get(token, self.src_vocab.get('<unk>', 3)) 
                   for token in tokens]
        
        return indices
    
    def detokenize_target(self, indices):
        """
        표적의 색인을 다시 텍스트로 바꾼다
        
        인수:
            indices: 토큰 색인의 목록이나 텐서
            
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
        탐욕적 복호로 번역을 만든다
        
        인수:
            src_text: 원본 텍스트 문자열이나 토큰의 목록
            max_len: 만들 순차열의 최대 길이
            
        반환값:
            translation: 번역된 텍스트
            attention_weights: 어텐션 가중치 (있을 때)
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
            max_len: 만들 순차열의 최대 길이
            length_penalty: 길이 정규화 벌점
            
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
            method: 복호 방법 ('greedy' 또는 'beam')
            **kwargs: 복호에 쓸 추가 인자
            
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
            
            # 탐욕적 복호
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
        어텐션 가중치 그려 보기
        
        인수:
            src_text: 원본 텍스트
            translation: 번역된 텍스트
            attention_weights: 어텐션 가중치 텐서
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
        
        # 짧음 벌점
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
        """짧음 벌점을 계산한다"""
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
    
    # 임시 낱말 몇 개 더하기
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


def main():
    """주된 학습과 평가의 흐름"""
    
    print("=" * 60)
    print("Seq2Seq Translation Example: English to French")
    print("=" * 60)
    
    # 재현성을 위한 난수 시드 설정
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    # 장치 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # ========== 1단계: 데이터 준비 ==========
    print("\n" + "=" * 60)
    print("Step 1: Preparing Data")
    print("=" * 60)
    
    # 예제 영어-프랑스어 병렬 말뭉치
    en_texts = [
        "Hello, how are you?",
        "I am fine, thank you.",
        "What is your name?",
        "My name is John.",
        "Nice to meet you.",
        "Where are you from?",
        "I am from Paris.",
        "Do you speak English?",
        "Yes, I speak English.",
        "What time is it?",
        "It is three o'clock.",
        "I like to read books.",
        "She is a teacher.",
        "He works in a hospital.",
        "We are students.",
        "They are learning French.",
        "I have a cat.",
        "The weather is nice today.",
        "Can you help me?",
        "Of course, I can help you.",
        "Thank you very much.",
        "You are welcome.",
        "Good morning.",
        "Good night.",
        "See you later.",
        "How old are you?",
        "I am twenty years old.",
        "What do you do?",
        "I am a doctor.",
        "Where do you live?",
    ]
    
    fr_texts = [
        "Bonjour, comment allez-vous?",
        "Je vais bien, merci.",
        "Quel est votre nom?",
        "Je m'appelle John.",
        "Enchanté.",
        "D'où venez-vous?",
        "Je viens de Paris.",
        "Parlez-vous anglais?",
        "Oui, je parle anglais.",
        "Quelle heure est-il?",
        "Il est trois heures.",
        "J'aime lire des livres.",
        "Elle est enseignante.",
        "Il travaille dans un hôpital.",
        "Nous sommes étudiants.",
        "Ils apprennent le français.",
        "J'ai un chat.",
        "Il fait beau aujourd'hui.",
        "Pouvez-vous m'aider?",
        "Bien sûr, je peux vous aider.",
        "Merci beaucoup.",
        "De rien.",
        "Bonjour.",
        "Bonne nuit.",
        "À plus tard.",
        "Quel âge avez-vous?",
        "J'ai vingt ans.",
        "Que faites-vous?",
        "Je suis médecin.",
        "Où habitez-vous?",
    ]
    
    print(f"Total samples: {len(en_texts)}")
    
    # 데이터 나누기
    (train_en, train_fr), (val_en, val_fr), (test_en, test_fr) = split_data(
        (en_texts, fr_texts), train_ratio=0.7, val_ratio=0.15
    )
    
    print(f"Training samples: {len(train_en)}")
    print(f"Validation samples: {len(val_en)}")
    print(f"Test samples: {len(test_en)}")
    
    # ========== 2단계: 어휘 만들기 ==========
    print("\n" + "=" * 60)
    print("Step 2: Building Vocabularies")
    print("=" * 60)
    
    # 토큰 나누개 만들기
    tokenizer = Tokenizer(lower=True, remove_punct=False)
    
    # 어휘 만들기
    en_vocab = Vocabulary(max_size=1000, min_freq=1)
    en_vocab.build_vocab(train_en, tokenizer.tokenize)
    
    fr_vocab = Vocabulary(max_size=1000, min_freq=1)
    fr_vocab.build_vocab(train_fr, tokenizer.tokenize)
    
    print(f"English vocabulary size: {len(en_vocab)}")
    print(f"French vocabulary size: {len(fr_vocab)}")
    
    # ========== 3단계: 데이터셋과 데이터로더 만들기 ==========
    print("\n" + "=" * 60)
    print("Step 3: Creating Datasets")
    print("=" * 60)
    
    train_dataset = ParallelDataset(
        train_en, train_fr, en_vocab, fr_vocab,
        src_tokenizer=tokenizer, trg_tokenizer=tokenizer
    )
    
    val_dataset = ParallelDataset(
        val_en, val_fr, en_vocab, fr_vocab,
        src_tokenizer=tokenizer, trg_tokenizer=tokenizer
    )
    
    test_dataset = ParallelDataset(
        test_en, test_fr, en_vocab, fr_vocab,
        src_tokenizer=tokenizer, trg_tokenizer=tokenizer
    )
    
    BATCH_SIZE = 4
    PAD_IDX = en_vocab.pad_idx
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, PAD_IDX)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, PAD_IDX)
    )
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # ========== 4단계: 모델 만들기 ==========
    print("\n" + "=" * 60)
    print("Step 4: Creating Model")
    print("=" * 60)
    
    # 모델의 초파라메터
    ENC_EMB_DIM = 128
    DEC_EMB_DIM = 128
    HID_DIM = 256
    N_LAYERS = 2
    ENC_DROPOUT = 0.3
    DEC_DROPOUT = 0.3
    
    # 인코더 만들기
    encoder = BasicEncoder(
        input_size=len(en_vocab),
        embedding_dim=ENC_EMB_DIM,
        hidden_size=HID_DIM,
        num_layers=N_LAYERS,
        dropout=ENC_DROPOUT,
        bidirectional=True,
        rnn_type='LSTM'
    )
    
    # 디코더 만들기
    decoder = AttentionDecoder(
        output_size=len(fr_vocab),
        embedding_dim=DEC_EMB_DIM,
        hidden_size=HID_DIM * 2,  # 인코더가 양방향이라 2를 곱한다
        encoder_hidden_size=HID_DIM * 2,
        num_layers=N_LAYERS,
        dropout=DEC_DROPOUT,
        rnn_type='LSTM'
    )
    
    # seq2seq 모델 만들기
    model = Seq2SeqAttention(encoder, decoder, device, PAD_IDX).to(device)
    
    # 웨이트 초기화
    initialize_weights(model)
    
    print(f"Model created with {count_parameters(model):,} trainable parameters")
    
    # ========== 5단계: 모델 학습시키기 ==========
    print("\n" + "=" * 60)
    print("Step 5: Training Model")
    print("=" * 60)
    
    # 학습 설정
    LEARNING_RATE = 0.001
    N_EPOCHS = 20
    CLIP = 1.0
    
    # 최적화기와 로스 함수 만들기
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    
    # 학습기 만들기
    trainer = Seq2SeqTrainer(model, optimizer, criterion, device, PAD_IDX, CLIP)
    
    # 모델을 학습시킨다
    train_losses, val_losses = trainer.train(
        train_loader,
        val_loader,
        num_epochs=N_EPOCHS,
        checkpoint_dir='checkpoints',
        teacher_forcing_ratio=0.5,
        save_every=5
    )
    
    # ========== 6단계: 모델 평가하기 ==========
    print("\n" + "=" * 60)
    print("Step 6: Evaluating Model")
    print("=" * 60)
    
    # 추론 객체 만들기
    en_vocab_dict = en_vocab.token2idx
    fr_vocab_dict = fr_vocab.token2idx
    
    inference = Seq2SeqInference(
        model, en_vocab_dict, fr_vocab_dict, device,
        sos_idx=fr_vocab.sos_idx,
        eos_idx=fr_vocab.eos_idx,
        pad_idx=PAD_IDX
    )
    
    # 번역 시험
    print("\nTest Translations:")
    print("-" * 60)
    
    bleu_scores = []
    
    for i, (en_text, fr_text) in enumerate(zip(test_en, test_fr)):
        translation, _ = inference.greedy_decode(en_text, max_len=50)
        
        # BLEU 점수 계산
        bleu = BLEU.compute_bleu(fr_text, translation)
        bleu_scores.append(bleu)
        
        print(f"\nExample {i+1}:")
        print(f"English:    {en_text}")
        print(f"French:     {fr_text}")
        print(f"Predicted:  {translation}")
        print(f"BLEU:       {bleu:.4f}")
    
    avg_bleu = np.mean(bleu_scores)
    print(f"\nAverage BLEU Score: {avg_bleu:.4f}")
    
    # ========== 7단계: 대화식 번역 ==========
    print("\n" + "=" * 60)
    print("Step 7: Interactive Translation")
    print("=" * 60)
    
    # 번역 예제
    test_sentences = [
        "Hello, my friend.",
        "How are you today?",
        "I am learning French.",
        "What is the time?"
    ]
    
    print("\nAdditional Examples:")
    print("-" * 60)
    
    for sent in test_sentences:
        translation, _ = inference.greedy_decode(sent)
        print(f"EN: {sent}")
        print(f"FR: {translation}\n")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

## 2. 논의

30개 문장을 앞에서부터 21개 학습, 4개 검증, 5개 테스트로 나눈다. 어휘는 학습 구간에서만 만들므로 검증·테스트에는 OOV가 생길 수 있다. 매우 작은 자료의 결과를 일반 번역 성능으로 해석할 수 없다. torch와 NumPy 시드는 설정하지만 티처 포싱에서 쓰는 Python random은 설정하지 않는다.

현재 ParallelDataset은 정수 목록을 반환하지만 collate_fn은 pad_sequence에 텐서 목록을 전달하는 것을 기대한다. 따라서 이 두 모듈을 그대로 연결하면 배치 구성 단계에서 맞지 않는다. 데이터셋이나 collate 단계에서 long 텐서로 변환하는 계약이 필요하다.

양방향 인코더의 최종 상태는 2H로 이어 붙여지므로 디코더도 그 피처 크기에 맞춰 설정한다. 학습은 검증 최고 체크포인트를 저장하지만 이 예제는 이를 다시 불러오지 않고 마지막 모델로 테스트한다. 최고 검증 모델의 성능으로 보고해서는 안 된다.

학습 토큰화는 구두점을 분리하지만 추론은 소문자·공백만 사용해 불일치한다. BLEU 도구에는 n-그램 분모 오류가 있고, 참조·출력의 대소문자·구두점 처리도 맞추지 않는다. 문장 BLEU를 평균한 값은 코퍼스 BLEU와 다르다. 따라서 이 코드에 보고되는 점수를 표준 번역 성능으로 사용하면 안 된다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
30개 문장을 이 예제의 비율로 나누었을 때 크기와 분할 방식을 설명하라.

</div>

??? success "풀이"
    학습 21개, 검증 4개, 테스트 5개다. 비율에 따른 정수 절삭을 사용하고 데이터를 섞지 않는다. 작은 수와 문장 순서 때문에 분포 차이가 클 수 있으며 학습 사전에 없는 단어도 발생할 수 있다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
ParallelDataset과 collate_fn 사이의 불일치를 설명하라.

</div>

??? success "풀이"
    데이터셋은 ID의 파이썬 목록을 반환하고 pad_sequence는 텐서를 요구한다. 배치 패딩 전에 long 텐서로 바꾸어야 하며 빈 입력과 PAD ID도 검사한다. 모델 구조가 맞아도 데이터 계약이 다르면 학습까지 도달하지 못한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
최종 번역 평가 전에 점검해야 할 세 항목을 제시하라.

</div>

??? success "풀이"
    평가할 체크포인트가 선택한 상태인지 확인하고, 학습·추론의 토큰화를 맞춘다. 표준 BLEU의 n-그램 발생 수·토큰화와 코퍼스 집계 방식을 사용한다. 현재 작은 예제의 점수로 일반화나 다른 모델 대비 우열을 확정하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래의 테스트 코드 조각을 검토하라. 정의되지 않은 클래스와 잘못된 이름, 입력 모양 가정을 찾아 이 페이지의 실제 함수에 맞는 점검 항목을 제안하라.

</div>

??? success "풀이"
    아래 조각은 실제 실행 가능한 테스트가 아니다. 함수 이름과 모델 이름부터 이 예제에 맞게 작성해야 한다.
    ```python
    def test_complete example():
        model = Complete Example(...)
        # 보통의 입력
        assert model(normal_input).shape == expected_shape
        # 원소가 하나인 배치
        assert model(single_input).shape == (1, ...)
        # 큰 값 (넘침을 확인한다)
        out = model(torch.ones(...) * 1000)
        assert torch.isfinite(out).all()
        # 경사의 흐름
        out = model(normal_input)
        out.sum().backward()
        for p in model.parameters():
            assert p.grad is not None
    ```
    이 페이지는 완성된 사용자 정의 모델 클래스를 제공하지 않는다. 실제 함수의 입력·출력 모양, 허용 자료형, 유한한 값과 경사의 유무를 기준으로 검사한다. 빈 입력을 처리하는 방식은 연산마다 다르며, 무조건 정상적인 로스나 경사가 나와야 한다고 가정하지 않는다.

## 정리하며

번역 파이프라인에서는 데이터 목록·배치 텐서, 토큰화·사전, 체크포인트와 지표의 규약이 모두 맞아야 한다. 이 예제의 연결상 제약을 해소하기 전에는 완성된 평가 결과로 해석할 수 없다.
