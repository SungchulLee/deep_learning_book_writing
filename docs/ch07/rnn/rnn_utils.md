# RNN 유틸리티

rnn_utils.py RNN 실습을 위한 종합 유틸리티 모듈

순차열 모형은 시간적이고 순서가 있는 데이터를 다루는 데 바탕이 된다. 이 구현은 순환 신경망의 핵심 착상을 다루며, 순환 계산과 학습된 표현이 시각 사이의 의존을 어떻게 붙잡는지 보인다.

## 코드

```python
"""
rnn_utils.py
============
RNN 실습을 위한 종합 유틸리티 모듈

다음을 위한 공통 함수를 준다.
- 인자 구문 분석
- 데이터 적재와 전처리
- RNN 모델 구조
- 학습과 평가
- 시각화
- 텍스트 처리 도구

지은이: PyTorch RNN 실습
날짜: 2025년 11월
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random
from torch.utils.data import Dataset, DataLoader

# =============================================================================
# 설정과 준비
# =============================================================================

def parse_args():
    """RNN 학습을 위한 명령줄 인자 구문 분석"""
    parser = argparse.ArgumentParser(description='PyTorch RNN Training')
    
    # 학습 하이퍼파라미터
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden-size', type=int, default=128)
    parser.add_argument('--num-layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.5)
    
    # 순차열 매개변수
    parser.add_argument('--sequence-length', type=int, default=50)
    parser.add_argument('--embedding-dim', type=int, default=100)
    
    # 시스템 설정
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--log-interval', type=int, default=100)
    
    # 모델 저장
    parser.add_argument('--save-model', action='store_true')
    parser.add_argument('--path', type=str, default='./model.pth')
    parser.add_argument('--eval-only', action='store_true')
    
    args = parser.parse_args()
    
    # 장치 지정
    if args.device == 'auto':
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device(args.device)
    
    return args

def set_seed(seed=42):
    """재현성을 위해 무작위 씨앗 정하기"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =============================================================================
# RNN 모델 구조
# =============================================================================

class SimpleRNN(nn.Module):
    """순차열 분류를 위한 기본 RNN"""
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden=None):
        # x: (배치, seq_len, input_size)
        out, hidden = self.rnn(x, hidden)
        # 마지막 출력 가져오기
        out = self.fc(out[:, -1, :])
        return out, hidden

class LSTMClassifier(nn.Module):
    """순차열 분류를 위한 LSTM"""
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size, 
                 num_layers=1, dropout=0.5):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x: (배치, seq_len)
        embedded = self.embedding(x)  # (배치, seq_len, embedding_dim)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # 마지막 숨은 상태 쓰기
        out = self.dropout(hidden[-1])
        out = self.fc(out)
        return out

class GRUPredictor(nn.Module):
    """순차열 예측(시계열 따위)을 위한 GRU"""
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.2):
        super(GRUPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x: (배치, seq_len, input_size)
        gru_out, hidden = self.gru(x)
        # 마지막 출력 쓰기
        out = self.fc(gru_out[:, -1, :])
        return out

class BiLSTM(nn.Module):
    """양방향 LSTM"""
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size,
                 num_layers=1, dropout=0.5):
        super(BiLSTM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers,
                           batch_first=True, bidirectional=True,
                           dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        # 양방향이라 2를 곱한다
        self.fc = nn.Linear(hidden_size * 2, output_size)
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # 마지막 순방향과 역방향 숨은 상태 이어 붙이기
        hidden_fwd = hidden[-2]
        hidden_bwd = hidden[-1]
        hidden_cat = torch.cat([hidden_fwd, hidden_bwd], dim=1)
        out = self.dropout(hidden_cat)
        out = self.fc(out)
        return out

# =============================================================================
# 학습과 평가
# =============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device, clip=None):
    """한 에포크 동안 학습한다"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        
        loss = criterion(output, target)
        loss.backward()
        
        # 기울기 폭발을 막으려고 기울기 자르기
        if clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        total_loss += loss.item()
        
        # 분류의 정확도 계산
        if output.dim() > 1 and output.size(1) > 1:
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total if total > 0 else 0
    
    return avg_loss, accuracy

def evaluate(model, dataloader, criterion, device):
    """모델 평가"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            loss = criterion(output, target)
            total_loss += loss.item()
            
            if output.dim() > 1 and output.size(1) > 1:
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total if total > 0 else 0
    
    return avg_loss, accuracy

# =============================================================================
# 텍스트 처리 도구
# =============================================================================

class Vocabulary:
    """텍스트 데이터의 어휘를 만들고 관리하기"""
    def __init__(self):
        self.word2idx = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
        self.idx2word = {0: '<PAD>', 1: '<UNK>', 2: '<SOS>', 3: '<EOS>'}
        self.word_count = {}
        self.n_words = 4
    
    def add_sentence(self, sentence):
        for word in sentence.split():
            self.add_word(word)
    
    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.n_words
            self.idx2word[self.n_words] = word
            self.word_count[word] = 1
            self.n_words += 1
        else:
            self.word_count[word] += 1
    
    def __len__(self):
        return self.n_words

def tokenize(text):
    """간단한 토큰 나누개"""
    return text.lower().split()

def text_to_sequence(text, vocab, max_length=None):
    """텍스트를 색인의 순차열로 바꾸기"""
    tokens = tokenize(text)
    sequence = [vocab.word2idx.get(word, vocab.word2idx['<UNK>']) for word in tokens]
    
    if max_length:
        if len(sequence) < max_length:
            sequence += [vocab.word2idx['<PAD>']] * (max_length - len(sequence))
        else:
            sequence = sequence[:max_length]
    
    return sequence

# =============================================================================
# 시각화
# =============================================================================

def plot_training_curves(train_losses, val_losses, train_accs=None, val_accs=None):
    """학습 곡선 그리기"""
    fig, axes = plt.subplots(1, 2 if train_accs else 1, figsize=(15 if train_accs else 10, 5))
    
    if train_accs is None:
        axes = [axes]
    
    # 손실 그래프
    axes[0].plot(train_losses, label='Train Loss')
    axes[0].plot(val_losses, label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # 정확도 그림 (있을 때)
    if train_accs:
        axes[1].plot(train_accs, label='Train Acc')
        axes[1].plot(val_accs, label='Val Acc')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)
    
    plt.tight_layout()
    return fig

# =============================================================================
# 모델 저장
# =============================================================================

def save_model(model, path, vocab=None):
    """모델과, 원하면 어휘도 저장하기"""
    save_dict = {'model_state_dict': model.state_dict()}
    if vocab:
        save_dict['vocab'] = vocab
    torch.save(save_dict, path)
    print(f"Model saved to {path}")

def load_model(model_class, path, device, **model_kwargs):
    """모델 불러오기"""
    checkpoint = torch.load(path, map_location=device)
    model = model_class(**model_kwargs).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    vocab = checkpoint.get('vocab', None)
    print(f"Model loaded from {path}")
    return model, vocab

# =============================================================================
# 데이터 생성 도구
# =============================================================================

def generate_sine_wave(seq_length, num_samples):
    """시계열을 위한 사인파 데이터 만들기"""
    X = []
    y = []
    for _ in range(num_samples):
        start = np.random.rand() * 2 * np.pi
        x = np.sin(np.linspace(start, start + 2 * np.pi, seq_length + 1))
        X.append(x[:-1])
        y.append(x[1:])
    return np.array(X), np.array(y)

__version__ = "1.0.0"
__author__ = "PyTorch RNN Tutorial"


if __name__ == "__main__":
    pass
```

## 논의

이 구현은 클래스 다섯 개(`SimpleRNN`, `LSTMClassifier`, `GRUPredictor`, `BiLSTM`, 그리고 하나 더)를 정의하며, 이들이 어우러져 완전한 순환 신경망 구조를 이룬다. 클래스마다 별개의 부품을 감싸므로 코드가 모듈식이고 넓히기 쉽다. `forward` 메서드가 PyTorch의 자동 미분이 쓰는 계산 그래프를 정의한다.

손실 계산은 모델의 출력을 최적화 목표와 이어 준다. 알맞은 손실 함수를 고르는 일은 결정적으로 중요하다. 손실 함수가 모델이 무엇을 최적화하도록 배울지를 정하며, 학습된 표현과 결정 경계를 직접 빚어내기 때문이다.

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다. 그림을 그리는 코드는 학습된 표현, 수렴의 움직임, 평가 지표에 대한 통찰을 주어 추상적인 계산을 손에 잡히게 만든다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 넓혀진다. 초매개변수와 구조의 변형, 다른 데이터셋으로 실험해 보면 이해가 깊어지고 순차열 모형 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
기본 초기화를 쓴 `SimpleRNN`의 학습 가능한 매개변수 총수를 계산하라. 가중치와 편향을 모두 넣어 층별로 나누어 세어라.

??? success "연습문제 1 풀이"
    `nn.Linear(in_features, out_features)` 각각에는 `in_features * out_features`개의 가중치 매개변수와 (`bias=False`가 아닌 한) `out_features`개의 편향 매개변수가 있다. `nn.Conv2d(in_c, out_c, k)`에는 `in_c * out_c * k * k`개의 가중치와 `out_c`개의 편향이 있다. `nn.Embedding(num, dim)`에는 `num * dim`개의 매개변수가 있다. 모든 층에 대해 더하면 된다. `sum(p.numel() for p in model.parameters())`로 확인할 수 있다.

---

**연습문제 2.**
입력이 기대하는 모양과 자료형을 갖는지 확인하도록 주 함수나 클래스에 입력 검증을 추가하라. 잘못된 입력에는 유익한 오류 메시지를 내라.

??? success "연습문제 2 풀이"
    `forward` 메서드(또는 해당 함수)의 첫머리에 다음과 같은 검사를 추가한다. `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`와 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'`. 모양을 검증할 때는 중요한 차원을 확인한다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 유익한 오류 메시지는 디버깅 속도를 크게 높이고 코드를 재사용하기에도 더 견고하게 만든다.

---

**연습문제 3.**
은닉 크기가 $h$이고 입력 크기가 $x$로 같을 때 LSTM 셀과 GRU 셀의 매개변수 개수를 비교하라. 어느 쪽이 더 적으며 그 이유는 무엇인가?

??? success "연습문제 3 풀이"
    LSTM에는 4개의 게이트(입력, 망각, 셀, 출력)가 있고 각 게이트가 입력과 은닉 상태 양쪽에 대한 가중치 행렬을 가지므로 $4 \times (x \cdot h + h \cdot h + h) = 4(xh + h^2 + h)$개의 매개변수를 갖는다. GRU에는 3개의 게이트(재설정, 갱신, 새 상태)가 있어 $3 \times (x \cdot h + h \cdot h + h) = 3(xh + h^2 + h)$개이다. GRU는 게이트를 4개 대신 3개 쓰고 셀 상태와 은닉 상태를 합치므로 LSTM의 75%에 해당하는 매개변수를 갖는다. 실무에서 GRU는 매개변수가 적은데도 LSTM에 견줄 만한 성능을 내는 경우가 많다.

---

**연습문제 4.**
층이나 블록의 수를 설정할 수 있도록 `SimpleRNN`을 확장하라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`으로 깊이가 변하는 구조를 만들어라. 층 2개, 4개, 8개로 시험하라.

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`으로 훑는다. (보통의 파이썬 리스트가 아니라) `nn.ModuleList`을 써야 PyTorch가 최적화를 위해 모든 매개변수를 등록한다. `for n in [2, 4, 8]: model = SimpleRNN(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.
