# RNN 유틸리티

RNN 실습에 사용하는 모델, 학습·평가 함수와 텍스트·저장 도구를 모은 참조 페이지다. 네 모델은 서로 다른 대안이며 하나의 모델로 연결되어 있지 않다. 공통 함수가 모든 모델의 반환값과 과제에 바로 호환되는지 확인해야 한다.

## 1. 코드

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
    
    # 순차열 파라메터
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
        
        # 그래디언트 폭발을 막으려고 그래디언트 자르기
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
    
    # 로스 그래프
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

## 2. 논의

`SimpleRNN`은 연속 피처의 마지막 시점으로 분류 로짓과 은닉 상태의 튜플을 반환한다. `LSTMClassifier`와 `BiLSTM`은 토큰 ID를 임베딩하고 분류 로짓만 반환한다. `GRUPredictor`는 연속 피처로부터 마지막 시점의 예측을 반환한다. `Vocabulary`는 뉴럴 넷트웍이 아닌 단어 사전이다.

`train_epoch`와 `evaluate`는 모델 반환값을 텐서로 가정한다. 따라서 튜플을 반환하는 SimpleRNN에는 그대로 사용할 수 없고 로짓을 분리해야 한다. 출력 차원이 2 이상이면 argmax 정확도를 계산하므로 여러 회귀 타깃이나 길이별 출력에는 적합한 지표가 아니다. 단일 이진 로짓에는 정확도가 0으로 보고될 수 있으므로 출력에 맞춘 평가가 필요하다.

로스는 배치 로스의 단순 평균이다. 마지막 배치가 작으면 표본 평균과 다르며 빈 로더에서는 0으로 나누게 된다. `evaluate`는 평가 모드를 유지하므로 이후 학습에서는 다시 train 모드를 설정해야 한다. 모델들은 길이 정보를 받지 않아 패딩 위치도 처리한다.

저장 함수는 모델 state_dict와 선택적인 어휘 객체를 저장한다. 구조 인수와 전처리 규칙, 옵티마이저·스케줄러 상태와 난수 상태는 따로 관리해야 한다. 사용자 정의 Vocabulary 객체를 저장하면 최신 PyTorch의 기본 weights_only 로드에서 허용되지 않을 수 있다. 사전 데이터를 기본 자료형으로 저장하는 방식과 신뢰할 수 있는 객체 복원을 구분한다.

`generate_sine_wave`는 각 표본에서 한 칸 이동한 입력·정답 배열을 만든다. 길이 전체를 예측하는 정답과 마지막 출력만 쓰는 GRUPredictor의 출력은 그대로 대응하지 않을 수 있다. 모델 출력과 타깃의 형태를 명시적으로 맞춰 의도하지 않은 브로드캐스팅을 방지한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
입력 차원 x, 은닉 차원 h, 출력 차원 o, 순환 레이어수 L인 SimpleRNN의 파라메터 수를 구하라.

</div>

??? success "연습문제 1 풀이"
    첫 순환 레이어는 $h(x+h)+2h$개다. 추가 레이어는 각각 $2h^2+2h$개이고 출력 레이어는 $oh+o$개다. 따라서 합계는 $h(x+h)+2h+(L-1)(2h^2+2h)+oh+o$다. 시퀀스 길이에 비례해 파라메터가 늘지는 않는다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
공통 학습 함수를 네 모델에 적용하기 전에 확인할 반환값과 과제 조건을 설명하라.

</div>

??? success "연습문제 2 풀이"
    SimpleRNN의 튜플에서 로짓을 분리해야 한다. 분류의 정수 클래스와 회귀의 실수 타깃을 구분하고 로스에 맞는 모양을 확인한다. 출력 열 수가 여러 개라는 사실만으로 분류 정확도를 계산해서는 안 된다. 길이와 패딩도 별도 처리한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
같은 입력·은닉 크기의 기본 LSTM과 GRU에서 바이어스를 포함한 셀 파라메터 수를 비교하라.

</div>

??? success "연습문제 3 풀이"
    PyTorch의 bias=True, 단일 레이어·방향, 투영 없는 설정에서는 LSTM이 $4h(x+h+2)$개, GRU가 $3h(x+h+2)$개다. 두 바이어스 벡터를 포함한 계산이며 임베딩과 출력 레이어는 제외한다. LSTM의 네 계산 묶음은 세 게이트와 셀 후보다.

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

공통 유틸리티는 반환 형식, 입력과 타깃의 형태, 로스와 지표가 맞을 때 재사용할 수 있다. 이 페이지의 모델별 차이와 저장·복원의 범위를 확인한 뒤 연결해야 한다.
