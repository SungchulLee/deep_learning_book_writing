# 신경 말 모델
## 학습 목표

이 절을 마치면 다음을 할 수 있게 된다.

- 셈 바탕에서 신경 말 모델로의 흐름을 이해한다
- PyTorch로 앞먹임 신경 말 모델을 짠다
- 맥락 길이가 바뀌는 되돌이 그물 바탕 말 모델을 세운다
- 멀리 떨어진 얽힘을 다루는 LSTM 말 모델을 짠다
- 스스로 눈길을 쓰는 변환기 바탕 말 모델을 꾸민다
- 일에 따라 알맞은 얼개를 견주고 고른다

---

## n-그램에서 신경 모델로

n-그램 모델에는 근본 한계가 있다. 곧 붙박이 맥락 창, 자료의 성김, 뜻으로 두루 통하지 못함이다. 신경 말 모델은 낱말을 이어진 벡터 공간에 묻는 **흩뿌린 나타냄**을 배워 이를 다룬다.

### 신경 말 모델의 핵심 이점

| 갈래 | n-그램 | 신경 |
|--------|--------|--------|
| 맥락 | 붙박이, 제한됨 | 바뀌거나 제한 없음 |
| 나타냄 | 띄엄띄엄한 셈 | 이어진 묻힘 |
| 두루 통함 | 없음 | 뜻의 닮음 |
| 기억 | 드러난 셈 | 배운 매개변수 |
| 부드럽게 하기 | 드러난 재주 | 묻힘으로 넌지시 |

---

## 앞먹임 신경 말 모델

Bengio 외(2003)의 선구적인 연구는 다음 얼개로 신경 말 모델을 들여왔다:

### 구조 훑어보기

```
Input: [w_{t-n+1}, ..., w_{t-1}]  (context words)
         ↓
    [Embedding Layer]  → Lookup word vectors
         ↓
    [Concatenate]      → Join embeddings
         ↓
    [Hidden Layer]     → Non-linear transformation
         ↓
    [Output Layer]     → Scores over vocabulary
         ↓
    [Softmax]          → Probability distribution
         ↓
Output: P(w_t | context)
```

### 수식으로 나타내기

앞뒤 흐름 낱말 $w_{t-n+1}, \ldots, w_{t-1}$이 주어지면

1. **Embedding**: $\mathbf{e}_i = C(w_i) \in \mathbb{R}^d$
2. **이어 붙이기**: $\mathbf{x} = [\mathbf{e}_{t-n+1}; \ldots; \mathbf{e}_{t-1}] \in \mathbb{R}^{(n-1) \cdot d}$
3. **Hidden layer**: $\mathbf{h} = \tanh(\mathbf{W}\mathbf{x} + \mathbf{b})$
4. **Output**: $\mathbf{s} = \mathbf{U}\mathbf{h} + \mathbf{c}$
5. **Softmax**: $P(w_t = v | \text{context}) = \frac{\exp(s_v)}{\sum_{v'} \exp(s_{v'})}$

### PyTorch 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple


class Vocabulary:
    """신경 말 모델의 낱말 곳간 다스리기."""
    
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.PAD = "<pad>"
        self.UNK = "<unk>"
        self.START = "<s>"
        self.END = "</s>"
        
        # 특수 토큰 더하기
        for token in [self.PAD, self.UNK, self.START, self.END]:
            self._add(token)
    
    def _add(self, word: str) -> int:
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        return self.word2idx[word]
    
    def build(self, corpus: List[str], min_freq: int = 1):
        """말뭉치에서 낱말 곳간 세우기."""
        from collections import Counter
        counts = Counter()
        for sentence in corpus:
            counts.update(sentence.lower().split())
        
        for word, count in counts.items():
            if count >= min_freq:
                self._add(word)
        
        print(f"Vocabulary size: {len(self)}")
    
    def encode(self, word: str) -> int:
        return self.word2idx.get(word.lower(), self.word2idx[self.UNK])
    
    def decode(self, idx: int) -> str:
        return self.idx2word[idx]
    
    def __len__(self) -> int:
        return len(self.word2idx)


class FeedforwardLMDataset(Dataset):
    """앞먹임 말 모델 익히기용 자료 뭉치."""
    
    def __init__(self, corpus: List[str], vocab: Vocabulary, context_size: int):
        self.vocab = vocab
        self.context_size = context_size
        self.examples = []
        
        for sentence in corpus:
            words = sentence.lower().split()
            # 시작 토막으로 덧대기
            words = [vocab.START] * context_size + words + [vocab.END]
            indices = [vocab.encode(w) for w in words]
            
            # (맥락, 목표) 짝 만들기
            for i in range(context_size, len(indices)):
                context = indices[i - context_size:i]
                target = indices[i]
                self.examples.append((context, target))
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        context, target = self.examples[idx]
        return torch.tensor(context), torch.tensor(target)


class FeedforwardLM(nn.Module):
    """
    앞먹임 신경 말 모델(Bengio 외, 2003).
    
    구조:
        묻힘 → 이어 붙이기 → 숨은 층 → 내놓음 → 소프트맥스
    
    인수:
        vocab_size: 낱말 곳간의 크기
        embedding_dim: 낱말 임베딩의 차원
        context_size: 맥락 낱말의 개수
        hidden_dim: 숨은 층의 차원
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int,
                 context_size: int, hidden_dim: int):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(context_size * embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        """익히기를 낫게 하는 자비에 첫자리매김."""
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        앞먹임.
        
        인수:
            x: (batch_size, context_size) 맥락 낱말 번호
            
        반환값:
            (batch_size, vocab_size) 로짓
        """
        # 묻힘: (batch, context_size) → (batch, context_size, embed_dim)
        embeds = self.embedding(x)
        
        # 펴기: (batch, context_size * embed_dim)
        embeds = embeds.view(x.size(0), -1)
        
        # tanh 깨어남을 갖춘 숨은 층
        hidden = torch.tanh(self.fc1(embeds))
        
        # 내놓는 로짓
        logits = self.fc2(hidden)
        
        return logits
    
    def get_next_word_probs(self, context: List[int]) -> torch.Tensor:
        """다음 낱말에 대한 확률 분포 얻기."""
        self.eval()
        with torch.no_grad():
            x = torch.tensor([context])
            logits = self(x)
            probs = F.softmax(logits, dim=-1)
        return probs[0]


def train_feedforward_lm(corpus: List[str], context_size: int = 3,
                         embedding_dim: int = 64, hidden_dim: int = 128,
                         epochs: int = 20, batch_size: int = 32,
                         learning_rate: float = 0.001):
    """앞먹임 말 모델 익히기."""
    
    # 어휘 만들기
    vocab = Vocabulary()
    vocab.build(corpus)
    
    # 데이터셋 생성
    dataset = FeedforwardLMDataset(corpus, vocab, context_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 모형을 시작한다
    model = FeedforwardLM(
        vocab_size=len(vocab),
        embedding_dim=embedding_dim,
        context_size=context_size,
        hidden_dim=hidden_dim
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 학습 루프
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for contexts, targets in loader:
            logits = model(contexts)
            loss = criterion(logits, targets)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        perplexity = torch.exp(torch.tensor(avg_loss))
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, PPL={perplexity:.2f}")
    
    return model, vocab
```

### 앞먹임 모델의 한계

1. **붙박이 맥락 창**: 아무리 먼 얽힘도 다루지 못한다
2. **매개변수를 나눠 쓰지 않음**: 자리마다 무게가 따로 있다
3. **셈 값**: 내놓는 소프트맥스에 $O(V)$

---

## 되돌이 그물 말 모델

되돌이 신경망은 때 걸음을 가로질러 앎을 나르는 **숨은 상태**를 지녀 붙박이 맥락의 한계를 다룬다.

### 구조

```
For each time step t:
    h_t = tanh(W_hh · h_{t-1} + W_xh · x_t + b_h)
    y_t = W_hy · h_t + b_y
    P(w_t | w_1,...,w_{t-1}) = softmax(y_t)
```

숨은 상태 $\mathbf{h}_t$은 지나온 $w_1, \ldots, w_{t-1}$ 모두를 간추린다.

### 구현

```python
class RNNLanguageModel(nn.Module):
    """
    맥락 길이가 바뀌는 되돌이 그물 말 모델.
    
    숨은 상태가 아무리 먼 지난 이야기의 앎도 나른다.
    
    인수:
        vocab_size: 낱말 곳간의 크기
        embedding_dim: 낱말 임베딩의 차원
        hidden_dim: 되돌이 그물 숨은 상태의 차원
        num_layers: 쌓은 되돌이 그물 층의 개수
        dropout: 드롭아웃 확률
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int,
                 num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        init_range = 0.1
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.fc.weight.data.uniform_(-init_range, init_range)
        self.fc.bias.data.zero_()
    
    def forward(self, x: torch.Tensor, hidden: torch.Tensor = None):
        """
        되돌이 그물을 지나는 앞먹임.
        
        인수:
            x: (batch, seq_len) 들임 토막 번호
            hidden: 처음 숨은 상태(없어도 됨)
            
        반환값:
            logits: (batch, seq_len, vocab_size)
            hidden: 마지막 숨은 상태
        """
        # 묻힘: (batch, seq_len, embed_dim)
        embeds = self.dropout(self.embedding(x))
        
        # 되돌이 그물: (batch, seq_len, hidden_dim)
        output, hidden = self.rnn(embeds, hidden)
        output = self.dropout(output)
        
        # 어휘로 사영한다
        logits = self.fc(output)
        
        return logits, hidden
    
    def init_hidden(self, batch_size: int) -> torch.Tensor:
        """숨은 상태를 0으로 첫자리매김."""
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim)


class RNNLMDataset(Dataset):
    """되돌이 그물 말 모델용 자료 뭉치(차례에서 차례로)."""
    
    def __init__(self, corpus: List[str], vocab: Vocabulary, max_len: int = 35):
        self.sequences = []
        
        for sentence in corpus:
            words = sentence.lower().split()
            words = [vocab.START] + words + [vocab.END]
            indices = [vocab.encode(w) for w in words]
            
            # 익히기를 위해 덩이로 쪼개기
            for i in range(0, len(indices) - 1, max_len):
                seq = indices[i:i + max_len + 1]
                if len(seq) > 1:
                    self.sequences.append(seq)
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = self.sequences[idx]
        # 들임: 마지막 토막을 뺀 전부, 목표: 첫 토막을 뺀 전부
        return torch.tensor(seq[:-1]), torch.tensor(seq[1:])


def collate_sequences(batch):
    """길이가 들쭉날쭉한 차례를 덧대어 모으기."""
    inputs, targets = zip(*batch)
    
    # 묶음 안 최대 길이까지 덧대기
    max_len = max(len(x) for x in inputs)
    
    padded_inputs = []
    padded_targets = []
    
    for inp, tgt in zip(inputs, targets):
        pad_len = max_len - len(inp)
        if pad_len > 0:
            inp = torch.cat([inp, torch.zeros(pad_len, dtype=torch.long)])
            tgt = torch.cat([tgt, torch.zeros(pad_len, dtype=torch.long)])
        padded_inputs.append(inp)
        padded_targets.append(tgt)
    
    return torch.stack(padded_inputs), torch.stack(padded_targets)
```

### 때를 거슬러 뒤로 퍼뜨리기(BPTT)

되돌이 그물은 셈 그래프를 때에 걸쳐 펼쳐 놓고 뒤로 퍼뜨리기를 써서 익힌다. 길이 $T$인 차례에 대해:

$$\frac{\partial L}{\partial W} = \sum_{t=1}^{T} \frac{\partial L_t}{\partial W}$$

기울기 항마다 야코비 행렬의 곱이 들어 있어 기울기가 **사라지거나 터질** 수 있다.

### 잘라 낸 때 거슬러 퍼뜨리기

긴 차례에서는 앞먹임의 숨은 상태는 그대로 두고 뒤로 퍼뜨리기만 붙박이 창으로 잘라 낸다:

```python
def train_rnn_truncated_bptt(model, data, hidden, seq_len=35):
    """잘라 낸 때 거슬러 퍼뜨리기로 익히기."""
    model.train()
    
    for i in range(0, data.size(1) - 1, seq_len):
        # 묶음 얻기
        seqlen = min(seq_len, data.size(1) - 1 - i)
        inputs = data[:, i:i+seqlen]
        targets = data[:, i+1:i+1+seqlen]
        
        # 숨은 상태를 발자취에서 떼어 내기
        hidden = hidden.detach()
        
        # 앞먹임과 되돌림
        logits, hidden = model(inputs, hidden)
        loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    
    return hidden
```

---

## LSTM 말 모델

긴 짧은 기억 그물은 앎의 흐름을 다스리는 **문 얼개**로 기울기 사라짐 문제를 다룬다.

### LSTM 식

$$\mathbf{f}_t = \sigma(\mathbf{W}_f[\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f) \quad \text{(Forget gate)}$$

$$\mathbf{i}_t = \sigma(\mathbf{W}_i[\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i) \quad \text{(Input gate)}$$

$$\tilde{\mathbf{c}}_t = \tanh(\mathbf{W}_c[\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c) \quad \text{(Candidate)}$$

$$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t \quad \text{(Cell update)}$$

$$\mathbf{o}_t = \sigma(\mathbf{W}_o[\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o) \quad \text{(Output gate)}$$

$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t) \quad \text{(Hidden state)}$$

### 구현

```python
class LSTMLanguageModel(nn.Module):
    """
    먼 거리 얽힘을 위해 문을 둔 LSTM 말 모델.
    
    칸 상태가 기울기 흐름의 "고속도로"가 된다.
    
    인수:
        vocab_size: 낱말 곳간의 크기
        embedding_dim: 낱말 임베딩의 차원
        hidden_dim: LSTM 숨은 상태의 차원
        num_layers: 쌓은 LSTM 층의 개수
        dropout: 드롭아웃 확률
        tie_weights: 묻힘 무게와 내놓는 무게를 묶을지 여부
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int,
                 num_layers: int = 2, dropout: float = 0.5, 
                 tie_weights: bool = True):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        # 무게 묶기: 묻힘과 내놓음이 무게를 나눠 쓴다
        if tie_weights and embedding_dim == hidden_dim:
            self.fc.weight = self.embedding.weight
        
        self._init_weights()
    
    def _init_weights(self):
        init_range = 0.1
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.fc.bias.data.zero_()
        if self.fc.weight is not self.embedding.weight:
            self.fc.weight.data.uniform_(-init_range, init_range)
    
    def forward(self, x: torch.Tensor, hidden: Tuple = None):
        """
        LSTM을 지나는 앞먹임.
        
        인수:
            x: (batch, seq_len) 들임 토막
            hidden: 처음 상태 (h_0, c_0) 튜플
            
        반환값:
            logits: (batch, seq_len, vocab_size)
            hidden: 마지막 상태 (h_n, c_n) 튜플
        """
        embeds = self.dropout(self.embedding(x))
        output, hidden = self.lstm(embeds, hidden)
        output = self.dropout(output)
        logits = self.fc(output)
        
        return logits, hidden
    
    def init_hidden(self, batch_size: int):
        """숨은 상태와 칸 상태 첫자리매김."""
        device = next(self.parameters()).device
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return (h, c)


def train_lstm_lm(corpus: List[str], embedding_dim: int = 256,
                  hidden_dim: int = 512, num_layers: int = 2,
                  epochs: int = 30, batch_size: int = 32,
                  learning_rate: float = 0.001):
    """LSTM 말 모델 익히기."""
    
    vocab = Vocabulary()
    vocab.build(corpus)
    
    dataset = RNNLMDataset(corpus, vocab)
    loader = DataLoader(dataset, batch_size=batch_size, 
                        shuffle=True, collate_fn=collate_sequences)
    
    model = LSTMLanguageModel(
        vocab_size=len(vocab),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers
    )
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 채움을 무시한다
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for inputs, targets in loader:
            batch_size = inputs.size(0)
            hidden = model.init_hidden(batch_size)
            
            logits, _ = model(inputs, hidden)
            
            # 손실을 위해 꼴 바꾸기: (batch * seq_len, vocab_size)
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
            
            loss = criterion(logits, targets)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        ppl = torch.exp(torch.tensor(avg_loss))
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, PPL={ppl:.2f}")
    
    return model, vocab
```

### AWD-LSTM: 벌준 LSTM

가장 앞선 LSTM 말 모델은 벌주기를 세게 쓴다:

1. **무게 떨구기**: 되돌이 무게에 떨구기
2. **묻힘 떨구기**: 묻힘 행렬에 떨구기
3. **잠근 떨구기**: 때 걸음마다 같은 떨구기 마스크
4. **무게 묶기**: 묻힘 무게와 내놓는 무게를 나눠 쓰기
5. **길이가 바뀌는 때 거슬러 퍼뜨리기**: 차례 길이를 마구잡이로 뽑기

---

## 변환기 말 모델

변환기는 되돌이를 **스스로 눈길**로 갈음해 나란히 익히기와 더 나은 먼 거리 나타내기를 가능하게 한다.

### 스스로 눈길 얼개

나타냄의 이음 $\mathbf{X} = [\mathbf{x}_1, \ldots, \mathbf{x}_n]$이 주어지면

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

여기서 각 기호는 다음과 같다.

- $\mathbf{Q} = \mathbf{X}\mathbf{W}^Q$ (queries)
- $\mathbf{K} = \mathbf{X}\mathbf{W}^K$ (keys)
- $\mathbf{V} = \mathbf{X}\mathbf{W}^V$ (values)

### 인과 가리기

말 나타내기에서는 앞으로 올 토막에 눈길을 주지 못하게 막아야 한다:

$$\text{mask}_{ij} = \begin{cases} 0 & \text{if } j \leq i \\ -\infty & \text{otherwise} \end{cases}$$

이러면 $P(w_t | w_1, \ldots, w_{t-1})$이 지나간 낱말에만 기대게 된다.

### 구현

```python
import math


class PositionalEncoding(nn.Module):
    """'Attention Is All You Need'의 사인파 위치 인코딩."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 버퍼로 등록(매개변수 아님)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """들임에 자리 부호 더하기."""
        return x + self.pe[:, :x.size(1)]


class TransformerLM(nn.Module):
    """
    GPT 방식 변환기 말 모델.
    
    말 나타내기를 위해 인과(자기되돌리기) 가리기를 쓴다.
    
    인수:
        vocab_size: 낱말 곳간의 크기
        d_model: 모형 차원
        nhead: 눈길 머리의 개수
        num_layers: 변환기 층의 개수
        dim_feedforward: 앞먹임 그물의 안쪽 차원
        dropout: 드롭아웃 확률
        max_len: 순차열의 최대 길이
    """
    
    def __init__(self, vocab_size: int, d_model: int = 512, nhead: int = 8,
                 num_layers: int = 6, dim_feedforward: int = 2048,
                 dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        
        self.d_model = d_model
        
        # 토큰 임베딩
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 위치 인코딩
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        
        # 변환기 풀개 층
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # 출력 사영
        self.fc = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
    
    def _init_weights(self):
        init_range = 0.1
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.fc.weight.data.uniform_(-init_range, init_range)
        self.fc.bias.data.zero_()
    
    def generate_causal_mask(self, size: int) -> torch.Tensor:
        """인과 눈길 마스크 만들기."""
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        인과 가림을 쓰는 앞먹임.
        
        인수:
            x: (batch, seq_len) 들임 토막
            
        반환값:
            (batch, seq_len, vocab_size) 로짓
        """
        seq_len = x.size(1)
        
        # 잣수를 맞춘 묻힘
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.dropout(x)
        
        # 인과 가림
        mask = self.generate_causal_mask(seq_len).to(x.device)
        
        # 변환기 앞먹임(스스로 눈길만)
        output = self.transformer(x, x, tgt_mask=mask)
        
        # 어휘로 사영한다
        logits = self.fc(output)
        
        return logits


def train_transformer_lm(corpus: List[str], d_model: int = 256,
                         nhead: int = 4, num_layers: int = 4,
                         epochs: int = 30, batch_size: int = 32):
    """변환기 말 모델 익히기."""
    
    vocab = Vocabulary()
    vocab.build(corpus)
    
    dataset = RNNLMDataset(corpus, vocab, max_len=50)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, collate_fn=collate_sequences)
    
    model = TransformerLM(
        vocab_size=len(vocab),
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers
    )
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for inputs, targets in loader:
            logits = model(inputs)
            
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
            
            loss = criterion(logits, targets)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        ppl = torch.exp(torch.tensor(avg_loss))
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, PPL={ppl:.2f}")
    
    return model, vocab
```

---

## 글 만들어 내기

신경 말 모델은 여러 만들어 내기 전략을 받쳐 준다:

```python
def generate_text(model, vocab, max_length: int = 50,
                  temperature: float = 1.0, top_k: int = None,
                  top_p: float = None) -> str:
    """
    익힌 말 모델로 글 만들어 내기.
    
    인수:
        model: 익힌 말 모델(LSTM 또는 변환기)
        vocab: 낱말 곳간 개체
        max_length: 만들어 낼 최대 토막 수
        temperature: 표집 온도
        top_k: 상위 k 거르기(없어도 됨)
        top_p: 알갱이 표집 문턱값(없어도 됨)
    """
    model.eval()
    
    # START 토막으로 시작하기
    generated = [vocab.encode(vocab.START)]
    
    # LSTM 숨은 상태 다루기
    hidden = None
    if hasattr(model, 'init_hidden'):
        hidden = model.init_hidden(1)
    
    with torch.no_grad():
        for _ in range(max_length):
            # 들임 갖추기
            x = torch.tensor([generated[-50:]])  # 마지막 토막 50개 쓰기
            
            # 순전파
            if hasattr(model, 'lstm'):
                logits, hidden = model(x, hidden)
            else:
                logits = model(x)
            
            # 마지막 자리의 로짓을 얻는다
            logits = logits[0, -1, :] / temperature
            
            # 상위 k 거르기를 적용한다
            if top_k is not None:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][-1]
                logits[indices_to_remove] = float('-inf')
            
            # 알갱이(상위 p) 거르기 쓰기
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    0, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')
            
            # 뽑기
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            
            # END 토막인지 살피기
            if next_token == vocab.encode(vocab.END):
                break
            
            generated.append(next_token)
    
    # 토막 풀어내기
    words = [vocab.decode(idx) for idx in generated[1:]]  # START 건너뛰기
    return ' '.join(words)
```

---

## 모형 견줌

| 갈래 | 앞먹임 | 되돌이 그물 | LSTM | 변환기 |
|--------|-------------|-----|------|-------------|
| 맥락 | 붙박이 창 | 제한 없음 | 제한 없음 | 온 차례 |
| 익히기 | 나란히 | 차례대로 | 차례대로 | 나란히 |
| 먼 거리 얽힘 | 나쁨 | 나쁨 | 좋음 | 아주 좋음 |
| 기억 | O(창) | O(숨은 층) | O(숨은 층) | O(차례²) |
| 요즘 쓰임 | 드묾 | 드묾 | 보통 | 판을 잡음 |

### 흔한 헷갈림도(Penn Treebank)

| 모델 | 매개변수 | 헷갈림도 |
|-------|------------|------------|
| 앞먹임(Bengio) | 약 1000만 | 약 140 |
| LSTM(2층) | 약 2000만 | 약 80~100 |
| AWD-LSTM | 약 2400만 | 약 57 |
| 변환기(6층) | 약 4000만 | 약 60~70 |
| GPT-2(소형) | 1억 1700만 | 약 35 |

---

## 미리 익힌 말 모델

요즘은 미리 익힌 모델을 특정 일에 곱게 다듬어 써먹는다:

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 미리 익힌 GPT-2 읽어 들이기
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# 글 만들어 내기
input_text = "The quick brown fox"
input_ids = tokenizer.encode(input_text, return_tensors='pt')

output = model.generate(
    input_ids,
    max_length=50,
    temperature=0.7,
    top_p=0.95,
    do_sample=True
)

print(tokenizer.decode(output[0]))
```

---

## 요약

신경 말 모델은 단순한 앞먹임 그물에서 정교한 변환기 얼개로 나아왔다:

1. **앞먹임 말 모델**은 이어진 낱말 나타냄을 들여왔지만 맥락이 붙박이이다
2. **되돌이 그물 말 모델**은 길이가 바뀌는 차례를 다루지만 기울기 탈이 있다
3. **LSTM 말 모델**은 문 얼개로 기울기 사라짐을 다룬다
4. **변환기 말 모델**은 나란히 익히기를 가능하게 하고 먼 거리 얽힘을 담아낸다

요즘의 큰 말 모델(GPT-4, Claude, LLaMA)은 매개변수가 수십억 개인, 크게 키운 변환기 말 모델이다.

---

## 참고 문헌

1. Bengio, Y., et al. (2003). A neural probabilistic language model. *JMLR*.
2. Mikolov, T., et al. (2010). Recurrent neural network based language model. *INTERSPEECH*.
3. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*.
4. Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS*.
5. Merity, S., et al. (2017). Regularizing and optimizing LSTM language models. *ICLR*.

## 연습문제

1. **묻힘 그려 보기**: 앞먹임 말 모델을 익히고 t-SNE로 낱말 묻힘을 그려 보라. 비슷한 낱말이 함께 뭉치는가?

2. **LSTM과 GRU**: GRU 말 모델을 짜고 같은 자료에서 LSTM과 헷갈림도를 견주어라.

3. **눈길 그려 보기**: 변환기 말 모델의 눈길 무늬를 그려 보라. 들임 갈래에 따라 어떤 무늬가 나타나는가?

4. **만들어 낸 글의 좋음**: 같은 시킴말에 대해 n-그램, LSTM, 변환기 모델이 만든 글을 견주어라.

5. **곱게 다듬기**: 분야별 말뭉치(보기로 금융 뉴스)로 GPT-2를 곱게 다듬고 분야 맞추기를 값매김하여라.

---
