# 뉴럴 넷트웍 언어 모델의 문맥·학습·생성

뉴럴 넷트웍 언어 모델은 embedding과 공유된 함수로 다음 토큰의 로짓을 계산한다. 빈도표의 각 문맥을 독립적으로 저장하는 방식과 달리 여러 단어·문맥 사이에서 파라메터를 공유할 수 있다. 하지만 학습한 표현이 항상 의미적 유사성을 정확히 반영하거나 미관측 자료에서 일반화를 보장하는 것은 아니다. 문맥 길이·어휘·시간 상태·인과성은 구조와 구현에서 각각 확인해야 한다.

## 1. 고정 문맥의 순방향 모델

[Bengio 등의 뉴럴 확률 언어 모델](https://jmlr.org/papers/v3/bengio03a.html)은 고정 길이 문맥의 단어 embedding을 연결해 다음 단어를 예측하는 대표적인 출발점이다. 같은 어휘 embedding을 위치마다 공유하지만 연결 뒤 선형 레이어는 위치별 입력 웨이트를 갖는다. 따라서 ‘파라메터를 전혀 공유하지 않는다’는 설명은 맞지 않는다.

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

문맥 m개·embedding 차원D이면 연결은 mD차원이고, tanh 은닉을 선형 변환해 어휘 크기V의 로짓을 만든다. 출력 softmax의 비용에는 보통 은닉H와 어휘V의 곱도 들어간다. 고정 m개 밖의 입력은 직접 보지 못한다.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple

class Vocabulary:
    """뉴럴 말 모델의 낱말 곳간 다스리기."""
    
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
    """피드포워드 말 모델 익히기용 자료 뭉치."""
    
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
    피드포워드 뉴럴 말 모델(Bengio 외, 2003).
    
    구조:
        묻힘 → 이어 붙이기 → 숨은 레이어 → 내놓음 → 소프트맥스
    
    인수:
        vocab_size: 낱말 곳간의 크기
        embedding_dim: 낱말 임베딩의 차원
        context_size: 맥락 낱말의 개수
        hidden_dim: 숨은 레이어의 차원
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
        피드포워드.
        
        인수:
            x: (batch_size, context_size) 맥락 낱말 번호
            
        반환값:
            (batch_size, vocab_size) 로짓
        """
        # 묻힘: (batch, context_size) → (batch, context_size, embed_dim)
        embeds = self.embedding(x)
        
        # 펴기: (batch, context_size * embed_dim)
        embeds = embeds.view(x.size(0), -1)
        
        # tanh 활성화를 갖춘 숨은 레이어
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
    """피드포워드 말 모델 익히기."""
    
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

Vocabulary는 PAD·UNK·시작·종료를 먼저 등록하고 소문자 공백 토큰으로 어휘를 만든다. 데이터는 문장 앞에 m개 시작 토큰, 뒤에 종료를 넣어 고정 문맥과 다음 단어 목표를 만든다. 어휘는 학습 분할에서만 구성하고 평가의 미등록 단어는 같은 UNK 규칙을 적용한다. 모델 출력에는 PAD·시작 같은 특수 토큰도 있어 생성 시 허용 폴리시가 필요하다.

학습은 CE에 raw 로짓을 넣고 Adam으로 갱신한다. 출력 PPL은 학습 배치 로스를 단순 평균한 exp이며 마지막 배치 크기·평가 토큰 수를 가중한 독립 검증 PPL과 다르다. 빈 loader도 처리하지 않는다. get_next_word_probs는 CPU 입력을 만들고 eval로 바꾼 채 돌려주므로 장치·모드의 부수 효과를 확인한다.

## 2. RNN 상태와 잘린 역전파

RNN은 단어를 순서대로 읽으며 이전 은닉 상태를 갱신한다. h_t를 w_t까지 처리한 상태로 정의했다면 그 출력은 다음 w_{t+1} 예측에 사용한다. 입력·목표를 한 칸 이동하는 계약을 명확히 해야 한다. 유한한 상태와 실제 학습 길이가 있어 ‘모든 과거를 제한 없이 정확히 기억한다’는 뜻은 아니다.

```
For each time step t:
    h_t = tanh(W_hh · h_{t-1} + W_xh · x_t + b_h)
    y_t = W_hy · h_t + b_y
    P(w_t | w_1,...,w_{t-1}) = softmax(y_t)
```

```python
class RNNLanguageModel(nn.Module):
    """
    맥락 길이가 바뀌는 리커런트 넷트웍 말 모델.
    
    숨은 상태가 아무리 먼 지난 이야기의 앎도 나른다.
    
    인수:
        vocab_size: 낱말 곳간의 크기
        embedding_dim: 낱말 임베딩의 차원
        hidden_dim: 리커런트 넷트웍 숨은 상태의 차원
        num_layers: 쌓은 리커런트 넷트웍 레이어의 개수
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
        리커런트 넷트웍을 지나는 피드포워드.
        
        인수:
            x: (batch, seq_len) 들임 토막 번호
            hidden: 처음 숨은 상태(없어도 됨)
            
        반환값:
            logits: (batch, seq_len, vocab_size)
            hidden: 마지막 숨은 상태
        """
        # 묻힘: (batch, seq_len, embed_dim)
        embeds = self.dropout(self.embedding(x))
        
        # 리커런트 넷트웍: (batch, seq_len, hidden_dim)
        output, hidden = self.rnn(embeds, hidden)
        output = self.dropout(output)
        
        # 어휘로 사영한다
        logits = self.fc(output)
        
        return logits, hidden
    
    def init_hidden(self, batch_size: int) -> torch.Tensor:
        """숨은 상태를 0으로 첫자리매김."""
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim)

class RNNLMDataset(Dataset):
    """리커런트 넷트웍 말 모델용 자료 뭉치(차례에서 차례로)."""
    
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
    
    # 배치 안 최대 길이까지 덧대기
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

RNNLMDataset은 시작·종료를 포함한 각 문장을 max_len 길이 구간으로 나누고 입력 seq[:-1], 목표 seq[1:]을 만든다. collate는 PAD ID0으로 길이를 맞춘다. 긴 문장의 다음 구간에서도 상태를 초기화하면 이전 구간의 문맥은 잃는다. 독립 문장을 무작위 배치로 섞는 방식과 연속 스트림의 같은 배치 행에 상태를 이어 주는 방식은 다르다.

```python
def train_rnn_truncated_bptt(model, data, hidden, seq_len=35):
    """잘라 낸 때 거슬러 퍼뜨리기로 익히기."""
    model.train()
    
    for i in range(0, data.size(1) - 1, seq_len):
        # 배치 얻기
        seqlen = min(seq_len, data.size(1) - 1 - i)
        inputs = data[:, i:i+seqlen]
        targets = data[:, i+1:i+1+seqlen]
        
        # 숨은 상태를 발자취에서 떼어 내기
        hidden = hidden.detach()
        
        # 피드포워드와 되돌림
        logits, hidden = model(inputs, hidden)
        loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    
    return hidden
```

잘린 BPTT는 상태 값은 다음 구간에 전달하되 detach로 경사 연결을 자른다. 위 함수는 RNN의 단일 hidden 텐서를 가정하며 LSTM의(h,c) 튜플에는 그대로 적용되지 않는다. criterion·optimizer·vocab_size도 외부 정의가 필요하다. targets는 배치의 열 구간 slicing이라 비연속일 수 있어 view(-1)가 실패할 수 있다. 배치 행이 같은 스트림을 계속 나타내는지와 경계에서의 상태 초기화도 필요하다. RNN init_hidden은 CPU 기본 dtype 텐서를 만들므로 모델의 장치·dtype과 맞춰야 한다.

## 3. LSTM과 weight tying

LSTM은 망각·입력·출력 gate와 후보 셀을 사용해 셀 상태 c_t=f_t⊙c_{t−1}+i_t⊙c̃_t를 갱신하고 h_t=o_t⊙tanh(c_t)를 만든다. 경사와 기억 경로를 개선하는 설계지만 긴 의존을 항상 보존하거나 경사 문제가 완전히 사라지는 것은 아니다.

```python
class LSTMLanguageModel(nn.Module):
    """
    먼 거리 얽힘을 위해 문을 둔 LSTM 말 모델.
    
    칸 상태가 그래디언트 흐름의 "고속도로"가 된다.
    
    인수:
        vocab_size: 낱말 곳간의 크기
        embedding_dim: 낱말 임베딩의 차원
        hidden_dim: LSTM 숨은 상태의 차원
        num_layers: 쌓은 LSTM 레이어의 개수
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
        LSTM을 지나는 피드포워드.
        
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
            
            # 로스를 위해 꼴 바꾸기: (batch * seq_len, vocab_size)
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

LSTMLanguageModel은 embedding·LSTM·dropout·출력 head다. tie_weights=True여도 embedding_dim과 hidden_dim이 같을 때만 실제 웨이트를 공유한다. 학습 함수의 기본256·512는 달라 기본 호출에서는 tying이 일어나지 않는다. 공유가 일어나면 embedding과 출력 웨이트는 같은 Parameter이며 두 번 세지 않는다.

이 구현의 일반 dropout은 [AWD-LSTM 연구](https://arxiv.org/abs/1708.02182)의 recurrent weight dropout·embedding dropout·시간에 공유하는 locked mask 전체를 구현한 것이 아니다. AWD-LSTM이라는 이름의 성능과 이 단순 구조를 동일시하지 않는다. 해당 연구의 ICLR 발표는2018년이며 preprint는2017년이다.

학습 루프는 배치마다(h,c)를 새로 초기화하므로 잘라 나눈 문장 구간 사이의 상태도 이어지지 않는다. PAD 목표0을 CE에서 제외하지만 출력 후보에서 PAD를 제거하는 것은 아니다. 기록한 배치 평균 로스의 exp도 유효 토큰별 전체 NLL에서 계산한 PPL과 다를 수 있다. 검증·시험 분할이나 최고 체크포인트 평가도 포함하지 않는다.

## 4. 인과 attention과 decoder memory

자기회귀 모델은 현재까지의 입력에서 다음 토큰을 예측해야 한다. causal self-attention에서는 i위치가 j≤i의 입력만 볼 수 있게 한다. 학습 목표를 다음 토큰으로 이동한 상태에서 현재 입력을 보는 것은 적절하지만 다음 입력 위치를 보면 정답이 누출될 수 있다.

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
        
        # 버퍼로 등록(파라메터 아님)
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
        nhead: 어텐션 머리의 개수
        num_layers: 변환기 레이어의 개수
        dim_feedforward: 피드포워드 넷트웍의 안쪽 차원
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
        
        # 변환기 디코더 레이어
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
        """인과 어텐션 마스크 만들기."""
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        인과 가림을 쓰는 피드포워드.
        
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
        
        # 변환기 피드포워드(셀프 어텐션만)
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

현재 TransformerLM은 **TransformerDecoder에 x를 target과 memory로 동시에 전달하면서 tgt_mask만 지정**한다. [Decoder API](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerDecoder.html)에서 target self-attention 마스크와 memory cross-attention 마스크는 별개다. 이 코드의 cross-attention은 뒤 위치의 x도 읽을 수 있어, 학습 때 다음 토큰 정답이 memory에 있는 미래 입력에서 누출될 수 있다. 따라서 올바른 causal decoder-only LM 구현으로 설명할 수 없다.

입력·memory padding 마스크도 없어 학습 로스에서 PAD를 무시한다고 attention이 패딩을 자동 제외하는 것은 아니다. 위치 embedding은 max_len까지이고 현재 sin/cos 할당은 홀수 d_model에서 차원이 맞지 않을 수 있다. head 수로 나누어지는 d_model 조건도 필요하다. 이러한 인과성·입력 계약을 해결하기 전의 낮은 학습 로스는 적절한 자기회귀 성능의 근거가 아니다.

## 5. 공통 생성 함수의 인터페이스 한계

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
        top_p: 알갱이 표집 임계값(없어도 됨)
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

generate_text는 모든 구조를 지원하는 함수처럼 보이지만 서로 다른 호출 계약을 섞는다. LSTM에서는 이전 hidden을 유지하면서 매번 늘어난 prefix를 다시 입력해 과거 토큰을 중복 처리한다. 상태를 유지하면 새 토큰만, 전체 prefix를 다시 넣으면 초기 상태에서 계산하는 일관된 폴리시가 필요하다.

RNN에는 lstm 속성이 없어 else로 가지만 반환값은(logits,hidden) 튜플이라 뒤의 텐서 인덱싱과 맞지 않는다. FeedforwardLM은 고정 context_size와 B×V 출력을 기대하는데 이 함수는 가변 prefix와 B×L×V를 가정한다. Transformer도 위 인과 누출 문제가 남고 최대50토큰으로 자르면 원래 관찰 범위·위치 인덱스가 달라진다.

생성 입력은 CPU에서 만들어 모델 장치가 바뀌면 맞지 않을 수 있다. 온도 양수·top-k 범위·top-p 범위, 허용 특수 토큰·종료와 최대 길이를 검증하는 기능도 제한적이다. 한 함수에 구조 이름만 바꿔 넣으면 동일한 확률 모델의 생성이 된다고 가정하지 않는다.

## 8. 사전 학습 모델과 비교 기준

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

마지막은 앞의 단어 모델과 별개의 GPT-2 모델·토크나이저를 불러 샘플링하는 예다. 웨이트·패키지·환경이 필요하며 로컬 모델을 이 코드에서 사전 학습한 것은 아니다. max_length는 프롬프트를 포함한 길이이고 토큰화도 앞의 공백 단어와 다르다. 이 편집에서는 다운로드·추론·학습을 실행하지 않는다.

순방향·RNN·LSTM·트랜스포머의 문맥·속도·메모리를 고정 순위로 제시하지 않는다. RNN은 길이에 순차적인 상태 갱신을 하고 attention은 학습 위치를 병렬 계산할 수 있지만 자기회귀 생성은 다음 토큰에 의존한다. dense attention의 쌍별O(L²D)·projection O(LD²), recurrent 활성 저장·BPTT·cache와 구현도 구분한다.

PPL은 같은 평가 자료·어휘·토큰화·UNK·경계·사전 학습 조건에서 비교한다. 서로 다른 조건의 Penn Treebank나 GPT-2 숫자를 한 표의 구조 순위로 제시하지 않는다. 큰 모델의 구체적 구조·파라메터가 공개되지 않았을 때 특정 상용 모델을 일률적인 크기·구조의 예로 단정하지 않는다.

## 연습문제

1. **입력·목표 이동**: seq=[시작,a,b,종료]에서 입력과 목표를 적어라. → 입력은[시작,a,b], 목표는[a,b,종료]다. 각 출력은 처리한 입력의 다음 토큰을 예측한다.

2. **상태와 prefix**: hidden을 유지한 LSTM에 매번 전체 prefix를 다시 넣으면 무엇이 달라지는가? → 이미 처리한 토큰을 다시 상태에 누적한다. 새 토큰만 넣거나 초기 상태에서 prefix를 다시 계산해야 한다.

3. **인과 마스크**: 현재 Transformer의 tgt_mask만으로 충분한가? → 아니다. x를 memory로도 넣어 unmasked cross-attention이 미래 입력을 읽는다. 모델 전체 경로에서 미래 비참조를 보장해야 한다.

4. **웨이트 공유**: 학습 함수의 기본256차원 embedding·512차원 hidden에서 tie_weights가 적용되는가? → 코드 조건이 같지 않아 적용되지 않는다. 구조·차원과 실제 Parameter 공유를 확인한다.

5. **PPL 집계**: 길이가 다른 배치 로스를 같은 웨이트로 평균한 exp는 토큰 PPL인가? → 일반적으로 아니다. 무시하지 않은 토큰의 NLL 합을 같은 사건 수로 나눈 뒤 exp해야 한다. 독립 검증 자료와 인과성도 필요하다.

## 정리하며

뉴럴 언어 모델은 표현을 공유하지만 문맥·상태·패딩·인과성의 계약이 필요하다. 이 코드의 decoder memory 누출과 구조별 생성 인터페이스 차이를 이해하고, 실제 웨이트 공유·학습 범위·토큰별 평가를 확인해야 한다.

**참고 문헌**

1. Bengio, Y., et al. (2003). A neural probabilistic language model. *JMLR*.
2. Mikolov, T., et al. (2010). Recurrent neural network based language model. *INTERSPEECH*.
3. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*.
4. Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS*.
5. Merity, S., et al. (2018). Regularizing and Optimizing LSTM Language Models. *ICLR*.
