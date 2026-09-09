# 트랜스포머의 학습과 추론

트랜스포머 구조에서 학습과 추론은 근본적으로 다른 방식을 따른다. 학습 중에는 모형이 원문과 목표문 수열을 모두 받아 효율적인 병렬 계산을 위해 **스승 강제**를 쓴다. 추론 중에는 모형이 자기 회귀로 토큰을 만들며, 한 번에 하나씩 내고 그것을 다음 단계의 입력으로 되먹인다.

---

## 1. 학습 파이프라인

### 입력 쌍

트랜스포머는 짝지어진 수열 $(x, y)$으로 학습한다.

- $x$: 원문 수열 (이를테면 영어 문장)
- $y$: 목표 수열 (이를테면 프랑스어 번역)

인코더는 $x$을 그대로 받고 디코더는 $y$을 **오른쪽으로 민** 것을 받는다.

### 스승 강제

스승 강제는 학습 중에 모형 자신의 예측 대신 참 목표 수열을 디코더에 넣는 것이다. 그러면 모든 목표 자리를 한꺼번에 병렬로 셈할 수 있다.

목표 수열 $y = [y_1, y_2, \ldots, y_T]$에 대해 다음과 같다.

- **디코더 입력**(오른쪽으로 민 것): $[\langle\text{start}\rangle, y_1, y_2, \ldots, y_{T-1}]$
- **참 이름표**: $[y_1, y_2, \ldots, y_T, \langle\text{end}\rangle]$

오른쪽으로 밀어 두면 토큰 $y_t$을 맞히는 일이 $y_t$ 자신이 아니라 토큰 $y_1, \ldots, y_{t-1}$에만 매인다.

### 보기: 기계 번역

"The cat sat on the mat"을 "Le chat était assis sur le tapis"로 옮길 때 다음과 같다.

$$
\begin{aligned}
\text{Encoder input } (x) &: [\text{The}, \text{cat}, \text{sat}, \text{on}, \text{the}, \text{mat}] \\
\text{Decoder input } (y_{\text{shifted}}) &: [\langle\text{start}\rangle, \text{Le}, \text{chat}, \text{était}, \text{assis}, \text{sur}, \text{le}] \\
\text{Target labels} &: [\text{Le}, \text{chat}, \text{était}, \text{assis}, \text{sur}, \text{le}, \text{tapis}, \langle\text{end}\rangle]
\end{aligned}
$$

### 손실 셈하기

학습 손실은 예측한 확률 분포와 참 토큰 사이의 교차 엔트로피를 모든 자리에 걸쳐 더한 것이다.

$$
\mathcal{L} = -\sum_{t=1}^{T} \log P_\theta(y_t \mid y_{<t}, x)
$$

여기서 $P_\theta(y_t \mid y_{<t}, x)$은 앞선 목표 토큰 $y_{<t}$과 원문 수열 $x$ 전체를 조건으로 하여 자리 $t$의 올바른 토큰 $y_t$에 모형이 매긴 확률이다.

자리마다 디코더는 어휘 전체에 대한 확률 분포를 낸다. 교차 엔트로피 손실은 모형이 올바른 다음 토큰에 낮은 확률을 매기면 벌을 준다.

### 임베딩 학습

낱말 임베딩은 대개 트랜스포머의 나머지와 함께 학습한다. 임베딩 층은 무작위로 초기화하고 역전파로 갱신한다. 흔한 방식이 셋 있다.

1. **맨바닥부터 학습**: 무작위 초기화. 임베딩이 학습 중에 과제에 맞는 표현을 배운다.
2. **사전 학습으로 초기화**: Word2Vec, GloVe, 또는 맥락 임베딩으로 초기화한 뒤 미세 조정하거나 얼려 둔다.
3. **부분 낱말 임베딩**: 바이트 쌍 부호화(BPE)나 WordPiece를 쓰는 모형은 부분 낱말 단위의 임베딩을 배워 어휘를 더 넓게 덮고 드문 낱말을 더 잘 다룬다.

---

## 2. 파이토치 학습 구현

```python
import torch
import torch.nn as nn
import torch.optim as optim

def create_masks(src, tgt, src_pad_idx=0, tgt_pad_idx=0):
    """
    트랜스포머 학습을 위한 채움 가림과 인과 가림을 만든다.
    
    인수:
        src: 원문 토큰 [batch_size, src_len]
        tgt: 목표 토큰 [batch_size, tgt_len]
        src_pad_idx: 원문의 채움 토큰 색인
        tgt_pad_idx: 목표문의 채움 토큰 색인
    
    반환값:
        src_key_padding_mask: [batch_size, src_len] - 채운 자리가 True
        tgt_key_padding_mask: [batch_size, tgt_len] - 채운 자리가 True
        tgt_mask: [tgt_len, tgt_len] - 인과 가림
        memory_key_padding_mask: src_key_padding_mask와 같다
    """
    # 채움 가림: 채운 자리가 True
    src_key_padding_mask = (src == src_pad_idx)       # [batch_size, src_len]
    tgt_key_padding_mask = (tgt == tgt_pad_idx)       # [batch_size, tgt_len]
    
    # 디코더를 위한 인과 가림: 앞으로의 토큰에 주의하지 못하게 한다
    tgt_len = tgt.size(1)
    tgt_mask = torch.triu(                            # [tgt_len, tgt_len]
        torch.ones(tgt_len, tgt_len, device=tgt.device),
        diagonal=1
    ).bool()
    
    return src_key_padding_mask, tgt_key_padding_mask, tgt_mask

def train_step(model, optimizer, criterion, src, tgt, src_pad_idx=0, tgt_pad_idx=0):
    """
    트랜스포머의 학습 단계 하나.
    
    인수:
        model: 트랜스포머 모형
        optimizer: 최적화기 객체
        criterion: 손실 함수(CrossEntropyLoss)
        src: 원문 토큰 [batch_size, src_len]
        tgt: 목표 수열 전체 [batch_size, tgt_len]
        src_pad_idx: 원문의 채움 색인
        tgt_pad_idx: 목표문의 채움 색인
    
    반환값:
        loss: 스칼라 손실 값
    """
    model.train()
    
    # 디코더 입력(오른쪽으로 민 목표)과 이름표를 마련한다
    # 디코더가 보는 것: [<start>, y1, y2, ..., y_{T-1}]
    # 이름표: [y1, y2, ..., y_T]
    tgt_input = tgt[:, :-1]     # [batch_size, tgt_len - 1]
    tgt_labels = tgt[:, 1:]     # [batch_size, tgt_len - 1]
    
    # 가림을 만든다
    src_pad_mask, tgt_pad_mask, tgt_causal_mask = create_masks(
        src, tgt_input, src_pad_idx, tgt_pad_idx
    )
    
    # 순전파
    optimizer.zero_grad()
    output = model(                                   # [batch_size, tgt_len-1, vocab_size]
        src, tgt_input,
        src_mask=None,
        tgt_mask=tgt_causal_mask,
        src_key_padding_mask=src_pad_mask,
        tgt_key_padding_mask=tgt_pad_mask,
        memory_key_padding_mask=src_pad_mask
    )
    
    # 손실을 셈한다 (교차 엔트로피를 위해 편다)
    vocab_size = output.size(-1)
    loss = criterion(
        output.reshape(-1, vocab_size),               # [batch_size * (tgt_len-1), vocab_size]
        tgt_labels.reshape(-1)                        # [batch_size * (tgt_len-1)]
    )
    
    # 역전파와 매개변수 갱신
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    return loss.item()

def train_epoch(model, dataloader, optimizer, criterion, device):
    """
    한 세대를 학습한다.
    
    인수:
        model: 트랜스포머 모형
        dataloader: (src, tgt) 배치를 내보내는 DataLoader
        optimizer: 최적화기 객체
        criterion: 손실 함수
        device: torch.device
    
    반환값:
        average_loss: 모든 배치에 걸친 평균 손실
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for src, tgt in dataloader:
        src = src.to(device)                          # [batch_size, src_len]
        tgt = tgt.to(device)                          # [batch_size, tgt_len]
        
        loss = train_step(model, optimizer, criterion, src, tgt)
        total_loss += loss
        num_batches += 1
    
    return total_loss / num_batches
```

---

## 3. 추론 과정

### 자기 회귀 생성

추론 중에는 원문 수열 $x$만 주어진다. 디코더는 한 번에 토큰 하나씩 만들며 맞힌 토큰마다 다음 단계의 입력으로 되먹인다.

생성 고리는 다음과 같이 돈다.

$$
\hat{y}_t = \arg\max_{v \in \mathcal{V}} P_\theta(v \mid \hat{y}_1, \ldots, \hat{y}_{t-1}, x)
$$

1. 인코더로 원문 수열 $x$을 인코딩한다(한 번만 한다)
2. 디코더 입력을 $\langle\text{start}\rangle$ 토큰으로 시작한다
3. 단계 $t$마다 다음을 한다.
   - 지금까지 만든 토큰 $[\langle\text{start}\rangle, \hat{y}_1, \ldots, \hat{y}_{t-1}]$을 모두 디코더에 넣는다
   - 마지막 자리의 예측을 꺼낸다
   - 다음 토큰을 고른다(탐욕 디코딩, 빔 탐색, 또는 표집으로)
   - 고른 토큰을 디코더 입력에 덧붙인다
4. $\langle\text{end}\rangle$이 나오거나 최대 길이에 이르면 멈춘다

### 파이토치 추론 구현

```python
@torch.no_grad()
def greedy_decode(
    model,
    src: torch.Tensor,
    max_len: int = 100,
    start_token: int = 1,
    end_token: int = 2,
    pad_idx: int = 0
) -> torch.Tensor:
    """
    트랜스포머의 자기 회귀 탐욕 디코딩.
    
    인수:
        model: 학습된 트랜스포머 모형
        src: 원문 수열 [1, src_len]
        max_len: 생성할 최대 길이
        start_token: 수열 시작 토큰의 번호
        end_token: 수열 끝 토큰의 번호
        pad_idx: 채움 토큰의 번호
    
    반환값:
        generated: 만들어진 토큰 수열 [1, gen_len]
    """
    model.eval()
    device = src.device
    
    # 원문을 인코딩한다 (한 번만 셈한다)
    src_pad_mask = (src == pad_idx)                    # [1, src_len]
    memory = model.encode(src, src_key_padding_mask=src_pad_mask)
    
    # 디코더 입력을 <start> 토큰으로 시작한다
    generated = torch.tensor([[start_token]], device=device)  # [1, 1]
    
    for _ in range(max_len):
        tgt_len = generated.size(1)
        
        # 지금 수열 길이에 맞는 인과 가림을 만든다
        tgt_mask = torch.triu(                        # [tgt_len, tgt_len]
            torch.ones(tgt_len, tgt_len, device=device),
            diagonal=1
        ).bool()
        
        # 디코딩
        output = model.decode(                        # [1, tgt_len, vocab_size]
            generated, memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=src_pad_mask
        )
        
        # 마지막 자리의 예측을 얻는다
        next_token_logits = output[:, -1, :]          # [1, vocab_size]
        next_token = next_token_logits.argmax(dim=-1, keepdim=True)  # [1, 1]
        
        # 만들어진 수열에 덧붙인다
        generated = torch.cat([generated, next_token], dim=1)  # [1, tgt_len + 1]
        
        # <end> 토큰이 나오면 멈춘다
        if next_token.item() == end_token:
            break
    
    return generated

@torch.no_grad()
def translate(
    model,
    src_tokens: torch.Tensor,
    idx_to_token: dict,
    max_len: int = 100,
    start_token: int = 1,
    end_token: int = 2
) -> str:
    """
    원문 수열을 글로 옮긴다.
    
    인수:
        model: 학습된 트랜스포머 모형
        src_tokens: 원문 토큰 번호 [1, src_len]
        idx_to_token: 토큰 색인에서 글자로의 잇댐
        max_len: 생성할 최대 길이
        start_token: 시작 토큰의 번호
        end_token: 끝 토큰의 번호
    
    반환값:
        translated_text: 만들어진 번역 글
    """
    generated = greedy_decode(
        model, src_tokens, max_len, start_token, end_token
    )
    
    # <start>와 <end>를 빼고 토큰 번호를 글자로 바꾼다
    tokens = generated.squeeze().tolist()
    words = [
        idx_to_token.get(idx, "<unk>")
        for idx in tokens
        if idx not in (start_token, end_token, 0)
    ]
    
    return " ".join(words)
```

---

## 4. 학습과 추론 견주기

| 측면 | 학습 | 추론 |
|--------|----------|-----------|
| **인코더 입력** | 원문 수열 $x$ | 원문 수열 $x$ |
| **디코더 입력** | 오른쪽으로 민 참값 $y_{\text{shifted}}$ | 앞서 만든 토큰 $\hat{y}_{<t}$ |
| **목표가 필요한가** | 그렇다 (손실 셈에) | 아니다 |
| **병렬성** | 온전함 (모든 자리를 한꺼번에 셈한다) | 차례대로 (단계마다 토큰 하나) |
| **방식** | 스승 강제 | 자기 회귀 (탐욕, 빔, 표집) |
| **인과 가림** | 앞을 엿보지 못하게 학습 중에 적용한다 | 생성 단계마다 적용한다 |
| **계산** | 수열 전체에 앞먹임 한 번 | 길이 $T$의 출력에 앞먹임 $T$번 |

---

## 5. 완전한 학습 예제

```python
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class PositionalEncoding(nn.Module):
    """사인파 위치 인코딩."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))     # [1, max_len, d_model]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerSeq2Seq(nn.Module):
    """임베딩 층과 출력 층을 갖춘 nn.Transformer의 감싸개."""
    
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float
    ):
        super().__init__()
        self.d_model = d_model
        
        # 임베딩과 위치 인코딩
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # 트랜스포머 알맹이
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # 출력 사영
        self.output_proj = nn.Linear(d_model, tgt_vocab_size)
    
    def encode(self, src, src_key_padding_mask=None):
        """원문 수열을 인코딩한다."""
        x = self.src_embed(src) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        return self.transformer.encoder(x, src_key_padding_mask=src_key_padding_mask)
    
    def decode(self, tgt, memory, tgt_mask=None, memory_key_padding_mask=None):
        """인코더의 기억이 주어졌을 때 목표 수열을 디코딩한다."""
        x = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer.decoder(
            x, memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        return self.output_proj(x)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None,
                src_key_padding_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        """온전한 앞먹임."""
        src_emb = self.src_embed(src) * math.sqrt(self.d_model)
        tgt_emb = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)
        tgt_emb = self.pos_encoder(tgt_emb)
        
        output = self.transformer(
            src_emb, tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        return self.output_proj(output)

def build_transformer(
    src_vocab_size: int,
    tgt_vocab_size: int,
    d_model: int = 512,
    nhead: int = 8,
    num_encoder_layers: int = 6,
    num_decoder_layers: int = 6,
    dim_feedforward: int = 2048,
    dropout: float = 0.1
) -> TransformerSeq2Seq:
    """
    사비에르 초기화를 쓴 트랜스포머 모형을 세운다.
    
    임베딩 층과 출력 사영을 갖춘 TransformerSeq2Seq를 돌려준다.
    """
    model = TransformerSeq2Seq(
        src_vocab_size, tgt_vocab_size,
        d_model, nhead,
        num_encoder_layers, num_decoder_layers,
        dim_feedforward, dropout
    )
    
    # 사비에르 초기화
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return model

# 학습 고리 보기
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 초매개변수
    src_vocab_size = 5000
    tgt_vocab_size = 5000
    d_model = 256
    nhead = 8
    num_layers = 3
    batch_size = 32
    num_epochs = 20
    learning_rate = 1e-4
    
    # 모형을 세운다
    model = build_transformer(
        src_vocab_size, tgt_vocab_size,
        d_model, nhead, num_layers, num_layers
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 채움을 무시한다
    
    # 임시 데이터 (실제 데이터셋으로 바꾸어라)
    src_data = torch.randint(1, src_vocab_size, (640, 20))  # 표본 640개, 길이 20
    tgt_data = torch.randint(1, tgt_vocab_size, (640, 22))  # <start>와 <end>를 넣는다
    
    dataset = TensorDataset(src_data, tgt_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 학습
    for epoch in range(num_epochs):
        avg_loss = train_epoch(model, dataloader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
```

---

## 6. 이름표 평활화

본디 트랜스포머 논문은 $\epsilon = 0.1$의 이름표 매끄럽게 하기를 쓰는데, 확률의 작은 몫을 어휘의 모든 토큰에 흩뿌려 모형이 지나치게 자신하지 않게 한다.

$$
y_{\text{smooth}}(k) = (1 - \epsilon) \cdot \mathbf{1}_{k=y} + \frac{\epsilon}{|\mathcal{V}|}
$$

여기서 $|\mathcal{V}|$은 어휘 크기이고 $y$은 참 토큰이다.

```python
class LabelSmoothingLoss(nn.Module):
    """레이블 평활화를 쓰는 교차 엔트로피 손실."""
    
    def __init__(self, vocab_size: int, smoothing: float = 0.1, pad_idx: int = 0):
        super().__init__()
        self.smoothing = smoothing
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
    
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        인수:
            logits: [batch_size * seq_len, vocab_size]
            target: [batch_size * seq_len]
        """
        log_probs = torch.log_softmax(logits, dim=-1)       # [N, vocab_size]
        
        # 매끄럽게 한 목표 분포를 만든다
        smooth_target = torch.full_like(log_probs, self.smoothing / (self.vocab_size - 1))
        smooth_target.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        
        # 채운 자리를 0으로 만든다
        non_pad_mask = target != self.pad_idx                # [N]
        smooth_target[~non_pad_mask] = 0.0
        
        # KL 발산 손실
        loss = -(smooth_target * log_probs).sum(dim=-1)      # [N]
        loss = loss[non_pad_mask].mean()
        
        return loss
```

---

## 연습문제

**연습문제 1.**
표준 트랜스포머 학습 파이프라인을 데이터 마련, 토큰 나누기, 배치 묶기, 학습 고리로 나누어 설명하라.

??? success "연습문제 1 풀이"

    1. 글을 토큰으로 나눈다(BPE/WordPiece). 2. 배치 안에서 수열을 같은 길이로 채운다. 3. 채움을 위한 주의 가림을 만든다. 4. 트랜스포머로 앞먹임한다. 5. 다음 토큰 예측(또는 가린 토큰)에 대해 교차 엔트로피 손실을 셈한다. 6. 역전파하고 AdamW에 예열과 코사인 감쇠를 곁들여 갱신한다.

---

**연습문제 2.**
효율적인 자기 회귀 추론을 위한 KV 캐싱을 설명하라.

??? success "연습문제 2 풀이"
    생성 중에 단계마다 모든 열쇠·값 쌍을 다시 셈하는 것은 아깝다. KV 캐시는 앞선 모든 자리의 K, V 행렬을 담아 둔다. 새 토큰마다 그 Q, K, V만 셈하고 K, V를 캐시에 덧붙인 뒤 캐시에 담긴 K, V 전체에 주의한다. 그러면 단계마다의 계산이 $O(n^2 d)$에서 $O(nd)$으로 준다.

---

**연습문제 3.**
스승 강제와 자기 회귀 디코딩의 차이는 무엇인가?

??? success "연습문제 3 풀이"
    스승 강제(학습)는 참 토큰을 디코더 입력으로 넣고 다음 토큰을 맞힌다. 빠르고 안정적이지만 노출 편향이 생긴다. 자기 회귀 디코딩(추론)은 모형 자신의 예측을 입력으로 넣는다. 느리고 오차가 쌓일 수 있다. 예정된 표집은 학습 중에 둘을 섞어 그 틈을 메운다.

---

**연습문제 4.**
파이토치에서 트랜스포머 언어 모형의 간단한 학습 단계를 구현하라.

??? success "연습문제 4 풀이"
    ```python
    logits = model(input_ids[:, :-1])  # 다음 토큰을 맞힌다
    targets = input_ids[:, 1:]  # 민 목표
    loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    ```

## 정리하며

트랜스포머의 학습 과정과 추론 과정은 디코더를 다루는 방식에서 근본적으로 다르다.

1. **학습**은 오른쪽으로 민 참값과 함께 스승 강제를 써서 자리에 걸쳐 온전히 병렬로 하고 참인 다음 토큰에 대해 손실을 셈한다.
2. **추론**은 자기 회귀로 만들며 단계마다 토큰 하나를 내고 그 예측을 디코더 입력으로 되먹인다.
3. **이름표 매끄럽게 하기**는 지나친 자신을 막고 규제 노릇을 한다.
4. **가림 만들기**는 (길이가 제각각인 수열을 위한) 채움 가림과 (자기 회귀 생성을 위한) 인과 가림을 함께 쓴다.

이 구분을 이해하는 것은 트랜스포머 기반 체계를 구현하고 고치고 다듬는 데 꼭 필요하다.

**참고 문헌**

1. Vaswani, A., et al. (2017). "Attention Is All You Need." NeurIPS.
2. Szegedy, C., et al. (2016). "Rethinking the Inception Architecture for Computer Vision." CVPR. (Label smoothing)
3. Williams, R. J., & Zipser, D. (1989). "A Learning Algorithm for Continually Running Fully Recurrent Neural Networks." (Teacher forcing)
