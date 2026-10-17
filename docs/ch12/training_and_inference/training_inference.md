# 트랜스포머의 학습과 추론

인코더·디코더 번역 모델의 학습과 자기 회귀 생성을 비교한다. 학습에서는 정답 접두를 입력하는 티처 포싱으로 여러 목표 위치를 병렬 계산할 수 있다. 생성에서는 새로 선택한 토큰을 다음 입력에 추가한다. 모든 트랜스포머 과제가 번역처럼 원문·목표 쌍을 쓰는 것은 아니다.

## 1. 학습 파이프라인

### 입력 쌍

이 절의 Seq2Seq 모델은 원문·목표 쌍 (x,y)을 사용한다. 인과 언어 모델은 한 텍스트를 이동해 입력·목표를 만들고, MLM은 손상 입력에서 일부 원래 토큰을 복원하는 등 학습 구성은 목표마다 다르다.

- $x$: 원문 시퀀스 (이를테면 영어 문장)
- $y$: 목표 시퀀스 (이를테면 프랑스어 번역)

인코더는 $x$을 그대로 받고 디코더는 $y$을 **오른쪽으로 민** 것을 받는다.

### 티처 포싱

티처 포싱는 학습 중에 모델 자신의 예측 대신 참 목표 시퀀스을 디코더에 넣는 것이다. 그러면 모든 목표 자리를 한꺼번에 병렬로 셈할 수 있다.

목표 시퀀스 $y = [y_1, y_2, \ldots, y_T]$에 대해 다음과 같다.

- **디코더 입력**(오른쪽으로 민 것): $[\langle\text{start}\rangle,y_1,y_2,\ldots,y_T]$
- **참 레이블**: $[y_1, y_2, \ldots, y_T, \langle\text{end}\rangle]$

디코더 입력과 정답은 같은 길이를 갖게 정렬한다. 위처럼 EOS를 정답에 추가하면 디코더 입력에도 마지막 원래 토큰 y_T까지 포함해야 한다. 여기에 인과 마스크를 함께 적용해야 정답 토큰 자체나 미래 목표에 접근하지 못한다.

### 보기: 기계 번역

"The cat sat on the mat"을 "Le chat était assis sur le tapis"로 옮길 때 다음과 같다.

$$
\begin{aligned}
\text{Encoder input } (x) &: [\text{The}, \text{cat}, \text{sat}, \text{on}, \text{the}, \text{mat}] \\
\text{Decoder input } (y_{\text{shifted}}) &: [\langle\text{start}\rangle,\text{Le},\text{chat},\text{était},\text{assis},\text{sur},\text{le},\text{tapis}] \\
\text{Target labels} &: [\text{Le}, \text{chat}, \text{était}, \text{assis}, \text{sur}, \text{le}, \text{tapis}, \langle\text{end}\rangle]
\end{aligned}
$$

### 로스 계산

다음 식은 EOS를 제외한 목표 위치의 음의 로그 확률 합이다. 실제 구현에서는 EOS도 유효 목표에 포함하고 PAD를 제외하며, 유효 토큰 평균인지 합인지 집계 규칙을 명시한다.

$$
\mathcal{L} = -\sum_{t=1}^{T} \log P_\theta(y_t \mid y_{<t}, x)
$$

여기서 $P_\theta(y_t \mid y_{<t}, x)$은 앞선 목표 토큰 $y_{<t}$과 원문 시퀀스 $x$ 전체를 조건으로 하여 위치 $t$의 올바른 토큰 $y_t$에 모델이 매긴 확률이다.

위치마다 디코더는 어휘 전체에 대한 확률 분포를 낸다. 교차 엔트로피 로스는 모델이 올바른 다음 토큰에 낮은 확률을 매기면 벌을 준다.

### 임베딩 학습

단어 임베딩은 대개 트랜스포머의 나머지와 함께 학습한다. 임베딩 레이어는 무작위로 초기화하고 역전파로 갱신한다. 흔한 방식이 셋 있다.

1. **맨바닥부터 학습**: 무작위 초기화. 임베딩이 학습 중에 과제에 맞는 표현을 배운다.
2. **사전 학습 웨이트 사용**: 같은 어휘·토큰화·차원을 갖는 임베딩이나 모델 체크포인트를 불러와 갱신하거나 동결한다. 문맥 임베딩은 문장과 위치마다 달라 단일 고정 어휘 테이블과 동일하게 직접 복사할 수는 없다.

---

## 2. 파이토치 학습 구현

아래 train_step은 tgt에 BOS·목표 토큰·EOS가 포함된 구성을 가정한다. 모델이 src/tgt_key_padding_mask와 memory_key_padding_mask를 받고 어휘 로짓을 반환해야 한다. 앞의 불완전한 Transformer 클래스와 인터페이스가 다르며 이 페이지 뒤의 TransformerSeq2Seq와 연결해 읽는다.

패딩 마스크의 True는 차단이다. 로스 함수의 ignore_index는 tgt_pad_idx와 맞춰야 하며 mask를 전달하는 것만으로 PAD 정답이 제외되지는 않는다. train_epoch는 배치 평균 로스를 단순 평균하므로 유효 목표 길이·배치 크기가 다르면 전체 토큰 평균과 다르다. 빈 로더·유효 정답이 없는 배치도 별도 처리한다.

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
        criterion: 로스 함수(CrossEntropyLoss)
        src: 원문 토큰 [batch_size, src_len]
        tgt: 목표 수열 전체 [batch_size, tgt_len]
        src_pad_idx: 원문의 채움 색인
        tgt_pad_idx: 목표문의 채움 색인
    
    반환값:
        loss: 스칼라 로스 값
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
    
    # 로스를 셈한다 (교차 엔트로피를 위해 편다)
    vocab_size = output.size(-1)
    loss = criterion(
        output.reshape(-1, vocab_size),               # [batch_size * (tgt_len-1), vocab_size]
        tgt_labels.reshape(-1)                        # [batch_size * (tgt_len-1)]
    )
    
    # 역전파와 파라메터 갱신
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
        criterion: 로스 함수
        device: torch.device
    
    반환값:
        average_loss: 모든 배치에 걸친 평균 로스
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

추론 중에는 원문 시퀀스 $x$만 주어진다. 디코더는 한 번에 토큰 하나씩 만들며 맞힌 토큰마다 다음 단계의 입력으로 되먹인다.

생성 고리는 다음과 같이 돈다.

$$
\hat{y}_t = \arg\max_{v \in \mathcal{V}} P_\theta(v \mid \hat{y}_1, \ldots, \hat{y}_{t-1}, x)
$$

1. 인코더로 원문 시퀀스 $x$을 인코딩한다(한 번만 한다)
2. 디코더 입력을 $\langle\text{start}\rangle$ 토큰으로 시작한다
3. 단계 $t$마다 다음을 한다.
   - 지금까지 만든 토큰 $[\langle\text{start}\rangle, \hat{y}_1, \ldots, \hat{y}_{t-1}]$을 모두 디코더에 넣는다
   - 마지막 자리의 예측을 꺼낸다
   - 다음 토큰을 고른다(탐욕 디코딩, 빔 탐색, 또는 샘플링으로)
   - 고른 토큰을 디코더 입력에 덧붙인다
4. $\langle\text{end}\rangle$이 나오거나 최대 길이에 이르면 멈춘다

### 파이토치 추론 구현

이 greedy_decode는 배치 1개만 지원한다. generated를 (1,1)로 만들고 next_token.item()으로 종료를 판단하기 때문이다. max_len은 새 토큰 반복 횟수라 반환에는 시작 토큰까지 포함된다. model.decode는 은닉 표현이 아니라 어휘 로짓을 반환해야 하며 캐시는 사용하지 않는다.

translate는 PAD를 0으로 고정해 제거하고 토큰을 공백으로 연결한다. 서브워드 토크나이저의 실제 복호화와 같지 않으며, 생성 0회이면 squeeze가 스칼라를 만들어 후속 순회가 실패할 수 있다.

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

## 4. 학습과 추론 비교

| 측면 | 학습 | 추론 |
|--------|----------|-----------|
| **인코더 입력** | 원문 시퀀스 $x$ | 원문 시퀀스 $x$ |
| **디코더 입력** | 오른쪽으로 민 참값 $y_{\text{shifted}}$ | 앞서 만든 토큰 $\hat{y}_{<t}$ |
| **목표가 필요한가** | 그렇다 (로스 계산에) | 아니다 |
| **병렬성** | 온전함 (모든 자리를 한꺼번에 계산한다) | 차례대로 (단계마다 토큰 하나) |
| **방식** | 티처 포싱 | 자기 회귀 (탐욕, 빔, 샘플링) |
| **인과 마스크** | 앞을 엿보지 못하게 학습 중에 적용한다 | 생성 단계마다 적용한다 |
| **계산** | 시퀀스 전체에 순방향 한 번 | 길이 $T$의 출력에 순방향 $T$번 |

---

## 5. 학습 루프 예제

아래 TransformerSeq2Seq는 PyTorch 인코더·디코더와 목표 시프트·마스크를 연결한다. 다만 main은 무작위 원문·목표를 독립적으로 생성하며 BOS·EOS 구성이나 실제 번역 관계가 없다. 보존된 로스 출력은 번역 성능 평가가 아니다. 검증 분할·모델 선택·번역 평가도 포함하지 않는다.

사인·코사인 표는 짝수 피처 차원과 max_len 이내의 입력을 가정한다. 모델 forward와 decode의 마스크 인수가 다르므로 학습·생성 호출 규약을 구분한다.

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
    """임베딩 레이어와 출력 레이어를 갖춘 nn.Transformer의 감싸개."""
    
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
        
        # 트랜스포머 커널
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
        """온전한 피드포워드."""
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
    
    임베딩 레이어와 출력 사영을 갖춘 TransformerSeq2Seq를 돌려준다.
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
    
    # 초파라메터
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

**출력:**

```
Epoch 1/20, Loss: 8.5710
Epoch 2/20, Loss: 8.4871
Epoch 3/20, Loss: 8.4326
Epoch 4/20, Loss: 8.3878
Epoch 5/20, Loss: 8.3326
Epoch 6/20, Loss: 8.2584
Epoch 7/20, Loss: 8.1662
Epoch 8/20, Loss: 8.0654
Epoch 9/20, Loss: 7.9797
Epoch 10/20, Loss: 7.8929
Epoch 11/20, Loss: 7.8092
Epoch 12/20, Loss: 7.7274
Epoch 13/20, Loss: 7.6537
Epoch 14/20, Loss: 7.5746
Epoch 15/20, Loss: 7.4908
Epoch 16/20, Loss: 7.4138
Epoch 17/20, Loss: 7.3334
Epoch 18/20, Loss: 7.2542
Epoch 19/20, Loss: 7.1734
Epoch 20/20, Loss: 7.0938
```

---

## 6. 레이블 스무딩

아래 LabelSmoothingLoss는 정답 확률을 1−epsilon, 나머지를 epsilon/(V−1)로 두어 위의 균등 혼합 식과 다르다. 유효 행에서도 PAD 클래스에 스무딩 확률을 주며 PAD 정답 행만 제외한다. vocab_size>1, 0≤smoothing≤1, logits=(N,V), target=(N)의 정수 ID를 가정한다. target=-100을 scatter 전에 처리하지 않으므로 그 값을 직접 넘기면 실패하고, 모두 PAD인 입력의 mean도 유효하지 않다.

레이블 스무딩은 정답 원핫 분포의 일부 질량을 다른 클래스에 나눠 주는 정규화 방법이다. 아래 식은 전체 어휘의 균등 분포와 섞는 정의다. 정답 외 클래스에만 나누는 정의와 PAD 제외 방식도 있으므로 식·구현을 맞춰야 한다.

$$
y_{\text{smooth}}(k) = (1 - \epsilon) \cdot \mathbf{1}_{k=y} + \frac{\epsilon}{|\mathcal{V}|}
$$

여기서 $|\mathcal{V}|$은 어휘 크기이고 $y$은 참 토큰이다.

```python
class LabelSmoothingLoss(nn.Module):
    """레이블 평활화를 쓰는 교차 엔트로피 로스."""
    
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
        
        # KL 발산 로스
        loss = -(smooth_target * log_probs).sum(dim=-1)      # [N]
        loss = loss[non_pad_mask].mean()
        
        return loss
```

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
Seq2Seq 학습에서 데이터·목표 정렬·마스크·로스를 연결해 설명하라.

</div>

??? success "풀이"
    분할한 원문·목표를 토큰화하고 목표에 BOS·EOS를 구성한다. 입력은 tgt[:-1], 정답은 tgt[1:]로 정렬하고 원문·목표·메모리 PAD 마스크와 인과 마스크를 적용한다. 로짓에 PAD 제외 교차 엔트로피를 계산해 역전파·갱신한다. 옵티마이저·스케줄은 선택이며 항상 AdamW·코사인이 필수인 것은 아니다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
KV 캐시가 한 단계와 전체 생성에 주는 비용 차이를 설명하라.

</div>

??? success "풀이"
    이전 레이어별 K·V를 저장해 새 토큰의 투영·계산만 추가한다. 길이 t의 한 쿼리 어텐션은 피처 차원 고정 시 O(t), 전체 n개 생성은 어텐션만 O(n²)다. 캐시 메모리도 길이에 선형으로 늘며 원문·목표 위치와 패딩 규약을 맞춰야 한다. 이 페이지 greedy_decode는 캐시를 실제로 사용하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff easy" title="쉬움"></span>
티처 포싱과 자기 회귀 생성의 문맥 차이 및 scheduled sampling의 한계를 설명하라.

</div>

??? success "풀이"
    티처 포싱은 정답 과거 토큰, 생성은 모델이 선택한 과거 토큰을 입력한다. 그 차이로 생성 오류가 이후 문맥에 영향을 줄 수 있다. Scheduled sampling은 둘을 섞는 시도지만 항상 문제를 해결하거나 품질을 높인다는 보장은 없고 트랜스포머의 완전 병렬 학습 구성도 달라질 수 있다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 인과 언어 모델 학습 조각이 가정하는 모델·마스크·정답·갱신 조건을 설명하라.

</div>

??? success "풀이"
    model·input_ids·vocab_size·optimizer·scheduler와 F는 외부에 정의되어야 한다. 모델은 미래 접근을 차단한 로짓을 반환하고 PAD 정답을 로스에서 제외해야 한다. 조각은 함수 시작의 경사 초기화를 보여 주지 않으므로 이전 배치의 경사가 남지 않게 해야 하며, 모델이 내부 시프트하는 API라면 외부 시프트와 중복하지 않는다.

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

티처 포싱 학습은 정답 입력·목표 시프트와 인과 마스크를 함께 사용한다. 생성은 선택 토큰을 입력에 이어 붙이며 배치·캐시·종료 규약을 따로 설계한다. 무작위 데이터 학습 루프와 번역 품질을 구분하고, 스무딩 식과 로스의 PAD·평균 처리도 일치시킨다.

**참고 문헌**

1. Vaswani, A., et al. (2017). "Attention Is All You Need." NeurIPS.
2. Szegedy, C., et al. (2016). "Rethinking the Inception Architecture for Computer Vision." CVPR. (Label smoothing)
3. Williams, R. J., & Zipser, D. (1989). "A Learning Algorithm for Continually Running Fully Recurrent Neural Networks." (Teacher forcing)
