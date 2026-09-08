# 트랜스포머의 변형

| 구조 | 병렬화 | 먼 거리 | 복잡도 | 귀납 편향 |
|--------------|-----------------|------------|------------|----------------|
| 순환 신경망/LSTM | 차례대로 | 어렵다 | O(n) | 시간 |
| 합성곱 신경망 | 병렬 | 제한적 | O(n) | 국소 무늬 |
| 트랜스포머 | 병렬 | 쉽다 | O(n²) | 없음 (배운다) |

---

## 1. 순환 신경망/LSTM

### 구조

```
x₁ → [h₁] → x₂ → [h₂] → x₃ → [h₃] → ...
        ↓         ↓         ↓
       y₁        y₂        y₃
```

### 성질

```python
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        x = self.embed(x)
        out, (h, c) = self.lstm(x)
        return self.fc(out)
```

**좋은 점:**

- O(n) 복잡도
- 수열에 좋은 귀납 편향
- 기억을 아낀다

**나쁜 점:**

- 차례대로 처리한다 (학습이 느리다)
- 긴 수열에서 기울기가 사라진다
- 병렬로 하기 어렵다

---

## 2. 수열을 위한 합성곱 신경망

### 구조

```
[x₁ x₂ x₃ x₄ x₅]
   \_____/
    Conv1
      \_____/
       Conv2
         ...
```

### 성질

```python
class CNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_filters, kernel_sizes):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k, padding=k//2)
            for k in kernel_sizes
        ])
        self.fc = nn.Linear(num_filters * len(kernel_sizes), vocab_size)
    
    def forward(self, x):
        x = self.embed(x).transpose(1, 2)  # [B, C, L]
        conv_outs = [F.relu(conv(x)) for conv in self.convs]
        pooled = [F.max_pool1d(c, c.size(2)).squeeze(2) for c in conv_outs]
        return self.fc(torch.cat(pooled, dim=1))
```

**좋은 점:**

- 온전히 병렬로 할 수 있다
- 국소 무늬에 좋다
- 크기가 고정된 맥락에 효율적이다

**나쁜 점:**

- 받는 영역이 좁다 (층이 많이 필요하다)
- 아주 먼 의존에는 알맞지 않다

---

## 3. 트랜스포머

### 구조

```
[x₁ x₂ x₃ x₄ x₅]
    ↓↓↓↓↓
Self-Attention (all-to-all)
    ↓↓↓↓↓
[y₁ y₂ y₃ y₄ y₅]
```

### 성질

```python
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(512, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        pos = torch.arange(x.size(1), device=x.device)
        x = self.embed(x) + self.pos(pos)
        return self.fc(self.transformer(x))
```

**좋은 점:**

- 온전히 병렬로 할 수 있다
- 먼 거리를 곧바로 잇는다
- 표현력이 매우 좋다

**나쁜 점:**

- O(n²) 복잡도
- 차례에 대한 붙박이 편향이 없다
- 긴 수열에서 기억을 많이 쓴다

---

## 4. 자세히 견주기

### 복잡도 분석

| 연산 | 순환 신경망 | 합성곱 신경망 | 트랜스포머 |
|-----------|-----|-----|-------------|
| 차례로 하는 연산 | O(n) | O(1) | O(1) |
| 층마다의 복잡도 | O(n·d²) | O(k·n·d²) | O(n²·d) |
| 최대 경로 길이 | O(n) | O(log_k(n)) | O(1) |

### 기억 사용

수열 길이가 n이고 차원이 d일 때 다음과 같다.

| 모형 | 학습 기억 | 추론 기억 |
|-------|-----------------|------------------|
| 순환 신경망 | O(n·d) | O(d) |
| 합성곱 신경망 | O(n·d) | O(n·d) |
| 트랜스포머 | O(n²·h + n·d) | KV 캐시를 쓰면 O(n²·h) |

### 수열 길이에 따른 성능

```
Short (< 512):    Transformer ≈ CNN > RNN
Medium (512-2K):  Transformer > CNN > RNN
Long (2K-8K):     Efficient Transformers > CNN > RNN
Very Long (>8K):  State Space Models / Linear Attention
```

---

## 5. 성능 수치 (어림값)

### 언어 모형화 (당혹도, 낮을수록 좋다)

| 모형 | WikiText-103 | 매개변수 |
|-------|--------------|------------|
| LSTM | 약 35 | 1억 5000만 |
| 트랜스포머 | 약 18 | 1억 5000만 |
| GPT-2 | 약 15 | 15억 |

### 글 분류 (정확도)

| 모형 | IMDB | SST-2 |
|-------|------|-------|
| LSTM | 89% | 87% |
| 합성곱 신경망 | 90% | 88% |
| BERT | 95% | 94% |

### 기계 번역 (BLEU)

| 모형 | WMT 영어-독일어 |
|-------|-----------|
| LSTM 수열 대 수열 | 28.4 |
| 트랜스포머 | 34.4 |

---

## 6. 혼합형 접근

### 트랜스포머와 합성곱 신경망

```python
class ConvTransformer(nn.Module):
    """국소 합성곱과 전역 주의."""
    def __init__(self, d_model, num_heads, kernel_size=3):
        super().__init__()
        self.local_conv = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size//2)
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
    
    def forward(self, x):
        # 국소 특징
        local = self.local_conv(x.transpose(1, 2)).transpose(1, 2)
        # 전역 주의
        global_out, _ = self.attention(x, x, x)
        return local + global_out
```

### 트랜스포머와 순환 신경망

```python
class TransformerWithRecurrence(nn.Module):
    """되풀이 기억을 갖춘 트랜스포머."""
    def __init__(self, d_model, num_heads, memory_size):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.memory_rnn = nn.GRU(d_model, d_model, batch_first=True)
        self.memory = None
    
    def forward(self, x):
        # 기억을 고친다
        if self.memory is not None:
            x = torch.cat([self.memory, x], dim=1)
        out, _ = self.attention(x, x, x)
        # 기억으로 눌러 담는다
        self.memory, _ = self.memory_rnn(out)
        return out[:, -x.size(1):, :]
```

---

## 7. 언제 무엇을 쓰는가

### 순환 신경망을 쓸 때
- 흘러드는 데이터 (실시간 처리)
- 기억이 매우 빠듯할 때
- 수열의 차례가 매우 중요할 때
- 짧은 수열 (100 미만)

### 합성곱 신경망을 쓸 때
- 국소 무늬가 가장 중요할 때
- 크기가 고정된 분류
- 빠른 추론이 필요할 때
- 수열 길이가 어지간할 때

### 트랜스포머를 쓸 때
- 먼 거리 의존이 중요할 때
- 병렬 학습을 할 수 있을 때
- 최고 수준이 필요할 때
- 사전 학습된 모형을 쓸 수 있을 때

---

## 8. 요즘의 대안

아주 긴 수열에서는 트랜스포머 같은 성능을 지키면서 일차 복잡도를 주는 구조가 여럿 있다.

| 모형 | 복잡도 | 방식 | 핵심 혁신 |
|-------|------------|----------|----------------|
| 선형 주의 | $O(n)$ | 소프트맥스의 핵 어림 | $\text{softmax}(QK^T)V$을 $\phi(Q)(\phi(K)^TV)$으로 바꾼다 |
| Mamba/S4 | $O(n)$ | 가려 뽑는 상태 공간 모형 | 하드웨어를 고려한 훑기와 데이터에 따른 상태 전이 |
| RWKV | $O(n)$ | 트랜스포머처럼 학습하는 순환 신경망 | 되풀이로 셈할 수 있는 선형 주의 정식화 |
| Longformer | $O(n \cdot w)$ | 국소와 전역을 섞은 성긴 주의 | 과제에 맞는 전역 토큰을 곁들인 미끄러지는 창 주의 |
| Hyena | $O(n \log n)$ | 긴 합성곱 | 주의를 암묵적으로 매개변수화한 합성곱으로 바꾼다 |

맥락 창이 토큰 10만 개 이상으로 늘어 표준 트랜스포머 주의를 쓰기 어려워지는 지금 이 방법들이 특히 중요하다.

---

## 9. 성긴 주의 무늬

핵심 통찰은 온전한 주의 행렬이 경험상 성기다는 것이다. 주의 가중치 대부분이 자리의 작은 일부에 몰린다. 성긴 주의는 주의 무늬를 미리 제한하여 이를 정식화하고, 가중치가 낮은 항목의 계산을 아예 건너뛴다.

##### 복잡도 문제

수열 길이가 $N$이고 모형 차원이 $d$일 때 다음과 같다.

| 부품 | 시간 복잡도 | 기억 |
|-----------|----------------|--------|
| 온전한 주의 | O(N²d) | O(N²) |
| **목표** | O(Nd) 또는 O(N log N · d) | O(N) |

(어지간한 문서인) 토큰 $N = 16{,}384$개의 수열에서 온전한 주의 행렬의 항목은 $2.7 \times 10^8$개이다. 32비트 정밀도라면 머리 하나에 대한 이 행렬을 담아 두는 데만 약 1GB가 든다.

##### 흔한 성긴 무늬

##### 1. 국소(미끄러지는 창) 주의

자리마다 창 안의 가까운 자리에만 주의한다.

$$
\text{Attend}(i) = \{j : |i - j| \leq w\}
$$

**복잡도**: O(Nw)이며 여기서 $w$은 창 크기이다

```
Window size = 3:
Position 5 attends to: [2, 3, 4, 5, 6, 7, 8]
```

**까닭**: 자연어의 의존 대부분은 국소적이다. 미끄러지는 창은 이를 잡아내면서 뚜렷한 주의 가중치를 좀처럼 받지 않는 먼 자리를 무시한다.

##### 2. 팽창(걸음 있는) 주의

정해진 간격의 자리에 주의하여 온전한 주의 없이도 먼 거리를 아우른다.

$$
\text{Attend}(i) = \{j : (i - j) \mod d = 0, |i-j| \leq w \cdot d\}
$$

**복잡도**: O(Nw)

```
Dilation = 2, Window = 3:
Position 6 attends to: [0, 2, 4, 6, 8, 10, 12]
```

**까닭**: 팽창 합성곱과 비슷하다. $d$번째 자리마다 주의하면 계산을 늘리지 않고도 실효 수용 영역이 $d$배로 커진다.

##### 3. 전역 주의

어떤 자리를 모든 자리와 서로 주의하는 "전역"으로 정한다.

$$
\text{Attend}(i) = \begin{cases}
\{1, ..., N\} & \text{if } i \in \mathcal{G} \\
\mathcal{G} \cup \text{Local}(i) & \text{otherwise}
\end{cases}
$$

Longformer와 BigBird가 쓴다. 흔한 전역 토큰으로는 `[CLS]`, 과제에 맞는 토큰, 구간마다의 첫 토큰이나 마지막 토큰이 있다.

##### 4. 블록 성긴 주의

수열을 블록으로 나누고 정해진 블록 안에서 또는 블록 사이에서 주의한다.

```
Block pattern:
[■ ■ □ □ ■]
[■ ■ ■ □ □]
[□ ■ ■ ■ □]
[□ □ ■ ■ ■]
[■ □ □ ■ ■]
```

##### 5. 무작위 주의

자리마다 정해진 수의 다른 자리에 무작위로 주의한다.

$$
\text{Attend}(i) = \text{Random}(k) \cup \text{Local}(i)
$$

**이론적인 뜻**: 무작위 주의는 두 토큰 사이의 기대 경로 길이를 $O(\log N)$으로 만들어 정보가 퍼지는 데 대한 이론적 보장(그래프 팽창 성질)을 준다.

##### 6. 위계(여러 크기) 주의

주의를 여러 알갱이 수준에 걸쳐 짜 놓는다.

**1수준**: 구간 안의 토큰 수준 국소 주의
**2수준**: 구간 요약 사이의 구간 수준 주의
**3수준**: 문서 요약 사이의 문서 수준 주의

이는 위계를 이루는 정보 흐름을 만든다.

$$\text{Local tokens} \to \text{Segment summaries} \to \text{Global representation}$$

위계 주의는 (절, 문단 같은) 짜임이 있는 문서를 자연스레 다루며, 잔 정보는 국소로 굵은 정보는 전역으로 처리하여 복잡도를 $O(N \sqrt{N})$ 이하로 줄인다.

**Sparse Transformer**(Child 외, 2019)는 주의 무늬를 성긴 부분 둘, 곧 국소 맥락에 주의하는 것과 걸음 있는 자리에 주의하는 것으로 인수분해하여 $O(N\sqrt{N})$ 복잡도를 이루며 이 길을 열었다.

##### 무늬 섞기

가장 잘 통하는 성긴 주의 방법은 여러 무늬를 섞는다. BigBird의 이론 결과(Zaheer 외, 2020)는 무작위와 국소와 전역 주의를 함께 쓰면 보편 근사에 넉넉함을 증명한다. 성긴 모형이 어떤 온전한 주의 트랜스포머든 흉내 낼 수 있다.

| 부품 | 목적 | 홀로 넉넉한가 |
|-----------|---------|-------------------|
| 국소 | 이웃한 의존을 잡아낸다 | 아니다 (먼 거리를 놓친다) |
| 전역 | 수열 전체의 정보를 모은다 | 아니다 (자리가 너무 적다) |
| 무작위 | 기대 경로 길이를 짧게 한다 | 아니다 (짜임을 놓친다) |
| **섞기** | **위의 모두** | **그렇다 (튜링 완전)** |

##### 파이토치 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List

def create_local_attention_mask(
    seq_len: int,
    window_size: int,
    device: torch.device = None
) -> torch.Tensor:
    """
    미끄러지는 창 주의 가림을 만든다.
    
    인수:
        seq_len: 수열 길이
        window_size: 주의 창의 크기(한쪽)
        device: 텐서를 둘 장치
        
    반환값:
        가림 [seq_len, seq_len]. True면 가린다(주의하지 않는다)
    """
    # 자리 번호를 만든다
    rows = torch.arange(seq_len, device=device).unsqueeze(1)
    cols = torch.arange(seq_len, device=device).unsqueeze(0)
    
    # 거리를 셈한다
    distance = torch.abs(rows - cols)
    
    # 창 바깥의 자리를 가린다
    mask = distance > window_size
    
    return mask

def create_dilated_attention_mask(
    seq_len: int,
    window_size: int,
    dilation: int,
    device: torch.device = None
) -> torch.Tensor:
    """
    팽창(걸음 있는) 주의 가림을 만든다.
    
    인수:
        seq_len: 수열 길이
        window_size: 주의할 자리의 개수
        dilation: 주의하는 자리 사이의 걸음
        device: 텐서를 둘 장치
    """
    rows = torch.arange(seq_len, device=device).unsqueeze(1)
    cols = torch.arange(seq_len, device=device).unsqueeze(0)
    
    # 팽창 공간에서의 거리
    distance = torch.abs(rows - cols)
    
    # 팽창 창 안에 있고 팽창 간격에 맞으면 쓸 수 있다
    within_window = distance <= window_size * dilation
    aligned = (rows - cols) % dilation == 0
    
    mask = ~(within_window & aligned)
    return mask

def create_block_sparse_mask(
    seq_len: int,
    block_size: int,
    num_random_blocks: int = 1,
    device: torch.device = None
) -> torch.Tensor:
    """
    블록 성긴 주의 가림을 만든다.
    
    블록마다 제 자신과 이웃한 블록과 무작위 블록에 주의한다.
    """
    num_blocks = (seq_len + block_size - 1) // block_size
    
    # 가림을 시작한다 (모두 가림)
    mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
    
    for i in range(num_blocks):
        i_start = i * block_size
        i_end = min((i + 1) * block_size, seq_len)
        
        for j in range(num_blocks):
            j_start = j * block_size
            j_end = min((j + 1) * block_size, seq_len)
            
            # 제 블록과 이웃한 블록
            if abs(i - j) <= 1:
                mask[i_start:i_end, j_start:j_end] = False
            
            # 무작위 블록
            elif torch.rand(1).item() < num_random_blocks / num_blocks:
                mask[i_start:i_end, j_start:j_end] = False
    
    return mask

class LocalAttention(nn.Module):
    """
    국소(미끄러지는 창) 주의.
    
    자리마다 정해진 창 안의 자리에만 주의한다.
    복잡도: O(N * window_size * d)
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        window_size: int,
        dropout: float = 0.0
    ):
        super().__init__()
        
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        """
        국소 주의를 쓰는 앞먹임.
        
        인수:
            x: 입력 [batch, seq_len, d_model]
            causal: 창 안에서 인과 가림을 적용할지 여부
        """
        batch_size, seq_len, _ = x.shape
        
        # Q, K, V를 사영한다
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 다중 머리에 맞게 꼴을 바꾼다
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 온전한 주의 점수를 셈한다
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # 국소 주의 가림을 만든다
        local_mask = create_local_attention_mask(seq_len, self.window_size, x.device)
        
        # 필요하면 인과 가림을 적용한다
        if causal:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            local_mask = local_mask | causal_mask
        
        # 가림을 적용한다
        scores = scores.masked_fill(local_mask, float('-inf'))
        
        # 소프트맥스를 하고 값에 적용한다
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, v)
        
        # 꼴을 바꾸고 사영한다
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.o_proj(output)

class LongformerAttention(nn.Module):
    """
    다음을 아우르는 Longformer 방식 주의:
    1. 국소 미끄러지는 창 주의 (모든 토큰에)
    2. 전역 주의 ([CLS] 같은 정해진 토큰에)
    
    복잡도: O(N * (w + g)). 여기서 w는 창, g는 전역 토큰이다
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        window_size: int,
        dropout: float = 0.0
    ):
        super().__init__()
        
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5
        
        # 국소 주의 사영
        self.q_local = nn.Linear(d_model, d_model)
        self.k_local = nn.Linear(d_model, d_model)
        self.v_local = nn.Linear(d_model, d_model)
        
        # 전역 주의 사영 (따로 둔 매개변수)
        self.q_global = nn.Linear(d_model, d_model)
        self.k_global = nn.Linear(d_model, d_model)
        self.v_global = nn.Linear(d_model, d_model)
        
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        global_attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        인수:
            x: 입력 [batch, seq_len, d_model]
            global_attention_mask: 참거짓 [batch, seq_len]. 전역 토큰이 True
        """
        batch_size, seq_len, _ = x.shape
        
        # 국소 사영
        q_local = self.q_local(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_local = self.k_local(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v_local = self.v_local(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 전역 사영
        q_global = self.q_global(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_global = self.k_global(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 국소 주의 점수를 셈한다
        local_scores = torch.matmul(q_local, k_local.transpose(-2, -1)) * self.scale
        
        # 국소 가림을 적용한다
        local_mask = create_local_attention_mask(seq_len, self.window_size, x.device)
        local_scores = local_scores.masked_fill(local_mask, float('-inf'))
        
        # 전역 자리는 모든 자리에 대한 주의를 셈한다
        # 다른 자리는 전역 자리에 대한 주의를 더한다
        global_indices = global_attention_mask.nonzero(as_tuple=True)
        
        if len(global_indices[0]) > 0:
            # 전역 자리는 모든 것에 주의할 수 있다
            for b, idx in zip(global_indices[0], global_indices[1]):
                local_scores[b, :, idx, :] = torch.matmul(
                    q_global[b, :, idx:idx+1, :],
                    k_global[b].transpose(-2, -1)
                ) * self.scale
                
                # 모든 자리가 전역 자리에 주의할 수 있다
                local_scores[b, :, :, idx] = torch.matmul(
                    q_local[b],
                    k_global[b, :, idx:idx+1, :].transpose(-2, -1)
                ).squeeze(-1) * self.scale
        
        # 소프트맥스
        attn_weights = F.softmax(local_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 출력을 셈한다
        output = torch.matmul(attn_weights, v_local)
        
        # 꼴을 바꾸고 사영한다
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.o_proj(output)

class BigBirdAttention(nn.Module):
    """
    다음을 아우르는 BigBird 방식 주의:
    1. 무작위 주의
    2. 창(국소) 주의
    3. 전역 주의
    
    O(N) 복잡도를 이룬다.
    수열 함수의 보편 근사기임이 이론으로 증명되었다.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        block_size: int = 64,
        num_global_tokens: int = 2,
        num_random_blocks: int = 3,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.block_size = block_size
        self.num_global_tokens = num_global_tokens
        self.num_random_blocks = num_random_blocks
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def _create_bigbird_mask(
        self,
        seq_len: int,
        device: torch.device
    ) -> torch.Tensor:
        """BigBird 성긴 주의 가림을 만든다."""
        
        # 모두 가린 채로 시작한다
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
        
        # 1. 전역 토큰 (앞의 num_global_tokens개 자리)
        mask[:self.num_global_tokens, :] = False
        mask[:, :self.num_global_tokens] = False
        
        # 2. 국소·미끄러지는 창 (제 블록과 이웃한 블록 안)
        num_blocks = (seq_len + self.block_size - 1) // self.block_size
        
        for i in range(seq_len):
            block_i = i // self.block_size
            
            # 같은 블록에 주의한다
            start = block_i * self.block_size
            end = min((block_i + 1) * self.block_size, seq_len)
            mask[i, start:end] = False
            
            # 이웃한 블록에 주의한다
            if block_i > 0:
                prev_start = (block_i - 1) * self.block_size
                mask[i, prev_start:start] = False
            if block_i < num_blocks - 1:
                next_end = min((block_i + 2) * self.block_size, seq_len)
                mask[i, end:next_end] = False
        
        # 3. 무작위 주의
        for i in range(0, seq_len, self.block_size):
            block_end = min(i + self.block_size, seq_len)
            
            # 무작위 블록을 고른다
            valid_blocks = [b for b in range(num_blocks) 
                          if abs(b - i // self.block_size) > 1]
            
            if valid_blocks:
                random_blocks = torch.tensor(valid_blocks)[
                    torch.randperm(len(valid_blocks))[:self.num_random_blocks]
                ]
                
                for rb in random_blocks:
                    rb_start = rb * self.block_size
                    rb_end = min((rb + 1) * self.block_size, seq_len)
                    mask[i:block_end, rb_start:rb_end] = False
        
        return mask
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """BigBird 주의를 쓰는 앞먹임."""
        batch_size, seq_len, _ = x.shape
        
        # 사영한다
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 꼴을 바꾼다
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 점수를 셈한다
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # BigBird 가림을 적용한다
        mask = self._create_bigbird_mask(seq_len, x.device)
        scores = scores.masked_fill(mask, float('-inf'))
        
        # 소프트맥스를 하고 적용한다
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, v)
        
        # 꼴을 바꾸고 사영한다
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.o_proj(output)

def visualize_sparse_patterns(seq_len: int = 64):
    """여러 성긴 주의 무늬를 그려 본다."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    patterns = [
        ("Full Attention", torch.zeros(seq_len, seq_len).bool()),
        ("Local (w=8)", create_local_attention_mask(seq_len, window_size=8)),
        ("Dilated (w=4, d=4)", create_dilated_attention_mask(seq_len, 4, 4)),
        ("Block Sparse", create_block_sparse_mask(seq_len, 16, 1)),
        ("Causal Local", create_local_attention_mask(seq_len, 8) | 
         torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()),
    ]
    
    # BigBird 무늬
    bigbird = BigBirdAttention(64, 1, block_size=16, num_global_tokens=2)
    bigbird_mask = bigbird._create_bigbird_mask(seq_len, torch.device('cpu'))
    patterns.append(("BigBird", bigbird_mask))
    
    for idx, (name, mask) in enumerate(patterns):
        ax = axes[idx // 3, idx % 3]
        # 그림으로 보려고 가림을 뒤집는다 (흰색은 주의, 검은색은 가림)
        ax.imshow(~mask.float().numpy(), cmap='gray', aspect='auto')
        ax.set_title(name)
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        
        # 주의하는 쌍의 수를 센다
        attending = (~mask).sum().item()
        density = attending / (seq_len * seq_len) * 100
        ax.text(0.02, 0.98, f'Density: {density:.1f}%', 
                transform=ax.transAxes, fontsize=9,
                verticalalignment='top', color='red')
    
    plt.suptitle('Sparse Attention Patterns (White = Attend, Black = Masked)')
    plt.tight_layout()
    plt.savefig('sparse_attention_patterns.png', dpi=150)
    plt.close()

# 사용 예
if __name__ == "__main__":
    d_model = 256
    num_heads = 4
    seq_len = 256
    batch_size = 2
    
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 국소 주의를 시험한다
    print("--- Local Attention ---")
    local_attn = LocalAttention(d_model, num_heads, window_size=32)
    out_local = local_attn(x)
    print(f"Input: {x.shape}, Output: {out_local.shape}")
    
    # Longformer 주의를 시험한다
    print("\n--- Longformer Attention ---")
    longformer_attn = LongformerAttention(d_model, num_heads, window_size=32)
    out_longformer = longformer_attn(x)
    print(f"Input: {x.shape}, Output: {out_longformer.shape}")
    
    # BigBird 주의를 시험한다
    print("\n--- BigBird Attention ---")
    bigbird_attn = BigBirdAttention(d_model, num_heads, block_size=32)
    out_bigbird = bigbird_attn(x)
    print(f"Input: {x.shape}, Output: {out_bigbird.shape}")
    
    # 무늬를 그려 본다
    visualize_sparse_patterns(64)
    print("\nVisualization saved to 'sparse_attention_patterns.png'")
```

##### 복잡도 견주기

| 방법 | 시간 | 기억 | 전역 맥락 |
|--------|------|--------|----------------|
| 온전한 주의 | O(N²) | O(N²) | ✓ 온전함 |
| 국소 창 | O(Nw) | O(Nw) | ✗ 제한적 |
| 팽창 | O(Nw) | O(Nw) | ✗ 제한적 (더 넓다) |
| Sparse Transformer | O(N√N) | O(N√N) | ✓ 걸음 있는 무늬로 |
| Longformer | O(Nw + Ng) | O(N) | ✓ 전역으로 |
| BigBird | O(N) | O(N) | ✓ 무작위와 전역으로 |

##### 효율적인 주의와의 관계

성긴 주의(주의 행렬의 짜임 있는 성김)는 효율적인 주의를 얻는 여러 길 가운데 하나이다. 그 지형은 다음과 같다.

| 길 | 방법 | 보기 |
|----------|----------|----------|
| **성긴 무늬** | 어느 쌍이 주의를 셈할지 제한한다 | Longformer, BigBird, Sparse Transformer |
| **낮은 계수 어림** | 주의 행렬을 낮은 계수로 인수분해해 어림한다 | Linformer, Nyström |
| **핵 방법** | 소프트맥스를 선형화할 수 있는 핵으로 어림한다 | Performer, Random Feature Attention |
| **입출력을 고려한 계산** | 기억 접근 방식을 다듬는다 | FlashAttention |

성긴 주의와 이 다른 길들은 서로를 채워 준다. 이를테면 FlashAttention이 성긴 무늬를 빠르게 할 수 있고 낮은 계수 방법을 국소 창과 함께 쓸 수 있다.

##### 간추림

성긴 주의 무늬는 긴 수열을 효율적으로 처리하게 해 준다.

1. **국소 주의**: 빠르지만 맥락이 좁다
2. **전역 토큰**: 수열 전체의 정보를 지킨다
3. **무작위 주의**: 표현력에 대한 이론적 보장
4. **위계 주의**: 여러 크기의 정보 흐름
5. **섞은 무늬**: 모든 길의 좋은 점을 모으며 보편적임을 증명할 수 있다

##### 참고 문헌

1. Beltagy, I., et al. (2020). "Longformer: The Long-Document Transformer."
2. Zaheer, M., et al. (2020). "Big Bird: Transformers for Longer Sequences." NeurIPS.
3. Child, R., et al. (2019). "Generating Long Sequences with Sparse Transformers."
4. Kitaev, N., et al. (2020). "Reformer: The Efficient Transformer." ICLR.
5. Dao, T., et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention." NeurIPS.

---

## 10. 트랜스포머 키우기

##### 큰 규모 모형 학습하기

##### GPT-3

OpenAI가 만든 GPT-3은 디코더 전용 트랜스포머를 키우면 강한 소수 예시 학습 능력이 나옴을 보였다.

**구조**: 매개변수 1750억의 디코더 전용 트랜스포머로 층 96개, 숨은 차원 12,288, 주의 머리 96개를 쓴다.

**학습 데이터**: 걸러 낸 Common Crawl의 일부에 고른 데이터셋(WebText, Books, 위키백과)을 더해 모두 약 3000억 토큰이다. 학습 데이터의 다양함이 과제에 걸친 일반화에 매우 중요하다.

**학습 계산**: GPT-3에는 약 3.14 × 10²³ FLOP이 들었고 모형 병렬과 데이터 병렬로 GPU 수천 개에 걸쳐 학습했다.

**핵심 통찰**: 기울기 갱신 없이 프롬프트의 몇몇 예만으로 과제를 해내는 GPT-3의 맥락 안 학습 능력은 규모가 커지며 창발했고 작은 모형에서는 보이지 않았다.

##### PaLM

구글이 만든 PaLM(경로 언어 모형)은 몇 가지 구조를 다듬어 매개변수 5400억까지 키웠다.

**구조**: SwiGLU 활성, 주의와 순전파의 병렬 계산, 다중 질의 주의, RoPE 위치 인코딩을 쓰는 디코더 전용 트랜스포머이다.

**학습 데이터**: 웹 문서, 책, 코드, 대화 데이터에 걸친 토큰 7800억 개의 여러 언어 말뭉치이다.

**학습 기반**: PaLM은 여러 TPU 묶음에 걸친 계산을 효율적으로 지휘하는 Pathways 체계로 TPU v4 칩 6,144개에 걸쳐 학습했다.

**핵심 통찰**: PaLM은 모형 규모가 커지며 뚝 끊긴 듯 나타나는 추론 과제(이를테면 생각의 사슬 프롬프트)의 "돌파" 능력을 보였다.

##### LLaMA

LLaMA(메타 AI의 대형 언어 모형)는 데이터가 넉넉하면 더 작지만 잘 학습된 모형이 훨씬 큰 모형의 성능과 맞먹거나 앞설 수 있음을 보였다.

**구조**: 앞 정규화(RMSNorm), SwiGLU 활성, RoPE를 쓰는 디코더 전용 트랜스포머이며 모형 크기는 매개변수 70억에서 650억까지이다.

**학습 데이터**: 공개된 데이터만으로 1.4조 토큰이며, 그 모형 크기에 흔한 것보다 더 많은 토큰으로 학습했다.

**핵심 통찰**: 계산을 가장 아끼는 모형은 제 매개변수 수보다 훨씬 많은 토큰으로 학습되며, 이는 토큰을 적게 쓰고 모형을 키우는 쪽을 편들던 앞선 규모 법칙 가정에 맞선다(친칠라 규모 법칙).

##### 규모 법칙

Kaplan 외(2020)와 Hoffmann 외(2022)는 모형 성능과 계산 예산 사이의 경험적 관계를 밝혔다.

##### 캐플런 규모 법칙

(교차 엔트로피 손실 $L$으로 잰) 성능은 모형 크기 $N$, 데이터셋 크기 $D$, 계산 $C$에 대해 거듭제곱 법칙을 따른다.

$$
L(N) \approx \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad
L(D) \approx \left(\frac{D_c}{D}\right)^{\alpha_D}, \quad
L(C) \approx \left(\frac{C_c}{C}\right)^{\alpha_C}
$$

여기서 $\alpha_N \approx 0.076$, $\alpha_D \approx 0.095$, $\alpha_C \approx 0.050$이다.

##### 친칠라 규모 법칙 (Hoffmann 외)

계산 최적 방식은 정해진 계산 예산 $C$을 모형 크기 $N$과 데이터 크기 $D$을 똑같이 키우며 나눈다.

$$
N_{\text{opt}} \propto C^{0.5}, \quad D_{\text{opt}} \propto C^{0.5}
$$

실제로 가장 좋은 토큰 수는 매개변수 수의 약 20배이다. 매개변수 100억짜리 모형은 토큰 약 2000억 개로 학습해야 한다.

##### 큰 규모 학습의 어려움

##### 데이터의 어려움

**양과 질**: 큰 모형에는 질 좋은 토큰이 수조 개 필요하다. 웹에서 긁은 데이터는 질을 따져 꼼꼼히 거르고 겹친 것을 없애고 분야에 걸쳐 고르게 해야 한다. 데이터 오염(평가 잣대와 겹침)을 찾아내어 없애야 한다.

**큰 규모의 앞처리**: 테라바이트 규모 데이터셋의 토큰 나누기와 거르기와 뒤섞기에는 분산 데이터 파이프라인이 필요하다. 학습 데이터를 병목 없이 가속기 수천 개에 대어 주어야 한다.

##### 계산의 어려움

**비용**: GPT-3 학습에는 계산에만 수백만 달러가 들었다고 어림된다. 비용은 전체 FLOP에 대략 비례하는데, FLOP은 매개변수 $N$과 토큰 $D$에 대해 $O(N \cdot D)$으로 는다.

**학습 안정성**: 큰 모형은 학습 손실이 갑자기 치솟아 학습을 어그러뜨리는 손실 튐이 잦다. 학습률을 낮추기, 기울기 자르기, 조심스러운 초기화로 누그러뜨린다.

**되풀이 가능성**: 가속기 수천 개에서 도는 학습은 부동소수점 연산 순서에서 오는 비결정성을 낳아 똑같이 되풀이하기 어렵게 만든다.

##### 기억의 병목

float32으로 Adam을 써서 학습하는 매개변수 $N$개의 모형에서는 다음과 같다.

| 부품 | 매개변수당 기억 | 모두 (매개변수 1750억) |
|-----------|---------------------|---------------------|
| 매개변수 | 4바이트 | 700GB |
| 기울기 | 4바이트 | 700GB |
| Adam 최적화기 상태 ($m$, $v$) | 8바이트 | 1,400GB |
| **모두** | **16바이트** | **약 2.8TB** |

이는 (대개 40~80GB인) 어떤 가속기 하나의 기억도 넘어서므로 병렬화 방법이 필요해진다.

##### 병렬화 방법

##### 데이터 병렬

일꾼마다 모형을 통째로 하나씩 지니고 서로 다른 데이터 조각을 처리한다. 기울기는 all-reduce로 일꾼들에 걸쳐 평균 낸다.

$$
g_{\text{avg}} = \frac{1}{K} \sum_{k=1}^{K} g_k
$$

여기서 $K$은 일꾼의 수이고 $g_k$은 일꾼 $k$의 기울기이다.

**한계**: 일꾼마다 모형 전체를 담아야 하므로 데이터 병렬만으로는 장치 하나의 기억을 넘는 모형을 다룰 수 없다.

```python
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_data_parallel(rank: int, world_size: int):
    """데이터 병렬을 위해 분산 프로세스 묶음을 시작한다."""
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def train_with_ddp(rank: int, world_size: int, model: nn.Module):
    """
    여러 GPU 학습을 위해 모형을 DistributedDataParallel으로 감싼다.
    
    GPU마다 다른 작은 배치를 처리하고, 기울기는 optimizer.step() 전에
    all-reduce로 저절로 평균 낸다.
    """
    setup_data_parallel(rank, world_size)
    
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])           # 기울기를 맞추려고 모형을 감싼다
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # DistributedSampler로 순위마다 다른 데이터 조각을 받는다
    # 기울기는 순위에 걸쳐 저절로 평균 낸다
    for batch in dataloader:
        optimizer.zero_grad()
        loss = model(batch).loss
        loss.backward()                              # 여기서 기울기 all-reduce가 일어난다
        optimizer.step()
```

##### 기울기 모으기

바라는 실효 배치 크기가 GPU 기억을 넘으면, 기울기 모으기가 매개변수를 갱신하기 전에 앞먹임과 역전파를 여러 번 하며 기울기를 쌓아 더 큰 배치를 흉내 낸다.

$$
g_{\text{accumulated}} = \frac{1}{A} \sum_{a=1}^{A} g_a
$$

여기서 $A$은 모으기 단계의 수이다.

```python
def train_with_gradient_accumulation(
    model: nn.Module,
    dataloader,
    optimizer,
    accumulation_steps: int = 8,
    max_grad_norm: float = 1.0
):
    """
    기울기 모으기를 쓰는 학습 고리.
    
    실효 배치 크기 = micro_batch_size × accumulation_steps × num_gpus
    이를테면 4 × 8 × GPU 4개 = 실효 배치 크기 128.
    """
    model.train()
    optimizer.zero_grad()
    
    for step, batch in enumerate(dataloader):
        # 앞먹임과 역전파 (기울기가 .grad에 쌓인다)
        loss = model(**batch).loss
        loss = loss / accumulation_steps              # 모으기 단계 수로 나눈다
        loss.backward()
        
        if (step + 1) % accumulation_steps == 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
```

##### 모형 병렬 (텐서 병렬)

층 하나하나를 여러 장치에 나눈다. 선형 층 $Y = XW$에서는 가중치 행렬 $W$을 열 방향으로 장치 $K$개에 나눈다.

$$
W = [W_1 \mid W_2 \mid \cdots \mid W_K]
$$

장치마다 $Y_k = X W_k$을 셈하고 그 결과를 (나누는 방식에 따라) 이어 붙이거나 줄인다. Megatron-LM이 쓰는 방식이다.

##### 파이프라인 병렬

층마다 다른 장치를 맡긴다. 입력 작은 배치가 파이프라인을 따라 흐른다.

$$
\text{Device 1: Layers 1–24} \rightarrow \text{Device 2: Layers 25–48} \rightarrow \text{Device 3: Layers 49–72} \rightarrow \text{Device 4: Layers 73–96}
$$

**어려움**: 순진한 파이프라인 병렬은 장치가 앞먹임이나 역전파를 기다리며 노는 "파이프라인 거품"을 만든다. 작은 배치로 나누기(GPipe)와 엇갈린 일정 짜기(PipeDream)가 노는 시간을 줄인다.

##### 3차원 병렬

큰 규모 학습은 세 방법을 모두 아우른다.

- 장치 묶음에 걸친 **데이터 병렬**
- 묶음 안(대개 노드 하나 안)의 **텐서 병렬**
- 묶음에 걸친 **파이프라인 병렬**

이를테면 Megatron-Turing NLG(5300억)는 노드마다 8갈래 텐서 병렬, 노드에 걸쳐 35갈래 파이프라인 병렬, 파이프라인 복제본에 걸쳐 데이터 병렬을 쓴다.

##### 효율적으로 키우기 위해 떠오르는 구조

##### Megatron-LM

Megatron-LM(엔비디아)은 주의와 순전파 계산을 나누어 트랜스포머 층을 위한 효율적인 텐서 병렬을 준다.

- **주의**: $Q$, $K$, $V$ 사영을 열 방향으로 장치에 나눈다. 장치마다 따로 주의를 셈한다. 출력 사영은 행 방향으로 나누고 결과를 all-reduce로 더한다.
- **순전파**: 첫 선형 층은 열 방향으로, 둘째는 행 방향으로 나눈다. all-reduce 한 번으로 출력을 맞춘다.

이 방식은 트랜스포머 층마다 all-reduce가 두 번만 필요하여 주고받는 짐이 가볍다.

##### 전문가 섞기 (MoE)

Switch Transformer 같은 전문가 섞기 모형은 입력 토큰마다 매개변수의 일부만 켜서, 계산 비용은 덜 늘리면서 모형 전체의 그릇을 훨씬 크게 한다.

문 얼개가 토큰마다 상위 $k$명의 전문가를 고른다.

$$
G(x) = \text{TopK}\left(\text{softmax}(x \cdot W_g)\right)
$$

$$
\text{MoE}(x) = \sum_{i \in \text{TopK}} G(x)_i \cdot E_i(x)
$$

여기서 $E_i$은 $i$번째 전문가(대개 순전파 신경망)이고 $G(x)_i$은 문 가중치이다.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    """전문가 하나: 표준 자리별 순전파 신경망."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)

class TopKGatingMoE(nn.Module):
    """
    상위 K 문을 갖춘 전문가 섞기 층.
    
    트랜스포머 블록의 표준 순전파 신경망을 대신한다.
    토큰마다 학습되는 문 함수에 따라 상위 k명의 전문가에게
    보내진다.
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # 문 신경망: 입력을 전문가 점수로 사영한다
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        
        # 전문가 신경망
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff, dropout)
            for _ in range(num_experts)
        ])
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        인수:
            x: [batch_size, seq_len, d_model]
        
        반환값:
            output: [batch_size, seq_len, d_model]
            aux_loss: 부하를 고르게 하는 손실(스칼라)
        """
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)                  # [B*S, d_model]
        
        # 문 점수를 셈한다
        gate_logits = self.gate(x_flat)                # [B*S, num_experts]
        gate_probs = F.softmax(gate_logits, dim=-1)    # [B*S, num_experts]
        
        # 토큰마다 상위 k명의 전문가를 고른다
        top_k_probs, top_k_indices = torch.topk(       # [B*S, top_k] each
            gate_probs, self.top_k, dim=-1
        )
        
        # 고른 확률을 정규화한다
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # 전문가의 출력을 셈한다 (간추린 것. 실제로는 scatter/gather를 쓴다)
        output = torch.zeros_like(x_flat)              # [B*S, d_model]
        for k in range(self.top_k):
            expert_idx = top_k_indices[:, k]           # [B*S]
            weight = top_k_probs[:, k].unsqueeze(-1)   # [B*S, 1]
            
            for i in range(self.num_experts):
                mask = (expert_idx == i)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[i](expert_input)
                    output[mask] += weight[mask] * expert_output
        
        # 부하를 고르게 하는 보조 손실
        # f_i: 전문가 i에게 간 토큰의 비율
        # p_i: 전문가 i의 평균 문 확률
        f = torch.zeros(self.num_experts, device=x.device)
        for k in range(self.top_k):
            for i in range(self.num_experts):
                f[i] += (top_k_indices[:, k] == i).float().mean()
        f = f / self.top_k
        
        p = gate_probs.mean(dim=0)                     # [num_experts]
        aux_loss = self.num_experts * (f * p).sum()
        
        return output.view(batch_size, seq_len, d_model), aux_loss
```

**Switch Transformer**: $k=1$(전문가 하나로 보냄)을 써서 같은 질의 빽빽한 모형보다 최대 7배 빠르다. 전체 매개변수 1.6조인 Switch Transformer가 앞먹임마다 쓰는 매개변수는 약 1000억뿐이다.

**부하 고르기**: 전문가에게 보내는 일이 고르지 않을 수 있다. 위의 보조 손실이 전문가를 고르게 쓰도록 북돋운다.

$$
\mathcal{L}_{\text{aux}} = \alpha \cdot N_E \sum_{i=1}^{N_E} f_i \cdot p_i
$$

여기서 $f_i$은 전문가 $i$에게 간 토큰의 비율, $p_i$은 전문가 $i$의 평균 문 확률, $N_E$은 전문가의 수이다.

##### 효율적으로 키우는 기법

**양자화**: 매개변수의 정밀도를 float32에서 int8이나 int4로 낮추면 질을 거의 잃지 않고 기억이 4~8배 준다. 학습 뒤 양자화(GPTQ, AWQ)와 양자화를 고려한 학습(QAT)을 모두 쓴다.

**가지치기**: 군더더기 가중치를 없애면 모형이 작아진다. 짜임 있는 가지치기(주의 머리나 순전파 차원을 통째로 없앰)가 짜임 없는(가중치 하나하나) 가지치기보다 하드웨어에 잘 맞는다.

**증류**: 더 작은 "학생" 모형이 더 큰 "스승" 모형을 흉내 내도록 학습시킨다. DistilBERT는 매개변수를 40% 줄이고 추론을 60% 빠르게 하면서 BERT 성능의 97%를 낸다.

##### 학습 기반 견주기

| 시스템 | 모형 크기 | 하드웨어 | 병렬화 | 학습 시간 |
|--------|-----------|----------|-------------|---------------|
| GPT-3 | 1750억 | V100 GPU | 데이터와 모형 | 몇 달 |
| PaLM | 5400억 | TPU v4 (칩 6144개) | 데이터와 모형과 파이프라인 | 몇 주 |
| LLaMA-65B | 650억 | A100 GPU (2048개) | 데이터와 텐서와 파이프라인 | 약 21일 |
| Chinchilla | 700억 | TPU v3/v4 | 데이터와 모형 | 몇 주 |

##### 간추림

트랜스포머를 키우는 일은 데이터와 계산과 기억에 걸친 복잡한 맞바꿈의 공간을 헤쳐 가는 일이다.

1. **규모 법칙**은 계산을 모형 크기와 학습 데이터에 어떻게 나누는 것이 가장 좋은지 길잡이를 준다.
2. **병렬화 방법**(데이터, 텐서, 파이프라인)은 장치 하나의 기억을 넘는 모형을 학습할 수 있게 한다.
3. **전문가 섞기**는 토큰마다 모형의 일부만 켜서 매개변수를 아낀다.
4. **학습 뒤 효율** 기법(양자화, 가지치기, 증류)은 큰 모형을 실제로 쓸 만하게 만든다.
5. **학습 안정성**에는 조심스러운 학습률 일정과 기울기 자르기와 초기화가 필요하다.

이 분야는 빠르게 흘러가고 있으며 새로운 구조와 학습 방법이 꾸준히 나오고 있다.

##### 참고 문헌

1. Brown, T., et al. (2020). "Language Models are Few-Shot Learners." NeurIPS. (GPT-3)
2. Chowdhery, A., et al. (2022). "PaLM: Scaling Language Modeling with Pathways." arXiv.
3. Touvron, H., et al. (2023). "LLaMA: Open and Efficient Foundation Language Models." arXiv.
4. Kaplan, J., et al. (2020). "Scaling Laws for Neural Language Models." arXiv.
5. Hoffmann, J., et al. (2022). "Training Compute-Optimal Large Language Models." NeurIPS. (Chinchilla)
6. Shoeybi, M., et al. (2020). "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism." arXiv.
7. Fedus, W., et al. (2022). "Switch Transformers: Scaling to Trillion Parameter Models." JMLR.

---

## 연습문제

**연습문제 1.**
구조, 사전 학습, 잘 맞는 쓰임새의 면에서 BERT, GPT, T5, XLNet을 견주어라.

??? success "연습문제 1 풀이"
    BERT는 인코더만 쓰고 가린 언어 모형화를 하며 이해 과제(분류, 개체명 인식)에 가장 좋다. GPT는 디코더만 쓰고 자기 회귀 언어 모형화를 하며 생성에 가장 좋다. T5는 인코더-디코더에 구간 망가뜨리기를 쓰며 수열 대 수열 과제에 가장 좋다. XLNet은 순열 언어 모형을 쓰는 디코더로 자기 회귀이면서도 양방향 맥락을 잡아낸다. 저마다 이해와 생성 사이의 다른 맞바꿈을 나타낸다.

---

**연습문제 2.**
BERT에 견준 RoBERTa의 핵심 혁신을 설명하라.

??? success "연습문제 2 풀이"
    RoBERTa(Liu 외, 2019)는 (1) (도움이 안 된다고 밝혀진) NSP 목표를 없애고, (2) 더 많은 데이터로 더 오래 학습하고, (3) 동적 가리기(세대마다 다른 가림)를 쓰고, (4) 배치를 더 크게 하고, (5) WordPiece 대신 바이트 쌍 부호화를 써서 BERT를 낫게 한다. 구조는 바꾸지 않고 학습 조리법만 낫게 한 것이다.

---

**연습문제 3.**
ALBERT와 BERT의 차이는 무엇인가? ALBERT는 어떻게 매개변수를 줄이는가?

??? success "연습문제 3 풀이"
    ALBERT는 (1) 인수분해한 임베딩($V \times H$ 대신 $V \times E + E \times H$으로, $E \ll H$일 때 매개변수를 아낀다)과 (2) 층끼리 매개변수 함께 쓰기(모든 층이 같은 가중치를 쓴다)로 매개변수를 약 10배 줄이면서 BERT 성능의 대부분을 지킨다.

---

**연습문제 4.**
'사전 학습한 뒤 미세 조정하기'라는 개념과 그것이 어떻게 흘러왔는지 설명하라.

??? success "연습문제 4 풀이"
    본디 방식은 큰 말뭉치로 사전 학습한 뒤 과제 데이터로 미세 조정하는 것이다. 그 뒤 (1) 어댑터는 작은 학습 모듈을 더하고 사전 학습된 가중치는 얼려 둔다. (2) 프롬프트 조정은 학습되는 프롬프트 토큰을 더하고 모형은 얼려 둔다. (3) 맥락 안 학습(GPT-3)은 매개변수를 갱신하지 않고 프롬프트에 예만 준다. 흐름은 사전 학습된 모형을 덜 건드리는 쪽이다.

## 정리하며

이 세 구조는 수열 모형화에 대한 서로 다른 설계 철학을 나타낸다.

- **순환 신경망**: 수열이 본디 시간을 따른다는 귀납 편향 위에 세워졌다. 차례를 따르는 성질이 순서를 자연스레 지키지만 계산과 기울기 흐름에 병목을 만든다. 흘러드는 데이터를 다루는 응용에는 여전히 쓸모 있다.
- **합성곱 신경망**: 병렬 합성곱으로 국소 수용 영역의 무늬를 효율적으로 적용한다. 무늬 찾기에 뛰어나지만 먼 거리 의존에는 층을 많이 쌓아야 한다.
- **트랜스포머**: 수열의 짜임에 대해 아무 가정도 하지 않고 모든 관계를 주의로 배운다. 이차 복잡도를 대가로 가장 힘 있고 잘 커진다.

이 분야는 대체로 트랜스포머로 모였지만, 세 구조를 모두 이해하면 특정 제약(지연, 기억, 수열 길이)에 맞는 연장을 고르는 데 도움이 된다.

**참고 문헌**

1. Vaswani, A., et al. (2017). "Attention Is All You Need."
2. Hochreiter, S., & Schmidhuber, J. (1997). "Long Short-Term Memory."
3. Kim, Y. (2014). "Convolutional Neural Networks for Sentence Classification."
4. Gu, A., et al. (2022). "Efficiently Modeling Long Sequences with Structured State Spaces."

---
