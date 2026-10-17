# 트랜스포머의 변형

RNN·CNN·트랜스포머의 연결 구조를 비교하고 희소 어텐션, 혼합 모델과 분산 학습을 살펴본다. 길이에 대한 복잡도와 실제 속도·메모리, 구조의 가능성과 예제의 구현 범위를 구분한다.

## 1. 순환 뉴럴 넷트웍/LSTM

아래 LSTMModel은 (B,L,V)의 위치별 로짓을 반환한다. 스트리밍 상태를 받거나 다음 호출에 돌려주는 인터페이스는 없으므로, 현재 코드가 자동으로 호출 사이의 문맥을 유지하는 것은 아니다.

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

- 피처 차원과 레이어 수를 고정하면 기본 상태 갱신 비용이 길이에 선형이다.
- 시간 순서와 상태 갱신이라는 귀납 바이어스가 있다.
- 추론에서 과거 전체 활성 대신 고정 크기 상태를 유지할 수 있다.

**나쁜 점:**

- 같은 시퀀스의 상태 갱신이 순차적으로 연결된다.
- 긴 역전파의 야코비안 곱으로 그래디언트 소실·폭발이 생길 수 있다.
- 배치·피처 연산은 병렬화할 수 있지만 시간축 의존은 남는다.

---

## 2. 시퀀스을 위한 컨볼루션 뉴럴 넷트웍

CNNModel은 전역 최대 풀링으로 (B,V)의 표본별 로짓을 반환해 위 LSTM의 위치별 출력과 과제가 다르다. 이 코드 블록에는 torch.nn.functional을 F로 가져오는 import가 빠져 있다. 양쪽 패딩 컨볼루션은 미래 토큰도 읽으므로 자기 회귀 모델로 사용하려면 별도 인과 구성이 필요하다.

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
import torch
import torch.nn as nn


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

- 단일 작은 커널의 리셉티브 필드는 국소적이다.
- 긴 의존을 연결하려면 깊이·팽창·스트라이드·풀링 등을 설계해야 한다.
- 먼 의존을 원천적으로 다룰 수 없는 구조는 아니다.

---

## 3. 트랜스포머

TransformerModel은 양방향 인코더와 어휘 출력 레이어다. 인과·패딩 마스크를 전달하지 않아 다음 토큰 언어 모델로 곧바로 사용할 수 없다. 학습 위치 표는 512개로 고정되어 있다.

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

- 전역 점수 내적·가중합은 길이의 제곱에 비례한다.
- 위치 표현과 마스크 등 순서 관련 설계를 별도로 정한다.
- 명시적인 전체 점수 저장은 긴 입력에서 큰 메모리를 사용한다.

---

## 4. 자세히 비교

### 복잡도 분석

| 기준 | 기본 RNN | 1차원 CNN | 전역 셀프 어텐션 |
|---|---|---|---|
| 학습 시간축 의존 | n번 상태 갱신 | 레이어 안의 위치 병렬화 | 쿼리 위치 병렬화 |
| 대표 연산 비용 | $O(nd^2)$ | $O(knd^2)$ | 점수·가중합 $O(n^2d)$, 투영 $O(nd^2)$ |
| 먼 위치 연결 | 중간 상태를 거침 | 커널·깊이·팽창에 의존 | 허용된 위치끼리 한 레이어에서 직접 연결 |

CNN의 경로가 항상 O(log n)인 것은 아니다. 이는 특정 계층·팽창·다운샘플 구성의 가정이 필요하다. 자기 회귀 생성의 순차성은 학습 병렬성과도 구분한다.

### 메모리 사용

시퀀스 길이가 n이고 차원이 d일 때 다음과 같다.

| 구조·상황 | 길이에 따른 대표 저장 |
|---|---|
| RNN 학습 | 시간별 활성 $O(nd)$, 전체 역전파 가정 |
| RNN 점진 추론 | 고정 크기 상태 $O(d)$, 레이어·상태 차원 고정 |
| CNN | 활성·리셉티브 필드·스트리밍 구현에 따라 다름 |
| 명시적 전역 어텐션 학습 | 점수 $O(hn^2)$와 토큰 활성 |
| 캐시 기반 한 토큰 생성 | K·V 캐시 $O(nd)$, 현재 쿼리의 점수 $O(hn)$ |

배치·레이어·파라메터·옵티마이저 저장은 생략한 비교다. KV 캐시 자체가 O(n²)인 것은 아니며 메모리 효율 어텐션도 전체 점수 저장을 피할 수 있다.

### 시퀀스 길이에 따른 성능

```
Short (< 512):    Transformer ≈ CNN > RNN
Medium (512-2K):  Transformer > CNN > RNN
Long (2K-8K):     Efficient Transformers > CNN > RNN
Very Long (>8K):  State Space Models / Linear Attention
```

---

## 5. 성능 비교의 조건

모델 이름별로 출처 없는 정확도·BLEU·퍼플렉시티를 나란히 놓으면 공정한 비교가 되지 않는다. 학습 데이터·토큰화·사전 학습·파라메터·튜닝 예산과 평가 분할을 함께 명시해야 한다.

이 페이지의 코드에는 WikiText·IMDB·SST-2·WMT를 학습·평가하는 실험이 없다. 따라서 특정 구조의 정확도나 번역 점수를 이 구현의 결과로 제시하지 않는다. 속도와 메모리도 같은 과제·입력 길이·장치·정밀도에서 측정한다.

## 6. 혼합형 접근

ConvTransformer는 국소 컨볼루션과 전역 어텐션을 더하는 블록이며 학습·평가 과제는 제공하지 않는다. 커널이 짝수이면 padding=k//2가 길이를 한 칸 늘려 덧셈이 맞지 않을 수 있다. 인과·패딩 마스크도 없다.

TransformerWithRecurrence는 memory_size를 사용하지 않고 과거 출력 전체를 self.memory에 저장한다. 호출할수록 길이가 증가하며 마지막 슬라이스도 이미 늘어난 x의 길이를 사용해 현재 입력 부분만 반환하지 않는다. detach·상태 초기화·표본 대응·장치 이동 처리도 없어 제한된 고정 크기 메모리나 안전한 스트리밍 모델로 해석해서는 안 된다.

### 트랜스포머와 컨볼루션 뉴럴 넷트웍

```python
class ConvTransformer(nn.Module):
    """국소 컨볼루션과 전역 주의."""
    def __init__(self, d_model, num_heads, kernel_size=3):
        super().__init__()
        self.local_conv = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size//2)
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
    
    def forward(self, x):
        # 국소 피처
        local = self.local_conv(x.transpose(1, 2)).transpose(1, 2)
        # 전역 주의
        global_out, _ = self.attention(x, x, x)
        return local + global_out
```

### 트랜스포머와 순환 뉴럴 넷트웍

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

- **RNN**: 고정 상태로 이어지는 시간 처리와 낮은 추론 메모리가 중요한 경우의 출발점이다.
- **CNN**: 국소 패턴과 제어된 리셉티브 필드를 효율적으로 추출하는 데 유용하다.
- **트랜스포머**: 위치 간 정보 결합과 사용 가능한 사전 학습 모델을 활용할 때 유용하다.

“길이 100 미만” 같은 보편적인 선택 기준이나 최고 성능 구조는 없다. 과제·학습 예산과 실제 지연·메모리로 비교한다.

## 8. 요즘의 대안

긴 시퀀스 비용을 줄이기 위해 연결 범위를 제한하거나 점수 계산을 근사하거나 순환 상태를 사용하는 방법이 있다. 같은 계산 복잡도가 같은 표현력·품질을 뜻하지는 않는다.

| 방법 | 길이 비용의 대표 형태 | 필요한 가정 |
|---|---|---|
| 선형 어텐션 | $O(n)$ | 고정 피처 차원과 적절한 커널·정규화 |
| 상태 공간 모델 | 주로 $O(n)$ 또는 구성별 변형 | 상태 크기·스캔 알고리즘 고정, S4와 Mamba의 선택 규칙은 다름 |
| RWKV 계열 | 상태 갱신에 선형 | 모델 버전·상태 차원 고정 |
| Longformer | $O(nw+ng)$ | 국소 폭 w·전역 토큰 g와 희소 커널 |
| Hyena 계열 | 컨볼루션 알고리즘에 따라 $O(n\log n)$ 등 | 필터·구현 조건 확인 |

처리 가능한 길이, 저장 비용과 긴 입력에서의 품질은 별도의 기준이다. 같은 모델 이름이라도 구현·장치·학습 길이에 따라 비교 결과가 달라질 수 있다.

---

## 9. 희소 어텐션 무늬

희소 어텐션은 계산할 쿼리·키 쌍을 미리 제한한다. 학습된 전체 어텐션의 웨이트가 언제나 일부 위치에만 몰린다는 법칙을 전제로 하지 않는다. 연결을 줄이면 일부 정보 경로가 사라지므로 과제별 품질을 확인해야 한다.

### 복잡도 문제

시퀀스 길이가 $N$이고 모델 차원이 $d$일 때 다음과 같다.

| 부품 | 시간 복잡도 | 메모리 |
|-----------|----------------|--------|
| 전역 어텐션 | O(N²d) | O(N²) |
| **목표** | O(Nd) 또는 O(N log N · d) | O(N) |

(어지간한 문서인) 토큰 $N = 16{,}384$개의 시퀀스에서 전역 어텐션 행렬의 항목은 $2.7 \times 10^8$개이다. 32비트 정밀도라면 헤드 하나에 대한 이 행렬을 담아 두는 데만 약 1GB가 든다.

### 흔한 성긴 무늬

### 1. 국소(슬라이딩 윈도) 주의

위치마다 창 안의 가까운 자리에만 주의한다.

$$
\text{Attend}(i) = \{j : |i - j| \leq w\}
$$

**복잡도**: 피처 차원을 고정하고 허용된 연결만 계산하는 커널에서는 O(Nw)다. 여기서 w는 반경이라 내부 위치의 허용 키는 최대 2w+1개다. 아래 마스크 함수는 N×N 텐서를 직접 만들므로 생성·저장 비용은 O(N²)다.

```
Window size = 3:
Position 5 attends to: [2, 3, 4, 5, 6, 7, 8]
```

**설계 근거**: 국소 관계를 우선 처리하며 여러 레이어·전역 토큰 등으로 먼 정보를 연결할 수 있다. 먼 토큰이 항상 중요하지 않다고 가정하는 것은 아니다.

### 2. 팽창(스텝 있는) 주의

정해진 간격의 자리에 주의하여 전역 어텐션 없이도 먼 거리를 아우른다.

$$
\text{Attend}(i) = \{j : (i - j) \mod d = 0, |i-j| \leq w \cdot d\}
$$

**복잡도**: O(Nw)

```
Dilation = 2, Window = 3:
Position 6 attends to: [0, 2, 4, 6, 8, 10, 12]
```

**까닭**: 팽창 컨볼루션과 비슷하다. $d$번째 위치마다 주의하면 계산을 늘리지 않고도 실효 리셉티브 필드가 $d$배로 커진다.

### 3. 전역 어텐션

어떤 자리를 모든 자리와 서로 주의하는 "전역"으로 정한다.

$$
\text{Attend}(i) = \begin{cases}
\{1, ..., N\} & \text{if } i \in \mathcal{G} \\
\mathcal{G} \cup \text{Local}(i) & \text{otherwise}
\end{cases}
$$

Longformer와 BigBird가 쓴다. 흔한 전역 토큰으로는 `[CLS]`, 과제에 맞는 토큰, 구간마다의 첫 토큰이나 마지막 토큰이 있다.

### 4. 블록 희소 어텐션

시퀀스을 블록으로 나누고 정해진 블록 안에서 또는 블록 사이에서 주의한다.

```
Block pattern:
[■ ■ □ □ ■]
[■ ■ ■ □ □]
[□ ■ ■ ■ □]
[□ □ ■ ■ ■]
[■ □ □ ■ ■]
```

### 5. 무작위 어텐션

위치마다 정해진 수의 다른 자리에 무작위로 주의한다.

$$
\text{Attend}(i) = \text{Random}(k) \cup \text{Local}(i)
$$

**이론 해석**: 무작위 연결은 일부 그래프 구성에서 경로 길이를 줄일 수 있다. 연결 수·방향·전역 노드와 깊이의 조건을 생략하고 모든 무작위 어텐션에 O(log N) 정보 전달 보장이 있다고 해석해서는 안 된다.

### 6. 위계(여러 크기) 주의

토큰·구간·문서 요약 등 여러 수준에서 어텐션을 구성한다.

**1수준**: 구간 안의 토큰 수준 국소 어텐션
**2수준**: 구간 요약 사이의 구간 수준 주의
**3수준**: 문서 요약 사이의 문서 수준 주의

이는 위계를 이루는 정보 흐름을 만든다.

$$\text{Local tokens} \to \text{Segment summaries} \to \text{Global representation}$$

계층 어텐션은 구간 요약을 통해 정보를 전달한다. 세부 토큰을 압축하는 로스가 있을 수 있고 복잡도는 구간 크기·단계 수·요약 개수에 따라 달라져 O(N√N) 이하가 자동으로 보장되지는 않는다.

**Sparse Transformer**는 국소·스트라이드 연결을 조합한 대표적인 설계에서 O(N√N) 비용을 제시했다. 임의의 계층 어텐션이 자동으로 같은 비용을 얻는 것은 아니다.

### 무늬 섞기

여러 희소 연결을 조합하면 국소·전역 정보 전달을 함께 설계할 수 있다. BigBird의 보편 근사·튜링 완전성 결과는 논문의 구조와 이론적 가정에 따른 표현력 분석이다. 유한한 학습 모델이 임의의 전체 어텐션 모델을 같은 자원으로 그대로 재현하거나 같은 품질을 낸다는 보장은 아니다.

| 연결 | 역할 | 한계 |
|---|---|---|
| 국소 | 이웃 위치 결합 | 한 레이어의 먼 위치 직접 접근 제한 |
| 전역 | 선택 토큰을 통한 정보 교환 | 요약·연결의 병목 가능 |
| 무작위 | 추가 장거리 경로 | 연결 규칙·재현성에 의존 |
| 혼합 | 여러 경로 조합 | 실제 학습 품질·커널 효율을 별도 검증 |

### 파이토치 구현

이 절의 LocalAttention·LongformerAttention·BigBirdAttention은 모두 전체 N×N 점수를 만든 뒤 마스크한다. 따라서 희소한 그림을 만들더라도 실제 메모리·주요 내적 비용은 이차로 남는다. 패딩 처리도 제공하지 않는다.

LongformerAttention은 v_global을 선언하지만 사용하지 않고, 전역 행 대입에서 (H,1,N)을 (H,N)에 넣어 축이 맞지 않을 수 있다. 전역 마스크가 없는 main 예제는 이 경로를 확인하지 않는다. 전체 Longformer 구현으로 보아서는 안 된다.

BigBirdAttention은 eval 모드에서도 매번 무작위 마스크를 다시 만들며, 헤드·배치에 공유한다. create_block_sparse_mask의 num_random_blocks도 실제 선택 개수가 아니라 선택 확률에 쓰이므로 항상 정해진 개수를 보장하지 않는다. 음수 반경·0 dilation·블록 크기나 맞지 않는 헤드 차원 등 유효 입력 조건도 확인해야 한다.

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
    팽창(스텝 있는) 주의 가림을 만든다.
    
    인수:
        seq_len: 수열 길이
        window_size: 주의할 자리의 개수
        dilation: 주의하는 자리 사이의 스텝
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
        국소 주의를 쓰는 피드포워드.
        
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
        
        # 전역 주의 사영 (따로 둔 파라메터)
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
        # global_attention_mask는 선택 인수라 None일 수 있다. 그대로 두면
        # NoneType에 nonzero를 부르게 되므로, 없으면 전역 자리가 없는 것으로 본다.
        if global_attention_mask is None:
            global_indices = (torch.empty(0, dtype=torch.long, device=x.device),
                              torch.empty(0, dtype=torch.long, device=x.device))
        else:
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
        """BigBird 주의를 쓰는 피드포워드."""
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
        # ~는 .float().numpy() 보다 늦게 묶이므로 괄호가 있어야 한다.
        # 괄호가 없으면 실수 배열에 ~를 걸어 TypeError가 난다.
        ax.imshow((~mask).float().numpy(), cmap='gray', aspect='auto')
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

**출력:**

```
--- Local Attention ---
Input: torch.Size([2, 256, 256]), Output: torch.Size([2, 256, 256])

--- Longformer Attention ---
Input: torch.Size([2, 256, 256]), Output: torch.Size([2, 256, 256])

--- BigBird Attention ---
Input: torch.Size([2, 256, 256]), Output: torch.Size([2, 256, 256])

Visualization saved to 'sparse_attention_patterns.png'
```

### 복잡도 비교

| 방식 | 허용 연결만 계산할 때의 점수 비용 | 조건 |
|---|---|---|
| 전역 | $O(N^2)$ | 모든 쌍 |
| 국소·팽창 | $O(Nw)$ | 고정 연결 폭 |
| Sparse Transformer의 대표 패턴 | $O(N\sqrt N)$ | 해당 분해 설계 |
| 국소+전역 | $O(Nw+Ng)$ | 윈도 w, 전역 g |
| BigBird 형태 혼합 | 길이에 선형 가능 | 전역·국소·무작위 연결 수 고정 |

실제 예제처럼 전체 행렬을 만든 뒤 가리는 구현에는 이 절감이 적용되지 않는다. 피처 투영·FFN·학습 저장은 별도다.

### 효율적인 주의와의 관계

희소 어텐션은 허용된 위치 쌍만 계산하는 접근이다. 저랭크·커널 근사와 정확한 어텐션의 메모리 효율화는 바꾸는 요소가 다르다.

| 접근 | 바꾸는 요소 | 예시 |
|---|---|---|
| 희소 패턴 | 계산할 위치 쌍 | Longformer·BigBird |
| 저랭크 근사 | 위치 축 표현·행렬 근사 | Linformer·Nyström 계열 |
| 커널 방법 | 결합 순서를 바꿀 수 있는 피처 표현 | Performer 계열 |
| 메모리 접근 최적화 | 정확한 어텐션의 타일·누산 순서 | FlashAttention |

근사·연결 제한과 정확한 어텐션의 메모리 효율화는 효과가 다르다. 지원되는 조합과 커널을 확인해야 한다.

희소 어텐션과 이 다른 길들은 서로를 채워 준다. 이를테면 FlashAttention이 성긴 무늬를 빠르게 할 수 있고 저랭크 방법을 국소 창과 함께 쓸 수 있다.

### 핵심 내용

희소 어텐션 무늬는 긴 시퀀스을 효율적으로 처리하게 해 준다.

연결 패턴, 리셉티브 필드와 실제 희소 커널을 함께 설계한다. 전역 토큰·무작위 경로가 모든 정보를 로스 없이 유지하거나 학습 성능을 보장하지는 않는다.

### 참고 문헌

1. Beltagy, I., et al. (2020). "Longformer: The Long-Document Transformer."
2. Zaheer, M., et al. (2020). "Big Bird: Transformers for Longer Sequences." NeurIPS.
3. Child, R., et al. (2019). "Generating Long Sequences with Sparse Transformers."
4. Kitaev, N., et al. (2020). "Reformer: The Efficient Transformer." ICLR.
5. Dao, T., et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention." NeurIPS.

---

## 10. 트랜스포머 키우기

### 큰 규모 모델 학습하기

### GPT-3

OpenAI가 만든 GPT-3은 디코더 전용 트랜스포머를 키우면 강한 퓨샷 학습 능력이 나옴을 보였다.

**구조**: 파라메터 1750억의 디코더 전용 트랜스포머로 레이어 96개, 숨은 차원 12,288, 어텐션 헤드 96개를 쓴다.

**학습 데이터**: 걸러 낸 Common Crawl의 일부에 고른 데이터셋(WebText, Books, 위키백과)을 더해 모두 약 3000억 토큰이다. 학습 데이터의 다양함이 과제에 걸친 일반화에 매우 중요하다.

**학습 계산**: 모델 크기와 사용한 토큰 외에도 실제 연산·장치·병렬화 효율을 확인한다. 이 페이지는 GPT-3 학습을 재현하거나 장치 수·실행 시간을 측정하지 않는다.

**핵심 통찰**: 프롬프트의 예시를 활용하는 성능이 모델 규모와 함께 개선될 수 있다. 작은 모델에는 맥락 내 학습이 전혀 없다는 뜻은 아니며, 과제·프롬프트·지표가 함께 영향을 준다.

### PaLM

구글이 만든 PaLM(경로 언어 모델)은 몇 가지 구조를 다듬어 파라메터 5400억까지 키웠다.

**구조**: SwiGLU 활성, 주의와 순전파의 병렬 계산, 멀티쿼리 어텐션, RoPE 위치 인코딩을 쓰는 디코더 전용 트랜스포머이다.

**학습 데이터**: 웹 문서, 책, 코드, 대화 데이터에 걸친 토큰 7800억 개의 여러 언어 말뭉치이다.

**학습 기반**: PaLM은 여러 TPU 배치에 걸친 계산을 효율적으로 지휘하는 Pathways 체계로 TPU v4 칩 6,144개에 걸쳐 학습했다.

**핵심 통찰**: 일부 추론 평가에서는 모델 규모에 따른 급격한 점수 개선이 보고되었다. 이를 보편적인 능력 발현 임계로 읽지 않고 지표·프롬프트·데이터 조건을 함께 본다.

### LLaMA

LLaMA(메타 AI의 대형 언어 모델)는 데이터가 넉넉하면 더 작지만 잘 학습된 모델이 훨씬 큰 모델의 성능과 맞먹거나 앞설 수 있음을 보였다.

**구조**: Pre-LN(RMSNorm), SwiGLU 활성, RoPE를 쓰는 디코더 전용 트랜스포머이며 모델 크기는 파라메터 70억에서 650억까지이다.

**학습 데이터**: 원래 LLaMA의 데이터 구성과 학습 토큰 수는 모델 규모에 따라 달랐다. 최대 모델의 보고를 모든 크기에 동일하게 적용하지 않는다.

**핵심 통찰**: 더 많은 학습 토큰을 사용해 작은 모델의 추론 비용과 품질을 절충할 수 있다. 학습 FLOP만 최적화하는 친칠라 분석과 배포 시 반복 추론 비용까지 고려하는 선택은 목적이 다르다.

### 스케일링 법칙

Kaplan 외(2020)와 Hoffmann 외(2022)는 모델 성능과 계산 예산 사이의 경험적 관계를 밝혔다.

### 캐플런 스케일링 법칙

(교차 엔트로피 로스 $L$으로 잰) 성능은 모델 크기 $N$, 데이터셋 크기 $D$, 계산 $C$에 대해 거듭제곱 법칙을 따른다.

$$
L(N) \approx \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad
L(D) \approx \left(\frac{D_c}{D}\right)^{\alpha_D}, \quad
L(C) \approx \left(\frac{C_c}{C}\right)^{\alpha_C}
$$

여기서 $\alpha_N \approx 0.076$, $\alpha_D \approx 0.095$, $\alpha_C \approx 0.050$이다.

### 친칠라 스케일링 법칙 (Hoffmann 외)

계산 최적 방식은 정해진 계산 예산 $C$을 모델 크기 $N$과 데이터 크기 $D$을 똑같이 키우며 나눈다.

$$
N_{\text{opt}} \propto C^{0.5}, \quad D_{\text{opt}} \propto C^{0.5}
$$

친칠라 연구의 실험 범위에서 흔히 쓰는 근사는 D≈20N이다. 파라메터 100억에 토큰 약 2000억이라는 예는 학습 계산 예산에 대한 출발점이며 반드시 그 수만큼 학습해야 한다는 규칙은 아니다. 데이터 품질과 추론 예산에 따라 배분이 달라진다.

### 큰 규모 학습의 어려움

### 데이터의 어려움

**양과 질**: 큰 모델에는 질 좋은 토큰이 수조 개 필요하다. 웹에서 긁은 데이터는 질을 따져 꼼꼼히 거르고 겹친 것을 없애고 분야에 걸쳐 고르게 해야 한다. 데이터 오염(평가 잣대와 겹침)을 찾아내어 없애야 한다.

**큰 규모의 앞처리**: 테라바이트 규모 데이터셋의 토큰 나누기와 거르기와 뒤섞기에는 분산 데이터 파이프라인이 필요하다. 학습 데이터를 병목 없이 가속기 수천 개에 대어 주어야 한다.

### 계산의 어려움

**비용**: 주요 dense 학습 연산은 대략 계수 N과 토큰 D의 곱에 비례할 수 있다. 실제 금액은 하드웨어 가격·통신·가동률·실패 재시도 등에 의존하므로 FLOP만으로 확정하지 않는다.

**학습 안정성**: 큰 모델은 학습 로스가 갑자기 치솟아 학습을 어그러뜨리는 로스 튐이 잦다. 러닝레이트를 낮추기, 경사 클리핑, 조심스러운 초기화로 누그러뜨린다.

**되풀이 가능성**: 가속기 수천 개에서 도는 학습은 부동소수점 연산 순서에서 오는 비결정성을 낳아 똑같이 되풀이하기 어렵게 만든다.

### 메모리의 병목

float32으로 Adam을 써서 학습하는 파라메터 $N$개의 모델에서는 다음과 같다.

| 부품 | 파라메터당 메모리 | 모두 (파라메터 1750억) |
|-----------|---------------------|---------------------|
| 파라메터 | 4바이트 | 700GB |
| 그래디언트 | 4바이트 | 700GB |
| Adam 옵티마이저 상태 ($m$, $v$) | 8바이트 | 1,400GB |
| **모두** | **16바이트** | **약 2.8TB** |

표는 십진 GB/TB 기준의 웨이트·경사·Adam 상태 합이며 활성·임시 작업 공간과 통신 버퍼는 빠져 있다. 혼합 정밀도의 master weight나 샤딩을 쓰면 구성이 달라진다. 일반적인 단일 장치보다 큰 저장 요구량을 어떻게 분산할지 설계해야 한다.

### 병렬화 방법

### 데이터 병렬

아래 DDP 조각은 dataloader와 분산 실행 환경 변수를 외부에 가정한다. DistributedSampler로 표본을 나누고 에포크마다 셔플 시드를 갱신하며 배치를 해당 장치로 이동하는 단계가 필요하다. DDP가 데이터 분할을 자동 수행하지는 않는다.

워커마다 모델을 통째로 하나씩 지니고 서로 다른 데이터 조각을 처리한다. 그래디언트는 all-reduce로 워커들에 걸쳐 평균 낸다.

$$
g_{\text{avg}} = \frac{1}{K} \sum_{k=1}^{K} g_k
$$

여기서 $K$은 워커의 수이고 $g_k$은 워커 $k$의 그래디언트이다.

**한계**: 워커마다 모델 전체를 담아야 하므로 데이터 병렬만으로는 장치 하나의 메모리를 넘는 모델을 다룰 수 없다.

```python
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_data_parallel(rank: int, world_size: int):
    """데이터 병렬을 위해 분산 프로세스 배치를 시작한다."""
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def train_with_ddp(rank: int, world_size: int, model: nn.Module):
    """
    여러 GPU 학습을 위해 모형을 DistributedDataParallel으로 감싼다.
    
    GPU마다 다른 작은 배치를 처리하고, 그래디언트는 optimizer.step() 전에
    all-reduce로 저절로 평균 낸다.
    """
    setup_data_parallel(rank, world_size)
    
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])           # 그래디언트를 맞추려고 모형을 감싼다
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # DistributedSampler로 순위마다 다른 데이터 조각을 받는다
    # 그래디언트는 순위에 걸쳐 저절로 평균 낸다
    for batch in dataloader:
        optimizer.zero_grad()
        loss = model(batch).loss
        loss.backward()                              # 여기서 그래디언트 all-reduce가 일어난다
        optimizer.step()
```

### 경사 누적

아래 함수는 배치 수가 accumulation_steps로 나누어떨어지지 않으면 마지막 미완성 그룹을 갱신하지 않고 남긴다. 배치 크기가 다를 때 고정 배수로 로스를 나누는 것도 정확한 전체 표본·토큰 평균과 다를 수 있다. DDP와 결합하면 no_sync와 동기화 시점도 별도로 설계해야 한다.

바라는 실효 배치 크기가 GPU 메모리를 넘으면, 경사 누적가 파라메터를 갱신하기 전에 순방향과 역전파를 여러 번 하며 그래디언트를 쌓아 더 큰 배치를 흉내 낸다.

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
    그래디언트 모으기를 쓰는 학습 고리.
    
    실효 배치 크기 = micro_batch_size × accumulation_steps × num_gpus
    이를테면 4 × 8 × GPU 4개 = 실효 배치 크기 128.
    """
    model.train()
    optimizer.zero_grad()
    
    for step, batch in enumerate(dataloader):
        # 피드포워드와 역전파 (그래디언트가 .grad에 쌓인다)
        loss = model(**batch).loss
        loss = loss / accumulation_steps              # 모으기 단계 수로 나눈다
        loss.backward()
        
        if (step + 1) % accumulation_steps == 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
```

### 모델 병렬 (텐서 병렬)

레이어 하나하나를 여러 장치에 나눈다. 선형 레이어 $Y = XW$에서는 웨이트 행렬 $W$을 열 방향으로 장치 $K$개에 나눈다.

$$
W = [W_1 \mid W_2 \mid \cdots \mid W_K]
$$

장치마다 $Y_k = X W_k$을 계산하고 그 결과를 (나누는 방식에 따라) 이어 붙이거나 줄인다. Megatron-LM이 쓰는 방식이다.

### 파이프라인 병렬

레이어마다 다른 장치를 맡긴다. 입력 작은 배치가 파이프라인을 따라 흐른다.

$$
\text{Device 1: Layers 1–24} \rightarrow \text{Device 2: Layers 25–48} \rightarrow \text{Device 3: Layers 49–72} \rightarrow \text{Device 4: Layers 73–96}
$$

**어려움**: 순진한 파이프라인 병렬은 장치가 순방향이나 역전파를 기다리며 노는 "파이프라인 거품"을 만든다. 작은 배치로 나누기(GPipe)와 엇갈린 일정 짜기(PipeDream)가 노는 시간을 줄인다.

### 3차원 병렬

큰 규모 학습은 세 방법을 모두 아우른다.

- 장치 배치에 걸친 **데이터 병렬**
- 배치 안(대개 노드 하나 안)의 **텐서 병렬**
- 배치에 걸친 **파이프라인 병렬**

이를테면 Megatron-Turing NLG(5300억)는 노드마다 8클래스 텐서 병렬, 노드에 걸쳐 35클래스 파이프라인 병렬, 파이프라인 복제본에 걸쳐 데이터 병렬을 쓴다.

### 효율적으로 키우기 위해 떠오르는 구조

### Megatron-LM

Megatron-LM(엔비디아)은 주의와 순전파 계산을 나누어 트랜스포머 레이어를 위한 효율적인 텐서 병렬을 준다.

- **어텐션**: Q·K·V의 투영을 분할하고 출력 투영의 분할 결과를 합친다. 실제 통신은 병렬화 방식과 구현에 따라 정한다.

통신 횟수·량은 순방향·역전파·병렬 배치와 구현에 따라 달라진다. 특정 계산 도식의 all-reduce 횟수를 모든 트랜스포머 레이어·학습 단계에 동일한 상수로 적용하지 않는다.

### 전문가 혼합 (MoE)

Switch Transformer 같은 전문가 혼합 모델은 입력 토큰마다 파라메터의 일부만 켜서, 계산 비용은 덜 늘리면서 모델 전체의 그릇을 훨씬 크게 한다.

게이트가 토큰마다 상위 $k$명의 전문가를 고른다.

$$
G(x) = \text{TopK}\left(\text{softmax}(x \cdot W_g)\right)
$$

$$
\text{MoE}(x) = \sum_{i \in \text{TopK}} G(x)_i \cdot E_i(x)
$$

여기서 $E_i$은 $i$번째 전문가(대개 순전파 뉴럴 넷트웍)이고 $G(x)_i$은 게이트 웨이트이다.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    """전문가 하나: 표준 자리별 순전파 뉴럴 넷트웍."""
    
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
    상위 K 문을 갖춘 전문가 섞기 레이어.
    
    트랜스포머 블록의 표준 순전파 뉴럴 넷트웍을 대신한다.
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
        
        # 문 뉴럴 넷트웍: 입력을 전문가 점수로 사영한다
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        
        # 전문가 뉴럴 넷트웍
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
            aux_loss: 부하를 고르게 하는 로스(스칼라)
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
        
        # 부하를 고르게 하는 보조 로스
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

**Switch Transformer**: 대표적인 top-1 라우팅 MoE다. 전체 계수와 한 토큰의 활성 계수·통신량은 다르며, 속도 배율을 이 작은 예제나 모든 dense 비교에 그대로 적용할 수 없다.

**부하 균형**: 보조 로스는 일부 전문가에만 토큰이 몰리는 현상을 완화하려는 목표다. 라우팅 균등화나 품질을 보장하지는 않는다. 반환한 aux_loss를 과제 로스에 실제로 더해야 학습에 반영된다.

이 구현은 top-k 선택 확률을 다시 정규화한다. k=1이면 선택 웨이트가 항상 1이 되어 주 로스에서 gate로 가는 연속 경사가 사라질 수 있고, 보조 로스의 역할이 중요해진다. expert capacity·토큰 드롭·분산 all-to-all은 구현하지 않는다. 비연속 입력의 view와 1≤top_k≤전문가 수 조건도 확인한다.

$$
\mathcal{L}_{\text{aux}} = \alpha \cdot N_E \sum_{i=1}^{N_E} f_i \cdot p_i
$$

여기서 $f_i$은 전문가 $i$에게 간 토큰의 비율, $p_i$은 전문가 $i$의 평균 게이트 확률, $N_E$은 전문가의 수이다.

### 효율적으로 키우는 기법

**양자화**: FP32 원시 웨이트를 INT8·INT4로 저장하면 계수당 데이터는 각각 4분의 1·8분의 1이다. 스케일·제로포인트·활성·캐시·런타임 메모리는 별도이므로 전체 메모리가 정확히 같은 배율로 줄지는 않는다. 품질 로스와 속도는 양자화 방식·커널별로 평가한다.

**가지치기**: 군더더기 웨이트를 없애면 모델이 작아진다. 구조 있는 가지치기(어텐션 헤드나 순전파 차원을 통째로 없앰)가 구조 없는(웨이트 하나하나) 가지치기보다 하드웨어에 잘 맞는다.

**증류**: 큰 교사 모델의 출력·표현을 작은 학생 모델의 학습 신호로 사용한다. 계수·속도·성능 절감 비율은 모델·데이터·장치에 따라 달라 특정 보고를 보편적인 수치로 제시하지 않는다.

### 학습 기반 비교

대규모 학습 시스템의 실제 시간은 데이터·토큰 수·정밀도·장치·통신망·가동률을 함께 보고해야 비교할 수 있다. 이 페이지에는 시스템별 학습 시간 벤치마크가 없으므로 “몇 주·몇 달”의 출처 없는 표로 효율을 비교하지 않는다.

### 핵심 내용

트랜스포머를 키우는 일은 데이터와 계산과 메모리에 걸친 복잡한 트레이드오프의 공간을 헤쳐 가는 일이다.

1. **스케일링 법칙**은 계산을 모델 크기와 학습 데이터에 어떻게 나누는 것이 가장 좋은지 길잡이를 준다.
2. **병렬화 방법**(데이터, 텐서, 파이프라인)은 장치 하나의 메모리를 넘는 모델을 학습할 수 있게 한다.
3. **MoE**는 토큰당 활성 계산과 전체 저장 계수를 분리한다. 전체 계수가 작아지는 방법은 아니다.

이 분야는 빠르게 흘러가고 있으며 새로운 구조와 학습 방법이 꾸준히 나오고 있다.

### 참고 문헌

1. Brown, T., et al. (2020). "Language Models are Few-Shot Learners." NeurIPS. (GPT-3)
2. Chowdhery, A., et al. (2022). "PaLM: Scaling Language Modeling with Pathways." arXiv.
3. Touvron, H., et al. (2023). "LLaMA: Open and Efficient Foundation Language Models." arXiv.
4. Kaplan, J., et al. (2020). "Scaling Laws for Neural Language Models." arXiv.
5. Hoffmann, J., et al. (2022). "Training Compute-Optimal Large Language Models." NeurIPS. (Chinchilla)
6. Shoeybi, M., et al. (2020). "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism." arXiv.
7. Fedus, W., et al. (2022). "Switch Transformers: Scaling to Trillion Parameter Models." JMLR.

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
BERT·GPT·T5·XLNet의 학습 문맥과 목표를 비교하라.

</div>

??? success "풀이"
    BERT는 양방향 손상 입력에서 MLM을, GPT는 인과 문맥에서 다음 토큰을, T5는 손상 원문과 이전 목표로 제거 구간을 예측한다. XLNet은 순열 기반 자기 회귀 목표와 정보 유출을 막는 계산 규칙을 사용한다. 문맥·목표의 차이를 모든 과제에서의 우열로 바꾸어 해석하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
RoBERTa와 BERT의 비교에서 바뀐 조건을 설명하라.

</div>

??? success "풀이"
    NSP 제거, 동적 마스킹, 데이터·배치·학습 예산과 바이트 기반 BPE 등 여러 요소가 달라졌다. 성능 차이를 NSP 하나의 효과로 단정하거나 NSP가 모든 설정에서 쓸모없다는 결론으로 일반화하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff easy" title="쉬움"></span>
ALBERT의 임베딩 분해와 레이어 공유가 계수·계산에 미치는 영향을 설명하라.

</div>

??? success "풀이"
    임베딩을 $VH$ 대신 $VE+EH$로 표현하고 레이어 그룹 간 웨이트를 공유해 고유 파라메터를 줄일 수 있다. 공유해도 각 레이어의 계산은 반복하므로 FLOP·활성 메모리가 같은 비율로 줄지는 않는다. 공유 범위·폭·구성에 따라 계수 절감률이 달라진다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
전체 미세 조정·어댑터·프롬프트 튜닝·맥락 내 학습을 구분하라.

</div>

??? success "풀이"
    전체 미세 조정은 기반 계수를 갱신한다. 어댑터나 학습 프롬프트는 주로 작은 추가 계수를 갱신하고 기반을 동결한다. 맥락 내 학습은 추론 시 텍스트 예시를 주며 경사 갱신하지 않는다. 이들이 시간순으로 서로를 완전히 대체한 것은 아니며 목적·비용에 따라 선택한다.

## 정리하며

순환 상태, 컨볼루션 리셉티브 필드와 어텐션 연결은 서로 다른 비용·문맥 제약을 갖는다. 희소 패턴의 수식과 실제 커널, 전체 계수와 토큰당 활성 계수, 학습 FLOP와 배포 비용을 구분한다. 페이지의 코드 조각과 출처 없는 성능 순위를 근거로 보편적인 우열을 판단하지 않는다.

**참고 문헌**

1. Vaswani, A., et al. (2017). "Attention Is All You Need."
2. Hochreiter, S., & Schmidhuber, J. (1997). "Long Short-Term Memory."
3. Kim, Y. (2014). "Convolutional Neural Networks for Sentence Classification."
4. Gu, A., et al. (2022). "Efficiently Modeling Long Sequences with Structured State Spaces."

---
