# 셀프 어텐션

셀프 어텐션은 Q·K·V가 같은 입력 시퀀스의 표현에서 나오는 연산이다. 마스크가 허용하는 위치의 정보를 결합하며, 전체 문맥 또는 과거·현재 문맥을 활용하도록 설계할 수 있다.

셀프 어텐션은 트랜스포머에 힘을 주는 핵심 장치로, 이해를 위한 구조(BERT, RoBERTa)와 생성을 위한 구조(GPT)의 기초를 이룬다. 순차적인 처리를 병렬적이고 내용에 기반한 상호작용으로 바꾸어 순차 데이터를 다루는 방식을 뒤집어 놓았다.

---

## 1. 수학적 정식화

### 셀프 어텐션의 정의

자리가 $n$개이고 임베딩 차원이 $d$인 입력 시퀀스 $\mathbf{X} \in \mathbb{R}^{n \times d}$이 주어졌을 때 다음과 같다.

$$\mathbf{Q} = \mathbf{X}\mathbf{W}^Q, \quad \mathbf{K} = \mathbf{X}\mathbf{W}^K, \quad \mathbf{V} = \mathbf{X}\mathbf{W}^V$$

$$\text{SelfAttention}(\mathbf{X}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

여기서 $\mathbf{W}^Q, \mathbf{W}^K \in \mathbb{R}^{d \times d_k}$이고 $\mathbf{W}^V \in \mathbb{R}^{d \times d_v}$이다.

**일반적인 어텐션과의 핵심 차이는 Q, K, V이 모두 같은 출처 X에서 나온다는 점이다.**

### 왜 $\sqrt{d_k}$으로 배율을 조정하는가

스케일 $1/\sqrt{d_k}$는 독립적인 평균 0·분산 1 성분의 내적에서 분산이 차원에 비례하는 현상을 완화한다. 실제 Q·K는 학습된 투영으로 서로 연관될 수 있다. 스케일만으로 포화를 없애거나 전체 학습이 멈추지 않도록 보장하지는 않는다.

엄밀히 보면 $q_i, k_j$이 서로 독립으로 $\mathcal{N}(0, 1)$을 따를 때 다음이 성립한다.

$$\text{Var}(\mathbf{q}^T \mathbf{k}) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = d_k$$

$\sqrt{d_k}$으로 나누면 분산이 1로 정규화되어 소프트맥스가 얌전한 영역에 머문다.

### Q, K, V의 의미로 보는 해석

토큰마다 서로 다른 표현 세 가지를 만든다.

| 투영 | 의미상의 구실 | 직관 |
|------------|---------------|-----------|
| **쿼리** $\mathbf{q}_i$ | "나는 어떤 정보를 찾는가?" | 토큰이 던지는 물음 |
| **키** $\mathbf{k}_i$ | "나는 어떤 정보를 주는가?" | 토큰이 제 내용을 알리는 방식 |
| **값** $\mathbf{v}_i$ | "뽑히면 나는 무엇을 내놓는가?" | 실제로 모아질 정보 |

이렇게 나누면 토큰이 주는 정보와 찾는 정보가 서로 다를 수 있다. 주어와 동사의 의존처럼 비대칭인 관계를 다루는 데 매우 중요하다.

### 어텐션 행렬 읽기

어텐션 행렬 $\mathbf{A} \in \mathbb{R}^{n \times n}$의 성분 $a_{ij}$은 자리 $i$이 자리 $j$에 얼마나 주목하는지를 나타낸다.

$$a_{ij} = \frac{\exp(\mathbf{q}_i^T \mathbf{k}_j / \sqrt{d_k})}{\sum_{l=1}^{n} \exp(\mathbf{q}_i^T \mathbf{k}_l / \sqrt{d_k})}$$

**성질:**

- 행마다 합이 1이다 (원본 자리에 대한 올바른 확률 분포)
- $a_{ii}$은 자기 자신에 대한 주목을 나타낸다
- 이 행렬은 대체로 **대칭이 아니다** ($a_{ij} \neq a_{ji}$)
- 이 행렬은 **정사각**이다. 자리마다 모든 자리에 주목한다

### 어텐션 행렬 그려 보기

문장 "The cat sat"에 대해 살펴보자.

```
         Keys
         The  cat  sat
        ┌────┬────┬────┐
    The │ .6 │ .2 │ .2 │  Query "The" attends mostly to itself
        ├────┼────┼────┤
Q   cat │ .1 │ .7 │ .2 │  Query "cat" attends mostly to itself
        ├────┼────┼────┤
    sat │ .2 │ .5 │ .3 │  Query "sat" attends to "cat" (subject)
        └────┴────┴────┘
```

이 행렬은 설명을 위한 웨이트 예시다. 드롭아웃 전의 유효 행은 합이 1이며 대칭일 필요는 없다. 숫자 예시만으로 실제 모델이 주어나 의미 관계를 학습했다는 결론을 내릴 수는 없다.

---

## 2. 셀프 어텐션이 전역 문맥을 가능하게 하는 까닭

### 먼 거리 의존 문제

*"The cat sat on the mat because it was soft."*를 생각해 보자.

자리 8의 "it"을 자리 6의 "mat"으로 이어 주려면 모델이 멀리 떨어진 두 자리를 잇대야 한다.

**전통적인 RNN 방식** (순차 처리):

- "mat"의 정보가 자리 7을 지나 자리 8로 전해져야 한다
- 경로의 길이: $O(|i - j|)$
- 그래디언트가 여러 걸음을 지나야 해서 소실하거나 폭발한다

**셀프 어텐션 방식** (병렬 처리):

- 자리 8("it")이 자리 6("mat")에 곧바로 주목한다
- 경로의 길이: $O(1)$ — 거리와 상관없이 일정하다
- 어떤 두 자리 사이에도 그래디언트가 곧바로 흐른다

### 경로 길이 비교

| 구조 | 경로 길이 (자리 $i$에서 $j$까지) | 병렬화 |
|--------------|-----------------------------------|-----------------|
| RNN/LSTM | $O(\|i - j\|)$ | 순차적 |
| CNN | 스트라이드·dilation에 따라 다름 | 병렬 |
| 셀프 어텐션 | $O(1)$ | 완전 병렬 |

허용된 위치 사이의 짧은 경로는 장거리 정보 전달을 도울 수 있다. 하지만 작은 어텐션 웨이트나 다른 연산의 영향도 있으므로 직접 연결이 강한 그래디언트나 좋은 일반화를 보장하지 않는다. 인과 마스크가 막은 방향에는 해당 직접 경로도 없다.

---

## 3. 셀프 어텐션과 크로스 어텐션

| 항목 | 셀프 어텐션 | 크로스 어텐션 |
|--------|----------------|-----------------|
| Q의 출처 | 같은 시퀀스 X | 디코더 시퀀스 |
| K와 V의 출처 | 같은 시퀀스 X | 인코더 시퀀스 |
| 목적 | 내부 문맥 다루기 | 바깥을 참조하고 붙들기 |
| 어텐션의 모양 | 정사각 $(n \times n)$ | 직사각 $(n_q \times n_k)$ |
| 흔한 쓰임 | 인코더나 디코더 안에서 | 디코더가 인코더에 주목할 때 |

셀프 어텐션은 시퀀스 **안**의 관계를 붙잡고, 크로스 어텐션은 시퀀스 **사이**를 잇는다.

---

## 4. 양방향 셀프 어텐션과 인과 셀프 어텐션

### 양방향 (인코더 방식)

자리마다 다른 모든 자리를 볼 수 있다. 양쪽 방향의 온전한 문맥이다.

$$\mathbf{A} = \begin{pmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \\ a_{31} & a_{32} & a_{33} \end{pmatrix}$$

**쓰임새:** 온전한 문맥을 쓸 수 있는 이해 과제(BERT, 문장 분류, 개체명 인식, 쿼리응답).

### 인과·마스크 (디코더 방식)

자리 $i$은 자리 $1, \ldots, i$에만 주목할 수 있다(과거와 현재이며 미래는 아니다).

$$\mathbf{A} = \begin{pmatrix} a_{11} & 0 & 0 \\ a_{21} & a_{22} & 0 \\ a_{31} & a_{32} & a_{33} \end{pmatrix}$$

**쓰임새:** 다음 토큰을 자기회귀적으로 예측하는 생성 과제(GPT, 언어 모델, 텍스트 생성).

인과 마스크는 소프트맥스에 앞서 미래 자리를 $-\infty$으로 두어 구현한다.

$$\text{scores}_{ij} = \begin{cases} \mathbf{q}_i^T \mathbf{k}_j / \sqrt{d_k} & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}$$

### 멀티헤드 어텐션에서의 마스크

멀티헤드 상황에서 인과 마스크를 쓰면 같은 마스크가 모든 헤드에 퍼진다. 헤드마다 서로 다른 어텐션 무늬를 따로 배우지만 모두 같은 인과 제약을 지킨다. 곧 헤드 $h$은 다음을 계산한다.

$$\text{head}_h = \text{softmax}\left(\frac{\mathbf{Q}_h \mathbf{K}_h^T}{\sqrt{d_k}} + \mathbf{M}\right)\mathbf{V}_h$$

여기서 $j \leq i$이면 $\mathbf{M}_{ij} = 0$이고 그렇지 않으면 $\mathbf{M}_{ij} = -\infty$이다. (곱셈이 아니라) 덧셈으로 쓰면 소프트맥스를 지나는 그래디언트가 깨끗하게 흐른다.

### 인코더-디코더의 마스크 방식

온전한 인코더-디코더 트랜스포머에는 서로 다른 마스크 방식 세 가지가 함께 있다.

| 부품 | 마스크 | 까닭 |
|-----------|---------|--------|
| 인코더 셀프 어텐션 | 없음 (양방향) | 원본을 온전히 쓸 수 있다 |
| 디코더 셀프 어텐션 | 인과 마스크 | 자기회귀 성질을 지킨다 |
| 디코더 크로스 어텐션 | 없음 (인코더에 대해) | 원본의 모든 자리에 닿을 수 있다 |

디코더는 덧댐 토큰을 무시하려고 셀프 어텐션과 크로스 어텐션 모두에 **덧댐 마스크**도 쓴다.

---

## 5. 근본적인 성질

### 순열 동변성

위치 정보와 고정 마스크가 없고 드롭아웃이 꺼져 있으면 입력 순열에 대해 출력도 같은 순열을 따른다. 마스크를 쓰면 위치 두 축도 함께 순열해야 같은 성질을 비교할 수 있다.

$$\text{SelfAttn}(\mathbf{P}\mathbf{X}) = \mathbf{P} \cdot \text{SelfAttn}(\mathbf{X})$$

여기서 $\mathbf{P}$은 순열 행렬이다.

**뜻하는 바:** 셀프 어텐션은 자리를 대칭적으로 다루며 순서라는 개념을 스스로 갖고 있지 않다. 위치 정보는 위치 부호화(사인·코사인, 학습형, 회전형 등)로 따로 넣어 주어야 한다.

### 지역성이나 순서에 대한 귀납 편향이 없다

(순차적 바이어스가 있는) RNN이나 (지역 리셉티브 필드가 있는) CNN과 달리 셀프 어텐션에는 다음에 대한 가정이 없다.

- 위치(어떤 토큰이 "가까운지")
- 방향(왼쪽 문맥과 오른쪽 문맥)
- 지역성(가까운 토큰이 더 쓸모 있다는 것)

이는 (어떤 무늬든 배울 수 있는 유연함이라는) **강점**이면서 (데이터가 더 필요하고 위치 부호화를 따로 넣어야 하는) **약점**이기도 하다.

### 계산 복잡도

| 자원 | 복잡도 | 병목 |
|----------|------------|------------|
| 시간 | $O(n^2 d)$ | 시퀀스 길이에 이차 |
| 메모리 | $O(n^2)$ | 어텐션 행렬을 담는 데 든다 |

$O(n^2)$으로 커지는 것이 긴 시퀀스에서 가장 큰 한계이며, 그 때문에 선형 어텐션, Linformer, Performer, FlashAttention 같은 효율적인 판본이 나왔다.

---

## 6. 셀프 어텐션과 완전 연결 레이어

입력 전체를 길이 nd의 벡터로 펼쳐 같은 크기로 보내는 일반 선형 레이어는 O(n²d²)의 웨이트를 가질 수 있다. 반면 토큰마다 공유하는 d→d 선형 레이어는 O(d²)이고 위치 간 정보를 직접 섞지 않는다. “완전 연결 레이어”을 비교할 때 어느 연산을 뜻하는지 명시해야 한다.

셀프 어텐션은 공유된 Q·K·V 투영으로 입력에 따른 위치 간 결합 웨이트를 만든다. 학습된 투영 계수는 고정되어도 그 결합은 입력에 따라 달라진다.

## 7. 셀프 어텐션이 배우는 것

일부 모델 분석에서는 가까운 위치, 구두점이나 문법적 관계에 대응하는 헤드 패턴을 관찰한다. 하지만 레이어별 역할이 모든 모델에서 정해져 있거나 뒤쪽 레이어가 반드시 추론을 담당하는 것은 아니다.

웨이트 히트맵, 탐침 과제와 입력 변경에 따른 반응은 서로 다른 근거다. 어텐션이 큰 위치와 예측의 인과적 기여도는 같지 않으며, 무작위 초기 모델의 형태 시연을 학습된 의미 구조로 해석해서는 안 된다.

## 8. PyTorch 구현

### 기본 셀프 어텐션

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class SelfAttention(nn.Module):
    """
    자기 어텐션 레이어
    
    질의와 열쇠와 값이 모두 같은 입력에서 나오는 어텐션을 계산한다.
    트랜스포머 인코더 레이어에서 쓴다.
    """
    
    def __init__(
        self, 
        d_model: int, 
        d_k: Optional[int] = None, 
        d_v: Optional[int] = None,
        dropout: float = 0.0
    ):
        super().__init__()
        self.d_k = d_k or d_model
        self.d_v = d_v or d_model
        self.scale = self.d_k ** -0.5
        
        # Q, K, V의 선형 사영
        self.W_q = nn.Linear(d_model, self.d_k)
        self.W_k = nn.Linear(d_model, self.d_k)
        self.W_v = nn.Linear(d_model, self.d_v)
        
        # 출력 사영
        self.out_proj = nn.Linear(self.d_v, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        인수:
            x: 입력 순차열 (배치 크기, seq_len, d_model)
            mask: 선택적인 어텐션 가림막 (배치 크기, seq_len, seq_len)
                  0은 가릴 자리를 뜻한다
            
        반환값:
            output: 자기 어텐션을 거친 출력 (배치 크기, seq_len, d_model)
            attention_weights: 어텐션 행렬 (배치 크기, seq_len, seq_len)
        """
        # Q, K, V로 사영 (모두 같은 입력 x에서 나온다)
        Q = self.W_q(x)  # (배치, seq_len, d_k)
        K = self.W_k(x)  # (배치, seq_len, d_k)
        V = self.W_v(x)  # (배치, seq_len, d_v)
        
        # 배율 조정 내적 어텐션 점수 계산
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # 가림막이 있으면 씌우기
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # 열쇠(마지막 차원)에 대해 소프트맥스
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 값의 가중합
        attended = torch.matmul(attention_weights, V)
        
        # 출력 사영
        output = self.out_proj(attended)
        
        return output, attention_weights

def demonstrate_self_attention():
    """기본 자기 어텐션 시연."""
    d_model = 512
    seq_len = 10
    batch_size = 2
    
    self_attn = SelfAttention(d_model)
    X = torch.randn(batch_size, seq_len, d_model)
    
    output, weights = self_attn(X)
    
    print(f"Input shape:     {X.shape}")        # (2, 10, 512)
    print(f"Output shape:    {output.shape}")   # (2, 10, 512)
    print(f"Attention shape: {weights.shape}")  # (2, 10, 10)
    print(f"\nAttention matrix is square: {weights.shape[-2]} x {weights.shape[-1]}")
    print(f"Each row sums to 1: {weights[0, 0].sum().item():.4f}")
```

### 인과 셀프 어텐션 (자기회귀 모델용)

고정 크기의 하삼각 마스크 버퍼를 사용하므로 입력 길이는 max_seq_len 이하여야 한다. PAD 키를 제외하는 별도 인수는 없고, return_attention=False여도 웨이트 배열을 내부에서 계산한다. 드롭아웃 적용 후 웨이트는 행 합이 1이 아닐 수 있다.

```python
class CausalSelfAttention(nn.Module):
    """
    다중 머리를 지원하는 인과(가림막) 자기 어텐션
    
    자리마다 뒤의 자리에 주목하지 못하게 막는다.
    자기회귀 생성을 하는 GPT 같은 디코더 전용 모델에서 쓴다.
    """
    
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int, 
        max_seq_len: int = 2048, 
        dropout: float = 0.0
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # 효율을 위해 QKV 사영을 합침 (행렬 곱 세 번 대신 한 번)
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 인과 가림막을 버퍼로 등록 (파라메터는 아니지만 모델과 함께 움직인다)
        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer('causal_mask', causal_mask.view(1, 1, max_seq_len, max_seq_len))
        
    def forward(
        self, 
        x: torch.Tensor,
        return_attention: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        인수:
            x: 입력 순차열 (배치 크기, seq_len, embed_dim)
            return_attention: 어텐션 웨이트를 돌려줄지 여부
            
        반환값:
            output: 어텐션을 거친 출력 (배치 크기, seq_len, embed_dim)
            attention_weights: 선택적으로 (배치 크기, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape
        
        # 효율적인 연산 한 번으로 Q, K, V 사영
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, 배치, heads, seq, head_dim)
        Q, K, V = qkv[0], qkv[1], qkv[2]
        
        # 어텐션 점수 계산: (배치, heads, seq, seq)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # 인과 가림막 씌우기 (자리마다 과거와 현재에만 주목할 수 있다)
        mask = self.causal_mask[:, :, :seq_len, :seq_len]
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # 소프트맥스와 선택적인 드롭아웃
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 값에 어텐션 적용
        attended = torch.matmul(attention_weights, V)
        
        # 머리를 이어 붙이고 되사영
        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(attended)
        
        if return_attention:
            return output, attention_weights
        return output, None

def demonstrate_causal_attention():
    """인과 가림막의 무늬를 보인다."""
    batch_size, seq_len, embed_dim, num_heads = 1, 5, 64, 4
    
    x = torch.randn(batch_size, seq_len, embed_dim)
    causal_attn = CausalSelfAttention(embed_dim, num_heads)
    
    output, weights = causal_attn(x)
    
    print("Causal Attention Pattern (first head):")
    print("Each row shows what that position attends to.")
    print("Position i can only attend to positions <= i (lower triangular).\n")
    print(weights[0, 0].detach().numpy().round(3))
    print("\nNote: Upper triangle is 0 (future positions masked)")

if __name__ == "__main__":
    demonstrate_causal_attention()
```

**출력:**
```
Causal Attention Pattern (first head):
Each row shows what that position attends to.
Position i can only attend to positions <= i (lower triangular).

[[1.    0.    0.    0.    0.   ]
 [0.423 0.577 0.    0.    0.   ]
 [0.298 0.351 0.351 0.    0.   ]
 [0.221 0.264 0.258 0.257 0.   ]
 [0.178 0.213 0.207 0.201 0.201]]

Note: Upper triangle is 0 (future positions masked)
```

---

## 9. 다른 장치와 비교

### 셀프 어텐션과 RNN

| 항목 | 셀프 어텐션 | RNN/LSTM |
|--------|---------------|----------|
| 먼 거리 의존 | 경로 길이 $O(1)$ | 경로 길이 $O(n)$ |
| 병렬화 | 완전 병렬 | 순차적 (본디 직렬) |
| 레이어당 계산 | $O(n^2 d)$ | $O(n d^2)$ |
| 메모리 | $O(n^2)$ | $O(n)$ |
| 그래디언트의 흐름 | 곧바른 연결 | 순환 걸음을 거침 |
| 귀납 바이어스 | 없음 | 순차적·시간적 |

**셀프 어텐션이 나을 때:** 전역 문맥이 필요한 과제, 병렬 학습, 먼 거리 의존이 중요한 긴 시퀀스.

**RNN이 나을 때:** 흐름 처리 응용, 메모리가 빠듯한 환경, 순차적·시간적 구조가 뚜렷한 과제.

### 셀프 어텐션과 컨볼루션

| 항목 | 셀프 어텐션 | 컨볼루션 |
|--------|---------------|-------------|
| 리셉티브 필드 | 전역 (시퀀스 전체) | 지역 (커널 크기 $k$) |
| 파라메터 공유 | 어디서나 같은 Q, K, V | 어디서나 같은 커널 |
| 귀납 바이어스 | 없음 | 평행 이동 동변성, 지역성 |
| 계산 | 점수·가중합 $O(n^2d)$ | 일반적인 d→d Conv1d는 $O(knd^2)$ |
| 먼 거리 | 곧바로 | 쌓거나 팽창해야 한다 |

**착상:** CNN은 가까운 원소끼리 어울린다는 지역성 바이어스가 강하다. 셀프 어텐션은 거리와 상관없이 어떤 원소끼리 어울려야 하는지를 배운다.

---

## 10. 응용

### 트랜스포머 인코더 블록 (BERT 방식)

내장 MultiheadAttention의 불리언 마스크는 True가 제외 위치다. 위 사용자 정의 클래스의 0 제외 규칙과 반대이므로 그대로 넘기지 않는다. 이 블록은 어텐션과 FFN의 구조 예시이며 토큰·위치 임베딩과 학습 목표까지 포함한 BERT는 아니다.

```python
class TransformerEncoderBlock(nn.Module):
    """
    양방향 자기 어텐션이 있는 트랜스포머 인코더 블록 하나.
    
    구조: 자기 어텐션 → 더하기와 정규화 → 순방향 뉴럴 넷트웍 → 더하기와 정규화
    """
    
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int, 
        ff_dim: int, 
        dropout: float = 0.1
    ):
        super().__init__()
        
        # 다중 머리 자기 어텐션 (양방향)
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # 자리별 순방향 뉴럴 넷트웍
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        src_mask: torch.Tensor = None,
        src_key_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        인수:
            x: 입력 (배치, seq_len, embed_dim)
            src_mask: 어텐션 가림막 (seq_len, seq_len)
            src_key_padding_mask: 덧댐 가림막 (배치, seq_len)
        """
        # 잔차 연결이 있는 자기 어텐션 (사전 레이어 정규화 판본)
        attn_out, _ = self.self_attn(
            x, x, x, 
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )
        x = self.norm1(x + self.dropout(attn_out))
        
        # 잔차 연결이 있는 순방향 뉴럴 넷트웍
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x
```

### 비전 트랜스포머(ViT)의 셀프 어텐션

이미지를 시퀀스로 다루어 이미지 조각에 셀프 어텐션을 적용한다.

```python
class VisionSelfAttention(nn.Module):
    """
    이미지 조각을 위한 자기 어텐션 (비전 트랜스포머 방식).
    
    이미지를 조각으로 나누어 펼친 뒤 순차열로 다룬다.
    조각마다 다른 모든 조각에 전역적으로 주목할 수 있다.
    """
    
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int,
        dropout: float = 0.0
    ):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, 
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, patches: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        인수:
            patches: 펼친 이미지 조각 (배치, num_patches, embed_dim)
                     대체로 첫 자리에 [CLS] 토큰이 있다
        
        반환값:
            output: 자기 어텐션을 거친 조각 (배치, num_patches, embed_dim)
            weights: 어텐션 웨이트 (배치, num_patches, num_patches)
        """
        # 조각마다 (CLS 토큰을 포함한) 다른 모든 조각에 주목한다
        attended, weights = self.attention(patches, patches, patches)
        
        # 잔차 연결
        output = self.norm(patches + attended)
        
        return output, weights
```

### 여러 맥락에서의 셀프 어텐션

| 맥락 | 어텐션의 종류 | 핵심 피처 |
|---------|---------------|---------------------|
| **BERT 인코더** | 양방향 | 온전한 문맥, 모으기에 [CLS] 토큰 사용 |
| **GPT 디코더** | 인과 | 자기회귀적으로 다음 토큰을 예측 |
| **비전 트랜스포머** | 양방향 | 조각을 토큰으로, 분류에 [CLS] 사용 |
| **음향 (Wav2Vec)** | 양방향 | 날 파형이나 스펙트럼 피처 |
| **단백질 (ESM)** | 양방향 | 아미노산을 토큰으로 |

---

## 11. 효율적인 셀프 어텐션 판본

$O(n^2)$의 복잡도 때문에 효율적인 대안이 많이 나왔다.

| 방법 | 복잡도 | 핵심 착상 |
|--------|------------|----------|
| **성긴 어텐션** | $O(n\sqrt{n})$ | 정해진 무늬(지역과 스트라이드)에만 주목한다 |
| **Linformer** | $O(n)$ | K와 V를 낮은 계수로 투영한다 |
| **Performer** | $O(n)$ | 무작위 피처로 소프트맥스를 어림한다 |
| **선형 어텐션** | $O(n)$ | 소프트맥스를 없애고 커널 트릭을 쓴다 |
| **FlashAttention** | 시간 $O(n^2)$, 메모리 $O(n)$ | 입출력을 고려한 타일 계산 |
| **미끄럼창** | $O(nw)$ | 창 크기가 $w$인 지역 어텐션 |

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
셀프 어텐션과 크로스 어텐션에서 Q·K·V의 출처를 설명하라.

</div>

??? success "연습문제 1 풀이"
    셀프 어텐션은 같은 시퀀스에서 Q·K·V를 각각 투영한다. 출처가 같아도 투영 웨이트가 달라 Q=K=V라고 할 수는 없다. 크로스 어텐션은 쿼리와 키·값의 출처가 다르며 길이도 다를 수 있다. 어느 위치를 볼지는 마스크가 추가로 정한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
위치 표현과 마스크가 없는 셀프 어텐션의 순열 동변성을 보여라.

</div>

??? success "연습문제 2 풀이"
    순열 P에 대해 투영은 $Q\mapsto PQ,K\mapsto PK,V\mapsto PV$다. 점수는 $S\mapsto PSP^\top$이고 행별 소프트맥스도 같은 행·열 순열을 따른다. 따라서 $A\mapsto PAP^\top$이고 출력은 $PAP^\top PV=PAV$다. 고정 마스크나 위치 표현을 그대로 두면 임의의 입력 순열에 같은 주장을 할 수 없다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff easy" title="쉬움"></span>
전체 어텐션 행렬을 저장할 때 n=4096인 경우의 메모리를 계산하고 한계를 설명하라.

</div>

??? success "연습문제 3 풀이"
    float32 행렬 하나는 $4096^2\times4=64$ MiB다. 배치 1·12헤드·12레이어를 모두 보관하면 9216 MiB, 즉 9 GiB다. 다른 상태와 그래디언트는 별도다. 실제 메모리는 융합 커널·자료형·저장 전략에 따라 달라지며 이 수치가 모든 트랜스포머의 최대 길이를 512~2048로 제한한다는 뜻은 아니다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
미래 자리에 주목하지 못하게 마스크를 씌워 인과(자기회귀) 셀프 어텐션을 구현하라.

</div>

??? success "연습문제 4 풀이"
    ```python
    def causal_self_attention(X, WQ, WK, WV, d_k):
        Q, K, V = X @ WQ, X @ WK, X @ WV
        scores = Q @ K.T / d_k**0.5
        # 인과 가림막: 미래 자리에 -inf
        mask = torch.triu(torch.ones(n, n), diagonal=1).bool()
        scores.masked_fill_(mask, float('-inf'))
        return torch.softmax(scores, dim=-1) @ V
    ```

    마지막 연습문제의 함수는 외부 변수 n에 의존하고 `K.T`를 사용하므로 2차원 입력을 가정한다. n은 입력 길이에서 구해야 하며 배치 입력에는 마지막 두 축 전치와 브로드캐스트 가능한 마스크가 필요하다. 코드 조각을 모든 입력 형태의 완성 구현으로 해석하지 않는다.

## 정리하며

셀프 어텐션은 같은 시퀀스의 표현을 내용에 따라 결합한다. 접근 가능한 위치와 순열 성질은 마스크·위치 표현에 달려 있다. 직접 경로와 병렬 학습의 이점을 전체 그래디언트 보존이나 생성 병렬화 보장으로 확대하지 않는다.

**참고 문헌**

- Vaswani et al., "Attention Is All You Need" (2017) — 최초의 트랜스포머 논문
- Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers" (2019)
- Radford et al., "Language Models are Unsupervised Multitask Learners" (GPT-2, 2019)
- Dosovitskiy et al., "An Image is Worth 16x16 Words" (ViT, 2020)
- Clark et al., "What Does BERT Look At?" (2019) — 어텐션 분석
