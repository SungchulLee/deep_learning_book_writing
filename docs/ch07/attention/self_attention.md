# 자기 어텐션
## 들어가며

자기 어텐션은 질의와 열쇠와 값이 모두 **같은 입력 순차열**에서 나오는 어텐션의 특수한 경우이다. 순차열의 자리마다 (자기 자신을 포함한) 모든 자리에 주목할 수 있어, 바깥 문맥 없이도 한 순차열 안의 관계와 의존을 붙잡는다.

자기 어텐션은 트랜스포머에 힘을 주는 핵심 장치로, 이해를 위한 구조(BERT, RoBERTa)와 생성을 위한 구조(GPT)의 바탕을 이룬다. 순차적인 처리를 병렬적이고 내용에 기반한 상호작용으로 바꾸어 순차 데이터를 다루는 방식을 뒤집어 놓았다.

## 수학적 정식화

### 자기 어텐션의 정의

자리가 $n$개이고 임베딩 차원이 $d$인 입력 순차열 $\mathbf{X} \in \mathbb{R}^{n \times d}$이 주어졌을 때 다음과 같다.

$$\mathbf{Q} = \mathbf{X}\mathbf{W}^Q, \quad \mathbf{K} = \mathbf{X}\mathbf{W}^K, \quad \mathbf{V} = \mathbf{X}\mathbf{W}^V$$

$$\text{SelfAttention}(\mathbf{X}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

여기서 $\mathbf{W}^Q, \mathbf{W}^K \in \mathbb{R}^{d \times d_k}$이고 $\mathbf{W}^V \in \mathbb{R}^{d \times d_v}$이다.

**일반적인 어텐션과의 핵심 차이는 Q, K, V이 모두 같은 출처 X에서 나온다는 점이다.**

### 왜 $\sqrt{d_k}$으로 배율을 조정하는가

배율 인수 $\frac{1}{\sqrt{d_k}}$은 내적의 크기가 너무 커지는 것을 막는다. $d_k$이 크면 (성분이 서로 독립이고 평균 0, 분산 1이라 할 때) 내적 $\mathbf{q}_i^T \mathbf{k}_j$의 분산이 $d_k$에 비례하는 경향이 있다. 내적이 크면 소프트맥스가 기울기가 아주 작은 영역으로 밀려나 사실상 딱딱한 argmax처럼 움직이고 학습이 멈춘다.

엄밀히 보면 $q_i, k_j$이 서로 독립으로 $\mathcal{N}(0, 1)$을 따를 때 다음이 성립한다.

$$\text{Var}(\mathbf{q}^T \mathbf{k}) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = d_k$$

$\sqrt{d_k}$으로 나누면 분산이 1로 정규화되어 소프트맥스가 얌전한 영역에 머문다.

### Q, K, V의 의미로 보는 해석

토큰마다 서로 다른 표현 세 가지를 만든다.

| 사영 | 의미상의 구실 | 직관 |
|------------|---------------|-----------|
| **질의** $\mathbf{q}_i$ | "나는 어떤 정보를 찾는가?" | 토큰이 던지는 물음 |
| **열쇠** $\mathbf{k}_i$ | "나는 어떤 정보를 주는가?" | 토큰이 제 내용을 알리는 방식 |
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

행마다 확률 분포이다(합이 1이다). 비대칭성은 "sat"이 (주어를 찾느라) "cat"에 세게 주목하지만 "cat"은 "sat"에 그만큼 주목하지 않음을 보여 준다.

## 자기 어텐션이 전역 문맥을 가능하게 하는 까닭

### 먼 거리 의존 문제

*"The cat sat on the mat because it was soft."*를 생각해 보자.

자리 8의 "it"을 자리 6의 "mat"으로 이어 주려면 모델이 멀리 떨어진 두 자리를 잇대야 한다.

**전통적인 RNN 방식** (순차 처리):

- "mat"의 정보가 자리 7을 지나 자리 8로 전해져야 한다
- 경로의 길이: $O(|i - j|)$
- 기울기가 여러 걸음을 지나야 해서 소실하거나 폭발한다

**자기 어텐션 방식** (병렬 처리):

- 자리 8("it")이 자리 6("mat")에 곧바로 주목한다
- 경로의 길이: $O(1)$ — 거리와 상관없이 일정하다
- 어떤 두 자리 사이에도 기울기가 곧바로 흐른다

### 경로 길이 견주기

| 구조 | 경로 길이 (자리 $i$에서 $j$까지) | 병렬화 |
|--------------|-----------------------------------|-----------------|
| RNN/LSTM | $O(\|i - j\|)$ | 순차적 |
| CNN | 핵이 $k$일 때 $O(\log_{k}\|i - j\|)$ | 병렬 |
| 자기 어텐션 | $O(1)$ | 완전 병렬 |

경로가 짧으면 기울기가 잘 흘러 먼 거리 의존을 배우기 좋다. 그래서 전역적인 이해가 필요한 과제에서 트랜스포머가 뛰어나다.

## 자기 어텐션과 교차 어텐션

| 항목 | 자기 어텐션 | 교차 어텐션 |
|--------|----------------|-----------------|
| Q의 출처 | 같은 순차열 X | 복호기 순차열 |
| K와 V의 출처 | 같은 순차열 X | 부호기 순차열 |
| 목적 | 내부 문맥 다루기 | 바깥을 참조하고 붙들기 |
| 어텐션의 모양 | 정사각 $(n \times n)$ | 직사각 $(n_q \times n_k)$ |
| 흔한 쓰임 | 부호기나 복호기 안에서 | 복호기가 부호기에 주목할 때 |

자기 어텐션은 순차열 **안**의 관계를 붙잡고, 교차 어텐션은 순차열 **사이**를 잇는다.

## 양방향 자기 어텐션과 인과 자기 어텐션

### 양방향 (부호기 방식)

자리마다 다른 모든 자리를 볼 수 있다. 양쪽 방향의 온전한 문맥이다.

$$\mathbf{A} = \begin{pmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \\ a_{31} & a_{32} & a_{33} \end{pmatrix}$$

**쓰임새:** 온전한 문맥을 쓸 수 있는 이해 과제(BERT, 문장 분류, 개체명 인식, 질의응답).

### 인과·가림막 (복호기 방식)

자리 $i$은 자리 $1, \ldots, i$에만 주목할 수 있다(과거와 현재이며 미래는 아니다).

$$\mathbf{A} = \begin{pmatrix} a_{11} & 0 & 0 \\ a_{21} & a_{22} & 0 \\ a_{31} & a_{32} & a_{33} \end{pmatrix}$$

**쓰임새:** 다음 토큰을 자기회귀적으로 예측하는 생성 과제(GPT, 언어 모형, 텍스트 생성).

인과 가림막은 소프트맥스에 앞서 미래 자리를 $-\infty$으로 두어 구현한다.

$$\text{scores}_{ij} = \begin{cases} \mathbf{q}_i^T \mathbf{k}_j / \sqrt{d_k} & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}$$

### 다중 머리 어텐션에서의 가림막

다중 머리 상황에서 인과 가림막을 쓰면 같은 가림막이 모든 머리에 퍼진다. 머리마다 서로 다른 어텐션 무늬를 따로 배우지만 모두 같은 인과 제약을 지킨다. 곧 머리 $h$은 다음을 계산한다.

$$\text{head}_h = \text{softmax}\left(\frac{\mathbf{Q}_h \mathbf{K}_h^T}{\sqrt{d_k}} + \mathbf{M}\right)\mathbf{V}_h$$

여기서 $j \leq i$이면 $\mathbf{M}_{ij} = 0$이고 그렇지 않으면 $\mathbf{M}_{ij} = -\infty$이다. (곱셈이 아니라) 덧셈으로 쓰면 소프트맥스를 지나는 기울기가 깨끗하게 흐른다.

### 부호기-복호기의 가림막 방식

온전한 부호기-복호기 트랜스포머에는 서로 다른 가림막 방식 세 가지가 함께 있다.

| 부품 | 가림막 | 까닭 |
|-----------|---------|--------|
| 부호기 자기 어텐션 | 없음 (양방향) | 원본을 온전히 쓸 수 있다 |
| 복호기 자기 어텐션 | 인과 가림막 | 자기회귀 성질을 지킨다 |
| 복호기 교차 어텐션 | 없음 (부호기에 대해) | 원본의 모든 자리에 닿을 수 있다 |

복호기는 덧댐 토큰을 무시하려고 자기 어텐션과 교차 어텐션 모두에 **덧댐 가림막**도 쓴다.

## 근본적인 성질

### 순열 동변성

입력의 자리를 뒤섞으면 출력도 똑같이 뒤섞인다.

$$\text{SelfAttn}(\mathbf{P}\mathbf{X}) = \mathbf{P} \cdot \text{SelfAttn}(\mathbf{X})$$

여기서 $\mathbf{P}$은 순열 행렬이다.

**뜻하는 바:** 자기 어텐션은 자리를 대칭적으로 다루며 순서라는 개념을 스스로 갖고 있지 않다. 위치 정보는 위치 부호화(사인·코사인, 학습형, 회전형 등)로 따로 넣어 주어야 한다.

### 지역성이나 순서에 대한 귀납 편향이 없다

(순차적 편향이 있는) RNN이나 (지역 수용 영역이 있는) CNN과 달리 자기 어텐션에는 다음에 대한 가정이 없다.

- 위치(어떤 토큰이 "가까운지")
- 방향(왼쪽 문맥과 오른쪽 문맥)
- 지역성(가까운 토큰이 더 쓸모 있다는 것)

이는 (어떤 무늬든 배울 수 있는 유연함이라는) **강점**이면서 (데이터가 더 필요하고 위치 부호화를 따로 넣어야 하는) **약점**이기도 하다.

### 계산 복잡도

| 자원 | 복잡도 | 병목 |
|----------|------------|------------|
| 시간 | $O(n^2 d)$ | 순차열 길이에 이차 |
| 메모리 | $O(n^2)$ | 어텐션 행렬을 담는 데 든다 |

$O(n^2)$으로 커지는 것이 긴 순차열에서 가장 큰 한계이며, 그 때문에 선형 어텐션, Linformer, Performer, FlashAttention 같은 효율적인 판본이 나왔다.

## 자기 어텐션과 완전 연결층

자기 어텐션이 완전 연결층과 비슷해 보일 수 있지만 근본적으로 다르다.

| 항목 | 완전 연결 | 자기 어텐션 |
|--------|-----------------|----------------|
| 가중치 | 정적 (학습 뒤 고정) | 동적 (입력에서 계산) |
| 자리 의존 | 자리마다 가중치가 다름 | 모든 자리가 같은 Q, K, V 행렬을 씀 |
| 적응성 | 뻣뻣함 (늘 같은 변환) | 내용에 맞춤 (입력에 따라 섞음) |
| 매개변수 수 | 순차열에 대해 $O(n^2 d^2)$ | 순차열 길이와 무관하게 $O(d^2)$ |

**핵심 착상:** 자기 어텐션은 **동적이고 내용에 기반한** 섞음 가중치를 계산하고, 완전 연결층은 **정적이고 학습된** 변환을 적용한다.

## 자기 어텐션이 배우는 것

탐침 분류기와 어텐션 시각화 연구는 층마다 서로 다른 무늬를 배운다는 것을 밝혔다.

### 앞쪽 층
- 지역적인 무늬 (이웃 토큰에 대한 주목)
- 위치의 무늬 ("앞 낱말에 주목" 같은 고정된 자리 차이)
- 기본적인 문법 (문장 부호, 기능어, 관사)

### 중간 층
- 문법적 관계 (주어와 동사의 일치, 수식어와 머리)
- 상호 참조 해소 (대명사와 선행사 잇기)
- 개체명 인식의 무늬
- 구의 구조

### 뒤쪽 층
- 과제에 특화된 무늬
- 의미 관계와 추론
- 먼 거리 의존
- 추상적인 특징의 조합

## PyTorch 구현

### 기본 자기 어텐션

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class SelfAttention(nn.Module):
    """
    자기 어텐션 층
    
    질의와 열쇠와 값이 모두 같은 입력에서 나오는 어텐션을 계산한다.
    트랜스포머 부호기 층에서 쓴다.
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

### 인과 자기 어텐션 (자기회귀 모델용)

```python
class CausalSelfAttention(nn.Module):
    """
    다중 머리를 지원하는 인과(가림막) 자기 어텐션
    
    자리마다 뒤의 자리에 주목하지 못하게 막는다.
    자기회귀 생성을 하는 GPT 같은 복호기 전용 모델에서 쓴다.
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
        
        # 인과 가림막을 버퍼로 등록 (매개변수는 아니지만 모델과 함께 움직인다)
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
            return_attention: 어텐션 가중치를 돌려줄지 여부
            
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

## 다른 장치와 견주기

### 자기 어텐션과 RNN

| 항목 | 자기 어텐션 | RNN/LSTM |
|--------|---------------|----------|
| 먼 거리 의존 | 경로 길이 $O(1)$ | 경로 길이 $O(n)$ |
| 병렬화 | 완전 병렬 | 순차적 (본디 직렬) |
| 층당 계산 | $O(n^2 d)$ | $O(n d^2)$ |
| 메모리 | $O(n^2)$ | $O(n)$ |
| 기울기의 흐름 | 곧바른 연결 | 순환 걸음을 거침 |
| 귀납 편향 | 없음 | 순차적·시간적 |

**자기 어텐션이 나을 때:** 전역 문맥이 필요한 과제, 병렬 학습, 먼 거리 의존이 중요한 긴 순차열.

**RNN이 나을 때:** 흐름 처리 응용, 메모리가 빠듯한 환경, 순차적·시간적 구조가 뚜렷한 과제.

### 자기 어텐션과 합성곱

| 항목 | 자기 어텐션 | 합성곱 |
|--------|---------------|-------------|
| 수용 영역 | 전역 (순차열 전체) | 지역 (핵 크기 $k$) |
| 매개변수 공유 | 어디서나 같은 Q, K, V | 어디서나 같은 핵 |
| 귀납 편향 | 없음 | 평행 이동 동변성, 지역성 |
| 계산 | $O(n^2 d)$ | $O(k n d)$ |
| 먼 거리 | 곧바로 | 쌓거나 팽창해야 한다 |

**착상:** CNN은 가까운 원소끼리 어울린다는 지역성 편향이 강하다. 자기 어텐션은 거리와 상관없이 어떤 원소끼리 어울려야 하는지를 배운다.

## 응용

### 트랜스포머 부호기 블록 (BERT 방식)

```python
class TransformerEncoderBlock(nn.Module):
    """
    양방향 자기 어텐션이 있는 트랜스포머 부호기 블록 하나.
    
    구조: 자기 어텐션 → 더하기와 정규화 → 순방향 신경망 → 더하기와 정규화
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
        
        # 자리별 순방향 신경망
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
        # 잔차 연결이 있는 자기 어텐션 (사전 층 정규화 판본)
        attn_out, _ = self.self_attn(
            x, x, x, 
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )
        x = self.norm1(x + self.dropout(attn_out))
        
        # 잔차 연결이 있는 순방향 신경망
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x
```

### 비전 트랜스포머(ViT)의 자기 어텐션

이미지를 순차열로 다루어 이미지 조각에 자기 어텐션을 적용한다.

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
            weights: 어텐션 가중치 (배치, num_patches, num_patches)
        """
        # 조각마다 (CLS 토큰을 포함한) 다른 모든 조각에 주목한다
        attended, weights = self.attention(patches, patches, patches)
        
        # 잔차 연결
        output = self.norm(patches + attended)
        
        return output, weights
```

### 여러 맥락에서의 자기 어텐션

| 맥락 | 어텐션의 종류 | 핵심 특징 |
|---------|---------------|---------------------|
| **BERT 부호기** | 양방향 | 온전한 문맥, 모으기에 [CLS] 토큰 사용 |
| **GPT 복호기** | 인과 | 자기회귀적으로 다음 토큰을 예측 |
| **비전 트랜스포머** | 양방향 | 조각을 토큰으로, 분류에 [CLS] 사용 |
| **음향 (Wav2Vec)** | 양방향 | 날 파형이나 스펙트럼 특징 |
| **단백질 (ESM)** | 양방향 | 아미노산을 토큰으로 |

## 효율적인 자기 어텐션 판본

$O(n^2)$의 복잡도 때문에 효율적인 대안이 많이 나왔다.

| 방법 | 복잡도 | 핵심 착상 |
|--------|------------|----------|
| **성긴 어텐션** | $O(n\sqrt{n})$ | 정해진 무늬(지역과 보폭)에만 주목한다 |
| **Linformer** | $O(n)$ | K와 V를 낮은 계수로 사영한다 |
| **Performer** | $O(n)$ | 무작위 특징으로 소프트맥스를 어림한다 |
| **선형 어텐션** | $O(n)$ | 소프트맥스를 없애고 핵 요령을 쓴다 |
| **FlashAttention** | 시간 $O(n^2)$, 메모리 $O(n)$ | 입출력을 고려한 타일 계산 |
| **미끄럼창** | $O(nw)$ | 창 크기가 $w$인 지역 어텐션 |

## 요약

자기 어텐션의 특징은 다음과 같다.

| 성질 | 설명 |
|----------|-------------|
| **출처** | Q, K, V이 모두 같은 순차열에서 나온다 |
| **범위** | 자리마다 모든 자리에 주목한다 |
| **경로 길이** | 어떤 두 자리 사이에도 $O(1)$ |
| **복잡도** | 시간 $O(n^2 d)$, 공간 $O(n^2)$ |
| **귀납 편향** | 없음 (위치 부호화가 필요하다) |
| **핵심 이점** | 병렬 계산으로 얻는 전역 문맥 |

자기 어텐션은 자리마다 다른 모든 자리에 주목하게 하여 순차열 안의 관계를 곧바로 다룬다. 트랜스포머에 힘을 주는 장치이며, 자연어 처리와 시각을 비롯한 여러 분야를 뒤집어 놓은 이해(BERT)와 생성(GPT) 구조의 바탕을 이룬다.

**핵심 착상:**

1. **역할의 분리:** Q, K, V 덕분에 토큰이 답하는 것과 다른 물음을 던질 수 있어 비대칭인 관계를 다룰 수 있다.

2. **내용으로 찾기:** RNN이나 CNN의 고정된 연결과 달리 어텐션의 무늬는 입력의 내용에 따라 그때그때 달라진다.

3. **위치에 무관:** 구조 자체는 위치를 모르므로 따로 넣어 주어야 하며, 그 덕분에 위치를 담는 방식을 자유롭게 고를 수 있다.

4. **규모의 맞바꿈:** 전역 문맥에는 이차 비용이 따르며, 그 때문에 효율적인 어텐션 판본에 대한 연구가 풍성하다.

## 참고 문헌

- Vaswani et al., "Attention Is All You Need" (2017) — 최초의 트랜스포머 논문
- Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers" (2019)
- Radford et al., "Language Models are Unsupervised Multitask Learners" (GPT-2, 2019)
- Dosovitskiy et al., "An Image is Worth 16x16 Words" (ViT, 2020)
- Clark et al., "What Does BERT Look At?" (2019) — 어텐션 분석

## 연습문제

**연습문제 1.**
자기 어텐션과 교차 어텐션의 차이를 설명하라.

??? success "연습문제 1 풀이"
    자기 어텐션에서는 질의와 열쇠와 값이 모두 같은 순차열에서 나온다($Q = K = V = X$). 자리마다 같은 순차열의 모든 자리에 주목한다. 교차 어텐션에서는 질의가 한 순차열에서, 열쇠와 값이 다른 순차열에서 온다(seq2seq에서 복호기가 부호기의 출력에 주목하는 것 따위).

---

**연습문제 2.**
자기 어텐션이 순열 동변임을 보여라. 곧 입력의 순서를 바꾸면 출력의 순서도 똑같이 바뀜을 보여라.

??? success "연습문제 2 풀이"
    $\pi$을 순열이라 하고 $X' = \pi(X)$이라 하자. 그러면 $Q' = X'W_Q = \pi(X)W_Q = \pi(Q)$이고 $K'$과 $V'$도 마찬가지이다. 어텐션은 $\text{softmax}(Q'K'^\top/\sqrt{d})V' = \pi(\text{softmax}(QK^\top/\sqrt{d})V)$이 된다. 출력의 순서가 똑같이 바뀐다. 그래서 위치 부호화가 필요하다.

---

**연습문제 3.**
순차열의 길이가 $n$일 때 자기 어텐션의 메모리 복잡도는 얼마인가? 그것이 실제 쓰임을 어떻게 제한하는가?

??? success "연습문제 3 풀이"
    어텐션 행렬이 $n \times n$이라 메모리가 $O(n^2)$ 든다. float32에서 $n = 4096$이면 층마다 머리마다 $4096^2 \times 4 \approx 64$MB이다. 머리 12개에 층 12개면 어텐션 행렬에만 약 9GB가 든다. 그래서 표준 트랜스포머가 $n \approx 512{\sim}2048$으로 제한된다.

---

**연습문제 4.**
미래 자리에 주목하지 못하게 가림막을 씌워 인과(자기회귀) 자기 어텐션을 구현하라.

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
