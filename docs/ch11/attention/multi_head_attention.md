# 멀티헤드 어텐션

멀티헤드 어텐션은 모델이 서로 다른 자리에서 **서로 다른 표현 부분공간**의 정보에 함께 주목하게 해 준다. 어텐션 함수를 하나만 쓰는 대신, 저마다 학습된 투영을 갖춘 어텐션 연산 $h$개를 나란히 계산한다.

$$\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\mathbf{W}_O$$

여기서 헤드 하나하나는 다음과 같다.

$$\text{head}_i = \text{Attention}(\mathbf{X}\mathbf{W}_Q^{(i)}, \mathbf{X}\mathbf{W}_K^{(i)}, \mathbf{X}\mathbf{W}_V^{(i)})$$

---

## 1. 동기: 여러 결합 가중치

단일 헤드는 쿼리마다 위치에 대한 가중치 분포 하나를 계산한다. 멀티헤드는 여러 투영에서 각각 분포를 계산하고 결과를 결합한다. 한 헤드도 여러 관계의 정보를 섞어 표현할 수 있으므로 단일 헤드가 문법·의미를 동시에 표현할 수 없다고 단정해서는 안 된다.

### 하나의 점수 함수

헤드가 하나인 어텐션에서 어텐션의 무늬는 다음이 정한다.

$$\mathbf{A} = \text{softmax}\left(\frac{\mathbf{X}\mathbf{W}_Q\mathbf{W}_K^T\mathbf{X}^T}{\sqrt{d_k}}\right)$$

행렬 $\mathbf{W}_Q\mathbf{W}_K^T \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$이 관련성을 계산하는 **쌍선형 형식 하나**를 정의한다.

### 무늬 하나로는 모자란 까닭

"The animal didn't cross the street because it was too tired."를 생각해 보자.

"it"을 풀려면 모델이 다음에 주목해야 한다.

1. **문법적 선행사**: "animal" (명사구의 구조)
2. **의미적 제약**: "tired" (생물성. 거리는 피곤해지지 않는다)
3. **지역 문맥**: "didn't cross" (설명되고 있는 행동)

어텐션 분포가 하나뿐이면 타협해야 한다.

- "animal"에 뾰족하게 몰리면 "tired"의 정보를 잃는다
- 관련된 토큰에 고루 퍼지면 저마다 받는 가중치가 묽어진다

### 멀티헤드라는 해법

헤드가 $h$개면 서로 **독립인** 어텐션 분포 $h$개를 계산한다.

$$\mathbf{A}^{(i)} = \text{softmax}\left(\frac{\mathbf{X}\mathbf{W}_Q^{(i)}(\mathbf{W}_K^{(i)})^T\mathbf{X}^T}{\sqrt{d_k}}\right)$$

헤드마다 다른 관련도와 값 표현을 학습할 수 있다. 다음은 가능한 역할 분담을 설명하는 예시이며 헤드 번호별로 이런 기능이 자동 배정되는 것은 아니다.

- $\text{head}_1$: 문법적 의존 (animal)
- $\text{head}_2$: 의미적 서술 (tired)
- $\text{head}_3$: 지역적인 동사구 (didn't cross)

---

## 2. 구조

### 차원

다음이 주어졌다고 하자.

- 모델의 차원: $d_{\text{model}}$
- 헤드의 수: $h$
- 헤드당 차원: $d_k = d_v = d_{\text{model}} / h$

헤드마다 차원이 더 낮은 부분공간에서 움직인다.

### 매개변수 행렬

헤드 $i \in \{1, \ldots, h\}$마다 다음과 같다.

$$\mathbf{W}_Q^{(i)} \in \mathbb{R}^{d_{\text{model}} \times d_k}$$

$$\mathbf{W}_K^{(i)} \in \mathbb{R}^{d_{\text{model}} \times d_k}$$

$$\mathbf{W}_V^{(i)} \in \mathbb{R}^{d_{\text{model}} \times d_v}$$

출력 투영은 다음과 같다.

$$\mathbf{W}_O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$$

### 전체 매개변수

$$\underbrace{h \cdot 3 \cdot d_{\text{model}} \cdot d_k}_{\text{QKV projections}} + \underbrace{d_{\text{model}}^2}_{\mathbf{W}_O} = 4 d_{\text{model}}^2$$

전체 차원을 d로 고정하고 각 헤드를 d/h로 나누는 일반 MHA에서는 편향을 제외한 매개변수가 $4d^2$다. 네 편향을 쓰면 $4d$를 더한다. 같은 투영·출력 차원의 단일 헤드와 같은 개수지만 실제 시간과 가중치 저장량까지 같다는 뜻은 아니다.

---

## 3. 출력 투영의 역할

### 왜 필요한가

출력 투영 $\mathbf{W}_O$은 **잔차 연결** 때문에 순방향 신경망에 흡수될 수 없다.

$$\mathbf{X}' = \text{LayerNorm}(\mathbf{X} + \text{Concat}(\ldots)\mathbf{W}_O)$$

어텐션의 출력은 순방향 신경망에 앞서 **$\mathbf{X}$에 더해진다**. $\mathbf{W}_O$이 없으면 다음과 같다.

$$\mathbf{X}' = \text{LayerNorm}(\mathbf{X} + \text{Concat}(\ldots))$$

순방향 신경망의 $\mathbf{W}_1$은 이미 잔차를 품은 $\mathbf{X}'$에 작용한다. 계산 그래프가 $\mathbf{W}_O$과 $\mathbf{W}_1$을 갈라놓는다.

### 헤드 사이의 소통

더 중요하게도 $\mathbf{W}_O$은 **헤드 사이의 섞임**을 가능하게 한다.

**$\mathbf{W}_O$ 이전**: 헤드들이 완전히 독립이며 벡터를 나란히 붙여 놓았을 뿐이다.

$$\text{Concat}(\text{head}_1, \ldots, \text{head}_h) = [\mathbf{z}^{(1)} | \mathbf{z}^{(2)} | \ldots | \mathbf{z}^{(h)}]$$

**$\mathbf{W}_O$ 이후**: 헤드 사이로 정보가 흐른다.

$$\text{Concat}(\ldots)\mathbf{W}_O = \sum_{j=1}^{h} \mathbf{z}^{(j)} \mathbf{W}_O^{(j)}$$

여기서 모델이 **서로 다른 어텐션 무늬를 어떻게 엮을지**를 배운다.

### 구체적인 예

다음과 같다고 하자.

- 1번 헤드가 문법 정보를 찾았다: "주어는 'cat'이다"
- 2번 헤드가 의미 정보를 찾았다: "서술어가 생물성을 함의한다"
- 3번 헤드가 위치 정보를 찾았다: "가까운 토큰이 움직임에 관한 것이다"

**$\mathbf{W}_O$ 이전** (이어 붙이기):

$$[\text{syntax} | \text{semantics} | \text{position}]$$

**$\mathbf{W}_O$ 이후** (학습된 섞음):

$$\text{output} = 0.5 \cdot \text{syntax} + 0.3 \cdot \text{semantics} + 0.2 \cdot \text{position}$$

---

## 4. 정보 이론의 관점

### 헤드 하나: 볼록 껍질 안의 한 점

헤드 하나의 어텐션 출력은 다음과 같다.

$$\mathbf{z}_i = \sum_{j=1}^{n} A_{ij} \mathbf{v}_j$$

드롭아웃이 없고 유효 위치가 있는 행에서는 가중치가 음이 아니며 합이 1이므로 출력은 값 벡터들의 볼록 껍질 안에 있다. 어텐션 드롭아웃 뒤에는 이 조건이 성립하지 않을 수 있다. 출력 투영·편향·잔차를 더한 최종 출력은 원래 값의 볼록 껍질로 제한되지 않는다.

### 여러 헤드: 더 풍부한 부분공간

헤드마다 (차원이 더 낮은) 제 볼록 껍질 안의 **서로 다른 점**에 닿는다.

$$\mathbf{z}_i = \left[\sum_j A_{ij}^{(1)} \mathbf{v}_j^{(1)}; \ldots; \sum_j A_{ij}^{(h)} \mathbf{v}_j^{(h)}\right]\mathbf{W}_O$$

헤드별로 서로 다른 값 투영과 가중치가 있으므로 여러 결합을 독립적으로 계산할 수 있다. 그러나 이어 붙이기만으로 전체 출력 공간을 자유롭게 채운다는 보장은 아니다. 출력 투영은 헤드 차원의 일반 선형 혼합이며 스칼라 계수의 볼록 평균으로 제한되지 않는다.

---

## 5. PyTorch 구현

### 멀티헤드 셀프 어텐션

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class MultiHeadSelfAttention(nn.Module):
    """
    다중 머리 자기 어텐션
    
    임베딩 차원을 여러 머리로 나누어, 모델이 서로 다른 표현 부분공간의
    정보에 주목할 수 있게 한다.
    """
    
    def __init__(
        self, 
        d_model: int, 
        n_heads: int, 
        dropout: float = 0.0,
        bias: bool = True
    ):
        super().__init__()
        
        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = self.d_k ** -0.5
        
        # Q, K, V의 선형 사영
        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)
        
        # 출력 사영 (머리 사이의 소통을 가능하게 한다)
        self.W_o = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        
        # 사영하고 모양 바꾸기: (배치, seq, d_model) -> (배치, heads, seq, d_k)
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 모든 머리에 대해 배율 조정 내적 어텐션을 나란히 계산
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 값에 어텐션 적용
        context = torch.matmul(attn_weights, V)
        
        # 모양 바꾸고 사영: (배치, heads, seq, d_k) -> (배치, seq, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(context)
        
        return output, attn_weights

def demonstrate_multi_head():
    """다중 머리 자기 어텐션 시연."""
    batch_size, seq_len, d_model, n_heads = 2, 10, 64, 8
    
    x = torch.randn(batch_size, seq_len, d_model)
    mha = MultiHeadSelfAttention(d_model, n_heads)
    
    output, weights = mha(x)
    
    print("Multi-Head Attention Demonstration")
    print("-" * 40)
    print(f"Input:    {x.shape}")
    print(f"Output:   {output.shape}")
    print(f"Weights:  {weights.shape}")
    print(f"Heads:    {n_heads}")
    print(f"Head dim: {d_model // n_heads}")

if __name__ == "__main__":
    demonstrate_multi_head()
```

**출력:**

```
Multi-Head Attention Demonstration
----------------------------------------
Input:    torch.Size([2, 10, 64])
Output:   torch.Size([2, 10, 64])
Weights:  torch.Size([2, 8, 10, 10])
Heads:    8
Head dim: 8
```

### 일반적인 멀티헤드 어텐션 (교차 어텐션용)

```python
class MultiHeadAttention(nn.Module):
    """
    일반적인 다중 머리 어텐션
    
    자기 어텐션(Q=K=V=X)이나 교차 어텐션(Q는 디코더, K와 V는 인코더)에 모두 쓸 수 있다.
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = self.d_k ** -0.5
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        Q: torch.Tensor, 
        K: torch.Tensor, 
        V: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        인수:
            Q: 질의의 출처 (배치, seq_q, d_model)
            K: 열쇠의 출처 (배치, seq_k, d_model)
            V: 값의 출처 (배치, seq_v, d_model), 대체로 seq_k == seq_v
            mask: 선택적인 어텐션 가림막
        """
        batch_size = Q.size(0)
        seq_q, seq_k = Q.size(1), K.size(1)
        
        # 사영하고 모양 바꾸기
        Q = self.W_q(Q).view(batch_size, seq_q, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, seq_k, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, seq_k, self.n_heads, self.d_k).transpose(1, 2)
        
        # 어텐션
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        
        # 모양 바꾸고 출력 사영
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_q, self.d_model)
        output = self.W_o(context)
        
        return output, attn_weights
```

---

## 6. 실험으로 본 헤드의 분업

학습된 트랜스포머를 분석한 연구는 기능의 분업을 확인해 준다.

| 헤드의 종류 | 무늬 | 예 |
|-----------|---------|---------|
| **위치 헤드** | 앞이나 뒤 토큰에 주목 | "The [cat]" → "The"에 주목 |
| **문법 헤드** | 주어와 동사의 의존을 붙잡음 | "cats [run]" → "cats"에 주목 |
| **상호 참조 헤드** | 대명사와 선행사의 관계를 좇음 | "[it] was tired" → "animal"에 주목 |
| **베끼기 헤드** | 드문 단어와 고유 명사에 집중 | 이름, 전문 용어 |
| **구분자 헤드** | 문장 부호와 문장 경계에 주목 | 마침표, [SEP] 토큰 |

특정 모델의 제거 실험은 일부 헤드의 역할과 중복성을 보여 줄 수 있다. 모든 헤드가 하나의 독립 기능에 대응하거나 하나를 없애면 그 기능만 사라지는 것은 아니다.

---

## 7. 단순한 병렬화가 아니다

흔한 오해가 있다. 멀티헤드 어텐션이 같은 것을 나란히 계산할 뿐이라는 생각이다.

**헤드 하나**는 자리마다 어텐션으로 가중한 결합을 **하나** 계산한다.

**여러 헤드**는 서로 **독립인 결합 $h$개**를 계산한 뒤 그것들을 **가장 좋게 섞는 법을 배운다**.

여러 가중치 분포는 단일 분포와 다른 표현 구조를 제공한다. 표현력의 엄밀한 비교에는 헤드·특징 차원, 층수와 출력 함수 등의 조건이 필요하며, 멀티헤드가 차원과 상관없이 모든 단일 헤드 구조보다 우세하다는 주장으로 일반화할 수 없다.

---

## 8. 계산 효율

멀티헤드 어텐션은 이론적인 복잡도가 헤드 하나일 때와 같다.

| 연산 | 헤드 하나 | 여러 헤드 |
|-----------|-------------|------------|
| 투영 | $3 \cdot n \cdot d^2$ | $3 \cdot n \cdot d^2$ |
| 어텐션 | $n^2 \cdot d$ | $h \cdot n^2 \cdot (d/h) = n^2 \cdot d$ |
| 출력 투영 | $n \cdot d^2$ | $n \cdot d^2$ |

전체 특징 차원을 고정하면 주된 행렬 곱의 점근적 계산량은 같은 차수다. 하지만 명시적으로 저장하는 가중치는 h×n×n이므로 헤드 수에 따라 늘어난다. 실제 실행 시간은 헤드 차원과 커널·장치에 따라 달라진다.

---

## 9. 하이퍼파라미터: 헤드의 수

흔한 설정은 다음과 같다.

| 모델 | $d_{\text{model}}$ | 헤드 수 | $d_k$ |
|-------|-------------------|-------|-------|
| BERT-base | 768 | 12 | 64 |
| BERT-large | 1024 | 16 | 64 |
| GPT-2 | 768 | 12 | 64 |
| GPT-3 (175B) | 12288 | 96 | 128 |
| LLaMA-7B | 4096 | 32 | 128 |

**어림 규칙**: $d_k = 64$이나 $d_k = 128$이 흔하다. $d_{\text{model}}$에 맞추어 헤드 수를 조정한다.

---

## 10. 변형

### 그룹 쿼리 어텐션 (GQA)

GQA는 여러 쿼리 헤드가 더 적은 키·값 헤드를 공유하는 구조다. 아래 클래스는 투영층 생성자만 있고 forward가 없어 실제 어텐션 계산이나 KV 반복을 구현하지 않는다. d_model의 헤드 수 나눗셈 조건도 추가로 확인해야 한다.

```python
class GroupedQueryAttention(nn.Module):
    """
    GQA: 질의 머리 n_heads개가 열쇠-값 머리 n_kv_heads개를 나누어 쓴다.
    추론할 때 KV 캐시의 크기를 줄인다.
    """
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        assert n_heads % n_kv_heads == 0
        
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_heads // n_kv_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)  # Q에는 온전한 머리
        self.W_k = nn.Linear(d_model, n_kv_heads * self.d_k)  # K에는 줄인 머리
        self.W_v = nn.Linear(d_model, n_kv_heads * self.d_k)  # V에는 줄인 머리
        self.W_o = nn.Linear(d_model, d_model)
```

### 다중 쿼리 어텐션 (MQA)

극단적인 경우로, 모든 쿼리 헤드가 키-값 헤드 하나를 함께 쓴다($n_{kv} = 1$).

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
전체 특징 차원이 같을 때 여러 헤드가 제공하는 차이와 그 한계를 설명하라.

</div>

??? success "연습문제 1 풀이"
    각 헤드가 다른 투영에서 별도의 위치 가중치 분포를 계산할 수 있다. 대신 헤드당 특징 차원은 작아지고 가중치 저장량과 실행 효율도 달라진다. 헤드가 특정 기능을 반드시 분담하거나 더 많은 헤드가 항상 더 좋은 성능을 내는 것은 아니다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff easy" title="쉬움"></span>
$d_{\text{model}} = 512$이고 헤드가 $h = 8$개인 멀티헤드 어텐션의 매개변수 수를 계산하라.

</div>

??? success "연습문제 2 풀이"
    투영 행렬 $W_Q, W_K, W_V$이 각각 $\mathbb{R}^{512 \times 512}$이고 출력 투영 $W_O \in \mathbb{R}^{512 \times 512}$이 있다. 모두 $4 \times 512 \times 512 = 1{,}048{,}576$개의 매개변수이며 512짜리 편향 4개(2048개)가 더해진다. 전부 약 105만 개이다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
(`nn.MultiheadAttention`을 쓰지 않고) 멀티헤드 어텐션을 PyTorch로 밑바닥부터 구현하라.

</div>

??? success "연습문제 3 풀이"
    ```python
    class MultiHeadAttn(nn.Module):
        def __init__(self, d_model, num_heads):
            super().__init__()
            self.h = num_heads
            self.d_k = d_model // num_heads
            self.WQ = nn.Linear(d_model, d_model)
            self.WK = nn.Linear(d_model, d_model)
            self.WV = nn.Linear(d_model, d_model)
            self.WO = nn.Linear(d_model, d_model)
        def forward(self, Q, K, V, mask=None):
            B, N, _ = Q.shape
            q = self.WQ(Q).view(B, N, self.h, self.d_k).transpose(1, 2)
            k = self.WK(K).view(B, -1, self.h, self.d_k).transpose(1, 2)
            v = self.WV(V).view(B, -1, self.h, self.d_k).transpose(1, 2)
            attn = torch.softmax(q @ k.transpose(-2,-1) / self.d_k**0.5, dim=-1)
            out = (attn @ v).transpose(1,2).reshape(B, N, -1)
            return self.WO(out)
    ```

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
학습한 모델의 헤드별 어텐션을 비교할 때 어떤 근거를 보고해야 하는가?

</div>

??? success "연습문제 4 풀이"
    입력 토큰·층·헤드 번호, 마스크와 평가 모드를 명시하고 같은 조건에서 비교한다. 지역성이나 특정 관계와의 상관을 관찰할 수 있지만 고정된 역할을 미리 가정하지 않는다. 실제 예측 기여를 확인하려면 입력 변경·제거 실험 등 추가 근거가 필요하다.

    연습문제 3의 MultiHeadAttn은 mask 인수를 받지만 실제 점수에 적용하지 않는다. 따라서 인과·패딩 제약이 필요한 경우 그대로 사용할 수 없다. 헤드 수는 양수이고 d_model을 나누어야 하며 K·V의 길이도 같아야 한다. 생성자와 reshape만으로 이 조건이 검증되는 것은 아니다.

## 정리하며

멀티헤드는 여러 투영의 결합 가중치를 계산하고 출력 투영으로 섞는다. 같은 모델 차원에서 매개변수 수가 같아도 헤드별 가중치 메모리와 실제 속도는 달라진다. 기능 분업은 관측할 대상이며 보장이 아니다. GQA 조각과 마스크를 무시하는 연습 코드의 범위도 확인한다.

**참고 문헌**

1. Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS*.

2. Voita, E., et al. (2019). Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned. *ACL*.

3. Clark, K., et al. (2019). What Does BERT Look At? An Analysis of BERT's Attention. *BlackboxNLP*.

4. Shazeer, N. (2019). Fast Transformer Decoding: One Write-Head is All You Need. *arXiv*.
