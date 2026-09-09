# 다중 머리 어텐션

다중 머리 어텐션은 모델이 서로 다른 자리에서 **서로 다른 표현 부분공간**의 정보에 함께 주목하게 해 준다. 어텐션 함수를 하나만 쓰는 대신, 저마다 학습된 사영을 갖춘 어텐션 연산 $h$개를 나란히 계산한다.

$$\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\mathbf{W}_O$$

여기서 머리 하나하나는 다음과 같다.

$$\text{head}_i = \text{Attention}(\mathbf{X}\mathbf{W}_Q^{(i)}, \mathbf{X}\mathbf{W}_K^{(i)}, \mathbf{X}\mathbf{W}_V^{(i)})$$

---

## 1. 동기: 머리 하나의 병목

머리가 하나인 어텐션에는 근본적인 한계가 있다. $d_{\text{model}}$이 아무리 커도 질의 자리마다 어텐션 분포가 **하나**뿐이다.

### 계수의 제약

머리가 하나인 어텐션에서 어텐션의 무늬는 다음이 정한다.

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

### 다중 머리라는 해법

머리가 $h$개면 서로 **독립인** 어텐션 분포 $h$개를 계산한다.

$$\mathbf{A}^{(i)} = \text{softmax}\left(\frac{\mathbf{X}\mathbf{W}_Q^{(i)}(\mathbf{W}_K^{(i)})^T\mathbf{X}^T}{\sqrt{d_k}}\right)$$

머리마다 **서로 다른 관련성의 개념**을 배운다.

- $\text{head}_1$: 문법적 의존 (animal)
- $\text{head}_2$: 의미적 서술 (tired)
- $\text{head}_3$: 지역적인 동사구 (didn't cross)

---

## 2. 구조

### 차원

다음이 주어졌다고 하자.

- 모델의 차원: $d_{\text{model}}$
- 머리의 수: $h$
- 머리당 차원: $d_k = d_v = d_{\text{model}} / h$

머리마다 차원이 더 낮은 부분공간에서 움직인다.

### 매개변수 행렬

머리 $i \in \{1, \ldots, h\}$마다 다음과 같다.

$$\mathbf{W}_Q^{(i)} \in \mathbb{R}^{d_{\text{model}} \times d_k}$$

$$\mathbf{W}_K^{(i)} \in \mathbb{R}^{d_{\text{model}} \times d_k}$$

$$\mathbf{W}_V^{(i)} \in \mathbb{R}^{d_{\text{model}} \times d_v}$$

출력 사영은 다음과 같다.

$$\mathbf{W}_O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$$

### 전체 매개변수

$$\underbrace{h \cdot 3 \cdot d_{\text{model}} \cdot d_k}_{\text{QKV projections}} + \underbrace{d_{\text{model}}^2}_{\mathbf{W}_O} = 4 d_{\text{model}}^2$$

차원이 $d_{\text{model}}$인 큰 어텐션 머리 하나와 매개변수 수가 같다.

---

## 3. $\mathbf{W}_O$의 구실

### 왜 필요한가

출력 사영 $\mathbf{W}_O$은 **잔차 연결** 때문에 순방향 신경망에 흡수될 수 없다.

$$\mathbf{X}' = \text{LayerNorm}(\mathbf{X} + \text{Concat}(\ldots)\mathbf{W}_O)$$

어텐션의 출력은 순방향 신경망에 앞서 **$\mathbf{X}$에 더해진다**. $\mathbf{W}_O$이 없으면 다음과 같다.

$$\mathbf{X}' = \text{LayerNorm}(\mathbf{X} + \text{Concat}(\ldots))$$

순방향 신경망의 $\mathbf{W}_1$은 이미 잔차를 품은 $\mathbf{X}'$에 작용한다. 계산 그래프가 $\mathbf{W}_O$과 $\mathbf{W}_1$을 갈라놓는다.

### 머리 사이의 소통

더 중요하게도 $\mathbf{W}_O$은 **머리 사이의 섞임**을 가능하게 한다.

**$\mathbf{W}_O$ 이전**: 머리들이 완전히 독립이며 벡터를 나란히 붙여 놓았을 뿐이다.

$$\text{Concat}(\text{head}_1, \ldots, \text{head}_h) = [\mathbf{z}^{(1)} | \mathbf{z}^{(2)} | \ldots | \mathbf{z}^{(h)}]$$

**$\mathbf{W}_O$ 이후**: 머리 사이로 정보가 흐른다.

$$(\text{Concat}(\ldots)\mathbf{W}_O)_i = \sum_{j=1}^{h} \mathbf{z}^{(j)} \mathbf{W}_O^{(j \to \text{out})}$$

여기서 모델이 **서로 다른 어텐션 무늬를 어떻게 엮을지**를 배운다.

### 구체적인 예

다음과 같다고 하자.

- 1번 머리가 문법 정보를 찾았다: "주어는 'cat'이다"
- 2번 머리가 의미 정보를 찾았다: "서술어가 생물성을 함의한다"
- 3번 머리가 위치 정보를 찾았다: "가까운 토큰이 움직임에 관한 것이다"

**$\mathbf{W}_O$ 이전** (이어 붙이기):

$$[\text{syntax} | \text{semantics} | \text{position}]$$

**$\mathbf{W}_O$ 이후** (학습된 섞음):

$$\text{output} = 0.5 \cdot \text{syntax} + 0.3 \cdot \text{semantics} + 0.2 \cdot \text{position}$$

---

## 4. 정보 이론의 관점

### 머리 하나: 볼록 껍질 안의 한 점

머리 하나의 어텐션 출력은 다음과 같다.

$$\mathbf{z}_i = \sum_{j=1}^{n} A_{ij} \mathbf{v}_j$$

$\sum_j A_{ij} = 1$이므로 이는 $\{\mathbf{v}_1, \ldots, \mathbf{v}_n\}$의 **볼록 껍질** 안의 한 점이다.

### 여러 머리: 더 풍부한 부분공간

머리마다 (차원이 더 낮은) 제 볼록 껍질 안의 **서로 다른 점**에 닿는다.

$$\mathbf{z}_i = \left[\sum_j A_{ij}^{(1)} \mathbf{v}_j^{(1)}; \ldots; \sum_j A_{ij}^{(h)} \mathbf{v}_j^{(h)}\right]\mathbf{W}_O$$

이어 붙인 것은 **훨씬 풍부한 부분공간**을 펼치며, $\mathbf{W}_O$은 이 $h$개의 서로 다른 문맥 요약의 **어떤 선형 결합**이든 만들어 낼 수 있다.

---

## 5. PyTorch 구현

### 다중 머리 자기 어텐션

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

### 일반적인 다중 머리 어텐션 (교차 어텐션용)

```python
class MultiHeadAttention(nn.Module):
    """
    일반적인 다중 머리 어텐션
    
    자기 어텐션(Q=K=V=X)이나 교차 어텐션(Q는 복호기, K와 V는 부호기)에 모두 쓸 수 있다.
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

## 6. 실험으로 본 머리의 분업

학습된 트랜스포머를 분석한 연구는 기능의 분업을 확인해 준다.

| 머리의 종류 | 무늬 | 예 |
|-----------|---------|---------|
| **위치 머리** | 앞이나 뒤 토큰에 주목 | "The [cat]" → "The"에 주목 |
| **문법 머리** | 주어와 동사의 의존을 붙잡음 | "cats [run]" → "cats"에 주목 |
| **상호 참조 머리** | 대명사와 선행사의 관계를 좇음 | "[it] was tired" → "animal"에 주목 |
| **베끼기 머리** | 드문 낱말과 고유 명사에 집중 | 이름, 전문 용어 |
| **구분자 머리** | 문장 부호와 문장 경계에 주목 | 마침표, [SEP] 토큰 |

제거 실험을 해 보면 머리를 하나씩 없앨 때 특정 능력만 나빠지고 나머지는 멀쩡하다.

---

## 7. 단순한 병렬화가 아니다

흔한 오해가 있다. 다중 머리 어텐션이 같은 것을 나란히 계산할 뿐이라는 생각이다.

**머리 하나**는 자리마다 어텐션으로 가중한 결합을 **하나** 계산한다.

**여러 머리**는 서로 **독립인 결합 $h$개**를 계산한 뒤 그것들을 **가장 좋게 섞는 법을 배운다**.

표현력의 차이는 상당하다. 차원과 상관없이 다중 머리 어텐션은 머리 하나로는 나타낼 수 없는 함수를 나타낼 수 있다.

---

## 8. 계산 효율

다중 머리 어텐션은 이론적인 복잡도가 머리 하나일 때와 같다.

| 연산 | 머리 하나 | 여러 머리 |
|-----------|-------------|------------|
| 사영 | $3 \cdot n \cdot d^2$ | $3 \cdot n \cdot d^2$ |
| 어텐션 | $n^2 \cdot d$ | $h \cdot n^2 \cdot (d/h) = n^2 \cdot d$ |
| 출력 사영 | $n \cdot d^2$ | $n \cdot d^2$ |

나란한 짜임이라 GPU에 매우 잘 맞는다. 배치 행렬 곱 한 번으로 모든 머리가 한꺼번에 계산된다.

---

## 9. 초매개변수: 머리의 수

흔한 설정은 다음과 같다.

| 모델 | $d_{\text{model}}$ | 머리 수 | $d_k$ |
|-------|-------------------|-------|-------|
| BERT-base | 768 | 12 | 64 |
| BERT-large | 1024 | 16 | 64 |
| GPT-2 | 768 | 12 | 64 |
| GPT-3 (175B) | 12288 | 96 | 128 |
| LLaMA-7B | 4096 | 32 | 128 |

**어림 규칙**: $d_k = 64$이나 $d_k = 128$이 흔하다. $d_{\text{model}}$에 맞추어 머리 수를 조정한다.

---

## 10. 변형

### 묶음 질의 어텐션 (GQA)

LLaMA-2를 비롯한 효율적인 모델이 쓴다. 질의 머리 여러 개가 열쇠-값 머리를 나누어 쓴다.

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

### 다중 질의 어텐션 (MQA)

극단적인 경우로, 모든 질의 머리가 열쇠-값 머리 하나를 함께 쓴다($n_{kv} = 1$).

---

## 연습문제

**연습문제 1.**
전체 차원이 같을 때 어텐션 머리를 여러 개 두는 편이 하나만 두는 것보다 나은 까닭을 설명하라.

??? success "연습문제 1 풀이"
    머리가 여럿이면 모델이 서로 다른 자리에서 서로 다른 표현 부분공간의 정보에 한꺼번에 주목할 수 있다. 머리가 하나면 가중 평균을 하나만 계산하므로 서로 다른 종류의 관계가 뭉개질 수 있다. 차원이 $d_k = d/h$인 머리 $h$개가 있으면 문법적·의미적·위치적 무늬처럼 서로 다른 어텐션 무늬 $h$개를 나란히 붙잡을 수 있다.

---

**연습문제 2.**
$d_{\text{model}} = 512$이고 머리가 $h = 8$개인 다중 머리 어텐션의 매개변수 수를 계산하라.

??? success "연습문제 2 풀이"
    사영 행렬 $W_Q, W_K, W_V$이 각각 $\mathbb{R}^{512 \times 512}$이고 출력 사영 $W_O \in \mathbb{R}^{512 \times 512}$이 있다. 모두 $4 \times 512 \times 512 = 1{,}048{,}576$개의 매개변수이며 512짜리 편향 4개(2048개)가 더해진다. 전부 약 105만 개이다.

---

**연습문제 3.**
(`nn.MultiheadAttention`을 쓰지 않고) 다중 머리 어텐션을 PyTorch로 밑바닥부터 구현하라.

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

**연습문제 4.**
번역으로 학습한 트랜스포머의 머리마다 어텐션 무늬를 그려 보라. 어떤 무늬가 나타나는가?

??? success "연습문제 4 풀이"
    흔한 무늬는 이렇다. 1번 머리는 앞 낱말에 주목하고(지역적), 2번 머리는 문법적 의존(동사와 주어)에, 3번 머리는 위치의 무늬(같은 상대 자리에 주목)에, 4번 머리는 의미적 유사성에 주목한다. 어떤 머리는 넓고 고른 어텐션을 보이며 전역 문맥을 담는다. 이런 다양함 덕분에 여러 머리가 하나보다 낫다.

## 정리하며

다중 머리 어텐션은 다음을 준다.

1. **여러 어텐션 무늬**: 문법적·의미적·위치적으로 근본이 다른 관계를 붙잡는다
2. **부분공간 분해**: 머리마다 특화된 특징을 뽑는다
3. **학습된 결합**: $\mathbf{W}_O$이 머리 사이의 섞음으로 여러 관점을 아우른다
4. **기능의 분업**: 학습하면서 저절로 나타난다
5. **같은 계산 비용**: 매개변수 수가 큰 머리 하나와 같다

**핵심 착상**: 어텐션 분포 하나로는 문법적·의미적·위치적 관계를 한꺼번에 나타낼 수 없다. 다중 머리 어텐션은 계산 효율을 지키면서 이 근본적인 한계를 푼다.

출력 사영 $\mathbf{W}_O$은 형식뿐인 것이 아니다. 머리마다 주는 서로 다른 "시선"을 하나의 표현으로 엮는 법을 모델이 배우는 곳이 바로 여기이다.

**참고 문헌**

1. Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS*.

2. Voita, E., et al. (2019). Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned. *ACL*.

3. Clark, K., et al. (2019). What Does BERT Look At? An Analysis of BERT's Attention. *BlackboxNLP*.

4. Shazeer, N. (2019). Fast Transformer Decoding: One Write-Head is All You Need. *arXiv*.
