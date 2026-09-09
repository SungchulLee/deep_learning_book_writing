# 어텐션의 기초
어텐션 장치는 신경망이 모든 정보를 똑같이 다루는 대신 입력의 쓸모 있는 부분에 그때그때 집중하게 해 준다. 본디 순차열 대 순차열 모델의 한계를 풀려고 나왔지만, 이제는 트랜스포머를 비롯한 요즘 구조의 바탕이 되었다.

---

## 1. 핵심 직관: 부드러운 사전 찾기로 본 어텐션

어텐션을 이해하는 가장 밝은 길 가운데 하나는 **부드러운 사전 찾기**, 곧 열쇠-값 검색을 미분 가능하게 일반화한 것으로 보는 관점이다. 전통적인 사전은 정확히 맞추어 찾는다. 질의가 주어지면 딱 맞는 열쇠에 딸린 값을 돌려준다. 어텐션은 이 개념을 넓혀 유사도에 따라 모든 항목에 걸쳐 **부드러운 가중 검색**을 한다.

### 딱딱한 검색과 부드러운 검색

보통의 사전은 정확히 맞추어 열쇠를 값으로 보낸다.

```python
dictionary = {
    "cat": [0.2, 0.8, 0.1],   # 열쇠 -> 값
    "dog": [0.3, 0.7, 0.2],
    "mat": [0.1, 0.1, 0.9],
}

result = dictionary["cat"]  # [0.2, 0.8, 0.1]을 돌려준다
```

어텐션은 이를 연속이고 미분 가능한 검색으로 일반화한다.

```python
def soft_lookup(query, keys, values):
    similarities = [dot(query, k) for k in keys]
    weights = softmax(similarities)
    result = sum(w * v for w, v in zip(weights, values))
    return result
```

| 항목 | 딱딱한 사전 | 부드러운 어텐션 |
|--------|-----------------|----------------|
| 맞추기 | 열쇠가 정확히 일치 | 유사도로 가중 |
| 고르기 | 이분법 (0 아니면 1) | 연속 $[0, 1]$ |
| 출력 | 값 하나 | 가중 결합 |
| 미분 가능 | 아니다 | 그렇다 |
| 학습 가능 | 아니다 | 그렇다 (Q, K, V 사영) |

딱딱한 맞추기에서 부드러운 맞추기로 옮겨 가는 이 간단한 변화가 학습 가능하고 미분 가능한 기억 접근의 힘을 열어 준다. 질의가 주어지면 어텐션은 모든 열쇠와의 유사도 점수를 계산하고 소프트맥스로 정규화한 뒤 그에 딸린 값들의 가중합을 돌려준다.

### 사회 관계망 비유

| 사회 관계망 | 어텐션 | 구실 |
|--------------|-----------|---------|
| 검색어 | 질의 ($\mathbf{Q}$) | 필요한 정보를 나타냄 |
| 해시태그 | 열쇠 ($\mathbf{K}$) | 찾을 수 있게 함 |
| 게시물 내용 | 값 ($\mathbf{V}$) | 실제 정보를 줌 |

"노을 사진 요령"을 검색할 때 해시태그(#사진, #노을)가 관련 게시물로 데려다주지만, 정작 원한 것은 게시물의 내용이다. **열쇠는 찾히기 좋게 맞추어져 있고 값이 알맹이를 나른다.**

### 부드러운 사전 관점이 중요한 까닭

**역할의 분리.** 사전에서 찾을 때 쓰는 열쇠는 돌려받는 값과 다르다(`{"isbn-123": "위대한 개츠비"}`). 마찬가지로 열쇠는 **찾히기 좋게**, 값은 **담은 정보가 풍부하게** 맞추어진다. 그래서 모델이 "어떻게 찾히느냐"와 "무슨 정보를 주느냐"에 대해 서로 다른 표현을 배울 수 있다.

**내용으로 찾는 기억.** 전통적인 컴퓨터 메모리는 자리로 주소를 매긴다(`memory[address]`). 어텐션은 내용으로 주소를 매기게 하여(`memory[content_similar_to_query]`) 아무 뜻 없는 자리가 아니라 뜻에 따라 꺼내 온다.

**부드러운 성능 저하.** 딱딱한 찾기는 조금만 달라져도 완전히 실패한다. 열쇠가 정확하지 않으면 아무것도 얻지 못한다. 부드러운 찾기는 완만하게 나빠진다. 비슷한 질의는 비슷한 가중 결합을 꺼내 오고, 작은 흔들림은 출력의 작은 변화로 이어진다.

**조합적인 검색.** 두 열쇠 사이에 있는 질의는 두 값을 섞어 꺼내 오므로, 모델이 저장된 재료로 새 응답을 빚어낼 수 있다.

### 온도: 부드러움과 딱딱함을 다스리기

소프트맥스의 온도가 찾기가 얼마나 "딱딱해질지"를 다스린다.

$$\alpha_i = \frac{\exp(s_i / T)}{\sum_j \exp(s_j / T)}$$

| 온도 | 거동 | 비유 |
|-------------|----------|---------|
| $T \to 0$ | 딱딱한 어텐션 (argmax) | 정확한 사전 찾기 |
| $T = 1$ | 표준 소프트맥스 | 부드러운 섞음 |
| $T \to \infty$ | 고른 가중치 | 모든 값의 평균을 돌려줌 |

트랜스포머 어텐션의 배율 인수 $\sqrt{d_k}$은 차원에 맞추어지는 온도 노릇을 한다.

---

## 2. 역사적 배경: Seq2Seq의 병목

### 문제

어텐션이 나오기 전에 부호기-복호기 모델은 입력 순차열 전체를 길이가 고정된 문맥 벡터 하나로 눌러 담았다.

$$\mathbf{c} = f_{\text{encoder}}(x_1, x_2, \ldots, x_T)$$

그러면 **정보 병목**이 생긴다. $O(Td)$비트를 $O(d)$비트로 눌러 담으면 긴 순차열에서는 정보를 잃을 수밖에 없다. 순차열이 길어질수록 성능이 크게 나빠졌다.

### 어텐션이라는 해법

Bahdanau 등(2014)은 복호기가 걸음마다 서로 다른 부호기 자리에 "주목"하게 하자고 제안했다.

$$\mathbf{c}_t = \sum_{i=1}^{T} \alpha_{ti} \mathbf{h}_i$$

여기서 $\alpha_{ti}$은 시각 $t$의 출력을 만들 때 복호기가 부호기 상태 $\mathbf{h}_i$에 얼마나 집중해야 하는지를 나타낸다. 어텐션 가중치는 $\sum_i \alpha_{ti} = 1$과 $\alpha_{ti} \geq 0$을 만족한다.

이렇게 하면 복호하는 내내 $O(Td)$만큼의 정보에 닿을 수 있어 병목을 비켜 간다.

### 정렬로 보는 해석

번역에서 어텐션 가중치는 원본과 표적의 자리 사이의 **부드러운 정렬**을 나타낸다.

```
Source:  The  cat  sat  on  the  mat
Weights: 0.1  0.7  0.1  0.0  0.0  0.1  → "chat" (French for cat)
         0.0  0.1  0.0  0.0  0.1  0.8  → "tapis" (French for mat)
```

모델은 따로 가르쳐 주지 않아도 정렬의 짜임을 암묵적으로 배운다.

---

## 3. 질의-열쇠-값 형식

요즘 어텐션은 입력 $\mathbf{X} \in \mathbb{R}^{n \times d}$에서 얻은 세 가지 사영으로 움직인다.

$$\mathbf{Q} = \mathbf{X}\mathbf{W}_Q, \quad \mathbf{K} = \mathbf{X}\mathbf{W}_K, \quad \mathbf{V} = \mathbf{X}\mathbf{W}_V$$

이렇게 나누는 데는 저마다의 까닭이 있다.

- **질의**는 찾으려는 뜻, 곧 지금 자리에 무엇이 필요한지를 담는다
- **열쇠**는 맞추기에 맞추어져 있다. 찾히기 좋은 간결한 기술자이다
- **값**은 내용을 나른다. 맞는 것을 찾았을 때 건네줄 풍부한 정보이다

덕분에 모델이 찾기와 정보 전달에 서로 다른 표현을 배울 유연함을 얻는다. 질의어(SQL)가 무엇을 찾을지 정하고, 색인이 빠른 찾기를 돕고, 저장된 레코드가 실제 데이터를 담는 전통적인 데이터베이스의 짜임과 닮았다.

### 배율 조정 내적 어텐션

표준 형식은 다음과 같다(Vaswani 등, 2017).

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}$$

**한 걸음씩 보면 다음과 같다.**

1. **점수 계산**: $\mathbf{S} = \mathbf{Q}\mathbf{K}^\top$ — 성분 $S_{ij}$은 자리 $i$의 질의와 자리 $j$의 열쇠가 얼마나 맞는지를 잰다
2. **배율 조정**: $\mathbf{S} / \sqrt{d_k}$ — 차원이 커질 때 내적이 커지는 것을 막아 소프트맥스의 포화를 피한다
3. **정규화**: $\mathbf{A} = \text{softmax}(\mathbf{S})$ — $i$번째 행이 자리에 대한 분포가 된다
4. **모으기**: $\mathbf{Z} = \mathbf{A}\mathbf{V}$ — 출력마다 값 벡터의 가중합이다

---

## 4. 점수 함수

점수 함수마다 표현력과 효율을 저울질한다.

**내적** ($d_q = d_k$이어야 한다):

$$\text{score}(\mathbf{q}, \mathbf{k}) = \mathbf{q}^\top \mathbf{k}$$

**배율 조정 내적** (트랜스포머의 표준):

$$\text{score}(\mathbf{q}, \mathbf{k}) = \frac{\mathbf{q}^\top \mathbf{k}}{\sqrt{d_k}}$$

**일반형** (학습 가능하며 차원이 달라도 된다):

$$\text{score}(\mathbf{q}, \mathbf{k}) = \mathbf{q}^\top \mathbf{W} \mathbf{k}$$

**덧셈형 / 바다나우** (표현력이 크고 느리다):

$$\text{score}(\mathbf{q}, \mathbf{k}) = \mathbf{v}^\top \tanh(\mathbf{W}_q \mathbf{q} + \mathbf{W}_k \mathbf{k})$$

**코사인 유사도** (크기에 무관하다):

$$\text{score}(\mathbf{q}, \mathbf{k}) = \frac{\mathbf{q}^\top \mathbf{k}}{\|\mathbf{q}\| \|\mathbf{k}\|}$$

| 점수 함수 | 매개변수 | 복잡도 | 쓰임새 |
|----------------|------------|------------|----------|
| 내적 | 0 | $O(d)$ | Q와 K의 차원이 같을 때 |
| 배율 조정 내적 | 0 | $O(d)$ | 트랜스포머 (기울기가 안정적) |
| 일반형 | $d_q \times d_k$ | $O(d_q d_k)$ | 차원이 다를 때 |
| 덧셈형 | $d_a(d_q + d_k) + d_a$ | $O(d_a)$ | RNN seq2seq |
| 코사인 | 0 | $O(d)$ | 유사도가 묶여 있을 때 |

### 배율 조정이 중요한 까닭

$d$차원의 무작위 단위 벡터에서 내적의 분산은 약 $d$이다. 배율을 조정하지 않으면 $d$이 클 때 소프트맥스가 기울기가 사라지는 포화 영역으로 밀려난다. $\sqrt{d_k}$ 인수가 분산을 1로 정규화하여 기울기가 잘 흐르게 한다. 자세한 분석은 [배율 조정 내적](scaled_dot_product.md)을 보라.

---

## 5. 어텐션의 갈래

| 갈래 | Q의 출처 | K와 V의 출처 | 쓰임새 |
|---------|----------|-------------|----------|
| **자기 어텐션** | 같은 순차열 | 같은 순차열 | 내부 문맥 (BERT 부호기) |
| **교차 어텐션** | 복호기 | 부호기 | 부호기와 복호기 잇기 (번역) |
| **인과 어텐션** | 같은 순차열 | 과거 자리만 | 자기회귀 생성 (GPT) |

### 자기 어텐션

모든 사영이 같은 입력에서 나온다.

$$\mathbf{Q} = \mathbf{X}\mathbf{W}^Q, \quad \mathbf{K} = \mathbf{X}\mathbf{W}^K, \quad \mathbf{V} = \mathbf{X}\mathbf{W}^V$$

자리마다 모든 자리에 주목할 수 있어 전역적인 의존을 붙잡는다. 자세한 내용은 [자기 어텐션](self_attention.md)을 보라.

### 교차 어텐션

질의는 한 순차열(복호기)에서, 열쇠와 값은 다른 순차열(부호기)에서 온다.

$$\mathbf{Q} = \mathbf{X}_{\text{dec}}\mathbf{W}^Q, \quad \mathbf{K} = \mathbf{X}_{\text{enc}}\mathbf{W}^K, \quad \mathbf{V} = \mathbf{X}_{\text{enc}}\mathbf{W}^V$$

이것이 seq2seq 모델에서 부호기와 복호기를 잇는다. 교차 어텐션의 무늬는 어텐션의 무늬 절에서 다룬다.

### 인과 (가림막) 어텐션

자리 $i$은 자리 $j \leq i$에만 주목할 수 있다.

$$\alpha_{ij} = 0 \quad \text{for } j > i$$

미래 토큰을 쓸 수 없는 자기회귀 생성에 꼭 필요하다.

---

## 6. 가림막 씌우기

### 덧댐 가림막

길이가 다른 순차열을 묶을 때 덧댐 토큰에 주목하지 않게 막는다.

```python
def padding_mask(seq, pad_idx=0):
    """(배치, seq_len) -> (배치, 1, 1, seq_len)"""
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)
```

### 인과 가림막

자기회귀 모델을 위한 아래 삼각 가림막이다.

```python
def causal_mask(size):
    """(1, 1, size, size)"""
    return torch.tril(torch.ones(size, size)).unsqueeze(0).unsqueeze(0)
```

가림막은 소프트맥스에 앞서 점수를 $-\infty$으로 두어 씌운다.

---

## 7. PyTorch 구현

### 배율 조정 내적 어텐션

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    인수:
        Q: (배치, ..., seq_q, d_k)
        K: (배치, ..., seq_k, d_k)
        V: (배치, ..., seq_k, d_v)
        mask: (배치, ..., seq_q, seq_k)로 방송 가능

    반환값:
        output: (배치, ..., seq_q, d_v)
        weights: (배치, ..., seq_q, seq_k)
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    weights = F.softmax(scores, dim=-1)
    output = torch.matmul(weights, V)
    return output, weights
```

### 사영을 갖춘 어텐션 층

```python
class Attention(nn.Module):
    def __init__(self, d_model, d_k=None, d_v=None):
        super().__init__()
        d_k = d_k or d_model
        d_v = d_v or d_model

        self.W_q = nn.Linear(d_model, d_k)
        self.W_k = nn.Linear(d_model, d_k)
        self.W_v = nn.Linear(d_model, d_v)
        self.d_k = d_k

    def forward(self, query, key, value, mask=None):
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        return scaled_dot_product_attention(Q, K, V, mask)
```

### 부드러운 사전 모듈

```python
class SoftDictionary(nn.Module):
    """
    부드러운 사전 찾기로 본 어텐션.
    
    사전으로 보는 해석을 강조한다.
    - 열쇠가 사전의 색인이다
    - 값이 담긴 내용이다
    - 질의가 부드러운 맞추기로 쓸모 있는 내용을 꺼내 온다
    """
    
    def __init__(self, key_dim: int, value_dim: int, num_entries: int = None):
        super().__init__()
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.scale = key_dim ** -0.5
        
        if num_entries is not None:
            self.keys = nn.Parameter(torch.randn(num_entries, key_dim))
            self.values = nn.Parameter(torch.randn(num_entries, value_dim))
        else:
            self.keys = None
            self.values = None
            
    def forward(self, query, keys=None, values=None, temperature=1.0):
        """
        인수:
            query: (배치, query_dim) 또는 (배치, num_queries, query_dim)
            keys: (배치, num_entries, key_dim), None이면 학습된 열쇠를 쓴다
            values: (배치, num_entries, value_dim), None이면 학습된 값을 쓴다
            temperature: 날카로움을 다스린다 (낮을수록 딱딱한 선택)
        """
        if keys is None:
            keys = self.keys.unsqueeze(0).expand(query.size(0), -1, -1)
        if values is None:
            values = self.values.unsqueeze(0).expand(query.size(0), -1, -1)
            
        if query.dim() == 2:
            query = query.unsqueeze(1)
            
        scores = torch.matmul(query, keys.transpose(-2, -1)) * self.scale
        scores = scores / temperature
        weights = F.softmax(scores, dim=-1)
        retrieved = torch.matmul(weights, values)
        
        return retrieved.squeeze(1), weights.squeeze(1)
```

---

## 8. 어텐션의 성질

### 순열 동변성

입력의 순서를 바꾸면 출력도 똑같이 바뀐다.

$$\text{Attention}(\mathbf{P}\mathbf{X}) = \mathbf{P} \cdot \text{Attention}(\mathbf{X})$$

어텐션은 자리를 대칭적으로 다루므로 위치 정보를 따로 넣어 주어야 한다.

### 계산 복잡도

순차열의 길이가 $n$이고 차원이 $d$일 때 다음과 같다.

| 연산 | 시간 | 공간 |
|-----------|------|-------|
| 점수 계산 | $O(n^2 d)$ | $O(n^2)$ |
| 소프트맥스와 모으기 | $O(n^2)$ | $O(n^2)$ |
| **합계** | $O(n^2 d)$ | $O(n^2)$ |

복잡도가 이차라 아주 긴 순차열에는 쓰기 어려운데, 그래서 효율적인 판본(성긴 어텐션, 선형 어텐션, FlashAttention)이 나왔다.

---

## 9. 다른 개념과의 이음새

### 기억 신경망

기억 신경망(Weston 등, 2015)은 기억으로 보는 어텐션의 관점을 명시적으로 정식화한다.

$$\mathbf{o} = \sum_i p_i \mathbf{c}_i$$

여기서 $p_i = \text{softmax}(\mathbf{q}^T \mathbf{m}_i)$은 주소 가중치, $\mathbf{m}_i$은 기억의 열쇠, $\mathbf{c}_i$은 기억의 값이다. 트랜스포머는 기억을 순차열 자체로 삼아 기억 신경망을 일반화하는데, 그것이 바로 자기 어텐션이다.

### 홉필드 신경망과 연상 기억

요즘의 홉필드 신경망(Ramsauer 등, 2020)은 어텐션이 연상 기억을 연속으로 느슨하게 푼 것임을 드러낸다.

$$\text{new state} = \text{softmax}(\beta \cdot \text{state} \cdot \text{patterns}^T) \cdot \text{patterns}$$

이는 정확히 어텐션의 식이다! 이 이음새는 어텐션이 **무늬 완성**을 구현함을 보여 준다. 무늬의 일부(질의)가 주어지면 저장된 무늬(값)를 꺼내 오는 것이다. 요즘 홉필드 신경망의 지수적인 저장 용량이 트랜스포머가 긴 문맥을 다룰 수 있는 까닭을 설명해 준다.

### 검색 증강 생성 (RAG)

RAG 시스템은 어텐션과 비슷한 검색을 큰 규모로 쓴다.

| RAG의 부품 | 어텐션에서의 대응 |
|---------------|-------------------|
| 질의 부호기 | 질의 사영 $\mathbf{W}_Q$ |
| 문서 색인 | 열쇠 |
| 문서 내용 | 값 |
| 검색 | 어텐션 계산 |
| 읽개 | 뒤따르는 처리 |

RAG는 바깥 지식 기반에 대한 어텐션으로 볼 수 있으며, 모델의 "기억"을 매개변수 너머로 넓힌다.

---

## 10. Seq2Seq 어텐션에서 트랜스포머로

RNN 어텐션에서 트랜스포머로 나아가며 핵심적인 통찰이 드러났다.

1. **어텐션이 궂은일을 다 한다**: 어텐션을 쓰는 seq2seq에서 정보는 대부분 순환이 아니라 어텐션을 타고 흐른다
2. **자기 어텐션**: 어텐션이 부호기와 복호기를 잇는다면 같은 순차열 안의 자리끼리는 왜 안 되겠는가?
3. **병렬화**: 순환을 없애면 대규모 병렬 처리가 가능해진다
4. **위치 부호화**: 순환이 없으면 위치를 따로 담아 주어야 한다

| 연도 | 발전 | 영향 |
|------|-------------|--------|
| 2014 | LSTM을 쓰는 seq2seq | 부호기-복호기 틀 |
| 2015 | 바다나우 어텐션 | 병목 문제를 풀었다 |
| 2015 | 루옹 어텐션 | 효율적인 대안 |
| 2017 | 트랜스포머 | 자기 어텐션이 순환을 대신했다 |

---

## 연습문제

**연습문제 1.**
어텐션의 질의-열쇠-값 틀과 정보 검색과의 비유를 설명하라.

??? success "연습문제 1 풀이"
    질의는 찾으려는 것이다. 열쇠는 있는 정보의 색인이다. 값은 정보 그 자체이다. 어텐션은 질의와 열쇠마다의 맞음 정도를 계산한 뒤 값들의 가중합을 돌려준다. 비유하면 데이터베이스를 (질의로) 검색하여 레코드의 머리말(열쇠)과 맞추어 보고 관련 레코드(값)를 돌려받는 것이다.

---

**연습문제 2.**
$e_{ij} = \text{score}(q_i, k_j)$일 때 어텐션 가중치 $\alpha_{ij} = \text{softmax}(e_{ij})$을 유도하라.

??? success "연습문제 2 풀이"
    점수 함수는 질의와 열쇠가 맞는 정도를 잰다. 흔히 쓰는 것은 내적($q^\top k$), 배율 조정 내적($q^\top k / \sqrt{d_k}$), 덧셈형($v^\top \tanh(W_1 q + W_2 k)$)이다. 소프트맥스가 점수를 열쇠에 대한 확률 분포로 정규화한다: $\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_l \exp(e_{il})}$.

---

**연습문제 3.**
어텐션을 왜 '부드러운' 정렬 장치라 부르는가?

??? success "연습문제 3 풀이"
    원본의 자리 하나를 고르는 딱딱한 정렬과 달리 부드러운 어텐션은 모든 자리에 걸친 가중 평균을 계산한다. 그러면 기울기가 매끄럽고 연속으로 흘러 역전파로 처음부터 끝까지 학습할 수 있다. 딱딱한 어텐션은 미분할 수 없어 강화 학습 기법이 필요하다.

---

**연습문제 4.**
기본적인 어텐션 장치(바다나우 방식 덧셈 어텐션)를 PyTorch로 구현하라.

??? success "연습문제 4 풀이"
    ```python
    class AdditiveAttention(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.W1 = nn.Linear(d, d)
            self.W2 = nn.Linear(d, d)
            self.v = nn.Linear(d, 1)
        def forward(self, query, keys, values):
            scores = self.v(torch.tanh(self.W1(query).unsqueeze(1) + self.W2(keys)))
            weights = torch.softmax(scores, dim=1)
            return (weights * values).sum(dim=1)
    ```

## 정리하며

| 개념 | 설명 |
|---------|-------------|
| **핵심 착상** | 미분 가능한 부드러운 사전. 질의와 열쇠가 맞는 정도에 따른 값의 가중 결합 |
| **Q, K, V의 구실** | Q는 찾으려는 뜻, K는 찾히기, V는 내용 |
| **점수 함수** | 내적(빠름), 덧셈형(표현력), 배율 조정(안정적) |
| **갈래** | 자기(순차열 안), 교차(순차열 사이), 인과(자기회귀) |
| **복잡도** | 시간 $O(n^2 d)$, 공간 $O(n^2)$ |
| **온도** | 부드러움과 딱딱함의 정도를 다스린다. $\sqrt{d_k}$ 배율이 차원에 맞추어진다 |

**참고 문헌**

1. Bahdanau, D., Cho, K., & Bengio, Y. (2014). "Neural Machine Translation by Jointly Learning to Align and Translate." *arXiv:1409.0473*.
2. Luong, M.-T., Pham, H., & Manning, C. D. (2015). "Effective Approaches to Attention-based Neural Machine Translation." *EMNLP*.
3. Vaswani, A., et al. (2017). "Attention Is All You Need." *NeurIPS*.
4. Weston, J., Chopra, S., & Bordes, A. (2015). "Memory Networks." *ICLR*.
5. Sukhbaatar, S., et al. (2015). "End-To-End Memory Networks." *NeurIPS*.
6. Ramsauer, H., et al. (2020). "Hopfield Networks is All You Need." *ICLR*.
7. Graves, A., Wayne, G., & Danihelka, I. (2014). "Neural Turing Machines." *arXiv:1410.5401*.
