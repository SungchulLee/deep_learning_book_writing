# 스케일 내적 어텐션

스케일 내적 어텐션은 트랜스포머 구조의 근본 구성 블록이다. Vaswani 등(2017)이 "Attention Is All You Need"에서 내놓았으며, 요즘 하드웨어에서 크게 병렬화할 수 있는 행렬 연산으로 어텐션 웨이트를 효율적이고 효과적으로 계산한다.

핵심 혁신은 내적 어텐션의 계산 효율에 학습 중 그래디언트를 안정시키는 배율 인수를 더한 데 있다. 사소해 보이는 이 손질이 어텐션 기반의 깊은 모델을 학습시키는 데 결정적이었다.

---

## 1. 수학적 정식화

### 어텐션 함수

쿼리 $\mathbf{Q} \in \mathbb{R}^{n \times d_k}$, 키 $\mathbf{K} \in \mathbb{R}^{m \times d_k}$, 값 $\mathbf{V} \in \mathbb{R}^{m \times d_v}$이 주어지면 스케일 내적 어텐션은 다음을 계산한다.

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

### 한 걸음씩 계산하기

**1단계: 날 어텐션 점수 계산**

$$\mathbf{S} = \mathbf{Q}\mathbf{K}^T \in \mathbb{R}^{n \times m}$$

성분 $s_{ij}$은 쿼리 $i$과 키 $j$의 유사도를 잰다.

$$s_{ij} = \mathbf{q}_i^T \mathbf{k}_j = \sum_{l=1}^{d_k} q_{il} k_{jl}$$

내적이 클수록 쿼리와 키가 임베딩 공간에서 비슷한 방향을 가리킨다는 뜻이다.

**2단계: 배율 조정 적용**

$$\mathbf{S}_{\text{scaled}} = \frac{\mathbf{S}}{\sqrt{d_k}}$$

배율 인수 $\frac{1}{\sqrt{d_k}}$은 학습의 안정에 매우 중요하다(아래의 자세한 분석을 보라).

**3단계: 행마다 소프트맥스 적용**

$$\alpha_{ij} = \frac{\exp(s_{ij}^{\text{scaled}})}{\sum_{l=1}^{m} \exp(s_{il}^{\text{scaled}})}$$

$\mathbf{A}$의 행마다 자리에 대한 확률 분포가 된다.

- 모든 $i$에 대해 $\sum_j A_{ij} = 1$
- 모든 $i, j$에 대해 $A_{ij} \geq 0$

**4단계: 값의 가중합**

$$\mathbf{O} = \mathbf{A}\mathbf{V} \in \mathbb{R}^{n \times d_v}$$

출력의 행마다 어텐션으로 가중한 값 벡터의 볼록 결합이다.

$$\mathbf{o}_i = \sum_{j=1}^{m} \alpha_{ij} \mathbf{v}_j$$

### 차원 분석

| 텐서 | 모양 | 설명 |
|--------|-------|-------------|
| $\mathbf{Q}$ | $(n, d_k)$ | 차원이 $d_k$인 쿼리 벡터 $n$개 |
| $\mathbf{K}$ | $(m, d_k)$ | 차원이 $d_k$인 키 벡터 $m$개 |
| $\mathbf{V}$ | $(m, d_v)$ | 차원이 $d_v$인 값 벡터 $m$개 |
| $\mathbf{S}$ | $(n, m)$ | 어텐션 점수 |
| $\mathbf{A}$ | $(n, m)$ | 어텐션 웨이트 (소프트맥스 뒤) |
| $\mathbf{O}$ | $(n, d_v)$ | 출력 벡터 |

**핵심 제약:**

- 쿼리와 키의 차원이 같아야 한다($d_k$)
- 키와 값의 시퀀스 길이가 같아야 한다($m$)
- 값의 차원($d_v$)은 아무것이나 되며 그것이 출력 차원이 된다
- 셀프 어텐션에서는 $n = m$이다(쿼리와 키·값이 같은 시퀀스다)

### 배치와 헤드가 있을 때

실제로는 배치와 여러 헤드를 가진 텐서를 다룬다.

| 텐서 | 모양 | 설명 |
|--------|-------|-------------|
| $\mathbf{Q}$ | $(B, H, n, d_k)$ | 배치 멀티헤드 쿼리 |
| $\mathbf{K}$ | $(B, H, m, d_k)$ | 배치 멀티헤드 키 |
| $\mathbf{V}$ | $(B, H, m, d_v)$ | 배치 멀티헤드 값 |
| $\mathbf{O}$ | $(B, H, n, d_v)$ | 배치 멀티헤드 출력 |

여기서 $B$은 배치 크기이고 $H$은 어텐션 헤드의 수이다.

---

## 2. 왜 내적인가

내적 $\mathbf{q}^T \mathbf{k}$은 관련성을 재기에 알맞은 좋은 성질을 지닌다.

### 기하학적 해석

$$\mathbf{q}^T \mathbf{k} = \|\mathbf{q}\| \|\mathbf{k}\| \cos\theta$$

여기서 $\theta$은 두 벡터 사이의 각이다. 벡터가 비슷한 방향을 가리키거나($\theta$이 작거나) 크기가 크면 내적이 크다. 이는 직관적인 유사도를 준다. 임베딩 공간에서 "정렬된" 쿼리와 키가 높은 어텐션 점수를 낸다.

### 계산 효율

내적은 행렬 곱으로 계산할 수 있고, 이는 GPU에서 매우 잘 최적화되어 있다.

```python
scores = torch.matmul(Q, K.transpose(-2, -1))  # GEMM 연산 한 번
```

덕분에 모든 쿼리-키 쌍을 한꺼번에 대규모로 병렬 처리할 수 있다.

### 대안과 비교

| 어텐션의 종류 | 점수 함수 | 복잡도 | 참고 |
|----------------|------------------|------------|-------|
| 내적 | $\mathbf{q}^T \mathbf{k}$ | $O(d)$ | 가장 간단하고 빠르다 |
| 덧셈형 (바다나우) | $\mathbf{v}^T \tanh(\mathbf{W}_q\mathbf{q} + \mathbf{W}_k\mathbf{k})$ | $O(d)$에 비선형 | 표현력이 크고 느리다 |
| 곱셈형 | $\mathbf{q}^T \mathbf{W} \mathbf{k}$ | $O(d^2)$ | 학습되는 상호작용 |

학습된 Q·K 투영을 포함하면 내적 점수도 학습 가능한 관련도를 표현한다. 가산 점수와 내적 점수의 표현력·속도는 차원과 투영의 재사용, 하드웨어 구현에 따라 달라진다.

---

## 3. 배율 조정의 결정적인 구실

q와 k의 모든 성분이 서로 독립이고 평균 0·분산 1이면 내적의 평균은 0, 분산은 $d_k$다. 따라서 $\sqrt{d_k}$로 나누면 이 가정 아래 분산이 1이 된다. 내적의 분포가 정확한 정규분포라는 뜻은 아니며, 적절한 조건에서 큰 차원의 근사로 볼 수 있다. 학습된 Q·K는 이 독립성이나 분산 가정을 따르지 않을 수 있다.

소프트맥스는 공통 상수를 더해도 변하지 않는다. 따라서 점수의 절댓값보다 위치 사이의 차이가 중요하다. 한 값이 압도하면 소프트맥스 야코비안 $\partial\alpha_i/\partial s_j=\alpha_i(\delta_{ij}-\alpha_j)$가 작아져 Q·K 쪽 학습 신호를 약화할 수 있다. 값 V로 가는 경로나 잔차 경로의 모든 그래디언트까지 사라진다는 뜻은 아니다.

점수 $(20,22,18,21)$의 소프트맥스는 약 $(0.0889,0.6572,0.0120,0.2418)$이다. 이를 $\sqrt{512}$로 나눈 점수의 소프트맥스는 약 $(0.2467,0.2695,0.2259,0.2579)$다. 스케일은 이 예제의 분포를 완만하게 하지만 모든 점수가 특정 구간에 머물거나 모든 학습이 안정된다는 보장은 없다.

스케일을 양수 온도와 연결해 볼 수도 있다. 다만 온도는 점수의 날카로움을 조절하는 파라메터이며 탐험·활용이나 모델 규모 확장의 성공을 자동으로 보장하지 않는다.

## 4. 내용으로 찾는 부드러운 기억으로 본 어텐션

어텐션은 미분 가능한 기억 시스템으로 볼 수 있다.

| 전통적인 기억 | 어텐션에서의 대응 |
|--------------------|---------------------|
| 메모리 주소 | 키 벡터 $\mathbf{k}$ |
| 메모리 내용 | 값 벡터 $\mathbf{v}$ |
| 쿼리·찾기 | 쿼리 벡터 $\mathbf{q}$ |
| 딱딱한 주소 맞추기 | 부드러운 유사도 맞추기 |
| 읽기 연산 | 가중된 값 꺼내기 |

**쓰기** $(k, v)$: 기억 창고에 키-값 쌍을 더한다.

쿼리 $q$으로 **읽기**: $\text{output} = \sum_j \text{similarity}(q, k_j) \cdot v_j$

이는 내용으로 찾는 방식이다. 어디에 담겼는지가 아니라 *무엇을* 찾는지로 꺼내 온다. 이산적인 기억 찾기와 달리 연산 전체로 그래디언트가 흘러 처음부터 끝까지 학습할 수 있다.

---

## 5. 그래디언트의 흐름과 부드러운 선택

유효 위치가 하나 이상인 행에서 소프트맥스는 웨이트 합을 1로 만든다. 특정 위치의 웨이트가 크면 그 값 벡터가 가중합에 크게 반영되지만, Q·K의 그래디언트와 V의 그래디언트는 서로 다른 경로를 따른다.

차원이 커질 때의 분산 계산은 스케일의 설계 근거다. 실제 모델에서의 포화나 학습 병목은 엑티베이션과 그래디언트를 측정해 판단하며, 레이어 수나 차원만으로 학습 정지를 단정하지 않는다.

## 6. PyTorch 구현

### 핵심 모듈

아래 모듈의 nan_to_num은 모든 위치가 가려진 행의 NaN 웨이트를 0으로 바꾸지만, 잘못된 입력에서 생긴 NaN도 감출 수 있다. 입력 유한성 검사를 대신하지 않는다. return_weights=False여도 내부 웨이트 행렬은 계산하므로 그 자체로 메모리 절약 커널이 되지는 않는다.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class ScaledDotProductAttention(nn.Module):
    """
    배율 조정 내적 어텐션.
    
    계산: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    
    배치 계산과 다중 머리 어텐션, 선택적인 가림막,
    어텐션 웨이트의 드롭아웃을 지원한다.
    """
    
    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_weights: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        인수:
            query: (..., seq_q, d_k)
            key: (..., seq_k, d_k)
            value: (..., seq_k, d_v)
            mask: (..., seq_q, seq_k)로 방송 가능. 0이면 가린다.
            
        반환값:
            output: (..., seq_q, d_v)
            attention_weights: return_weights이면 (..., seq_q, seq_k)
        """
        d_k = query.size(-1)
        
        # 점수: Q @ K^T / sqrt(d_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = torch.nan_to_num(attention_weights, nan=0.0)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, value)
        
        if return_weights:
            return output, attention_weights
        return output, None
```

### 함수형 인터페이스

```python
def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    training: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    함수형 배율 조정 내적 어텐션.
    
    인수:
        query: (..., seq_q, d_k)
        key: (..., seq_k, d_k)
        value: (..., seq_k, d_v)
        mask: (..., seq_q, seq_k)로 방송 가능
        
    반환값:
        output: (..., seq_q, d_v)
        weights: (..., seq_q, seq_k)
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    weights = F.softmax(scores, dim=-1)
    weights = torch.nan_to_num(weights, nan=0.0)
    
    if dropout_p > 0.0 and training:
        weights = F.dropout(weights, p=dropout_p, training=True)
    
    return torch.matmul(weights, value), weights
```

### PyTorch 내장 함수 쓰기

PyTorch 2.0 이상은 최적화된 구현을 제공한다.

```python
from torch.nn.functional import scaled_dot_product_attention as torch_sdpa

output = torch_sdpa(
    query, key, value,
    attn_mask=mask,
    dropout_p=0.1,
    is_causal=False
)
```

[PyTorch 내장 SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)는 장치·자료형·형태 등의 조건에 따라 가능한 구현을 선택한다. 지정된 dropout_p는 모델의 eval 모드와 무관하게 적용되므로 평가에서는 0을 넘겨야 한다. 불리언 마스크는 True가 허용 위치이고 실수 마스크는 점수에 더해지는 바이어스여서, 위 사용자 정의 함수의 0·1 마스크와 그대로 교환할 수 없다.

### 배율 조정을 실험으로 확인하기

```python
def verify_scaling_effect():
    """배율 조정이 분산을 정규화함을 실험으로 확인한다."""
    torch.manual_seed(42)
    
    dims = [16, 64, 256, 512, 1024]
    num_samples = 10000
    
    print("Empirical verification of dot-product variance:")
    print("-" * 70)
    print(f"{'d_k':>6} | {'Unscaled Var':>12} | {'Scaled Var':>12} | {'sqrt(d_k)':>10}")
    print("-" * 70)
    
    for d_k in dims:
        Q = torch.randn(num_samples, d_k)
        K = torch.randn(num_samples, d_k)
        
        unscaled = (Q * K).sum(dim=1)
        scaled = unscaled / (d_k ** 0.5)
        
        print(f"{d_k:>6} | {unscaled.var().item():>12.2f} | "
              f"{scaled.var().item():>12.4f} | {d_k**0.5:>10.2f}")

verify_scaling_effect()
```

**출력:**
```
Empirical verification of dot-product variance:
----------------------------------------------------------------------
   d_k | Unscaled Var |   Scaled Var |   sqrt(d_k)
----------------------------------------------------------------------
    16 |        16.05 |       1.0032 |       4.00
    64 |        63.81 |       0.9970 |       8.00
   256 |       257.42 |       1.0055 |      16.00
   512 |       513.68 |       1.0034 |      22.63
  1024 |      1026.39 |       1.0024 |      32.00
----------------------------------------------------------------------
```

---

## 7. 마스크 전략

마스크는 어떤 자리가 어떤 자리에 주목할 수 있는지를 다스린다.

### 마스크의 종류

1. **패딩 마스크**: 길이가 다른 시퀀스에서 덧댐 토큰을 무시한다
2. **인과 마스크**: 미래 토큰에 주목하지 못하게 한다 (자기회귀 모델)
3. **맞춤 마스크**: 성긴 어텐션 방식을 구현한다

### 구현

```python
def create_attention_masks(batch_size: int, seq_q: int, seq_k: int):
    """여러 가림막 전략을 시연한다."""
    
    # 덧댐 가림막: (배치, 1, 1, seq_k)
    pad_mask = torch.ones(batch_size, 1, 1, seq_k)
    pad_mask[:, :, :, -2:] = 0  # 마지막 두 자리는 덧댐이다
    
    # 인과 가림막: (1, 1, seq_q, seq_k)
    causal_mask = torch.tril(torch.ones(1, 1, seq_q, seq_k))
    
    # 합친 가림막
    combined_mask = pad_mask * causal_mask
    
    return pad_mask, causal_mask, combined_mask
```

```
Causal Mask (seq=4):          Padding Mask (pad last 2):
┌─────────────┐               ┌─────────────┐
│ 1 0 0 0 │               │ 1 1 0 0 │
│ 1 1 0 0 │               │ 1 1 0 0 │
│ 1 1 1 0 │               │ 1 1 0 0 │
│ 1 1 1 1 │               │ 1 1 0 0 │
└─────────────┘               └─────────────┘
```

---

## 8. 다른 배율 조정 전략

### 쿼리와 키에 나누어 배율 조정하기

마지막 점수의 배율을 조정하는 대신 쿼리와 키의 배율을 따로 조정한다.

$$\text{score} = \left(\frac{\mathbf{q}}{d_k^{1/4}}\right)^T \left(\frac{\mathbf{k}}{d_k^{1/4}}\right) = \frac{\mathbf{q}^T \mathbf{k}}{\sqrt{d_k}}$$

수학적으로는 같고 수치적인 성질이 더 나을 수 있다.

```python
class BalancedScaledAttention(nn.Module):
    def __init__(self, d_k: int):
        super().__init__()
        self.scale = d_k ** (-0.25)  # 네제곱근
    
    def forward(self, Q, K, V, mask=None):
        Q_scaled = Q * self.scale
        K_scaled = K * self.scale
        scores = torch.matmul(Q_scaled, K_scaled.transpose(-2, -1))
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        return F.softmax(scores, dim=-1) @ V
```

### 학습되는 온도

아래 temperature는 제한 없는 파라메터여서 학습 중 0이나 음수가 될 수 있다. 양수 온도라는 해석을 유지하려면 softplus나 지수 변환 등의 파라메터화가 필요하다.

어떤 모델은 학습되는 배율 인수를 쓴다.

```python
class TemperatureScaledAttention(nn.Module):
    def __init__(self, d_k: int):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(math.sqrt(d_k)))
    
    def forward(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.temperature
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        return F.softmax(scores, dim=-1) @ V
```

### 코사인 유사도

쿼리와 키를 단위 벡터로 정규화하면 차원과 상관없이 점수가 $[-1, 1]$에 묶인다.

```python
class CosineAttention(nn.Module):
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, Q, K, V, mask=None):
        Q_norm = F.normalize(Q, dim=-1)
        K_norm = F.normalize(K, dim=-1)
        scores = torch.matmul(Q_norm, K_norm.transpose(-2, -1)) / self.temperature
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        return F.softmax(scores, dim=-1) @ V
```

---

## 9. 계산량 분석

쿼리의 길이가 $n$, 키·값의 길이가 $m$, 차원이 $d$일 때 다음과 같다.

| 연산 | 시간 복잡도 | 공간 복잡도 |
|-----------|-----------------|------------------|
| $\mathbf{Q}\mathbf{K}^T$ | $O(nmd)$ | $O(nm)$ |
| 배율 조정 | $O(nm)$ | $O(1)$ |
| 소프트맥스 | $O(nm)$ | $O(nm)$ |
| $\mathbf{A}\mathbf{V}$ | $O(nmd)$ | $O(nd)$ |
| **합계** | $O(nmd)$ | $O(nm + nd)$ |

$n = m$인 셀프 어텐션에서는 시간이 $O(n^2d)$, 공간이 $O(n^2)$이다. 어텐션 행렬을 담는 데 드는 이차 메모리가 긴 시퀀스를 다루는 데 가장 큰 병목이다.

### 수치적 안정성

PyTorch의 `F.softmax`은 로그-합-지수 요령을 저절로 적용한다.

$$\text{softmax}(\mathbf{x})_i = \frac{\exp(x_i - \max(\mathbf{x}))}{\sum_j \exp(x_j - \max(\mathbf{x}))}$$

---

## 10. 메모리를 아끼는 판본

### 청크로 나눈 어텐션

아래 함수는 쿼리 축을 나누므로 한 번의 점수 배열은 줄지만 학습 자동 미분이 청크별 중간값을 모두 보관하면 전체 학습 메모리가 같은 비율로 줄지는 않는다. 마스크·드롭아웃은 없고 출력 버퍼는 장치만 맞추므로 입력 자료형과 맞출 필요가 있다.

어텐션을 덩이로 나누어 처리하여 최대 메모리를 줄인다.

```python
def chunked_attention(query, key, value, chunk_size=64):
    """메모리: O(seq_len^2)이 아니라 O(chunk_size * seq_len)이다."""
    batch, seq_q, d_k = query.shape
    d_v = value.size(-1)
    output = torch.zeros(batch, seq_q, d_v, device=query.device)
    
    for i in range(0, seq_q, chunk_size):
        q_chunk = query[:, i:i+chunk_size]
        scores = torch.matmul(q_chunk, key.transpose(-2, -1)) / math.sqrt(d_k)
        weights = F.softmax(scores, dim=-1)
        output[:, i:i+chunk_size] = torch.matmul(weights, value)
    
    return output
```

### FlashAttention

FlashAttention은 전체 점수 행렬을 HBM에 저장하지 않고 타일 단위로 정확한 어텐션을 계산하는 방법이다. 고정 피처 차원에서 작업 메모리의 길이 의존성을 줄이지만, 모든 쿼리·키 쌍을 계산하는 시간까지 선형으로 바꾸지는 않는다. PyTorch는 지원 조건에 따라 이 계열 커널을 사용할 수 있다. 아래 컨텍스트가 모든 환경에서 FlashAttention을 강제하는 것은 아니다.

```python
with torch.backends.cuda.sdp_kernel(
    enable_flash=True,
    enable_math=True,
    enable_mem_efficient=True
):
    output = F.scaled_dot_product_attention(query, key, value)
```

---

## 11. 고차원의 기하

### 측도의 집중

고차원에서는 흥미로운 기하 현상이 일어난다.

- **부피의 대부분이 겉면 가까이 있다**: 고차원 구의 부피는 거의 전부가 겉면 부근에 몰려 있다
- **거의 직교함**: 무작위 벡터는 높은 확률로 거의 직교한다
- **평균 둘레로의 집중**: 내적이 기댓값 둘레에 몰린다

무작위 단위 벡터 $\mathbf{u}, \mathbf{v} \in \mathbb{R}^d$에 대해 다음이 성립한다.

$$\mathbb{E}[\mathbf{u}^T \mathbf{v}] = 0, \quad \text{Var}(\mathbf{u}^T \mathbf{v}) = \frac{1}{d}$$

위 단위 벡터의 식에서는 내적 분산이 차원에 따라 작아진다. 성분 분산이 1인 앞선 벡터는 길이 제곱의 평균이 d이므로 내적 분산이 커진다. 두 정규화 조건을 섞어 같은 결론을 내리면 안 된다.

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
내적 스케일의 분산 근거와 한계를 설명하라.

</div>

??? success "연습문제 1 풀이"
    모든 성분이 독립이고 평균 0·분산 1이면 내적 분산이 $d_k$여서 $\sqrt{d_k}$로 나누면 1이다. 학습된 Q·K가 같은 가정을 만족한다고 보장되지는 않으며 스케일만으로 포화·소실·폭발을 모두 막을 수 없다. 단위 길이 벡터의 분산 계산과도 구분한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
$d_k = 2$일 때 $Q = [[1,0],[0,1]]$, $K = [[1,0],[0,1]]$, $V = [[1,2],[3,4]]$에 대한 어텐션 출력을 계산하라.

</div>

??? success "연습문제 2 풀이"
    점수: $QK^\top / \sqrt{2} = [[1,0],[0,1]] / \sqrt{2} = [[0.707, 0], [0, 0.707]]$. 웨이트: 행마다 소프트맥스를 취하면 대략 $[[0.67, 0.33], [0.33, 0.67]]$이다. 출력: 웨이트 $\times V = [[0.67 \cdot 1 + 0.33 \cdot 3, 0.67 \cdot 2 + 0.33 \cdot 4], [0.33 \cdot 1 + 0.67 \cdot 3, 0.33 \cdot 2 + 0.67 \cdot 4]] = [[1.66, 2.66], [2.34, 3.34]]$.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
행렬 연산으로 스케일 내적 어텐션을 PyTorch로 구현하라.

</div>

??? success "연습문제 3 풀이"
    ```python
    def scaled_dot_product_attention(Q, K, V, mask=None):
        d_k = Q.size(-1)
        scores = Q @ K.transpose(-2, -1) / d_k**0.5
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=-1)
        return weights @ V
    ```

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff easy" title="쉬움"></span>
시퀀스의 길이가 $n$이고 차원이 $d$일 때 스케일 내적 어텐션의 계산 복잡도는 얼마인가?

</div>

??? success "연습문제 4 풀이"
    $QK^\top$을 계산하는 데 $O(n^2 d)$, 소프트맥스에 $O(n^2)$, $V$을 곱하는 데 $O(n^2 d)$이 들어 모두 $O(n^2 d)$이다. $n$에 대한 이차 의존이 긴 시퀀스에서 가장 큰 병목이며, 그 때문에 선형 어텐션이나 성긴 어텐션 같은 효율적인 판본이 나왔다.

## 정리하며

스케일 내적 어텐션은 Q·K 내적을 차원에 맞춰 조절한 뒤 V를 결합한다. 스케일의 분산 설명에는 통계 가정이 필요하다. 마스크 의미와 평가 드롭아웃, 전체 행 마스킹과 메모리 구현을 확인해야 한다.

**참고 문헌**

1. Vaswani, A., et al. (2017). "Attention Is All You Need." *NeurIPS*.
2. Bahdanau, D., Cho, K., & Bengio, Y. (2015). "Neural Machine Translation by Jointly Learning to Align and Translate." *ICLR*.
3. Dao, T., et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." *NeurIPS*.
4. Rabe, M. N., & Staats, C. (2021). "Self-attention Does Not Need $O(n^2)$ Memory." *arXiv:2112.05682*.
5. Xiong, R., et al. (2020). "On Layer Normalization in the Transformer Architecture." *ICML*.
6. Noci, L., et al. (2022). "Signal Propagation in Transformers: Theoretical Perspectives and the Role of Rank Collapse." *NeurIPS*.
