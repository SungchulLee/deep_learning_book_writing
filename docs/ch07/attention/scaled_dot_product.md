# 배율 조정 내적 어텐션
## 들어가며

배율 조정 내적 어텐션은 트랜스포머 구조의 근본 구성 블록이다. Vaswani 등(2017)이 "Attention Is All You Need"에서 내놓았으며, 요즘 하드웨어에서 크게 병렬화할 수 있는 행렬 연산으로 어텐션 가중치를 효율적이고 효과적으로 계산한다.

핵심 혁신은 내적 어텐션의 계산 효율에 학습 중 기울기를 안정시키는 배율 인수를 더한 데 있다. 사소해 보이는 이 손질이 어텐션 기반의 깊은 모델을 학습시키는 데 결정적이었다.

## 수학적 정식화

### 어텐션 함수

질의 $\mathbf{Q} \in \mathbb{R}^{n \times d_k}$, 열쇠 $\mathbf{K} \in \mathbb{R}^{m \times d_k}$, 값 $\mathbf{V} \in \mathbb{R}^{m \times d_v}$이 주어지면 배율 조정 내적 어텐션은 다음을 계산한다.

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

### 한 걸음씩 계산하기

**1단계: 날 어텐션 점수 계산**

$$\mathbf{S} = \mathbf{Q}\mathbf{K}^T \in \mathbb{R}^{n \times m}$$

성분 $s_{ij}$은 질의 $i$과 열쇠 $j$의 유사도를 잰다.

$$s_{ij} = \mathbf{q}_i^T \mathbf{k}_j = \sum_{l=1}^{d_k} q_{il} k_{jl}$$

내적이 클수록 질의와 열쇠가 임베딩 공간에서 비슷한 방향을 가리킨다는 뜻이다.

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
| $\mathbf{Q}$ | $(n, d_k)$ | 차원이 $d_k$인 질의 벡터 $n$개 |
| $\mathbf{K}$ | $(m, d_k)$ | 차원이 $d_k$인 열쇠 벡터 $m$개 |
| $\mathbf{V}$ | $(m, d_v)$ | 차원이 $d_v$인 값 벡터 $m$개 |
| $\mathbf{S}$ | $(n, m)$ | 어텐션 점수 |
| $\mathbf{A}$ | $(n, m)$ | 어텐션 가중치 (소프트맥스 뒤) |
| $\mathbf{O}$ | $(n, d_v)$ | 출력 벡터 |

**핵심 제약:**

- 질의와 열쇠의 차원이 같아야 한다($d_k$)
- 열쇠와 값의 순차열 길이가 같아야 한다($m$)
- 값의 차원($d_v$)은 아무것이나 되며 그것이 출력 차원이 된다
- 자기 어텐션에서는 $n = m$이다(질의와 열쇠·값이 같은 순차열이다)

### 배치와 머리가 있을 때

실제로는 배치와 여러 머리를 가진 텐서를 다룬다.

| 텐서 | 모양 | 설명 |
|--------|-------|-------------|
| $\mathbf{Q}$ | $(B, H, n, d_k)$ | 배치 다중 머리 질의 |
| $\mathbf{K}$ | $(B, H, m, d_k)$ | 배치 다중 머리 열쇠 |
| $\mathbf{V}$ | $(B, H, m, d_v)$ | 배치 다중 머리 값 |
| $\mathbf{O}$ | $(B, H, n, d_v)$ | 배치 다중 머리 출력 |

여기서 $B$은 배치 크기이고 $H$은 어텐션 머리의 수이다.

## 왜 내적인가

내적 $\mathbf{q}^T \mathbf{k}$은 관련성을 재기에 알맞은 좋은 성질을 지닌다.

### 기하학적 해석

$$\mathbf{q}^T \mathbf{k} = \|\mathbf{q}\| \|\mathbf{k}\| \cos\theta$$

여기서 $\theta$은 두 벡터 사이의 각이다. 벡터가 비슷한 방향을 가리키거나($\theta$이 작거나) 크기가 크면 내적이 크다. 이는 직관적인 유사도를 준다. 임베딩 공간에서 "정렬된" 질의와 열쇠가 높은 어텐션 점수를 낸다.

### 계산 효율

내적은 행렬 곱으로 계산할 수 있고, 이는 GPU에서 매우 잘 최적화되어 있다.

```python
scores = torch.matmul(Q, K.transpose(-2, -1))  # GEMM 연산 한 번
```

덕분에 모든 질의-열쇠 쌍을 한꺼번에 대규모로 병렬 처리할 수 있다.

### 대안과 견주기

| 어텐션의 종류 | 점수 함수 | 복잡도 | 참고 |
|----------------|------------------|------------|-------|
| 내적 | $\mathbf{q}^T \mathbf{k}$ | $O(d)$ | 가장 간단하고 빠르다 |
| 덧셈형 (바다나우) | $\mathbf{v}^T \tanh(\mathbf{W}_q\mathbf{q} + \mathbf{W}_k\mathbf{k})$ | $O(d)$에 비선형 | 표현력이 크고 느리다 |
| 곱셈형 | $\mathbf{q}^T \mathbf{W} \mathbf{k}$ | $O(d^2)$ | 학습되는 상호작용 |

(다중 머리 어텐션처럼) 학습된 사영과 함께 쓰는 배율 조정 내적은 덧셈형 어텐션과 표현력이 비슷하면서도 행렬 곱의 하드웨어 최적화 덕분에 훨씬 빠르다.

## 배율 조정의 결정적인 구실

### 분산 폭발 문제

성분마다 평균 $\mu = 0$, 분산 $\sigma^2 = 1$인 분포에서 서로 독립으로 뽑은 무작위 벡터 $\mathbf{q}, \mathbf{k} \in \mathbb{R}^{d_k}$ 둘을 생각해 보자.

**내적의 평균:**

$$\mathbb{E}[\mathbf{q}^T \mathbf{k}] = \sum_{i=1}^{d_k} \mathbb{E}[q_i] \mathbb{E}[k_i] = 0$$

**내적의 분산:**

$$\text{Var}(\mathbf{q}^T \mathbf{k}) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = \sum_{i=1}^{d_k} \mathbb{E}[q_i^2]\mathbb{E}[k_i^2] = d_k$$

분산이 **차원에 선형으로 커진다**.

$$\mathbf{q}^T \mathbf{k} \sim \mathcal{N}(0, d_k)$$

흔한 트랜스포머 차원에서는 다음과 같다.

| $d_k$ | $\sqrt{d_k}$ | 흔한 점수 범위 ($\pm 2\sigma$) |
|-------|--------------|-------------------------------------|
| 16 | 4 | [-8, 8] |
| 64 | 8 | [-16, 16] |
| 128 | 11.3 | [-22.6, 22.6] |
| 512 | 22.6 | [-45.2, 45.2] |

### 소프트맥스의 포화

소프트맥스의 입력이 크면 분포가 거의 원-핫이 된다.

$$\text{softmax}([100, 0, 0]) \approx [1.0, 0.0, 0.0]$$

그러면 다음이 일어난다.

1. **기울기 소실**: 포화된 영역에서 $\frac{\partial \text{softmax}}{\partial x} \approx 0$이다
2. **정보의 손실**: 미묘한 어텐션 무늬가 딱딱한 선택으로 무너진다
3. **학습의 불안정**: 기울기가 잡음이 많고 믿을 수 없게 된다

### 소프트맥스의 야코비 행렬

소프트맥스의 기울기는 다음과 같은 꼴이다.

$$\frac{\partial \alpha_i}{\partial s_j} = \alpha_i (\delta_{ij} - \alpha_j)$$

소프트맥스가 포화하면($\alpha_1 \approx 1$이고 나머지가 $\approx 0$이면) 다음과 같다.

$$\frac{\partial \alpha_1}{\partial s_j} \approx 1 \cdot (1 - 1) = 0 \quad \text{for } j = 1$$

$$\frac{\partial \alpha_1}{\partial s_j} \approx 1 \cdot (0 - 0) = 0 \quad \text{for } j \neq 1$$

**모든 기울기가 사라진다.** 모델이 어텐션 가중치를 고치는 법을 배울 수 없다.

기울기는 어텐션이 불확실할 때(0.5 언저리) 가장 세고 확신할 때(0이나 1 가까이) 가장 약해, 자연스럽게 애매한 경우에 학습이 집중된다.

### 배율 조정이라는 해법

$\sqrt{d_k}$으로 나누면 분산이 정규화된다.

$$\text{Var}\left[\frac{\mathbf{q}^T \mathbf{k}}{\sqrt{d_k}}\right] = \frac{d_k}{d_k} = 1$$

그러면 점수가 소프트맥스의 기울기가 멀쩡한 알맞은 범위(보통 $[-3, 3]$)에 머문다.

### 수치 예제

$d_k = 512$이라 하자. 어텐션 점수가 $\mathbf{s} = (20, 22, 18, 21)$이라고 하자.

**배율을 조정하지 않은 소프트맥스:**

$$\text{softmax}(20, 22, 18, 21) \approx (0.018, 0.731, 0.002, 0.249)$$

**배율을 조정한 경우 ($\sqrt{512} \approx 22.6$으로 나눔):**

$$\mathbf{s}_{\text{scaled}} = (0.88, 0.97, 0.80, 0.93)$$

$$\text{softmax}(0.88, 0.97, 0.80, 0.93) \approx (0.227, 0.249, 0.210, 0.314)$$

배율을 조정한 쪽이 분포가 훨씬 매끄러워 모든 자리에서 뜻있는 기울기를 얻는다.

### 왜 하필 $\sqrt{d_k}$인가

| 배율 인수 | 나오는 분산 | 효과 |
|----------------|-------------------|--------|
| 없음 | $d_k$ | 소프트맥스가 포화한다 |
| $d_k$ | $1/d_k$ | 점수가 너무 작아 어텐션이 거의 고르다 |
| $\sqrt{d_k}$ | $1$ | 딱 알맞다. 분산이 1이 된다 |

$\sqrt{d_k}$이라는 선택은 표준적인 초기화를 가정할 때 분산을 1로 맞추어 주므로 우아하다.

### 온도로 보는 해석

배율 인수는 통계 역학의 역온도처럼 움직인다.

$$\text{softmax}\left(\frac{\mathbf{s}}{\sqrt{d_k}}\right) = \text{softmax}\left(\frac{\mathbf{s}}{T}\right) \text{ with } T = \sqrt{d_k}$$

- **높은 온도** ($T$이 큼): 고른 어텐션, 탐험
- **낮은 온도** ($T$이 작음): 몰린 어텐션, 활용

배율 조정은 차원에 맞추어지는 온도를 준다. 모델이 클수록 온도가 저절로 높아져 규모가 커져도 포화하지 않는다.

### 기울기 소실의 연쇄

트랜스포머에서 기울기는 여러 부품을 지나야 한다. 출력 사영 → 값 모으기 → 어텐션 가중치(소프트맥스) → 점수 계산 → Q·K 사영 → 층 정규화 → 잔차 연결 → 앞선 층이다. 어텐션 가중치가 포화하면 기울기가 어텐션 장치를 제대로 지나지 못해 병목이 생기고 신경망 전체의 학습이 멈춘다.

## 내용으로 찾는 부드러운 기억으로 본 어텐션

어텐션은 미분 가능한 기억 시스템으로 볼 수 있다.

| 전통적인 기억 | 어텐션에서의 대응 |
|--------------------|---------------------|
| 메모리 주소 | 열쇠 벡터 $\mathbf{k}$ |
| 메모리 내용 | 값 벡터 $\mathbf{v}$ |
| 질의·찾기 | 질의 벡터 $\mathbf{q}$ |
| 딱딱한 주소 맞추기 | 부드러운 유사도 맞추기 |
| 읽기 연산 | 가중된 값 꺼내기 |

**쓰기** $(k, v)$: 기억 창고에 열쇠-값 쌍을 더한다.

질의 $q$으로 **읽기**: $\text{output} = \sum_j \text{similarity}(q, k_j) \cdot v_j$

이는 내용으로 찾는 방식이다. 어디에 담겼는지가 아니라 *무엇을* 찾는지로 꺼내 온다. 이산적인 기억 찾기와 달리 연산 전체로 기울기가 흘러 처음부터 끝까지 학습할 수 있다.

## 기울기의 흐름과 부드러운 선택

### 자리들 사이의 경쟁

소프트맥스는 경쟁을 만든다. 한 자리에 주목하면 다른 자리에 대한 주목이 줄어든다. 한 자리가 도맡으면(어떤 $j$에 대해 $A_{ij} \approx 1$이면) 기울기가 주로 그 자리를 타고 흐르며, 미분 가능한 채로 남는 **부드러운 이산 선택**을 구현한다.

### 배율 조정이 특히 중요할 때

**임베딩 차원이 클 때.** 요즘 모델은 차원이 크다.

| 모델 | $d_{\text{model}}$ | $d_k$ (머리당) |
|-------|-------------------|------------------|
| BERT-base | 768 | 64 |
| GPT-2 | 768~1600 | 64 |
| GPT-3 | 12288 | 128 |

배율을 조정하지 않으면 점수의 표준편차가 8~11이 되어 소프트맥스가 포화한다.

**깊은 신경망.** 깊은 트랜스포머(12층 이상)에서는 기울기의 흐름이 매우 중요하다. 앞쪽 층의 어텐션이 포화하면 심각한 기울기 병목이 생긴다.

**규모 확장성.** 배율 조정은 모델의 크기가 달라도 소프트맥스의 거동을 한결같게 하여 다시 맞추지 않고도 구조의 규모를 키울 수 있게 한다.

## PyTorch 구현

### 핵심 모듈

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
    어텐션 가중치의 드롭아웃을 지원한다.
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

이 내장 함수는 하드웨어와 입력의 성질에 따라 가장 효율적인 구현(FlashAttention, 메모리 효율 어텐션, 표준)을 저절로 고른다.

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

## 가림막 전략

가림막은 어떤 자리가 어떤 자리에 주목할 수 있는지를 다스린다.

### 가림막의 종류

1. **덧댐 가림막**: 길이가 다른 순차열에서 덧댐 토큰을 무시한다
2. **인과 가림막**: 미래 토큰에 주목하지 못하게 한다 (자기회귀 모델)
3. **맞춤 가림막**: 성긴 어텐션 방식을 구현한다

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

## 다른 배율 조정 전략

### 질의와 열쇠에 나누어 배율 조정하기

마지막 점수의 배율을 조정하는 대신 질의와 열쇠의 배율을 따로 조정한다.

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

질의와 열쇠를 단위 벡터로 정규화하면 차원과 상관없이 점수가 $[-1, 1]$에 묶인다.

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

## 계산량 분석

질의의 길이가 $n$, 열쇠·값의 길이가 $m$, 차원이 $d$일 때 다음과 같다.

| 연산 | 시간 복잡도 | 공간 복잡도 |
|-----------|-----------------|------------------|
| $\mathbf{Q}\mathbf{K}^T$ | $O(nmd)$ | $O(nm)$ |
| 배율 조정 | $O(nm)$ | $O(1)$ |
| 소프트맥스 | $O(nm)$ | $O(nm)$ |
| $\mathbf{A}\mathbf{V}$ | $O(nmd)$ | $O(nd)$ |
| **합계** | $O(nmd)$ | $O(nm + nd)$ |

$n = m$인 자기 어텐션에서는 시간이 $O(n^2d)$, 공간이 $O(n^2)$이다. 어텐션 행렬을 담는 데 드는 이차 메모리가 긴 순차열을 다루는 데 가장 큰 병목이다.

### 수치적 안정성

PyTorch의 `F.softmax`은 로그-합-지수 요령을 저절로 적용한다.

$$\text{softmax}(\mathbf{x})_i = \frac{\exp(x_i - \max(\mathbf{x}))}{\sum_j \exp(x_j - \max(\mathbf{x}))}$$

## 메모리를 아끼는 판본

### 덩이로 나눈 어텐션

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

### 플래시 눈길

FlashAttention(Dao 등, 2022)은 데이터를 빠른 SRAM에 두는 타일 알고리즘으로 정확한 어텐션을 $O(n)$ 메모리에 계산한다. PyTorch 2.0 이상은 쓸 수 있으면 저절로 FlashAttention을 쓴다.

```python
with torch.backends.cuda.sdp_kernel(
    enable_flash=True,
    enable_math=True,
    enable_mem_efficient=True
):
    output = F.scaled_dot_product_attention(query, key, value)
```

## 고차원의 기하

### 측도의 집중

고차원에서는 흥미로운 기하 현상이 일어난다.

- **부피의 대부분이 겉면 가까이 있다**: 고차원 구의 부피는 거의 전부가 겉면 부근에 몰려 있다
- **거의 직교함**: 무작위 벡터는 높은 확률로 거의 직교한다
- **평균 둘레로의 집중**: 내적이 기댓값 둘레에 몰린다

무작위 단위 벡터 $\mathbf{u}, \mathbf{v} \in \mathbb{R}^d$에 대해 다음이 성립한다.

$$\mathbb{E}[\mathbf{u}^T \mathbf{v}] = 0, \quad \text{Var}(\mathbf{u}^T \mathbf{v}) = \frac{1}{d}$$

배율을 조정하지 않으면 정렬의 작은 차이가 커지는 분산 때문에 부풀려지고, 차원에 따라 소프트맥스가 보는 점수의 범위가 크게 달라진다. 배율을 조정하면 차원과 상관없이 분산이 일정하게 정규화되어 모델의 크기가 달라도 거동이 한결같다.

## 요약

### 핵심 식

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

### 핵심 설계 원리

1. **내적 점수 매기기**: 행렬 곱으로 효율적인 병렬 계산을 하며 GPU 최적화를 활용한다
2. **$\frac{1}{\sqrt{d_k}}$으로 배율 조정**: 분산을 1로 지켜 소프트맥스에서 기울기가 사라지는 것을 막는다
3. **소프트맥스 정규화**: 올바른 확률 분포를 만들어 부드러운 선택을 가능하게 한다
4. **값 모으기**: 볼록 결합으로 가중해 모아 정보를 지킨다

### 깊은 통찰

1. **기하적 관점**: 내적은 임베딩 공간에서의 정렬을 잰다. 질의는 비슷한 방향을 가리키는 열쇠를 찾는다
2. **기억의 관점**: 어텐션은 부드러운 검색을 하는 미분 가능한 내용 주소 기억을 구현한다
3. **온도의 관점**: 배율 인수가 어텐션의 날카로움을 다스려 탐험과 활용의 균형을 잡는다
4. **기울기의 관점**: 소프트맥스는 기울기의 흐름을 지키면서 자리들 사이에 경쟁을 만든다

| 항목 | 배율 조정 없음 | $\sqrt{d_k}$ 배율 조정 |
|--------|-----------------|--------------------------|
| 점수의 분산 | $d_k$ (차원에 따라 커짐) | $1$ (안정적) |
| 소프트맥스의 거동 | 포화한다 | 기울기가 매끄럽다 |
| 어텐션의 분포 | 거의 원-핫 | 고루 퍼짐 |
| 학습 | 기울기 소실 | 안정된 학습 |

## 참고 문헌

1. Vaswani, A., et al. (2017). "Attention Is All You Need." *NeurIPS*.
2. Bahdanau, D., Cho, K., & Bengio, Y. (2015). "Neural Machine Translation by Jointly Learning to Align and Translate." *ICLR*.
3. Dao, T., et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." *NeurIPS*.
4. Rabe, M. N., & Staats, C. (2021). "Self-attention Does Not Need $O(n^2)$ Memory." *arXiv:2112.05682*.
5. Xiong, R., et al. (2020). "On Layer Normalization in the Transformer Architecture." *ICML*.
6. Noci, L., et al. (2022). "Signal Propagation in Transformers: Theoretical Perspectives and the Role of Rank Collapse." *NeurIPS*.

## 연습문제

**연습문제 1.**
내적을 $1/\sqrt{d_k}$으로 배율 조정하는 까닭과 그러지 않으면 어떻게 되는지 설명하라.

??? success "연습문제 1 풀이"
    성분의 분산이 1인 무작위 벡터 $q, k \in \mathbb{R}^{d_k}$에 대해 $q^\top k$의 분산은 $d_k$이다. $d_k$이 크면 내적이 커져 소프트맥스가 기울기가 사라지는 포화 영역으로 밀려난다. $1/\sqrt{d_k}$으로 배율을 조정하면 분산이 1로 정규화되어 소프트맥스가 민감한 영역에 머문다.

---

**연습문제 2.**
$d_k = 2$일 때 $Q = [[1,0],[0,1]]$, $K = [[1,0],[0,1]]$, $V = [[1,2],[3,4]]$에 대한 어텐션 출력을 계산하라.

??? success "연습문제 2 풀이"
    점수: $QK^\top / \sqrt{2} = [[1,0],[0,1]] / \sqrt{2} = [[0.707, 0], [0, 0.707]]$. 가중치: 행마다 소프트맥스를 취하면 대략 $[[0.67, 0.33], [0.33, 0.67]]$이다. 출력: 가중치 $\times V = [[0.67 \cdot 1 + 0.33 \cdot 3, 0.67 \cdot 2 + 0.33 \cdot 4], [0.33 \cdot 1 + 0.67 \cdot 3, 0.33 \cdot 2 + 0.67 \cdot 4]] = [[1.66, 2.66], [2.34, 3.34]]$.

---

**연습문제 3.**
행렬 연산으로 배율 조정 내적 어텐션을 PyTorch로 구현하라.

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

**연습문제 4.**
순차열의 길이가 $n$이고 차원이 $d$일 때 배율 조정 내적 어텐션의 계산 복잡도는 얼마인가?

??? success "연습문제 4 풀이"
    $QK^\top$을 계산하는 데 $O(n^2 d)$, 소프트맥스에 $O(n^2)$, $V$을 곱하는 데 $O(n^2 d)$이 들어 모두 $O(n^2 d)$이다. $n$에 대한 이차 의존이 긴 순차열에서 가장 큰 병목이며, 그 때문에 선형 어텐션이나 성긴 어텐션 같은 효율적인 판본이 나왔다.
