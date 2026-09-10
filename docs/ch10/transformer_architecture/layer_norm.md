# 트랜스포머의 층 정규화

층 정규화는 트랜스포머 구조의 매우 중요한 부품으로, 아래 층(자기 주의와 순전파 신경망)마다 그 뒤에 적용되어 학습을 안정되게 하고 깊게 쌓을 수 있게 한다. 정규화를 아래 층 앞에 두느냐 뒤에 두느냐는 학습의 움직임에 큰 영향을 주며, 요즘 구조는 효율을 위해 RMSNorm 같은 대안으로 모여 왔다.

---

## 1. 층 정규화

### 수식으로 나타내기

층 정규화는 토큰마다 따로 특징 차원에 걸쳐 정규화한다.

$$
\text{LayerNorm}(\mathbf{x}) = \gamma \odot \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

여기서:

- $\mu = \frac{1}{d} \sum_{i=1}^{d} x_i$은 특징에 걸친 평균이다
- $\sigma^2 = \frac{1}{d} \sum_{i=1}^{d} (x_i - \mu)^2$은 특징에 걸친 분산이다
- $\gamma, \beta \in \mathbb{R}^d$은 학습되는 크기 조정과 옮김 매개변수이다
- $\epsilon$은 수치 안정성을 위한 작은 상수이다(대개 $10^{-5}$이나 $10^{-6}$)

### 왜 (배치 정규화가 아니라) 층 정규화인가

배치 정규화는 배치 차원에 걸쳐 정규화하며 특징마다 작은 배치의 모든 토큰에 대한 통계를 셈한다. 트랜스포머에는 다음 까닭으로 알맞지 않다.

1. **수열 길이가 제각각이다**: 배치 안의 수열마다 길이가 달라 채움 자리에 대한 배치 통계가 뜻을 잃는다.
2. **자기 회귀 생성**: 추론 중에 모형이 (배치 크기 1로) 한 번에 토큰 하나를 처리하므로 배치 통계를 쓸 수 없다.
3. **자리에 따른 다름**: 자리마다 토큰 표현의 분포가 다른데 배치에 걸쳐 정규화하면 이를 뭉뚱그린다.

층 정규화는 토큰마다 통계를 셈하여 이 셋을 모두 피한다.

### PyTorch 구현

```python
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    """
    트랜스포머에서 쓰는 층 정규화.
    
    마지막 차원(특징 차원)에 걸쳐 정규화한다.
    """
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        인수:
            x: 입력 텐서 [..., d_model]
        반환값:
            정규화된 텐서 [..., d_model]
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta
```

---

## 2. 앞 정규화와 뒤 정규화

층 정규화를 아래 층에 견주어 어디에 두느냐가 학습의 안정성과 모형의 질에 큰 영향을 준다.

### 뒤 정규화 (본디 트랜스포머)

본디 "Attention Is All You Need" 논문은 잔차를 더한 뒤에 정규화를 둔다.

$$
\mathbf{x}' = \text{LayerNorm}(\mathbf{x} + \text{SubLayer}(\mathbf{x}))
$$

**성질:**

- 정규화가 잔차 연결 뒤의 출력 크기를 다스린다
- 기울기가 층 정규화를 거치므로 기울기의 크기가 제한될 수 있다
- 학습이 흔들리지 않게 하려면 학습률 예열을 조심스레 해야 한다
- 학습이 잘되면 마지막 성능이 조금 더 나은 편이다

### 앞 정규화 (요즘의 표준)

요즘 트랜스포머(GPT-2 이후, LLaMA, T5 v1.1)는 아래 층 앞에 정규화를 둔다.

$$
\mathbf{x}' = \mathbf{x} + \text{SubLayer}(\text{LayerNorm}(\mathbf{x}))
$$

**성질:**

- 기울기가 정규화를 거치지 않고 잔차 연결로 자유로이 흐른다
- 깊은 모형 학습에 더 안정적이다(많은 경우 예열이 필요 없다)
- 잔차 흐름이 정규화되지 않은 표현을 나르며 층마다의 몫을 쌓아 간다
- 층이 대략 12개보다 깊은 모형의 표준 선택이다

### 기울기 흐름 분석

핵심 차이는 기울기가 퍼지는 방식에 있다. 앞 정규화에서는 앞쪽 층의 출력에 대한 손실의 기울기가 곧바로 가는 길을 가진다.

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{x}^{(l)}} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}^{(L)}} + \sum_{k=l}^{L-1} \frac{\partial \mathcal{L}}{\partial \mathbf{x}^{(L)}} \cdot \frac{\partial \text{SubLayer}^{(k)}}{\partial \mathbf{x}^{(l)}}
$$

첫 항은 (잔차를 통한 항등인) 곧은 기울기 길이고 둘째 항은 아래 층의 야코비 행렬을 낀다. 뒤 정규화에서는 이 곧은 길이 정규화 함수에 끊긴다.

### 파이토치 구현 견주기

```python
import torch
import torch.nn as nn
from typing import Optional, Tuple

class PostNormTransformerBlock(nn.Module):
    """뒤 정규화: 잔차를 더한 뒤의 층 정규화(본디 트랜스포머)."""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 뒤 정규화: 잔차 뒤에 정규화한다
        attn_out, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        ff_out = self.ffn(x)
        x = self.norm2(x + ff_out)
        return x

class PreNormTransformerBlock(nn.Module):
    """앞 정규화: 아래 층 앞의 층 정규화(요즘의 표준)."""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 앞 정규화: 아래 층 앞에 정규화한다
        residual = x
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, attn_mask=mask)
        x = residual + self.dropout(attn_out)
        
        residual = x
        x_norm = self.norm2(x)
        ff_out = self.ffn(x_norm)
        x = residual + ff_out
        return x
```

### 마지막 층 정규화

앞 정규화 구조에서는 마지막 트랜스포머 블록 뒤, 출력 사영 앞에 마지막 층 정규화를 둔다. 잔차 흐름이 정규화되지 않은 몫을 쌓아 가므로 이것이 필요하다.

```python
class PreNormTransformer(nn.Module):
    def __init__(self, d_model, num_layers, ...):
        super().__init__()
        self.layers = nn.ModuleList([
            PreNormTransformerBlock(d_model, ...) for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)  # 앞 정규화에 매우 중요하다
        self.output_proj = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)  # 출력 사영 앞에서 정규화한다
        return self.output_proj(x)
```

뒤 정규화 구조는 층마다 출력이 이미 정규화되어 있으므로 이 마지막 정규화가 필요 없다.

---

## 3. RMSNorm

제곱평균제곱근 층 정규화(RMSNorm)는 평균 맞추기 단계를 없애 층 정규화를 간단하게 한 것으로 LLaMA, Mistral을 비롯한 요즘 대형 언어 모형에 쓰인다.

$$
\text{RMSNorm}(\mathbf{x}) = \gamma \odot \frac{\mathbf{x}}{\text{RMS}(\mathbf{x}) + \epsilon}
$$

여기서:

$$
\text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2}
$$

### 동기

RMSNorm은 층 정규화의 중심 다시 맞추기(평균 빼기)가 필요 없고 크기 다시 맞추기(분산 정규화)가 학습 안정성의 주된 몫이라고 본다. 평균 맞추기를 없애면 다음과 같다.

1. **계산을 줄인다**: 정규화마다 축약 연산이 하나 줄어든다
2. **기울기를 간단하게 한다**: 역전파의 항이 줄어든다
3. **경험으로 보아 같다**: 모형의 질이 떨어지지 않는다

### PyTorch 구현

```python
class RMSNorm(nn.Module):
    """
    제곱평균제곱근 층 정규화.
    
    LLaMA, Mistral을 비롯한 요즘 대형 언어 모형이 쓴다.
    층 정규화보다 간단하고 빠르면서 성능은 같다.
    """
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        인수:
            x: 입력 텐서 [..., d_model]
        반환값:
            정규화된 텐서 [..., d_model]
        """
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.gamma * x / rms
```

### LayerNorm과의 비교

| 측면 | 층 정규화 | RMSNorm |
|--------|-----------|---------|
| 평균 빼기 | 한다 | 안 한다 |
| 학습되는 옮김 ($\beta$) | 있다 | 없다 |
| 매개변수 | $2d$ ($\gamma, \beta$) | $d$ ($\gamma$) |
| 계산 | 축약 2번 (평균, 분산) | 축약 1번 (제곱평균제곱근) |
| 쓰이는 곳 | BERT, GPT-2, T5 | LLaMA, Mistral, Gemma |
| 성능 | 기준 | 같다 |

---

## 4. 구조마다의 정규화 자리

| 모형 | 정규화 종류 | 자리 |
|-------|-----------|-----------|
| 본디 트랜스포머 | 층 정규화 | 뒤 정규화 |
| BERT | 층 정규화 | 뒤 정규화 |
| GPT-2 | 층 정규화 | 앞 정규화 |
| T5 v1.0 | 층 정규화 | 앞 정규화 |
| T5 v1.1 | RMSNorm | 앞 정규화 |
| LLaMA | RMSNorm | 앞 정규화 |
| Mistral | RMSNorm | 앞 정규화 |

이 분야는 새로운 큰 규모 모형의 기본으로 RMSNorm을 쓰는 앞 정규화로 모여 왔다.

---

## 5. 실용적인 고려

### 가중치 감쇠와 정규화

정규화 매개변수($\gamma$, $\beta$)는 대개 가중치 감쇠에서 뺀다.

```python
# 최적화기를 위해 매개변수를 가른다
decay_params = []
no_decay_params = []
for name, param in model.named_parameters():
    if 'norm' in name or 'bias' in name:
        no_decay_params.append(param)
    else:
        decay_params.append(param)

optimizer = torch.optim.AdamW([
    {'params': decay_params, 'weight_decay': 0.1},
    {'params': no_decay_params, 'weight_decay': 0.0}
], lr=3e-4)
```

### 수치적 안정성

섞인 정밀도 학습(FP16/BF16)을 할 때 정규화 층은 수치 안정성을 지키려고 대개 FP32으로 둔다.

```python
# 섞인 정밀도 학습에서 정규화는 FP32으로 셈한다
class StableLayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=eps)
    
    def forward(self, x):
        # 정규화를 위해 FP32으로 올린다
        return self.norm(x.float()).type_as(x)
```

---

## 연습문제

**연습문제 1.**
통계를 특징 차원에서 셈하는 층 정규화 $\text{LN}(x) = \gamma \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$을 이끌어 내라.

??? success "연습문제 1 풀이"
    입력 $x \in \mathbb{R}^d$에 대해 $\mu = \frac{1}{d}\sum_i x_i$, $\sigma^2 = \frac{1}{d}\sum_i(x_i-\mu)^2$이다. 층 정규화는 배치에 걸쳐 정규화하는 배치 정규화와 달리 토큰마다 따로 특징에 걸쳐 정규화한다. 그래서 길이가 제각각인 수열과 작은 배치 크기에 알맞다.

---

**연습문제 2.**
앞 정규화 트랜스포머 구조와 뒤 정규화 트랜스포머 구조를 견주어라.

??? success "연습문제 2 풀이"
    뒤 정규화(본디 방식)는 $x + \text{LN}(\text{SubLayer}(x))$이고 앞 정규화는 $x + \text{SubLayer}(\text{LN}(x))$이다. 앞 정규화는 잔차 길이 깔끔하게 남으므로(건너뛰기에 정규화가 없다) 깊은 모형에 더 안정적이다. 뒤 정규화는 학습률 예열을 조심스레 하면 마지막 성능이 더 나을 수 있다.

---

**연습문제 3.**
트랜스포머에서 배치 정규화보다 층 정규화를 더 좋아하는 까닭은 무엇인가?

??? success "연습문제 3 풀이"
    배치 정규화는 배치 차원에 걸쳐 통계를 셈하는데 이는 (1) 배치 크기에 따라 달라지고, (2) 길이가 제각각인 수열에 알맞지 않으며(채움이 통계를 흔든다), (3) 추론을 위해 누적 통계가 필요하다. 층 정규화는 토큰마다 통계를 셈하여 이 문제를 모두 피한다.

---

**연습문제 4.**
층 정규화를 맨바닥부터 구현하고 `nn.LayerNorm`과 맞는지 확인하라.

??? success "연습문제 4 풀이"
    ```python
    def layer_norm(x, gamma, beta, eps=1e-5):
        mu = x.mean(dim=-1, keepdim=True)
        sigma = x.std(dim=-1, keepdim=True, unbiased=False)
        return gamma * (x - mu) / (sigma + eps) + beta
    ```

## 정리하며

트랜스포머의 층 정규화에는 핵심 결정이 셋 있다.

1. **정규화 종류**: 층 정규화(전통)와 RMSNorm(요즘, 더 빠르고 질은 같다)
2. **자리**: 뒤 정규화(본디 방식, 최고 성능이 조금 낫지만 학습이 어렵다)와 앞 정규화(요즘의 표준, 학습이 안정적이고 깊은 모형을 가능케 한다)
3. **마지막 정규화**: 앞 정규화 구조에서 출력 사영 앞에 필요하다

요즘의 모범은 LLaMA, Mistral 같은 최고 수준 모형이 쓰는 **앞 정규화 RMSNorm**이다.

**참고 문헌**

1. Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). "Layer Normalization." arXiv.
2. Xiong, R., et al. (2020). "On Layer Normalization in the Transformer Architecture." ICML.
3. Zhang, B., & Sennrich, R. (2019). "Root Mean Square Layer Normalization." NeurIPS.
4. Vaswani, A., et al. (2017). "Attention Is All You Need." NeurIPS.
