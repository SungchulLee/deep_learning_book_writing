# 트랜스포머의 LayerNorm

LayerNorm은 각 토큰의 피처 축에서 통계를 계산해 표현의 스케일을 조절한다. 잔차 연결 앞뒤의 배치에 따라 그래디언트 경로가 달라지며, 평균을 빼지 않는 RMSNorm도 대안으로 쓰인다. 정규화의 위치와 종류만으로 학습 안정성이나 최종 성능을 보장할 수는 없다.

## 1. LayerNorm

### 수식으로 나타내기

LayerNorm은 토큰마다 따로 피처 차원에 걸쳐 정규화한다.

$$
\text{LayerNorm}(\mathbf{x}) = \gamma \odot \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

여기서:

- $\mu = \frac{1}{d} \sum_{i=1}^{d} x_i$은 피처에 걸친 평균이다
- $\sigma^2 = \frac{1}{d} \sum_{i=1}^{d} (x_i - \mu)^2$은 피처에 걸친 분산이다
- $\gamma, \beta \in \mathbb{R}^d$은 학습되는 크기 조정과 이동 파라메터이다
- $\epsilon$은 수치 안정성을 위한 작은 상수이다(대개 $10^{-5}$이나 $10^{-6}$)

### 왜 (배치 정규화가 아니라) LayerNorm인가

배치 정규화는 지정한 피처 축을 제외한 축의 통계를 사용하므로 입력 배치와 배치 구성의 영향을 받는다. 시퀀스에 적용할 때는 축과 패딩의 영향을 주의해야 한다. 평가 시에는 기본적으로 저장된 이동 통계를 쓰므로 배치 크기가 1이라는 이유만으로 사용할 수 없는 것은 아니다.

LayerNorm(d)는 각 토큰의 d개 피처를 따로 정규화한다. 다른 표본과 위치의 통계가 섞이지 않고, 학습·평가에서 같은 계산을 사용하므로 가변 길이 입력과 자기 회귀 모델에 연결하기 편하다. 패딩의 어텐션·로스 마스킹은 별도로 필요하다.

이 비교는 일반적인 트랜스포머의 토큰별 LayerNorm 설정에 대한 설명이다. 모든 뉴럴 넷트웍에서 배치 정규화를 사용할 수 없다는 뜻은 아니다.

### PyTorch 구현

```python
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    """
    트랜스포머에서 쓰는 레이어 정규화.
    
    마지막 차원(피처 차원)에 걸쳐 정규화한다.
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

## 2. Pre-LN와 Post-LN

LayerNorm을 하위 레이어에 비교하여 어디에 두느냐가 학습의 안정성과 모델의 질에 큰 영향을 준다.

### Post-LN (원래 트랜스포머)

원래 "Attention Is All You Need" 논문은 잔차를 더한 뒤에 정규화를 둔다.

$$
\mathbf{x}' = \text{LayerNorm}(\mathbf{x} + \text{SubLayer}(\mathbf{x}))
$$

**성질:**

- 잔차를 더한 결과가 정규화를 통과한다.
- 역전파의 잔차 경로에도 정규화의 야코비안이 들어간다.
- 초기화·러닝레이트에 따라 워밍업이 안정적인 학습에 도움이 될 수 있다.
- Pre-LN와의 성능 우열은 모델과 학습 조건에 따라 다르다.

### Pre-LN (요즘의 표준)

GPT-2와 T5처럼 하위 레이어의 입력에 정규화를 적용하는 구조가 있다. 아래 식은 LayerNorm 표기이며 실제 모델은 RMS 기반 정규화를 쓸 수도 있다.

$$
\mathbf{x}' = \mathbf{x} + \text{SubLayer}(\text{LayerNorm}(\mathbf{x}))
$$

**성질:**

- 잔차 연결의 항등 경로가 정규화 바깥에 남는다.
- 깊은 모델의 초기 그래디언트 흐름에 유리할 수 있다.
- 워밍업이 불필요하다는 보장은 없으며 러닝레이트·초기화와 함께 결정한다.
- “12레이어 이상” 같은 보편적인 전환 기준은 없다.

### 그래디언트 흐름 분석

Pre-LN 블록을 $x_{l+1}=x_l+F_l(N_l(x_l))$로 쓰면 입력에 대한 야코비안은 다음과 같다.

$$
J_l=I+J_{F_l}(N_l(x_l))J_{N_l}(x_l)

여러 블록의 역전파에는 이 야코비안들의 전치 곱이 나타난다. 각 블록에 항등 항이 있어도 다른 항과의 상쇄·확대가 가능하므로 그래디언트의 크기가 항상 유지되는 것은 아니다. Post-LN 블록의 야코비안은 정규화의 야코비안을 잔차 합 바깥에 곱한 형태다.

### 파이토치 구현 비교

```python
import torch
import torch.nn as nn
from typing import Optional, Tuple

class PostNormTransformerBlock(nn.Module):
    """뒤 정규화: 잔차를 더한 뒤의 레이어 정규화(본디 트랜스포머)."""
    
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
    """앞 정규화: 아래 레이어 앞의 레이어 정규화(요즘의 표준)."""
    
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
        # 앞 정규화: 아래 레이어 앞에 정규화한다
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

### 마지막 LayerNorm

Pre-LN 구조에서는 블록 스택 뒤, 출력 투영 앞에 마지막 정규화를 두는 경우가 많다. 각 블록의 잔차 출력 자체는 정규화하지 않으므로 마지막 표현의 스케일을 조절하는 역할이다. 이는 구조의 선택이며 모든 모델에서의 수학적 필수 조건은 아니다.

```
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

Post-LN는 각 블록의 잔차 합 뒤에 이미 정규화가 있어 추가 마지막 정규화를 생략하는 구성이 일반적이다.

---

## 3. RMSNorm

제곱평균제곱근 LayerNorm(RMSNorm)는 평균 맞추기 단계를 없애 LayerNorm을 간단하게 한 것으로 LLaMA, Mistral을 비롯한 요즘 대형 언어 모델에 쓰인다.

$$
\text{RMSNorm}(\mathbf{x}) = \gamma \odot \frac{\mathbf{x}}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2+\epsilon}}
$$

여기서:

$$
\text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2}
$$

### 동기

RMSNorm은 평균을 빼지 않고 제곱평균으로 스케일을 조절한다. 아래 구현은 epsilon을 제곱근 안에 더하며 LayerNorm과 다르게 평행 이동에 불변하지 않다. 평균 빼기가 모든 과제에서 불필요하다는 결론으로 일반화해서는 안 된다.

학습 가능한 배율만 사용하는 기본형은 피처 차원 d에 대해 d개의 파라메터를 갖는다. LayerNorm의 배율·이동 기본형은 2d개다. 연산과 메모리 접근을 줄일 여지가 있지만 실제 속도와 예측 성능은 커널·장치·모델 설정에 따라 비교해야 한다.

### PyTorch 구현

```python
class RMSNorm(nn.Module):
    """
    제곱평균제곱근 레이어 정규화.
    
    LLaMA, Mistral을 비롯한 요즘 대형 언어 모형이 쓴다.
    레이어 정규화보다 간단하고 빠르면서 성능은 같다.
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

| 측면 | LayerNorm 기본형 | RMSNorm 기본형 |
|---|---|---|
| 평균 빼기 | 적용 | 생략 |
| 정규화 기준 | 평균으로 중심화한 분산 | 원소 제곱의 평균 |
| 학습 계수 | 배율·이동, $2d$개 | 배율, $d$개 |
| epsilon | 분산에 더한 뒤 제곱근 | 제곱평균에 더한 뒤 제곱근 |
| 속도·성능 | 실험 조건에 따라 비교 | 실험 조건에 따라 비교 |

---

## 4. 구조마다의 정규화 자리

| 구조 | 정규화 | 배치 |
|---|---|---|
| 원래 트랜스포머 | LayerNorm | Post-LN |
| BERT | LayerNorm | Post-LN |
| GPT-2 | LayerNorm | Pre-LN |
| T5 | 평균을 빼지 않는 RMS 기반 정규화 | Pre-LN |
| LLaMA | RMSNorm | Pre-LN |

T5의 정규화는 이름이 LayerNorm으로 표시되더라도 일반 LayerNorm처럼 평균을 빼거나 바이어스를 더하지 않는다. 모델 이름만으로 모든 변형의 세부 설정을 단정하지 말고 [실제 정규화 구현](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py)을 확인한다.

## 5. 실용적인 고려

### 웨이트 감쇠와 정규화

정규화의 배율·이동을 웨이트 감쇠에서 제외하는 설정이 흔하지만 필수 규칙은 아니다. 아래 조각은 외부의 model을 가정하며 이름에 norm 또는 bias가 들어가는지를 기준으로 분류한다. 실제 모듈 종류를 확인하지 않으므로 의도한 모든 정규화 계수를 정확히 분리한다고 보장할 수 없다.

```python
# 최적화기를 위해 파라메터를 가른다
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

혼합 정밀도에서는 통계량 계산을 FP32로 수행하고 출력을 원래 자료형으로 돌려 수치 오차를 줄일 수 있다. 아래 StableLayerNorm은 입력만 float로 바꾼다. 모델 전체를 half로 변환해 정규화 계수도 FP16이 되면 지원 여부와 자료형 호환성을 따로 확인해야 한다. FP32 누산과 모든 파라메터를 FP32로 보관하는 것은 다르다.

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

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
토큰별 LayerNorm의 평균·분산과 학습 계수를 설명하라.

</div>

??? success "풀이"
    피처 d개에서 평균과 $d$로 나눈 중심 제곱 편차를 계산하고 $\sqrt{\sigma^2+\epsilon}$으로 나눈다. 이어 피처별 배율·이동을 적용한다. epsilon과 학습 계수가 있어 최종 출력의 분산이 항상 정확히 1인 것은 아니다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
Post-LN과 Pre-LN의 잔차·정규화 순서를 식으로 비교하라.

</div>

??? success "풀이"
    Post-LN은 $N(x+F(x))$, Pre-LN은 $x+F(N(x))$다. 후자는 항등 잔차 경로가 정규화 바깥에 남는다. 이는 그래디언트 경로의 차이이며 Pre-LN의 워밍업 불필요나 최종 성능 우위를 보장하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
작은 배치와 패딩이 있는 입력에서 LayerNorm의 장점과 한계를 설명하라.

</div>

??? success "풀이"
    다른 표본이나 토큰의 통계를 사용하지 않고 학습·평가가 같은 계산이라 편리하다. 다만 패딩 토큰도 정규화되므로 어텐션과 로스의 패딩 마스크를 대체하지 않는다. BatchNorm도 평가 모드에서는 이동 통계를 사용할 수 있다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 풀이가 `nn.LayerNorm`과 정확히 같은 식인지 검토하라. 특히 epsilon의 위치와 분산의 정의를 비교하라.

</div>

??? success "풀이"
    아래 조각은 표준편차에 epsilon을 더하므로 분산에 더한 뒤 제곱근을 취하는 nn.LayerNorm과 다르다. unbiased=False는 모집단 분산 기준과 맞지만 epsilon 위치도 맞춰야 한다. 같은 계수·epsilon·자료형에서 비교해야 한다.

    ```python
    def layer_norm(x, gamma, beta, eps=1e-5):
        mu = x.mean(dim=-1, keepdim=True)
        sigma = x.std(dim=-1, keepdim=True, unbiased=False)
        return gamma * (x - mu) / (sigma + eps) + beta
    ```

## 정리하며

정규화의 종류, 잔차 연결에 대한 위치와 통계 계산의 자료형을 함께 확인한다. Pre-LN의 항등 경로와 RMSNorm의 연산 단순화는 유용한 설계 근거지만 모델 품질·속도의 보편적인 우열을 뜻하지 않는다.

**참고 문헌**

1. Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). "Layer Normalization." arXiv.
2. Xiong, R., et al. (2020). "On Layer Normalization in the Transformer Architecture." ICML.
3. Zhang, B., & Sennrich, R. (2019). "Root Mean Square Layer Normalization." NeurIPS.
4. Vaswani, A., et al. (2017). "Attention Is All You Need." NeurIPS.
