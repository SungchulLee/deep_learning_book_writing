# 가린 자기 주의
## 들어가며

가린 자기 주의는 트랜스포머 디코더에서 자기 회귀 생성을 가능케 하는 얼개이다. 자리마다 앞으로의 자리에 주의하지 못하게 막아 왼쪽에서 오른쪽으로의 수열 생성에 필요한 인과 짜임을 지킨다.

## 인과성이라는 요구

언어 모형화에서는 앞선 토큰을 모두 주고 다음 토큰을 맞힌다.

$$
P(x_1, x_2, \ldots, x_n) = \prod_{t=1}^{n} P(x_t | x_1, \ldots, x_{t-1})
$$

이 인수분해는 $x_t$을 맞힐 때 $x_1, \ldots, x_{t-1}$의 정보만 쓸 것을 요구한다. 가린 주의가 병렬 학습 중에 이 제약을 지키게 한다.

## 수학적 정식화

### 표준 자기 주의

가림이 없으면 주의는 자리마다 모든 자리에 주의하게 한다.

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

### 인과(가린) 자기 주의

소프트맥스에 앞서 앞으로의 자리를 $-\infty$으로 두는 가림 $M$을 더한다.

$$
\text{MaskedAttention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V
$$

여기서 가림 $M \in \mathbb{R}^{n \times n}$은 다음과 같다.

$$
M_{ij} = \begin{cases} 0 & \text{if } j \leq i \text{ (can attend)} \\ -\infty & \text{if } j > i \text{ (masked)} \end{cases}
$$

### 주의 가중치에 미치는 영향

$-\infty$ 가림과 함께 소프트맥스를 적용하면 다음과 같다.

$$
\text{softmax}(-\infty) = 0
$$

이는 앞으로의 자리에 대한 주의 가중치를 0으로 만든다.

$$
\alpha_{ij} = \begin{cases} \frac{\exp(s_{ij})}{\sum_{k \leq i} \exp(s_{ik})} & \text{if } j \leq i \\ 0 & \text{if } j > i \end{cases}
$$

## 가림 만들기

### 아래 삼각 가림

```python
def create_causal_mask(seq_len: int, device: torch.device = None) -> torch.Tensor:
    """
    인과 주의 가림을 만든다.
    
    mask[i,j] = True이면 자리 i가 j에 주의할 수 **없다**는 뜻인 가림을 돌려준다.
    
    seq_len=4일 때의 보기:
    [[False,  True,  True,  True],
     [False, False,  True,  True],
     [False, False, False,  True],
     [False, False, False, False]]
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    return mask

def create_causal_mask_float(seq_len: int, device: torch.device = None) -> torch.Tensor:
    """
    가린 자리를 -inf로 둔 인과 가림을 만든다.
    
    주의 점수에 곧바로 더할 수 있는 가림을 돌려준다.
    """
    mask = torch.triu(
        torch.full((seq_len, seq_len), float('-inf'), device=device),
        diagonal=1
    )
    return mask
```

### 눈으로 보기

```
Attention Pattern for Position 4:
                    Position
              1    2    3    4    5
            ┌───┬───┬───┬───┬───┐
Position 4  │ ✓ │ ✓ │ ✓ │ ✓ │ ✗ │
            └───┴───┴───┴───┴───┘
              ↑                ↑
           Attend          Masked
```

## PyTorch 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class MaskedSelfAttention(nn.Module):
    """
    가린(인과) 자기 주의 얼개.
    
    자리마다 제 자신과 앞선 자리에만 주의할 수 있는
    자기 회귀 주의를 구현한다.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        max_len: int = 2048
    ):
        """
        인수:
            d_model: 모형 차원
            num_heads: 주의 머리의 수
            dropout: 주의 드롭아웃 확률
            max_len: 미리 셈해 둘 가림의 최대 수열 길이
        """
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        
        # 선형 사영
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # 드롭아웃
        self.dropout = nn.Dropout(dropout)
        
        # 인과 가림을 미리 셈해 등록한다
        mask = torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()
        self.register_buffer('causal_mask', mask)
    
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        텐서를 여러 머리로 나눈다.
        
        입력: [batch, seq_len, d_model]
        출력: [batch, num_heads, seq_len, head_dim]
        """
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)
    
    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        머리를 다시 텐서 하나로 합친다.
        
        입력: [batch, num_heads, seq_len, head_dim]
        출력: [batch, seq_len, d_model]
        """
        batch_size, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        인과 가림을 쓰는 앞먹임.
        
        인수:
            x: 입력 텐서 [batch_size, seq_len, d_model]
            padding_mask: 선택으로 주는 채움 가림 [batch_size, seq_len]
                가릴 자리(채움 토큰)가 True
            return_attention: 어텐션 가중치를 돌려줄지 여부
            
        반환값:
            output: 바뀐 텐서 [batch_size, seq_len, d_model]
            attention_weights: 선택 [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.shape
        
        # 1단계: 선형 사영
        Q = self.W_q(x)  # [batch, seq, d_model]
        K = self.W_k(x)
        V = self.W_v(x)
        
        # 2단계: 머리로 나눈다
        Q = self._split_heads(Q)  # [batch, heads, seq, head_dim]
        K = self._split_heads(K)
        V = self._split_heads(V)
        
        # 3단계: 스케일 조정 내적 주의
        # [batch, heads, seq, seq]
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # 4단계: 인과 가림을 적용한다 (앞으로의 자리에 주의하지 못하게)
        causal_mask = self.causal_mask[:seq_len, :seq_len]
        attention_scores = attention_scores.masked_fill(
            causal_mask.unsqueeze(0).unsqueeze(0),
            float('-inf')
        )
        
        # 5단계: 채움 가림을 적용한다 (선택)
        if padding_mask is not None:
            # padding_mask: [batch, seq] -> [batch, 1, 1, seq]
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
            attention_scores = attention_scores.masked_fill(
                padding_mask,
                float('-inf')
            )
        
        # 6단계: 소프트맥스
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # 모두 가려진 행에서 나오는 NaN을 다룬다 (인과 가림에서는 생기지 않아야 한다)
        attention_weights = torch.nan_to_num(attention_weights, 0.0)
        
        # 7단계: 드롭아웃
        attention_weights = self.dropout(attention_weights)
        
        # 8단계: 값에 주의를 적용한다
        # [batch, heads, seq, head_dim]
        context = torch.matmul(attention_weights, V)
        
        # 9단계: 머리를 합치고 마지막 사영을 한다
        context = self._merge_heads(context)  # [batch, seq, d_model]
        output = self.W_o(context)
        
        if return_attention:
            return output, attention_weights
        return output, None

def visualize_causal_attention(seq_len: int = 10):
    """인과 주의 무늬를 그려 본다."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 보기용 주의 가중치를 만든다 (허락된 자리 안에서 고르게)
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    attention = torch.ones(seq_len, seq_len)
    attention = attention.masked_fill(mask, 0.0)
    
    # 행을 정규화한다
    attention = attention / attention.sum(dim=-1, keepdim=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 인과 가림
    ax1 = axes[0]
    im1 = ax1.imshow(mask.float().numpy(), cmap='Reds')
    ax1.set_xlabel('Key Position (j)')
    ax1.set_ylabel('Query Position (i)')
    ax1.set_title('Causal Mask (Red = Masked)')
    ax1.set_xticks(range(seq_len))
    ax1.set_yticks(range(seq_len))
    plt.colorbar(im1, ax=ax1)
    
    # 주의 무늬
    ax2 = axes[1]
    im2 = ax2.imshow(attention.numpy(), cmap='Blues')
    ax2.set_xlabel('Key Position (j)')
    ax2.set_ylabel('Query Position (i)')
    ax2.set_title('Attention Pattern (After Masking)')
    ax2.set_xticks(range(seq_len))
    ax2.set_yticks(range(seq_len))
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig('causal_attention_pattern.png', dpi=150)
    plt.close()

# 사용 예
if __name__ == "__main__":
    # 설정
    d_model = 512
    num_heads = 8
    batch_size = 4
    seq_len = 20
    
    # 모듈을 만든다
    masked_attn = MaskedSelfAttention(
        d_model=d_model,
        num_heads=num_heads,
        dropout=0.1
    )
    
    # 시험 입력
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 순전파
    output, attn_weights = masked_attn(x, return_attention=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    
    # 인과성을 확인한다: 주의 가중치가 아래 삼각이어야 한다
    # 위 삼각(대각선 제외)이 모두 0인지 살핀다
    for b in range(batch_size):
        for h in range(num_heads):
            upper = torch.triu(attn_weights[b, h], diagonal=1)
            assert torch.allclose(upper, torch.zeros_like(upper), atol=1e-6), \
                "Attention weights should be zero for future positions!"
    
    print("\n✓ Causality verified: No attention to future positions")
    
    # 첫 머리의 주의 무늬를 보인다
    print("\nAttention pattern (first batch, first head):")
    pattern = attn_weights[0, 0].detach()
    print(f"Sum of each row (should be 1.0): {pattern.sum(dim=-1)[:5].tolist()}")
    
    # 시각화한다
    visualize_causal_attention(10)
    print("\nVisualization saved to 'causal_attention_pattern.png'")
```

## 인과 가림과 채움 가림 섞기

실제로는 인과 가림과 채움 가림을 자주 함께 쓴다.

```python
def create_combined_mask(
    seq_len: int,
    padding_mask: torch.Tensor,
    device: torch.device = None
) -> torch.Tensor:
    """
    인과 가림과 채움 가림을 합쳐 만든다.
    
    인수:
        seq_len: 수열 길이
        padding_mask: [batch, seq]. 채운 자리가 True
        device: 가림 텐서를 둘 장치
        
    반환값:
        합친 가림 [batch, 1, seq, seq]
    """
    # 인과 가림: [seq, seq]
    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, device=device),
        diagonal=1
    ).bool()
    
    # [1, 1, seq, seq]로 넓힌다
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
    
    # 채움 가림: [batch, seq] -> [batch, 1, 1, seq]
    padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
    
    # 합친다: 둘 중 하나라도 참이면 가린다
    combined_mask = causal_mask | padding_mask
    
    return combined_mask
```

## 플래시 주의로 효율적으로 구현하기

긴 수열에서는 플래시 주의가 기억을 아끼는 인과 주의를 준다.

```python
# 파이토치의 scaled_dot_product_attention에 is_causal=True를 쓴다
def efficient_causal_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dropout_p: float = 0.0
) -> torch.Tensor:
    """
    파이토치의 SDPA를 쓰는 효율적인 인과 주의.
    
    인수:
        query, key, value: [batch, heads, seq, head_dim]
        dropout_p: 드롭아웃 확률
        
    반환값:
        출력 텐서 [batch, heads, seq, head_dim]
    """
    return F.scaled_dot_product_attention(
        query, key, value,
        attn_mask=None,
        dropout_p=dropout_p,
        is_causal=True  # 인과 가림을 저절로 적용한다
    )
```

## 접두 언어 모형: 양방향 접두와 인과 접미

어떤 모형(T5 디코더, UL2 같은)은 섞은 방식을 쓴다.

```python
def create_prefix_lm_mask(
    prefix_len: int,
    total_len: int,
    device: torch.device = None
) -> torch.Tensor:
    """
    접두 언어 모형 주의 가림을 만든다.
    
    접두 자리(1부터 prefix_len까지)는 서로 양방향으로 주의할 수 있다.
    접미 자리는 접두와 앞선 접미 자리에 주의할 수 있다.
    
    인수:
        prefix_len: 양방향 접두의 길이
        total_len: 수열 전체의 길이
        
    반환값:
        가림 텐서 [total_len, total_len]
    """
    mask = torch.zeros(total_len, total_len, device=device)
    
    # 접미 자리는 앞으로의 접미에 주의할 수 없다
    suffix_mask = torch.triu(
        torch.ones(total_len - prefix_len, total_len - prefix_len, device=device),
        diagonal=1
    )
    mask[prefix_len:, prefix_len:] = suffix_mask
    
    return mask.bool()

# 보기: 토큰 5개짜리 접두를 가진 접두 언어 모형
# 자리 1~5: 양방향 (서로 볼 수 있다)
# 자리 6 이상: 인과 (접두와 앞선 접미를 볼 수 있다)
```

## 미끄러지는 창 주의

긴 수열에서 효율을 위해 미끄러지는 창은 주의를 가까운 자리로 제한한다.

```python
def create_sliding_window_causal_mask(
    seq_len: int,
    window_size: int,
    device: torch.device = None
) -> torch.Tensor:
    """
    미끄러지는 창 인과 주의 가림을 만든다.
    
    자리마다 많아야 window_size개의 앞선 자리에 주의할 수 있다.
    
    인수:
        seq_len: 수열 길이
        window_size: 주의 창의 크기
        
    반환값:
        가림 텐서 [seq_len, seq_len]
    """
    # 인과 가림으로 시작한다
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    
    # 창 바깥의 자리도 가린다
    for i in range(seq_len):
        start = max(0, i - window_size + 1)
        mask[i, :start] = 1
    
    return mask.bool()
```

## 학습과 추론

### 학습 (병렬)

학습 중에는 인과 가림과 함께 모든 자리를 병렬로 셈한다.

```python
# 모든 자리를 한꺼번에 셈한다
output = masked_attention(input_sequence)  # [batch, seq_len, d_model]
```

### 추론 (자기 회귀)

생성 중에는 토큰을 하나씩 만든다.

```python
# 토큰을 하나씩 만든다
for t in range(max_tokens):
    # 마지막 자리만 셈한다
    logits = model(generated_tokens)[:, -1, :]
    next_token = sample(logits)
    generated_tokens = torch.cat([generated_tokens, next_token], dim=1)
```

KV 캐싱을 쓰면 앞선 자리의 주의를 다시 셈하지 않아도 된다.

## 요약

가린 자기 주의는 자기 회귀 모형에 꼭 필요하다.

1. **인과성을 지킨다**: 앞에서 뒤로의 정보 흐름을 막는다
2. **병렬 학습을 가능케 한다**: 모든 자리를 한꺼번에 셈한다
3. **생성 순서를 지킨다**: 토큰을 왼쪽에서 오른쪽으로 만든다
4. **다른 가림과 어울린다**: 채움, 미끄러지는 창 따위와 함께 쓸 수 있다

가린 주의를 이해하는 것은 언어 모형과 글 생성 체계를 구현하는 데 매우 중요하다.

## 참고 문헌

1. Vaswani, A., et al. (2017). "Attention Is All You Need." NeurIPS.
2. Radford, A., et al. (2019). "Language Models are Unsupervised Multitask Learners."
3. Dao, T., et al. (2022). "FlashAttention: Fast and Memory-Efficient Attention."
4. Beltagy, I., et al. (2020). "Longformer: The Long-Document Transformer."

## 연습문제

**연습문제 1.**
자기 회귀 언어 모형에 인과 가림이 왜 필요한지 설명하라.

??? success "연습문제 1 풀이"
    자기 회귀 생성에서 모형은 $P(x_t|x_{<t})$을 맞힌다. 학습 중에 모형이 앞으로의 토큰에 주의할 수 있으면 답을 맞히는 대신 '베끼기'를 배우게 된다. 인과 가림은 주의 행렬의 위 삼각을 $-\infty$으로 채워 자리마다 자신과 앞선 자리에만 주의하게 한다.

---

**연습문제 2.**
파이토치에서 효율적인 인과 주의 가림을 구현하라.

??? success "연습문제 2 풀이"
    ```python
    def causal_mask(n):
        return torch.triu(torch.ones(n, n), diagonal=1).bool()  # True면 가린다

    # 주의에서: scores.masked_fill_(mask, float('-inf'))
    ```

---

**연습문제 3.**
인과 가림과 채움 가림을 견주어라. 각각 언제 쓰는가?

??? success "연습문제 3 풀이"
    인과 가림은 위 삼각 꼴로 앞으로의 자리에 주의하지 못하게 막는다. 디코더와 자기 회귀 모형에 쓴다. 채움 가림은 (길이가 제각각인 수열에서) 채운 자리를 표시한다. 인코더와 디코더 모두에 쓴다. `mask = causal_mask | padding_mask`처럼 함께 쓸 수 있다.

---

**연습문제 4.**
KV 캐싱이 인과 가림을 어떻게 이용해 자기 회귀 추론을 빠르게 하는지 설명하라.

??? success "연습문제 4 풀이"
    생성 중에 새 토큰마다 앞선 토큰 모두에만 주의하면 된다. KV 캐시를 쓰면 앞 단계의 열쇠와 값 사영을 담아 둔다. 새 토큰에 대해서는 그 질의만 셈해 담아 둔 K, V에 주의한다. 그러면 단계마다의 비용이 $O(n^2)$에서 $O(n)$으로 줄고 앞선 토큰의 표현을 다시 셈하지 않아도 된다.
