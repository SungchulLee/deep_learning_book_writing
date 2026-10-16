# 마스크 셀프 어텐션

인과 마스크는 각 위치가 미래 입력 토큰을 참조하지 못하게 한다. 이를 통해 다음 토큰 예측을 학습할 때 여러 위치의 로스를 병렬로 계산할 수 있다. 패딩 마스크와 결합할 때는 마스크의 True가 허용인지 차단인지 먼저 확인한다.

## 1. 인과성이라는 요구

언어 모델링에서는 앞선 토큰을 모두 주고 다음 토큰을 맞힌다.

$$
P(x_1, x_2, \ldots, x_n) = \prod_{t=1}^{n} P(x_t | x_1, \ldots, x_{t-1})
$$

이 인수분해는 $x_t$을 맞힐 때 $x_1, \ldots, x_{t-1}$의 정보만 쓸 것을 요구한다. 마스크 어텐션가 병렬 학습 중에 이 제약을 지키게 한다.

---

## 2. 수학적 정식화

### 표준 셀프 어텐션

마스크가 없으면 주의는 위치마다 모든 자리에 주의하게 한다.

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

### 인과(가린) 셀프 어텐션

소프트맥스에 앞서 앞으로의 자리를 $-\infty$으로 두는 마스크 $M$을 더한다.

$$
\text{MaskedAttention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V
$$

여기서 마스크 $M \in \mathbb{R}^{n \times n}$은 다음과 같다.

$$
M_{ij} = \begin{cases} 0 & \text{if } j \leq i \text{ (can attend)} \\ -\infty & \text{if } j > i \text{ (masked)} \end{cases}
$$

### 어텐션 웨이트에 미치는 영향

$-\infty$ 마스크과 함께 소프트맥스를 적용하면 다음과 같다.

$$
\text{softmax}(-\infty) = 0
$$

같은 행에 적어도 하나의 유한한 점수가 있을 때 -inf인 위치의 웨이트는 0이다. 모든 키가 가려진 행은 일반 소프트맥스에서 NaN이 될 수 있다. 위 식은 단일 스칼라에 소프트맥스를 적용한다는 뜻이 아니라 벡터 안의 마스킹된 성분을 나타낸다.

$$
\alpha_{ij} = \begin{cases} \frac{\exp(s_{ij})}{\sum_{k \leq i} \exp(s_{ik})} & \text{if } j \leq i \\ 0 & \text{if } j > i \end{cases}
$$

---

## 3. 마스크 만들기

### 아래 삼각 마스크

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

---

## 4. PyTorch 구현

이 클래스의 불리언 마스크는 True가 차단을 뜻한다. 입력은 (B,L,E), padding_mask는 (B,L)이고 저장된 max_len 이내여야 한다. 패딩 마스크는 키 위치만 제외하며 패딩 쿼리의 출력을 자동으로 0으로 만들지 않는다.

모든 키가 가려진 행의 NaN 웨이트를 0으로 치환하지만 출력 투영의 바이어스 때문에 최종 출력이 반드시 0인 것은 아니다. 반환 웨이트는 드롭아웃 뒤 값이어서 학습 모드에서는 행 합이 정확히 1이 아닐 수 있다. 예제의 행 합 출력도 이 조건으로 읽는다.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class MaskedSelfAttention(nn.Module):
    """
    가린(인과) 자기 주의 구조.
    
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
        인과 가림을 쓰는 피드포워드.
        
        인수:
            x: 입력 텐서 [batch_size, seq_len, d_model]
            padding_mask: 선택으로 주는 채움 가림 [batch_size, seq_len]
                가릴 자리(채움 토큰)가 True
            return_attention: 어텐션 웨이트를 돌려줄지 여부
            
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
    
    # 보기용 주의 웨이트를 만든다 (허락된 자리 안에서 고르게)
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
    
    # 인과성을 확인한다: 주의 웨이트가 아래 삼각이어야 한다
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

**출력:**

```
Input shape: torch.Size([4, 20, 512])
Output shape: torch.Size([4, 20, 512])
Attention weights shape: torch.Size([4, 8, 20, 20])

✓ Causality verified: No attention to future positions

Attention pattern (first batch, first head):
Sum of each row (should be 1.0): [1.1111111640930176, 1.1111111640930176, 1.1111111640930176, 0.8720628023147583, 0.9415875673294067]

Visualization saved to 'causal_attention_pattern.png'
```

---

## 5. 인과 마스크과 패딩 마스크 섞기

아래 함수는 True가 차단인 두 불리언 마스크를 OR로 합쳐 (B,1,L,L)을 반환한다. 마스크를 생성하는 장치와 padding_mask의 장치가 같아야 한다.

실제로는 인과 마스크과 패딩 마스크를 자주 함께 쓴다.

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

---

## 6. FlashAttention로 효율적으로 구현하기

scaled_dot_product_attention은 조건에 따라 메모리 효율 커널을 선택한다. 아래 함수가 항상 FlashAttention을 사용한다고 단정할 수는 없다. 함수형 API의 dropout_p는 모델의 eval 모드와 무관하게 적용되므로 평가에서는 0을 넘긴다.

이 API의 불리언 마스크는 True가 허용이라는 점도 앞의 사용자 정의 클래스와 다르다. is_causal=True의 비정사각 마스크 정렬은 KV 캐시의 위치 오프셋과 맞는지 확인해야 한다.

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

---

## 7. 접두 언어 모델: 양방향 접두와 인과 접미

접두부는 양방향으로 읽고 그 뒤는 인과적으로 생성하는 목적을 prefix LM이라고 한다. 일반적인 T5 디코더 전체가 이 마스크를 사용한다는 뜻은 아니다.

아래 함수는 접두부 행이 접미부 키를 보지 못하게 가리지 않아 접미부 정답이 접두부를 통해 유출될 수 있다. 의도한 prefix LM이라면 접두부 쿼리에서 접미부 키로 가는 블록도 차단해야 한다. 0≤prefix_len≤total_len 조건도 필요하다.

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

---

## 8. 슬라이딩 윈도 주의

슬라이딩 윈도 마스크는 각 쿼리가 자신과 가까운 과거 위치만 보게 한다. window_size는 양수여야 한다. 이 코드는 전체 L×L 마스크를 생성하므로 윈도 제한만으로 연산·메모리가 선형으로 바뀌지는 않는다. 윈도 전용 커널이나 희소 연산이 필요하다.

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

---

## 9. 학습과 추론

### 학습 (병렬)

학습에서는 입력과 목표를 한 토큰 어긋나게 맞춘다. 위치 i의 입력까지 읽어 다음 토큰을 예측하므로 마스크의 대각선은 허용해도 된다. 아래 호출은 외부에서 정의한 입력과 모델을 가정하는 개념 조각이다.

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

아래 추론 조각은 매번 전체 generated_tokens를 모델에 넘기며 KV 캐시를 실제로 사용하지 않는다. 캐시 구현은 각 레이어의 과거 K·V를 저장하고 새 쿼리와 결합해야 한다. 캐시 길이를 반영한 위치 ID와 인과 마스크 정렬도 필요하다.

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
다음 토큰 예측에서 인과 마스크와 목표 시프트가 각각 필요한 이유를 설명하라.

</div>

??? success "연습문제 1 풀이"
    인과 마스크는 위치 i가 i보다 뒤의 입력을 보지 못하게 한다. 목표 시프트는 그 위치의 출력이 현재 입력 토큰이 아니라 다음 토큰을 예측하게 한다. 자기 자신을 볼 수 있는 대각 마스크에 같은 토큰을 정답으로 주면 의도한 다음 토큰 학습이 아니다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
파이토치에서 효율적인 인과 어텐션 마스크를 구현하라.

</div>

??? success "연습문제 2 풀이"
    ```python
    def causal_mask(n):
        return torch.triu(torch.ones(n, n), diagonal=1).bool()  # True면 가린다

    # 주의에서: scores.masked_fill_(mask, float('-inf'))
    ```

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
인과 마스크와 패딩 마스크의 결합 규약을 설명하라.

</div>

??? success "연습문제 3 풀이"
    둘 다 True가 차단이라면 OR로 합칠 수 있다. True가 허용인 API에는 그대로 넘길 수 없다. 패딩 키 제외와 패딩 쿼리의 출력·로스 제외도 구분하고, 한 행의 모든 키가 가려질 때의 처리 규칙을 정한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
KV 캐시가 줄이는 연산과 추가로 사용하는 저장 공간을 설명하라.

</div>

??? success "연습문제 4 풀이"
    과거 위치의 K·V와 블록 출력을 다시 계산하는 일을 줄이고 새 쿼리로 과거 키 전체에 어텐션한다. 헤드 수·피처 차원 등을 고정하면 단계 t의 어텐션 내적·가중합은 O(t), 캐시 저장은 O(t)다. 전체 길이 n의 생성 동안 어텐션 합계는 O(n²)이며 토큰 투영·피드포워드·어휘 출력의 비용은 별도다. 캐시는 생성 전체를 상수 시간으로 만들지 않는다.

## 정리하며

인과 마스크는 미래에서 현재로의 정보 유출을 막고, 목표를 한 토큰 이동한 병렬 학습을 가능하게 한다. 불리언 마스크 규약, 패딩 행, 드롭아웃과 캐시 위치를 함께 확인한다. 슬라이딩 윈도 마스크나 함수 이름만으로 효율적인 커널 사용이 보장되지는 않는다.

**참고 문헌**

1. Vaswani, A., et al. (2017). "Attention Is All You Need." NeurIPS.
2. Radford, A., et al. (2019). "Language Models are Unsupervised Multitask Learners."
3. Dao, T., et al. (2022). "FlashAttention: Fast and Memory-Efficient Attention."
4. Beltagy, I., et al. (2020). "Longformer: The Long-Document Transformer."
