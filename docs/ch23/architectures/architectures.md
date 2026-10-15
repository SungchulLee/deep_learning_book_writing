# 대규모 언어 모델의 구조 변형

정규화 위치, FFN, 위치 표현과 키·값 head 공유는 트랜스포머의 계산·메모리·학습에 영향을 준다. 여러 모델이 이 선택들을 조합하지만 모든 대규모 언어 모델이 같은 구성을 사용하는 것은 아니다. 구조의 이론적 이점과 아래 부분 구현에서 실제 얻는 효과도 구분한다.

## 1. 정규화 위치와 RMSNorm

post-norm은 LayerNorm(x+Sublayer(x)), pre-norm은 x+Sublayer(LayerNorm(x)) 형태다. pre-norm은 잔차 경로의 기울기 전달에 도움이 될 수 있지만 모든 설정에서 최적이라는 뜻은 아니다. GPT-2 이후의 pre-norm을 원래 트랜스포머의 post-norm과 혼동하지 않는다.

[RMSNorm](https://arxiv.org/abs/1910.07467)은 평균을 빼지 않고 마지막 특징 축의 제곱평균으로 스케일을 조절한다.

$$\operatorname{RMSNorm}(x)=\gamma\odot\frac{x}{\sqrt{d^{-1}\sum_i x_i^2+\epsilon}}.$$

```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """제곱평균제곱근 층 정규화."""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 제곱 평균 제곱근 셈하기
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_normed = x / rms
        return x_normed * self.weight
```

현재 구현의 학습 매개변수는 d개의 γ이며 편향은 없다. 평균 0을 보장하는 정규화가 아니고 저정밀 x²의 overflow나 누적 dtype도 검토해야 한다. 단순한 연산이라는 이점은 있지만 ‘항상 15% 빠르다’는 고정 속도 향상은 없다. 실제 커널·dtype·모양에서 비교해야 한다.

## 2. SwiGLU와 FFN 폭

[SwiGLU](https://arxiv.org/abs/2002.05202)는 SiLU를 적용한 사영과 다른 사영을 원소별로 곱하고 출력 사영한다.

```python
class SwiGLU(nn.Module):
    """앞먹임 그물을 위한 SwiGLU 활성화."""
    
    def __init__(self, dim: int, hidden_dim: int = None, bias: bool = False):
        super().__init__()
        hidden_dim = hidden_dim or int(dim * 8/3)  # LLaMA는 8/3 곱수를 쓴다
        # 효율을 위해 256의 배수로 반올림한다
        hidden_dim = 256 * ((hidden_dim + 255) // 256)
        
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)  # 빗장
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)  # 내림 쏘기
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)  # 올림 쏘기
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: swish(x @ W1) * (x @ W3)
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))
```

편향을 제외하면 중간 차원 m의 두 행렬 FFN은 2dm개, 세 행렬 SwiGLU는 3dm개다. 공정하게 비교하려면 폭과 전체 매개변수·계산 예산을 맞춘다. m≈8d/3이면 폭 4d의 일반 FFN과 비슷한 가중치 수지만 코드가 m을 256의 배수로 올림하므로 정확한 수는 달라진다. GELU·SwiGLU에 고정된 1%·2% 성능 향상을 붙일 수는 없다.

## 3. RoPE의 회전과 상대 위치

[RoPE](https://arxiv.org/abs/2104.09864)는 Q·K의 차원 쌍을 위치에 따른 각도로 회전한다. 같은 주파수의 회전에 대해 (R_m q)ᵀ(R_n k)=qᵀR_(n−m)k이므로 내적에 상대 위치가 반영된다. 실제 점수는 Q·K의 내용에도 의존한다.

```python
class RotaryEmbedding(nn.Module):
    """회전 위치 임베딩(RoPE)."""
    
    def __init__(self, dim: int, max_seq_len: int = 8192, base: int = 10000):
        super().__init__()
        
        # 잦기를 셈한다
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # cos/sin을 미리 셈한다
        t = torch.arange(max_seq_len)
        freqs = torch.outer(t, inv_freq)
        
        self.register_buffer('cos_cached', freqs.cos())
        self.register_buffer('sin_cached', freqs.sin())
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor):
        """
        물음과 열쇠에 돌림 묻힘을 적용한다.
        
        인수:
            q, k: (배치, 머리, 차례 길이, 머리 차원)
            positions: (차례 길이,) 자리 번호
        """
        cos = self.cos_cached[positions]  # (차례 길이, 머리 차원/2)
        sin = self.sin_cached[positions]
        
        q_rotated = self._apply_rotary(q, cos, sin)
        k_rotated = self._apply_rotary(k, cos, sin)
        
        return q_rotated, k_rotated
    
    def _apply_rotary(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """텐서에 돌림을 적용한다."""
        # 짝으로 쪼갠다
        x1, x2 = x[..., ::2], x[..., 1::2]
        
        # 회전
        rotated = torch.stack([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)
        
        return rotated.flatten(-2)
```

회전 차원은 짝수여야 하고 positions는 미리 계산한 표의 유효 인덱스여야 한다. B×L×H×d 같은 head 축이 있으면 cos·sin의 broadcast 축을 명시적으로 맞춰야 한다. 현재 forward는 그 축을 자동 삽입하지 않는다. 위치 표를 계산할 수 있다는 사실만으로 학습 길이 밖 성능이 보장되지는 않는다.

```python
def ntk_scaled_rope(dim: int, max_seq_len: int, base: int = 10000, scale: float = 2.0):
    """맥락을 넓히기 위한 NTK를 헤아린 돌림 자리 묻힘 잣수 맞추기."""
    # 더 긴 차례를 위해 밑을 늘린다
    scaled_base = base * (scale ** (dim / (dim - 2)))
    inv_freq = 1.0 / (scaled_base ** (torch.arange(0, dim, 2).float() / dim))
    return inv_freq
```

ntk_scaled_rope는 base'=base·scale^(d/(d−2))로 주파수를 바꾸는 예다. max_seq_len은 사용하지 않고 d=2이면 분모가 0이다. 유효 d>2와 양의 base·scale이 필요하다. 주파수 재조정은 실제 학습·평가와 함께 검토할 선택이며 단순히 캐시 길이를 늘리는 것과도 다르다.

## 4. MHA·MQA·GQA

MHA는 각 쿼리 head에 키·값 head를 대응시킨다. MQA는 모든 쿼리가 한 KV head를 공유하고, [GQA](https://arxiv.org/abs/2305.13245)는 그 사이의 개수로 KV head를 공유한다.

```python
class MultiQueryAttention(nn.Module):
    """여러 물음 어텐션: 열쇠-값 머리 하나, 물음 머리 여럿."""
    
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.q_proj = nn.Linear(dim, dim)  # 물음 머리 여럿
        self.k_proj = nn.Linear(dim, self.head_dim)  # 열쇠-값 머리 하나
        self.v_proj = nn.Linear(dim, self.head_dim)
        self.o_proj = nn.Linear(dim, dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, L, 1, self.head_dim)  # 머리 하나
        v = self.v_proj(x).view(B, L, 1, self.head_dim)
        
        # k, v를 물음 머리에 걸쳐 퍼뜨린다
        k = k.expand(-1, -1, self.num_heads, -1)
        v = v.expand(-1, -1, self.num_heads, -1)
        
        # 여느 어텐션
        scores = torch.einsum('blhd,bmhd->bhlm', q, k) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        out = torch.einsum('bhlm,bmhd->blhd', attn, v)
        
        return self.o_proj(out.reshape(B, L, -1))
```

```python
class GroupedQueryAttention(nn.Module):
    """무리 지은 물음 어텐션: 열쇠-값 머리 수 < 물음 머리 수."""
    
    def __init__(
        self, 
        dim: int, 
        num_heads: int, 
        num_kv_heads: int  # LLaMA-2 70B는 열쇠-값 머리 8개, 물음 머리 64개를 쓴다
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_groups = num_heads // num_kv_heads
        self.head_dim = dim // num_heads
        
        self.q_proj = nn.Linear(dim, num_heads * self.head_dim)
        self.k_proj = nn.Linear(dim, num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(dim, num_kv_heads * self.head_dim)
        self.o_proj = nn.Linear(dim, dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim)
        
        # 물음 머리 수에 맞추려 열쇠-값 머리를 되풀이한다
        k = k.repeat_interleave(self.num_groups, dim=2)
        v = v.repeat_interleave(self.num_groups, dim=2)
        
        # 여느 어텐션 셈하기
        q = q.transpose(1, 2)  # (배치, 머리, 길이, 차원)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        
        out = out.transpose(1, 2).reshape(B, L, -1)
        return self.o_proj(out)
```

모델 차원 d는 쿼리 head 수 H로 나누어져야 하며 H는 KV head 수 H_kv로 나누어져야 한다. H_kv=H면 MHA, H_kv=1이면 MQA다. 이 코드는 두 어텐션 모두 인과·padding 마스크와 KV cache가 없어 전체 입력을 양방향으로 본다. 그대로 자기회귀 언어 모델에 사용하면 미래 정보가 누출된다.

길이 T·배치 B·층 수 L·head 차원 d_h에서 KV 원소 수는 2LBTH_kv d_h다. 실제 바이트는 dtype당 바이트를 곱한다. MHA 대비 H_kv/H 비율로 저장량을 줄일 수 있지만 GQA 코드의 repeat_interleave는 계산 시 head를 다시 복제한다. 캐시를 작은 head 수로 실제 저장하는 구현과 구분해야 한다. 품질·속도 우열은 학습·실행 조건에 달려 있다.

## 5. 슬라이딩 윈도 어텐션

국소 인과 어텐션은 위치 i에서 가까운 과거와 현재만 보도록 제한한다. [Mistral 7B](https://arxiv.org/abs/2310.06825)는 이런 구조와 GQA를 활용한 모델의 예다.

```python
def sliding_window_mask(seq_len: int, window_size: int) -> torch.Tensor:
    """미끄러지는 창 주의 가림을 만든다."""
    mask = torch.zeros(seq_len, seq_len)
    
    for i in range(seq_len):
        start = max(0, i - window_size)
        mask[i, start:i+1] = 1
    
    return mask

class SlidingWindowAttention(nn.Module):
    """효율을 위해 미끄러지는 창을 쓰는 어텐션."""
    
    def __init__(self, dim: int, num_heads: int, window_size: int = 4096):
        super().__init__()
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.num_heads = num_heads
        
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        
        # 미끄러지는 창 가림막을 만든다
        mask = sliding_window_mask(L, self.window_size).to(x.device)
        mask = mask.masked_fill(mask == 0, float('-inf'))
        
        # 가림막을 쓴 어텐션
        scores = torch.einsum('bnhd,bmhd->bhnm', q, k) / (self.head_dim ** 0.5)
        scores = scores + mask
        attn = torch.softmax(scores, dim=-1)
        
        out = torch.einsum('bhnm,bmhd->bnhd', attn, v)
        return self.out(out.reshape(B, L, -1))
```

현재 마스크는 i−window_size부터 i까지를 포함하므로 최대 window_size+1개 위치를 허용한다. 허용 위치의 값 1에 −∞ 차단을 더하는 형태라 모든 허용 점수에 공통 1이 추가되지만 소프트맥스에는 영향을 주지 않는다.

그러나 scores 자체를 B×H×T×T의 밀집 행렬로 만들므로 이 코드는 O(TW)의 희소 메모리·계산을 달성하지 않는다. 윈도 연산 전용 커널이나 필요한 위치만 구성하는 구현이 있어야 한다. 층 수를 늘리면 정보 경로가 더 멀리 갈 수 있지만 고정 깊이에서 임의 길이 전체 문맥을 항상 참조하는 것은 아니다.

## 6. 블록 연결과 구현 범위

```python
class LLaMABlock(nn.Module):
    """요즘 새것을 담은 LLaMA 꼴 변환기 덩이."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        ffn_dim_multiplier: float = 8/3,
        norm_eps: float = 1e-6
    ):
        super().__init__()
        
        # 앞 고르게 맞추기
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)
        
        # 무리 지은 물음 어텐션
        self.attention = GroupedQueryAttention(dim, num_heads, num_kv_heads)
        
        # SwiGLU 앞먹임 그물
        self.ffn = SwiGLU(dim, int(dim * ffn_dim_multiplier))
        
        # 돌림 자리 묻힘
        self.rope = RotaryEmbedding(dim // num_heads)
    
    def forward(
        self, 
        x: torch.Tensor, 
        positions: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        # 남는 이음을 곁들인 어텐션
        h = self.attention_norm(x)
        h = self.attention(h, positions, mask)
        x = x + h
        
        # 잔차를 곁들인 순전파
        h = self.ffn_norm(x)
        h = self.ffn(h)
        x = x + h
        
        return x
```

LLaMABlock은 두 pre-norm 잔차 경로와 GQA·SwiGLU의 연결 의도를 보여 준다. 하지만 GroupedQueryAttention.forward는 x 하나만 받는데 이 블록은 h·positions·mask를 넘겨 호출 시 인자 오류가 난다. rope를 선언하지만 Q·K에 적용하는 경로도 없다. 따라서 완성된 인과 LLaMA 블록이 아니다.

실제 블록에는 Q·K 회전·유효 마스크·cache와 위치 인덱스, dtype·shape 계약을 일치시켜야 한다. Llama 2도 크기별로 어텐션 구성이 다르므로 계열 전체에 GQA를 일률적으로 표시하지 않는다. 이 페이지는 부품 설명이며 논문의 전체 모델이나 벤치마크를 실행·재현한 결과가 아니다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
bias 없는 중간 폭 m의 일반 FFN과 SwiGLU 매개변수 수를 비교하라.

</div>

??? success "연습문제 1 풀이"

    일반 FFN은 d→m→d의 두 행렬로 2dm개다. SwiGLU는 두 d→m 사영과 한 m→d 출력으로 3dm개다. 같은 가중치 수를 위해 일반 FFN의 폭 4d에 대해 SwiGLU 폭을 약 8d/3로 둘 수 있지만 코드의 256 배수 올림을 반영해야 실제 수가 나온다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff easy" title="쉬움"></span>
GQA에서 H=32, H_kv=8일 때 이론적인 KV 저장량과 현재 repeat_interleave의 차이를 설명하라.

</div>

??? success "연습문제 2 풀이"

    같은 배치·길이·층·head 차원·dtype에서 작은 KV 캐시의 저장량은 MHA의 8/32=1/4다. 그러나 코드가 계산 전에 KV를 32개 head로 복제하므로 중간 텐서는 커진다. 실제 캐시도 구현하지 않아 저장량 감소를 측정한 결과는 아니다. 어텐션 점수의 T×T 메모리도 별도다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
마스크를 국소화한 코드가 곧바로 O(TW) 메모리를 얻지 못하는 이유를 설명하라.

</div>

??? success "연습문제 3 풀이"

    먼 위치의 점수를 −∞로 바꾸더라도 이미 T×T 밀집 scores를 만들었다면 할당량은 여전히 이차적이다. 필요한 윈도 위치만 계산·저장하는 구현이 필요하다. 또한 현재 window_size는 과거 위치 수이므로 현재를 포함한 최대 후보 수는 하나 더 많다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
현재 LLaMABlock을 인과 언어 모델의 완성 블록으로 볼 수 없는 이유를 설명하라.

</div>

??? success "연습문제 4 풀이"

    어텐션 메서드 인자 수가 맞지 않아 호출 오류가 나고 RoPE 모듈도 실제 Q·K에 적용되지 않는다. GQA에는 인과·padding 마스크와 cache가 없다. 선언된 부품 이름만으로 계산 경로와 정보 접근이 구현되었다고 판단할 수 없다. 코드 수정·실행 없이 이 계약을 본문에서 구분한다.



## 정리하며

정규화·게이트 FFN·RoPE·KV head 공유·국소 어텐션은 서로 다른 구조 선택이다. 매개변수·캐시·밀집 중간 텐서 비용을 구분하고 실제 연결·마스크·차원과 평가로 효과를 확인해야 한다.

**참고 문헌**

1. Zhang, B., & Sennrich, R. (2019). Root Mean Square Layer Normalization.
2. Shazeer, N. (2020). GLU Variants Improve Transformer.
3. Su, J., et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding.
4. Ainslie, J., et al. (2023). GQA: Training Generalized Multi-Query Transformer Models.
