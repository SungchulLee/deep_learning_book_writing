# 큰 말 모델 얼개의 새로움
## 학습 목표

- 맨 변환기에서 요즘 큰 말 모델까지의 핵심 얼개 바뀜을 이해한다
- RMSNorm, SwiGLU, 돌림 묻힘, 묶은 물음 눈길을 짠다
- 여러 얼개 고름의 맞바꿈을 살핀다
- GPT, LLaMA, Mistral 갈래의 얼개를 견준다

## 들어가며

요즘 큰 말 모델은 처음 변환기를 넘어 수많은 얼개의 새로움을 담고 있다. 이 고침은 고갱이 눈길 얼개는 그대로 두면서 익히기의 든든함, 셈의 효율, 모델의 능력을 낫게 한다.

## 고르게 맞추기 변종

### 앞 고르게 맞추기와 뒤 고르게 맞추기

**처음 변환기(뒤 고르게 맞추기)**:

$$\text{output} = \text{LayerNorm}(x + \text{Sublayer}(x))$$

**요즘 큰 말 모델(앞 고르게 맞추기)**:

$$\text{output} = x + \text{Sublayer}(\text{LayerNorm}(x))$$

앞 고르게 맞추기는 깊은 모델에서 기울기가 더 잘 흐르게 한다.

### RMSNorm

제곱 평균 제곱근 고르게 맞추기는 평균 빼기를 없앤다:

$$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \cdot \gamma$$

여기서:

$$\text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2}$$

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

**좋은 점**: LayerNorm보다 15% 빠르고 성능은 비슷하다.

## 깨어남 함수

### SwiGLU

스위시 깨어남을 쓴 문 달린 선형 낱:

$$\text{SwiGLU}(x) = \text{Swish}(xW_1) \otimes (xW_2)$$

여기서 $\text{Swish}(x) = x \cdot \sigma(x)$이고 $\sigma$은 시그모이드다.

```python
class SwiGLU(nn.Module):
    """앞먹임 그물을 위한 SwiGLU 깨어남."""
    
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

### 앞먹임 그물 견줌

| 깨어남 | 매개변수 | 성능 | 쓰이는 곳 |
|------------|------------|-------------|---------|
| ReLU | $2 \cdot d \cdot d_{ff}$ | 밑금 | 맨 처음 변환기 |
| GELU | $2 \cdot d \cdot d_{ff}$ | +1% | GPT-2, BERT |
| SwiGLU | $3 \cdot d \cdot d_{ff}$ | +2% | LLaMA, 미스트랄 |

## 자리 부호

### 돌림 자리 묻힘(RoPE)

RoPE는 복소 공간에서의 돌림으로 자리를 부호화한다:

$$\text{RoPE}(x_m, m) = x_m e^{im\theta}$$

실수 벡터에서는 차원 짝마다 돌림을 쓴다:

$$R_\theta^m = \begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{pmatrix}$$

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
            q, k: (묶음, 머리, 차례 길이, 머리 차원)
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

### RoPE의 좋은 점

1. **상대 자리**: 눈길이 절대 자리가 아니라 $(m - n)$에 달렸다
2. **밖으로 늘리기**: 절대 묻힘보다 길이에 두루 통한다
3. **매개변수를 더하지 않음**: 자리를 돌림으로 부호화한다

### 넓힌 맥락: NTK를 헤아린 잣수 맞추기

익힐 때보다 긴 차례에서는 바탕 잦기의 잣수를 맞춘다:

$$\theta'_i = \theta_i \cdot \alpha^{-2i/d}$$

```python
def ntk_scaled_rope(dim: int, max_seq_len: int, base: int = 10000, scale: float = 2.0):
    """맥락을 넓히기 위한 NTK를 헤아린 돌림 자리 묻힘 잣수 맞추기."""
    # 더 긴 차례를 위해 밑을 늘린다
    scaled_base = base * (scale ** (dim / (dim - 2)))
    inv_freq = 1.0 / (scaled_base ** (torch.arange(0, dim, 2).float() / dim))
    return inv_freq
```

## 어텐션의 갈래

### 다중 질의 어텐션 (MQA)

모든 물음 머리가 열쇠-값 머리 하나를 나눠 쓴다:

```python
class MultiQueryAttention(nn.Module):
    """여러 물음 눈길: 열쇠-값 머리 하나, 물음 머리 여럿."""
    
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
        
        # 여느 눈길
        scores = torch.einsum('blhd,bmhd->bhlm', q, k) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        out = torch.einsum('bhlm,bmhd->blhd', attn, v)
        
        return self.o_proj(out.reshape(B, L, -1))
```

### 묶음 질의 어텐션 (GQA)

여러 머리 눈길과 여러 물음 눈길의 사이. 곧 물음 머리 묶음이 열쇠-값 머리를 나눠 쓴다:

```python
class GroupedQueryAttention(nn.Module):
    """무리 지은 물음 눈길: 열쇠-값 머리 수 < 물음 머리 수."""
    
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
        
        # 여느 눈길 셈하기
        q = q.transpose(1, 2)  # (묶음, 머리, 길이, 차원)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        
        out = out.transpose(1, 2).reshape(B, L, -1)
        return self.o_proj(out)
```

### 눈길 견줌

| 갈래 | 열쇠-값 머리 | 열쇠-값 곳간 크기 | 좋음 | 쓰이는 곳 |
|------|----------|---------------|---------|---------|
| MHA | H | $2 \cdot H \cdot d_h$ | Best | GPT-3 |
| 묶은 물음 눈길 | H/G | $2 \cdot H/G \cdot d_h$ | 여러 머리 눈길에 가깝다 | LLaMA-2 70B |
| MQA | 1 | $2 \cdot d_h$ | Good | PaLM |

## 미끄러지는 창 주의

### Mistral의 방식

창 크기가 $W$인 가까운 자리 눈길:

```python
def sliding_window_mask(seq_len: int, window_size: int) -> torch.Tensor:
    """미끄러지는 창 주의 가림을 만든다."""
    mask = torch.zeros(seq_len, seq_len)
    
    for i in range(seq_len):
        start = max(0, i - window_size)
        mask[i, start:i+1] = 1
    
    return mask

class SlidingWindowAttention(nn.Module):
    """효율을 위해 미끄러지는 창을 쓰는 눈길."""
    
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
        
        # 가림막을 쓴 눈길
        scores = torch.einsum('bnhd,bmhd->bhnm', q, k) / (self.head_dim ** 0.5)
        scores = scores + mask
        attn = torch.softmax(scores, dim=-1)
        
        out = torch.einsum('bhnm,bmhd->bnhd', attn, v)
        return self.out(out.reshape(B, L, -1))
```

### 좋은 점

- **기억 공간**: O(n²) 대신 O(n·W)
- **실효 맥락**: 층을 쌓아 여전히 온 차례에 눈길을 준다

## 온전한 LLaMA 방식 덩이

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
        
        # 무리 지은 물음 눈길
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
        # 남는 이음을 곁들인 눈길
        h = self.attention_norm(x)
        h = self.attention(h, positions, mask)
        x = x + h
        
        # 잔차를 곁들인 순전파
        h = self.ffn_norm(x)
        h = self.ffn(h)
        x = x + h
        
        return x
```

## 구조 견주기

| 조각 | GPT-3 | LLaMA-2 | Mistral |
|-----------|-------|---------|---------|
| 고르게 맞추기 | LayerNorm | RMSNorm | RMSNorm |
| 고르게 맞추기 자리 | 뒤 | 앞 | 앞 |
| 깨어남 | GELU | SwiGLU | SwiGLU |
| 자리 | 배움 | RoPE | RoPE |
| 눈길 | 여러 머리 눈길 | 묶은 물음 눈길 | 미끄러지는 창 + 묶은 물음 눈길 |
| 맥락 | 2K/4K | 4K | 8K(미끄러지는 창) |

## 요약

요즘 큰 말 모델 얼개는 다음을 담고 있다:

1. **RMSNorm**: 평균 빼기 없는 더 빠른 고르게 맞추기
2. **SwiGLU**: 나타내는 힘을 키우는 문 달린 깨어남
3. **RoPE**: 돌림에 바탕한 상대 자리 부호
4. **묶은 물음 눈길/여러 물음 눈길**: 효율적인 미룸을 위해 줄인 열쇠-값 곳간
5. **미끄러지는 창**: 한 줄 복잡도의 눈길

## 핵심 통찰

$$\boxed{\text{Modern LLMs} = \text{Transformer} + \text{Pre-Norm} + \text{RoPE} + \text{SwiGLU} + \text{GQA}}$$

## 참고 문헌

1. Zhang, B., & Sennrich, R. (2019). Root Mean Square Layer Normalization.
2. Shazeer, N. (2020). GLU Variants Improve Transformer.
3. Su, J., et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding.
4. Ainslie, J., et al. (2023). GQA: Training Generalized Multi-Query Transformer Models.

## 연습문제

**연습문제 1.**
GPT, BERT, T5의 얼개 차이를 견주어라. 미리 익히기 목표는 어떻게 다른가?

??? success "연습문제 1 풀이"
    | 모델 | 얼개 | 미리 익히기 목표 | 방향 |
    |-------|-------------|----------------------|----------------|
    | **GPT** | 풀개만의 변환기 | 인과 말 나타내기(다음 토막 어림) | 왼쪽에서 오른쪽만 |
    | **BERT** | 부호기만의 변환기 | 가린 말 나타내기 + 다음 월 어림 | 두 방향 |
    | **T5** | 부호기-풀개 변환기 | 구간 망가뜨리기(잡음 없애기) | 부호기: 두 방향, 풀개: 왼쪽에서 오른쪽 |

    GPT는 만들어 내기에, BERT는 이해와 갈래 매기기에 뛰어나며, T5는 모든 일을 글에서 글로 세워 둘 다 잘한다.

---

**연습문제 2.**
GPT-1에서 GPT-4까지의 흐름을 좇아라. 걸음마다의 핵심 규모 눈썰미는 무엇인가?

??? success "연습문제 2 풀이"
    **GPT-1**(매개변수 1억 1700만): 살펴보지 않는 미리 익히기와 살펴 배우는 곱게 다듬기가 여러 일에 통함을 보였다. **GPT-2**(15억): 규모만으로 영 발 성능이 나옴을 보였고 "말 모델은 살펴보지 않는 여러 일 배우개"라는 눈썰미를 들여왔다. **GPT-3**(1750억): 기울기 고침 없이 몇 발 맥락 안에서 배우기를 보였고 규모 법칙을 세웠다. **GPT-4**(크기 미공개): 여러 갈래(글 + 그림)이며 사람 되먹임 북돋움 배움으로 따짐, 시킴 따르기, 안전을 크게 낫게 했다. 핵심 눈썰미: 세대마다 매개변수, 자료, 셈을 키우면 작은 규모에는 없던 떠오르는 능력이 나옴을 보였다.

---

**연습문제 3.**
자기되돌리기 말 나타내기와 가린 말 나타내기의 차이는 무엇인가? 요즘 큰 말 모델은 왜 대부분 자기되돌리기인가?

??? success "연습문제 3 풀이"
    **자기되돌리기** 모델은 앞선 토막을 모두 보고 다음 토막을 미루어 본다. $p(x_t | x_{<t})$이다. **가린** 모델은 가리지 않은 토막을 모두 보고 아무렇게나 가린 토막을 미루어 본다. $p(x_t | x_{\setminus t})$이다. 자기되돌리기 모델이 널리 쓰이는 까닭은 이렇다. (1) 사람이 읽고 쓰는 것과 맞게 왼쪽에서 오른쪽으로 자연스레 글을 만든다. (2) 인과 짜임 덕에 미룸 때 열쇠-값 갈무리가 잘 든다. (3) 크기 법칙이 자기되돌리기 목표에 이롭다(크게 키울 때 표본을 더 아낀다). (4) 앞뒤 흐름 속 배우기와 지시 따르기가 다음 토막 미루어 보기에서 더 자연스레 돋아난다.

---

**연습문제 4.**
맥락 안에서 배우기라는 생각을 밝혀라. 왜 놀라우며 한계는 무엇인가?

??? success "연습문제 4 풀이"
    맥락 안에서 배우기(ICL)는 큰 말 모델이 기울기를 조금도 고치지 않고 시킴말에 든 보기에 조건을 걸어 일을 해내는 힘이다. "Translate English to French: sea otter => loutre de mer, cheese => " 같은 시킴말을 주면 모델이 "fromage"를 올바로 내놓는다. 모델을 옮김에 대놓고 익힌 적이 없는데도 그렇다는 점이 놀랍다. 갖가지 글로 미리 익히는 동안 보기에서 일의 무늬를 미루는 법을 배운 것이다. **한계**: (1) 여러 걸음 따짐이 필요한 복잡한 일에서는 성능이 떨어진다. (2) 보기의 차례와 꼴에 민감하다. (3) 맥락 창의 길이에 매인다. (4) 특화된 일에서는 곱게 다듬은 성능에 못 미친다. (5) 그 얼개가 이론으로 온전히 밝혀지지 않았다.
