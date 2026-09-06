# 플래시 눈길
## 들어가며

Dao 외(2022)가 내놓은 플래시 눈길은 들고남을 헤아리는 정확한 눈길 알고리즘으로, 기억 공간 씀씀이를 O(N²)에서 O(N)으로 줄이면서 보통의 눈길보다 2~4배 빠르다. GPU의 기억 공간 다가감 무늬를 조심스레 짜서 이를 이룬다.

## 기억 공간의 병목

### 보통 눈길의 기억 공간 탈

보통의 눈길은 다음을 셈한다:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

This requires materializing the full $N \times N$ attention matrix:

```
Memory usage: O(N²) for storing attention scores
```

For a 100K token sequence with float16: $100K \times 100K \times 2 = 20$ GB just for attention!

### 들고남 복잡도

GPU 연산은 셈이 아니라 기억 공간에 묶이는 일이 흔하다:

| 연산 | 셈 | 기억 공간 다가감 |
|-----------|---------|---------------|
| 행렬 곱셈 | 많음 | 보통 |
| 소프트맥스 | 적음 | 많음(온 행렬 읽고 쓰기) |
| 떨구기 | 아주 적음 | 많음 |

플래시 눈길은 FLOPs가 아니라 **기억 공간 들고남**을 다듬는다.

## 핵심 생각

### 1. 타일 나누기

온 눈길을 셈하는 대신 타일 단위로 다룬다:

```
For each tile of Q:
    For each tile of K, V:
        Compute partial attention for this tile
        Update running statistics (online softmax)
```

### 2. 흐르며 하는 소프트맥스

보통의 소프트맥스는 두 번 훑어야 한다:

1. 수치를 든든히 하려 최댓값을 셈한다
2. 지수를 셈하고 고르게 맞춘다

흐르며 하는 소프트맥스는 온 행렬을 담지 않고 조금씩 셈한다:

$$
m_{new} = \max(m_{old}, \max(\mathbf{x}_{new}))
$$

$$
\ell_{new} = e^{m_{old} - m_{new}} \ell_{old} + \sum e^{x_i - m_{new}}
$$

### 3. 다시 셈하기

뒤먹임 동안 눈길 무게를 담아 두는 대신 다시 셈한다:

- 기억 공간을 아낀다: O(N²) 눈길 행렬을 담지 않는다
- 셈은 조금 더 들지만 기억 공간 들고남이 줄어 더 빠르다

## 알고리즘

### 순전파

```
Algorithm: Flash Attention Forward

Input: Q, K, V ∈ ℝ^{N×d}, block sizes Br, Bc
Output: O ∈ ℝ^{N×d}

1. Initialize O = 0, ℓ = 0, m = -∞ (running statistics)
2. Divide Q into Tr blocks of size Br
3. Divide K, V into Tc blocks of size Bc

4. for j = 1 to Tc:  # K, V 덩이를 되풀이한다
       Load Kj, Vj from HBM to SRAM
       
       for i = 1 to Tr:  # Q 덩이를 되풀이한다
           Load Qi, Oi, ℓi, mi from HBM to SRAM
           
           # 이 덩이의 눈길을 셈한다
           Sij = Qi @ Kj.T / √d
           
           # 흐르는 최댓값을 새로 고친다
           mij = rowmax(Sij)
           Pij = exp(Sij - mij)
           ℓij = rowsum(Pij)
           
           # 이동 통계 갱신
           mi_new = max(mi, mij)
           ℓi_new = exp(mi - mi_new) * ℓi + exp(mij - mi_new) * ℓij
           
           # 내놓기를 새로 고친다
           Oi = (ℓi * exp(mi - mi_new) * Oi + exp(mij - mi_new) * Pij @ Vj) / ℓi_new
           
           # 통계 갱신
           mi = mi_new
           ℓi = ℓi_new
           
           Store Oi, ℓi, mi to HBM

5. Return O
```

## PyTorch 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


def flash_attention_forward_reference(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    block_size: int = 64,
    causal: bool = False
) -> torch.Tensor:
    """
    플래시 눈길의 본보기 짜기(이해를 돕기 위해).
    
    참고: 이는 가르치려고 간추린 판이다.
    실제 플래시 눈길은 효율을 위해 CUDA로 짠다.
    
    인수:
        Q: 물음 [묶음, 머리, 차례 길이, 머리 차원]
        K: 열쇠 [묶음, 머리, 차례 길이, 머리 차원]
        V: 값 [묶음, 머리, 차례 길이, 머리 차원]
        block_size: 덩이로 나눌 때의 타일 크기
        causal: 인과 가림막을 쓸지 여부
        
    반환값:
        내놓기 [묶음, 머리, 차례 길이, 머리 차원]
    """
    batch_size, num_heads, seq_len, head_dim = Q.shape
    scale = head_dim ** -0.5
    
    # 덩이 수
    num_blocks = (seq_len + block_size - 1) // block_size
    
    # 내놓기와 통계를 첫자리매김한다
    O = torch.zeros_like(Q)
    L = torch.zeros(batch_size, num_heads, seq_len, 1, device=Q.device)
    M = torch.full((batch_size, num_heads, seq_len, 1), float('-inf'), device=Q.device)
    
    # 덩이 단위로 처리한다
    for j in range(num_blocks):
        # 열쇠-값 덩이 번호
        kv_start = j * block_size
        kv_end = min((j + 1) * block_size, seq_len)
        
        Kj = K[:, :, kv_start:kv_end, :]
        Vj = V[:, :, kv_start:kv_end, :]
        
        for i in range(num_blocks):
            # 물음 덩이 번호
            q_start = i * block_size
            q_end = min((i + 1) * block_size, seq_len)
            
            Qi = Q[:, :, q_start:q_end, :]
            
            # 이 덩이의 눈길 점수를 셈한다
            Sij = torch.matmul(Qi, Kj.transpose(-2, -1)) * scale
            
            # 필요하면 인과 가림을 적용한다
            if causal:
                # 이 덩이의 가림막을 만든다
                q_positions = torch.arange(q_start, q_end, device=Q.device)
                kv_positions = torch.arange(kv_start, kv_end, device=Q.device)
                mask = q_positions.unsqueeze(1) < kv_positions.unsqueeze(0)
                Sij = Sij.masked_fill(mask, float('-inf'))
            
            # 흐르는 부드러운 최댓값 새로 고침
            # 지금 덩이의 최댓값
            mij = Sij.max(dim=-1, keepdim=True).values
            
            # 앞선 통계를 불러온다
            mi = M[:, :, q_start:q_end, :]
            li = L[:, :, q_start:q_end, :]
            oi = O[:, :, q_start:q_end, :]
            
            # 새 최댓값
            mi_new = torch.maximum(mi, mij)
            
            # 수치가 안정되게 눈길 무게를 셈한다
            Pij = torch.exp(Sij - mi_new)
            lij = Pij.sum(dim=-1, keepdim=True)
            
            # 흐르는 합을 새로 고친다
            li_new = torch.exp(mi - mi_new) * li + lij
            
            # 내놓기를 새로 고친다
            oi_new = (
                torch.exp(mi - mi_new) * li * oi + 
                torch.matmul(Pij, Vj)
            ) / li_new
            
            # 새로 고친 값을 담는다
            O[:, :, q_start:q_end, :] = oi_new
            L[:, :, q_start:q_end, :] = li_new
            M[:, :, q_start:q_end, :] = mi_new
    
    return O


class FlashAttention(nn.Module):
    """
    PyTorch가 가장 좋게 다듬은 짜기를 쓰는 플래시 눈길 모듈.
    
    torch.nn.functional.scaled_dot_product_attention을 쓰며,
    쓸 수 있을 때 플래시 눈길을 저절로 쓴다.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        causal: bool = False
    ):
        super().__init__()
        
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.causal = causal
        self.dropout = dropout
        
        # 사영
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        플래시 눈길을 쓰는 앞먹임.
        
        인수:
            x: 입력 [batch, seq_len, d_model]
            attention_mask: 선택으로 주는 가림
            
        반환값:
            내놓기 [묶음, 차례 길이, d_model]
        """
        batch_size, seq_len, _ = x.shape
        
        # Q, K, V를 사영한다
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 다중 머리 주의에 맞게 꼴을 바꾼다
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # PyTorch가 가장 좋게 다듬은 SDPA를 쓴다(플래시 눈길이 들어 있다)
        # 가장 좋은 짜기를 저절로 고른다
        output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=self.causal
        )
        
        # 꼴을 바꾸고 출력을 사영한다
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.o_proj(output)
        
        return output


class FlashAttentionWithKVCache(nn.Module):
    """
    만들어 내기를 위해 열쇠-값 곳간을 받치는 플래시 눈길.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0
    ):
        super().__init__()
        
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        열쇠-값 곳간을 곁들일 수도 있는 앞먹임.
        """
        batch_size, seq_len, _ = x.shape
        
        # 사영한다
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 꼴을 바꾼다
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # KV 캐시를 다룬다
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
        
        present_key_value = (k, v) if use_cache else None
        
        # 플래시 눈길
        # 곳간을 쓰는 만들어 내기에서는 물음만 짧다
        is_causal = (past_key_value is None and seq_len > 1)
        
        output = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal
        )
        
        # 꼴을 바꾸고 사영한다
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.o_proj(output)
        
        return output, present_key_value


def benchmark_attention(seq_lengths: list, d_model: int = 512, num_heads: int = 8):
    """플래시 눈길과 여느 눈길의 잣대를 잰다."""
    import time
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    head_dim = d_model // num_heads
    
    results = []
    
    for seq_len in seq_lengths:
        print(f"\nSequence length: {seq_len}")
        
        # 들임을 만든다
        q = torch.randn(1, num_heads, seq_len, head_dim, device=device)
        k = torch.randn(1, num_heads, seq_len, head_dim, device=device)
        v = torch.randn(1, num_heads, seq_len, head_dim, device=device)
        
        # 여느 눈길
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start = time.time()
        
        for _ in range(10):
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
            attn = F.softmax(scores, dim=-1)
            out_standard = torch.matmul(attn, v)
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        time_standard = (time.time() - start) / 10
        
        # 플래시 눈길(SDPA로)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start = time.time()
        
        for _ in range(10):
            out_flash = F.scaled_dot_product_attention(q, k, v)
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        time_flash = (time.time() - start) / 10
        
        # 메모리 사용량
        memory_standard = seq_len * seq_len * 4 / (1024 ** 2)  # float32 눈길 행렬의 MB
        
        print(f"  Standard: {time_standard*1000:.2f}ms, ~{memory_standard:.1f}MB attention matrix")
        print(f"  Flash:    {time_flash*1000:.2f}ms (no attention matrix stored)")
        print(f"  Speedup:  {time_standard/time_flash:.2f}x")
        
        results.append({
            'seq_len': seq_len,
            'standard_ms': time_standard * 1000,
            'flash_ms': time_flash * 1000,
            'speedup': time_standard / time_flash
        })
    
    return results


# 사용 예
if __name__ == "__main__":
    print("Flash Attention Demo")
    print("=" * 50)
    
    # 설정
    d_model = 512
    num_heads = 8
    batch_size = 2
    seq_len = 1024
    
    # 플래시 눈길 모듈을 시험한다
    flash_attn = FlashAttention(d_model, num_heads, causal=True)
    
    x = torch.randn(batch_size, seq_len, d_model)
    output = flash_attn(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # 열쇠-값 곳간으로 시험한다
    print("\n--- Testing with KV Cache ---")
    flash_attn_cache = FlashAttentionWithKVCache(d_model, num_heads)
    
    # 프롬프트 처리
    prompt = torch.randn(1, 100, d_model)
    _, kv_cache = flash_attn_cache(prompt, use_cache=True)
    print(f"Prompt processed, cache size: {kv_cache[0].shape}")
    
    # 토막을 만든다
    for i in range(5):
        new_token = torch.randn(1, 1, d_model)
        out, kv_cache = flash_attn_cache(new_token, past_key_value=kv_cache, use_cache=True)
        print(f"Generated token {i+1}, cache size: {kv_cache[0].shape}")
    
    # 참고 짜기 시험
    print("\n--- Reference Implementation Test ---")
    Q = torch.randn(1, 4, 64, 32)
    K = torch.randn(1, 4, 64, 32)
    V = torch.randn(1, 4, 64, 32)
    
    # 여느 눈길
    scale = 32 ** -0.5
    standard_out = torch.matmul(
        F.softmax(torch.matmul(Q, K.transpose(-2, -1)) * scale, dim=-1),
        V
    )
    
    # 플래시 눈길 참고
    flash_out = flash_attention_forward_reference(Q, K, V, block_size=16)
    
    # 서로 맞는지 살핀다
    max_diff = (standard_out - flash_out).abs().max().item()
    print(f"Max difference between standard and flash: {max_diff:.2e}")
    print("✓ Outputs match!" if max_diff < 1e-5 else "✗ Outputs differ!")
    
    # 잣대 재기(GPU가 있으면)
    if torch.cuda.is_available():
        print("\n--- Benchmarking ---")
        benchmark_attention([512, 1024, 2048, 4096])
```

## 기억 공간과 빠르기 견줌

### 기억 사용

| 차례 길이 | 보통 눈길 | 플래시 눈길 |
|-----------------|-------------------|-----------------|
| 1K | 4 MB | O(block_size) |
| 4K | 64 MB | O(block_size) |
| 16K | 1 GB | O(block_size) |
| 64K | 16 GB | O(block_size) |

### 빠르기(흔한 경우)

| 차례 길이 | 빨라짐 |
|-----------------|---------|
| 512 | 1.5-2x |
| 2048 | 2-3x |
| 8192 | 3-4x |

## 플래시 눈길 2

플래시 눈길 2는 다음을 더 다듬는다:

1. **더 나은 나란히 하기**: 묶음이 아니라 차례 길이로 쪼갠다
2. **행렬곱이 아닌 FLOPs 줄이기**: 소프트맥스 연산을 가장 적게 한다
3. **더 나은 일 나누기**: GPU 점유율에 맞춰 다듬는다

흔한 빨라짐: **플래시 눈길 1보다 2배**

## PyTorch에 아우르기

```python
# PyTorch 2.0 이상은 가능하면 플래시 눈길을 저절로 쓴다
output = F.scaled_dot_product_attention(
    query, key, value,
    is_causal=True  # 녹여 붙인 인과 가림막을 쓸 수 있게 한다
)

# 어느 뒷단을 쓰는지 살핀다
with torch.backends.cuda.sdp_kernel(
    enable_flash=True,
    enable_math=False,
    enable_mem_efficient=False
):
    output = F.scaled_dot_product_attention(q, k, v)
```

## 요약

플래시 눈길은 눈길 셈하기의 판을 뒤집는다:

1. **O(N) 기억 공간**: N×N 눈길 행렬을 담을 필요가 없다
2. **2~4배 빠름**: 들고남을 헤아리는 알고리즘이 기억 공간 대역폭을 줄인다
3. **정확한 셈**: 보통의 눈길과 같은 결과
4. **긴 차례**: 토막 10만 개 넘는 차례로도 익힐 수 있다

## 참고 문헌

1. Dao, T., et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." NeurIPS.
2. Dao, T. (2023). "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning."
3. PyTorch 설명서: scaled_dot_product_attention

## 연습문제

**연습문제 1.**
열쇠-값 곳간이 자기되돌리기 풀기를 어떻게 빠르게 하는지 밝혀라. 기억 공간의 맞바꿈은 무엇인가?

??? success "연습문제 1 풀이"
    During autoregressive generation, each new token attends to all previous tokens. Without caching, generating token $t$ recomputes the key and value projections for all $t-1$ previous tokens, giving $O(t^2)$ total computation for a sequence of length $T$. With KV-caching, keys and values from previous steps are stored and reused, so only the new token's K/V are computed at each step, reducing computation to $O(T)$ total. The trade-off: KV-cache memory grows as $O(T \cdot L \cdot d)$ where $L$ is the number of layers and $d$ is the hidden dimension. For long sequences with large models, this can consume significant GPU memory.

---

**연습문제 2.**
플래시 눈길의 고갱이 생각을 설명하여라. 수학으로는 같은 셈을 하는데 왜 빨라지는가?

??? success "연습문제 2 풀이"
    Flash Attention exploits the GPU memory hierarchy. Standard attention materializes the $N \times N$ attention matrix in HBM (slow GPU memory), causing memory-bound computation. Flash Attention tiles the computation into blocks that fit in SRAM (fast on-chip memory), computing attention block-by-block without ever materializing the full attention matrix. It uses online softmax (tracking running max and sum) to compute exact attention incrementally. The speedup comes from reduced HBM reads/writes (IO complexity drops from $O(N^2 d)$ to $O(N^2 d^2 / M)$ where $M$ is SRAM size), not from fewer FLOPs. This provides 2-4x wall-clock speedup and $O(N)$ memory.

---

**연습문제 3.**
미리 짚어 풀기란 무엇이며 내놓는 분포를 바꾸지 않고 어떻게 미룸을 빠르게 하는가?

??? success "연습문제 3 풀이"
    미리 짚어 풀기는 작고 빠른 "밑그림" 모델로 후보 토막 $K$개를 만든 뒤 큰 "목표" 모델로 그 $K$개를 나란히 확인한다. 목표 모델이 밑그림의 어림에 동의하면 목표 모델 앞먹임 한 번으로 $K$개를 모두 받아들인다(차례차례 $K$번 대신). 어긋나면 물리치기 표집으로 다룬다. 곧 처음 물리친 토막을 맞춘 분포에서 다시 뽑아 내놓는 분포가 목표 모델의 것과 같게 한다. 빨라짐은 밑그림 모델의 받아들임 비율에 달렸고, 좋은 밑그림 모델이면 2~3배가 보통이다. 핵심 눈썰미는 확인은 나란히 되지만 만들어 내기는 차례차례라는 것이다.

---

**연습문제 4.**
익힌 뒤 양자화(PTQ)와 양자화를 헤아린 익히기(QAT)를 견주어라. 저마다 언제 쓰는 것이 좋은가?

??? success "연습문제 4 풀이"
    **익힌 뒤 양자화**는 더 익히지 않고 눈금 맞추기 자료로 잣수 인자를 정해 미리 익힌 모델의 무게(그리고 원하면 깨어남)를 양자화한다. 빠르고 단순하지만 특히 8비트 아래에서는 정확도가 떨어질 수 있다. **양자화를 헤아린 익히기**는 기울기에 곧바로 지나가기 어림개를 써서 익히는 동안 양자화를 흉내내어 모델이 낮은 정밀도에 맞춰지게 한다. 낮은 자릿수(4비트, 2비트)에서 정확도가 더 낫지만 온전한 익히기를 한 번 더 돌려야 한다. 빠르기가 중요하고 8비트면 넉넉한 (펼치기) 장면에서는 **익힌 뒤 양자화가 낫다**. 4비트처럼 세게 양자화해야 하고 익힐 자원이 있으면 **양자화를 헤아린 익히기가 낫다**.
