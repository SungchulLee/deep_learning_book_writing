# FlashAttention

[FlashAttention](https://arxiv.org/abs/2205.14135)은 GPU의 메모리 계층을 고려해 전체 어텐션 행렬을 저장하지 않고 타일 단위로 계산하는 알고리즘이다. 수학적인 밀집 어텐션을 근사하지 않으면서 메모리 이동·중간 저장을 줄인다. 부동소수점 연산 순서에 따른 차이는 있을 수 있고 속도 향상도 하드웨어·dtype·길이·구현에 따라 달라진다.

## 1. 전체 행렬 저장의 비용

보통의 직접 구현은 softmax(QKᵀ/√d)V에서 QKᵀ와 확률 행렬을 저장한다.

```
Memory usage: O(N²) for storing attention scores
```

길이 100,000의 2바이트 점수 행렬 하나는 한 배치·한 head만으로 20GB다. 전체 head·배치와 역전파 저장량은 더 커진다. 이는 메모리 예산 계산이며 모든 일반 어텐션 API가 반드시 이 행렬을 외부에 저장한다는 뜻은 아니다.

## 2. 타일과 온라인 소프트맥스

```
For each tile of Q:
    For each tile of K, V:
        Compute partial attention for this tile
        Update running statistics (online softmax)
```

키 블록을 순회하며 각 쿼리 행의 최대값 m, 지수 합 ℓ와 출력의 가중 합을 갱신할 수 있다. 새 최대값이 바뀌면 이전 합과 출력도 같은 기준으로 재조정한다. 단순히 각 블록의 softmax를 계산해 평균하는 방식과 다르다.

$$m'=\max(m,\max_j s_j),\qquad\ell'=e^{m-m'}\ell+\sum_j e^{s_j-m'}.$$

어텐션 상호작용의 계산 자체는 밀집 O(T²d)이며 전체 모델 계산이 선형으로 바뀌는 것은 아니다. Q·K·V·출력과 행별 통계는 길이에 비례해 저장하고 타일 작업 공간도 필요하다. 전체 메모리를 O(block_size)만이라고 쓰면 이 저장량을 빠뜨린다.

## 3. 개념 알고리즘

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
           
           # 이 덩이의 어텐션을 셈한다
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

이 의사코드는 순전파의 재조정 원리를 보여 준다. 실제 고성능 GPU 커널은 타일·thread·warp 배분과 메모리 읽기·쓰기를 구현해야 한다. 역전파에서도 필요한 값을 재계산해 전체 확률 행렬 저장을 피하는 구성이 중요하다. 재계산은 연산을 추가할 수 있지만 메모리 이동 감소로 전체 실행이 빨라질 수 있다.

## 4. 참조 코드와 SDPA 래퍼

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
    플래시 어텐션의 본보기 짜기(이해를 돕기 위해).
    
    참고: 이는 가르치려고 간추린 판이다.
    실제 플래시 어텐션은 효율을 위해 CUDA로 짠다.
    
    인수:
        Q: 물음 [배치, 머리, 차례 길이, 머리 차원]
        K: 열쇠 [배치, 머리, 차례 길이, 머리 차원]
        V: 값 [배치, 머리, 차례 길이, 머리 차원]
        block_size: 덩이로 나눌 때의 타일 크기
        causal: 인과 가림막을 쓸지 여부
        
    반환값:
        내놓기 [배치, 머리, 차례 길이, 머리 차원]
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
            
            # 이 덩이의 어텐션 점수를 셈한다
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
            
            # 수치가 안정되게 어텐션 무게를 셈한다
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
    PyTorch가 가장 좋게 다듬은 짜기를 쓰는 플래시 어텐션 모듈.
    
    torch.nn.functional.scaled_dot_product_attention을 쓰며,
    쓸 수 있을 때 플래시 어텐션을 저절로 쓴다.
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
        플래시 어텐션을 쓰는 피드포워드.
        
        인수:
            x: 입력 [batch, seq_len, d_model]
            attention_mask: 선택으로 주는 가림
            
        반환값:
            내놓기 [배치, 차례 길이, d_model]
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
        
        # PyTorch가 가장 좋게 다듬은 SDPA를 쓴다(플래시 어텐션이 들어 있다)
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
    만들어 내기를 위해 열쇠-값 곳간을 받치는 플래시 어텐션.
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
        열쇠-값 곳간을 곁들일 수도 있는 피드포워드.
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
        
        # 플래시 어텐션
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
    """플래시 어텐션과 여느 어텐션의 잣대를 잰다."""
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
        
        # 여느 어텐션
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start = time.time()
        
        for _ in range(10):
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
            attn = F.softmax(scores, dim=-1)
            out_standard = torch.matmul(attn, v)
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        time_standard = (time.time() - start) / 10
        
        # 플래시 어텐션(SDPA로)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start = time.time()
        
        for _ in range(10):
            out_flash = F.scaled_dot_product_attention(q, k, v)
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        time_flash = (time.time() - start) / 10
        
        # 메모리 사용량
        memory_standard = seq_len * seq_len * 4 / (1024 ** 2)  # float32 어텐션 행렬의 MB
        
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
    
    # 플래시 어텐션 모듈을 시험한다
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
    
    # 여느 어텐션
    scale = 32 ** -0.5
    standard_out = torch.matmul(
        F.softmax(torch.matmul(Q, K.transpose(-2, -1)) * scale, dim=-1),
        V
    )
    
    # 플래시 어텐션 참고
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

flash_attention_forward_reference는 Python·PyTorch 타일 반복으로 개념을 보여 주는 코드다. SRAM·HBM 배치나 최적화된 전용 역전파를 구현하지 않는다. 일반 autograd로 추적할 때의 저장·in-place 갱신 계약도 별도로 검토해야 하므로 학습 메모리·속도 특성을 원 논문 커널과 같게 볼 수 없다. 현재 참조는 같은 길이의 Q·K·V와 같은 값 차원을 전제로 한다.

FlashAttention 클래스는 이름과 달리 [PyTorch SDPA](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)를 호출한다. 실제 backend는 입력·장치·dtype·지원 조건에 따라 선택되므로 호출만으로 Flash 커널 사용을 입증하지 않는다. bool attn_mask의 True는 허용 위치로 MultiheadAttention의 padding mask와 반대이며 broadcast 모양·가산 마스크 규칙도 맞춰야 한다.

SDPA는 지정한 dropout_p를 적용하므로 이 래퍼가 eval에서 0으로 바꾸는 것이 중요하다. is_causal과 별도 마스크의 조합은 설치 버전의 제약을 확인한다. FlashAttentionWithKVCache는 과거가 있으면 is_causal을 끈다. 새 토큰 하나에는 맞을 수 있지만 여러 새 토큰을 동시에 넣으면 그 구간의 미래를 볼 수 있어 별도 오프셋 마스크가 필요하다. padding 마스크도 제공하지 않는다.

**기존 출력:**

```
Flash Attention Demo
==================================================
Input shape: torch.Size([2, 1024, 512])
Output shape: torch.Size([2, 1024, 512])

--- Testing with KV Cache ---
Prompt processed, cache size: torch.Size([1, 8, 100, 64])
Generated token 1, cache size: torch.Size([1, 8, 101, 64])
Generated token 2, cache size: torch.Size([1, 8, 102, 64])
Generated token 3, cache size: torch.Size([1, 8, 103, 64])
Generated token 4, cache size: torch.Size([1, 8, 104, 64])
Generated token 5, cache size: torch.Size([1, 8, 105, 64])

--- Reference Implementation Test ---
Max difference between standard and flash: 2.38e-07
✓ Outputs match!
```

이 출력의 작은 비인과 Q·K·V 비교는 모든 길이·마스크·dtype·역전파와 cache 경로의 정확성을 증명하지 않는다. 캐시 시연도 새 구간 길이 1만 사용한다. 이 편집에서는 재실행하지 않았다.

## 5. 벤치마크 해석

benchmark_attention은 무작위 FP32 텐서에 직접 구현과 SDPA를 비교한다. 전용 Flash backend를 강제·확인하지 않고 warmup도 없으므로 출력 이름만으로 FlashAttention의 고정 속도 향상을 주장할 수 없다. GPU 동기화는 하지만 반복·순서·dtype·커널 선택의 영향을 분리해야 한다.

memory_standard는 T²×4만 세어 배치·head 수를 빠뜨린다. 실제 시연은 head 8개이므로 점수 하나도 그 여덟 배이며 확률·출력·기타 활성화는 별도다. SDPA 측정 중 앞의 scores·attn이 참조로 남는 점도 전체 peak memory 해석에 영향을 준다. 이 함수는 실제 peak memory를 측정하지 않는다.

## 6. FlashAttention-2와 커널 선택

[FlashAttention-2](https://arxiv.org/abs/2307.08691)는 비행렬곱 연산과 thread block·warp의 작업 분할 등을 개선한다. 논문에서 보고한 특정 가속 비율을 모든 길이·모델의 고정 효과로 일반화하지 않는다. prefill과 한 토큰 decode의 병목도 서로 다를 수 있다.

```python
# PyTorch 2.0 이상은 가능하면 플래시 어텐션을 저절로 쓴다
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

위 호출은 API 사용의 짧은 예이며 query·key·value가 정의된 완성 스크립트는 아니다. backend 제어 API와 지원 조건은 설치 버전에 맞춰 확인해야 한다. 특정 커널을 강제하면 지원하지 않는 입력에서 실패할 수 있다. 긴 문맥에서도 전체 모델·KV cache·학습 저장과 이차적 계산 비용은 남는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
FlashAttention의 선형 메모리와 밀집 어텐션의 이차적 계산을 구분하라.

</div>

??? success "연습문제 1 풀이"

    전체 T×T 중간 행렬을 저장하지 않지만 모든 쿼리·키의 밀집 상호작용은 계산한다. 고정 차원에서 추가 저장은 Q·K·V·출력·통계와 타일 작업 공간이고 계산은 O(T²d)다. 모델 전체나 KV cache까지 상수 메모리가 되는 것은 아니다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
블록별 softmax의 단순 평균이 전체 softmax와 다른 이유를 설명하라.

</div>

??? success "연습문제 2 풀이"

    각 블록의 정규화 분모가 달라 상대 확률 질량을 잃는다. 전체 기준 최대값과 지수 합을 유지하고 기준이 바뀔 때 이전 출력·합을 재조정해야 원래 어텐션을 계산할 수 있다. 반올림에 따른 수치 차이는 별도다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff easy" title="쉬움"></span>
현재 cache 래퍼에서 새 토큰 하나와 여러 토큰의 인과성이 다른 이유를 설명하라.

</div>

??? success "연습문제 3 풀이"

    과거가 있으면 is_causal=False여서 새 쿼리는 모든 현재 K를 본다. 새 토큰 하나는 미래 위치가 없지만 여러 개면 앞 토큰이 뒤 토큰을 볼 수 있다. 과거 길이를 반영한 오프셋 마스크와 padding·위치 처리가 필요하다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
이 페이지의 속도·메모리 출력만으로 Flash 커널 효과를 확정할 수 없는 이유를 설명하라.

</div>

??? success "연습문제 4 풀이"

    SDPA backend·warmup·dtype·반복 조건을 확인하지 않고 메모리 값도 T²×4의 단일 head 추정이다. 전용 커널 사용과 실제 peak memory·오차·역전파를 같은 환경에서 측정해야 한다. 짧은 참조 비교와 전체 구현의 정확성·속도 검증은 다르다.



## 정리하며

FlashAttention은 밀집 어텐션의 중간 저장과 메모리 이동을 줄인다. 참조 반복·SDPA backend·전용 커널을 구분하고 실제 마스크·cache·dtype·측정 조건에서 정확성과 비용을 확인해야 한다.

**참고 문헌**

1. Dao, T., et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." NeurIPS.
2. Dao, T. (2023). "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning."
3. PyTorch 설명서: scaled_dot_product_attention
