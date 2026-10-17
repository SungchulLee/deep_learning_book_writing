# KV 캐시와 자기회귀 추론

KV 캐시는 각 레이어에서 이전 토큰의 키·값을 저장해 다음 토큰 처리에 재사용한다. 과거 전체의 상태와 사영을 매번 다시 계산하는 비용을 줄이는 대신 시퀀스 길이에 비례하는 저장 공간을 사용한다. 한 토큰의 계산과 전체 생성의 계산, 어텐션과 모델 전체의 비용을 구분해야 한다.

## 1. Prefill과 토큰별 디코딩

캐시가 없으면 다음 토큰마다 전체 접두사를 다시 넣는 구현을 사용할 수 있다.

```
Generate token 1: Compute attention for [prompt]
Generate token 2: Compute attention for [prompt, token1]  # 시킴말을 다시 셈한다!
Generate token 3: Compute attention for [prompt, token1, token2]  # 둘 다 다시 셈한다!
...
Generate token N: Compute attention for entire sequence
```

캐시가 있으면 처음 프롬프트를 처리하는 prefill 뒤 새 토큰만 입력한다.

```
Generate token 1: Compute K, V for [prompt], cache them
Generate token 2: Load cached K, V; compute only for token1
Generate token 3: Load cached K, V; compute only for token2
...
```

고정 차원에서 길이 t 전체 접두사의 밀집 어텐션은 O(t²), 새 쿼리 하나가 t개 키를 보는 디코딩은 O(t)다. 처음부터 T개를 생성하는 이상화된 어텐션 합계는 각각 O(T³)와 O(T²)다. 캐시를 쓴 전체 생성이 O(T)가 되는 것은 아니다. QKV·FFN의 차원별 비용, 어휘 출력 레이어·프롬프트 길이·메모리 이동은 별도로 포함된다.

길이 M 프롬프트와 G개 새 토큰에서는 prefill의 O(M²) 뒤 디코딩의 O(GM+G²) 어텐션이 더해진다. 이론적인 연산 차수와 실제 지연·처리량의 향상 비율은 같지 않다.

## 2. 상태·위치·마스크의 계약

새 토큰의 q_t·k_t·v_t를 만들고 기존 K·V 뒤에 k_t·v_t를 연결한다. 이전 캐시가 있는 상태에서 전체 접두사를 다시 입력하면 과거를 중복 추가하므로 새 구간만 전달해야 한다. 각 레이어의 cache는 같은 요청·시퀀스·모델 웨이트에서 온 것이어야 한다.

새 위치 ID는 캐시 길이에서 시작한다. 여러 새 토큰을 한 번에 넣으면 과거는 모두 보되 새 구간 안의 미래를 차단해야 한다. 배치별 길이·padding·EOS와 beam 재정렬도 관리해야 하며 서로 다른 사용자의 캐시를 섞어서는 안 된다.

## 3. PyTorch 축약 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from dataclasses import dataclass

@dataclass
class KVCache:
    """갈무리한 열쇠-값 짝을 담는 그릇."""
    key: torch.Tensor
    value: torch.Tensor
    
    @property
    def seq_len(self) -> int:
        return self.key.size(2)

class AttentionWithKVCache(nn.Module):
    """열쇠-값 곳간을 곁들인 인과 셀프 어텐션."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        use_cache: bool = True
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        batch_size, seq_len, _ = x.shape
        
        # Q, K, V를 셈한다
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 곳간을 새로 고친다
        if kv_cache is not None:
            k = torch.cat([kv_cache.key, k], dim=2)
            v = torch.cat([kv_cache.value, v], dim=2)
        
        new_cache = KVCache(k, v) if use_cache else None
        
        # 어텐션
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # 인과 가림막(토막이 여럿일 때만)
        if seq_len > 1:
            total_len = k.size(2)
            mask = torch.triu(torch.ones(seq_len, total_len, device=x.device), 
                            diagonal=total_len - seq_len + 1).bool()
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.o_proj(output), new_cache

class GPTWithKVCache(nn.Module):
    """만들어 내기를 위해 열쇠-값 곳간을 갖춘 GPT 모델."""
    
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, 
                 num_layers: int, d_ff: int, max_len: int = 2048):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        
        self.blocks = nn.ModuleList([
            self._make_block(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
    
    def _make_block(self, d_model, num_heads, d_ff):
        return nn.ModuleDict({
            'attn': AttentionWithKVCache(d_model, num_heads),
            'ffn': nn.Sequential(
                nn.Linear(d_model, d_ff), nn.GELU(),
                nn.Linear(d_ff, d_model)
            ),
            'norm1': nn.LayerNorm(d_model),
            'norm2': nn.LayerNorm(d_model)
        })
    
    def forward(self, input_ids, past_caches=None, use_cache=True):
        batch_size, seq_len = input_ids.shape
        past_len = past_caches[0].seq_len if past_caches else 0
        
        pos_ids = torch.arange(past_len, past_len + seq_len, device=input_ids.device)
        x = self.token_emb(input_ids) + self.pos_emb(pos_ids)
        
        if past_caches is None:
            past_caches = [None] * self.num_layers
        
        new_caches = []
        for i, block in enumerate(self.blocks):
            residual = x
            x = block['norm1'](x)
            x, cache = block['attn'](x, past_caches[i], use_cache)
            x = residual + x
            x = x + block['ffn'](block['norm2'](x))
            new_caches.append(cache)
        
        return self.lm_head(self.ln_f(x)), new_caches if use_cache else None
    
    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens=50, temperature=1.0, top_k=50):
        logits, caches = self(prompt_ids, use_cache=True)
        generated = prompt_ids
        
        for _ in range(max_new_tokens):
            next_logits = logits[:, -1, :] / temperature
            if top_k:
                v, _ = torch.topk(next_logits, top_k)
                next_logits[next_logits < v[:, -1:]] = float('-inf')
            
            next_token = torch.multinomial(F.softmax(next_logits, dim=-1), 1)
            generated = torch.cat([generated, next_token], dim=1)
            logits, caches = self(next_token, past_caches=caches, use_cache=True)
        
        return generated

# 예
if __name__ == "__main__":
    model = GPTWithKVCache(vocab_size=1000, d_model=256, num_heads=4, 
                           num_layers=4, d_ff=1024)
    
    prompt = torch.randint(0, 1000, (1, 10))
    generated = model.generate(prompt, max_new_tokens=20)
    print(f"Generated: {prompt.shape} -> {generated.shape}")
```

캐시는 레이어별 B×H×T×d_h 모양이다. 어텐션은 과거 길이를 반영한 오프셋 인과 마스크를 사용해 여러 새 토큰의 미래를 차단한다. 모델은 캐시 첫 레이어의 길이로 위치 ID를 만들므로 모든 레이어의 길이·장치·dtype이 일치해야 한다. num_layers·head 수·차원과 max_len도 유효해야 한다.

코드는 padding mask·가변 길이 요청·EOS 종료를 구현하지 않는다. AttentionWithKVCache의 dropout 모듈도 선언만 하고 적용하지 않는다. generate는 no_grad지만 eval로 자동 전환하지 않으며 temperature>0과 top_k≤어휘 크기 등의 검증이 없다. top-k 경계 동점은 k개보다 많이 남길 수 있다.

매 단계 torch.cat은 기존 cache를 새 텐서로 복사한다. 정적 버퍼나 paged 관리와 같은 효율을 가정하지 않는다. 마지막 토큰을 고른 뒤에도 추가 forward를 수행해 반환에 필요 없는 계산을 한다. main은 무작위 초기화 모델·프롬프트로 모양을 시연하며 학습된 생성 품질을 평가하지 않는다.

**기존 출력:**

```
Generated: torch.Size([1, 10]) -> torch.Size([1, 30])
```

## 4. 캐시 메모리 계산

레이어 수 L, 배치 B, 길이 T, KV head 수 H_kv, head 차원 d_h, 원소당 b바이트일 때 KV 값 저장량은 2LBTH_kv d_h b다. cache 메타데이터·할당 여유·웨이트·중간 활성화는 별도다. GQA에서는 쿼리 head가 아닌 KV head 수를 사용한다.

예를 들어 32레이어·32 KV head·head 차원 128·2바이트·배치 1이면 길이 2048에서 1GiB, 8192에서 4GiB, 32768에서 16GiB다. 해당 숫자는 가정한 원소 저장량이며 모든 ‘7B’ 모델의 캐시 크기와 같지 않다. GB와 GiB의 단위도 구분한다.

## 5. 저장 관리와 다른 최적화

[PagedAttention](https://arxiv.org/abs/2309.06180)은 논리적 KV를 물리 메모리 블록에 매핑해 할당·단편화·공유 관리를 개선한다. 공유 가능한 동일 prefix와 소유·수명 폴리시가 필요하며 임의의 서로 다른 토큰 캐시를 자유롭게 공유하는 것은 아니다.

GQA·MQA는 KV head 수를 줄이는 구조 선택으로 배치 수만큼 캐시를 줄이는 기법이 아니다. 최근 토큰만 남기는 sliding cache는 원 모델의 어텐션 범위·위치·학습과 맞춰야 하며 임의의 full-attention 모델에 적용해도 출력이 그대로라는 보장은 없다.

[FlashAttention](flash_attention.md)은 어텐션의 중간 행렬·메모리 이동을 줄이는 접근이고 KV cache는 과거 계산을 재사용하는 접근이다. 함께 쓸 수 있지만 같은 최적화는 아니다. 이 편집에서는 생성·시간 측정·그림 재생성을 하지 않았다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
캐시가 있을 때 한 토큰과 T토큰 전체의 어텐션 차수를 구분하라.

</div>

??? success "풀이"

    t번째 새 쿼리는 과거 키 t개를 보므로 O(t), T토큰을 생성하는 합은 O(T²)다. 매번 전체 접두사를 재계산하면 단계당 O(t²), 합계 O(T³)다. 이는 고정 차원의 어텐션 비교이며 모델 전체 사영·FFN·어휘 출력·메모리 비용은 추가된다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
예제 모델의 4레이어·4head·head 차원 64, B=1, T=30, FP32 cache 저장량을 계산하라.

</div>

??? success "풀이"

    2×4×1×4×30×64×4=245,760바이트로 240KiB다. 웨이트·임시 텐서·할당 오버헤드는 제외한다. GQA라면 같은 공식에서 실제 KV head 수를 사용해야 한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff easy" title="쉬움"></span>
cache가 있는 여러 새 토큰에서 필요한 마스크와 입력 범위를 설명하라.

</div>

??? success "풀이"

    과거 cache를 한 번만 포함하고 새 구간만 입력한다. 새 구간의 위치 i는 과거와 자신까지 볼 수 있지만 뒤의 새 토큰은 보지 못하도록 오프셋 인과 마스크가 필요하다. 위치 ID와 모든 레이어 cache 길이·배치 재정렬·padding도 일치해야 한다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
PagedAttention·GQA·sliding cache가 서로 다른 이유를 설명하라.

</div>

??? success "풀이"

    PagedAttention은 물리 저장과 공유 관리, GQA는 KV head 공유 구조, sliding cache는 참조할 과거 범위를 제한하는 폴리시다. 메모리 절감 원인과 출력 의미가 다르며 sliding 제한은 원 모델과 맞지 않으면 결과를 바꿀 수 있다. 이론적 저장량과 실제 지연은 별도 평가한다.



## 정리하며

KV cache는 과거 계산을 재사용하지만 전체 생성의 어텐션과 저장량은 길이에 따라 늘어난다. 새 입력·위치·인과 마스크와 요청별 상태를 맞추고, 저장 관리·구조 변경·어텐션 커널의 효과를 구분한다.

**참고 문헌**

1. Pope, R., et al. (2022). "Efficiently Scaling Transformer Inference."
2. Kwon, W., et al. (2023). "Efficient Memory Management for LLM Serving with PagedAttention."
