# 열쇠-값 곳간과 미룸 다듬기

열쇠-값 곳간은 앞선 토막의 열쇠와 값 텐서를 갈무리해 겹치는 셈을 없애는, 자기되돌리기 글 만들어 내기의 근본 다듬기 재주이다. 만드는 토막마다의 미룸 시간을 O(N²)에서 O(N)으로 줄인다.

---

## 1. 겹치는 셈 문제

### 열쇠-값 곳간이 없을 때

자기되돌리기로 만들어 낼 때 새 토막마다 온 차례의 눈길을 다시 셈해야 한다:

```
Generate token 1: Compute attention for [prompt]
Generate token 2: Compute attention for [prompt, token1]  # 시킴말을 다시 셈한다!
Generate token 3: Compute attention for [prompt, token1, token2]  # 둘 다 다시 셈한다!
...
Generate token N: Compute attention for entire sequence
```

**전체 눈길 연산**: O(N³)

### 열쇠-값 곳간이 있을 때

앞선 자리의 K와 V를 갈무리하고 새 토막만 셈한다:

```
Generate token 1: Compute K, V for [prompt], cache them
Generate token 2: Load cached K, V; compute only for token1
Generate token 3: Load cached K, V; compute only for token2
...
```

**전체 눈길 연산**: O(N²)

---

## 2. 수학적 바탕

때 걸음 $t$에서 새 토막 $x_t$과 갈무리한 $K_{1:t-1}$, $V_{1:t-1}$이 주어지면

$$
\begin{aligned}
q_t &= x_t W^Q \\
k_t &= x_t W^K \\
v_t &= x_t W^V \\
K_{1:t} &= [K_{1:t-1}; k_t] \\
V_{1:t} &= [V_{1:t-1}; v_t] \\
\text{out}_t &= \text{softmax}\left(\frac{q_t K_{1:t}^T}{\sqrt{d_k}}\right)V_{1:t}
\end{aligned}
$$

---

## 3. PyTorch 구현

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
    """열쇠-값 곳간을 곁들인 인과 스스로 눈길."""
    
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

---

## 4. 기억 공간에서 헤아릴 점

### 층마다의 곳간 크기

$$
\text{Size} = 2 \times B \times H \times L \times d_h \times \text{bytes}
$$

여기서 $B$ = 묶음, $H$ = 머리, $L$ = 차례 길이, $d_h$ = 머리 차원이다.

### 보기: LLaMA-7B

| 차례 길이 | 곳간 크기 |
|-----------------|------------|
| 2K | 약 1 GB |
| 8K | 약 4 GB |
| 32K | 약 16 GB |

---

## 5. 앞선 다듬기

### 쪽 나눈 눈길(vLLM)

열쇠-값 곳간을 가상 기억 공간의 쪽처럼 다스린다:

- 잇닿지 않은 기억 공간 나눠 주기
- 차례끼리 기억 공간 나눠 쓰기
- 기억 공간 조각남을 줄인다

### 묶은 물음 눈길(GQA)

여러 물음 머리가 열쇠-값 머리를 나눠 쓴다:

- 곳간 크기를 묶음 크기만큼 줄인다
- LLaMA 2, Mistral이 쓴다

### 미끄러지는 창 곳간

긴 차례에서는 최근 토막만 갈무리한다.

---

## 연습문제

**연습문제 1.**
열쇠-값 곳간이 자기되돌리기 풀기를 어떻게 빠르게 하는지 밝혀라. 기억 공간의 맞바꿈은 무엇인가?

??? success "연습문제 1 풀이"
    자기되돌리기로 글을 만들 때 새 토막마다 앞선 토막 모두에 눈길을 준다. 갈무리하지 않으면 토막 $t$을 만들 때 앞선 토막 $t-1$개의 열쇠와 값 되비춤을 다시 셈하므로 길이 $T$의 이음에 온 셈이 $O(t^2)$이 된다. 열쇠-값 갈무리를 쓰면 앞선 걸음의 열쇠와 값을 담아 두었다가 되쓰므로 걸음마다 새 토막의 열쇠와 값만 셈하면 되어 온 셈이 $O(T)$으로 준다. 맞바꿈은 이렇다. 열쇠-값 갈무리의 기억 자리가 $O(T \cdot L \cdot d)$으로 늘어난다. 여기서 $L$은 층 수, $d$은 숨은 차수다. 큰 모델로 긴 이음을 다루면 GPU 기억 자리를 크게 잡아먹을 수 있다.

---

**연습문제 2.**
플래시 눈길의 고갱이 생각을 설명하여라. 수학으로는 같은 셈을 하는데 왜 빨라지는가?

??? success "연습문제 2 풀이"
    플래시 눈길은 GPU 기억 자리의 층 얼개를 쓴다. 여느 눈길은 $N \times N$ 눈길 행렬을 HBM(느린 GPU 기억 자리)에 실제로 만들어 기억 자리에 매인 셈이 된다. 플래시 눈길은 셈을 SRAM(빠른 칩 안 기억 자리)에 들어가는 덩이로 쪼개어, 온전한 눈길 행렬을 한 번도 만들지 않고 덩이마다 눈길을 셈한다. 이어 가는 소프트맥스(달리는 최댓값과 합을 좇는다)로 딱 맞는 눈길을 조금씩 셈한다. 빨라지는 까닭은 뜨는 셈 횟수가 줄어서가 아니라 HBM 읽고 쓰기가 줄어서다(입출력 복잡도가 $O(N^2 d)$에서 $O(N^2 d^2 / M)$으로 떨어지며 $M$은 SRAM 크기다). 그래서 벽시계 시간이 2~4배 빨라지고 기억 자리는 $O(N)$이 된다.

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

## 정리하며

열쇠-값 곳간은 효율적인 큰 말 모델 미룸에 꼭 필요하다:

1. **겹치는 셈을 없앤다**: 전체가 O(N³)이 아니라 O(N²)
2. **기억 공간 맞바꿈**: 미룸은 빨라지지만 기억 공간이 더 든다
3. **다듬기의 바탕**: 묶음 짓기, 미리 짚어 풀기를 가능하게 한다

**참고 문헌**

1. Pope, R., et al. (2022). "Efficiently Scaling Transformer Inference."
2. Kwon, W., et al. (2023). "Efficient Memory Management for LLM Serving with PagedAttention."
