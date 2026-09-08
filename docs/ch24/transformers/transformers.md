# 자기 되돌이 변환기

Vaswani 외(2017)가 내놓은 변환기 얼개는 자기 되돌이 차례 나타내기의 으뜸 틀이 되었다. 차례를 한 걸음씩 다루는 되돌이 신경망과 달리 변환기는 **스스로 눈길**로 익히는 동안 나타냄을 나란히 셈하면서 **인과 가림막**으로 자기 되돌이 성질을 지킨다. 이 아우름, 곧 나란한 익히기와 차례대로 만들어 내기가 GPT, LLaMA와 그 뒤를 잇는 요즘 큰 말 모델의 바탕이다.

---

## 1. 되돌이 그물에서 변환기로

### 되돌이 모델의 한계

예전의 자기 되돌이 모델(되돌이 신경망, 긴 짧은 기억, 문 달린 되돌이 낱개)에는 바탕이 되는 한계가 있다.

1. **차례대로 셈하기**: 때 $t$의 숨은 상태가 $t-1$의 상태에 매인다
2. **나란히 하기의 제한**: 익히기가 GPU의 나란함을 온전히 쓰지 못한다
3. **기울기 문제**: 긴 차례에서 기울기가 사라지거나 터진다
4. **붙박이 맥락 누르기**: 모든 지난 일이 크기가 붙박인 숨은 상태로 눌린다

### 어텐션이라는 해법

눈길 얼개는 다음으로 이 문제를 다룬다.

- 어느 두 자리 사이든 곧바로 이음을 셈한다
- 익히는 동안 온전히 나란히 할 수 있게 한다
- 기울기가 흐를 뚜렷한 길을 준다
- 그때그때 내용에 바탕한 맥락 모으기를 할 수 있게 한다

---

## 2. 인과(가린) 스스로 눈길

### 표준 자기 주의

들임 차례 $\mathbf{X} \in \mathbb{R}^{T \times d}$이 주어질 때 스스로 눈길은 다음을 셈한다.

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}$$

여기서 $\mathbf{Q} = \mathbf{X}\mathbf{W}_Q$, $\mathbf{K} = \mathbf{X}\mathbf{W}_K$, $\mathbf{V} = \mathbf{X}\mathbf{W}_V$이다.

### 인과 지키게 하기

자기 되돌이로 나타내려면 자리 $t$이 $> t$인 자리를 보아서는 안 된다. 이는 **인과 가림막**으로 이룬다.

$$\text{CausalAttention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}} + \mathbf{M}\right)\mathbf{V}$$

여기서 $\mathbf{M}$은 $-\infty$으로 이루어진 위 삼각 행렬이다.

$$M_{ij} = \begin{cases} 0 & \text{if } i \geq j \\ -\infty & \text{if } i < j \end{cases}$$

$-\infty$ 값은 소프트맥스 뒤 0이 되어 앞날 자리를 보지 못하게 한다.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CausalSelfAttention(nn.Module):
    """
    자기 되돌이 모델을 위한 인과(가린) 스스로 눈길.
    
    자리마다 제 자신과 앞선 자리에만 주의할 수 있다.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Q, K, V의 선형 사영
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        
        # 출력 사영
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        # 인과 마스크 만들기(하삼각)
        # 버퍼로 등록해 모델과 함께 갈무리되되 익혀지지는 않게 한다
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        self.register_buffer('causal_mask', mask)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        인과 가림을 쓰는 앞먹임.
        
        인수:
            x: 입력 텐서 [batch_size, seq_len, d_model]
            attention_mask: 덧붙일 수 있는 가림막 [묶음 크기, 차례 길이]
            
        반환값:
            출력 텐서 [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        
        # Q, K, V를 셈한다
        Q = self.W_q(x)  # [묶음, 차례 길이, d_model]
        K = self.W_k(x)
        V = self.W_v(x)
        
        # 다중 머리 주의에 맞게 꼴을 바꾼다
        # [묶음, 차례 길이, 머리 수, d_k] -> [묶음, 머리 수, 차례 길이, d_k]
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 주의 점수를 셈한다
        # [묶음, 머리 수, 차례 길이, 차례 길이]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 인과 가림을 적용한다
        causal_mask = self.causal_mask[:seq_len, :seq_len]
        scores = scores.masked_fill(causal_mask, float('-inf'))
        
        # 있으면 눈길 가림막을 쓴다(예컨대 채우기)
        if attention_mask is not None:
            # attention_mask: [묶음, 차례 길이] -> [묶음, 1, 1, 차례 길이]
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # 소프트맥스와 드롭아웃
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 값에 어텐션 적용
        # [묶음, 머리 수, 차례 길이, d_k]
        context = torch.matmul(attention_weights, V)
        
        # 다시 꼴 되돌리기
        # [묶음, 차례 길이, d_model]
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # 출력 사영
        output = self.W_o(context)
        
        return output
```

---

## 3. 변환기 풀개 덩이

### 구조

여느 변환기 풀개 덩이는 다음으로 이루어진다.

1. 남은 이음과 층 고르게 맞추기를 갖춘 인과 스스로 눈길
2. 남은 이음과 층 고르게 맞추기를 갖춘 앞먹임 신경망

```python
class TransformerBlock(nn.Module):
    """
    변환기 풀개 덩이 하나.
    
    짜임:
        x -> 층 고르게 맞추기 -> 인과 눈길 -> + -> 층 고르게 맞추기 -> 앞먹임 신경망 -> +
             |__________________________|      |___________________|
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int = None,
        dropout: float = 0.1,
        max_seq_len: int = 2048
    ):
        super().__init__()
        
        d_ff = d_ff or 4 * d_model  # 여느 방식: 앞먹임 신경망은 모델 차원의 4배
        
        # 층 고르게 맞추기(앞 층 고르게 맞추기 얼개)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
        # 인과 스스로 눈길
        self.attention = CausalSelfAttention(
            d_model, n_heads, max_seq_len, dropout
        )
        
        # 순방향 신경망
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        변환기 덩이를 지나는 앞먹임.
        
        인수:
            x: 들임 [묶음 크기, 차례 길이, d_model]
            attention_mask: 쓸 수도 있는 가림막 [묶음 크기, 차례 길이]
            
        반환값:
            내놓기 [묶음 크기, 차례 길이, d_model]
        """
        # 잔차를 곁들인 자기 주의
        x = x + self.attention(self.ln1(x), attention_mask)
        
        # 잔차를 곁들인 순전파
        x = x + self.ffn(self.ln2(x))
        
        return x
```

### 앞 층 고르게 맞추기와 뒤 층 고르게 맞추기

**뒤 층 고르게 맞추기**(본디 변환기):
```
x -> Attention -> + -> LayerNorm -> FFN -> + -> LayerNorm
     |____________|                 |______|
```

**앞 층 고르게 맞추기**(GPT-2, 요즘 모델):
```
x -> LayerNorm -> Attention -> + -> LayerNorm -> FFN -> +
                               |                        |
                               x ---------------------->|
```

앞 층 고르게 맞추기는 깊은 모델을 익히는 데 더 안정되며 이제 여느 방식이다.

---

## 4. GPT 얼개

### 온전한 모델

```python
class GPT(nn.Module):
    """
    GPT 꼴 자기 되돌이 변환기.
    
    구조:
        - 토큰 박아 넣기 + 자리 박아 넣기
        - 변환기 풀개 덩이 쌓기
        - 낱말로의 내놓기 쏘기
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        d_ff: int = None,
        max_seq_len: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # 토큰 임베딩과 자리 임베딩
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # 트랜스포머 블록
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, max_seq_len)
            for _ in range(n_layers)
        ])
        
        # 마지막 층 정규화
        self.ln_f = nn.LayerNorm(d_model)
        
        # 내놓기 쏘기(흔히 토큰 박아 넣기와 묶는다)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # 무게 묶기
        self.lm_head.weight = self.token_embedding.weight
        
        # 가중치 초기화
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """GPT-2을 따라 무게를 첫자리매김한다."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        앞먹임.
        
        인수:
            input_ids: 토큰 어깨수 [묶음 크기, 차례 길이]
            attention_mask: 쓸 수도 있는 가림막 [묶음 크기, 차례 길이]
            
        반환값:
            로짓 [묶음 크기, 차례 길이, 낱말 수]
        """
        batch_size, seq_len = input_ids.shape
        
        assert seq_len <= self.max_seq_len, f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}"
        
        # 묻힘을 얻는다
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        
        tok_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(positions)
        
        x = self.dropout(tok_emb + pos_emb)
        
        # 변환기 덩이를 지난다
        for block in self.blocks:
            x = block(x, attention_mask)
        
        # 마지막 층 고르게 맞추기와 쏘기
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits
    
    def compute_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        다음 토큰 헤아리기의 어긋 엔트로피 손실을 셈한다.
        """
        # 다음 토큰 맞히기를 위해 민다
        inputs = input_ids[:, :-1]
        targets = input_ids[:, 1:]
        
        # 로짓을 얻는다
        logits = self.forward(inputs, attention_mask[:, :-1] if attention_mask is not None else None)
        
        # 교차 엔트로피 손실
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-100  # 채움을 무시한다
        )
        
        return loss
```

### 모델 짜임새

```python
# GPT-2 짜임새
GPT2_CONFIGS = {
    'gpt2-small': {
        'vocab_size': 50257,
        'd_model': 768,
        'n_layers': 12,
        'n_heads': 12,
        'max_seq_len': 1024
    },
    'gpt2-medium': {
        'vocab_size': 50257,
        'd_model': 1024,
        'n_layers': 24,
        'n_heads': 16,
        'max_seq_len': 1024
    },
    'gpt2-large': {
        'vocab_size': 50257,
        'd_model': 1280,
        'n_layers': 36,
        'n_heads': 20,
        'max_seq_len': 1024
    },
    'gpt2-xl': {
        'vocab_size': 50257,
        'd_model': 1600,
        'n_layers': 48,
        'n_heads': 25,
        'max_seq_len': 1024
    }
}

def create_gpt2(config_name: str) -> GPT:
    """짜임새 이름으로 GPT-2 모델을 만든다."""
    config = GPT2_CONFIGS[config_name]
    return GPT(**config)
```

---

## 5. 자리 부호

### 배운 자리 박아 넣기

GPT은 배운 자리 박아 넣기, 곧 단순한 찾아보기 표를 쓴다.

```python
self.position_embedding = nn.Embedding(max_seq_len, d_model)
```

이점:

- 단순하고 잘 듣는다
- 자료에서 배운다

한계:

- 최대 길이가 붙박여 있다
- 익힌 길이 너머로 늘려 헤아리지 못한다

### 사인 꼴 자리 부호화

본디 변환기는 붙박인 사인 꼴 부호화를 썼다.

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

```python
class SinusoidalPositionalEncoding(nn.Module):
    """붙박인 사인 꼴 자리 부호화."""
    
    def __init__(self, d_model: int, max_seq_len: int = 5000):
        super().__init__()
        
        # 위치 인코딩 행렬을 만든다
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # [1, max_seq_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """들임에 자리 부호 더하기."""
        return x + self.pe[:, :x.size(1)]
```

### 돌림 자리 박아 넣기(RoPE)

요즘 모델(LLaMA 등)은 돌림으로 자리를 담는 RoPE을 쓴다.

```python
class RotaryPositionalEmbedding(nn.Module):
    """
    회전 위치 임베딩(RoPE).
    
    물음과 열쇠 벡터를 돌려 자리를 담는다.
    길이를 더 잘 늘려 헤아릴 수 있게 한다.
    """
    
    def __init__(self, d_model: int, max_seq_len: int = 4096, base: int = 10000):
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # 돌림 잦기를 셈한다
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
        
        # 코사인과 사인을 미리 셈한다
        self._precompute_cache(max_seq_len)
    
    def _precompute_cache(self, seq_len: int):
        """코사인과 사인 값을 미리 셈한다."""
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)  # [차례 길이, d_model/2]
        
        # 사인과 코사인 둘 다에 쓰려 겹친다
        emb = torch.cat([freqs, freqs], dim=-1)  # [차례 길이, d_model]
        
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len: int
    ) -> tuple:
        """
        물음과 열쇠에 돌림 박아 넣기를 쓴다.
        
        인수:
            q: 물음 텐서 [묶음, 머리 수, 차례 길이, d_k]
            k: 열쇠 텐서 [묶음, 머리 수, 차례 길이, d_k]
            seq_len: 수열 길이
            
        반환값:
            (돌린 q, 돌린 k)
        """
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]
        
        q_rot = self._apply_rotation(q, cos, sin)
        k_rot = self._apply_rotation(k, cos, sin)
        
        return q_rot, k_rot
    
    def _apply_rotation(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """텐서에 돌림을 적용한다."""
        # 두 반으로 가른다
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        
        # 회전
        rotated = torch.cat([-x2, x1], dim=-1)
        
        return x * cos + rotated * sin
```

---

## 6. 글 만들어 내기

### 자기 되돌이 뽑기

```python
@torch.no_grad()
def generate(
    model: GPT,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = None,
    top_p: float = None,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    자기 회귀로 글을 짓는다.
    
    인수:
        model: 익힌 GPT 모델
        prompt_ids: 출발 토큰 어깨수 [1, prompt_len]
        max_new_tokens: 만들 토큰의 최대 개수
        temperature: 표집 온도
        top_k: 상위 k 표집 매개변수
        top_p: 웃 p(핵) 뽑기 매개변수
        device: 만들어 낼 기기
        
    반환값:
        만든 토큰 어깨수 [1, prompt_len + max_new_tokens]
    """
    model.eval()
    model = model.to(device)
    
    generated = prompt_ids.clone().to(device)
    
    for _ in range(max_new_tokens):
        # 최대 차례 길이로 자른다
        context = generated[:, -model.max_seq_len:]
        
        # 다음 토큰의 로짓을 얻는다
        logits = model(context)
        next_token_logits = logits[:, -1, :]  # [1, 낱말 수]
        
        # 온도를 적용한다
        next_token_logits = next_token_logits / temperature
        
        # 상위 k 거르기를 적용한다
        if top_k is not None:
            indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
            next_token_logits[indices_to_remove] = float('-inf')
        
        # 상위 p(핵) 거르기를 적용한다
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # 쌓인 확률이 문턱값을 넘는 토막 없애기
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            next_token_logits[indices_to_remove] = float('-inf')
        
        # 분포에서 뽑는다
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # 만들어진 수열에 덧붙인다
        generated = torch.cat([generated, next_token], dim=1)
        
        # 차례 끝 토큰이 있는지 살핀다(뜻매김되어 있으면)
        # if next_token.item() == eos_token_id:
        #     break
    
    return generated
```

### 효율 좋은 만들어 내기를 위한 열쇠-값 저장턱

만들어 내는 동안 열쇠-값 짝을 저장턱에 담아 겹치는 셈을 피할 수 있다.

```python
class CausalSelfAttentionWithCache(nn.Module):
    """효율 좋은 만들어 내기를 위한 열쇠-값 저장턱을 갖춘 스스로 눈길."""
    
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 2048):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        # 만들어 내기를 위한 저장턱
        self.cache_k = None
        self.cache_v = None
    
    def forward(
        self,
        x: torch.Tensor,
        use_cache: bool = False,
        past_cache: tuple = None
    ) -> tuple:
        """
        저장턱을 쓸 수도 있는 앞먹임.
        
        인수:
            x: 입력 [batch, seq_len, d_model]
            use_cache: 저장턱을 쓰거나 고칠지 여부
            past_cache: 앞선 (K, V) 저장턱
            
        반환값:
            (내놓기, 저장턱). use_cache이면 저장턱은 (K, V), 아니면 None
        """
        batch_size, seq_len, _ = x.shape
        
        # 새 토큰의 Q, K, V을 셈한다
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 쓰고 있으면 저장턱에 덧붙인다
        if past_cache is not None:
            past_k, past_v = past_cache
            K = torch.cat([past_k, K], dim=2)
            V = torch.cat([past_v, V], dim=2)
        
        # 다음 되풀이를 위해 저장턱을 담아 둔다
        cache = (K, V) if use_cache else None
        
        # 어텐션 계산
        total_len = K.size(2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 새 자리가 옛 자리를 볼 때만 인과 가림막을 쓴다
        # 지난 자리는 모두 쓸 수 있다
        if seq_len > 1:
            mask = torch.triu(torch.ones(seq_len, total_len, device=x.device), diagonal=total_len - seq_len + 1).bool()
            scores = scores.masked_fill(mask, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, V)
        
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(context)
        
        return output, cache

@torch.no_grad()
def generate_with_cache(
    model,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    열쇠-값 저장턱을 쓴 효율 좋은 만들어 내기.
    
    맥락에는 저장턱의 K/V을 쓰고 새 토큰의 눈길만 셈한다.
    """
    model.eval()
    
    # 채근 글에 대한 첫 앞먹임
    generated = prompt_ids.clone()
    past_cache = None
    
    for _ in range(max_new_tokens):
        # 새 토큰만 다룬다
        if past_cache is None:
            input_ids = generated
        else:
            input_ids = generated[:, -1:]
        
        # 저장턱과 함께 앞먹임한다
        logits, past_cache = model.forward_with_cache(input_ids, past_cache)
        
        # 다음 토큰을 뽑는다
        next_logits = logits[:, -1, :] / temperature
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
        
        generated = torch.cat([generated, next_token], dim=1)
    
    return generated
```

---

## 7. 요즘 얼개의 개선

### 다중 질의 어텐션 (MQA)

머리 사이에 K, V을 나누어 써서 기억과 셈을 줄인다.

```python
class MultiQueryAttention(nn.Module):
    """
    여러 물음 눈길: 모든 머리가 K, V 하나를 나누어 쓴다.
    
    열쇠-값 저장턱 크기를 n_heads 갑절만큼 줄인다.
    """
    
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model, bias=False)  # 물음 머리 여럿
        self.W_k = nn.Linear(d_model, self.d_k, bias=False)  # K 하나
        self.W_v = nn.Linear(d_model, self.d_k, bias=False)  # V 하나
        self.W_o = nn.Linear(d_model, d_model, bias=False)
```

### 묶음 질의 어텐션 (GQA)

여러 머리 눈길과 여러 물음 눈길 사이의 절충으로 머리 무리가 K, V을 나누어 쓴다.

```python
class GroupedQueryAttention(nn.Module):
    """
    무리 물음 눈길: 무리 안에서 K, V을 나누어 쓴다.
    
    n_kv_heads < n_heads이면 기억과 셈을 아낀다.
    """
    
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        assert n_heads % n_kv_heads == 0
        
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_heads // n_kv_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.W_v = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
```

### SwiGLU 깨움

LLaMA과 요즘 모델은 앞먹임 신경망에 SwiGLU을 쓴다.

```python
class SwiGLU(nn.Module):
    """
    SwiGLU 깨움: 스위시 문 달린 선형 낱개.
    
    SwiGLU(x) = Swish(xW1) * (xW2)
    여기서 Swish(x) = x * 시그모이드(x)
    """
    
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_ff, bias=False)
        self.W2 = nn.Linear(d_model, d_ff, bias=False)
        self.W3 = nn.Linear(d_ff, d_model, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W3(F.silu(self.W1(x)) * self.W2(x))
```

### RMSNorm

층 고르게 맞추기보다 효율이 좋다.

```python
class RMSNorm(nn.Module):
    """
    제곱평균제곱근 층 정규화.
    
    층 고르게 맞추기보다 단순하고 빠르다(평균을 빼지 않는다).
    """
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight
```

---

## 8. 학습할 때 살필 점

### 배움 빠르기 차례표

```python
def get_lr_schedule(
    step: int,
    warmup_steps: int,
    max_steps: int,
    max_lr: float,
    min_lr: float
) -> float:
    """
    몸 풀기를 갖춘 코사인 배움 빠르기 차례표.
    """
    if step < warmup_steps:
        # 선형 워밍업
        return max_lr * step / warmup_steps
    elif step > max_steps:
        return min_lr
    else:
        # 코사인 감쇠
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))
```

### 기울기 검문점

큰 모델을 기억을 아껴 익히려면:

```python
from torch.utils.checkpoint import checkpoint

class GPTWithCheckpointing(GPT):
    """기억을 줄이려 기울기 되짚을 자리를 둔 GPT."""
    
    def forward(self, input_ids, attention_mask=None):
        # ... 박아 넣기 ...
        
        for block in self.blocks:
            # 기억을 아끼려 덩이마다 되짚을 자리를 둔다
            x = checkpoint(block, x, attention_mask)
        
        # ... 내놓기 쏘기 ...
```

---

## 9. 계량 금융에서의 쓰임

### 돈살림 글 만들어 내기

```python
class FinancialGPT(GPT):
    """
    돈살림 글 만들어 내기에 미세 조정한 GPT.
    
    쓰임새:
    - 보고서 간추리기
    - 되짚어 시험하기를 위한 소식 만들어 내기
    - 시나리오 이야기 만들어 내기
    """
    
    def __init__(self, financial_vocab_size: int, **kwargs):
        super().__init__(vocab_size=financial_vocab_size, **kwargs)
        
        # 덧붙인 돈살림 낱개 박아 넣기
        self.entity_embedding = nn.Embedding(1000, kwargs.get('d_model', 768))
```

### 시계열 예측

변환기는 돈살림 시계열을 나타낼 수 있다.

```python
class TimeSeriesTransformer(nn.Module):
    """
    돈살림 시계열 내다보기를 위한 변환기.
    
    값의 지난 일을 이어진 값의 "문장"으로 본다.
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        forecast_horizon: int = 10
    ):
        super().__init__()
        
        # 이어진 값을 d_model으로 쏜다
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 트랜스포머 블록
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads)
            for _ in range(n_layers)
        ])
        
        # 내놓기: 분포 매개변수를 헤아린다
        self.output_projection = nn.Linear(d_model, input_dim * 2)  # 평균, 표준 편차
        self.forecast_horizon = forecast_horizon
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        인수:
            x: 지난 값 [묶음, 차례 길이, 들임 차원]
            
        반환값:
            내다보기 분포의 (평균, 표준 편차)
        """
        h = self.input_projection(x)
        
        for block in self.blocks:
            h = block(h)
        
        params = self.output_projection(h[:, -1, :])
        mean, log_std = params.chunk(2, dim=-1)
        
        return mean, F.softplus(log_std)
```

---

## 연습문제

1. **눈길 그려 보기**: 모델이 무엇을 보는지 알아보려 눈길 머리 그려 보기를 짜라.

2. **자리 부호화 견주기**: 인공 과제에서 배운 박아 넣기, 사인 꼴, RoPE을 견주어라.

3. **열쇠-값 저장턱 짜기**: GPT 모델에 온전한 열쇠-값 저장턱을 짜고 빨라진 정도를 재라.

4. **돈살림 미세 조정**: 작은 GPT 모델을 돈살림 소식으로 미세 조정하고 만들어 내기 품질을 따져 보라.

5. **얼개 떼어 보기**: 여러 눈길 변형(여러 머리 눈길, 여러 물음 눈길, 무리 물음 눈길)으로 모델 솜씨를 견주어라.

## 정리하며

자기 되돌이 변환기는 차례 나타내기의 으뜸 얼개가 되었다.

1. **인과 가림막**은 나란한 익히기와 함께 자기 되돌이로 나타내기를 가능하게 한다
2. **스스로 눈길**은 어느 자리 사이든 곧바로 이음을 준다
3. **자리 부호화**(배운 것, 사인 꼴, 돌림)는 차례의 순서를 담는다
4. **열쇠-값 저장턱**은 효율 좋은 만들어 내기를 가능하게 한다
5. **요즘의 개선**(무리 물음 눈길, SwiGLU, 제곱평균제곱근 고르게 맞추기)은 효율과 솜씨를 높인다

표현력과 익히기 효율, 키울 수 있음이 어우러져 변환기는 요즘 말 모델의 바탕이 되었고 돈살림을 비롯한 다른 마당에서도 점점 중요해지고 있다.

**참고 문헌**

1. Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS*.
2. Radford, A., et al. (2018). Improving Language Understanding by Generative Pre-Training. *OpenAI*.
3. Radford, A., et al. (2019). Language Models are Unsupervised Multitask Learners. *OpenAI*.
4. Brown, T., et al. (2020). Language Models are Few-Shot Learners. *NeurIPS*.
5. Touvron, H., et al. (2023). LLaMA: Open and Efficient Foundation Language Models. *arXiv*.
6. Su, J., et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding. *arXiv*.

---
