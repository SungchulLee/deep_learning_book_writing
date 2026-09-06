# GPT: 생성 사전 학습 트랜스포머
## 들어가며

GPT(생성 사전 학습 트랜스포머)는 자기 회귀 언어 모형화를 위해 설계된 디코더 전용 트랜스포머 구조이다. BERT의 양방향 방식과 달리 GPT는 한 방향(왼쪽에서 오른쪽) 주의를 써서 글 생성 과제에 자연스레 맞는다.

## GPT의 흐름

| 모형 | 매개변수 | 학습 데이터 | 맥락 길이 |
|-------|------------|---------------|----------------|
| GPT-1 (2018) | 117M | BooksCorpus | 512 |
| GPT-2 (2019) | 1.5B | WebText | 1024 |
| GPT-3 (2020) | 175B | 300B tokens | 2048 |
| GPT-4 (2023) | ~1.8T* | Undisclosed | 8K-128K |

*보도에 바탕한 어림

## 구조

GPT는 인과 자기 주의를 갖춘 디코더 전용 트랜스포머 블록의 더미를 쓴다.

$$
\text{GPT} = \text{TransformerDecoder}^L
$$

### BERT와의 핵심 차이

| 측면 | GPT | BERT |
|--------|-----|------|
| 구조 | 디코더만 | 인코더만 |
| 주의 | 인과 (한 방향) | 양방향 |
| 사전 학습 | 다음 토큰 맞히기 | 가린 언어 모형화 |
| 주된 쓰임 | 생성 | 이해 |

### 모형 설정 (GPT-2)

| 모형 | 층 | 머리 | $d_{\text{model}}$ | 매개변수 |
|-------|--------|-------|---------|------------|
| 작음 | 12 | 12 | 768 | 1억 1700만 |
| 중간 | 24 | 16 | 1024 | 3억 4500만 |
| 큼 | 36 | 20 | 1280 | 7억 6200만 |
| 아주 큼 | 48 | 25 | 1600 | 15억 |

## 사전 학습 목표

GPT는 표준 언어 모형화(다음 토큰 맞히기)를 쓴다.

$$
\mathcal{L} = -\sum_{t=1}^{T} \log P(x_t | x_1, \ldots, x_{t-1}; \theta)
$$

여기서 확률은 어휘에 대한 소프트맥스로 셈한다.

$$
P(x_t | x_{<t}) = \text{softmax}(h_t W_e^T)
$$

여기서 $h_t$은 자리 $t$의 숨은 상태이고 $W_e$은 임베딩 행렬이다(가중치 묶기).

## PyTorch 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Tuple, List

class GPTConfig:
    """GPT 모형 설정."""
    
    def __init__(
        self,
        vocab_size: int = 50257,
        n_positions: int = 1024,
        n_embd: int = 768,
        n_layer: int = 12,
        n_head: int = 12,
        n_inner: int = None,
        dropout: float = 0.1,
        layer_norm_epsilon: float = 1e-5
    ):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner or 4 * n_embd
        self.dropout = dropout
        self.layer_norm_epsilon = layer_norm_epsilon

class CausalSelfAttention(nn.Module):
    """다중 머리 인과 자기 주의."""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        
        assert config.n_embd % config.n_head == 0
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.scale = self.head_dim ** -0.5
        
        # QKV를 한데 모은 사영
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # 인과 가림
        self.register_buffer(
            'bias',
            torch.tril(torch.ones(config.n_positions, config.n_positions))
                .view(1, 1, config.n_positions, config.n_positions)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """KV 캐시를 선택으로 쓰는 앞먹임."""
        batch_size, seq_len, _ = x.shape
        
        # Q, K, V를 셈한다
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=-1)
        
        # 다중 머리 주의에 맞게 꼴을 바꾼다
        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        
        # 담아 둔 KV를 다룬다
        if layer_past is not None:
            past_k, past_v = layer_past
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        
        present = (k, v) if use_cache else None
        
        # 어텐션
        kv_seq_len = k.size(2)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # 인과 가림을 적용한다
        causal_mask = self.bias[:, :, kv_seq_len - seq_len:kv_seq_len, :kv_seq_len]
        attn_weights = attn_weights.masked_fill(causal_mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # 값에 적용한다
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_embd)
        output = self.c_proj(output)
        output = self.resid_dropout(output)
        
        return output, present

class GPTBlock(nn.Module):
    """GPT 트랜스포머 블록 하나."""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, config.n_inner),
            nn.GELU(),
            nn.Linear(config.n_inner, config.n_embd),
            nn.Dropout(config.dropout)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """앞 정규화 구조의 앞먹임."""
        attn_output, present = self.attn(self.ln_1(x), layer_past, use_cache)
        x = x + attn_output
        x = x + self.mlp(self.ln_2(x))
        return x, present

class GPTModel(nn.Module):
    """GPT 언어 모형."""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        # 임베딩
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        
        # 트랜스포머 블록
        self.h = nn.ModuleList([GPTBlock(config) for _ in range(config.n_layer)])
        
        # 마지막 층 정규화
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        
        # 언어 모형 머리 (가중치를 묶었다)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """앞먹임."""
        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        
        past_length = past_key_values[0][0].size(2) if past_key_values else 0
        
        if position_ids is None:
            position_ids = torch.arange(
                past_length, past_length + seq_len,
                dtype=torch.long, device=device
            ).unsqueeze(0)
        
        # 임베딩
        hidden_states = self.drop(self.wte(input_ids) + self.wpe(position_ids))
        
        # 블록
        presents = [] if use_cache else None
        for i, block in enumerate(self.h):
            layer_past = past_key_values[i] if past_key_values else None
            hidden_states, present = block(hidden_states, layer_past, use_cache)
            if use_cache:
                presents.append(present)
        
        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
        
        return {'loss': loss, 'logits': logits, 'past_key_values': presents}
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True
    ) -> torch.Tensor:
        """자기 회귀로 글을 짓는다."""
        past_key_values = None
        
        for _ in range(max_new_tokens):
            model_input = input_ids[:, -1:] if past_key_values else input_ids
            
            outputs = self.forward(model_input, past_key_values=past_key_values, use_cache=True)
            logits = outputs['logits'][:, -1, :] / temperature
            past_key_values = outputs['past_key_values']
            
            # 상위 k 거르기
            if top_k is not None:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits = logits.masked_fill(indices_to_remove, float('-inf'))
            
            # 상위 p 거르기
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits = logits.masked_fill(indices_to_remove, float('-inf'))
            
            # 뽑기
            if do_sample:
                next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            input_ids = torch.cat([input_ids, next_token], dim=-1)
        
        return input_ids

# 표집 방법
class SamplingStrategies:
    """글 생성의 표집 방법."""
    
    @staticmethod
    def greedy(logits: torch.Tensor) -> torch.Tensor:
        """탐욕 디코딩."""
        return torch.argmax(logits, dim=-1, keepdim=True)
    
    @staticmethod
    def temperature_sampling(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """온도 표집."""
        probs = F.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1)
    
    @staticmethod
    def top_k_sampling(logits: torch.Tensor, k: int = 50, temperature: float = 1.0) -> torch.Tensor:
        """상위 k 표집."""
        logits = logits / temperature
        values, indices = torch.topk(logits, k, dim=-1)
        logits_filtered = torch.full_like(logits, float('-inf'))
        logits_filtered.scatter_(1, indices, values)
        probs = F.softmax(logits_filtered, dim=-1)
        return torch.multinomial(probs, num_samples=1)
    
    @staticmethod
    def nucleus_sampling(logits: torch.Tensor, p: float = 0.9, temperature: float = 1.0) -> torch.Tensor:
        """핵(상위 p) 표집."""
        logits = logits / temperature
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

# 사용 예
if __name__ == "__main__":
    config = GPTConfig(vocab_size=50257, n_positions=1024, n_embd=768, n_layer=12, n_head=12)
    model = GPTModel(config)
    
    input_ids = torch.randint(0, config.vocab_size, (2, 64))
    outputs = model(input_ids, labels=input_ids)
    
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Loss: {outputs['loss'].item():.4f}")
    
    generated = model.generate(input_ids[:, :10], max_new_tokens=30, temperature=0.8, top_k=50)
    print(f"Generated shape: {generated.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

## 맥락 안 학습

GPT-3은 모형이 프롬프트 안의 예를 보고 과제에 맞추어 가는 맥락 안 학습을 들여왔다.

### 영 예시
```
Translate English to French:
sea otter =>
```

### 소수 예시
```
Translate English to French:
sea otter => loutre de mer
peppermint => menthe poivrée
cheese =>
```

## 규모 법칙

GPT-3은 규모가 커질 때의 변화를 내다볼 수 있음을 보였다.

$$
L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}
$$

여기서 $N$은 매개변수이며, 큰 모형일수록 표본을 더 아낌을 뜻한다.

### 친칠라 규모 법칙 (Hoffmann 외, 2022)

나중 연구는 GPT-3의 규모 조리법이 최선이 아니었음을 보였다. 친칠라 규모 법칙은 학습 토큰의 수가 모형 매개변수에 비례해 늘어야 한다고 내다본다.

$$N_{\text{opt}} \approx 20 \cdot D$$

여기서 $N_{\text{opt}}$은 가장 좋은 매개변수 수이고 $D$은 데이터 예산이다. 이는 GPT-3(매개변수 1750억, 토큰 3000억)이 크게 덜 학습되었음을 뜻한다. 그 연산 예산에서 계산 최적 모형이라면 매개변수 약 700억으로 토큰 약 1.4조 개를 학습했을 것이다.

## 창발하는 능력

GPT 모형의 규모가 커지면 작은 모형에는 없던 질적으로 새로운 능력이 창발한다.

| 능력 | 처음 관찰된 곳 | 규모 |
|------------|---------------|-------|
| 소수 예시 학습 | GPT-3 (1750억) | 매개변수 100억 초과 |
| 생각의 사슬 추론 | 매개변수 약 1000억 | 600억 초과 |
| 코드 생성 | Codex (약 120억) | 100억 초과 |
| 지시 따르기 | InstructGPT (13억 이상) | 인간 피드백 강화 학습과 함께 |

창발의 얼개는 아직 논란거리이다. 손실 지형의 상전이를 나타낼 수도 있고 그저 평가 지표의 해상도가 좋아진 것일 수도 있다.

## GPT에서 ChatGPT로: 정렬

GPT 모형은 다음 토큰을 맞히도록 학습되는데, 그것이 늘 도움이 되고 해롭지 않고 정직한 것과 들어맞지는 않는다. GPT에서 ChatGPT로 가는 길은 다음으로 이루어진다.

1. **지도 미세 조정(SFT)**: 바라는 행동을 사람이 써 보인 예로 학습한다
2. **보상 모형**: 출력들 사이의 사람의 선호를 맞히는 모형을 학습한다
3. **인간 피드백 강화 학습(RLHF)**: PPO로 보상 모형을 가장 크게 하도록 정책을 다듬는다

이 정렬 과정이 다음 토큰 예측기를 지시를 따르는 도우미로 바꾼다.

## 요약

GPT는 자기 회귀 언어 모형화 방식을 자리 잡게 했다.

1. **디코더 전용 구조**: 인코더-디코더보다 간단하고 깔끔하게 커진다
2. **인과 주의**: 생성에 자연스럽고 효율적인 추론을 위한 KV 캐싱을 가능케 한다
3. **규모**: 큰 모형은 작은 규모에는 없던 창발 능력을 보인다
4. **맥락 안 학습**: 프롬프트의 예로 미세 조정 없이 과제에 맞춘다
5. **가중치 묶기**: 입력 임베딩 층과 출력 사영이 임베딩 행렬 $W_e$을 함께 쓰면($P(x_t | x_{<t}) = \text{softmax}(h_t W_e^T)$) 매개변수가 줄고 한결같음이 나아진다

### 디코더 전용이 주류가 된 까닭

GPT의 디코더 전용 방식이 대형 언어 모형의 주류가 된 것은 매개변수마다 모든 예측에 이바지하기 때문이다. 인코더-디코더 모형에서는 생성 중에 인코더의 매개변수가 놀고 있다. 게다가 하나로 모은 다음 토큰 맞히기 목표는 간단하고 데이터를 아끼며(자리마다 학습 신호를 준다) 큰 규모에서 이해와 생성을 자연스레 함께 받친다.

## 참고 문헌

1. Radford, A., et al. (2018). "Improving Language Understanding by Generative Pre-Training." (GPT-1)
2. Radford, A., et al. (2019). "Language Models are Unsupervised Multitask Learners." (GPT-2)
3. Brown, T., et al. (2020). "Language Models are Few-Shot Learners." NeurIPS. (GPT-3)
4. Hoffmann, J., et al. (2022). "Training Compute-Optimal Large Language Models." (Chinchilla)
5. Ouyang, L., et al. (2022). "Training Language Models to Follow Instructions with Human Feedback." (InstructGPT)

---

## GPT로 하는 글 생성

#### 생성 과정

```
Prompt: "The quick brown"
     ↓
Step 1: P(fox|The quick brown) → "fox"
Step 2: P(jumps|The quick brown fox) → "jumps"
Step 3: P(over|...) → "over"
...
```

#### 파이토치 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Callable

class GPTGenerator:
    """GPT 모형으로 하는 글 생성."""
    
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        stop_tokens: Optional[List[int]] = None
    ) -> str:
        """
        프롬프트에서 글을 짓는다.
        
        인수:
            prompt: 입력 글
            max_new_tokens: 만들 토큰의 최대 개수
            temperature: 표본 추출의 온도 (높을수록 더 무작위)
            top_k: 상위 k개 토큰에서 뽑는다
            top_p: 핵 표집의 문턱값
            repetition_penalty: 토큰을 되풀이할 때의 벌점
            stop_tokens: 생성을 멈추는 토큰 번호
        """
        # 프롬프트를 인코딩한다
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        generated = input_ids
        past_kv = None
        
        for _ in range(max_new_tokens):
            # 모형의 출력을 얻는다
            if past_kv is not None:
                outputs = self.model(generated[:, -1:], past_key_values=past_kv, use_cache=True)
            else:
                outputs = self.model(generated, use_cache=True)
            
            logits = outputs.logits[:, -1, :]
            past_kv = outputs.past_key_values
            
            # 되풀이 벌점을 적용한다
            if repetition_penalty != 1.0:
                for token_id in set(generated[0].tolist()):
                    logits[0, token_id] /= repetition_penalty
            
            # 온도를 적용한다
            logits = logits / temperature
            
            # 상위 k 거르기를 적용한다
            if top_k is not None:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # 상위 p(핵) 거르기를 적용한다
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # 뽑기
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # 멈춤 토큰인지 살핀다
            if stop_tokens and next_token.item() in stop_tokens:
                break
            
            generated = torch.cat([generated, next_token], dim=-1)
        
        return self.tokenizer.decode(generated[0], skip_special_tokens=True)
    
    @torch.no_grad()
    def generate_beam_search(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        num_beams: int = 5,
        length_penalty: float = 1.0,
        early_stopping: bool = True
    ) -> str:
        """빔 탐색으로 짓는다."""
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        # 빔을 시작한다: (로그 확률, 수열)
        beams = [(0.0, input_ids)]
        
        for _ in range(max_new_tokens):
            all_candidates = []
            
            for log_prob, seq in beams:
                outputs = self.model(seq)
                logits = outputs.logits[:, -1, :]
                log_probs = F.log_softmax(logits, dim=-1)
                
                # 상위 k개 토큰을 얻는다
                topk_log_probs, topk_indices = torch.topk(log_probs, num_beams)
                
                for i in range(num_beams):
                    new_log_prob = log_prob + topk_log_probs[0, i].item()
                    new_seq = torch.cat([seq, topk_indices[:, i:i+1]], dim=-1)
                    
                    # 길이 벌점
                    score = new_log_prob / (len(new_seq[0]) ** length_penalty)
                    all_candidates.append((score, new_log_prob, new_seq))
            
            # 상위 빔을 고른다
            all_candidates.sort(key=lambda x: x[0], reverse=True)
            beams = [(c[1], c[2]) for c in all_candidates[:num_beams]]
        
        # 가장 좋은 수열을 돌려준다
        best_seq = beams[0][1]
        return self.tokenizer.decode(best_seq[0], skip_special_tokens=True)
    
    @torch.no_grad()
    def generate_contrastive(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        k: int = 4,
        alpha: float = 0.6
    ) -> str:
        """
        대조 탐색: 질과 다양함의 균형을 잡는다.
        
        점수 = (1-α) * 확률 - α * 맥락과의 최대 비슷함
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        generated = input_ids
        
        for _ in range(max_new_tokens):
            outputs = self.model(generated, output_hidden_states=True)
            logits = outputs.logits[:, -1, :]
            hidden = outputs.hidden_states[-1][:, -1, :]  # 마지막 층, 마지막 토큰
            
            # 상위 k개 후보를 얻는다
            top_probs, top_indices = torch.topk(F.softmax(logits, dim=-1), k)
            
            best_score = float('-inf')
            best_token = None
            
            for i in range(k):
                token_id = top_indices[0, i]
                prob = top_probs[0, i].item()
                
                # 이 토큰의 숨은 상태를 얻는다
                candidate_seq = torch.cat([generated, token_id.unsqueeze(0).unsqueeze(0)], dim=-1)
                candidate_out = self.model(candidate_seq, output_hidden_states=True)
                candidate_hidden = candidate_out.hidden_states[-1][:, -1, :]
                
                # 앞선 맥락과의 최대 비슷함을 셈한다
                context_hiddens = outputs.hidden_states[-1][0, :-1, :]  # 마지막만 뺀 전부
                similarities = F.cosine_similarity(
                    candidate_hidden.expand(context_hiddens.size(0), -1),
                    context_hiddens,
                    dim=-1
                )
                max_sim = similarities.max().item()
                
                # 대조 점수
                score = (1 - alpha) * prob - alpha * max_sim
                
                if score > best_score:
                    best_score = score
                    best_token = token_id
            
            generated = torch.cat([generated, best_token.unsqueeze(0).unsqueeze(0)], dim=-1)
        
        return self.tokenizer.decode(generated[0], skip_special_tokens=True)

# 흘려보내며 생성하기
class StreamingGenerator:
    """만들어지는 대로 토큰을 내보내는 생성기."""
    
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
    
    @torch.no_grad()
    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50
    ):
        """토큰을 하나씩 내보낸다."""
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        past_kv = None
        
        for _ in range(max_new_tokens):
            if past_kv is not None:
                outputs = self.model(input_ids[:, -1:], past_key_values=past_kv, use_cache=True)
            else:
                outputs = self.model(input_ids, use_cache=True)
            
            logits = outputs.logits[:, -1, :] / temperature
            past_kv = outputs.past_key_values
            
            # 상위 k 표집
            if top_k:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            # 새 토큰을 풀어 내보낸다
            token_str = self.tokenizer.decode(next_token[0])
            yield token_str

# 제약을 둔 생성
class ConstrainedGenerator:
    """제약을 두고 짓는다(이를테면 어떤 낱말을 꼭 넣어야 한다)."""
    
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
    
    @torch.no_grad()
    def generate_with_keywords(
        self,
        prompt: str,
        keywords: List[str],
        max_new_tokens: int = 100,
        temperature: float = 1.0
    ) -> str:
        """주어진 열쇳말을 꼭 담는 글을 짓는다."""
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        # 열쇳말을 인코딩한다
        keyword_ids = [self.tokenizer.encode(kw, add_special_tokens=False) for kw in keywords]
        remaining_keywords = set(range(len(keywords)))
        
        for step in range(max_new_tokens):
            outputs = self.model(input_ids)
            logits = outputs.logits[:, -1, :] / temperature
            
            # 아직 쓰지 않은 열쇳말 토큰의 확률을 올린다
            if remaining_keywords and step < max_new_tokens - 10:  # 이어 쓸 자리를 남긴다
                for kw_idx in remaining_keywords:
                    for token_id in keyword_ids[kw_idx]:
                        logits[0, token_id] += 5.0  # 올린다
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # 열쇳말이 나왔는지 살핀다
            for kw_idx in list(remaining_keywords):
                if next_token.item() in keyword_ids[kw_idx]:
                    remaining_keywords.discard(kw_idx)
            
            input_ids = torch.cat([input_ids, next_token], dim=-1)
        
        return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

# 사용 예
if __name__ == "__main__":
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    
    # 모델을 불러온다
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    generator = GPTGenerator(model, tokenizer, device='cpu')
    
    prompt = "The future of artificial intelligence"
    
    # 여러 가지 표집 방법
    print("=== Sampling Strategies ===\n")
    
    print("Greedy (temperature=0.1):")
    print(generator.generate(prompt, max_new_tokens=50, temperature=0.1))
    
    print("\nTop-k (k=50):")
    print(generator.generate(prompt, max_new_tokens=50, top_k=50))
    
    print("\nNucleus (top_p=0.9):")
    print(generator.generate(prompt, max_new_tokens=50, top_p=0.9))
    
    print("\nWith repetition penalty:")
    print(generator.generate(prompt, max_new_tokens=50, top_p=0.9, repetition_penalty=1.2))
```

#### 표집 방법 견주기

| 방법 | 좋은 점 | 나쁜 점 | 쓰임새 |
|----------|------|------|----------|
| 탐욕 | 결정론적 | 되풀이됨 | 사실을 내야 할 때 |
| 온도 | 다스리기 간단 | 앞뒤가 안 맞을 수 있음 | 창작 |
| 상위 k | 나쁜 토큰을 막음 | 자르는 지점이 고정됨 | 두루 쓰기 |
| 핵 (상위 p) | 자르는 지점이 유동적 | 좋은 토큰을 자를 수 있음 | 가장 두루 쓰인다 |
| 빔 탐색 | 모형 기준 최적 | 뻔한 출력 | 번역 |
| 대조 | 다양하고 앞뒤 맞음 | 느림 | 질이 높아야 할 때 |

#### 좋은 방법

1. 창작 과제에는 **온도 0.7~0.9**
2. 기본으로 **상위 p 0.9~0.95**
3. 고리를 줄이려면 **되풀이 벌점 1.1~1.3**
4. 가장 좋은 결과를 얻으려면 **상위 k와 상위 p를 함께**

#### 간추림

GPT의 생성은 다음으로 이루어진다.

1. 프롬프트를 모형에 넣어 처리한다
2. 분포에서 다음 토큰을 뽑는다
3. 표집 방법(온도, 상위 k, 상위 p)을 적용한다
4. 토큰을 덧붙이고 되풀이한다

#### 참고 문헌

1. Holtzman, A., et al. (2020). "The Curious Case of Neural Text Degeneration."
2. Su, Y., et al. (2022). "A Contrastive Framework for Neural Text Generation."

## 연습문제

**연습문제 1.**
GPT의 자기 회귀 사전 학습 목표를 설명하고 BERT의 MLM과 어떻게 다른지 밝혀라.

??? success "연습문제 1 풀이"
    GPT는 인과(왼쪽에서 오른쪽) 주의로 $\sum_t \log P(x_t | x_{<t})$을 가장 크게 한다. BERT의 MLM은 양방향 맥락으로 가린 토큰을 맞힌다. GPT의 목표는 글 생성을, BERT의 목표는 이해를 자연스레 받친다. GPT는 토큰마다 왼쪽 맥락을 보고, BERT는 가린 자리에 대해 왼쪽과 오른쪽 맥락을 모두 본다.

---

**연습문제 2.**
규모와 능력의 면에서 GPT-1에서 GPT-4까지의 흐름을 설명하라.

??? success "연습문제 2 풀이"
    GPT-1(2018)은 매개변수 1억 1700만으로 사전 학습과 미세 조정을 통한 전이 학습을 보였다. GPT-2(2019)는 15억으로 프롬프트를 통한 영 예시 과제 수행을 보였다. GPT-3(2020)은 1750억으로 소수 예시 맥락 안 학습을 들여왔다. GPT-4(2023)는 여러 양식(글과 그림)을 다루며 추론이 크게 나아졌다.

---

**연습문제 3.**
파이토치에서 간단한 GPT 방식 디코더 블록을 구현하라.

??? success "연습문제 3 풀이"
    ```python
    class GPTBlock(nn.Module):
        def __init__(self, d, h):
            super().__init__()
            self.ln1 = nn.LayerNorm(d)
            self.attn = nn.MultiheadAttention(d, h, batch_first=True)
            self.ln2 = nn.LayerNorm(d)
            self.ffn = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
        def forward(self, x, mask):
            x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=mask)[0]
            x = x + self.ffn(self.ln2(x))
            return x
    ```

---

**연습문제 4.**
대형 언어 모형의 '창발 능력' 개념을 설명하라.

??? success "연습문제 4 풀이"
    창발 능력이란 모형의 규모가 커질 때 갑자기 나타나며 작은 모형에는 없는 능력이다. 생각의 사슬 추론(매개변수 약 1000억), 소수 예시 학습(약 100억), 지시 따르기가 그 보기이다. 그 얼개는 논란거리이다. 매끄러운 능력 곡선을 끊긴 지표로 잰 탓이라는 주장도 있다(Schaeffer 외, 2023).
