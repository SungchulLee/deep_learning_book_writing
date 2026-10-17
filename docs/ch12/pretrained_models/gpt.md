# GPT: 생성 사전 학습 트랜스포머

GPT 계열은 인과 셀프 어텐션으로 앞선 문맥을 읽고 다음 토큰을 예측하는 디코더 전용 언어 모델이다. 이 절에서는 역사적 모델의 피처, 학습 목표와 생성 구현을 살펴본다. 같은 GPT 이름이라도 모델별 구조·데이터·후처리와 공개된 정보가 다르다.

## 1. GPT의 흐름

| 모델 | 발표 당시의 대표 피처 | 비교 범위 |
|---|---|---|
| GPT-1, 2018 | 약 117M 계수, BooksCorpus 사전 학습·미세 조정 | 초기 전이 학습 실험 |
| GPT-2, 2019 | 최대 약 1.5B, WebText | 모델 크기별 체크포인트를 구분 |
| GPT-3, 2020 | 최대 175B, 프롬프트 기반 퓨샷 실험 | 모델 규모·과제별 결과를 구분 |
| GPT-4, 2023 | 텍스트·이미지 입력을 다룬 기술 보고 | 계수 수·구조 세부는 비공개 |

[GPT-4 기술 보고서](https://arxiv.org/abs/2303.08774)가 공개하지 않은 계수 수를 보도상의 추정으로 확정하지 않는다. 이후 제품·체크포인트의 컨텍스트 길이를 2023년의 모델 전체에 묶어 적용해서도 안 된다.

## 2. 구조

GPT는 인과 셀프 어텐션을 갖춘 디코더 전용 트랜스포머 블록의 스택를 쓴다.

$$
\text{GPT} = \text{TransformerDecoder}^L
$$

### BERT와의 핵심 차이

| 측면 | GPT | BERT |
|--------|-----|------|
| 구조 | 디코더만 | 인코더만 |
| 주의 | 인과 (한 방향) | 양방향 |
| 사전 학습 | 다음 토큰 예측 | 마스크 언어 모델링 |
| 주된 쓰임 | 생성 | 이해 |

### 모델 설정 (GPT-2)

| GPT-2 크기 | 레이어 | 헤드 | 모델 차원 | 대표 공개 체크포인트의 계수 수 |
|---|---|---|---|---|
| Small | 12 | 12 | 768 | 약 124M |
| Medium | 24 | 16 | 1024 | 약 355M |
| Large | 36 | 20 | 1280 | 약 774M |
| XL | 48 | 25 | 1600 | 약 1.558B |

초기 논문의 반올림 표기와 실제 공개 체크포인트의 집계가 다를 수 있다. 아래 코드는 사전 학습 체크포인트를 읽지 않는 GPT-2 형태의 별도 구현이다.

---

## 3. 사전 학습 목표

GPT는 표준 언어 모델링(다음 토큰 예측)를 쓴다.

$$
\mathcal{L} = -\sum_{t=1}^{T} \log P(x_t | x_1, \ldots, x_{t-1}; \theta)
$$

여기서 확률은 어휘에 대한 소프트맥스로 계산한다.

$$
P(x_t\mid x_{<t})=\operatorname{softmax}(h_{t-1}W_e^T)_{x_t}
$$

여기서 $h_{t-1}$은 토큰 $x_{t-1}$까지 읽은 마지막 은닉 상태이고, 소프트맥스 벡터의 정답 토큰 $x_t$ 성분을 고른다. 첫 토큰 예측은 시작 토큰이나 빈 문맥의 규칙이 필요하다. 입력 임베딩 행렬 $W_e$와 출력 투영을 공유하는 구성을 가정했다.

---

## 4. PyTorch 구현

이 GPTModel은 Pre-LN, 학습 절대 위치와 입력·출력 어휘 웨이트 공유를 사용한다. labels를 내부에서 한 토큰 이동하므로 외부에서 다시 시프트하지 않는다. PAD 정답은 -100으로 제외해야 하지만 어텐션 패딩 마스크 인수는 없다. 길이 1 또는 모두 제외한 정답의 평균 로스도 별도 처리해야 한다.

캐시는 레이어별 과거 K·V를 저장하고 위치 ID에 과거 길이를 반영한다. 캐시 전체 길이는 고정 위치 표·인과 마스크의 n_positions 이내여야 한다. generate는 no_grad를 사용하지만 eval로 바꾸지 않아 호출자가 평가 모드를 설정해야 한다. EOS 종료와 길이 초과 처리는 없고 temperature>0, 1≤top_k≤어휘 크기, 0<top_p≤1 조건을 가정한다.

main은 무작위 모델·토큰의 모양을 확인한다. 보존된 로스와 생성 길이는 언어 품질 평가가 아니다. 아래 별도 GPTGenerator와 반환 인터페이스도 다르다.

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
        """KV 캐시를 선택으로 쓰는 피드포워드."""
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
        """앞 정규화 구조의 피드포워드."""
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
        
        # 마지막 레이어 정규화
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        
        # 언어 모형 머리 (웨이트를 묶었다)
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
        """피드포워드."""
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

**출력:**

```
Logits shape: torch.Size([2, 64, 50257])
Loss: 10.9469
Generated shape: torch.Size([2, 40])
Parameters: 124,439,808
```

---

## 5. 맥락 내 학습

맥락 내 학습은 웨이트를 갱신하지 않고 프롬프트의 지시·예시를 이용해 과제를 수행하는 방식이다. GPT-3 연구는 이를 여러 규모와 과제에서 체계적으로 평가했다. 프롬프트에 예시가 있다고 정확한 규칙 적용이 보장되지는 않는다.

### 제로샷
```
Translate English to French:
sea otter =>
```

### 퓨샷
```
Translate English to French:
sea otter => loutre de mer
peppermint => menthe poivrée
cheese =>
```

---

## 6. 스케일링 법칙

스케일링 법칙은 특정 데이터·모델 계열에서 관찰한 로스와 모델 크기·학습 토큰·계산량의 관계를 근사한다. 파라메터만 늘리면 모든 과제의 품질이나 표본 효율이 보장된다는 법칙은 아니다.

친칠라 연구는 고정된 학습 계산량 안에서 계수 수 N과 학습 토큰 D를 함께 배분하는 문제를 다룬다. 흔히 인용하는 근사 비율은 다음 방향이다.

$$
D\approx20N
$$

즉 파라메터당 약 20개 학습 토큰이지 그 반대가 아니다. 이 수는 연구의 실험 범위에서 얻은 학습 계산 최적화의 근사이며 데이터 품질·추론 비용·재학습 예산에 따라 적절한 비율은 달라질 수 있다.

친칠라의 70B 모델·1.4T 토큰 실험은 Gopher와 학습 계산량을 맞춘 비교다. 이를 GPT-3의 계산 예산에 대한 직접적인 수치 대입으로 바꾸어 해석하지 않는다. [친칠라 원 논문](https://arxiv.org/abs/2203.15556)을 참고한다.

## 7. 창발하는 능력

규모 증가에 따라 일부 과제 점수가 급격히 오르는 관찰을 창발 능력이라고 부른다. 특정 계수 수를 넘으면 추론·코드·지시 수행이 반드시 나타나는 보편적 경계가 있는 것은 아니다. 데이터·프롬프트·추가 학습도 함께 영향을 준다.

정확 일치 같은 불연속 지표에서는 매끄러운 토큰별 개선도 급격한 점수 변화처럼 보일 수 있다. 연속적인 로스·부분 점수와 통계적 불확실성을 함께 비교해야 한다. 이 해석상의 쟁점은 [평가 지표와 창발에 관한 연구](https://arxiv.org/abs/2304.15004)를 참고한다.

## 8. GPT에서 ChatGPT로: 정렬

초기의 InstructGPT 연구는 지도 미세 조정(SFT), 인간 선호로 학습한 리워드 모델과 PPO 기반 RLHF를 연결했다. 리워드 점수만 무제한 높이는 것이 아니라 기준 폴리시에서 지나치게 벗어나지 않도록 제약하는 과정도 포함한다.

이는 후속 학습의 한 사례이며 모든 대화 모델이나 이후 ChatGPT 버전의 전체 학습법을 확정하는 설명은 아니다. 선호 최적화가 사실성·안전성·지시 준수를 완벽하게 보장하지도 않는다.

## 9. GPT로 하는 글 생성

이 절은 Hugging Face 모델의 outputs.logits·past_key_values 같은 속성을 가정한다. 앞의 사용자 정의 GPTModel은 사전을 반환하므로 그대로 교환할 수 없다. 기본 device="cuda"도 해당 장치가 없는 환경에서는 바꿔야 한다.

**반복 패널티:** 코드가 기존 토큰의 로짓을 모두 양수 패널티로 나누므로 음수 로짓은 오히려 커진다. 반복을 억제하려는 보편적인 구현으로 볼 수 없다. 보통 양·음수 로짓에 다른 변환이 필요하며 토큰 단위 패널티가 의미 반복까지 막는 것도 아니다.

**빔·대조 탐색:** generate_beam_search는 EOS를 처리하지 않고 early_stopping 인수도 사용하지 않는다. 같은 단계의 모든 후보 길이가 같아 길이 정규화가 그 단계의 순위를 바꾸지도 않는다. 빔 탐색은 제한된 후보만 유지하는 근사이며 전역 최적을 보장하지 않는다. 대조 탐색은 마지막 접두 토큰을 비교 문맥에서 제외하고, 길이 1 접두에서는 빈 유사도에 max를 적용해 실패할 수 있다.

**스트리밍·키워드:** StreamingGenerator와 ConstrainedGenerator는 eval 전환을 하지 않는다. 토큰 하나씩 독립 디코딩해 이어 붙이면 전체 토큰을 함께 디코딩한 문자열과 다를 수 있다. 키워드 구현은 키워드의 토큰 하나만 나와도 완료로 간주해 전체 단어·구문 포함을 보장하지 않는다. main의 “Greedy (temperature=0.1)”도 argmax가 아니라 낮은 온도 샘플링이다.

### 생성 과정

```
Prompt: "The quick brown"
     ↓
Step 1: P(fox|The quick brown) → "fox"
Step 2: P(jumps|The quick brown fox) → "jumps"
Step 3: P(over|...) → "over"
...
```

### 파이토치 구현

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
            top_p: 핵 표집의 임계값
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
            
            # 상위 p(커널) 거르기를 적용한다
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
            hidden = outputs.hidden_states[-1][:, -1, :]  # 마지막 레이어, 마지막 토큰
            
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

### 샘플링 방법 비교

| 방법 | 선택 방식 | 해석의 한계 |
|---|---|---|
| 탐욕 | 각 단계 argmax | 사실성·전체 시퀀스 확률 최적을 보장하지 않음 |
| 온도 샘플링 | logits/T에서 표본 추출 | T>0이며 품질은 과제별로 비교 |
| top-k | 상위 k개 재정규화 | 낮은 순위 제외가 오류 제거를 뜻하지 않음 |
| top-p | 누적 확률 기준 후보 선택 | 후보 수가 달라지며 템플릿·분포에 의존 |
| 빔 탐색 | 제한된 후보의 누적 점수 탐색 | 전역 최적이나 번역 품질 보장 없음 |
| 대조 탐색 | 확률과 표현 중복 점수 결합 | 추가 계산과 문맥 비교 구현을 확인 |

### 좋은 방법

온도·top-k·top-p·반복 패널티는 정해진 최적 조합이 없다. 과제별 정확성·다양성·반복률을 같은 프롬프트와 예산에서 비교한다. 반복 패널티를 고르기 전에 코드의 음수 로짓 처리부터 확인한다.

### 핵심 내용

GPT의 생성은 다음으로 이루어진다.

1. 프롬프트에서 다음 토큰의 로짓을 계산한다.
2. 온도·후보 필터 등 선택 규칙을 적용한다.
3. 정규화한 분포에서 샘플링하거나 argmax를 고른다.
4. 토큰을 추가하고 EOS·길이 등의 종료 조건까지 반복한다.

### 참고 문헌

1. Holtzman, A., et al. (2020). "The Curious Case of Neural Text Degeneration."
2. Su, Y., et al. (2022). "A Contrastive Framework for Neural Text Generation."

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
GPT의 다음 토큰 목표와 BERT의 MLM을 비교하라.

</div>

??? success "풀이"
    인과 언어 모델은 $-\sum_t\log P(x_t\mid x_{<t})$를 최소화한다. BERT는 손상 입력의 양방향 문맥에서 선택 위치를 복원한다. 확률의 조건과 대상 위치가 다르며 어느 구조가 모든 이해·생성 과제에서 우수한지를 이 목표만으로 단정하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
초기 GPT 모델의 역사와 공개 정보의 범위를 설명하라.

</div>

??? success "풀이"
    GPT-1은 사전 학습·미세 조정, GPT-2는 대규모 언어 모델과 제로샷 평가, GPT-3은 프롬프트 기반 퓨샷 실험을 대표적으로 보여 주었다. GPT-4 기술 보고는 텍스트·이미지 입력을 설명하지만 계수 수·구조 세부를 공개하지 않았다. 공개되지 않은 추정을 확정된 사양으로 넣지 않는다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
파이토치에서 간단한 GPT 방식 디코더 블록을 구현하라.

</div>

??? success "풀이"
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

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
“창발 능력”의 관찰에서 평가 지표와 모델 규모를 어떻게 해석해야 하는가?

</div>

??? success "풀이"
    일부 과제 점수가 규모에 따라 급격히 변한다는 관찰을 가리킨다. 불연속 지표나 작은 표본에서도 그런 형태가 생길 수 있어 연속 로스·부분 점수와 불확실성을 함께 본다. 특정 계수 수를 능력 발현의 보편적인 임계로 제시해서는 안 된다.

## 정리하며

GPT의 인과 목표, 웨이트 공유·KV 캐시와 토큰 선택 규칙을 연결한다. 모델의 역사적 보고와 현재 제품 사양, 학습 계산 최적화와 일반적인 능력 보장을 구분한다. 생성 코드는 반환 형식·장치·모드·종료 조건과 필터의 실제 동작을 확인해야 한다.

**참고 문헌**

1. Radford, A., et al. (2018). "Improving Language Understanding by Generative Pre-Training." (GPT-1)
2. Radford, A., et al. (2019). "Language Models are Unsupervised Multitask Learners." (GPT-2)
3. Brown, T., et al. (2020). "Language Models are Few-Shot Learners." NeurIPS. (GPT-3)
4. Hoffmann, J., et al. (2022). "Training Compute-Optimal Large Language Models." (Chinchilla)
5. Ouyang, L., et al. (2022). "Training Language Models to Follow Instructions with Human Feedback." (InstructGPT)

---
