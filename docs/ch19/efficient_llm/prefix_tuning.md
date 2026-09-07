# 앞가지 다듬기
## 학습 목표

- 앞가지 다듬기가 무게를 고치지 않고 모델을 맞추는 법을 이해한다
- 부호기와 풀개 모델 모두에 앞가지 다듬기를 짠다
- 앞가지 다듬기를 시킴말 다듬기와 다른 부드러운 시킴말 방법과 견준다
- 앞가지 길이와 매개변수 다시 매기기 전략을 정한다

## 들어가며

앞가지 다듬기(Li & Liang, 2021)는 변환기 층마다 열쇠와 값 앞에 익힐 수 있는 이어진 벡터("앞가지")를 붙이는, 매개변수를 아끼는 곱게 다듬기 방법이다. 띄엄띄엄한 시킴말과 달리 이 앞가지는 배운 묻힘이어서 일에 맞춘 앎을 더 효율적으로 담을 수 있다.

## 핵심 개념

### 띄엄띄엄한 시킴말과 이어진 시킴말

**띄엄띄엄한 시킴말**(예로부터의 것):
```
"Translate English to French: The cat sat on the mat"
```
낱말 곳간의 토막에 갇히고 손수 빚어야 한다.

**이어진 시킴말**(앞가지 다듬기):
```
[P1][P2][P3]...[Pn] "The cat sat on the mat"
```
이어진 공간에서 배운 묻힘이며 끝에서 끝까지 가장 좋게 한다.

### 어떻게 되는가

앞가지 다듬기는 눈길 무게를 고치는 대신 눈길이 보는 것을 고친다:

$$
\text{Attention}(Q, [P_K; K], [P_V; V])
$$

여기서 $P_K, P_V$은 열쇠와 값 앞에 붙이는 배울 수 있는 앞가지 묻힘이다.

## 수학적 바탕

### 보통의 눈길

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

### 앞가지를 붙인 눈길

길이 $l$인 앞가지에서:

$$
K' = [P_K; K] \in \mathbb{R}^{(l+n) \times d_k}
$$

$$
V' = [P_V; V] \in \mathbb{R}^{(l+n) \times d_v}
$$

$$
\text{Attention}(Q, K', V') = \text{softmax}\left(\frac{QK'^T}{\sqrt{d_k}}\right)V'
$$

앞가지 항목은 실제 토막이 모두 눈길을 줄 수 있는 "가상 토막"이 되는 셈이다.

### 매개변수의 수

층이 $L$개, 눈길 머리가 $H$개, 머리 차원이 $d_h$인 모델에서:

$$
\text{Prefix params} = l \times L \times 2 \times H \times d_h = 2 \times l \times L \times d_{model}
$$

**Example**: GPT-2 Medium ($L=24$, $d_{model}=1024$) with prefix length 10:

- Prefix parameters: $2 \times 10 \times 24 \times 1024 = 491,520$ (0.14% of 345M)

## 구현

### 기본 앞가지 단원

```python
import torch
import torch.nn as nn
from typing import Tuple, Optional, List


class PrefixEncoder(nn.Module):
    """
    모든 층의 앞가지를 부호로 만든다.
    
    작은 여러 층 인식개로 배울 수 있는 들임 묻힘에서 앞가지 묻힘을
    만든다. 이렇게 다시 매개변수화하면 익히기가 더 안정된다.
    """
    
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        prefix_length: int,
        hidden_dim: int = 512,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.prefix_length = prefix_length
        
        # 전체 차원: 층 수 * 2(K와 V) * 머리 수 * 머리 차원
        self.total_dim = num_layers * 2 * num_heads * head_dim
        
        # 배울 수 있는 앞가지 묻힘
        self.prefix_tokens = nn.Embedding(prefix_length, hidden_dim)
        
        # 다시 매개변수화하는 여러 층 인식개(익히기가 나아진다)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, self.total_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        모든 층의 앞가지 열쇠-값 짝을 만든다.
        
        반환값:
            prefix_keys: [묶음, 층 수, 머리 수, 앞가지 길이, 머리 차원]
            prefix_values: [묶음, 층 수, 머리 수, 앞가지 길이, 머리 차원]
        """
        device = self.prefix_tokens.weight.device
        
        # 앞가지 토막 번호를 얻는다
        prefix_ids = torch.arange(self.prefix_length, device=device)
        prefix_ids = prefix_ids.unsqueeze(0).expand(batch_size, -1)
        
        # 묻고 바꾼다
        prefix_emb = self.prefix_tokens(prefix_ids)  # [묶음, 앞가지 길이, 숨은]
        prefix = self.mlp(prefix_emb)  # [묶음, 앞가지 길이, 전체 차원]
        prefix = self.dropout(prefix)
        
        # 꼴 바꾸기: [묶음, 앞가지 길이, 층, 2, 머리, 머리 차원]
        prefix = prefix.view(
            batch_size,
            self.prefix_length,
            self.num_layers,
            2,
            self.num_heads,
            self.head_dim
        )
        
        # [묶음, 층, 머리, 앞가지 길이, 머리 차원, 2]로 자리를 바꾼다
        prefix = prefix.permute(0, 2, 4, 1, 5, 3)
        
        # 열쇠와 값으로 쪼갠다
        prefix_keys = prefix[..., 0].contiguous()
        prefix_values = prefix[..., 1].contiguous()
        
        return prefix_keys, prefix_values


class PrefixTuningModel(nn.Module):
    """
    미리 익힌 모델에 앞가지 다듬기를 더하는 감개.
    
    바탕 모델을 얼리고 앞가지 매개변수만 익힌다.
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        prefix_length: int = 20,
        hidden_dim: int = 512
    ):
        super().__init__()
        
        self.base_model = base_model
        self.prefix_length = prefix_length
        
        # 바탕 모델을 얼린다
        for param in base_model.parameters():
            param.requires_grad = False
        
        # 앞가지 부호기를 만든다
        self.prefix_encoder = PrefixEncoder(
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            prefix_length=prefix_length,
            hidden_dim=hidden_dim
        )
    
    def get_prefix(self, batch_size: int):
        """앞가지 열쇠-값 짝을 얻는다."""
        return self.prefix_encoder(batch_size)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None, **kwargs):
        batch_size = input_ids.size(0)
        
        # 앞가지를 만든다
        prefix_keys, prefix_values = self.get_prefix(batch_size)
        
        # 앞가지에 맞춰 눈길 가림막을 넓힌다
        if attention_mask is not None:
            prefix_mask = torch.ones(
                batch_size, self.prefix_length,
                device=attention_mask.device,
                dtype=attention_mask.dtype
            )
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        
        # 앞가지를 곁들여 앞먹임한다(짜기는 바탕 모델에 따라 다르다)
        # 이는 간추린 겉면이며, 실제 짜기는
        # 눈길 층마다 prefix_keys, prefix_values를 넣어야 한다
        return self.base_model(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=self._format_prefix(prefix_keys, prefix_values),
            **kwargs
        )
    
    def _format_prefix(self, prefix_keys, prefix_values):
        """허깅페이스 꼴 past_key_values에 맞게 앞가지를 꾸민다."""
        # 층마다 하나씩 (열쇠, 값) 짝의 목록을 돌려준다
        past_key_values = []
        for layer_idx in range(prefix_keys.size(1)):
            layer_key = prefix_keys[:, layer_idx]  # [묶음, 머리, 앞가지 길이, 머리 차원]
            layer_value = prefix_values[:, layer_idx]
            past_key_values.append((layer_key, layer_value))
        return tuple(past_key_values)
```

### 눈길 층과 아우르기

```python
class PrefixAttention(nn.Module):
    """
    앞가지를 받치는 여러 머리 눈길.
    
    눈길을 셈하기 앞서 열쇠와 값 앞에 앞가지를 붙인다.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        prefix_key: Optional[torch.Tensor] = None,
        prefix_value: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        앞가지를 곁들일 수도 있는 앞먹임.
        
        인수:
            hidden_states: [묶음, 차례 길이, d_model]
            attention_mask: [묶음, 차례 길이] 또는 [묶음, 1, 차례 길이, 전체 길이]
            prefix_key: [묶음, 머리 수, 앞가지 길이, 머리 차원]
            prefix_value: [묶음, 머리 수, 앞가지 길이, 머리 차원]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Q, K, V를 셈한다
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # [묶음, 머리, 차례 길이, 머리 차원] 꼴로 바꾼다
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # K와 V 앞에 앞가지를 붙인다
        if prefix_key is not None and prefix_value is not None:
            k = torch.cat([prefix_key, k], dim=2)  # [묶음, 머리, 앞가지+차례, 머리 차원]
            v = torch.cat([prefix_value, v], dim=2)
        
        # 어텐션 계산
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # 주의 가림을 적용한다
        if attention_mask is not None:
            # 앞가지에 맞춰 가림막을 넓힌다(앞가지는 늘 본다)
            if prefix_key is not None:
                prefix_len = prefix_key.size(2)
                if attention_mask.dim() == 2:
                    # [묶음, 차례] -> [묶음, 1, 1, 앞가지+차례]
                    prefix_mask = torch.ones(batch_size, prefix_len, device=attention_mask.device)
                    attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
                    attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            
            attn_weights = attn_weights.masked_fill(attention_mask == 0, float('-inf'))
        
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 값에 어텐션 적용
        attn_output = torch.matmul(attn_weights, v)
        
        # 꼴을 바꾸고 사영한다
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        attn_output = self.o_proj(attn_output)
        
        return attn_output
```

## 왜 매개변수를 다시 매기는가?

앞가지 묻힘을 곧바로 가장 좋게 하면 흔들릴 수 있다. 다층 퍼셉트론으로 다시 매기면:

1. **가장 좋게 하기의 지형이 나아진다** — 다층 퍼셉트론이 더 작은 공간에서 대응시킨다
2. **무게를 나눠 쓸 수 있다** — 다층 퍼셉트론 무게 한 벌이 모든 앞가지를 만든다
3. **익히기가 든든해진다** — 기울기가 다층 퍼셉트론의 비선형을 지나 흐른다

```python
class DirectPrefixEncoder(nn.Module):
    """
    곧바른 앞가지 매개변수화(여러 층 인식개 없이).
    
    더 단순하나 익히는 동안 덜 안정될 수 있다.
    """
    
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        prefix_length: int
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.prefix_length = prefix_length
        
        # 곧바로 배우는 매개변수
        # 꼴: [앞가지 길이, 층, 2, 머리, 머리 차원]
        self.prefix = nn.Parameter(
            torch.randn(prefix_length, num_layers, 2, num_heads, head_dim) * 0.01
        )
    
    def forward(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # 묶음에 맞춰 넓힌다
        prefix = self.prefix.unsqueeze(0).expand(batch_size, -1, -1, -1, -1, -1)
        
        # [묶음, 층, 머리, 앞가지 길이, 머리 차원, 2]로 자리를 바꾼다
        prefix = prefix.permute(0, 2, 4, 1, 5, 3)
        
        return prefix[..., 0].contiguous(), prefix[..., 1].contiguous()
```

## 견줌: 앞가지 다듬기와 시킴말 다듬기

| 갈래 | 앞가지 다듬기 | 시킴말 다듬기 |
|--------|---------------|---------------|
| 쓰이는 곳 | 층마다(K, V) | 들임 묻힘만 |
| Parameters | $2 \times l \times L \times d$ | $l \times d$ |
| 나타내는 힘 | 높음 | 낮음 |
| 성능 | 더 좋음(특히 자료가 적을 때) | 자료가 많을 때 좋음 |
| 복잡도 | 높음 | 더 단순 |

### 시킴말 다듬기 짜기

```python
class PromptTuning(nn.Module):
    """
    시킴말 다듬기: 들임 앞에만 붙는 배울 수 있는 묻힘.
    
    앞가지 다듬기보다 단순하나 나타내는 힘이 약하다.
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        num_virtual_tokens: int = 20,
        embedding_dim: int = 768,
        init_from_vocab: bool = True,
        init_text: str = None
    ):
        super().__init__()
        
        self.base_model = base_model
        self.num_virtual_tokens = num_virtual_tokens
        
        # 바탕 모델을 얼린다
        for param in base_model.parameters():
            param.requires_grad = False
        
        # 배울 수 있는 시킴말 묻힘
        self.prompt_embeddings = nn.Embedding(num_virtual_tokens, embedding_dim)
        
        # 고를 수 있음: 낱말 곳간에서 첫자리매김한다
        if init_from_vocab and init_text is not None:
            self._init_from_text(init_text)
    
    def _init_from_text(self, text: str):
        """글 토막으로 시킴말을 첫자리매김한다(토막내개가 필요하다)."""
        # 짜기에서는 글을 토막내고 묻힘을 베낄 것이다
        pass
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None, **kwargs):
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # 바탕 모델에서 들임 묻힘을 얻는다
        if hasattr(self.base_model, 'get_input_embeddings'):
            input_embeds = self.base_model.get_input_embeddings()(input_ids)
        else:
            input_embeds = self.base_model.embed_tokens(input_ids)
        
        # 시킴말 묻힘을 얻는다
        prompt_ids = torch.arange(self.num_virtual_tokens, device=device)
        prompt_embeds = self.prompt_embeddings(prompt_ids)
        prompt_embeds = prompt_embeds.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 시킴말을 들임에 잇는다
        inputs_embeds = torch.cat([prompt_embeds, input_embeds], dim=1)
        
        # 눈길 가림막을 넓힌다
        if attention_mask is not None:
            prompt_mask = torch.ones(batch_size, self.num_virtual_tokens, device=device)
            attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)
        
        # 모델을 지나 앞먹임한다
        return self.base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs
        )
```

## 웃매개변수 길잡이

### 앞가지 길이

| 앞가지 길이 | 매개변수(GPT-2 Medium) | 쓰임새 |
|---------------|---------------------------|----------|
| 5 | 약 25만 | 단순한 일 |
| 10 | 약 50만 | 붙박이 |
| 20 | 약 100만 | 복잡한 일 |
| 50 | 약 250만 | 담는 힘이 클 때 |
| 100 | 약 500만 | 곱게 다듬기에 가까움 |

**어림 규칙**: 10~20으로 시작하고 덜 맞으면 늘린다.

### 숨은 차원

다층 퍼셉트론으로 다시 매길 때:

- 숨은 차원이 작으면(256~512): 매개변수가 적고 덜 맞을 수 있다
- 숨은 차원이 크면(768~1024): 더 잘 나타내지만 지나치게 맞을 위험이 있다

### 배움 비율

온전한 곱게 다듬기보다 큰 것이 보통이다:

- 앞가지 다듬기: 1e-3 ~ 5e-3
- 온전한 곱게 다듬기: 1e-5 ~ 5e-5

## 쓰임새

### 알맞은 곳

1. **만들어 내기 일** — 간추리기, 옮김, 대화
2. **몇 발 배우기** — 익힘 자료가 적을 때
3. **여러 일 배우기** — 일마다 다른 앞가지
4. **미리 익힌 앎 지키기** — 얼린 바탕 모델

### 덜 알맞은 곳

1. **이름표가 많은 갈래 매기기** — 흔히 LoRA가 낫다
2. **아주 긴 차례** — 앞가지가 차례 길이를 늘린다
3. **부호기만의 일** — 본디 만들어 내기를 위해 꾸며졌다

## 완전한 학습 예제

```python
def train_prefix_tuning(
    base_model,
    train_dataloader,
    eval_dataloader,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    prefix_length: int = 20,
    num_epochs: int = 5,
    learning_rate: float = 3e-3,
    device: str = 'cuda'
):
    """앞가지 다듬기로 모델을 익힌다."""
    
    # 앞가지를 다듬은 모델을 만든다
    model = PrefixTuningModel(
        base_model=base_model,
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        prefix_length=prefix_length
    ).to(device)
    
    # 앞가지 매개변수만 익힐 수 있다
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # 가장 좋게 하개(앞가지 매개변수만)
    optimizer = torch.optim.AdamW(
        model.prefix_encoder.parameters(),
        lr=learning_rate,
        weight_decay=0.01
    )
    
    # 학습 루프
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")
    
    return model
```

## 요약

| 갈래 | 자세히 |
|--------|---------|
| **얼개** | 층마다 배울 수 있는 K, V를 앞에 붙인다 |
| **매개변수** | 모델의 약 0.1~1% |
| **알맞은 곳** | 만들어 내기, 몇 발, 여러 일 |
| **핵심 웃매개변수** | 앞가지 길이(10~50) |
| **익히기** | 곱게 다듬기보다 큰 배움 비율 |

## 참고 문헌

1. Li, X. L., & Liang, P. (2021). "Prefix-Tuning: Optimizing Continuous Prompts for Generation." ACL.
2. Lester, B., Al-Rfou, R., & Constant, N. (2021). "The Power of Scale for Parameter-Efficient Prompt Tuning." EMNLP.
3. Liu, X., et al. (2022). "P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks."

## 연습문제

**연습문제 1.**
LoRA의 고갱이 생각을 밝혀라. 모델의 좋음을 지키면서 익힐 매개변수를 어떻게 줄이는가?

??? success "연습문제 1 풀이"
    LoRA(낮은 계수 맞추기)는 미리 익힌 무게 행렬 $W_0 \in \mathbb{R}^{d \times k}$을 얼리고 낮은 계수 고침 $\Delta W = BA$을 더한다. 여기서 $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$이고 $r \ll \min(d, k)$이다. 앞으로 걸음은 $h = (W_0 + BA)x$이 된다. $A$과 $B$만 익히므로 익힐 매개변수가 $dk$개에서 $r(d+k)$개로 준다. $d = k = 4096$이고 $r = 8$인 흔한 층이면 $256\times$ 줄어든다. 고갱이 눈썰미는 곱게 다듬는 동안의 무게 고침이 타고난 계수가 낮다는 것이며, 그래서 낮은 계수 쪼갬이 종요로운 맞춤을 담아낸다.

---

**연습문제 2.**
LoRA, 앞가지 다듬기, 어댑터 층을 견주어라. 기억 공간, 미룸 빠르기, 일의 성능에서 맞바꿈은 무엇인가?

??? success "연습문제 2 풀이"
    | 방법 | 익힐 매개변수 | 미룸 덧짐 | 기억 공간 | 여러 일 |
    |--------|-----------------|-------------------|--------|------------|
    | **LoRA** | $\sim$0.1~1% | 없음(무게를 어울린다) | 낮음 | $A, B$ 행렬을 갈아 끼운다 |
    | **머리말 맞추기** | $\sim$0.1% | 조금 있음(덧붙은 토막) | 낮음 | 머리말 벡터를 갈아 끼운다 |
    | **맞춤개** | $\sim$1~5% | 어느 정도 있음(덧붙은 층) | 보통 | 맞춤개 단원을 갈아 끼운다 |

    $BA$을 $W_0$에 어울릴 수 있으므로 LoRA는 미룸 덧짐이 0이다. 앞가지 다듬기는 맥락 창을 먹는 가상 토막을 더한다. 어댑터는 늦음을 늘리는 병목 층을 더한다. 셋 다 대부분의 일에서 온전한 곱게 다듬기에 가까운 성능을 내며, 단순하고 효율적이어서 LoRA가 가장 널리 쓰인다.

---

**연습문제 3.**
매개변수 700억 모델을 온전히 곱게 다듬는 것이 대부분의 조직에 왜 실전에 맞지 않는가? 기억 공간 요구량을 수로 나타내어라.

??? success "연습문제 3 풀이"
    fp16으로 된 700억 매개변수 모델은 무게만 해도 $70 \times 10^9 \times 2$바이트 = 140GB가 든다. 곱게 다듬으려면 여기에 더해 가장 좋게 하개 상태(Adam은 매개변수마다 상태 2개를 담으므로 fp32으로 280GB), 기울기(fp16으로 140GB), 되짚기용 살림 값이 든다. 모두 GPU 기억 자리가 $\sim$700GB 넘게 든다. 기울기 되짚음 저장과 섞인 촘촘함을 써도 A100 80GB GPU가 8장 넘게 든다. LoRA은 익힐 매개변수를 $\sim$7000만 개로 줄여 가장 좋게 하개 상태와 기울기 기억 자리를 $1000\times$ 줄이므로 GPU 한두 장으로도 곱게 다듬을 수 있다.

---

**연습문제 4.**
양자화를 헤아린 LoRA(QLoRA)란 무엇이며 큰 말 모델 곱게 다듬기를 누구나 하게 만드는 데 왜 뜻깊은가?

??? success "연습문제 4 풀이"
    QLoRA은 밑 모델을 4비트로 수 줄이고 LoRA 맞춤개는 fp16/bf16으로 두어 둘을 아우른다. 밑 무게 $W_0$은 4비트 NormalFloat 꼴로 담기므로(매개변수마다 $\sim$0.5바이트) 700억 모델의 기억 자리가 140GB에서 $\sim$35GB으로 준다. LoRA 맞춤개는 익힘이 든든하도록 더 촘촘한 꼴로 남는다. 여기에 겹 수 줄이기(수 줄이기 상수를 다시 수 줄이기)와 쪽 넘김 가장 좋게 하개(가장 좋게 하개 상태가 치솟을 때 CPU 기억 자리를 씀)라는 새로움이 더해진다. 그래서 48GB GPU(A6000) 한 장으로 650억 모델을 곱게 다듬을 수 있고, 값비싼 여러 GPU 무리 없이도 연구자와 작은 조직이 큰 말 모델을 맞출 수 있게 된다.
