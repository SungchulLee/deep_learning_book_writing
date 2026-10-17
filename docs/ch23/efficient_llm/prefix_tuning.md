# Prefix tuning과 prompt tuning

[Prefix tuning](https://arxiv.org/abs/2101.00190)은 기반 모델을 동결하고 각 레이어가 참조할 연속 prefix 표현을 학습하는 방법이다. [Prompt tuning](https://arxiv.org/abs/2104.08691)은 주로 입력 임베딩 앞에 학습 벡터를 추가한다. 일반 텍스트 프롬프팅과 달리 두 방식은 학습 자료·로스·역전파로 새 파라메터를 최적화한다. prefix 언어 모델링이라는 사전 학습 마스크와도 다른 개념이다.

## 1. 텍스트 지시와 학습 벡터

```
"Translate English to French: The cat sat on the mat"
```

이것은 일반적인 텍스트 프롬프트다. 사람이 직접 만들 수도 있고 탐색으로 선택할 수도 있으며, 텍스트라는 이유만으로 반드시 수작업이어야 하는 것은 아니다.

```
[P1][P2][P3]...[Pn] "The cat sat on the mat"
```

P1 등의 표시는 가상 벡터의 개념도이며 어휘의 실제 토큰 ID나 모델 호출이 아니다. prefix를 넣으면 기반 웨이트가 고정이어도 어텐션 점수·출력은 달라진다. ‘어텐션 웨이트를 바꾸지 않는다’는 표현은 학습되는 사영 행렬과 동적으로 계산되는 어텐션 확률을 혼동할 수 있다.

## 2. 키·값 prefix의 모양

각 레이어에서 K'=[P_K;K], V'=[P_V;V]로 연결하고 Q가 이 추가 위치를 참조하게 할 수 있다. prefix는 단어의 출력 목표가 아니라 추가 조건이다. 마스크·위치·cache 형식을 해당 기반 모델과 맞춰야 한다.

MHA의 L레이어·모델 차원 d·prefix 길이 l에서 직접 저장하는 KV 값은 2lLd개다. GQA·MQA에서는 쿼리 head 수가 아니라 실제 KV head 수에 따른다. 예를 들어 L=24, d=1024, l=10이면 직접 KV는 491,520개다. 이는 MLP로 리파라메리제이션한 학습 모듈의 전체 파라메터 수가 아니다.

## 3. PrefixEncoder와 기반 연결

```python
import torch
import torch.nn as nn
from typing import Tuple, Optional, List

class PrefixEncoder(nn.Module):
    """
    모든 레이어의 앞가지를 부호로 만든다.
    
    작은 여러 레이어 인식개로 배울 수 있는 들임 묻힘에서 앞가지 묻힘을
    만든다. 이렇게 다시 파라메터화하면 익히기가 더 안정된다.
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
        
        # 전체 차원: 레이어 수 * 2(K와 V) * 머리 수 * 머리 차원
        self.total_dim = num_layers * 2 * num_heads * head_dim
        
        # 배울 수 있는 앞가지 묻힘
        self.prefix_tokens = nn.Embedding(prefix_length, hidden_dim)
        
        # 다시 파라메터화하는 여러 레이어 인식개(익히기가 나아진다)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, self.total_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        모든 레이어의 앞가지 열쇠-값 짝을 만든다.
        
        반환값:
            prefix_keys: [배치, 레이어 수, 머리 수, 앞가지 길이, 머리 차원]
            prefix_values: [배치, 레이어 수, 머리 수, 앞가지 길이, 머리 차원]
        """
        device = self.prefix_tokens.weight.device
        
        # 앞가지 토막 번호를 얻는다
        prefix_ids = torch.arange(self.prefix_length, device=device)
        prefix_ids = prefix_ids.unsqueeze(0).expand(batch_size, -1)
        
        # 묻고 바꾼다
        prefix_emb = self.prefix_tokens(prefix_ids)  # [배치, 앞가지 길이, 숨은]
        prefix = self.mlp(prefix_emb)  # [배치, 앞가지 길이, 전체 차원]
        prefix = self.dropout(prefix)
        
        # 꼴 바꾸기: [배치, 앞가지 길이, 레이어, 2, 머리, 머리 차원]
        prefix = prefix.view(
            batch_size,
            self.prefix_length,
            self.num_layers,
            2,
            self.num_heads,
            self.head_dim
        )
        
        # [배치, 레이어, 머리, 앞가지 길이, 머리 차원, 2]로 자리를 바꾼다
        prefix = prefix.permute(0, 2, 4, 1, 5, 3)
        
        # 열쇠와 값으로 쪼갠다
        prefix_keys = prefix[..., 0].contiguous()
        prefix_values = prefix[..., 1].contiguous()
        
        return prefix_keys, prefix_values

class PrefixTuningModel(nn.Module):
    """
    미리 익힌 모델에 앞가지 다듬기를 더하는 감개.
    
    바탕 모델을 얼리고 앞가지 파라메터만 익힌다.
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
        
        # 앞가지 인코더를 만든다
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
        
        # 앞가지에 맞춰 어텐션 가림막을 넓힌다
        if attention_mask is not None:
            prefix_mask = torch.ones(
                batch_size, self.prefix_length,
                device=attention_mask.device,
                dtype=attention_mask.dtype
            )
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        
        # 앞가지를 곁들여 피드포워드한다(짜기는 바탕 모델에 따라 다르다)
        # 이는 간추린 겉면이며, 실제 짜기는
        # 어텐션 레이어마다 prefix_keys, prefix_values를 넣어야 한다
        return self.base_model(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=self._format_prefix(prefix_keys, prefix_values),
            **kwargs
        )
    
    def _format_prefix(self, prefix_keys, prefix_values):
        """허깅페이스 꼴 past_key_values에 맞게 앞가지를 꾸민다."""
        # 레이어마다 하나씩 (열쇠, 값) 짝의 목록을 돌려준다
        past_key_values = []
        for layer_idx in range(prefix_keys.size(1)):
            layer_key = prefix_keys[:, layer_idx]  # [배치, 머리, 앞가지 길이, 머리 차원]
            layer_value = prefix_values[:, layer_idx]
            past_key_values.append((layer_key, layer_value))
        return tuple(past_key_values)
```

PrefixEncoder는 embedding과 두 Linear의 MLP로 모든 레이어의 KV를 만든다. 출력 K·V는 각각 B×L×H×l×d_h다. 은닉 폭 h, F=2LHd_h이면 학습 파라메터는 lh+(h²+h)+(hF+F)개다. L=24, d=1024, l=10, h=512면 25,482,752개로 직접 KV의 491,520개보다 훨씬 많다.

리파라메리제이션이 최적화를 도울 수 있지만 항상 더 작은 공간·적은 파라메터·안정성을 보장하는 것은 아니다. 추론에서 고정된 prefix 값을 미리 계산해 둘 수도 있으나 현재 클래스는 매번 MLP를 호출한다.

PrefixTuningModel은 기반을 동결하고 과거 KV tuple로 prefix를 전달한다. 모든 인코더·디코더가 같은 tuple cache 인터페이스를 받지는 않는다. 모델·버전별 cache 객체·레이어/헤드 차원·위치와 마스크 길이를 맞춰야 하며 encoder–decoder의 여러 어텐션 종류도 자동 지원하는 것이 아니다. 기반을 동결해도 prefix까지 역전파할 경로는 유지해야 한다.

## 4. 어텐션 부품과 마스크

```python
class PrefixAttention(nn.Module):
    """
    앞가지를 받치는 여러 머리 어텐션.
    
    어텐션을 셈하기 앞서 열쇠와 값 앞에 앞가지를 붙인다.
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
        앞가지를 곁들일 수도 있는 피드포워드.
        
        인수:
            hidden_states: [배치, 차례 길이, d_model]
            attention_mask: [배치, 차례 길이] 또는 [배치, 1, 차례 길이, 전체 길이]
            prefix_key: [배치, 머리 수, 앞가지 길이, 머리 차원]
            prefix_value: [배치, 머리 수, 앞가지 길이, 머리 차원]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Q, K, V를 셈한다
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # [배치, 머리, 차례 길이, 머리 차원] 꼴로 바꾼다
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # K와 V 앞에 앞가지를 붙인다
        if prefix_key is not None and prefix_value is not None:
            k = torch.cat([prefix_key, k], dim=2)  # [배치, 머리, 앞가지+차례, 머리 차원]
            v = torch.cat([prefix_value, v], dim=2)
        
        # 어텐션 계산
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # 주의 가림을 적용한다
        if attention_mask is not None:
            # 앞가지에 맞춰 가림막을 넓힌다(앞가지는 늘 본다)
            if prefix_key is not None:
                prefix_len = prefix_key.size(2)
                if attention_mask.dim() == 2:
                    # [배치, 차례] -> [배치, 1, 1, 앞가지+차례]
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

PrefixAttention은 새 Q·K·V·출력 사영을 만드는 독립 부품이며 사전 학습 어텐션을 이식·동결하는 클래스가 아니다. 인과 마스크도 없으므로 자기회귀 모델에 그대로 넣으면 미래 토큰을 볼 수 있다.

코드는 prefix가 있고 2차원 마스크를 받은 경우에만 B×1×1×길이로 확장한다. prefix가 없는 일반 B×T 마스크는 score의 head·query 축에 올바르게 broadcast되지 않을 수 있다. prefix K·V는 둘 다 제공하고 길이·head·장치를 맞춰야 하며, 모두 차단된 행의 softmax도 처리해야 한다. 1=유효라는 마스크 규칙을 다른 API의 bool 차단 규칙과 구분한다.

## 5. 직접 prefix와 입력 prompt

```python
class DirectPrefixEncoder(nn.Module):
    """
    곧바른 앞가지 파라메터화(여러 레이어 인식개 없이).
    
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
        
        # 곧바로 배우는 파라메터
        # 꼴: [앞가지 길이, 레이어, 2, 머리, 머리 차원]
        self.prefix = nn.Parameter(
            torch.randn(prefix_length, num_layers, 2, num_heads, head_dim) * 0.01
        )
    
    def forward(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # 배치에 맞춰 넓힌다
        prefix = self.prefix.unsqueeze(0).expand(batch_size, -1, -1, -1, -1, -1)
        
        # [배치, 레이어, 머리, 앞가지 길이, 머리 차원, 2]로 자리를 바꾼다
        prefix = prefix.permute(0, 2, 4, 1, 5, 3)
        
        return prefix[..., 0].contiguous(), prefix[..., 1].contiguous()
```

DirectPrefixEncoder는 KV 값을 직접 학습한다. 비교 시 같은 길이뿐 아니라 리파라메리제이션 여부·KV head 수·추가 학습 모듈을 포함해 학습/배포 저장량을 구분한다.

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
        
        # 어텐션 가림막을 넓힌다
        if attention_mask is not None:
            prompt_mask = torch.ones(batch_size, self.num_virtual_tokens, device=device)
            attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)
        
        # 모델을 지나 피드포워드한다
        return self.base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs
        )
```

PromptTuning은 입력 임베딩 앞에 l×d 학습 벡터를 연결한다. embedding_dim이 기반 임베딩과 같아야 한다. _init_from_text는 pass라 텍스트 기반 초기화가 실제로 구현되어 있지 않다.

현재 forward는 kwargs의 labels를 확장하지 않는다. 인과 모델의 입력 길이는 l+T가 되었는데 목표를 원래 T 길이로 넘기면 정렬이 맞지 않는다. 가상 입력 위치의 목표를 −100으로 제외하는 등 모델별 로스 계약을 구성해야 한다. KV prefix는 입력 토큰을 직접 늘리는 방식과 목표 정렬이 다르다. 위치 한도·추가 KV·어텐션 비용도 함께 검토한다.

## 6. 학습 함수와 평가

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
    
    # 앞가지 파라메터만 익힐 수 있다
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # 가장 좋게 하개(앞가지 파라메터만)
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

train_prefix_tuning은 prefix_encoder를 갱신하지만 eval_dataloader는 사용하지 않는다. 검증·최적 체크포인트·저장·학습 재개는 구현하지 않으며 배치 평균 로스만 출력한다. labels·padding·위치·장치·cache API의 계약이 맞아야 outputs.loss를 얻을 수 있다. 빈 loader와 유효 목표가 없는 배치도 처리해야 한다.

모델 전체 train 호출은 동결 기반의 dropout도 활성화한다. 웨이트 동결과 실행 모드는 다르므로 의도한 학습 폴리시를 명시한다. 이 편집에서는 기반 로딩·학습·평가를 실행하지 않았다.

prefix 길이·MLP 폭·러닝레이트를 늘리면 언제나 품질이 좋아지는 것은 아니다. 같은 과제·데이터·평가 예산에서 prompt·prefix·LoRA·어댑터의 품질·전체 메모리·지연을 비교한다. 단순 파라메터 비율로 표현력이나 모든 과제의 성능 순위를 확정하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
직접 KV prefix와 MLP 리파라메리제이션의 파라메터 수를 비교하라.

</div>

??? success "풀이"

    MHA 직접 KV는 2lLd다. 현재 MLP는 lh+h²+h+hF+F이며 F=2Ld다. L=24,d=1024,l=10,h=512면 직접 값은 491,520개지만 MLP는 25,482,752개다. 생성된 KV 저장량과 학습 모듈 파라메터 수는 같지 않다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
prefix tuning·prompt tuning·prefix 언어 모델링을 구분하라.

</div>

??? success "풀이"

    prefix tuning은 주로 레이어별 KV 조건, prompt tuning은 입력 임베딩의 가상 벡터를 학습한다. prefix 언어 모델링은 prefix 내부 양방향·suffix 인과 어텐션과 목표를 정하는 사전 학습 방식이다. 이름이 비슷해도 무엇이 학습되고 어느 위치에서 로스를 계산하는지가 다르다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
PromptTuning에서 T 길이 labels를 그대로 넘길 때의 문제를 설명하라.

</div>

??? success "풀이"

    입력 임베딩 앞에 l개를 붙여 출력 위치는 l+T가 되므로 목표도 정렬해야 한다. 가상 위치를 제외하도록 −100 목표를 붙이는 등 기반 모델의 내부 shift와 일치하는 구성이 필요하다. 현재 forward에는 이 처리가 없으며 _init_from_text도 미구현이다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
train_prefix_tuning이 완성된 범용 인코더·디코더 학습 예제가 아닌 이유를 설명하라.

</div>

??? success "풀이"

    cache tuple·head·위치·마스크 지원은 모델별로 다르고 현재 함수는 검증 loader도 사용하지 않는다. 목표·입력 계약, 동결과 dropout 모드, 검증·저장·재개를 별도로 구성해야 한다. 구조 이름만으로 모든 기반 모델에 같은 방식으로 연결할 수는 없다.



## 정리하며

학습 prefix·입력 prompt·사전 학습의 prefix 마스크는 구분해야 한다. KV와 MLP의 파라메터 수, cache·마스크·위치·목표 정렬을 맞추고 실제 품질과 운영 비용으로 방법을 비교한다.

**참고 문헌**

1. Li, X. L., & Liang, P. (2021). "Prefix-Tuning: Optimizing Continuous Prompts for Generation." ACL.
2. Lester, B., Al-Rfou, R., & Constant, N. (2021). "The Power of Scale for Parameter-Efficient Prompt Tuning." EMNLP.
3. Liu, X., et al. (2022). "P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks."
