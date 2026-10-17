# BERT: 트랜스포머 기반 양방향 인코더 표현

BERT는 모든 인코더 레이어에서 양방향 문맥을 활용하도록 사전 학습한 언어 표현 모델이다. 마스크 언어 모델링(MLM)으로 일부 토큰을 복원하고, 원래 모델은 다음 문장 예측(NSP)도 함께 학습했다. 구조·학습 목표와 실제 체크포인트를 구분해 살펴본다.

## 1. 핵심 혁신

### 1. 양방향 맥락

인과 언어 모델은 토큰을 앞선 문맥으로 예측한다. BERT의 MLM은 손상된 입력의 왼쪽·오른쪽 문맥을 함께 사용한다. 한 위치만 가렸을 때 다음 조건부 확률로 설명할 수 있다.

$$
P(x_i | x_1, \ldots, x_{i-1}, x_{i+1}, \ldots, x_n)
$$

마스크 언어 모델(MLM) 사전 학습 목표로 이를 이룬다.

### 2. 사전 학습과 미세 조정 방식

BERT는 앞선 전이 학습 연구를 이어, 사전 학습한 양방향 인코더를 여러 과제에 미세 조정하는 방식을 널리 보였다.

1. **사전 학습**: 레이블 없는 큰 말뭉치에서 일반적인 언어 표현을 배운다
2. **미세 조정**: 구조를 거의 바꾸지 않고 특정 과제에 맞춘다

---

## 2. 구조

BERT는 여러 레이어의 트랜스포머 인코더를 쓴다.

$$
\text{BERT} = \text{TransformerEncoder}^L
$$

### 모델의 크기

| 모델 | 레이어 (L) | 은닉 차원 (H) | 헤드 (A) | 파라메터 |
|-------|------------|------------|-----------|------------|
| BERT-Base | 12 | 768 | 12 | 110M |
| BERT-Large | 24 | 1024 | 16 | 340M |

### 입력 표현

BERT의 입력은 임베딩 셋의 합이다.

$$
\mathbf{E}_{\text{input}} = \mathbf{E}_{\text{token}} + \mathbf{E}_{\text{segment}} + \mathbf{E}_{\text{position}}
$$

**특수 토큰:**

- `[CLS]`: 분류 토큰 (첫 자리)
- `[SEP]`: 구간을 가르는 토큰
- `[MASK]`: 가린 토큰의 표식
- `[PAD]`: 패딩 토큰

**입력 꼴:**

```
[CLS] Token1 Token2 ... [SEP] Token1 Token2 ... [SEP]
  ↓                       ↓                      ↓
Segment A                Segment B           (Segment B)
```

---

## 3. 사전 학습 목표

### 1. 마스크 언어 모델 (MLM)

원래 MLM은 입력 토큰 중 약 15%를 예측 대상으로 고른다. 이 중 일부는 원래 토큰 그대로 남으므로 전체 15%가 모두 [MASK]로 바뀌는 것은 아니다.

$$
\mathcal{L}_{\text{MLM}} = -\sum_{i \in \mathcal{M}} \log P_\theta(x_i\mid\widetilde{\mathbf{x}})
$$

여기서 $\mathcal{M}$은 예측 대상으로 고른 위치이며 $\widetilde{\mathbf{x}}$는 80-10-10 규칙으로 손상한 입력이다. 로스를 합으로 쓸지 대상 토큰 평균으로 쓸지도 학습 설정에서 정한다.

**입력 손상 방식 (80-10-10 규칙):**

- 80%: `[MASK]`으로 바꾼다
- 10%: 무작위 토큰으로 바꾼다
- 10%: 그대로 둔다

그러면 모델이 `[MASK]` 토큰만 다룰 줄 알게 되는 일을 막는다.

### 2. 다음 문장 예측 (NSP)

이진 분류이다. 문장 B가 실제로 A 다음 문장인가?

$$
\mathcal{L}_{\text{NSP}} = -[y \log P(\text{IsNext}) + (1-y) \log P(\text{NotNext})]
$$

**참고:** 나중 연구(RoBERTa)는 NSP가 필요 없을 수 있고 어떤 과제에서는 성능을 해칠 수도 있음을 보였다.

### MLM의 독립 가정

일반적인 MLM 로스는 같은 손상 입력에서 각 대상 위치의 정답 로그 확률을 별도로 더한다. 다음 곱은 그 학습 목표의 해석이며 실제 토큰들이 언어적으로 독립이라는 주장이 아니다.

$$
P(\mathbf{x}_{\mathcal{M}}\mid\widetilde{\mathbf{x}}) \approx \prod_{i \in \mathcal{M}} P_\theta(x_i\mid\widetilde{\mathbf{x}})
$$

같은 손상 입력을 처리하는 은닉 표현들은 셀프 어텐션으로 서로 상호작용한다. 다만 여러 정답 토큰을 순서대로 조건에 추가하는 자기 회귀 결합 확률을 직접 학습하는 것과는 다르다. MLM의 토큰 확률을 그대로 곱해 일반 언어 모델의 시퀀스 확률이나 퍼플렉시티로 해석해서는 안 된다.

### 합친 로스

$$
\mathcal{L} = \mathcal{L}_{\text{MLM}} + \mathcal{L}_{\text{NSP}}
$$

### 사전 학습 데이터와 계산

원래 BERT는 BooksCorpus와 영어 위키백과를 이용해 사전 학습했다. 논문은 배치 256개로 100만 단계를 사용하고 처음 90% 단계는 길이 128, 나머지는 길이 512를 사용했다고 설명한다. 모든 배치가 항상 512토큰이었던 것은 아니다. 세부 설정은 [원 논문의 학습 절차](https://arxiv.org/html/1810.04805v2)를 참고한다.

- **배치와 길이**: 시퀀스 256개이며 학습 단계에 따라 최대 길이가 달랐다.
- **옵티마이저**: Adam과 워밍업·선형 러닝레이트 감쇠를 사용했다.
- **어휘**: 체크포인트의 WordPiece 어휘와 대소문자 처리 규칙을 함께 확인한다.
- **학습 비용**: 논문의 하드웨어·설정에 대한 보고이며 현재 구현의 실행 시간으로 일반화하지 않는다.

---

## 4. PyTorch 구현

다음 사용자 정의 BertModel은 무작위로 초기화된다. main의 입력·레이블도 무작위이므로 출력 모양과 계수 수는 구조 확인용이며 사전 학습·분류 품질의 결과가 아니다.

attention_mask는 (B,L)에서 1이 유효, 0이 패딩이고 내부에서는 점수에 더하는 -10000 바이어스로 바꾼다. 유한한 바이어스는 모든 키가 패딩인 행을 안전하게 0으로 만드는 폴리시가 아니다. 반환 어텐션은 드롭아웃 이후라 학습 중 행 합이 항상 1인 것도 아니다.

BertForMaskedLM은 입력 임베딩과 어휘 투영 웨이트를 공유하지만 원래 MLM의 중간 dense·GELU·LayerNorm을 생략한다. labels는 선택한 예측 위치만 유효 ID로 두고 나머지는 -100으로 표시해야 한다. BertModel의 풀러는 MLM 로짓 계산에 사용하지 않으므로 MLM 로스만으로는 학습되지 않는다. 모든 labels가 제외되면 평균 로스의 유효 분모가 없다.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict

class BertEmbeddings(nn.Module):
    """BERT 임베딩 레이어: 토큰 + 구간 + 자리"""
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-12
    ):
        super().__init__()
        
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.segment_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        인수:
            input_ids: 토큰 번호 [batch_size, seq_len]
            token_type_ids: 구간 번호 [batch_size, seq_len]
            position_ids: 자리 번호 [batch_size, seq_len]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # 기본 자리 번호
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        
        # 기본 구간 번호 (모두 0)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        # 임베딩을 더한다
        embeddings = (
            self.token_embeddings(input_ids) +
            self.position_embeddings(position_ids) +
            self.segment_embeddings(token_type_ids)
        )
        
        # 레이어 정규화와 드롭아웃
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings

class BertSelfAttention(nn.Module):
    """BERT의 자기 주의(양방향)."""
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        assert hidden_size % num_attention_heads == 0
        
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """다중 머리 주의에 맞게 꼴을 바꾼다."""
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size)
        return x.permute(0, 2, 1, 3)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        인수:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, 1, 1, seq_len]
        """
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # 주의 점수
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # 주의 가림을 적용한다
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # 정규화
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # 값에 적용한다
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        
        batch_size, seq_len = context_layer.shape[:2]
        context_layer = context_layer.view(batch_size, seq_len, self.all_head_size)
        
        if output_attentions:
            return context_layer, attention_probs
        return context_layer, None

class BertLayer(nn.Module):
    """BERT 인코더 레이어 하나."""
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-12
    ):
        super().__init__()
        
        # 자기 주의
        self.attention = BertSelfAttention(
            hidden_size, num_attention_heads, dropout
        )
        self.attention_output = nn.Linear(hidden_size, hidden_size)
        self.attention_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        # 순전파
        self.intermediate = nn.Linear(hidden_size, intermediate_size)
        self.output = nn.Linear(intermediate_size, hidden_size)
        self.output_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """BERT 레이어를 지나는 피드포워드."""
        
        # 자기 주의
        attn_output, attn_weights = self.attention(
            hidden_states, attention_mask, output_attentions
        )
        attn_output = self.attention_output(attn_output)
        attn_output = self.dropout(attn_output)
        hidden_states = self.attention_norm(hidden_states + attn_output)
        
        # 순전파
        intermediate_output = self.intermediate(hidden_states)
        intermediate_output = F.gelu(intermediate_output)
        layer_output = self.output(intermediate_output)
        layer_output = self.dropout(layer_output)
        hidden_states = self.output_norm(hidden_states + layer_output)
        
        return hidden_states, attn_weights

class BertEncoder(nn.Module):
    """BERT 인코더 더미."""
    
    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            BertLayer(
                hidden_size,
                num_attention_heads,
                intermediate_size,
                dropout
            )
            for _ in range(num_layers)
        ])
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False
    ) -> Dict[str, torch.Tensor]:
        """모든 인코더 레이어를 지나는 피드포워드."""
        
        all_hidden_states = [] if output_hidden_states else None
        all_attentions = [] if output_attentions else None
        
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            
            hidden_states, attn_weights = layer(
                hidden_states, attention_mask, output_attentions
            )
            
            if output_attentions:
                all_attentions.append(attn_weights)
        
        if output_hidden_states:
            all_hidden_states.append(hidden_states)
        
        return {
            'last_hidden_state': hidden_states,
            'hidden_states': all_hidden_states,
            'attentions': all_attentions
        }

class BertPooler(nn.Module):
    """[CLS] 토큰 표현을 풀링한다."""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """[CLS] 토큰을 가져와 풀링을 적용한다."""
        cls_token = hidden_states[:, 0]
        pooled = self.dense(cls_token)
        pooled = self.activation(pooled)
        return pooled

class BertModel(nn.Module):
    """
    온전한 BERT 모형.
    
    아래쪽 과제의 바탕으로 쓸 수 있다.
    """
    
    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embeddings = BertEmbeddings(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            dropout=dropout
        )
        
        self.encoder = BertEncoder(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            dropout=dropout
        )
        
        self.pooler = BertPooler(hidden_size)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        피드포워드.
        
        인수:
            input_ids: 토큰 번호 [batch_size, seq_len]
            attention_mask: 실제 토큰은 1, 채움은 0 [batch_size, seq_len]
            token_type_ids: 구간 번호 [batch_size, seq_len]
            position_ids: 자리 번호 [batch_size, seq_len]
        """
        # 주의 레이어를 위한 주의 가림을 만든다
        if attention_mask is not None:
            # [batch, seq]를 [batch, 1, 1, seq]로 바꾼다
            # 0은 -inf로, 1은 0으로
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        else:
            extended_attention_mask = None
        
        # 임베딩
        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids
        )
        
        # 인코더
        encoder_outputs = self.encoder(
            hidden_states=embedding_output,
            attention_mask=extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )
        
        # 풀러
        pooled_output = self.pooler(encoder_outputs['last_hidden_state'])
        
        return {
            'last_hidden_state': encoder_outputs['last_hidden_state'],
            'pooler_output': pooled_output,
            'hidden_states': encoder_outputs.get('hidden_states'),
            'attentions': encoder_outputs.get('attentions')
        }

class BertForMaskedLM(nn.Module):
    """가린 언어 모형화 머리를 갖춘 BERT."""
    
    def __init__(self, config: dict):
        super().__init__()
        
        self.bert = BertModel(**config)
        self.cls = nn.Linear(config['hidden_size'], config['vocab_size'])
        
        # 임베딩과 웨이트를 묶는다
        self.cls.weight = self.bert.embeddings.token_embeddings.weight
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """가린 언어 모형 로스를 셈하는 피드포워드."""
        
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # 가린 언어 모형 예측
        prediction_scores = self.cls(outputs['last_hidden_state'])
        
        # 이름표가 있으면 로스를 셈한다
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                prediction_scores.view(-1, prediction_scores.size(-1)),
                labels.view(-1),
                ignore_index=-100  # 가리지 않은 토큰은 무시한다
            )
        
        return {
            'loss': loss,
            'logits': prediction_scores,
            'hidden_states': outputs['last_hidden_state']
        }

class BertForSequenceClassification(nn.Module):
    """수열 분류(이를테면 감성 분석)를 위한 BERT."""
    
    def __init__(self, config: dict, num_labels: int):
        super().__init__()
        
        self.bert = BertModel(**config)
        self.dropout = nn.Dropout(config.get('dropout', 0.1))
        self.classifier = nn.Linear(config['hidden_size'], num_labels)
        self.num_labels = num_labels
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """분류 로스를 셈하는 피드포워드."""
        
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # [CLS]의 풀링된 출력을 쓴다
        pooled_output = outputs['pooler_output']
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        # 로스를 계산한다
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # 회귀
                loss = F.mse_loss(logits.squeeze(), labels.squeeze())
            else:
                # 분류
                loss = F.cross_entropy(logits, labels)
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs['last_hidden_state']
        }

# 사용 예
if __name__ == "__main__":
    # BERT-Base 설정
    config = {
        'vocab_size': 30522,
        'hidden_size': 768,
        'num_layers': 12,
        'num_attention_heads': 12,
        'intermediate_size': 3072,
        'max_position_embeddings': 512,
        'type_vocab_size': 2,
        'dropout': 0.1
    }
    
    # 모델 생성
    model = BertModel(**config)
    
    # 예제 입력
    batch_size = 4
    seq_len = 128
    
    input_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    attention_mask[:, -10:] = 0  # 채움을 흉내 낸다
    token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    token_type_ids[:, 64:] = 1  # 둘째 구간
    
    # 순전파
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids
    )
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Last hidden state shape: {outputs['last_hidden_state'].shape}")
    print(f"Pooler output shape: {outputs['pooler_output'].shape}")
    
    # 파라메터 개수 세기
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    # 분류 모형을 시험한다
    print("\n--- Testing Classification Model ---")
    classifier = BertForSequenceClassification(config, num_labels=2)
    
    labels = torch.randint(0, 2, (batch_size,))
    cls_outputs = classifier(input_ids, attention_mask, token_type_ids, labels)
    
    print(f"Classification logits shape: {cls_outputs['logits'].shape}")
    print(f"Classification loss: {cls_outputs['loss'].item():.4f}")
```

**출력:**

```
Input shape: torch.Size([4, 128])
Last hidden state shape: torch.Size([4, 128, 768])
Pooler output shape: torch.Size([4, 768])

Total parameters: 109,482,240

--- Testing Classification Model ---
Classification logits shape: torch.Size([4, 2])
Classification loss: 0.5852
```

---

## 5. 다운스트림 과제를 위한 미세 조정

아래 세 래퍼는 hidden_size=768을 고정하므로 다른 폭의 인코더에는 출력 레이어도 맞춰야 한다. 제공한 BertModel에 학습 웨이트를 읽는 코드는 없으며, 이미 사전 학습한 객체를 넘기지 않는 한 미세 조정이 아니다. NER에는 서브워드·특수 토큰·PAD의 라벨 정렬이, 추출형 QA에는 질문·PAD 제외와 유효한 답 구간 선택이 필요하다.

### 텍스트 분류

```python
# [CLS] 토큰에 분류 머리를 얹는다
class BertClassifier(nn.Module):
    def __init__(self, bert_model, num_classes):
        super().__init__()
        self.bert = bert_model
        self.classifier = nn.Linear(768, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        cls_output = outputs['pooler_output']
        return self.classifier(cls_output)
```

### 토큰 분류 (개체명 인식)

```python
# 토큰마다 분류 머리를 얹는다
class BertNER(nn.Module):
    def __init__(self, bert_model, num_labels):
        super().__init__()
        self.bert = bert_model
        self.classifier = nn.Linear(768, num_labels)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        sequence_output = outputs['last_hidden_state']
        return self.classifier(sequence_output)
```

### 질문 답변

```python
# 시작 자리와 끝 자리를 맞힌다
class BertQA(nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.bert = bert_model
        self.qa_outputs = nn.Linear(768, 2)  # 시작, 끝
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        logits = self.qa_outputs(outputs['last_hidden_state'])
        start_logits, end_logits = logits.split(1, dim=-1)
        return start_logits.squeeze(-1), end_logits.squeeze(-1)
```

---

## 6. BERT의 변형

| 모델 | 대표 변화 | 비교 시 확인할 점 |
|---|---|---|
| RoBERTa | NSP 제외, 동적 마스킹·학습 조건 변경 | 목표 외의 데이터·계산도 달라짐 |
| ALBERT | 임베딩 분해·레이어 간 공유·문장 순서 예측 | 설정별 계수 수와 실제 연산량은 별개 |
| DistilBERT | 지식 증류와 축소된 인코더 | 품질·속도 비율은 과제와 장치별로 비교 |
| ELECTRA | 교체 토큰 판별 | 생성기·판별기의 학습 비용을 함께 고려 |
| DeBERTa | 내용·위치를 분리한 어텐션 | 모델 버전과 학습 목표를 구분 |
| SpanBERT | 구간 마스킹·경계 기반 예측 | 구간 중심 과제에서 조건을 맞춰 비교 |

### 사전 학습 모델 지형에서 BERT의 자리

양방향 인코더는 문장 전체가 주어진 분류·추출 등에 유용하다. BERT의 MLM은 자기 회귀 다음 토큰 생성과 다른 목표이며 바로 같은 생성 루프를 적용할 수는 없다. 그러나 생성용 변형의 가능성을 부정하거나 인과 모델이 이해 과제에서 항상 열세라고 단정해서도 안 된다.

---

## 7. BERT로 하는 텍스트 분류

이 절은 별도의 Hugging Face 예제다. BertModel 이름을 다시 가져오므로 앞의 사용자 정의 클래스와 구분한다. from_pretrained는 실제 체크포인트를 읽고 새 분류 헤드는 별도로 초기화한다.

train_classifier는 워밍업 없이 처음 러닝레이트부터 선형 감쇠한다. 로스는 배치 평균들의 단순 평균이며, 반환값은 마지막 모델로서 저장한 최고 검증 F1 모델을 다시 불러오지 않는다. 검증 F1이 초기 best_f1=0을 넘지 않으면 체크포인트가 저장되지 않을 수도 있다.

freeze_bert는 경사 갱신만 막는다. model.train()을 호출하면 인코더의 드롭아웃도 켜져 고정 피처를 매번 동일하게 만드는 설정은 아니다. predict의 기본 device는 CPU인데 학습은 CUDA를 선택할 수 있어 모델과 입력 장치를 맞춰야 한다. 추론 토큰화의 길이도 학습 max_length=128과 일관되게 지정한다.

BertClassifierWithPooling의 cls는 원시 첫 위치 출력으로 pooler_output과 다르다. concat은 마지막 네 레이어가 있어야 선언한 4배 폭과 맞고, max는 은닉 텐서를 제자리 변경한다. 지원하지 않는 pooling 이름이나 전부 PAD인 입력은 별도 검증이 필요하다. train_multilabel은 로스와 예측을 계산할 뿐 optimizer 갱신을 수행하지 않는다.

### 분류를 위한 구조

```
Input: [CLS] This movie was great! [SEP]
          ↓
    BERT Encoder (12 layers)
          ↓
    [CLS] hidden state
          ↓
    Classification Head
          ↓
    Softmax → Probabilities
```

### 파이토치 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

class BertClassifier(nn.Module):
    """BERT 기반 글 분류기."""
    
    def __init__(
        self,
        model_name: str = 'bert-base-uncased',
        num_labels: int = 2,
        dropout: float = 0.1,
        freeze_bert: bool = False
    ):
        super().__init__()
        
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
    
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        피드포워드.
        
        인수:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            token_type_ids: [batch, seq_len] (선택)
        
        반환값:
            logits: [batch, num_labels]
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # [CLS] 토큰 표현을 쓴다
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        return self.classifier(pooled_output)

class TextClassificationDataset(Dataset):
    """글 분류를 위한 데이터셋."""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def train_epoch(model, dataloader, optimizer, scheduler, device):
    """한 에폭을 학습한다."""
    model.train()
    total_loss = 0
    predictions, true_labels = [], []
    
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        logits = model(input_ids, attention_mask)
        loss = F.cross_entropy(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        predictions.extend(logits.argmax(dim=-1).cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
    
    acc = accuracy_score(true_labels, predictions)
    return total_loss / len(dataloader), acc

@torch.no_grad()
def evaluate(model, dataloader, device):
    """모형을 평가한다."""
    model.eval()
    total_loss = 0
    predictions, true_labels = [], []
    
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        logits = model(input_ids, attention_mask)
        loss = F.cross_entropy(logits, labels)
        
        total_loss += loss.item()
        predictions.extend(logits.argmax(dim=-1).cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
    
    acc = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted')
    
    return total_loss / len(dataloader), acc, f1

def train_classifier(
    train_texts, train_labels,
    val_texts, val_labels,
    model_name='bert-base-uncased',
    num_labels=2,
    epochs=3,
    batch_size=16,
    learning_rate=2e-5,
    max_length=128
):
    """온전한 학습 파이프라인."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 토큰 나누개와 모형
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertClassifier(model_name, num_labels).to(device)
    
    # 데이터셋
    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # 최적화기와 일정 조정기
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_steps
    )
    
    # 학습 루프
    best_f1 = 0
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, device)
        
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), 'best_model.pt')
    
    return model, tokenizer

class BertClassifierWithPooling(nn.Module):
    """여러 풀링 방법을 갖춘 BERT 분류기."""
    
    def __init__(self, model_name, num_labels, pooling='cls'):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.pooling = pooling
        
        hidden_size = self.bert.config.hidden_size
        if pooling == 'concat':
            hidden_size *= 4  # 마지막 네 레이어
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_labels)
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids, attention_mask,
            output_hidden_states=True
        )
        
        if self.pooling == 'cls':
            pooled = outputs.last_hidden_state[:, 0]
        
        elif self.pooling == 'mean':
            # 토큰에 대한 평균 풀링
            token_embeddings = outputs.last_hidden_state
            attention_mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_embeddings = (token_embeddings * attention_mask_expanded).sum(1)
            sum_mask = attention_mask_expanded.sum(1).clamp(min=1e-9)
            pooled = sum_embeddings / sum_mask
        
        elif self.pooling == 'max':
            # 최대 풀링
            token_embeddings = outputs.last_hidden_state
            token_embeddings[attention_mask == 0] = -1e9
            pooled = token_embeddings.max(dim=1)[0]
        
        elif self.pooling == 'concat':
            # 마지막 네 레이어의 [CLS]를 이어 붙인다
            hidden_states = outputs.hidden_states
            pooled = torch.cat([h[:, 0] for h in hidden_states[-4:]], dim=-1)
        
        return self.classifier(pooled)

# 여러 이름표 분류
class BertMultiLabelClassifier(nn.Module):
    """여러 이름표 분류를 위한 BERT."""
    
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        logits = self.classifier(outputs.pooler_output)
        return logits  # 소프트맥스 없음. BCEWithLogitsLoss를 쓴다

def train_multilabel(model, batch, device, threshold=0.5):
    """여러 이름표 분류를 위한 학습 단계."""
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device).float()
    
    logits = model(input_ids, attention_mask)
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    
    # 예측
    preds = (torch.sigmoid(logits) > threshold).int()
    
    return loss, preds

# 사용 예
if __name__ == "__main__":
    # 예제 데이터
    train_texts = [
        "This movie was amazing!",
        "Terrible waste of time.",
        "Pretty good overall.",
        "Not worth watching."
    ]
    train_labels = [1, 0, 1, 0]  # 1은 긍정, 0은 부정
    
    val_texts = ["Great film!", "Boring movie."]
    val_labels = [1, 0]
    
    # 학습
    model, tokenizer = train_classifier(
        train_texts, train_labels,
        val_texts, val_labels,
        epochs=2,
        batch_size=2
    )
    
    # 추론
    @torch.no_grad()
    def predict(model, tokenizer, text, device='cpu'):
        model.eval()
        encoding = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        logits = model(encoding['input_ids'].to(device), encoding['attention_mask'].to(device))
        probs = F.softmax(logits, dim=-1)
        return probs.cpu().numpy()
    
    print("\nPredictions:")
    for text in ["This is wonderful!", "This is terrible!"]:
        probs = predict(model, tokenizer, text)
        print(f"  '{text}': {probs}")
```

### 좋은 방법

러닝레이트·배치 크기·길이·에포크·워밍업·웨이트 감쇠는 데이터와 모델에 따라 검증셋에서 선택한다. 코드의 소규모 4개 학습·2개 검증 문장으로 일반적인 성능을 판단할 수 없고, 최종 평가는 별도 테스트셋에서 수행한다. 특정 범위를 모든 과제의 충분조건으로 제시하지 않는다.

### 핵심 내용

BERT 분류는 다음으로 이루어진다.

1. [CLS]와 [SEP]을 넣어 입력을 토큰으로 나눈다
2. [CLS] 표현을 꺼낸다
3. 분류 헤드를 적용한다
4. 교차 엔트로피 로스로 미세 조정한다

### 참고 문헌

1. Devlin, J., et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers."
2. Sun, C., et al. (2019). "How to Fine-Tune BERT for Text Classification."

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
원래 BERT의 MLM과 NSP 목표를 설명하라.

</div>

??? success "풀이"
    MLM은 선택한 약 15% 위치의 원래 토큰을 손상 입력에서 예측한다. 선택된 위치 중 80%를 MASK, 10%를 무작위 토큰으로 바꾸고 10%는 그대로 둔다. NSP는 두 구간의 이어짐을 분류한다. NSP 없이 학습한 후속 모델도 있지만 모든 설정에서 NSP가 무의미하다고 결론내리지는 않는다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
BERT와 인과 언어 모델이 볼 수 있는 문맥과 학습 목표를 비교하라.

</div>

??? success "풀이"
    BERT는 손상된 전체 입력을 양방향으로 읽고 선택한 위치를 복원한다. 인과 모델은 각 출력의 미래 접근을 막고 다음 토큰을 예측한다. 이 차이는 문맥·확률 분해 방식이며 이해·생성 능력의 보편적인 우열을 뜻하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
이 페이지 BertModel 기본 설정의 파라메터 수를 구성별로 계산하라.

</div>

??? success "풀이"
    임베딩·그 LayerNorm은 23,837,184개, 인코더 블록 하나는 7,087,872개로 12레이어는 85,054,464개, 풀러는 590,592개다. 합은 109,482,240개다. 별도의 MLM·NSP·분류 헤드는 포함하지 않는다. BERT-Base의 약 110M 표기는 이러한 구성·반올림 기준과 함께 읽는다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
문장 분류·토큰 분류·추출형 QA에 필요한 출력과 정답 처리를 설명하라.

</div>

??? success "풀이"
    문장 분류는 표본별 로짓, 토큰 분류는 위치별 클래스 로짓, 추출형 QA는 시작·끝 위치 로짓을 사용한다. 토큰 과제는 서브워드·PAD·특수 토큰 라벨을 맞추고 QA는 원문 안의 유효 구간을 선택해야 한다. 전체 미세 조정 외에 동결·일부 갱신도 가능하므로 갱신 대상을 명시한다.

## 정리하며

BERT의 양방향 MLM, 구간·위치 표현과 과제별 머리를 연결해 이해한다. 사용자 정의 구조와 체크포인트를 읽는 예제를 구분하고, 마스크·라벨 정렬·풀러의 학습 여부와 모델 선택·추론 장치를 확인한다.

**참고 문헌**

1. Devlin, J., et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers." NAACL.
2. Liu, Y., et al. (2019). "RoBERTa: A Robustly Optimized BERT Pretraining Approach."
3. Clark, K., et al. (2020). "ELECTRA: Pre-training Text Encoders as Discriminators."
4. He, P., et al. (2021). "DeBERTa: Decoding-enhanced BERT with Disentangled Attention."

---
