# BERT: 트랜스포머 기반 양방향 인코더 표현
## 들어가며

BERT(트랜스포머 기반 양방향 인코더 표현)는 깊은 양방향 사전 학습을 들여와 자연어 처리를 뒤바꾸었다. 왼쪽에서 오른쪽으로만 읽거나 왼쪽에서 오른쪽 모형과 오른쪽에서 왼쪽 모형을 얕게 이어 붙이던 앞선 모형과 달리, BERT는 "가린 언어 모형"(MLM) 목표로 참된 양방향 표현 학습을 이룬다.

## 핵심 혁신

### 1. 양방향 맥락

GPT 같은 앞선 언어 모형은 한 방향(왼쪽에서 오른쪽)이었다. BERT는 왼쪽과 오른쪽 맥락을 한꺼번에 조건으로 삼는다.

$$
P(x_i | x_1, \ldots, x_{i-1}, x_{i+1}, \ldots, x_n)
$$

가린 언어 모형(MLM) 사전 학습 목표로 이를 이룬다.

### 2. 사전 학습과 미세 조정 방식

BERT는 요즘의 두 단계 방식을 자리 잡게 했다.

1. **사전 학습**: 이름표 없는 큰 말뭉치에서 일반적인 언어 표현을 배운다
2. **미세 조정**: 구조를 거의 바꾸지 않고 특정 과제에 맞춘다

## 구조

BERT는 여러 층의 트랜스포머 인코더를 쓴다.

$$
\text{BERT} = \text{TransformerEncoder}^L
$$

### 모형의 크기

| 모형 | 층 (L) | 숨은 차원 (H) | 머리 (A) | 매개변수 |
|-------|------------|------------|-----------|------------|
| BERT-Base | 12 | 768 | 12 | 110M |
| BERT-Large | 24 | 1024 | 16 | 340M |

### 입력 표현

BERT의 입력은 임베딩 셋의 합이다.

$$
\mathbf{E}_{\text{input}} = \mathbf{E}_{\text{token}} + \mathbf{E}_{\text{segment}} + \mathbf{E}_{\text{position}}
$$

**특별 토큰:**

- `[CLS]`: 분류 토큰 (첫 자리)
- `[SEP]`: 구간을 가르는 토큰
- `[MASK]`: 가린 토큰의 자리 지킴이
- `[PAD]`: 채움 토큰

**입력 꼴:**

```
[CLS] Token1 Token2 ... [SEP] Token1 Token2 ... [SEP]
  ↓                       ↓                      ↓
Segment A                Segment B           (Segment B)
```

## 사전 학습 목표

### 1. 가린 언어 모형 (MLM)

입력 토큰의 15%를 무작위로 가리고 맞힌다.

$$
\mathcal{L}_{\text{MLM}} = -\sum_{i \in \mathcal{M}} \log P(x_i | \mathbf{x}_{\backslash \mathcal{M}})
$$

여기서 $\mathcal{M}$은 가린 자리의 집합이다.

**가리는 방식 (80-10-10 규칙):**

- 80%: `[MASK]`으로 바꾼다
- 10%: 무작위 토큰으로 바꾼다
- 10%: 그대로 둔다

그러면 모형이 `[MASK]` 토큰만 다룰 줄 알게 되는 일을 막는다.

### 2. 다음 문장 맞히기 (NSP)

이진 분류이다. 문장 B가 실제로 A 다음 문장인가?

$$
\mathcal{L}_{\text{NSP}} = -[y \log P(\text{IsNext}) + (1-y) \log P(\text{NotNext})]
$$

**참고:** 나중 연구(RoBERTa)는 NSP가 필요 없을 수 있고 어떤 과제에서는 성능을 해칠 수도 있음을 보였다.

### MLM의 독립 가정

MLM 목표는 조건부 독립 가정을 둔다. 가리지 않은 맥락이 주어지면 가린 토큰을 서로 독립으로 맞힌다.

$$
P(\mathbf{x}_{\mathcal{M}} | \mathbf{x}_{\backslash \mathcal{M}}) \approx \prod_{i \in \mathcal{M}} P(x_i | \mathbf{x}_{\backslash \mathcal{M}})
$$

이는 단순화이다. 실제로 가린 토큰끼리 서로 얽혀 있을 수 있다. XLNet(Yang 외, 2019)은 맞히는 토큰 사이의 의존을 잡아내는 순열 기반 학습으로 이를 다룬다.

### 합친 손실

$$
\mathcal{L} = \mathcal{L}_{\text{MLM}} + \mathcal{L}_{\text{NSP}}
$$

### 사전 학습 데이터와 계산

BERT는 BooksCorpus(8억 낱말)와 영어 위키백과(25억 낱말)로 약 40 세대 사전 학습했다. 학습의 핵심 세부는 다음과 같다.

- **배치 크기**: 수열 256개 × 토큰 512개 = 배치마다 토큰 131,072개
- **최적화기**: 학습률 예열과 선형 감쇠를 곁들인 Adam
- **학습 시간**: TPU 칩 16개로 나흘(BERT-Base), TPU 칩 64개로 나흘(BERT-Large)
- **어휘**: 토큰 30,522개의 WordPiece 토큰 나누개

## PyTorch 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict

class BertEmbeddings(nn.Module):
    """BERT 임베딩 층: 토큰 + 구간 + 자리"""
    
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
        
        # 층 정규화와 드롭아웃
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
    """BERT 인코더 층 하나."""
    
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
        """BERT 층을 지나는 앞먹임."""
        
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
        """모든 인코더 층을 지나는 앞먹임."""
        
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
        앞먹임.
        
        인수:
            input_ids: 토큰 번호 [batch_size, seq_len]
            attention_mask: 실제 토큰은 1, 채움은 0 [batch_size, seq_len]
            token_type_ids: 구간 번호 [batch_size, seq_len]
            position_ids: 자리 번호 [batch_size, seq_len]
        """
        # 주의 층을 위한 주의 가림을 만든다
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
        
        # 부호기
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
        
        # 임베딩과 가중치를 묶는다
        self.cls.weight = self.bert.embeddings.token_embeddings.weight
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """가린 언어 모형 손실을 셈하는 앞먹임."""
        
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # 가린 언어 모형 예측
        prediction_scores = self.cls(outputs['last_hidden_state'])
        
        # 이름표가 있으면 손실을 셈한다
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
        """분류 손실을 셈하는 앞먹임."""
        
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # [CLS]의 풀링된 출력을 쓴다
        pooled_output = outputs['pooler_output']
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        # 손실을 계산한다
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
    
    # 매개변수 개수 세기
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

## 아래쪽 과제를 위한 미세 조정

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

### 질의응답

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

## BERT의 변형

| 모형 | 핵심 변화 | 영향 |
|-------|-------------|--------|
| **RoBERTa** | NSP 없앰, 동적 가리기, 더 많은 데이터, 더 큰 배치 | BERT가 크게 덜 학습되었음을 보였다 |
| **ALBERT** | 인수분해한 임베딩, 층끼리 매개변수 함께 쓰기, 문장 순서 맞히기 | BERT-Large보다 매개변수가 18배 적다 |
| **DistilBERT** | 지식 증류로 40% 작게 | BERT 성능의 97%를 60% 속도로 |
| **ELECTRA** | MLM 대신 바뀐 토큰 찾기 | 표본을 더 아낀다. 모든 토큰이 신호를 준다 |
| **DeBERTa** | 얽힘 푼 주의(내용과 자리를 나눔), 개선된 가림 디코더 | SuperGLUE에서 최고 수준 |
| **SpanBERT** | 이어진 구간을 가림, 구간 경계 목표 | 뽑아내는 과제(질의응답, 상호 참조)에 더 낫다 |

### 사전 학습 모형 지형에서 BERT의 자리

BERT는 인코더만 쓰는 사전 학습과 미세 조정 방식을 자리 잡게 했지만 그 영향은 더 멀리 뻗는다. MLM 사전 학습 목표는 이해 과제에서 양방향 맥락이 한 방향(GPT 방식) 모형보다 나은 표현을 낸다는 것을 보였다. 그러나 BERT는 글을 자기 회귀로 지을 수 없어 생성 과제에는 쓰기 어렵다. 그 자리는 디코더만 쓰는(GPT) 구조와 인코더-디코더(T5, BART) 구조가 메운다.

## 요약

BERT는 요즘 자연어 처리를 지배하는 사전 학습과 미세 조정 방식을 자리 잡게 했다.

1. **양방향 맥락**: MLM이 깊은 양방향 표현을 가능케 한다
2. **전이 학습**: 사전 학습된 모형이 여러 과제로 옮겨 간다
3. **간단한 미세 조정**: 과제에 맞는 구조가 최소한만 필요하다
4. **튼튼한 기준선**: BERT의 변형들이 여전히 겨룰 만하다

## 참고 문헌

1. Devlin, J., et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers." NAACL.
2. Liu, Y., et al. (2019). "RoBERTa: A Robustly Optimized BERT Pretraining Approach."
3. Clark, K., et al. (2020). "ELECTRA: Pre-training Text Encoders as Discriminators."
4. He, P., et al. (2021). "DeBERTa: Decoding-enhanced BERT with Disentangled Attention."

---

## BERT로 하는 글 분류

#### 분류를 위한 구조

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

#### 파이토치 구현

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
        앞먹임.
        
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
            hidden_size *= 4  # 마지막 네 층
        
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
            # 마지막 네 층의 [CLS]를 이어 붙인다
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

#### 좋은 방법

1. **학습률**: BERT에서는 2e-5에서 5e-5
2. **세대**: 대개 2~4이면 넉넉하다
3. **배치 크기**: 16~32 (기울기 모으기를 쓰면 더 크게)
4. **예열**: 전체 단계의 10%
5. **가중치 감쇠**: 0.01
6. **최대 길이**: 과제에 따라 128~512

#### 간추림

BERT 분류는 다음으로 이루어진다.

1. [CLS]와 [SEP]을 넣어 입력을 토큰으로 나눈다
2. [CLS] 표현을 꺼낸다
3. 분류 머리를 적용한다
4. 교차 엔트로피 손실로 미세 조정한다

#### 참고 문헌

1. Devlin, J., et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers."
2. Sun, C., et al. (2019). "How to Fine-Tune BERT for Text Classification."

## 연습문제

**연습문제 1.**
BERT의 두 사전 학습 목표인 가린 언어 모형화(MLM)와 다음 문장 맞히기(NSP)를 설명하라.

??? success "연습문제 1 풀이"
    MLM은 토큰의 15%를 무작위로 가리고(80%는 [MASK], 10%는 무작위, 10%는 그대로) 본디 토큰을 맞힌다. 그래서 양방향 모형이 된다. NSP는 문장 둘이 주어졌을 때 둘째가 첫째 다음인지 맞힌다. NSP는 나중에 쓸모가 덜하다고 밝혀져 RoBERTa 같은 뒤이은 모형에서 빠졌다.

---

**연습문제 2.**
BERT는 양방향이고 GPT는 한 방향인 까닭은 무엇인가? 맞바꿈은 무엇인가?

??? success "연습문제 2 풀이"
    BERT는 토큰마다 모든 자리에 주의할 수 있게 하는 가림을 써서 양방향 맥락을 얻는다. GPT는 인과 가림(왼쪽에서 오른쪽만)을 쓴다. 양방향 맥락은 분류나 질의응답 같은 이해 과제(NLU)에 도움이 된다. 생성(NLG)에는 한 방향이 필요하다. BERT는 글을 자기 회귀로 지을 수 없고, GPT는 지을 수 있지만 이해 과제의 표현이 약하다.

---

**연습문제 3.**
BERT-Base($L=12, H=768, A=12$)의 매개변수 수를 셈하라.

??? success "연습문제 3 풀이"
    임베딩은 어휘(30522) × 768 + 자리(512) × 768 + 종류(2) × 768 = 약 2380만이다. 층마다 다중 머리 주의가 $4 \times 768^2$ = 236만, 순전파가 $2 \times 768 \times 3072$ = 472만, 층 정규화가 4 × 768 = 3천이다. 층마다 약 710만이고 12층이면 약 8500만이다. 모두 약 1억 1천만 개이다.

---

**연습문제 4.**
BERT를 (가) 문장 분류, (나) 토큰 분류(개체명 인식), (다) 질의응답에 각각 어떻게 미세 조정하는지 설명하라.

??? success "연습문제 4 풀이"
    (가) 분류는 [CLS] 토큰 표현에 선형 분류기를 얹는다. (나) 개체명 인식은 토큰마다의 표현에 토큰별 분류기를 얹는다. (다) 질의응답은 토큰 표현 위의 선형 머리 둘로 답 구간의 시작과 끝 자리를 맞힌다. 모든 과제에서 과제에 맞는 머리를 붙여 BERT 모형 전체를 미세 조정한다.
