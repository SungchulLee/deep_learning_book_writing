# BERT: Bidirectional Encoder Representations from Transformers

## Introduction

BERT (Bidirectional Encoder Representations from Transformers) revolutionized NLP by introducing deep bidirectional pre-training. Unlike previous models that were either left-to-right or used shallow concatenation of left-to-right and right-to-left models, BERT uses a "masked language model" (MLM) objective to enable true bidirectional representation learning.

## Key Innovations

### 1. Bidirectional Context

Previous language models like GPT were unidirectional (left-to-right). BERT conditions on both left and right context simultaneously:

$$
P(x_i | x_1, \ldots, x_{i-1}, x_{i+1}, \ldots, x_n)
$$

This is achieved through the Masked Language Model (MLM) pre-training objective.

### 2. Pre-training + Fine-tuning Paradigm

BERT established the modern two-stage approach:
1. **Pre-training**: Learn general language representations on large unlabeled corpus
2. **Fine-tuning**: Adapt to specific tasks with minimal architecture changes

## Architecture

BERT uses a multi-layer Transformer encoder:

$$
\text{BERT} = \text{TransformerEncoder}^L
$$

### Model Sizes

| Model | Layers (L) | Hidden (H) | Heads (A) | Parameters |
|-------|------------|------------|-----------|------------|
| BERT-Base | 12 | 768 | 12 | 110M |
| BERT-Large | 24 | 1024 | 16 | 340M |

### Input Representation

BERT's input is a sum of three embeddings:

$$
\mathbf{E}_{\text{input}} = \mathbf{E}_{\text{token}} + \mathbf{E}_{\text{segment}} + \mathbf{E}_{\text{position}}
$$

**Special Tokens:**
- `[CLS]`: Classification token (first position)
- `[SEP]`: Separator between segments
- `[MASK]`: Placeholder for masked tokens
- `[PAD]`: Padding token

**Input Format:**

```
[CLS] Token1 Token2 ... [SEP] Token1 Token2 ... [SEP]
  ↓                       ↓                      ↓
Segment A                Segment B           (Segment B)
```

## Pre-training Objectives

### 1. Masked Language Model (MLM)

Randomly mask 15% of input tokens and predict them:

$$
\mathcal{L}_{\text{MLM}} = -\sum_{i \in \mathcal{M}} \log P(x_i | \mathbf{x}_{\backslash \mathcal{M}})
$$

Where $\mathcal{M}$ is the set of masked positions.

**Masking Strategy (80-10-10 rule):**
- 80%: Replace with `[MASK]`
- 10%: Replace with random token
- 10%: Keep unchanged

This prevents the model from only learning to handle `[MASK]` tokens.

### 2. Next Sentence Prediction (NSP)

Binary classification: Is sentence B the actual next sentence after A?

$$
\mathcal{L}_{\text{NSP}} = -[y \log P(\text{IsNext}) + (1-y) \log P(\text{NotNext})]
$$

**Note:** Later work (RoBERTa) showed NSP may not be necessary.

### Combined Loss

$$
\mathcal{L} = \mathcal{L}_{\text{MLM}} + \mathcal{L}_{\text{NSP}}
$$

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict


class BertEmbeddings(nn.Module):
    """BERT Embedding Layer: Token + Segment + Position"""
    
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
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            token_type_ids: Segment IDs [batch_size, seq_len]
            position_ids: Position IDs [batch_size, seq_len]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Default position IDs
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        
        # Default segment IDs (all zeros)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        # Sum embeddings
        embeddings = (
            self.token_embeddings(input_ids) +
            self.position_embeddings(position_ids) +
            self.segment_embeddings(token_type_ids)
        )
        
        # Layer norm and dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


class BertSelfAttention(nn.Module):
    """BERT Self-Attention (bidirectional)."""
    
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
        """Reshape for multi-head attention."""
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
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, 1, 1, seq_len]
        """
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # Attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Apply attention mask
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Normalize
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply to values
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        
        batch_size, seq_len = context_layer.shape[:2]
        context_layer = context_layer.view(batch_size, seq_len, self.all_head_size)
        
        if output_attentions:
            return context_layer, attention_probs
        return context_layer, None


class BertLayer(nn.Module):
    """Single BERT Encoder Layer."""
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-12
    ):
        super().__init__()
        
        # Self-attention
        self.attention = BertSelfAttention(
            hidden_size, num_attention_heads, dropout
        )
        self.attention_output = nn.Linear(hidden_size, hidden_size)
        self.attention_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        # Feed-forward
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
        """Forward pass through BERT layer."""
        
        # Self-attention
        attn_output, attn_weights = self.attention(
            hidden_states, attention_mask, output_attentions
        )
        attn_output = self.attention_output(attn_output)
        attn_output = self.dropout(attn_output)
        hidden_states = self.attention_norm(hidden_states + attn_output)
        
        # Feed-forward
        intermediate_output = self.intermediate(hidden_states)
        intermediate_output = F.gelu(intermediate_output)
        layer_output = self.output(intermediate_output)
        layer_output = self.dropout(layer_output)
        hidden_states = self.output_norm(hidden_states + layer_output)
        
        return hidden_states, attn_weights


class BertEncoder(nn.Module):
    """BERT Encoder Stack."""
    
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
        """Forward pass through all encoder layers."""
        
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
    """Pool the [CLS] token representation."""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Take [CLS] token and apply pooling."""
        cls_token = hidden_states[:, 0]
        pooled = self.dense(cls_token)
        pooled = self.activation(pooled)
        return pooled


class BertModel(nn.Module):
    """
    Complete BERT Model.
    
    Can be used as base for downstream tasks.
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
        Forward pass.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: 1 for real tokens, 0 for padding [batch_size, seq_len]
            token_type_ids: Segment IDs [batch_size, seq_len]
            position_ids: Position IDs [batch_size, seq_len]
        """
        # Create attention mask for attention layers
        if attention_mask is not None:
            # Convert [batch, seq] to [batch, 1, 1, seq]
            # 0 -> -inf, 1 -> 0
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        else:
            extended_attention_mask = None
        
        # Embeddings
        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids
        )
        
        # Encoder
        encoder_outputs = self.encoder(
            hidden_states=embedding_output,
            attention_mask=extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )
        
        # Pooler
        pooled_output = self.pooler(encoder_outputs['last_hidden_state'])
        
        return {
            'last_hidden_state': encoder_outputs['last_hidden_state'],
            'pooler_output': pooled_output,
            'hidden_states': encoder_outputs.get('hidden_states'),
            'attentions': encoder_outputs.get('attentions')
        }


class BertForMaskedLM(nn.Module):
    """BERT with Masked Language Modeling head."""
    
    def __init__(self, config: dict):
        super().__init__()
        
        self.bert = BertModel(**config)
        self.cls = nn.Linear(config['hidden_size'], config['vocab_size'])
        
        # Tie weights with embeddings
        self.cls.weight = self.bert.embeddings.token_embeddings.weight
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with MLM loss computation."""
        
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # MLM predictions
        prediction_scores = self.cls(outputs['last_hidden_state'])
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                prediction_scores.view(-1, prediction_scores.size(-1)),
                labels.view(-1),
                ignore_index=-100  # Ignore non-masked tokens
            )
        
        return {
            'loss': loss,
            'logits': prediction_scores,
            'hidden_states': outputs['last_hidden_state']
        }


class BertForSequenceClassification(nn.Module):
    """BERT for sequence classification (e.g., sentiment analysis)."""
    
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
        """Forward pass with classification loss."""
        
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Use [CLS] pooled output
        pooled_output = outputs['pooler_output']
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        # Compute loss
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Regression
                loss = F.mse_loss(logits.squeeze(), labels.squeeze())
            else:
                # Classification
                loss = F.cross_entropy(logits, labels)
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs['last_hidden_state']
        }


# Example usage
if __name__ == "__main__":
    # BERT-Base configuration
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
    
    # Create model
    model = BertModel(**config)
    
    # Sample input
    batch_size = 4
    seq_len = 128
    
    input_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    attention_mask[:, -10:] = 0  # Simulate padding
    token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    token_type_ids[:, 64:] = 1  # Second segment
    
    # Forward pass
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids
    )
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Last hidden state shape: {outputs['last_hidden_state'].shape}")
    print(f"Pooler output shape: {outputs['pooler_output'].shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    # Test classification model
    print("\n--- Testing Classification Model ---")
    classifier = BertForSequenceClassification(config, num_labels=2)
    
    labels = torch.randint(0, 2, (batch_size,))
    cls_outputs = classifier(input_ids, attention_mask, token_type_ids, labels)
    
    print(f"Classification logits shape: {cls_outputs['logits'].shape}")
    print(f"Classification loss: {cls_outputs['loss'].item():.4f}")
```

## Fine-tuning for Downstream Tasks

### Text Classification

```python
# Add classification head on [CLS] token
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

### Token Classification (NER)

```python
# Add classification head on each token
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

### Question Answering

```python
# Predict start and end positions
class BertQA(nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.bert = bert_model
        self.qa_outputs = nn.Linear(768, 2)  # start, end
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        logits = self.qa_outputs(outputs['last_hidden_state'])
        start_logits, end_logits = logits.split(1, dim=-1)
        return start_logits.squeeze(-1), end_logits.squeeze(-1)
```

## BERT Variants

| Model | Key Changes |
|-------|-------------|
| **RoBERTa** | Removes NSP, dynamic masking, more data |
| **ALBERT** | Parameter sharing, sentence-order prediction |
| **DistilBERT** | 40% smaller via knowledge distillation |
| **ELECTRA** | Replaced token detection instead of MLM |
| **DeBERTa** | Disentangled attention, enhanced mask decoder |

## Summary

BERT established the pre-train + fine-tune paradigm that dominates modern NLP:

1. **Bidirectional Context**: MLM enables deep bidirectional representations
2. **Transfer Learning**: Pre-trained models transfer to many tasks
3. **Simple Fine-tuning**: Minimal task-specific architecture
4. **Strong Baselines**: BERT variants remain competitive

## References

1. Devlin, J., et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers." NAACL.
2. Liu, Y., et al. (2019). "RoBERTa: A Robustly Optimized BERT Pretraining Approach."
3. Clark, K., et al. (2020). "ELECTRA: Pre-training Text Encoders as Discriminators."
4. He, P., et al. (2021). "DeBERTa: Decoding-enhanced BERT with Disentangled Attention."
