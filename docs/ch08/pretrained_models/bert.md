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

**Note:** Later work (RoBERTa) showed NSP may not be necessary and can even hurt performance on some tasks.

### Independence Assumption in MLM

The MLM objective makes a conditional independence assumption: masked tokens are predicted independently given the unmasked context:

$$
P(\mathbf{x}_{\mathcal{M}} | \mathbf{x}_{\backslash \mathcal{M}}) \approx \prod_{i \in \mathcal{M}} P(x_i | \mathbf{x}_{\backslash \mathcal{M}})
$$

This is a simplification—in reality, masked tokens can be correlated. XLNet (Yang et al., 2019) addresses this by using permutation-based training that captures dependencies among predicted tokens.

### Combined Loss

$$
\mathcal{L} = \mathcal{L}_{\text{MLM}} + \mathcal{L}_{\text{NSP}}
$$

### Pre-training Data and Compute

BERT was pre-trained on BooksCorpus (800M words) and English Wikipedia (2,500M words) for approximately 40 epochs. Key training details:

- **Batch size**: 256 sequences × 512 tokens = 131,072 tokens per batch
- **Optimizer**: Adam with learning rate warmup and linear decay
- **Training time**: 4 days on 16 TPU chips (BERT-Base), 4 days on 64 TPU chips (BERT-Large)
- **Vocabulary**: WordPiece tokenizer with 30,522 tokens

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

| Model | Key Changes | Impact |
|-------|-------------|--------|
| **RoBERTa** | Removes NSP, dynamic masking, more data, larger batches | Showed BERT was significantly undertrained |
| **ALBERT** | Factorized embeddings, cross-layer parameter sharing, sentence-order prediction | 18x fewer parameters than BERT-Large |
| **DistilBERT** | 40% smaller via knowledge distillation | 97% of BERT performance at 60% speed |
| **ELECTRA** | Replaced token detection instead of MLM | More sample-efficient; all tokens provide signal |
| **DeBERTa** | Disentangled attention (separate content/position), enhanced mask decoder | State-of-the-art on SuperGLUE |
| **SpanBERT** | Masks contiguous spans, span boundary objective | Better for extractive tasks (QA, coreference) |

### BERT's Position in the Pre-trained Model Landscape

BERT established the encoder-only pre-train + fine-tune paradigm, but its influence extends further. The MLM pre-training objective demonstrated that bidirectional context produces superior representations for understanding tasks compared to unidirectional (GPT-style) models. However, BERT cannot generate text autoregressively, which limits its applicability to generation tasks—this niche is filled by decoder-only (GPT) and encoder-decoder (T5, BART) architectures.

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

---

## Text Classification with BERT

#### Architecture for Classification

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

#### PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
from sklearn.metrics import accuracy_score, f1_score
import numpy as np


class BertClassifier(nn.Module):
    """BERT-based text classifier."""
    
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
        Forward pass.
        
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            token_type_ids: [batch, seq_len] (optional)
        
        Returns:
            logits: [batch, num_labels]
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        return self.classifier(pooled_output)


class TextClassificationDataset(Dataset):
    """Dataset for text classification."""
    
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
    """Train for one epoch."""
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
    """Evaluate model."""
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
    """Full training pipeline."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertClassifier(model_name, num_labels).to(device)
    
    # Datasets
    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_steps
    )
    
    # Training loop
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
    """BERT classifier with different pooling strategies."""
    
    def __init__(self, model_name, num_labels, pooling='cls'):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.pooling = pooling
        
        hidden_size = self.bert.config.hidden_size
        if pooling == 'concat':
            hidden_size *= 4  # Last 4 layers
        
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
            # Mean pooling over tokens
            token_embeddings = outputs.last_hidden_state
            attention_mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_embeddings = (token_embeddings * attention_mask_expanded).sum(1)
            sum_mask = attention_mask_expanded.sum(1).clamp(min=1e-9)
            pooled = sum_embeddings / sum_mask
        
        elif self.pooling == 'max':
            # Max pooling
            token_embeddings = outputs.last_hidden_state
            token_embeddings[attention_mask == 0] = -1e9
            pooled = token_embeddings.max(dim=1)[0]
        
        elif self.pooling == 'concat':
            # Concatenate last 4 layers' [CLS]
            hidden_states = outputs.hidden_states
            pooled = torch.cat([h[:, 0] for h in hidden_states[-4:]], dim=-1)
        
        return self.classifier(pooled)


# Multi-label classification
class BertMultiLabelClassifier(nn.Module):
    """BERT for multi-label classification."""
    
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        logits = self.classifier(outputs.pooler_output)
        return logits  # No softmax - use BCEWithLogitsLoss


def train_multilabel(model, batch, device, threshold=0.5):
    """Training step for multi-label classification."""
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device).float()
    
    logits = model(input_ids, attention_mask)
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    
    # Predictions
    preds = (torch.sigmoid(logits) > threshold).int()
    
    return loss, preds


# Example usage
if __name__ == "__main__":
    # Sample data
    train_texts = [
        "This movie was amazing!",
        "Terrible waste of time.",
        "Pretty good overall.",
        "Not worth watching."
    ]
    train_labels = [1, 0, 1, 0]  # 1=positive, 0=negative
    
    val_texts = ["Great film!", "Boring movie."]
    val_labels = [1, 0]
    
    # Train
    model, tokenizer = train_classifier(
        train_texts, train_labels,
        val_texts, val_labels,
        epochs=2,
        batch_size=2
    )
    
    # Inference
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

#### Best Practices

1. **Learning Rate**: 2e-5 to 5e-5 for BERT
2. **Epochs**: 2-4 usually sufficient
3. **Batch Size**: 16-32 (larger with gradient accumulation)
4. **Warmup**: 10% of total steps
5. **Weight Decay**: 0.01
6. **Max Length**: Task-dependent, 128-512

#### Summary

BERT classification involves:
1. Tokenize input with [CLS] and [SEP]
2. Extract [CLS] representation
3. Apply classification head
4. Fine-tune with cross-entropy loss

#### References

1. Devlin, J., et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers."
2. Sun, C., et al. (2019). "How to Fine-Tune BERT for Text Classification."
