# Transformer-Based Named Entity Recognition

## Learning Objectives

- Fine-tune pre-trained transformers (BERT, RoBERTa) for NER
- Handle subword tokenization alignment
- Implement token classification heads
- Combine transformers with CRF layers

## Architecture Overview

```
Input Text → Tokenizer → [CLS] t₁ t₂ ... tₙ [SEP]
                              ↓
                    Pre-trained Transformer
                              ↓
                    h₁, h₂, ..., hₙ (hidden states)
                              ↓
                    Classification Head (Linear)
                              ↓
                    logits → Softmax/CRF → Tags
```

## Subword Tokenization Handling

### The Challenge

Transformers use subword tokenization, splitting words into pieces:

```
"Washington" → ["Wash", "##ing", "##ton"]
```

Only the first subword should receive the entity label; others are ignored.

### Alignment Implementation

```python
import torch
from transformers import AutoTokenizer, AutoModel

def align_labels(
    word_labels: list,
    word_ids: list,  # From tokenizer output
    label_all_tokens: bool = False
) -> list:
    """
    Align word-level labels to subword tokens.
    
    Args:
        word_labels: Labels for each word
        word_ids: word_ids from tokenizer (None for special tokens)
        label_all_tokens: If True, propagate labels to all subwords
    """
    aligned_labels = []
    previous_word_idx = None
    
    for word_idx in word_ids:
        if word_idx is None:
            # Special token ([CLS], [SEP], [PAD])
            aligned_labels.append(-100)  # Ignored in loss
        elif word_idx != previous_word_idx:
            # First subword of a word
            aligned_labels.append(word_labels[word_idx])
        else:
            # Continuation subword
            if label_all_tokens:
                label = word_labels[word_idx]
                # Convert B- to I- for continuation
                if label.startswith('B-'):
                    label = 'I-' + label[2:]
                aligned_labels.append(label)
            else:
                aligned_labels.append(-100)
        
        previous_word_idx = word_idx
    
    return aligned_labels
```

## PyTorch Implementation

### BERT for Token Classification

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class BertForNER(nn.Module):
    """BERT-based NER model with token classification head."""
    
    def __init__(
        self,
        model_name: str = 'bert-base-cased',
        num_labels: int = 9,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        
        self.num_labels = num_labels
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(
                logits.view(-1, self.num_labels),
                labels.view(-1)
            )
        
        return {'loss': loss, 'logits': logits}
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            predictions = torch.argmax(outputs['logits'], dim=-1)
        return predictions
```

### BERT + CRF Model

```python
class BertCRFForNER(nn.Module):
    """BERT with CRF layer for structured prediction."""
    
    def __init__(
        self,
        model_name: str = 'bert-base-cased',
        num_labels: int = 9,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        
        # CRF layer
        self.crf = CRF(num_labels, batch_first=True)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None
    ):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        emissions = self.classifier(sequence_output)
        
        if labels is not None:
            # CRF loss (negative log-likelihood)
            loss = self.crf(emissions, labels, mask=attention_mask.bool())
            return {'loss': loss, 'emissions': emissions}
        
        return {'emissions': emissions}
    
    def decode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> list:
        """Viterbi decoding for best tag sequence."""
        outputs = self.forward(input_ids, attention_mask)
        return self.crf.decode(outputs['emissions'], mask=attention_mask.bool())
```

## Training Pipeline

```python
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader

def train_transformer_ner(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 3,
    learning_rate: float = 5e-5,
    warmup_ratio: float = 0.1,
    device: str = 'cuda'
):
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    best_f1 = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(input_ids, attention_mask, labels)
            loss = outputs['loss']
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
        
        # Evaluation
        model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels']
                
                preds = model.predict(input_ids, attention_mask)
                
                # Collect valid predictions (non-padding)
                for pred, label, mask in zip(preds, labels, attention_mask):
                    valid_len = mask.sum().item()
                    all_preds.append(pred[:valid_len].cpu().tolist())
                    all_labels.append(label[:valid_len].tolist())
        
        # Compute F1
        f1 = compute_ner_f1(all_labels, all_preds)
        
        print(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, F1={f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), 'best_ner_model.pt')
    
    return best_f1
```

## Key Considerations

### Model Selection

| Model | Size | Speed | Accuracy | Best For |
|-------|------|-------|----------|----------|
| BERT-base | 110M | Fast | Good | General use |
| RoBERTa-base | 125M | Fast | Better | Most tasks |
| BERT-large | 340M | Slow | Best | Maximum accuracy |
| DistilBERT | 66M | Fastest | Acceptable | Production/speed |

### Hyperparameters

- **Learning rate**: 2e-5 to 5e-5 (lower than pre-training)
- **Batch size**: 16-32 (depends on GPU memory)
- **Epochs**: 3-5 (transformer models converge quickly)
- **Warmup**: 10% of training steps
- **Max length**: 128-512 (task dependent)

### CRF vs Softmax

| Aspect | Softmax | CRF |
|--------|---------|-----|
| Speed | Faster | Slower |
| Accuracy | Good | Slightly better |
| Transition modeling | No | Yes |
| Use when | Speed critical | Maximum accuracy |

## Summary

1. **Fine-tune** pre-trained transformers for NER
2. **Align** subword tokens to word labels carefully
3. **Add CRF** layer for structured prediction
4. Use **lower learning rates** and **warmup** for stability
5. **Evaluate** at entity level with micro F1
