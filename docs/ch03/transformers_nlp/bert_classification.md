# Text Classification with BERT

## Overview

BERT excels at text classification tasks like sentiment analysis, topic classification, and intent detection. This document covers the complete pipeline from data preparation to deployment.

## Architecture for Classification

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

## PyTorch Implementation

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

## Best Practices

1. **Learning Rate**: 2e-5 to 5e-5 for BERT
2. **Epochs**: 2-4 usually sufficient
3. **Batch Size**: 16-32 (larger with gradient accumulation)
4. **Warmup**: 10% of total steps
5. **Weight Decay**: 0.01
6. **Max Length**: Task-dependent, 128-512

## Summary

BERT classification involves:
1. Tokenize input with [CLS] and [SEP]
2. Extract [CLS] representation
3. Apply classification head
4. Fine-tune with cross-entropy loss

## References

1. Devlin, J., et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers."
2. Sun, C., et al. (2019). "How to Fine-Tune BERT for Text Classification."
