# Transfer Learning for NLP

Transfer learning has transformed NLP through pretrained language models. This section covers the key strategies for adapting models like BERT, GPT, and their successors to downstream text tasks.

## The NLP Transfer Learning Revolution

Unlike computer vision where ImageNet pretraining dominated for years, NLP transfer underwent a rapid evolution:

| Era | Approach | Transfer mechanism |
|-----|----------|-------------------|
| Pre-2018 | Word2Vec, GloVe | Frozen word embeddings |
| 2018 | ELMo | Contextual embeddings (frozen) |
| 2018–2019 | BERT, GPT | Full model fine-tuning |
| 2020+ | GPT-3, T5 | Prompting, few-shot, adapters |

## Fine-Tuning BERT for Classification

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class BERTClassifier(nn.Module):
    """BERT-based text classifier with transfer learning."""
    
    def __init__(self, model_name='bert-base-uncased', num_labels=2, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels)
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0]  # [CLS] token
        return self.classifier(cls_output)


# Setup
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = BERTClassifier(num_labels=2)

# Fine-tuning hyperparameters (from BERT paper)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
```

### Recommended Hyperparameters

| Parameter | Range | Notes |
|-----------|-------|-------|
| Learning rate | 1e-5 to 5e-5 | Lower than pretraining |
| Batch size | 16–32 | Larger if memory allows |
| Epochs | 2–4 | Risk of overfitting beyond 4 |
| Warmup | 6–10% of steps | Prevents early divergence |
| Weight decay | 0.01 | Standard regularisation |

## Feature Extraction for NLP

When data is very limited or compute is constrained:

```python
class BERTFeatureExtractor(nn.Module):
    """Frozen BERT encoder with trainable classifier."""
    
    def __init__(self, model_name='bert-base-uncased', num_labels=2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_labels)
        )
    
    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.encoder(input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0]
        return self.classifier(cls_output)
```

## ULMFiT-Style Transfer

Howard & Ruder (2018) introduced three techniques that remain relevant:

1. **Discriminative fine-tuning**: Different learning rates per layer (see [discriminative LR](../transfer_learning/discriminative_lr.md))
2. **Slanted triangular learning rates**: Warmup then linear decay
3. **Gradual unfreezing**: Unfreeze from top layer down (see [layer freezing](../transfer_learning/layer_freezing.md))

```python
def slanted_triangular_schedule(optimizer, num_steps, cut_frac=0.1, ratio=32):
    """Slanted triangular learning rate from ULMFiT."""
    cut = int(num_steps * cut_frac)
    
    def lr_lambda(step):
        if step < cut:
            return step / cut  # Linear warmup
        else:
            return 1 - (1 - 1/ratio) * (step - cut) / (num_steps - cut)
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

## Domain-Adaptive Pretraining

For specialised domains (legal, medical, financial), additional pretraining on domain text before task-specific fine-tuning improves performance:

```
General Pretraining → Domain Pretraining → Task Fine-tuning
   (Wikipedia)         (Financial texts)    (Sentiment)
```

This three-stage approach is particularly effective for quantitative finance applications where domain vocabulary and language patterns differ significantly from general text.

## Summary

| Setting | Strategy | Expected improvement |
|---------|----------|---------------------|
| Large labeled data | Full fine-tuning | Highest accuracy |
| Small labeled data | Feature extraction + MLP | Fast, prevents overfitting |
| Domain-specific | Domain pretraining → fine-tune | Best for specialised tasks |
| Very limited data | Prompt-based / few-shot | No fine-tuning needed |

## References

1. Devlin, J., et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers." *NAACL*.
2. Howard, J., & Ruder, S. (2018). "Universal Language Model Fine-tuning for Text Classification." *ACL*.
3. Gururangan, S., et al. (2020). "Don't Stop Pretraining: Adapt Language Models to Domains and Tasks." *ACL*.
