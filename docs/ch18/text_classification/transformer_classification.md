# Transformer-Based Text Classification

## Overview

Fine-tuning pretrained Transformers is the dominant approach. BERT uses the `[CLS]` token:

$$P(y | x) = \text{softmax}(\mathbf{W} \cdot \mathbf{h}_{[CLS]} + \mathbf{b})$$

```python
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
trainer = Trainer(model=model, args=TrainingArguments(
    output_dir="./results", num_train_epochs=3,
    per_device_train_batch_size=16, learning_rate=2e-5,
))
trainer.train()
```

## Model Comparison

| Model | GLUE Avg | SST-2 |
|-------|----------|-------|
| BERT-base | 79.6 | 93.5 |
| RoBERTa-base | 83.2 | 94.8 |
| DeBERTa-v3 | 88.1 | 96.0 |

## References

1. Devlin, J., et al. (2019). BERT. *NAACL-HLT*.
2. Sun, C., et al. (2019). How to Fine-Tune BERT for Text Classification. *CCL*.
