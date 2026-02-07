# BERT for Token Classification

## Learning Objectives

- Fine-tune BERT for NER as a token classification task
- Handle subword-to-word label alignment
- Apply the HuggingFace Transformers pipeline for NER
- Compare BERT variants for token classification performance

---

## BERT Token Classification

BERT treats NER as **token classification**: each token receives an independent label prediction from the final hidden state.

### Architecture

$$P(y_i | \mathbf{x}) = \text{softmax}(\mathbf{W} \cdot \mathbf{h}_i^{[L]} + \mathbf{b})$$

where $\mathbf{h}_i^{[L]}$ is BERT's final-layer hidden state at position $i$.

### HuggingFace Implementation

```python
from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers import TrainingArguments, Trainer
from datasets import load_dataset
import numpy as np

# Load model and tokenizer
model_name = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load CoNLL-2003
dataset = load_dataset("conll2003")
label_list = dataset["train"].features["ner_tags"].feature.names

model = AutoModelForTokenClassification.from_pretrained(
    model_name, num_labels=len(label_list)
)

def tokenize_and_align(examples):
    tokenized = tokenizer(
        examples["tokens"], truncation=True,
        is_split_into_words=True, padding="max_length", max_length=128
    )
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        label_ids = []
        prev_word = None
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != prev_word:
                label_ids.append(label[word_id])
            else:
                label_ids.append(-100)  # Ignore subword continuations
            prev_word = word_id
        labels.append(label_ids)
    tokenized["labels"] = labels
    return tokenized

tokenized_dataset = dataset.map(tokenize_and_align, batched=True)

training_args = TrainingArguments(
    output_dir="./ner_model",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=5e-5,
    warmup_ratio=0.1,
    evaluation_strategy="epoch",
)

trainer = Trainer(
    model=model, args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
)
trainer.train()
```

---

## Model Comparison

| Model | CoNLL-2003 F1 | Speed | Memory |
|-------|---------------|-------|--------|
| BERT-base-cased | 92.4 | Fast | 440MB |
| BERT-large-cased | 92.8 | Slow | 1.3GB |
| RoBERTa-base | 92.6 | Fast | 500MB |
| RoBERTa-large | 93.2 | Slow | 1.4GB |
| DeBERTa-v3-base | 93.0 | Fast | 560MB |

---

## Summary

1. BERT treats NER as per-token classification with a linear head on the final layer
2. Subword alignment via `word_ids()` maps labels to the first subword of each word
3. Fine-tuning with low learning rates (2e-5 to 5e-5) and warmup is standard
4. Adding a CRF layer on top of BERT provides modest further improvement

---

## References

1. Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers. *NAACL-HLT*.
2. Wolf, T., et al. (2020). Transformers: State-of-the-Art NLP. *EMNLP Demo*.
