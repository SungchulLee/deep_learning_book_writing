# Sentiment Analysis

## Task Variants

| Variant | Output | Example |
|---------|--------|---------|
| Binary | pos/neg | "Great movie!" → positive |
| Fine-grained | 1-5 scale | "Decent but slow" → 3/5 |
| Aspect-level | (aspect, sentiment) | "Food great, service slow" → (food,+), (service,-) |

## Financial Sentiment

Domain-specific challenges: "Revenue declined 5%" may be positive if expectations were -8%.

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
# Labels: positive, negative, neutral
```

## Datasets

| Dataset | Task | Size | Classes |
|---------|------|------|---------|
| SST-2 | Binary | 67K | 2 |
| IMDB | Binary | 50K | 2 |
| Financial PhraseBank | Financial | 4.8K | 3 |

## References

1. Socher, R., et al. (2013). Recursive Deep Models for Semantic Compositionality. *EMNLP*.
2. Araci, D. (2019). FinBERT: Financial Sentiment Analysis with Pre-Trained Language Models.
