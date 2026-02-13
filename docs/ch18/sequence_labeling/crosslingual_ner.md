# Cross-Lingual NER

## Overview

Cross-lingual NER transfers entity recognition capabilities from high-resource languages (typically English) to low-resource target languages.

## Transfer Approaches

| Method | Requires Target Data | Approach |
|--------|---------------------|----------|
| Direct Transfer | No | Train on source, apply to target using multilingual model |
| Translate-Train | No target NER data | Translate source data, project annotations |
| Few-Shot | Small target set | Fine-tune multilingual model on few target examples |

### Direct Transfer with Multilingual BERT

```python
from transformers import AutoModelForTokenClassification

# Train on English CoNLL-2003
model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-multilingual-cased", num_labels=9
)
# Train on English data...

# Apply directly to German/Spanish/etc.
# mBERT's shared representation enables zero-shot transfer
```

### Translate-Train

1. Translate source sentences to target language
2. Project entity annotations using word alignments
3. Train on translated + projected data

## Multilingual Models

| Model | Languages | Parameters |
|-------|-----------|-----------|
| mBERT | 104 | 110M |
| XLM-RoBERTa | 100 | 270M / 550M |
| RemBERT | 110 | 575M |

## Benchmark: WikiANN

WikiANN provides NER annotations in 282 languages using Wikipedia anchor texts.

Typical zero-shot transfer F1 from English:
- Spanish: ~75 F1
- German: ~70 F1
- Chinese: ~50 F1
- Low-resource African languages: ~40 F1

## Summary

1. Multilingual pretrained models enable zero-shot cross-lingual NER
2. XLM-RoBERTa generally outperforms mBERT for cross-lingual transfer
3. Translate-train can improve over direct transfer when translation quality is high
4. Performance degrades significantly for typologically distant languages
