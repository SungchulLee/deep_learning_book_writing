# Subword Token Alignment

## Learning Objectives

- Understand the challenge of aligning subword tokens to word-level labels
- Implement robust alignment strategies for different tokenizers
- Handle edge cases in label propagation across subwords

---

## The Alignment Problem

Pretrained Transformers use subword tokenization, but NER annotations are at the word level:

```
Words:    "Washington"  "visited"  "Goldman"  "Sachs"
Labels:   B-PER         O          B-ORG      I-ORG

Subwords: "Wash" "##ing" "##ton"  "visited"  "Gold" "##man"  "Sachs"
Labels:   B-PER  ???     ???       O          B-ORG  ???      I-ORG
```

---

## Alignment Strategies

### Strategy 1: First Subword Only (Standard)

Only the first subword of each word receives the label; others get a special ignore index:

```python
def align_first_subword(word_labels, word_ids, ignore_index=-100):
    aligned = []
    prev_word = None
    for word_id in word_ids:
        if word_id is None:
            aligned.append(ignore_index)
        elif word_id != prev_word:
            aligned.append(word_labels[word_id])
        else:
            aligned.append(ignore_index)
        prev_word = word_id
    return aligned
```

### Strategy 2: Propagate Labels

All subwords of a word receive the same label, with B- converted to I- for continuations:

```python
def align_propagate(word_labels, word_ids, label_map, ignore_index=-100):
    aligned = []
    prev_word = None
    for word_id in word_ids:
        if word_id is None:
            aligned.append(ignore_index)
        elif word_id != prev_word:
            aligned.append(word_labels[word_id])
        else:
            label = word_labels[word_id]
            # Convert B- to I- for continuation subwords
            if isinstance(label, str) and label.startswith('B-'):
                label = 'I-' + label[2:]
            aligned.append(label if isinstance(label, int) else label_map.get(label, label))
        prev_word = word_id
    return aligned
```

### Strategy 3: Last Subword

Use the last subword's representation (captures complete morphological information).

---

## Prediction Alignment

At inference, aggregate subword predictions back to word level:

```python
def aggregate_predictions(subword_preds, word_ids, strategy='first'):
    word_preds = {}
    for idx, word_id in enumerate(word_ids):
        if word_id is None:
            continue
        if strategy == 'first':
            if word_id not in word_preds:
                word_preds[word_id] = subword_preds[idx]
        elif strategy == 'max':
            if word_id not in word_preds:
                word_preds[word_id] = subword_preds[idx]
            else:
                # Keep prediction with higher confidence
                word_preds[word_id] = max(
                    word_preds[word_id], subword_preds[idx],
                    key=lambda x: x.max() if hasattr(x, 'max') else x
                )
    return [word_preds[i] for i in sorted(word_preds)]
```

---

## Summary

1. Subword alignment is critical for accurate Transformer-based NER
2. **First subword only** is the standard strategy, balancing simplicity and effectiveness
3. Label propagation can help when subword boundaries carry entity information
4. Prediction aggregation must reverse the alignment at inference time

---

## References

1. Devlin, J., et al. (2019). BERT. *NAACL-HLT*.
2. Souza, F., Nogueira, R., & Lotufo, R. (2020). Portuguese NER by leveraging BERT. *PROPOR*.
