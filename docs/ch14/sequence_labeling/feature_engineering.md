# Feature Engineering for NER

## Learning Objectives

- Design effective hand-crafted features for NER systems
- Implement orthographic, lexical, and contextual feature extractors
- Understand feature templates for CRF-based NER
- Recognize the transition from manual features to neural representations

---

## Overview

Before neural NER models, feature engineering was the primary determinant of system quality. Understanding these features remains valuable: they inform neural architecture design, serve as interpretable baselines, and augment neural models in low-resource settings.

---

## Feature Categories

### Word-Level Features

```python
def extract_word_features(word: str, position: int, sentence_len: int) -> dict:
    """Extract features for a single word."""
    return {
        'word.lower': word.lower(),
        'word.isupper': word.isupper(),
        'word.istitle': word.istitle(),
        'word.isdigit': word.isdigit(),
        'word.length': len(word),
        'word.has_hyphen': '-' in word,
        'word.has_digit': any(c.isdigit() for c in word),
        'word.prefix_1': word[0] if word else '',
        'word.prefix_2': word[:2] if len(word) >= 2 else word,
        'word.prefix_3': word[:3] if len(word) >= 3 else word,
        'word.suffix_1': word[-1] if word else '',
        'word.suffix_2': word[-2:] if len(word) >= 2 else word,
        'word.suffix_3': word[-3:] if len(word) >= 3 else word,
        'word.is_start': position == 0,
        'word.is_end': position == sentence_len - 1,
    }
```

### Context Window Features

```python
def extract_context_features(tokens, position, window=2):
    """Extract features from surrounding context."""
    features = {}
    for offset in range(-window, window + 1):
        idx = position + offset
        prefix = f'w[{offset:+d}]'
        if 0 <= idx < len(tokens):
            features[f'{prefix}.word'] = tokens[idx].lower()
            features[f'{prefix}.istitle'] = tokens[idx].istitle()
            features[f'{prefix}.isupper'] = tokens[idx].isupper()
        else:
            features[f'{prefix}.BOS'] = True if idx < 0 else False
            features[f'{prefix}.EOS'] = True if idx >= len(tokens) else False
    return features
```

### Feature Templates for CRFs

CRF feature templates define combinations of observations and labels:

| Template | Description |
|----------|-------------|
| $f(y_i, x_i)$ | Current word + current label |
| $f(y_i, x_{i-1})$ | Previous word + current label |
| $f(y_{i-1}, y_i)$ | Transition: previous label + current label |
| $f(y_i, \text{prefix}_3(x_i))$ | 3-char prefix + current label |
| $f(y_i, \text{shape}(x_i))$ | Word shape + current label |

---

## Word Shape Features

Encode orthographic patterns:

```python
def word_shape(word: str) -> str:
    """Convert word to shape pattern (e.g., 'Apple' -> 'Xxxxx')."""
    shape = []
    for c in word:
        if c.isupper():
            shape.append('X')
        elif c.islower():
            shape.append('x')
        elif c.isdigit():
            shape.append('d')
        else:
            shape.append(c)
    return ''.join(shape)

def brief_word_shape(word: str) -> str:
    """Compressed shape (collapse consecutive same-type chars)."""
    shape = word_shape(word)
    brief = [shape[0]]
    for c in shape[1:]:
        if c != brief[-1]:
            brief.append(c)
    return ''.join(brief)

# Examples:
# "Apple" -> "Xxxxx" (full) / "Xx" (brief)
# "U.S.A." -> "X.X.X." (full) / "X.X." (brief)
# "COVID-19" -> "XXXXX-dd" (full) / "X-d" (brief)
```

---

## Transition from Features to Neural Representations

| Era | Approach | Example |
|-----|----------|---------|
| Classical | Hand-crafted features + CRF | CoNLL-2003 baseline |
| Embedding era | Word2Vec/GloVe + BiLSTM + CRF | Lample et al. 2016 |
| Pretrained era | BERT representations + Linear/CRF | Devlin et al. 2019 |

Neural models learn representations that implicitly capture many hand-crafted features. However, explicit features still help in low-resource settings and provide interpretability.

---

## Summary

1. Orthographic features (capitalization, shape, prefixes/suffixes) are the most discriminative for NER
2. Context window features capture local patterns around each token
3. Feature templates formalize the interaction between observations and labels in CRFs
4. Neural models have largely replaced manual features but understanding them informs architecture design

---

## References

1. Ratinov, L., & Roth, D. (2009). Design Challenges in NER. *CoNLL*.
2. Zhang, S., & Elhadad, N. (2013). Unsupervised Biomedical NER. *J. Biomedical Informatics*.
