# Bag-of-Words and TF-IDF

## Bag-of-Words

BoW represents documents as word count vectors $\mathbf{x} \in \mathbb{R}^{|V|}$, ignoring word order.

### PyTorch Implementation

```python
import torch
import torch.nn as nn

class BoWClassifier(nn.Module):
    def __init__(self, vocab_size, num_labels):
        super().__init__()
        self.linear = nn.Linear(vocab_size, num_labels)

    def forward(self, bow_vec):
        return self.linear(bow_vec)
```

*See accompanying code: `bow_classifier.py` for a complete Spanish/English classifier example.*

## TF-IDF

$$\text{TF-IDF}(t, d) = \underbrace{\frac{\text{count}(t, d)}{|d|}}_{\text{TF}} \times \underbrace{\log \frac{N}{\text{DF}(t)}}_{\text{IDF}}$$

High TF-IDF indicates terms frequent in a document but rare across the corpus.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train = tfidf.fit_transform(train_texts)
clf = LogisticRegression(max_iter=1000).fit(X_train, train_labels)
```

## Limitations

No word order, no semantics, high dimensionality. These motivate neural representations.

## References

1. Salton, G., & Buckley, C. (1988). Term-Weighting Approaches in Automatic Text Retrieval.
