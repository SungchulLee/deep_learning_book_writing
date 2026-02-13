# Naive Bayes

Naive Bayes classifiers apply Bayes' theorem with the "naive" assumption of conditional independence between features given the class. Despite this strong assumption, they perform surprisingly well on text classification, spam filtering, and as fast baselines for high-dimensional data.

## Bayes' Theorem

$$P(y \mid \mathbf{x}) = \frac{P(\mathbf{x} \mid y) \, P(y)}{P(\mathbf{x})}$$

The naive independence assumption simplifies the likelihood:

$$P(\mathbf{x} \mid y) = \prod_{j=1}^{d} P(x_j \mid y)$$

Classification selects the class maximising the posterior:

$$\hat{y} = \arg\max_{c} \; P(y = c) \prod_{j=1}^{d} P(x_j \mid y = c)$$

In log-space (numerically stable):

$$\hat{y} = \arg\max_{c} \left[ \log P(y = c) + \sum_{j=1}^{d} \log P(x_j \mid y = c) \right]$$

## GaussianNB

Assumes each feature follows a Gaussian distribution within each class:

$$P(x_j \mid y = c) = \frac{1}{\sqrt{2\pi \sigma_{jc}^2}} \exp\!\left(-\frac{(x_j - \mu_{jc})^2}{2\sigma_{jc}^2}\right)$$

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np

X, y = make_classification(n_samples=1000, n_features=10, n_informative=5,
                           n_classes=3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gnb = GaussianNB()
gnb.fit(X_train, y_train)

print(f"Accuracy: {gnb.score(X_test, y_test):.4f}")
print(f"Class priors: {gnb.class_prior_.round(3)}")
print(f"Class means shape: {gnb.theta_.shape}")    # (n_classes, n_features)
print(f"Class variances shape: {gnb.var_.shape}")   # (n_classes, n_features)
```

### Variance Smoothing

```python
# Add small constant to variances for numerical stability
gnb = GaussianNB(var_smoothing=1e-9)  # default
```

### Incremental Learning

```python
# Partial fit for streaming / large data
gnb = GaussianNB()
gnb.partial_fit(X_batch_1, y_batch_1, classes=[0, 1, 2])
gnb.partial_fit(X_batch_2, y_batch_2)
```

### When to Use

- Continuous features that are roughly Gaussian within each class
- Quick baseline before trying more complex models
- When features are reasonably independent

## MultinomialNB

Models feature counts using a multinomial distribution. Standard for text classification with word counts or TF-IDF:

$$P(x_j \mid y = c) \propto \theta_{jc}^{x_j}, \quad \theta_{jc} = \frac{N_{jc} + \alpha}{N_c + \alpha d}$$

where $N_{jc}$ is the count of feature $j$ in class $c$, $N_c$ is the total count in class $c$, and $\alpha$ is the Laplace smoothing parameter.

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline

# Text classification example
texts = [
    "earnings beat expectations revenue growth", "stock price target raised upgrade",
    "quarterly loss guidance lowered", "debt downgrade credit risk default",
    "dividend increase buyback shareholder", "profit margin expansion efficiency",
    "bankruptcy filing restructuring", "revenue decline market share loss",
]
labels = [1, 1, 0, 0, 1, 1, 0, 0]  # 1=positive, 0=negative

pipe = make_pipeline(
    TfidfVectorizer(),
    MultinomialNB(alpha=1.0)  # Laplace smoothing
)

pipe.fit(texts, labels)
print(pipe.predict(["strong earnings growth exceeded"]))  # [1]
```

### Alpha (Smoothing)

```python
# alpha=1.0: Laplace smoothing (default)
# alpha=0.0: No smoothing (risk of zero probabilities)
# alpha<1.0: Lidstone smoothing
mnb = MultinomialNB(alpha=0.1)
```

### Feature Log-Probabilities

```python
mnb = MultinomialNB()
mnb.fit(X_counts, y)

# Log P(x_j | y=c)
print(mnb.feature_log_prob_.shape)  # (n_classes, n_features)

# Most informative features per class
feature_names = vectorizer.get_feature_names_out()
for c in range(mnb.classes_.shape[0]):
    top_indices = mnb.feature_log_prob_[c].argsort()[-10:][::-1]
    top_words = [feature_names[i] for i in top_indices]
    print(f"Class {c}: {', '.join(top_words)}")
```

## BernoulliNB

Models binary features (presence/absence). Each feature is Bernoulli-distributed:

$$P(x_j \mid y = c) = p_{jc}^{x_j} (1 - p_{jc})^{(1 - x_j)}$$

```python
from sklearn.naive_bayes import BernoulliNB

# Binary features (e.g., word presence in document)
bnb = BernoulliNB(alpha=1.0, binarize=0.0)
bnb.fit(X_binary, y)
```

The `binarize` parameter thresholds continuous inputs:

```python
# TF-IDF values → binary (present if > 0)
bnb = BernoulliNB(binarize=0.0)
# Converts: x > 0 → 1, x ≤ 0 → 0
```

**Key difference from MultinomialNB**: BernoulliNB explicitly penalises the *absence* of features, making it better when non-occurrence is informative (e.g., a word NOT appearing in a spam email).

## ComplementNB

Designed for imbalanced text classification. Uses statistics from the *complement* of each class:

```python
from sklearn.naive_bayes import ComplementNB

cnb = ComplementNB(alpha=1.0, norm=True)
cnb.fit(X_tfidf, y)
# Often outperforms MultinomialNB on imbalanced datasets
```

## Comparison of Variants

| Variant | Feature Type | Likelihood | Best For |
|---------|-------------|------------|----------|
| `GaussianNB` | Continuous | Gaussian | General continuous data |
| `MultinomialNB` | Counts / TF-IDF | Multinomial | Text classification |
| `BernoulliNB` | Binary | Bernoulli | Binary features, short texts |
| `ComplementNB` | Counts / TF-IDF | Complement | Imbalanced text data |
| `CategoricalNB` | Categorical | Categorical | Discrete non-ordered features |

## The Independence Assumption

The assumption $P(\mathbf{x} \mid y) = \prod_j P(x_j \mid y)$ is almost always violated in practice. Despite this:

1. **Classification** only needs the correct $\arg\max$, not calibrated probabilities. Even with wrong independence assumptions, the ranking of classes can still be correct.
2. **Calibration**: The raw probabilities from `predict_proba` are typically poorly calibrated. Use `CalibratedClassifierCV` if calibrated probabilities are needed:

```python
from sklearn.calibration import CalibratedClassifierCV

gnb = GaussianNB()
calibrated = CalibratedClassifierCV(gnb, cv=5, method='isotonic')
calibrated.fit(X_train, y_train)
proba_calibrated = calibrated.predict_proba(X_test)
```

## Pipeline Example

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV

pipe = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB()),
])

param_grid = {
    'tfidf__max_features': [1000, 5000, 10000],
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'clf__alpha': [0.01, 0.1, 1.0],
}

grid = GridSearchCV(pipe, param_grid, cv=5, scoring='f1_weighted')
grid.fit(texts, labels)
print(f"Best: {grid.best_params_}")
```

## Quantitative Finance: News Sentiment Classification

Naive Bayes is a natural fit for classifying financial news sentiment due to its speed and effectiveness with text:

```python
# Classify news headlines as positive/negative for asset returns
news_pipe = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
    ('clf', ComplementNB(alpha=0.1)),   # handles sentiment imbalance
])

# Train on labelled financial news
news_pipe.fit(headline_texts, sentiment_labels)

# Score incoming news in real-time
new_headlines = ["Fed signals rate cut", "Earnings miss analyst estimates"]
sentiments = news_pipe.predict(new_headlines)
probas = news_pipe.predict_proba(new_headlines)
```

## Summary

| Aspect | Detail |
|--------|--------|
| **Training** | $O(n \cdot d)$ — single pass through data |
| **Prediction** | $O(d)$ per sample |
| **Scaling** | Not needed (except for `GaussianNB` may benefit) |
| **Key parameter** | `alpha` (smoothing) |
| **Strengths** | Extremely fast, works well with small data, handles high $d$ |
| **Weaknesses** | Independence assumption, poor probability calibration |
| **Best for** | Text classification, fast baselines, incremental learning |
