# Credit Scoring

Credit scoring predicts default probability from borrower characteristics. It presents specific ML challenges: severe class imbalance, regulatory interpretability requirements, and the need for calibrated probabilities. Scikit-learn provides the tools to address each of these.

## The Imbalanced Classification Problem

Default rates are typically 1–5%, creating extreme class imbalance where a "predict all non-default" baseline achieves >95% accuracy:

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Simulate credit data: 3% default rate
X, y = make_classification(
    n_samples=10000, n_features=20, n_informative=10,
    weights=[0.97, 0.03],  # 97% non-default
    random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"Default rate: {y.mean():.2%}")
print(f"Naive accuracy: {1 - y.mean():.2%}")
```

## Handling Imbalance

### Class Weights

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Automatic balancing: weight inversely proportional to class frequency
lr = LogisticRegression(class_weight='balanced', max_iter=1000)
rf = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)
```

### SMOTE (Synthetic Oversampling)

```python
# pip install imbalanced-learn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import StandardScaler

pipe = ImbPipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('clf', LogisticRegression(max_iter=1000)),
])

# SMOTE applied only to training folds in cross-validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='roc_auc')
print(f"CV AUC: {scores.mean():.4f}")
```

### Threshold Tuning

The default 0.5 threshold is rarely optimal for imbalanced problems:

```python
from sklearn.metrics import precision_recall_curve

lr = LogisticRegression(class_weight='balanced', max_iter=1000)
lr.fit(X_train, y_train)
y_proba = lr.predict_proba(X_test)[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

# Find threshold for target recall (e.g., catch 80% of defaults)
target_recall = 0.80
idx = np.argmin(np.abs(recalls[:-1] - target_recall))
optimal_threshold = thresholds[idx]
print(f"Threshold for {target_recall:.0%} recall: {optimal_threshold:.3f}")
print(f"Precision at this threshold: {precisions[idx]:.3f}")
```

## Metrics for Credit Scoring

Accuracy is misleading; use these instead:

```python
from sklearn.metrics import (
    classification_report, roc_auc_score,
    average_precision_score, brier_score_loss
)

y_pred = (y_proba >= optimal_threshold).astype(int)
print(classification_report(y_test, y_pred, target_names=['Non-Default', 'Default']))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
print(f"PR-AUC:  {average_precision_score(y_test, y_proba):.4f}")
print(f"Brier:   {brier_score_loss(y_test, y_proba):.4f}")
```

### Gini Coefficient

Standard in banking credit risk:

```python
def gini_coefficient(y_true, y_score):
    auc = roc_auc_score(y_true, y_score)
    return 2 * auc - 1

gini = gini_coefficient(y_test, y_proba)
print(f"Gini: {gini:.4f}")
```

### KS Statistic

Kolmogorov–Smirnov statistic measures the maximum separation between default and non-default score distributions:

```python
def ks_statistic(y_true, y_score):
    from scipy.stats import ks_2samp
    default_scores = y_score[y_true == 1]
    non_default_scores = y_score[y_true == 0]
    stat, _ = ks_2samp(default_scores, non_default_scores)
    return stat

ks = ks_statistic(y_test, y_proba)
print(f"KS Statistic: {ks:.4f}")
```

## Probability Calibration

Raw model probabilities are often poorly calibrated. Regulatory requirements demand that a predicted 5% PD means approximately 5% of such borrowers actually default:

```python
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

# Calibrate using Platt scaling or isotonic regression
calibrated = CalibratedClassifierCV(
    LogisticRegression(class_weight='balanced', max_iter=1000),
    cv=5,
    method='isotonic'
)
calibrated.fit(X_train, y_train)
y_proba_cal = calibrated.predict_proba(X_test)[:, 1]

# Check calibration
fraction_of_positives, mean_predicted_value = calibration_curve(
    y_test, y_proba_cal, n_bins=10
)
```

## Scorecard Development

Traditional scorecards use logistic regression with binned features for interpretability:

```python
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.pipeline import Pipeline

scorecard_pipe = Pipeline([
    ('binner', KBinsDiscretizer(n_bins=10, encode='onehot-dense', strategy='quantile')),
    ('clf', LogisticRegression(class_weight='balanced', max_iter=1000)),
])

scorecard_pipe.fit(X_train, y_train)
y_score = scorecard_pipe.predict_proba(X_test)[:, 1]

# Convert to points-based scorecard
def proba_to_score(proba, pdo=20, base_score=600, base_odds=50):
    """Convert probability to credit score (higher = better)."""
    odds = (1 - proba) / np.maximum(proba, 1e-10)
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    return offset + factor * np.log(odds)

scores = proba_to_score(y_score)
print(f"Score range: [{scores.min():.0f}, {scores.max():.0f}]")
```

## Interpretability

### Feature Importance

```python
from sklearn.inspection import permutation_importance

rf = RandomForestClassifier(class_weight='balanced', n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

perm_imp = permutation_importance(rf, X_test, y_test, n_repeats=10,
                                   scoring='roc_auc', random_state=42)

import pandas as pd
imp_df = pd.DataFrame({
    'feature': feature_names,
    'importance': perm_imp.importances_mean,
    'std': perm_imp.importances_std,
}).sort_values('importance', ascending=False)

print(imp_df.head(10))
```

### Partial Dependence

```python
from sklearn.inspection import PartialDependenceDisplay

fig, ax = plt.subplots(figsize=(12, 4))
PartialDependenceDisplay.from_estimator(
    rf, X_test, features=[0, 1, 2],  # top 3 features
    feature_names=feature_names, ax=ax
)
```

## Model Comparison Pipeline

```python
from sklearn.model_selection import StratifiedKFold

models = {
    'LR': LogisticRegression(class_weight='balanced', max_iter=1000),
    'RF': RandomForestClassifier(class_weight='balanced', n_estimators=200),
    'GBM': GradientBoostingClassifier(n_estimators=200),
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    pipe = Pipeline([('scaler', StandardScaler()), ('clf', model)])
    auc_scores = cross_val_score(pipe, X_train, y_train, cv=skf, scoring='roc_auc')
    print(f"{name:4s}: AUC = {auc_scores.mean():.4f} ± {auc_scores.std():.4f}")
```

## Summary

| Challenge | Solution |
|-----------|----------|
| Class imbalance | `class_weight='balanced'`, SMOTE, threshold tuning |
| Metric selection | ROC-AUC, PR-AUC, Gini, KS (not accuracy) |
| Calibration | `CalibratedClassifierCV` (isotonic or Platt) |
| Interpretability | Logistic regression, permutation importance, PDP |
| Scorecard format | Binning + logistic regression + probability-to-score mapping |
| Regulatory compliance | Calibrated PDs, interpretable features, stable models |
