# Regression Metrics

Regression metrics quantify how well predictions match continuous target values. Different metrics emphasize different aspects of error, so choose based on your application's priorities.

---

## Mean Squared Error (MSE)

### 1. Definition and Usage

```python
from sklearn.metrics import mean_squared_error
import numpy as np

y_true = np.array([3, -0.5, 2, 7])
y_pred = np.array([2.5, 0.0, 2, 8])

mse = mean_squared_error(y_true, y_pred)
print(f"MSE: {mse:.4f}")  # 0.375
```

### 2. Mathematical Definition

$$MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

```python
# Manual calculation
mse_manual = np.mean((y_true - y_pred) ** 2)
print(f"Manual MSE: {mse_manual:.4f}")
```

### 3. Properties

- **Always non-negative** (MSE ≥ 0)
- **Sensitive to outliers** (squared errors)
- **Same units as y²** (not easily interpretable)
- **Differentiable** (good for optimization)

### 4. When to Use

- Standard loss function for training
- When large errors are particularly undesirable
- When outliers should be penalized heavily

---

## Root Mean Squared Error (RMSE)

### 1. Definition and Usage

```python
from sklearn.metrics import mean_squared_error

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
# Or with squared=False
rmse = mean_squared_error(y_true, y_pred, squared=False)
print(f"RMSE: {rmse:.4f}")  # 0.6124
```

### 2. Mathematical Definition

$$RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

### 3. Properties

- **Same units as target** (interpretable)
- **Comparable to standard deviation** of errors
- **Still sensitive to outliers**

### 4. Interpretation

```python
# RMSE tells you the typical magnitude of errors
# If RMSE = 5 and target ranges from 0-100
# Errors are typically around 5% of the range
```

---

## Mean Absolute Error (MAE)

### 1. Definition and Usage

```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_true, y_pred)
print(f"MAE: {mae:.4f}")  # 0.5
```

### 2. Mathematical Definition

$$MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

```python
# Manual calculation
mae_manual = np.mean(np.abs(y_true - y_pred))
print(f"Manual MAE: {mae_manual:.4f}")
```

### 3. Properties

- **Same units as target** (interpretable)
- **Less sensitive to outliers** than MSE
- **Not differentiable at zero** (gradient is undefined)
- **Robust to outliers**

### 4. MSE vs MAE

```python
# Example with outlier
y_true_outlier = np.array([1, 2, 3, 4, 100])  # 100 is outlier
y_pred_outlier = np.array([1.1, 2.1, 3.1, 4.1, 5])

mse_outlier = mean_squared_error(y_true_outlier, y_pred_outlier)
mae_outlier = mean_absolute_error(y_true_outlier, y_pred_outlier)

print(f"MSE with outlier: {mse_outlier:.2f}")  # Dominated by outlier
print(f"MAE with outlier: {mae_outlier:.2f}")  # More robust
```

---

## R² Score (Coefficient of Determination)

### 1. Definition and Usage

```python
from sklearn.metrics import r2_score

r2 = r2_score(y_true, y_pred)
print(f"R²: {r2:.4f}")
```

### 2. Mathematical Definition

$$R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2} = 1 - \frac{SS_{res}}{SS_{tot}}$$

```python
# Manual calculation
ss_res = np.sum((y_true - y_pred) ** 2)
ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
r2_manual = 1 - (ss_res / ss_tot)
print(f"Manual R²: {r2_manual:.4f}")
```

### 3. Interpretation

```python
# R² = 1: Perfect predictions
# R² = 0: Model predicts the mean (no better than baseline)
# R² < 0: Model is worse than predicting the mean

# Example interpretations
print("R² = 0.90: Model explains 90% of variance")
print("R² = 0.50: Model explains 50% of variance")
print("R² = 0.00: Model no better than mean baseline")
```

### 4. Adjusted R²

```python
# Adjusted R² penalizes for number of features
def adjusted_r2(r2, n_samples, n_features):
    return 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)

n_samples, n_features = 100, 10
r2 = 0.85

adj_r2 = adjusted_r2(r2, n_samples, n_features)
print(f"R²: {r2:.4f}")
print(f"Adjusted R²: {adj_r2:.4f}")
```

---

## Mean Absolute Percentage Error (MAPE)

### 1. Definition and Usage

```python
from sklearn.metrics import mean_absolute_percentage_error

y_true = np.array([100, 50, 30, 20])
y_pred = np.array([110, 45, 33, 22])

mape = mean_absolute_percentage_error(y_true, y_pred)
print(f"MAPE: {mape:.4f}")  # As decimal (0.15 = 15%)
print(f"MAPE: {mape*100:.2f}%")
```

### 2. Mathematical Definition

$$MAPE = \frac{1}{n}\sum_{i=1}^{n}\left|\frac{y_i - \hat{y}_i}{y_i}\right|$$

```python
# Manual calculation
mape_manual = np.mean(np.abs((y_true - y_pred) / y_true))
print(f"Manual MAPE: {mape_manual:.4f}")
```

### 3. Properties and Limitations

```python
# MAPE issues:
# 1. Undefined when y_true = 0
# 2. Asymmetric: penalizes under-predictions more

# Example of asymmetry
y_true_asym = np.array([100])
y_pred_over = np.array([150])  # Over-prediction by 50
y_pred_under = np.array([50])  # Under-prediction by 50

print(f"Over-prediction MAPE: {mean_absolute_percentage_error(y_true_asym, y_pred_over):.2%}")
print(f"Under-prediction MAPE: {mean_absolute_percentage_error(y_true_asym, y_pred_under):.2%}")
```

---

## Symmetric MAPE (sMAPE)

### 1. Definition

```python
def symmetric_mape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error"""
    return np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

smape = symmetric_mape(y_true, y_pred)
print(f"sMAPE: {smape:.4f}")
```

### 2. Properties

- Bounded between 0 and 2 (or 0% and 200%)
- More symmetric than MAPE
- Still undefined when both y_true and y_pred are 0

---

## Median Absolute Error

### 1. Definition and Usage

```python
from sklearn.metrics import median_absolute_error

med_ae = median_absolute_error(y_true, y_pred)
print(f"Median AE: {med_ae:.4f}")
```

### 2. Properties

- **Very robust to outliers**
- Reports the median error magnitude
- Useful when data has many outliers

---

## Max Error

### 1. Definition and Usage

```python
from sklearn.metrics import max_error

y_true = np.array([3, 2, 7, 1])
y_pred = np.array([4, 2, 7, 10])  # Last prediction is way off

max_err = max_error(y_true, y_pred)
print(f"Max Error: {max_err}")  # 9
```

### 2. Use Cases

- Quality control (worst-case error)
- When all predictions must be within tolerance
- Safety-critical applications

---

## Explained Variance Score

### 1. Definition and Usage

```python
from sklearn.metrics import explained_variance_score

evs = explained_variance_score(y_true, y_pred)
print(f"Explained Variance: {evs:.4f}")
```

### 2. Mathematical Definition

$$EVS = 1 - \frac{Var(y - \hat{y})}{Var(y)}$$

### 3. Difference from R²

```python
# R² uses sum of squared errors
# EVS uses variance of residuals

# They're equal when residuals have zero mean
# EVS can be higher than R² if model has bias
```

---

## Custom Metrics

### 1. Weighted MSE

```python
# Give more weight to certain samples
sample_weights = np.array([1, 1, 1, 10])  # Last sample more important

weighted_mse = mean_squared_error(y_true, y_pred, sample_weight=sample_weights)
print(f"Weighted MSE: {weighted_mse:.4f}")
```

### 2. Log-transformed Metrics

```python
def rmsle(y_true, y_pred):
    """Root Mean Squared Logarithmic Error"""
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))

# Good for targets with wide range (e.g., prices)
y_true_prices = np.array([100, 1000, 10000])
y_pred_prices = np.array([110, 900, 11000])

print(f"RMSE: {np.sqrt(mean_squared_error(y_true_prices, y_pred_prices)):.2f}")
print(f"RMSLE: {rmsle(y_true_prices, y_pred_prices):.4f}")
```

---

## Comprehensive Comparison

```python
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Generate data
X, y = make_regression(n_samples=1000, n_features=10, noise=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# All metrics
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    median_absolute_error, max_error, explained_variance_score
)

print("Regression Metrics Summary")
print("-" * 40)
print(f"MSE:               {mean_squared_error(y_test, y_pred):.4f}")
print(f"RMSE:              {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"MAE:               {mean_absolute_error(y_test, y_pred):.4f}")
print(f"Median AE:         {median_absolute_error(y_test, y_pred):.4f}")
print(f"Max Error:         {max_error(y_test, y_pred):.4f}")
print(f"R²:                {r2_score(y_test, y_pred):.4f}")
print(f"Explained Var:     {explained_variance_score(y_test, y_pred):.4f}")
```

---

## PyTorch Equivalents

```python
import torch
import torch.nn as nn

y_true_t = torch.FloatTensor(y_test)
y_pred_t = torch.FloatTensor(y_pred)

# MSE Loss
mse_loss = nn.MSELoss()
print(f"PyTorch MSE: {mse_loss(y_pred_t, y_true_t).item():.4f}")

# MAE Loss (L1 Loss)
mae_loss = nn.L1Loss()
print(f"PyTorch MAE: {mae_loss(y_pred_t, y_true_t).item():.4f}")

# Huber Loss (smooth L1, robust)
huber_loss = nn.SmoothL1Loss()
print(f"PyTorch Huber: {huber_loss(y_pred_t, y_true_t).item():.4f}")
```

---

## Choosing the Right Metric

| Metric | Sensitive to Outliers | Units | Interpretability | Best For |
|--------|----------------------|-------|------------------|----------|
| MSE | Very | y² | Low | Training |
| RMSE | Very | y | Medium | General |
| MAE | Less | y | High | Robust evaluation |
| MAPE | Less | % | High | Business reporting |
| R² | Very | None | High | Model comparison |
| Median AE | Robust | y | High | Outlier-heavy data |

**Guidelines:**
- Use **RMSE** as default metric
- Use **MAE** when outliers should be tolerated
- Use **MAPE** for percentage-based interpretation
- Use **R²** for comparing models on same dataset
- Use **Max Error** when worst-case matters
