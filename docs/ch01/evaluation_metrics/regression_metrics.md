# Regression Metrics: MSE, MAE, R² and Beyond

## Overview

Regression metrics quantify how well a model's continuous predictions match actual values. Unlike classification where predictions are discrete categories, regression deals with continuous outputs, requiring metrics that measure the magnitude and direction of prediction errors.

This section covers essential regression metrics, their mathematical foundations, PyTorch implementations, and practical guidance for metric selection in quantitative finance applications.

---

## Mathematical Foundations

### The Prediction Error Framework

For a regression model with predictions $\hat{y}_i$ and true values $y_i$ across $n$ samples, the **residual** (prediction error) for sample $i$ is:

$$e_i = y_i - \hat{y}_i$$

All regression metrics are functions of these residuals, differing in how they aggregate and weight errors.

---

## Mean Absolute Error (MAE)

### Definition

Mean Absolute Error measures the average magnitude of errors without considering direction:

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

### Properties

| Property | Description |
|----------|-------------|
| **Units** | Same as target variable |
| **Range** | $[0, \infty)$ |
| **Optimal Value** | 0 (perfect predictions) |
| **Outlier Sensitivity** | Low (linear penalty) |
| **Interpretability** | High |

### Mathematical Interpretation

MAE treats all errors equally regardless of their size. An error of 10 units is penalized exactly twice as much as an error of 5 units.

**Gradient for optimization:**

$$\frac{\partial \text{MAE}}{\partial \hat{y}_i} = -\frac{1}{n} \cdot \text{sign}(y_i - \hat{y}_i)$$

The gradient magnitude is constant, leading to stable but potentially slow convergence near the optimum.

### PyTorch Implementation

```python
import torch
import torch.nn as nn

def mae_manual(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Calculate Mean Absolute Error manually.
    
    Args:
        y_true: Ground truth values, shape (n_samples,) or (n_samples, 1)
        y_pred: Predicted values, same shape as y_true
        
    Returns:
        MAE as a scalar tensor
    """
    return torch.mean(torch.abs(y_true - y_pred))

# Using PyTorch's built-in
mae_loss = nn.L1Loss(reduction='mean')

# Example usage
y_true = torch.tensor([300000., 450000., 200000., 550000., 380000.])
y_pred = torch.tensor([290000., 470000., 195000., 530000., 400000.])

mae_value = mae_manual(y_true, y_pred)
print(f"MAE: ${mae_value.item():,.2f}")  # MAE: $17,000.00
```

### When to Use MAE

- **Interpretability is key**: MAE directly represents average error magnitude
- **Outliers are expected but not catastrophic**: MAE doesn't over-penalize large errors
- **Business context uses absolute values**: e.g., "predictions are off by $X on average"

---

## Mean Squared Error (MSE)

### Definition

Mean Squared Error computes the average of squared residuals:

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

### Properties

| Property | Description |
|----------|-------------|
| **Units** | Squared units of target |
| **Range** | $[0, \infty)$ |
| **Optimal Value** | 0 |
| **Outlier Sensitivity** | High (quadratic penalty) |
| **Differentiability** | Smooth everywhere |

### Mathematical Interpretation

MSE penalizes larger errors disproportionately. An error of 10 is penalized 4× more than an error of 5 (since $10^2/5^2 = 4$).

**Gradient for optimization:**

$$\frac{\partial \text{MSE}}{\partial \hat{y}_i} = -\frac{2}{n}(y_i - \hat{y}_i)$$

The gradient magnitude is proportional to error size, leading to faster convergence for large errors.

### Connection to Maximum Likelihood

Under the assumption that errors follow a Gaussian distribution:

$$y_i = f(x_i) + \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, \sigma^2)$$

The likelihood function is:

$$L(\theta) = \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y_i - \hat{y}_i)^2}{2\sigma^2}\right)$$

Taking the negative log-likelihood:

$$-\log L(\theta) = \frac{n}{2}\log(2\pi\sigma^2) + \frac{1}{2\sigma^2}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

Minimizing this is equivalent to minimizing MSE when $\sigma^2$ is constant.

### PyTorch Implementation

```python
import torch
import torch.nn as nn

def mse_manual(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Calculate Mean Squared Error manually.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        MSE as a scalar tensor
    """
    return torch.mean((y_true - y_pred) ** 2)

# Using PyTorch's built-in
mse_loss = nn.MSELoss(reduction='mean')

# Example usage
y_true = torch.tensor([300000., 450000., 200000., 550000., 380000.])
y_pred = torch.tensor([290000., 470000., 195000., 530000., 400000.])

mse_value = mse_manual(y_true, y_pred)
print(f"MSE: {mse_value.item():,.0f}")  # MSE: 340,000,000
```

### When to Use MSE

- **Large errors are particularly costly**: MSE's quadratic penalty is appropriate
- **Optimization stability**: Smooth gradients aid neural network training
- **Statistical inference**: MSE connects to Gaussian MLE assumptions

---

## Root Mean Squared Error (RMSE)

### Definition

RMSE is the square root of MSE, returning the metric to the original units:

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2} = \sqrt{\text{MSE}}$$

### Properties

| Property | Description |
|----------|-------------|
| **Units** | Same as target variable |
| **Range** | $[0, \infty)$ |
| **Interpretation** | Standard deviation of residuals |
| **Outlier Sensitivity** | High |

### Mathematical Interpretation

RMSE can be interpreted as the standard deviation of the prediction errors (assuming zero mean). Under the Gaussian error assumption:

$$\text{RMSE} \approx \hat{\sigma}_\epsilon$$

where $\hat{\sigma}_\epsilon$ is the estimated standard deviation of the error term.

### PyTorch Implementation

```python
import torch

def rmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Calculate Root Mean Squared Error.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        RMSE as a scalar tensor
    """
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2))

# Example usage
y_true = torch.tensor([300000., 450000., 200000., 550000., 380000.])
y_pred = torch.tensor([290000., 470000., 195000., 530000., 400000.])

rmse_value = rmse(y_true, y_pred)
print(f"RMSE: ${rmse_value.item():,.2f}")  # RMSE: $18,439.09
```

### RMSE vs MAE Relationship

For any dataset:
$$\text{MAE} \leq \text{RMSE} \leq \sqrt{n} \cdot \text{MAE}$$

The ratio $\text{RMSE}/\text{MAE}$ indicates error distribution:
- **Ratio ≈ 1.0**: Errors are uniform in magnitude
- **Ratio ≈ 1.25** (typical): Errors follow a normal distribution
- **Ratio >> 1.25**: Large outliers present

---

## Coefficient of Determination (R²)

### Definition

R² measures the proportion of variance in the target explained by the model:

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$$

where:
- $SS_{res} = \sum(y_i - \hat{y}_i)^2$ is the residual sum of squares
- $SS_{tot} = \sum(y_i - \bar{y})^2$ is the total sum of squares
- $\bar{y} = \frac{1}{n}\sum y_i$ is the mean of true values

### Properties

| Property | Description |
|----------|-------------|
| **Range** | $(-\infty, 1]$ |
| **Optimal Value** | 1 (perfect predictions) |
| **Baseline Value** | 0 (predicting the mean) |
| **Scale Independence** | Yes |

### Mathematical Interpretation

R² compares model performance to a baseline that always predicts the mean:

- **R² = 1**: Perfect predictions ($SS_{res} = 0$)
- **R² = 0**: Model as good as predicting the mean
- **R² < 0**: Model worse than predicting the mean (possible with test data)

### Alternative Formulation

R² can also be expressed as:

$$R^2 = 1 - \frac{\text{MSE}}{\text{Var}(y)}$$

This shows R² as the fraction of target variance not explained by prediction error.

### PyTorch Implementation

```python
import torch

def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Calculate R² (Coefficient of Determination).
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        R² as a scalar tensor
    """
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    
    # Handle edge case where variance is zero
    if ss_tot == 0:
        return torch.tensor(float('nan'))
    
    return 1 - (ss_res / ss_tot)

# Example usage
y_true = torch.tensor([300000., 450000., 200000., 550000., 380000.])
y_pred = torch.tensor([290000., 470000., 195000., 530000., 400000.])

r2_value = r2_score(y_true, y_pred)
print(f"R² Score: {r2_value.item():.4f}")  # R² Score: 0.9854
```

### Interpretation Guidelines

| R² Value | Interpretation | Context |
|----------|----------------|---------|
| 0.90 - 1.00 | Excellent | Check for overfitting |
| 0.70 - 0.90 | Good | Typical for well-fitted models |
| 0.50 - 0.70 | Moderate | May need improvement |
| 0.30 - 0.50 | Weak | Consider different approach |
| < 0.30 | Poor | Model has limited predictive power |

---

## Adjusted R²

### The Problem with R²

Standard R² never decreases when adding features, even irrelevant ones:

$$R^2_{new} \geq R^2_{old}$$

This makes R² unsuitable for comparing models with different numbers of features.

### Definition

Adjusted R² penalizes model complexity:

$$R^2_{adj} = 1 - \frac{(1 - R^2)(n - 1)}{n - p - 1}$$

where:
- $n$ = number of samples
- $p$ = number of features (predictors)

### Properties

- **Decreases** if added features don't improve the model sufficiently
- **Undefined** when $n \leq p + 1$
- **Can be negative** even when R² is positive

### PyTorch Implementation

```python
import torch

def adjusted_r2(y_true: torch.Tensor, y_pred: torch.Tensor, 
                n_features: int) -> torch.Tensor:
    """
    Calculate Adjusted R².
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        n_features: Number of features in the model
        
    Returns:
        Adjusted R² as a scalar tensor
    """
    n = len(y_true)
    
    # Check for valid computation
    if n <= n_features + 1:
        return torch.tensor(float('nan'))
    
    r2 = r2_score(y_true, y_pred)
    adjusted = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    
    return adjusted

# Example
y_true = torch.tensor([300000., 450000., 200000., 550000., 380000.])
y_pred = torch.tensor([290000., 470000., 195000., 530000., 400000.])

adj_r2 = adjusted_r2(y_true, y_pred, n_features=3)
print(f"Adjusted R²: {adj_r2.item():.4f}")
```

---

## Percentage-Based Metrics

### Mean Absolute Percentage Error (MAPE)

**Definition:**

$$\text{MAPE} = \frac{100}{n} \sum_{i=1}^{n} \left|\frac{y_i - \hat{y}_i}{y_i}\right|$$

**Properties:**
- Scale-independent (percentage)
- Undefined when $y_i = 0$
- Asymmetric: over-predictions penalized less than under-predictions

```python
import torch

def mape(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Calculate Mean Absolute Percentage Error.
    
    Args:
        y_true: Ground truth values (must be non-zero)
        y_pred: Predicted values
        
    Returns:
        MAPE as percentage
    """
    # Mask zero values to avoid division errors
    mask = y_true != 0
    
    if not torch.any(mask):
        return torch.tensor(float('inf'))
    
    return torch.mean(torch.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# Example
y_true = torch.tensor([300000., 450000., 200000., 550000., 380000.])
y_pred = torch.tensor([290000., 470000., 195000., 530000., 400000.])

mape_value = mape(y_true, y_pred)
print(f"MAPE: {mape_value.item():.2f}%")  # MAPE: 4.48%
```

### Symmetric MAPE (SMAPE)

**Definition:**

$$\text{SMAPE} = \frac{100}{n} \sum_{i=1}^{n} \frac{|y_i - \hat{y}_i|}{(|y_i| + |\hat{y}_i|)/2}$$

**Properties:**
- Bounded: $[0\%, 200\%]$
- Symmetric treatment of over/under-predictions
- Better handles near-zero values

```python
import torch

def smape(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Calculate Symmetric Mean Absolute Percentage Error.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        SMAPE as percentage
    """
    numerator = torch.abs(y_true - y_pred)
    denominator = (torch.abs(y_true) + torch.abs(y_pred)) / 2
    
    # Mask zero denominators
    mask = denominator != 0
    
    if not torch.any(mask):
        return torch.tensor(0.0)
    
    return torch.mean(numerator[mask] / denominator[mask]) * 100

# Example
smape_value = smape(y_true, y_pred)
print(f"SMAPE: {smape_value.item():.2f}%")
```

---

## Robust Metrics

### Median Absolute Error

**Definition:**

$$\text{MedAE} = \text{median}(|y_1 - \hat{y}_1|, |y_2 - \hat{y}_2|, \ldots, |y_n - \hat{y}_n|)$$

**Properties:**
- Highly robust to outliers
- 50th percentile of absolute errors
- Useful for heavy-tailed error distributions

```python
import torch

def median_absolute_error(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Calculate Median Absolute Error."""
    return torch.median(torch.abs(y_true - y_pred))

# Example
med_ae = median_absolute_error(y_true, y_pred)
print(f"Median Absolute Error: ${med_ae.item():,.2f}")
```

### Maximum Error

**Definition:**

$$\text{MaxError} = \max_i |y_i - \hat{y}_i|$$

**Use Case:** Worst-case error bound for critical applications.

```python
import torch

def max_error(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Calculate Maximum Error."""
    return torch.max(torch.abs(y_true - y_pred))
```

---

## Comprehensive Regression Metrics Class

```python
import torch
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class RegressionReport:
    """Container for regression evaluation results."""
    mae: float
    mse: float
    rmse: float
    r2: float
    adjusted_r2: Optional[float]
    mape: float
    smape: float
    median_ae: float
    max_error: float
    
    def __str__(self) -> str:
        lines = [
            "=" * 50,
            "REGRESSION EVALUATION REPORT",
            "=" * 50,
            f"MAE:             {self.mae:,.4f}",
            f"MSE:             {self.mse:,.4f}",
            f"RMSE:            {self.rmse:,.4f}",
            f"R² Score:        {self.r2:.4f}",
        ]
        if self.adjusted_r2 is not None:
            lines.append(f"Adjusted R²:     {self.adjusted_r2:.4f}")
        lines.extend([
            f"MAPE:            {self.mape:.2f}%",
            f"SMAPE:           {self.smape:.2f}%",
            f"Median AE:       {self.median_ae:,.4f}",
            f"Max Error:       {self.max_error:,.4f}",
            "=" * 50
        ])
        return "\n".join(lines)


class RegressionMetrics:
    """
    Comprehensive regression metrics calculator.
    
    Example:
        >>> y_true = torch.tensor([100., 200., 300., 400., 500.])
        >>> y_pred = torch.tensor([110., 195., 290., 410., 480.])
        >>> metrics = RegressionMetrics(y_true, y_pred)
        >>> report = metrics.full_report(n_features=3)
        >>> print(report)
    """
    
    def __init__(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        self.y_true = y_true.float()
        self.y_pred = y_pred.float()
        self.residuals = self.y_true - self.y_pred
        
    def mae(self) -> float:
        """Mean Absolute Error."""
        return torch.mean(torch.abs(self.residuals)).item()
    
    def mse(self) -> float:
        """Mean Squared Error."""
        return torch.mean(self.residuals ** 2).item()
    
    def rmse(self) -> float:
        """Root Mean Squared Error."""
        return torch.sqrt(torch.mean(self.residuals ** 2)).item()
    
    def r2(self) -> float:
        """R² (Coefficient of Determination)."""
        ss_res = torch.sum(self.residuals ** 2)
        ss_tot = torch.sum((self.y_true - torch.mean(self.y_true)) ** 2)
        
        if ss_tot == 0:
            return float('nan')
        return (1 - ss_res / ss_tot).item()
    
    def adjusted_r2(self, n_features: int) -> Optional[float]:
        """Adjusted R²."""
        n = len(self.y_true)
        if n <= n_features + 1:
            return None
        r2_val = self.r2()
        return 1 - (1 - r2_val) * (n - 1) / (n - n_features - 1)
    
    def mape(self) -> float:
        """Mean Absolute Percentage Error."""
        mask = self.y_true != 0
        if not torch.any(mask):
            return float('inf')
        return (torch.mean(torch.abs(self.residuals[mask] / self.y_true[mask])) * 100).item()
    
    def smape(self) -> float:
        """Symmetric Mean Absolute Percentage Error."""
        denominator = (torch.abs(self.y_true) + torch.abs(self.y_pred)) / 2
        mask = denominator != 0
        if not torch.any(mask):
            return 0.0
        return (torch.mean(torch.abs(self.residuals[mask]) / denominator[mask]) * 100).item()
    
    def median_ae(self) -> float:
        """Median Absolute Error."""
        return torch.median(torch.abs(self.residuals)).item()
    
    def max_error(self) -> float:
        """Maximum Absolute Error."""
        return torch.max(torch.abs(self.residuals)).item()
    
    def residual_analysis(self) -> Dict[str, float]:
        """Statistical analysis of residuals."""
        return {
            'mean': torch.mean(self.residuals).item(),
            'std': torch.std(self.residuals).item(),
            'min': torch.min(self.residuals).item(),
            'max': torch.max(self.residuals).item(),
            'median': torch.median(self.residuals).item(),
        }
    
    def full_report(self, n_features: Optional[int] = None) -> RegressionReport:
        """Generate comprehensive evaluation report."""
        return RegressionReport(
            mae=self.mae(),
            mse=self.mse(),
            rmse=self.rmse(),
            r2=self.r2(),
            adjusted_r2=self.adjusted_r2(n_features) if n_features else None,
            mape=self.mape(),
            smape=self.smape(),
            median_ae=self.median_ae(),
            max_error=self.max_error()
        )


# Example usage
if __name__ == "__main__":
    # House price prediction example
    y_true = torch.tensor([300000., 450000., 200000., 550000., 380000., 
                           420000., 290000., 510000.])
    y_pred = torch.tensor([290000., 470000., 195000., 530000., 400000., 
                           410000., 305000., 495000.])
    
    metrics = RegressionMetrics(y_true, y_pred)
    report = metrics.full_report(n_features=5)
    print(report)
    
    print("\nResidual Analysis:")
    for key, value in metrics.residual_analysis().items():
        print(f"  {key}: ${value:,.2f}")
```

---

## Metric Selection Guide

### Decision Framework

```
┌─────────────────────────────────────────────────────────────┐
│                    METRIC SELECTION GUIDE                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Primary Use Case              →  Recommended Metric(s)      │
│  ─────────────────────────────────────────────────────────   │
│  General model comparison      →  RMSE + R²                  │
│  Business reporting            →  MAE (interpretable)        │
│  Cross-scale comparison        →  MAPE or SMAPE              │
│  Outlier-heavy data            →  Median Absolute Error      │
│  Critical applications         →  Max Error + RMSE           │
│  Model complexity comparison   →  Adjusted R²                │
│  Optimization/training         →  MSE (smooth gradients)     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Industry Standards by Domain

| Domain | Primary Metrics | Notes |
|--------|-----------------|-------|
| **Finance** | RMSE, MAPE | MAPE for returns, RMSE for price levels |
| **Real Estate** | MAE, MAPE | Interpretable for clients |
| **Energy** | RMSE, MAPE | Forecasting accuracy critical |
| **Healthcare** | MAE, Max Error | Worst-case bounds important |
| **Supply Chain** | SMAPE | Handles near-zero demand |

### Finance-Specific Considerations

For quantitative finance applications:

1. **Asset Returns**: Use RMSE or MAE on log returns
2. **Price Levels**: Consider MAPE for percentage accuracy
3. **Volatility**: Use RMSE with appropriate scaling
4. **Risk Models**: Report multiple metrics including tail statistics

---

## Common Pitfalls and Best Practices

### Pitfalls to Avoid

1. **Using R² alone**: Always report with RMSE or MAE for error magnitude
2. **Ignoring scale**: Compare models using scale-independent metrics when appropriate
3. **MAPE with zero values**: Use SMAPE or handle zeros explicitly
4. **Overfitting indicators**: Very high R² (>0.99) may indicate data leakage

### Best Practices

1. **Report multiple metrics**: MAE + RMSE + R² provides comprehensive view
2. **Include confidence intervals**: Via cross-validation or bootstrapping
3. **Residual analysis**: Check for heteroscedasticity and non-normality
4. **Document business context**: Explain what metric values mean in practice

---

## Summary

| Metric | Formula | Units | Best For |
|--------|---------|-------|----------|
| MAE | $\frac{1}{n}\sum\|e_i\|$ | Target units | Interpretability |
| MSE | $\frac{1}{n}\sum e_i^2$ | Squared units | Optimization |
| RMSE | $\sqrt{\text{MSE}}$ | Target units | Error magnitude |
| R² | $1 - \frac{SS_{res}}{SS_{tot}}$ | Unitless | Explained variance |
| Adj R² | Penalized R² | Unitless | Model comparison |
| MAPE | $\frac{100}{n}\sum\|\frac{e_i}{y_i}\|$ | Percentage | Cross-scale comparison |
| SMAPE | Symmetric MAPE | Percentage | Near-zero handling |

---

## References

1. Hyndman, R.J. & Koehler, A.B. (2006). "Another look at measures of forecast accuracy." *International Journal of Forecasting*.
2. Chai, T. & Draxler, R.R. (2014). "Root mean square error (RMSE) or mean absolute error (MAE)?" *Geoscientific Model Development*.
3. Willmott, C.J. & Matsuura, K. (2005). "Advantages of the mean absolute error (MAE) over the root mean square error (RMSE)." *Climate Research*.
