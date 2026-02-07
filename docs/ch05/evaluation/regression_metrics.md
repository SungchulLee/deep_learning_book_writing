# Regression Metrics

## Overview

Regression metrics evaluate continuous-valued predictions. Each metric emphasizes different aspects of prediction quality—average error, outlier sensitivity, relative accuracy, or distributional properties.

## Mean Squared Error (MSE)

$$\text{MSE} = \frac{1}{N}\sum_{i=1}^N (y_i - \hat{y}_i)^2$$

MSE penalizes large errors quadratically, making it sensitive to outliers. It is the default loss function for regression and the natural metric when errors are Gaussian.

```python
mse = F.mse_loss(predictions, targets).item()
rmse = mse ** 0.5  # Root MSE, in the same units as the target
```

## Mean Absolute Error (MAE)

$$\text{MAE} = \frac{1}{N}\sum_{i=1}^N |y_i - \hat{y}_i|$$

MAE is more robust to outliers than MSE. It corresponds to the median regression under the Laplace error distribution.

```python
mae = F.l1_loss(predictions, targets).item()
```

## R² (Coefficient of Determination)

$$R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}$$

$R^2$ measures the proportion of variance explained by the model. $R^2 = 1$ indicates perfect prediction; $R^2 = 0$ indicates the model is no better than predicting the mean; $R^2 < 0$ indicates the model is worse than the mean.

## Mean Absolute Percentage Error (MAPE)

$$\text{MAPE} = \frac{100\%}{N}\sum_{i=1}^N \left|\frac{y_i - \hat{y}_i}{y_i}\right|$$

MAPE provides a scale-independent measure but is undefined when $y_i = 0$ and asymmetric (overestimates are penalized differently than underestimates of the same magnitude).

## Quantile Loss

For predicting conditional quantiles rather than the mean:

$$\mathcal{L}_\tau(y, \hat{y}) = \begin{cases} \tau(y - \hat{y}) & \text{if } y \geq \hat{y} \\ (1-\tau)(\hat{y} - y) & \text{if } y < \hat{y} \end{cases}$$

This is essential for financial risk metrics like Value-at-Risk (VaR).

## Financial Regression Metrics

| Metric | Use Case |
|---|---|
| MSE/RMSE | General-purpose, pricing model calibration |
| MAE | Robust evaluation, trading signal quality |
| R² | Predictive power relative to baseline |
| Quantile loss | VaR estimation, tail risk |
| Directional accuracy | Sign of return prediction |

## Key Takeaways

- MSE emphasizes large errors; MAE is robust to outliers.
- R² measures explanatory power relative to a mean-prediction baseline.
- In finance, directional accuracy and quantile losses are often more relevant than MSE.
