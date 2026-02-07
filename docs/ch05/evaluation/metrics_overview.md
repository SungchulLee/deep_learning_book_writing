# Metrics Overview

## Overview

Evaluation metrics quantify model performance and guide model selection. The choice of metric should align with the problem's objectives—a model optimized for accuracy may be poorly calibrated, and a model with low MSE may produce unacceptable tail errors.

## Classification vs. Regression

**Classification metrics** evaluate discrete predictions: accuracy, precision, recall, F1, ROC-AUC, and calibration. The appropriate metric depends on class balance, cost asymmetry, and whether probabilistic or hard predictions are needed.

**Regression metrics** evaluate continuous predictions: MSE, MAE, R², MAPE. The appropriate metric depends on error symmetry, outlier sensitivity, and scale.

## Metric Selection Framework

| Consideration | Preferred Metrics |
|---|---|
| Balanced classes | Accuracy, F1 |
| Imbalanced classes | Precision, Recall, F1 (macro/weighted), ROC-AUC |
| Cost-sensitive | Weighted precision/recall matching cost structure |
| Probabilistic outputs | Log loss, Brier score, calibration |
| Outlier-sensitive | MAE, quantile losses |
| Scale-independent | R², MAPE |

## Quantitative Finance Metrics

Standard ML metrics often fail to capture financial objectives:

- **Sharpe ratio**: Risk-adjusted return, more relevant than MSE for trading strategies.
- **Maximum drawdown**: Largest peak-to-trough decline, captures tail risk.
- **Calibration error**: Critical for option pricing—models must produce accurate probabilities.
- **P&L distribution**: The full distribution of profit and loss, not just its mean.

## Key Takeaways

- Choose metrics that align with the problem's economic or scientific objectives.
- Always report multiple metrics to provide a complete picture.
- In finance, standard ML metrics are necessary but insufficient—domain-specific metrics are essential.
