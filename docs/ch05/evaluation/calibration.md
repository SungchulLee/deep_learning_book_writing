# Calibration Metrics

## Overview

A model is **well-calibrated** when its predicted probabilities reflect true frequencies: among all samples predicted to be positive with probability 0.8, approximately 80% should actually be positive. Calibration is critical for decision-making under uncertainty, particularly in finance and risk management.

## Expected Calibration Error (ECE)

ECE partitions predictions into $B$ bins by predicted probability and computes the weighted average of the gap between predicted confidence and observed accuracy:

$$\text{ECE} = \sum_{b=1}^B \frac{n_b}{N} |\text{acc}(b) - \text{conf}(b)|$$

```python
def expected_calibration_error(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i+1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += mask.sum() / len(y_true) * abs(bin_acc - bin_conf)
    return ece
```

## Reliability Diagram

A reliability diagram plots observed frequency vs. predicted probability. A perfectly calibrated model lies on the diagonal.

## Brier Score

$$\text{Brier} = \frac{1}{N}\sum_{i=1}^N (p_i - y_i)^2$$

The Brier score combines calibration and discrimination into a single metric. It decomposes into reliability (calibration), resolution (discrimination), and uncertainty components.

## Calibration Methods

**Temperature scaling**: A simple post-hoc calibration method that learns a single temperature parameter $T$ on the validation set:

$$p_{\text{calibrated}} = \text{softmax}(z / T)$$

```python
class TemperatureScaling(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        return logits / self.temperature
```

**Platt scaling**: Fits a logistic regression on the logits using the validation set.

## Financial Importance

In option pricing, model-implied probabilities must be well-calibrated for hedging to work correctly. A neural pricing model that accurately predicts option prices but produces miscalibrated implied volatilities will generate poor hedging performance.

## Key Takeaways

- Calibration measures whether predicted probabilities are trustworthy.
- ECE and Brier score are the standard calibration metrics.
- Temperature scaling is a simple, effective post-hoc calibration method.
- Calibration is essential for financial applications where probabilities drive decisions.
