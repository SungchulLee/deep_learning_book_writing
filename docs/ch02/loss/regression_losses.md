# Regression Losses

Regression losses measure prediction errors for continuous target variables. The choice among them encodes assumptions about error distributions and desired robustness properties. This section examines the three primary regression losses and their mathematical foundations.

## Mean Squared Error (MSE) — L2 Loss

Mean Squared Error is the default choice for regression, arising from the assumption that errors follow a Gaussian distribution.

### Mathematical Definition

$$\mathcal{L}_{\text{MSE}} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

The squared term has two important consequences: all errors become positive, and larger errors receive disproportionately higher penalties. An error of magnitude 10 contributes 100 to the loss, while an error of magnitude 1 contributes only 1.

### Properties and Gradient Behavior

The gradient of MSE with respect to predictions is:

$$\frac{\partial \mathcal{L}_{\text{MSE}}}{\partial \hat{y}_i} = \frac{2}{n}(\hat{y}_i - y_i)$$

This linear gradient relationship means the optimization signal scales proportionally with error magnitude—larger errors produce stronger gradients, pushing the model to prioritize reducing them.

### PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Sample data with an outlier
actual = torch.tensor([85.0, 90.0, 88.0, 92.0, 15.0])  # Note: 15 is an outlier
predicted = torch.tensor([84.0, 89.0, 87.0, 91.0, 87.0])

# Using nn.MSELoss
mse_criterion = nn.MSELoss()
mse_loss = mse_criterion(predicted, actual)
print(f"MSE Loss: {mse_loss.item():.4f}")  # 1040.8

# Root Mean Squared Error for interpretability
rmse = torch.sqrt(mse_loss)
print(f"RMSE: {rmse.item():.4f}")  # ~32.26 (same units as target)
```

### Analyzing Outlier Sensitivity

```python
# Examine per-sample squared errors
squared_errors = (predicted - actual) ** 2
print(f"Squared errors: {squared_errors}")
# tensor([1., 1., 1., 1., 5184.])

# The outlier dominates the total loss
outlier_contribution = squared_errors[4] / squared_errors.sum()
print(f"Outlier contributes {outlier_contribution.item()*100:.1f}% of total loss")
# Outlier contributes 99.9% of total loss
```

**Characteristics:**

- Smooth, differentiable everywhere
- Sensitive to outliers (squared penalty)
- Optimal under Gaussian noise assumption
- Best when large errors are genuinely problematic

## Mean Absolute Error (MAE) — L1 Loss

Mean Absolute Error uses absolute differences, providing robustness to outliers at the cost of non-smooth gradients.

### Mathematical Definition

$$\mathcal{L}_{\text{MAE}} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

### Gradient Behavior

$$\frac{\partial \mathcal{L}_{\text{MAE}}}{\partial \hat{y}_i} = \frac{1}{n} \cdot \text{sign}(\hat{y}_i - y_i)$$

The gradient magnitude is constant (±1/n) regardless of error size. This means all errors contribute equally to the optimization signal—a double-edged property that provides outlier robustness but can slow convergence near the optimum.

!!! warning "Non-Differentiability"
    MAE is not differentiable at zero error. PyTorch handles this through subgradients, but this can occasionally cause optimization instability.

### PyTorch Implementation

```python
# Using nn.L1Loss (PyTorch's name for MAE)
mae_criterion = nn.L1Loss()
mae_loss = mae_criterion(predicted, actual)
print(f"MAE Loss: {mae_loss.item():.4f}")  # 14.8

# Compare individual contributions
absolute_errors = torch.abs(predicted - actual)
print(f"Absolute errors: {absolute_errors}")
# tensor([1., 1., 1., 1., 72.])
```

### Outlier Impact Comparison

```python
# Without outlier
mse_no_outlier = F.mse_loss(predicted[:4], actual[:4])
mae_no_outlier = F.l1_loss(predicted[:4], actual[:4])

print(f"MSE without outlier: {mse_no_outlier.item():.4f}")  # 1.0
print(f"MAE without outlier: {mae_no_outlier.item():.4f}")  # 1.0

# Impact analysis
mse_impact = ((mse_loss - mse_no_outlier) / mse_no_outlier * 100).item()
mae_impact = ((mae_loss - mae_no_outlier) / mae_no_outlier * 100).item()
print(f"MSE increased by: {mse_impact:.1f}%")  # 103,980%
print(f"MAE increased by: {mae_impact:.1f}%")  # 1,380%
```

The outlier inflates MSE by over 1000× but MAE by only ~14×—demonstrating MAE's superior robustness.

**Characteristics:**

- Robust to outliers (linear penalty)
- Constant gradient magnitude
- Non-differentiable at zero
- Corresponds to Laplace distribution assumption

## Smooth L1 Loss (Huber Loss)

Smooth L1 Loss combines the benefits of both MSE and MAE through a piecewise definition: quadratic for small errors, linear for large errors.

### Mathematical Definition

$$\mathcal{L}_{\text{Huber}} = \begin{cases} 
\frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| < \delta \\
\delta |y - \hat{y}| - \frac{1}{2}\delta^2 & \text{otherwise}
\end{cases}$$

where $\delta$ (often called `beta` in PyTorch) controls the transition point. PyTorch's `SmoothL1Loss` uses $\delta = 1$ by default.

### Gradient Behavior

$$\frac{\partial \mathcal{L}_{\text{Huber}}}{\partial \hat{y}} = \begin{cases}
\hat{y} - y & \text{if } |y - \hat{y}| < \delta \\
\delta \cdot \text{sign}(\hat{y} - y) & \text{otherwise}
\end{cases}$$

Near the optimum (small errors), gradients scale linearly like MSE, enabling precise fine-tuning. For large errors, gradients are bounded like MAE, preventing outlier-driven instability.

### PyTorch Implementation

```python
# Default SmoothL1Loss (beta=1.0)
smooth_l1_criterion = nn.SmoothL1Loss()
smooth_l1_loss = smooth_l1_criterion(predicted, actual)
print(f"Smooth L1 Loss: {smooth_l1_loss.item():.4f}")  # 14.3

# Custom beta (transition threshold)
smooth_l1_beta5 = nn.SmoothL1Loss(beta=5.0)
loss_beta5 = smooth_l1_beta5(predicted, actual)
print(f"Smooth L1 (beta=5): {loss_beta5.item():.4f}")
```

### Visualizing the Transition

```python
# Demonstrate which regime each error falls into
errors = actual - predicted
for i, error in enumerate(errors):
    abs_error = abs(error.item())
    regime = "MSE regime (quadratic)" if abs_error < 1 else "MAE regime (linear)"
    print(f"Error {i+1}: {error.item():6.1f} → {regime}")

# Output:
# Error 1:   -1.0 → MAE regime (linear)
# Error 2:   -1.0 → MAE regime (linear)
# Error 3:   -1.0 → MAE regime (linear)
# Error 4:   -1.0 → MAE regime (linear)
# Error 5:  -72.0 → MAE regime (linear)
```

**Characteristics:**

- Smooth, differentiable everywhere
- Quadratic near optimum (precise convergence)
- Linear for large errors (outlier robustness)
- Standard in object detection (bounding box regression)

## Comparative Analysis

### Experimental Comparison

```python
# Three scenarios
scenarios = {
    "Clean data": (
        torch.tensor([85.0, 90.0, 88.0, 92.0, 87.0]),
        torch.tensor([84.0, 89.0, 87.0, 91.0, 86.0])
    ),
    "Moderate errors": (
        torch.tensor([85.0, 90.0, 88.0, 92.0, 87.0]),
        torch.tensor([80.0, 85.0, 83.0, 87.0, 82.0])
    ),
    "With outlier": (
        torch.tensor([85.0, 90.0, 88.0, 92.0, 15.0]),
        torch.tensor([84.0, 89.0, 87.0, 91.0, 87.0])
    )
}

for name, (actual, pred) in scenarios.items():
    mse = F.mse_loss(pred, actual)
    mae = F.l1_loss(pred, actual)
    smooth = F.smooth_l1_loss(pred, actual)
    print(f"\n{name}:")
    print(f"  MSE:       {mse.item():8.2f}")
    print(f"  MAE:       {mae.item():8.2f}")
    print(f"  Smooth L1: {smooth.item():8.2f}")
```

### Gradient Magnitude Impact

```python
# For a large error (outlier), compare gradient magnitudes
outlier_error = torch.tensor(72.0, requires_grad=True)

# MSE gradient
mse_example = outlier_error ** 2 / 2
mse_example.backward()
print(f"MSE gradient magnitude: {abs(outlier_error.grad.item()):.1f}")  # 72.0

# Reset and compute MAE gradient
outlier_error = torch.tensor(72.0, requires_grad=True)
mae_example = torch.abs(outlier_error)
mae_example.backward()
print(f"MAE gradient magnitude: {abs(outlier_error.grad.item()):.1f}")  # 1.0
```

MSE produces a gradient 72× larger for this outlier, meaning the model will update much more aggressively to fit outliers when using MSE.

## Decision Framework

| Criterion | MSE | MAE | Smooth L1 |
|-----------|-----|-----|-----------|
| **Outlier sensitivity** | High | Low | Medium |
| **Gradient near optimum** | Smooth, scaled | Constant | Smooth, scaled |
| **Differentiability** | Everywhere | Not at 0 | Everywhere |
| **Distributional assumption** | Gaussian | Laplace | Hybrid |

### Use MSE when:
- Data is clean with minimal outliers
- Large errors are genuinely problematic
- You want smooth optimization dynamics
- Example: Price prediction in stable markets

### Use MAE when:
- Data contains outliers or heavy-tailed noise
- All errors should be treated equally
- You want error in the same units as the target
- Example: Delivery time prediction (traffic outliers common)

### Use Smooth L1 when:
- You want outlier robustness with smooth gradients
- The data has occasional anomalies
- Training stability is a concern
- Example: Object detection bounding box regression

## Key Takeaways

The choice of regression loss encodes assumptions about your data and optimization priorities. MSE provides smooth gradients and optimal performance under Gaussian noise but is vulnerable to outliers. MAE offers robustness but with constant gradients that can slow convergence. Smooth L1 provides a principled middle ground, making it an excellent default choice for real-world data where outliers are possible but not dominant. All three losses are fully compatible with PyTorch's autograd system and can be seamlessly integrated into training pipelines.
