# MSE and MAE

Mean Squared Error and Mean Absolute Error are the two foundational regression losses, each arising from maximum likelihood estimation under a specific noise model. MSE assumes Gaussian noise and penalizes large errors quadratically; MAE assumes Laplace noise and penalizes all errors linearly. This section derives both from first principles, establishes their mathematical properties, and demonstrates the complete PyTorch training pipeline.

## MSE: Gaussian Noise Model

### Data and Model

Given paired observations $\{(x^{(i)}, y^{(i)}): i=1,\ldots,m\}$, we posit a linear relationship corrupted by Gaussian noise:

$$y^{(i)} = \alpha + \beta x^{(i)} + \varepsilon^{(i)}, \qquad \varepsilon^{(i)} \sim \mathcal{N}(0, \sigma^2)$$

independently for each observation. The conditional distribution is:

$$y^{(i)} \mid x^{(i)} \sim \mathcal{N}\!\left(\alpha + \beta x^{(i)},\; \sigma^2\right)$$

### Likelihood and Log-Likelihood

Since the observations are independent, the joint likelihood under parameters $(\alpha, \beta)$ with fixed $\sigma^2$ is:

$$L(\alpha,\beta) = \prod_{i=1}^m \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left(-\frac{1}{2\sigma^2}\left(y^{(i)} - \alpha - \beta x^{(i)}\right)^2\right)$$

Taking the natural logarithm converts the product into a sum:

$$\ell(\alpha,\beta) = -\frac{1}{2\sigma^2}\sum_{i=1}^m \left(y^{(i)} - \alpha - \beta x^{(i)}\right)^2 - \frac{m}{2}\log(2\pi\sigma^2)$$

### From Log-Likelihood to Cost Function

Maximizing $\ell$ is equivalent to minimizing its negation. Dropping constant terms and positive scaling factors:

$$J(\alpha,\beta) = \frac{1}{2}\sum_{i=1}^m \left(y^{(i)} - \alpha - \beta x^{(i)}\right)^2$$

This is the **square cost function**. The factor of $\frac{1}{2}$ is a convention that cancels the power of 2 during differentiation. Normalizing by $m$ gives the Mean Squared Error:

$$\text{MSE} = \frac{1}{m}\sum_{i=1}^m \left(y^{(i)} - \alpha - \beta x^{(i)}\right)^2$$

The equivalence chain confirms that minimizing MSE is maximum likelihood estimation:

$$\underset{\alpha,\beta}{\operatorname{argmax}}\ L \quad\Leftrightarrow\quad \underset{\alpha,\beta}{\operatorname{argmax}}\ \ell \quad\Leftrightarrow\quad \underset{\alpha,\beta}{\operatorname{argmin}}\ J$$

The key insight is that **MSE is the negative log-likelihood of a Gaussian model**, up to an additive constant and a multiplicative factor of $\sigma^2$.

### Closed-Form MLE Solution

Setting the partial derivatives to zero yields the MLE estimates.

**For $\alpha$:**

$$\frac{\partial J}{\partial \alpha} = -\sum_{i=1}^m \left(y^{(i)} - \alpha - \beta x^{(i)}\right) = 0 \quad\Longrightarrow\quad \hat{\alpha} = \bar{y} - \hat{\beta}\bar{x}$$

where $\bar{x} = \frac{1}{m}\sum_{i} x^{(i)}$ and $\bar{y} = \frac{1}{m}\sum_{i} y^{(i)}$.

**For $\beta$:**

$$\frac{\partial J}{\partial \beta} = -\sum_{i=1}^m x^{(i)}\left(y^{(i)} - \alpha - \beta x^{(i)}\right) = 0 \quad\Longrightarrow\quad \hat{\beta} = \frac{\sum_{i=1}^m (x^{(i)} - \bar{x})(y^{(i)} - \bar{y})}{\sum_{i=1}^m (x^{(i)} - \bar{x})^2}$$

These are the **normal equations** whose closed-form solution gives the Ordinary Least Squares (OLS) estimators.

### Regression in Standardized Form

An elegant reformulation uses standardized variables:

$$z_x^{(i)} = \frac{x^{(i)} - \hat{\mu}_x}{\hat{\sigma}_x}, \qquad z_y^{(i)} = \frac{y^{(i)} - \hat{\mu}_y}{\hat{\sigma}_y}$$

In these coordinates, the regression line takes the remarkably simple form:

$$\frac{y - \hat{\mu}_y}{\hat{\sigma}_y} = \hat{\rho}\;\frac{x - \hat{\mu}_x}{\hat{\sigma}_x}$$

where $\hat{\rho} = \text{Corr}(x, y)$ is the sample correlation coefficient. This reveals that regression in standardized space has **zero intercept** and a **slope equal to the correlation**. The correlation $\hat{\rho}$ determines how much variation in $x$ translates to predicted variation in $y$: when $|\hat{\rho}| \approx 1$, predictions closely track the data; when $\hat{\rho} \approx 0$, the model predicts $\hat{\mu}_y$ regardless of $x$.

Converting back to original units recovers the usual estimates:

$$\hat{\beta} = \hat{\rho}\;\frac{\hat{\sigma}_y}{\hat{\sigma}_x}, \qquad \hat{\alpha} = \hat{\mu}_y - \hat{\beta}\;\hat{\mu}_x$$

### MSE Gradient Analysis

The gradient of MSE with respect to predictions is:

$$\frac{\partial \mathcal{L}_{\text{MSE}}}{\partial \hat{y}_i} = \frac{2}{m}(\hat{y}_i - y_i)$$

This linear relationship has important consequences. Near the optimum, gradients shrink naturally, enabling precise convergence without learning rate adjustment. For linear models, the MSE loss surface is a convex quadratic bowl with a unique global minimum (the OLS solution), and gradient descent is guaranteed to converge.

Under Gaussian noise, the MLE (which minimizes MSE) is the minimum-variance unbiased estimator by the Cramér–Rao bound. However, the squared penalty amplifies large errors—a single outlier with residual $r = 100$ contributes $10{,}000$ to the loss, potentially dominating the gradient signal.

### PyTorch: `nn.MSELoss` and `F.mse_loss`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

predictions = torch.tensor([140.0, 210.0, 245.0, 310.0, 360.0])
targets = torch.tensor([150.0, 200.0, 250.0, 300.0, 350.0])

# Functional API (stateless)
loss = F.mse_loss(predictions, targets)

# Module API (configurable, recommended)
criterion = nn.MSELoss()          # default: reduction='mean'
loss = criterion(predictions, targets)
```

Both compute the same quantity. The module API integrates cleanly with `nn.Module` containers and supports configuration via `reduction`.

### Complete Training Example

The following trains a linear model $y = wx + b$ on synthetic data from $y = 1 + 2x + \varepsilon$, $\varepsilon \sim \mathcal{N}(0, 0.01)$:

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

np.random.seed(1)
batch_size = 100

x_train = np.random.uniform(size=(batch_size, 1))
y_train = 1 + 2 * x_train + np.random.normal(scale=0.1, size=(batch_size, 1))

x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

train_ds = TensorDataset(x_train, y_train)
trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = nn.Linear(1, 1).to(device)
loss_fn = F.mse_loss
opt = optim.SGD(model.parameters(), lr=0.1)

w_trace, b_trace, loss_trace = [], [], []

for epoch in range(1000):
    for xb, yb in trainloader:
        xb, yb = xb.to(device), yb.to(device)

        preds = model(xb)          # 1. Forward pass
        loss = loss_fn(preds, yb)   # 2. Compute loss
        loss.backward()             # 3. Backpropagate
        opt.step()                  # 4. Update parameters
        opt.zero_grad()             # 5. Reset gradients

        w_trace.append(model.weight.item())
        b_trace.append(model.bias.item())
        loss_trace.append(loss.item())
```

The weight trace converges to approximately 2 (the true slope) and the bias to approximately 1 (the true intercept), with the loss decreasing monotonically toward the noise floor $\sigma^2 = 0.01$.

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(12, 3))

axes[0].plot(w_trace, label="estimated slope")
axes[0].axhline(y=2, color='r', linestyle='--', label="true slope")
axes[0].legend()

axes[1].plot(b_trace, label="estimated bias")
axes[1].axhline(y=1, color='r', linestyle='--', label="true bias")
axes[1].legend()

axes[2].plot(loss_trace, label="loss")
axes[2].legend()

plt.tight_layout()
plt.show()
```

---

## MAE: Laplace Noise Model

### Data and Model

Given paired observations $\{(x^{(i)}, y^{(i)}): i=1,\ldots,m\}$, suppose the relationship is corrupted by Laplace noise:

$$y^{(i)} = f_\theta(x^{(i)}) + \varepsilon^{(i)}, \qquad \varepsilon^{(i)} \sim \text{Laplace}(0, b)$$

where $b > 0$ is a fixed scale parameter. The Laplace distribution has density:

$$p(\varepsilon) = \frac{1}{2b}\exp\!\left(-\frac{|\varepsilon|}{b}\right)$$

Compared to the Gaussian, the Laplace distribution has heavier tails—its density decays exponentially as $e^{-|\varepsilon|}$ rather than as $e^{-\varepsilon^2}$. This means large deviations are more probable under the Laplace model, making it a natural choice when outliers are expected.

### Likelihood and Log-Likelihood

$$L(\theta) = \prod_{i=1}^m \frac{1}{2b}\exp\!\left(-\frac{|y^{(i)} - f_\theta(x^{(i)})|}{b}\right)$$

Taking the natural logarithm:

$$\ell(\theta) = -\frac{1}{b}\sum_{i=1}^m \left|y^{(i)} - f_\theta(x^{(i)})\right| - m\log(2b)$$

### From Log-Likelihood to Cost Function

Negating and dropping constant terms:

$$J(\theta) = \frac{1}{m}\sum_{i=1}^m \left|y^{(i)} - f_\theta(x^{(i)})\right|$$

This is the **Mean Absolute Error**. Just as MSE emerges from Gaussian MLE, MAE emerges from Laplace MLE:

$$\underset{\theta}{\operatorname{argmax}}\ L \quad\Leftrightarrow\quad \underset{\theta}{\operatorname{argmax}}\ \ell \quad\Leftrightarrow\quad \underset{\theta}{\operatorname{argmin}}\ J$$

### MLE for Location: The Median

For the simplest case of estimating a location parameter $\mu$ from i.i.d. Laplace samples $\{x^{(i)}\}_{i=1}^m$:

$$\hat{\mu}_{\text{MAE}} = \arg\min_\mu \sum_{i=1}^m |x^{(i)} - \mu| = \text{median}(x^{(1)}, \ldots, x^{(m)})$$

The MAE-optimal estimate is the **sample median**, whereas the MSE-optimal estimate is the **sample mean**. The median is a robust statistic: a single outlier moved to infinity changes the mean arbitrarily but leaves the median unaffected (provided $m \geq 3$).

### MAE Gradient Analysis

$$\frac{\partial \mathcal{L}_{\text{MAE}}}{\partial \hat{y}_i} = \frac{1}{m}\,\text{sign}(\hat{y}_i - y_i) = \frac{1}{m}\begin{cases} +1 & \hat{y}_i > y_i \\ -1 & \hat{y}_i < y_i \\ \text{undefined} & \hat{y}_i = y_i \end{cases}$$

Two consequences follow immediately.

**Constant gradient magnitude.** Unlike MSE where the gradient scales with the error, MAE produces a gradient of fixed magnitude $1/m$ regardless of whether the error is 0.01 or 100. This is the source of MAE's robustness: outliers receive the same gradient signal as typical samples.

**Non-differentiability at zero.** The absolute value function has a kink at zero. In practice, PyTorch uses subgradients (any value in $[-1, +1]$ at zero), which works reliably in optimization. However, the constant gradient magnitude near the optimum means MAE cannot "slow down" as it approaches the solution, potentially causing oscillation around the minimum.

### PyTorch: `nn.L1Loss` and `F.l1_loss`

PyTorch implements MAE under the name L1 loss (since $|x|$ is the L1 norm):

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

y_true = torch.tensor([1.0, 2.0, 3.0, 100.0])
y_pred = torch.tensor([1.1, 2.2, 2.8, 3.0])

# Class-based
criterion = nn.L1Loss()
loss_class = criterion(y_pred, y_true)

# Functional
loss_func = F.l1_loss(y_pred, y_true)

print(f"MAE (class):      {loss_class.item():.4f}")
print(f"MAE (functional): {loss_func.item():.4f}")
# Both: 24.3250
```

Reduction options work identically to MSE:

```python
loss_mean = nn.L1Loss(reduction='mean')(y_pred, y_true)   # 24.325
loss_sum  = nn.L1Loss(reduction='sum')(y_pred, y_true)    # 97.3
loss_none = nn.L1Loss(reduction='none')(y_pred, y_true)   # tensor([0.1, 0.2, 0.2, 97.0])
```

The per-sample output reveals that the outlier (sample 4 with error 97.0) contributes linearly to the total, unlike MSE where it would contribute $97^2 = 9409$.

---

## Comparative Analysis

### Distributional Lens

The choice between MSE and MAE is fundamentally a choice of noise model:

| Property | MSE (Gaussian) | MAE (Laplace) |
|----------|----------------|---------------|
| **Noise density** | $\propto \exp(-\varepsilon^2/2\sigma^2)$ | $\propto \exp(-\lvert\varepsilon\rvert/b)$ |
| **Tail weight** | Light (sub-Gaussian decay) | Heavy (exponential decay) |
| **Optimal estimator** | Sample mean | Sample median |
| **Robustness** | Sensitive to outliers | Robust to outliers |
| **Gradient near optimum** | Shrinks to zero | Constant magnitude |
| **Differentiability** | Everywhere | Not at $\hat{y} = y$ |

### Outlier Robustness Demonstration

```python
# Data with an outlier
actual = torch.tensor([85.0, 90.0, 88.0, 92.0, 15.0])
predicted = torch.tensor([84.0, 89.0, 87.0, 91.0, 87.0])

mse = F.mse_loss(predicted, actual)
mae = F.l1_loss(predicted, actual)

print(f"MSE: {mse.item():.2f}")   # 1040.80  (dominated by outlier)
print(f"MAE: {mae.item():.2f}")   # 14.80    (outlier contributes proportionally)

# Outlier contribution analysis
abs_errors = torch.abs(predicted - actual)
sq_errors = (predicted - actual) ** 2

outlier_mae_share = abs_errors[4] / abs_errors.sum()
outlier_mse_share = sq_errors[4] / sq_errors.sum()

print(f"Outlier share of MAE: {outlier_mae_share.item()*100:.1f}%")  # ~97.3%
print(f"Outlier share of MSE: {outlier_mse_share.item()*100:.1f}%")  # ~99.9%
```

### Gradient Magnitude Comparison

```python
# For a large error (outlier), compare gradient magnitudes
outlier_error = torch.tensor(72.0, requires_grad=True)

# MSE gradient
mse_loss = outlier_error ** 2 / 2
mse_loss.backward()
print(f"MSE gradient magnitude: {abs(outlier_error.grad.item()):.1f}")  # 72.0

# MAE gradient
outlier_error = torch.tensor(72.0, requires_grad=True)
mae_loss = torch.abs(outlier_error)
mae_loss.backward()
print(f"MAE gradient magnitude: {abs(outlier_error.grad.item()):.1f}")  # 1.0
```

MSE produces a gradient 72× larger for this outlier, meaning the model updates much more aggressively to fit outliers under MSE.

## Why MSE Implies Gaussian Noise

The derivation can be read in reverse: choosing MSE as a loss function implicitly assumes that the residuals $y^{(i)} - f_\theta(x^{(i)})$ follow a Gaussian distribution. If the true noise distribution has heavier tails than Gaussian (e.g., due to outliers), MSE assigns excessive weight to extreme errors, and alternative losses like MAE or Huber loss may be more appropriate.

The distributional correspondence generalizes beyond linear models. For any parametric model $f_\theta$, minimizing MSE is equivalent to maximum likelihood under $y \mid x \sim \mathcal{N}(f_\theta(x), \sigma^2)$, and minimizing MAE is equivalent to maximum likelihood under $y \mid x \sim \text{Laplace}(f_\theta(x), b)$.

## When to Use Each

**Use MSE when:**

- Data is clean with approximately Gaussian noise
- Large errors are genuinely more costly (squared loss reflects the true cost structure)
- Smooth, diminishing gradients near the optimum are desired for precise convergence

**Use MAE when:**

- Data contains outliers or heavy-tailed noise (financial returns, sensor data with occasional spikes)
- All prediction errors should contribute proportionally regardless of magnitude
- The median is a more meaningful summary statistic than the mean
- Robustness is prioritized over convergence speed near the optimum

**Consider Huber loss when:**

- You want MAE's outlier robustness but MSE's smooth gradients near the optimum
- The data has occasional outliers but the typical noise is well-behaved

## Key Takeaways

MSE and MAE are not arbitrary formula choices but natural consequences of distributional assumptions combined with the maximum likelihood principle. MSE emerges from Gaussian noise and produces the sample mean as the optimal estimator; MAE emerges from Laplace noise and produces the sample median. The MLE estimates for linear regression have closed-form solutions, expressible through sample statistics (means, standard deviations, correlation). The standardized regression equation $z_y = \hat{\rho}\, z_x$ reveals the geometric essence of linear regression: it projects the data onto the direction of maximum correlation. In PyTorch, MSE is implemented as `nn.MSELoss`/`F.mse_loss` and MAE as `nn.L1Loss`/`F.l1_loss`, both integrating seamlessly into standard training loops.
