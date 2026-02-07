# Regularization Overview

## Overview

Regularization encompasses a broad family of strategies that constrain the learning process to improve generalization. Every supervised learning model must navigate between **underfitting** (too simple to capture genuine patterns) and **overfitting** (memorizing training noise rather than learning generalizable structure). Regularization techniques systematically address overfitting by penalizing model complexity, injecting noise, expanding the effective training set, or limiting the optimization trajectory.

This section provides the conceptual and mathematical foundations — the bias-variance tradeoff, the geometry of constrained optimization, and a unified taxonomy of techniques — before the detailed treatments in subsequent sections.

## Overfitting and Underfitting

### The Model Complexity Spectrum

**Underfitting (High Bias).** A model underfits when it lacks the capacity to represent the true input–output relationship. Indicators include high training loss that fails to decrease with further training, training and validation losses that are both large and close in value, and predictions that miss systematic patterns in the data. For example, fitting a linear model to a quadratic relationship will always leave structured residuals, regardless of how much data is available.

**Overfitting (High Variance).** A model overfits when it fits the training data too closely, capturing noise and idiosyncrasies that do not generalize. Indicators include training loss that continues to decrease while validation loss increases, a growing gap between training and validation performance, and highly sensitive predictions that change substantially with small perturbations to the training set.

**The Optimal Region.** Between these extremes lies the sweet spot where the model has enough capacity to capture genuine patterns but is appropriately constrained to avoid memorizing noise. The goal of regularization is to keep the model in this region.

### Diagnosing Fit: Learning Curves

The most practical tool for diagnosing overfitting and underfitting is the **learning curve**, which plots training and validation loss as a function of training epochs.

| Pattern | Train Loss | Val Loss | Gap | Diagnosis |
|---------|-----------|----------|-----|-----------|
| Both high, close together | High | High | Small | Underfitting |
| Both low, close together | Low | Low | Small | Good fit |
| Train low, val rising | Low | High / Rising | Large | Overfitting |
| Both decreasing, gap stable | Decreasing | Decreasing | Moderate | Still learning (continue training) |

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split


def generate_polynomial_data(n_samples=500, noise_std=0.3, seed=42):
    """Generate synthetic data with polynomial ground truth."""
    torch.manual_seed(seed)
    x = torch.linspace(-3, 3, n_samples).unsqueeze(1)
    y_true = 0.5 * x**3 - 2 * x**2 + x + 1
    y = y_true + noise_std * torch.randn_like(y_true)
    return x, y


def compute_learning_curves(model, train_loader, val_loader, epochs=200, lr=0.01):
    """
    Train model and record training/validation loss per epoch.

    Returns:
        dict with keys 'train_loss' and 'val_loss', each a list of floats.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        train_loss_sum = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * X_batch.size(0)

        # --- Validation ---
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                val_loss_sum += criterion(model(X_batch), y_batch).item() * X_batch.size(0)

        history['train_loss'].append(train_loss_sum / len(train_loader.dataset))
        history['val_loss'].append(val_loss_sum / len(val_loader.dataset))

    return history
```

### Comparing Model Complexities

```python
class PolynomialModel(nn.Module):
    """Linear regression on polynomial features of specified degree."""

    def __init__(self, degree):
        super().__init__()
        self.degree = degree
        self.linear = nn.Linear(degree, 1)

    def forward(self, x):
        # Build polynomial feature matrix [x, x^2, ..., x^degree]
        features = torch.cat([x ** i for i in range(1, self.degree + 1)], dim=1)
        return self.linear(features)


class DeepNetwork(nn.Module):
    """Overparameterized deep network to demonstrate overfitting."""

    def __init__(self, hidden_dim=256, n_layers=4):
        super().__init__()
        layers = [nn.Linear(1, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# --- Prepare data ---
x, y = generate_polynomial_data(n_samples=200, noise_std=0.5)
dataset = TensorDataset(x, y)
train_set, val_set = random_split(dataset, [140, 60])
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64)

# Case 1: Underfitting — linear model on cubic data
underfit_model = PolynomialModel(degree=1)
underfit_history = compute_learning_curves(underfit_model, train_loader, val_loader)

# Case 2: Good fit — cubic model on cubic data
goodfit_model = PolynomialModel(degree=3)
goodfit_history = compute_learning_curves(goodfit_model, train_loader, val_loader)

# Case 3: Overfitting — deep network on small dataset
overfit_model = DeepNetwork(hidden_dim=256, n_layers=4)
overfit_history = compute_learning_curves(overfit_model, train_loader, val_loader, epochs=500)
```

## The Bias-Variance Tradeoff

### Mathematical Derivation

Consider a regression problem where the true relationship is $y = f(x) + \epsilon$, with noise $\epsilon$ satisfying $\mathbb{E}[\epsilon] = 0$ and $\text{Var}[\epsilon] = \sigma^2$. A learning algorithm produces an estimator $\hat{f}_{\mathcal{D}}(x)$ trained on a dataset $\mathcal{D}$ drawn from the data-generating distribution.

For a fixed test point $x_0$, the expected squared error over both the noise and the randomness in $\mathcal{D}$ is:

$$
\text{EPE}(x_0) = \mathbb{E}_{\mathcal{D}, \epsilon}\left[\left(y_0 - \hat{f}_{\mathcal{D}}(x_0)\right)^2\right]
$$

Expand by writing $y_0 = f(x_0) + \epsilon$ and introducing $\bar{f}(x_0) = \mathbb{E}_{\mathcal{D}}[\hat{f}_{\mathcal{D}}(x_0)]$:

$$
\text{EPE}(x_0) = \mathbb{E}\left[\left(\epsilon + f(x_0) - \hat{f}_{\mathcal{D}}(x_0)\right)^2\right]
$$

Since $\epsilon$ is independent of $\hat{f}_{\mathcal{D}}(x_0)$:

$$
= \sigma^2 + \mathbb{E}_{\mathcal{D}}\left[\left(f(x_0) - \hat{f}_{\mathcal{D}}(x_0)\right)^2\right]
$$

Adding and subtracting $\bar{f}(x_0)$ inside the squared term and expanding — noting that the cross term vanishes because $\mathbb{E}_{\mathcal{D}}[\hat{f}_{\mathcal{D}}(x_0) - \bar{f}(x_0)] = 0$ — yields the three-term decomposition:

$$
\boxed{\text{EPE}(x_0) = \underbrace{\sigma^2}_{\text{Irreducible Error}} + \underbrace{\left(f(x_0) - \bar{f}(x_0)\right)^2}_{\text{Bias}^2} + \underbrace{\mathbb{E}_{\mathcal{D}}\left[\left(\hat{f}_{\mathcal{D}}(x_0) - \bar{f}(x_0)\right)^2\right]}_{\text{Variance}}}
$$

| Term | Meaning | Depends On |
|------|---------|------------|
| Irreducible error $\sigma^2$ | Noise inherent in the data | Data-generating process |
| Bias$^2$ | Systematic error of the average model | Model family (capacity) |
| Variance | Sensitivity of $\hat{f}$ to the choice of $\mathcal{D}$ | Model complexity, dataset size |

### The U-Shaped Curve and Double Descent

As model complexity increases from very simple to very flexible, **bias** monotonically decreases (a more flexible model can fit more patterns), **variance** monotonically increases (a more flexible model is more sensitive to training data), and **total error** follows a U-shape with a minimum at the optimal complexity.

For a fixed model complexity, increasing the training set size $n$ leaves bias approximately unchanged but decreases variance roughly as $O(1/n)$, explaining why collecting more data is particularly effective for high-variance models.

Modern deep learning research has revealed the **double descent** phenomenon: beyond the interpolation threshold (where the model first achieves zero training error), error can *decrease again* as model complexity continues to increase. This does not invalidate the bias-variance framework — the decomposition still holds — but shows that variance can decrease in the overparameterized regime due to implicit regularization from gradient descent, initialization, and early stopping.

### Empirical Bias-Variance Estimation

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


def true_function(x):
    """Ground-truth function: cubic polynomial."""
    return 0.5 * x**3 - x**2 + 0.5 * x + 1


def generate_dataset(n_samples=50, noise_std=0.5, seed=None):
    """Generate a noisy dataset from the true function."""
    if seed is not None:
        torch.manual_seed(seed)
    x = torch.linspace(-2, 2, n_samples).unsqueeze(1)
    y = true_function(x) + noise_std * torch.randn_like(x)
    return x, y


def train_model(model, x, y, epochs=1000, lr=0.01):
    """Train a model on a single dataset."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for _ in range(epochs):
        optimizer.zero_grad()
        criterion(model(x), y).backward()
        optimizer.step()
    return model


def estimate_bias_variance(degree, n_datasets=200, n_samples=50, noise_std=0.5):
    """
    Estimate bias^2 and variance for a model of given polynomial degree
    by training on many bootstrap datasets.

    Returns:
        bias_sq: float — squared bias averaged over test points
        variance: float — variance averaged over test points
        mse: float — mean squared error averaged over test points and datasets
    """
    x_test = torch.linspace(-2, 2, 100).unsqueeze(1)
    y_true = true_function(x_test)

    # Collect predictions from each dataset
    all_predictions = []
    for seed in range(n_datasets):
        x_train, y_train = generate_dataset(n_samples, noise_std, seed=seed)
        model = PolynomialModel(degree)
        train_model(model, x_train, y_train)

        with torch.no_grad():
            pred = model(x_test)
        all_predictions.append(pred)

    predictions = torch.stack(all_predictions)  # (n_datasets, n_test, 1)
    mean_pred = predictions.mean(dim=0)          # E[f_hat(x)]

    bias_sq = ((y_true - mean_pred) ** 2).mean().item()
    variance = predictions.var(dim=0).mean().item()
    mse = ((predictions - y_true.unsqueeze(0)) ** 2).mean().item()

    return bias_sq, variance, mse


# Sweep over model complexities
degrees = [1, 2, 3, 5, 8, 12, 18]
results = []

for d in degrees:
    bias_sq, var, mse = estimate_bias_variance(d, n_datasets=100)
    results.append({'degree': d, 'bias_sq': bias_sq, 'variance': var, 'mse': mse})
    print(f"Degree {d:2d}:  Bias²={bias_sq:.4f}  Var={var:.4f}  MSE={mse:.4f}")
```

### Visualizing the Tradeoff

```python
import matplotlib.pyplot as plt

degrees_plot = [r['degree'] for r in results]
biases = [r['bias_sq'] for r in results]
variances = [r['variance'] for r in results]
mses = [r['mse'] for r in results]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(degrees_plot, biases, 'b-o', label='Bias²')
ax.plot(degrees_plot, variances, 'r-s', label='Variance')
ax.plot(degrees_plot, mses, 'k--^', label='Total MSE')
ax.set_xlabel('Polynomial Degree (Model Complexity)')
ax.set_ylabel('Error')
ax.set_title('Bias-Variance Tradeoff')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('bias_variance_tradeoff.png', dpi=150)
plt.show()
```

## Taxonomy of Regularization Techniques

Regularization methods can be organized by *where* in the learning pipeline they intervene.

### 1. Explicit Parameter Penalties

These methods add a penalty term directly to the loss function to discourage large or unnecessary weights. The regularized objective takes the general form:

$$
\mathcal{L}_{\text{reg}}(\theta) = \mathcal{L}_{\text{data}}(\theta) + \Omega(\theta)
$$

where $\Omega(\theta)$ is the penalty and the relative weight controls the bias-variance tradeoff.

| Method | Penalty Term | Key Property |
|--------|-------------|--------------|
| L2 (Ridge) | $\lambda \| w \|_2^2$ | Shrinks all weights toward zero |
| L1 (Lasso) | $\lambda \| w \|_1$ | Drives some weights to exactly zero (sparsity) |
| Elastic Net | $\lambda_1 \| w \|_1 + \lambda_2 \| w \|_2^2$ | Combines sparsity with grouping stability |

### Constraint Geometry

Each parameter penalty restricts the feasible region of parameter space. Consider the L2 constraint for a two-dimensional weight vector $(w_1, w_2)$: without regularization the optimizer finds the global minimum of $\mathcal{L}_{\text{data}}$ anywhere in $\mathbb{R}^2$; with L2 the effective feasible set is $\{w : \|w\|_2^2 \leq t\}$ for some threshold $t$ determined by $\lambda$, and the regularized solution lies where the loss contour is tangent to the constraint boundary. L1 constrains to the diamond $\{w : \|w\|_1 \leq t\}$, whose corners lie on the coordinate axes — explaining why L1 produces sparse solutions.

### Effective Degrees of Freedom

Regularization reduces the effective number of parameters the model uses. For Ridge regression:

$$
\text{df}(\lambda) = \text{tr}\left[X(X^TX + \lambda I)^{-1}X^T\right] = \sum_{j=1}^{p} \frac{\mu_j}{\mu_j + \lambda}
$$

where $\mu_j$ are eigenvalues of $X^TX$. As $\lambda \to 0$, $\text{df} \to p$ (full model); as $\lambda \to \infty$, $\text{df} \to 0$.

### How Regularization Shifts the Bias-Variance Tradeoff

| Technique | Effect on Bias | Effect on Variance | Net Effect |
|-----------|---------------|-------------------|------------|
| L2 (Ridge) | Slight increase | Significant decrease | Lower total error |
| L1 (Lasso) | Slight increase | Significant decrease | Sparser models |
| Dropout | Slight increase | Significant decrease | Ensemble effect |
| Early Stopping | Slight increase | Significant decrease | Limits effective capacity |
| Data Augmentation | Unchanged or decreased | Decreased | Larger effective dataset |

### 2. Architectural Constraints

These methods modify the network architecture or the forward pass to limit effective capacity.

**Dropout** randomly disables neurons during training:

$$
\tilde{h} = \frac{m \odot h}{1-p}, \quad m_i \sim \text{Bernoulli}(1-p)
$$

This prevents co-adaptation and creates an implicit ensemble of $2^d$ sub-networks for a layer with $d$ units.

**DropConnect** generalizes dropout by randomly zeroing individual *weights* rather than entire activations, providing finer-grained stochastic regularization.

**Batch Normalization** normalizes activations within each mini-batch:

$$
\hat{x}_i = \frac{x_i - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}}
$$

While designed primarily for training stability, batch normalization has a regularizing effect through the noise introduced by mini-batch statistics.

### 3. Data-Side Regularization

Instead of constraining the model, these methods expand or modify the training data.

**Data Augmentation** creates transformed copies of training examples that preserve semantic content (random crops, flips, rotations, color jitter for images; synonym replacement, back-translation for text).

**Label Smoothing** replaces hard one-hot targets with soft targets that distribute a small probability mass across all classes, preventing the model from becoming overconfident.

**Mixup** and **CutMix** create virtual training examples by blending pairs of inputs and their labels, encouraging linear behavior between training examples.

**Cutout** (Random Erasing) randomly masks rectangular regions of input images, forcing the model to attend to the full spatial extent of objects rather than relying on local discriminative patches.

**Noise Injection** adds random perturbations to inputs, activations, or gradients during training. For linear regression with input noise $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$, this is equivalent to L2 regularization:

$$
\mathbb{E}_\epsilon[\|y - (x + \epsilon)^T w\|^2] = \|y - x^T w\|^2 + \sigma^2 \|w\|^2
$$

### 4. Optimization-Based Regularization

These methods control the training process itself rather than modifying the model or data.

**Early Stopping** halts training when validation loss stops improving, preventing the model from entering the overfitting regime. For linear models, early stopping with gradient descent is mathematically equivalent to L2 regularization with effective strength $\lambda_{\text{eff}} \approx 1/(\eta t)$, where $\eta$ is the learning rate and $t$ is the iteration count.

**Learning Rate Scheduling** and **Gradient Clipping** also have regularizing effects, though they are not traditionally classified as regularization techniques.

## Combining Regularization Techniques

In practice, multiple regularization methods are used simultaneously.

### Typical Deep Learning Recipe

```python
import torch.nn as nn
import torch.optim as optim

class RegularizedCNN(nn.Module):
    """CNN combining multiple regularization techniques."""

    def __init__(self, num_classes=10, dropout_rate=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),          # Batch normalization
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),  # Spatial dropout

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),    # Standard dropout
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


model = RegularizedCNN()
optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-2  # L2 regularization (decoupled)
)
```

### Guidelines for Combining Techniques

| Combination | Interaction | Recommendation |
|-------------|------------|----------------|
| Dropout + BatchNorm | BatchNorm reduces the need for dropout | Use lower dropout rate (0.1–0.2) |
| L2 + Dropout | Both reduce effective capacity | Reduce one if val loss increases |
| Augmentation + Dropout | Complementary (data-side + model-side) | Generally safe to combine |
| Early Stopping + any method | Always beneficial as a safety net | Always use alongside other methods |
| L1 + L2 (Elastic Net) | L1 for sparsity, L2 for stability | Use Elastic Net when features are correlated |
| Label Smoothing + Mixup | Both soften targets | Reduce smoothing factor when using Mixup |

## Selecting Regularization Strategies

### Decision Framework

1. **Start with data augmentation and early stopping** — low risk, broadly applicable
2. **Add weight decay** (L2) — default in modern optimizers like AdamW
3. **Add dropout** if the model has dense layers prone to overfitting
4. **Use batch normalization** for training stability and mild regularization
5. **Apply label smoothing** for classification tasks to prevent overconfidence
6. **Apply L1 or Elastic Net** if feature selection or sparsity is desired
7. **Use cross-validation** to tune all regularization hyperparameters

### Architecture-Specific Defaults

| Architecture | Recommended Techniques |
|--------------|----------------------|
| MLPs | Dropout (0.5), weight decay, early stopping |
| CNNs | BatchNorm, spatial dropout (0.2), augmentation, weight decay |
| RNNs/LSTMs | Dropout between layers (0.3), weight decay |
| Transformers | Dropout (0.1), label smoothing, weight decay |
| Classical ML (linear/tree) | L1/L2/Elastic Net, cross-validation |

## Common Causes and Remedies

### Causes of Underfitting

1. **Insufficient model capacity**: too few layers, too few parameters
2. **Excessive regularization**: overly large weight decay or dropout rate
3. **Inadequate training**: too few epochs or learning rate too low
4. **Poor feature representation**: missing informative features

### Causes of Overfitting

1. **Excess model capacity** relative to the amount and complexity of data
2. **Insufficient training data** to constrain a high-capacity model
3. **Training too long** without validation-based stopping
4. **No regularization** applied to a flexible model

### Remedies

**For underfitting:** increase model capacity, reduce regularization strength, train longer with appropriate learning rate schedule, engineer better features or use richer input representations.

**For overfitting:** apply regularization (dropout, weight decay, data augmentation), use early stopping with patience, collect more training data, reduce model capacity.

## References

1. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer. Chapters 3, 7.
2. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. Chapters 5, 7.
3. Geman, S., Bienenstock, E., & Doursat, R. (1992). Neural Networks and the Bias/Variance Dilemma. *Neural Computation*, 4(1), 1-58.
4. Belkin, M., et al. (2019). Reconciling Modern Machine Learning Practice and the Classical Bias-Variance Trade-off. *PNAS*, 116(32), 15849-15854.
5. Kukačka, J., Golkov, V., & Cremers, D. (2017). Regularization for Deep Learning: A Taxonomy. arXiv:1710.10686.
6. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer. Section 1.1.
