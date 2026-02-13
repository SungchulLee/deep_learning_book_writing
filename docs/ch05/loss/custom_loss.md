# Custom Loss Functions

Standard loss functions encode common assumptions, but real-world problems often require specialized objectives. This section covers the design and implementation of custom loss functions in PyTorch, including practical examples for segmentation, multi-task learning, and dynamic weight balancing.

## When to Create Custom Losses

Custom loss functions become necessary when:

1. **Domain-specific objectives:** Medical imaging requires overlap metrics (Dice), object detection needs bounding box losses (IoU)
2. **Data characteristics:** Hard example mining, class-weighted losses, noise-aware objectives
3. **Multi-task learning:** Combining multiple objectives with learned or fixed weights
4. **Custom constraints:** Physics-informed losses, smoothness penalties, adversarial objectives
5. **Research:** Testing novel loss formulations

## Implementation Approaches

### Method 1: Function-Based (Simple Cases)

For quick experiments, define loss as a simple function:

```python
import torch
import torch.nn.functional as F

def custom_mse_loss(predictions, targets):
    """Custom MSE for demonstration."""
    squared_diff = (predictions - targets) ** 2
    return torch.mean(squared_diff)

# Usage
pred = torch.tensor([1.0, 2.0, 3.0])
target = torch.tensor([1.5, 2.5, 3.5])
loss = custom_mse_loss(pred, target)
print(f"Custom MSE: {loss.item():.4f}")  # 0.2500
```

!!! note "Critical Requirements"
    All operations must use PyTorch tensors (not NumPy), the output must be a scalar tensor, and all operations must be differentiable for autograd to compute gradients.

### Method 2: Class-Based (Recommended)

The `nn.Module` approach offers configurability and integrates with PyTorch conventions:

```python
import torch.nn as nn

class WeightedMSELoss(nn.Module):
    """MSE Loss with per-sample weights."""

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, predictions, targets, weights=None):
        squared_error = (predictions - targets) ** 2

        if weights is not None:
            squared_error = squared_error * weights

        if self.reduction == 'mean':
            return torch.mean(squared_error)
        elif self.reduction == 'sum':
            return torch.sum(squared_error)
        return squared_error  # 'none'

# Usage
criterion = WeightedMSELoss()
weights = torch.tensor([0.5, 1.0, 2.0])
loss = criterion(pred, target, weights)
```

## Dice Loss: Segmentation Overlap

Dice Loss optimizes the Dice coefficient (also called F1 score for sets), measuring overlap between predicted and ground truth segmentation masks. It is particularly effective for imbalanced segmentation tasks where the foreground region is small relative to the background.

### Mathematical Formulation

The Dice coefficient between sets $A$ and $B$:

$$\text{Dice} = \frac{2|A \cap B|}{|A| + |B|}$$

For soft (differentiable) predictions:

$$\mathcal{L}_{\text{Dice}} = 1 - \frac{2\sum_i p_i g_i + \epsilon}{\sum_i p_i + \sum_i g_i + \epsilon}$$

where $p_i$ are predicted probabilities, $g_i$ are ground truth values, and $\epsilon$ provides numerical stability.

### Implementation

```python
class DiceLoss(nn.Module):
    """Dice Loss for binary segmentation.

    Args:
        smooth: Smoothing factor for numerical stability. Default: 1.0.
    """

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        """
        Args:
            predictions: Model outputs after sigmoid, values in [0, 1].
            targets: Binary ground truth mask.
        """
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        intersection = (predictions * targets).sum()
        union = predictions.sum() + targets.sum()

        dice = (2. * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice
```

### Demonstration

```python
# Simple 5x5 segmentation
true_mask = torch.tensor([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
], dtype=torch.float32)

# Good prediction (high overlap)
good_pred = torch.tensor([
    [0, 0, 0, 0, 0],
    [0, 0.9, 0.8, 0.9, 0],
    [0, 0.85, 0.95, 0.85, 0],
    [0, 0.9, 0.8, 0.9, 0],
    [0, 0, 0, 0, 0]
], dtype=torch.float32)

# Bad prediction (shifted, poor overlap)
bad_pred = torch.tensor([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0.8, 0.9, 0.85, 0, 0],
    [0.9, 0.8, 0.9, 0, 0],
    [0, 0, 0, 0, 0]
], dtype=torch.float32)

dice_criterion = DiceLoss()
print(f"Good prediction Dice Loss: {dice_criterion(good_pred, true_mask):.4f}")  # ~0.12
print(f"Bad prediction Dice Loss:  {dice_criterion(bad_pred, true_mask):.4f}")   # ~0.70
```

### Dice + BCE Combination

In practice, combining Dice Loss with BCE often outperforms either alone:

```python
class DiceBCELoss(nn.Module):
    """Combined Dice and BCE loss for segmentation."""

    def __init__(self, dice_weight=0.5, bce_weight=0.5, smooth=1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        dice_loss = self.dice(probs, targets)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets)
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss
```

## Combining Multiple Loss Terms

Many applications require optimizing multiple objectives simultaneously. A common pattern is the weighted sum:

$$\mathcal{L}_{\text{total}} = \sum_i \alpha_i \mathcal{L}_i$$

### Fixed-Weight Combination

```python
class CombinedLoss(nn.Module):
    """Combines multiple loss functions with configurable weights."""

    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights or {'mse': 1.0, 'l1': 1.0}
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, predictions, targets):
        mse = self.mse_loss(predictions, targets)
        l1 = self.l1_loss(predictions, targets)

        total = self.weights['mse'] * mse + self.weights['l1'] * l1

        # Return components for logging
        return total, {'mse': mse.item(), 'l1': l1.item()}

# Usage
combined = CombinedLoss(weights={'mse': 0.7, 'l1': 0.3})
total_loss, components = combined(pred, target)
print(f"Total: {total_loss.item():.4f}")
print(f"Components: {components}")
```

### Dynamic Weight Balancing (Uncertainty Weighting)

For multi-task learning, task weights can be learned via homoscedastic uncertainty (Kendall et al., 2018):

$$\mathcal{L}_{\text{total}} = \sum_i \frac{1}{2\sigma_i^2}\mathcal{L}_i + \log\sigma_i$$

The log-variance terms $\log\sigma_i$ are learnable parameters. As $\sigma_i$ increases, the weight on task $i$ decreases, and the $\log\sigma_i$ regularizer prevents all $\sigma_i$ from going to infinity.

```python
class UncertaintyWeightedLoss(nn.Module):
    """Learns task weights via homoscedastic uncertainty.

    Reference: "Multi-Task Learning Using Uncertainty to Weigh Losses"
    (Kendall, Gal, Cipolla, 2018)
    """

    def __init__(self, num_tasks):
        super().__init__()
        # Log variance (learnable); initialized to 0 → sigma^2 = 1
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses):
        """
        Args:
            losses: List of individual task losses.
        """
        total = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])  # 1/sigma^2
            total += precision * loss + self.log_vars[i]
        return total
```

!!! note "Training the Weights"
    The `log_vars` parameters are included in the optimizer alongside model parameters. They are updated by backpropagation, automatically balancing the task weights during training.

## Best Practices

### Design Guidelines

1. **Use `nn.Module`**: Enables configuration, parameter tracking, and device management
2. **Return a scalar**: Loss must reduce to a single value for backpropagation
3. **Document thoroughly**: Include mathematical formulation, parameter meanings, and usage examples
4. **Add numerical stability**: Include smoothing terms, clamps for log arguments

### Testing Custom Losses

```python
def test_custom_loss(loss_fn, input_shape=(10,)):
    """Validate a custom loss function."""

    # Create test inputs with gradients
    pred = torch.randn(input_shape, requires_grad=True)
    target = torch.randn(input_shape)

    # Test 1: Output is scalar
    loss = loss_fn(pred, target)
    assert loss.dim() == 0, "Loss must be a scalar"
    print("✓ Shape check passed")

    # Test 2: Gradients flow
    loss.backward()
    assert pred.grad is not None, "Gradients must flow"
    print("✓ Gradient check passed")

    # Test 3: Numerical stability
    edge_pred = torch.tensor([0.0, 1e-10, 1e10])
    edge_target = torch.tensor([0.0, 0.0, 1e10])
    edge_loss = loss_fn(edge_pred, edge_target)
    assert not torch.isnan(edge_loss) and not torch.isinf(edge_loss)
    print("✓ Numerical stability check passed")

# Example
test_custom_loss(WeightedMSELoss())
```

### Common Pitfalls

| Mistake | Problem | Solution |
|---------|---------|----------|
| Using `.item()` inside loss | Breaks gradient tracking | Keep as tensor until final logging |
| Non-differentiable operations | Zero gradients | Use differentiable alternatives |
| Missing reduction | Wrong gradient scaling | Always reduce to scalar |
| No stability terms | NaN from `log(0)` | Add epsilon: `log(x + 1e-8)` |
| Detaching intermediate tensors | Breaks computation graph | Only detach for targets/labels |

## Key Takeaways

Custom loss functions extend PyTorch's built-in losses to domain-specific objectives. The class-based approach (`nn.Module`) is preferred for its flexibility, parameter tracking, and integration with PyTorch conventions. Dice Loss optimizes overlap metrics for segmentation and is often combined with BCE for best results. Multi-task losses can use fixed weights, but uncertainty-weighted losses (with learnable $\log\sigma^2$ parameters) automatically balance competing objectives during training. Always validate custom losses for correct shape, gradient flow, and numerical stability before deploying in training.
