# Problem Specification

A precise problem specification prevents wasted effort from solving the wrong problem. In deep learning, the specification defines the input domain, output format, loss function, and evaluation metric before any model is designed.

## Definition

A problem specification is the formal triple:

$$
\text{Problem} = (\mathcal{X},\; \mathcal{Y},\; \text{Objective})
$$

where $\mathcal{X}$ is the input space, $\mathcal{Y}$ is the output space, and the objective defines what constitutes a correct or optimal solution.

## Explanation

In deep learning, a complete specification requires:

- **Input specification**: Tensor shape, dtype, value range, and preprocessing. Example: "RGB images of shape $(3, 224, 224)$, pixel values in $[0, 1]$, normalized by ImageNet statistics."
- **Output specification**: What the model should produce. Classification logits? Bounding boxes? Generated text? The output format determines the model architecture's final layer.
- **Loss function**: The differentiable objective that training minimizes. Cross-entropy for classification, MSE for regression, etc.
- **Evaluation metric**: The non-differentiable metric that matters in practice (accuracy, F1, BLEU). Often differs from the loss.
- **Constraints**: Latency budget, model size limit, minimum accuracy threshold.

A common failure mode is misalignment between the loss and the evaluation metric. For example, training with MSE loss but evaluating with accuracy on a classification task will produce suboptimal results.

## Examples

```python
import torch
import torch.nn as nn

# Complete problem specification for binary classification
spec = {
    "input": "tensor of shape (batch, 10), float32, standardized",
    "output": "tensor of shape (batch, 1), logits",
    "loss": "binary cross-entropy with logits",
    "metric": "accuracy",
    "constraint": "model < 10K parameters",
}
for k, v in spec.items():
    print(f"  {k}: {v}")

# Implement according to spec
model = nn.Sequential(nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 1))
n_params = sum(p.numel() for p in model.parameters())
print(f"\nParameters: {n_params} (limit: 10000)")

# Verify input/output contract
x = torch.randn(4, 10)
logits = model(x)
assert logits.shape == (4, 1), f"Output shape {logits.shape} != (4, 1)"

# Loss matches specification
y = torch.tensor([1.0, 0.0, 1.0, 0.0]).unsqueeze(1)
loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
print(f"Loss: {loss.item():.4f}")

# Metric matches specification
preds = (torch.sigmoid(logits) > 0.5).float()
accuracy = (preds == y).float().mean()
print(f"Accuracy: {accuracy.item():.2f}")
```
