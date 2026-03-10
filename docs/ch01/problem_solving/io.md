# Input and Output

Every algorithm and neural network defines a mapping from inputs to outputs. Understanding the input-output contract precisely is the first step in designing any model or data pipeline.

## Definition

An algorithm (or model) is a function:

$$
f: \mathcal{X} \rightarrow \mathcal{Y}
$$

where $\mathcal{X}$ is the input space and $\mathcal{Y}$ is the output space. In deep learning, $\mathcal{X}$ is typically a tensor space (e.g., $\mathbb{R}^{B \times C \times H \times W}$ for images) and $\mathcal{Y}$ is a probability distribution, a scalar, or another tensor.

## Explanation

Specifying the input-output contract precisely prevents bugs and clarifies model design:

- **Input shape and dtype**: A model expecting $(B, 3, 224, 224)$ float32 tensors will fail silently or produce garbage if given $(B, 224, 224, 3)$ (channels-last format). Always document and validate tensor shapes.
- **Output semantics**: Classification models output logits (unnormalized), probabilities (after softmax), or log-probabilities (after log-softmax). Confusing these corrupts the loss computation.
- **Preprocessing contract**: The mapping from raw data (text, images, tabular rows) to model-ready tensors is part of the input specification. A model trained on normalized data will fail on unnormalized inputs.

Data loading in PyTorch follows a clear input-output chain: raw files go through a `Dataset` (which defines `__getitem__` returning tensors), then through a `DataLoader` (which batches and shuffles), and finally into the model.

## Examples

```python
import torch
import torch.nn as nn

# Define a model with explicit input/output contract
class Classifier(nn.Module):
    """Input: (batch, 10) float32. Output: (batch, 3) logits."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 2 and x.shape[1] == 10, f"Expected (B, 10), got {x.shape}"
        return self.net(x)

model = Classifier()
x = torch.randn(4, 10)
logits = model(x)
probs = torch.softmax(logits, dim=1)
print(f"Input shape:  {x.shape}")
print(f"Logits shape: {logits.shape}")
print(f"Probs sum:    {probs.sum(dim=1)}")  # should be [1, 1, 1, 1]

# Demonstrate the importance of matching preprocessing
x_raw = torch.randn(4, 10) * 100  # unscaled
x_normalized = (x_raw - x_raw.mean(0)) / (x_raw.std(0) + 1e-8)
print(f"Raw logits range:   [{model(x_raw).min():.1f}, {model(x_raw).max():.1f}]")
print(f"Norm logits range:  [{model(x_normalized).min():.1f}, {model(x_normalized).max():.1f}]")
```
