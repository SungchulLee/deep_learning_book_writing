# DropConnect

## Overview

DropConnect is a generalization of dropout that randomly sets individual *weights* (connections) to zero during training rather than entire neuron activations. Proposed by Wan et al. (2013), DropConnect operates at the weight level, providing a finer-grained form of stochastic regularization. While dropout masks activations — effectively removing entire neurons from the forward pass — DropConnect masks individual elements of the weight matrix, allowing each neuron to participate partially in every forward pass.

## Mathematical Formulation

### Standard Fully Connected Layer

A standard fully connected layer computes:

$$
y = \sigma(Wx + b)
$$

where $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ is the weight matrix, $x \in \mathbb{R}^{d_{\text{in}}}$ is the input, $b \in \mathbb{R}^{d_{\text{out}}}$ is the bias, and $\sigma$ is a nonlinearity.

### DropConnect Forward Pass

During training, DropConnect samples a binary mask $M \in \{0, 1\}^{d_{\text{out}} \times d_{\text{in}}}$ with each entry drawn independently:

$$
M_{ij} \sim \text{Bernoulli}(1 - p)
$$

where $p$ is the drop probability. The forward pass becomes:

$$
y = \sigma\left((M \odot W) x + b\right)
$$

where $\odot$ denotes element-wise (Hadamard) multiplication.

### Inverted DropConnect

Analogous to inverted dropout, we scale by $\frac{1}{1-p}$ during training so that no rescaling is needed at inference:

$$
y_{\text{train}} = \sigma\left(\frac{M \odot W}{1-p} \, x + b\right)
$$

$$
y_{\text{inference}} = \sigma(Wx + b)
$$

### Expected Output

With inverted scaling, the expected output of the masked layer matches the unmasked layer:

$$
\mathbb{E}_M\left[\frac{M \odot W}{1-p}\right] = W
$$

This ensures consistent behavior between training and inference.

### Distribution of the Pre-Activation

The pre-activation $u = (M \odot W) x$ is a sum of independent random variables. For the $j$-th output unit:

$$
u_j = \sum_{i=1}^{d_{\text{in}}} M_{ji} W_{ji} x_i
$$

Each term $M_{ji} W_{ji} x_i$ has:

$$
\mathbb{E}[M_{ji} W_{ji} x_i] = (1-p) W_{ji} x_i
$$

$$
\text{Var}[M_{ji} W_{ji} x_i] = p(1-p)(W_{ji} x_i)^2
$$

By the Central Limit Theorem, for sufficiently large $d_{\text{in}}$, $u_j$ is approximately Gaussian:

$$
u_j \;\dot{\sim}\; \mathcal{N}\!\left((1-p) \sum_i W_{ji} x_i, \;\; p(1-p) \sum_i (W_{ji} x_i)^2\right)
$$

This Gaussian approximation is used for efficient inference in the original paper (though inverted scaling is more common in practice).

## Comparison with Dropout

### Where the Mask is Applied

| Aspect | Dropout | DropConnect |
|--------|---------|-------------|
| Mask target | Activations $h$ | Weights $W$ |
| Mask shape | $(d,)$ per layer | $(d_{\text{out}}, d_{\text{in}})$ per layer |
| Granularity | Drops entire neurons | Drops individual connections |
| Possible sub-networks | $2^d$ per layer | $2^{d_{\text{out}} \times d_{\text{in}}}$ per layer |
| Sparsity pattern | Structured (rows/columns) | Unstructured (arbitrary entries) |

### Dropout as a Special Case

Dropout can be viewed as a special case of DropConnect where the mask has a structured form. In dropout, when neuron $i$ is dropped, *all* outgoing weights from neuron $i$ are zeroed simultaneously. In DropConnect, each weight is dropped independently. Formally, the dropout mask on weights is:

$$
M_{ji}^{\text{dropout}} = m_i \quad \forall j, \qquad m_i \sim \text{Bernoulli}(1-p)
$$

while the DropConnect mask is:

$$
M_{ji}^{\text{dropconnect}} \sim \text{Bernoulli}(1-p) \quad \text{independently for all } (j, i)
$$

### Implicit Ensemble Size

For a single layer with $d_{\text{in}}$ inputs and $d_{\text{out}}$ outputs:

- **Dropout** implicitly averages over $2^{d_{\text{in}}}$ sub-networks
- **DropConnect** implicitly averages over $2^{d_{\text{in}} \times d_{\text{out}}}$ sub-networks

This exponentially larger ensemble provides richer regularization at the cost of higher variance during training.

## PyTorch Implementation

### Basic DropConnect Layer

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class DropConnect(nn.Module):
    """
    DropConnect: randomly zeros individual weights during training.
    
    Uses inverted scaling so no modification is needed at inference.
    
    Args:
        in_features: Size of each input sample
        out_features: Size of each output sample
        p: Probability of dropping each weight (default: 0.5)
        bias: If True, adds a learnable bias (default: True)
    """
    
    def __init__(self, in_features: int, out_features: int, 
                 p: float = 0.5, bias: bool = True):
        super().__init__()
        if not 0 <= p < 1:
            raise ValueError(f"drop probability must be in [0, 1), got {p}")
        
        self.p = p
        self.linear = nn.Linear(in_features, out_features, bias=bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.p > 0:
            # Sample binary mask for weights
            mask = torch.bernoulli(
                torch.full_like(self.linear.weight, 1 - self.p)
            )
            # Apply mask with inverted scaling
            effective_weight = self.linear.weight * mask / (1 - self.p)
            return F.linear(x, effective_weight, self.linear.bias)
        
        return self.linear(x)
```

### DropConnect as a Wrapper

```python
class DropConnectWrapper(nn.Module):
    """
    Wrap any nn.Linear layer with DropConnect regularization.
    
    Args:
        linear_layer: An existing nn.Linear module
        p: Drop probability for each weight
    """
    
    def __init__(self, linear_layer: nn.Linear, p: float = 0.5):
        super().__init__()
        self.linear = linear_layer
        self.p = p
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.p > 0:
            mask = torch.bernoulli(
                torch.full_like(self.linear.weight, 1 - self.p)
            )
            weight = self.linear.weight * mask / (1 - self.p)
            return F.linear(x, weight, self.linear.bias)
        return self.linear(x)


def apply_dropconnect(model: nn.Module, p: float = 0.5) -> nn.Module:
    """
    Replace all nn.Linear layers in a model with DropConnect-wrapped versions.
    
    Args:
        model: Neural network
        p: Drop probability
        
    Returns:
        Model with DropConnect applied to all linear layers
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name, DropConnectWrapper(module, p=p))
        else:
            apply_dropconnect(module, p)
    return model
```

### Network with DropConnect

```python
class NetworkWithDropConnect(nn.Module):
    """Feedforward network using DropConnect layers."""
    
    def __init__(self, input_dim, hidden_dims, output_dim, drop_prob=0.5):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                DropConnect(prev_dim, hidden_dim, p=drop_prob),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        
        # No DropConnect on the output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


# Example usage
model = NetworkWithDropConnect(
    input_dim=784, hidden_dims=[512, 256], output_dim=10, drop_prob=0.5
)
```

### Monte Carlo DropConnect for Uncertainty

Like MC Dropout, DropConnect can be kept active at inference to estimate predictive uncertainty:

```python
class MCDropConnectModel(nn.Module):
    """Model supporting Monte Carlo DropConnect for uncertainty estimation."""
    
    def __init__(self, input_dim, hidden_dims, output_dim, drop_prob=0.5):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                DropConnect(prev_dim, hidden_dim, p=drop_prob),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
    
    def predict_with_uncertainty(self, x, n_samples=100):
        """
        Make predictions with uncertainty using MC DropConnect.
        
        Args:
            x: Input tensor
            n_samples: Number of stochastic forward passes
            
        Returns:
            mean: Mean prediction across samples
            std: Standard deviation (epistemic uncertainty)
        """
        self.train()  # Keep DropConnect active
        
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)
        
        return mean, std
```

## DropConnect for Convolutional Layers

DropConnect can also be applied to convolutional filters:

```python
class DropConnectConv2d(nn.Module):
    """
    Conv2d with DropConnect applied to the convolutional filters.
    
    Each weight in the kernel is independently dropped.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, p=0.5, bias=True):
        super().__init__()
        self.p = p
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias
        )
    
    def forward(self, x):
        if self.training and self.p > 0:
            mask = torch.bernoulli(
                torch.full_like(self.conv.weight, 1 - self.p)
            )
            weight = self.conv.weight * mask / (1 - self.p)
            return F.conv2d(
                x, weight, self.conv.bias,
                stride=self.conv.stride,
                padding=self.conv.padding
            )
        return self.conv(x)
```

## Training Example

```python
import torch.optim as optim
from torch.utils.data import DataLoader

def train_with_dropconnect(
    model, train_loader, val_loader, epochs=100, lr=0.001
):
    """Train a DropConnect model and compare with dropout baseline."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # Training — DropConnect active
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation — DropConnect disabled
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += y_batch.size(0)
                val_correct += predicted.eq(y_batch).sum().item()
        
        history['train_loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(val_correct / val_total)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.4f}, "
                  f"Val Acc={val_correct/val_total:.4f}")
    
    return history
```

## Computational Considerations

DropConnect introduces additional computational cost compared to standard dropout because the mask is sampled over the full weight matrix rather than just the activation vector. For a layer with $d_{\text{in}}$ inputs and $d_{\text{out}}$ outputs:

| Operation | Dropout | DropConnect |
|-----------|---------|-------------|
| Mask size | $d_{\text{in}}$ | $d_{\text{in}} \times d_{\text{out}}$ |
| Mask sampling | $O(d_{\text{in}})$ | $O(d_{\text{in}} \times d_{\text{out}})$ |
| Memory overhead | Negligible | Proportional to weight matrix |
| Inference | Same as standard | Same as standard |

In practice, the overhead is modest for most architectures because the mask generation is highly parallelizable on GPUs and the weight matrix is already in memory.

## Practical Guidelines

### When to Use DropConnect over Dropout

1. **Dense layers with many parameters**: DropConnect provides finer regularization
2. **When dropout is insufficient**: If the model still overfits with aggressive dropout
3. **Uncertainty estimation**: The larger implicit ensemble may yield better calibrated uncertainties
4. **Experimentation**: Worth trying when standard dropout gives suboptimal results

### Recommended Drop Rates

| Architecture | Dropout | DropConnect |
|-------------|---------|-------------|
| Fully connected | 0.5 | 0.5 |
| Convolutional | 0.2 – 0.3 | 0.3 – 0.5 |
| Output layer | Not applied | Not applied |

### Common Practices

- **Mode switching**: Like dropout, always use `model.train()` and `model.eval()` to toggle DropConnect
- **Bias terms**: Typically do not apply DropConnect to bias parameters
- **Combining with other regularization**: DropConnect can be combined with weight decay; reduce one if overfitting persists

## References

1. Wan, L., Zeiler, M., Zhang, S., Le Cun, Y., & Fergus, R. (2013). Regularization of Neural Networks using DropConnect. *Proceedings of the 30th International Conference on Machine Learning (ICML)*.
2. Srivastava, N., et al. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting. *JMLR*, 15(1), 1929-1958.
3. Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning. *ICML*.
