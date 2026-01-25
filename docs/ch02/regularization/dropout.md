# Dropout

## Overview

Dropout is a regularization technique that randomly sets a fraction of neuron activations to zero during training. This prevents co-adaptation of neurons and forces the network to learn more robust features that are useful in conjunction with many different random subsets of other neurons.

## Mathematical Formulation

### Basic Dropout Operation

During training, for a layer with activation vector $h \in \mathbb{R}^d$ and dropout probability $p$:

$$
\tilde{h}_i = \begin{cases}
0 & \text{with probability } p \\
\frac{h_i}{1-p} & \text{with probability } 1-p
\end{cases}
$$

The scaling by $\frac{1}{1-p}$ (called **inverted dropout**) ensures the expected value remains unchanged:

$$
\mathbb{E}[\tilde{h}_i] = p \cdot 0 + (1-p) \cdot \frac{h_i}{1-p} = h_i
$$

### Mask Formulation

Using a binary mask $m \sim \text{Bernoulli}(1-p)^d$:

$$
\tilde{h} = \frac{m \odot h}{1-p}
$$

where $\odot$ denotes element-wise multiplication.

### Forward Pass with Dropout

For a feedforward layer with weights $W$, bias $b$, and activation $\sigma$:

**Training:**
$$
y = \sigma\left( W \cdot \frac{m \odot x}{1-p} + b \right)
$$

**Inference:**
$$
y = \sigma(Wx + b)
$$

No dropout is applied during inference due to the inverted dropout scaling during training.

## Theoretical Interpretation

### Ensemble Interpretation

Dropout can be viewed as training an exponential ensemble of neural networks:

- For a network with $d$ units, there are $2^d$ possible dropout masks
- Each training step samples one sub-network from this ensemble
- At test time, the full network approximates the ensemble's average prediction

### Bayesian Interpretation

Dropout approximates Bayesian inference in deep learning. Gal & Ghahramani (2016) showed that dropout training minimizes an approximation to the KL divergence between an approximate posterior and the true posterior over weights.

### Noise Injection View

Dropout injects multiplicative noise into the network:

$$
\tilde{h} = h \odot \epsilon, \quad \epsilon_i \sim \begin{cases}
\frac{1}{1-p} & \text{with prob } 1-p \\
0 & \text{with prob } p
\end{cases}
$$

This multiplicative noise has variance $\text{Var}[\epsilon_i] = \frac{p}{1-p}$.

## PyTorch Implementation

### Built-in Dropout

```python
import torch
import torch.nn as nn

class NetworkWithDropout(nn.Module):
    """Standard network with dropout layers."""
    
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.5):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


# Training vs evaluation mode
model = NetworkWithDropout(784, [512, 256], 10, dropout_rate=0.5)
model.train()  # Dropout active
model.eval()   # Dropout disabled
```

### Custom Dropout Implementation

```python
class CustomDropout(nn.Module):
    """Custom dropout implementation showing the internals."""
    
    def __init__(self, p: float = 0.5):
        super().__init__()
        if not 0 <= p <= 1:
            raise ValueError(f"dropout probability must be in [0, 1], got {p}")
        self.p = p
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0:
            return x
        
        # Generate binary mask (1 = keep, 0 = drop)
        mask = torch.bernoulli(torch.full_like(x, 1 - self.p))
        
        # Apply mask with inverted dropout scaling
        return x * mask / (1 - self.p)
```

### Dropout Variants

```python
# 1D Dropout - for sequences (drops entire channels)
dropout_1d = nn.Dropout1d(p=0.5)  # Input: (batch, channels, length)

# 2D Spatial Dropout - for images (drops entire feature maps)  
dropout_2d = nn.Dropout2d(p=0.5)  # Input: (batch, channels, H, W)

# 3D Spatial Dropout - for video/3D data
dropout_3d = nn.Dropout3d(p=0.5)  # Input: (batch, channels, D, H, W)

# Alpha Dropout - for SELU activations (Self-Normalizing Networks)
alpha_dropout = nn.AlphaDropout(p=0.5)
```

### Monte Carlo Dropout for Uncertainty

```python
class MCDropoutModel(nn.Module):
    """Model supporting Monte Carlo Dropout for uncertainty estimation."""
    
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.5):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
    
    def predict_with_uncertainty(self, x, n_samples=100):
        """
        Make predictions with uncertainty using MC Dropout.
        
        Returns:
            mean: Mean prediction
            std: Standard deviation (epistemic uncertainty)
        """
        self.train()  # Keep dropout active
        
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

## Training with Dropout

```python
import torch.optim as optim
from torch.utils.data import DataLoader

def train_with_dropout(model, train_loader, val_loader, epochs=100, lr=0.001):
    """Train a model with dropout regularization."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # Training phase - dropout ACTIVE
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += y_batch.size(0)
            train_correct += predicted.eq(y_batch).sum().item()
        
        # Validation phase - dropout DISABLED
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
        history['train_acc'].append(train_correct / train_total)
        history['val_acc'].append(val_correct / val_total)
    
    return history
```

## Advanced Techniques

### DropConnect

Drops connections (weights) instead of activations:

```python
class DropConnect(nn.Module):
    """DropConnect: drops weights instead of activations."""
    
    def __init__(self, in_features, out_features, p=0.5):
        super().__init__()
        self.p = p
        self.weight = nn.Parameter(torch.randn(out_features, in_features) / 
                                   (in_features ** 0.5))
        self.bias = nn.Parameter(torch.zeros(out_features))
    
    def forward(self, x):
        if self.training and self.p > 0:
            mask = (torch.rand_like(self.weight) > self.p).float()
            effective_weight = self.weight * mask / (1 - self.p)
        else:
            effective_weight = self.weight
        return nn.functional.linear(x, effective_weight, self.bias)
```

### Spatial Dropout for CNNs

```python
class CNNWithSpatialDropout(nn.Module):
    """CNN using spatial dropout (drops entire feature maps)."""
    
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=dropout_rate),  # Spatial dropout
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=dropout_rate),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),  # Regular dropout
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
```

### Dropout in Transformers

```python
class TransformerBlockWithDropout(nn.Module):
    """Transformer block with dropout in standard positions."""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout1(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        return x
```

## Practical Guidelines

### Recommended Dropout Rates

| Architecture | Location | Typical Rate |
|--------------|----------|--------------|
| Fully connected | Hidden layers | 0.5 |
| CNNs | After conv layers | 0.2 - 0.3 |
| CNNs | Before final FC | 0.5 |
| RNNs/LSTMs | Between layers | 0.2 - 0.5 |
| Transformers | Attention/FFN | 0.1 |

### When to Use Dropout

1. **Large networks** with many parameters
2. **Limited training data**
3. **Clear overfitting** (train >> val performance)
4. **Dense layers** (more effective than conv layers)

### When NOT to Use Dropout

1. **Very small networks** - may hurt performance
2. **With batch normalization** - use lower dropout rates
3. **Sufficient data** - may not be needed
4. **Already using strong augmentation**

### Common Mistakes

1. **Forgetting mode switching**: Always use `model.train()` and `model.eval()`
2. **Dropout after output**: Never put dropout before final layer
3. **Same rate everywhere**: Different layers may need different rates
4. **Rate too high**: Start with 0.2-0.3, increase if needed

## Combining with Other Regularization

```python
class RegularizedNetwork(nn.Module):
    """Network combining dropout with batch norm and weight decay."""
    
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

# Use with weight decay (L2)
model = RegularizedNetwork(784, 256, 10)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

## References

1. Srivastava, N., et al. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting. *JMLR*, 15(1), 1929-1958.
2. Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation. *ICML*.
3. Wan, L., et al. (2013). Regularization of Neural Networks using DropConnect. *ICML*.
