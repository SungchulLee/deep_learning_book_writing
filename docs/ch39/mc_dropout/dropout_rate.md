# Dropout Rate Selection for MC Dropout

## Overview

The dropout rate $p$ (probability of dropping a unit) is a critical hyperparameter that affects both training regularization and the quality of uncertainty estimates. This document provides principled approaches to dropout rate selection for Monte Carlo Dropout applications.

## Theoretical Foundations

### The Prior-Posterior Trade-off

From the variational inference perspective, the dropout rate $p$ implicitly specifies a prior over the weights. The relationship between dropout rate, weight decay $\lambda$, and the prior is:

$$
\lambda = \frac{p \ell^2}{2N\tau}
$$

where:

- $\ell^2$ is the prior length-scale
- $N$ is the dataset size
- $\tau$ is the model precision (inverse observation variance)

**Implications:**

- **Higher $p$** → stronger regularization → wider posteriors → higher uncertainty
- **Lower $p$** → weaker regularization → narrower posteriors → lower uncertainty

### Variance of Dropout Distribution

For a weight $w$ with learned mean $m$, the dropout distribution has:

$$
\mathbb{E}[w] = (1-p) \cdot m
$$

$$
\text{Var}[w] = p(1-p) \cdot m^2
$$

The coefficient of variation:

$$
\text{CV}[w] = \frac{\sqrt{\text{Var}[w]}}{\mathbb{E}[w]} = \sqrt{\frac{p}{1-p}}
$$

| $p$ | CV |
|-----|-----|
| 0.1 | 0.33 |
| 0.2 | 0.50 |
| 0.3 | 0.65 |
| 0.5 | 1.00 |
| 0.7 | 1.53 |

Higher $p$ induces greater weight variability and thus wider predictive distributions.

### Output Variance Propagation

For a single layer $y = \sigma(Wx + b)$ with dropout rate $p$, the variance propagates approximately as:

$$
\text{Var}[y_j] \approx \frac{p}{1-p} \cdot \mathbb{E}[y_j]^2 + (1-p) \cdot \text{Var}[\text{pre-activation}]
$$

For deep networks, this compounds across layers, making deeper layers particularly sensitive to dropout rate.

## Layer-Specific Dropout Rates

### Principles for Layer Assignment

Different layers benefit from different dropout rates:

1. **Early layers** (close to input): Lower rates (0.1-0.3)
   - Learn general features that should be stable
   - Dropping too many corrupts the feature hierarchy

2. **Middle layers**: Moderate rates (0.3-0.5)
   - Balance between regularization and information flow

3. **Late layers** (close to output): Higher rates (0.4-0.6)
   - Most prone to overfitting
   - MC Dropout here captures output uncertainty most directly

4. **Convolutional layers**: Lower rates (0.1-0.3)
   - Spatial structure should be preserved
   - Use spatial dropout (drop entire feature maps)

5. **Fully connected layers**: Higher rates (0.3-0.5)
   - More parameters, more prone to overfitting

### Architecture-Specific Guidelines

```python
def get_recommended_dropout_rates(architecture: str) -> dict:
    """
    Get recommended dropout rates by architecture type.
    
    Returns dict mapping layer type to dropout rate.
    """
    recommendations = {
        'mlp': {
            'hidden': 0.5,
            'input': 0.2,  # Optional, often omitted
        },
        'cnn': {
            'conv_early': 0.1,
            'conv_late': 0.25,
            'fc': 0.5,
        },
        'resnet': {
            'after_conv': 0.0,  # BatchNorm handles regularization
            'after_block': 0.2,
            'fc': 0.5,
        },
        'transformer': {
            'attention': 0.1,
            'ffn': 0.1,
            'embedding': 0.1,
        },
        'rnn': {
            'between_layers': 0.3,
            'recurrent': 0.0,  # Use variational dropout instead
            'output': 0.5,
        },
        'vae': {
            'encoder': 0.2,
            'decoder': 0.2,
            # Note: dropout interferes with VAE latent space
        }
    }
    return recommendations.get(architecture, {'default': 0.5})
```

## Data-Dependent Selection

### Relationship to Dataset Size

Larger datasets require less regularization:

$$
p_{\text{optimal}} \propto \frac{1}{\sqrt{N}}
$$

**Empirical guidelines:**

| Dataset Size | Recommended $p$ |
|--------------|-----------------|
| < 1,000 | 0.5 - 0.7 |
| 1,000 - 10,000 | 0.4 - 0.5 |
| 10,000 - 100,000 | 0.3 - 0.4 |
| > 100,000 | 0.1 - 0.3 |

### Model Capacity Scaling

Larger models need more regularization:

$$
p_{\text{optimal}} \propto \log(\text{num\_params})
$$

```python
def suggest_dropout_from_model_size(
    num_params: int,
    dataset_size: int,
    base_rate: float = 0.5
) -> float:
    """
    Suggest dropout rate based on model and data size.
    
    Uses heuristic: larger models / smaller data → higher dropout
    """
    import math
    
    # Capacity ratio: params per datapoint
    capacity_ratio = num_params / dataset_size
    
    # Scale factor (empirical)
    if capacity_ratio > 100:
        scale = 1.2
    elif capacity_ratio > 10:
        scale = 1.0
    elif capacity_ratio > 1:
        scale = 0.8
    else:
        scale = 0.6
    
    suggested = base_rate * scale
    
    # Clamp to reasonable range
    return max(0.1, min(0.7, suggested))
```

## Calibration-Based Selection

### Dropout Rate and Calibration

The dropout rate directly affects uncertainty calibration. Too low → overconfident; too high → underconfident.

**Expected Calibration Error (ECE):**

$$
\text{ECE} = \sum_{b=1}^{B} \frac{n_b}{N} |\text{acc}(b) - \text{conf}(b)|
$$

where $\text{acc}(b)$ is accuracy in bin $b$ and $\text{conf}(b)$ is mean confidence.

### Grid Search for Optimal Calibration

```python
import torch
import numpy as np
from typing import List, Tuple


def calibration_grid_search(
    model_class,
    train_loader,
    val_loader,
    dropout_rates: List[float],
    mc_samples: int = 50,
    n_bins: int = 15,
    **model_kwargs
) -> Tuple[float, dict]:
    """
    Find dropout rate that minimizes ECE on validation set.
    
    Args:
        model_class: Model class to instantiate
        train_loader: Training data
        val_loader: Validation data  
        dropout_rates: List of dropout rates to try
        mc_samples: MC samples for evaluation
        n_bins: Number of calibration bins
        
    Returns:
        best_rate: Optimal dropout rate
        results: Dict mapping rate to metrics
    """
    results = {}
    
    for p in dropout_rates:
        print(f"Training with dropout rate {p}")
        
        # Train model
        model = model_class(dropout_rate=p, **model_kwargs)
        train_model(model, train_loader)  # Your training function
        
        # Evaluate calibration
        ece, mce, brier = evaluate_calibration(
            model, val_loader, mc_samples, n_bins
        )
        
        results[p] = {
            'ece': ece,
            'mce': mce,
            'brier': brier
        }
        
        print(f"  ECE: {ece:.4f}, MCE: {mce:.4f}, Brier: {brier:.4f}")
    
    # Find best by ECE
    best_rate = min(results.keys(), key=lambda p: results[p]['ece'])
    
    return best_rate, results


def evaluate_calibration(
    model,
    data_loader,
    mc_samples: int = 50,
    n_bins: int = 15
) -> Tuple[float, float, float]:
    """
    Compute calibration metrics using MC Dropout predictions.
    
    Returns:
        ece: Expected Calibration Error
        mce: Maximum Calibration Error
        brier: Brier score
    """
    model.eval()
    model.enable_mc_dropout()
    
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in data_loader:
            # MC Dropout prediction
            probs_samples = []
            for _ in range(mc_samples):
                logits = model(x)
                probs = torch.softmax(logits, dim=-1)
                probs_samples.append(probs)
            
            mean_probs = torch.stack(probs_samples).mean(dim=0)
            all_probs.append(mean_probs.cpu())
            all_labels.append(y.cpu())
    
    all_probs = torch.cat(all_probs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Compute metrics
    confidences, predictions = all_probs.max(dim=1)
    accuracies = predictions.eq(all_labels).float()
    
    # Binning
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    mce = 0.0
    
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        prop_in_bin = in_bin.float().mean()
        
        if prop_in_bin > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].mean()
            
            gap = abs(avg_accuracy - avg_confidence)
            ece += prop_in_bin * gap
            mce = max(mce, gap.item())
    
    # Brier score
    one_hot = torch.zeros_like(all_probs)
    one_hot.scatter_(1, all_labels.unsqueeze(1), 1)
    brier = ((all_probs - one_hot) ** 2).sum(dim=1).mean()
    
    return ece.item(), mce, brier.item()
```

## Uncertainty-Quality Trade-offs

### Sharpness vs. Calibration

**Sharpness** measures how concentrated the predictive distribution is:

$$
\text{Sharpness} = -\mathbb{E}[\mathbb{H}[\hat{p}]] = \mathbb{E}\left[\sum_c \hat{p}_c \log \hat{p}_c\right]
$$

Higher dropout → less sharp (wider) distributions.

**The trade-off:**

- Low $p$: Sharp but potentially overconfident
- High $p$: Well-calibrated but potentially too uncertain

```python
def compute_sharpness(probs: torch.Tensor) -> float:
    """
    Compute sharpness (negative entropy) of predictions.
    Higher = more confident/sharp predictions.
    """
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
    return -entropy.mean().item()


def compute_calibration_sharpness_curve(
    model,
    val_loader,
    dropout_rates: List[float],
    mc_samples: int = 50
) -> dict:
    """
    Compute calibration and sharpness for different dropout rates.
    
    Useful for visualizing the trade-off.
    """
    results = {'dropout_rate': [], 'ece': [], 'sharpness': []}
    
    for p in dropout_rates:
        # Temporarily modify dropout rate
        original_rates = []
        for m in model.modules():
            if hasattr(m, 'p'):
                original_rates.append(m.p)
                m.p = p
        
        # Evaluate
        all_probs = []
        all_labels = []
        
        model.enable_mc_dropout()
        with torch.no_grad():
            for x, y in val_loader:
                samples = [torch.softmax(model(x), -1) for _ in range(mc_samples)]
                mean_probs = torch.stack(samples).mean(dim=0)
                all_probs.append(mean_probs)
                all_labels.append(y)
        
        all_probs = torch.cat(all_probs)
        all_labels = torch.cat(all_labels)
        
        # Metrics
        ece, _, _ = evaluate_calibration(model, val_loader, mc_samples)
        sharpness = compute_sharpness(all_probs)
        
        results['dropout_rate'].append(p)
        results['ece'].append(ece)
        results['sharpness'].append(sharpness)
        
        # Restore original rates
        idx = 0
        for m in model.modules():
            if hasattr(m, 'p'):
                m.p = original_rates[idx]
                idx += 1
    
    return results
```

### Predictive Performance Trade-off

Higher dropout during training can hurt accuracy:

```python
def accuracy_dropout_sweep(
    model_class,
    train_loader,
    val_loader,
    dropout_rates: List[float],
    **kwargs
) -> dict:
    """
    Measure accuracy vs dropout rate trade-off.
    """
    results = {'p': [], 'train_acc': [], 'val_acc': [], 'gap': []}
    
    for p in dropout_rates:
        model = model_class(dropout_rate=p, **kwargs)
        history = train_model(model, train_loader, val_loader)
        
        results['p'].append(p)
        results['train_acc'].append(history['train_acc'][-1])
        results['val_acc'].append(history['val_acc'][-1])
        results['gap'].append(
            history['train_acc'][-1] - history['val_acc'][-1]
        )
    
    return results
```

## Concrete Dropout: Learning the Dropout Rate

### Motivation

Instead of treating $p$ as a hyperparameter, we can learn it during training. Concrete Dropout (Gal et al., 2017) uses a continuous relaxation of Bernoulli dropout.

### The Concrete Distribution

The Concrete (or Gumbel-Softmax) relaxation:

$$
z = \sigma\left( \frac{1}{\tau} \left( \log \frac{p}{1-p} + \log \frac{u}{1-u} \right) \right)
$$

where $u \sim \text{Uniform}(0, 1)$ and $\tau$ is the temperature.

As $\tau \to 0$, this approaches a Bernoulli sample. During training, we use $\tau > 0$ for gradient flow.

### Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConcreteDropout(nn.Module):
    """
    Concrete Dropout layer with learnable dropout probability.
    
    The dropout rate is learned via gradient descent through the
    concrete relaxation of Bernoulli dropout.
    """
    
    def __init__(
        self,
        init_p: float = 0.5,
        temperature: float = 0.1,
        weight_regularizer: float = 1e-6,
        dropout_regularizer: float = 1e-5
    ):
        super().__init__()
        
        # Learnable logit for dropout probability
        # p = sigmoid(logit_p)
        init_logit = torch.log(torch.tensor(init_p / (1 - init_p)))
        self.logit_p = nn.Parameter(init_logit)
        
        self.temperature = temperature
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        
    @property
    def p(self) -> torch.Tensor:
        """Current dropout probability."""
        return torch.sigmoid(self.logit_p)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.p
        
        if self.training:
            # Concrete relaxation
            u = torch.rand_like(x).clamp(1e-8, 1 - 1e-8)
            s = torch.sigmoid(
                (torch.log(u) - torch.log(1 - u) + 
                 torch.log(p) - torch.log(1 - p)) / self.temperature
            )
            z = 1 - s
        else:
            # Hard dropout for inference
            z = torch.bernoulli(torch.full_like(x, 1 - p))
        
        # Scale by 1/(1-p) for inverted dropout
        return x * z / (1 - p + 1e-8)
    
    def regularization_loss(
        self,
        layer_weight: torch.Tensor,
        num_data: int
    ) -> torch.Tensor:
        """
        Compute the regularization loss for this layer.
        
        Includes:
        1. Weight regularization (scaled by dropout)
        2. Entropy regularization on p (encourages learning non-trivial p)
        """
        p = self.p
        
        # Weight regularization: λ * (1-p) * ||W||^2
        weight_reg = self.weight_regularizer * (1 - p) * (layer_weight ** 2).sum()
        
        # Dropout regularization: encourages p away from 0 and 1
        # Uses KL divergence to uniform Bernoulli
        dropout_reg = self.dropout_regularizer * num_data * (
            p * torch.log(p + 1e-8) + (1 - p) * torch.log(1 - p + 1e-8)
        )
        
        return weight_reg + dropout_reg


class ConcreteDropoutLinear(nn.Module):
    """
    Linear layer with Concrete Dropout.
    
    Wraps nn.Linear with learnable dropout applied to input.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        init_p: float = 0.5,
        weight_regularizer: float = 1e-6,
        dropout_regularizer: float = 1e-5
    ):
        super().__init__()
        
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = ConcreteDropout(
            init_p=init_p,
            weight_regularizer=weight_regularizer,
            dropout_regularizer=dropout_regularizer
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        return self.linear(x)
    
    def regularization_loss(self, num_data: int) -> torch.Tensor:
        return self.dropout.regularization_loss(self.linear.weight, num_data)
    
    @property
    def p(self) -> float:
        return self.dropout.p.item()


class ConcreteDropoutNetwork(nn.Module):
    """
    Network with learnable dropout rates via Concrete Dropout.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        init_p: float = 0.5,
        weight_regularizer: float = 1e-6,
        dropout_regularizer: float = 1e-5
    ):
        super().__init__()
        
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            self.layers.append(
                ConcreteDropoutLinear(
                    prev_dim, hidden_dim,
                    init_p=init_p,
                    weight_regularizer=weight_regularizer,
                    dropout_regularizer=dropout_regularizer
                )
            )
            prev_dim = hidden_dim
        
        # Output layer (no dropout)
        self.output = nn.Linear(prev_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.output(x)
    
    def regularization_loss(self, num_data: int) -> torch.Tensor:
        """Total regularization loss for all Concrete Dropout layers."""
        reg = 0
        for layer in self.layers:
            reg = reg + layer.regularization_loss(num_data)
        return reg
    
    def get_dropout_rates(self) -> List[float]:
        """Get learned dropout rates for all layers."""
        return [layer.p for layer in self.layers]


def train_concrete_dropout(
    model: ConcreteDropoutNetwork,
    train_loader,
    val_loader,
    epochs: int = 100,
    lr: float = 1e-3
):
    """
    Training loop for Concrete Dropout with proper regularization.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    num_data = len(train_loader.dataset)
    
    for epoch in range(epochs):
        model.train()
        
        for x, y in train_loader:
            optimizer.zero_grad()
            
            output = model(x)
            nll_loss = criterion(output, y)
            reg_loss = model.regularization_loss(num_data)
            
            loss = nll_loss + reg_loss
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            rates = model.get_dropout_rates()
            print(f"Epoch {epoch+1}: Learned dropout rates: {rates}")
    
    return model
```

## Practical Recommendations

### Decision Framework

```
1. Start with architecture-specific defaults
   - MLP: 0.5 for hidden layers
   - CNN: 0.25 conv, 0.5 FC
   - Transformer: 0.1

2. Adjust for dataset size
   - Small data (<10k): increase by 0.1-0.2
   - Large data (>100k): decrease by 0.1-0.2

3. Validate on calibration metrics
   - If overconfident (low ECE but wrong predictions confident): increase p
   - If underconfident (high uncertainty even when correct): decrease p

4. Consider task requirements
   - Safety-critical: prefer higher p (conservative uncertainty)
   - Real-time inference: prefer lower p (fewer MC samples needed)

5. Use Concrete Dropout for automatic tuning
   - When uncertain about optimal rate
   - When different layers may need different rates
```

### Common Pitfalls

1. **Same rate everywhere:** Different layer types need different rates

2. **Ignoring model capacity:** Large models need more dropout

3. **Training vs. inference mismatch:** Ensure MC Dropout uses the training rate

4. **Ignoring calibration:** Accuracy alone doesn't indicate good uncertainty

5. **Not considering the task:** OOD detection may benefit from higher $p$ than in-distribution prediction

## References

1. Gal, Y., Hron, J., & Kendall, A. (2017). Concrete Dropout. *NeurIPS*.

2. Srivastava, N., et al. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting. *JMLR*.

3. Guo, C., et al. (2017). On Calibration of Modern Neural Networks. *ICML*.

4. Gal, Y. (2016). Uncertainty in Deep Learning. *PhD Thesis*.
