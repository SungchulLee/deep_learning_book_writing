# Deep Ensemble Fundamentals

## Overview

Deep ensembles (Lakshminarayanan et al., 2017) provide state-of-the-art uncertainty estimation through a remarkably simple idea: train multiple networks independently and use their disagreement as a measure of uncertainty. Despite lacking a formal Bayesian interpretation, ensembles consistently outperform more principled methods on calibration and out-of-distribution detection benchmarks.

## Why Ensembles Work

### Multiple Modes of the Loss Landscape

Neural network loss landscapes contain many local minima. Different random initializations lead to different solutions that may agree on in-distribution data but disagree on ambiguous or out-of-distribution inputs. This disagreement naturally captures epistemic uncertainty.

### The Ensemble Predictive Distribution

Given $M$ independently trained models $\{f_{\theta_m}\}_{m=1}^M$, the ensemble prediction is:

$$p(y|\mathbf{x}, \mathcal{D}) \approx \frac{1}{M} \sum_{m=1}^M p(y|\mathbf{x}, \theta_m)$$

For regression with Gaussian outputs $(\mu_m, \sigma_m^2)$ from each member:

$$\bar{\mu} = \frac{1}{M}\sum_{m=1}^M \mu_m, \quad \bar{\sigma}^2 = \frac{1}{M}\sum_{m=1}^M (\sigma_m^2 + \mu_m^2) - \bar{\mu}^2$$

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import numpy as np


class EnsembleMember(nn.Module):
    """Single ensemble member for classification."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int],
                 output_dim: int, dropout: float = 0.0):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DeepEnsemble(nn.Module):
    """
    Deep Ensemble for uncertainty estimation.
    
    Trains M independent models and uses prediction disagreement
    as an uncertainty measure.
    """
    
    def __init__(self, n_members: int, input_dim: int,
                 hidden_dims: List[int], output_dim: int,
                 task: str = 'classification'):
        super().__init__()
        self.n_members = n_members
        self.task = task
        self.members = nn.ModuleList([
            EnsembleMember(input_dim, hidden_dims, output_dim)
            for _ in range(n_members)
        ])
    
    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Ensemble prediction with uncertainty.
        
        Returns:
            mean_probs/mean: Ensemble mean prediction
            epistemic: Prediction variance across members
            predictions: Predicted classes (classification)
        """
        all_outputs = []
        for member in self.members:
            member.eval()
            all_outputs.append(member(x))
        
        outputs = torch.stack(all_outputs)  # (M, batch, dim)
        
        if self.task == 'classification':
            probs = F.softmax(outputs, dim=-1)  # (M, batch, classes)
            mean_probs = probs.mean(dim=0)
            predictions = mean_probs.argmax(dim=-1)
            
            # Epistemic: variance across members
            epistemic = probs.var(dim=0).mean(dim=-1)
            
            # Predictive entropy
            epsilon = 1e-10
            entropy = -torch.sum(
                mean_probs * torch.log(mean_probs + epsilon), dim=-1
            )
            
            return {
                'probs': mean_probs,
                'predictions': predictions,
                'epistemic': epistemic,
                'entropy': entropy,
                'all_probs': probs
            }
        else:
            mean = outputs.mean(dim=0)
            epistemic_var = outputs.var(dim=0)
            
            return {
                'mean': mean,
                'epistemic_var': epistemic_var,
                'std': torch.sqrt(epistemic_var)
            }


def train_ensemble(
    ensemble: DeepEnsemble,
    train_loader,
    epochs: int = 50,
    lr: float = 1e-3,
    device: str = 'cpu'
) -> Dict[str, List[float]]:
    """
    Train each ensemble member independently with different
    random initialization (already handled by PyTorch default init).
    """
    history = {f'member_{m}_loss': [] for m in range(ensemble.n_members)}
    criterion = nn.CrossEntropyLoss()
    
    for m, member in enumerate(ensemble.members):
        member = member.to(device)
        optimizer = torch.optim.Adam(member.parameters(), lr=lr)
        
        print(f"\nTraining member {m+1}/{ensemble.n_members}")
        
        for epoch in range(epochs):
            member.train()
            epoch_loss = 0
            
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                if x.dim() > 2:
                    x = x.view(x.size(0), -1)
                
                optimizer.zero_grad()
                loss = criterion(member(x), y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            history[f'member_{m}_loss'].append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}: Loss = {avg_loss:.4f}")
    
    return history
```

## Ensemble Size Selection

Empirical studies show diminishing returns beyond 5 members:

| Ensemble Size | Relative ECE Improvement | Compute Cost |
|--------------|-------------------------|--------------|
| 1 (baseline) | 0% | 1× |
| 3 | ~70% of max improvement | 3× |
| 5 | ~90% of max improvement | 5× |
| 10 | ~97% of max improvement | 10× |
| 20 | ~99% of max improvement | 20× |

**Recommendation**: Use $M = 5$ as default. Use $M = 3$ when compute-constrained.

## Key Takeaways

!!! success "Summary"
    1. **Simple and effective**: Train M models independently, average predictions
    2. **State-of-the-art uncertainty**: Consistently outperforms Bayesian methods on benchmarks
    3. **Captures epistemic uncertainty** through inter-model disagreement
    4. **5 members typically sufficient** with diminishing returns beyond
    5. **Main cost**: M× training and inference compute

## References

- Lakshminarayanan, B., et al. (2017). "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles." NeurIPS.
- Fort, S., et al. (2019). "Deep Ensembles: A Loss Landscape Perspective." arXiv.
