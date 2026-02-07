# Free Adversarial Training

## Introduction

**Free Adversarial Training** (Shafahi et al., 2019) addresses the primary bottleneck of adversarial training: computational cost. Standard PGD-AT requires $K$ forward-backward passes per batch for the inner maximization, making it 7-10× slower than standard training. Free AT achieves comparable robustness at **nearly the cost of standard training** by recycling gradients.

## Motivation

In standard adversarial training, the computation graph looks like:

```
For each batch:
    1. PGD inner loop (K forward-backward passes) → generates x_adv
    2. Forward pass on x_adv → compute loss
    3. Backward pass → update θ
```

Most of the cost is in step 1. Free AT observes that the gradients computed during step 1 contain useful information for updating model parameters $\theta$, not just the perturbation $\boldsymbol{\delta}$.

## Mathematical Foundation

### Key Insight: Gradient Reuse

During PGD, we compute $\nabla_\mathbf{x} \mathcal{L}$ to update the perturbation. But via the chain rule:

$$
\nabla_\mathbf{x} \mathcal{L}(f_\theta(\mathbf{x} + \boldsymbol{\delta}), y)
$$

also depends on model parameters through $f_\theta$. The **same backward pass** can simultaneously compute $\nabla_\theta \mathcal{L}$ for the model update.

### Free AT Algorithm

**Algorithm: Free Adversarial Training**

For each epoch, replay each minibatch $m$ times:

```
For each mini-batch (x, y), repeat m times:
    1. Compute logits: z = f_θ(x + δ)
    2. Compute loss: L = CrossEntropy(z, y)
    3. Single backward pass → get ∇_x L and ∇_θ L simultaneously
    4. Update perturbation: δ ← δ + ε · sign(∇_x L), then project
    5. Update parameters: θ ← θ - η · ∇_θ L
```

The perturbation $\boldsymbol{\delta}$ persists across the $m$ replays, effectively performing $m$ PGD steps while also updating the model $m$ times.

### Epoch Equivalence

If standard training runs for $E$ epochs with batch size $B$, Free AT runs for $E/m$ epochs with $m$ replays per batch. The total number of gradient computations is the same as standard training, but each gradient serves dual purpose.

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, List
from tqdm import tqdm

class FreeAdversarialTrainer:
    """
    Free Adversarial Training.
    
    Achieves adversarial robustness at nearly the cost of
    standard training by reusing gradients for both perturbation
    updates and parameter updates.
    
    Parameters
    ----------
    model : nn.Module
        Model to train
    epsilon : float
        Perturbation budget
    m : int
        Number of minibatch replays (PGD steps per batch)
    """
    
    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 8/255,
        m: int = 8,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.epsilon = epsilon
        self.m = m
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.model.to(self.device)
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer
    ) -> Dict[str, float]:
        """
        Train for one epoch with Free AT.
        
        Each batch is replayed m times. The perturbation
        persists and accumulates across replays.
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Free AT')
        for x, y in pbar:
            x, y = x.to(self.device), y.to(self.device)
            
            # Initialize perturbation (persists across m replays)
            delta = torch.zeros_like(x, requires_grad=False)
            
            for _ in range(self.m):
                # Apply current perturbation
                x_adv = torch.clamp(x + delta, 0, 1)
                x_adv.requires_grad_(True)
                
                # Forward pass
                logits = self.model(x_adv)
                loss = F.cross_entropy(logits, y)
                
                # Single backward pass: computes both ∇_x and ∇_θ
                optimizer.zero_grad()
                loss.backward()
                
                # Update model parameters (using ∇_θ)
                optimizer.step()
                
                # Update perturbation (using ∇_x)
                with torch.no_grad():
                    grad = x_adv.grad.data
                    delta = delta + self.epsilon * grad.sign()
                    delta = torch.clamp(delta, -self.epsilon, self.epsilon)
                
                total_loss += loss.item() * len(y)
                correct += (logits.argmax(1) == y).sum().item()
                total += len(y)
            
            pbar.set_postfix({
                'loss': f'{total_loss/total:.4f}',
                'acc': f'{correct/total:.2%}'
            })
        
        return {
            'loss': total_loss / total,
            'accuracy': correct / total
        }
    
    def train(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int,
        optimizer: Optional[optim.Optimizer] = None
    ) -> List[Dict]:
        """
        Complete training loop.
        
        Note: Run for epochs/m epochs since each batch
        is processed m times.
        """
        if optimizer is None:
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=0.1, momentum=0.9, weight_decay=5e-4
            )
        
        # Adjust epochs for replay factor
        adjusted_epochs = max(epochs // self.m, 1)
        
        history = []
        for epoch in range(1, adjusted_epochs + 1):
            metrics = self.train_epoch(train_loader, optimizer)
            history.append(metrics)
            print(f"Epoch {epoch}/{adjusted_epochs}: "
                  f"Loss={metrics['loss']:.4f}, "
                  f"Acc={metrics['accuracy']:.2%}")
        
        return history
```

## Computational Comparison

| Method | Forward Passes/Batch | Backward Passes/Batch | Relative Cost |
|--------|---------------------|----------------------|---------------|
| Standard Training | 1 | 1 | 1× |
| PGD-AT ($K=10$) | 11 | 11 | ~10× |
| Free AT ($m=8$) | 8 | 8 | ~1.2× (amortized) |
| Fast AT | 2 | 2 | ~2× |

Free AT achieves ~8× speedup over PGD-AT by amortizing the cost across replays.

## Robustness Results

CIFAR-10, $\varepsilon = 8/255$:

| Method | Training Time | Clean Acc | Robust Acc (PGD-20) |
|--------|-------------|-----------|---------------------|
| Standard | 1× | 95% | 0% |
| PGD-AT | 10× | 85% | 48% |
| Free AT ($m=8$) | ~1.2× | 83% | 43% |

Free AT trades a small amount of robustness for dramatic computational savings.

## Limitations

- **Slightly weaker robustness**: ~3-5% lower robust accuracy than PGD-AT
- **Catastrophic overfitting risk**: Can suffer from sudden robustness collapse
- **Hyperparameter sensitivity**: The replay factor $m$ must be chosen carefully

## References

1. Shafahi, A., et al. (2019). "Adversarial Training for Free!" NeurIPS.
