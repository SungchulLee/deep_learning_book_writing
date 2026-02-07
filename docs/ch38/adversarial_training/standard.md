# Adversarial Training

## Introduction

**Adversarial training** is the most effective defense against adversarial attacks. It augments the training process with adversarial examples, teaching the model to be robust within an $\varepsilon$-ball around each training point. This section covers standard adversarial training (PGD-AT), TRADES, MART, and practical considerations.

## Mathematical Foundation

### Standard Training vs Robust Training

**Standard Empirical Risk Minimization (ERM):**
$$
\min_\theta \mathbb{E}_{(\mathbf{x}, y) \sim \mathcal{D}} \left[ \mathcal{L}(f_\theta(\mathbf{x}), y) \right]
$$

This optimizes for average-case performance but ignores adversarial perturbations.

**Robust Optimization (Adversarial Training):**
$$
\min_\theta \mathbb{E}_{(\mathbf{x}, y) \sim \mathcal{D}} \left[ \max_{\|\boldsymbol{\delta}\|_p \leq \varepsilon} \mathcal{L}(f_\theta(\mathbf{x} + \boldsymbol{\delta}), y) \right]
$$

This is a **min-max** problem:
- **Inner maximization**: Find worst-case perturbation (the attack)
- **Outer minimization**: Train to be robust against it

### Interpretation

For each training example $(\mathbf{x}, y)$:
1. Find the adversarial perturbation $\boldsymbol{\delta}^*$ that maximizes loss
2. Update parameters to minimize loss on $\mathbf{x} + \boldsymbol{\delta}^*$

The model learns to correctly classify not just $\mathbf{x}$, but the entire $\varepsilon$-ball around $\mathbf{x}$.

### PGD-Based Adversarial Training

Since the inner maximization is intractable, we approximate it with PGD:

$$
\boldsymbol{\delta}^* \approx \text{PGD}(\mathbf{x}, y, \varepsilon, \alpha, K)
$$

**Algorithm: PGD Adversarial Training**

```
For each epoch:
    For each mini-batch (x, y):
        1. Generate adversarial examples: x_adv = PGD(x, y, ε, α, K)
        2. Compute loss: L = CrossEntropy(f_θ(x_adv), y)
        3. Update parameters: θ ← θ - η∇_θ L
```

## TRADES: Theoretically Principled Trade-off

### Motivation

Standard adversarial training can sacrifice too much clean accuracy. **TRADES** (Zhang et al., 2019) explicitly balances the trade-off between clean and robust accuracy.

### Formulation

TRADES decomposes the robust loss:

$$
\mathcal{L}_{\text{TRADES}} = \mathcal{L}_{\text{CE}}(f_\theta(\mathbf{x}), y) + \beta \cdot \text{KL}(f_\theta(\mathbf{x}) \| f_\theta(\mathbf{x}_{\text{adv}}))
$$

where:
- First term: Standard cross-entropy (clean accuracy)
- Second term: KL divergence between clean and adversarial predictions (local smoothness)
- $\beta$: Trade-off parameter (typically 1-6)

### Intuition

- **Clean loss**: Ensures predictions are correct on original data
- **KL term**: Encourages predictions to be **consistent** between clean and perturbed inputs
- Together: Robust predictions that are also accurate

### Key Difference from Standard AT

| Aspect | Standard AT | TRADES |
|--------|-------------|--------|
| Loss target | Adversarial examples only | Clean + consistency |
| Trade-off control | Implicit (via $\varepsilon$) | Explicit (via $\beta$) |
| Clean accuracy | Lower | Higher |
| Robust accuracy | Higher | Slightly lower |

## MART: Misclassification-Aware Robust Training

### Motivation

Not all examples are equally important. **MART** (Wang et al., 2020) focuses more on misclassified examples.

### Formulation

$$
\mathcal{L}_{\text{MART}} = \text{BCE}(f_\theta(\mathbf{x}_{\text{adv}}), y) + \lambda \cdot (1 - p_y(\mathbf{x})) \cdot \text{KL}(f_\theta(\mathbf{x}) \| f_\theta(\mathbf{x}_{\text{adv}}))
$$

where:
- $p_y(\mathbf{x})$ is the probability of true class on clean input
- $(1 - p_y(\mathbf{x}))$ upweights examples that are already difficult

### Intuition

- Misclassified examples (low $p_y$) get higher weight in the regularization
- Correctly classified examples (high $p_y$) get lower weight
- Focus defense effort where it's most needed

## PyTorch Implementation

### Standard Adversarial Training

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, List
from tqdm import tqdm

class AdversarialTrainer:
    """
    Standard PGD-based adversarial training.
    
    Solves: min_θ E[ max_{||δ||≤ε} L(f_θ(x+δ), y) ]
    
    Parameters
    ----------
    model : nn.Module
        Model to train
    epsilon : float
        Perturbation budget (default: 8/255 for CIFAR-10)
    alpha : float
        PGD step size
    num_iter : int
        PGD iterations during training
    """
    
    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 8/255,
        alpha: float = 2/255,
        num_iter: int = 10,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_iter = num_iter
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model.to(self.device)
    
    def _pgd_attack(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        """Generate PGD adversarial examples."""
        x_adv = x + torch.empty_like(x).uniform_(-self.epsilon, self.epsilon)
        x_adv = torch.clamp(x_adv, 0, 1)
        
        for _ in range(self.num_iter):
            x_adv.requires_grad_(True)
            loss = F.cross_entropy(self.model(x_adv), y)
            
            self.model.zero_grad()
            loss.backward()
            
            with torch.no_grad():
                x_adv = x_adv + self.alpha * x_adv.grad.sign()
                delta = torch.clamp(x_adv - x, -self.epsilon, self.epsilon)
                x_adv = torch.clamp(x + delta, 0, 1)
        
        return x_adv.detach()
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for x, y in pbar:
            x, y = x.to(self.device), y.to(self.device)
            
            # Generate adversarial examples
            x_adv = self._pgd_attack(x, y)
            
            # Forward pass on adversarial examples
            optimizer.zero_grad()
            logits = self.model(x_adv)
            loss = F.cross_entropy(logits, y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track metrics
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
    
    def evaluate(
        self,
        test_loader: DataLoader,
        attack_iter: int = 20
    ) -> Dict[str, float]:
        """Evaluate clean and robust accuracy."""
        self.model.eval()
        
        clean_correct = 0
        robust_correct = 0
        total = 0
        
        # Use stronger attack for evaluation
        original_iter = self.num_iter
        self.num_iter = attack_iter
        
        for x, y in tqdm(test_loader, desc='Evaluating'):
            x, y = x.to(self.device), y.to(self.device)
            
            # Clean accuracy
            with torch.no_grad():
                clean_pred = self.model(x).argmax(1)
                clean_correct += (clean_pred == y).sum().item()
            
            # Robust accuracy
            x_adv = self._pgd_attack(x, y)
            with torch.no_grad():
                robust_pred = self.model(x_adv).argmax(1)
                robust_correct += (robust_pred == y).sum().item()
            
            total += len(y)
        
        self.num_iter = original_iter
        
        return {
            'clean_accuracy': clean_correct / total,
            'robust_accuracy': robust_correct / total
        }
    
    def train(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        save_path: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """Complete training loop."""
        if optimizer is None:
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=0.1, momentum=0.9, weight_decay=5e-4
            )
        
        if scheduler is None:
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[int(0.5*epochs), int(0.75*epochs)],
                gamma=0.1
            )
        
        history = {
            'train_loss': [], 'train_acc': [],
            'clean_acc': [], 'robust_acc': []
        }
        
        best_robust = 0
        
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_loader, optimizer)
            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics['accuracy'])
            
            # Evaluate
            eval_metrics = self.evaluate(test_loader)
            history['clean_acc'].append(eval_metrics['clean_accuracy'])
            history['robust_acc'].append(eval_metrics['robust_accuracy'])
            
            print(f"  Clean: {eval_metrics['clean_accuracy']:.2%}, "
                  f"Robust: {eval_metrics['robust_accuracy']:.2%}")
            
            # Save best
            if save_path and eval_metrics['robust_accuracy'] > best_robust:
                best_robust = eval_metrics['robust_accuracy']
                torch.save(self.model.state_dict(), save_path)
                print(f"  Saved best model (robust: {best_robust:.2%})")
            
            scheduler.step()
        
        return history


class TRADESTrainer(AdversarialTrainer):
    """
    TRADES: Theoretically Principled Trade-off.
    
    Loss: L_CE(f(x), y) + β · KL(f(x) || f(x_adv))
    """
    
    def __init__(self, *args, beta: float = 6.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer
    ) -> Dict[str, float]:
        """TRADES training epoch."""
        self.model.train()
        
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='TRADES Training')
        for x, y in pbar:
            x, y = x.to(self.device), y.to(self.device)
            
            # Clean forward pass
            logits_clean = self.model(x)
            loss_natural = F.cross_entropy(logits_clean, y)
            
            # Generate adversarial examples (maximize KL)
            x_adv = x.clone().detach() + torch.empty_like(x).uniform_(-self.epsilon, self.epsilon)
            x_adv = torch.clamp(x_adv, 0, 1)
            
            for _ in range(self.num_iter):
                x_adv.requires_grad_(True)
                with torch.no_grad():
                    p_clean = F.softmax(logits_clean, dim=1)
                
                loss_kl = F.kl_div(
                    F.log_softmax(self.model(x_adv), dim=1),
                    p_clean,
                    reduction='batchmean'
                )
                
                self.model.zero_grad()
                loss_kl.backward()
                
                with torch.no_grad():
                    x_adv = x_adv + self.alpha * x_adv.grad.sign()
                    delta = torch.clamp(x_adv - x, -self.epsilon, self.epsilon)
                    x_adv = torch.clamp(x + delta, 0, 1)
            
            # TRADES loss
            logits_adv = self.model(x_adv)
            loss_robust = F.kl_div(
                F.log_softmax(logits_adv, dim=1),
                F.softmax(logits_clean.detach(), dim=1),
                reduction='batchmean'
            )
            
            loss = loss_natural + self.beta * loss_robust
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(y)
            correct += (logits_clean.argmax(1) == y).sum().item()
            total += len(y)
            
            pbar.set_postfix({'loss': f'{total_loss/total:.4f}'})
        
        return {'loss': total_loss / total, 'accuracy': correct / total}
```

### Usage Example

```python
import torchvision
import torchvision.transforms as transforms

# Load CIFAR-10
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

# Create model
model = torchvision.models.resnet18(num_classes=10)

# Standard adversarial training
trainer = AdversarialTrainer(model, epsilon=8/255)
history = trainer.train(train_loader, test_loader, epochs=100, save_path='robust_model.pth')

# Or use TRADES
trainer_trades = TRADESTrainer(model, epsilon=8/255, beta=6.0)
history = trainer_trades.train(train_loader, test_loader, epochs=100)
```

## Practical Considerations

### Hyperparameters

| Parameter | Standard AT | TRADES | Notes |
|-----------|-------------|--------|-------|
| $\varepsilon$ | 8/255 | 8/255 | Perturbation budget |
| $\alpha$ | 2/255 | 2/255 | Step size |
| $K$ (train) | 7-10 | 10 | PGD iterations |
| $K$ (eval) | 20-100 | 20-100 | Stronger for evaluation |
| $\beta$ | N/A | 1-6 | TRADES trade-off |
| Learning rate | 0.1 | 0.1 | With decay |
| Epochs | 100-200 | 100-200 | More than standard training |

### Computational Cost

Adversarial training is **7-10× slower** than standard training:
- Each batch requires $K$ forward-backward passes for PGD
- Typical: 10 PGD steps means 10× more gradient computations

### Common Issues

1. **Catastrophic overfitting**: Robust accuracy suddenly drops
   - Solution: Monitor robust accuracy, use early stopping

2. **Clean-robust trade-off**: Robust models have lower clean accuracy
   - Expected: ~5-15% drop in clean accuracy
   - Use TRADES to control trade-off

3. **Overfitting to training attacks**: Model robust to PGD but not others
   - Solution: Evaluate with multiple attacks (AutoAttack)

### Expected Results (CIFAR-10)

| Method | Clean Acc | Robust Acc (PGD-20) |
|--------|-----------|---------------------|
| Standard training | 95% | 0% |
| Standard AT | 85% | 48% |
| TRADES (β=6) | 87% | 46% |
| State-of-art | 90% | 60% |

## Summary

| Method | Formula | Key Feature |
|--------|---------|-------------|
| **Standard AT** | $\min_\theta \mathbb{E}[\max_\delta \mathcal{L}(f(\mathbf{x}+\boldsymbol{\delta}), y)]$ | Max-loss on adversarial |
| **TRADES** | $\mathcal{L}_{\text{CE}} + \beta \cdot \text{KL}$ | Explicit accuracy trade-off |
| **MART** | Weighted by $(1-p_y)$ | Focus on hard examples |

Adversarial training remains the gold standard for achieving robust models, despite the computational overhead and accuracy trade-off.

## References

1. Madry, A., et al. (2018). "Towards Deep Learning Models Resistant to Adversarial Attacks." ICLR.
2. Zhang, H., et al. (2019). "Theoretically Principled Trade-off between Robustness and Accuracy." ICML.
3. Wang, Y., et al. (2020). "Improving Adversarial Robustness Requires Revisiting Misclassified Examples." ICLR.
