# Joint Energy Models

## Learning Objectives

After completing this section, you will be able to:

1. Understand how classifiers implicitly define energy-based models
2. Derive the JEM framework unifying classification and generation
3. Implement JEM training combining discriminative and generative objectives
4. Apply JEM for calibrated classification, OOD detection, and adversarial robustness

## Introduction

Joint Energy Models (JEM), introduced by Grathwohl et al. (2020) in "Your Classifier is Secretly an Energy Based Model," represent one of the most elegant unifications in modern machine learning. The key insight is that any classifier that outputs logits $f_\theta(x) \in \mathbb{R}^K$ implicitly defines an energy function over inputs:

$$E_\theta(x) = -\text{LogSumExp}(f_\theta(x)) = -\log \sum_{k=1}^K \exp(f_\theta(x)_k)$$

This means every trained classifier already has a built-in generative model, density estimator, and anomaly detector—it just needs to be trained to use these capabilities.

## Mathematical Framework

### From Logits to Energy

Given a neural network $f_\theta: \mathbb{R}^d \to \mathbb{R}^K$ that outputs logits for $K$ classes, we define:

**Joint distribution**: $p_\theta(x, y) \propto \exp(f_\theta(x)_y)$

**Marginal over labels**: $p_\theta(x) = \sum_y p_\theta(x, y) \propto \sum_y \exp(f_\theta(x)_y)$

**Conditional (classifier)**: $p_\theta(y|x) = \frac{\exp(f_\theta(x)_y)}{\sum_{y'} \exp(f_\theta(x)_{y'})} = \text{softmax}(f_\theta(x))_y$

**Marginal energy**: $E_\theta(x) = -\log \sum_y \exp(f_\theta(x)_y) = -\text{LogSumExp}(f_\theta(x))$

The classifier $p_\theta(y|x)$ is simply the standard softmax—exactly what we already use for classification. The novelty is recognizing that the unnormalized density $p_\theta(x) \propto \exp(-E_\theta(x))$ provides a generative model "for free."

### Class-Conditional Energies

Each logit $f_\theta(x)_y$ serves as the negative class-conditional energy:

$$E_\theta(x, y) = -f_\theta(x)_y$$

This means the classifier's logit for class $y$ measures how compatible input $x$ is with class $y$—exactly the energy interpretation.

## Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class JointEnergyModel(nn.Module):
    """
    Joint Energy-Based Model that unifies classification and generation.
    
    Any classifier backbone can be used — the JEM framework just
    reinterprets the logits as (negative) energies.
    """
    
    def __init__(self, backbone: nn.Module, n_classes: int):
        super().__init__()
        self.backbone = backbone
        self.n_classes = n_classes
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits for classification."""
        return self.backbone(x)
    
    def energy(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        """
        Compute energy.
        
        Parameters
        ----------
        x : torch.Tensor
            Input data
        y : torch.Tensor, optional
            Class labels. If None, returns marginal energy E(x).
            If provided, returns class-conditional energy E(x, y).
        """
        logits = self.forward(x)
        
        if y is None:
            # Marginal energy: E(x) = -logsumexp(logits)
            return -torch.logsumexp(logits, dim=1)
        else:
            # Class-conditional energy: E(x, y) = -logit_y
            return -logits.gather(1, y.unsqueeze(1)).squeeze(1)
    
    def classify(self, x: torch.Tensor) -> torch.Tensor:
        """Standard classification via softmax."""
        return F.softmax(self.forward(x), dim=1)
    
    def ood_score(self, x: torch.Tensor) -> torch.Tensor:
        """OOD score: higher energy = more likely OOD."""
        return self.energy(x)


class SimpleBackbone(nn.Module):
    """Simple backbone for demonstration."""
    
    def __init__(self, input_dim, hidden_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_classes)
        )
    
    def forward(self, x):
        return self.net(x)


def train_jem(model, labeled_data, labels, n_epochs=100,
              lr=1e-3, langevin_steps=20, langevin_lr=0.01,
              gen_weight=1.0):
    """
    Train JEM with combined classification and generation objectives.
    
    Loss = L_classification + λ * L_generation
    
    L_classification = Cross-entropy
    L_generation = E(x_data) - E(x_samples)  (contrastive)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    buffer = []
    
    metrics = {'cls_loss': [], 'gen_loss': [], 'accuracy': []}
    
    for epoch in range(n_epochs):
        perm = torch.randperm(len(labeled_data))
        batch_size = 64
        epoch_cls, epoch_gen = 0, 0
        correct, total = 0, 0
        n_batches = 0
        
        for i in range(0, len(labeled_data), batch_size):
            x = labeled_data[perm[i:i+batch_size]]
            y = labels[perm[i:i+batch_size]]
            
            # === Classification loss ===
            logits = model(x)
            cls_loss = F.cross_entropy(logits, y)
            
            # Accuracy
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += len(y)
            
            # === Generation loss (contrastive) ===
            # Positive: energy on data
            pos_energy = model.energy(x).mean()
            
            # Negative: energy on Langevin samples
            if buffer and torch.rand(1) > 0.05:
                indices = torch.randint(len(buffer), (x.shape[0],))
                init = torch.stack([buffer[idx] for idx in indices])
            else:
                init = torch.randn_like(x)
            
            # Langevin dynamics
            neg_x = init.clone()
            for _ in range(langevin_steps):
                neg_x.requires_grad_(True)
                energy = model.energy(neg_x).sum()
                grad = torch.autograd.grad(energy, neg_x)[0]
                neg_x = neg_x.detach() - langevin_lr * grad
                neg_x = neg_x + 0.005 * torch.randn_like(neg_x)
            
            neg_energy = model.energy(neg_x.detach()).mean()
            gen_loss = pos_energy - neg_energy
            
            # Update buffer
            for s in neg_x.detach():
                buffer.append(s.cpu())
            buffer = buffer[-5000:]
            
            # === Combined loss ===
            loss = cls_loss + gen_weight * gen_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_cls += cls_loss.item()
            epoch_gen += gen_loss.item()
            n_batches += 1
        
        metrics['cls_loss'].append(epoch_cls / n_batches)
        metrics['gen_loss'].append(epoch_gen / n_batches)
        metrics['accuracy'].append(correct / total)
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}: cls_loss={epoch_cls/n_batches:.4f}, "
                  f"gen_loss={epoch_gen/n_batches:.4f}, "
                  f"acc={correct/total:.3f}")
    
    return metrics
```

## Applications

### Out-of-Distribution Detection

JEM provides OOD detection as a natural byproduct of the energy function:

```python
def jem_ood_detection(model, in_dist_data, ood_data):
    """
    OOD detection using JEM energy.
    
    In-distribution data → low energy
    Out-of-distribution data → high energy
    """
    from sklearn.metrics import roc_auc_score
    
    with torch.no_grad():
        in_energy = model.energy(in_dist_data).numpy()
        ood_energy = model.energy(ood_data).numpy()
    
    labels = np.concatenate([
        np.zeros(len(in_energy)),
        np.ones(len(ood_energy))
    ])
    scores = np.concatenate([in_energy, ood_energy])
    
    auroc = roc_auc_score(labels, scores)
    
    print(f"AUROC: {auroc:.4f}")
    print(f"In-dist energy: {in_energy.mean():.2f} ± {in_energy.std():.2f}")
    print(f"OOD energy: {ood_energy.mean():.2f} ± {ood_energy.std():.2f}")
    
    return auroc
```

### Adversarial Purification

JEM can defend against adversarial attacks by "purifying" adversarial examples back to the data manifold via energy minimization:

```python
def adversarial_purification(model, adv_images, n_steps=50,
                             step_size=0.01):
    """
    Purify adversarial examples via energy minimization.
    
    Idea: adversarial perturbations push inputs to high-energy
    regions. Energy minimization pushes them back to the data manifold.
    """
    x = adv_images.clone()
    
    for _ in range(n_steps):
        x.requires_grad_(True)
        energy = model.energy(x)
        grad = torch.autograd.grad(energy.sum(), x)[0]
        
        # Move toward lower energy
        x = x.detach() - step_size * grad
        x = x.clamp(0, 1)
    
    return x
```

## Key Takeaways

!!! success "Core Concepts"
    1. Any classifier with logit outputs implicitly defines an energy function via $E(x) = -\text{LogSumExp}(f(x))$
    2. JEM training adds a generative loss to the standard classification loss, teaching the model to use its energy function
    3. The resulting model simultaneously classifies, detects OOD inputs, generates samples, and provides calibrated uncertainties
    4. OOD detection comes "for free"—in-distribution inputs have low energy, OOD inputs have high energy
    5. Adversarial purification uses energy minimization to push adversarial examples back to the data manifold

!!! tip "Practical Benefit"
    JEM demonstrates that generative and discriminative objectives are complementary, not competing. Adding generative training often improves classification calibration and robustness, making JEM attractive even when generation quality is not the primary goal.

## Exercises

1. **JEM on MNIST**: Train a JEM classifier on MNIST. Evaluate classification accuracy, then test OOD detection using Fashion-MNIST as the OOD dataset.

2. **Energy calibration**: Plot histograms of energy values for correctly classified, misclassified, and OOD inputs. Is energy a good predictor of prediction reliability?

3. **Generative weight**: Sweep the generative weight $\lambda$ from 0 (pure classifier) to 10 (generation-dominated). How does classification accuracy change? How does OOD detection change?

## References

- Grathwohl, W., et al. (2020). Your Classifier is Secretly an Energy Based Model and You Should Treat It Like One. *ICLR*.
- Liu, W., et al. (2021). Energy-Based Out-of-Distribution Detection. *NeurIPS*.
- Hill, M., et al. (2022). Stochastic Security: Adversarial Defense Using Long-Run Dynamics of Energy-Based Models. *ICLR*.
