# Fair Representation Learning

## Overview

**Fair Representation Learning** (Zemel et al., 2013) learns a latent representation $Z = g(X)$ that preserves information useful for prediction while removing information about the protected attribute. The representation is then used as input to any downstream classifier.

## Mathematical Formulation

The goal is to learn an encoder $g: \mathcal{X} \to \mathcal{Z}$ such that:

1. **Utility**: $Z$ retains predictive information about $Y$: $I(Z; Y)$ is maximized
2. **Fairness**: $Z$ removes information about $A$: $I(Z; A)$ is minimized

This is formalized as:

$$\min_g \; \underbrace{-I(Z; Y)}_{\text{prediction loss}} + \lambda \underbrace{I(Z; A)}_{\text{fairness penalty}}$$

## PyTorch Implementation

```python
import torch
import torch.nn as nn
from typing import Dict, Tuple

class FairAutoencoder(nn.Module):
    """
    Autoencoder that learns fair representations by jointly:
    1. Reconstructing inputs (utility)
    2. Predicting labels (task performance)
    3. Preventing an adversary from predicting protected attributes (fairness)
    """
    
    def __init__(self, input_dim: int, latent_dim: int = 16, hidden_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1), nn.Sigmoid(),
        )
        self.adversary = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1), nn.Sigmoid(),
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        z = self.encode(x)
        x_recon = self.decoder(z)
        y_pred = self.predictor(z).squeeze(-1)
        a_pred = self.adversary(z).squeeze(-1)
        return z, x_recon, y_pred, a_pred


def train_fair_representation(
    model: FairAutoencoder,
    X: torch.Tensor,
    y: torch.Tensor,
    A: torch.Tensor,
    epochs: int = 200,
    fairness_weight: float = 1.0,
    recon_weight: float = 0.5,
    lr: float = 0.001,
) -> Dict[str, list]:
    """Train fair autoencoder with adversarial debiasing on the representation."""
    
    # Separate optimizers for encoder/decoder/predictor vs adversary
    enc_params = (
        list(model.encoder.parameters()) +
        list(model.decoder.parameters()) +
        list(model.predictor.parameters())
    )
    enc_optimizer = torch.optim.Adam(enc_params, lr=lr)
    adv_optimizer = torch.optim.Adam(model.adversary.parameters(), lr=lr)
    
    history = {'task_loss': [], 'adv_loss': []}
    
    for epoch in range(epochs):
        model.train()
        
        # Step 1: Train adversary to predict A from Z
        z = model.encode(X).detach()
        a_pred = model.adversary(z).squeeze(-1)
        adv_loss = nn.functional.binary_cross_entropy(a_pred, A.float())
        adv_optimizer.zero_grad()
        adv_loss.backward()
        adv_optimizer.step()
        
        # Step 2: Train encoder to fool adversary + predict Y + reconstruct X
        z, x_recon, y_pred, a_pred = model(X)
        task_loss = nn.functional.binary_cross_entropy(y_pred, y.float())
        recon_loss = nn.functional.mse_loss(x_recon, X)
        adv_fool = nn.functional.binary_cross_entropy(a_pred, A.float())
        
        # Minimize task + recon, MAXIMIZE adversary loss (gradient reversal)
        total = task_loss + recon_weight * recon_loss - fairness_weight * adv_fool
        enc_optimizer.zero_grad()
        total.backward()
        enc_optimizer.step()
        
        history['task_loss'].append(task_loss.item())
        history['adv_loss'].append(adv_loss.item())
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}: task={task_loss.item():.4f}, "
                  f"adv={adv_loss.item():.4f}")
    
    return history
```

## Summary

- Learns **latent representations** that are informative for prediction but invariant to group membership
- Uses **adversarial training** to remove protected information from the representation
- The representation can be used with **any downstream model**
- Tradeoff controlled by $\lambda$: higher values produce fairer but potentially less useful representations

## Next Steps

- [Data Augmentation](augmentation.md): Augmentation strategies for fairness
- [Adversarial Debiasing](../inprocessing/adversarial.md): Adversarial approach applied directly during model training
