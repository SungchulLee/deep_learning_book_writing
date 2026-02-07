# Barlow Twins

Barlow Twins learns representations by making the cross-correlation matrix between the embeddings of two distorted versions of a sample close to the identity matrix. This avoids both negative samples and asymmetric architectures.

## Objective

Given two views $z_A$ and $z_B$ of the same batch, the Barlow Twins loss operates on the cross-correlation matrix $\mathcal{C}$:

$$\mathcal{C}_{ij} = \frac{\sum_b z_{b,i}^A \, z_{b,j}^B}{\sqrt{\sum_b (z_{b,i}^A)^2} \sqrt{\sum_b (z_{b,j}^B)^2}}$$

$$\mathcal{L}_{\text{BT}} = \underbrace{\sum_i (1 - \mathcal{C}_{ii})^2}_{\text{invariance}} + \lambda \underbrace{\sum_i \sum_{j \neq i} \mathcal{C}_{ij}^2}_{\text{redundancy reduction}}$$

The first term pushes diagonal elements toward 1 (representations of the same image should agree), and the second term pushes off-diagonal elements toward 0 (different dimensions should be decorrelated).

## Implementation

```python
import torch
import torch.nn as nn
from torchvision import models


class BarlowTwins(nn.Module):
    """
    Barlow Twins: Self-Supervised Learning via Redundancy Reduction.
    
    Args:
        base_encoder: Backbone architecture
        projection_dim: Output dimension of projector
        lambd: Weight on redundancy reduction term
    """
    def __init__(self, base_encoder='resnet50', projection_dim=8192, lambd=0.0051):
        super().__init__()
        self.lambd = lambd
        
        self.encoder = models.resnet50(weights=None)
        encoder_dim = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()
        
        self.projector = nn.Sequential(
            nn.Linear(encoder_dim, 8192),
            nn.BatchNorm1d(8192),
            nn.ReLU(inplace=True),
            nn.Linear(8192, 8192),
            nn.BatchNorm1d(8192),
            nn.ReLU(inplace=True),
            nn.Linear(8192, projection_dim),
        )
        
        self.bn = nn.BatchNorm1d(projection_dim, affine=False)
    
    def forward(self, x1, x2):
        z1 = self.projector(self.encoder(x1))
        z2 = self.projector(self.encoder(x2))
        
        z1_norm = self.bn(z1)
        z2_norm = self.bn(z2)
        
        batch_size = z1.shape[0]
        
        # Cross-correlation matrix (D × D)
        c = (z1_norm.T @ z2_norm) / batch_size
        
        on_diag = (1 - torch.diagonal(c)).pow(2).sum()
        off_diag = self._off_diagonal(c).pow(2).sum()
        
        loss = on_diag + self.lambd * off_diag
        return loss
    
    @staticmethod
    def _off_diagonal(x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
```

## Comparison with Other Methods

| Method | Negatives | Momentum | Batch sensitivity | Key mechanism |
|--------|-----------|----------|------------------|---------------|
| SimCLR | ✅ Required | ❌ | High | Large-batch negatives |
| MoCo | ✅ Queue | ✅ | Low | Momentum queue |
| BYOL | ❌ | ✅ | Low | Predictor + EMA |
| SimSiam | ❌ | ❌ | Moderate | Stop-gradient |
| Barlow Twins | ❌ | ❌ | Low | Cross-correlation |

## References

1. Zbontar, J., et al. (2021). "Barlow Twins: Self-Supervised Learning via Redundancy Reduction." *ICML*.
