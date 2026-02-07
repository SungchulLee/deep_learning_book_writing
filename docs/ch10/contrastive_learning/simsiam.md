# SimSiam: Simple Siamese Representation Learning

SimSiam (Simple Siamese) is the simplest non-contrastive self-supervised learning method. It requires no negative samples, no momentum encoder, and no large batches—just a stop-gradient operation.

## Key Insight

SimSiam shows that the stop-gradient operation alone is sufficient to prevent collapse. The predictor creates asymmetry between the two branches, and stop-gradient on the target branch forces the predictor to learn meaningful representations rather than trivial solutions.

## Architecture

```
Branch 1: Input → Encoder → Projector → Predictor → p₁
Branch 2: Input → Encoder → Projector → z₂ (stop-gradient)

Loss: -cos(p₁, stopgrad(z₂)) - cos(p₂, stopgrad(z₁))
```

## Implementation

class SimSiam(nn.Module):
    """
    SimSiam: Exploring Simple Siamese Representation Learning
    
    Even simpler than BYOL:
    - No momentum encoder
    - No target network  
    - Just stop-gradient!
    
    The key insight: stop-gradient prevents collapse by creating
    an implicit moving target.
    
    Args:
        base_encoder: Backbone architecture
        projection_dim: Dimension of projection
        prediction_dim: Hidden dimension of predictor
    """
    def __init__(
        self,
        base_encoder='resnet50',
        projection_dim=2048,
        prediction_dim=512
    ):
        super().__init__()
        
        if base_encoder == 'resnet50':
            self.encoder = models.resnet50(weights=None)
            encoder_dim = 2048
            self.encoder.fc = nn.Identity()
        else:
            raise ValueError(f"Unknown encoder: {base_encoder}")
        
        self.encoder_dim = encoder_dim
        
        # Projector: 3-layer MLP with output BN
        self.projector = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.BatchNorm1d(encoder_dim),
            nn.ReLU(inplace=True),
            nn.Linear(encoder_dim, encoder_dim),
            nn.BatchNorm1d(encoder_dim),
            nn.ReLU(inplace=True),
            nn.Linear(encoder_dim, projection_dim),
            nn.BatchNorm1d(projection_dim, affine=False)
        )
        
        # Predictor: 2-layer MLP with bottleneck
        self.predictor = nn.Sequential(
            nn.Linear(projection_dim, prediction_dim),
            nn.BatchNorm1d(prediction_dim),
            nn.ReLU(inplace=True),
            nn.Linear(prediction_dim, projection_dim)
        )
    
    def forward(self, x1, x2):
        """
        Forward pass with stop-gradient.
        
        CRITICAL: z.detach() is what prevents collapse!
        """
        z1 = self.projector(self.encoder(x1))
        z2 = self.projector(self.encoder(x2))
        
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        
        # Stop-gradient on z (target)
        loss = self.cosine_loss(p1, z2.detach())
        loss += self.cosine_loss(p2, z1.detach())
        return loss / 2
    
    def cosine_loss(self, p, z):
        """Negative cosine similarity."""
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)
        return -(p * z).sum(dim=1).mean()
    
    def get_representation(self, x):
        return self.encoder(x)




## Collapse Analysis

class CollapseAnalysis:
    """Tools for understanding why BYOL/SimSiam don't collapse."""
    
    @staticmethod
    def check_collapse(model, dataloader, device, num_batches=10):
        """
        Check if representations have collapsed.
        
        Signs of collapse:
        - Low variance in representations
        - High cosine similarity between random samples
        """
        model.eval()
        representations = []
        
        with torch.no_grad():
            for i, (x, _) in enumerate(dataloader):
                if i >= num_batches:
                    break
                x = x.to(device)
                rep = model.get_representation(x)
                representations.append(rep)
        
        reps = torch.cat(representations, dim=0)
        
        # Metrics
        std_rep = reps.std(dim=0).mean()
        
        reps_norm = F.normalize(reps, dim=1)
        n = min(1000, reps.shape[0])
        idx1 = torch.randperm(reps.shape[0])[:n]
        idx2 = torch.randperm(reps.shape[0])[:n]
        random_cos_sim = (reps_norm[idx1] * reps_norm[idx2]).sum(dim=1).mean()
        
        print(f"Collapse Analysis:")
        print(f"  Representation std: {std_rep.item():.4f}")
        print(f"  Random pair cos sim: {random_cos_sim.item():.4f}")
        
        if std_rep < 0.1:
            print("  ⚠️ WARNING: Low variance - possible collapse!")
        if random_cos_sim > 0.9:
            print("  ⚠️ WARNING: High similarity - possible collapse!")
        
        return {'std': std_rep.item(), 'random_cos_sim': random_cos_sim.item()}




## Why Stop-Gradient Prevents Collapse

```
Without stop-gradient (COLLAPSES):
  ∂L/∂θ = ∂L/∂p · ∂p/∂θ + ∂L/∂z · ∂z/∂θ
  → Both branches push toward trivial solution f(x) = 0

With stop-gradient (WORKS):
  ∂L/∂θ = ∂L/∂p · ∂p/∂θ + 0
  → Only predictor branch optimized
  → Target z provides a "moving target"
  → Predictor must learn to PREDICT, not just COPY
```

## Comparison with BYOL

| Aspect | BYOL | SimSiam |
|--------|------|---------|
| Target network | Momentum EMA | Stop-gradient |
| Momentum | 0.996 → 1.0 | N/A |
| Parameters | 2× encoder | 1× encoder |
| Memory | Higher | Lower |
| Batch size | Less sensitive | Requires larger |
| Performance | Slightly higher | Competitive |

## Summary

SimSiam demonstrates that the essential ingredients for non-contrastive SSL are remarkably simple: a Siamese architecture with a predictor and stop-gradient. No momentum, no negative pairs, no large batches.

## References

1. Chen, X., & He, K. (2021). "Exploring Simple Siamese Representation Learning." *CVPR*.
2. Tian, Y., et al. (2021). "Understanding Self-Supervised Learning Dynamics without Contrastive Pairs." *ICML*.
