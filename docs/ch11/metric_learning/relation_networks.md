# Relation Networks

Relation Networks (Sung et al., 2018) replace fixed distance metrics with a **learned similarity function**. Instead of using cosine similarity or Euclidean distance, a neural network learns to compare query and support embeddings.

## Key Idea

Rather than computing $d(f(x_q), f(x_s))$ with a fixed metric, Relation Networks learn:

$$r_{i,j} = g_\phi(\text{concat}(f_\theta(x_i), f_\theta(x_j)))$$

where $g_\phi$ is a learnable relation module that outputs a similarity score in $[0, 1]$.

## Architecture

```
Support image → Embedding CNN → support embedding ─┐
                                                    ├─ Concatenate → Relation Module → score ∈ [0,1]
Query image   → Embedding CNN → query embedding  ──┘
```

## Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingModule(nn.Module):
    """CNN encoder for feature extraction."""
    
    def __init__(self, in_channels=3):
        super().__init__()
        self.encoder = nn.Sequential(
            self._conv_block(in_channels, 64),
            self._conv_block(64, 64),
            self._conv_block(64, 64),
            self._conv_block(64, 64),
        )
    
    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
    
    def forward(self, x):
        return self.encoder(x)


class RelationModule(nn.Module):
    """Learned similarity function."""
    
    def __init__(self, input_channels=128, hidden_size=8):
        super().__init__()
        self.relation = nn.Sequential(
            self._conv_block(input_channels, 64),
            self._conv_block(64, 64),
            nn.Flatten(),
            nn.Linear(64 * hidden_size * hidden_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output similarity score in [0, 1]
        )
    
    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
    
    def forward(self, x):
        return self.relation(x).squeeze(-1)


class RelationNetwork(nn.Module):
    """
    Learning to Compare: Relation Network for Few-Shot Learning.
    
    Uses MSE loss instead of cross-entropy since the relation
    module outputs a continuous similarity score.
    """
    
    def __init__(self, in_channels=3, feature_channels=64):
        super().__init__()
        self.embedding = EmbeddingModule(in_channels)
        self.relation = RelationModule(input_channels=feature_channels * 2)
    
    def forward(self, support_x, support_y, query_x, n_way, k_shot):
        """
        Args:
            support_x: (N*K, C, H, W)
            support_y: (N*K,)
            query_x: (N*Q, C, H, W)
            n_way: number of classes
            k_shot: shots per class
        
        Returns:
            relations: (N*Q, N) similarity scores
        """
        # Encode
        support_features = self.embedding(support_x)  # (N*K, D, h, w)
        query_features = self.embedding(query_x)      # (N*Q, D, h, w)
        
        # Compute class prototypes by averaging support features
        _, D, h, w = support_features.shape
        support_features = support_features.view(n_way, k_shot, D, h, w)
        prototypes = support_features.mean(dim=1)  # (N, D, h, w)
        
        # Compare each query with each prototype
        n_query = query_features.size(0)
        
        # Expand for pairwise comparison
        query_expanded = query_features.unsqueeze(1).expand(-1, n_way, -1, -1, -1)
        proto_expanded = prototypes.unsqueeze(0).expand(n_query, -1, -1, -1, -1)
        
        # Concatenate along channel dimension
        pairs = torch.cat([query_expanded, proto_expanded], dim=2)  # (N*Q, N, 2D, h, w)
        pairs = pairs.view(-1, 2 * D, h, w)
        
        # Compute relation scores
        relations = self.relation(pairs)  # (N*Q * N,)
        relations = relations.view(n_query, n_way)  # (N*Q, N)
        
        return relations
    
    def compute_loss(self, relations, query_y, n_way):
        """MSE loss against one-hot targets."""
        one_hot = F.one_hot(query_y, n_way).float()
        return F.mse_loss(relations, one_hot)
    
    def predict(self, support_x, support_y, query_x):
        n_way = support_y.unique().size(0)
        k_shot = support_x.size(0) // n_way
        relations = self.forward(support_x, support_y, query_x, n_way, k_shot)
        return relations.argmax(dim=1)
```

## Key Design Choices

| Choice | Rationale |
|--------|-----------|
| Learned similarity | More expressive than fixed metrics |
| MSE loss | Regression to similarity score (not classification) |
| Sigmoid output | Bounded score in [0, 1] |
| Concatenation | Simple feature combination |

## Comparison with Fixed-Metric Methods

| Method | Distance function | Learnable? | Loss |
|--------|------------------|-----------|------|
| Siamese | Euclidean / cosine | Fixed | Contrastive |
| Prototypical | Euclidean | Fixed | Cross-entropy |
| Matching | Cosine | Fixed | Cross-entropy |
| **Relation** | **Neural network** | **Yes** | **MSE** |

## References

1. Sung, F., et al. (2018). "Learning to Compare: Relation Network for Few-Shot Learning." *CVPR*.
