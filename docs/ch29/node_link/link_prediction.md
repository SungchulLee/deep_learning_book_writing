# Link Prediction

Many real-world graphs are incomplete: social networks lack undiscovered
friendships, knowledge graphs miss relations, and financial networks have
unobserved transactions.  Link prediction addresses this by scoring how
likely an edge is to exist between two nodes, given the observed graph
structure and node features.

## Problem Formulation

Given a graph $G = (V, E)$ with node features $X$, the goal is to learn a
scoring function

$$
s(u, v) = f(\mathbf{h}_u, \mathbf{h}_v) \in \mathbb{R}
$$

where $\mathbf{h}_u, \mathbf{h}_v$ are learned node embeddings (typically
from a GNN), and $f$ outputs a high score when the edge $(u, v)$ is likely
to exist.

## Scoring Functions

### Dot Product

The simplest approach:

$$
s(u, v) = \mathbf{h}_u^{\top} \mathbf{h}_v
$$

Efficient to compute but assumes symmetric relationships.

### Bilinear (DistMult)

Introduce a learnable diagonal matrix:

$$
s(u, v) = \mathbf{h}_u^{\top} \text{diag}(\mathbf{r}) \, \mathbf{h}_v
$$

where $\mathbf{r}$ is a relation-specific parameter vector.  This is
widely used in knowledge graph completion.

### MLP Decoder

Concatenate embeddings and pass through a neural network:

$$
s(u, v) = \text{MLP}([\mathbf{h}_u \| \mathbf{h}_v])
$$

More expressive than dot product but slower at inference.

### TransE (Translation-Based)

Model edges as translations in embedding space:

$$
s(u, v) = -\|\mathbf{h}_u + \mathbf{r} - \mathbf{h}_v\|
$$

where $\mathbf{r}$ is a relation-specific translation vector.  Higher
scores (closer to zero) indicate more likely edges.

## Training

### Negative Sampling

For each observed edge $(u, v) \in E$ (positive sample), generate $k$
negative samples $(u, v')$ where $(u, v') \notin E$ by randomly selecting
$v'$.  The loss encourages positive edges to score higher than negatives.

### Loss Functions

**Binary cross-entropy:**

$$
\mathcal{L} = -\sum_{(u,v) \in E} \log \sigma(s(u,v)) - \sum_{(u,v') \notin E} \log(1 - \sigma(s(u,v')))
$$

**Margin-based (hinge) loss:**

$$
\mathcal{L} = \sum_{(u,v) \in E} \sum_{(u,v') \notin E} \max\bigl(0, \, \gamma - s(u,v) + s(u,v')\bigr)
$$

where $\gamma > 0$ is a margin hyperparameter.

## Evaluation Metrics

| Metric | Description |
|---|---|
| AUC-ROC | Area under the ROC curve; measures ranking quality |
| Average Precision | Area under precision-recall curve; robust to class imbalance |
| Hits@K | Fraction of true edges ranked in the top $K$ |
| MRR | Mean reciprocal rank of true edges |

!!! note "Data Splitting"
    Link prediction typically uses a temporal or random split: training edges
    form the observed graph, and the model predicts held-out edges.  Message
    edges (used for GNN computation) and supervision edges (used for loss)
    should be handled carefully to avoid data leakage.

## Implementation

```python
"""
Link prediction with GNN embeddings and dot-product scoring.

Demonstrates training with negative sampling and BCE loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# === Simple GNN Encoder ===
class GCNEncoder(nn.Module):
    """Two-layer GCN producing node embeddings."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.w1 = nn.Linear(in_dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # Simplified GCN: A * X * W
        h = F.relu(self.w1(adj @ x))
        h = self.w2(adj @ h)
        return h


# === Link Predictor ===
class LinkPredictor(nn.Module):
    """Dot-product link prediction model."""

    def __init__(self, in_dim: int, hidden_dim: int, emb_dim: int):
        super().__init__()
        self.encoder = GCNEncoder(in_dim, hidden_dim, emb_dim)

    def encode(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, adj)

    def decode(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        return (z[src] * z[dst]).sum(dim=1)

    def forward(
        self, x: torch.Tensor, adj: torch.Tensor,
        pos_edges: torch.Tensor, neg_edges: torch.Tensor
    ) -> torch.Tensor:
        z = self.encode(x, adj)
        pos_scores = self.decode(z, pos_edges)
        neg_scores = self.decode(z, neg_edges)
        scores = torch.cat([pos_scores, neg_scores])
        labels = torch.cat([
            torch.ones_like(pos_scores),
            torch.zeros_like(neg_scores),
        ])
        return F.binary_cross_entropy_with_logits(scores, labels)


# === Example ===
if __name__ == "__main__":
    torch.manual_seed(42)
    n, d = 6, 4
    x = torch.randn(n, d)
    adj = torch.eye(n)
    adj[0, 1] = adj[1, 0] = 1
    adj[1, 2] = adj[2, 1] = 1
    adj[2, 3] = adj[3, 2] = 1

    model = LinkPredictor(in_dim=d, hidden_dim=8, emb_dim=4)
    pos_edges = torch.tensor([[0, 1, 2], [1, 2, 3]])
    neg_edges = torch.tensor([[0, 3, 4], [4, 5, 5]])

    loss = model(x, adj, pos_edges, neg_edges)
    print(f"Loss: {loss.item():.4f}")
```

## Reference

- Zhang, M. & Chen, Y. "Link Prediction Based on Graph Neural Networks."
  NeurIPS 2018.
- Bordes, A. et al. "Translating Embeddings for Modeling Multi-relational
  Data." NeurIPS 2013.
