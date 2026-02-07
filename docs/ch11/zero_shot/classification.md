# Zero-Shot Classification

## Overview

Visual-semantic embedding models learn deep neural network mappings between visual features (from CNNs) and semantic representations (word embeddings). This section covers DeViSE, bilinear compatibility models, and advanced architectures for zero-shot learning.

## DeViSE: Deep Visual-Semantic Embedding

### Architecture Overview

DeViSE (Frome et al., 2013) learns to project CNN visual features into the semantic embedding space:

$$M: \mathbb{R}^{d_v} \rightarrow \mathbb{R}^{d_s}$$

such that:
$$M \cdot \mathbf{v}(\text{image}) \approx \mathbf{s}(\text{class\_name})$$

### Components

**Visual Encoder**: Pre-trained CNN (typically VGG, ResNet)
- Extract features from penultimate layer
- Frozen or fine-tuned during training

**Projection Network**: Learned transformation
- Linear or non-linear mapping
- Projects visual features to semantic space

**Semantic Embeddings**: Pre-trained word vectors
- Word2Vec, GloVe, or FastText
- Fixed during training (typically)

### Loss Function

DeViSE uses a margin-based ranking loss:

$$\mathcal{L} = \sum_{(\mathbf{x}, y)} \sum_{j \neq y} \max(0, \gamma - \mathbf{s}_y^\top \hat{\mathbf{v}} + \mathbf{s}_j^\top \hat{\mathbf{v}})$$

where:
- $\hat{\mathbf{v}} = M \cdot \phi(\mathbf{x})$ is the projected visual embedding
- $\gamma$ is the margin parameter
- The sum over $j \neq y$ can be approximated by sampling negatives

### PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DeViSE(nn.Module):
    """
    Deep Visual-Semantic Embedding Model (DeViSE).
    
    Projects CNN features into word embedding space.
    """
    
    def __init__(self, visual_dim: int, semantic_dim: int, 
                 embedding_dim: int = 300, dropout: float = 0.5):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Visual projection network
        self.visual_projection = nn.Sequential(
            nn.Linear(visual_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, embedding_dim)
        )
        
        # Semantic projection (optional)
        self.semantic_projection = nn.Linear(semantic_dim, embedding_dim)
        
        # Temperature for softmax (learnable)
        self.temperature = nn.Parameter(torch.ones(1))
    
    def project_visual(self, visual_features):
        """Project visual features to embedding space."""
        projected = self.visual_projection(visual_features)
        return F.normalize(projected, p=2, dim=1)
    
    def project_semantic(self, semantic_embeddings):
        """Project semantic embeddings to embedding space."""
        projected = self.semantic_projection(semantic_embeddings)
        return F.normalize(projected, p=2, dim=1)
    
    def forward(self, visual_features, semantic_pos, semantic_neg=None):
        """
        Forward pass for training.
        """
        v_proj = self.project_visual(visual_features)
        s_pos_proj = self.project_semantic(semantic_pos)
        
        pos_score = torch.sum(v_proj * s_pos_proj, dim=1)
        
        if semantic_neg is not None:
            s_neg_proj = self.project_semantic(semantic_neg)
            neg_score = torch.sum(v_proj * s_neg_proj, dim=1)
            return pos_score, neg_score
        
        return pos_score
    
    def predict(self, visual_features, class_embeddings, class_names):
        """Predict class for visual features."""
        self.eval()
        
        with torch.no_grad():
            v_proj = self.project_visual(visual_features)
            
            s_all = torch.stack([
                torch.tensor(class_embeddings[c], dtype=torch.float32)
                for c in class_names
            ])
            s_proj = self.project_semantic(s_all)
            
            scores = v_proj @ s_proj.T
            pred_indices = torch.argmax(scores, dim=1)
            predictions = [class_names[i] for i in pred_indices.numpy()]
        
        return predictions, scores
```

## Bilinear Compatibility Models

### Concept

Bilinear models capture pairwise interactions between visual and semantic dimensions:

$$F(\mathbf{v}, \mathbf{s}) = \mathbf{v}^\top W \mathbf{s}$$

where $W \in \mathbb{R}^{d_v \times d_s}$ is a learned weight matrix.

### Low-Rank Approximation

Reduce parameters with low-rank factorization:

$$W = U V^\top$$

This is equivalent to projecting both modalities to a lower dimension:

$$F(\mathbf{v}, \mathbf{s}) = (U^\top \mathbf{v})^\top (V^\top \mathbf{s})$$

### Implementation

```python
class BilinearCompatibility(nn.Module):
    """
    Bilinear compatibility model for ZSL.
    """
    
    def __init__(self, visual_dim: int, semantic_dim: int, 
                 hidden_dim: int = 512, use_low_rank: bool = True):
        super().__init__()
        
        self.use_low_rank = use_low_rank
        
        if use_low_rank:
            self.visual_proj = nn.Sequential(
                nn.Linear(visual_dim, hidden_dim * 2),
                nn.BatchNorm1d(hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
            
            self.semantic_proj = nn.Sequential(
                nn.Linear(semantic_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
        else:
            self.W = nn.Parameter(torch.randn(visual_dim, semantic_dim) * 0.01)
            self.visual_bn = nn.BatchNorm1d(visual_dim)
    
    def forward(self, visual, semantic):
        """Compute bilinear compatibility scores."""
        if self.use_low_rank:
            v_proj = F.normalize(self.visual_proj(visual), dim=1)
            s_proj = F.normalize(self.semantic_proj(semantic), dim=-1)
            
            if len(s_proj.shape) == 2 and s_proj.shape[0] != visual.shape[0]:
                scores = v_proj @ s_proj.T
            else:
                scores = torch.sum(v_proj * s_proj, dim=1)
        else:
            visual = self.visual_bn(visual)
            if len(semantic.shape) == 2 and semantic.shape[0] != visual.shape[0]:
                scores = visual @ self.W @ semantic.T
            else:
                scores = torch.sum(visual @ self.W * semantic, dim=1)
        
        return scores
```

## Loss Functions

### Margin Ranking Loss

```python
class RankingLoss(nn.Module):
    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin
    
    def forward(self, pos_score, neg_score):
        loss = torch.clamp(self.margin - pos_score + neg_score, min=0)
        return loss.mean()
```

### Triplet Loss

```python
class TripletLoss(nn.Module):
    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        pos_dist = torch.sum((anchor - positive) ** 2, dim=1)
        neg_dist = torch.sum((anchor - negative) ** 2, dim=1)
        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0)
        return loss.mean()
```

### InfoNCE / Contrastive Loss

```python
def info_nce_loss(visual_emb, semantic_emb, temperature=0.07):
    """InfoNCE loss for contrastive learning."""
    v_norm = F.normalize(visual_emb, dim=1)
    s_norm = F.normalize(semantic_emb, dim=1)
    
    logits = v_norm @ s_norm.T / temperature
    labels = torch.arange(len(visual_emb), device=visual_emb.device)
    
    loss_v = F.cross_entropy(logits, labels)
    loss_s = F.cross_entropy(logits.T, labels)
    
    return (loss_v + loss_s) / 2
```

## Advanced Architectures

### Cross-Modal Attention

```python
class CrossModalAttention(nn.Module):
    """Cross-modal attention for visual-semantic alignment."""
    
    def __init__(self, visual_dim: int, semantic_dim: int, 
                 hidden_dim: int = 256, n_heads: int = 8):
        super().__init__()
        
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.semantic_proj = nn.Linear(semantic_dim, hidden_dim)
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=0.1,
            batch_first=True
        )
        
        self.output_proj = nn.Linear(hidden_dim, 1)
    
    def forward(self, visual, semantic):
        v_proj = self.visual_proj(visual)
        s_proj = self.semantic_proj(semantic)
        
        if len(v_proj.shape) == 2:
            v_proj = v_proj.unsqueeze(1)
        s_proj = s_proj.unsqueeze(1)
        
        attended, _ = self.attention(
            query=s_proj, key=v_proj, value=v_proj
        )
        
        score = self.output_proj(attended.squeeze(1))
        return score.squeeze(-1)
```

### Two-Branch Network

```python
class TwoBranchNetwork(nn.Module):
    """Two-branch network with separate encoders."""
    
    def __init__(self, visual_dim: int, semantic_dim: int, 
                 embedding_dim: int = 512):
        super().__init__()
        
        self.visual_branch = nn.Sequential(
            nn.Linear(visual_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, embedding_dim)
        )
        
        self.semantic_branch = nn.Sequential(
            nn.Linear(semantic_dim, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )
    
    def forward(self, visual, semantic):
        v_emb = F.normalize(self.visual_branch(visual), dim=1)
        s_emb = F.normalize(self.semantic_branch(semantic), dim=-1)
        
        if len(s_emb.shape) == 2 and s_emb.shape[0] != visual.shape[0]:
            return v_emb @ s_emb.T
        else:
            return torch.sum(v_emb * s_emb, dim=1)
```

## Training Best Practices

### Hard Negative Mining

Select negatives that are most confusing to the model:

```python
def hard_negative_mining(model, visual, positive_idx, all_embeddings, seen_classes):
    """Select hardest negative for each sample."""
    with torch.no_grad():
        scores = model(visual, all_embeddings)
        
        # Mask out positive class
        for i, pos_i in enumerate(positive_idx):
            scores[i, pos_i] = -float('inf')
        
        # Select hardest negative
        hard_neg_indices = torch.argmax(scores, dim=1)
    
    return all_embeddings[hard_neg_indices]
```

### Learning Rate Scheduling

```python
import math

def cosine_warmup_scheduler(optimizer, num_epochs, warmup_epochs=10):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

## Summary

Visual-semantic embedding models form the backbone of modern ZSL:

1. **DeViSE** projects CNN features to word embedding space using ranking loss
2. **Bilinear models** capture rich interactions between modalities
3. **Cross-modal attention** learns fine-grained alignments
4. **Loss functions** include ranking, triplet, cross-entropy, and InfoNCE variants

Key considerations:
- Pre-trained features (both visual and semantic) are crucial
- Hard negative mining improves discriminative learning
- Regularization prevents overfitting to seen classes
- Proper normalization ensures stable training
