# Prototypical Networks

## Introduction

Prototypical Networks, introduced by Snell et al. (2017), are among the most elegant and effective approaches to few-shot learning. The key insight is remarkably simple: classes can be represented by the mean (prototype) of their support embeddings, and classification is performed by finding the nearest prototype.

## Core Concept

### The Prototype Idea

Given an embedding function $f_\theta$ and a support set $\mathcal{S}_c = \{(x_1^c, y_1^c), \ldots, (x_{K}^c, y_K^c)\}$ for class $c$, the **prototype** for class $c$ is:

$$\mathbf{c}_c = \frac{1}{|\mathcal{S}_c|} \sum_{(x_i, y_i) \in \mathcal{S}_c} f_\theta(x_i)$$

This is simply the centroid of class embeddings in the learned representation space.

### Classification via Distance

Given a query $x$, classification is performed by computing distances to all prototypes and applying softmax:

$$p(y = c | x) = \frac{\exp(-d(f_\theta(x), \mathbf{c}_c))}{\sum_{c'} \exp(-d(f_\theta(x), \mathbf{c}_{c'}))}$$

where $d$ is a distance function (typically squared Euclidean distance).

### Why Prototypes Work

The effectiveness of prototypical networks can be understood through several lenses:

**Bregman Divergence Perspective**: Snell et al. (2017) show that for any Bregman divergence, the optimal classification is achieved by comparing to cluster means. Squared Euclidean distance is a Bregman divergence.

**Mixup Interpretation**: Computing prototypes can be viewed as a form of data augmentation through averaging, improving generalization.

**Robustness**: Averaging multiple examples reduces the impact of noisy or atypical support examples.

## Mathematical Foundation

### Bregman Divergences

A Bregman divergence associated with a strictly convex, differentiable function $\phi$ is:

$$d_\phi(z, z') = \phi(z) - \phi(z') - (z - z')^T \nabla\phi(z')$$

**Theorem**: For exponential family mixture models with Bregman divergence $d_\phi$, the optimal cluster representative minimizing expected divergence is the cluster mean.

The squared Euclidean distance is a Bregman divergence with $\phi(z) = \|z\|_2^2$:

$$d_{\phi}(z, z') = \|z\|^2 - \|z'\|^2 - 2z'^T(z - z') = \|z - z'\|^2$$

### Connection to Gaussian Mixture Models

Prototypical networks can be viewed as learning a mixture of Gaussians with shared covariance:

$$p(x | c) = \mathcal{N}(f_\theta(x); \mu_c, \sigma^2 I)$$

where $\mu_c$ is the prototype and $\sigma^2$ is shared across classes. Under this model, the posterior over classes is exactly the softmax over negative squared distances.

## PyTorch Implementation

### Complete Prototypical Network

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class PrototypicalNetwork(nn.Module):
    """
    Prototypical Networks for Few-Shot Learning.
    
    Reference: Snell et al. "Prototypical Networks for Few-shot Learning" 
    NeurIPS 2017
    """
    
    def __init__(
        self, 
        encoder: nn.Module,
        distance: str = 'euclidean',
        temperature: float = 1.0
    ):
        """
        Args:
            encoder: Embedding network f_θ
            distance: 'euclidean' or 'cosine'
            temperature: Scaling factor for logits
        """
        super().__init__()
        self.encoder = encoder
        self.distance = distance
        self.temperature = temperature
    
    def compute_prototypes(
        self, 
        support_embeddings: torch.Tensor, 
        support_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute class prototypes from support embeddings.
        
        Args:
            support_embeddings: (n_support, embed_dim)
            support_labels: (n_support,) with values in {0, ..., n_way-1}
        
        Returns:
            prototypes: (n_way, embed_dim)
        """
        n_way = support_labels.max().item() + 1
        embed_dim = support_embeddings.size(1)
        
        prototypes = torch.zeros(n_way, embed_dim, device=support_embeddings.device)
        
        for c in range(n_way):
            class_mask = (support_labels == c)
            prototypes[c] = support_embeddings[class_mask].mean(dim=0)
        
        return prototypes
    
    def compute_distances(
        self, 
        query_embeddings: torch.Tensor, 
        prototypes: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute distances from queries to prototypes.
        
        Args:
            query_embeddings: (n_query, embed_dim)
            prototypes: (n_way, embed_dim)
        
        Returns:
            distances: (n_query, n_way)
        """
        if self.distance == 'euclidean':
            # Squared Euclidean: ||q - p||^2 = ||q||^2 + ||p||^2 - 2*q^T*p
            query_norm = (query_embeddings ** 2).sum(dim=1, keepdim=True)
            proto_norm = (prototypes ** 2).sum(dim=1, keepdim=True).t()
            cross_term = torch.mm(query_embeddings, prototypes.t())
            
            distances = query_norm + proto_norm - 2 * cross_term
            distances = torch.clamp(distances, min=0.0)
            
        elif self.distance == 'cosine':
            query_norm = F.normalize(query_embeddings, p=2, dim=1)
            proto_norm = F.normalize(prototypes, p=2, dim=1)
            similarity = torch.mm(query_norm, proto_norm.t())
            distances = 1 - similarity
        else:
            raise ValueError(f"Unknown distance: {self.distance}")
        
        return distances
    
    def forward(
        self, 
        support: torch.Tensor, 
        support_labels: torch.Tensor, 
        query: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute classification logits for query examples.
        
        Args:
            support: Support set images (n_support, *input_shape)
            support_labels: Support labels (n_support,)
            query: Query images (n_query, *input_shape)
        
        Returns:
            logits: (n_query, n_way) classification logits
        """
        n_support = support.size(0)
        
        # Concatenate for efficient batch encoding
        all_images = torch.cat([support, query], dim=0)
        all_embeddings = self.encoder(all_images)
        
        # Split back
        support_embeddings = all_embeddings[:n_support]
        query_embeddings = all_embeddings[n_support:]
        
        # Compute prototypes
        prototypes = self.compute_prototypes(support_embeddings, support_labels)
        
        # Compute distances
        distances = self.compute_distances(query_embeddings, prototypes)
        
        # Convert to logits (negative distance)
        logits = -distances / self.temperature
        
        return logits
    
    def loss(
        self, 
        support: torch.Tensor, 
        support_labels: torch.Tensor, 
        query: torch.Tensor, 
        query_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute loss and accuracy for an episode.
        
        Returns:
            loss: Cross-entropy loss
            accuracy: Classification accuracy
        """
        logits = self.forward(support, support_labels, query)
        loss = F.cross_entropy(logits, query_labels)
        
        predictions = logits.argmax(dim=1)
        accuracy = (predictions == query_labels).float().mean()
        
        return loss, accuracy
```

### Standard Encoder Architecture

```python
class Conv4Encoder(nn.Module):
    """
    4-layer convolutional encoder commonly used in few-shot learning.
    """
    
    def __init__(self, in_channels: int = 1, hidden_dim: int = 64):
        super().__init__()
        
        self.encoder = nn.Sequential(
            self._conv_block(in_channels, hidden_dim),
            self._conv_block(hidden_dim, hidden_dim),
            self._conv_block(hidden_dim, hidden_dim),
            self._conv_block(hidden_dim, hidden_dim),
        )
        self.output_dim = hidden_dim
    
    def _conv_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return F.adaptive_avg_pool2d(features, 1).view(features.size(0), -1)
```

## Training Procedure

### Episodic Training Loop

```python
def train_prototypical_network(
    model: PrototypicalNetwork,
    train_dataset,
    optimizer: torch.optim.Optimizer,
    n_way: int = 5,
    k_shot: int = 5,
    n_query: int = 15,
    n_episodes: int = 100,
    device: str = 'cuda'
):
    """
    Train prototypical network for one epoch.
    """
    model.train()
    total_loss = 0
    total_acc = 0
    
    for episode in range(n_episodes):
        # Sample episode
        support, support_labels, query, query_labels = sample_episode(
            train_dataset, n_way, k_shot, n_query
        )
        
        support = support.to(device)
        support_labels = support_labels.to(device)
        query = query.to(device)
        query_labels = query_labels.to(device)
        
        # Forward and backward
        optimizer.zero_grad()
        loss, acc = model.loss(support, support_labels, query, query_labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_acc += acc.item()
    
    return total_loss / n_episodes, total_acc / n_episodes
```

### Training Configuration

Key hyperparameters for prototypical networks:

| Parameter | Typical Value | Notes |
|-----------|--------------|-------|
| N-way (train) | 20-60 | Higher N during training often helps |
| K-shot (train) | 5-15 | More shots than test |
| Query per class | 15 | Standard choice |
| Learning rate | 1e-3 | With Adam optimizer |
| Embedding dim | 64-1600 | Depends on complexity |
| Episodes per epoch | 100-1000 | More for complex tasks |

**Training Trick**: Train with higher N-way than test. Training 30-way and testing 5-way often improves performance.

## Variants and Extensions

### Infinite Mixture Prototypes

Handle class uncertainty by modeling each class with multiple prototypes:

```python
class InfiniteMixturePrototypes(nn.Module):
    """
    Represent each class with multiple prototypes using
    a mixture model.
    """
    
    def __init__(self, encoder: nn.Module, n_prototypes: int = 3):
        super().__init__()
        self.encoder = encoder
        self.n_prototypes = n_prototypes
    
    def compute_cluster_prototypes(
        self, 
        support_embeddings: torch.Tensor,
        support_labels: torch.Tensor
    ):
        """
        Use k-means to find multiple prototypes per class.
        """
        from sklearn.cluster import KMeans
        
        n_way = support_labels.max().item() + 1
        embed_dim = support_embeddings.size(1)
        
        all_prototypes = []
        prototype_weights = []
        prototype_classes = []
        
        for c in range(n_way):
            class_mask = support_labels == c
            class_embeddings = support_embeddings[class_mask].cpu().numpy()
            
            n_samples = class_embeddings.shape[0]
            n_clusters = min(self.n_prototypes, n_samples)
            
            if n_clusters == 1:
                prototypes = class_embeddings.mean(axis=0, keepdims=True)
                weights = [1.0]
            else:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                kmeans.fit(class_embeddings)
                prototypes = kmeans.cluster_centers_
                
                # Weight by cluster size
                labels = kmeans.labels_
                weights = [(labels == k).sum() / len(labels) 
                          for k in range(n_clusters)]
            
            for i, (proto, weight) in enumerate(zip(prototypes, weights)):
                all_prototypes.append(proto)
                prototype_weights.append(weight)
                prototype_classes.append(c)
        
        return (
            torch.tensor(all_prototypes, device=support_embeddings.device),
            torch.tensor(prototype_weights, device=support_embeddings.device),
            torch.tensor(prototype_classes, device=support_embeddings.device)
        )
```

### Task-Adaptive Prototypes

Adapt prototypes based on query information:

```python
class TaskAdaptivePrototypes(nn.Module):
    """
    Refine prototypes using attention over support-query pairs.
    """
    
    def __init__(self, encoder: nn.Module, embed_dim: int = 64):
        super().__init__()
        self.encoder = encoder
        
        # Attention mechanism
        self.query_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=4,
            batch_first=True
        )
    
    def forward(self, support, support_labels, query):
        n_support = support.size(0)
        n_way = support_labels.max().item() + 1
        
        # Encode
        all_images = torch.cat([support, query], dim=0)
        all_embeddings = self.encoder(all_images)
        
        support_embeddings = all_embeddings[:n_support]
        query_embeddings = all_embeddings[n_support:]
        
        # Initial prototypes
        prototypes = []
        for c in range(n_way):
            mask = support_labels == c
            prototypes.append(support_embeddings[mask].mean(dim=0))
        prototypes = torch.stack(prototypes)  # (n_way, embed_dim)
        
        # Refine with attention to queries
        # Query: prototypes, Key/Value: all support
        refined_prototypes, _ = self.query_attention(
            prototypes.unsqueeze(0),
            support_embeddings.unsqueeze(0),
            support_embeddings.unsqueeze(0)
        )
        refined_prototypes = refined_prototypes.squeeze(0)
        
        # Combine original and refined
        prototypes = 0.5 * prototypes + 0.5 * refined_prototypes
        
        # Compute distances
        distances = torch.cdist(query_embeddings, prototypes, p=2) ** 2
        
        return -distances
```

### Semi-Supervised Prototypical Networks

Use unlabeled data to refine prototypes:

```python
class SemiSupervisedProtoNet(nn.Module):
    """
    Incorporate unlabeled examples through soft assignment.
    """
    
    def __init__(self, encoder: nn.Module, n_refine_steps: int = 3):
        super().__init__()
        self.encoder = encoder
        self.n_refine_steps = n_refine_steps
    
    def forward(
        self, 
        support: torch.Tensor,
        support_labels: torch.Tensor,
        query: torch.Tensor,
        unlabeled: Optional[torch.Tensor] = None
    ):
        # Encode all
        support_emb = self.encoder(support)
        query_emb = self.encoder(query)
        
        n_way = support_labels.max().item() + 1
        
        # Initial prototypes from labeled support
        prototypes = self._compute_prototypes(support_emb, support_labels, n_way)
        
        if unlabeled is not None:
            unlabeled_emb = self.encoder(unlabeled)
            
            # Iteratively refine prototypes
            for _ in range(self.n_refine_steps):
                # Soft assign unlabeled to prototypes
                distances = torch.cdist(unlabeled_emb, prototypes, p=2) ** 2
                soft_labels = F.softmax(-distances, dim=1)  # (n_unlabeled, n_way)
                
                # Recompute prototypes
                new_prototypes = []
                for c in range(n_way):
                    # Weighted mean including unlabeled
                    labeled_mask = support_labels == c
                    labeled_contrib = support_emb[labeled_mask].sum(dim=0)
                    labeled_count = labeled_mask.sum()
                    
                    unlabeled_weights = soft_labels[:, c]
                    unlabeled_contrib = (unlabeled_emb * unlabeled_weights.unsqueeze(1)).sum(dim=0)
                    unlabeled_count = unlabeled_weights.sum()
                    
                    prototype = (labeled_contrib + unlabeled_contrib) / (labeled_count + unlabeled_count)
                    new_prototypes.append(prototype)
                
                prototypes = torch.stack(new_prototypes)
        
        # Final classification
        distances = torch.cdist(query_emb, prototypes, p=2) ** 2
        return -distances
    
    def _compute_prototypes(self, embeddings, labels, n_way):
        prototypes = []
        for c in range(n_way):
            mask = labels == c
            prototypes.append(embeddings[mask].mean(dim=0))
        return torch.stack(prototypes)
```

## Theoretical Analysis

### Sample Complexity

For prototypical networks with $K$ support examples per class and embedding dimension $d$:

**Prototype Estimation Error**:
$$\mathbb{E}\left[\|\hat{\mathbf{c}}_c - \mathbf{c}_c^*\|_2^2\right] = O\left(\frac{\sigma^2}{K}\right)$$

where $\sigma^2$ is the variance of class embeddings.

**Classification Error**: Under mild assumptions, the excess classification risk decreases as $O(1/\sqrt{K})$.

### Comparison with Other Methods

| Method | Prototype Computation | Classification | Complexity |
|--------|----------------------|----------------|------------|
| Prototypical | Mean | Nearest centroid | $O(NK)$ |
| Matching Networks | Identity | Attention-weighted | $O(NK \cdot Q)$ |
| Relation Networks | Mean | Learned relation | $O(N \cdot Q)$ |

## Practical Recommendations

### Distance Function Selection

**Euclidean** (default):
- Works well with L2-normalized embeddings
- Better for high-dimensional spaces
- Computationally efficient

**Cosine**:
- Scale-invariant
- Works well without normalization
- May help with varying embedding magnitudes

### Embedding Normalization

```python
def forward_with_normalization(self, support, support_labels, query):
    # Encode
    support_emb = self.encoder(support)
    query_emb = self.encoder(query)
    
    # L2 normalize (often improves performance)
    support_emb = F.normalize(support_emb, p=2, dim=1)
    query_emb = F.normalize(query_emb, p=2, dim=1)
    
    # Rest of forward pass...
```

### Data Augmentation

Strong augmentation during episodic training:

```python
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(84, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

## Experimental Results

### Benchmark Performance (5-way accuracy)

| Method | Omniglot 1-shot | Omniglot 5-shot | mini-ImageNet 1-shot | mini-ImageNet 5-shot |
|--------|-----------------|-----------------|---------------------|---------------------|
| Matching Networks | 98.1% | 98.9% | 43.6% | 55.3% |
| **Prototypical Networks** | **98.8%** | **99.7%** | **49.4%** | **68.2%** |
| MAML | 98.7% | 99.9% | 48.7% | 63.1% |

## Summary

Prototypical Networks offer:

1. **Simplicity**: Just compute class means and nearest-neighbor classify
2. **Efficiency**: No iterative optimization at test time
3. **Strong Performance**: Competitive with more complex methods
4. **Theoretical Foundation**: Grounded in Bregman divergence theory

Key insights for practitioners:
- Train with higher N-way than test
- Use squared Euclidean distance with normalized embeddings
- Consider multiple prototypes for complex classes
- Data augmentation is crucial for generalization

## References

1. Snell, J., et al. "Prototypical Networks for Few-shot Learning." NeurIPS 2017.
2. Fort, S. "Gaussian Prototypical Networks for Few-Shot Learning on Omniglot." CVPR Workshop 2017.
3. Allen, K., et al. "Infinite Mixture Prototypes for Few-Shot Learning." ICML 2019.
4. Ren, M., et al. "Meta-Learning for Semi-Supervised Few-Shot Classification." ICLR 2018.
