# Siamese Networks for One-Shot Learning

## Introduction

Siamese networks were among the first neural network architectures specifically designed for one-shot learning. Introduced by Bromley et al. (1993) for signature verification and popularized for one-shot image recognition by Koch et al. (2015), these networks learn to determine whether two inputs belong to the same class by comparing their learned representations.

## Architecture Overview

### The Twin Network Design

A Siamese network consists of two identical subnetworks (twins) that share the same weights. Each subnetwork processes one input, and their outputs are compared to determine similarity:

```
Input 1 ──→ [Encoder f_θ] ──→ Embedding 1 ──┐
                                            ├──→ [Distance/Similarity] ──→ Same/Different
Input 2 ──→ [Encoder f_θ] ──→ Embedding 2 ──┘
        (shared weights)
```

The key insight is **weight sharing**: both inputs pass through the exact same network, ensuring that similar inputs produce similar embeddings regardless of which branch processes them.

### Mathematical Formulation

Given two inputs $x_1$ and $x_2$, the Siamese network computes:

**Embeddings**:
$$z_1 = f_\theta(x_1), \quad z_2 = f_\theta(x_2)$$

**Similarity/Distance**:
$$s(x_1, x_2) = g(|z_1 - z_2|)$$

where $f_\theta$ is the encoder network, $|z_1 - z_2|$ computes element-wise absolute difference, and $g$ is typically a learned function mapping the difference to a similarity score.

### Why Weight Sharing Matters

Weight sharing ensures that the learned embedding function is:
1. **Symmetric**: The same transformation is applied regardless of input order
2. **Consistent**: Similar inputs always map to similar regions
3. **Efficient**: Only one network needs to be stored and trained

## Encoder Architectures

### Convolutional Encoder for Images

The original Koch et al. architecture for Omniglot:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseConvEncoder(nn.Module):
    """
    Convolutional encoder for Siamese networks.
    
    Architecture follows Koch et al. (2015) with modifications
    for different input sizes.
    """
    
    def __init__(
        self, 
        input_channels: int = 1, 
        input_size: int = 105,
        hidden_dim: int = 64,
        embedding_dim: int = 4096
    ):
        """
        Args:
            input_channels: Number of input channels (1 for grayscale)
            input_size: Input image size (assumes square images)
            hidden_dim: Base number of convolutional filters
            embedding_dim: Final embedding dimension
        """
        super().__init__()
        
        # Convolutional layers with increasing filter sizes
        self.conv1 = nn.Conv2d(input_channels, hidden_dim, kernel_size=10)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=7)
        self.conv3 = nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=4)
        self.conv4 = nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=4)
        
        # Calculate flattened size after convolutions
        self._conv_output_size = self._get_conv_output_size(
            input_channels, input_size
        )
        
        # Fully connected layer to embedding
        self.fc = nn.Linear(self._conv_output_size, embedding_dim)
    
    def _get_conv_output_size(self, channels: int, size: int) -> int:
        """Calculate output size after conv layers."""
        with torch.no_grad():
            dummy = torch.zeros(1, channels, size, size)
            dummy = self._conv_forward(dummy)
            return dummy.view(1, -1).size(1)
    
    def _conv_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolutional layers."""
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.conv4(x))
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute embedding for input.
        
        Args:
            x: Input images (batch, channels, height, width)
        
        Returns:
            embeddings: (batch, embedding_dim)
        """
        x = self._conv_forward(x)
        x = x.view(x.size(0), -1)
        x = torch.sigmoid(self.fc(x))
        
        return x
```

### Modern Encoder with ResNet Backbone

```python
import torchvision.models as models


class ResNetSiameseEncoder(nn.Module):
    """
    Siamese encoder using pre-trained ResNet backbone.
    """
    
    def __init__(
        self, 
        pretrained: bool = True,
        embedding_dim: int = 256
    ):
        super().__init__()
        
        # Load pre-trained ResNet
        resnet = models.resnet18(pretrained=pretrained)
        
        # Remove final classification layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Projection head to embedding space
        self.projection = nn.Sequential(
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        self.embedding_dim = embedding_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute embedding."""
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        embeddings = self.projection(features)
        
        # L2 normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
```

## Complete Siamese Network

### Standard Implementation

```python
class SiameseNetwork(nn.Module):
    """
    Complete Siamese Network for similarity learning.
    
    Computes whether two inputs belong to the same class
    by comparing their learned embeddings.
    """
    
    def __init__(self, encoder: nn.Module, use_distance: bool = True):
        """
        Args:
            encoder: Shared encoder network
            use_distance: If True, output distance; else output similarity
        """
        super().__init__()
        self.encoder = encoder
        self.use_distance = use_distance
        
        # Optional: learn a function on top of the distance
        if hasattr(encoder, 'embedding_dim'):
            embed_dim = encoder.embedding_dim
        else:
            embed_dim = 4096  # Default
        
        self.similarity_network = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )
    
    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        """Encode single input."""
        return self.encoder(x)
    
    def forward(
        self, 
        x1: torch.Tensor, 
        x2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute similarity between two inputs.
        
        Args:
            x1: First input batch (batch, *input_shape)
            x2: Second input batch (batch, *input_shape)
        
        Returns:
            If use_distance: L2 distance (batch,)
            Else: similarity score in [0, 1] (batch,)
        """
        # Encode both inputs
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        
        if self.use_distance:
            # L2 distance
            distance = F.pairwise_distance(z1, z2, p=2)
            return distance
        else:
            # Learned similarity on absolute difference
            diff = torch.abs(z1 - z2)
            similarity = self.similarity_network(diff)
            return similarity.squeeze(-1)
    
    def predict_same_class(
        self, 
        x1: torch.Tensor, 
        x2: torch.Tensor, 
        threshold: float = 0.5
    ) -> torch.Tensor:
        """
        Predict whether inputs are from same class.
        
        Args:
            x1, x2: Input batches
            threshold: Decision threshold
        
        Returns:
            predictions: Boolean tensor (batch,)
        """
        if self.use_distance:
            distances = self.forward(x1, x2)
            return distances < threshold
        else:
            similarities = self.forward(x1, x2)
            return similarities > threshold
```

## Loss Functions

### Contrastive Loss

The standard loss for training Siamese networks:

$$\mathcal{L}(z_1, z_2, y) = (1-y) \cdot \frac{1}{2} d^2 + y \cdot \frac{1}{2} \max(0, m - d)^2$$

where $y=0$ for similar pairs and $y=1$ for dissimilar pairs.

```python
class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for Siamese networks.
    
    Pulls similar pairs together and pushes dissimilar pairs apart
    beyond a margin.
    """
    
    def __init__(self, margin: float = 1.0):
        """
        Args:
            margin: Minimum distance for dissimilar pairs
        """
        super().__init__()
        self.margin = margin
    
    def forward(
        self, 
        z1: torch.Tensor, 
        z2: torch.Tensor, 
        y: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            z1: First embeddings (batch, embed_dim)
            z2: Second embeddings (batch, embed_dim)
            y: Labels - 0 for similar, 1 for dissimilar
        
        Returns:
            loss: Scalar loss value
        """
        # Compute Euclidean distance
        distance = F.pairwise_distance(z1, z2, p=2)
        
        # Loss for similar pairs: minimize distance
        similar_loss = (1 - y) * 0.5 * distance ** 2
        
        # Loss for dissimilar pairs: maximize distance up to margin
        dissimilar_loss = y * 0.5 * F.relu(self.margin - distance) ** 2
        
        loss = similar_loss + dissimilar_loss
        
        return loss.mean()


class MarginContrastiveLoss(nn.Module):
    """
    Variant with separate margins for similar and dissimilar pairs.
    """
    
    def __init__(
        self, 
        pos_margin: float = 0.0, 
        neg_margin: float = 1.0
    ):
        super().__init__()
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
    
    def forward(
        self, 
        z1: torch.Tensor, 
        z2: torch.Tensor, 
        y: torch.Tensor
    ) -> torch.Tensor:
        distance = F.pairwise_distance(z1, z2, p=2)
        
        # Similar pairs should be closer than pos_margin
        pos_loss = (1 - y) * F.relu(distance - self.pos_margin) ** 2
        
        # Dissimilar pairs should be farther than neg_margin
        neg_loss = y * F.relu(self.neg_margin - distance) ** 2
        
        return (pos_loss + neg_loss).mean()
```

### Binary Cross-Entropy Loss

Alternative loss treating similarity as binary classification:

```python
class SiameseBCELoss(nn.Module):
    """
    Binary cross-entropy loss for Siamese similarity prediction.
    
    Treats the problem as binary classification:
    is this pair from the same class?
    """
    
    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.label_smoothing = label_smoothing
    
    def forward(
        self, 
        similarity: torch.Tensor, 
        y: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            similarity: Predicted similarity in [0, 1]
            y: Labels - 1 for same class, 0 for different
        """
        # Optional label smoothing
        if self.label_smoothing > 0:
            y = y * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        return F.binary_cross_entropy(similarity, y.float())
```

## One-Shot Classification

### Classification by Comparison

For one-shot classification, we compare a query image against one example from each class:

```python
def one_shot_classify(
    model: SiameseNetwork,
    support_set: torch.Tensor,
    support_labels: torch.Tensor,
    query: torch.Tensor,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Perform one-shot classification using a Siamese network.
    
    Args:
        model: Trained Siamese network
        support_set: One example per class (n_classes, *input_shape)
        support_labels: Class labels (n_classes,)
        query: Query images to classify (n_query, *input_shape)
        device: Computation device
    
    Returns:
        predictions: Predicted class labels (n_query,)
    """
    model.eval()
    model = model.to(device)
    
    support_set = support_set.to(device)
    query = query.to(device)
    
    n_classes = support_set.size(0)
    n_query = query.size(0)
    
    predictions = []
    
    with torch.no_grad():
        # Encode support set once
        support_embeddings = model.forward_one(support_set)
        
        for i in range(n_query):
            query_embedding = model.forward_one(query[i:i+1])
            
            # Compute distance to each support example
            distances = []
            for j in range(n_classes):
                dist = F.pairwise_distance(
                    query_embedding,
                    support_embeddings[j:j+1]
                )
                distances.append(dist.item())
            
            # Predict class with minimum distance
            pred_idx = np.argmin(distances)
            predictions.append(support_labels[pred_idx].item())
    
    return torch.tensor(predictions, device=device)


def one_shot_classify_efficient(
    model: SiameseNetwork,
    support_set: torch.Tensor,
    support_labels: torch.Tensor,
    query: torch.Tensor
) -> torch.Tensor:
    """
    Efficient one-shot classification using batched operations.
    """
    model.eval()
    
    n_classes = support_set.size(0)
    n_query = query.size(0)
    
    with torch.no_grad():
        # Encode all examples
        support_embeddings = model.forward_one(support_set)  # (C, D)
        query_embeddings = model.forward_one(query)  # (Q, D)
        
        # Compute all pairwise distances
        # (Q, C) distance matrix
        distances = torch.cdist(query_embeddings, support_embeddings, p=2)
        
        # Get nearest support example for each query
        nearest_idx = distances.argmin(dim=1)
        predictions = support_labels[nearest_idx]
    
    return predictions
```

## Training Strategies

### Pair Sampling

Effective training requires balanced sampling of similar and dissimilar pairs:

```python
import random
from collections import defaultdict


class PairSampler:
    """
    Sample balanced pairs for Siamese network training.
    """
    
    def __init__(
        self, 
        labels: torch.Tensor,
        positive_ratio: float = 0.5
    ):
        """
        Args:
            labels: All class labels
            positive_ratio: Fraction of similar pairs to sample
        """
        self.labels = labels
        self.positive_ratio = positive_ratio
        
        # Build class-to-indices mapping
        self.class_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.class_indices[label.item()].append(idx)
        
        self.classes = list(self.class_indices.keys())
    
    def sample_pairs(self, batch_size: int):
        """
        Sample a batch of pairs.
        
        Returns:
            idx1, idx2: Indices of paired examples
            labels: 0 for same class, 1 for different class
        """
        idx1, idx2, pair_labels = [], [], []
        
        n_positive = int(batch_size * self.positive_ratio)
        n_negative = batch_size - n_positive
        
        # Sample positive pairs (same class)
        for _ in range(n_positive):
            # Choose a class with at least 2 examples
            valid_classes = [c for c in self.classes 
                          if len(self.class_indices[c]) >= 2]
            cls = random.choice(valid_classes)
            
            # Sample two different examples from this class
            i1, i2 = random.sample(self.class_indices[cls], 2)
            
            idx1.append(i1)
            idx2.append(i2)
            pair_labels.append(0)  # Same class
        
        # Sample negative pairs (different classes)
        for _ in range(n_negative):
            # Choose two different classes
            cls1, cls2 = random.sample(self.classes, 2)
            
            i1 = random.choice(self.class_indices[cls1])
            i2 = random.choice(self.class_indices[cls2])
            
            idx1.append(i1)
            idx2.append(i2)
            pair_labels.append(1)  # Different classes
        
        # Shuffle
        combined = list(zip(idx1, idx2, pair_labels))
        random.shuffle(combined)
        idx1, idx2, pair_labels = zip(*combined)
        
        return (
            torch.tensor(idx1),
            torch.tensor(idx2),
            torch.tensor(pair_labels)
        )


class HardPairMiner:
    """
    Online hard pair mining for more effective training.
    """
    
    def __init__(self, model: SiameseNetwork, margin: float = 0.5):
        self.model = model
        self.margin = margin
    
    def mine_hard_pairs(
        self, 
        embeddings: torch.Tensor, 
        labels: torch.Tensor
    ):
        """
        Find hard positive and negative pairs.
        
        Hard positives: Same class but far apart
        Hard negatives: Different class but close together
        """
        n = embeddings.size(0)
        
        # Compute all pairwise distances
        distances = torch.cdist(embeddings, embeddings, p=2)
        
        # Create label masks
        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
        
        hard_pos_pairs = []
        hard_neg_pairs = []
        
        for i in range(n):
            # Hard positive: same class, max distance
            pos_mask = labels_eq[i].clone()
            pos_mask[i] = False  # Exclude self
            
            if pos_mask.any():
                pos_distances = distances[i][pos_mask]
                hardest_pos_idx = pos_mask.nonzero()[pos_distances.argmax()]
                
                if distances[i, hardest_pos_idx] > self.margin:
                    hard_pos_pairs.append((i, hardest_pos_idx.item(), 0))
            
            # Hard negative: different class, min distance
            neg_mask = ~labels_eq[i]
            
            if neg_mask.any():
                neg_distances = distances[i][neg_mask]
                hardest_neg_idx = neg_mask.nonzero()[neg_distances.argmin()]
                
                if distances[i, hardest_neg_idx] < self.margin:
                    hard_neg_pairs.append((i, hardest_neg_idx.item(), 1))
        
        return hard_pos_pairs, hard_neg_pairs
```

### Complete Training Loop

```python
class SiameseTrainer:
    """
    Training loop for Siamese networks.
    """
    
    def __init__(
        self,
        model: SiameseNetwork,
        train_data: torch.Tensor,
        train_labels: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.train_data = train_data.to(device)
        self.train_labels = train_labels.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        
        self.pair_sampler = PairSampler(train_labels)
    
    def train_epoch(
        self, 
        n_iterations: int = 1000,
        batch_size: int = 32,
        log_interval: int = 100
    ):
        """
        Train for one epoch.
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for iteration in range(n_iterations):
            # Sample pairs
            idx1, idx2, labels = self.pair_sampler.sample_pairs(batch_size)
            
            x1 = self.train_data[idx1]
            x2 = self.train_data[idx2]
            labels = labels.float().to(self.device)
            
            # Forward pass
            z1 = self.model.encoder(x1)
            z2 = self.model.encoder(x2)
            
            # Compute loss
            loss = self.loss_fn(z1, z2, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Compute accuracy (for monitoring)
            with torch.no_grad():
                distances = F.pairwise_distance(z1, z2)
                predictions = (distances > 0.5).float()
                correct += (predictions == labels).sum().item()
                total += batch_size
            
            if (iteration + 1) % log_interval == 0:
                avg_loss = total_loss / log_interval
                accuracy = correct / total
                print(f"Iteration {iteration+1}/{n_iterations} | "
                      f"Loss: {avg_loss:.4f} | Acc: {accuracy:.4f}")
                total_loss = 0
                correct = 0
                total = 0
        
        return total_loss / max(n_iterations % log_interval, 1)
    
    def evaluate(
        self,
        test_data: torch.Tensor,
        test_labels: torch.Tensor,
        n_way: int = 5,
        n_episodes: int = 100
    ):
        """
        Evaluate one-shot classification performance.
        """
        self.model.eval()
        
        test_data = test_data.to(self.device)
        test_labels = test_labels.to(self.device)
        
        # Get unique classes
        unique_classes = torch.unique(test_labels)
        
        accuracies = []
        
        for _ in range(n_episodes):
            # Sample n_way classes
            episode_classes = unique_classes[
                torch.randperm(len(unique_classes))[:n_way]
            ]
            
            # Sample 1 support and 1 query per class
            support_data = []
            query_data = []
            query_labels = []
            
            for new_label, cls in enumerate(episode_classes):
                cls_mask = test_labels == cls
                cls_indices = cls_mask.nonzero(as_tuple=True)[0]
                
                perm = cls_indices[torch.randperm(len(cls_indices))]
                support_idx = perm[0]
                query_idx = perm[1] if len(perm) > 1 else perm[0]
                
                support_data.append(test_data[support_idx])
                query_data.append(test_data[query_idx])
                query_labels.append(new_label)
            
            support = torch.stack(support_data)
            query = torch.stack(query_data)
            query_labels = torch.tensor(query_labels, device=self.device)
            support_labels = torch.arange(n_way, device=self.device)
            
            # Classify
            predictions = one_shot_classify_efficient(
                self.model, support, support_labels, query
            )
            
            acc = (predictions == query_labels).float().mean().item()
            accuracies.append(acc)
        
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        
        return mean_acc, std_acc
```

## Variants and Extensions

### Triplet Siamese Networks

Combine Siamese architecture with triplet loss:

```python
class TripletSiameseNetwork(nn.Module):
    """
    Siamese network trained with triplet loss.
    """
    
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder
    
    def forward(
        self, 
        anchor: torch.Tensor, 
        positive: torch.Tensor, 
        negative: torch.Tensor
    ):
        """
        Compute embeddings for triplet.
        """
        z_a = self.encoder(anchor)
        z_p = self.encoder(positive)
        z_n = self.encoder(negative)
        
        return z_a, z_p, z_n
    
    def triplet_loss(
        self, 
        anchor: torch.Tensor, 
        positive: torch.Tensor, 
        negative: torch.Tensor,
        margin: float = 0.2
    ):
        """Compute triplet loss."""
        z_a, z_p, z_n = self.forward(anchor, positive, negative)
        
        d_ap = F.pairwise_distance(z_a, z_p)
        d_an = F.pairwise_distance(z_a, z_n)
        
        loss = F.relu(d_ap - d_an + margin)
        
        return loss.mean()
```

### Cross-Domain Siamese Networks

For domain adaptation scenarios:

```python
class CrossDomainSiamese(nn.Module):
    """
    Siamese network with domain-specific encoders.
    
    Useful when comparing examples from different domains
    (e.g., sketch to photo matching).
    """
    
    def __init__(
        self, 
        encoder_a: nn.Module, 
        encoder_b: nn.Module,
        shared_dim: int = 256
    ):
        super().__init__()
        self.encoder_a = encoder_a
        self.encoder_b = encoder_b
        
        # Project to shared space
        self.project_a = nn.Linear(encoder_a.embedding_dim, shared_dim)
        self.project_b = nn.Linear(encoder_b.embedding_dim, shared_dim)
    
    def forward(self, x_a: torch.Tensor, x_b: torch.Tensor):
        """
        Compare examples from two domains.
        """
        z_a = self.project_a(self.encoder_a(x_a))
        z_b = self.project_b(self.encoder_b(x_b))
        
        # L2 normalize
        z_a = F.normalize(z_a, p=2, dim=1)
        z_b = F.normalize(z_b, p=2, dim=1)
        
        return z_a, z_b
```

## Summary

Siamese networks provide a foundational approach to one-shot learning by:

1. **Learning similarity through comparison**: Twin networks with shared weights
2. **Enabling nearest-neighbor classification**: Query classified by finding most similar support example
3. **Transferring to novel classes**: Similarity function generalizes beyond training classes

Key implementation considerations:
- Balance positive and negative pairs during training
- Use hard negative mining for more effective learning
- Consider using pre-trained backbones for better representations
- Monitor both contrastive loss and one-shot accuracy during training

## References

1. Bromley, J., et al. "Signature Verification using a Siamese Time Delay Neural Network." NeurIPS 1993.
2. Koch, G., et al. "Siamese Neural Networks for One-shot Image Recognition." ICML Deep Learning Workshop 2015.
3. Chopra, S., et al. "Learning a Similarity Metric Discriminatively, with Application to Face Verification." CVPR 2005.
4. Schroff, F., et al. "FaceNet: A Unified Embedding for Face Recognition and Clustering." CVPR 2015.
