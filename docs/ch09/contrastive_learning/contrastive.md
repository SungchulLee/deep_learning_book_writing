# Contrastive Learning

## Learning Objectives

By the end of this section, you will be able to:

- Understand the fundamental principles of contrastive representation learning
- Derive and implement the InfoNCE loss and its variants
- Explain why contrastive learning works from an information-theoretic perspective
- Implement the core components of contrastive learning systems
- Analyze the role of negative samples, batch size, and temperature in contrastive learning

## Introduction

Contrastive learning has emerged as the dominant paradigm in self-supervised representation learning. The core idea is elegantly simple: learn representations by pulling similar (positive) samples together while pushing dissimilar (negative) samples apart in the embedding space.

### The Contrastive Learning Framework

Given an input $x$, contrastive learning creates:
- **Positive pairs** $(x, x^+)$: Different views of the same sample
- **Negative pairs** $(x, x^-)$: Views from different samples

The objective is to learn an encoder $f_\theta$ such that:

$$\text{sim}(f_\theta(x), f_\theta(x^+)) \gg \text{sim}(f_\theta(x), f_\theta(x^-))$$

where $\text{sim}(\cdot, \cdot)$ is a similarity function (typically cosine similarity).

## Mathematical Foundations

### Noise Contrastive Estimation (NCE)

Contrastive learning builds upon Noise Contrastive Estimation, which frames density estimation as classification:

$$p(C=1 | x) = \frac{p_\theta(x)}{p_\theta(x) + k \cdot p_n(x)}$$

where:
- $C=1$ indicates the sample is from the data distribution
- $p_\theta(x)$ is the model distribution
- $p_n(x)$ is the noise distribution
- $k$ is the ratio of noise to data samples

### InfoNCE Loss

The Information Noise Contrastive Estimation (InfoNCE) loss is the workhorse of modern contrastive learning:

$$\mathcal{L}_{\text{InfoNCE}} = -\mathbb{E}\left[\log \frac{\exp(f(x) \cdot f(x^+) / \tau)}{\sum_{i=0}^{K} \exp(f(x) \cdot f(x_i) / \tau)}\right]$$

where:
- $f(x)$ is the encoded representation (typically L2-normalized)
- $x^+$ is the positive sample
- $x_0 = x^+$ and $x_1, ..., x_K$ are negative samples
- $\tau$ is the temperature parameter

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class InfoNCELoss(nn.Module):
    """
    InfoNCE Loss (Normalized Temperature-scaled Cross Entropy).
    
    This is the core loss function for contrastive learning methods
    like SimCLR, MoCo, and many others.
    
    Mathematical formulation:
    L = -log(exp(sim(z_i, z_j) / τ) / Σ_k exp(sim(z_i, z_k) / τ))
    
    where z_i and z_j are positive pairs, and k iterates over all samples
    including the positive pair (i.e., the denominator sums over 1 positive
    and N-1 negatives per sample in the batch).
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z_i, z_j, labels=None):
        """
        Compute InfoNCE loss.
        
        Args:
            z_i: First view embeddings (B, D), L2-normalized
            z_j: Second view embeddings (B, D), L2-normalized
            labels: Optional labels for supervised contrastive (B,)
        
        Returns:
            loss: Scalar loss value
        """
        batch_size = z_i.shape[0]
        device = z_i.device
        
        # Normalize embeddings
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        # Concatenate embeddings from both views
        # representations: (2B, D)
        representations = torch.cat([z_i, z_j], dim=0)
        
        # Compute similarity matrix: (2B, 2B)
        # sim[i, j] = z_i · z_j
        similarity_matrix = torch.mm(representations, representations.t())
        
        # Scale by temperature
        similarity_matrix = similarity_matrix / self.temperature
        
        # Create labels for positive pairs
        # For sample i in z_i, its positive is at position i + B in z_j
        # For sample i in z_j, its positive is at position i in z_i
        if labels is None:
            # Self-supervised: positives are augmented views of same image
            labels = torch.arange(batch_size, device=device)
            labels = torch.cat([labels + batch_size, labels], dim=0)
        else:
            # Supervised contrastive: same class samples are positives
            labels = torch.cat([labels, labels], dim=0)
        
        # Mask out self-similarity (diagonal elements)
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -float('inf'))
        
        # Compute loss using cross-entropy
        # This is equivalent to InfoNCE when labels specify positive indices
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss


class NTXentLoss(nn.Module):
    """
    NT-Xent Loss (Normalized Temperature-scaled Cross Entropy Loss).
    
    Used in SimCLR. Equivalent to InfoNCE but with explicit positive/negative
    handling for clarity.
    """
    def __init__(self, temperature=0.5, eps=1e-6):
        super().__init__()
        self.temperature = temperature
        self.eps = eps
    
    def forward(self, z_i, z_j):
        """
        Compute NT-Xent loss.
        
        For each sample i, the loss is:
        -log(exp(sim(z_i, z_j) / τ) / Σ_{k≠i} exp(sim(z_i, z_k) / τ))
        
        Args:
            z_i: First view embeddings (B, D)
            z_j: Second view embeddings (B, D)
        
        Returns:
            loss: Scalar loss value
        """
        batch_size = z_i.shape[0]
        device = z_i.device
        
        # Normalize
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        # Concatenate: [z_i; z_j] shape (2B, D)
        z = torch.cat([z_i, z_j], dim=0)
        
        # Full similarity matrix (2B, 2B)
        sim = torch.mm(z, z.t()) / self.temperature
        
        # Get the positive pairs
        # For z_i[k], positive is z_j[k] (at index B+k)
        # For z_j[k], positive is z_i[k] (at index k)
        sim_i_j = torch.diag(sim, batch_size)  # (B,) - z_i to z_j similarities
        sim_j_i = torch.diag(sim, -batch_size)  # (B,) - z_j to z_i similarities
        
        # Positive similarities for all 2B samples
        positives = torch.cat([sim_i_j, sim_j_i], dim=0)  # (2B,)
        
        # Mask to remove self-similarity
        mask = ~torch.eye(2 * batch_size, dtype=torch.bool, device=device)
        
        # Sum of exponentials for denominator (excluding self)
        negatives = sim.masked_select(mask).view(2 * batch_size, -1)  # (2B, 2B-1)
        
        # Log-sum-exp for numerical stability
        logsumexp_neg = torch.logsumexp(negatives, dim=1)  # (2B,)
        
        # Also include positive in denominator
        # logsumexp(positives ∪ negatives) = logsumexp([pos, logsumexp_neg])
        max_val = torch.max(positives, logsumexp_neg)
        logsumexp_all = max_val + torch.log(
            torch.exp(positives - max_val) + torch.exp(logsumexp_neg - max_val)
        )
        
        # Loss: -log(exp(pos) / exp(logsumexp_all)) = logsumexp_all - pos
        loss = (logsumexp_all - positives).mean()
        
        return loss
```

### Information-Theoretic Interpretation

InfoNCE provides a lower bound on mutual information:

$$I(X; Y) \geq \log(K+1) - \mathcal{L}_{\text{InfoNCE}}$$

where $K$ is the number of negative samples.

This means:
- Minimizing InfoNCE loss maximizes mutual information between views
- More negatives ($K$) provide a tighter bound
- The bound becomes equality as $K \to \infty$

```python
def compute_mutual_information_bound(loss, num_negatives):
    """
    Compute the lower bound on mutual information from InfoNCE loss.
    
    The InfoNCE loss provides a lower bound:
    I(X; Y) >= log(K + 1) - L_InfoNCE
    
    Args:
        loss: InfoNCE loss value
        num_negatives: Number of negative samples (K)
    
    Returns:
        mi_lower_bound: Lower bound on mutual information in nats
    """
    mi_lower_bound = math.log(num_negatives + 1) - loss
    
    # Convert to bits
    mi_lower_bound_bits = mi_lower_bound / math.log(2)
    
    return mi_lower_bound, mi_lower_bound_bits
```

## Core Components

### 1. Encoder Architecture

The encoder maps inputs to a representation space:

```python
class ContrastiveEncoder(nn.Module):
    """
    Encoder for contrastive learning.
    
    Architecture: Backbone -> Global Average Pooling -> Projection Head
    
    The projection head is crucial for contrastive learning:
    - Original SimCLR paper shows significant improvement with projection head
    - Representations before projection head transfer better
    """
    def __init__(self, backbone, feature_dim=128, hidden_dim=2048):
        super().__init__()
        
        # Backbone encoder (e.g., ResNet without final FC)
        self.backbone = backbone
        
        # Determine backbone output dimension
        # Try forward pass with dummy input
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            out = self.backbone(dummy)
            if len(out.shape) > 2:
                out = F.adaptive_avg_pool2d(out, 1).flatten(1)
            backbone_dim = out.shape[1]
        
        # Projection head (MLP)
        # Maps backbone features to contrastive embedding space
        self.projection_head = nn.Sequential(
            nn.Linear(backbone_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim),
        )
    
    def forward(self, x, return_features=False):
        """
        Forward pass.
        
        Args:
            x: Input images (B, C, H, W)
            return_features: If True, return both features and projections
        
        Returns:
            z: Projected embeddings (B, feature_dim)
            h: (optional) Backbone features before projection
        """
        # Backbone features
        h = self.backbone(x)
        
        # Global average pooling if needed
        if len(h.shape) > 2:
            h = F.adaptive_avg_pool2d(h, 1).flatten(1)
        
        # Project to contrastive space
        z = self.projection_head(h)
        
        if return_features:
            return z, h
        return z


class ProjectionHead(nn.Module):
    """
    MLP Projection Head for contrastive learning.
    
    Design choices:
    - 2-3 layers with BN and ReLU
    - Hidden dimension typically same as backbone output
    - Output dimension typically smaller (128-256)
    - No output normalization (done in loss function)
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        
        layers = []
        
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = output_dim if i == num_layers - 1 else hidden_dim
            
            layers.extend([
                nn.Linear(in_dim, out_dim),
            ])
            
            # No BN/ReLU on final layer
            if i < num_layers - 1:
                layers.extend([
                    nn.BatchNorm1d(out_dim),
                    nn.ReLU(inplace=True),
                ])
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x)
```

### 2. Data Augmentation Pipeline

Data augmentation is critical for defining positive pairs:

```python
import torchvision.transforms as transforms
from PIL import ImageFilter
import random

class GaussianBlur:
    """Gaussian blur augmentation in SimCLR."""
    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = sigma
    
    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class ContrastiveAugmentation:
    """
    Augmentation pipeline for contrastive learning.
    
    Key insight: The choice of augmentations defines what invariances
    the model should learn. Common invariances:
    - Geometric: crop, flip, rotation
    - Photometric: color jitter, grayscale
    - Blur/noise: Gaussian blur, noise injection
    
    SimCLR paper findings:
    - Color distortion is most important
    - Random crop is crucial
    - Gaussian blur helps
    - Composition of augmentations > individual augmentations
    """
    def __init__(self, 
                 size=224,
                 scale=(0.2, 1.0),
                 color_strength=1.0,
                 blur_prob=0.5):
        
        # Color jitter parameters
        color_jitter = transforms.ColorJitter(
            brightness=0.8 * color_strength,
            contrast=0.8 * color_strength,
            saturation=0.8 * color_strength,
            hue=0.2 * color_strength
        )
        
        self.augmentation = transforms.Compose([
            # Random resized crop - most important augmentation
            transforms.RandomResizedCrop(size=size, scale=scale),
            
            # Random horizontal flip
            transforms.RandomHorizontalFlip(p=0.5),
            
            # Color jitter with probability 0.8
            transforms.RandomApply([color_jitter], p=0.8),
            
            # Random grayscale
            transforms.RandomGrayscale(p=0.2),
            
            # Gaussian blur
            transforms.RandomApply([GaussianBlur(sigma=(0.1, 2.0))], p=blur_prob),
            
            # Convert to tensor and normalize
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    
    def __call__(self, x):
        """
        Apply augmentation twice to create two views.
        
        Args:
            x: Input image (PIL Image)
        
        Returns:
            view1, view2: Two augmented views of the input
        """
        return self.augmentation(x), self.augmentation(x)


class ContrastiveDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper for contrastive learning.
    
    Applies augmentation twice to each image to create positive pairs.
    """
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        image, _ = self.base_dataset[idx]
        
        # Create two augmented views
        view1, view2 = self.transform(image)
        
        return view1, view2
```

### 3. Temperature Parameter

Temperature controls the sharpness of the softmax distribution:

```python
def analyze_temperature_effect(similarities, temperatures=[0.01, 0.07, 0.5, 1.0]):
    """
    Analyze how temperature affects the contrastive loss landscape.
    
    Low temperature (τ → 0):
    - Sharper distribution
    - Model focuses on hard negatives
    - Can lead to instability
    
    High temperature (τ → ∞):
    - Flatter distribution
    - All negatives contribute equally
    - May not learn fine distinctions
    
    Args:
        similarities: Similarity scores (B, K+1) where K is num negatives
        temperatures: List of temperatures to analyze
    
    Returns:
        Analysis results for each temperature
    """
    results = {}
    
    for temp in temperatures:
        # Scaled similarities
        scaled_sim = similarities / temp
        
        # Softmax probabilities
        probs = F.softmax(scaled_sim, dim=1)
        
        # Entropy of the distribution
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1).mean()
        
        # How much probability mass on the positive (index 0)
        positive_prob = probs[:, 0].mean()
        
        # Effective number of negatives (based on entropy)
        effective_negatives = torch.exp(entropy)
        
        results[temp] = {
            'entropy': entropy.item(),
            'positive_prob': positive_prob.item(),
            'effective_negatives': effective_negatives.item(),
            'gradient_magnitude': (probs * (1 - probs)).mean().item(),  # Sigmoid-like
        }
        
        print(f"Temperature τ={temp}:")
        print(f"  Entropy: {results[temp]['entropy']:.4f}")
        print(f"  P(positive): {results[temp]['positive_prob']:.4f}")
        print(f"  Effective negatives: {results[temp]['effective_negatives']:.2f}")
    
    return results


class LearnableTemperature(nn.Module):
    """
    Learnable temperature parameter.
    
    Some methods (e.g., CLIP) learn the temperature jointly with the model.
    Initialize to log(1/0.07) ≈ 2.66 for initial temperature of 0.07.
    """
    def __init__(self, init_temp=0.07, min_temp=0.01, max_temp=0.5):
        super().__init__()
        
        # Store as log for numerical stability and easier optimization
        self.log_temp = nn.Parameter(torch.tensor(math.log(init_temp)))
        
        # Bounds
        self.log_min = math.log(min_temp)
        self.log_max = math.log(max_temp)
    
    @property
    def temperature(self):
        """Get current temperature value (clamped)."""
        clamped_log = torch.clamp(self.log_temp, self.log_min, self.log_max)
        return torch.exp(clamped_log)
    
    def forward(self, similarities):
        """Scale similarities by temperature."""
        return similarities / self.temperature
```

### 4. Hard Negative Mining

Not all negatives are equally informative:

```python
class HardNegativeMining:
    """
    Hard negative mining strategies for contrastive learning.
    
    Hard negatives: Samples that are similar to the query but from
    different classes (or different instances in self-supervised setting).
    
    Semi-hard negatives: Negatives that are further than positive but
    still within a margin.
    
    Curriculum: Start with random negatives, gradually introduce harder ones.
    """
    
    @staticmethod
    def select_hard_negatives(query, keys, labels=None, num_hard=None, 
                              margin=None, strategy='hardest'):
        """
        Select hard negative samples.
        
        Args:
            query: Query embeddings (B, D)
            keys: Key embeddings (N, D), typically from memory bank
            labels: Optional labels for supervised setting
            num_hard: Number of hard negatives to select
            margin: Margin for semi-hard negative mining
            strategy: 'hardest', 'semi-hard', 'curriculum'
        
        Returns:
            hard_negative_indices: Indices of selected hard negatives
        """
        # Compute similarities
        similarities = torch.mm(query, keys.t())  # (B, N)
        
        if labels is not None:
            # Mask out same-class samples
            label_mask = labels.unsqueeze(1) == labels.unsqueeze(0)
            similarities = similarities.masked_fill(label_mask, -float('inf'))
        
        B, N = similarities.shape
        
        if strategy == 'hardest':
            # Select the most similar negatives (hardest)
            if num_hard is None:
                num_hard = min(100, N)
            
            _, indices = similarities.topk(num_hard, dim=1, largest=True)
            return indices
        
        elif strategy == 'semi-hard':
            # Select negatives within a margin from positive
            if margin is None:
                margin = 0.1
            
            # Assuming positive similarity is in diagonal or first position
            pos_sim = similarities.diag() if labels is not None else similarities[:, 0]
            
            # Semi-hard: positive - margin < negative < positive
            mask = (similarities < pos_sim.unsqueeze(1)) & \
                   (similarities > pos_sim.unsqueeze(1) - margin)
            
            # Randomly sample from valid negatives
            indices = []
            for b in range(B):
                valid = mask[b].nonzero(as_tuple=True)[0]
                if len(valid) == 0:
                    # Fall back to hardest if no semi-hard
                    valid = similarities[b].topk(num_hard or 10, largest=True)[1]
                else:
                    valid = valid[torch.randperm(len(valid))[:num_hard or 10]]
                indices.append(valid)
            
            return indices
        
        elif strategy == 'curriculum':
            # Mix of random and hard based on training progress
            raise NotImplementedError("Curriculum strategy requires training state")
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    @staticmethod
    def debiased_contrastive_loss(z_i, z_j, temperature=0.5, tau_plus=0.1):
        """
        Debiased contrastive loss (DCL) from Chuang et al.
        
        Corrects for the sampling bias in standard contrastive loss
        where some "negatives" might actually be semantically similar.
        
        Args:
            z_i: First view embeddings (B, D)
            z_j: Second view embeddings (B, D)
            temperature: Temperature parameter
            tau_plus: Estimated probability of positive pairs in negatives
        
        Returns:
            Debiased contrastive loss
        """
        batch_size = z_i.shape[0]
        device = z_i.device
        
        # Normalize
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        # Concatenate
        z = torch.cat([z_i, z_j], dim=0)
        
        # Similarity matrix
        sim = torch.mm(z, z.t()) / temperature
        
        # Positive similarities
        pos_sim = torch.cat([
            torch.diag(sim, batch_size),
            torch.diag(sim, -batch_size)
        ], dim=0)
        
        # Sum of negative similarities (excluding diagonal)
        mask = ~torch.eye(2 * batch_size, dtype=torch.bool, device=device)
        neg_sim = sim[mask].view(2 * batch_size, -1)
        
        # Debiased negative term
        N = 2 * batch_size - 2
        neg_exp = torch.exp(neg_sim)
        
        # Correction factor
        correction = N * tau_plus
        
        # Debiased denominator
        debiased_neg = (neg_exp.sum(dim=1) - correction * torch.exp(pos_sim)) / (1 - tau_plus)
        debiased_neg = torch.clamp(debiased_neg, min=N * math.exp(-1 / temperature))
        
        # Loss
        loss = -pos_sim + torch.log(torch.exp(pos_sim) + debiased_neg)
        
        return loss.mean()
```

## Training Procedure

```python
def train_contrastive_epoch(model, dataloader, optimizer, criterion, 
                            device, scheduler=None, epoch=0):
    """
    Train contrastive learning model for one epoch.
    
    Args:
        model: ContrastiveEncoder model
        dataloader: Data loader yielding (view1, view2) pairs
        optimizer: Optimizer
        criterion: Contrastive loss function
        device: Device
        scheduler: Optional learning rate scheduler
        epoch: Current epoch (for logging)
    
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0
    total_samples = 0
    
    for batch_idx, (view1, view2) in enumerate(dataloader):
        view1 = view1.to(device)
        view2 = view2.to(device)
        
        # Forward pass
        z1 = model(view1)
        z2 = model(view2)
        
        # Compute loss
        loss = criterion(z1, z2)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update scheduler (if per-iteration)
        if scheduler is not None and hasattr(scheduler, 'step_batch'):
            scheduler.step_batch()
        
        # Track metrics
        batch_size = view1.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        
        # Logging
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}: Loss = {loss.item():.4f}")
    
    # Update scheduler (if per-epoch)
    if scheduler is not None and not hasattr(scheduler, 'step_batch'):
        scheduler.step()
    
    avg_loss = total_loss / total_samples
    return avg_loss


def linear_evaluation(encoder, train_loader, test_loader, device, 
                      num_classes=10, epochs=100, lr=0.1):
    """
    Linear evaluation protocol for contrastive learning.
    
    Freeze encoder and train a linear classifier on top.
    This is the standard way to evaluate learned representations.
    
    Args:
        encoder: Trained encoder (will be frozen)
        train_loader: Training data loader
        test_loader: Test data loader
        device: Device
        num_classes: Number of output classes
        epochs: Training epochs for linear classifier
        lr: Learning rate for linear classifier
    
    Returns:
        Test accuracy
    """
    encoder.eval()
    
    # Extract features
    print("Extracting features...")
    train_features, train_labels = extract_features(encoder, train_loader, device)
    test_features, test_labels = extract_features(encoder, test_loader, device)
    
    # Create linear classifier
    feature_dim = train_features.shape[1]
    classifier = nn.Linear(feature_dim, num_classes).to(device)
    
    optimizer = torch.optim.SGD(classifier.parameters(), lr=lr, 
                                 momentum=0.9, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    # Train linear classifier
    print("Training linear classifier...")
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(train_features), torch.tensor(train_labels)
    )
    train_loader_linear = torch.utils.data.DataLoader(
        train_dataset, batch_size=256, shuffle=True
    )
    
    for epoch in range(epochs):
        classifier.train()
        for features, labels in train_loader_linear:
            features = features.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits = classifier(features)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
    
    # Evaluate
    classifier.eval()
    test_features_tensor = torch.tensor(test_features).to(device)
    test_labels_tensor = torch.tensor(test_labels).to(device)
    
    with torch.no_grad():
        logits = classifier(test_features_tensor)
        predictions = logits.argmax(dim=1)
        accuracy = (predictions == test_labels_tensor).float().mean().item() * 100
    
    print(f"Linear evaluation accuracy: {accuracy:.2f}%")
    return accuracy


def extract_features(encoder, dataloader, device):
    """
    Extract features using the encoder backbone (before projection head).
    
    Args:
        encoder: ContrastiveEncoder model
        dataloader: Data loader
        device: Device
    
    Returns:
        features: numpy array (N, D)
        labels: numpy array (N,)
    """
    features_list = []
    labels_list = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            
            # Get features before projection head
            _, features = encoder(images, return_features=True)
            
            features_list.append(features.cpu().numpy())
            labels_list.append(labels.numpy())
    
    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    
    return features, labels
```

## Theoretical Analysis

### Why Does Contrastive Learning Work?

Several theoretical perspectives explain the success of contrastive learning:

```python
def alignment_uniformity_metrics(z, positive_pairs):
    """
    Compute alignment and uniformity metrics.
    
    Wang & Isola (2020) show that contrastive loss optimizes:
    1. Alignment: Positive pairs should have similar representations
    2. Uniformity: Features should be uniformly distributed on the hypersphere
    
    Args:
        z: Embeddings (N, D), normalized
        positive_pairs: List of (i, j) index pairs
    
    Returns:
        alignment: Average distance between positive pairs (lower is better)
        uniformity: Log average pairwise similarity (lower = more uniform)
    """
    z = F.normalize(z, dim=1)
    
    # Alignment: E[||f(x) - f(x+)||^2]
    alignment_distances = []
    for i, j in positive_pairs:
        dist = (z[i] - z[j]).pow(2).sum()
        alignment_distances.append(dist)
    
    alignment = torch.stack(alignment_distances).mean()
    
    # Uniformity: log E[e^{-2||f(x) - f(x')||^2}]
    # Higher uniformity = features more evenly spread
    sq_pdist = torch.cdist(z, z, p=2).pow(2)
    
    # Exclude diagonal (self-distances)
    n = z.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=z.device)
    sq_pdist = sq_pdist[mask].view(n, n-1)
    
    uniformity = torch.log(torch.exp(-2 * sq_pdist).mean())
    
    return alignment.item(), uniformity.item()


class ContrastiveLossAnalysis:
    """
    Analysis tools for understanding contrastive loss behavior.
    """
    
    @staticmethod
    def gradient_analysis(z_i, z_j, temperature=0.5):
        """
        Analyze gradients of contrastive loss.
        
        Key insight: Gradient pushes representations of negatives apart
        proportional to their similarity (harder negatives get larger gradients).
        
        Args:
            z_i: First view embeddings (B, D), requires_grad=True
            z_j: Second view embeddings (B, D)
            temperature: Temperature
        
        Returns:
            Gradient statistics
        """
        z_i = z_i.clone().requires_grad_(True)
        z_j = z_j.clone().requires_grad_(True)
        
        # Compute loss
        z_i_norm = F.normalize(z_i, dim=1)
        z_j_norm = F.normalize(z_j, dim=1)
        
        z = torch.cat([z_i_norm, z_j_norm], dim=0)
        sim = torch.mm(z, z.t()) / temperature
        
        B = z_i.shape[0]
        labels = torch.cat([
            torch.arange(B) + B,
            torch.arange(B)
        ])
        
        mask = ~torch.eye(2 * B, dtype=torch.bool)
        sim = sim.masked_fill(~mask, -float('inf'))
        
        loss = F.cross_entropy(sim, labels)
        loss.backward()
        
        # Analyze gradients
        grad_norm_i = z_i.grad.norm(dim=1).mean()
        grad_norm_j = z_j.grad.norm(dim=1).mean()
        
        # Correlation between similarity and gradient magnitude
        with torch.no_grad():
            neg_sims = sim[:B, B:][mask[:B, B:]].view(B, B-1)
            neg_grad_contribution = torch.abs(z_i.grad).sum(dim=1)
        
        return {
            'mean_grad_norm_i': grad_norm_i.item(),
            'mean_grad_norm_j': grad_norm_j.item(),
            'loss': loss.item(),
        }
    
    @staticmethod
    def representation_quality(z, labels):
        """
        Measure quality of learned representations.
        
        Metrics:
        1. Intra-class compactness: How close same-class samples are
        2. Inter-class separability: How far different-class samples are
        3. Silhouette score: Combined measure
        
        Args:
            z: Embeddings (N, D)
            labels: Class labels (N,)
        
        Returns:
            Quality metrics dictionary
        """
        from sklearn.metrics import silhouette_score
        
        z_norm = F.normalize(z, dim=1).numpy()
        labels_np = labels.numpy() if isinstance(labels, torch.Tensor) else labels
        
        # Compute pairwise distances
        distances = torch.cdist(
            torch.tensor(z_norm), 
            torch.tensor(z_norm), 
            p=2
        ).numpy()
        
        # Intra-class distances (compactness)
        unique_labels = np.unique(labels_np)
        intra_distances = []
        for label in unique_labels:
            mask = labels_np == label
            intra_dist = distances[mask][:, mask]
            intra_distances.append(intra_dist[np.triu_indices_from(intra_dist, k=1)].mean())
        
        avg_intra = np.mean(intra_distances)
        
        # Inter-class distances (separability)
        inter_distances = []
        for i, label_i in enumerate(unique_labels):
            for label_j in unique_labels[i+1:]:
                mask_i = labels_np == label_i
                mask_j = labels_np == label_j
                inter_dist = distances[mask_i][:, mask_j].mean()
                inter_distances.append(inter_dist)
        
        avg_inter = np.mean(inter_distances)
        
        # Silhouette score
        try:
            silhouette = silhouette_score(z_norm, labels_np)
        except:
            silhouette = 0.0
        
        return {
            'intra_class_distance': avg_intra,
            'inter_class_distance': avg_inter,
            'separability_ratio': avg_inter / (avg_intra + 1e-6),
            'silhouette_score': silhouette,
        }
```

## Summary

Contrastive learning provides a principled framework for self-supervised representation learning:

| Aspect | Key Insight |
|--------|-------------|
| **Objective** | Pull positive pairs together, push negatives apart |
| **Loss** | InfoNCE/NT-Xent provides lower bound on mutual information |
| **Temperature** | Controls hardness of the contrastive task |
| **Augmentation** | Defines what invariances to learn |
| **Batch Size** | More negatives = tighter MI bound |
| **Evaluation** | Linear probing on frozen features |

The subsequent sections will cover specific instantiations of this framework: SimCLR, MoCo, BYOL, and others.

## References

1. Chen, T., et al. (2020). A Simple Framework for Contrastive Learning of Visual Representations. ICML.
2. He, K., et al. (2020). Momentum Contrast for Unsupervised Visual Representation Learning. CVPR.
3. Oord, A. v. d., Li, Y., & Vinyals, O. (2018). Representation Learning with Contrastive Predictive Coding. arXiv.
4. Wang, T., & Isola, P. (2020). Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere. ICML.
5. Chuang, C. Y., et al. (2020). Debiased Contrastive Learning. NeurIPS.
