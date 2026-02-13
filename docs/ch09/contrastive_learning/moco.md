# MoCo: Momentum Contrast for Unsupervised Visual Representation Learning

## Learning Objectives

By the end of this section, you will be able to:

- Understand the motivation behind MoCo and the dictionary look-up perspective
- Implement MoCo v1/v2/v3 architectures in PyTorch
- Explain the momentum encoder and queue mechanism
- Compare MoCo's memory bank approach with SimCLR's in-batch negatives
- Train MoCo models with limited computational resources

## Introduction

MoCo (Momentum Contrast) approaches contrastive learning from the perspective of building a **dynamic dictionary**. Unlike SimCLR which requires large batch sizes for many negatives, MoCo maintains a queue of encoded keys, enabling effective contrastive learning with smaller batches.

### Key Insight

Contrastive learning can be viewed as training an encoder to perform dictionary look-up:

- **Query**: Encoded representation of the input
- **Keys**: Encoded representations in the dictionary
- **Positive key**: The matching key for the query
- **Negative keys**: Non-matching keys

The challenge is building a dictionary that is:
1. **Large**: More negatives improve the contrastive learning bound
2. **Consistent**: Keys should be encoded by a similar encoder

## MoCo Architecture Evolution

### MoCo v1: Queue + Momentum Encoder

```
Query Encoder f_q                 Key Encoder f_k (momentum-updated)
      │                                      │
      ▼                                      ▼
   ┌─────┐                              ┌─────┐
   │  q  │                              │  k  │ ← Current batch keys
   └──┬──┘                              └──┬──┘
      │                                    │
      │            ┌──────────────────────┘
      │            │
      ▼            ▼
   ┌─────────────────────────────────────────────┐
   │              Dictionary (Queue)              │
   │  [k₁, k₂, k₃, ..., k_K]                     │
   │   ↑                    ↑                     │
   │  oldest              newest                  │
   └─────────────────────────────────────────────┘
                      │
                      ▼
              InfoNCE Loss
```

### MoCo v2: + SimCLR Augmentations + MLP Head

### MoCo v3: - Queue + Symmetric Loss (SimSiam-style)

## Complete Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import copy
from PIL import ImageFilter
import random


# =============================================================================
# Data Augmentation (same as SimCLR v2 for MoCo v2+)
# =============================================================================

class GaussianBlur:
    """Gaussian blur augmentation."""
    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = sigma
    
    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        return x.filter(ImageFilter.GaussianBlur(radius=sigma))


class MoCoAugmentation:
    """
    Data augmentation for MoCo v2.
    
    MoCo v2 adopts SimCLR's augmentation strategy:
    - Random resized crop
    - Color jitter
    - Grayscale
    - Gaussian blur (applied with probability 0.5)
    """
    def __init__(self, size=224):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur()], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def __call__(self, x):
        """Return two augmented views."""
        return self.transform(x), self.transform(x)


# =============================================================================
# MoCo v1/v2
# =============================================================================

class MoCo(nn.Module):
    """
    Momentum Contrast (MoCo) v1/v2
    
    Key components:
    1. Query encoder f_q: Standard encoder, gradients flow through
    2. Key encoder f_k: Momentum-updated copy, no gradients
    3. Queue: FIFO queue of encoded keys from previous batches
    
    The momentum update ensures keys are encoded consistently:
        θ_k ← m * θ_k + (1 - m) * θ_q
    
    Args:
        base_encoder: Backbone architecture
        dim: Feature dimension (output of projection head)
        K: Queue size (number of negative keys)
        m: Momentum coefficient for key encoder update
        T: Temperature for contrastive loss
        mlp: Whether to use MLP projection head (MoCo v2)
    """
    def __init__(
        self,
        base_encoder='resnet50',
        dim=128,
        K=65536,
        m=0.999,
        T=0.07,
        mlp=True  # True for MoCo v2
    ):
        super().__init__()
        
        self.K = K
        self.m = m
        self.T = T
        
        # Build encoders
        self.encoder_q, encoder_dim = self._build_encoder(base_encoder)
        self.encoder_k, _ = self._build_encoder(base_encoder)
        
        # Build projection heads
        if mlp:
            # MoCo v2: 2-layer MLP head
            self.encoder_q.fc = nn.Sequential(
                nn.Linear(encoder_dim, encoder_dim),
                nn.ReLU(),
                nn.Linear(encoder_dim, dim)
            )
            self.encoder_k.fc = nn.Sequential(
                nn.Linear(encoder_dim, encoder_dim),
                nn.ReLU(),
                nn.Linear(encoder_dim, dim)
            )
        else:
            # MoCo v1: Linear head
            self.encoder_q.fc = nn.Linear(encoder_dim, dim)
            self.encoder_k.fc = nn.Linear(encoder_dim, dim)
        
        # Initialize key encoder with query encoder weights
        for param_q, param_k in zip(self.encoder_q.parameters(), 
                                     self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False  # Key encoder not updated by gradients
        
        # Create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = F.normalize(self.queue, dim=0)
        
        # Queue pointer
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
    
    def _build_encoder(self, base_encoder):
        """Build encoder backbone."""
        if base_encoder == 'resnet50':
            encoder = models.resnet50(weights=None)
            dim = 2048
        elif base_encoder == 'resnet18':
            encoder = models.resnet18(weights=None)
            dim = 512
        else:
            raise ValueError(f"Unknown encoder: {base_encoder}")
        return encoder, dim
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder.
        
        θ_k ← m * θ_k + (1 - m) * θ_q
        
        This provides slowly evolving keys, ensuring consistency.
        """
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                     self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """
        Update the queue with new keys.
        
        FIFO: Remove oldest keys, add newest keys.
        """
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        
        # Ensure queue size is divisible by batch size
        assert self.K % batch_size == 0, "Queue size must be divisible by batch size"
        
        # Replace keys at ptr position
        self.queue[:, ptr:ptr + batch_size] = keys.T
        
        # Move pointer
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr
    
    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle for distributed training.
        
        This prevents batch normalization from leaking information
        between samples in distributed training.
        
        In single-GPU training, this is a simple random shuffle.
        """
        batch_size = x.shape[0]
        
        # Random shuffle indices
        idx_shuffle = torch.randperm(batch_size, device=x.device)
        
        # Index to restore original order
        idx_unshuffle = torch.argsort(idx_shuffle)
        
        return x[idx_shuffle], idx_unshuffle
    
    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """Undo batch shuffle."""
        return x[idx_unshuffle]
    
    def forward(self, im_q, im_k):
        """
        Forward pass.
        
        Args:
            im_q: Query images (B, 3, H, W)
            im_k: Key images (B, 3, H, W)
        
        Returns:
            logits: Similarity logits (B, K+1)
            labels: Ground truth labels (B,) - always 0 (positive is first)
        """
        # Compute query features
        q = self.encoder_q(im_q)  # (B, dim)
        q = F.normalize(q, dim=1)
        
        # Compute key features
        with torch.no_grad():
            # Update key encoder
            self._momentum_update_key_encoder()
            
            # Shuffle for batch norm
            im_k_shuffled, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            
            # Encode keys
            k = self.encoder_k(im_k_shuffled)  # (B, dim)
            k = F.normalize(k, dim=1)
            
            # Unshuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)
        
        # Positive logits: Nx1
        # Query and its corresponding key
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)  # (B, 1)
        
        # Negative logits: NxK
        # Query against all keys in queue
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])  # (B, K)
        
        # Concatenate: logits = [positive, negatives]
        logits = torch.cat([l_pos, l_neg], dim=1)  # (B, 1+K)
        
        # Apply temperature
        logits /= self.T
        
        # Labels: positive is always at index 0
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        
        # Update queue
        self._dequeue_and_enqueue(k)
        
        return logits, labels


def train_moco_epoch(model, dataloader, optimizer, device, epoch):
    """
    Train MoCo for one epoch.
    
    Args:
        model: MoCo model
        dataloader: Data loader yielding (query_imgs, key_imgs)
        optimizer: Optimizer
        device: Device
        epoch: Current epoch
    
    Returns:
        Average loss and accuracy
    """
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0
    
    for batch_idx, (im_q, im_k) in enumerate(dataloader):
        im_q = im_q.to(device)
        im_k = im_k.to(device)
        
        # Forward pass
        logits, labels = model(im_q, im_k)
        
        # Compute loss
        loss = F.cross_entropy(logits, labels)
        
        # Compute accuracy (how often positive is ranked first)
        acc = (logits.argmax(dim=1) == labels).float().mean()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_acc += acc.item()
        num_batches += 1
        
        if batch_idx % 100 == 0:
            print(f"  Epoch {epoch}, Batch {batch_idx}: "
                  f"Loss = {loss.item():.4f}, Acc = {acc.item():.4f}")
    
    return total_loss / num_batches, total_acc / num_batches


# =============================================================================
# MoCo v3
# =============================================================================

class MoCoV3(nn.Module):
    """
    MoCo v3: An Empirical Study of Training Self-Supervised Vision Transformers
    
    Key differences from MoCo v1/v2:
    1. No queue - uses large batches like SimCLR
    2. Symmetric loss - both views are queries and keys
    3. Prediction head - like BYOL/SimSiam
    4. Works with Vision Transformers
    
    Architecture:
        x1 -> f -> g -> z1 -> h -> p1   } Symmetric
        x2 -> f -> g -> z2 -> h -> p2   } loss
    
    Loss: sim(p1, z2.detach()) + sim(p2, z1.detach())
    
    Args:
        base_encoder: Backbone architecture
        dim: Dimension of projection output
        pred_dim: Dimension of prediction head hidden layer
        T: Temperature
    """
    def __init__(
        self,
        base_encoder='resnet50',
        dim=256,
        mlp_dim=4096,
        pred_dim=4096,
        T=1.0
    ):
        super().__init__()
        
        self.T = T
        
        # Build backbone
        if base_encoder == 'resnet50':
            self.encoder = models.resnet50(weights=None)
            encoder_dim = 2048
            self.encoder.fc = nn.Identity()
        else:
            raise ValueError(f"Unknown encoder: {base_encoder}")
        
        # Projection head g(·): 3-layer MLP
        self.projector = nn.Sequential(
            nn.Linear(encoder_dim, mlp_dim),
            nn.BatchNorm1d(mlp_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, mlp_dim),
            nn.BatchNorm1d(mlp_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, dim),
            nn.BatchNorm1d(dim, affine=False)  # No learnable affine params
        )
        
        # Prediction head h(·): 2-layer MLP
        self.predictor = nn.Sequential(
            nn.Linear(dim, pred_dim),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pred_dim, dim)
        )
    
    def forward(self, x1, x2):
        """
        Forward pass with symmetric loss.
        
        Args:
            x1, x2: Two views of images
        
        Returns:
            loss: Contrastive loss
        """
        # Encode both views
        f1 = self.encoder(x1)
        f2 = self.encoder(x2)
        
        # Project
        z1 = self.projector(f1)
        z2 = self.projector(f2)
        
        # Predict
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        
        # Symmetric loss
        loss = self.contrastive_loss(p1, z2) + self.contrastive_loss(p2, z1)
        loss = loss / 2
        
        return loss
    
    def contrastive_loss(self, q, k):
        """
        InfoNCE loss with detached keys.
        
        Args:
            q: Predictions (B, D)
            k: Projections (B, D) - will be detached
        
        Returns:
            loss: Contrastive loss
        """
        # Normalize
        q = F.normalize(q, dim=1)
        k = F.normalize(k, dim=1).detach()  # Stop gradient to keys
        
        # Cosine similarity
        # In distributed training, keys would be gathered from all GPUs
        logits = torch.mm(q, k.T) / self.T  # (B, B)
        
        # Labels: diagonal is positive
        labels = torch.arange(q.shape[0], device=q.device)
        
        return F.cross_entropy(logits, labels)


# =============================================================================
# Comparison Analysis
# =============================================================================

class MoCoAnalysis:
    """
    Analysis tools for understanding MoCo design choices.
    """
    
    @staticmethod
    def analyze_queue_dynamics(queue_size, batch_size, num_epochs, samples_per_epoch):
        """
        Analyze how the queue evolves during training.
        
        Args:
            queue_size: Size of the queue (K)
            batch_size: Batch size
            num_epochs: Number of training epochs
            samples_per_epoch: Number of samples per epoch
        
        Returns:
            Analysis statistics
        """
        batches_per_epoch = samples_per_epoch // batch_size
        total_batches = num_epochs * batches_per_epoch
        
        # How many times the queue is completely refreshed
        queue_refreshes = (total_batches * batch_size) / queue_size
        
        # Average "age" of keys in the queue (in batches)
        avg_age_batches = queue_size / (2 * batch_size)
        
        # Age in epochs
        avg_age_epochs = avg_age_batches / batches_per_epoch
        
        print("MoCo Queue Analysis:")
        print(f"  Queue size: {queue_size}")
        print(f"  Batch size: {batch_size}")
        print(f"  Total batches: {total_batches}")
        print(f"  Queue refreshes: {queue_refreshes:.1f}")
        print(f"  Average key age: {avg_age_batches:.1f} batches "
              f"({avg_age_epochs:.3f} epochs)")
        
        return {
            'queue_refreshes': queue_refreshes,
            'avg_age_batches': avg_age_batches,
            'avg_age_epochs': avg_age_epochs
        }
    
    @staticmethod
    def analyze_momentum_effect(momentum, num_epochs, updates_per_epoch):
        """
        Analyze how momentum affects encoder divergence.
        
        With momentum m, after n updates the key encoder has:
        θ_k = Σ_{i=0}^{n-1} (1-m) * m^i * θ_q(n-1-i)
        
        This is an exponential moving average with effective window ~1/(1-m).
        """
        total_updates = num_epochs * updates_per_epoch
        
        # Effective window size
        effective_window = 1 / (1 - momentum)
        
        # Half-life in updates
        half_life = -math.log(0.5) / math.log(momentum)
        half_life = math.log(2) / (1 - momentum)  # Approximation
        
        print("Momentum Analysis:")
        print(f"  Momentum coefficient: {momentum}")
        print(f"  Effective window: {effective_window:.0f} updates")
        print(f"  Half-life: {half_life:.0f} updates")
        print(f"  Total updates: {total_updates}")
        
        # After training, what fraction of old weights remain?
        remaining_old = momentum ** total_updates
        print(f"  Remaining initial weight after training: {remaining_old:.2e}")
    
    @staticmethod
    def compare_moco_simclr(batch_size_simclr, queue_size_moco, batch_size_moco):
        """
        Compare effective number of negatives in MoCo vs SimCLR.
        
        SimCLR: 2(N-1) negatives per sample (both views of other samples)
        MoCo: K negatives from queue
        """
        simclr_negatives = 2 * (batch_size_simclr - 1)
        moco_negatives = queue_size_moco
        
        print("MoCo vs SimCLR Comparison:")
        print(f"\nSimCLR (batch size {batch_size_simclr}):")
        print(f"  Negatives per sample: {simclr_negatives}")
        print(f"  GPU memory: O(N²) for similarity matrix")
        
        print(f"\nMoCo (queue size {queue_size_moco}, batch size {batch_size_moco}):")
        print(f"  Negatives per sample: {moco_negatives}")
        print(f"  GPU memory: O(NK) for similarity computation")
        print(f"  Ratio: {moco_negatives / simclr_negatives:.1f}x more negatives")
        
        # Memory comparison (rough)
        simclr_memory_factor = batch_size_simclr ** 2
        moco_memory_factor = batch_size_moco * queue_size_moco
        
        print(f"\nMemory efficiency:")
        print(f"  SimCLR similarity matrix: {simclr_memory_factor}")
        print(f"  MoCo similarity computation: {moco_memory_factor}")


# =============================================================================
# MoCo vs SimCLR Design Choices
# =============================================================================

"""
+------------------+------------------------+------------------------+
|     Aspect       |       SimCLR           |         MoCo           |
+------------------+------------------------+------------------------+
| Negatives source | In-batch (2N-1)        | Queue (K, e.g., 65536) |
| Batch size req.  | Large (4096)           | Small (256) sufficient |
| Memory usage     | O(N²)                  | O(NK), but K is fixed  |
| Key consistency  | N/A (all same encoder) | Momentum encoder       |
| Loss             | NT-Xent                | InfoNCE                |
| Augmentation     | Strong (v1)            | Same as SimCLR (v2+)   |
| Projection head  | 2-layer MLP            | 2-layer MLP (v2+)      |
+------------------+------------------------+------------------------+
"""


# =============================================================================
# Training
# =============================================================================

def train_moco(
    model,
    train_dataset,
    epochs=200,
    batch_size=256,
    learning_rate=0.03,
    momentum=0.9,
    weight_decay=1e-4,
    device='cuda'
):
    """
    Full MoCo training loop.
    
    Default hyperparameters from MoCo paper:
    - Batch size: 256
    - Learning rate: 0.03
    - SGD momentum: 0.9
    - Weight decay: 1e-4
    - Epochs: 200
    - Queue size: 65536
    - Momentum coefficient: 0.999
    - Temperature: 0.07
    """
    model = model.to(device)
    
    # Create dataset with augmentations
    augmentation = MoCoAugmentation(size=224)
    
    class MoCoDataset(torch.utils.data.Dataset):
        def __init__(self, base_dataset, transform):
            self.base_dataset = base_dataset
            self.transform = transform
        
        def __len__(self):
            return len(self.base_dataset)
        
        def __getitem__(self, idx):
            img, _ = self.base_dataset[idx]
            return self.transform(img)
    
    moco_dataset = MoCoDataset(train_dataset, augmentation)
    
    dataloader = torch.utils.data.DataLoader(
        moco_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    # Optimizer - SGD with momentum
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay
    )
    
    # Learning rate schedule - step decay
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[120, 160],  # Decay at epochs 120 and 160
        gamma=0.1
    )
    
    # Training loop
    losses = []
    accs = []
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(f"  Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        loss, acc = train_moco_epoch(model, dataloader, optimizer, device, epoch)
        
        losses.append(loss)
        accs.append(acc)
        scheduler.step()
        
        print(f"  Average loss: {loss:.4f}")
        print(f"  Average accuracy: {acc:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, f'moco_checkpoint_epoch_{epoch+1}.pt')
    
    return model, losses, accs


# =============================================================================
# Usage Example
# =============================================================================

if __name__ == "__main__":
    import math
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create MoCo v2 model
    print("\n=== MoCo v2 ===")
    moco_v2 = MoCo(
        base_encoder='resnet50',
        dim=128,
        K=65536,
        m=0.999,
        T=0.07,
        mlp=True  # MoCo v2
    )
    
    print(f"Queue size: {moco_v2.K}")
    print(f"Momentum: {moco_v2.m}")
    print(f"Temperature: {moco_v2.T}")
    
    # Test forward pass
    im_q = torch.randn(32, 3, 224, 224)
    im_k = torch.randn(32, 3, 224, 224)
    
    moco_v2.eval()
    logits, labels = moco_v2(im_q, im_k)
    
    print(f"\nForward pass test:")
    print(f"  Input shape: {im_q.shape}")
    print(f"  Logits shape: {logits.shape} (batch_size, 1 + queue_size)")
    print(f"  Labels: {labels}")
    
    # Compute loss
    loss = F.cross_entropy(logits, labels)
    print(f"  Loss: {loss.item():.4f}")
    
    # Create MoCo v3 model
    print("\n=== MoCo v3 ===")
    moco_v3 = MoCoV3(
        base_encoder='resnet50',
        dim=256,
        mlp_dim=4096,
        T=1.0
    )
    
    # Test forward pass
    x1 = torch.randn(32, 3, 224, 224)
    x2 = torch.randn(32, 3, 224, 224)
    
    moco_v3.eval()
    loss_v3 = moco_v3(x1, x2)
    
    print(f"  Loss: {loss_v3.item():.4f}")
    
    # Analysis
    print("\n=== Analysis ===")
    MoCoAnalysis.compare_moco_simclr(
        batch_size_simclr=4096,
        queue_size_moco=65536,
        batch_size_moco=256
    )
    
    print()
    MoCoAnalysis.analyze_momentum_effect(
        momentum=0.999,
        num_epochs=200,
        updates_per_epoch=5000
    )
```

## Summary

| Version | Key Features | Negatives | Batch Size |
|---------|-------------|-----------|------------|
| MoCo v1 | Queue + momentum encoder | 65536 | 256 |
| MoCo v2 | + MLP head + SimCLR augmentations | 65536 | 256 |
| MoCo v3 | No queue + symmetric loss + predictor | In-batch | 4096 |

MoCo's key insight—using a momentum encoder to maintain consistent key representations—has influenced many subsequent methods including BYOL and SimSiam.

## References

1. He, K., Fan, H., Wu, Y., Xie, S., & Girshick, R. (2020). Momentum Contrast for Unsupervised Visual Representation Learning. CVPR.
2. Chen, X., Fan, H., Girshick, R., & He, K. (2020). Improved Baselines with Momentum Contrastive Learning. arXiv.
3. Chen, X., Xie, S., & He, K. (2021). An Empirical Study of Training Self-Supervised Vision Transformers. ICCV.
