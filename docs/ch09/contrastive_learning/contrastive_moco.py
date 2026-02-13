"""
MoCo: Momentum Contrast for Unsupervised Visual Representation Learning
Implementation of MoCo v2 algorithm for self-supervised learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import copy


class MoCo(nn.Module):
    """
    Momentum Contrast (MoCo) Model
    
    Uses a momentum encoder and queue mechanism for efficient contrastive learning.
    
    Args:
        base_encoder: backbone architecture
        dim: feature dimension
        K: queue size
        m: momentum coefficient
        T: temperature
    """
    def __init__(self, base_encoder='resnet50', dim=128, K=65536, m=0.999, T=0.07):
        super().__init__()
        
        self.K = K
        self.m = m
        self.T = T
        
        # Create query encoder
        if base_encoder == 'resnet50':
            self.encoder_q = models.resnet50(pretrained=False)
            feature_dim = 2048
        elif base_encoder == 'resnet18':
            self.encoder_q = models.resnet18(pretrained=False)
            feature_dim = 512
        else:
            raise ValueError(f"Unknown encoder: {base_encoder}")
        
        # Replace fc layer with projection head
        self.encoder_q.fc = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, dim)
        )
        
        # Create key encoder (momentum encoder)
        self.encoder_k = copy.deepcopy(self.encoder_q)
        
        # Freeze key encoder - it's updated via momentum
        for param in self.encoder_k.parameters():
            param.requires_grad = False
        
        # Create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = F.normalize(self.queue, dim=0)
        
        # Queue pointer
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """
        Update queue with new keys
        """
        batch_size = keys.shape[0]
        
        ptr = int(self.queue_ptr)
        
        # Replace the keys at ptr position
        if ptr + batch_size <= self.K:
            self.queue[:, ptr:ptr + batch_size] = keys.T
        else:
            # Wrap around if needed
            remaining = self.K - ptr
            self.queue[:, ptr:] = keys[:remaining].T
            self.queue[:, :batch_size - remaining] = keys[remaining:].T
        
        # Move pointer
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr
    
    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle for distributed training (simplified version)
        In full implementation, this shuffles across GPUs
        """
        idx_shuffle = torch.randperm(x.shape[0], device=x.device)
        idx_unshuffle = torch.argsort(idx_shuffle)
        return x[idx_shuffle], idx_unshuffle
    
    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle
        """
        return x[idx_unshuffle]
    
    def forward(self, im_q, im_k):
        """
        Forward pass
        
        Args:
            im_q: query images (batch_size, 3, H, W)
            im_k: key images (batch_size, 3, H, W)
        
        Returns:
            logits: (batch_size, 1 + K)
            labels: (batch_size,)
        """
        # Compute query features
        q = self.encoder_q(im_q)
        q = F.normalize(q, dim=1)
        
        # Compute key features
        with torch.no_grad():
            # Update momentum encoder
            self._momentum_update_key_encoder()
            
            # Shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            
            k = self.encoder_k(im_k)
            k = F.normalize(k, dim=1)
            
            # Unshuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)
        
        # Compute logits
        # Positive logits: (batch_size, 1)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        
        # Negative logits: (batch_size, K)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        
        # Concatenate logits
        logits = torch.cat([l_pos, l_neg], dim=1)
        
        # Apply temperature
        logits /= self.T
        
        # Labels: positives are at index 0
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        
        # Dequeue and enqueue
        self._dequeue_and_enqueue(k)
        
        return logits, labels


def train_step_moco(model, optimizer, batch_views, device):
    """
    Single training step for MoCo
    
    Args:
        model: MoCo model
        optimizer: optimizer
        batch_views: tuple of (view1, view2) - two augmented views
        device: torch device
    """
    model.train()
    optimizer.zero_grad()
    
    im_q, im_k = batch_views
    im_q, im_k = im_q.to(device), im_k.to(device)
    
    # Forward pass
    logits, labels = model(im_q, im_k)
    
    # Compute loss (cross-entropy)
    loss = F.cross_entropy(logits, labels)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    return loss.item()


class MoCoV3(nn.Module):
    """
    MoCo v3: An improved version using Vision Transformers
    Simplified implementation focusing on the core concepts
    """
    def __init__(self, base_encoder='resnet50', dim=256, mlp_dim=4096, T=0.2):
        super().__init__()
        
        self.T = T
        
        # Base encoder
        if base_encoder == 'resnet50':
            backbone = models.resnet50(pretrained=False)
            feature_dim = 2048
            backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Unknown encoder: {base_encoder}")
        
        self.encoder = backbone
        
        # Projection head (3-layer MLP)
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, mlp_dim),
            nn.BatchNorm1d(mlp_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, mlp_dim),
            nn.BatchNorm1d(mlp_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, dim),
            nn.BatchNorm1d(dim)
        )
        
        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.BatchNorm1d(mlp_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, dim)
        )
    
    def forward(self, x1, x2):
        """
        Forward pass for MoCo v3
        Uses symmetric loss without momentum encoder
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
        
        # Compute symmetric loss
        loss = self.contrastive_loss(p1, z2) / 2 + self.contrastive_loss(p2, z1) / 2
        
        return loss
    
    def contrastive_loss(self, q, k):
        """
        Contrastive loss for MoCo v3
        """
        # Normalize
        q = F.normalize(q, dim=1)
        k = F.normalize(k, dim=1)
        
        # Gather all targets (in distributed training)
        # Here we use the batch itself
        logits = torch.mm(q, k.t()) / self.T
        labels = torch.arange(logits.shape[0], device=logits.device)
        
        loss = F.cross_entropy(logits, labels)
        return loss


if __name__ == "__main__":
    # Example usage
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize MoCo v2
    print("Testing MoCo v2...")
    model_v2 = MoCo(base_encoder='resnet50', dim=128, K=4096).to(device)
    optimizer = torch.optim.SGD(model_v2.parameters(), lr=0.03, momentum=0.9, weight_decay=1e-4)
    
    # Test forward pass
    im_q = torch.randn(32, 3, 224, 224).to(device)
    im_k = torch.randn(32, 3, 224, 224).to(device)
    
    logits, labels = model_v2(im_q, im_k)
    loss = F.cross_entropy(logits, labels)
    
    print(f"MoCo v2 initialized successfully!")
    print(f"Queue size: {model_v2.K}")
    print(f"Momentum coefficient: {model_v2.m}")
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
    
    # Initialize MoCo v3
    print("\nTesting MoCo v3...")
    model_v3 = MoCoV3(base_encoder='resnet50', dim=256).to(device)
    
    loss_v3 = model_v3(im_q, im_k)
    print(f"MoCo v3 initialized successfully!")
    print(f"Loss: {loss_v3.item():.4f}")
