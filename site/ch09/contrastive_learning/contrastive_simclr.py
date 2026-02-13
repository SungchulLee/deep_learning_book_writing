"""
SimCLR: A Simple Framework for Contrastive Learning of Visual Representations
Implementation of the SimCLR algorithm for self-supervised learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ProjectionHead(nn.Module):
    """Projection head for SimCLR"""
    def __init__(self, input_dim=2048, hidden_dim=2048, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)


class SimCLR(nn.Module):
    """
    SimCLR Model for Contrastive Learning
    
    Args:
        base_model: backbone architecture (resnet18, resnet50, etc.)
        projection_dim: dimension of projection head output
    """
    def __init__(self, base_model='resnet50', projection_dim=128):
        super().__init__()
        
        # Load backbone
        if base_model == 'resnet18':
            self.encoder = models.resnet18(pretrained=False)
            feature_dim = 512
        elif base_model == 'resnet50':
            self.encoder = models.resnet50(pretrained=False)
            feature_dim = 2048
        else:
            raise ValueError(f"Unknown model: {base_model}")
        
        # Remove the final classification layer
        self.encoder.fc = nn.Identity()
        
        # Add projection head
        self.projection_head = ProjectionHead(
            input_dim=feature_dim,
            hidden_dim=feature_dim,
            output_dim=projection_dim
        )
    
    def forward(self, x):
        # Extract features
        features = self.encoder(x)
        # Project to contrastive space
        projections = self.projection_head(features)
        return features, projections


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (NT-Xent)
    The contrastive loss function used in SimCLR
    """
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z_i, z_j):
        """
        Args:
            z_i: projections from augmented view 1, shape (batch_size, projection_dim)
            z_j: projections from augmented view 2, shape (batch_size, projection_dim)
        """
        batch_size = z_i.shape[0]
        
        # Normalize embeddings
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        # Concatenate both views
        representations = torch.cat([z_i, z_j], dim=0)  # (2*batch_size, projection_dim)
        
        # Compute similarity matrix
        similarity_matrix = F.cosine_similarity(
            representations.unsqueeze(1),
            representations.unsqueeze(0),
            dim=2
        )  # (2*batch_size, 2*batch_size)
        
        # Create labels: positive pairs are diagonal blocks
        labels = torch.arange(batch_size, device=z_i.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)
        
        # Mask out self-similarities
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z_i.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -float('inf'))
        
        # Apply temperature scaling
        similarity_matrix = similarity_matrix / self.temperature
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss


def train_step(model, optimizer, criterion, batch_views, device):
    """
    Single training step for SimCLR
    
    Args:
        model: SimCLR model
        optimizer: optimizer
        criterion: NTXentLoss
        batch_views: tuple of (view1, view2) - two augmented views of the batch
        device: torch device
    """
    model.train()
    optimizer.zero_grad()
    
    view1, view2 = batch_views
    view1, view2 = view1.to(device), view2.to(device)
    
    # Forward pass through both views
    _, projections1 = model(view1)
    _, projections2 = model(view2)
    
    # Compute contrastive loss
    loss = criterion(projections1, projections2)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    return loss.item()


def extract_features(model, dataloader, device):
    """
    Extract features using trained SimCLR encoder
    Useful for downstream tasks
    """
    model.eval()
    features_list = []
    labels_list = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            features, _ = model(images)
            features_list.append(features.cpu())
            labels_list.append(labels)
    
    features = torch.cat(features_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    
    return features, labels


if __name__ == "__main__":
    # Example usage
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = SimCLR(base_model='resnet50', projection_dim=128).to(device)
    
    # Initialize loss and optimizer
    criterion = NTXentLoss(temperature=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Example training loop (pseudo-code)
    print("SimCLR Model initialized successfully!")
    print(f"Encoder: {model.encoder.__class__.__name__}")
    print(f"Projection head output dim: 128")
    
    # Create dummy batch for testing
    dummy_view1 = torch.randn(32, 3, 224, 224).to(device)
    dummy_view2 = torch.randn(32, 3, 224, 224).to(device)
    
    _, proj1 = model(dummy_view1)
    _, proj2 = model(dummy_view2)
    
    loss = criterion(proj1, proj2)
    print(f"\nTest forward pass successful!")
    print(f"Loss value: {loss.item():.4f}")
