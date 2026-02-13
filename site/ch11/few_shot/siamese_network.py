"""
Siamese Networks for One-Shot Learning

Siamese networks learn a similarity metric between pairs of examples.
They use twin networks with shared weights to embed inputs, then compute
similarity between embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseEncoder(nn.Module):
    """
    CNN encoder for Siamese network.
    """
    def __init__(self, input_channels=1, hidden_dim=64, embedding_dim=128):
        super(SiameseEncoder, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, hidden_dim, kernel_size=10, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 2
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 4
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=1),
            nn.ReLU(),
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim * 4, embedding_dim),
            nn.Sigmoid()  # Constrain embeddings to [0, 1]
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc(x)
        return x


class SiameseNetwork(nn.Module):
    """
    Siamese Network that computes similarity between pairs.
    """
    def __init__(self, encoder):
        super(SiameseNetwork, self).__init__()
        self.encoder = encoder
    
    def forward(self, x1, x2):
        """
        Compute embeddings and distance for a pair of inputs.
        
        Args:
            x1: First input (batch_size, channels, height, width)
            x2: Second input (batch_size, channels, height, width)
        
        Returns:
            distance: L1 distance between embeddings
        """
        # Get embeddings
        embedding1 = self.encoder(x1)
        embedding2 = self.encoder(x2)
        
        # Compute L1 distance
        distance = torch.abs(embedding1 - embedding2)
        
        return distance
    
    def predict_similarity(self, x1, x2):
        """
        Predict similarity score (0 = different, 1 = same).
        """
        distance = self.forward(x1, x2)
        # Add a final layer to convert distance to similarity
        similarity = torch.sigmoid(distance.sum(dim=1))
        return similarity


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for training Siamese networks.
    
    Loss = (1-Y) * 0.5 * D^2 + Y * 0.5 * max(margin - D, 0)^2
    
    where Y=0 for similar pairs and Y=1 for dissimilar pairs,
    D is the Euclidean distance between embeddings.
    """
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, distance, label):
        """
        Args:
            distance: Euclidean distance between embeddings
            label: 0 for similar pairs, 1 for dissimilar pairs
        """
        loss = (1 - label) * torch.pow(distance, 2) + \
               label * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        return loss.mean()


class TripletLoss(nn.Module):
    """
    Triplet loss: pulls anchor closer to positive, pushes away from negative.
    
    Loss = max(d(anchor, positive) - d(anchor, negative) + margin, 0)
    """
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        """
        Args:
            anchor: Anchor embeddings (batch_size, embedding_dim)
            positive: Positive embeddings (same class as anchor)
            negative: Negative embeddings (different class)
        """
        pos_distance = F.pairwise_distance(anchor, positive, p=2)
        neg_distance = F.pairwise_distance(anchor, negative, p=2)
        
        loss = F.relu(pos_distance - neg_distance + self.margin)
        return loss.mean()


def train_siamese(model, x1, x2, labels, optimizer, criterion):
    """
    Training step for Siamese network with contrastive loss.
    
    Args:
        x1, x2: Pair of inputs
        labels: 0 for similar, 1 for dissimilar
    """
    model.train()
    optimizer.zero_grad()
    
    # Get embeddings
    emb1 = model.encoder(x1)
    emb2 = model.encoder(x2)
    
    # Compute Euclidean distance
    distance = F.pairwise_distance(emb1, emb2)
    
    # Compute loss
    loss = criterion(distance, labels)
    
    loss.backward()
    optimizer.step()
    
    return loss.item()


def one_shot_classification(model, support_set, support_labels, query):
    """
    Perform one-shot classification using a trained Siamese network.
    
    Args:
        support_set: (n_classes, *input_shape) - One example per class
        support_labels: (n_classes,) - Class labels
        query: (n_query, *input_shape) - Queries to classify
    
    Returns:
        predictions: (n_query,) - Predicted class labels
    """
    model.eval()
    n_classes = support_set.shape[0]
    n_query = query.shape[0]
    
    with torch.no_grad():
        # Get embeddings for support set
        support_embeddings = model.encoder(support_set)
        
        predictions = []
        for i in range(n_query):
            # Get embedding for this query
            query_embedding = model.encoder(query[i:i+1])
            
            # Compute distances to all support examples
            distances = []
            for j in range(n_classes):
                dist = F.pairwise_distance(
                    query_embedding, 
                    support_embeddings[j:j+1]
                )
                distances.append(dist)
            
            distances = torch.stack(distances)
            
            # Predict class with minimum distance
            predicted_idx = torch.argmin(distances)
            predictions.append(support_labels[predicted_idx])
        
        return torch.tensor(predictions)


# Example usage
if __name__ == "__main__":
    # Create model
    encoder = SiameseEncoder(input_channels=1, hidden_dim=64, embedding_dim=128)
    model = SiameseNetwork(encoder)
    
    # Training with contrastive loss
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Example training batch
    batch_size = 32
    x1 = torch.randn(batch_size, 1, 28, 28)
    x2 = torch.randn(batch_size, 1, 28, 28)
    labels = torch.randint(0, 2, (batch_size,)).float()  # 0=similar, 1=dissimilar
    
    loss = train_siamese(model, x1, x2, labels, optimizer, criterion)
    print(f"Training loss: {loss:.4f}")
    
    # One-shot classification example
    n_classes = 5
    n_query = 10
    support_set = torch.randn(n_classes, 1, 28, 28)
    support_labels = torch.arange(n_classes)
    query = torch.randn(n_query, 1, 28, 28)
    
    predictions = one_shot_classification(model, support_set, support_labels, query)
    print(f"Predictions: {predictions}")
    
    # Alternative: Triplet loss training
    triplet_criterion = TripletLoss(margin=1.0)
    anchor = torch.randn(batch_size, 1, 28, 28)
    positive = torch.randn(batch_size, 1, 28, 28)
    negative = torch.randn(batch_size, 1, 28, 28)
    
    anchor_emb = model.encoder(anchor)
    positive_emb = model.encoder(positive)
    negative_emb = model.encoder(negative)
    
    triplet_loss = triplet_criterion(anchor_emb, positive_emb, negative_emb)
    print(f"Triplet loss: {triplet_loss.item():.4f}")
