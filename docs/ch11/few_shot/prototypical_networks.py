"""
Prototypical Networks for Few-Shot Learning

Reference: Snell et al. "Prototypical Networks for Few-shot Learning" (2017)

The key idea: Compute a prototype representation for each class by averaging
the embeddings of support examples, then classify queries based on distance
to these prototypes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvEncoder(nn.Module):
    """
    Simple 4-layer convolutional encoder for embedding images.
    Commonly used in few-shot learning papers.
    """
    def __init__(self, input_channels=1, hidden_dim=64, output_dim=64):
        super(ConvEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            self._conv_block(input_channels, hidden_dim),
            self._conv_block(hidden_dim, hidden_dim),
            self._conv_block(hidden_dim, hidden_dim),
            self._conv_block(hidden_dim, output_dim),
        )
        
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
    
    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)


class PrototypicalNetwork(nn.Module):
    """
    Prototypical Network for N-way K-shot classification.
    
    Args:
        encoder: Neural network that embeds inputs into a feature space
    """
    def __init__(self, encoder):
        super(PrototypicalNetwork, self).__init__()
        self.encoder = encoder
    
    def forward(self, support, support_labels, query):
        """
        Args:
            support: (n_support, *input_shape) - Support set examples
            support_labels: (n_support,) - Labels for support set
            query: (n_query, *input_shape) - Query examples to classify
        
        Returns:
            logits: (n_query, n_classes) - Classification logits
        """
        # Embed all examples
        n_classes = len(torch.unique(support_labels))
        n_support = support.shape[0]
        n_query = query.shape[0]
        
        # Concatenate support and query for efficient encoding
        all_examples = torch.cat([support, query], dim=0)
        embeddings = self.encoder(all_examples)
        
        # Split back into support and query
        support_embeddings = embeddings[:n_support]
        query_embeddings = embeddings[n_support:]
        
        # Compute prototypes for each class
        prototypes = self._compute_prototypes(support_embeddings, support_labels, n_classes)
        
        # Compute distances from queries to prototypes
        logits = self._compute_logits(query_embeddings, prototypes)
        
        return logits
    
    def _compute_prototypes(self, embeddings, labels, n_classes):
        """
        Compute prototype for each class as the mean of support embeddings.
        """
        prototypes = []
        for c in range(n_classes):
            # Find all support examples for class c
            class_mask = (labels == c)
            class_embeddings = embeddings[class_mask]
            # Compute mean (prototype)
            prototype = class_embeddings.mean(dim=0)
            prototypes.append(prototype)
        
        return torch.stack(prototypes)
    
    def _compute_logits(self, query_embeddings, prototypes):
        """
        Compute negative squared Euclidean distance from queries to prototypes.
        Negative distance acts as logits (closer = higher probability).
        """
        # Expand dimensions for broadcasting
        # query: (n_query, 1, embedding_dim)
        # prototypes: (1, n_classes, embedding_dim)
        query_expanded = query_embeddings.unsqueeze(1)
        prototypes_expanded = prototypes.unsqueeze(0)
        
        # Compute squared Euclidean distance
        distances = torch.sum((query_expanded - prototypes_expanded) ** 2, dim=2)
        
        # Return negative distances as logits
        return -distances


def train_step(model, support, support_labels, query, query_labels, optimizer):
    """
    Single training step for prototypical network.
    """
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    logits = model(support, support_labels, query)
    
    # Compute loss
    loss = F.cross_entropy(logits, query_labels)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    # Compute accuracy
    predictions = torch.argmax(logits, dim=1)
    accuracy = (predictions == query_labels).float().mean()
    
    return loss.item(), accuracy.item()


def evaluate(model, support, support_labels, query, query_labels):
    """
    Evaluate model on a few-shot task.
    """
    model.eval()
    with torch.no_grad():
        logits = model(support, support_labels, query)
        loss = F.cross_entropy(logits, query_labels)
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == query_labels).float().mean()
    
    return loss.item(), accuracy.item()


# Example usage
if __name__ == "__main__":
    # Create model
    encoder = ConvEncoder(input_channels=1, hidden_dim=64, output_dim=64)
    model = PrototypicalNetwork(encoder)
    
    # Example 5-way 1-shot task
    n_way = 5
    k_shot = 1
    n_query = 15
    
    # Dummy data (batch_size, channels, height, width)
    support = torch.randn(n_way * k_shot, 1, 28, 28)
    support_labels = torch.arange(n_way).repeat_interleave(k_shot)
    query = torch.randn(n_query, 1, 28, 28)
    query_labels = torch.randint(0, n_way, (n_query,))
    
    # Forward pass
    logits = model(support, support_labels, query)
    print(f"Logits shape: {logits.shape}")  # Should be (15, 5)
    
    # Training example
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss, acc = train_step(model, support, support_labels, query, query_labels, optimizer)
    print(f"Loss: {loss:.4f}, Accuracy: {acc:.4f}")
