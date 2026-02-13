"""
Matching Networks for One-Shot Learning

Reference: Vinyals et al. "Matching Networks for One Shot Learning" (2016)

Key idea: Use attention mechanisms to compare query examples with support set,
weighting each support example's contribution to the prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionEncoder(nn.Module):
    """
    Encoder with attention mechanism for matching networks.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(AttentionEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
    
    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # Take the last output
        return self.fc(lstm_out[:, -1, :])


class SimpleEncoder(nn.Module):
    """
    Simple CNN encoder for images.
    """
    def __init__(self, input_channels=1, hidden_dim=64, output_dim=64):
        super(SimpleEncoder, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(hidden_dim, output_dim, 3, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        return x.view(x.size(0), -1)


class MatchingNetwork(nn.Module):
    """
    Matching Network with cosine similarity attention.
    
    The model classifies queries by computing attention weights over the support set,
    where attention is based on cosine similarity between embeddings.
    """
    def __init__(self, encoder, use_full_context_embeddings=False):
        super(MatchingNetwork, self).__init__()
        self.encoder = encoder
        self.use_fce = use_full_context_embeddings
        
    def forward(self, support, support_labels, query):
        """
        Args:
            support: (n_support, *input_shape) - Support set
            support_labels: (n_support,) - One-hot or label indices
            query: (n_query, *input_shape) - Query set
        
        Returns:
            predictions: (n_query, n_classes) - Prediction probabilities
        """
        # Encode support and query
        support_embeddings = self.encoder(support)
        query_embeddings = self.encoder(query)
        
        # Normalize embeddings for cosine similarity
        support_embeddings = F.normalize(support_embeddings, p=2, dim=1)
        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
        
        # Compute attention weights using cosine similarity
        # Shape: (n_query, n_support)
        attention = torch.mm(query_embeddings, support_embeddings.t())
        attention = F.softmax(attention, dim=1)
        
        # Convert support_labels to one-hot if needed
        n_classes = support_labels.max().item() + 1
        if support_labels.dim() == 1:
            support_labels_one_hot = F.one_hot(support_labels, n_classes).float()
        else:
            support_labels_one_hot = support_labels.float()
        
        # Weighted sum of support labels based on attention
        # Shape: (n_query, n_classes)
        predictions = torch.mm(attention, support_labels_one_hot)
        
        return predictions
    
    def predict(self, support, support_labels, query):
        """
        Get class predictions for query examples.
        """
        predictions = self.forward(support, support_labels, query)
        return torch.argmax(predictions, dim=1)


def cosine_distance(x, y):
    """
    Compute cosine distance between two sets of embeddings.
    """
    # Normalize
    x = F.normalize(x, p=2, dim=1)
    y = F.normalize(y, p=2, dim=1)
    
    # Cosine similarity
    similarity = torch.mm(x, y.t())
    
    return 1 - similarity


def train_matching_network(model, support, support_labels, query, query_labels, optimizer):
    """
    Training step for matching network.
    """
    model.train()
    optimizer.zero_grad()
    
    # Get predictions
    predictions = model(support, support_labels, query)
    
    # Compute loss (cross-entropy with soft targets)
    query_labels_one_hot = F.one_hot(query_labels, predictions.shape[1]).float()
    loss = F.binary_cross_entropy(predictions, query_labels_one_hot)
    
    # Or use cross-entropy with hard labels
    # loss = F.cross_entropy(predictions, query_labels)
    
    loss.backward()
    optimizer.step()
    
    # Compute accuracy
    predicted_classes = torch.argmax(predictions, dim=1)
    accuracy = (predicted_classes == query_labels).float().mean()
    
    return loss.item(), accuracy.item()


def evaluate_matching_network(model, support, support_labels, query, query_labels):
    """
    Evaluate matching network.
    """
    model.eval()
    with torch.no_grad():
        predictions = model(support, support_labels, query)
        query_labels_one_hot = F.one_hot(query_labels, predictions.shape[1]).float()
        loss = F.binary_cross_entropy(predictions, query_labels_one_hot)
        
        predicted_classes = torch.argmax(predictions, dim=1)
        accuracy = (predicted_classes == query_labels).float().mean()
    
    return loss.item(), accuracy.item()


# Example usage
if __name__ == "__main__":
    # Create model
    encoder = SimpleEncoder(input_channels=1, hidden_dim=64, output_dim=64)
    model = MatchingNetwork(encoder)
    
    # Example 5-way 5-shot task
    n_way = 5
    k_shot = 5
    n_query = 15
    
    # Create dummy data
    support = torch.randn(n_way * k_shot, 1, 28, 28)
    support_labels = torch.arange(n_way).repeat_interleave(k_shot)
    query = torch.randn(n_query, 1, 28, 28)
    query_labels = torch.randint(0, n_way, (n_query,))
    
    # Forward pass
    predictions = model(support, support_labels, query)
    print(f"Predictions shape: {predictions.shape}")  # (15, 5)
    
    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss, acc = train_matching_network(model, support, support_labels, query, query_labels, optimizer)
    print(f"Loss: {loss:.4f}, Accuracy: {acc:.4f}")
    
    # Make predictions
    predicted_classes = model.predict(support, support_labels, query)
    print(f"Predicted classes: {predicted_classes}")
