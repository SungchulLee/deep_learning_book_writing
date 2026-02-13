"""
Transformer-based Few-Shot Learning

Modern approach using self-attention to process support and query sets together,
allowing the model to reason about relationships between examples.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """
    Add positional information to embeddings.
    """
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]


class TransformerFewShotClassifier(nn.Module):
    """
    Transformer-based few-shot classifier.
    
    The model processes support and query examples together, allowing
    attention mechanisms to compare and relate examples across the sets.
    """
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=2, dim_feedforward=512):
        super(TransformerFewShotClassifier, self).__init__()
        
        # Input embedding
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Linear(d_model, 1)
    
    def forward(self, support, support_labels, query):
        """
        Args:
            support: (n_support, input_dim)
            support_labels: (n_support,)
            query: (n_query, input_dim)
        
        Returns:
            logits: (n_query, n_classes)
        """
        n_support = support.shape[0]
        n_query = query.shape[0]
        n_classes = support_labels.max().item() + 1
        
        # Combine support and query
        all_examples = torch.cat([support, query], dim=0)  # (n_support + n_query, input_dim)
        
        # Project to d_model
        embeddings = self.input_projection(all_examples)  # (n_support + n_query, d_model)
        
        # Add positional encoding
        embeddings = self.pos_encoder(embeddings.unsqueeze(0)).squeeze(0)
        
        # Apply transformer
        transformed = self.transformer(embeddings.unsqueeze(0)).squeeze(0)
        
        # Split back into support and query
        support_transformed = transformed[:n_support]
        query_transformed = transformed[n_support:]
        
        # Compute logits for each query against each class
        logits = []
        for query_emb in query_transformed:
            class_logits = []
            for c in range(n_classes):
                # Get support examples for this class
                class_mask = (support_labels == c)
                class_support = support_transformed[class_mask]
                
                # Compute similarity between query and class prototypes
                query_expanded = query_emb.unsqueeze(0).expand(class_support.shape[0], -1)
                similarities = F.cosine_similarity(query_expanded, class_support, dim=1)
                class_logit = similarities.mean()
                class_logits.append(class_logit)
            
            logits.append(torch.stack(class_logits))
        
        return torch.stack(logits)


class SetTransformer(nn.Module):
    """
    Set-based transformer for few-shot learning.
    
    Uses induced set attention to create permutation-invariant representations
    of support and query sets.
    """
    def __init__(self, input_dim, d_model=128, num_heads=4, num_inds=32):
        super(SetTransformer, self).__init__()
        
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Induced set attention blocks
        self.inducing_points = nn.Parameter(torch.randn(num_inds, d_model))
        
        self.mab1 = MultiheadAttentionBlock(d_model, num_heads)
        self.mab2 = MultiheadAttentionBlock(d_model, num_heads)
        
        self.output_proj = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, set_size, input_dim)
        
        Returns:
            output: (batch_size, d_model) - Set embedding
        """
        # Project input
        x = self.input_proj(x)  # (batch_size, set_size, d_model)
        
        # Induced set attention
        batch_size = x.shape[0]
        inds = self.inducing_points.unsqueeze(0).expand(batch_size, -1, -1)
        
        h = self.mab1(inds, x)  # (batch_size, num_inds, d_model)
        h = self.mab2(h, h)     # (batch_size, num_inds, d_model)
        
        # Pool over inducing points
        output = h.mean(dim=1)  # (batch_size, d_model)
        
        return self.output_proj(output)


class MultiheadAttentionBlock(nn.Module):
    """
    Multihead attention block for Set Transformer.
    """
    def __init__(self, d_model, num_heads):
        super(MultiheadAttentionBlock, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.layer_norm2 = nn.LayerNorm(d_model)
    
    def forward(self, query, key_value):
        """
        Args:
            query: (batch_size, query_len, d_model)
            key_value: (batch_size, kv_len, d_model)
        """
        # Attention
        attn_out, _ = self.attention(query, key_value, key_value)
        query = self.layer_norm(query + attn_out)
        
        # FFN
        ffn_out = self.ffn(query)
        query = self.layer_norm2(query + ffn_out)
        
        return query


def train_transformer_fewshot(model, support, support_labels, query, query_labels, optimizer):
    """
    Training step for transformer few-shot model.
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


# Example usage
if __name__ == "__main__":
    # Model configuration
    input_dim = 784  # Flattened 28x28 images
    d_model = 128
    nhead = 4
    num_layers = 2
    
    # Create model
    model = TransformerFewShotClassifier(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers
    )
    
    # Example 5-way 5-shot task
    n_way = 5
    k_shot = 5
    n_query = 15
    
    support = torch.randn(n_way * k_shot, input_dim)
    support_labels = torch.arange(n_way).repeat_interleave(k_shot)
    query = torch.randn(n_query, input_dim)
    query_labels = torch.randint(0, n_way, (n_query,))
    
    # Forward pass
    logits = model(support, support_labels, query)
    print(f"Logits shape: {logits.shape}")  # (15, 5)
    
    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss, acc = train_transformer_fewshot(
        model, support, support_labels, query, query_labels, optimizer
    )
    print(f"Loss: {loss:.4f}, Accuracy: {acc:.4f}")
    
    # Example with Set Transformer
    set_model = SetTransformer(input_dim=input_dim, d_model=128, num_heads=4)
    support_set = support.unsqueeze(0)  # Add batch dimension
    set_embedding = set_model(support_set)
    print(f"Set embedding shape: {set_embedding.shape}")  # (1, 128)
