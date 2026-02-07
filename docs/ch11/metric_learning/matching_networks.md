# Matching Networks

Matching Networks (Vinyals et al., 2016) introduced the episodic training paradigm for few-shot learning and perform classification through a weighted nearest-neighbour approach using attention over the support set.

## Key Idea

Classification of a query $\hat{x}$ is performed as a weighted sum over support set labels:

$$P(y = c \mid \hat{x}, \mathcal{S}) = \sum_{(x_i, y_i) \in \mathcal{S}} a(\hat{x}, x_i) \, \mathbf{1}(y_i = c)$$

where $a(\hat{x}, x_i)$ is a softmax attention kernel:

$$a(\hat{x}, x_i) = \frac{\exp(\text{cos}(f(\hat{x}), g(x_i)))}{\sum_j \exp(\text{cos}(f(\hat{x}), g(x_j)))}$$

## Architecture

Matching Networks use separate encoders for the support and query sets:

- $g$: Support set encoder (optionally with bidirectional LSTM for full context)
- $f$: Query encoder (optionally with attention LSTM conditioned on support)

The Full Context Embeddings (FCE) allow each support embedding to attend to all other support examples, providing richer representations.

## Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class MatchingNetwork(nn.Module):
    """
    Matching Networks for One-Shot Learning.
    
    Classifies queries by soft attention over support set embeddings.
    """
    
    def __init__(self, encoder, use_fce=False):
        super().__init__()
        self.encoder = encoder
        self.use_fce = use_fce
        
        if use_fce:
            feature_dim = 512  # Adjust based on encoder
            self.support_lstm = nn.LSTM(
                feature_dim, feature_dim // 2,
                bidirectional=True, batch_first=True
            )
    
    def forward(self, support_x, support_y, query_x, n_way):
        """
        Args:
            support_x: (N*K, C, H, W) support images
            support_y: (N*K,) support labels
            query_x: (N*Q, C, H, W) query images
            n_way: number of classes
        
        Returns:
            log_probs: (N*Q, N) log-probabilities
        """
        # Encode all images
        support_embeddings = self.encoder(support_x)  # (N*K, D)
        query_embeddings = self.encoder(query_x)      # (N*Q, D)
        
        # Optional: Full Context Embeddings
        if self.use_fce:
            support_embeddings = self._fce(support_embeddings)
        
        # L2 normalise
        support_embeddings = F.normalize(support_embeddings, dim=1)
        query_embeddings = F.normalize(query_embeddings, dim=1)
        
        # Cosine similarity: (N*Q, N*K)
        similarities = torch.mm(query_embeddings, support_embeddings.t())
        
        # Softmax attention
        attention = F.softmax(similarities, dim=1)  # (N*Q, N*K)
        
        # Convert support labels to one-hot: (N*K, N)
        one_hot = F.one_hot(support_y, n_way).float()
        
        # Weighted vote: (N*Q, N)
        log_probs = torch.log(torch.mm(attention, one_hot) + 1e-8)
        
        return log_probs
    
    def _fce(self, embeddings):
        """Full Context Embeddings using bidirectional LSTM."""
        embeddings = embeddings.unsqueeze(0)  # Add batch dim
        output, _ = self.support_lstm(embeddings)
        return output.squeeze(0)
    
    def predict(self, support_x, support_y, query_x):
        n_way = support_y.unique().size(0)
        log_probs = self.forward(support_x, support_y, query_x, n_way)
        return log_probs.argmax(dim=1)
```

## Training

```python
def train_matching_network(model, train_loader, optimizer, n_way, device, num_episodes=100):
    """Train Matching Network with episodic training."""
    model.train()
    total_loss = 0
    
    for episode_idx, (support_x, support_y, query_x, query_y) in enumerate(train_loader):
        if episode_idx >= num_episodes:
            break
        
        support_x = support_x.to(device)
        support_y = support_y.to(device)
        query_x = query_x.to(device)
        query_y = query_y.to(device)
        
        log_probs = model(support_x, support_y, query_x, n_way)
        loss = F.nll_loss(log_probs, query_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / min(num_episodes, episode_idx + 1)
```

## Summary

| Aspect | Detail |
|--------|--------|
| Classification | Soft nearest-neighbour via attention |
| Distance | Cosine similarity |
| Context | Optional FCE with bidirectional LSTM |
| Training | Episodic (first to propose this) |
| Strength | Simple, effective baseline |
| Weakness | No class prototypes; attention over all support |

## References

1. Vinyals, O., et al. (2016). "Matching Networks for One Shot Learning." *NeurIPS*.
