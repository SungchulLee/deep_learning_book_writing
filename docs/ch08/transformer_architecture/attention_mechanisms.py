"""
Attention Mechanisms for RNNs
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class BahdanauAttention(nn.Module):
    """Bahdanau (Additive) Attention"""
    def __init__(self, hidden_size):
        super().__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)
    
    def forward(self, query, keys):
        # query: [batch, hidden] - decoder hidden
        # keys: [batch, seq_len, hidden] - encoder outputs
        scores = self.V(torch.tanh(
            self.W1(query).unsqueeze(1) + self.W2(keys)
        ))  # [batch, seq_len, 1]
        attention_weights = F.softmax(scores, dim=1)
        context = torch.sum(attention_weights * keys, dim=1)
        return context, attention_weights

class LuongAttention(nn.Module):
    """Luong (Multiplicative) Attention"""
    def __init__(self, hidden_size, method='dot'):
        super().__init__()
        self.method = method
        if method == 'general':
            self.W = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, query, keys):
        if self.method == 'dot':
            scores = torch.bmm(query.unsqueeze(1), keys.transpose(1, 2))
        elif self.method == 'general':
            scores = torch.bmm(self.W(query).unsqueeze(1), keys.transpose(1, 2))
        
        attention_weights = F.softmax(scores, dim=-1)
        context = torch.bmm(attention_weights, keys).squeeze(1)
        return context, attention_weights


# ---------------------------------------------------------------------------
# Masked Softmax Utility
# ---------------------------------------------------------------------------
# In many sequence tasks, inputs have variable lengths. When computing
# attention scores, positions beyond the actual sequence length must be
# masked to -inf so that softmax assigns them zero probability.
# This avoids attending to padding tokens and is essential for both
# encoder self-attention and cross-attention in Transformers.

def masked_softmax(X, valid_lens):
    """Perform softmax by masking positions beyond valid lengths.

    Args:
        X: 3D tensor of shape (batch_size, num_queries, num_keys)
        valid_lens: 1D tensor (batch_size,) or 2D tensor (batch_size, num_queries)
            Each element specifies how many keys are valid for that query.
    Returns:
        Softmax output with the same shape as X, where masked positions are 0.
    """
    if valid_lens is None:
        return F.softmax(X, dim=-1)

    shape = X.shape
    if valid_lens.dim() == 1:
        # Same valid length for all queries in each batch element
        valid_lens = valid_lens.repeat_interleave(shape[1])
    else:
        valid_lens = valid_lens.reshape(-1)

    # Build mask: positions >= valid_len get -1e6 so exp(-1e6) ≈ 0
    X_flat = X.reshape(-1, shape[-1])
    maxlen = X_flat.size(1)
    mask = torch.arange(maxlen, device=X.device)[None, :] < valid_lens[:, None]
    X_flat[~mask] = -1e6

    return F.softmax(X_flat.reshape(shape), dim=-1)
