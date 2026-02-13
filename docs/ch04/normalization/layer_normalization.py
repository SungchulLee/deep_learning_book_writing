"""
Layer Normalization Implementation and Examples
================================================

Layer Normalization normalizes inputs across the feature dimension.
Unlike Batch Norm, it doesn't depend on batch size - great for RNNs and small batches.

Paper: "Layer Normalization" (Ba, Kiros & Hinton, 2016)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNormNumPy:
    """
    Layer Normalization implementation from scratch using NumPy.
    Normalizes across the feature dimension (not batch dimension).
    """
    
    def __init__(self, normalized_shape, eps=1e-5):
        """
        Args:
            normalized_shape: Shape of the features to normalize over
            eps: Small constant for numerical stability
        """
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        # Learnable parameters
        self.gamma = np.ones(normalized_shape)  # Scale parameter
        self.beta = np.zeros(normalized_shape)  # Shift parameter
        
    def forward(self, x):
        """
        Forward pass of Layer Normalization.
        
        Args:
            x: Input of shape (batch_size, *normalized_shape)
            
        Returns:
            Normalized output of same shape as input
        """
        # Calculate mean and variance over the last len(normalized_shape) dimensions
        axes = tuple(range(-len(self.normalized_shape), 0))
        
        mean = np.mean(x, axis=axes, keepdims=True)
        var = np.var(x, axis=axes, keepdims=True)
        
        # Normalize
        x_normalized = (x - mean) / np.sqrt(var + self.eps)
        
        # Scale and shift
        out = self.gamma * x_normalized + self.beta
        
        return out


class RNNWithLayerNorm(nn.Module):
    """
    RNN cell with Layer Normalization.
    LayerNorm is particularly useful for RNNs.
    """
    
    def __init__(self, input_size, hidden_size):
        super(RNNWithLayerNorm, self).__init__()
        
        self.hidden_size = hidden_size
        
        # Input and hidden transformations
        self.W_ih = nn.Linear(input_size, hidden_size, bias=False)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Layer normalization
        self.ln = nn.LayerNorm(hidden_size)
        
    def forward(self, x, hidden=None):
        """
        Args:
            x: Input of shape (batch_size, seq_len, input_size)
            hidden: Initial hidden state (batch_size, hidden_size)
            
        Returns:
            outputs: All hidden states (batch_size, seq_len, hidden_size)
            hidden: Final hidden state (batch_size, hidden_size)
        """
        batch_size, seq_len, _ = x.size()
        
        if hidden is None:
            hidden = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        outputs = []
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            
            # Compute new hidden state
            hidden = self.W_ih(x_t) + self.W_hh(hidden)
            
            # Apply layer normalization
            hidden = self.ln(hidden)
            
            # Apply activation
            hidden = torch.tanh(hidden)
            
            outputs.append(hidden.unsqueeze(1))
        
        outputs = torch.cat(outputs, dim=1)
        
        return outputs, hidden


class TransformerBlockWithLayerNorm(nn.Module):
    """
    Simplified Transformer block using Layer Normalization.
    This is the standard normalization for Transformers.
    """
    
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerBlockWithLayerNorm, self).__init__()
        
        # Multi-head attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer normalization (2 instances)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, attn_mask=None, key_padding_mask=None):
        """
        Args:
            x: Input of shape (seq_len, batch_size, d_model)
            
        Returns:
            Output of same shape as input
        """
        # Self-attention with residual connection and layer norm
        x2 = self.self_attn(x, x, x, attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask)[0]
        x = x + self.dropout1(x2)
        x = self.norm1(x)
        
        # Feed-forward with residual connection and layer norm
        x2 = self.linear2(self.dropout(F.relu(self.linear1(x))))
        x = x + self.dropout2(x2)
        x = self.norm2(x)
        
        return x


class SimpleNetworkWithLayerNorm(nn.Module):
    """
    Simple feedforward network with Layer Normalization.
    """
    
    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super(SimpleNetworkWithLayerNorm, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        
        self.fc3 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.ln1(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        x = self.ln2(x)
        x = F.relu(x)
        
        x = self.fc3(x)
        
        return x


def demonstrate_layer_norm():
    """
    Demonstrate how Layer Normalization works.
    """
    print("=" * 60)
    print("Layer Normalization Demonstration")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Create sample data
    batch_size = 4
    num_features = 5
    
    # Each sample has features with different scales
    x = np.array([
        [1.0, 2.0, 3.0, 4.0, 5.0],      # Sample 1
        [10.0, 20.0, 30.0, 40.0, 50.0],  # Sample 2 (larger scale)
        [0.1, 0.2, 0.3, 0.4, 0.5],      # Sample 3 (smaller scale)
        [5.0, 5.0, 5.0, 5.0, 5.0],      # Sample 4 (constant)
    ])
    
    print("\nOriginal data:")
    print(x)
    print(f"\nMean per sample: {np.mean(x, axis=1)}")
    print(f"Std per sample:  {np.std(x, axis=1)}")
    
    # Apply layer normalization
    ln = LayerNormNumPy(num_features)
    x_normalized = ln.forward(x)
    
    print("\nAfter Layer Normalization:")
    print(x_normalized)
    print(f"\nMean per sample: {np.mean(x_normalized, axis=1)}")
    print(f"Std per sample:  {np.std(x_normalized, axis=1)}")
    
    print("\nKey observations:")
    print("- Each SAMPLE is normalized independently")
    print("- Mean ≈ 0 and Std ≈ 1 for EACH sample")
    print("- Works well for variable batch sizes")
    print("- No dependence on other samples in the batch")


def compare_batchnorm_layernorm():
    """
    Compare Batch Normalization vs Layer Normalization.
    """
    print("\n" + "=" * 60)
    print("Batch Norm vs Layer Norm Comparison")
    print("=" * 60)
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create sample data
    x = torch.randn(8, 10)  # 8 samples, 10 features
    
    print("\nOriginal data shape:", x.shape)
    print("Original data:\n", x[:2])  # Show first 2 samples
    
    # Batch Normalization
    bn = nn.BatchNorm1d(10)
    bn.eval()  # Use eval mode to avoid using running stats
    x_bn = bn(x)
    
    # Layer Normalization
    ln = nn.LayerNorm(10)
    x_ln = ln(x)
    
    print("\nAfter Batch Normalization:")
    print("Mean per feature (across batch):", x_bn.mean(dim=0).detach().numpy()[:5])
    print("Mean per sample (across features):", x_bn.mean(dim=1).detach().numpy()[:3])
    
    print("\nAfter Layer Normalization:")
    print("Mean per feature (across batch):", x_ln.mean(dim=0).detach().numpy()[:5])
    print("Mean per sample (across features):", x_ln.mean(dim=1).detach().numpy()[:3])
    
    print("\n" + "-" * 60)
    print("Key Differences:")
    print("-" * 60)
    print("Batch Normalization:")
    print("  - Normalizes across the BATCH dimension")
    print("  - Each feature is normalized using batch statistics")
    print("  - Depends on batch size (problematic for small batches)")
    print("  - Different behavior in train vs eval mode")
    print("  - Best for: CNNs, large batches, feedforward networks")
    
    print("\nLayer Normalization:")
    print("  - Normalizes across the FEATURE dimension")
    print("  - Each sample is normalized independently")
    print("  - Independent of batch size")
    print("  - Same behavior in train and eval mode")
    print("  - Best for: RNNs, Transformers, small batches")


def demonstrate_small_batch_problem():
    """
    Show why LayerNorm is better for small batches.
    """
    print("\n" + "=" * 60)
    print("Small Batch Problem")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    # Small batch
    x_small = torch.randn(2, 10)
    
    # Large batch
    x_large = torch.randn(64, 10)
    
    # Batch Normalization
    bn = nn.BatchNorm1d(10)
    bn.train()
    
    # Layer Normalization
    ln = nn.LayerNorm(10)
    
    print("\nWith Batch Normalization:")
    with torch.no_grad():
        out_small_bn = bn(x_small)
        out_large_bn = bn(x_large)
    
    print(f"Small batch (n=2) std: {out_small_bn.std():.4f}")
    print(f"Large batch (n=64) std: {out_large_bn.std():.4f}")
    
    print("\nWith Layer Normalization:")
    with torch.no_grad():
        out_small_ln = ln(x_small)
        out_large_ln = ln(x_large)
    
    print(f"Small batch (n=2) std: {out_small_ln.std():.4f}")
    print(f"Large batch (n=64) std: {out_large_ln.std():.4f}")
    
    print("\nObservation:")
    print("- BatchNorm is sensitive to batch size")
    print("- LayerNorm is consistent across batch sizes")
    print("- Use LayerNorm when batch size is small or variable")


if __name__ == "__main__":
    demonstrate_layer_norm()
    compare_batchnorm_layernorm()
    demonstrate_small_batch_problem()
    
    print("\n" + "=" * 60)
    print("When to use Layer Normalization:")
    print("=" * 60)
    print("✓ RNNs and LSTMs")
    print("✓ Transformers (standard choice)")
    print("✓ Small batch sizes")
    print("✓ Online learning (batch size = 1)")
    print("✓ When batch statistics are unreliable")
