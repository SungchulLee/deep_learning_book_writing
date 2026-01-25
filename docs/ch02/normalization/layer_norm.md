# Layer Normalization

## Overview

Layer Normalization (LayerNorm), introduced by Ba, Kiros, and Hinton in 2016, normalizes activations across the feature dimension for each sample independently. Unlike Batch Normalization, LayerNorm's statistics are computed per sample, making it independent of batch size and the de facto standard for Transformers and sequence models.

## Mathematical Formulation

### Forward Pass

For an input $x \in \mathbb{R}^{H}$ where $H$ is the number of features (normalized shape), LayerNorm computes:

**Step 1: Compute per-sample statistics**

$$\mu = \frac{1}{H} \sum_{i=1}^{H} x_i$$

$$\sigma^2 = \frac{1}{H} \sum_{i=1}^{H} (x_i - \mu)^2$$

**Step 2: Normalize**

$$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$$

**Step 3: Scale and shift**

$$y_i = \gamma_i \hat{x}_i + \beta_i$$

Where:
- $\gamma, \beta \in \mathbb{R}^{H}$ are learnable parameters (element-wise)
- $\epsilon$ is a small constant (typically $10^{-5}$)

### Key Difference from BatchNorm

| Aspect | BatchNorm | LayerNorm |
|--------|-----------|-----------|
| Statistics computed over | Batch dimension | Feature dimension |
| For input (N, C, H, W) | Mean over (N, H, W) per C | Mean over (C, H, W) per N |
| Batch dependence | Yes | No |
| Train/eval difference | Yes | No |
| Running statistics | Required | Not needed |

### Visualization of Normalization Axes

```
Input: (N, C, H, W) = (Batch, Channels, Height, Width)

BatchNorm:  Normalizes ████ for each channel
            N × H × W elements per statistic
            
LayerNorm:  Normalizes ████ for each sample
            C × H × W elements per statistic
```

## PyTorch Implementation

### From Scratch

```python
import torch
import torch.nn as nn

class LayerNormFromScratch(nn.Module):
    """
    Layer Normalization implementation from scratch.
    Normalizes across the feature dimension for each sample.
    """
    
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        # Learnable parameters
        if self.elementwise_affine:
            self.gamma = nn.Parameter(torch.ones(normalized_shape))
            self.beta = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)
    
    def forward(self, x):
        """
        Args:
            x: Input of shape (..., *normalized_shape)
        
        Returns:
            Normalized output of same shape
        """
        # Compute number of dimensions to normalize over
        num_dims = len(self.normalized_shape)
        
        # Normalize over the last num_dims dimensions
        dims = tuple(range(-num_dims, 0))
        
        # Compute mean and variance
        mean = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, keepdim=True, unbiased=False)
        
        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # Apply affine transformation
        if self.elementwise_affine:
            x_norm = self.gamma * x_norm + self.beta
        
        return x_norm
    
    def extra_repr(self):
        return (f'{self.normalized_shape}, eps={self.eps}, '
                f'elementwise_affine={self.elementwise_affine}')
```

### Using PyTorch Built-in

```python
import torch
import torch.nn as nn

# For 1D features (e.g., hidden states in Transformer)
ln = nn.LayerNorm(normalized_shape=512)  # Input: (..., 512)

# For 2D features (e.g., image features)
ln = nn.LayerNorm(normalized_shape=[64, 32, 32])  # Input: (..., 64, 32, 32)

# Common parameters
ln = nn.LayerNorm(
    normalized_shape=512,
    eps=1e-5,
    elementwise_affine=True  # Learnable gamma and beta
)
```

### Verifying Correctness

```python
def verify_layer_norm():
    """Verify our implementation matches PyTorch."""
    torch.manual_seed(42)
    
    # Test 1: Simple 2D input
    x = torch.randn(4, 256)
    
    ln_torch = nn.LayerNorm(256)
    ln_custom = LayerNormFromScratch(256)
    
    # Copy parameters
    ln_custom.gamma.data = ln_torch.weight.data.clone()
    ln_custom.beta.data = ln_torch.bias.data.clone()
    
    out_torch = ln_torch(x)
    out_custom = ln_custom(x)
    
    print(f"Max difference (2D): {(out_torch - out_custom).abs().max():.2e}")
    
    # Test 2: 4D input (image-like)
    x = torch.randn(2, 64, 8, 8)
    
    ln_torch = nn.LayerNorm([64, 8, 8])
    ln_custom = LayerNormFromScratch([64, 8, 8])
    
    ln_custom.gamma.data = ln_torch.weight.data.clone()
    ln_custom.beta.data = ln_torch.bias.data.clone()
    
    out_torch = ln_torch(x)
    out_custom = ln_custom(x)
    
    print(f"Max difference (4D): {(out_torch - out_custom).abs().max():.2e}")

verify_layer_norm()
```

## Gradient Derivation

### Backward Pass

For LayerNorm with input $x \in \mathbb{R}^{H}$ and output $y$:

**Gradient w.r.t. learnable parameters:**

$$\frac{\partial \mathcal{L}}{\partial \gamma_i} = \frac{\partial \mathcal{L}}{\partial y_i} \cdot \hat{x}_i$$

$$\frac{\partial \mathcal{L}}{\partial \beta_i} = \frac{\partial \mathcal{L}}{\partial y_i}$$

**Gradient w.r.t. normalized input:**

$$\frac{\partial \mathcal{L}}{\partial \hat{x}_i} = \frac{\partial \mathcal{L}}{\partial y_i} \cdot \gamma_i$$

**Gradient w.r.t. input:**

Let $\sigma = \sqrt{\sigma^2 + \epsilon}$. Then:

$$\frac{\partial \mathcal{L}}{\partial x_i} = \frac{1}{\sigma} \left[ \frac{\partial \mathcal{L}}{\partial \hat{x}_i} - \frac{1}{H} \sum_{j=1}^{H} \frac{\partial \mathcal{L}}{\partial \hat{x}_j} - \frac{\hat{x}_i}{H} \sum_{j=1}^{H} \frac{\partial \mathcal{L}}{\partial \hat{x}_j} \hat{x}_j \right]$$

### Custom Autograd Implementation

```python
class LayerNormFunction(torch.autograd.Function):
    """Custom autograd function for LayerNorm."""
    
    @staticmethod
    def forward(ctx, x, gamma, beta, eps=1e-5):
        # Compute statistics over last dimension
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        std = torch.sqrt(var + eps)
        
        # Normalize
        x_centered = x - mean
        x_norm = x_centered / std
        
        # Scale and shift
        out = gamma * x_norm + beta
        
        # Save for backward
        ctx.save_for_backward(x_norm, gamma, std)
        ctx.eps = eps
        
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        x_norm, gamma, std = ctx.saved_tensors
        H = x_norm.shape[-1]
        
        # Gradient w.r.t. gamma and beta
        grad_gamma = (grad_output * x_norm).sum(dim=tuple(range(grad_output.dim()-1)))
        grad_beta = grad_output.sum(dim=tuple(range(grad_output.dim()-1)))
        
        # Gradient w.r.t. x_norm
        grad_x_norm = grad_output * gamma
        
        # Gradient w.r.t. x (using the derived formula)
        grad_x = (1.0 / std) * (
            grad_x_norm 
            - grad_x_norm.mean(dim=-1, keepdim=True)
            - x_norm * (grad_x_norm * x_norm).mean(dim=-1, keepdim=True)
        )
        
        return grad_x, grad_gamma, grad_beta, None
```

## Layer Normalization in Transformers

### Standard Transformer Architecture

LayerNorm is applied in two places in each Transformer block:

1. **After attention** (with residual connection)
2. **After feed-forward network** (with residual connection)

```python
class TransformerBlock(nn.Module):
    """Standard Transformer block with LayerNorm (Post-LN)."""
    
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        
        # Multi-head attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, attn_mask=None, key_padding_mask=None):
        # Self-attention with residual and LayerNorm
        attn_out, _ = self.self_attn(x, x, x, attn_mask=attn_mask,
                                      key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.dropout1(attn_out))
        
        # FFN with residual and LayerNorm
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        
        return x
```

### Pre-Layer Normalization (Pre-LN)

Modern architectures often use Pre-LN for improved training stability:

```python
class PreLNTransformerBlock(nn.Module):
    """Transformer block with Pre-Layer Normalization."""
    
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),  # GELU is common in modern transformers
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, attn_mask=None, key_padding_mask=None):
        # Pre-LN: normalize before attention
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm,
                                      attn_mask=attn_mask,
                                      key_padding_mask=key_padding_mask)
        x = x + self.dropout1(attn_out)
        
        # Pre-LN: normalize before FFN
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + self.dropout2(ffn_out)
        
        return x
```

### Comparison: Post-LN vs Pre-LN

| Aspect | Post-LN | Pre-LN |
|--------|---------|--------|
| Gradient flow | Can have issues in deep networks | More stable gradients |
| Training speed | May require warmup | Often faster to converge |
| Final performance | Slightly better (with proper training) | Good, more robust |
| Ease of training | Requires careful tuning | More forgiving |

## Layer Normalization in RNNs

### LSTM with LayerNorm

```python
class LayerNormLSTMCell(nn.Module):
    """LSTM cell with Layer Normalization."""
    
    def __init__(self, input_size, hidden_size):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Combined input-to-hidden weights
        self.W_ih = nn.Linear(input_size, 4 * hidden_size, bias=False)
        self.W_hh = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        
        # Layer normalization for each gate
        self.ln_i = nn.LayerNorm(hidden_size)  # Input gate
        self.ln_f = nn.LayerNorm(hidden_size)  # Forget gate
        self.ln_g = nn.LayerNorm(hidden_size)  # Cell gate
        self.ln_o = nn.LayerNorm(hidden_size)  # Output gate
        self.ln_c = nn.LayerNorm(hidden_size)  # Cell state (for output)
    
    def forward(self, x, states):
        """
        Args:
            x: Input of shape (batch, input_size)
            states: Tuple of (h, c), each (batch, hidden_size)
        """
        h, c = states
        
        # Compute all gates at once
        gates = self.W_ih(x) + self.W_hh(h)
        
        # Split into individual gates
        i_gate, f_gate, g_gate, o_gate = gates.chunk(4, dim=1)
        
        # Apply LayerNorm and activations
        i = torch.sigmoid(self.ln_i(i_gate))
        f = torch.sigmoid(self.ln_f(f_gate))
        g = torch.tanh(self.ln_g(g_gate))
        o = torch.sigmoid(self.ln_o(o_gate))
        
        # Update cell state
        c_new = f * c + i * g
        
        # Compute hidden state
        h_new = o * torch.tanh(self.ln_c(c_new))
        
        return h_new, c_new


class LayerNormLSTM(nn.Module):
    """Multi-layer LSTM with LayerNorm."""
    
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.cells = nn.ModuleList([
            LayerNormLSTMCell(
                input_size if i == 0 else hidden_size,
                hidden_size
            )
            for i in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x, states=None):
        """
        Args:
            x: Input of shape (batch, seq_len, input_size)
            states: Optional initial states
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Initialize states if not provided
        if states is None:
            h = [torch.zeros(batch_size, self.hidden_size, device=device)
                 for _ in range(self.num_layers)]
            c = [torch.zeros(batch_size, self.hidden_size, device=device)
                 for _ in range(self.num_layers)]
        else:
            h, c = states
        
        outputs = []
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            
            for layer in range(self.num_layers):
                h[layer], c[layer] = self.cells[layer](x_t, (h[layer], c[layer]))
                x_t = self.dropout(h[layer]) if layer < self.num_layers - 1 else h[layer]
            
            outputs.append(h[-1].unsqueeze(1))
        
        return torch.cat(outputs, dim=1), (h, c)
```

## Batch Independence Demonstration

```python
def demonstrate_batch_independence():
    """Show that LayerNorm is independent of batch size."""
    
    ln = nn.LayerNorm(256)
    
    # Same sample, different batch contexts
    x_single = torch.randn(1, 256)
    
    # Replicate to different batch sizes
    batch_sizes = [1, 2, 8, 32, 128]
    
    print("LayerNorm output for same sample in different batch sizes:")
    print("-" * 60)
    
    for bs in batch_sizes:
        # Create batch with our sample at position 0
        x_batch = torch.randn(bs, 256)
        x_batch[0] = x_single[0]
        
        out = ln(x_batch)
        
        print(f"Batch size {bs:3d}: first 5 values = {out[0, :5].tolist()}")
    
    print("\nObservation: Output is IDENTICAL regardless of batch size!")

demonstrate_batch_independence()
```

**Output:**
```
LayerNorm output for same sample in different batch sizes:
------------------------------------------------------------
Batch size   1: first 5 values = [0.234, -1.456, 0.789, ...]
Batch size   2: first 5 values = [0.234, -1.456, 0.789, ...]
Batch size   8: first 5 values = [0.234, -1.456, 0.789, ...]
Batch size  32: first 5 values = [0.234, -1.456, 0.789, ...]
Batch size 128: first 5 values = [0.234, -1.456, 0.789, ...]

Observation: Output is IDENTICAL regardless of batch size!
```

## Comparison with Batch Normalization

```python
def compare_ln_bn():
    """Compare LayerNorm and BatchNorm behavior."""
    
    print("=" * 60)
    print("LayerNorm vs BatchNorm Comparison")
    print("=" * 60)
    
    # Create data
    torch.manual_seed(42)
    x = torch.randn(8, 256)
    
    ln = nn.LayerNorm(256)
    bn = nn.BatchNorm1d(256)
    
    # Apply both
    out_ln = ln(x)
    out_bn = bn(x)
    
    print("\n1. Statistics after normalization:")
    print(f"   LayerNorm:  mean per sample  = {out_ln.mean(dim=1)[:3].tolist()}")
    print(f"   BatchNorm:  mean per feature = {out_bn.mean(dim=0)[:3].tolist()}")
    
    print("\n2. Same behavior in train/eval:")
    ln.eval()
    bn.eval()
    
    out_ln_eval = ln(x)
    out_bn_eval = bn(x)
    
    ln_diff = (out_ln - out_ln_eval).abs().max().item()
    bn_diff = (out_bn - out_bn_eval).abs().max().item()
    
    print(f"   LayerNorm train/eval diff: {ln_diff:.6f}")
    print(f"   BatchNorm train/eval diff: {bn_diff:.6f}")
    
    print("\n3. Single sample behavior:")
    x_single = torch.randn(1, 256)
    
    ln.train()
    bn.train()
    
    out_ln_single = ln(x_single)
    # BatchNorm with batch=1 is problematic!
    try:
        out_bn_single = bn(x_single)
        bn_single_std = out_bn_single.std().item()
    except:
        bn_single_std = "ERROR"
    
    print(f"   LayerNorm std with batch=1: {out_ln_single.std():.4f}")
    print(f"   BatchNorm std with batch=1: {bn_single_std}")

compare_ln_bn()
```

## When to Use Layer Normalization

### Good Use Cases

✅ **Transformers** (BERT, GPT, T5, etc.)  
✅ **RNNs and LSTMs** (stabilizes training)  
✅ **Variable or small batch sizes**  
✅ **Online learning** (batch size = 1)  
✅ **Sequence models** with variable-length inputs  
✅ **When train/eval consistency is important**

### Avoid When

❌ **Image classification with CNNs** (BatchNorm often better)  
❌ **Large batch training** where BatchNorm is stable  
❌ **Style transfer** (InstanceNorm is preferred)

## Best Practices

### 1. Placement in Architecture

```python
# For Transformers: Post-LN (original) or Pre-LN (modern)
class PostLNBlock(nn.Module):
    def forward(self, x):
        x = self.norm1(x + self.attn(x))  # Normalize after residual
        x = self.norm2(x + self.ffn(x))
        return x

class PreLNBlock(nn.Module):
    def forward(self, x):
        x = x + self.attn(self.norm1(x))  # Normalize before sublayer
        x = x + self.ffn(self.norm2(x))
        return x
```

### 2. Initialization

```python
# Default initialization (gamma=1, beta=0) is usually good
ln = nn.LayerNorm(512)

# For very deep networks, consider scaling
def scale_init(module):
    if isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
```

### 3. Combined with Dropout

```python
# Common pattern in Transformers
class TransformerSublayer(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, sublayer_fn):
        return self.norm(x + self.dropout(sublayer_fn(x)))
```

## Summary

Layer Normalization is essential for:

1. **Transformers** - standard choice for all major architectures
2. **RNNs** - improves training stability
3. **Small/variable batches** - no batch dependence

Key properties:
- Normalizes across **features**, not batch
- **Same behavior** in training and inference
- **No running statistics** needed
- **Batch-independent** computation

## References

1. Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). Layer Normalization. *arXiv preprint arXiv:1607.06450*.

2. Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS*.

3. Xiong, R., et al. (2020). On Layer Normalization in the Transformer Architecture. *ICML*.
