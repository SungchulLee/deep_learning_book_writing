# Quantization for LLMs

## Introduction

Quantization reduces model size and accelerates inference by representing weights and activations with lower-precision data types. For LLMs, this is critical for deployment on consumer hardware and reducing serving costs.

## Fundamentals

### Precision Formats

| Format | Bits | Range | Use Case |
|--------|------|-------|----------|
| FP32 | 32 | ±3.4e38 | Training (baseline) |
| FP16 | 16 | ±65504 | Mixed precision training |
| BF16 | 16 | ±3.4e38 | Training (better range) |
| INT8 | 8 | -128 to 127 | Inference |
| INT4 | 4 | -8 to 7 | LLM inference |
| FP8 | 8 | Various | Emerging standard |

### Memory Savings

For a 7B parameter model:

| Precision | Size | Memory |
|-----------|------|--------|
| FP32 | 28 GB | ~32 GB |
| FP16/BF16 | 14 GB | ~16 GB |
| INT8 | 7 GB | ~8 GB |
| INT4 | 3.5 GB | ~4 GB |

## Quantization Theory

### Linear Quantization

Map floating-point values to integers:

$$
x_q = \text{round}\left(\frac{x}{\Delta}\right) + z
$$

$$
\hat{x} = \Delta(x_q - z)
$$

Where:
- $\Delta$ = scale factor
- $z$ = zero point
- $x_q$ = quantized value

### Scale and Zero Point Computation

**Symmetric quantization** ($z = 0$):

$$
\Delta = \frac{\max(|x|)}{2^{b-1} - 1}
$$

**Asymmetric quantization**:

$$
\Delta = \frac{x_{max} - x_{min}}{2^b - 1}, \quad z = \text{round}\left(-\frac{x_{min}}{\Delta}\right)
$$

### Quantization Error

The error from quantization:

$$
\epsilon = x - \hat{x} = x - \Delta \cdot \text{round}\left(\frac{x}{\Delta}\right)
$$

For uniform distribution of values within a bin:

$$
\mathbb{E}[\epsilon^2] = \frac{\Delta^2}{12}
$$

## Weight Quantization Methods

### Post-Training Quantization (PTQ)

Quantize pre-trained model without retraining.

```python
import torch
import torch.nn as nn
from typing import Tuple, Optional


def compute_scale_zero(
    tensor: torch.Tensor,
    bits: int = 8,
    symmetric: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute quantization parameters.
    """
    if symmetric:
        max_val = tensor.abs().max()
        scale = max_val / (2 ** (bits - 1) - 1)
        zero_point = torch.zeros(1, device=tensor.device)
    else:
        min_val, max_val = tensor.min(), tensor.max()
        scale = (max_val - min_val) / (2 ** bits - 1)
        zero_point = torch.round(-min_val / scale)
    
    return scale, zero_point


def quantize_tensor(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    bits: int = 8
) -> torch.Tensor:
    """
    Quantize a tensor to fixed-point representation.
    """
    q_min = -(2 ** (bits - 1))
    q_max = 2 ** (bits - 1) - 1
    
    quantized = torch.round(tensor / scale) + zero_point
    quantized = torch.clamp(quantized, q_min, q_max)
    
    return quantized.to(torch.int8)


def dequantize_tensor(
    quantized: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor
) -> torch.Tensor:
    """
    Dequantize back to floating point.
    """
    return scale * (quantized.float() - zero_point)


class QuantizedLinear(nn.Module):
    """
    Linear layer with quantized weights.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int = 8,
        group_size: Optional[int] = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size or in_features
        
        # Number of groups for group-wise quantization
        self.num_groups = in_features // self.group_size
        
        # Quantized weights (stored as int8)
        self.register_buffer(
            'weight_quantized',
            torch.zeros(out_features, in_features, dtype=torch.int8)
        )
        
        # Scale per group
        self.register_buffer(
            'scale',
            torch.ones(out_features, self.num_groups)
        )
        
        # Zero point per group
        self.register_buffer(
            'zero_point',
            torch.zeros(out_features, self.num_groups)
        )
        
        # Optional bias
        self.bias = nn.Parameter(torch.zeros(out_features))
    
    @classmethod
    def from_float(
        cls,
        linear: nn.Linear,
        bits: int = 8,
        group_size: Optional[int] = None
    ) -> 'QuantizedLinear':
        """
        Create quantized layer from floating-point layer.
        """
        quant_linear = cls(
            linear.in_features,
            linear.out_features,
            bits=bits,
            group_size=group_size
        )
        
        weight = linear.weight.data
        group_size = quant_linear.group_size
        
        # Quantize each group
        for i in range(quant_linear.num_groups):
            start = i * group_size
            end = (i + 1) * group_size
            
            group_weight = weight[:, start:end]
            scale, zp = compute_scale_zero(group_weight, bits)
            
            quant_linear.scale[:, i] = scale
            quant_linear.zero_point[:, i] = zp
            quant_linear.weight_quantized[:, start:end] = quantize_tensor(
                group_weight, scale, zp, bits
            )
        
        if linear.bias is not None:
            quant_linear.bias.data = linear.bias.data.clone()
        
        return quant_linear
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize weights
        weight = torch.zeros(
            self.out_features, self.in_features,
            device=x.device, dtype=x.dtype
        )
        
        for i in range(self.num_groups):
            start = i * self.group_size
            end = (i + 1) * self.group_size
            
            weight[:, start:end] = dequantize_tensor(
                self.weight_quantized[:, start:end],
                self.scale[:, i:i+1],
                self.zero_point[:, i:i+1]
            )
        
        return nn.functional.linear(x, weight, self.bias)
```

### GPTQ (Accurate Post-Training Quantization)

GPTQ uses second-order information to minimize quantization error:

$$
\arg\min_{W_q} \|WX - W_q X\|_2^2
$$

Key insight: Quantize one weight at a time, adjusting remaining weights to compensate.

```python
import torch
import torch.nn as nn
from typing import List


class GPTQ:
    """
    GPTQ quantization algorithm.
    
    Quantizes weights column by column, using Hessian information
    to minimize output error.
    """
    
    def __init__(
        self,
        layer: nn.Linear,
        bits: int = 4,
        group_size: int = 128,
        block_size: int = 128
    ):
        self.layer = layer
        self.bits = bits
        self.group_size = group_size
        self.block_size = block_size
        
        self.rows = layer.weight.shape[0]
        self.cols = layer.weight.shape[1]
        
        # Hessian accumulator
        self.H = torch.zeros(self.cols, self.cols, device=layer.weight.device)
        self.nsamples = 0
    
    def add_batch(self, inp: torch.Tensor):
        """
        Accumulate Hessian from input batch.
        
        H = X^T X (input outer product)
        """
        if len(inp.shape) == 3:
            inp = inp.reshape(-1, inp.shape[-1])
        
        self.H += inp.T @ inp
        self.nsamples += inp.shape[0]
    
    def quantize(self) -> torch.Tensor:
        """
        Perform GPTQ quantization.
        """
        W = self.layer.weight.data.clone()
        
        # Finalize Hessian
        H = self.H / self.nsamples
        
        # Add damping for numerical stability
        damp = 0.01 * torch.diag(H).mean()
        H += damp * torch.eye(self.cols, device=H.device)
        
        # Cholesky decomposition
        H_inv = torch.linalg.cholesky(H)
        H_inv = torch.cholesky_inverse(H_inv)
        
        # Quantized weights
        Q = torch.zeros_like(W)
        
        # Process in blocks
        for i1 in range(0, self.cols, self.block_size):
            i2 = min(i1 + self.block_size, self.cols)
            
            W_block = W[:, i1:i2].clone()
            Q_block = torch.zeros_like(W_block)
            Err_block = torch.zeros_like(W_block)
            H_inv_block = H_inv[i1:i2, i1:i2]
            
            for i in range(i2 - i1):
                w = W_block[:, i]
                d = H_inv_block[i, i]
                
                # Determine group for this column
                group_idx = (i1 + i) // self.group_size
                
                # Compute scale for group
                group_start = (group_idx * self.group_size) - i1
                group_end = min(group_start + self.group_size, i2 - i1)
                
                if i == max(0, group_start):
                    # Compute scale at start of group
                    group_weights = W_block[:, max(0, group_start):group_end]
                    scale, zp = compute_scale_zero(group_weights, self.bits)
                
                # Quantize
                q = quantize_tensor(w, scale, zp, self.bits)
                Q_block[:, i] = q.float()
                
                # Compute error
                err = (w - dequantize_tensor(q, scale, zp)) / d
                
                # Update remaining weights in block
                W_block[:, i:] -= err.unsqueeze(1) * H_inv_block[i, i:].unsqueeze(0)
            
            Q[:, i1:i2] = Q_block
        
        return Q
```

### AWQ (Activation-aware Weight Quantization)

AWQ identifies and protects salient weights based on activation magnitudes:

$$
\text{saliency}_j = \|X_j\|_2
$$

Weights with high corresponding activation magnitudes are scaled up before quantization, then scaled back during inference.

```python
def compute_awq_scales(
    weight: torch.Tensor,
    activations: torch.Tensor,
    bits: int = 4
) -> torch.Tensor:
    """
    Compute AWQ scaling factors.
    
    Scale = activation_magnitude^alpha where alpha balances
    weight and activation quantization error.
    """
    # Compute activation magnitudes
    act_scales = activations.abs().mean(dim=0)
    
    # Compute weight magnitudes
    weight_scales = weight.abs().mean(dim=0)
    
    # Optimal scale (balances weight and activation error)
    # Higher activation = protect more
    alpha = 0.5  # Tunable hyperparameter
    scales = act_scales.pow(alpha) / weight_scales.pow(1 - alpha)
    
    # Clip scales
    scales = torch.clamp(scales, min=1e-5, max=1e5)
    
    return scales
```

## Activation Quantization

### Dynamic Quantization

Compute scales at runtime for each input:

```python
class DynamicQuantizedLinear(nn.Module):
    """
    Linear with dynamically quantized activations.
    """
    
    def __init__(self, in_features: int, out_features: int, bits: int = 8):
        super().__init__()
        self.bits = bits
        # Pre-quantized weights
        self.weight_scale = nn.Parameter(torch.ones(1))
        self.register_buffer(
            'weight_q', 
            torch.zeros(out_features, in_features, dtype=torch.int8)
        )
        self.bias = nn.Parameter(torch.zeros(out_features))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dynamically quantize input
        x_scale, x_zp = compute_scale_zero(x, self.bits)
        x_q = quantize_tensor(x, x_scale, x_zp, self.bits)
        
        # Integer matrix multiply (simulated)
        # In practice, use specialized int8 kernels
        out_q = torch.matmul(x_q.float(), self.weight_q.T.float())
        
        # Dequantize output
        out = out_q * (x_scale * self.weight_scale)
        
        return out + self.bias
```

### Static Quantization

Calibrate scales offline using representative data:

```python
class CalibrationCollector:
    """
    Collect activation statistics for static quantization.
    """
    
    def __init__(self, method: str = 'minmax'):
        self.method = method
        self.min_vals = []
        self.max_vals = []
        self.histograms = []
    
    def collect(self, tensor: torch.Tensor):
        """Record statistics from a tensor."""
        self.min_vals.append(tensor.min().item())
        self.max_vals.append(tensor.max().item())
        
        if self.method == 'entropy':
            # Histogram for entropy calibration
            hist = torch.histc(tensor, bins=2048)
            self.histograms.append(hist)
    
    def compute_scale(self, bits: int = 8) -> Tuple[float, float]:
        """Compute final scale and zero point."""
        if self.method == 'minmax':
            min_val = min(self.min_vals)
            max_val = max(self.max_vals)
            
        elif self.method == 'percentile':
            # Use 99.9th percentile
            all_mins = torch.tensor(self.min_vals)
            all_maxs = torch.tensor(self.max_vals)
            min_val = torch.quantile(all_mins, 0.001).item()
            max_val = torch.quantile(all_maxs, 0.999).item()
            
        scale = (max_val - min_val) / (2 ** bits - 1)
        zero_point = round(-min_val / scale)
        
        return scale, zero_point
```

## LLM-Specific Techniques

### KV Cache Quantization

Quantize cached keys and values to reduce memory:

```python
class QuantizedKVCache:
    """
    KV cache with quantized storage.
    """
    
    def __init__(self, bits: int = 8):
        self.bits = bits
        self.keys_q = None
        self.values_q = None
        self.key_scales = []
        self.value_scales = []
    
    def update(
        self,
        new_keys: torch.Tensor,
        new_values: torch.Tensor
    ):
        """Add new keys/values to cache."""
        # Quantize new entries
        k_scale, k_zp = compute_scale_zero(new_keys, self.bits)
        v_scale, v_zp = compute_scale_zero(new_values, self.bits)
        
        new_k_q = quantize_tensor(new_keys, k_scale, k_zp, self.bits)
        new_v_q = quantize_tensor(new_values, v_scale, v_zp, self.bits)
        
        # Append to cache
        if self.keys_q is None:
            self.keys_q = new_k_q
            self.values_q = new_v_q
        else:
            self.keys_q = torch.cat([self.keys_q, new_k_q], dim=2)
            self.values_q = torch.cat([self.values_q, new_v_q], dim=2)
        
        self.key_scales.append((k_scale, k_zp))
        self.value_scales.append((v_scale, v_zp))
    
    def get_keys_values(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve dequantized keys and values."""
        # For simplicity, use most recent scale
        # In practice, use per-token or grouped scales
        k_scale, k_zp = self.key_scales[-1]
        v_scale, v_zp = self.value_scales[-1]
        
        keys = dequantize_tensor(self.keys_q, k_scale, k_zp)
        values = dequantize_tensor(self.values_q, v_scale, v_zp)
        
        return keys, values
```

### Mixed-Precision Quantization

Different bits for different components:

```python
MIXED_PRECISION_CONFIG = {
    'embedding': 8,      # Embeddings: 8-bit
    'attention.qkv': 4,  # QKV projections: 4-bit
    'attention.out': 4,  # Output projection: 4-bit
    'mlp.gate': 4,       # MLP layers: 4-bit
    'mlp.up': 4,
    'mlp.down': 4,
    'lm_head': 8,        # Output head: 8-bit (sensitive)
    'layernorm': 32,     # Keep full precision
}


def quantize_model_mixed(
    model: nn.Module,
    config: dict
) -> nn.Module:
    """
    Apply mixed-precision quantization.
    """
    for name, module in model.named_modules():
        for pattern, bits in config.items():
            if pattern in name:
                if isinstance(module, nn.Linear) and bits < 32:
                    # Replace with quantized version
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1]
                    parent = model.get_submodule(parent_name)
                    
                    quant_module = QuantizedLinear.from_float(module, bits=bits)
                    setattr(parent, child_name, quant_module)
                break
    
    return model
```

## Quantization-Aware Training (QAT)

### Straight-Through Estimator (STE)

Gradient through quantization:

$$
\frac{\partial L}{\partial w} = \frac{\partial L}{\partial \hat{w}} \cdot \mathbf{1}_{|w| \leq \text{clip}}
$$

```python
class STEQuantize(torch.autograd.Function):
    """
    Straight-Through Estimator for quantization.
    """
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: torch.Tensor, bits: int):
        q_min = -(2 ** (bits - 1))
        q_max = 2 ** (bits - 1) - 1
        
        # Quantize
        x_q = torch.round(x / scale)
        x_q = torch.clamp(x_q, q_min, q_max)
        
        # Dequantize
        x_dq = x_q * scale
        
        return x_dq
    
    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through: pass gradients unchanged
        return grad_output, None, None


class QATLinear(nn.Module):
    """
    Linear layer with quantization-aware training.
    """
    
    def __init__(self, in_features: int, out_features: int, bits: int = 4):
        super().__init__()
        self.bits = bits
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Learnable scale
        self.weight_scale = nn.Parameter(torch.ones(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fake quantize weights during training
        w_q = STEQuantize.apply(self.weight, self.weight_scale, self.bits)
        
        return nn.functional.linear(x, w_q, self.bias)
```

## Evaluation and Comparison

### Perplexity Impact

Typical perplexity increase on WikiText-2:

| Method | Bits | PPL Increase |
|--------|------|--------------|
| Round-to-nearest | 8 | +0.1 |
| Round-to-nearest | 4 | +0.5-2.0 |
| GPTQ | 4 | +0.1-0.3 |
| AWQ | 4 | +0.1-0.3 |
| GPTQ | 3 | +0.3-1.0 |

### Speed Comparison

Inference speedup on A100 (LLaMA-7B):

| Precision | Tokens/sec | Memory |
|-----------|------------|--------|
| FP16 | 100 | 14 GB |
| INT8 | 150 | 7 GB |
| INT4 (GPTQ) | 180 | 4 GB |
| INT4 (AWQ) | 200 | 4 GB |

## Practical Guidelines

### Recommended Configurations

| Use Case | Method | Bits | Notes |
|----------|--------|------|-------|
| Server (quality) | FP16/BF16 | 16 | Best quality |
| Server (balanced) | GPTQ/AWQ | 4 | Good trade-off |
| Consumer GPU | GPTQ | 4 | Fits 7B in 8GB |
| Edge device | AWQ | 4 | With group size 128 |
| Extreme compression | GPTQ | 3 | Noticeable quality loss |

### Best Practices

1. **Calibration data**: Use 128-512 samples representative of target domain
2. **Group size**: 128 provides good balance (vs per-tensor or per-channel)
3. **Sensitive layers**: Keep embeddings and output head at higher precision
4. **Validation**: Always evaluate perplexity and downstream tasks
5. **Combine techniques**: Quantization + KV cache + Flash Attention

## Summary

Quantization is essential for LLM deployment:

1. **4-bit weights**: Near-lossless with GPTQ/AWQ
2. **Memory reduction**: 4-8x smaller models
3. **Speed improvement**: 1.5-2x faster inference
4. **Accessibility**: Run 7B models on consumer GPUs

Key trade-off: **Quality vs. Memory vs. Speed**

## References

1. Frantar, E., et al. (2023). "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers."
2. Lin, J., et al. (2023). "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration."
3. Dettmers, T., et al. (2022). "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale."
4. Dettmers, T., et al. (2023). "QLoRA: Efficient Finetuning of Quantized LLMs."
5. Xiao, G., et al. (2023). "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models."
