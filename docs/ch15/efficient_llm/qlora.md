# QLoRA: Quantized Low-Rank Adaptation

## Learning Objectives

- Understand how QLoRA enables fine-tuning large models on consumer hardware
- Learn the key innovations: NF4 quantization, double quantization, paged optimizers
- Implement QLoRA components and understand memory savings
- Configure QLoRA for different model sizes and hardware constraints

## Introduction

QLoRA (Dettmers et al., 2023) makes fine-tuning large language models accessible on consumer hardware by combining 4-bit quantization of the base model with LoRA adapters in full precision. This enables fine-tuning a 65B parameter model on a single 48GB GPU—previously requiring multiple 80GB GPUs.

## Key Innovations

QLoRA introduces four key techniques:

1. **4-bit NormalFloat (NF4)** - Information-theoretically optimal quantization for normally distributed weights
2. **Double Quantization** - Quantize the quantization constants to save additional memory
3. **Paged Optimizers** - Use CPU memory for optimizer states during GPU memory spikes
4. **LoRA Adapters** - Train low-rank adapters in full precision (bfloat16)

## Memory Analysis

### Why Quantization Matters

| Model | FP16 Size | 4-bit Size | Savings |
|-------|-----------|------------|---------|
| 7B | 14 GB | 3.5 GB | 4× |
| 13B | 26 GB | 6.5 GB | 4× |
| 30B | 60 GB | 15 GB | 4× |
| 65B | 130 GB | 32.5 GB | 4× |

### Full Training Memory Breakdown

For fine-tuning with AdamW optimizer:

| Component | FP16 | QLoRA (4-bit) |
|-----------|------|---------------|
| Model weights | 2 bytes/param | 0.5 bytes/param |
| Gradients | 2 bytes/param | Only for LoRA (~0.1%) |
| Optimizer states | 8 bytes/param | Only for LoRA |
| Activations | Variable | Variable |

**Example: 7B model**
- Full FP16 fine-tuning: ~56 GB (weights + gradients + optimizer)
- QLoRA: ~6 GB (4-bit weights + LoRA overhead)

## NF4: 4-bit NormalFloat Quantization

### The Problem with Uniform Quantization

Standard INT4 uses uniform quantization levels:
```
INT4: -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7
```

But neural network weights follow a normal distribution centered at zero. Uniform quantization wastes precision on unlikely values.

### NormalFloat Solution

NF4 places quantization levels based on the normal distribution's quantiles, ensuring equal probability mass between adjacent levels:

```python
import torch
import scipy.stats as stats
import numpy as np


def compute_nf4_levels() -> torch.Tensor:
    """
    Compute NF4 quantization levels.
    
    16 levels placed at quantiles of N(0,1) such that each
    bin contains equal probability mass (1/16 each).
    """
    # For 4 bits, we have 16 levels
    # Place them at quantiles that divide N(0,1) into 16 equal-probability bins
    
    # Quantile positions (midpoints of 16 equal-probability bins)
    quantiles = np.linspace(0, 1, 17)  # 17 edges for 16 bins
    bin_midpoints = (quantiles[:-1] + quantiles[1:]) / 2
    
    # Get the values at these quantiles from N(0,1)
    levels = stats.norm.ppf(bin_midpoints)
    
    # Normalize to [-1, 1] range
    levels = levels / np.abs(levels).max()
    
    return torch.tensor(levels, dtype=torch.float32)


# Pre-computed NF4 levels (normalized to [-1, 1])
NF4_LEVELS = torch.tensor([
    -1.0000, -0.6962, -0.5251, -0.3949,
    -0.2844, -0.1848, -0.0911,  0.0000,
     0.0796,  0.1609,  0.2461,  0.3379,
     0.4407,  0.5626,  0.7230,  1.0000
])

# Notice: levels are denser near zero where most weights lie
```

### NF4 Quantization Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class NF4Tensor:
    """
    4-bit NormalFloat quantized tensor.
    
    Stores weights in 4-bit NF4 format with per-block scaling.
    """
    
    # Pre-computed NF4 quantization levels
    NF4_LEVELS = torch.tensor([
        -1.0000, -0.6962, -0.5251, -0.3949,
        -0.2844, -0.1848, -0.0911,  0.0000,
         0.0796,  0.1609,  0.2461,  0.3379,
         0.4407,  0.5626,  0.7230,  1.0000
    ])
    
    def __init__(
        self,
        data: torch.Tensor,
        block_size: int = 64,
        double_quant: bool = True
    ):
        """
        Quantize a tensor to NF4 format.
        
        Args:
            data: Tensor to quantize
            block_size: Number of elements per quantization block
            double_quant: Whether to apply double quantization to scales
        """
        self.shape = data.shape
        self.block_size = block_size
        self.double_quant = double_quant
        self.dtype = data.dtype
        self.device = data.device
        
        # Flatten for block processing
        flat = data.reshape(-1).float()
        
        # Pad to multiple of block_size
        self.numel = flat.numel()
        padded_size = ((self.numel + block_size - 1) // block_size) * block_size
        if padded_size > self.numel:
            flat = F.pad(flat, (0, padded_size - self.numel))
        
        # Reshape into blocks
        blocks = flat.reshape(-1, block_size)
        self.num_blocks = blocks.shape[0]
        
        # Compute per-block scales (absmax)
        self.scales = blocks.abs().max(dim=1).values
        
        # Normalize blocks
        normalized = blocks / (self.scales.unsqueeze(1) + 1e-8)
        
        # Quantize to nearest NF4 level
        nf4 = self.NF4_LEVELS.to(data.device)
        distances = (normalized.unsqueeze(-1) - nf4).abs()
        self.quantized = distances.argmin(dim=-1).to(torch.uint8)
        
        # Double quantization: quantize the scales themselves
        if double_quant:
            self.scale_scale = self.scales.abs().max()
            self.scales_quantized = (self.scales / (self.scale_scale + 1e-8) * 127).round().to(torch.int8)
        else:
            self.scale_scale = None
            self.scales_quantized = None
    
    def dequantize(self) -> torch.Tensor:
        """Dequantize back to full precision."""
        # Get NF4 values
        nf4 = self.NF4_LEVELS.to(self.device)
        values = nf4[self.quantized.long()]
        
        # Get scales
        if self.double_quant:
            scales = self.scales_quantized.float() / 127.0 * self.scale_scale
        else:
            scales = self.scales
        
        # Scale back
        dequantized = values * scales.unsqueeze(1)
        
        # Reshape and trim padding
        return dequantized.reshape(-1)[:self.numel].reshape(self.shape).to(self.dtype)
    
    def memory_size(self) -> dict:
        """Calculate memory usage."""
        # Quantized weights: 4 bits per element (stored as uint8 for simplicity)
        quant_bytes = self.quantized.numel()  # Could be halved with packing
        
        # Scales
        if self.double_quant:
            scale_bytes = self.scales_quantized.numel() + 4  # int8 scales + float32 scale_scale
        else:
            scale_bytes = self.scales.numel() * 4  # float32 scales
        
        return {
            'quantized_weights': quant_bytes,
            'scales': scale_bytes,
            'total': quant_bytes + scale_bytes,
            'compression_ratio': (self.numel * 4) / (quant_bytes + scale_bytes)  # vs float32
        }


class NF4Linear(nn.Module):
    """
    Linear layer with NF4 quantized weights.
    
    Weights are stored in 4-bit NF4 format and dequantized on-the-fly
    during forward pass.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        block_size: int = 64,
        double_quant: bool = True
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.double_quant = double_quant
        
        # Placeholder for quantized weights
        self.weight_nf4: Optional[NF4Tensor] = None
        
        # Bias stays in full precision
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
    
    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        block_size: int = 64,
        double_quant: bool = True
    ) -> 'NF4Linear':
        """Create NF4Linear from existing linear layer."""
        nf4_linear = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            block_size=block_size,
            double_quant=double_quant
        )
        
        # Quantize weights
        nf4_linear.weight_nf4 = NF4Tensor(
            linear.weight.data,
            block_size=block_size,
            double_quant=double_quant
        )
        
        # Copy bias
        if linear.bias is not None:
            nf4_linear.bias.data = linear.bias.data.clone()
        
        return nf4_linear
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize weights on-the-fly
        weight = self.weight_nf4.dequantize()
        return F.linear(x, weight, self.bias)
```

## Double Quantization

Double quantization applies quantization to the quantization constants (scales), providing additional memory savings:

```python
def double_quantize_scales(
    scales: torch.Tensor,
    block_size: int = 256
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply double quantization to scale factors.
    
    Instead of storing scales as FP32 (4 bytes each), quantize them
    to FP8/INT8 with their own scale factor.
    
    Memory per original block:
    - Without double quant: 4 bytes (FP32 scale)
    - With double quant: 1 byte (INT8 scale) + 4 bytes / 256 (scale of scales)
    
    Savings: ~75% on scale storage
    """
    # Group scales into blocks
    num_scales = scales.numel()
    padded_size = ((num_scales + block_size - 1) // block_size) * block_size
    
    scales_padded = F.pad(scales.reshape(-1), (0, padded_size - num_scales))
    scales_blocks = scales_padded.reshape(-1, block_size)
    
    # Compute scale of scales (one FP32 per block of 256 scales)
    scale_scales = scales_blocks.abs().max(dim=1).values
    
    # Quantize scales to INT8
    normalized = scales_blocks / (scale_scales.unsqueeze(1) + 1e-8)
    scales_int8 = (normalized * 127).round().clamp(-128, 127).to(torch.int8)
    
    return scales_int8, scale_scales, num_scales


def double_dequantize_scales(
    scales_int8: torch.Tensor,
    scale_scales: torch.Tensor,
    num_scales: int
) -> torch.Tensor:
    """Dequantize doubly-quantized scales."""
    scales_float = scales_int8.float() / 127.0 * scale_scales.unsqueeze(1)
    return scales_float.reshape(-1)[:num_scales]
```

## Paged Optimizers

During training, optimizer states can cause GPU memory spikes. Paged optimizers offload to CPU when needed:

```python
class PagedAdamW:
    """
    AdamW optimizer with CPU paging for memory spikes.
    
    Automatically moves optimizer states to CPU when GPU memory is low,
    and back to GPU when needed.
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        page_threshold: float = 0.9  # Page when GPU usage exceeds 90%
    ):
        self.params = list(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.page_threshold = page_threshold
        
        # Initialize optimizer states
        self.state = {}
        for p in self.params:
            self.state[p] = {
                'step': 0,
                'exp_avg': torch.zeros_like(p, device='cpu'),  # Start on CPU
                'exp_avg_sq': torch.zeros_like(p, device='cpu'),
                'on_gpu': False
            }
    
    def _check_memory(self) -> float:
        """Return GPU memory utilization (0-1)."""
        if torch.cuda.is_available():
            used = torch.cuda.memory_allocated()
            total = torch.cuda.get_device_properties(0).total_memory
            return used / total
        return 0.0
    
    def _page_to_cpu(self, param: torch.Tensor):
        """Move optimizer state to CPU."""
        state = self.state[param]
        if state['on_gpu']:
            state['exp_avg'] = state['exp_avg'].cpu()
            state['exp_avg_sq'] = state['exp_avg_sq'].cpu()
            state['on_gpu'] = False
    
    def _page_to_gpu(self, param: torch.Tensor):
        """Move optimizer state to GPU."""
        state = self.state[param]
        if not state['on_gpu']:
            device = param.device
            state['exp_avg'] = state['exp_avg'].to(device)
            state['exp_avg_sq'] = state['exp_avg_sq'].to(device)
            state['on_gpu'] = True
    
    def step(self):
        """Perform optimization step with automatic paging."""
        for p in self.params:
            if p.grad is None:
                continue
            
            # Check memory and page if needed
            if self._check_memory() > self.page_threshold:
                # Page out states for params we're not currently updating
                for other_p in self.params:
                    if other_p is not p:
                        self._page_to_cpu(other_p)
            
            # Ensure current param's state is on GPU
            self._page_to_gpu(p)
            
            state = self.state[p]
            state['step'] += 1
            
            # Get state tensors
            exp_avg = state['exp_avg']
            exp_avg_sq = state['exp_avg_sq']
            
            beta1, beta2 = self.betas
            
            # Weight decay
            if self.weight_decay != 0:
                p.data.add_(p.data, alpha=-self.lr * self.weight_decay)
            
            # Update biased first moment estimate
            exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)
            
            # Update biased second moment estimate
            exp_avg_sq.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)
            
            # Bias correction
            bias_correction1 = 1 - beta1 ** state['step']
            bias_correction2 = 1 - beta2 ** state['step']
            
            # Compute step
            denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(self.eps)
            step_size = self.lr / bias_correction1
            
            p.data.addcdiv_(exp_avg, denom, value=-step_size)
    
    def zero_grad(self):
        """Zero gradients."""
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
```

## Complete QLoRA Layer

```python
class QLoRALayer(nn.Module):
    """
    QLoRA: NF4-quantized base weights + full-precision LoRA adapters.
    
    Combines:
    - 4-bit NF4 quantization of frozen base weights
    - bfloat16 LoRA adapters for training
    """
    
    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        block_size: int = 64,
        double_quant: bool = True
    ):
        super().__init__()
        
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        # Quantize base weights to NF4
        self.base = NF4Linear.from_linear(
            original_layer,
            block_size=block_size,
            double_quant=double_quant
        )
        
        # Freeze base (it's already not trainable due to quantization)
        for param in self.base.parameters():
            param.requires_grad = False
        
        # LoRA adapters in bfloat16
        self.rank = rank
        self.scaling = alpha / rank
        
        self.lora_A = nn.Parameter(
            torch.randn(in_features, rank, dtype=torch.bfloat16) * 0.01
        )
        self.lora_B = nn.Parameter(
            torch.zeros(rank, out_features, dtype=torch.bfloat16)
        )
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base forward (dequantizes NF4 weights on the fly)
        base_output = self.base(x)
        
        # LoRA forward in bfloat16
        x_bf16 = x.to(torch.bfloat16)
        lora_output = self.dropout(x_bf16) @ self.lora_A @ self.lora_B
        lora_output = lora_output * self.scaling
        
        return base_output + lora_output.to(base_output.dtype)
    
    def memory_savings(self) -> dict:
        """Calculate memory savings vs full precision."""
        in_f = self.base.in_features
        out_f = self.base.out_features
        
        fp16_bytes = in_f * out_f * 2  # FP16 weights
        nf4_bytes = self.base.weight_nf4.memory_size()['total']
        lora_bytes = (self.lora_A.numel() + self.lora_B.numel()) * 2  # bfloat16
        
        return {
            'fp16_size': fp16_bytes,
            'qlora_size': nf4_bytes + lora_bytes,
            'compression_ratio': fp16_bytes / (nf4_bytes + lora_bytes)
        }
```

## Applying QLoRA to a Model

```python
from dataclasses import dataclass, field
from typing import Set


@dataclass
class QLoRAConfig:
    """Configuration for QLoRA."""
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: Set[str] = field(default_factory=lambda: {'q_proj', 'v_proj'})
    block_size: int = 64
    double_quant: bool = True


def apply_qlora(
    model: nn.Module,
    config: QLoRAConfig
) -> nn.Module:
    """Apply QLoRA to all matching modules."""
    replacements = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(target in name for target in config.target_modules):
                replacements.append((name, module))
    
    for name, module in replacements:
        parts = name.rsplit('.', 1)
        if len(parts) == 2:
            parent = model.get_submodule(parts[0])
            child_name = parts[1]
        else:
            parent = model
            child_name = name
        
        qlora_layer = QLoRALayer(
            module,
            rank=config.rank,
            alpha=config.alpha,
            dropout=config.dropout,
            block_size=config.block_size,
            double_quant=config.double_quant
        )
        setattr(parent, child_name, qlora_layer)
        
        print(f"Applied QLoRA to {name}")
    
    return model
```

## Memory Requirements by Model Size

| Model | FP16 Training | LoRA (FP16 base) | QLoRA |
|-------|---------------|------------------|-------|
| 7B | ~56 GB | ~16 GB | **~6 GB** |
| 13B | ~104 GB | ~28 GB | **~10 GB** |
| 30B | ~240 GB | ~65 GB | **~24 GB** |
| 65B | ~520 GB | ~140 GB | **~48 GB** |

## Best Practices

### Hyperparameters

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| Rank | 8-64 | Higher for complex tasks |
| Alpha | 16-32 | Typically 2× rank |
| Block size | 64 | Balance compression/accuracy |
| Double quant | True | Always use for memory savings |
| LR | 1e-4 to 2e-4 | Similar to LoRA |

### Hardware Requirements

| Model Size | Minimum GPU | Recommended |
|------------|-------------|-------------|
| 7B | 8 GB (RTX 3070) | 12 GB (RTX 3080) |
| 13B | 12 GB | 16 GB (RTX 4080) |
| 30B | 24 GB (RTX 3090) | 40 GB (A6000) |
| 65B | 40 GB (A100-40) | 48 GB (A6000-48) |

### Training Tips

1. **Use gradient checkpointing** for additional memory savings
2. **Use bfloat16** for LoRA adapters (better numerical stability than fp16)
3. **Batch size 1 + gradient accumulation** for memory-constrained setups
4. **Monitor for NaN** - quantization can cause instability

## Summary

QLoRA enables fine-tuning of very large models on consumer hardware through:

| Innovation | Benefit |
|------------|---------|
| NF4 Quantization | 4× compression with minimal quality loss |
| Double Quantization | Additional ~0.4 bits/param savings |
| Paged Optimizers | Handle memory spikes gracefully |
| BF16 LoRA | Full-precision adaptation on quantized base |

**Result**: Fine-tune a 65B model on a single 48GB GPU!

## References

1. Dettmers, T., et al. (2023). "QLoRA: Efficient Finetuning of Quantized LLMs." NeurIPS.
2. Dettmers, T., et al. (2022). "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale."
3. Dettmers, T., et al. (2023). "The case for 4-bit precision: k-bit Inference Scaling Laws."
