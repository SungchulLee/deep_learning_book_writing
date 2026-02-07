# Parameter-Efficient Fine-tuning (PEFT)

## Learning Objectives

- Understand why parameter-efficient methods are essential for LLMs
- Implement LoRA and understand its mathematical foundation
- Compare different PEFT methods: LoRA, QLoRA, Prefix Tuning, Adapters
- Choose the appropriate method based on constraints and requirements

## Introduction

Parameter-Efficient Fine-Tuning (PEFT) methods adapt large language models while training only a small fraction of parameters. This is essential for LLMs where full fine-tuning is prohibitively expensive or impossible.

### Why PEFT?

| Model | Parameters | Full FT Memory | LoRA Memory |
|-------|------------|----------------|-------------|
| BERT-base | 110M | ~2 GB | ~2 GB |
| LLaMA-7B | 7B | ~56 GB | ~8 GB |
| LLaMA-65B | 65B | ~520 GB | ~40 GB |

For general fine-tuning theory (gradual unfreezing, discriminative LR, catastrophic forgetting), see [Ch 8.1 Transfer Learning](../ch08/transfer_learning/).

## LoRA (Low-Rank Adaptation)

### Mathematical Foundation

Instead of updating weight matrix $W \in \mathbb{R}^{d_{out} \times d_{in}}$ directly, LoRA adds a low-rank decomposition:

$$
W' = W + \Delta W = W + BA
$$

Where:
- $B \in \mathbb{R}^{d_{out} \times r}$ (down-projection)
- $A \in \mathbb{R}^{r \times d_{in}}$ (up-projection)
- $r \ll \min(d_{in}, d_{out})$ is the rank

**Parameter savings**: Instead of $d_{in} \times d_{out}$ parameters, LoRA uses $r \times (d_{in} + d_{out})$.

For $d_{in} = d_{out} = 4096$ and $r = 8$:
- Full: 16.7M parameters
- LoRA: 65K parameters (0.4%)

### Scaling Factor

LoRA uses a scaling factor $\alpha$ to control the magnitude of updates:

$$
W' = W + \frac{\alpha}{r} BA
$$

Typically $\alpha = 2r$ or $\alpha = r$, so the effective scaling is constant regardless of rank choice.

### Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class LoRALayer(nn.Module):
    """
    LoRA (Low-Rank Adaptation) layer.
    
    Wraps an existing linear layer and adds trainable low-rank matrices.
    Original weights are frozen; only A and B are trained.
    """
    
    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.original = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        # Freeze original weights
        for param in self.original.parameters():
            param.requires_grad = False
        
        # Low-rank matrices
        # A is initialized with Kaiming/He initialization
        self.lora_A = nn.Parameter(torch.empty(in_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        
        # B is initialized to zero so ΔW = BA = 0 at start
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original forward pass (frozen)
        original_out = self.original(x)
        
        # LoRA forward pass
        # x @ A @ B * scaling
        lora_out = self.dropout(x) @ self.lora_A @ self.lora_B * self.scaling
        
        return original_out + lora_out
    
    def merge_weights(self) -> nn.Linear:
        """
        Merge LoRA weights into original layer for inference.
        
        Returns a standard nn.Linear with no overhead.
        """
        merged = nn.Linear(
            self.original.in_features,
            self.original.out_features,
            bias=self.original.bias is not None
        )
        
        # W' = W + (α/r) * B @ A
        merged.weight.data = self.original.weight.data + \
            (self.lora_A @ self.lora_B).T * self.scaling
        
        if self.original.bias is not None:
            merged.bias.data = self.original.bias.data
        
        return merged


class LoRAConfig:
    """Configuration for LoRA adaptation."""
    
    def __init__(
        self,
        rank: int = 8,
        alpha: float = 16,
        dropout: float = 0.0,
        target_modules: list = None
    ):
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        # Default: apply to attention projections
        self.target_modules = target_modules or ['q_proj', 'v_proj']


def apply_lora(
    model: nn.Module,
    config: LoRAConfig
) -> nn.Module:
    """
    Apply LoRA to specified modules in a model.
    
    Args:
        model: The model to adapt
        config: LoRA configuration
        
    Returns:
        Model with LoRA layers applied
    """
    for name, module in model.named_modules():
        # Check if this module should have LoRA applied
        if any(target in name for target in config.target_modules):
            if isinstance(module, nn.Linear):
                # Get parent module and attribute name
                parts = name.rsplit('.', 1)
                if len(parts) == 2:
                    parent_name, child_name = parts
                    parent = model.get_submodule(parent_name)
                else:
                    parent = model
                    child_name = name
                
                # Replace with LoRA layer
                lora_layer = LoRALayer(
                    module,
                    rank=config.rank,
                    alpha=config.alpha,
                    dropout=config.dropout
                )
                setattr(parent, child_name, lora_layer)
                print(f"Applied LoRA to {name}")
    
    return model


def get_lora_params(model: nn.Module) -> list:
    """Get only the LoRA parameters for optimization."""
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora_' in name:
            lora_params.append(param)
    return lora_params


def count_parameters(model: nn.Module) -> dict:
    """Count trainable and total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        'trainable': trainable,
        'total': total,
        'percentage': 100 * trainable / total
    }
```

### Which Layers to Target?

Research shows different layers have different impacts:

| Target Modules | Quality | Parameters |
|----------------|---------|------------|
| Query only (q) | Good | Lowest |
| Query + Value (q, v) | Better | Low |
| All attention (q, k, v, o) | Best | Medium |
| Attention + MLP | Marginal gain | High |

**Recommendation**: Start with `q_proj` and `v_proj` for best quality/parameter trade-off.

### Rank Selection

| Rank | Parameters | Use Case |
|------|------------|----------|
| 4 | Minimal | Simple tasks, very limited memory |
| 8 | Low | Default, good for most tasks |
| 16 | Medium | Complex tasks, large datasets |
| 64 | Higher | Near full fine-tuning quality |
| 256+ | High | Approaches full fine-tuning |

## QLoRA (Quantized LoRA)

QLoRA (Dettmers et al., 2023) enables fine-tuning very large models on consumer hardware by combining:

1. **4-bit NormalFloat (NF4) quantization** of base model
2. **LoRA adapters** in full precision (bfloat16)
3. **Double quantization** of quantization constants
4. **Paged optimizers** for memory management

```python
import torch
import torch.nn as nn
from typing import Tuple


class NF4Linear(nn.Module):
    """
    Simplified NF4 (4-bit NormalFloat) quantized linear layer.
    
    Note: This is a reference implementation. Production use should
    leverage bitsandbytes or similar optimized libraries.
    """
    
    # NF4 quantization levels (normalized for N(0,1) distribution)
    NF4_LEVELS = torch.tensor([
        -1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
        0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0
    ])
    
    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor = None,
        block_size: int = 64
    ):
        super().__init__()
        
        self.out_features, self.in_features = weight.shape
        self.block_size = block_size
        self.bias = nn.Parameter(bias) if bias is not None else None
        
        # Quantize weights
        self.register_buffer('weight_quantized', self._quantize(weight))
        self.register_buffer('scales', self._compute_scales(weight))
    
    def _compute_scales(self, weight: torch.Tensor) -> torch.Tensor:
        """Compute per-block scaling factors."""
        # Reshape into blocks
        weight_flat = weight.reshape(-1)
        num_blocks = (weight_flat.numel() + self.block_size - 1) // self.block_size
        
        # Pad if necessary
        padded_size = num_blocks * self.block_size
        if weight_flat.numel() < padded_size:
            weight_flat = F.pad(weight_flat, (0, padded_size - weight_flat.numel()))
        
        weight_blocks = weight_flat.reshape(num_blocks, self.block_size)
        
        # Scale = max absolute value per block
        scales = weight_blocks.abs().max(dim=1).values
        return scales
    
    def _quantize(self, weight: torch.Tensor) -> torch.Tensor:
        """Quantize weights to 4-bit NF4."""
        # Simplified: just store indices into NF4_LEVELS
        # Real implementation would pack 2 values per byte
        weight_flat = weight.reshape(-1)
        
        # Normalize by block scales
        scales = self._compute_scales(weight)
        num_blocks = scales.numel()
        
        # Reshape and normalize
        padded_size = num_blocks * self.block_size
        if weight_flat.numel() < padded_size:
            weight_flat = F.pad(weight_flat, (0, padded_size - weight_flat.numel()))
        
        weight_blocks = weight_flat.reshape(num_blocks, self.block_size)
        weight_normalized = weight_blocks / (scales.unsqueeze(1) + 1e-8)
        
        # Find nearest NF4 level
        distances = (weight_normalized.unsqueeze(-1) - self.NF4_LEVELS.to(weight.device)).abs()
        indices = distances.argmin(dim=-1)
        
        return indices.to(torch.uint8)
    
    def _dequantize(self) -> torch.Tensor:
        """Dequantize weights for forward pass."""
        # Get NF4 values from indices
        weight_nf4 = self.NF4_LEVELS.to(self.weight_quantized.device)[self.weight_quantized.long()]
        
        # Scale back
        weight_scaled = weight_nf4 * self.scales.unsqueeze(1)
        
        # Reshape to original
        weight = weight_scaled.reshape(-1)[:self.out_features * self.in_features]
        return weight.reshape(self.out_features, self.in_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize on-the-fly
        weight = self._dequantize()
        output = F.linear(x, weight, self.bias)
        return output


class QLoRALayer(nn.Module):
    """
    QLoRA: LoRA with quantized base weights.
    
    Base model in 4-bit, LoRA adapters in bfloat16.
    """
    
    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16,
        dropout: float = 0.0
    ):
        super().__init__()
        
        # Quantize original weights to NF4
        self.quantized = NF4Linear(
            original_layer.weight.data,
            original_layer.bias.data if original_layer.bias is not None else None
        )
        
        # LoRA in full precision (bfloat16)
        self.rank = rank
        self.scaling = alpha / rank
        
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        self.lora_A = nn.Parameter(
            torch.randn(in_features, rank, dtype=torch.bfloat16) * 0.01
        )
        self.lora_B = nn.Parameter(
            torch.zeros(rank, out_features, dtype=torch.bfloat16)
        )
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantized forward (dequantizes on the fly)
        base_out = self.quantized(x)
        
        # LoRA forward in bfloat16
        x_bf16 = x.to(torch.bfloat16)
        lora_out = (self.dropout(x_bf16) @ self.lora_A @ self.lora_B * self.scaling)
        
        return base_out + lora_out.to(base_out.dtype)
```

### QLoRA Memory Savings

| Model | Full FT | LoRA (16-bit) | QLoRA (4-bit) |
|-------|---------|---------------|---------------|
| 7B | 56 GB | ~16 GB | ~6 GB |
| 13B | 104 GB | ~28 GB | ~10 GB |
| 65B | 520 GB | ~130 GB | ~40 GB |

## Prefix Tuning

Prepend trainable continuous prompts to the input, modifying attention without changing model weights:

```python
class PrefixTuning(nn.Module):
    """
    Prefix Tuning: Learn continuous prompts prepended to input.
    
    Instead of discrete prompt tokens, learn continuous embeddings
    that are prepended to the key-value pairs in each attention layer.
    """
    
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        prefix_length: int = 20,
        hidden_dim: int = 512
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.prefix_length = prefix_length
        
        # Total size: layers * 2 (key + value) * heads * head_dim
        prefix_size = num_layers * 2 * num_heads * head_dim
        
        # Learnable prefix embeddings
        self.prefix_embedding = nn.Embedding(prefix_length, hidden_dim)
        
        # MLP to project to full prefix size (for stability)
        self.prefix_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, prefix_size)
        )
        
        self.num_heads = num_heads
        self.head_dim = head_dim
    
    def forward(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate prefix key-value pairs.
        
        Returns:
            Tuple of (prefix_keys, prefix_values), each of shape
            [batch, num_layers, num_heads, prefix_length, head_dim]
        """
        # Get prefix indices
        prefix_ids = torch.arange(self.prefix_length, device=self.prefix_embedding.weight.device)
        prefix_ids = prefix_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Embed and project
        prefix = self.prefix_embedding(prefix_ids)  # [batch, prefix_len, hidden]
        prefix = self.prefix_mlp(prefix)  # [batch, prefix_len, prefix_size]
        
        # Reshape to [batch, prefix_len, num_layers, 2, num_heads, head_dim]
        prefix = prefix.view(
            batch_size,
            self.prefix_length,
            self.num_layers,
            2,  # key and value
            self.num_heads,
            self.head_dim
        )
        
        # Rearrange to [batch, num_layers, num_heads, prefix_len, head_dim]
        prefix = prefix.permute(0, 2, 4, 1, 5, 3)  # Move layer and head dims
        
        prefix_keys = prefix[..., 0]    # [batch, layers, heads, prefix_len, head_dim]
        prefix_values = prefix[..., 1]  # [batch, layers, heads, prefix_len, head_dim]
        
        return prefix_keys, prefix_values


class PrefixAttention(nn.Module):
    """
    Attention layer modified to use prefix tuning.
    """
    
    def __init__(self, original_attention: nn.Module, prefix_length: int):
        super().__init__()
        self.attention = original_attention
        self.prefix_length = prefix_length
        
        # Freeze original attention
        for param in self.attention.parameters():
            param.requires_grad = False
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        prefix_key: torch.Tensor,
        prefix_value: torch.Tensor,
        attention_mask: torch.Tensor = None
    ):
        """
        Forward with prefix prepended to keys and values.
        """
        # Compute Q, K, V from hidden states
        # (Implementation depends on specific attention architecture)
        
        # Prepend prefix to K and V
        # K = [prefix_key; K]
        # V = [prefix_value; V]
        
        # Extend attention mask for prefix (always attend to prefix)
        if attention_mask is not None:
            prefix_mask = torch.ones(
                attention_mask.shape[0], self.prefix_length,
                device=attention_mask.device
            )
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        
        # Compute attention with extended K, V
        # ...
        
        pass  # Full implementation depends on model architecture
```

## Adapter Layers

Insert small trainable bottleneck modules between frozen transformer layers:

```python
class Adapter(nn.Module):
    """
    Adapter module: down-project → nonlinearity → up-project + residual.
    
    Inserted after attention and/or FFN sublayers.
    """
    
    def __init__(
        self,
        hidden_size: int,
        bottleneck_size: int = 64,
        activation: str = 'gelu'
    ):
        super().__init__()
        
        self.down_proj = nn.Linear(hidden_size, bottleneck_size)
        self.up_proj = nn.Linear(bottleneck_size, hidden_size)
        
        if activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'relu':
            self.act = nn.ReLU()
        else:
            self.act = nn.Tanh()
        
        # Initialize up_proj to near-zero for stable start
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Bottleneck with residual connection
        return x + self.up_proj(self.act(self.down_proj(x)))


class AdapterTransformerBlock(nn.Module):
    """
    Transformer block with adapters inserted.
    
    Architecture:
        x → Attention → Adapter → LayerNorm → FFN → Adapter → LayerNorm → output
    """
    
    def __init__(
        self,
        original_block: nn.Module,
        bottleneck_size: int = 64
    ):
        super().__init__()
        
        self.original = original_block
        
        # Freeze original block
        for param in original_block.parameters():
            param.requires_grad = False
        
        # Infer hidden size from the block
        hidden_size = self._get_hidden_size(original_block)
        
        # Add adapters after attention and FFN
        self.adapter_attn = Adapter(hidden_size, bottleneck_size)
        self.adapter_ffn = Adapter(hidden_size, bottleneck_size)
    
    def _get_hidden_size(self, block: nn.Module) -> int:
        """Try to infer hidden size from block architecture."""
        for name, module in block.named_modules():
            if isinstance(module, nn.Linear):
                return module.out_features
        raise ValueError("Could not infer hidden size")
    
    def forward(self, hidden_states: torch.Tensor, **kwargs):
        # This is a simplified version; actual implementation depends on
        # the specific transformer architecture
        
        # Attention
        attn_output = self.original.attention(hidden_states, **kwargs)
        if isinstance(attn_output, tuple):
            attn_output = attn_output[0]
        attn_output = self.adapter_attn(attn_output)  # Apply adapter
        
        hidden_states = hidden_states + attn_output
        hidden_states = self.original.ln_1(hidden_states)
        
        # FFN
        ffn_output = self.original.mlp(hidden_states)
        ffn_output = self.adapter_ffn(ffn_output)  # Apply adapter
        
        hidden_states = hidden_states + ffn_output
        hidden_states = self.original.ln_2(hidden_states)
        
        return hidden_states


def add_adapters(model: nn.Module, bottleneck_size: int = 64) -> nn.Module:
    """Add adapters to all transformer blocks."""
    # Implementation depends on model architecture
    # This is a template for the general approach
    pass
```

### Adapter Variants

| Variant | Placement | Notes |
|---------|-----------|-------|
| Serial Adapter | After sublayer | Original design (Houlsby et al.) |
| Parallel Adapter | Parallel to sublayer | Better for some tasks |
| AdapterFusion | Combine multiple adapters | Multi-task learning |

## Comprehensive Comparison

### Parameter Efficiency

| Method | Trainable Params | Memory Overhead | Inference Overhead |
|--------|-----------------|-----------------|-------------------|
| Full Fine-tuning | 100% | High | None |
| LoRA | 0.1-1% | Low | None (after merge) |
| QLoRA | 0.1-1% | Very Low | Slight (quantization) |
| Prefix Tuning | 0.1% | Low | Slight (longer sequence) |
| Adapters | 1-5% | Low | Slight (extra layers) |

### Quality vs Efficiency Trade-off

```
Quality
  ↑
  │     ┌─────────────────┐
  │     │ Full Fine-tuning │
  │     └────────┬────────┘
  │              │
  │     ┌────────┴────────┐
  │     │      LoRA       │
  │     └────────┬────────┘
  │              │
  │     ┌────────┴────────┐
  │     │     QLoRA       │
  │     └────────┬────────┘
  │              │
  │     ┌────────┴────────┐
  │     │    Adapters     │
  │     └────────┬────────┘
  │              │
  │     ┌────────┴────────┐
  │     │  Prefix Tuning  │
  │     └─────────────────┘
  └─────────────────────────────→ Efficiency
```

### When to Use What

| Scenario | Recommended Method |
|----------|-------------------|
| Consumer GPU (≤24GB), 7B model | QLoRA |
| Production, need zero overhead | LoRA (merged) |
| Multiple tasks, single model | Adapters |
| Very limited parameters | Prefix Tuning |
| Maximum quality, sufficient resources | Full Fine-tuning |
| 65B+ model, limited hardware | QLoRA |

## Combining Methods

Methods can be combined for specific needs:

```python
# QLoRA + Gradient Checkpointing for maximum memory efficiency
# LoRA on attention + Adapters on MLP
# Prefix Tuning + LoRA for extra capacity
```

## Summary

| Method | Key Idea | Best For |
|--------|----------|----------|
| **LoRA** | Low-rank weight updates | General use, production |
| **QLoRA** | 4-bit base + LoRA | Large models, limited memory |
| **Prefix Tuning** | Learnable soft prompts | Generation tasks |
| **Adapters** | Bottleneck modules | Multi-task, modular |

## References

1. Hu, E., et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." ICLR.
2. Dettmers, T., et al. (2023). "QLoRA: Efficient Finetuning of Quantized LLMs." NeurIPS.
3. Li, X., & Liang, P. (2021). "Prefix-Tuning: Optimizing Continuous Prompts for Generation." ACL.
4. Houlsby, N., et al. (2019). "Parameter-Efficient Transfer Learning for NLP." ICML.
5. Pfeiffer, J., et al. (2020). "AdapterHub: A Framework for Adapting Transformers." EMNLP.
