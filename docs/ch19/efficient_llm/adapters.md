# Adapter Layers

## Learning Objectives

- Understand the adapter architecture and design principles
- Implement adapters for transformer models
- Compare adapter variants (serial, parallel, AdapterFusion)
- Configure adapter bottleneck size and placement

## Introduction

Adapters (Houlsby et al., 2019) are small trainable modules inserted between the layers of a frozen pre-trained model. Each adapter consists of a down-projection, nonlinearity, and up-projection with a residual connection. This enables task-specific adaptation while keeping the base model frozen.

## Architecture

### Basic Adapter Module

The adapter follows a bottleneck architecture:

$$
\text{Adapter}(x) = x + f(xW_{down})W_{up}
$$

Where:
- $W_{down} \in \mathbb{R}^{d \times r}$ projects to bottleneck dimension $r$
- $f$ is a nonlinearity (GELU, ReLU)
- $W_{up} \in \mathbb{R}^{r \times d}$ projects back to original dimension
- Residual connection ensures initial behavior matches pre-trained model

### Parameter Count

For a model with dimension $d$ and bottleneck $r$:

$$
\text{Adapter params} = 2 \times d \times r + r + d
$$

With adapters after both attention and FFN in each layer:

$$
\text{Total params} = L \times 2 \times (2dr + r + d)
$$

**Example**: BERT-base ($L=12$, $d=768$) with $r=64$:
- Per adapter: $2 \times 768 \times 64 + 64 + 768 = 99,136$
- Total: $12 \times 2 \times 99,136 = 2,379,264$ (2.2% of 110M)

## Implementation

### Core Adapter Module

```python
import torch
import torch.nn as nn
from typing import Optional, List


class Adapter(nn.Module):
    """
    Adapter module: down-projection → activation → up-projection + residual.
    
    Args:
        input_dim: Input/output dimension (model hidden size)
        bottleneck_dim: Bottleneck dimension (typically 64-256)
        activation: Activation function
        init_scale: Scale for output projection initialization
    """
    
    def __init__(
        self,
        input_dim: int,
        bottleneck_dim: int = 64,
        activation: str = 'gelu',
        init_scale: float = 1e-3,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        
        # Down-projection
        self.down_proj = nn.Linear(input_dim, bottleneck_dim)
        
        # Activation
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.SiLU()
        
        # Up-projection
        self.up_proj = nn.Linear(bottleneck_dim, input_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize for stable residual
        self._init_weights(init_scale)
    
    def _init_weights(self, scale: float):
        """Initialize weights for stable residual connection."""
        nn.init.normal_(self.down_proj.weight, std=0.02)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.normal_(self.up_proj.weight, std=scale)
        nn.init.zeros_(self.up_proj.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        down = self.down_proj(x)
        activated = self.activation(down)
        activated = self.dropout(activated)
        up = self.up_proj(activated)
        return x + up
    
    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class ScaledAdapter(Adapter):
    """Adapter with learnable scaling factor."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale = nn.Parameter(torch.ones(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        down = self.down_proj(x)
        activated = self.activation(down)
        activated = self.dropout(activated)
        up = self.up_proj(activated)
        return x + self.scale * up
```

### Serial Adapters (Original Design)

```python
class SerialAdapterTransformerLayer(nn.Module):
    """
    Transformer layer with serial adapters after each sublayer.
    
    Architecture:
        x → Attention → Adapter → Add & Norm → FFN → Adapter → Add & Norm
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        bottleneck_dim: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Adapters
        self.adapter_attn = Adapter(d_model, bottleneck_dim)
        self.adapter_ffn = Adapter(d_model, bottleneck_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Attention + adapter
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=attention_mask)
        attn_out = self.adapter_attn(attn_out)
        x = self.norm1(x + self.dropout(attn_out))
        
        # FFN + adapter
        ffn_out = self.ffn(x)
        ffn_out = self.adapter_ffn(ffn_out)
        x = self.norm2(x + ffn_out)
        
        return x
```

### Parallel Adapters

```python
class ParallelAdapterTransformerLayer(nn.Module):
    """
    Transformer layer with parallel adapters.
    
    Adapters run in parallel with sublayers, not after them.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        bottleneck_dim: int = 64,
        dropout: float = 0.1,
        adapter_scale: float = 1.0
    ):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.adapter_attn = Adapter(d_model, bottleneck_dim)
        self.adapter_ffn = Adapter(d_model, bottleneck_dim)
        self.adapter_scale = adapter_scale
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Parallel: adapter and attention both take x
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=attention_mask)
        adapter_out = self.adapter_attn(x) - x  # Remove residual
        x = self.norm1(x + self.dropout(attn_out) + self.adapter_scale * adapter_out)
        
        # Parallel: adapter and FFN both take x
        ffn_out = self.ffn(x)
        adapter_out = self.adapter_ffn(x) - x
        x = self.norm2(x + ffn_out + self.adapter_scale * adapter_out)
        
        return x
```

## AdapterFusion: Multi-Task Learning

```python
class AdapterFusion(nn.Module):
    """
    Learn to combine multiple task-specific adapters.
    
    Each adapter is trained separately, then fusion learns
    optimal weighting for a target task.
    """
    
    def __init__(
        self,
        adapters: nn.ModuleList,
        input_dim: int
    ):
        super().__init__()
        
        self.adapters = adapters
        self.num_adapters = len(adapters)
        
        # Freeze individual adapters
        for adapter in adapters:
            for param in adapter.parameters():
                param.requires_grad = False
        
        # Fusion attention
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = x.shape
        
        # Get adapter outputs
        adapter_outputs = []
        for adapter in self.adapters:
            adapter_out = adapter(x) - x  # Contribution without residual
            adapter_outputs.append(adapter_out)
        
        # Stack: [batch, seq, num_adapters, hidden]
        adapter_stack = torch.stack(adapter_outputs, dim=2)
        
        # Attention over adapters
        query = self.query(x).unsqueeze(2)
        keys = self.key(adapter_stack)
        
        scores = (query * keys).sum(dim=-1) / (hidden_dim ** 0.5)
        scores = scores / self.temperature
        weights = torch.softmax(scores, dim=-1)
        
        # Weighted combination
        values = self.value(adapter_stack)
        fused = (weights.unsqueeze(-1) * values).sum(dim=2)
        
        return x + fused
```

## Multi-Task Adapter Model

```python
class MultiTaskAdapterModel(nn.Module):
    """Model with switchable task-specific adapters."""
    
    def __init__(
        self,
        base_model: nn.Module,
        task_names: List[str],
        hidden_dim: int,
        bottleneck_dim: int = 64
    ):
        super().__init__()
        
        self.base_model = base_model
        self.task_names = task_names
        
        # Freeze base
        for param in base_model.parameters():
            param.requires_grad = False
        
        # One adapter per task
        self.adapters = nn.ModuleDict({
            name: Adapter(hidden_dim, bottleneck_dim)
            for name in task_names
        })
        
        self.current_task: Optional[str] = None
    
    def set_task(self, task_name: str):
        """Activate a specific task adapter."""
        if task_name not in self.task_names:
            raise ValueError(f"Unknown task: {task_name}")
        
        self.current_task = task_name
        
        # Only current task adapter is trainable
        for name, adapter in self.adapters.items():
            for param in adapter.parameters():
                param.requires_grad = (name == task_name)
    
    def forward(self, x: torch.Tensor, task: Optional[str] = None):
        task = task or self.current_task
        if task is None:
            raise ValueError("No task specified")
        
        hidden = self.base_model(x)
        return self.adapters[task](hidden)
```

## Hyperparameter Guide

### Bottleneck Dimension

| Bottleneck | % of BERT-base | Use Case |
|------------|----------------|----------|
| 8 | 0.2% | Extreme efficiency |
| 32 | 0.9% | Low resource |
| **64** | 1.8% | **Default** |
| 128 | 3.5% | Complex tasks |
| 256 | 7.0% | High capacity |

### Placement Strategy

| Strategy | Adapters/Layer | Recommendation |
|----------|----------------|----------------|
| Attention only | 1 | Minimal |
| FFN only | 1 | Often sufficient |
| **Both** | 2 | **Recommended** |

### Learning Rate

- Adapters: 1e-4 to 1e-3
- Full fine-tuning: 1e-5 to 5e-5

## Comparison

| Method | Params | Inference Overhead | Multi-task | Modular |
|--------|--------|-------------------|------------|---------|
| Adapters | 1-5% | Yes | Easy | Yes |
| LoRA | 0.1-1% | None (merge) | Harder | Yes |
| Prefix Tuning | 0.1-1% | Yes | Easy | Yes |

**Use Adapters when:**
- Multi-task learning with task switching
- Modular deployment
- Inference overhead is acceptable

## Training Example

```python
def train_with_adapters(
    base_model: nn.Module,
    train_loader,
    hidden_dim: int,
    bottleneck_dim: int = 64,
    num_epochs: int = 5,
    lr: float = 1e-4,
    device: str = 'cuda'
):
    # Freeze base model
    for param in base_model.parameters():
        param.requires_grad = False
    
    # Add adapter (simplified - actual integration varies by model)
    adapter = Adapter(hidden_dim, bottleneck_dim).to(device)
    
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward through base + adapter
            with torch.no_grad():
                hidden = base_model(**batch).last_hidden_state
            output = adapter(hidden)
            
            loss = compute_loss(output, batch['labels'])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}")
    
    return adapter


def save_adapter(adapter: nn.Module, path: str):
    """Save adapter weights."""
    torch.save(adapter.state_dict(), path)


def load_adapter(adapter: nn.Module, path: str):
    """Load adapter weights."""
    adapter.load_state_dict(torch.load(path))
```

## Summary

| Aspect | Details |
|--------|---------|
| **Architecture** | Down-proj → Activation → Up-proj + Residual |
| **Parameters** | 1-5% of model |
| **Placement** | After attention and/or FFN |
| **Key advantage** | Modular, easy multi-task |
| **Trade-off** | Inference overhead |

## References

1. Houlsby, N., et al. (2019). "Parameter-Efficient Transfer Learning for NLP." ICML.
2. Pfeiffer, J., et al. (2020). "AdapterHub: A Framework for Adapting Transformers." EMNLP.
3. Pfeiffer, J., et al. (2021). "AdapterFusion: Non-Destructive Task Composition." EACL.
4. He, J., et al. (2022). "Towards a Unified View of Parameter-Efficient Transfer Learning." ICLR.
