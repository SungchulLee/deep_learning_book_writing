# LoRA: Low-Rank Adaptation

## Learning Objectives

- Understand the mathematical foundation of low-rank adaptation
- Implement LoRA from scratch and apply it to transformer models
- Configure LoRA hyperparameters (rank, alpha, target modules)
- Merge LoRA weights for zero-overhead inference

## Introduction

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning method that freezes pre-trained model weights and injects trainable low-rank decomposition matrices into each layer. This reduces trainable parameters by 10,000x while achieving comparable performance to full fine-tuning.

## Mathematical Foundation

### The Core Idea

Instead of updating a weight matrix $W_0 \in \mathbb{R}^{d_{out} \times d_{in}}$ directly, LoRA constrains the update to a low-rank decomposition:

$$
W = W_0 + \Delta W = W_0 + BA
$$

Where:
- $B \in \mathbb{R}^{d_{out} \times r}$ (down-projection)
- $A \in \mathbb{R}^{r \times d_{in}}$ (up-projection)  
- $r \ll \min(d_{in}, d_{out})$ is the rank

### Why Low-Rank Works

Research shows that the weight updates during fine-tuning have low "intrinsic rank"—the effective dimensionality of the update is much smaller than the full parameter space. LoRA exploits this by explicitly parameterizing updates as low-rank matrices.

### Forward Pass

For input $x$:

$$
h = W_0 x + \Delta W x = W_0 x + BAx
$$

The original weights $W_0$ are frozen; only $A$ and $B$ are trained.

### Scaling Factor

LoRA uses a scaling factor to control update magnitude:

$$
h = W_0 x + \frac{\alpha}{r} BAx
$$

Where $\alpha$ is a constant (typically $\alpha = 2r$ or $\alpha = r$). This scaling ensures:
- The magnitude of $\Delta W$ is independent of rank choice
- Hyperparameter transfer: same $\alpha$ works across different ranks
- Stable training dynamics

### Parameter Efficiency

For a linear layer with dimensions $d_{in} \times d_{out}$:

| Method | Parameters |
|--------|------------|
| Full fine-tuning | $d_{in} \times d_{out}$ |
| LoRA (rank $r$) | $r \times (d_{in} + d_{out})$ |

**Example**: For $d_{in} = d_{out} = 4096$, $r = 8$:
- Full: 16,777,216 parameters
- LoRA: 65,536 parameters (0.39%)

## Implementation

### Core LoRA Layer

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Dict, Any


class LoRALayer(nn.Module):
    """
    LoRA layer that wraps an existing linear layer.
    
    Implements: h = W₀x + (α/r)BAx
    
    Args:
        original_layer: The linear layer to adapt
        rank: Rank of the low-rank decomposition
        alpha: Scaling factor (typically alpha = 2*rank)
        dropout: Dropout probability on LoRA path
    """
    
    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
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
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False
        
        # LoRA matrices
        # A: in_features -> rank (down projection)
        self.lora_A = nn.Parameter(torch.empty(in_features, rank))
        # B: rank -> out_features (up projection)
        self.lora_B = nn.Parameter(torch.empty(rank, out_features))
        
        # Initialize
        self._init_weights()
        
        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # For tracking
        self.merged = False
    
    def _init_weights(self):
        """
        Initialize LoRA weights.
        
        A: Kaiming uniform (same as nn.Linear default)
        B: Zero (so initial ΔW = BA = 0)
        """
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: W₀x + (α/r)BAx
        """
        if self.merged:
            # If weights are merged, just use original layer
            return self.original(x)
        
        # Original path (frozen)
        original_output = self.original(x)
        
        # LoRA path: x @ A @ B * scaling
        lora_output = self.dropout(x) @ self.lora_A @ self.lora_B * self.scaling
        
        return original_output + lora_output
    
    def merge_weights(self):
        """
        Merge LoRA weights into original layer for inference.
        
        W' = W₀ + (α/r)BA^T
        
        After merging, forward pass has zero overhead.
        """
        if self.merged:
            return
        
        # Compute ΔW = (α/r) * A @ B, then transpose for weight format
        delta_w = (self.lora_A @ self.lora_B * self.scaling).T
        self.original.weight.data += delta_w
        self.merged = True
    
    def unmerge_weights(self):
        """
        Unmerge LoRA weights (reverse of merge).
        
        Useful for continued training or switching adapters.
        """
        if not self.merged:
            return
        
        delta_w = (self.lora_A @ self.lora_B * self.scaling).T
        self.original.weight.data -= delta_w
        self.merged = False
    
    def get_delta_weight(self) -> torch.Tensor:
        """Return the LoRA weight update ΔW."""
        return (self.lora_A @ self.lora_B * self.scaling).T
    
    @property
    def num_parameters(self) -> int:
        """Number of trainable LoRA parameters."""
        return self.lora_A.numel() + self.lora_B.numel()


class LoRALinear(nn.Module):
    """
    Standalone LoRA linear layer (not wrapping existing layer).
    
    Useful for creating new models with LoRA built-in.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        bias: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank
        
        # Frozen base weight
        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False) if bias else None
        
        # Trainable LoRA
        self.lora_A = nn.Parameter(torch.empty(in_features, rank))
        self.lora_B = nn.Parameter(torch.empty(rank, out_features))
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        lora = self.dropout(x) @ self.lora_A @ self.lora_B * self.scaling
        return base + lora
```

### Applying LoRA to a Model

```python
from dataclasses import dataclass, field
from typing import Set


@dataclass
class LoRAConfig:
    """Configuration for LoRA adaptation."""
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    # Which module names to apply LoRA to
    target_modules: Set[str] = field(default_factory=lambda: {'q_proj', 'v_proj'})
    # Modules to exclude even if they match target_modules
    exclude_modules: Set[str] = field(default_factory=set)


def apply_lora_to_model(
    model: nn.Module,
    config: LoRAConfig
) -> nn.Module:
    """
    Apply LoRA to all matching modules in a model.
    
    Args:
        model: The model to adapt
        config: LoRA configuration
        
    Returns:
        Model with LoRA layers (original weights frozen)
    """
    # Collect modules to replace (can't modify during iteration)
    replacements = []
    
    for name, module in model.named_modules():
        # Check if this module should get LoRA
        should_apply = (
            isinstance(module, nn.Linear) and
            any(target in name for target in config.target_modules) and
            not any(exclude in name for exclude in config.exclude_modules)
        )
        
        if should_apply:
            replacements.append((name, module))
    
    # Apply replacements
    for name, module in replacements:
        # Navigate to parent module
        parts = name.rsplit('.', 1)
        if len(parts) == 2:
            parent_name, child_name = parts
            parent = model.get_submodule(parent_name)
        else:
            parent = model
            child_name = name
        
        # Create and set LoRA layer
        lora_layer = LoRALayer(
            module,
            rank=config.rank,
            alpha=config.alpha,
            dropout=config.dropout
        )
        setattr(parent, child_name, lora_layer)
        
        print(f"Applied LoRA to {name}: {module.in_features} -> {module.out_features}, rank={config.rank}")
    
    print(f"\nTotal LoRA layers: {len(replacements)}")
    return model


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Extract only LoRA parameters from model state dict.
    
    Useful for saving/loading just the adapter weights.
    """
    return {
        name: param for name, param in model.state_dict().items()
        if 'lora_' in name
    }


def load_lora_state_dict(model: nn.Module, state_dict: Dict[str, torch.Tensor]):
    """Load LoRA parameters into model."""
    model_state = model.state_dict()
    
    for name, param in state_dict.items():
        if name in model_state:
            model_state[name].copy_(param)
        else:
            print(f"Warning: {name} not found in model")
    
    model.load_state_dict(model_state, strict=False)
```

### Training Utilities

```python
def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """Get only LoRA parameters for optimizer."""
    params = []
    for name, param in model.named_parameters():
        if 'lora_' in name:
            params.append(param)
    return params


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count trainable vs total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    lora = sum(p.numel() for n, p in model.named_parameters() if 'lora_' in n)
    
    return {
        'trainable': trainable,
        'total': total,
        'lora': lora,
        'trainable_percent': 100.0 * trainable / total,
        'lora_percent': 100.0 * lora / total
    }


def freeze_non_lora(model: nn.Module):
    """Freeze all parameters except LoRA."""
    for name, param in model.named_parameters():
        if 'lora_' not in name:
            param.requires_grad = False


def merge_lora_weights(model: nn.Module):
    """Merge all LoRA weights in model for inference."""
    for module in model.modules():
        if isinstance(module, LoRALayer):
            module.merge_weights()


def unmerge_lora_weights(model: nn.Module):
    """Unmerge all LoRA weights in model."""
    for module in model.modules():
        if isinstance(module, LoRALayer):
            module.unmerge_weights()
```

## Hyperparameter Guide

### Rank Selection

| Rank | Parameters | Quality | Use Case |
|------|------------|---------|----------|
| 1-4 | Minimal | Lower | Very simple tasks, extreme compression |
| 8 | Low | Good | **Default**, most tasks |
| 16 | Medium | Better | Complex tasks, larger datasets |
| 32-64 | Higher | Near full FT | Tasks requiring high capacity |
| 128+ | High | ~Full FT | When approaching full fine-tuning |

**Rule of thumb**: Start with rank 8, increase if underfitting, decrease if overfitting or memory-constrained.

### Alpha Selection

Common strategies:
- $\alpha = r$: Conservative scaling
- $\alpha = 2r$: **Default**, balanced
- $\alpha = 4r$: Aggressive updates

The ratio $\alpha/r$ determines the effective learning rate for LoRA parameters. Higher ratio = larger updates.

### Target Module Selection

For transformer models:

| Target | Modules | Quality | Efficiency |
|--------|---------|---------|------------|
| Minimal | q_proj | Baseline | Highest |
| **Standard** | q_proj, v_proj | Good | High |
| Extended | q_proj, k_proj, v_proj, o_proj | Better | Medium |
| Full attention | All attention + output | Best | Lower |
| Everything | Attention + MLP | Marginal gain | Lowest |

```python
# Common configurations
LORA_CONFIGS = {
    'minimal': LoRAConfig(rank=8, target_modules={'q_proj'}),
    'standard': LoRAConfig(rank=8, target_modules={'q_proj', 'v_proj'}),
    'extended': LoRAConfig(rank=8, target_modules={'q_proj', 'k_proj', 'v_proj', 'o_proj'}),
    'full': LoRAConfig(rank=16, target_modules={'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'}),
}
```

### Learning Rate

LoRA typically uses higher learning rates than full fine-tuning:

| Method | Typical LR |
|--------|------------|
| Full fine-tuning | 1e-5 to 5e-5 |
| LoRA | 1e-4 to 3e-4 |

## Advanced Topics

### LoRA for Multiple Tasks

Save and load different adapters:

```python
class LoRAManager:
    """Manage multiple LoRA adapters for a single base model."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.adapters: Dict[str, Dict[str, torch.Tensor]] = {}
        self.current_adapter: Optional[str] = None
    
    def save_adapter(self, name: str):
        """Save current LoRA weights as named adapter."""
        self.adapters[name] = get_lora_state_dict(self.model)
    
    def load_adapter(self, name: str):
        """Load a saved adapter."""
        if name not in self.adapters:
            raise ValueError(f"Adapter '{name}' not found")
        
        # Unmerge current if merged
        unmerge_lora_weights(self.model)
        
        # Load new adapter
        load_lora_state_dict(self.model, self.adapters[name])
        self.current_adapter = name
    
    def delete_adapter(self, name: str):
        """Delete a saved adapter."""
        if name in self.adapters:
            del self.adapters[name]
```

### LoRA+ (Improved Learning Rates)

LoRA+ uses different learning rates for A and B matrices:

```python
def get_lora_plus_params(model: nn.Module, lr: float, lr_ratio: float = 16.0):
    """
    LoRA+ parameter groups with different LRs for A and B.
    
    B gets higher LR since it's initialized to zero.
    """
    params_A = []
    params_B = []
    
    for name, param in model.named_parameters():
        if 'lora_A' in name:
            params_A.append(param)
        elif 'lora_B' in name:
            params_B.append(param)
    
    return [
        {'params': params_A, 'lr': lr},
        {'params': params_B, 'lr': lr * lr_ratio}
    ]
```

### Rank-Stabilized LoRA (rsLoRA)

Adjusts scaling for better stability at high ranks:

$$
h = W_0 x + \frac{\alpha}{\sqrt{r}} BAx
$$

```python
class rsLoRALayer(LoRALayer):
    """Rank-stabilized LoRA with sqrt scaling."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Use sqrt(r) instead of r for scaling
        self.scaling = self.alpha / math.sqrt(self.rank)
```

## Complete Training Example

```python
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


def train_lora(
    model_name: str,
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    config: LoRAConfig,
    num_epochs: int = 3,
    learning_rate: float = 2e-4,
    device: str = 'cuda'
):
    """Complete LoRA training pipeline."""
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = apply_lora_to_model(model, config)
    model = model.to(device)
    
    # Setup optimizer (only LoRA params)
    lora_params = get_lora_parameters(model)
    optimizer = torch.optim.AdamW(lora_params, lr=learning_rate)
    
    # Count parameters
    param_counts = count_parameters(model)
    print(f"Trainable: {param_counts['trainable']:,} ({param_counts['trainable_percent']:.2f}%)")
    print(f"LoRA: {param_counts['lora']:,} ({param_counts['lora_percent']:.2f}%)")
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        # Evaluation
        model.eval()
        eval_loss = 0
        with torch.no_grad():
            for batch in eval_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                eval_loss += outputs.loss.item()
        
        print(f"  Eval Loss: {eval_loss / len(eval_dataloader):.4f}")
        model.train()
    
    # Save LoRA weights
    lora_state = get_lora_state_dict(model)
    torch.save(lora_state, 'lora_weights.pt')
    
    # Merge for inference
    merge_lora_weights(model)
    
    return model


# Example usage
if __name__ == "__main__":
    config = LoRAConfig(
        rank=8,
        alpha=16,
        dropout=0.05,
        target_modules={'q_proj', 'v_proj'}
    )
    
    # model = train_lora("meta-llama/Llama-2-7b-hf", train_dl, eval_dl, config)
```

## Summary

| Aspect | Recommendation |
|--------|----------------|
| **Rank** | Start with 8, adjust based on task complexity |
| **Alpha** | Use 2× rank as default |
| **Targets** | q_proj + v_proj for most tasks |
| **Learning rate** | 1e-4 to 3e-4 (higher than full FT) |
| **Dropout** | 0.05-0.1 for regularization |

## References

1. Hu, E., et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022.
2. Dettmers, T., et al. (2023). "QLoRA: Efficient Finetuning of Quantized LLMs."
3. Hayou, S., et al. (2024). "LoRA+: Efficient Low Rank Adaptation of Large Models."
4. Kalajdzievski, D. (2023). "Rank-Stabilized LoRA: Unlocking the Potential of LoRA Fine-Tuning."
