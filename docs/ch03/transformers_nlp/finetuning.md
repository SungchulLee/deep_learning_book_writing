# Fine-tuning Strategies

## Overview

Fine-tuning adapts pre-trained models to specific tasks. This document covers strategies from full fine-tuning to parameter-efficient methods.

## Full Fine-tuning

Update all model parameters on task-specific data:

```python
import torch
import torch.nn as nn
from transformers import AutoModel

class FullFineTuning(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids, attention_mask)
        pooled = outputs.last_hidden_state[:, 0]  # [CLS] token
        return self.classifier(pooled)

# All parameters are trainable
model = FullFineTuning("bert-base-uncased", num_labels=2)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
```

**Pros:** Maximum performance potential
**Cons:** Expensive, risk of catastrophic forgetting

## Feature Extraction (Frozen Encoder)

Freeze pre-trained weights, only train task head:

```python
class FeatureExtraction(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.encoder(input_ids, attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        return self.classifier(pooled)
```

**Pros:** Fast, no forgetting
**Cons:** Limited adaptation

## Gradual Unfreezing

Progressively unfreeze layers during training:

```python
class GradualUnfreeze:
    def __init__(self, model, num_layers):
        self.model = model
        self.num_layers = num_layers
        self.unfrozen_layers = 0
        
        # Initially freeze all encoder layers
        for param in model.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_next_layer(self):
        """Unfreeze from top to bottom."""
        if self.unfrozen_layers >= self.num_layers:
            return
        
        layer_idx = self.num_layers - 1 - self.unfrozen_layers
        for param in self.model.encoder.layer[layer_idx].parameters():
            param.requires_grad = True
        
        self.unfrozen_layers += 1
        print(f"Unfroze layer {layer_idx}")

# Usage
scheduler = GradualUnfreeze(model, num_layers=12)

for epoch in range(epochs):
    if epoch > 0 and epoch % 2 == 0:
        scheduler.unfreeze_next_layer()
    train_epoch(model, train_loader)
```

## Discriminative Learning Rates

Apply different learning rates to different layers:

```python
def get_layer_lrs(model, base_lr=2e-5, decay=0.95):
    """Higher LR for top layers, lower for bottom."""
    params = []
    num_layers = len(model.encoder.layer)
    
    # Embeddings - lowest LR
    params.append({
        'params': model.encoder.embeddings.parameters(),
        'lr': base_lr * (decay ** num_layers)
    })
    
    # Encoder layers
    for i, layer in enumerate(model.encoder.layer):
        lr = base_lr * (decay ** (num_layers - i - 1))
        params.append({'params': layer.parameters(), 'lr': lr})
    
    # Classifier - highest LR
    params.append({'params': model.classifier.parameters(), 'lr': base_lr})
    
    return params

optimizer = torch.optim.AdamW(get_layer_lrs(model))
```

## LoRA (Low-Rank Adaptation)

Add trainable low-rank matrices to attention layers:

$$
W' = W + BA, \quad B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times d}
$$

```python
class LoRALayer(nn.Module):
    def __init__(self, original_layer, rank=8, alpha=16):
        super().__init__()
        self.original = original_layer
        self.rank = rank
        self.alpha = alpha
        
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        # Freeze original
        for param in self.original.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        original_out = self.original(x)
        lora_out = (x @ self.lora_A @ self.lora_B) * (self.alpha / self.rank)
        return original_out + lora_out


def apply_lora(model, rank=8):
    """Apply LoRA to all attention layers."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and 'attention' in name:
            parent_name = '.'.join(name.split('.')[:-1])
            layer_name = name.split('.')[-1]
            parent = model.get_submodule(parent_name)
            setattr(parent, layer_name, LoRALayer(module, rank))
    return model
```

**Pros:** ~0.1% trainable parameters, no inference overhead (merge weights)
**Cons:** Slightly lower than full fine-tuning

## Prefix Tuning

Prepend trainable continuous prompts:

```python
class PrefixTuning(nn.Module):
    def __init__(self, model, prefix_length=20, hidden_dim=512):
        super().__init__()
        self.model = model
        self.prefix_length = prefix_length
        
        # Freeze base model
        for param in model.parameters():
            param.requires_grad = False
        
        # Learnable prefix
        self.prefix_embedding = nn.Embedding(prefix_length, hidden_dim)
        self.prefix_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, model.config.num_layers * 2 * model.config.hidden_size)
        )
    
    def get_prefix(self, batch_size):
        prefix_ids = torch.arange(self.prefix_length).unsqueeze(0).expand(batch_size, -1)
        prefix = self.prefix_embedding(prefix_ids)
        prefix = self.prefix_mlp(prefix)
        # Reshape for key-value injection
        return prefix.view(batch_size, self.prefix_length, -1, 2, self.model.config.hidden_size)
    
    def forward(self, input_ids, attention_mask):
        batch_size = input_ids.size(0)
        prefix = self.get_prefix(batch_size)
        # Inject prefix into attention layers...
        return self.model(input_ids, attention_mask, past_key_values=prefix)
```

## Adapter Layers

Insert small trainable modules between frozen layers:

```python
class Adapter(nn.Module):
    def __init__(self, hidden_size, bottleneck_size=64):
        super().__init__()
        self.down_proj = nn.Linear(hidden_size, bottleneck_size)
        self.up_proj = nn.Linear(bottleneck_size, hidden_size)
        self.act = nn.GELU()
    
    def forward(self, x):
        return x + self.up_proj(self.act(self.down_proj(x)))


class AdapterTransformerLayer(nn.Module):
    def __init__(self, original_layer, bottleneck_size=64):
        super().__init__()
        self.original = original_layer
        
        # Freeze original
        for param in original_layer.parameters():
            param.requires_grad = False
        
        hidden_size = original_layer.attention.self.query.out_features
        self.adapter_attn = Adapter(hidden_size, bottleneck_size)
        self.adapter_ffn = Adapter(hidden_size, bottleneck_size)
    
    def forward(self, hidden_states, attention_mask=None):
        # Original attention
        attn_output = self.original.attention(hidden_states, attention_mask)[0]
        attn_output = self.adapter_attn(attn_output)  # Add adapter
        
        hidden_states = hidden_states + attn_output
        hidden_states = self.original.layernorm1(hidden_states)
        
        # Original FFN
        ffn_output = self.original.ffn(hidden_states)
        ffn_output = self.adapter_ffn(ffn_output)  # Add adapter
        
        return hidden_states + ffn_output
```

## Comparison

| Method | Trainable Params | Memory | Performance |
|--------|-----------------|--------|-------------|
| Full Fine-tuning | 100% | High | Best |
| Feature Extraction | <1% | Low | Moderate |
| LoRA | ~0.1% | Low | Near-best |
| Prefix Tuning | ~0.1% | Low | Good |
| Adapters | ~1-5% | Moderate | Good |

## Best Practices

1. **Start with feature extraction** to establish baseline
2. **Use LoRA** for production with limited resources
3. **Full fine-tuning** when performance is critical
4. **Gradual unfreezing** to prevent catastrophic forgetting
5. **Early stopping** and validation monitoring

## Summary

Choose fine-tuning strategy based on:
- Available compute and memory
- Dataset size
- Performance requirements
- Need to preserve pre-trained knowledge

## References

1. Howard, J., & Ruder, S. (2018). "Universal Language Model Fine-tuning for Text Classification."
2. Hu, E., et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models."
3. Li, X., & Liang, P. (2021). "Prefix-Tuning: Optimizing Continuous Prompts for Generation."
4. Houlsby, N., et al. (2019). "Parameter-Efficient Transfer Learning for NLP."
