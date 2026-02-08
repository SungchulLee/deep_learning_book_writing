# Fine-tuning Strategies for Transfer Learning

## Learning Objectives

- Understand the spectrum from feature extraction to full fine-tuning
- Implement gradual unfreezing and discriminative learning rates
- Recognize and mitigate catastrophic forgetting
- Choose appropriate fine-tuning strategies based on task requirements

## Introduction

Fine-tuning adapts pre-trained models to specific downstream tasks. The key challenge is balancing adaptation to new data while preserving valuable pre-trained knowledge. This document covers general fine-tuning strategies; for parameter-efficient methods like LoRA and adapters, see [Ch 15.6 Efficient LLM Fine-tuning](../../ch15/efficient_llm/efficiency_overview.md).

## The Fine-tuning Spectrum

```
Feature Extraction ←――――――――――――――――――→ Full Fine-tuning
(Frozen encoder)                        (All params trainable)

Less adaptation                         More adaptation
Lower risk of forgetting               Higher risk of forgetting
Faster training                        Slower training
Lower performance ceiling              Higher performance ceiling
```

## Full Fine-tuning

Update all model parameters on task-specific data:

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class FullFineTuning(nn.Module):
    """
    Full fine-tuning: all parameters are trainable.
    
    Best when:
    - Large task-specific dataset available
    - Task differs significantly from pretraining
    - Maximum performance is required
    """
    
    def __init__(self, model_name: str, num_labels: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]  # [CLS] token
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


# All parameters are trainable
model = FullFineTuning("bert-base-uncased", num_labels=2)
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Typical hyperparameters for full fine-tuning
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
```

### Full Fine-tuning Best Practices

| Hyperparameter | Typical Range | Notes |
|----------------|---------------|-------|
| Learning rate | 1e-5 to 5e-5 | Lower than pretraining |
| Batch size | 16-32 | Larger if memory allows |
| Epochs | 2-4 | Early stopping recommended |
| Warmup | 6-10% of steps | Prevents early divergence |
| Weight decay | 0.01 | Regularization |

## Feature Extraction (Frozen Encoder)

Freeze pre-trained weights, only train the task-specific head:

```python
class FeatureExtraction(nn.Module):
    """
    Feature extraction: encoder frozen, only classifier trained.
    
    Best when:
    - Very small dataset
    - Task is similar to pretraining
    - Fast iteration needed
    - Preserving pretrained knowledge is critical
    """
    
    def __init__(self, model_name: str, num_labels: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # Freeze all encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Only classifier is trainable
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_labels)
        )
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        # No gradient computation for encoder
        with torch.no_grad():
            outputs = self.encoder(input_ids, attention_mask=attention_mask)
        
        pooled = outputs.last_hidden_state[:, 0]
        return self.classifier(pooled)
    
    @property
    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


model = FeatureExtraction("bert-base-uncased", num_labels=2)
print(f"Trainable parameters: {model.num_trainable_params:,}")  # ~590K vs 109M total

# Can use higher learning rate since only training small head
optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3)
```

## Gradual Unfreezing

Progressively unfreeze layers during training, starting from the top (closest to output):

```python
from typing import List


class GradualUnfreezeScheduler:
    """
    Gradually unfreeze layers from top to bottom.
    
    Rationale: Top layers are more task-specific, bottom layers
    contain more general features. Unfreezing top-first allows
    the model to adapt gradually without disrupting lower-level
    representations.
    """
    
    def __init__(self, model: nn.Module, encoder_attr: str = 'encoder'):
        self.model = model
        self.encoder = getattr(model, encoder_attr)
        
        # Get encoder layers (works for BERT-like models)
        if hasattr(self.encoder, 'layer'):
            self.layers = list(self.encoder.layer)
        elif hasattr(self.encoder, 'layers'):
            self.layers = list(self.encoder.layers)
        else:
            raise ValueError("Could not find encoder layers")
        
        self.num_layers = len(self.layers)
        self.unfrozen_count = 0
        
        # Initially freeze all encoder layers
        self._freeze_all()
    
    def _freeze_all(self):
        """Freeze all encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_layer(self, layer_idx: int):
        """Unfreeze a specific layer."""
        for param in self.layers[layer_idx].parameters():
            param.requires_grad = True
    
    def unfreeze_next(self) -> bool:
        """
        Unfreeze the next layer (top to bottom).
        
        Returns:
            True if a layer was unfrozen, False if all layers already unfrozen
        """
        if self.unfrozen_count >= self.num_layers:
            return False
        
        # Unfreeze from top (highest index) to bottom
        layer_idx = self.num_layers - 1 - self.unfrozen_count
        self.unfreeze_layer(layer_idx)
        self.unfrozen_count += 1
        
        print(f"Unfroze layer {layer_idx} ({self.unfrozen_count}/{self.num_layers})")
        return True
    
    def unfreeze_n_layers(self, n: int):
        """Unfreeze n layers at once."""
        for _ in range(n):
            if not self.unfreeze_next():
                break
    
    def unfreeze_embeddings(self):
        """Unfreeze embedding layer (usually done last)."""
        if hasattr(self.encoder, 'embeddings'):
            for param in self.encoder.embeddings.parameters():
                param.requires_grad = True
            print("Unfroze embeddings")


# Usage example
model = FullFineTuning("bert-base-uncased", num_labels=2)
scheduler = GradualUnfreezeScheduler(model)

num_epochs = 12
unfreeze_every = 2  # Unfreeze a new layer every 2 epochs

for epoch in range(num_epochs):
    # Unfreeze layers progressively
    if epoch > 0 and epoch % unfreeze_every == 0:
        scheduler.unfreeze_next()
    
    # Optionally unfreeze embeddings at the end
    if epoch == num_epochs - 2:
        scheduler.unfreeze_embeddings()
    
    # train_epoch(model, train_loader, optimizer)
    print(f"Epoch {epoch}: training...")
```

## Discriminative Learning Rates

Apply different learning rates to different layers—lower for bottom layers (general features), higher for top layers (task-specific):

```python
from typing import List, Dict, Any


def get_discriminative_lr_params(
    model: nn.Module,
    base_lr: float = 2e-5,
    lr_decay: float = 0.95,
    weight_decay: float = 0.01
) -> List[Dict[str, Any]]:
    """
    Create parameter groups with discriminative learning rates.
    
    Learning rate schedule:
    - Embeddings: base_lr * decay^num_layers (lowest)
    - Layer 0: base_lr * decay^(num_layers-1)
    - Layer 1: base_lr * decay^(num_layers-2)
    - ...
    - Layer N-1: base_lr * decay (second highest)
    - Classifier: base_lr (highest)
    
    Args:
        model: Model with encoder and classifier
        base_lr: Learning rate for top layer/classifier
        lr_decay: Multiplicative decay per layer (0 < decay < 1)
        weight_decay: Weight decay for regularization
    """
    param_groups = []
    
    # Get number of encoder layers
    if hasattr(model.encoder, 'layer'):
        encoder_layers = model.encoder.layer
    elif hasattr(model.encoder, 'layers'):
        encoder_layers = model.encoder.layers
    else:
        raise ValueError("Could not find encoder layers")
    
    num_layers = len(encoder_layers)
    
    # Embeddings - lowest learning rate
    if hasattr(model.encoder, 'embeddings'):
        emb_lr = base_lr * (lr_decay ** (num_layers + 1))
        param_groups.append({
            'params': model.encoder.embeddings.parameters(),
            'lr': emb_lr,
            'weight_decay': weight_decay,
            'name': 'embeddings'
        })
    
    # Encoder layers - gradually increasing LR
    for i, layer in enumerate(encoder_layers):
        layer_lr = base_lr * (lr_decay ** (num_layers - i))
        param_groups.append({
            'params': layer.parameters(),
            'lr': layer_lr,
            'weight_decay': weight_decay,
            'name': f'layer_{i}'
        })
    
    # Classifier head - highest learning rate
    if hasattr(model, 'classifier'):
        param_groups.append({
            'params': model.classifier.parameters(),
            'lr': base_lr,
            'weight_decay': weight_decay,
            'name': 'classifier'
        })
    
    return param_groups


def print_lr_schedule(param_groups: List[Dict[str, Any]]):
    """Print the learning rate schedule."""
    print("Learning Rate Schedule:")
    print("-" * 40)
    for group in param_groups:
        print(f"  {group.get('name', 'unnamed'):15s}: {group['lr']:.2e}")


# Usage
model = FullFineTuning("bert-base-uncased", num_labels=2)
param_groups = get_discriminative_lr_params(model, base_lr=2e-5, lr_decay=0.9)
print_lr_schedule(param_groups)

optimizer = torch.optim.AdamW(param_groups)
```

### Example Learning Rate Schedule (BERT-base, 12 layers)

| Component | Layer | LR (decay=0.9) | LR (decay=0.95) |
|-----------|-------|----------------|-----------------|
| Embeddings | - | 5.7e-7 | 1.1e-6 |
| Layer 0 | Bottom | 5.1e-6 | 6.9e-6 |
| Layer 5 | Middle | 1.2e-5 | 1.3e-5 |
| Layer 11 | Top | 1.8e-5 | 1.9e-5 |
| Classifier | - | 2.0e-5 | 2.0e-5 |

## Catastrophic Forgetting

A central challenge in fine-tuning is **catastrophic forgetting**: the model loses pre-trained knowledge when adapted to a specific task.

### Why It Happens

1. **Distribution shift**: Fine-tuning data differs from pretraining data
2. **Gradient interference**: Task gradients overwrite general knowledge
3. **Capacity reallocation**: Model repurposes neurons for new task

### Mitigation Strategies

#### 1. Regularization-Based (L2-SP)

Penalize deviation from pre-trained weights:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda \sum_i \|\theta_i - \theta_i^{\text{pre}}\|^2
$$

```python
class L2SPRegularizer:
    """
    L2-SP: L2 regularization toward Starting Point (pretrained weights).
    """
    
    def __init__(self, model: nn.Module, pretrained_state: dict, lambda_l2sp: float = 0.01):
        self.model = model
        self.pretrained_state = {k: v.clone() for k, v in pretrained_state.items()}
        self.lambda_l2sp = lambda_l2sp
    
    def penalty(self) -> torch.Tensor:
        """Compute L2 distance from pretrained weights."""
        penalty = 0.0
        for name, param in self.model.named_parameters():
            if name in self.pretrained_state:
                penalty += torch.sum((param - self.pretrained_state[name].to(param.device)) ** 2)
        return self.lambda_l2sp * penalty


# Usage
pretrained_state = {k: v.clone() for k, v in model.state_dict().items()}
regularizer = L2SPRegularizer(model, pretrained_state, lambda_l2sp=0.01)

# In training loop
loss = criterion(outputs, labels)
loss = loss + regularizer.penalty()
loss.backward()
```

#### 2. Elastic Weight Consolidation (EWC)

Weight important parameters more heavily using Fisher information:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \sum_i \frac{\lambda}{2} F_i (\theta_i - \theta_i^{\text{pre}})^2
$$

```python
class EWCRegularizer:
    """
    Elastic Weight Consolidation for preventing catastrophic forgetting.
    """
    
    def __init__(
        self,
        model: nn.Module,
        fisher_dict: dict,
        pretrained_state: dict,
        lambda_ewc: float = 1000
    ):
        self.model = model
        self.fisher = fisher_dict
        self.pretrained = pretrained_state
        self.lambda_ewc = lambda_ewc
    
    @classmethod
    def compute_fisher(
        cls,
        model: nn.Module,
        dataloader,
        num_samples: int = 1000
    ) -> dict:
        """Estimate Fisher information from data."""
        fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
        model.eval()
        
        count = 0
        for batch in dataloader:
            if count >= num_samples:
                break
            
            outputs = model(**batch)
            # Use log-likelihood for Fisher
            log_probs = torch.log_softmax(outputs.logits, dim=-1)
            labels = batch['labels']
            loss = -log_probs.gather(1, labels.unsqueeze(1)).mean()
            
            model.zero_grad()
            loss.backward()
            
            for n, p in model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad ** 2
            
            count += batch['input_ids'].size(0)
        
        # Normalize
        for n in fisher:
            fisher[n] /= count
        
        return fisher
    
    def penalty(self) -> torch.Tensor:
        penalty = 0.0
        for n, p in self.model.named_parameters():
            if n in self.fisher and n in self.pretrained:
                penalty += (self.fisher[n] * (p - self.pretrained[n].to(p.device)) ** 2).sum()
        return self.lambda_ewc * penalty
```

#### 3. Replay-Based

Mix task-specific data with samples from pretraining distribution:

```python
class ReplayDataLoader:
    """
    Mix task data with replay data from pretraining distribution.
    """
    
    def __init__(
        self,
        task_loader,
        replay_loader,
        replay_ratio: float = 0.1
    ):
        self.task_loader = task_loader
        self.replay_loader = replay_loader
        self.replay_ratio = replay_ratio
        self.replay_iter = iter(replay_loader)
    
    def __iter__(self):
        for task_batch in self.task_loader:
            yield task_batch
            
            # Occasionally yield replay batch
            if torch.rand(1).item() < self.replay_ratio:
                try:
                    replay_batch = next(self.replay_iter)
                except StopIteration:
                    self.replay_iter = iter(self.replay_loader)
                    replay_batch = next(self.replay_iter)
                yield replay_batch
```

#### 4. Architecture-Based

Use parameter-efficient methods (LoRA, adapters) that freeze pretrained weights—see [Ch 15.6](../../ch15/efficient_llm/efficiency_overview.md).

## Choosing a Strategy

### Decision Framework

```
                    Dataset Size
                    Small (<1K)    Medium (1K-100K)    Large (>100K)
Task Similarity     
to Pretraining

High              Feature         Discriminative LR    Full Fine-tune
                  Extraction      + Gradual Unfreeze   (careful LR)

Medium            LoRA/Adapters   Gradual Unfreeze     Full Fine-tune
                                  + L2-SP              

Low               LoRA with       Full Fine-tune       Full Fine-tune
                  larger rank     + EWC/Replay         
```

### Quick Reference

| Scenario | Recommended Strategy |
|----------|---------------------|
| Limited compute | Feature extraction or LoRA |
| Small dataset | Feature extraction → gradual unfreeze |
| Large dataset, similar task | Full fine-tune with low LR |
| Large dataset, different task | Full fine-tune with warmup |
| Need to preserve knowledge | LoRA, adapters, or EWC |
| Multiple tasks | Adapters (one per task) |

## Summary

| Method | Trainable Params | Forgetting Risk | Best For |
|--------|-----------------|-----------------|----------|
| Full Fine-tuning | 100% | High | Maximum performance |
| Feature Extraction | <1% | None | Small data, fast iteration |
| Gradual Unfreezing | Progressive | Medium | Balanced adaptation |
| Discriminative LR | 100% | Medium | Preserving low-level features |
| L2-SP / EWC | 100% | Low | Continual learning |

For parameter-efficient methods (LoRA, QLoRA, Prefix Tuning, Adapters), see [Ch 15.6 Efficient LLM Fine-tuning](../../ch15/efficient_llm/).

## References

1. Howard, J., & Ruder, S. (2018). "Universal Language Model Fine-tuning for Text Classification." ACL.
2. Kirkpatrick, J., et al. (2017). "Overcoming Catastrophic Forgetting in Neural Networks." PNAS.
3. Li, X., et al. (2018). "Explicit Inductive Bias for Transfer Learning with Convolutional Networks." ICML.
4. Peters, M., et al. (2019). "To Tune or Not to Tune? Adapting Pretrained Representations to Diverse Tasks."
