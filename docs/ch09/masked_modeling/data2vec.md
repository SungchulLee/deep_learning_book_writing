# Data2Vec: A General Framework for Self-Supervised Learning

Data2Vec provides a unified self-supervised learning framework that works across modalities (vision, speech, language) by predicting latent representations of the full input from a masked view.

## Key Innovation

Unlike BEiT (discrete tokens) or MAE (raw pixels), Data2Vec predicts **contextualised latent representations** from a teacher network:

$$\mathcal{L} = \frac{1}{|M|} \sum_{t \in M} \| f_\theta(\tilde{x})_t - \text{sg}(g_\xi(x))_t \|^2$$

where $\tilde{x}$ is the masked input, $f_\theta$ is the student encoder, $g_\xi$ is the teacher encoder (EMA of student), $M$ is the set of masked positions, and $\text{sg}$ denotes stop-gradient.

## Architecture

```
Student (trained):
  Masked input → Transformer Encoder → Predict representations at masked positions

Teacher (EMA updated):
  Full input → Transformer Encoder → Target representations (top-K layer average)
```

## Implementation

```python
import torch
import torch.nn as nn
import copy


class Data2Vec(nn.Module):
    """Data2Vec: predict contextualised representations from masked input."""
    
    def __init__(self, encoder, top_k_layers=8, momentum=0.999):
        super().__init__()
        self.student = encoder
        self.teacher = copy.deepcopy(encoder)
        self.top_k_layers = top_k_layers
        self.momentum = momentum
        
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        hidden_dim = 768
        self.regression_head = nn.Linear(hidden_dim, hidden_dim)
    
    @torch.no_grad()
    def update_teacher(self):
        for s_param, t_param in zip(self.student.parameters(), self.teacher.parameters()):
            t_param.data = self.momentum * t_param.data + (1 - self.momentum) * s_param.data
    
    def forward(self, x, mask):
        with torch.no_grad():
            teacher_output = self.teacher(x, output_hidden_states=True)
            top_k = teacher_output.hidden_states[-self.top_k_layers:]
            teacher_targets = torch.stack(top_k).mean(dim=0).detach()
        
        student_output = self.student(x, mask=mask)
        student_predictions = self.regression_head(student_output)
        
        loss = nn.functional.smooth_l1_loss(
            student_predictions[mask], teacher_targets[mask], beta=2.0
        )
        return loss
```

## Cross-Modal Generality

| Modality | Masking strategy | Target |
|----------|-----------------|--------|
| Vision | Random patch masking | Top-K layer representations |
| Speech | Contiguous span masking | Top-K layer representations |
| Language | Token masking (BERT-style) | Top-K layer representations |

## References

1. Baevski, A., et al. (2022). "data2vec: A General Framework for Self-Supervised Learning in Speech, Vision and Language." *ICML*.
