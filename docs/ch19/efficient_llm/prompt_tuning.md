# Prompt Tuning

## Learning Objectives

- Understand soft prompt tuning vs. discrete prompting
- Compare prompt tuning with prefix tuning
- Implement prompt tuning for financial tasks

## Core Concept

**Prompt tuning** (Lester et al., 2021) prepends learnable **soft prompt** embeddings to the input, while keeping all model parameters frozen:

$$\hat{y} = \text{LLM}([P_1, P_2, \ldots, P_m, x_1, x_2, \ldots, x_n]; \theta_{\text{frozen}})$$

where $P_1, \ldots, P_m$ are continuous embedding vectors (not tied to any vocabulary tokens) optimized via backpropagation.

## Prompt Tuning vs. Prefix Tuning

| Aspect | Prompt Tuning | Prefix Tuning |
|--------|--------------|---------------|
| Where added | Input embeddings only | Every layer's K, V |
| Trainable params | $m \times d$ | $m \times L \times 2d$ |
| Expressiveness | Lower | Higher |
| Task performance | Good at scale | Good at all scales |

Where $m$ = prompt length, $d$ = hidden dim, $L$ = number of layers.

## Implementation

```python
import torch
import torch.nn as nn


class PromptTuning(nn.Module):
    def __init__(self, base_model, n_prompt_tokens=20, hidden_size=4096):
        super().__init__()
        self.base_model = base_model
        self.n_prompt_tokens = n_prompt_tokens

        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Learnable soft prompt embeddings
        self.soft_prompt = nn.Parameter(
            torch.randn(n_prompt_tokens, hidden_size) * 0.01
        )

    def forward(self, input_ids, attention_mask=None):
        batch_size = input_ids.shape[0]

        # Get input embeddings
        input_embeds = self.base_model.get_input_embeddings()(input_ids)

        # Prepend soft prompt
        prompt_embeds = self.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1)
        combined_embeds = torch.cat([prompt_embeds, input_embeds], dim=1)

        # Adjust attention mask
        if attention_mask is not None:
            prompt_mask = torch.ones(batch_size, self.n_prompt_tokens,
                                   device=attention_mask.device)
            attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)

        return self.base_model(inputs_embeds=combined_embeds,
                              attention_mask=attention_mask)
```

## Scaling Behavior

Lester et al. (2021) showed that prompt tuning approaches full fine-tuning quality as model size increases:

| Model Size | Prompt Tuning | Full Fine-Tuning |
|-----------|--------------|-----------------|
| T5-Small (60M) | 82.1% | 89.5% |
| T5-Base (220M) | 87.3% | 90.1% |
| T5-Large (770M) | 89.5% | 90.4% |
| T5-XXL (11B) | 90.2% | 90.5% |

*SuperGLUE average accuracy*

## References

1. Lester, B., et al. (2021). "The Power of Scale for Parameter-Efficient Prompt Tuning." *EMNLP*.
2. Li, X. & Liang, P. (2021). "Prefix-Tuning: Optimizing Continuous Prompts for Generation." *ACL*.
