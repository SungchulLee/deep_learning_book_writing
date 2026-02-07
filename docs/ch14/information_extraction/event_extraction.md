# Event Extraction

## Learning Objectives

- Understand event trigger detection and argument extraction
- Distinguish between sentence-level and document-level event extraction
- Apply event extraction to financial text

## Task Definition

Event extraction identifies **structured event records** from text. Each event consists of:

- **Trigger**: The word(s) that most clearly express the event occurrence
- **Event Type**: Category from a predefined ontology
- **Arguments**: Participants and attributes with defined roles

### Example

Input: *"Apple acquired Beats Electronics for approximately $3 billion on May 28, 2014."*

| Component | Value | Role |
|-----------|-------|------|
| Trigger | acquired | -- |
| Event Type | Acquisition | -- |
| Argument | Apple | Buyer |
| Argument | Beats Electronics | Target |
| Argument | $3 billion | Price |
| Argument | May 28, 2014 | Date |

## Two-Stage Pipeline

### Stage 1: Trigger Detection

Identify event trigger words and classify their event type. This is a token-level classification task:

$$P(\text{type}_i \mid x_i, \mathbf{x}) = \text{softmax}(\mathbf{W}_t \mathbf{h}_i + \mathbf{b}_t)$$

where $\mathbf{h}_i$ is the contextualized representation of token $i$.

### Stage 2: Argument Extraction

For each detected trigger, identify argument spans and assign roles:

$$P(\text{role} \mid x_{i:j}, \text{trigger}, \mathbf{x}) = \text{softmax}(\mathbf{W}_a [\mathbf{h}_{i:j}; \mathbf{h}_{\text{trigger}}] + \mathbf{b}_a)$$

## BERT-Based Event Extraction

```python
import torch
import torch.nn as nn
from transformers import AutoModel

class EventExtractor(nn.Module):
    def __init__(self, model_name, num_event_types, num_roles, hidden_dim=768):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.trigger_classifier = nn.Linear(hidden_dim, num_event_types)
        self.argument_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_roles),
        )

    def forward(self, input_ids, attention_mask, trigger_idx=None):
        h = self.encoder(input_ids, attention_mask=attention_mask).last_hidden_state

        # Trigger detection: classify each token
        trigger_logits = self.trigger_classifier(h)  # (B, L, num_event_types)

        # Argument extraction: conditioned on trigger position
        if trigger_idx is not None:
            trigger_h = h[torch.arange(h.size(0)), trigger_idx]  # (B, H)
            trigger_expanded = trigger_h.unsqueeze(1).expand_as(h)
            combined = torch.cat([h, trigger_expanded], dim=-1)
            arg_logits = self.argument_classifier(combined)  # (B, L, num_roles)
            return trigger_logits, arg_logits

        return trigger_logits, None
```

## Document-Level Event Extraction

Sentence-level models miss arguments spread across multiple sentences:

*"Apple announced a major deal on Monday. The tech giant will pay $3 billion for Beats Electronics. Tim Cook called it a great acquisition."*

Arguments for the acquisition event span three sentences. Document-level models use cross-sentence attention or entity coreference to link distributed arguments.

## Financial Event Types

| Event Type | Trigger Examples | Key Arguments |
|------------|-----------------|---------------|
| Earnings | reported, posted | company, revenue, EPS, period |
| M&A | acquired, merged | buyer, target, price, date |
| IPO | went public, listed | company, exchange, price, shares |
| Bankruptcy | filed, defaulted | company, chapter, date, liabilities |
| Executive Change | appointed, resigned | person, role, company |
| Dividend | declared, distributed | company, amount, ex-date |
| Stock Split | split, divided | company, ratio, record date |

## Evaluation

- **Trigger Identification**: F1 on trigger span detection
- **Trigger Classification**: F1 on trigger span + event type
- **Argument Identification**: F1 on argument span detection
- **Argument Classification**: F1 on argument span + role assignment

ACE 2005 benchmark results (approximate):

| Model | Trigger F1 | Argument F1 |
|-------|-----------|------------|
| DMCNN (2015) | 67.6 | 45.7 |
| JMEE (2018) | 73.7 | 51.1 |
| OneIE (2020) | 74.7 | 56.8 |
| DEGREE (2022) | 76.3 | 58.2 |

## References

1. Chen, Y., et al. (2015). Event Extraction via Dynamic Multi-Pooling CNNs. *ACL*.
2. Lin, Y., et al. (2020). A Joint Neural Model for IE with Global Features. *ACL*.
3. Hsu, I., et al. (2022). DEGREE: A Data-Efficient Generation-Based Event Extraction Model. *NAACL*.
