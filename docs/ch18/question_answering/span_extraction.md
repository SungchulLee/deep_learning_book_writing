# Span Extraction

## Mathematical Formulation

Given question $q$ and context $c$, predict start position $s$ and end position $e$:

$$P(s=i | q, c) = \frac{\exp(\mathbf{w}_s^T \mathbf{h}_i)}{\sum_j \exp(\mathbf{w}_s^T \mathbf{h}_j)}$$

$$P(e=j | q, c) = \frac{\exp(\mathbf{w}_e^T \mathbf{h}_j)}{\sum_k \exp(\mathbf{w}_e^T \mathbf{h}_k)}$$

### Training Loss

$$\mathcal{L} = -\log P(s = s^*) - \log P(e = e^*)$$

### Inference

Select span $(i, j)$ with $i \leq j$ and $j - i < L_{\max}$ maximizing:

$$\hat{s}, \hat{e} = \arg\max_{i \leq j} \left[\log P(s=i) + \log P(e=j)\right]$$

## Implementation

```python
import torch
import torch.nn as nn
from transformers import AutoModel

class SpanExtractor(nn.Module):
    def __init__(self, model_name="bert-base-uncased"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.start_head = nn.Linear(hidden_size, 1)
        self.end_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.encoder(
            input_ids, attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        hidden = outputs.last_hidden_state  # (batch, seq_len, hidden)
        start_logits = self.start_head(hidden).squeeze(-1)
        end_logits = self.end_head(hidden).squeeze(-1)
        return start_logits, end_logits

    def predict(self, start_logits, end_logits, max_span_length=30):
        batch_size = start_logits.size(0)
        starts, ends = [], []
        for b in range(batch_size):
            s_scores = start_logits[b].softmax(dim=-1)
            e_scores = end_logits[b].softmax(dim=-1)
            best_score, best_s, best_e = -1, 0, 0
            for i in range(len(s_scores)):
                for j in range(i, min(i + max_span_length, len(e_scores))):
                    score = s_scores[i] * e_scores[j]
                    if score > best_score:
                        best_score, best_s, best_e = score, i, j
            starts.append(best_s)
            ends.append(best_e)
        return starts, ends
```

## No-Answer Prediction (SQuAD 2.0)

For unanswerable questions, compare the best span score against a null score:

$$\text{has\_answer} = \max_{i,j} (s_i + e_j) > s_{[CLS]} + e_{[CLS]} + \tau$$

where $\tau$ is a learned threshold.

## Summary

1. Span extraction predicts independent start/end distributions over context tokens
2. Cross-entropy loss on gold start/end positions is the standard objective
3. Efficient inference selects the highest-scoring valid span
4. No-answer prediction compares span scores against a null hypothesis
