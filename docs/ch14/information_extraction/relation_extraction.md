# Relation Extraction

## Learning Objectives

- Understand pipeline and joint approaches to relation extraction
- Implement BERT-based relation classification
- Apply distant supervision for large-scale RE training

## Task Definition

Given a sentence with two identified entities, predict the semantic relation between them:

$$r^* = \arg\max_{r \in \mathcal{R}} P(r \mid e_1, e_2, \mathbf{x})$$

where $\mathcal{R}$ is a predefined set of relation types (e.g., `founded-by`, `headquartered-in`, `subsidiary-of`, `no-relation`).

### Example

Input: `[Tim Cook]_{e1}` is the CEO of `[Apple Inc.]_{e2}`.
Output: `CEO-of(Tim Cook, Apple Inc.)`

## Pipeline Approach

### Step 1: Entity Pair Enumeration

After NER, enumerate all entity pairs within a sentence or document window. Type constraints reduce the combinatorial space — e.g., `CEO-of` requires (PERSON, ORG).

### Step 2: Relation Classification

Concatenate entity representations with context:

$$P(r \mid e_1, e_2, \mathbf{x}) = \text{softmax}\left(\mathbf{W} \left[\mathbf{h}_{e_1}; \mathbf{h}_{e_2}; \mathbf{h}_{[\text{CLS}]}\right] + \mathbf{b}\right)$$

### BERT-Based Implementation

```python
import torch
import torch.nn as nn
from transformers import AutoModel

class BERTRelationClassifier(nn.Module):
    def __init__(self, model_name, num_relations, hidden_dim=768):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_relations),
        )

    def forward(self, input_ids, attention_mask, e1_mask, e2_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        h = outputs.last_hidden_state  # (B, L, H)

        # Entity representations via mean pooling over entity tokens
        e1_h = (h * e1_mask.unsqueeze(-1)).sum(1) / e1_mask.sum(1, keepdim=True)
        e2_h = (h * e2_mask.unsqueeze(-1)).sum(1) / e2_mask.sum(1, keepdim=True)
        cls_h = h[:, 0]  # [CLS] token

        combined = torch.cat([cls_h, e1_h, e2_h], dim=-1)
        return self.classifier(combined)
```

## Distant Supervision

Manually labeling relation instances is expensive. **Distant supervision** (Mintz et al., 2009) automatically generates training data by aligning knowledge base triples with text.

**Assumption**: If entity pair $(e_1, e_2)$ participates in relation $r$ in the KB, then any sentence mentioning both entities may express $r$.

This assumption is noisy. Multi-instance learning addresses this:

$$P(r \mid \text{bag}) = \max_{x_i \in \text{bag}} P(r \mid e_1, e_2, x_i)$$

### Noise Reduction

1. **Multi-instance learning**: Aggregate evidence across multiple sentences
2. **Attention over instances**: Weight sentences by relevance
3. **Reinforcement learning**: Train a selector to choose informative instances

## Joint Entity and Relation Extraction

Joint models extract entities and relations simultaneously, avoiding error propagation. Span-based approaches enumerate candidate spans, classify entity types, then classify relations for each entity pair.

## Datasets

| Dataset | Relations | Instances | Domain |
|---------|-----------|-----------|--------|
| SemEval-2010 Task 8 | 9 + Other | 10,717 | General |
| TACRED | 41 + no_rel | 106,264 | Newswire |
| DocRED | 96 | 63,427 | Wikipedia |
| FewRel | 100 | 70,000 | Few-shot |

## Financial Relation Extraction

Key financial relations include `subsidiary-of`, `CEO-of`, `invested-in`, `competitor-of`, `supplied-by`, `audited-by`, and `regulated-by`. These enable construction of financial knowledge graphs for risk analysis, supply chain modeling, and portfolio management.

## References

1. Mintz, M., et al. (2009). Distant Supervision for RE Without Labeled Data. *ACL*.
2. Baldini Soares, L., et al. (2019). Matching the Blanks: Distributional Similarity for RE. *ACL*.
3. Zhong, Z., & Chen, D. (2021). A Frustratingly Easy Approach for Entity and Relation Extraction. *NAACL*.
