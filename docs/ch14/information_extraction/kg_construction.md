# Knowledge Graph Construction

## Learning Objectives

- Understand the end-to-end KG construction pipeline
- Implement entity linking and KG embedding methods
- Apply KG construction to financial domain

## What Is a Knowledge Graph?

A knowledge graph represents structured knowledge as a collection of triples:

$$(h, r, t) \in \mathcal{E} \times \mathcal{R} \times \mathcal{E}$$

where $h$ is the head entity, $r$ is the relation, and $t$ is the tail entity. Example: `(Apple Inc., headquartered_in, Cupertino)`.

## Construction Pipeline

### Step 1: Entity Linking

Map textual mentions to canonical KB entities, resolving ambiguity. "Apple" could refer to Apple Inc. (company) or apple (fruit). Approach: candidate generation via surface form matching, then candidate ranking using contextual similarity:

$$e^* = \arg\max_{e \in \mathcal{C}(m)} \left[\text{sim}(\mathbf{h}_m, \mathbf{h}_e) + \text{prior}(e \mid m)\right]$$

### Step 2: Triple Extraction

Apply relation extraction to linked entities. From *"Apple, headquartered in Cupertino, was founded by Steve Jobs"*, extract: (Apple_Inc, headquartered_in, Cupertino) and (Apple_Inc, founded_by, Steve_Jobs).

### Step 3: Triple Validation

Validate using confidence thresholding, type constraints (subject/object types must match relation signature), and temporal constraints.

### Step 4: Graph Completion

Infer missing triples via embedding methods.

## KG Embedding Methods

| Method | Scoring Function | Key Idea |
|--------|-----------------|----------|
| TransE | $\lVert\mathbf{h} + \mathbf{r} - \mathbf{t}\rVert$ | Translation in embedding space |
| DistMult | $\langle \mathbf{h}, \mathbf{r}, \mathbf{t} \rangle$ | Bilinear diagonal |
| ComplEx | $\text{Re}(\langle \mathbf{h}, \mathbf{r}, \bar{\mathbf{t}} \rangle)$ | Complex-valued embeddings |
| RotatE | $\lVert\mathbf{h} \circ \mathbf{r} - \mathbf{t}\rVert$ | Rotation in complex space |

```python
import torch
import torch.nn as nn

class TransE(nn.Module):
    def __init__(self, num_entities, num_relations, embed_dim=100, margin=1.0):
        super().__init__()
        self.entity_emb = nn.Embedding(num_entities, embed_dim)
        self.relation_emb = nn.Embedding(num_relations, embed_dim)
        self.margin = margin
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

    def score(self, h, r, t):
        h_emb = self.entity_emb(h)
        r_emb = self.relation_emb(r)
        t_emb = self.entity_emb(t)
        return -torch.norm(h_emb + r_emb - t_emb, p=2, dim=-1)

    def loss(self, pos_h, pos_r, pos_t, neg_h, neg_r, neg_t):
        pos_score = self.score(pos_h, pos_r, pos_t)
        neg_score = self.score(neg_h, neg_r, neg_t)
        return torch.relu(self.margin - pos_score + neg_score).mean()
```

## Financial Knowledge Graphs

### Entity Types

Companies, Persons (executives), Products, Locations, Financial Instruments, Regulatory Bodies.

### Key Relations

| Relation | Example |
|----------|---------|
| subsidiary_of | (Instagram, subsidiary_of, Meta) |
| board_member | (Jensen Huang, board_member, NVIDIA) |
| competitor | (Visa, competitor, Mastercard) |
| supplies_to | (TSMC, supplies_to, Apple) |
| listed_on | (Apple, listed_on, NASDAQ) |
| sector | (JPMorgan, sector, Financial Services) |

### Applications

- **Portfolio risk**: Identify hidden correlations through supply chain links
- **Event propagation**: Trace how events impact connected entities
- **Compliance**: Map ownership structures for sanctions screening
- **Alpha generation**: Discover non-obvious relationships for trading signals

## References

1. Bordes, A., et al. (2013). Translating Embeddings for Modeling Multi-Relational Data. *NeurIPS*.
2. Ji, S., et al. (2022). A Survey on Knowledge Graphs. *IEEE TNNLS*.
3. Hogan, A., et al. (2021). Knowledge Graphs. *ACM Computing Surveys*.
