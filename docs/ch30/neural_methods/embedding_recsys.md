# Embedding-Based Recommender Systems

## Learning Objectives

- Understand the two-tower architecture and why it enables scalable retrieval
- Connect recommendation embeddings to word embeddings and metric learning
- Implement a two-tower model with contrastive learning in PyTorch
- Understand approximate nearest neighbor (ANN) retrieval for large-scale recommendation
- Analyze the tradeoffs between dot-product and cross-feature models

## From Embeddings to Retrieval

The matrix factorization models in Section 29.1.3 learn user embeddings $\mathbf{p}_u$ and item embeddings $\mathbf{q}_i$ such that their dot product predicts ratings. This embedding view has a powerful implication: **recommendation becomes nearest neighbor search in embedding space**.

Given a user embedding $\mathbf{p}_u$, the top-$K$ recommended items are those whose embeddings $\mathbf{q}_i$ have the largest dot product (or cosine similarity) with $\mathbf{p}_u$:

$$\text{Top-}K(u) = \underset{i \in \mathcal{I}}{\text{argmax-}K} \; \mathbf{p}_u^\top \mathbf{q}_i$$

This formulation enables sub-linear retrieval using approximate nearest neighbor (ANN) data structures, making it practical for catalogs with millions of items.

## The Two-Tower Architecture

### Design

The two-tower (or dual-encoder) architecture processes users and items through **separate** encoder networks, producing embeddings that are compared via dot product:

$$\hat{y}_{ui} = \phi_u(u; \theta_u)^\top \phi_i(i; \theta_i)$$

where $\phi_u$ and $\phi_i$ are the user and item tower networks, respectively.

```python
class TwoTowerModel(nn.Module):
    """
    Two-tower retrieval model for recommendations.
    
    Each tower independently encodes its input into a shared
    embedding space. Scoring is a dot product, enabling
    efficient ANN retrieval at inference time.
    """
    def __init__(self, num_users, num_items, user_feat_dim=0,
                 item_feat_dim=0, emb_size=64, tower_dim=64):
        super().__init__()
        
        # User tower
        self.user_emb = nn.Embedding(num_users, emb_size)
        user_input_dim = emb_size + user_feat_dim
        self.user_tower = nn.Sequential(
            nn.Linear(user_input_dim, tower_dim * 2),
            nn.ReLU(),
            nn.Linear(tower_dim * 2, tower_dim),
            nn.LayerNorm(tower_dim)
        )
        
        # Item tower
        self.item_emb = nn.Embedding(num_items, emb_size)
        item_input_dim = emb_size + item_feat_dim
        self.item_tower = nn.Sequential(
            nn.Linear(item_input_dim, tower_dim * 2),
            nn.ReLU(),
            nn.Linear(tower_dim * 2, tower_dim),
            nn.LayerNorm(tower_dim)
        )
    
    def encode_user(self, user_ids, user_features=None):
        """Encode users into the shared embedding space."""
        x = self.user_emb(user_ids)
        if user_features is not None:
            x = torch.cat([x, user_features], dim=1)
        return self.user_tower(x)
    
    def encode_item(self, item_ids, item_features=None):
        """Encode items into the shared embedding space."""
        x = self.item_emb(item_ids)
        if item_features is not None:
            x = torch.cat([x, item_features], dim=1)
        return self.item_tower(x)
    
    def forward(self, user_ids, item_ids, 
                user_features=None, item_features=None):
        u = self.encode_user(user_ids, user_features)
        v = self.encode_item(item_ids, item_features)
        return (u * v).sum(1)
```

### Why Two Towers?

The critical property is that **user and item representations are computed independently**. This enables:

1. **Precomputation**: Item embeddings can be computed offline and stored in an index. At serving time, only the user tower runs.
2. **ANN retrieval**: The dot product between the user embedding and all item embeddings can be approximated using ANN data structures in sub-millisecond time.
3. **Asynchronous updates**: Item embeddings can be updated on a different schedule than user embeddings.

The tradeoff is that the two towers cannot model **cross-features** between users and items (unlike NCF, which concatenates embeddings and passes them through a joint MLP). This limits expressiveness but enables massive scale.

## Training with Contrastive Learning

### In-Batch Negatives

For implicit feedback, the two-tower model is typically trained with **contrastive learning** using in-batch negatives. Given a batch of $B$ positive (user, item) pairs, each item serves as a negative for all other users in the batch:

$$\mathcal{L} = -\frac{1}{B} \sum_{k=1}^B \log \frac{\exp(\mathbf{u}_k^\top \mathbf{v}_k / \tau)}{\sum_{j=1}^B \exp(\mathbf{u}_k^\top \mathbf{v}_j / \tau)}$$

where $\tau$ is a temperature parameter. This is the **InfoNCE** loss, also known as the NT-Xent loss from SimCLR.

```python
def contrastive_loss(user_embs, item_embs, temperature=0.1):
    """
    In-batch contrastive loss for two-tower training.
    
    Args:
        user_embs: (batch, dim) — user tower outputs
        item_embs: (batch, dim) — item tower outputs
        temperature: scaling parameter
    
    Returns:
        Scalar loss
    """
    logits = torch.mm(user_embs, item_embs.t()) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, labels)
```

### Temperature Scaling

The temperature $\tau$ controls the difficulty of the contrastive task:

- **Low $\tau$** (e.g., 0.05): Sharp distribution — the model must strongly distinguish positive from negative items. Produces more discriminative embeddings but harder to optimize.
- **High $\tau$** (e.g., 1.0): Soft distribution — easier to optimize but less discriminative.

### Hard Negative Mining

In-batch negatives are often too easy — random items from the batch are unlikely to be confusing. **Hard negative mining** selects negatives that are similar to the positive item:

- **Static hard negatives**: Items from the same category or with similar features.
- **Dynamic hard negatives**: Items with high model scores that are not actually relevant.
- **Semi-hard negatives**: Items closer to the user than the positive but not the closest, avoiding collapsed representations.

## Approximate Nearest Neighbor Retrieval

### The Retrieval Problem

At inference, we need to find the top-$K$ items from a catalog of $n$ items:

$$\text{Top-}K(u) = \underset{i \in \{1, \ldots, n\}}{\text{argmax-}K} \; \mathbf{u}^\top \mathbf{v}_i$$

Brute-force computation is $O(n \cdot d)$ per query. For $n = 10^6$ items and $d = 128$, this is ~$10^8$ operations per request — too slow for real-time serving.

### ANN Methods

Approximate nearest neighbor algorithms trade exact results for speed:

| Method | Idea | Complexity |
|--------|------|-----------|
| **LSH** | Hash vectors so similar ones map to same bucket | Sub-linear |
| **IVF** | Partition space into Voronoi cells, search nearby cells | $O(\sqrt{n} \cdot d)$ |
| **HNSW** | Navigate a hierarchical graph of neighbors | $O(\log n \cdot d)$ |
| **Product Quantization** | Compress vectors, compute approximate distances | Reduces $d$ |

FAISS (Facebook AI Similarity Search) is the most widely used library:

```python
import faiss

# Build index from item embeddings
dim = 128
index = faiss.IndexFlatIP(dim)           # Inner product (dot product)
index.add(item_embeddings.numpy())       # Add all item vectors

# Query: find top-K items for a user
k = 100
scores, indices = index.search(user_embedding.numpy(), k)
```

For large catalogs, use an approximate index:

```python
nlist = 1000    # Number of Voronoi cells
m = 16          # Number of PQ sub-quantizers
quantizer = faiss.IndexFlatIP(dim)
index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, 8)
index.train(item_embeddings.numpy())
index.add(item_embeddings.numpy())
index.nprobe = 10   # Search 10 nearest cells
```

### The Retrieval–Ranking Pipeline

Production systems use a multi-stage pipeline:

1. **Retrieval** (two-tower + ANN): Generate ~1000 candidates from millions of items. Must be fast (<10ms).
2. **Ranking** (cross-feature model): Re-score the candidates with a more expressive model (e.g., NCF, deep hybrid). Can be slower (~50ms).
3. **Re-ranking** (business rules): Apply diversity, freshness, and fairness constraints.

The two-tower model powers stage 1, while more expressive NCF-style models (Section 29.2.1) power stage 2.

## Connection to Other Embedding Methods

### Item2Vec

Item2Vec (Barkan and Koenigstein, 2016) directly applies the Word2Vec skip-gram architecture to recommendation. Items co-occurring in user sessions are treated like words co-occurring in sentences:

- **"Sentences"**: User interaction sequences
- **"Words"**: Item IDs
- **Training**: Skip-gram with negative sampling

This produces item embeddings where co-consumed items are close in embedding space. It is a special case of MF where the "rating" is binary (co-occurrence) and the loss is the skip-gram objective.

### Metric Learning

The two-tower architecture can be trained with metric learning losses that directly optimize the embedding geometry:

**Triplet loss**: Given anchor user $u$, positive item $i^+$, and negative item $i^-$:

$$\mathcal{L}_{\text{triplet}} = \max\bigl(0, \|\mathbf{u} - \mathbf{v}^+\|^2 - \|\mathbf{u} - \mathbf{v}^-\|^2 + \alpha\bigr)$$

where $\alpha$ is a margin. This pushes positive items closer and negative items farther from the user in embedding space.

### Graph Neural Network Embeddings

For user-item interaction graphs, graph neural networks (GNNs) provide an alternative embedding approach. **PinSage** (Ying et al., 2018) learns item embeddings by aggregating information from neighboring items in the interaction graph. **LightGCN** (He et al., 2020) simplifies the GNN approach by removing nonlinearities and using only neighborhood aggregation:

$$\mathbf{e}_u^{(l+1)} = \sum_{i \in \mathcal{N}_u} \frac{1}{\sqrt{|\mathcal{N}_u|}\sqrt{|\mathcal{N}_i|}} \mathbf{e}_i^{(l)}$$

The final embeddings are the average across all layers, capturing multi-hop collaborative signals.

## Embedding Quality Analysis

### Embedding Visualization

After training, embeddings can be visualized with t-SNE or UMAP to verify that they capture meaningful structure. Well-trained item embeddings should cluster by genre, style, or other semantic attributes — even though these attributes were never explicitly used as features.

```python
def analyze_embeddings(model, df_train, movies_df):
    """Analyze learned embeddings after training."""
    item_emb = model.item_emb.weight.data
    
    # Find nearest neighbors for a sample item
    sample_idx = 0
    sample_vec = item_emb[sample_idx].unsqueeze(0)
    cos_sim = F.cosine_similarity(sample_vec, item_emb, dim=1)
    top_k = cos_sim.topk(6)  # Top 5 + self
    
    for idx, sim in zip(top_k.indices[1:], top_k.values[1:]):
        print(f"  Item {idx.item()} | similarity = {sim.item():.4f}")
```

### Bias in Embeddings

Learned embeddings inherit biases from the training data. The source code's `analyze_embeddings` function reveals item biases — systematically high or low learned bias values correspond to universally popular or unpopular items. Similarly, user embeddings may cluster by demographic patterns present in the interaction data.

## Embedding-Based RecSys in Finance

Embedding-based retrieval is particularly valuable in financial applications:

- **Similar security search**: Learn embeddings from co-holding patterns (securities frequently held together) to find similar stocks, bonds, or funds. Analogous to Item2Vec with portfolio holdings as "sentences."
- **Client matching**: Embed clients based on their transaction history and portfolio composition. Find similar clients for peer comparison or product cross-selling.
- **Trade idea retrieval**: Embed research reports and trade ideas. When a market event occurs, retrieve the most relevant historical analyses via ANN search.

!!! info "Connection to Factor Models"
    In the MF framework, item embeddings $\mathbf{q}_i$ are analogous to factor loadings in asset pricing models. The user embedding $\mathbf{p}_u$ represents the investor's factor preferences. The dot product $\mathbf{p}_u^\top \mathbf{q}_i$ is the expected return of the asset given the investor's factor tilts. This connection deepens when item features (risk factors) are explicitly included in the item tower of a two-tower model.

## Summary

Embedding-based recommender systems frame recommendation as nearest neighbor search in a learned embedding space. The two-tower architecture enables scalable retrieval by computing user and item representations independently, while contrastive learning with in-batch negatives provides an efficient training signal for implicit feedback. ANN libraries like FAISS make it practical to search over millions of items in milliseconds. The multi-stage retrieval–ranking pipeline — fast embedding-based candidate generation followed by expressive cross-feature re-ranking — is the dominant architecture in production recommender systems.

---

## Exercises

1. **Two-tower on MovieLens**: Implement the `TwoTowerModel` and train it on MovieLens using contrastive loss. Convert the explicit ratings to implicit feedback (rating ≥ 4 = positive). Compare Hit Rate@10 against MF.

2. **Temperature sweep**: Train the two-tower model with $\tau \in \{0.01, 0.05, 0.1, 0.5, 1.0\}$. Plot Hit Rate@10 vs temperature. What is the optimal $\tau$?

3. **ANN accuracy**: Using FAISS, build both an exact (`IndexFlatIP`) and approximate (`IndexIVFPQ`) index from trained item embeddings. Measure recall@100 (fraction of exact top-100 items recovered by the approximate index) as a function of `nprobe`.

4. **Item2Vec**: Implement Item2Vec using PyTorch. Treat each user's chronological item sequence as a "sentence" and train skip-gram embeddings. Visualize the resulting item embeddings with t-SNE, colored by genre.

5. **Embedding analysis**: After training a two-tower model, compute pairwise cosine similarities between all item embeddings. Do items in the same genre have higher similarity? Quantify using the average intra-genre vs inter-genre similarity.

6. **Retrieval–ranking pipeline**: Build a two-stage system: (1) two-tower retrieval to get top-100 candidates, (2) NCF re-ranking of the candidates. Compare end-to-end Hit Rate@10 against using either model alone.
