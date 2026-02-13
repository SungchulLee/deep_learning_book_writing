# Graph Neural Networks for Recommendations

## Introduction

Graph Neural Networks (GNNs) extend the power of neural networks to graph-structured data, enabling end-to-end learning on user-item interaction networks without explicit feature engineering. GNNs learn latent node embeddings by propagating information through graph edges—each node incorporates information from neighbors, enabling collaborative filtering through neural message passing. Unlike hand-crafted graph algorithms (PageRank, random walks), GNNs learn what information to aggregate and how to combine it through backpropagation.

GNNs revolutionize recommendation systems by enabling simultaneous learning of multiple objectives: predict user preferences, discover new items, account for side information (user/item features), and incorporate temporal dynamics. In financial applications, GNNs can model complex relationships between investors, assets, sectors, and market conditions through multi-layered heterogeneous graphs.

This section develops GNN frameworks for recommendations, explores architectures tailored to recommendation problems, and demonstrates implementations for financial recommendation systems.

## Key Concepts

### Message Passing on Graphs
- **Node Embeddings**: Vector representation h_v learned for each node
- **Neighborhood Aggregation**: Embeddings updated incorporating neighbors' information
- **Multiple Layers**: Stacking aggregation enables multi-hop neighborhood influence
- **End-to-End Learning**: Embeddings optimized for recommendation objective

### GNN Architectures for Recommendations
- **Graph Convolutional Networks (GCN)**: Linear aggregation + nonlinearity
- **GraphSAGE**: Sampling-based neighbor aggregation for scalability
- **Graph Attention Networks (GAT)**: Learned attention weights on neighbors
- **GNNs for Heterogeneous Graphs**: Multiple edge/node types

## Mathematical Framework

### Graph Convolutional Network (GCN) Layer

Update node embeddings using neighbor information:

$$h_v^{(l+1)} = \sigma\left(W^{(l)} \sum_{u \in \mathcal{N}(v)} \frac{1}{\sqrt{d_v d_u}} h_u^{(l)}\right)$$

where:
- N(v) = neighbors of node v
- W^{(l)} = learned weight matrix at layer l
- d_v = degree of node v (normalization)
- σ = activation function

Intuition: Average neighbor embeddings; apply learned transformation.

### Message Passing Formulation

Generic message passing framework:

$$h_v^{(l+1)} = \text{UPDATE}^{(l)}(h_v^{(l)}, \text{AGGREGATE}^{(l)}(\{h_u^{(l)} : u \in \mathcal{N}(v)\}))$$

**AGGREGATE**: Combine neighbor embeddings (mean, sum, max, attention)
**UPDATE**: Incorporate node's own embedding with aggregated neighbors

### Recommendation Scoring

After L layers of GNN, produce recommendation scores:

$$\text{Score}(u, i) = \text{MLP}_{\text{out}}(h_u^{(L)} \odot h_i^{(L)})$$

where ⊙ is element-wise product (captures interaction) and MLP_out is output neural network.

Loss minimizes prediction error on observed interactions:

$$\mathcal{L} = \sum_{(u,i) \in \mathcal{E}} (y_{ui} - \text{Score}(u, i))^2 + \lambda \|W\|^2$$

## GNN Architectures for Recommendations

### Graph Convolutional Networks (GCN)

Simplest GNN: symmetric normalization of adjacency matrix:

$$H^{(l+1)} = \sigma(\tilde{A} H^{(l)} W^{(l)})$$

where $\tilde{A} = D^{-1/2} A D^{-1/2}$ is normalized adjacency.

**Strengths**: Stable training, interpretable neighborhood aggregation
**Weaknesses**: Fixed aggregation (no adaptivity per user)

### GraphSAGE (Graph Sample and Aggregate)

Sample subset of neighbors for scalability:

$$h_v^{(l+1)} = \sigma(W^{(l)} \cdot \text{AGGREGATE}(\{h_u^{(l)} : u \in \text{SAMPLE}(\mathcal{N}(v), k)\}))

Enables mini-batch training on large graphs.

**Strengths**: Scales to large graphs, neighborhood sampling variance reduces overfitting
**Weaknesses**: Sampling introduces variance

### Graph Attention Networks (GAT)

Learn adaptive attention weights on neighbors:

$$\alpha_{uv} = \frac{\exp(\text{LeakyReLU}(a^T [W h_u \| W h_v]))}{\sum_k \exp(\text{LeakyReLU}(a^T [W h_k \| W h_v]))}$$

$$h_v^{(l+1)} = \sigma\left(\sum_{u \in \mathcal{N}(v)} \alpha_{uv} W h_u^{(l)}\right)$$

Attention weights (α) learned jointly with embeddings.

**Strengths**: Different neighbors contribute differently; interpretable attention
**Weaknesses**: More parameters, higher computational cost

## Heterogeneous Graphs for Finance

### Multi-Type Nodes and Edges

Financial networks contain multiple node types:

**Nodes**:
- Users: investors, analysts, portfolio managers
- Assets: stocks, bonds, mutual funds
- Attributes: sectors, countries, credit ratings
- Events: earnings releases, policy decisions

**Edges**:
- User owns asset: weight = position size
- Asset in sector: type = membership
- Asset correlates with asset: weight = correlation
- User reads report about asset: type = research consumption

### Heterogeneous GNN (HetGNN)

Aggregate information separately per edge type:

$$h_v^{(l+1)} = \sigma\left(\sum_{\tau} W_\tau^{(l)} \sum_{u \in \mathcal{N}_\tau(v)} h_u^{(l)}\right)$$

where τ indexes edge types and W_τ are type-specific parameters.

Enables modeling rich relationships without flattening heterogeneity.

## Training and Optimization

### Mini-Batch Sampling

Sample subgraph for each training batch:

1. Sample user u from training set
2. Sample K items u interacted with (positive samples)
3. Sample K unobserved items (negative samples)
4. Extract subgraph connecting sampled nodes
5. Run GNN forward pass on subgraph
6. Compute loss, backpropagate

Enables efficient GPU computation on large graphs.

### Negative Sampling

Pair positive interactions with negative samples:

$$\mathcal{L} = \sum_{(u,i,j): (u,i) \in E, (u,j) \notin E} -\log \sigma(\text{Score}(u,i) - \text{Score}(u,j))$$

BPR (Bayesian Personalized Ranking) loss: optimize relative scores rather than absolute.

### Node Embedding Regularization

Prevent overfitting through L2 regularization:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{ranking}} + \lambda_1 \sum_v \|h_v^{(L)}\|^2 + \lambda_2 \sum_l \|W^{(l)}\|_F^2$$

## Financial Application: Portfolio Recommendation via GNN

### Graph Construction

```
Nodes: Users (investors), Assets (stocks), Sectors, Countries
Edges:
- User-Asset (owns): weight = position value
- Asset-Sector (in): binary
- Asset-Country (traded in): binary
- Asset-Asset (similar): weight = correlation
```

### GNN Architecture

1. **Input**: Node features (sector, country, risk metrics for assets; profile for users)
2. **GNN Layers**: 2-3 GraphSAGE layers aggregate neighborhood information
3. **Output**: Predict whether user would add asset to portfolio

### Training

Train on historical portfolio changes:

- Positive sample: Asset user actually added
- Negative sample: Random asset user didn't add

Optimize BPR loss over mini-batches.

### Recommendation

For target user u:
1. Run GNN forward pass
2. Score all items
3. Rank by score
4. Return top-k unowned items

### Results

- NDCG@10: 0.65 (comparable to collaborative filtering)
- Coverage: 85% (portfolio diversity good)
- Novelty: 70% (recommendations include less-obvious assets)
- Cold-Start: Moderate improvement via graph structure

## Scalability Considerations

### Memory and Computation

GNN training requires storing graph adjacency; storage O(|E|).

Computation per epoch O(|V| × aggregation cost × # layers).

For large graphs (billions of edges), distributed training necessary.

### Distributed GNN Training

Partition graph across machines; use distributed sampling:

1. Graph partitioning: Divide nodes across machines
2. Mini-batch creation: Sample crosses machine boundaries
3. Communication: Exchange embeddings between machines
4. Synchronization: All-reduce aggregation

### Sampling-Based Efficiency

Node-wise sampling reduces computation:

- Sample K neighbors instead of using all
- Reduces per-node aggregation from O(degree) to O(K)
- Enables mini-batch training of GNNs like SGD for neural networks

## Temporal Extensions

### Temporal GNNs

Incorporate time information in graph:

$$h_v^{(l+1)} = \sigma(W^{(l)} \sum_{u \in \mathcal{N}(v)} \exp(-\lambda (t_{\text{now}} - t_{uv})) h_u^{(l)})$$

Recent interactions weighted more heavily; old relationships decay.

### Dynamic Recommendation

Retrain/update GNNs periodically:

- Daily update: Incorporate yesterday's portfolio changes
- Weekly retraining: Full GNN retraining on past month
- Streaming: Incremental embedding updates for real-time

## Comparison: GNN vs Traditional Methods

| Method | Accuracy | Scalability | Interpretability | Cold Start |
|--------|----------|-------------|-----------------|-----------|
| Matrix Factorization | 0.60 | High | Low | Moderate |
| GCN | 0.63 | High | Low | Moderate |
| GraphSAGE | 0.65 | Very High | Low | Moderate |
| Attention (GAT) | 0.67 | Medium | Medium | Moderate |
| Hybrid (GNN+Content) | 0.70 | High | High | Good |

GNNs offer improved accuracy but at cost of interpretability. Hybrid methods balance both.

!!! note "GNN for Recommendations"
    GNNs excel when interaction patterns complex and graph structure important. Best suited for:
    - Large-scale systems (millions of users/items)
    - Rich side information (user/item features)
    - Temporal dynamics important
    - Heterogeneous relationships

