# Bipartite Graph Modeling for Recommendations

## Introduction

Many recommendation problems naturally decompose into bipartite graphs: two distinct node types (users and items) with edges representing interactions (ratings, purchases, clicks). Bipartite graph representation provides structured way to model user-item relationships and enables graph algorithms to discover patterns in interaction patterns. Beyond traditional edges representing single interactions, bipartite graphs can encode rich relationship structures: edge weights (interaction strength), edge attributes (timestamp, context), and additional constraints.

Bipartite graph representation unifies disparate recommendation problems under single framework: user-product recommendations (products and suppliers), user-research recommendations (analysts and reports), social recommendations (users and content). Graph algorithms—random walks, spectral clustering, belief propagation—provide principled methods to compute relevance scores and learn recommendations from graph structure alone.

This section develops bipartite graph theory for recommendations, explores algorithms exploiting graph structure, and demonstrates practical financial applications.

## Key Concepts

### Bipartite Graph Components
- **User Nodes**: U = {u₁, u₂, ..., u_m} representing users/investors
- **Item Nodes**: I = {i₁, i₂, ..., i_n} representing products/assets
- **Edges**: E ⊆ U × I representing interactions (only between different node types)
- **Edge Weights**: w(u,i) ∈ ℝ+ representing interaction strength

### Graph Properties
- **Degree**: Number of edges incident to node
- **Connectivity**: Graph path structure determining distance between nodes
- **Clustering**: Tendency of nodes to form connected clusters
- **Homophily**: Similar users tend to rate similar items

## Mathematical Framework

### Bipartite Graph Representation

Bipartite graph G = (U, I, E) with adjacency matrix:

$$A = \begin{pmatrix} 0 & B \\ B^T & 0 \end{pmatrix}$$

where $B \in \mathbb{R}^{|U| \times |I|}$ is user-item interaction matrix:

$$B_{ui} = \begin{cases}
w(u, i) & \text{if } (u, i) \in E \\
0 & \text{otherwise}
\end{cases}$$

### Graph Laplacian

Normalized Laplacian captures graph structure:

$$L = I - D^{-1/2} A D^{-1/2}$$

where D is degree matrix. Eigenvectors of L reveal graph structure (clusters, bottlenecks).

### User-Item Similarity via Graph Distance

Shortest path distance between user u and item i via graph:

$$d(u, i) = \text{min} \|p\|$$

where p is path from u to i. Shorter distance indicates greater relevance.

## Random Walk Based Recommendations

### Personalized PageRank

Random walk from user u exploring the graph:

$$\text{PageRank}_u(i) = (1-\alpha) \sum_j \frac{\text{PageRank}_u(j)}{d_j} + \alpha \delta(u_{\text{start}} = u)$$

Restart probability α ensures walker returns to starting user. High PageRank(i) indicates reachable from user u, suggesting relevance.

### Algorithm Pseudocode

```
Initialize: rank[u] = 1.0, rank[all others] = 0
For T iterations:
  new_rank = (1-α) × A × rank / degree + α × initial_rank
  rank = new_rank
Return: rank[items] as relevance scores
```

Computational complexity: O(T|E|) for T iterations.

## Spectral Methods for Graph Recommendations

### Spectral Clustering

Cluster users/items using graph Laplacian eigenvectors:

$$L v_k = \lambda_k v_k$$

Use lowest-k eigenvectors as features; cluster using k-means.

Interpretation: Users in same cluster have similar interaction patterns; recommend items popular in user's cluster.

### Similarity via Spectral Features

Define similarity between users/items using spectral representations:

$$\text{Sim}(u_1, u_2) = \langle v_1, v_2 \rangle$$

where v are spectral features (eigenvector components).

## Recommendation Algorithms on Bipartite Graphs

### Preferential Attachment (Popularity)

Recommend items that neighbors like:

$$\text{Score}(u, i) = \sum_{j: (u,j) \in E} w(u, j) \times \frac{1}{\text{degree}(i)}$$

Simple but effective baseline; popularity counter-balanced by inverse item degree.

### Common Neighbors

Recommend items users with common interests like:

$$\text{Score}(u, i) = |\{j: (u, j) \in E \text{ and } (j, i) \in E\}|$$

Items with many mutual-friend connections suggested.

### Hybrid Path-Based Scoring

Combine multiple path lengths:

$$\text{Score}(u, i) = \alpha_1 \cdot \text{length1}_{\text{paths}} + \alpha_2 \cdot \text{length2}_{\text{paths}} + \alpha_3 \cdot \text{length3}_{\text{paths}}$$

Weights α control relative importance of different path lengths.

## Bipartite Graph Construction for Finance

### User Nodes

**Explicit Users**:
- Individual investors
- Institutional asset managers
- Investment advisors

**Implicit Users** (treating as "users"):
- Portfolio positions
- Investment strategies
- Market regimes

### Item Nodes

**Financial Products**:
- Stocks
- Bonds
- Mutual funds
- ETFs

**Alternative Items**:
- Asset classes
- Trading strategies
- Research themes

### Interaction Edges

**Explicit Interactions**:
- Purchase: Edge with weight 1.0
- Holdings: Edge weight = position size / total portfolio
- Ratings: Edge weight = explicit rating / 5

**Implicit Interactions**:
- Views: Edge weight = view count
- Trading volume: Edge weight = volume / average volume
- Search interest: Edge weight = search frequency

### Edge Attributes

Additional edge information:

- **Timestamp**: When interaction occurred (enable temporal analysis)
- **Context**: Market regime, economic conditions
- **Duration**: How long held or viewed item
- **Performance**: Whether interaction profitable

## Practical Implementation

### Bipartite Graph Construction

```python
import networkx as nx

G = nx.Graph()

# Add user and item nodes
for u in users: G.add_node(u, node_type='user')
for i in items: G.add_node(i, node_type='item')

# Add edges from interactions
for (u, i, weight) in interactions:
    G.add_edge(u, i, weight=weight)
```

### PageRank-Based Recommendations

```python
from networkx.algorithms import pagerank

# Compute PageRank starting from user u
pr = pagerank(G, personalization={u: 1.0})

# Recommend top items by PageRank
recommendations = sorted(
    [(i, pr[i]) for i in items],
    key=lambda x: x[1],
    reverse=True
)[:k]
```

### Spectral Clustering for Discovery

```python
import scipy.sparse.linalg

# Compute Laplacian eigenvectors
L = nx.laplacian_matrix(G)
eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(L, k=5)

# Cluster nodes using eigenvectors
clusters = kmeans(eigenvectors, n_clusters=5)

# Recommend items in user's cluster
user_cluster = clusters[u]
recommendations = [i for i in items if clusters[i] == user_cluster]
```

## Advantages and Limitations

### Advantages

- **Interpretability**: Graph structure reveals recommendation logic
- **Efficiency**: Sparse graphs enable fast computation
- **Generality**: Applies to any bipartite interaction structure
- **Integration**: Easily combine multiple edge types

### Limitations

- **Cold Start**: New nodes (users/items) have few/no edges
- **Scalability**: Computing PageRank on massive graphs expensive
- **Sparsity**: Incomplete interaction data limits path discovery
- **Dynamics**: Graph structure changes over time; requires recomputation

## Temporal Extensions

### Time-Weighted Edges

Decay edge weights by recency:

$$w_{\text{decayed}}(u, i, t) = w(u, i) \times \exp(-\lambda (t_{\text{now}} - t_{\text{interaction}}))$$

Recent interactions more important than distant past.

### Temporal Random Walks

Random walk respecting temporal order:

$$\text{Next step} = \{j: \text{interaction}_j \text{ occurred after current edge}\}$$

Captures causal structure; if user u bought stocks after reading analyst report, paths credit report.

## Case Study: Analyst Research Recommendation

### Graph Construction

**User Nodes**: 200 equity analysts

**Item Nodes**: 10,000 research reports

**Edges**: Analyst reads report
- Weight: 1.0 if read > 50% of document
- Weight: 0.5 if read < 50%
- Timestamp: When read

**Attributes**: 
- Report publication date
- Stock tickers mentioned
- Sector focus

### Recommendation Algorithm

**PageRank with Personalization**:
1. Start from analyst u
2. Random walk through graph
3. Accumulate visits to reports
4. Recommend high-score unread reports

**Result**: 
- Analyst focused on tech reads tech research with high probability
- Collaborative signal: reports read by similar analysts suggested
- Novelty: Some probability of exploring other sectors

### Evaluation

- CTR on recommendations: 15% (analyst clicks recommended report)
- Read rate: 65% of clicked reports fully read
- Engagement: Reports recommended integrated into investment decisions 20% of time

!!! note "Bipartite Graphs"
    Bipartite graphs provide principled framework for representing user-item interactions. Graph algorithms (PageRank, spectral clustering) enable scalable recommendations without explicit embedding learning. Best suited for problems with explicit interaction data and interpretability importance.

