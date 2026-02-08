# 29.1.4 Node and Edge Features

## Introduction

Real-world graphs carry rich information beyond topology. **Node features** encode attributes of individual entities, while **edge features** encode properties of relationships. Feature engineering for graphs is critical for GNN performance, as the model's input representation directly impacts what patterns can be learned.

## Node Features

### Node Feature Matrix
The node feature matrix $X \in \mathbb{R}^{n \times d}$ assigns a $d$-dimensional feature vector $\mathbf{x}_v \in \mathbb{R}^d$ to each node $v$.

$$X = \begin{bmatrix} \mathbf{x}_1^T \\ \mathbf{x}_2^T \\ \vdots \\ \mathbf{x}_n^T \end{bmatrix}$$

### Types of Node Features

**Categorical Features**: Encoded via one-hot or learned embeddings.
- Example: Atom type in molecular graphs (C, N, O, ...)
- Example: User type in social networks

**Continuous Features**: Real-valued attributes.
- Example: Stock returns, volatility, market cap in financial graphs
- Example: Molecular properties (electronegativity, atomic mass)

**Structural Features**: Computed from graph topology.
- Node degree: $d(v) = |\mathcal{N}(v)|$
- Clustering coefficient: $C(v) = \frac{2|\{(u,w) : u,w \in \mathcal{N}(v), (u,w) \in E\}|}{d(v)(d(v)-1)}$
- PageRank, betweenness centrality, eigenvector centrality
- Local motif counts (triangles, cycles)

**Positional Features**: Encode node position in the graph.
- Random walk encodings
- Laplacian positional encodings (eigenvectors of the Laplacian)

## Edge Features

### Edge Feature Matrix
The edge feature matrix $E_{attr} \in \mathbb{R}^{m \times d_e}$ assigns a $d_e$-dimensional feature vector to each edge.

### Types of Edge Features

**Relationship Properties**:
- Bond type in molecular graphs (single, double, triple, aromatic)
- Transaction amount in financial graphs
- Distance or similarity between connected entities

**Derived Features**:
- Edge betweenness centrality
- Common neighbors between endpoints
- Jaccard similarity of neighborhoods

## Feature Normalization

For GNNs, feature normalization is important for training stability:

- **Standard normalization**: $\hat{x}_i = (x_i - \mu) / \sigma$
- **Min-max normalization**: $\hat{x}_i = (x_i - x_{min}) / (x_{max} - x_{min})$
- **Layer normalization**: Applied within GNN layers

## Graph-Level Features

For tasks requiring a single representation of the entire graph:

- **Global features**: Number of nodes, edges, density, diameter
- **Aggregated node features**: Mean, sum, max pooling of node features
- **Spectral features**: Eigenvalue statistics of the Laplacian

## Quantitative Finance Features

### Node Features for Financial Graphs
- **Asset nodes**: Returns (multi-horizon), volatility, sector encoding, market cap, P/E ratio, momentum signals
- **Institution nodes**: AUM, leverage ratio, portfolio concentration
- **Account nodes**: Transaction volume, account age, risk score

### Edge Features for Financial Graphs
- **Correlation edges**: Rolling correlation, partial correlation
- **Transaction edges**: Amount, frequency, time of day
- **Supply chain edges**: Revenue dependency percentage, contract duration

## Summary

Feature engineering bridges raw data and GNN models. The choice of node and edge features determines the expressiveness of the learned representations. In financial applications, combining topological features with domain-specific attributes creates powerful input representations for downstream tasks.
