# 29.7.4 Recommendation Systems

## Overview
GNNs model user-item interactions as bipartite graphs, learning embeddings that capture collaborative filtering signals and content features simultaneously.

## Approaches
- **LightGCN**: Simplified GCN on user-item bipartite graph (no feature transformation, no nonlinearity)
- **PinSage**: GraphSAGE on Pinterest item-item graph (billions of nodes)
- **NGCF**: Neural Graph Collaborative Filtering with embedding propagation
- **Knowledge-aware**: Incorporate knowledge graph side information

## Training
BPR (Bayesian Personalized Ranking) loss: $\mathcal{L} = -\sum \log \sigma(s_{ui} - s_{uj})$ where $i$ is positive and $j$ is negative item.

## Financial Applications
- **Fund recommendation**: Recommend investment products based on investor-fund interaction graph
- **News recommendation**: Financial news recommendation using reader-article graph
