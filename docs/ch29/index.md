# Chapter Overview

This chapter covers **Online and Streaming**.

<<<<<<< HEAD
# Reference

[Data Streams: Algorithms and Applications (Muthukrishnan)](https://www.cs.rutgers.edu/~muthu/stream-1-1.ps)
=======
Graph Neural Networks (GNNs) extend deep learning to non-Euclidean data structures -- graphs -- enabling powerful representation learning on relational data. From social networks and molecular structures to financial transaction networks and knowledge bases, graphs are ubiquitous in real-world applications. This chapter provides a comprehensive treatment of GNN theory, architectures, and applications.

## Contents

### 29.1 Graph Foundations

- [Graph Basics](foundations/graph_basics.md) -- Formal definitions of graphs, directed and undirected variants, and foundational graph terminology
- [Graph Representations](foundations/representations.md) -- Data structures for representing graphs including adjacency lists, matrices, and edge lists with computational trade-offs
- [Adjacency Matrix](foundations/adjacency_matrix.md) -- Deep dive into adjacency matrices, derived matrices, and their role in spectral graph theory and GNN formulations
- [Node and Edge Features](foundations/node_edge_features.md) -- Encoding node attributes and edge properties as feature matrices for GNN input
- [Graph Properties](foundations/graph_properties.md) -- Key topological and structural properties including connectivity, distance measures, and clustering
- [PyTorch Geometric Basics](foundations/pyg_basics.md) -- Introduction to PyG's Data objects, data loading, and graph manipulation utilities

### 29.2 Message Passing

- [Message Passing Framework](message_passing/framework.md) -- The unified MPNN paradigm: message computation, aggregation, and update steps
- [Aggregation Functions](message_passing/aggregation.md) -- Permutation-invariant aggregation operators (sum, mean, max) and their expressiveness properties
- [Update Functions](message_passing/update.md) -- Combining self-information with aggregated neighborhood messages to produce new node embeddings
- [MPNN](message_passing/mpnn.md) -- The Message Passing Neural Network formulation by Gilmer et al. unifying GNN architectures

### 29.3 Graph Convolutions

- [Spectral Graph Theory](convolutions/spectral_theory.md) -- Graph Laplacian eigendecomposition, graph signals, and the mathematical foundation for spectral convolutions
- [Graph Fourier Transform](convolutions/graph_fourier.md) -- Generalizing the Fourier transform to graph-structured data using Laplacian eigenvectors
- [ChebNet](convolutions/chebnet.md) -- Efficient spectral filtering via Chebyshev polynomial approximation of graph Laplacian filters
- [Graph Convolutional Network (GCN)](convolutions/gcn.md) -- Kipf and Welling's first-order spectral convolution with symmetric normalization
- [GraphSAGE](convolutions/graphsage.md) -- Inductive graph learning through neighborhood sampling and aggregation with multiple aggregator variants
- [Graph Attention Network (GAT)](convolutions/gat.md) -- Attention-based GNNs that learn adaptive, data-dependent neighbor importance weights
- [Graph Isomorphism Network (GIN)](convolutions/gin.md) -- Maximally expressive message passing GNN provably as powerful as the 1-WL isomorphism test

### 29.4 Advanced GNN Methods

- [Deep GNNs](advanced/deep_gnns.md) -- Techniques for building deeper GNNs including residual connections and normalization to combat over-smoothing
- [Over-Smoothing](advanced/over_smoothing.md) -- Analysis of node representation convergence in deep GNNs and mitigation strategies
- [Jumping Knowledge Networks](advanced/jumping_knowledge.md) -- Adaptive multi-layer aggregation enabling each node to select its optimal receptive field depth
- [Graph Transformers](advanced/graph_transformers.md) -- Applying transformer self-attention to graphs for global information exchange (Graphormer, GPS)
- [Heterogeneous Graphs](advanced/heterogeneous.md) -- GNNs for graphs with multiple node and edge types using RGCN and HAN architectures
- [Temporal Graphs](advanced/temporal.md) -- Dynamic graph neural networks for evolving structures with discrete-time and continuous-time approaches
- [Hypergraphs](advanced/hypergraphs.md) -- Neural networks on hypergraphs where hyperedges connect arbitrary numbers of nodes simultaneously

### 29.5 Graph-Level Tasks

- [Graph Classification](graph_tasks/graph_classification.md) -- Assigning labels to entire graphs using GNN layers, readout functions, and MLP classifiers
- [Graph Regression](graph_tasks/graph_regression.md) -- Predicting continuous values for whole graphs with applications in quantum chemistry and molecular solubility
- [Graph Pooling](graph_tasks/pooling.md) -- Flat pooling methods (sum, mean, max, attention-weighted) for compressing node embeddings into graph representations
- [Hierarchical Pooling](graph_tasks/hierarchical_pooling.md) -- Progressive graph coarsening with DiffPool, TopKPool, and SAGPool for multi-resolution representations
- [Set2Set](graph_tasks/set2set.md) -- Attention-based LSTM readout producing order-invariant graph representations more expressive than simple pooling

### 29.6 Node and Link Tasks

- [Node Classification](node_link/node_classification.md) -- Semi-supervised node labeling using graph structure and features in transductive and inductive settings
- [Link Prediction](node_link/link_prediction.md) -- Predicting missing or future edges using score-based methods and negative sampling
- [Node Embedding](node_link/node_embedding.md) -- Unsupervised node representation learning with DeepWalk, Node2Vec, LINE, and contrastive GNN methods
- [Community Detection](node_link/community_detection.md) -- Identifying densely connected node groups using GNN-based clustering and modularity optimization

### 29.7 GNN Applications

- [Molecular Property Prediction](applications/molecular.md) -- Predicting chemical properties from molecular graphs with atom and bond features
- [Drug Discovery](applications/drug_discovery.md) -- GNNs for virtual screening, drug-drug interaction prediction, and ADMET property estimation
- [Social Network Analysis](applications/social_networks.md) -- Community detection, influence prediction, cascade modeling, and bot detection in social graphs
- [Recommendation Systems](applications/recommendation.md) -- GNN-based collaborative filtering on user-item bipartite graphs with LightGCN and PinSage
- [Knowledge Graphs](applications/knowledge_graphs.md) -- Entity and relation embedding with R-GCN, CompGCN, and scoring functions for link prediction
- [Financial Networks](applications/financial_networks.md) -- Portfolio optimization, systemic risk propagation, contagion modeling, and fraud detection on financial graphs
>>>>>>> 96f31bd (...)
