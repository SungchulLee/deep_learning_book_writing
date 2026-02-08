# 29.1.6 PyTorch Geometric Basics

## Introduction

**PyTorch Geometric (PyG)** is the leading library for deep learning on graphs and other irregular structures. Built on PyTorch, it provides efficient data handling, a comprehensive collection of GNN layers, and utilities for graph manipulation. This section covers the essential PyG concepts needed throughout this chapter.

## Installation

```bash
pip install torch-geometric
# Or with specific CUDA version:
pip install torch-geometric -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

## The `Data` Object

The core data structure in PyG is `torch_geometric.data.Data`, which represents a single graph:

```python
from torch_geometric.data import Data

data = Data(
    x=x,              # [num_nodes, num_node_features]
    edge_index=edge_index,  # [2, num_edges] COO format
    edge_attr=edge_attr,    # [num_edges, num_edge_features]
    y=y,              # Target labels
)
```

### Key Attributes
- `data.x`: Node feature matrix
- `data.edge_index`: Edge connectivity in COO format
- `data.edge_attr`: Edge feature matrix
- `data.y`: Labels (node-level, edge-level, or graph-level)
- `data.pos`: Node position matrix (for spatial data)
- `data.num_nodes`: Number of nodes
- `data.num_edges`: Number of edges
- `data.num_node_features`: Dimensionality of node features

### Important Convention
`edge_index` uses COO format with shape `[2, num_edges]`:
- `edge_index[0]`: Source nodes
- `edge_index[1]`: Target nodes

For undirected graphs, each edge is stored twice (both directions).

## Batching

PyG batches multiple graphs into a single disconnected graph using `torch_geometric.loader.DataLoader`:

```python
from torch_geometric.loader import DataLoader

loader = DataLoader(dataset, batch_size=32, shuffle=True)
for batch in loader:
    # batch.x: concatenated node features
    # batch.edge_index: adjusted edge indices
    # batch.batch: mapping from node to graph index
    out = model(batch)
```

The `batch` vector maps each node to its corresponding graph in the batch, enabling graph-level operations like global pooling.

## Transforms

PyG provides transforms for data preprocessing:

```python
import torch_geometric.transforms as T

# Chain multiple transforms
transform = T.Compose([
    T.NormalizeFeatures(),
    T.AddSelfLoops(),
    T.ToUndirected(),
])

data = transform(data)
```

Common transforms:
- `NormalizeFeatures()`: Row-normalize node features
- `AddSelfLoops()`: Add self-loops to adjacency
- `ToUndirected()`: Convert directed to undirected
- `GCNNorm()`: Apply GCN normalization
- `RandomNodeSplit()`: Create train/val/test masks

## Built-in Datasets

PyG includes many standard benchmarks:

```python
from torch_geometric.datasets import Planetoid, TUDataset

# Node classification
cora = Planetoid(root='data/', name='Cora')

# Graph classification
proteins = TUDataset(root='data/', name='PROTEINS')
```

### Popular Datasets
- **Cora, CiteSeer, PubMed**: Citation networks (node classification)
- **PROTEINS, MUTAG, NCI1**: Molecular datasets (graph classification)
- **QM9**: Molecular property prediction (graph regression)
- **OGB**: Open Graph Benchmark (large-scale benchmarks)

## Message Passing Base Class

All PyG GNN layers inherit from `MessagePassing`:

```python
from torch_geometric.nn import MessagePassing

class MyGNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # Aggregation method
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return x_j  # Messages from neighbors

    def update(self, aggr_out):
        return self.lin(aggr_out)  # Update node embeddings
```

## Utility Functions

```python
from torch_geometric.utils import (
    to_networkx,          # Convert to NetworkX
    from_networkx,        # Convert from NetworkX
    degree,               # Compute node degrees
    add_self_loops,       # Add self-loops
    remove_self_loops,    # Remove self-loops
    to_dense_adj,         # Convert to dense adjacency
    to_dense_batch,       # Convert batched sparse to dense
    contains_self_loops,  # Check for self-loops
    is_undirected,        # Check if undirected
    sort_edge_index,      # Sort edges
    coalesce,             # Remove duplicate edges
)
```

## Summary

PyTorch Geometric provides an efficient, flexible framework for implementing GNNs. Its COO-based sparse representation, automatic batching, and rich library of layers and utilities make it the standard tool for graph deep learning research and applications.
