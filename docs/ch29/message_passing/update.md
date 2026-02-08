# 29.2.3 Update Functions

## Introduction

The **update function** combines a node's previous representation with the aggregated messages from its neighbors to produce the new node embedding. This step determines how self-information is preserved and integrated with neighborhood information across layers.

## Formal Definition

Given node $v$ with current representation $\mathbf{h}_v^{(l-1)}$ and aggregated neighborhood message $\mathbf{m}_v^{(l)}$:

$$\mathbf{h}_v^{(l)} = \text{UPD}\left(\mathbf{h}_v^{(l-1)}, \mathbf{m}_v^{(l)}\right)$$

## Update Function Variants

### Simple Replacement
$$\mathbf{h}_v^{(l)} = \sigma\left(\mathbf{m}_v^{(l)}\right)$$
Self-information is only preserved if self-loops are included in the graph. Used in basic GCN.

### Concatenation + Linear
$$\mathbf{h}_v^{(l)} = \sigma\left(W \cdot [\mathbf{h}_v^{(l-1)} \| \mathbf{m}_v^{(l)}]\right)$$
Explicitly separates self and neighbor information. Used in GraphSAGE.

### Residual Connection
$$\mathbf{h}_v^{(l)} = \mathbf{h}_v^{(l-1)} + \sigma\left(W \cdot \mathbf{m}_v^{(l)}\right)$$
Preserves original features, enables training deeper networks. Analogous to ResNet.

### GRU-based Update
$$\mathbf{h}_v^{(l)} = \text{GRU}\left(\mathbf{m}_v^{(l)}, \mathbf{h}_v^{(l-1)}\right)$$
Uses gating mechanisms to control information flow. The GRU decides how much old information to retain.

### LSTM-based Update
$$\mathbf{h}_v^{(l)}, \mathbf{c}_v^{(l)} = \text{LSTM}\left(\mathbf{m}_v^{(l)}, (\mathbf{h}_v^{(l-1)}, \mathbf{c}_v^{(l-1)})\right)$$
Maintains a cell state for long-range memory across layers.

### MLP-based Update
$$\mathbf{h}_v^{(l)} = \text{MLP}\left([\mathbf{h}_v^{(l-1)} \| \mathbf{m}_v^{(l)}]\right)$$
Maximum flexibility but more parameters. Used in GIN and other expressive architectures.

### Weighted Sum (with learnable $\epsilon$)
$$\mathbf{h}_v^{(l)} = \text{MLP}\left((1 + \epsilon) \cdot \mathbf{h}_v^{(l-1)} + \mathbf{m}_v^{(l)}\right)$$
The GIN update: $\epsilon$ controls the relative weight of self vs. neighbors.

## Design Considerations

### Self-Information Preservation
Without explicit self-connection, a node's own features may be diluted after multiple layers. Solutions:
1. Add self-loops to the graph
2. Use concatenation-based updates
3. Use residual connections

### Depth and Over-Smoothing
As depth increases, representations tend to converge. Update functions that preserve individuality help:
- Residual connections slow down smoothing
- GRU gating selectively retains information
- Skip connections from early layers (Jumping Knowledge)

### Nonlinearity
The activation function $\sigma$ (ReLU, ELU, GELU) applied in the update introduces nonlinearity, enabling the network to learn complex functions.

## Comparison

| Update Type | Self-Info | Depth-Friendly | Parameters | Expressiveness |
|-------------|-----------|----------------|------------|---------------|
| Replace | Via self-loops | Low | Low | Low |
| Concat+Linear | Explicit | Medium | Medium | Medium |
| Residual | Strong | High | Low | Medium |
| GRU | Gated | High | High | High |
| MLP | Explicit | Medium | High | High |
| GIN ($\epsilon$) | Weighted | Medium | Medium | High |

## Summary

The update function is the final step in each message passing iteration. The choice of update function balances expressiveness, computational cost, and trainability. For deep GNNs, residual connections or gated updates are recommended to combat over-smoothing and gradient issues.
