# 29.4.6 Temporal Graphs

## Introduction

**Temporal graphs** (dynamic graphs) evolve over time with changing nodes, edges, and features. They capture dynamic relationships like evolving financial networks, social interactions, and communication patterns.

## Types

**Discrete-time dynamic graphs (DTDG)**: Graph snapshots at regular intervals $G_1, G_2, \ldots, G_T$.

**Continuous-time dynamic graphs (CTDG)**: Events (edge additions/deletions) occur at continuous timestamps.

## Approaches

### Snapshot-based
Apply static GNN to each snapshot, then combine with RNN/transformer over time:
$$\mathbf{Z}_t = \text{GNN}(G_t), \quad \mathbf{H}_t = \text{RNN}(\mathbf{Z}_t, \mathbf{H}_{t-1})$$

### Temporal Message Passing
Extend message passing with temporal information: messages carry timestamps and temporal encodings.

### EvolveGCN
Evolve the GNN weights over time using an RNN:
$$W_t = \text{GRU}(H_t, W_{t-1})$$

### TGAT (Temporal Graph Attention)
Attention with time-aware encoding: $\Phi(t) = \cos(\omega t + \phi)$ as temporal positional encoding.

### TGN (Temporal Graph Network)
Maintains node memory that is updated with each interaction, combining memory, embedding, and message modules.

## Financial Applications
- **Portfolio rebalancing**: Evolving correlation networks
- **Fraud detection**: Temporal transaction patterns
- **Market regime detection**: Changing network structure signals regime shifts

## Summary

Temporal GNNs capture the dynamic nature of real-world networks. Combining static GNN architectures with temporal modeling (RNN, attention, memory) enables learning from evolving graph structures.
