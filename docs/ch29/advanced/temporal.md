# Temporal Graph Neural Networks

Real-world networks are rarely static: social connections form and dissolve, financial transactions occur at irregular intervals, and communication patterns shift over time. **Temporal graphs** (also called dynamic graphs) capture this evolution by associating timestamps with nodes, edges, or features. Temporal graph neural networks (Temporal GNNs) extend static GNN architectures to learn from these evolving structures, combining spatial message passing with temporal modeling.

## Types of Temporal Graphs

**Discrete-time dynamic graphs (DTDG)**: The graph is observed as a sequence of snapshots $G_1, G_2, \ldots, G_T$ at regular time intervals. Each snapshot is a complete static graph.

**Continuous-time dynamic graphs (CTDG)**: Events (edge additions, deletions, feature updates) occur at arbitrary timestamps. The graph evolves continuously rather than in fixed steps.

## Approaches

### Snapshot-Based Methods

Apply a static GNN independently to each snapshot, then combine the resulting node embeddings over time using an RNN or transformer:

$$
\mathbf{Z}_t = \text{GNN}(G_t), \quad \mathbf{H}_t = \text{RNN}(\mathbf{Z}_t, \mathbf{H}_{t-1})
$$

This approach is simple but treats each snapshot independently during the spatial aggregation step and requires discrete time steps.

### Temporal Message Passing

Extend standard message passing by including temporal information in the messages. Each message carries the timestamp of the interaction, and temporal encodings (analogous to positional encodings in transformers) enable the model to distinguish recent from distant interactions.

### EvolveGCN

Rather than evolving node embeddings over time, EvolveGCN evolves the GNN's **weight matrices** using an RNN:

$$
\mathbf{W}_t = \text{GRU}(\mathbf{H}_t, \mathbf{W}_{t-1})
$$

This allows the convolutional filters themselves to adapt as the graph structure changes.

### TGAT (Temporal Graph Attention)

TGAT applies the attention mechanism with time-aware positional encodings. A functional encoding $\Phi(t) = \cos(\omega t + \phi)$ injects temporal information into the attention scores, allowing the model to weight recent interactions more heavily.

### TGN (Temporal Graph Network)

TGN maintains a per-node **memory module** that is updated with each interaction. The architecture combines three components: a message function that summarizes each event, a memory updater (GRU or LSTM) that incorporates messages into node memory, and an embedding module that produces the final node representation from memory and graph structure.

## Financial Applications

- **Portfolio rebalancing**: Track evolving asset correlation networks to detect when portfolio weights need adjustment
- **Fraud detection**: Identify anomalous temporal transaction patterns that static analysis would miss
- **Market regime detection**: Monitor changes in network structure (e.g., sectoral correlations breaking down) as early signals of regime shifts

!!! note "Choosing Between DTDG and CTDG"
    If your data arrives at regular intervals (e.g., daily closing prices), snapshot-based methods are natural. If events occur at irregular timestamps (e.g., trade-level data), continuous-time methods avoid the information loss of binning into fixed windows.

## References

[Kazemi et al. -- Representation Learning for Dynamic Graphs: A Survey (JMLR 2020)](https://jmlr.org/papers/v21/19-447.html)
