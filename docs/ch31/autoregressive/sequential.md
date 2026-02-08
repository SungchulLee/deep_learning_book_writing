# Sequential Graph Generation

## Overview

Autoregressive graph generation decomposes the joint distribution $p_\theta(\mathcal{G})$ into a sequence of conditional distributions, generating one element at a time. This paradigm naturally handles variable-sized graphs and avoids the combinatorial challenge of producing an entire adjacency matrix simultaneously. The core idea is to impose an ordering on graph construction actions and factorize the likelihood using the chain rule of probability.

## Autoregressive Factorization

Given a graph $\mathcal{G} = (\mathbf{A}, \mathbf{X})$ with $n$ nodes, we define a generation sequence $\sigma = (a_1, a_2, \ldots, a_T)$ where each action $a_t$ adds a node, an edge, or terminates generation. The joint probability factorizes as:

$$
p_\theta(\mathcal{G}) = \sum_{\sigma \in \Sigma(\mathcal{G})} \prod_{t=1}^{T} p_\theta(a_t \mid a_{1:t-1})
$$

where $\Sigma(\mathcal{G})$ is the set of all valid generation sequences producing $\mathcal{G}$. Since marginalizing over orderings is intractable ($|\Sigma(\mathcal{G})|$ can be exponential), practical methods fix a canonical ordering $\sigma^*$ and optimize:

$$
\log p_\theta(\mathcal{G}) \geq \log \prod_{t=1}^{T} p_\theta(a_t^* \mid a_{1:t-1}^*)
$$

This lower bound becomes tight when the ordering is deterministic given the graph.

## Node-Level Decomposition

The most common decomposition adds one node at a time. At step $t$, the model:

1. **Decides whether to add a new node** or terminate: $p_\theta(\text{stop} \mid \mathcal{G}_{<t})$
2. **Generates node features**: $p_\theta(\mathbf{x}_t \mid \mathcal{G}_{<t})$
3. **Generates edges** to all existing nodes: $p_\theta(\mathbf{a}_t \mid \mathbf{x}_t, \mathcal{G}_{<t})$

where $\mathcal{G}_{<t}$ denotes the partially constructed graph and $\mathbf{a}_t \in \{0,1\}^{t-1}$ is the edge vector connecting node $t$ to nodes $1, \ldots, t-1$.

The log-likelihood for a graph with $n$ nodes under node ordering $\pi$ is:

$$
\log p_\theta(\mathcal{G} \mid \pi) = \sum_{t=1}^{n} \left[ \log p_\theta(\mathbf{x}_{\pi(t)} \mid \mathcal{G}_{<t}) + \sum_{s=1}^{t-1} \log p_\theta(A_{\pi(t),\pi(s)} \mid \mathbf{x}_{\pi(t)}, \mathcal{G}_{<t}) \right] + \log p_\theta(\text{stop} \mid \mathcal{G})
$$

## Edge-Level Decomposition

An alternative decomposes generation at the edge level, iterating over all possible node pairs:

$$
p_\theta(\mathbf{A} \mid \mathbf{X}) = \prod_{i=1}^{n} \prod_{j=1}^{i-1} p_\theta(A_{ij} \mid A_{<(i,j)}, \mathbf{X})
$$

where the product follows a raster-scan ordering of the upper triangular adjacency matrix. Each edge decision is a Bernoulli variable conditioned on all previously decided edges.

## Ordering Strategies

The choice of node ordering $\pi$ fundamentally affects generation quality and computational cost.

**BFS ordering** traverses the graph breadth-first from the highest-degree node. This concentrates edge decisions near the diagonal of the reordered adjacency matrix, enabling truncated context windows. The key property is that under BFS ordering, an edge $(i, j)$ with $i < j$ satisfies $j - i \leq B$ where $B$ is the BFS bandwidth. This means node $j$ only needs to consider connections to its $B$ most recent predecessors rather than all $j-1$ previous nodes.

**DFS ordering** produces similar locality but with different structural characteristics — depth-first traversal generates long chains before branching, which can be advantageous for tree-like graphs.

**Random ordering** with augmentation trains the model on multiple random orderings of each graph, approximating the marginalization over $\Sigma(\mathcal{G})$ through Monte Carlo sampling:

$$
\log p_\theta(\mathcal{G}) \approx \log \frac{1}{K} \sum_{k=1}^{K} \prod_{t=1}^{T} p_\theta(a_t^{(k)} \mid a_{1:t-1}^{(k)})
$$

## State Representation

At each generation step $t$, the model must encode the partially constructed graph $\mathcal{G}_{<t}$ into a fixed-dimensional state vector. Two approaches dominate:

**Recurrent encoding.** A GRU or LSTM maintains a hidden state $\mathbf{h}_t$ updated at each step:

$$
\mathbf{h}_t = \text{GRU}(\mathbf{h}_{t-1}, \mathbf{e}_t)
$$

where $\mathbf{e}_t$ encodes the action taken at step $t$ (node features and edge decisions). The hidden state implicitly summarizes the entire generation history.

**GNN encoding.** Apply a graph neural network to $\mathcal{G}_{<t}$ at each step to produce node embeddings, then aggregate to a graph-level representation:

$$
\mathbf{h}_t = \text{READOUT}(\text{GNN}(\mathcal{G}_{<t}))
$$

This is more expressive but more expensive, as the GNN must be re-applied at each step with a growing graph.

## Training Procedure

Training uses teacher forcing: at each step $t$, the model receives the ground-truth partial graph $\mathcal{G}_{<t}^*$ rather than its own predictions. The loss is the negative log-likelihood summed over all steps:

$$
\mathcal{L} = -\sum_{t=1}^{T} \log p_\theta(a_t^* \mid \mathcal{G}_{<t}^*)
$$

For edge predictions, this reduces to binary cross-entropy over each edge decision:

$$
\mathcal{L}_{\text{edge}} = -\sum_{t=2}^{n} \sum_{s=1}^{t-1} \left[ A_{ts}^* \log \hat{p}_{ts} + (1 - A_{ts}^*) \log (1 - \hat{p}_{ts}) \right]
$$

where $\hat{p}_{ts} = p_\theta(A_{ts} = 1 \mid \mathcal{G}_{<t}^*)$.

## Exposure Bias and Mitigation

Teacher forcing creates a train-test mismatch: during generation, the model conditions on its own (potentially erroneous) predictions rather than ground truth. This **exposure bias** can cause errors to compound over long sequences.

Mitigation strategies include:

- **Scheduled sampling**: With probability $\epsilon_t$ (annealed during training), use the model's own prediction instead of ground truth at step $t$
- **Sequence-level objectives**: Train with REINFORCE using graph-level metrics as rewards
- **Curriculum learning**: Start with small graphs and gradually increase size during training

## Finance Application: Sequential Network Construction

In financial network generation, sequential construction mirrors real-world network formation. Banks enter the interbank market over time, establishing lending relationships incrementally. The autoregressive framework can model this temporal evolution:

$$
p(\mathcal{G}_T) = \prod_{t=1}^{T} p(\text{bank}_t \text{ joins}) \cdot \prod_{s < t} p(\text{lends}(t, s) \mid \text{attributes}, \mathcal{G}_{<t})
$$

This captures preferential attachment (new banks preferentially connect to well-connected incumbents) and homophily (banks with similar attributes form connections), both of which are empirically observed in financial networks.

## Implementation: Sequential Graph Generator

```python
"""
Sequential (autoregressive) graph generation framework.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from collections import deque


class SequentialGraphGenerator(nn.Module):
    """
    Node-by-node autoregressive graph generator.
    
    At each step t:
    1. Graph-level RNN state summarizes G_{<t}
    2. Edge MLP predicts connections to previous nodes
    3. Stop predictor decides whether to add more nodes
    """

    def __init__(
        self,
        max_nodes: int,
        hidden_dim: int = 128,
        node_feature_dim: int = 0,
        rnn_type: str = "gru",
    ):
        super().__init__()
        self.max_nodes = max_nodes
        self.hidden_dim = hidden_dim
        self.node_feature_dim = node_feature_dim

        # Graph-level RNN: summarizes generation history
        rnn_input_dim = max_nodes  # edge vector padded to max_nodes
        if rnn_type == "gru":
            self.graph_rnn = nn.GRU(
                input_size=rnn_input_dim,
                hidden_size=hidden_dim,
                batch_first=True,
            )
        else:
            self.graph_rnn = nn.LSTM(
                input_size=rnn_input_dim,
                hidden_size=hidden_dim,
                batch_first=True,
            )

        # Edge predictor: given graph state, predict edges to previous nodes
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_nodes),
        )

        # Stop predictor
        self.stop_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Initial hidden state
        self.h0 = nn.Parameter(torch.zeros(1, 1, hidden_dim))

    def _get_init_hidden(self, batch_size: int) -> torch.Tensor:
        return self.h0.expand(1, batch_size, self.hidden_dim).contiguous()

    def forward(
        self,
        adj_sequences: torch.Tensor,
        lengths: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Training forward pass with teacher forcing.
        
        Args:
            adj_sequences: (B, max_nodes, max_nodes) padded edge vectors
                adj_sequences[b, t, :t] = edges from node t to nodes 0..t-1
            lengths: (B,) number of nodes in each graph
            
        Returns:
            Dictionary with 'edge_loss' and 'stop_loss'
        """
        batch_size = adj_sequences.size(0)
        device = adj_sequences.device
        h = self._get_init_hidden(batch_size).to(device)

        edge_loss = torch.tensor(0.0, device=device)
        stop_loss = torch.tensor(0.0, device=device)
        num_edge_preds = 0
        num_stop_preds = 0

        for t in range(1, self.max_nodes):
            # Input: edge vector from previous step
            if t == 1:
                x_t = torch.zeros(batch_size, 1, self.max_nodes, device=device)
            else:
                x_t = adj_sequences[:, t - 1, :].unsqueeze(1)  # (B, 1, max_nodes)

            # RNN step
            out, h = self.graph_rnn(x_t, h)
            graph_state = out.squeeze(1)  # (B, hidden_dim)

            # Edge predictions for step t
            edge_logits = self.edge_mlp(graph_state)  # (B, max_nodes)
            edge_targets = adj_sequences[:, t, :]  # (B, max_nodes)

            # Only compute loss for valid positions (node t connects to 0..t-1)
            # and for graphs that have at least t+1 nodes
            active = (lengths > t).float()  # (B,)
            if active.sum() > 0:
                # Mask: only positions 0..t-1 are valid edge targets
                mask = torch.zeros(batch_size, self.max_nodes, device=device)
                mask[:, :t] = 1.0
                mask = mask * active.unsqueeze(1)

                edge_bce = F.binary_cross_entropy_with_logits(
                    edge_logits, edge_targets, reduction="none"
                )
                edge_loss = edge_loss + (edge_bce * mask).sum()
                num_edge_preds += mask.sum().item()

            # Stop prediction
            stop_logits = self.stop_mlp(graph_state).squeeze(-1)  # (B,)
            # Target: stop=1 if this is the last node
            stop_target = (lengths == t + 1).float()
            # Only compute for graphs still being generated
            stop_active = (lengths >= t + 1).float()
            if stop_active.sum() > 0:
                stop_bce = F.binary_cross_entropy_with_logits(
                    stop_logits, stop_target, reduction="none"
                )
                stop_loss = stop_loss + (stop_bce * stop_active).sum()
                num_stop_preds += stop_active.sum().item()

        # Normalize
        edge_loss = edge_loss / max(num_edge_preds, 1)
        stop_loss = stop_loss / max(num_stop_preds, 1)

        return {
            "edge_loss": edge_loss,
            "stop_loss": stop_loss,
            "total_loss": edge_loss + stop_loss,
        }

    @torch.no_grad()
    def generate(
        self,
        num_graphs: int = 1,
        temperature: float = 1.0,
        device: str = "cpu",
    ) -> list[torch.Tensor]:
        """
        Generate graphs autoregressively.
        
        Returns:
            List of adjacency matrices (variable sizes)
        """
        self.eval()
        graphs = []

        for _ in range(num_graphs):
            h = self._get_init_hidden(1).to(device)
            edges = []
            x_t = torch.zeros(1, 1, self.max_nodes, device=device)

            for t in range(1, self.max_nodes):
                out, h = self.graph_rnn(x_t, h)
                graph_state = out.squeeze(1)

                # Sample edges to previous nodes
                edge_logits = self.edge_mlp(graph_state)[0, :t]  # (t,)
                edge_probs = torch.sigmoid(edge_logits / temperature)
                edge_sample = torch.bernoulli(edge_probs)
                edges.append(edge_sample)

                # Check stop
                stop_logit = self.stop_mlp(graph_state).squeeze()
                stop_prob = torch.sigmoid(stop_logit / temperature)
                if torch.bernoulli(stop_prob).item() > 0.5 and t >= 2:
                    break

                # Prepare next input
                x_t = torch.zeros(1, 1, self.max_nodes, device=device)
                x_t[0, 0, :t] = edge_sample

            # Reconstruct adjacency matrix
            n = len(edges) + 1
            adj = torch.zeros(n, n, device=device)
            for t, e in enumerate(edges):
                adj[t + 1, : t + 1] = e
                adj[: t + 1, t + 1] = e
            graphs.append(adj.cpu())

        return graphs


def bfs_node_ordering(adj: torch.Tensor) -> list[int]:
    """Compute BFS ordering starting from highest-degree node."""
    n = adj.size(0)
    start = adj.sum(dim=1).argmax().item()
    visited = {start}
    order = [start]
    queue = deque([start])

    while queue:
        node = queue.popleft()
        neighbors = torch.where(adj[node] > 0)[0].tolist()
        neighbors.sort(key=lambda x: -adj[x].sum().item())
        for nb in neighbors:
            if nb not in visited:
                visited.add(nb)
                order.append(nb)
                queue.append(nb)

    for i in range(n):
        if i not in visited:
            order.append(i)
    return order


def prepare_training_sequence(
    adj: torch.Tensor,
    max_nodes: int,
) -> tuple[torch.Tensor, int]:
    """
    Convert adjacency matrix to training sequence under BFS ordering.
    
    Returns:
        adj_sequence: (max_nodes, max_nodes) padded edge vectors
        num_nodes: actual number of nodes
    """
    order = bfs_node_ordering(adj)
    n = adj.size(0)

    # Reorder adjacency
    perm = torch.tensor(order)
    adj_ordered = adj[perm][:, perm]

    # Create sequence: row t contains edges to nodes 0..t-1
    seq = torch.zeros(max_nodes, max_nodes)
    for t in range(min(n, max_nodes)):
        for s in range(t):
            seq[t, s] = adj_ordered[t, s]

    return seq, n


if __name__ == "__main__":
    torch.manual_seed(42)
    max_n = 15

    # Create synthetic training data
    print("=== Preparing Training Data ===")
    graphs = []
    for _ in range(100):
        n = torch.randint(5, max_n, (1,)).item()
        adj = (torch.rand(n, n) < 0.2).float()
        adj = torch.triu(adj, diagonal=1)
        adj = adj + adj.t()
        graphs.append(adj)

    sequences = []
    lengths = []
    for adj in graphs:
        seq, n = prepare_training_sequence(adj, max_n)
        sequences.append(seq)
        lengths.append(n)

    adj_seq = torch.stack(sequences)  # (100, max_n, max_n)
    lens = torch.tensor(lengths)
    print(f"Training data: {adj_seq.shape}, lengths: {lens.float().mean():.1f} avg")

    # Train
    print("\n=== Training ===")
    model = SequentialGraphGenerator(max_nodes=max_n, hidden_dim=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(30):
        model.train()
        losses = model(adj_seq, lens)
        loss = losses["total_loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: loss={loss.item():.4f} "
                  f"(edge={losses['edge_loss'].item():.4f}, "
                  f"stop={losses['stop_loss'].item():.4f})")

    # Generate
    print("\n=== Generation ===")
    generated = model.generate(num_graphs=10)
    for i, g in enumerate(generated):
        n = g.size(0)
        e = int(g.sum().item()) // 2
        density = 2 * e / (n * (n - 1)) if n > 1 else 0
        print(f"Graph {i}: {n} nodes, {e} edges, density={density:.3f}")

    # Compare statistics
    ref_sizes = [adj.size(0) for adj in graphs]
    gen_sizes = [g.size(0) for g in generated]
    print(f"\nRef avg size: {sum(ref_sizes)/len(ref_sizes):.1f}")
    print(f"Gen avg size: {sum(gen_sizes)/len(gen_sizes):.1f}")
```
