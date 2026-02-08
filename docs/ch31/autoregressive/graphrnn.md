# GraphRNN

## Overview

GraphRNN (You et al., 2018) is the foundational autoregressive model for graph generation. It introduces a hierarchical RNN architecture that decomposes graph generation into two levels: a **graph-level RNN** that maintains the global generation state and generates one node per step, and an **edge-level RNN** that generates the edges connecting each new node to existing nodes. This two-level decomposition, combined with BFS ordering to exploit graph locality, makes GraphRNN both expressive and computationally tractable.

## Architecture

GraphRNN uses two recurrent neural networks in a hierarchical arrangement:

**Graph-level RNN ($f_{\text{graph}}$).** Maintains a hidden state $\mathbf{h}_t^G$ summarizing the graph constructed through step $t$. At each step, it receives the edge vector from the previous step and outputs an initial hidden state for the edge-level RNN:

$$
\mathbf{h}_t^G = f_{\text{graph}}(\mathbf{h}_{t-1}^G, \mathbf{a}_{t-1})
$$

where $\mathbf{a}_{t-1} \in \{0,1\}^{M}$ is the (possibly truncated) edge vector from step $t-1$, and $M$ is the BFS bandwidth.

**Edge-level RNN ($f_{\text{edge}}$).** Initialized with a state derived from $\mathbf{h}_t^G$, it sequentially predicts each edge in the vector $\mathbf{a}_t$:

$$
\mathbf{h}_{t,s}^E = f_{\text{edge}}(\mathbf{h}_{t,s-1}^E, a_{t,s-1})
$$
$$
p(a_{t,s} = 1) = \sigma(\text{MLP}(\mathbf{h}_{t,s}^E))
$$

where $a_{t,s}$ indicates the edge between node $t$ and node $t - s$ (in BFS order), and $\sigma$ is the sigmoid function.

## BFS Truncation

The key computational insight is that under BFS ordering, edges from a new node only reach back within the BFS bandwidth $M$. Instead of predicting connections to all $t-1$ previous nodes, GraphRNN only predicts $\min(t-1, M)$ edge decisions per step. This reduces the total number of edge predictions from $O(n^2)$ to $O(n \cdot M)$, where $M$ is typically much smaller than $n$ for sparse graphs.

The BFS bandwidth for a graph is:

$$
M = \max_{(u,v) \in \mathcal{E}} |\pi^{-1}(u) - \pi^{-1}(v)|
$$

where $\pi$ is the BFS ordering. For many graph families (community graphs, molecular graphs, social networks), $M \ll n$, making the truncation an effective approximation with minimal information loss.

## Training

GraphRNN is trained via maximum likelihood with teacher forcing at both levels. Given a graph with BFS ordering $\pi$ and bandwidth $M$:

$$
\mathcal{L} = -\sum_{t=2}^{n} \sum_{s=1}^{\min(t-1, M)} \left[ a_{t,s}^* \log p_\theta(a_{t,s} = 1 \mid \mathbf{h}_{t,s-1}^E) + (1 - a_{t,s}^*) \log p_\theta(a_{t,s} = 0 \mid \mathbf{h}_{t,s-1}^E) \right]
$$

During training, the ground-truth edge values $a_{t,s}^*$ are fed as inputs to the edge-level RNN (teacher forcing). The edge-level RNN is initialized at each step $t$ with a learned transformation of the graph-level state:

$$
\mathbf{h}_{t,0}^E = \text{MLP}_{\text{init}}(\mathbf{h}_t^G)
$$

## Simplified Variant: GraphRNN-S

For computational efficiency, GraphRNN-S replaces the edge-level RNN with a single MLP that predicts the entire edge vector at once:

$$
\hat{\mathbf{a}}_t = \sigma(\text{MLP}(\mathbf{h}_t^G)) \in [0,1]^{M}
$$

This removes sequential dependencies between edge decisions within a single step, trading expressiveness for parallelism. GraphRNN-S is significantly faster to train but may produce lower-quality graphs for datasets where intra-step edge correlations are important.

## Generation Procedure

Generation proceeds node by node until the model outputs a termination signal:

1. Initialize $\mathbf{h}_0^G$ as a learned or zero vector
2. For $t = 1, 2, \ldots, n_{\max}$:
    - Compute $\mathbf{h}_t^G = f_{\text{graph}}(\mathbf{h}_{t-1}^G, \mathbf{a}_{t-1})$
    - Initialize $\mathbf{h}_{t,0}^E = \text{MLP}_{\text{init}}(\mathbf{h}_t^G)$
    - For $s = 1, \ldots, M$:
        - Compute $p_{t,s} = \sigma(\text{MLP}(\mathbf{h}_{t,s}^E))$
        - Sample $a_{t,s} \sim \text{Bernoulli}(p_{t,s})$
        - If $a_{t,s} = 0$ for all $s$ (EOS pattern), stop generating
        - Update $\mathbf{h}_{t,s}^E = f_{\text{edge}}(\mathbf{h}_{t,s-1}^E, a_{t,s})$
3. Reconstruct adjacency matrix from edge sequences

The termination condition uses a special end-of-sequence (EOS) pattern: if the entire edge vector $\mathbf{a}_t$ is zero (no connections to any previous node), generation stops. This naturally produces graphs of variable size.

## Limitations

**Ordering sensitivity.** Different BFS traversals of the same graph yield different training sequences. While data augmentation with multiple orderings helps, the model still implicitly learns an ordering-dependent distribution. The gap between $p_\theta(\mathcal{G} \mid \pi)$ and the true $p(\mathcal{G})$ depends on how well the chosen ordering class captures structural regularities.

**Long-range dependencies.** The graph-level RNN must propagate information across potentially many steps. For large graphs, the hidden state may fail to capture early structural decisions that affect later generation choices.

**Scalability.** Training and generation are inherently sequential across nodes, limiting parallelism. For graphs with hundreds of nodes, training can be slow despite the BFS truncation.

## Finance Application: Interbank Network Generation

GraphRNN is well-suited for generating synthetic interbank lending networks, where the sequential construction mirrors temporal network formation. Banks join the interbank market and establish lending relationships incrementally. The BFS ordering naturally captures the hierarchical structure of financial networks — core banks (high degree) appear first, with peripheral institutions connecting primarily to nearby core nodes.

## Implementation: GraphRNN with BFS Truncation

```python
"""
GraphRNN: hierarchical RNN for graph generation with BFS truncation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from typing import Optional


class EdgeRNN(nn.Module):
    """Edge-level RNN: generates edge vector sequentially."""

    def __init__(self, hidden_dim: int, edge_dim: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rnn = nn.GRU(
            input_size=edge_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, edge_dim),
        )

    def forward(
        self,
        h_init: torch.Tensor,
        edge_targets: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Teacher-forced forward pass.
        
        Args:
            h_init: (B, hidden_dim) initial hidden state from graph RNN
            edge_targets: (B, M) ground-truth edge vectors
            lengths: (B,) valid length for each sample
            
        Returns:
            logits: (B, M) edge logits
        """
        batch_size, M = edge_targets.shape
        device = edge_targets.device

        # Prepare input: shifted targets with SOS token (zero)
        sos = torch.zeros(batch_size, 1, 1, device=device)
        inputs = edge_targets[:, :-1].unsqueeze(-1)  # (B, M-1, 1)
        inputs = torch.cat([sos, inputs], dim=1)  # (B, M, 1)

        # Run edge RNN
        h = h_init.unsqueeze(0)  # (1, B, hidden_dim)
        output, _ = self.rnn(inputs, h)  # (B, M, hidden_dim)
        logits = self.output(output).squeeze(-1)  # (B, M)

        return logits

    @torch.no_grad()
    def generate(
        self,
        h_init: torch.Tensor,
        max_edges: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate edge vector autoregressively.
        
        Args:
            h_init: (1, hidden_dim) initial hidden state
            max_edges: maximum number of edges to generate
            temperature: sampling temperature
            
        Returns:
            edges: (max_edges,) sampled binary edge vector
        """
        device = h_init.device
        h = h_init.unsqueeze(0)  # (1, 1, hidden_dim)
        edges = []
        x = torch.zeros(1, 1, 1, device=device)

        for s in range(max_edges):
            out, h = self.rnn(x, h)
            logit = self.output(out).squeeze()
            prob = torch.sigmoid(logit / temperature)
            edge = torch.bernoulli(prob)
            edges.append(edge.item())
            x = edge.view(1, 1, 1)

        return torch.tensor(edges, device=device)


class GraphRNN(nn.Module):
    """
    GraphRNN: two-level hierarchical RNN for graph generation.
    
    Graph-level RNN tracks global state; edge-level RNN generates
    connections for each new node.
    """

    def __init__(
        self,
        max_nodes: int,
        bfs_bandwidth: int,
        graph_hidden_dim: int = 128,
        edge_hidden_dim: int = 64,
        use_edge_rnn: bool = True,
    ):
        super().__init__()
        self.max_nodes = max_nodes
        self.M = bfs_bandwidth  # BFS bandwidth (truncation window)
        self.graph_hidden_dim = graph_hidden_dim
        self.use_edge_rnn = use_edge_rnn

        # Graph-level RNN
        self.graph_rnn = nn.GRU(
            input_size=bfs_bandwidth,
            hidden_size=graph_hidden_dim,
            batch_first=True,
        )
        self.h0 = nn.Parameter(torch.zeros(1, 1, graph_hidden_dim))

        if use_edge_rnn:
            # Full GraphRNN with edge-level RNN
            self.edge_rnn = EdgeRNN(edge_hidden_dim)
            self.edge_init = nn.Linear(graph_hidden_dim, edge_hidden_dim)
        else:
            # GraphRNN-S: MLP for entire edge vector
            self.edge_mlp = nn.Sequential(
                nn.Linear(graph_hidden_dim, graph_hidden_dim),
                nn.ReLU(),
                nn.Linear(graph_hidden_dim, bfs_bandwidth),
            )

    def _get_init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return self.h0.expand(1, batch_size, self.graph_hidden_dim).contiguous().to(device)

    def forward(
        self,
        edge_sequences: torch.Tensor,
        lengths: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Training forward pass.
        
        Args:
            edge_sequences: (B, max_nodes, M) BFS-truncated edge vectors
            lengths: (B,) number of nodes per graph
        """
        B, T, M = edge_sequences.shape
        device = edge_sequences.device
        h_graph = self._get_init_hidden(B, device)

        total_loss = torch.tensor(0.0, device=device)
        num_preds = 0

        for t in range(1, T):
            # Graph RNN input: edge vector from previous step
            if t == 1:
                x_t = torch.zeros(B, 1, M, device=device)
            else:
                x_t = edge_sequences[:, t - 1, :].unsqueeze(1)

            out, h_graph = self.graph_rnn(x_t, h_graph)
            graph_state = out.squeeze(1)  # (B, graph_hidden_dim)

            # Which samples are active (have node t)
            active = (lengths > t).float()
            if active.sum() == 0:
                break

            # Edge predictions
            targets = edge_sequences[:, t, :]  # (B, M)
            valid_len = min(t, M)

            if self.use_edge_rnn:
                h_edge_init = self.edge_init(graph_state)  # (B, edge_hidden_dim)
                edge_lengths = torch.full((B,), valid_len, device=device)
                logits = self.edge_rnn(h_edge_init, targets, edge_lengths)
            else:
                logits = self.edge_mlp(graph_state)

            # Mask for valid edge positions
            mask = torch.zeros(B, M, device=device)
            mask[:, :valid_len] = 1.0
            mask = mask * active.unsqueeze(1)

            loss = F.binary_cross_entropy_with_logits(
                logits, targets, reduction="none"
            )
            total_loss = total_loss + (loss * mask).sum()
            num_preds += mask.sum().item()

        total_loss = total_loss / max(num_preds, 1)

        return {"total_loss": total_loss}

    @torch.no_grad()
    def generate(
        self,
        num_graphs: int = 1,
        temperature: float = 1.0,
        device: str = "cpu",
    ) -> list[torch.Tensor]:
        """Generate graphs using ancestral sampling."""
        self.eval()
        graphs = []

        for _ in range(num_graphs):
            h_graph = self._get_init_hidden(1, torch.device(device))
            edge_vectors = []
            x_t = torch.zeros(1, 1, self.M, device=device)

            for t in range(1, self.max_nodes):
                out, h_graph = self.graph_rnn(x_t, h_graph)
                graph_state = out.squeeze(1)

                valid_len = min(t, self.M)

                if self.use_edge_rnn:
                    h_edge = self.edge_init(graph_state[0])
                    edges = self.edge_rnn.generate(
                        h_edge.unsqueeze(0), valid_len, temperature
                    )
                    # Pad to M
                    edge_vec = torch.zeros(self.M, device=device)
                    edge_vec[:valid_len] = edges[:valid_len]
                else:
                    logits = self.edge_mlp(graph_state)[0]
                    probs = torch.sigmoid(logits / temperature)
                    edge_vec = torch.bernoulli(probs)
                    edge_vec[valid_len:] = 0  # Mask invalid positions

                # Check EOS: all zeros in valid region
                if edge_vec[:valid_len].sum() == 0 and t > 1:
                    break

                edge_vectors.append(edge_vec[:valid_len].clone())
                x_t = edge_vec.unsqueeze(0).unsqueeze(0)

            # Reconstruct adjacency
            n = len(edge_vectors) + 1
            adj = torch.zeros(n, n, device=device)
            for t, ev in enumerate(edge_vectors):
                step = t + 1
                for s in range(len(ev)):
                    target_node = step - s - 1
                    if target_node >= 0 and ev[s] > 0:
                        adj[step, target_node] = 1.0
                        adj[target_node, step] = 1.0

            graphs.append(adj.cpu())

        return graphs


def bfs_edge_sequences(
    adj: torch.Tensor,
    max_nodes: int,
    bandwidth: int,
) -> tuple[torch.Tensor, int]:
    """
    Convert adjacency matrix to BFS-truncated edge sequences.
    
    Returns:
        sequences: (max_nodes, bandwidth) edge vectors
        n: number of nodes
    """
    n = adj.size(0)

    # BFS ordering from highest-degree node
    start = adj.sum(1).argmax().item()
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

    # Reorder adjacency
    perm = torch.tensor(order)
    adj_bfs = adj[perm][:, perm]

    # Build edge sequences with BFS truncation
    seq = torch.zeros(max_nodes, bandwidth)
    for t in range(1, min(n, max_nodes)):
        for s in range(min(t, bandwidth)):
            seq[t, s] = adj_bfs[t, t - s - 1]

    return seq, n


if __name__ == "__main__":
    torch.manual_seed(42)

    max_n = 20
    bw = 8  # BFS bandwidth

    # Create training data: community-structured graphs
    print("=== Preparing Training Data ===")
    graphs = []
    for _ in range(200):
        n = torch.randint(8, max_n, (1,)).item()
        # Two-community structure
        n1 = n // 2
        n2 = n - n1
        adj = torch.zeros(n, n)
        # Intra-community edges (dense)
        for i in range(n1):
            for j in range(i + 1, n1):
                if torch.rand(1) < 0.4:
                    adj[i, j] = adj[j, i] = 1
        for i in range(n1, n):
            for j in range(i + 1, n):
                if torch.rand(1) < 0.4:
                    adj[i, j] = adj[j, i] = 1
        # Inter-community edges (sparse)
        for i in range(n1):
            for j in range(n1, n):
                if torch.rand(1) < 0.05:
                    adj[i, j] = adj[j, i] = 1
        graphs.append(adj)

    # Prepare sequences
    sequences = []
    lengths = []
    for adj in graphs:
        seq, n = bfs_edge_sequences(adj, max_n, bw)
        sequences.append(seq)
        lengths.append(n)

    all_seqs = torch.stack(sequences)
    all_lens = torch.tensor(lengths)
    print(f"Data shape: {all_seqs.shape}, avg nodes: {all_lens.float().mean():.1f}")

    # Train GraphRNN
    print("\n=== Training GraphRNN (full) ===")
    model = GraphRNN(
        max_nodes=max_n,
        bfs_bandwidth=bw,
        graph_hidden_dim=128,
        edge_hidden_dim=64,
        use_edge_rnn=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(50):
        model.train()
        result = model(all_seqs, all_lens)
        loss = result["total_loss"]
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: loss={loss.item():.4f}")

    # Train GraphRNN-S
    print("\n=== Training GraphRNN-S (simplified) ===")
    model_s = GraphRNN(
        max_nodes=max_n,
        bfs_bandwidth=bw,
        graph_hidden_dim=128,
        use_edge_rnn=False,
    )
    optimizer_s = torch.optim.Adam(model_s.parameters(), lr=0.001)
    print(f"Parameters: {sum(p.numel() for p in model_s.parameters()):,}")

    for epoch in range(50):
        model_s.train()
        result = model_s(all_seqs, all_lens)
        loss = result["total_loss"]
        optimizer_s.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model_s.parameters(), 1.0)
        optimizer_s.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: loss={loss.item():.4f}")

    # Generate and compare
    print("\n=== Generation Results ===")
    for name, m in [("GraphRNN", model), ("GraphRNN-S", model_s)]:
        gen = m.generate(num_graphs=10)
        sizes = [g.size(0) for g in gen]
        edges = [int(g.sum().item()) // 2 for g in gen]
        densities = [2 * e / (n * (n - 1)) if n > 1 else 0
                     for n, e in zip(sizes, edges)]
        print(f"\n{name}:")
        print(f"  Avg nodes: {sum(sizes)/len(sizes):.1f} (ref: {all_lens.float().mean():.1f})")
        print(f"  Avg edges: {sum(edges)/len(edges):.1f}")
        print(f"  Avg density: {sum(densities)/len(densities):.3f}")
```
