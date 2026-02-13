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
