"""
Chapter 29.2.3: Update Functions
Various update strategies for message passing GNNs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================
# 1. Update Function Implementations
# ============================================================

class ReplaceUpdate(nn.Module):
    """Simple replacement: h_v = sigma(m_v)."""
    def __init__(self, channels):
        super().__init__()
    def forward(self, h_old, m_agg):
        return F.relu(m_agg)


class ConcatLinearUpdate(nn.Module):
    """Concatenation + Linear: h_v = sigma(W [h_v || m_v])."""
    def __init__(self, node_channels, msg_channels, out_channels):
        super().__init__()
        self.lin = nn.Linear(node_channels + msg_channels, out_channels)
    def forward(self, h_old, m_agg):
        return F.relu(self.lin(torch.cat([h_old, m_agg], dim=-1)))


class ResidualUpdate(nn.Module):
    """Residual: h_v = h_v + sigma(W * m_v)."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels)
        self.skip = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()
    def forward(self, h_old, m_agg):
        return self.skip(h_old) + F.relu(self.lin(m_agg))


class GRUUpdate(nn.Module):
    """GRU-based: h_v = GRU(m_v, h_v)."""
    def __init__(self, channels):
        super().__init__()
        self.gru = nn.GRUCell(channels, channels)
    def forward(self, h_old, m_agg):
        return self.gru(m_agg, h_old)


class LSTMUpdate(nn.Module):
    """LSTM-based update with cell state."""
    def __init__(self, channels):
        super().__init__()
        self.lstm_cell = nn.LSTMCell(channels, channels)
        self.cell_state = None

    def reset_cell(self, num_nodes, channels, device):
        self.cell_state = torch.zeros(num_nodes, channels, device=device)

    def forward(self, h_old, m_agg):
        if self.cell_state is None:
            self.cell_state = torch.zeros_like(h_old)
        h_new, c_new = self.lstm_cell(m_agg, (h_old, self.cell_state))
        self.cell_state = c_new
        return h_new


class MLPUpdate(nn.Module):
    """MLP-based: h_v = MLP([h_v || m_v])."""
    def __init__(self, node_channels, msg_channels, out_channels, hidden=None):
        super().__init__()
        hidden = hidden or out_channels * 2
        self.mlp = nn.Sequential(
            nn.Linear(node_channels + msg_channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_channels))
    def forward(self, h_old, m_agg):
        return self.mlp(torch.cat([h_old, m_agg], dim=-1))


class GINUpdate(nn.Module):
    """GIN: h_v = MLP((1 + eps) * h_v + m_v)."""
    def __init__(self, channels, hidden=None):
        super().__init__()
        hidden = hidden or channels * 2
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, channels))
    def forward(self, h_old, m_agg):
        return self.mlp((1 + self.eps) * h_old + m_agg)


# ============================================================
# 2. Message Passing Layer with Configurable Update
# ============================================================

class MPLayerWithUpdate(nn.Module):
    """Message passing layer with configurable update function."""

    def __init__(self, in_ch, out_ch, update_fn):
        super().__init__()
        self.msg_lin = nn.Linear(in_ch, out_ch)
        self.update_fn = update_fn

    def forward(self, x, edge_index):
        src, dst = edge_index[0], edge_index[1]
        n = x.shape[0]

        messages = self.msg_lin(x[src])
        agg = torch.zeros(n, messages.shape[1], device=x.device)
        agg.scatter_add_(0, dst.unsqueeze(1).expand_as(messages), messages)
        deg = torch.zeros(n, device=x.device)
        deg.scatter_add_(0, dst, torch.ones(dst.shape[0], device=x.device))
        agg = agg / deg.clamp(min=1).unsqueeze(1)

        return self.update_fn(x, agg)


def demo_update_functions():
    """Compare all update function variants."""
    print("=" * 60)
    print("Update Function Comparison")
    print("=" * 60)

    torch.manual_seed(42)
    edge_index = torch.tensor([
        [0, 1, 0, 2, 1, 2, 1, 3, 2, 4, 3, 4],
        [1, 0, 2, 0, 2, 1, 3, 1, 4, 2, 4, 3]], dtype=torch.long)
    x = torch.randn(5, 8)

    updates = {
        'Replace': ReplaceUpdate(16),
        'Concat+Linear': ConcatLinearUpdate(8, 16, 16),
        'Residual': ResidualUpdate(8, 16),
        'GRU': GRUUpdate(8),  # in=out for GRU
        'MLP': MLPUpdate(8, 16, 16),
        'GIN': GINUpdate(8),
    }

    for name, update_fn in updates.items():
        out_ch = 8 if name in ['GRU', 'GIN'] else 16
        layer = MPLayerWithUpdate(8, out_ch, update_fn)
        out = layer(x, edge_index)
        print(f"  {name:15s}: input={x.shape} -> output={out.shape}")


# ============================================================
# 3. Deep GNN with Different Updates
# ============================================================

class DeepGNN(nn.Module):
    """Multi-layer GNN to test update function effects on depth."""

    def __init__(self, in_ch, hidden_ch, num_layers, update_type='residual'):
        super().__init__()
        self.input_lin = nn.Linear(in_ch, hidden_ch)
        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            if update_type == 'replace':
                update = ReplaceUpdate(hidden_ch)
            elif update_type == 'residual':
                update = ResidualUpdate(hidden_ch, hidden_ch)
            elif update_type == 'gru':
                update = GRUUpdate(hidden_ch)
            elif update_type == 'gin':
                update = GINUpdate(hidden_ch)
            else:
                update = ReplaceUpdate(hidden_ch)
            self.layers.append(MPLayerWithUpdate(hidden_ch, hidden_ch, update))

    def forward(self, x, edge_index):
        x = self.input_lin(x)
        for layer in self.layers:
            x = layer(x, edge_index)
        return x


def demo_depth_comparison():
    """Compare update functions across different depths."""
    print("\n" + "=" * 60)
    print("Update Functions vs Depth (Over-Smoothing)")
    print("=" * 60)

    torch.manual_seed(42)
    import networkx as nx
    G = nx.karate_club_graph()
    edges = list(G.edges())
    src = [e[0] for e in edges] + [e[1] for e in edges]
    dst = [e[1] for e in edges] + [e[0] for e in edges]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    x = torch.randn(G.number_of_nodes(), 16)

    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Metric: mean pairwise distance between node embeddings")
    print(f"{'Layers':>8}", end="")
    for update_type in ['replace', 'residual', 'gru', 'gin']:
        print(f"  {update_type:>10}", end="")
    print()

    for n_layers in [1, 2, 4, 8, 16]:
        print(f"{n_layers:>8}", end="")
        for update_type in ['replace', 'residual', 'gru', 'gin']:
            model = DeepGNN(16, 16, n_layers, update_type)
            model.eval()
            with torch.no_grad():
                out = model(x, edge_index)
            mean_repr = out.mean(dim=0)
            dist = torch.norm(out - mean_repr, dim=1).mean().item()
            print(f"  {dist:10.4f}", end="")
        print()


# ============================================================
# 4. Gating Mechanism Analysis
# ============================================================

def demo_gating_analysis():
    """Analyze GRU gate activations during message passing."""
    print("\n" + "=" * 60)
    print("GRU Gating Analysis")
    print("=" * 60)

    torch.manual_seed(42)

    channels = 8
    gru = nn.GRUCell(channels, channels)

    h = torch.randn(5, channels)  # Previous node states
    m = torch.randn(5, channels)  # Aggregated messages

    # Access GRU internals
    with torch.no_grad():
        h_new = gru(m, h)

        # Manually compute gates
        W_ir, W_iz, W_in = gru.weight_ih.chunk(3, 0)
        W_hr, W_hz, W_hn = gru.weight_hh.chunk(3, 0)
        b_ir, b_iz, b_in = gru.bias_ih.chunk(3, 0)
        b_hr, b_hz, b_hn = gru.bias_hh.chunk(3, 0)

        r = torch.sigmoid(m @ W_ir.T + b_ir + h @ W_hr.T + b_hr)
        z = torch.sigmoid(m @ W_iz.T + b_iz + h @ W_hz.T + b_hz)

        print(f"Reset gate r (how much old info to forget):")
        print(f"  Mean: {r.mean():.4f}, Std: {r.std():.4f}")
        print(f"  Per node: {r.mean(dim=1).tolist()}")

        print(f"\nUpdate gate z (how much new info to accept):")
        print(f"  Mean: {z.mean():.4f}, Std: {z.std():.4f}")
        print(f"  Per node: {z.mean(dim=1).tolist()}")

        print(f"\nInterpretation:")
        print(f"  z close to 1 -> keep old state (ignore messages)")
        print(f"  z close to 0 -> use new information from messages")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    demo_update_functions()
    demo_depth_comparison()
    demo_gating_analysis()
