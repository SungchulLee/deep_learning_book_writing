"""
Chapter 29.2.4: Message Passing Neural Network (MPNN)
Complete MPNN implementation with graph-level prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List


# ============================================================
# 1. MPNN Framework Implementation
# ============================================================

class MPNN(nn.Module):
    """
    Full Message Passing Neural Network.
    Implements the Gilmer et al. (2017) framework with configurable
    message, update, and readout functions.
    """

    def __init__(self, node_features: int, edge_features: int,
                 hidden_dim: int, output_dim: int,
                 num_message_steps: int = 3,
                 readout: str = 'sum',
                 update_type: str = 'gru'):
        super().__init__()

        self.num_steps = num_message_steps
        self.hidden_dim = hidden_dim
        self.readout_type = readout

        # Input projection
        self.node_encoder = nn.Linear(node_features, hidden_dim)

        # Edge-conditioned message function
        self.edge_nn = nn.Sequential(
            nn.Linear(edge_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * hidden_dim)
        )

        # Backup: simple message function (when no edge features)
        self.message_lin = nn.Linear(hidden_dim, hidden_dim)

        # Update function
        if update_type == 'gru':
            self.update = nn.GRUCell(hidden_dim, hidden_dim)
        elif update_type == 'mlp':
            self.update = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim))

        self.update_type = update_type

        # Readout MLP
        self.readout_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        # Optional: Set2Set readout
        if readout == 'set2set':
            self.set2set_steps = 3
            self.set2set_lstm = nn.LSTMCell(2 * hidden_dim, hidden_dim)
            self.readout_mlp = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim))

    def message_function(self, h_src: torch.Tensor, h_dst: torch.Tensor,
                          edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute messages from source to destination."""
        if edge_attr is not None:
            # Edge-conditioned: M(h_u, e_vu) = Theta(e_vu) * h_u
            edge_weight = self.edge_nn(edge_attr)
            edge_weight = edge_weight.view(-1, self.hidden_dim, self.hidden_dim)
            msg = torch.bmm(edge_weight, h_src.unsqueeze(2)).squeeze(2)
            return msg
        else:
            return self.message_lin(h_src)

    def aggregate(self, messages: torch.Tensor, dst: torch.Tensor,
                   num_nodes: int) -> torch.Tensor:
        """Sum aggregation."""
        out = torch.zeros(num_nodes, self.hidden_dim, device=messages.device)
        out.scatter_add_(0, dst.unsqueeze(1).expand_as(messages), messages)
        return out

    def update_function(self, h: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """Update node hidden states."""
        if self.update_type == 'gru':
            return self.update(m, h)
        elif self.update_type == 'mlp':
            return self.update(torch.cat([h, m], dim=-1))

    def readout(self, h: torch.Tensor,
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Graph-level readout."""
        if batch is None:
            batch = torch.zeros(h.shape[0], dtype=torch.long, device=h.device)

        num_graphs = batch.max().item() + 1

        if self.readout_type == 'sum':
            graph_repr = torch.zeros(num_graphs, self.hidden_dim, device=h.device)
            graph_repr.scatter_add_(0, batch.unsqueeze(1).expand_as(h), h)
        elif self.readout_type == 'mean':
            graph_repr = torch.zeros(num_graphs, self.hidden_dim, device=h.device)
            graph_repr.scatter_add_(0, batch.unsqueeze(1).expand_as(h), h)
            counts = torch.zeros(num_graphs, device=h.device)
            counts.scatter_add_(0, batch, torch.ones(h.shape[0], device=h.device))
            graph_repr = graph_repr / counts.clamp(min=1).unsqueeze(1)
        elif self.readout_type == 'set2set':
            graph_repr = self._set2set_readout(h, batch, num_graphs)

        return self.readout_mlp(graph_repr)

    def _set2set_readout(self, h, batch, num_graphs):
        """Set2Set readout (simplified)."""
        q = torch.zeros(num_graphs, self.hidden_dim, device=h.device)
        c = torch.zeros(num_graphs, self.hidden_dim, device=h.device)

        for _ in range(self.set2set_steps):
            # Attention over nodes
            e = (h * q[batch]).sum(dim=-1)
            a = torch.zeros(h.shape[0], device=h.device)
            # Softmax per graph
            e_max = torch.zeros(num_graphs, device=h.device)
            e_max.scatter_reduce_(0, batch, e, reduce='amax')
            a = torch.exp(e - e_max[batch])
            a_sum = torch.zeros(num_graphs, device=h.device)
            a_sum.scatter_add_(0, batch, a)
            a = a / a_sum[batch].clamp(min=1e-10)

            # Weighted sum
            r = torch.zeros(num_graphs, self.hidden_dim, device=h.device)
            r.scatter_add_(0, batch.unsqueeze(1).expand_as(h), h * a.unsqueeze(1))

            q_input = torch.cat([q, r], dim=-1)
            q, c = self.set2set_lstm(q_input, (q, c))

        return torch.cat([q, r], dim=-1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None,
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Encode input
        h = self.node_encoder(x)

        # Message passing
        src, dst = edge_index[0], edge_index[1]
        for t in range(self.num_steps):
            messages = self.message_function(h[src], h[dst], edge_attr)
            m_agg = self.aggregate(messages, dst, h.shape[0])
            h = self.update_function(h, m_agg)

        # Readout
        return self.readout(h, batch)


# ============================================================
# 2. Demo: Molecular Property Prediction (Synthetic)
# ============================================================

def create_synthetic_molecules(n_molecules=200, max_atoms=15):
    """Create synthetic molecular graphs for demonstration."""
    torch.manual_seed(42)
    np.random.seed(42)

    graphs = []
    labels = []

    for i in range(n_molecules):
        n_atoms = np.random.randint(5, max_atoms + 1)

        # Node features: [atom_type (one-hot, 4 types), charge, mass]
        atom_types = np.random.randint(0, 4, n_atoms)
        x = torch.zeros(n_atoms, 6)
        for j in range(n_atoms):
            x[j, atom_types[j]] = 1.0
        x[:, 4] = torch.randn(n_atoms) * 0.5  # charge
        x[:, 5] = torch.rand(n_atoms) + 0.5    # mass

        # Random edges (roughly tree-like + some cycles)
        edges_src, edges_dst = [], []
        for j in range(1, n_atoms):
            parent = np.random.randint(0, j)
            edges_src.extend([j, parent])
            edges_dst.extend([parent, j])
        # Add a few extra edges
        n_extra = np.random.randint(0, min(3, n_atoms))
        for _ in range(n_extra):
            u, v = np.random.choice(n_atoms, 2, replace=False)
            edges_src.extend([u, v])
            edges_dst.extend([v, u])

        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)

        # Synthetic label: depends on graph structure + features
        label = (x[:, :4].sum() + n_atoms * 0.1 +
                 edge_index.shape[1] * 0.05 + torch.randn(1) * 0.1)

        graphs.append({'x': x, 'edge_index': edge_index, 'y': label})
        labels.append(label.item())

    return graphs


def demo_molecular_prediction():
    """End-to-end MPNN for molecular property prediction."""
    print("=" * 60)
    print("MPNN: Molecular Property Prediction (Synthetic)")
    print("=" * 60)

    graphs = create_synthetic_molecules(200)

    # Split
    train_graphs = graphs[:160]
    test_graphs = graphs[160:]

    # Model
    model = MPNN(
        node_features=6, edge_features=0,
        hidden_dim=32, output_dim=1,
        num_message_steps=3, readout='sum', update_type='gru')

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Training loop (mini-batch of single graphs for simplicity)
    model.train()
    for epoch in range(50):
        total_loss = 0
        np.random.shuffle(train_graphs)

        for g in train_graphs:
            optimizer.zero_grad()
            pred = model(g['x'], g['edge_index'])
            loss = criterion(pred.squeeze(), g['y'].squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            # Evaluate
            model.eval()
            test_loss = 0
            with torch.no_grad():
                for g in test_graphs:
                    pred = model(g['x'], g['edge_index'])
                    test_loss += criterion(pred.squeeze(), g['y'].squeeze()).item()
            print(f"Epoch {epoch+1:3d}: Train Loss={total_loss/len(train_graphs):.4f}, "
                  f"Test Loss={test_loss/len(test_graphs):.4f}")
            model.train()


# ============================================================
# 3. MPNN Variants Comparison
# ============================================================

def compare_mpnn_variants():
    """Compare different MPNN configurations."""
    print("\n" + "=" * 60)
    print("MPNN Variant Comparison")
    print("=" * 60)

    graphs = create_synthetic_molecules(100)
    train_g = graphs[:80]
    test_g = graphs[80:]

    configs = [
        {'num_message_steps': 1, 'readout': 'sum', 'update_type': 'gru'},
        {'num_message_steps': 3, 'readout': 'sum', 'update_type': 'gru'},
        {'num_message_steps': 3, 'readout': 'mean', 'update_type': 'gru'},
        {'num_message_steps': 3, 'readout': 'sum', 'update_type': 'mlp'},
        {'num_message_steps': 5, 'readout': 'sum', 'update_type': 'gru'},
    ]

    for cfg in configs:
        torch.manual_seed(42)
        model = MPNN(node_features=6, edge_features=0,
                      hidden_dim=32, output_dim=1, **cfg)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        model.train()
        for epoch in range(30):
            for g in train_g:
                optimizer.zero_grad()
                pred = model(g['x'], g['edge_index'])
                loss = criterion(pred.squeeze(), g['y'].squeeze())
                loss.backward()
                optimizer.step()

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for g in test_g:
                pred = model(g['x'], g['edge_index'])
                test_loss += criterion(pred.squeeze(), g['y'].squeeze()).item()
        test_loss /= len(test_g)

        desc = (f"T={cfg['num_message_steps']}, "
                f"readout={cfg['readout']}, "
                f"update={cfg['update_type']}")
        print(f"  {desc:40s} -> Test MSE: {test_loss:.4f}")


# ============================================================
# 4. Financial Network MPNN
# ============================================================

def demo_financial_mpnn():
    """MPNN for financial network risk prediction."""
    print("\n" + "=" * 60)
    print("Financial Network MPNN")
    print("=" * 60)

    torch.manual_seed(42)
    np.random.seed(42)

    # Create synthetic portfolio risk data
    n_samples = 100
    graphs = []

    for _ in range(n_samples):
        n_assets = np.random.randint(5, 12)

        # Node features: [return, vol, beta, sector_id]
        x = torch.randn(n_assets, 4)
        x[:, 1] = torch.abs(x[:, 1])  # Volatility is positive

        # Correlation-based edges
        src, dst = [], []
        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                if np.random.random() < 0.4:
                    src.extend([i, j])
                    dst.extend([j, i])
        if not src:  # Ensure at least one edge
            src, dst = [0, 1], [1, 0]
        edge_index = torch.tensor([src, dst], dtype=torch.long)

        # Portfolio risk label (synthetic)
        risk = x[:, 1].mean() + 0.1 * len(src) / n_assets + torch.randn(1) * 0.05
        graphs.append({'x': x, 'edge_index': edge_index, 'y': risk})

    # Train
    model = MPNN(node_features=4, edge_features=0,
                  hidden_dim=16, output_dim=1,
                  num_message_steps=2, readout='mean', update_type='gru')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(30):
        total_loss = 0
        for g in graphs[:80]:
            optimizer.zero_grad()
            pred = model(g['x'], g['edge_index'])
            loss = F.mse_loss(pred.squeeze(), g['y'].squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Loss={total_loss/80:.4f}")

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for g in graphs[80:]:
            pred = model(g['x'], g['edge_index'])
            test_loss += F.mse_loss(pred.squeeze(), g['y'].squeeze()).item()
    print(f"Test MSE: {test_loss/20:.4f}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    demo_molecular_prediction()
    compare_mpnn_variants()
    demo_financial_mpnn()
