"""
Chapter 29.3.3: ChebNet
Chebyshev polynomial spectral graph convolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx


class ChebConv(nn.Module):
    """Chebyshev spectral graph convolution layer."""

    def __init__(self, in_channels, out_channels, K=3):
        super().__init__()
        self.K = K
        self.lins = nn.ModuleList([
            nn.Linear(in_channels, out_channels, bias=False) for _ in range(K)
        ])
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x, edge_index, lambda_max=2.0):
        n = x.shape[0]
        # Build sparse Laplacian
        L = self._compute_laplacian(edge_index, n)
        L_tilde = (2.0 / lambda_max) * L - torch.eye(n, device=x.device)

        # Chebyshev recursion
        Tx_0 = x
        out = self.lins[0](Tx_0)

        if self.K > 1:
            Tx_1 = L_tilde @ x
            out = out + self.lins[1](Tx_1)

        for k in range(2, self.K):
            Tx_2 = 2 * L_tilde @ Tx_1 - Tx_0
            out = out + self.lins[k](Tx_2)
            Tx_0, Tx_1 = Tx_1, Tx_2

        return out + self.bias

    def _compute_laplacian(self, edge_index, n):
        A = torch.zeros(n, n, device=edge_index.device)
        src, dst = edge_index[0], edge_index[1]
        A[src, dst] = 1.0
        D = torch.diag(A.sum(dim=1))
        return D - A


class ChebNet(nn.Module):
    """Multi-layer ChebNet for node classification."""

    def __init__(self, in_ch, hidden_ch, out_ch, K=3, num_layers=2, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(ChebConv(in_ch, hidden_ch, K))
        for _ in range(num_layers - 2):
            self.convs.append(ChebConv(hidden_ch, hidden_ch, K))
        self.convs.append(ChebConv(hidden_ch, out_ch, K))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)


def demo_chebyshev_polynomials():
    """Visualize Chebyshev polynomials."""
    print("=" * 60)
    print("Chebyshev Polynomials")
    print("=" * 60)

    x = np.linspace(-1, 1, 200)
    T = [np.ones_like(x), x.copy()]
    for k in range(2, 6):
        T.append(2 * x * T[-1] - T[-2])

    for k in range(6):
        print(f"  T_{k}: max={np.max(np.abs(T[k])):.4f}, "
              f"zeros={np.sum(np.abs(np.diff(np.sign(T[k]))) > 0)}")


def demo_chebnet_node_classification():
    """ChebNet for node classification on Karate Club."""
    print("\n" + "=" * 60)
    print("ChebNet Node Classification")
    print("=" * 60)

    torch.manual_seed(42)
    G = nx.karate_club_graph()
    n = G.number_of_nodes()
    edges = list(G.edges())
    src = [e[0] for e in edges] + [e[1] for e in edges]
    dst = [e[1] for e in edges] + [e[0] for e in edges]
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    x = torch.eye(n, dtype=torch.float)
    y = torch.tensor([0 if G.nodes[i].get('club', '') == 'Mr. Hi' else 1
                       for i in range(n)], dtype=torch.long)

    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[::2] = True

    for K in [1, 2, 3, 5]:
        torch.manual_seed(42)
        model = ChebNet(n, 16, 2, K=K, num_layers=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

        model.train()
        for epoch in range(200):
            optimizer.zero_grad()
            out = model(x, edge_index)
            loss = F.cross_entropy(out[train_mask], y[train_mask])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            pred = model(x, edge_index).argmax(dim=1)
            test_acc = (pred[~train_mask] == y[~train_mask]).float().mean()
        print(f"  K={K}: Test Accuracy = {test_acc:.4f}")


def demo_filter_localization():
    """Show how K controls filter localization."""
    print("\n" + "=" * 60)
    print("Filter Localization (K-hop)")
    print("=" * 60)

    G = nx.path_graph(20)
    A = nx.adjacency_matrix(G).toarray().astype(float)
    D = np.diag(A.sum(axis=1))
    L = D - A
    n = 20
    lmax = np.max(np.linalg.eigvalsh(L))
    L_tilde = 2 * L / lmax - np.eye(n)

    # Impulse at center node
    impulse = np.zeros(n)
    impulse[10] = 1.0

    for K in [1, 2, 3, 5, 10]:
        T0 = impulse.copy()
        response = T0.copy()
        if K > 1:
            T1 = L_tilde @ impulse
            response += T1
        T_prev, T_curr = T0, T1 if K > 1 else T0
        for k in range(2, K):
            T_next = 2 * L_tilde @ T_curr - T_prev
            response += T_next
            T_prev, T_curr = T_curr, T_next

        nonzero = np.where(np.abs(response) > 1e-10)[0]
        spread = nonzero[-1] - nonzero[0] + 1 if len(nonzero) > 0 else 0
        print(f"  K={K:2d}: spread={spread} nodes, "
              f"support=[{nonzero[0] if len(nonzero)>0 else '-'}, "
              f"{nonzero[-1] if len(nonzero)>0 else '-'}]")


if __name__ == "__main__":
    demo_chebyshev_polynomials()
    demo_chebnet_node_classification()
    demo_filter_localization()
