"""
Chapter 29.3.6: Graph Attention Network (GAT)
Multi-head attention on graphs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx


class GATConvManual(nn.Module):
    """GAT convolution layer with multi-head attention."""

    def __init__(self, in_ch, out_ch, heads=1, concat=True,
                 dropout=0.0, negative_slope=0.2):
        super().__init__()
        self.heads = heads
        self.out_ch = out_ch
        self.concat = concat
        self.dropout = dropout

        self.W = nn.Linear(in_ch, heads * out_ch, bias=False)
        self.att = nn.Parameter(torch.randn(heads, 2 * out_ch))
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.bias = nn.Parameter(torch.zeros(heads * out_ch if concat else out_ch))

        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.att.unsqueeze(0))

    def forward(self, x, edge_index):
        n = x.shape[0]
        src, dst = edge_index[0], edge_index[1]

        # Add self-loops
        loop = torch.arange(n, device=x.device)
        src = torch.cat([src, loop])
        dst = torch.cat([dst, loop])

        # Linear transform
        h = self.W(x).view(n, self.heads, self.out_ch)

        # Attention scores
        h_src = h[src]  # [E, heads, out_ch]
        h_dst = h[dst]
        cat = torch.cat([h_src, h_dst], dim=-1)  # [E, heads, 2*out_ch]
        e = (cat * self.att.unsqueeze(0)).sum(dim=-1)  # [E, heads]
        e = self.leaky_relu(e)

        # Softmax per dst node
        e_max = torch.full((n, self.heads), float('-inf'), device=x.device)
        e_max.scatter_reduce_(0, dst.unsqueeze(1).expand_as(e), e, reduce='amax')
        alpha = torch.exp(e - e_max[dst])
        alpha_sum = torch.zeros(n, self.heads, device=x.device)
        alpha_sum.scatter_add_(0, dst.unsqueeze(1).expand_as(alpha), alpha)
        alpha = alpha / alpha_sum[dst].clamp(min=1e-10)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # Weighted aggregation
        weighted = h_src * alpha.unsqueeze(-1)
        out = torch.zeros(n, self.heads, self.out_ch, device=x.device)
        out.scatter_add_(0, dst.unsqueeze(1).unsqueeze(2).expand_as(weighted), weighted)

        if self.concat:
            out = out.view(n, self.heads * self.out_ch)
        else:
            out = out.mean(dim=1)

        return out + self.bias


class GAT(nn.Module):
    """Multi-layer GAT."""

    def __init__(self, in_ch, hidden_ch, out_ch, heads=4, num_layers=2, dropout=0.6):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATConvManual(in_ch, hidden_ch, heads=heads, concat=True, dropout=dropout))
        for _ in range(num_layers - 2):
            self.convs.append(GATConvManual(hidden_ch * heads, hidden_ch, heads=heads, concat=True, dropout=dropout))
        self.convs.append(GATConvManual(hidden_ch * heads, out_ch, heads=1, concat=False, dropout=dropout))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)


def demo_gat():
    """GAT on Karate Club."""
    print("=" * 60)
    print("GAT: Karate Club Node Classification")
    print("=" * 60)

    torch.manual_seed(42)
    G = nx.karate_club_graph()
    n = G.number_of_nodes()
    edges = list(G.edges())
    src = [e[0] for e in edges] + [e[1] for e in edges]
    dst = [e[1] for e in edges] + [e[0] for e in edges]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    x = torch.eye(n)
    y = torch.tensor([0 if G.nodes[i].get('club','')=='Mr. Hi' else 1
                       for i in range(n)], dtype=torch.long)
    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[[0,1,2,3,33,32,31,30]] = True

    for heads in [1, 4, 8]:
        torch.manual_seed(42)
        model = GAT(n, 8, 2, heads=heads, num_layers=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

        model.train()
        for epoch in range(300):
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x, edge_index)[train_mask], y[train_mask])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            pred = model(x, edge_index).argmax(1)
            test_acc = (pred[~train_mask] == y[~train_mask]).float().mean()
        print(f"  heads={heads}: Test Acc = {test_acc:.4f}")


def demo_attention_visualization():
    """Visualize attention coefficients."""
    print("\n" + "=" * 60)
    print("Attention Coefficient Analysis")
    print("=" * 60)

    torch.manual_seed(42)
    n = 6
    edge_index = torch.tensor([
        [0,1,0,2,1,2,1,3,2,4,3,5,4,5],
        [1,0,2,0,2,1,3,1,4,2,5,3,5,4]], dtype=torch.long)
    x = torch.randn(n, 4)

    layer = GATConvManual(4, 4, heads=1, concat=False)
    layer.eval()

    with torch.no_grad():
        # Manually extract attention
        src, dst = edge_index[0], edge_index[1]
        loop = torch.arange(n)
        src_all = torch.cat([src, loop])
        dst_all = torch.cat([dst, loop])

        h = layer.W(x).view(n, 1, 4)
        h_src = h[src_all]
        h_dst = h[dst_all]
        cat = torch.cat([h_src, h_dst], dim=-1)
        e = (cat * layer.att.unsqueeze(0)).sum(dim=-1)
        e = layer.leaky_relu(e)

        e_max = torch.full((n, 1), float('-inf'))
        e_max.scatter_reduce_(0, dst_all.unsqueeze(1).expand_as(e), e, reduce='amax')
        alpha = torch.exp(e - e_max[dst_all])
        alpha_sum = torch.zeros(n, 1)
        alpha_sum.scatter_add_(0, dst_all.unsqueeze(1).expand_as(alpha), alpha)
        alpha = alpha / alpha_sum[dst_all].clamp(min=1e-10)

    print("Edge attention coefficients:")
    for i in range(edge_index.shape[1]):
        print(f"  {src[i].item()} -> {dst[i].item()}: {alpha[i, 0].item():.4f}")


if __name__ == "__main__":
    demo_gat()
    demo_attention_visualization()
