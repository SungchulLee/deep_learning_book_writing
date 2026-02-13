"""
Chapter 29.2.2: Aggregation Functions
Various aggregation strategies for message passing GNNs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


# ============================================================
# 1. Standard Aggregation Functions
# ============================================================

class Aggregator:
    """Collection of aggregation functions for message passing."""

    @staticmethod
    def scatter_add(src: torch.Tensor, index: torch.Tensor,
                     num_nodes: int) -> torch.Tensor:
        out = torch.zeros(num_nodes, src.shape[1], device=src.device)
        out.scatter_add_(0, index.unsqueeze(1).expand_as(src), src)
        return out

    @staticmethod
    def scatter_count(index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        deg = torch.zeros(num_nodes, device=index.device)
        deg.scatter_add_(0, index, torch.ones_like(index, dtype=torch.float))
        return deg.clamp(min=1)

    @staticmethod
    def sum_aggr(messages: torch.Tensor, dst: torch.Tensor,
                  num_nodes: int) -> torch.Tensor:
        return Aggregator.scatter_add(messages, dst, num_nodes)

    @staticmethod
    def mean_aggr(messages: torch.Tensor, dst: torch.Tensor,
                   num_nodes: int) -> torch.Tensor:
        out = Aggregator.scatter_add(messages, dst, num_nodes)
        deg = Aggregator.scatter_count(dst, num_nodes)
        return out / deg.unsqueeze(1)

    @staticmethod
    def max_aggr(messages: torch.Tensor, dst: torch.Tensor,
                  num_nodes: int) -> torch.Tensor:
        out = torch.full((num_nodes, messages.shape[1]), float('-inf'),
                          device=messages.device)
        out.scatter_reduce_(0, dst.unsqueeze(1).expand_as(messages),
                            messages, reduce='amax')
        out[out == float('-inf')] = 0
        return out

    @staticmethod
    def min_aggr(messages: torch.Tensor, dst: torch.Tensor,
                  num_nodes: int) -> torch.Tensor:
        out = torch.full((num_nodes, messages.shape[1]), float('inf'),
                          device=messages.device)
        out.scatter_reduce_(0, dst.unsqueeze(1).expand_as(messages),
                            messages, reduce='amin')
        out[out == float('inf')] = 0
        return out

    @staticmethod
    def std_aggr(messages: torch.Tensor, dst: torch.Tensor,
                  num_nodes: int) -> torch.Tensor:
        mean = Aggregator.mean_aggr(messages, dst, num_nodes)
        sq_diff = (messages - mean[dst]) ** 2
        variance = Aggregator.mean_aggr(sq_diff, dst, num_nodes)
        return torch.sqrt(variance + 1e-8)


def demo_standard_aggregations():
    """Compare standard aggregation functions."""
    print("=" * 60)
    print("Standard Aggregation Functions")
    print("=" * 60)

    torch.manual_seed(42)

    # Graph: node 0 has neighbors {1, 2, 3}, node 4 has neighbor {3}
    edge_index = torch.tensor([
        [1, 2, 3, 3],
        [0, 0, 0, 4]], dtype=torch.long)
    num_nodes = 5
    messages = torch.tensor([
        [1.0, 2.0],  # from 1 to 0
        [3.0, 1.0],  # from 2 to 0
        [2.0, 3.0],  # from 3 to 0
        [2.0, 3.0],  # from 3 to 4
    ])
    dst = edge_index[1]

    print("Messages (to node 0): [[1,2], [3,1], [2,3]]")
    print("Messages (to node 4): [[2,3]]")

    for name, fn in [('Sum', Aggregator.sum_aggr),
                     ('Mean', Aggregator.mean_aggr),
                     ('Max', Aggregator.max_aggr),
                     ('Min', Aggregator.min_aggr),
                     ('Std', Aggregator.std_aggr)]:
        result = fn(messages, dst, num_nodes)
        print(f"  {name:5s}: node0={result[0].tolist()}, "
              f"node4={result[4].tolist()}")


# ============================================================
# 2. Attention-Weighted Aggregation
# ============================================================

class AttentionAggregation(nn.Module):
    """Attention-weighted aggregation (simplified GAT-style)."""

    def __init__(self, channels: int, heads: int = 1):
        super().__init__()
        self.heads = heads
        self.head_dim = channels // heads
        self.att = nn.Parameter(torch.randn(heads, 2 * self.head_dim))
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        num_nodes = x.shape[0]

        # Compute attention scores
        x_src = x[src]
        x_dst = x[dst]
        cat = torch.cat([x_src, x_dst], dim=-1)  # [E, 2*channels]

        # Reshape for multi-head
        cat = cat.view(-1, self.heads, 2 * self.head_dim)
        alpha = (cat * self.att.unsqueeze(0)).sum(dim=-1)  # [E, heads]
        alpha = self.leaky_relu(alpha)

        # Softmax per target node
        alpha_max = torch.zeros(num_nodes, self.heads, device=x.device)
        alpha_max.scatter_reduce_(0, dst.unsqueeze(1).expand_as(alpha),
                                   alpha, reduce='amax')
        alpha = alpha - alpha_max[dst]
        alpha = torch.exp(alpha)

        alpha_sum = torch.zeros(num_nodes, self.heads, device=x.device)
        alpha_sum.scatter_add_(0, dst.unsqueeze(1).expand_as(alpha), alpha)
        alpha = alpha / alpha_sum[dst].clamp(min=1e-10)

        # Weighted aggregation
        x_src_h = x_src.view(-1, self.heads, self.head_dim)
        weighted = x_src_h * alpha.unsqueeze(-1)

        out = torch.zeros(num_nodes, self.heads, self.head_dim, device=x.device)
        out.scatter_add_(0, dst.unsqueeze(1).unsqueeze(2).expand_as(weighted), weighted)

        return out.view(num_nodes, -1)


def demo_attention_aggregation():
    """Demonstrate attention-weighted aggregation."""
    print("\n" + "=" * 60)
    print("Attention-Weighted Aggregation")
    print("=" * 60)

    torch.manual_seed(42)
    edge_index = torch.tensor([
        [0, 1, 2, 3, 1, 0, 2, 1, 3, 2],
        [1, 0, 1, 2, 2, 2, 3, 3, 0, 0]], dtype=torch.long)
    x = torch.randn(4, 8)

    attn = AttentionAggregation(8, heads=2)
    out = attn(x, edge_index)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output[0]: {out[0, :4].tolist()}")


# ============================================================
# 3. Multi-Aggregation
# ============================================================

class MultiAggregation(nn.Module):
    """Combine multiple aggregation functions."""

    def __init__(self, aggregators: list = None):
        super().__init__()
        self.aggregators = aggregators or ['sum', 'mean', 'max', 'std']
        self.agg_fns = {
            'sum': Aggregator.sum_aggr,
            'mean': Aggregator.mean_aggr,
            'max': Aggregator.max_aggr,
            'min': Aggregator.min_aggr,
            'std': Aggregator.std_aggr,
        }

    def forward(self, messages, dst, num_nodes):
        results = []
        for name in self.aggregators:
            results.append(self.agg_fns[name](messages, dst, num_nodes))
        return torch.cat(results, dim=-1)


def demo_multi_aggregation():
    """Demonstrate multi-aggregation."""
    print("\n" + "=" * 60)
    print("Multi-Aggregation")
    print("=" * 60)

    messages = torch.tensor([
        [1.0, 2.0], [3.0, 1.0], [2.0, 3.0], [2.0, 3.0]])
    dst = torch.tensor([0, 0, 0, 4])
    num_nodes = 5

    multi_agg = MultiAggregation(['sum', 'mean', 'max', 'std'])
    out = multi_agg(messages, dst, num_nodes)
    print(f"Message dim: 2, Aggregators: 4")
    print(f"Output dim: {out.shape[1]} (2 x 4)")
    print(f"Node 0: {out[0].tolist()}")


# ============================================================
# 4. PNA (Principal Neighborhood Aggregation)
# ============================================================

class PNAAggregation(nn.Module):
    """
    Principal Neighborhood Aggregation.
    Combines multiple aggregators with degree-based scalers.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 aggregators=None, scalers=None):
        super().__init__()
        self.aggregators = aggregators or ['mean', 'max', 'min', 'std']
        self.scalers = scalers or ['identity', 'amplification', 'attenuation']

        total_dim = in_channels * len(self.aggregators) * len(self.scalers)
        self.lin = nn.Linear(total_dim, out_channels)

    def forward(self, messages, dst, num_nodes, deg=None):
        agg_fns = {
            'mean': Aggregator.mean_aggr,
            'max': Aggregator.max_aggr,
            'min': Aggregator.min_aggr,
            'std': Aggregator.std_aggr,
            'sum': Aggregator.sum_aggr,
        }

        if deg is None:
            deg = Aggregator.scatter_count(dst, num_nodes)

        results = []
        for agg_name in self.aggregators:
            agg_out = agg_fns[agg_name](messages, dst, num_nodes)
            for scaler_name in self.scalers:
                if scaler_name == 'identity':
                    results.append(agg_out)
                elif scaler_name == 'amplification':
                    scale = torch.log(deg + 1).unsqueeze(1)
                    results.append(agg_out * scale)
                elif scaler_name == 'attenuation':
                    scale = 1.0 / (deg + 1).unsqueeze(1)
                    results.append(agg_out * scale)

        combined = torch.cat(results, dim=-1)
        return self.lin(combined)


def demo_pna():
    """Demonstrate PNA aggregation."""
    print("\n" + "=" * 60)
    print("PNA Aggregation")
    print("=" * 60)

    torch.manual_seed(42)
    messages = torch.randn(10, 4)
    dst = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2, 3])
    num_nodes = 5

    pna = PNAAggregation(4, 8)
    out = pna(messages, dst, num_nodes)
    print(f"Input: {messages.shape[0]} messages, dim={messages.shape[1]}")
    print(f"4 aggregators x 3 scalers = 12 intermediate outputs")
    print(f"Output shape: {out.shape}")


# ============================================================
# 5. Expressiveness Comparison
# ============================================================

def demo_expressiveness():
    """Show where different aggregators fail to distinguish graphs."""
    print("\n" + "=" * 60)
    print("Aggregation Expressiveness Comparison")
    print("=" * 60)

    # Case 1: Different neighborhood sizes, same feature distribution
    print("\nCase 1: Different #neighbors, same feature mean")
    msg_a = torch.tensor([[1.0], [1.0], [1.0]])  # 3 neighbors, all = 1
    msg_b = torch.tensor([[1.0]])                  # 1 neighbor, = 1
    dst_a = torch.tensor([0, 0, 0])
    dst_b = torch.tensor([0])

    for name, fn in [('Sum', Aggregator.sum_aggr),
                     ('Mean', Aggregator.mean_aggr),
                     ('Max', Aggregator.max_aggr)]:
        ra = fn(msg_a, dst_a, 1)[0].item()
        rb = fn(msg_b, dst_b, 1)[0].item()
        dist = "SAME" if abs(ra - rb) < 1e-6 else "DIFF"
        print(f"  {name}: A={ra:.1f}, B={rb:.1f} -> {dist}")

    # Case 2: Same sum, different distribution
    print("\nCase 2: Same sum, different distribution")
    msg_a = torch.tensor([[1.0], [1.0], [1.0]])  # [1,1,1]
    msg_b = torch.tensor([[0.0], [0.0], [3.0]])  # [0,0,3]
    dst = torch.tensor([0, 0, 0])

    for name, fn in [('Sum', Aggregator.sum_aggr),
                     ('Mean', Aggregator.mean_aggr),
                     ('Max', Aggregator.max_aggr)]:
        ra = fn(msg_a, dst, 1)[0].item()
        rb = fn(msg_b, dst, 1)[0].item()
        dist = "SAME" if abs(ra - rb) < 1e-6 else "DIFF"
        print(f"  {name}: A={ra:.1f}, B={rb:.1f} -> {dist}")

    # Case 3: Same max, different multiplicity
    print("\nCase 3: Same max, different multiplicity")
    msg_a = torch.tensor([[3.0], [3.0], [3.0]])
    msg_b = torch.tensor([[3.0], [1.0], [1.0]])
    for name, fn in [('Sum', Aggregator.sum_aggr),
                     ('Mean', Aggregator.mean_aggr),
                     ('Max', Aggregator.max_aggr)]:
        ra = fn(msg_a, dst, 1)[0].item()
        rb = fn(msg_b, dst, 1)[0].item()
        dist = "SAME" if abs(ra - rb) < 1e-6 else "DIFF"
        print(f"  {name}: A={ra:.1f}, B={rb:.1f} -> {dist}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    demo_standard_aggregations()
    demo_attention_aggregation()
    demo_multi_aggregation()
    demo_pna()
    demo_expressiveness()
