"""
Chapter 29.1.1: Graph Basics
Fundamental graph concepts and data structures with Python implementations.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Set


# ============================================================
# 1. Basic Graph Class Implementation
# ============================================================

class SimpleGraph:
    """
    A basic graph implementation using adjacency list representation.
    Supports both directed and undirected graphs with optional weights.
    """

    def __init__(self, directed: bool = False, weighted: bool = False):
        self.directed = directed
        self.weighted = weighted
        self.adj_list: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        self.nodes: Set[int] = set()

    def add_node(self, node: int):
        """Add a node to the graph."""
        self.nodes.add(node)
        if node not in self.adj_list:
            self.adj_list[node] = []

    def add_edge(self, u: int, v: int, weight: float = 1.0):
        """Add an edge between nodes u and v."""
        self.nodes.add(u)
        self.nodes.add(v)
        self.adj_list[u].append((v, weight))
        if not self.directed:
            self.adj_list[v].append((u, weight))

    def get_neighbors(self, node: int) -> List[int]:
        """Return the neighbors of a node."""
        return [v for v, _ in self.adj_list[node]]

    def degree(self, node: int) -> int:
        """Return the degree of a node."""
        return len(self.adj_list[node])

    def in_degree(self, node: int) -> int:
        """Return in-degree for directed graphs."""
        if not self.directed:
            return self.degree(node)
        count = 0
        for u in self.nodes:
            for v, _ in self.adj_list[u]:
                if v == node:
                    count += 1
        return count

    def out_degree(self, node: int) -> int:
        """Return out-degree for directed graphs."""
        return len(self.adj_list[node])

    def num_nodes(self) -> int:
        return len(self.nodes)

    def num_edges(self) -> int:
        total = sum(len(neighbors) for neighbors in self.adj_list.values())
        return total if self.directed else total // 2

    def has_edge(self, u: int, v: int) -> bool:
        return any(neighbor == v for neighbor, _ in self.adj_list[u])

    def get_edge_weight(self, u: int, v: int) -> Optional[float]:
        for neighbor, weight in self.adj_list[u]:
            if neighbor == v:
                return weight
        return None

    def to_adjacency_matrix(self) -> np.ndarray:
        """Convert to adjacency matrix representation."""
        n = max(self.nodes) + 1
        matrix = np.zeros((n, n))
        for u in self.adj_list:
            for v, w in self.adj_list[u]:
                matrix[u][v] = w
        return matrix

    def to_edge_list(self) -> List[Tuple[int, int, float]]:
        """Convert to edge list representation."""
        edges = []
        seen = set()
        for u in self.adj_list:
            for v, w in self.adj_list[u]:
                edge = (min(u, v), max(u, v)) if not self.directed else (u, v)
                if edge not in seen:
                    edges.append((u, v, w))
                    if not self.directed:
                        seen.add(edge)
        return edges

    def __repr__(self):
        graph_type = "Directed" if self.directed else "Undirected"
        return f"{graph_type}Graph(nodes={self.num_nodes()}, edges={self.num_edges()})"


# ============================================================
# 2. Demonstrate Basic Graph Operations
# ============================================================

def demo_basic_graph():
    """Demonstrate basic graph creation and operations."""
    print("=" * 60)
    print("Basic Graph Operations")
    print("=" * 60)

    # Create an undirected graph
    g = SimpleGraph(directed=False)
    edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4)]
    for u, v in edges:
        g.add_edge(u, v)

    print(f"Graph: {g}")
    print(f"Neighbors of node 1: {g.get_neighbors(1)}")
    print(f"Degree of node 1: {g.degree(1)}")
    print(f"Has edge (0, 3): {g.has_edge(0, 3)}")
    print(f"Has edge (0, 1): {g.has_edge(0, 1)}")

    # Degree distribution
    degrees = [g.degree(n) for n in g.nodes]
    print(f"\nDegree sequence: {sorted(degrees, reverse=True)}")
    print(f"Sum of degrees: {sum(degrees)}")
    print(f"2 * |E| = {2 * g.num_edges()}")
    print(f"Handshaking lemma verified: {sum(degrees) == 2 * g.num_edges()}")

    # Adjacency matrix
    adj_matrix = g.to_adjacency_matrix()
    print(f"\nAdjacency Matrix:\n{adj_matrix}")

    # Edge list
    edge_list = g.to_edge_list()
    print(f"\nEdge List: {edge_list}")

    return g


# ============================================================
# 3. Directed Graph Example
# ============================================================

def demo_directed_graph():
    """Demonstrate directed graph operations."""
    print("\n" + "=" * 60)
    print("Directed Graph Operations")
    print("=" * 60)

    dg = SimpleGraph(directed=True)
    # A simple transaction network
    transactions = [
        (0, 1, 100.0),  # Account 0 sends $100 to Account 1
        (1, 2, 50.0),   # Account 1 sends $50 to Account 2
        (2, 0, 30.0),   # Account 2 sends $30 to Account 0
        (0, 3, 75.0),   # Account 0 sends $75 to Account 3
        (3, 1, 40.0),   # Account 3 sends $40 to Account 1
    ]

    for u, v, w in transactions:
        dg.add_edge(u, v, weight=w)

    print(f"Directed Graph: {dg}")
    for node in sorted(dg.nodes):
        print(f"Node {node}: in_degree={dg.in_degree(node)}, "
              f"out_degree={dg.out_degree(node)}")

    adj_matrix = dg.to_adjacency_matrix()
    print(f"\nWeighted Adjacency Matrix (transaction amounts):\n{adj_matrix}")

    return dg


# ============================================================
# 4. NetworkX Integration
# ============================================================

def demo_networkx():
    """Demonstrate graph operations using NetworkX."""
    print("\n" + "=" * 60)
    print("NetworkX Graph Operations")
    print("=" * 60)

    # Create a graph
    G = nx.karate_club_graph()
    print(f"Karate Club Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Basic statistics
    print(f"Average degree: {2 * G.number_of_edges() / G.number_of_nodes():.2f}")
    print(f"Density: {nx.density(G):.4f}")
    print(f"Is connected: {nx.is_connected(G)}")
    print(f"Diameter: {nx.diameter(G)}")
    print(f"Average shortest path: {nx.average_shortest_path_length(G):.4f}")
    print(f"Clustering coefficient: {nx.average_clustering(G):.4f}")

    # Degree distribution
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
    print(f"\nDegree distribution statistics:")
    print(f"  Max degree: {max(degree_sequence)}")
    print(f"  Min degree: {min(degree_sequence)}")
    print(f"  Mean degree: {np.mean(degree_sequence):.2f}")
    print(f"  Median degree: {np.median(degree_sequence):.2f}")

    return G


# ============================================================
# 5. Common Graph Types
# ============================================================

def demo_graph_types():
    """Demonstrate different common graph types."""
    print("\n" + "=" * 60)
    print("Common Graph Types")
    print("=" * 60)

    # Complete graph K5
    K5 = nx.complete_graph(5)
    print(f"Complete graph K5: {K5.number_of_nodes()} nodes, "
          f"{K5.number_of_edges()} edges (expected: {5*4//2})")

    # Star graph S6
    S6 = nx.star_graph(5)  # Central node + 5 peripheral
    print(f"Star graph S6: {S6.number_of_nodes()} nodes, "
          f"{S6.number_of_edges()} edges")

    # Tree (random)
    tree = nx.random_tree(10, seed=42)
    print(f"Random tree (n=10): {tree.number_of_nodes()} nodes, "
          f"{tree.number_of_edges()} edges (n-1={10-1})")
    print(f"  Is tree: {nx.is_tree(tree)}")

    # Bipartite graph
    B = nx.complete_bipartite_graph(3, 4)
    print(f"Complete bipartite K(3,4): {B.number_of_nodes()} nodes, "
          f"{B.number_of_edges()} edges (expected: {3*4})")
    print(f"  Is bipartite: {nx.is_bipartite(B)}")

    # DAG
    dag = nx.DiGraph()
    dag.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)])
    print(f"DAG: {dag.number_of_nodes()} nodes, {dag.number_of_edges()} edges")
    print(f"  Is DAG: {nx.is_directed_acyclic_graph(dag)}")
    print(f"  Topological sort: {list(nx.topological_sort(dag))}")


# ============================================================
# 6. Financial Network Example
# ============================================================

def demo_financial_network():
    """Create and analyze a simple financial correlation network."""
    print("\n" + "=" * 60)
    print("Financial Correlation Network")
    print("=" * 60)

    # Simulate stock returns for 5 assets
    np.random.seed(42)
    n_assets = 5
    n_days = 252
    asset_names = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

    # Generate correlated returns using a factor model
    market_factor = np.random.randn(n_days) * 0.01
    returns = np.zeros((n_days, n_assets))
    betas = [1.0, 0.9, 1.1, 1.2, 1.3]
    for i in range(n_assets):
        returns[:, i] = betas[i] * market_factor + np.random.randn(n_days) * 0.005

    # Compute correlation matrix
    corr_matrix = np.corrcoef(returns.T)
    print("Correlation Matrix:")
    for i, name in enumerate(asset_names):
        row = " ".join(f"{corr_matrix[i, j]:6.3f}" for j in range(n_assets))
        print(f"  {name}: {row}")

    # Build correlation graph (threshold = 0.3)
    threshold = 0.3
    G = nx.Graph()
    for i in range(n_assets):
        G.add_node(i, name=asset_names[i])
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            if abs(corr_matrix[i, j]) > threshold:
                G.add_edge(i, j, weight=corr_matrix[i, j])

    print(f"\nCorrelation graph (threshold={threshold}):")
    print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    for u, v, data in G.edges(data=True):
        print(f"  {asset_names[u]} -- {asset_names[v]}: {data['weight']:.3f}")

    # Graph metrics
    if G.number_of_edges() > 0:
        print(f"\n  Density: {nx.density(G):.4f}")
        degrees = dict(G.degree())
        most_connected = max(degrees, key=degrees.get)
        print(f"  Most connected: {asset_names[most_connected]} "
              f"(degree={degrees[most_connected]})")

    return G


# ============================================================
# 7. Visualization
# ============================================================

def visualize_graphs():
    """Visualize different graph types."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Simple undirected graph
    G1 = nx.petersen_graph()
    nx.draw(G1, ax=axes[0, 0], with_labels=True, node_color='lightblue',
            node_size=400, font_size=8)
    axes[0, 0].set_title("Petersen Graph")

    # 2. Complete graph
    G2 = nx.complete_graph(6)
    nx.draw(G2, ax=axes[0, 1], with_labels=True, node_color='lightgreen',
            node_size=400, font_size=8)
    axes[0, 1].set_title("Complete Graph K6")

    # 3. Star graph
    G3 = nx.star_graph(8)
    pos3 = nx.spring_layout(G3, seed=42)
    nx.draw(G3, pos=pos3, ax=axes[0, 2], with_labels=True,
            node_color='lightyellow', node_size=400, font_size=8)
    axes[0, 2].set_title("Star Graph S9")

    # 4. Directed graph
    G4 = nx.DiGraph([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)])
    pos4 = nx.spring_layout(G4, seed=42)
    nx.draw(G4, pos=pos4, ax=axes[1, 0], with_labels=True,
            node_color='lightsalmon', node_size=400, font_size=8,
            arrows=True, arrowsize=15)
    axes[1, 0].set_title("Directed Graph")

    # 5. Tree
    G5 = nx.balanced_tree(2, 3)
    pos5 = nx.spring_layout(G5, seed=42)
    nx.draw(G5, pos=pos5, ax=axes[1, 1], with_labels=True,
            node_color='plum', node_size=300, font_size=7)
    axes[1, 1].set_title("Balanced Tree (r=2, h=3)")

    # 6. Bipartite graph
    G6 = nx.complete_bipartite_graph(3, 4)
    pos6 = nx.bipartite_layout(G6, nodes=range(3))
    nx.draw(G6, pos=pos6, ax=axes[1, 2], with_labels=True,
            node_color=['lightblue'] * 3 + ['lightcoral'] * 4,
            node_size=400, font_size=8)
    axes[1, 2].set_title("Bipartite Graph K(3,4)")

    plt.tight_layout()
    plt.savefig("graph_types.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("\nVisualization saved to graph_types.png")


# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":
    g = demo_basic_graph()
    dg = demo_directed_graph()
    G = demo_networkx()
    demo_graph_types()
    demo_financial_network()
    visualize_graphs()
