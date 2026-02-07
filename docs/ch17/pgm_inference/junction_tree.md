# Junction Tree Algorithm

## Overview

The **Junction Tree Algorithm** extends exact inference to arbitrary graphical models by transforming the original graph into a tree of clusters (cliques). Once the junction tree is constructed, belief propagation on this tree yields exact marginals for all variables, regardless of the original graph's structure.

The junction tree algorithm is the most general exact inference method for discrete graphical models. Its complexity is determined by the treewidth of the graph — the same quantity that governs variable elimination complexity.

## Motivation

Belief propagation is exact on trees but can fail on graphs with cycles. Variable elimination handles arbitrary graphs but answers only one query at a time. The junction tree algorithm combines the best of both:

- **Handles cycles**: by clustering nodes into super-nodes that form a tree
- **All queries at once**: by running BP on the cluster tree
- **Exact**: provably correct marginals for all variables

## Algorithm Steps

### Step 1: Moralization (for Directed Models)

If starting from a Bayesian network, first convert to an undirected model:

1. For each node, connect all its parents with undirected edges ("marry the parents")
2. Replace all directed edges with undirected edges

### Step 2: Triangulation

A graph is **triangulated** (chordal) if every cycle of length 4 or more has a chord (an edge connecting two non-adjacent nodes in the cycle). Triangulation adds fill-in edges to ensure this property.

The triangulated graph determines the cliques that will form the junction tree nodes. Finding the optimal triangulation (minimizing the maximum clique size) is NP-hard, but heuristics like minimum-degree elimination work well.

```python
import torch
import networkx as nx
from typing import Dict, List, Set, Tuple, Optional
from itertools import combinations


def triangulate(graph: nx.Graph, 
                elimination_order: List[str] = None) -> Tuple[nx.Graph, List[str]]:
    """
    Triangulate an undirected graph using elimination ordering.
    
    Returns the triangulated graph and the elimination order used.
    """
    G = graph.copy()
    nodes = list(G.nodes())
    
    if elimination_order is None:
        # Min-degree heuristic
        elimination_order = []
        remaining = set(nodes)
        temp_G = G.copy()
        
        while remaining:
            min_node = min(remaining,
                          key=lambda v: temp_G.degree(v) if v in temp_G else 0)
            elimination_order.append(min_node)
            remaining.remove(min_node)
            
            if min_node in temp_G:
                neighbors = list(temp_G.neighbors(min_node))
                # Add fill-in edges
                for i, n1 in enumerate(neighbors):
                    for n2 in neighbors[i + 1:]:
                        if not G.has_edge(n1, n2):
                            G.add_edge(n1, n2)
                        if not temp_G.has_edge(n1, n2):
                            temp_G.add_edge(n1, n2)
                temp_G.remove_node(min_node)
    
    return G, elimination_order


def find_cliques(triangulated_graph: nx.Graph) -> List[Set[str]]:
    """Find all maximal cliques in a triangulated graph."""
    return [set(c) for c in nx.find_cliques(triangulated_graph)]
```

### Step 3: Identify Cliques

Find all **maximal cliques** of the triangulated graph. Each clique becomes a node in the junction tree.

### Step 4: Build the Junction Tree

Connect cliques to form a tree that satisfies the **running intersection property**: for any variable $X$, the set of clique nodes containing $X$ forms a connected subtree.

This is achieved by finding a **maximum weight spanning tree** over the clique graph, where the weight of an edge between cliques $C_i$ and $C_j$ is $|C_i \cap C_j|$ (the size of the separator).

```python
def build_junction_tree(cliques: List[Set[str]]) -> nx.Graph:
    """
    Build a junction tree from cliques.
    
    Uses maximum weight spanning tree to ensure the
    running intersection property.
    """
    n_cliques = len(cliques)
    
    # Build weighted clique graph
    clique_graph = nx.Graph()
    for i in range(n_cliques):
        clique_graph.add_node(i, variables=cliques[i])
    
    for i in range(n_cliques):
        for j in range(i + 1, n_cliques):
            separator = cliques[i] & cliques[j]
            if separator:
                # Weight = size of separator (maximize overlap)
                clique_graph.add_edge(i, j, weight=len(separator),
                                      separator=separator)
    
    # Maximum weight spanning tree
    junction_tree = nx.maximum_spanning_tree(clique_graph)
    
    # Copy node attributes
    for node in junction_tree.nodes():
        junction_tree.nodes[node]['variables'] = cliques[node]
    
    # Add separator info to edges
    for u, v in junction_tree.edges():
        sep = cliques[u] & cliques[v]
        junction_tree.edges[u, v]['separator'] = sep
    
    return junction_tree
```

### Step 5: Initialize Potentials

Assign each factor from the original model to exactly one clique that contains all variables in the factor's scope. The clique potential is initialized as the product of all assigned factors.

### Step 6: Message Passing (Calibration)

Run belief propagation on the junction tree. Messages between adjacent cliques $C_i$ and $C_j$ with separator $S_{ij} = C_i \cap C_j$:

$$\delta_{C_i \to C_j}(S_{ij}) = \sum_{C_i \setminus S_{ij}} \Phi_{C_i}(C_i) \prod_{k \in \mathcal{N}(i) \setminus \{j\}} \delta_{C_k \to C_i}(S_{ki})$$

where $\Phi_{C_i}$ is the clique potential.

After calibration, the marginal over clique $C_i$ is:

$$P(C_i) \propto \Phi_{C_i}(C_i) \prod_{k \in \mathcal{N}(i)} \delta_{C_k \to C_i}(S_{ki})$$

### Complete Implementation

```python
class JunctionTree:
    """
    Junction Tree algorithm for exact inference.
    
    Transforms an arbitrary graphical model into a tree of cliques,
    then runs belief propagation on the clique tree.
    """
    
    def __init__(self, variables: List[str],
                 cardinalities: Dict[str, int],
                 factors: List[Tuple[List[str], torch.Tensor]]):
        """
        Args:
            variables: All variable names
            cardinalities: Variable cardinalities
            factors: List of (variable_list, values_tensor) pairs
        """
        self.variables = variables
        self.cardinalities = cardinalities
        self.factors = factors
        
        self.cliques: List[Set[str]] = []
        self.tree: Optional[nx.Graph] = None
        self.clique_potentials: Dict[int, torch.Tensor] = {}
    
    def build(self):
        """Construct the junction tree."""
        # Build interaction graph
        interaction = nx.Graph()
        interaction.add_nodes_from(self.variables)
        for var_list, _ in self.factors:
            for i, v1 in enumerate(var_list):
                for v2 in var_list[i + 1:]:
                    interaction.add_edge(v1, v2)
        
        # Triangulate
        triangulated, elim_order = triangulate(interaction)
        
        # Find cliques
        self.cliques = find_cliques(triangulated)
        
        # Build junction tree
        self.tree = build_junction_tree(self.cliques)
        
        # Initialize clique potentials
        self._assign_factors()
    
    def _assign_factors(self):
        """Assign each factor to a clique containing its scope."""
        from itertools import product as cart_prod
        
        # Initialize all clique potentials to 1
        for i, clique in enumerate(self.cliques):
            clique_vars = sorted(clique)
            shape = tuple(self.cardinalities[v] for v in clique_vars)
            self.clique_potentials[i] = torch.ones(shape)
        
        # Assign each factor to the smallest containing clique
        for var_list, values in self.factors:
            scope = set(var_list)
            
            # Find containing clique
            for i, clique in enumerate(self.cliques):
                if scope.issubset(clique):
                    # Multiply factor into clique potential
                    clique_vars = sorted(clique)
                    shape = tuple(self.cardinalities[v] for v in clique_vars)
                    
                    for assignment in cart_prod(
                        *[range(self.cardinalities[v]) for v in clique_vars]
                    ):
                        assign_dict = dict(zip(clique_vars, assignment))
                        factor_idx = tuple(assign_dict[v] for v in var_list)
                        factor_val = values[factor_idx].item()
                        self.clique_potentials[i][assignment] *= factor_val
                    
                    break
    
    def calibrate(self) -> Dict[str, torch.Tensor]:
        """
        Run message passing on the junction tree and return marginals.
        """
        if self.tree is None:
            self.build()
        
        # Two-pass message schedule
        root = 0
        
        # Collect messages (leaves -> root)
        messages = {}
        visited = set()
        
        def collect(node, parent):
            visited.add(node)
            for neighbor in self.tree.neighbors(node):
                if neighbor not in visited:
                    collect(neighbor, node)
            
            if parent is not None:
                sep = self.tree.edges[node, parent]['separator']
                msg = self._compute_message(node, parent, sep, messages)
                messages[(node, parent)] = msg
        
        collect(root, None)
        
        # Distribute messages (root -> leaves)
        visited2 = set()
        
        def distribute(node):
            visited2.add(node)
            for neighbor in self.tree.neighbors(node):
                if neighbor not in visited2:
                    sep = self.tree.edges[node, neighbor]['separator']
                    msg = self._compute_message(node, neighbor, sep, messages)
                    messages[(node, neighbor)] = msg
                    distribute(neighbor)
        
        distribute(root)
        
        # Compute marginals
        return self._compute_marginals(messages)
    
    def _compute_message(self, source: int, target: int,
                         separator: Set[str],
                         messages: dict) -> torch.Tensor:
        """Compute message from source clique to target clique."""
        from itertools import product as cart_prod
        
        clique_vars = sorted(self.cliques[source])
        sep_vars = sorted(separator)
        sum_vars = [v for v in clique_vars if v not in separator]
        
        sep_shape = tuple(self.cardinalities[v] for v in sep_vars)
        msg = torch.zeros(sep_shape)
        
        sum_cards = [self.cardinalities[v] for v in sum_vars]
        
        for sep_vals in cart_prod(*[range(self.cardinalities[v]) for v in sep_vars]):
            total = 0.0
            for sum_vals in cart_prod(*[range(c) for c in sum_cards]):
                assign = dict(zip(sep_vars, sep_vals))
                assign.update(dict(zip(sum_vars, sum_vals)))
                
                clique_idx = tuple(assign[v] for v in clique_vars)
                val = self.clique_potentials[source][clique_idx].item()
                
                # Multiply by incoming messages (except from target)
                for neighbor in self.tree.neighbors(source):
                    if neighbor != target and (neighbor, source) in messages:
                        sep_n = self.tree.edges[neighbor, source]['separator']
                        sep_n_vars = sorted(sep_n)
                        n_idx = tuple(assign[v] for v in sep_n_vars)
                        val *= messages[(neighbor, source)][n_idx].item()
                
                total += val
            
            msg[sep_vals] = total
        
        return msg
    
    def _compute_marginals(self, messages) -> Dict[str, torch.Tensor]:
        """Compute variable marginals from calibrated clique potentials."""
        from itertools import product as cart_prod
        
        marginals = {}
        
        for var in self.variables:
            # Find a clique containing this variable
            for i, clique in enumerate(self.cliques):
                if var in clique:
                    clique_vars = sorted(clique)
                    other_vars = [v for v in clique_vars if v != var]
                    other_cards = [self.cardinalities[v] for v in other_vars]
                    
                    card = self.cardinalities[var]
                    marg = torch.zeros(card)
                    
                    for val in range(card):
                        total = 0.0
                        for other_vals in cart_prod(*[range(c) for c in other_cards]):
                            assign = dict(zip(other_vars, other_vals))
                            assign[var] = val
                            
                            clique_idx = tuple(assign[v] for v in clique_vars)
                            t = self.clique_potentials[i][clique_idx].item()
                            
                            for neighbor in self.tree.neighbors(i):
                                if (neighbor, i) in messages:
                                    sep = self.tree.edges[neighbor, i]['separator']
                                    sep_vars = sorted(sep)
                                    n_idx = tuple(assign[v] for v in sep_vars)
                                    t *= messages[(neighbor, i)][n_idx].item()
                            
                            total += t
                        marg[val] = total
                    
                    marg = marg / marg.sum()
                    marginals[var] = marg
                    break
        
        return marginals
```

## The Running Intersection Property

The correctness of the junction tree algorithm depends on the **running intersection property (RIP)**: for every variable $X$, the set of clique nodes containing $X$ forms a connected subtree of the junction tree.

This property ensures that information about $X$ can flow between any two cliques that share $X$ through a connected path of cliques that all contain $X$. Without RIP, messages about $X$ could take detours through cliques that have already marginalized out $X$, losing information.

## Complexity Analysis

The complexity of the junction tree algorithm is determined by the **treewidth** $w$ of the original graph:

$$\text{Time}: O(n \cdot d^{w+1}), \qquad \text{Space}: O(n \cdot d^{w+1})$$

where $d$ is the maximum cardinality. The treewidth equals the size of the largest clique in the optimal triangulation minus 1.

| Graph Structure | Treewidth | Junction Tree Clique Size |
|----------------|-----------|--------------------------|
| Chain | 1 | 2 variables per clique |
| Tree | 1 | 2 variables per clique |
| Grid ($n \times n$) | $n$ | Up to $n + 1$ variables |
| Complete graph | $n - 1$ | Single clique with all variables |

## When to Use Junction Trees

The junction tree algorithm is the method of choice when:

- The graph has moderate treewidth (say, $w \leq 15$ for binary variables)
- Multiple queries are needed (the junction tree is built once and reused)
- Exact answers are required (no approximation error)

For graphs with large treewidth, approximate methods (loopy BP, variational inference, sampling) become necessary.

## Summary

| Concept | Description |
|---------|-------------|
| **Junction Tree** | Tree of cliques obtained from triangulated graph |
| **Moralization** | Convert directed model to undirected |
| **Triangulation** | Add edges to make graph chordal |
| **Running Intersection** | Variable appears in connected subtree of clique tree |
| **Calibration** | Message passing on the clique tree |
| **Treewidth** | Determines complexity; size of largest clique minus 1 |

## Quantitative Finance Application

The junction tree algorithm enables exact computation of joint default probabilities in structured credit portfolios. Consider a Collateralized Debt Obligation (CDO) where firms are grouped into sectors with cross-sector dependencies. The graph has sector-level cliques (firms within a sector plus the sector factor) connected by macro factors. If the number of sectors is moderate and the within-sector structure is tree-like, the junction tree has manageable clique sizes, enabling exact loss distribution computation for tranche pricing — avoiding the Monte Carlo sampling noise that plagues numerical CDO valuation.
