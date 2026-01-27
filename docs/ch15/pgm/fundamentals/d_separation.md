# D-Separation in Bayesian Networks

## Overview

**D-separation** (directional separation) is the fundamental graphical criterion for determining conditional independence in Bayesian Networks. Given a DAG structure, d-separation allows us to read off which variables are conditionally independent without computing any probabilities.

## Formal Definition

### Blocked Paths

A path between nodes $X$ and $Y$ is **blocked** by a set of nodes $Z$ if there exists a node $W$ on the path such that either:

1. **Chain or Fork**: $W$ is in $Z$ and arrows on the path meet head-to-tail or tail-to-tail at $W$
   - Chain: $A \rightarrow W \rightarrow B$ with $W \in Z$
   - Fork: $A \leftarrow W \rightarrow B$ with $W \in Z$

2. **Collider**: $W$ is a collider (arrows meet head-to-head: $A \rightarrow W \leftarrow B$) and neither $W$ nor any descendant of $W$ is in $Z$

### D-Separation

Two sets of nodes $X$ and $Y$ are **d-separated** by a set $Z$, written $\text{d-sep}(X, Y \mid Z)$, if and only if every path from any node in $X$ to any node in $Y$ is blocked by $Z$.

### The Fundamental Theorem

If $X$ and $Y$ are d-separated by $Z$ in a DAG $G$, then $X$ and $Y$ are conditionally independent given $Z$ in every distribution $P$ that factors according to $G$:

$$\text{d-sep}_G(X, Y \mid Z) \implies X \perp\!\!\!\perp_P Y \mid Z$$

Moreover, if $X$ and $Y$ are **not** d-separated by $Z$, there exists some distribution compatible with $G$ where $X$ and $Y$ are dependent given $Z$.

## The Three Building Blocks

### 1. Chain (Serial Connection)

```
A ──→ B ──→ C
```

**D-separation analysis:**
- Path $A - B - C$: $B$ is in the middle with head-to-tail arrows
- If $B \in Z$: Path is **blocked** → $A \perp\!\!\!\perp C \mid B$
- If $B \notin Z$: Path is **open** → $A \not\perp\!\!\!\perp C$

**Intuition:** Information flows from $A$ to $C$ through $B$. Observing $B$ intercepts this flow.

```python
import torch
import numpy as np

def demonstrate_chain():
    """
    Chain structure: A → B → C
    Shows blocking when B is observed.
    """
    # P(A)
    p_a = torch.tensor([0.3, 0.7])  # A=0, A=1
    
    # P(B|A) - B depends strongly on A
    p_b_given_a = torch.tensor([
        [0.9, 0.1],  # P(B|A=0)
        [0.2, 0.8]   # P(B|A=1)
    ])
    
    # P(C|B) - C depends strongly on B
    p_c_given_b = torch.tensor([
        [0.8, 0.2],  # P(C|B=0)
        [0.3, 0.7]   # P(C|B=1)
    ])
    
    # Compute P(A, B, C) = P(A) * P(B|A) * P(C|B)
    p_joint = torch.zeros(2, 2, 2)
    for a in range(2):
        for b in range(2):
            for c in range(2):
                p_joint[a, b, c] = p_a[a] * p_b_given_a[a, b] * p_c_given_b[b, c]
    
    # Check marginal dependence: P(A, C)
    p_ac = p_joint.sum(dim=1)
    p_a_marginal = p_joint.sum(dim=(1, 2))
    p_c_marginal = p_joint.sum(dim=(0, 1))
    
    p_ac_independent = torch.outer(p_a_marginal, p_c_marginal)
    
    print("Chain Structure: A → B → C")
    print("=" * 50)
    print(f"\nP(A, C):\n{p_ac}")
    print(f"\nP(A) × P(C):\n{p_ac_independent}")
    print(f"\nA and C marginally independent? {torch.allclose(p_ac, p_ac_independent, atol=0.01)}")
    
    # Check conditional independence: P(A, C | B=b) for each b
    print("\nConditional Independence given B:")
    for b in range(2):
        p_ac_given_b = p_joint[:, b, :] / p_joint[:, b, :].sum()
        p_a_given_b = p_joint[:, b, :].sum(dim=1) / p_joint[:, b, :].sum()
        p_c_given_b_marginal = p_joint[:, b, :].sum(dim=0) / p_joint[:, b, :].sum()
        
        p_ac_independent_given_b = torch.outer(p_a_given_b, p_c_given_b_marginal)
        
        is_independent = torch.allclose(p_ac_given_b, p_ac_independent_given_b, atol=0.01)
        print(f"  A ⊥ C | B={b}? {is_independent}")

demonstrate_chain()
```

### 2. Fork (Common Cause)

```
A ←── B ──→ C
```

**D-separation analysis:**
- Path $A - B - C$: $B$ is in the middle with tail-to-tail arrows (fork)
- If $B \in Z$: Path is **blocked** → $A \perp\!\!\!\perp C \mid B$
- If $B \notin Z$: Path is **open** → $A \not\perp\!\!\!\perp C$

**Intuition:** $B$ is a common cause of both $A$ and $C$. They are marginally dependent (confounded), but independent once we condition on the common cause.

```python
def demonstrate_fork():
    """
    Fork structure: A ← B → C
    Common cause creates spurious correlation.
    """
    # P(B) - the common cause
    p_b = torch.tensor([0.4, 0.6])
    
    # P(A|B)
    p_a_given_b = torch.tensor([
        [0.9, 0.1],  # P(A|B=0)
        [0.2, 0.8]   # P(A|B=1)
    ])
    
    # P(C|B)
    p_c_given_b = torch.tensor([
        [0.7, 0.3],  # P(C|B=0)
        [0.1, 0.9]   # P(C|B=1)
    ])
    
    # P(A, B, C) = P(B) * P(A|B) * P(C|B)
    p_joint = torch.zeros(2, 2, 2)
    for a in range(2):
        for b in range(2):
            for c in range(2):
                p_joint[a, b, c] = p_b[b] * p_a_given_b[b, a] * p_c_given_b[b, c]
    
    # Check marginal dependence
    p_ac = p_joint.sum(dim=1)
    p_a = p_joint.sum(dim=(1, 2))
    p_c = p_joint.sum(dim=(0, 1))
    
    print("\nFork Structure: A ← B → C")
    print("=" * 50)
    print(f"\nP(A, C):\n{p_ac}")
    print(f"\nP(A) × P(C):\n{torch.outer(p_a, p_c)}")
    print(f"\nA and C marginally independent? {torch.allclose(p_ac, torch.outer(p_a, p_c), atol=0.01)}")
    
    # Conditional independence given B
    print("\nConditional Independence given B:")
    for b in range(2):
        p_ac_given_b = p_joint[:, b, :] / p_joint[:, b, :].sum()
        p_a_given_b_val = p_joint[:, b, :].sum(dim=1) / p_joint[:, b, :].sum()
        p_c_given_b_val = p_joint[:, b, :].sum(dim=0) / p_joint[:, b, :].sum()
        
        is_independent = torch.allclose(p_ac_given_b, torch.outer(p_a_given_b_val, p_c_given_b_val), atol=0.01)
        print(f"  A ⊥ C | B={b}? {is_independent}")

demonstrate_fork()
```

### 3. Collider (V-Structure)

```
A ──→ B ←── C
```

**D-separation analysis:**
- Path $A - B - C$: $B$ is a collider (arrows meet head-to-head)
- If $B \notin Z$ (and no descendant of $B$ in $Z$): Path is **blocked** → $A \perp\!\!\!\perp C$
- If $B \in Z$ (or any descendant of $B$ in $Z$): Path is **open** → $A \not\perp\!\!\!\perp C \mid B$

**Intuition:** This is the **explaining away** effect. $A$ and $C$ are independent causes, but observing their common effect $B$ creates a dependency.

```python
def demonstrate_collider():
    """
    Collider structure: A → B ← C
    Demonstrates the explaining away effect.
    """
    # P(A) and P(C) - independent causes
    p_a = torch.tensor([0.6, 0.4])
    p_c = torch.tensor([0.7, 0.3])
    
    # P(B|A,C) - B is caused by both A and C
    p_b_given_ac = torch.tensor([
        [[0.95, 0.05], [0.3, 0.7]],   # A=0: [C=0, C=1]
        [[0.4, 0.6], [0.1, 0.9]]      # A=1: [C=0, C=1]
    ])
    
    # P(A, B, C) = P(A) * P(C) * P(B|A,C)
    p_joint = torch.zeros(2, 2, 2)
    for a in range(2):
        for b in range(2):
            for c in range(2):
                p_joint[a, b, c] = p_a[a] * p_c[c] * p_b_given_ac[a, c, b]
    
    # Check marginal independence
    p_ac = p_joint.sum(dim=1)
    
    print("\nCollider Structure: A → B ← C")
    print("=" * 50)
    print(f"\nP(A, C):\n{p_ac}")
    print(f"\nP(A) × P(C):\n{torch.outer(p_a, p_c)}")
    print(f"\nA and C marginally independent? {torch.allclose(p_ac, torch.outer(p_a, p_c), atol=0.01)}")
    
    # Check conditional dependence given B (explaining away)
    print("\nConditional Dependence given B (Explaining Away):")
    for b in range(2):
        p_ac_given_b = p_joint[:, b, :] / p_joint[:, b, :].sum()
        p_a_given_b = p_joint[:, b, :].sum(dim=1) / p_joint[:, b, :].sum()
        p_c_given_b = p_joint[:, b, :].sum(dim=0) / p_joint[:, b, :].sum()
        
        is_independent = torch.allclose(p_ac_given_b, torch.outer(p_a_given_b, p_c_given_b), atol=0.01)
        print(f"  A ⊥ C | B={b}? {is_independent}")
        
    print("\nNote: A and C become DEPENDENT when we condition on their common effect B!")

demonstrate_collider()
```

## D-Separation Algorithm

Given a DAG $G$ and sets $X$, $Y$, $Z$, determine if $X$ and $Y$ are d-separated by $Z$:

```python
import networkx as nx
from typing import Set, List, Tuple
from collections import deque

class DSeparation:
    """
    Implementation of d-separation testing for Bayesian Networks.
    """
    
    def __init__(self, graph: nx.DiGraph):
        """
        Initialize with a directed acyclic graph.
        
        Args:
            graph: NetworkX DiGraph representing the Bayesian Network structure
        """
        self.graph = graph
        # Precompute ancestors for efficiency
        self._ancestors_cache = {}
    
    def get_ancestors(self, node: str) -> Set[str]:
        """Get all ancestors of a node (cached)."""
        if node not in self._ancestors_cache:
            self._ancestors_cache[node] = nx.ancestors(self.graph, node)
        return self._ancestors_cache[node]
    
    def get_descendants(self, node: str) -> Set[str]:
        """Get all descendants of a node."""
        return nx.descendants(self.graph, node)
    
    def is_collider(self, path: List[str], idx: int) -> bool:
        """
        Check if node at index idx in path is a collider.
        A collider has arrows pointing TO it from both neighbors.
        """
        if idx == 0 or idx == len(path) - 1:
            return False
        
        prev_node, curr_node, next_node = path[idx-1], path[idx], path[idx+1]
        
        # Check if both edges point to curr_node
        edge_from_prev = self.graph.has_edge(prev_node, curr_node)
        edge_from_next = self.graph.has_edge(next_node, curr_node)
        
        return edge_from_prev and edge_from_next
    
    def is_path_blocked(self, path: List[str], Z: Set[str]) -> bool:
        """
        Check if a path is blocked by the conditioning set Z.
        
        Args:
            path: List of nodes forming a path
            Z: Set of conditioning nodes
            
        Returns:
            True if path is blocked, False if path is open
        """
        # Check each intermediate node
        for i in range(1, len(path) - 1):
            node = path[i]
            
            if self.is_collider(path, i):
                # Collider: path is blocked unless node or descendant is in Z
                node_or_desc_in_z = node in Z or bool(self.get_descendants(node) & Z)
                if not node_or_desc_in_z:
                    return True  # Blocked by unobserved collider
            else:
                # Chain or fork: path is blocked if node is in Z
                if node in Z:
                    return True  # Blocked by observed non-collider
        
        return False  # Path is open
    
    def find_all_paths(self, source: str, target: str, max_length: int = 10) -> List[List[str]]:
        """
        Find all undirected paths between source and target.
        
        Note: We consider paths regardless of edge direction, as information
        can flow in either direction (subject to d-separation rules).
        """
        # Create undirected version for path finding
        undirected = self.graph.to_undirected()
        
        paths = []
        
        # BFS for paths
        queue = deque([(source, [source])])
        
        while queue:
            current, path = queue.popleft()
            
            if len(path) > max_length:
                continue
            
            if current == target and len(path) > 1:
                paths.append(path)
                continue
            
            for neighbor in undirected.neighbors(current):
                if neighbor not in path:  # Avoid cycles
                    queue.append((neighbor, path + [neighbor]))
        
        return paths
    
    def is_d_separated(self, 
                       X: Set[str], 
                       Y: Set[str], 
                       Z: Set[str]) -> bool:
        """
        Test if X and Y are d-separated given Z.
        
        Args:
            X: First set of nodes
            Y: Second set of nodes
            Z: Conditioning set
            
        Returns:
            True if X and Y are d-separated by Z
        """
        # Check all pairs of nodes
        for x in X:
            for y in Y:
                # Find all paths between x and y
                paths = self.find_all_paths(x, y)
                
                # If any path is open, X and Y are not d-separated
                for path in paths:
                    if not self.is_path_blocked(path, Z):
                        return False  # Found an open path
        
        return True  # All paths are blocked
    
    def find_minimal_d_separator(self, X: str, Y: str) -> Set[str]:
        """
        Find a minimal set Z that d-separates X from Y.
        This is useful for identifying what to condition on.
        """
        # Simple approach: try ancestors of X and Y
        all_nodes = set(self.graph.nodes()) - {X, Y}
        
        # Try smallest sets first
        from itertools import combinations
        
        for size in range(len(all_nodes) + 1):
            for Z in combinations(all_nodes, size):
                Z_set = set(Z)
                if self.is_d_separated({X}, {Y}, Z_set):
                    return Z_set
        
        return None  # No separator found (shouldn't happen for valid X, Y)


# Demonstration
def demonstrate_d_separation():
    """Demonstrate d-separation with the classic alarm network."""
    
    # Create the Alarm network
    # Burglary -> Alarm <- Earthquake
    # Alarm -> JohnCalls
    # Alarm -> MaryCalls
    
    G = nx.DiGraph()
    G.add_edges_from([
        ('Burglary', 'Alarm'),
        ('Earthquake', 'Alarm'),
        ('Alarm', 'JohnCalls'),
        ('Alarm', 'MaryCalls')
    ])
    
    d_sep = DSeparation(G)
    
    print("\nAlarm Network D-Separation Analysis")
    print("=" * 60)
    print("\nStructure:")
    print("  Burglary → Alarm ← Earthquake")
    print("              ↓")
    print("       JohnCalls  MaryCalls")
    
    # Test various d-separation queries
    tests = [
        ({'Burglary'}, {'Earthquake'}, set(), 
         "Burglary ⊥ Earthquake | ∅"),
        ({'Burglary'}, {'Earthquake'}, {'Alarm'}, 
         "Burglary ⊥ Earthquake | Alarm"),
        ({'JohnCalls'}, {'MaryCalls'}, set(), 
         "JohnCalls ⊥ MaryCalls | ∅"),
        ({'JohnCalls'}, {'MaryCalls'}, {'Alarm'}, 
         "JohnCalls ⊥ MaryCalls | Alarm"),
        ({'Burglary'}, {'JohnCalls'}, set(), 
         "Burglary ⊥ JohnCalls | ∅"),
        ({'Burglary'}, {'JohnCalls'}, {'Alarm'}, 
         "Burglary ⊥ JohnCalls | Alarm"),
    ]
    
    print("\nD-Separation Tests:")
    print("-" * 60)
    for X, Y, Z, description in tests:
        result = d_sep.is_d_separated(X, Y, Z)
        symbol = "✓" if result else "✗"
        print(f"  {symbol} {description}: {result}")
    
    print("\nExplanations:")
    print("-" * 60)
    print("• Burglary ⊥ Earthquake | ∅: TRUE (collider Alarm blocks path)")
    print("• Burglary ⊥ Earthquake | Alarm: FALSE (conditioning on collider opens path)")
    print("• JohnCalls ⊥ MaryCalls | ∅: FALSE (both have common cause Alarm)")
    print("• JohnCalls ⊥ MaryCalls | Alarm: TRUE (fork blocked by conditioning)")
    print("• Burglary ⊥ JohnCalls | ∅: FALSE (chain Burglary→Alarm→JohnCalls)")
    print("• Burglary ⊥ JohnCalls | Alarm: TRUE (chain blocked)")

demonstrate_d_separation()
```

## Markov Blanket

The **Markov Blanket** of a node $X$ is the minimal set of nodes that makes $X$ conditionally independent of all other nodes:

$$\text{MB}(X) = \text{Parents}(X) \cup \text{Children}(X) \cup \text{Parents of Children}(X)$$

Given its Markov blanket, a node is independent of all other nodes:
$$X \perp\!\!\!\perp \text{All other nodes} \mid \text{MB}(X)$$

```python
def markov_blanket(graph: nx.DiGraph, node: str) -> Set[str]:
    """
    Compute the Markov blanket of a node.
    
    MB(X) = Parents(X) ∪ Children(X) ∪ CoParents(X)
    """
    parents = set(graph.predecessors(node))
    children = set(graph.successors(node))
    
    # Parents of children (co-parents)
    co_parents = set()
    for child in children:
        co_parents.update(graph.predecessors(child))
    co_parents.discard(node)  # Remove self
    
    return parents | children | co_parents
```

## Summary Table

| Structure | Marginal | Conditional on Middle | Rule |
|-----------|----------|----------------------|------|
| Chain: $A \rightarrow B \rightarrow C$ | Dependent | Independent | Blocking |
| Fork: $A \leftarrow B \rightarrow C$ | Dependent | Independent | Blocking |
| Collider: $A \rightarrow B \leftarrow C$ | Independent | Dependent | Explaining Away |

## Key Insights

1. **D-separation is purely graphical**: We can determine conditional independence without knowing any probabilities

2. **Colliders are special**: They behave opposite to chains and forks—conditioning *opens* rather than *blocks* paths

3. **Descendants of colliders matter**: If any descendant of a collider is observed, the path through the collider opens

4. **The Markov blanket shields a node**: Conditioning on the Markov blanket makes a node independent of everything else

5. **D-separation implies independence**: If d-separated, variables are independent in *any* distribution compatible with the graph
