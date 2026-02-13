# D-Separation

## Overview

**D-separation** (directional separation) is the fundamental graphical criterion for determining conditional independence in Bayesian networks. Given a DAG structure, d-separation allows us to read off which variables are conditionally independent without computing any probabilities—a purely structural test.

## Formal Definition

### Blocked Paths

A path between nodes $X$ and $Y$ is **blocked** by a set of nodes $Z$ if there exists a node $W$ on the path such that either:

1. **Chain or Fork**: $W \in Z$ and arrows on the path meet head-to-tail ($A \to W \to B$) or tail-to-tail ($A \leftarrow W \to B$) at $W$.

2. **Collider**: $W$ is a collider ($A \to W \leftarrow B$) and neither $W$ nor any descendant of $W$ is in $Z$.

### D-Separation

Two sets of nodes $X$ and $Y$ are **d-separated** by a set $Z$, written $\text{d-sep}(X, Y \mid Z)$, if and only if every path from any node in $X$ to any node in $Y$ is blocked by $Z$.

### The Fundamental Theorem

If $X$ and $Y$ are d-separated by $Z$ in a DAG $G$, then $X$ and $Y$ are conditionally independent given $Z$ in every distribution $P$ that factors according to $G$:

$$\text{d-sep}_G(X, Y \mid Z) \implies X \perp\!\!\!\perp_P Y \mid Z$$

Moreover, if $X$ and $Y$ are **not** d-separated by $Z$, there exists some distribution compatible with $G$ where $X$ and $Y$ are dependent given $Z$. This completeness result (due to Verma and Pearl) means d-separation captures *exactly* the conditional independencies implied by the graph structure.

## The Three Building Blocks

### 1. Chain (Serial Connection)

```
A --> B --> C
```

**D-separation analysis:**

- Path $A - B - C$: $B$ is in the middle with head-to-tail arrows
- If $B \in Z$: path is **blocked** → $A \perp\!\!\!\perp C \mid B$
- If $B \notin Z$: path is **open** → $A \not\perp\!\!\!\perp C$

Information flows from $A$ to $C$ through $B$. Observing $B$ intercepts this flow.

```python
import torch
import numpy as np


def demonstrate_chain():
    """Chain structure: A -> B -> C. Blocking when B is observed."""
    p_a = torch.tensor([0.3, 0.7])
    
    p_b_given_a = torch.tensor([
        [0.9, 0.1],   # P(B|A=0)
        [0.2, 0.8]    # P(B|A=1)
    ])
    
    p_c_given_b = torch.tensor([
        [0.8, 0.2],   # P(C|B=0)
        [0.3, 0.7]    # P(C|B=1)
    ])
    
    # P(A, B, C) = P(A) * P(B|A) * P(C|B)
    p_joint = torch.zeros(2, 2, 2)
    for a in range(2):
        for b in range(2):
            for c in range(2):
                p_joint[a, b, c] = p_a[a] * p_b_given_a[a, b] * p_c_given_b[b, c]
    
    # Check marginal dependence: P(A, C) vs P(A) * P(C)
    p_ac = p_joint.sum(dim=1)
    p_a_marginal = p_joint.sum(dim=(1, 2))
    p_c_marginal = p_joint.sum(dim=(0, 1))
    p_ac_independent = torch.outer(p_a_marginal, p_c_marginal)
    
    print("Chain Structure: A -> B -> C")
    print(f"A and C marginally independent? "
          f"{torch.allclose(p_ac, p_ac_independent, atol=0.01)}")
    
    # Conditional independence given B
    for b in range(2):
        p_ac_given_b = p_joint[:, b, :] / p_joint[:, b, :].sum()
        p_a_given_b = p_joint[:, b, :].sum(dim=1) / p_joint[:, b, :].sum()
        p_c_given_b_val = p_joint[:, b, :].sum(dim=0) / p_joint[:, b, :].sum()
        p_ac_ind_given_b = torch.outer(p_a_given_b, p_c_given_b_val)
        is_ind = torch.allclose(p_ac_given_b, p_ac_ind_given_b, atol=0.01)
        print(f"  A _|_ C | B={b}? {is_ind}")


demonstrate_chain()
```

### 2. Fork (Common Cause)

```
A <-- B --> C
```

**D-separation analysis:**

- Path $A - B - C$: $B$ is in the middle with tail-to-tail arrows (fork)
- If $B \in Z$: path is **blocked** → $A \perp\!\!\!\perp C \mid B$
- If $B \notin Z$: path is **open** → $A \not\perp\!\!\!\perp C$

$B$ is a common cause of both $A$ and $C$. They are marginally dependent (confounded) but independent once we condition on the common cause.

```python
def demonstrate_fork():
    """Fork structure: A <- B -> C. Common cause creates spurious correlation."""
    p_b = torch.tensor([0.4, 0.6])
    
    p_a_given_b = torch.tensor([
        [0.9, 0.1],   # P(A|B=0)
        [0.2, 0.8]    # P(A|B=1)
    ])
    
    p_c_given_b = torch.tensor([
        [0.7, 0.3],   # P(C|B=0)
        [0.1, 0.9]    # P(C|B=1)
    ])
    
    # P(A, B, C) = P(B) * P(A|B) * P(C|B)
    p_joint = torch.zeros(2, 2, 2)
    for a in range(2):
        for b in range(2):
            for c in range(2):
                p_joint[a, b, c] = p_b[b] * p_a_given_b[b, a] * p_c_given_b[b, c]
    
    p_ac = p_joint.sum(dim=1)
    p_a = p_joint.sum(dim=(1, 2))
    p_c = p_joint.sum(dim=(0, 1))
    
    print("\nFork Structure: A <- B -> C")
    print(f"A and C marginally independent? "
          f"{torch.allclose(p_ac, torch.outer(p_a, p_c), atol=0.01)}")
    
    for b in range(2):
        p_ac_given_b = p_joint[:, b, :] / p_joint[:, b, :].sum()
        p_a_given_b_val = p_joint[:, b, :].sum(dim=1) / p_joint[:, b, :].sum()
        p_c_given_b_val = p_joint[:, b, :].sum(dim=0) / p_joint[:, b, :].sum()
        is_ind = torch.allclose(
            p_ac_given_b, torch.outer(p_a_given_b_val, p_c_given_b_val), atol=0.01
        )
        print(f"  A _|_ C | B={b}? {is_ind}")


demonstrate_fork()
```

### 3. Collider (V-Structure)

```
A --> B <-- C
```

**D-separation analysis:**

- Path $A - B - C$: $B$ is a collider (arrows meet head-to-head)
- If $B \notin Z$ (and no descendant of $B$ in $Z$): path is **blocked** → $A \perp\!\!\!\perp C$
- If $B \in Z$ (or any descendant of $B$ in $Z$): path is **open** → $A \not\perp\!\!\!\perp C \mid B$

This is the **explaining away** effect. $A$ and $C$ are independent causes, but observing their common effect $B$ creates a dependency.

```python
def demonstrate_collider():
    """Collider structure: A -> B <- C. Explaining away effect."""
    p_a = torch.tensor([0.6, 0.4])
    p_c = torch.tensor([0.7, 0.3])
    
    # P(B|A,C) - B is caused by both A and C
    p_b_given_ac = torch.tensor([
        [[0.95, 0.05], [0.3, 0.7]],    # A=0: [C=0, C=1]
        [[0.4, 0.6], [0.1, 0.9]]       # A=1: [C=0, C=1]
    ])
    
    # P(A, B, C) = P(A) * P(C) * P(B|A,C)
    p_joint = torch.zeros(2, 2, 2)
    for a in range(2):
        for b in range(2):
            for c in range(2):
                p_joint[a, b, c] = p_a[a] * p_c[c] * p_b_given_ac[a, c, b]
    
    p_ac = p_joint.sum(dim=1)
    
    print("\nCollider Structure: A -> B <- C")
    print(f"A and C marginally independent? "
          f"{torch.allclose(p_ac, torch.outer(p_a, p_c), atol=0.01)}")
    
    # Conditional dependence given B (explaining away)
    for b in range(2):
        p_ac_given_b = p_joint[:, b, :] / p_joint[:, b, :].sum()
        p_a_given_b = p_joint[:, b, :].sum(dim=1) / p_joint[:, b, :].sum()
        p_c_given_b = p_joint[:, b, :].sum(dim=0) / p_joint[:, b, :].sum()
        is_ind = torch.allclose(
            p_ac_given_b, torch.outer(p_a_given_b, p_c_given_b), atol=0.01
        )
        print(f"  A _|_ C | B={b}? {is_ind}")
    
    print("\nA and C become DEPENDENT when we condition on their common effect B!")


demonstrate_collider()
```

## D-Separation Algorithm

The following implementation provides a complete d-separation tester, including path enumeration, collider detection, and blocking analysis:

```python
import networkx as nx
from typing import Set, List
from collections import deque


class DSeparation:
    """Implementation of d-separation testing for Bayesian Networks."""
    
    def __init__(self, graph: nx.DiGraph):
        self.graph = graph
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
        A collider has arrows pointing TO it from both neighbors on the path.
        """
        if idx == 0 or idx == len(path) - 1:
            return False
        prev_node, curr_node, next_node = path[idx - 1], path[idx], path[idx + 1]
        edge_from_prev = self.graph.has_edge(prev_node, curr_node)
        edge_from_next = self.graph.has_edge(next_node, curr_node)
        return edge_from_prev and edge_from_next
    
    def is_path_blocked(self, path: List[str], Z: Set[str]) -> bool:
        """
        Check if a path is blocked by the conditioning set Z.
        
        A path is blocked if ANY intermediate node satisfies:
        - Non-collider (chain/fork) and the node is in Z, OR
        - Collider and neither the node nor any descendant is in Z.
        """
        for i in range(1, len(path) - 1):
            node = path[i]
            
            if self.is_collider(path, i):
                # Collider: blocked unless node or descendant in Z
                node_or_desc_in_z = node in Z or bool(self.get_descendants(node) & Z)
                if not node_or_desc_in_z:
                    return True
            else:
                # Chain or fork: blocked if node is in Z
                if node in Z:
                    return True
        
        return False
    
    def find_all_paths(self, source: str, target: str, 
                       max_length: int = 10) -> List[List[str]]:
        """
        Find all undirected paths between source and target.
        
        Paths are considered regardless of edge direction, as information
        flow direction is handled by the blocking rules.
        """
        undirected = self.graph.to_undirected()
        paths = []
        queue = deque([(source, [source])])
        
        while queue:
            current, path = queue.popleft()
            if len(path) > max_length:
                continue
            if current == target and len(path) > 1:
                paths.append(path)
                continue
            for neighbor in undirected.neighbors(current):
                if neighbor not in path:
                    queue.append((neighbor, path + [neighbor]))
        
        return paths
    
    def is_d_separated(self, X: Set[str], Y: Set[str], Z: Set[str]) -> bool:
        """
        Test if X and Y are d-separated given Z.
        
        Returns True if ALL paths between every pair (x, y) with x in X
        and y in Y are blocked by Z.
        """
        for x in X:
            for y in Y:
                paths = self.find_all_paths(x, y)
                for path in paths:
                    if not self.is_path_blocked(path, Z):
                        return False   # Found an open path
        return True
    
    def find_minimal_d_separator(self, X: str, Y: str) -> Set[str]:
        """Find a minimal set Z that d-separates X from Y."""
        from itertools import combinations
        
        all_nodes = set(self.graph.nodes()) - {X, Y}
        
        for size in range(len(all_nodes) + 1):
            for Z in combinations(all_nodes, size):
                Z_set = set(Z)
                if self.is_d_separated({X}, {Y}, Z_set):
                    return Z_set
        
        return None
```

### Demonstration: The Alarm Network

```python
def demonstrate_d_separation():
    """Demonstrate d-separation with the classic alarm network."""
    
    # Burglary -> Alarm <- Earthquake
    # Alarm -> JohnCalls, Alarm -> MaryCalls
    G = nx.DiGraph()
    G.add_edges_from([
        ('Burglary', 'Alarm'),
        ('Earthquake', 'Alarm'),
        ('Alarm', 'JohnCalls'),
        ('Alarm', 'MaryCalls')
    ])
    
    d_sep = DSeparation(G)
    
    print("Alarm Network D-Separation Analysis")
    print("=" * 60)
    print("Structure:")
    print("  Burglary -> Alarm <- Earthquake")
    print("               |")
    print("        JohnCalls  MaryCalls")
    
    tests = [
        ({'Burglary'}, {'Earthquake'}, set(),
         "Burglary _|_ Earthquake | empty"),
        ({'Burglary'}, {'Earthquake'}, {'Alarm'},
         "Burglary _|_ Earthquake | Alarm"),
        ({'JohnCalls'}, {'MaryCalls'}, set(),
         "JohnCalls _|_ MaryCalls | empty"),
        ({'JohnCalls'}, {'MaryCalls'}, {'Alarm'},
         "JohnCalls _|_ MaryCalls | Alarm"),
        ({'Burglary'}, {'JohnCalls'}, set(),
         "Burglary _|_ JohnCalls | empty"),
        ({'Burglary'}, {'JohnCalls'}, {'Alarm'},
         "Burglary _|_ JohnCalls | Alarm"),
    ]
    
    print("\nD-Separation Tests:")
    for X, Y, Z, description in tests:
        result = d_sep.is_d_separated(X, Y, Z)
        symbol = "T" if result else "F"
        print(f"  [{symbol}] {description}: {result}")
    
    print("\nExplanations:")
    print("  Burglary _|_ Earthquake | empty:  TRUE  (collider Alarm blocks path)")
    print("  Burglary _|_ Earthquake | Alarm:  FALSE (conditioning on collider opens path)")
    print("  JohnCalls _|_ MaryCalls | empty:  FALSE (common cause Alarm)")
    print("  JohnCalls _|_ MaryCalls | Alarm:  TRUE  (fork blocked by conditioning)")
    print("  Burglary _|_ JohnCalls | empty:   FALSE (chain Burglary->Alarm->JohnCalls)")
    print("  Burglary _|_ JohnCalls | Alarm:   TRUE  (chain blocked)")


demonstrate_d_separation()
```

## The Markov Blanket

The **Markov Blanket** of a node $X$ is the minimal set of nodes that makes $X$ conditionally independent of all other nodes in the network:

$$\text{MB}(X) = \text{Parents}(X) \cup \text{Children}(X) \cup \text{Parents of Children}(X)$$

Given its Markov blanket, a node is independent of all other nodes:

$$X \perp\!\!\!\perp (V \setminus \{X\} \setminus \text{MB}(X)) \mid \text{MB}(X)$$

The Markov blanket is important for both inference (it determines the minimal conditioning set) and learning (it defines the local neighborhood relevant to a variable).

```python
def markov_blanket(graph: nx.DiGraph, node: str) -> Set[str]:
    """
    Compute the Markov blanket of a node.
    
    MB(X) = Parents(X) U Children(X) U CoParents(X)
    """
    parents = set(graph.predecessors(node))
    children = set(graph.successors(node))
    
    co_parents = set()
    for child in children:
        co_parents.update(graph.predecessors(child))
    co_parents.discard(node)
    
    return parents | children | co_parents


# Example
G = nx.DiGraph()
G.add_edges_from([
    ('Burglary', 'Alarm'), ('Earthquake', 'Alarm'),
    ('Alarm', 'JohnCalls'), ('Alarm', 'MaryCalls')
])

mb_alarm = markov_blanket(G, 'Alarm')
print(f"Markov Blanket of Alarm: {mb_alarm}")
# {Burglary, Earthquake, JohnCalls, MaryCalls}
```

## The Bayes Ball Algorithm

An efficient alternative to enumerating all paths is the **Bayes Ball algorithm** (Shachter, 1998), which uses a message-passing scheme to determine d-separation in $O(|V| + |E|)$ time. The algorithm propagates "balls" through the graph, tracking whether they pass through nodes from the top (parent side) or bottom (child side), and checking whether the conditioning set blocks them.

The key rules for ball propagation at each node $W$:

| Ball arrives from | $W \notin Z$ | $W \in Z$ |
|---|---|---|
| Parent | Pass to children | Block; pass to parents |
| Child | Pass to both parents and children | Pass to children only |

This formulation avoids explicit path enumeration and scales to large networks.

## Summary

| Structure | Marginal | Conditioned on Middle | Rule |
|-----------|----------|----------------------|------|
| Chain: $A \to B \to C$ | Dependent | Independent | Blocking |
| Fork: $A \leftarrow B \to C$ | Dependent | Independent | Blocking |
| Collider: $A \to B \leftarrow C$ | Independent | Dependent | Explaining away |

## Key Insights

1. **D-separation is purely graphical**: conditional independence is determined without computing any probabilities.

2. **Colliders are special**: they behave opposite to chains and forks—conditioning *opens* rather than *blocks* paths.

3. **Descendants of colliders matter**: if any descendant of a collider is observed, the path through the collider opens.

4. **The Markov blanket shields a node**: conditioning on the Markov blanket makes a node independent of everything else.

5. **Completeness**: d-separation captures *exactly* the conditional independencies implied by the graph—no more, no less.

## Quantitative Finance Application

D-separation provides a formal framework for reasoning about which financial variables carry information about each other. Consider a credit risk network:

```
MacroEconomy -> IndustryHealth -> FirmDefault
                                       |
                                       v
                                  CreditSpread
```

D-separation tells us that $\text{MacroEconomy} \perp\!\!\!\perp \text{CreditSpread} \mid \text{FirmDefault}$: once we observe whether a firm defaulted, macroeconomic conditions provide no additional information about its credit spread. Conversely, without observing default status, macro conditions *are* informative about spreads through the chain. This kind of reasoning is essential for building efficient risk models that avoid redundant conditioning.
