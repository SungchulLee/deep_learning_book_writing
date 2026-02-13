# Factor Graphs

## Overview

**Factor graphs** provide a unified graphical representation that subsumes both Bayesian networks and Markov Random Fields. They make the factorization structure of a distribution explicit by introducing separate nodes for variables and factors, eliminating ambiguity about which variables participate in which factors.

## Definition

A factor graph is a bipartite graph $G = (V, F, E)$ where:

- **Variable nodes** $V = \{X_1, \ldots, X_n\}$ represent random variables (drawn as circles)
- **Factor nodes** $F = \{f_1, \ldots, f_m\}$ represent factors/functions (drawn as squares)
- **Edges** $E$ connect each factor node to the variable nodes in its scope

The joint distribution is:

$$P(X_1, \ldots, X_n) = \frac{1}{Z} \prod_{a=1}^{m} f_a(X_{\mathcal{N}(a)})$$

where $\mathcal{N}(a)$ denotes the set of variable nodes neighboring factor node $f_a$, and $Z$ is the partition function (equal to 1 when the factorization already defines a normalized distribution, as in Bayesian networks).

## Why Factor Graphs?

### Ambiguity in Undirected Models

Consider the factorization:

$$P(A, B, C) \propto \psi_1(A, B) \cdot \psi_2(B, C) \cdot \psi_3(A, C)$$

versus:

$$P(A, B, C) \propto \psi_1(A, B, C)$$

Both correspond to the same undirected graph (a complete graph over $\{A, B, C\}$), but they represent different factorization structures with different computational implications. The factor graph makes this distinction explicit:

**Three pairwise factors:**
```
  (A)---[f1]---(B)
   |             |
  [f3]         [f2]
   |             |
  (C)----------(C)
```

**One ternary factor:**
```
  (A)---[f1]---(B)
          |
         (C)
```

### Unified Message Passing

Factor graphs provide a single framework for message passing that works identically for directed and undirected models. The sum-product and max-product algorithms are most naturally expressed on factor graphs, where messages flow between variable nodes and factor nodes.

## Converting to Factor Graphs

### From Bayesian Networks

Each CPT $P(X_i \mid \text{Pa}(X_i))$ becomes a factor node connected to $X_i$ and all its parents:

```python
import torch
from typing import Dict, List, Optional


class FactorNode:
    """A factor node in a factor graph."""
    
    def __init__(self, name: str, variables: List[str],
                 cardinalities: Dict[str, int],
                 values: torch.Tensor):
        self.name = name
        self.variables = variables
        self.cardinalities = cardinalities
        self.values = values
    
    def evaluate(self, assignment: Dict[str, int]) -> float:
        index = tuple(assignment[v] for v in self.variables)
        return self.values[index].item()
    
    def __repr__(self) -> str:
        return f"FactorNode({self.name}, vars={self.variables})"


class FactorGraph:
    """
    Factor graph representation.
    
    A bipartite graph with variable nodes and factor nodes.
    """
    
    def __init__(self):
        self.variable_nodes: List[str] = []
        self.cardinalities: Dict[str, int] = {}
        self.factor_nodes: List[FactorNode] = []
        self.var_to_factors: Dict[str, List[int]] = {}
    
    def add_variable(self, name: str, cardinality: int):
        self.variable_nodes.append(name)
        self.cardinalities[name] = cardinality
        self.var_to_factors[name] = []
    
    def add_factor(self, name: str, variables: List[str],
                   values: torch.Tensor):
        factor_idx = len(self.factor_nodes)
        factor = FactorNode(name, variables, self.cardinalities, values)
        self.factor_nodes.append(factor)
        
        for var in variables:
            self.var_to_factors[var].append(factor_idx)
    
    def get_neighbors_of_variable(self, var: str) -> List[FactorNode]:
        """Get all factor nodes connected to a variable."""
        return [self.factor_nodes[i] for i in self.var_to_factors[var]]
    
    def get_neighbors_of_factor(self, factor_idx: int) -> List[str]:
        """Get all variable nodes connected to a factor."""
        return self.factor_nodes[factor_idx].variables
    
    def is_tree(self) -> bool:
        """Check if the factor graph is a tree (no cycles)."""
        n_vars = len(self.variable_nodes)
        n_factors = len(self.factor_nodes)
        n_edges = sum(
            len(f.variables) for f in self.factor_nodes
        )
        # A tree with (n_vars + n_factors) nodes has that many - 1 edges
        return n_edges == n_vars + n_factors - 1
    
    @staticmethod
    def from_bayesian_network(bn) -> 'FactorGraph':
        """Convert a Bayesian Network to a factor graph."""
        fg = FactorGraph()
        
        for var in bn.variables:
            fg.add_variable(var, bn.cardinalities[var])
        
        for var in bn.variables:
            cpt = bn.cpts[var]
            factor_vars = cpt.parents + [cpt.variable]
            fg.add_factor(
                f"f_{var}", factor_vars, cpt.values
            )
        
        return fg
    
    @staticmethod
    def from_mrf(mrf) -> 'FactorGraph':
        """Convert an MRF to a factor graph."""
        fg = FactorGraph()
        
        for var in mrf.variables:
            fg.add_variable(var, mrf.cardinalities[var])
        
        for i, factor in enumerate(mrf.factors):
            fg.add_factor(
                f"f_{i}", factor.variables, factor.values
            )
        
        return fg
```

### Example: Weather Network as Factor Graph

```python
def demonstrate_factor_graph():
    """Convert the weather Bayesian network to a factor graph."""
    # Build the weather network (from Section 17.2)
    from ch17.pgm_directed_models.bayesian_networks import build_weather_network
    
    bn = build_weather_network()
    fg = FactorGraph.from_bayesian_network(bn)
    
    print("Weather Network as Factor Graph")
    print("=" * 50)
    print(f"Variable nodes: {fg.variable_nodes}")
    print(f"Factor nodes: {[f.name for f in fg.factor_nodes]}")
    
    print("\nConnections:")
    for f in fg.factor_nodes:
        print(f"  {f.name} -- {f.variables}")
    
    print(f"\nIs tree? {fg.is_tree()}")
    # False: WetGrass has two parents creating a cycle
    # in the factor graph through f_Sprinkler, f_Rain, f_WetGrass
```

The factor graph structure for the weather network:

```
(Cloudy)---[f_Cloudy]
   |
   +-------[f_Sprinkler]---(Sprinkler)---+
   |                                      |
   +-------[f_Rain]-------(Rain)---------+
                                          |
                           [f_WetGrass]---+
```

## Message Passing on Factor Graphs

Factor graphs are the natural setting for message passing algorithms. Messages flow between variable nodes and factor nodes:

### Variable-to-Factor Messages

A variable node $X_i$ sends a message to factor node $f_a$:

$$\mu_{X_i \to f_a}(x_i) = \prod_{b \in \mathcal{N}(i) \setminus \{a\}} \mu_{f_b \to X_i}(x_i)$$

The message from a variable to a factor is the product of all incoming factor-to-variable messages *except* the one from the target factor.

### Factor-to-Variable Messages

A factor node $f_a$ sends a message to variable node $X_i$:

$$\mu_{f_a \to X_i}(x_i) = \sum_{X_{\mathcal{N}(a) \setminus \{i\}}} f_a(X_{\mathcal{N}(a)}) \prod_{j \in \mathcal{N}(a) \setminus \{i\}} \mu_{X_j \to f_a}(x_j)$$

The factor multiplies its potential by all incoming variable messages (except from $X_i$) and marginalizes out all variables except $X_i$.

### Marginals from Messages

Once messages converge, the marginal of variable $X_i$ is:

$$P(X_i) \propto \prod_{a \in \mathcal{N}(i)} \mu_{f_a \to X_i}(x_i)$$

```python
class SumProductOnFactorGraph:
    """
    Sum-product message passing on a factor graph.
    
    Computes exact marginals when the factor graph is a tree.
    """
    
    def __init__(self, fg: FactorGraph):
        self.fg = fg
        # Messages: (source, target) -> tensor
        self.var_to_factor_msgs: Dict[tuple, torch.Tensor] = {}
        self.factor_to_var_msgs: Dict[tuple, torch.Tensor] = {}
    
    def _init_messages(self):
        """Initialize all messages to uniform."""
        for var in self.fg.variable_nodes:
            card = self.fg.cardinalities[var]
            for fi in self.fg.var_to_factors[var]:
                self.var_to_factor_msgs[(var, fi)] = torch.ones(card)
                self.factor_to_var_msgs[(fi, var)] = torch.ones(card)
    
    def _compute_var_to_factor(self, var: str, target_fi: int):
        """Compute message from variable to factor."""
        card = self.fg.cardinalities[var]
        msg = torch.ones(card)
        
        for fi in self.fg.var_to_factors[var]:
            if fi != target_fi:
                msg = msg * self.factor_to_var_msgs.get(
                    (fi, var), torch.ones(card)
                )
        
        self.var_to_factor_msgs[(var, target_fi)] = msg
    
    def _compute_factor_to_var(self, factor_idx: int, target_var: str):
        """Compute message from factor to variable."""
        from itertools import product as cart_prod
        
        factor = self.fg.factor_nodes[factor_idx]
        other_vars = [v for v in factor.variables if v != target_var]
        target_card = self.fg.cardinalities[target_var]
        msg = torch.zeros(target_card)
        
        if not other_vars:
            # Factor only involves target variable
            target_idx = factor.variables.index(target_var)
            msg = factor.values.clone()
            return
        
        # Sum over all other variables
        other_cards = [self.fg.cardinalities[v] for v in other_vars]
        
        for target_val in range(target_card):
            total = 0.0
            for other_vals in cart_prod(*[range(c) for c in other_cards]):
                assignment = dict(zip(other_vars, other_vals))
                assignment[target_var] = target_val
                
                factor_val = factor.evaluate(assignment)
                
                # Multiply by incoming variable messages
                prod = 1.0
                for v, val in zip(other_vars, other_vals):
                    v_msg = self.var_to_factor_msgs.get(
                        (v, factor_idx), torch.ones(self.fg.cardinalities[v])
                    )
                    prod *= v_msg[val].item()
                
                total += factor_val * prod
            
            msg[target_val] = total
        
        self.factor_to_var_msgs[(factor_idx, target_var)] = msg
    
    def compute_marginals(self, n_iterations: int = 10) -> Dict[str, torch.Tensor]:
        """
        Run sum-product and compute marginals.
        
        For tree factor graphs, this converges in a number of iterations
        equal to the diameter of the tree.
        """
        self._init_messages()
        
        for _ in range(n_iterations):
            # Variable -> Factor messages
            for var in self.fg.variable_nodes:
                for fi in self.fg.var_to_factors[var]:
                    self._compute_var_to_factor(var, fi)
            
            # Factor -> Variable messages
            for fi, factor in enumerate(self.fg.factor_nodes):
                for var in factor.variables:
                    self._compute_factor_to_var(fi, var)
        
        # Compute marginals
        marginals = {}
        for var in self.fg.variable_nodes:
            card = self.fg.cardinalities[var]
            belief = torch.ones(card)
            for fi in self.fg.var_to_factors[var]:
                belief = belief * self.factor_to_var_msgs.get(
                    (fi, var), torch.ones(card)
                )
            belief = belief / belief.sum()
            marginals[var] = belief
        
        return marginals
```

## Factor Graph Properties

### Tree-Structured Factor Graphs

When the factor graph is a tree (no cycles), message passing computes **exact** marginals in a single pass (two sweeps: leaves to root, then root to leaves). This is the basis for the belief propagation algorithm discussed in Section 17.4.

### Cycle Detection

A factor graph has a cycle if and only if there exists a path from a variable node back to itself that alternates between variable and factor nodes. The presence of cycles means:

- Message passing is no longer guaranteed to converge
- If it converges, marginals may be approximate
- The junction tree algorithm (Section 17.4) can restore exactness by clustering variables

### Relationship to Treewidth

The treewidth of a factor graph determines the computational complexity of exact inference. A graph with treewidth $w$ admits exact inference in $O(n \cdot d^{w+1})$ time, where $d$ is the maximum cardinality. Trees have treewidth 1; complete graphs have treewidth $n - 1$.

## Factor Operations

Factor graphs rely on three fundamental operations that underpin all inference algorithms:

### Factor Product

Combine two factors by multiplying values for consistent assignments:

$$(\phi_1 \times \phi_2)(X, Y, Z) = \phi_1(X, Y) \cdot \phi_2(Y, Z)$$

### Factor Marginalization

Eliminate a variable by summing over its values:

$$\phi'(X) = \sum_Y \phi(X, Y)$$

### Factor Reduction

Fix a variable to an observed value:

$$\phi'(X) = \phi(X, Y = y)$$

These operations form the computational backbone of variable elimination, belief propagation, and the junction tree algorithm.

## Summary

| Concept | Description |
|---------|-------------|
| **Factor Graph** | Bipartite graph with variable nodes and factor nodes |
| **Variable Node** | Represents a random variable |
| **Factor Node** | Represents a factor/potential over its neighboring variables |
| **Sum-Product** | Message passing algorithm for computing marginals |
| **Tree Property** | Exact marginals when factor graph has no cycles |
| **Treewidth** | Determines complexity of exact inference |

## Key Advantages of Factor Graphs

1. **Explicit factorization**: No ambiguity about which variables participate in which factors.
2. **Unified framework**: Both directed and undirected models map naturally to factor graphs.
3. **Message passing**: The sum-product and max-product algorithms are most cleanly formulated on factor graphs.
4. **Modular construction**: Factors can be added, removed, or modified independently.
5. **Connection to neural architectures**: Graph neural networks and belief propagation neural networks operate on factor graph structures.

## Quantitative Finance Application

Factor graphs provide a natural framework for multi-asset derivative pricing under complex dependency structures. Consider pricing a basket option on $n$ correlated assets where the joint return distribution factors as:

$$P(R_1, \ldots, R_n) \propto \prod_{(i,j) \in E} \psi_{ij}(R_i, R_j) \prod_i \phi_i(R_i)$$

The factor graph makes the dependency structure explicit: unary factors $\phi_i$ encode marginal return distributions (e.g., from individual Black-Scholes models), while pairwise factors $\psi_{ij}$ encode copula-like dependencies. Message passing on this factor graph can efficiently compute marginal distributions and expected payoffs without brute-force Monte Carlo simulation of the full joint distribution.
