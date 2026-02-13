# Variable Elimination

## The Problem with Enumeration

Inference by enumeration has complexity $O(d^n)$ where $d$ is the maximum cardinality and $n$ is the number of variables. For 20 binary variables, this means summing over $2^{20} \approx 10^6$ terms.

**Variable Elimination (VE)** exploits the factored structure of graphical models to avoid this exponential blowup by working with local factors, eliminating variables one at a time, and only multiplying factors that share variables.

## The Key Insight: Pushing Sums Inside Products

Consider computing $P(X)$ by marginalizing over $Y$ and $Z$ where $P(X, Y, Z) = P(X) \cdot P(Y \mid X) \cdot P(Z \mid Y)$:

$$P(X) = \sum_Y \sum_Z P(X) \cdot P(Y \mid X) \cdot P(Z \mid Y)$$

Since $P(X)$ and $P(Y \mid X)$ do not depend on $Z$, we can rearrange:

$$P(X) = P(X) \cdot \sum_Y P(Y \mid X) \cdot \underbrace{\sum_Z P(Z \mid Y)}_{= 1}$$

This **distributive law** — pushing summations inside products when variables are absent from some factors — is the core principle behind variable elimination.

## Factors: The Fundamental Data Structure

A **factor** $\phi(X_1, \ldots, X_k)$ maps each assignment of values to a non-negative real number. Factors generalize both conditional distributions and potential functions. The three fundamental factor operations are:

**Factor Product**: $(\phi_1 \times \phi_2)(X, Y, Z) = \phi_1(X, Y) \cdot \phi_2(Y, Z)$

**Factor Marginalization**: $\phi'(X) = \sum_Y \phi(X, Y)$

**Factor Reduction**: $\phi'(X) = \phi(X, Y = y)$

### PyTorch Implementation

```python
import torch
from typing import Dict, List, Set, Optional
from itertools import product as cartesian_product


class Factor:
    """A factor over discrete random variables: phi(X1,...,Xk) -> R>=0."""
    
    def __init__(self, variables: List[str],
                 cardinalities: Dict[str, int],
                 values: Optional[torch.Tensor] = None):
        self.variables = list(variables)
        self.cardinalities = cardinalities
        self.shape = tuple(cardinalities[v] for v in self.variables)
        
        if values is None:
            self.values = torch.ones(self.shape)
        else:
            self.values = values.float()
    
    @property
    def scope(self) -> Set[str]:
        return set(self.variables)
    
    def multiply(self, other: 'Factor') -> 'Factor':
        """Factor product over the union of variables."""
        new_vars = self.variables.copy()
        for v in other.variables:
            if v not in new_vars:
                new_vars.append(v)
        
        result = Factor(new_vars, self.cardinalities)
        for assignment in self._enumerate(new_vars):
            val1 = self._get(assignment)
            val2 = other._get(assignment)
            result._set(assignment, val1 * val2)
        return result
    
    def marginalize(self, variables_to_sum: List[str]) -> 'Factor':
        """Sum out specified variables."""
        remaining = [v for v in self.variables if v not in variables_to_sum]
        if not remaining:
            return Factor([], self.cardinalities,
                         torch.tensor([self.values.sum()]))
        axes = tuple(self.variables.index(v) for v in variables_to_sum
                     if v in self.variables)
        return Factor(remaining, self.cardinalities, self.values.sum(dim=axes))
    
    def reduce(self, evidence: Dict[str, int]) -> 'Factor':
        """Fix evidence variables to observed values."""
        indices, remaining = [], []
        for v in self.variables:
            if v in evidence:
                indices.append(evidence[v])
            else:
                indices.append(slice(None))
                remaining.append(v)
        new_values = self.values[tuple(indices)]
        if not remaining:
            s = new_values.item() if new_values.dim() == 0 else new_values
            return Factor([], self.cardinalities, torch.tensor([s]))
        return Factor(remaining, self.cardinalities, new_values)
    
    def normalize(self) -> 'Factor':
        total = self.values.sum()
        if total > 0:
            return Factor(self.variables, self.cardinalities, self.values / total)
        return self
    
    def _get(self, assignment):
        if not self.variables:
            return self.values.item() if self.values.dim() == 0 else self.values[0].item()
        return self.values[tuple(assignment[v] for v in self.variables)].item()
    
    def _set(self, assignment, value):
        if not self.variables:
            self.values = torch.tensor([value])
        else:
            self.values[tuple(assignment[v] for v in self.variables)] = value
    
    def _enumerate(self, variables):
        if not variables:
            yield {}
            return
        cards = [self.cardinalities[v] for v in variables]
        for vals in cartesian_product(*[range(c) for c in cards]):
            yield dict(zip(variables, vals))
    
    def __repr__(self):
        return f"Factor({','.join(self.variables)}, shape={self.shape})"
```

## The Variable Elimination Algorithm

### Algorithm Steps

Given a graphical model with factors $\{\phi_1, \ldots, \phi_m\}$, query variables $Q$, and evidence $E = e$:

1. **Initialize**: Convert each CPT/potential to a factor
2. **Reduce**: Apply evidence by reducing all factors containing evidence variables
3. **Eliminate**: For each hidden variable $H_i$ in the chosen elimination order:
    - Collect all factors whose scope contains $H_i$
    - Multiply them into a single product factor
    - Sum out $H_i$
    - Replace collected factors with the new marginalized factor
4. **Multiply**: Multiply all remaining factors
5. **Normalize**: Divide by the sum to obtain $P(Q \mid E = e)$

```python
class VariableElimination:
    """
    Variable Elimination for exact inference.
    
    Complexity: O(n * d^(w+1)) where w is the induced width.
    """
    
    def __init__(self, bn):
        self.bn = bn
    
    def _cpt_to_factor(self, variable: str) -> Factor:
        cpt = self.bn.cpts[variable]
        variables = cpt.parents + [cpt.variable]
        return Factor(variables, self.bn.cardinalities, cpt.values)
    
    def query(self, query_vars: List[str],
              evidence: Dict[str, int] = None,
              elimination_order: List[str] = None,
              verbose: bool = False) -> torch.Tensor:
        """Compute P(query_vars | evidence) via variable elimination."""
        if evidence is None:
            evidence = {}
        
        if verbose:
            q_str = ', '.join(query_vars)
            e_str = ', '.join(f'{k}={v}' for k, v in evidence.items())
            print(f"Query: P({q_str}" + (f" | {e_str})" if evidence else ")"))
        
        # Step 1: Create factors from CPTs
        factors = [self._cpt_to_factor(var) for var in self.bn.variables]
        
        # Step 2: Reduce factors with evidence
        if evidence:
            factors = [f.reduce(evidence) for f in factors]
            factors = [f for f in factors if f.variables or f.values.numel() > 0]
        
        # Step 3: Determine elimination order
        all_vars = set(self.bn.variables)
        hidden_vars = all_vars - set(query_vars) - set(evidence.keys())
        
        if elimination_order is None:
            elimination_order = [
                v for v in reversed(self.bn.variables) if v in hidden_vars
            ]
        
        # Step 4: Eliminate hidden variables one by one
        for var in elimination_order:
            if verbose:
                print(f"  Eliminating {var}")
            
            relevant = [f for f in factors if var in f.scope]
            others = [f for f in factors if var not in f.scope]
            
            if not relevant:
                continue
            
            # Multiply relevant factors
            product = relevant[0]
            for f in relevant[1:]:
                product = product.multiply(f)
            
            # Sum out the variable
            marginalized = product.marginalize([var])
            factors = others + [marginalized]
        
        # Step 5: Multiply remaining factors
        result = factors[0]
        for f in factors[1:]:
            result = result.multiply(f)
        
        # Step 6: Normalize
        result = result.normalize()
        
        # Reorder dimensions to match query_vars order
        if result.variables != query_vars:
            perm = [result.variables.index(v) for v in query_vars
                    if v in result.variables]
            if perm:
                result.values = result.values.permute(*perm)
        
        return result.values
```

## Worked Example

Consider the chain $A \to B \to C$ with query $P(C \mid A = 0)$:

**Initial factors:** $\phi_A(A) = P(A)$, $\phi_B(A, B) = P(B \mid A)$, $\phi_C(B, C) = P(C \mid B)$

**Step 1 — Reduce with evidence $A = 0$:**

- $\phi'_A = \phi_A(A = 0)$ — scalar
- $\phi'_B(B) = \phi_B(A = 0, B)$ — now only over $B$
- $\phi_C(B, C)$ — unchanged

**Step 2 — Eliminate $B$:**

- Multiply: $\psi(B, C) = \phi'_B(B) \cdot \phi_C(B, C)$
- Marginalize: $\tau(C) = \sum_B \psi(B, C)$

**Step 3 — Multiply remaining:** $\phi'_A \cdot \tau(C)$

**Step 4 — Normalize:** $P(C \mid A = 0) = \tau(C) / \sum_C \tau(C)$

The key efficiency gain: we never construct the full joint table over $(A, B, C)$.

## Elimination Ordering

The efficiency of VE depends critically on the **elimination order**. Different orders produce intermediate factors of different sizes.

### Induced Width (Treewidth)

The **induced width** $w_\pi$ under elimination order $\pi$ is the maximum number of variables in any factor created during elimination, minus 1. It determines the complexity:

$$\text{Time}: O(n \cdot d^{w_\pi + 1}), \qquad \text{Space}: O(d^{w_\pi + 1})$$

### Complexity by Network Structure

| Network Structure | Induced Width | Complexity |
|-------------------|---------------|------------|
| Chain | 1 | $O(n \cdot d^2)$ |
| Tree | 1 | $O(n \cdot d^2)$ |
| Polytree | 1 | $O(n \cdot d^2)$ |
| Grid ($n \times n$) | $n - 1$ | $O(n^2 \cdot d^n)$ |
| Complete graph | $n - 1$ | $O(n \cdot d^n)$ |

### Finding Good Orders

Finding the optimal elimination order is NP-hard, but effective heuristics exist:

1. **Min-Degree**: Eliminate the variable with fewest neighbors in the current interaction graph
2. **Min-Fill**: Eliminate the variable that adds the fewest fill-in edges
3. **Min-Weight**: Eliminate the variable minimizing the product of neighbor cardinalities

```python
import networkx as nx


def min_degree_order(factors: List[Factor], hidden_vars: Set[str],
                     cardinalities: Dict[str, int]) -> List[str]:
    """Compute elimination order using the min-degree heuristic."""
    graph = nx.Graph()
    graph.add_nodes_from(hidden_vars)
    
    for f in factors:
        vars_in_hidden = [v for v in f.scope if v in hidden_vars]
        for i, v1 in enumerate(vars_in_hidden):
            for v2 in vars_in_hidden[i + 1:]:
                graph.add_edge(v1, v2)
    
    order = []
    remaining = set(hidden_vars)
    
    while remaining:
        min_var = min(remaining,
                      key=lambda v: graph.degree(v) if v in graph else 0)
        order.append(min_var)
        remaining.remove(min_var)
        
        if min_var in graph:
            neighbors = list(graph.neighbors(min_var))
            for i, n1 in enumerate(neighbors):
                for n2 in neighbors[i + 1:]:
                    graph.add_edge(n1, n2)
            graph.remove_node(min_var)
    
    return order
```

## Demonstration

```python
def demonstrate_variable_elimination():
    """Compare VE with enumeration on the Weather network."""
    # Assumes build_weather_network() from Section 17.2
    bn = build_weather_network()
    
    ve = VariableElimination(bn)
    
    # P(Rain | WetGrass=1) — diagnostic reasoning
    result = ve.query(['Rain'], {'WetGrass': 1}, verbose=True)
    print(f"\nP(Rain=0 | WetGrass=1) = {result[0]:.4f}")
    print(f"P(Rain=1 | WetGrass=1) = {result[1]:.4f}")
    
    # P(Rain, Sprinkler | WetGrass=1)
    result2 = ve.query(['Rain', 'Sprinkler'], {'WetGrass': 1})
    print(f"\nP(Rain, Sprinkler | WetGrass=1):")
    print(f"  Rain=0, Spr=0: {result2[0, 0]:.4f}")
    print(f"  Rain=0, Spr=1: {result2[0, 1]:.4f}")
    print(f"  Rain=1, Spr=0: {result2[1, 0]:.4f}")
    print(f"  Rain=1, Spr=1: {result2[1, 1]:.4f}")


demonstrate_variable_elimination()
```

## Summary

| Concept | Description |
|---------|-------------|
| **Factor** | Function $\phi(X_1,\ldots,X_k) \to \mathbb{R}_{\geq 0}$ |
| **Factor Product** | Combine by multiplication over union of scopes |
| **Marginalization** | Sum out a variable |
| **Variable Elimination** | Eliminate hidden variables one by one using factor operations |
| **Elimination Order** | Determines size of intermediate factors; critical for efficiency |
| **Induced Width** | Maximum factor size minus 1; determines complexity |

## Key Takeaways

1. **VE exploits structure**: works with local factors, not the full joint distribution.
2. **Order matters**: a good order yields polynomial time; a bad order yields exponential.
3. **Optimal for trees**: linear time ($O(n \cdot d^2)$) on tree-structured networks.
4. **Foundation for other algorithms**: the junction tree algorithm and belief propagation both build on the ideas of VE.
5. **Trade-off**: VE processes one query at a time; for multiple queries, the junction tree algorithm amortizes computation.

## Quantitative Finance Application

Variable elimination is directly applicable to computing conditional probabilities in credit risk networks. Consider a network of $n$ firms with sector-level dependencies:

$$P(\text{Default}_1, \ldots, \text{Default}_n \mid \text{MacroShock} = \text{severe})$$

Enumerating all $2^n$ default configurations is intractable for large portfolios. Variable elimination exploits the sparsity of the dependency graph — most firms interact only through shared sector factors — to compute portfolio loss distributions efficiently. The elimination order should process leaf firms (those not shared across sectors) first, keeping intermediate factors small and bounded by the number of sectors rather than the number of firms.
