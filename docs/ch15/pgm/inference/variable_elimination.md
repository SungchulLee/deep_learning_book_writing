# Variable Elimination Algorithm

## The Problem with Enumeration

Inference by enumeration has exponential complexity: $O(d^n)$ where $d$ is the maximum cardinality and $n$ is the number of variables. For a network with 20 binary variables, this means summing over $2^{20} \approx 1$ million terms!

**Variable Elimination** exploits the factored structure of Bayesian Networks to avoid this exponential blowup by:
1. Working with local factors instead of the full joint
2. Eliminating variables one at a time
3. Only multiplying factors that share variables

## The Key Insight: Pushing Sums Inside Products

Consider computing $P(X)$ by marginalizing over $Y$ and $Z$:

$$P(X) = \sum_Y \sum_Z P(X, Y, Z)$$

If the joint factors as $P(X, Y, Z) = P(X) \cdot P(Y|X) \cdot P(Z|Y)$:

$$P(X) = \sum_Y \sum_Z P(X) \cdot P(Y|X) \cdot P(Z|Y)$$

Since $P(X)$ doesn't depend on $Y$ or $Z$:

$$P(X) = P(X) \cdot \sum_Y P(Y|X) \cdot \sum_Z P(Z|Y)$$

The inner sum $\sum_Z P(Z|Y) = 1$ (normalization), and $\sum_Y P(Y|X) = 1$, so:

$$P(X) = P(X)$$

This shows how we can **push sums inside products** when variables don't appear in certain factors.

## Factors: The Fundamental Data Structure

A **factor** $\phi(X_1, \ldots, X_k)$ is a function that maps each assignment of values to a non-negative real number.

### Factor Operations

#### 1. Factor Product

Combines two factors by multiplying values for consistent assignments:

$$(\phi_1 \times \phi_2)(X, Y, Z) = \phi_1(X, Y) \cdot \phi_2(Y, Z)$$

The result is a factor over the union of variables.

#### 2. Factor Marginalization (Sum-Out)

Eliminates a variable by summing over its values:

$$\phi'(X) = \sum_Y \phi(X, Y)$$

#### 3. Factor Reduction

Fixes variables to observed values:

$$\phi'(X) = \phi(X, Y=y)$$

```python
import torch
from typing import Dict, List, Set, Optional, Tuple
from itertools import product as cartesian_product

class Factor:
    """
    A factor over discrete random variables.
    
    Represents a function φ(X₁, ..., Xₖ) → ℝ≥0
    
    Factors are the fundamental data structure for variable elimination.
    They generalize both probability distributions and potential functions.
    """
    
    def __init__(self,
                 variables: List[str],
                 cardinalities: Dict[str, int],
                 values: Optional[torch.Tensor] = None):
        """
        Initialize a factor.
        
        Args:
            variables: List of variable names (order matters!)
            cardinalities: Dict mapping variable names to number of values
            values: Tensor of factor values (optional)
        """
        self.variables = list(variables)
        self.cardinalities = cardinalities
        
        # Compute shape from variables
        self.shape = tuple(cardinalities[v] for v in self.variables)
        
        if values is None:
            self.values = torch.ones(self.shape)
        else:
            self.values = values.float()
            assert self.values.shape == self.shape, \
                f"Shape mismatch: {self.values.shape} vs expected {self.shape}"
    
    @property
    def scope(self) -> Set[str]:
        """Variables in this factor's scope."""
        return set(self.variables)
    
    def multiply(self, other: 'Factor') -> 'Factor':
        """
        Compute factor product: φ₁ × φ₂.
        
        (φ₁ × φ₂)(X, Y, Z) = φ₁(X, Y) · φ₂(Y, Z)
        
        Args:
            other: Factor to multiply with
            
        Returns:
            Product factor over union of variables
        """
        # Determine variables in result
        new_vars = self.variables.copy()
        for v in other.variables:
            if v not in new_vars:
                new_vars.append(v)
        
        # Create result factor
        result = Factor(new_vars, self.cardinalities)
        
        # Enumerate all assignments
        for assignment in self._enumerate_assignments(new_vars):
            # Get values from both factors
            val1 = self._get_value(assignment)
            val2 = other._get_value(assignment)
            
            # Set product
            result._set_value(assignment, val1 * val2)
        
        return result
    
    def marginalize(self, variables_to_sum: List[str]) -> 'Factor':
        """
        Sum out (marginalize) variables.
        
        φ'(X) = Σᵧ φ(X, Y)
        
        Args:
            variables_to_sum: Variables to eliminate
            
        Returns:
            Factor with specified variables summed out
        """
        # Variables remaining after marginalization
        remaining_vars = [v for v in self.variables if v not in variables_to_sum]
        
        if not remaining_vars:
            # Summing out all variables gives a scalar
            return Factor([], self.cardinalities, 
                         torch.tensor([self.values.sum()]))
        
        # Find axes to sum over
        axes = tuple(self.variables.index(v) for v in variables_to_sum if v in self.variables)
        
        # Sum over those axes
        new_values = self.values.sum(dim=axes)
        
        return Factor(remaining_vars, self.cardinalities, new_values)
    
    def reduce(self, evidence: Dict[str, int]) -> 'Factor':
        """
        Reduce factor by fixing evidence variables.
        
        φ'(X) = φ(X, E=e)
        
        Args:
            evidence: Dict mapping variable names to observed values
            
        Returns:
            Reduced factor with evidence variables removed
        """
        # Build indexing
        indices = []
        remaining_vars = []
        
        for v in self.variables:
            if v in evidence:
                indices.append(evidence[v])
            else:
                indices.append(slice(None))
                remaining_vars.append(v)
        
        # Extract slice
        new_values = self.values[tuple(indices)]
        
        if not remaining_vars:
            return Factor([], self.cardinalities, torch.tensor([new_values.item() if new_values.dim() == 0 else new_values]))
        
        return Factor(remaining_vars, self.cardinalities, new_values)
    
    def normalize(self) -> 'Factor':
        """Normalize to sum to 1 (convert to probability distribution)."""
        total = self.values.sum()
        if total > 0:
            return Factor(self.variables, self.cardinalities, self.values / total)
        return self
    
    def _get_value(self, assignment: Dict[str, int]) -> float:
        """Get factor value for an assignment."""
        if not self.variables:
            return self.values.item() if self.values.dim() == 0 else self.values[0].item()
        index = tuple(assignment[v] for v in self.variables)
        return self.values[index].item()
    
    def _set_value(self, assignment: Dict[str, int], value: float):
        """Set factor value for an assignment."""
        if not self.variables:
            self.values = torch.tensor([value])
        else:
            index = tuple(assignment[v] for v in self.variables)
            self.values[index] = value
    
    def _enumerate_assignments(self, variables: List[str]):
        """Generate all assignments to specified variables."""
        if not variables:
            yield {}
            return
        
        cards = [self.cardinalities[v] for v in variables]
        for values in cartesian_product(*[range(c) for c in cards]):
            yield dict(zip(variables, values))
    
    def __repr__(self) -> str:
        var_str = ','.join(self.variables) if self.variables else '∅'
        return f"Factor({var_str}, shape={self.shape})"
```

## The Variable Elimination Algorithm

### Algorithm Overview

Given:
- A Bayesian Network with factors $\{\phi_1, \ldots, \phi_m\}$
- Query variables $Q$
- Evidence variables $E = e$
- Hidden variables $H$ (all others)

**Goal**: Compute $P(Q \mid E = e)$

### Algorithm Steps

1. **Initialize**: Convert each CPT to a factor
2. **Reduce**: Apply evidence by reducing all factors containing evidence variables
3. **Eliminate**: For each hidden variable $H_i$ in elimination order:
   - Collect all factors containing $H_i$
   - Multiply them together
   - Sum out $H_i$
   - Add resulting factor back to factor list
4. **Multiply**: Multiply all remaining factors
5. **Normalize**: Divide by sum to get conditional probability

```python
class VariableElimination:
    """
    Variable Elimination algorithm for exact inference in Bayesian Networks.
    
    Time complexity: O(n · d^(w+1)) where:
    - n = number of variables
    - d = maximum cardinality
    - w = induced width (depends on elimination order)
    """
    
    def __init__(self, bn):
        """
        Initialize with a Bayesian Network.
        
        Args:
            bn: The Bayesian Network to perform inference on
        """
        self.bn = bn
    
    def _cpt_to_factor(self, variable: str) -> Factor:
        """Convert a CPT to a Factor."""
        cpt = self.bn.cpts[variable]
        variables = cpt.parents + [cpt.variable]
        return Factor(variables, self.bn.cardinalities, cpt.values)
    
    def query(self,
              query_vars: List[str],
              evidence: Dict[str, int] = None,
              elimination_order: List[str] = None,
              verbose: bool = False) -> torch.Tensor:
        """
        Perform inference using variable elimination.
        
        Args:
            query_vars: Variables to compute distribution over
            evidence: Observed variable values
            elimination_order: Order to eliminate hidden variables
            verbose: Whether to print progress
            
        Returns:
            Tensor of shape (card_q1, ...) with P(query | evidence)
        """
        if evidence is None:
            evidence = {}
        
        if verbose:
            print(f"\nVariable Elimination Query")
            print(f"Query: P({', '.join(query_vars)}" + 
                  (f" | {', '.join(f'{k}={v}' for k,v in evidence.items())})" if evidence else ")"))
            print("-" * 60)
        
        # Step 1: Create factors from CPTs
        factors = [self._cpt_to_factor(var) for var in self.bn.variables]
        
        if verbose:
            print(f"\nStep 1: Created {len(factors)} factors from CPTs")
            for f in factors:
                print(f"  {f}")
        
        # Step 2: Reduce factors with evidence
        if evidence:
            factors = [f.reduce(evidence) for f in factors]
            # Remove empty factors
            factors = [f for f in factors if f.variables or f.values.numel() > 0]
            
            if verbose:
                print(f"\nStep 2: Reduced factors with evidence")
                for f in factors:
                    print(f"  {f}")
        
        # Step 3: Determine elimination order
        all_vars = set(self.bn.variables)
        hidden_vars = all_vars - set(query_vars) - set(evidence.keys())
        
        if elimination_order is None:
            # Default: eliminate in reverse topological order
            elimination_order = [v for v in reversed(self.bn.variables) 
                                if v in hidden_vars]
        
        if verbose:
            print(f"\nStep 3: Elimination order: {elimination_order}")
        
        # Step 4: Eliminate hidden variables one by one
        for var in elimination_order:
            if verbose:
                print(f"\nEliminating {var}:")
            
            # Find factors containing this variable
            relevant = [f for f in factors if var in f.scope]
            others = [f for f in factors if var not in f.scope]
            
            if not relevant:
                continue
            
            if verbose:
                print(f"  Relevant factors: {[repr(f) for f in relevant]}")
            
            # Multiply relevant factors
            product = relevant[0]
            for f in relevant[1:]:
                product = product.multiply(f)
            
            if verbose:
                print(f"  Product scope: {product.variables}")
            
            # Sum out the variable
            marginalized = product.marginalize([var])
            
            if verbose:
                print(f"  After marginalizing {var}: {marginalized}")
            
            # Update factor list
            factors = others + [marginalized]
        
        # Step 5: Multiply remaining factors
        if verbose:
            print(f"\nStep 5: Multiplying {len(factors)} remaining factors")
        
        result = factors[0]
        for f in factors[1:]:
            result = result.multiply(f)
        
        # Step 6: Normalize
        result = result.normalize()
        
        if verbose:
            print(f"\nStep 6: Normalized result")
            print(f"Final factor: {result}")
        
        # Reorder dimensions to match query_vars order
        if result.variables != query_vars:
            perm = [result.variables.index(v) for v in query_vars if v in result.variables]
            if perm:
                result.values = result.values.permute(*perm)
                result.variables = query_vars
        
        return result.values
```

## Elimination Ordering

The efficiency of variable elimination depends critically on the **elimination order**. Different orders can lead to:

- **Best case**: Polynomial time (when network is tree-structured)
- **Worst case**: Exponential time (equivalent to enumeration)

### Induced Width

The **induced width** (or treewidth) is the maximum size of any factor created during elimination minus 1. It determines the complexity.

For elimination order $\pi$:
$$\text{Complexity} = O(n \cdot d^{w_\pi + 1})$$

where $w_\pi$ is the induced width under order $\pi$.

### Finding Good Orders

Finding the optimal elimination order is NP-hard, but heuristics work well:

1. **Min-Fill**: Eliminate the variable that adds the fewest edges to the induced graph
2. **Min-Degree**: Eliminate the variable with fewest neighbors
3. **Min-Weight**: Eliminate the variable minimizing factor size

```python
def min_degree_order(factors: List[Factor], hidden_vars: Set[str], 
                     cardinalities: Dict[str, int]) -> List[str]:
    """
    Compute elimination order using min-degree heuristic.
    
    At each step, eliminate the variable with fewest neighbors
    in the current interaction graph.
    """
    import networkx as nx
    
    # Build interaction graph (variables that appear together in factors)
    graph = nx.Graph()
    graph.add_nodes_from(hidden_vars)
    
    for f in factors:
        vars_in_hidden = [v for v in f.scope if v in hidden_vars]
        for i, v1 in enumerate(vars_in_hidden):
            for v2 in vars_in_hidden[i+1:]:
                graph.add_edge(v1, v2)
    
    order = []
    remaining = set(hidden_vars)
    
    while remaining:
        # Find variable with minimum degree
        min_var = min(remaining, 
                      key=lambda v: graph.degree(v) if v in graph else 0)
        order.append(min_var)
        remaining.remove(min_var)
        
        # Connect neighbors (fill-in edges)
        if min_var in graph:
            neighbors = list(graph.neighbors(min_var))
            for i, n1 in enumerate(neighbors):
                for n2 in neighbors[i+1:]:
                    graph.add_edge(n1, n2)
            graph.remove_node(min_var)
    
    return order
```

## Complexity Analysis

### Time Complexity

$$O(n \cdot d^{w+1})$$

where:
- $n$ = number of variables
- $d$ = maximum cardinality
- $w$ = induced width

### Space Complexity

$$O(d^{w+1})$$

The largest factor created has at most $w+1$ variables.

### When VE is Efficient

| Network Structure | Induced Width | Complexity |
|-------------------|---------------|------------|
| Chain | 1 | $O(n \cdot d^2)$ |
| Tree | 1 | $O(n \cdot d^2)$ |
| Polytree | 1 | $O(n \cdot d^2)$ |
| Grid (n×n) | $n-1$ | $O(n^2 \cdot d^n)$ |
| Complete graph | $n-1$ | $O(n \cdot d^n)$ |

## Worked Example

Consider the chain $A \rightarrow B \rightarrow C$ with query $P(C \mid A=0)$:

**Initial factors:**
- $\phi_A(A) = P(A)$
- $\phi_B(A, B) = P(B|A)$
- $\phi_C(B, C) = P(C|B)$

**Step 1: Reduce with evidence $A=0$:**
- $\phi'_A = \phi_A(A=0)$ (scalar)
- $\phi'_B(B) = \phi_B(A=0, B)$ (now only over $B$)
- $\phi_C(B, C)$ unchanged

**Step 2: Eliminate $B$:**
- Multiply: $\psi(B, C) = \phi'_B(B) \cdot \phi_C(B, C)$
- Marginalize: $\tau(C) = \sum_B \psi(B, C)$

**Step 3: Multiply remaining factors:**
- $\phi'_A$ is just a constant
- Result: $\tau(C)$

**Step 4: Normalize:**
- $P(C \mid A=0) = \tau(C) / \sum_C \tau(C)$

## Summary

| Concept | Description |
|---------|-------------|
| **Factor** | Function $\phi(X_1,\ldots,X_k) \rightarrow \mathbb{R}_{\geq 0}$ |
| **Factor Product** | Combine by multiplication |
| **Marginalization** | Sum out a variable |
| **Variable Elimination** | Eliminate hidden variables one by one |
| **Elimination Order** | Critical for efficiency |
| **Induced Width** | Determines complexity |

## Key Takeaways

1. **VE exploits structure**: Works with local factors, not full joint
2. **Order matters**: Good order → polynomial, bad order → exponential
3. **Optimal for trees**: Linear time on tree-structured networks
4. **Foundation for other algorithms**: Junction tree, belief propagation
5. **Trade-off**: Time vs space (can recompute or cache intermediate factors)
