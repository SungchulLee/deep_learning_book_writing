# Loopy Belief Propagation

## Overview

**Loopy Belief Propagation (Loopy BP)** applies the belief propagation message passing equations to graphs **with cycles**, where BP is no longer guaranteed to be exact. Despite lacking theoretical guarantees in the general case, loopy BP often produces remarkably good approximate marginals and is widely used in practice due to its simplicity and scalability.

## Motivation

Many important graphical models contain cycles:

- **Grid models** (image processing, spatial statistics): every pixel connects to its 4 or 8 neighbors, creating many short cycles
- **Fully connected CRFs** (semantic segmentation): all pairs of nodes interact
- **Financial networks**: institutions have complex, non-tree dependency structures

For these models, exact inference via the junction tree algorithm is intractable (the treewidth is too large). Loopy BP provides a practical alternative.

## The Algorithm

Loopy BP uses the same message update equations as tree BP, applied iteratively until convergence (or a maximum iteration count):

**Variable-to-factor message:**

$$\mu_{X_i \to f_a}(x_i) = \prod_{b \in \mathcal{N}(i) \setminus \{a\}} \mu_{f_b \to X_i}(x_i)$$

**Factor-to-variable message:**

$$\mu_{f_a \to X_i}(x_i) = \sum_{X_{\mathcal{N}(a) \setminus \{i\}}} f_a(X_{\mathcal{N}(a)}) \prod_{j \in \mathcal{N}(a) \setminus \{i\}} \mu_{X_j \to f_a}(x_j)$$

The key difference from tree BP: on a graph with cycles, these equations define a **fixed-point iteration** rather than a direct computation. Messages are updated repeatedly until they (hopefully) converge.

### Implementation

```python
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from itertools import product as cartesian_product


class LoopyBeliefPropagation:
    """
    Loopy Belief Propagation for approximate inference
    on factor graphs with cycles.
    """
    
    def __init__(self, factor_graph, damping: float = 0.5):
        """
        Args:
            factor_graph: The factor graph to perform inference on
            damping: Message damping factor in [0, 1].
                     0 = no damping (standard BP),
                     higher values = more damping (slower but more stable)
        """
        self.fg = factor_graph
        self.damping = damping
        
        self.var_to_factor: Dict[Tuple[str, int], torch.Tensor] = {}
        self.factor_to_var: Dict[Tuple[int, str], torch.Tensor] = {}
    
    def _init_messages(self):
        """Initialize all messages to uniform."""
        for var in self.fg.variable_nodes:
            card = self.fg.cardinalities[var]
            for fi in self.fg.var_to_factors[var]:
                self.var_to_factor[(var, fi)] = torch.ones(card) / card
                self.factor_to_var[(fi, var)] = torch.ones(card) / card
    
    def _update_var_to_factor(self, var: str, target_fi: int):
        """Update variable-to-factor message with damping."""
        card = self.fg.cardinalities[var]
        new_msg = torch.ones(card)
        
        for fi in self.fg.var_to_factors[var]:
            if fi != target_fi:
                new_msg = new_msg * self.factor_to_var.get(
                    (fi, var), torch.ones(card) / card
                )
        
        # Normalize
        if new_msg.sum() > 0:
            new_msg = new_msg / new_msg.sum()
        
        # Damping: blend new message with old
        old_msg = self.var_to_factor.get((var, target_fi), torch.ones(card) / card)
        self.var_to_factor[(var, target_fi)] = (
            (1 - self.damping) * new_msg + self.damping * old_msg
        )
    
    def _update_factor_to_var(self, factor_idx: int, target_var: str):
        """Update factor-to-variable message with damping."""
        factor = self.fg.factor_nodes[factor_idx]
        other_vars = [v for v in factor.variables if v != target_var]
        target_card = self.fg.cardinalities[target_var]
        new_msg = torch.zeros(target_card)
        
        if not other_vars:
            target_idx = factor.variables.index(target_var)
            for val in range(target_card):
                assignment = {target_var: val}
                new_msg[val] = factor.evaluate(assignment)
        else:
            other_cards = [self.fg.cardinalities[v] for v in other_vars]
            
            for target_val in range(target_card):
                total = 0.0
                for other_vals in cartesian_product(
                    *[range(c) for c in other_cards]
                ):
                    assignment = dict(zip(other_vars, other_vals))
                    assignment[target_var] = target_val
                    
                    factor_val = factor.evaluate(assignment)
                    prod = 1.0
                    for v, val in zip(other_vars, other_vals):
                        v_msg = self.var_to_factor.get(
                            (v, factor_idx),
                            torch.ones(self.fg.cardinalities[v])
                            / self.fg.cardinalities[v]
                        )
                        prod *= v_msg[val].item()
                    total += factor_val * prod
                new_msg[target_val] = total
        
        if new_msg.sum() > 0:
            new_msg = new_msg / new_msg.sum()
        
        old_msg = self.factor_to_var.get(
            (factor_idx, target_var),
            torch.ones(target_card) / target_card
        )
        self.factor_to_var[(factor_idx, target_var)] = (
            (1 - self.damping) * new_msg + self.damping * old_msg
        )
    
    def run(self, max_iterations: int = 100,
            tolerance: float = 1e-6,
            verbose: bool = False) -> Dict[str, torch.Tensor]:
        """
        Run loopy BP until convergence or max iterations.
        
        Returns approximate marginals for all variables.
        """
        self._init_messages()
        
        for iteration in range(max_iterations):
            old_messages = {
                k: v.clone() for k, v in self.factor_to_var.items()
            }
            
            # Update all messages
            for var in self.fg.variable_nodes:
                for fi in self.fg.var_to_factors[var]:
                    self._update_var_to_factor(var, fi)
            
            for fi, factor in enumerate(self.fg.factor_nodes):
                for var in factor.variables:
                    self._update_factor_to_var(fi, var)
            
            # Check convergence
            max_diff = 0.0
            for key in self.factor_to_var:
                if key in old_messages:
                    diff = torch.abs(
                        self.factor_to_var[key] - old_messages[key]
                    ).max().item()
                    max_diff = max(max_diff, diff)
            
            if verbose and (iteration + 1) % 10 == 0:
                print(f"  Iteration {iteration + 1}: max message change = {max_diff:.6f}")
            
            if max_diff < tolerance:
                if verbose:
                    print(f"  Converged at iteration {iteration + 1}")
                break
        
        # Compute beliefs
        marginals = {}
        for var in self.fg.variable_nodes:
            card = self.fg.cardinalities[var]
            belief = torch.ones(card)
            for fi in self.fg.var_to_factors[var]:
                belief = belief * self.factor_to_var.get(
                    (fi, var), torch.ones(card) / card
                )
            belief = belief / belief.sum()
            marginals[var] = belief
        
        return marginals
```

## Convergence and Accuracy

### When Does Loopy BP Work Well?

Loopy BP tends to produce good approximations when:

1. **Weak interactions**: The factors encode relatively weak dependencies (small coupling parameters)
2. **Long cycles**: The shortest cycle in the graph is long
3. **Low-degree nodes**: Variables participate in few factors
4. **Attractive potentials**: Pairwise potentials favor agreement (positive coupling)

### When Does It Fail?

Loopy BP can produce poor results or fail to converge when:

1. **Strong interactions**: Factors encode very strong preferences
2. **Short cycles**: Tight loops amplify errors in messages
3. **Frustrated systems**: Competing constraints that cannot all be satisfied (e.g., antiferromagnetic Ising on odd cycles)
4. **Multi-modal distributions**: BP tends to spread mass across modes rather than concentrating on them

### Damping

**Message damping** is the most common technique for improving convergence:

$$\mu^{(t+1)} = (1 - \alpha) \cdot \mu^{\text{new}} + \alpha \cdot \mu^{(t)}$$

where $\alpha \in [0, 1)$ is the damping factor. Higher damping slows convergence but reduces oscillation. A typical value is $\alpha = 0.5$.

## Connection to Variational Inference

Loopy BP has a deep connection to variational inference. The Bethe free energy approximation provides a variational interpretation:

$$F_{\text{Bethe}}(q) = \sum_a \sum_{x_a} q_a(x_a) \ln \frac{q_a(x_a)}{f_a(x_a)} - \sum_i (d_i - 1) \sum_{x_i} q_i(x_i) \ln q_i(x_i)$$

where $q_a$ are factor beliefs, $q_i$ are variable beliefs, and $d_i$ is the degree of variable $i$.

The fixed points of loopy BP correspond to **stationary points of the Bethe free energy**. This connection:

- Explains why loopy BP sometimes works well (the Bethe approximation is often accurate)
- Provides a convergence diagnostic (monitor the Bethe free energy)
- Motivates improvements like tree-reweighted BP and convex relaxations

## Variants and Improvements

### Tree-Reweighted BP (TRW-BP)

Decomposes the graph into a convex combination of spanning trees and runs BP on each tree. Provides an upper bound on the log-partition function, unlike standard loopy BP.

### Generalized BP

Uses larger clusters (regions) instead of individual factors, providing better approximations at higher computational cost. The cluster variational method generalizes the Bethe approximation to the Kikuchi approximation.

### Residual BP

Updates messages in order of their residual (how much they would change), focusing computation where it matters most. This adaptive schedule often converges faster than the standard parallel or random schedule.

## Demonstration: Ising Model on a Grid

```python
def demonstrate_loopy_bp():
    """Compare loopy BP with exact inference on a small Ising model."""
    # Create a 3x3 Ising model (too large for brute force, cycles present)
    from ch17.pgm_undirected_models.mrf import create_ising_model
    
    mrf = create_ising_model(3, 3, field=0.1, coupling=0.5)
    
    # Convert MRF to factor graph
    from ch17.pgm_undirected_models.factor_graphs import FactorGraph
    fg = FactorGraph.from_mrf(mrf)
    
    # Run loopy BP
    lbp = LoopyBeliefPropagation(fg, damping=0.3)
    marginals = lbp.run(max_iterations=50, verbose=True)
    
    print("\nApproximate marginals from Loopy BP:")
    for var in sorted(marginals.keys()):
        p1 = marginals[var][1].item()
        print(f"  P({var}=1) = {p1:.4f}")
    
    # Compare with exact (brute force) for this small model
    print("\nExact marginals (brute force):")
    for var in sorted(mrf.variables):
        exact_marg = mrf.marginal(var)
        print(f"  P({var}=1) = {exact_marg[1].item():.4f}")


demonstrate_loopy_bp()
```

## Summary

| Concept | Description |
|---------|-------------|
| **Loopy BP** | BP message passing on graphs with cycles |
| **Fixed-Point Iteration** | Messages updated iteratively until convergence |
| **Damping** | Blend old and new messages for stability |
| **Bethe Free Energy** | Variational objective whose stationary points = BP fixed points |
| **TRW-BP** | Tree-reweighted variant with theoretical guarantees |
| **Convergence** | Not guaranteed; depends on graph structure and coupling strength |

## Key Takeaways

1. **Practical workhorse**: Despite lack of guarantees, loopy BP is the go-to approximate inference method for many applications.
2. **Damping helps**: Almost always use damping ($\alpha \approx 0.3\text{--}0.5$) for graphs with short cycles.
3. **Monitor convergence**: Track message changes and Bethe free energy to assess reliability.
4. **Variational connection**: Loopy BP optimizes the Bethe free energy, providing theoretical insight into when and why it works.
5. **Scalable**: Linear in the number of edges per iteration, making it applicable to very large models.

## Quantitative Finance Application

Loopy BP is applicable to systemic risk assessment in financial networks where exact inference is intractable. Consider a network of 100+ financial institutions with complex cross-holdings and counterparty exposures forming a dense graph. Exact inference (junction tree) is infeasible due to the large treewidth, but loopy BP can efficiently approximate the marginal probability of each institution's distress given observed defaults elsewhere. The damped message passing converges reliably when coupling strengths (exposure levels) are moderate, which is typical in diversified portfolios. The Bethe free energy provides a diagnostic: if it fails to decrease monotonically, the approximation may be unreliable for that particular stress scenario.
