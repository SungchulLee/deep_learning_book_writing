# Belief Propagation

## Overview

**Belief Propagation (BP)**, also called the **sum-product algorithm**, is a message passing algorithm for computing exact marginals on tree-structured graphical models. It generalizes variable elimination by simultaneously computing marginals for *all* variables in a single pass, rather than answering one query at a time.

## From Variable Elimination to Message Passing

Variable elimination eliminates variables one at a time, producing intermediate factors. On a tree-structured graph, these intermediate computations can be organized as **messages** flowing along edges. Each message summarizes the "belief" that one part of the tree sends to another about a shared variable.

The key insight: if we run VE rooted at every variable, many intermediate computations are shared. Belief propagation caches these computations as messages, achieving the efficiency of computing *all* marginals in $O(n \cdot d^2)$ time on a tree (versus $O(n^2 \cdot d^2)$ for running VE separately for each variable).

## Message Passing on Trees

### Setup

Consider a tree-structured factor graph with variable nodes $\{X_1, \ldots, X_n\}$ and factor nodes $\{f_1, \ldots, f_m\}$. Messages pass between variable nodes and factor nodes along edges.

### Message Definitions

**Variable-to-factor message**: variable $X_i$ sends to factor $f_a$:

$$\mu_{X_i \to f_a}(x_i) = \prod_{b \in \mathcal{N}(i) \setminus \{a\}} \mu_{f_b \to X_i}(x_i)$$

This is the product of all incoming factor messages *except* from the target factor.

**Factor-to-variable message**: factor $f_a$ sends to variable $X_i$:

$$\mu_{f_a \to X_i}(x_i) = \sum_{X_{\mathcal{N}(a) \setminus \{i\}}} f_a(X_{\mathcal{N}(a)}) \prod_{j \in \mathcal{N}(a) \setminus \{i\}} \mu_{X_j \to f_a}(x_j)$$

The factor multiplies its potential by all incoming variable messages (except from $X_i$), then marginalizes out all variables except $X_i$.

### Computing Marginals (Beliefs)

Once all messages have been computed, the marginal of $X_i$ is:

$$P(X_i = x_i) \propto \prod_{a \in \mathcal{N}(i)} \mu_{f_a \to X_i}(x_i)$$

The joint marginal over variables in factor $f_a$'s scope is:

$$P(X_{\mathcal{N}(a)}) \propto f_a(X_{\mathcal{N}(a)}) \prod_{j \in \mathcal{N}(a)} \mu_{X_j \to f_a}(x_j)$$

## Algorithm: Two-Pass Schedule on Trees

On a tree, BP converges in exactly two passes:

1. **Collect phase** (leaves → root): Each leaf sends a message to its parent. Messages propagate inward until the root has received messages from all neighbors.

2. **Distribute phase** (root → leaves): The root sends messages to all neighbors. Messages propagate outward until all leaves have received messages.

After both passes, every edge has messages in both directions, and all marginals can be computed.

### Implementation

```python
import torch
from typing import Dict, List, Optional, Set, Tuple
from itertools import product as cartesian_product
from collections import deque


class BeliefPropagation:
    """
    Sum-product belief propagation on a factor graph.
    
    Computes exact marginals on tree-structured graphs.
    On graphs with cycles, this becomes loopy BP (approximate).
    """
    
    def __init__(self, factor_graph):
        self.fg = factor_graph
        # Messages indexed by (source_type, source_id, target_type, target_id)
        self.var_to_factor: Dict[Tuple[str, int], torch.Tensor] = {}
        self.factor_to_var: Dict[Tuple[int, str], torch.Tensor] = {}
    
    def _init_messages(self):
        """Initialize all messages to uniform."""
        for var in self.fg.variable_nodes:
            card = self.fg.cardinalities[var]
            for fi in self.fg.var_to_factors[var]:
                self.var_to_factor[(var, fi)] = torch.ones(card)
                self.factor_to_var[(fi, var)] = torch.ones(card)
    
    def _update_var_to_factor(self, var: str, target_fi: int):
        """
        mu_{Xi -> fa}(xi) = prod_{b != a} mu_{fb -> Xi}(xi)
        """
        card = self.fg.cardinalities[var]
        msg = torch.ones(card)
        
        for fi in self.fg.var_to_factors[var]:
            if fi != target_fi:
                msg = msg * self.factor_to_var.get((fi, var), torch.ones(card))
        
        # Normalize for numerical stability
        if msg.sum() > 0:
            msg = msg / msg.sum()
        
        self.var_to_factor[(var, target_fi)] = msg
    
    def _update_factor_to_var(self, factor_idx: int, target_var: str):
        """
        mu_{fa -> Xi}(xi) = sum_{Xn(a)\\i} fa(Xn(a)) prod_{j!=i} mu_{Xj->fa}(xj)
        """
        factor = self.fg.factor_nodes[factor_idx]
        other_vars = [v for v in factor.variables if v != target_var]
        target_card = self.fg.cardinalities[target_var]
        msg = torch.zeros(target_card)
        
        if not other_vars:
            # Unary factor
            target_idx = factor.variables.index(target_var)
            for val in range(target_card):
                assignment = {target_var: val}
                msg[val] = factor.evaluate(assignment)
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
                        )
                        prod *= v_msg[val].item()
                    
                    total += factor_val * prod
                msg[target_val] = total
        
        # Normalize
        if msg.sum() > 0:
            msg = msg / msg.sum()
        
        self.factor_to_var[(factor_idx, target_var)] = msg
    
    def run(self, n_iterations: int = 10) -> Dict[str, torch.Tensor]:
        """
        Run belief propagation and return marginals.
        
        For trees, converges in iterations = diameter of tree.
        """
        self._init_messages()
        
        for iteration in range(n_iterations):
            # Update all variable-to-factor messages
            for var in self.fg.variable_nodes:
                for fi in self.fg.var_to_factors[var]:
                    self._update_var_to_factor(var, fi)
            
            # Update all factor-to-variable messages
            for fi, factor in enumerate(self.fg.factor_nodes):
                for var in factor.variables:
                    self._update_factor_to_var(fi, var)
        
        # Compute beliefs (marginals)
        marginals = {}
        for var in self.fg.variable_nodes:
            card = self.fg.cardinalities[var]
            belief = torch.ones(card)
            for fi in self.fg.var_to_factors[var]:
                belief = belief * self.factor_to_var.get(
                    (fi, var), torch.ones(card)
                )
            belief = belief / belief.sum()
            marginals[var] = belief
        
        return marginals
```

## Max-Product Algorithm

By replacing summation with maximization in the factor-to-variable messages, belief propagation computes the **MAP (Maximum A Posteriori)** assignment instead of marginals:

$$\mu_{f_a \to X_i}^{\max}(x_i) = \max_{X_{\mathcal{N}(a) \setminus \{i\}}} f_a(X_{\mathcal{N}(a)}) \prod_{j \in \mathcal{N}(a) \setminus \{i\}} \mu_{X_j \to f_a}^{\max}(x_j)$$

In practice, it is common to work in log-space (the **max-sum** algorithm) to avoid numerical underflow:

$$\nu_{f_a \to X_i}(x_i) = \max_{X_{\mathcal{N}(a) \setminus \{i\}}} \left[\log f_a(X_{\mathcal{N}(a)}) + \sum_{j \neq i} \nu_{X_j \to f_a}(x_j)\right]$$

The MAP assignment is recovered by backtracking through the maximizing configurations at each step, analogous to the Viterbi algorithm for Hidden Markov Models.

## BP on Pairwise Models

For models with only pairwise factors $\psi_{ij}(X_i, X_j)$ and unary factors $\phi_i(X_i)$, the messages simplify to direct variable-to-variable messages:

$$m_{i \to j}(x_j) = \sum_{x_i} \psi_{ij}(x_i, x_j) \cdot \phi_i(x_i) \cdot \prod_{k \in \mathcal{N}(i) \setminus \{j\}} m_{k \to i}(x_i)$$

This is the form most commonly seen in textbooks and is equivalent to the factor graph formulation when each pairwise potential is represented as a single factor node.

## Convergence and Exactness

| Graph Structure | Convergence | Result |
|----------------|-------------|--------|
| Tree | Exact in 2 passes | Exact marginals |
| Tree (with evidence) | Exact in 2 passes | Exact posterior marginals |
| Graph with cycles | Not guaranteed | Approximate marginals |
| Single cycle | Usually converges | Often good approximation |
| Dense graph | May oscillate | Can be poor |

On trees, BP computes the same result as variable elimination but simultaneously for all variables. The complexity is $O(n \cdot d^2)$ for pairwise models and $O(n \cdot d^{k+1})$ for models with factors involving up to $k+1$ variables.

## Relationship to Other Algorithms

BP unifies several well-known algorithms as special cases:

- **Forward-backward algorithm** for HMMs: BP on a chain-structured factor graph
- **Viterbi algorithm**: Max-product BP on a chain
- **Kalman filter/smoother**: Gaussian BP on a chain with continuous variables
- **Pearl's algorithm**: The original formulation of BP on Bayesian network trees

## Summary

| Concept | Description |
|---------|-------------|
| **Belief Propagation** | Message passing algorithm for computing marginals |
| **Variable-to-Factor Message** | Product of all incoming factor messages except target |
| **Factor-to-Variable Message** | Factor times incoming variable messages, marginalized |
| **Belief** | Product of all incoming messages at a variable node |
| **Sum-Product** | Computes marginals |
| **Max-Product** | Computes MAP assignment |
| **Tree Exactness** | BP is exact on tree-structured graphs |

## Quantitative Finance Application

Belief propagation on tree-structured models is directly applicable to pricing and risk management in hierarchical financial systems. A tree-structured factor model with global market → sector → industry → firm layers can be solved exactly via BP, computing marginal default probabilities for every firm simultaneously. The forward-backward interpretation of BP on chains underlies the Baum-Welch algorithm used for calibrating Hidden Markov Models of market regimes, where latent states (bull/bear/crisis) drive observed returns.
