# Markov Random Fields and Undirected Graphical Models

## Overview

**Markov Random Fields (MRFs)**, also called undirected graphical models, use undirected graphs to represent probability distributions. Unlike Bayesian Networks, MRFs model symmetric relationships without directional semantics.

## Key Differences from Bayesian Networks

| Aspect | Bayesian Network | Markov Random Field |
|--------|------------------|---------------------|
| Graph type | Directed (DAG) | Undirected |
| Edge meaning | Causal/generative | Symmetric constraint |
| Factorization | CPTs: $P(X_i \mid \text{Pa})$ | Potentials: $\psi_C(X_C)$ |
| Partition function | Not needed (normalized) | Required: $Z = \sum_x \prod_C \psi_C$ |
| Independence test | D-separation | Graph separation |

## The Gibbs Distribution

MRFs define distributions using the **Gibbs distribution**:

$$P(X) = \frac{1}{Z} \prod_{C \in \mathcal{C}} \psi_C(X_C)$$

where:
- $\mathcal{C}$ is the set of cliques in the graph
- $\psi_C(X_C)$ is a **potential function** (non-negative) over clique $C$
- $Z = \sum_x \prod_{C \in \mathcal{C}} \psi_C(x_C)$ is the **partition function**

### The Partition Function Problem

Unlike Bayesian networks, MRFs require computing $Z$ to normalize the distribution. This sum over all configurations is exponential and often intractable:

$$Z = \sum_{x_1} \sum_{x_2} \cdots \sum_{x_n} \prod_{C \in \mathcal{C}} \psi_C(x_C)$$

This is a fundamental challenge in working with MRFs.

## Potential Functions

Potentials are **not probabilities**—they can take any non-negative value. Common forms:

### 1. Log-Linear (Exponential Family)

$$\psi_C(X_C) = \exp\left(\sum_k \theta_k f_k(X_C)\right)$$

where $f_k$ are **feature functions** and $\theta_k$ are parameters.

### 2. Pairwise Potentials

For edges $(i, j)$:

$$\psi_{ij}(X_i, X_j) = \exp(\theta_{ij}^{X_i, X_j})$$

Often represented as a matrix of compatibility scores.

```python
import torch
import torch.nn.functional as F
from typing import Dict, List, Set, Tuple, Optional
import numpy as np

class UndirectedFactor:
    """
    A factor/potential function for undirected graphical models.
    
    Unlike CPTs, factors need not sum to 1.
    """
    
    def __init__(self,
                 variables: List[str],
                 cardinalities: Dict[str, int],
                 values: Optional[torch.Tensor] = None):
        """
        Initialize a factor.
        
        Args:
            variables: Variables in this factor's scope
            cardinalities: Cardinality of each variable
            values: Potential values (non-negative)
        """
        self.variables = variables
        self.cardinalities = cardinalities
        self.shape = tuple(cardinalities[v] for v in variables)
        
        if values is None:
            self.values = torch.ones(self.shape)
        else:
            self.values = values.float()
            assert (self.values >= 0).all(), "Potentials must be non-negative"
    
    def evaluate(self, assignment: Dict[str, int]) -> float:
        """Evaluate potential for an assignment."""
        index = tuple(assignment[v] for v in self.variables)
        return self.values[index].item()


class MarkovRandomField:
    """
    Markov Random Field (Undirected Graphical Model).
    
    Defines: P(X) = (1/Z) ∏_C ψ_C(X_C)
    """
    
    def __init__(self):
        self.variables: List[str] = []
        self.cardinalities: Dict[str, int] = {}
        self.factors: List[UndirectedFactor] = []
        self._partition_function: Optional[float] = None
    
    def add_variable(self, name: str, cardinality: int):
        """Add a variable to the MRF."""
        self.variables.append(name)
        self.cardinalities[name] = cardinality
        self._partition_function = None  # Invalidate cache
    
    def add_factor(self, variables: List[str], values: torch.Tensor):
        """Add a potential function over specified variables."""
        factor = UndirectedFactor(variables, self.cardinalities, values)
        self.factors.append(factor)
        self._partition_function = None
    
    def unnormalized_probability(self, assignment: Dict[str, int]) -> float:
        """
        Compute unnormalized probability: ∏_C ψ_C(X_C).
        """
        prob = 1.0
        for factor in self.factors:
            prob *= factor.evaluate(assignment)
        return prob
    
    def compute_partition_function(self) -> float:
        """
        Compute partition function Z = Σ_x ∏_C ψ_C(x_C).
        
        Warning: Exponential in number of variables!
        """
        if self._partition_function is not None:
            return self._partition_function
        
        from itertools import product as cartesian_product
        
        Z = 0.0
        cards = [self.cardinalities[v] for v in self.variables]
        
        for values in cartesian_product(*[range(c) for c in cards]):
            assignment = dict(zip(self.variables, values))
            Z += self.unnormalized_probability(assignment)
        
        self._partition_function = Z
        return Z
    
    def probability(self, assignment: Dict[str, int]) -> float:
        """Compute normalized probability P(X = assignment)."""
        Z = self.compute_partition_function()
        return self.unnormalized_probability(assignment) / Z
    
    def marginal(self, variable: str) -> torch.Tensor:
        """Compute marginal distribution P(variable)."""
        from itertools import product as cartesian_product
        
        card = self.cardinalities[variable]
        marginal = torch.zeros(card)
        
        other_vars = [v for v in self.variables if v != variable]
        other_cards = [self.cardinalities[v] for v in other_vars]
        
        for val in range(card):
            prob_sum = 0.0
            for other_vals in cartesian_product(*[range(c) for c in other_cards]):
                assignment = dict(zip(other_vars, other_vals))
                assignment[variable] = val
                prob_sum += self.unnormalized_probability(assignment)
            marginal[val] = prob_sum
        
        # Normalize
        marginal = marginal / marginal.sum()
        return marginal
```

## The Hammersley-Clifford Theorem

This fundamental theorem connects graph structure to factorization:

**Theorem**: A strictly positive distribution $P(X) > 0$ satisfies the **local Markov property** with respect to an undirected graph $G$ if and only if $P$ factorizes according to the cliques of $G$:

$$P(X) = \frac{1}{Z} \prod_{C \in \mathcal{C}} \psi_C(X_C)$$

### Local Markov Property

A node is independent of all other nodes given its **neighbors** (Markov blanket):

$$X_i \perp\!\!\!\perp X_{V \setminus \{i\} \setminus N_i} \mid X_{N_i}$$

where $N_i$ are the neighbors of node $i$.

## Classic MRF Examples

### 1. Ising Model

Binary variables on a grid with pairwise interactions:

$$P(X) = \frac{1}{Z} \exp\left(\sum_{i} \theta_i X_i + \sum_{(i,j) \in E} \theta_{ij} X_i X_j\right)$$

**Applications**: Statistical physics, image denoising

### 2. Potts Model

Generalization of Ising to $k$ states:

$$P(X) = \frac{1}{Z} \exp\left(\sum_{(i,j) \in E} \beta \cdot \mathbf{1}[X_i = X_j]\right)$$

**Applications**: Image segmentation, community detection

```python
def create_ising_model(n_rows: int, n_cols: int, 
                       field: float = 0.0,
                       coupling: float = 1.0) -> MarkovRandomField:
    """
    Create an Ising model on a grid.
    
    Energy: E(X) = -h Σᵢ Xᵢ - J Σ_{i~j} Xᵢ Xⱼ
    P(X) ∝ exp(-E(X))
    
    Args:
        n_rows, n_cols: Grid dimensions
        field: External field strength (h)
        coupling: Coupling strength (J)
        
    Returns:
        MarkovRandomField
    """
    mrf = MarkovRandomField()
    
    # Add variables
    for i in range(n_rows):
        for j in range(n_cols):
            mrf.add_variable(f"X_{i}_{j}", 2)
    
    # Add unary factors (external field)
    for i in range(n_rows):
        for j in range(n_cols):
            var = f"X_{i}_{j}"
            # ψ(X) = exp(h·X) for X ∈ {0, 1}
            # Map {0,1} to {-1,+1}: spin = 2X - 1
            values = torch.tensor([
                np.exp(-field),  # X=0 → spin=-1
                np.exp(field)    # X=1 → spin=+1
            ])
            mrf.add_factor([var], values)
    
    # Add pairwise factors (coupling)
    for i in range(n_rows):
        for j in range(n_cols):
            var1 = f"X_{i}_{j}"
            
            # Horizontal neighbor
            if j < n_cols - 1:
                var2 = f"X_{i}_{j+1}"
                # ψ(X₁, X₂) = exp(J · (2X₁-1)(2X₂-1))
                values = torch.tensor([
                    [np.exp(coupling), np.exp(-coupling)],   # X₁=0
                    [np.exp(-coupling), np.exp(coupling)]    # X₁=1
                ])
                mrf.add_factor([var1, var2], values)
            
            # Vertical neighbor
            if i < n_rows - 1:
                var2 = f"X_{i+1}_{j}"
                values = torch.tensor([
                    [np.exp(coupling), np.exp(-coupling)],
                    [np.exp(-coupling), np.exp(coupling)]
                ])
                mrf.add_factor([var1, var2], values)
    
    return mrf


def demonstrate_ising_model():
    """Demonstrate the Ising model."""
    print("=" * 60)
    print("Ising Model Demonstration")
    print("=" * 60)
    
    # Create small Ising model (3x3 grid = 9 variables)
    mrf = create_ising_model(2, 2, field=0.0, coupling=1.0)
    
    print(f"\n2x2 Ising Model")
    print(f"Variables: {len(mrf.variables)}")
    print(f"Factors: {len(mrf.factors)}")
    
    # Compute partition function (only feasible for small models)
    Z = mrf.compute_partition_function()
    print(f"\nPartition function Z = {Z:.4f}")
    
    # Show some probabilities
    print("\nSome configurations:")
    configs = [
        {'X_0_0': 0, 'X_0_1': 0, 'X_1_0': 0, 'X_1_1': 0},  # All -1
        {'X_0_0': 1, 'X_0_1': 1, 'X_1_0': 1, 'X_1_1': 1},  # All +1
        {'X_0_0': 0, 'X_0_1': 1, 'X_1_0': 1, 'X_1_1': 0},  # Alternating
    ]
    
    for config in configs:
        prob = mrf.probability(config)
        spins = [2*v-1 for v in config.values()]
        print(f"  {spins}: P = {prob:.4f}")
    
    print("\nNote: Same-spin configurations are more probable (ferromagnetic coupling)")

demonstrate_ising_model()
```

## Conditional Random Fields (CRFs)

**Conditional Random Fields** are discriminative MRFs that model $P(Y \mid X)$ directly:

$$P(Y \mid X) = \frac{1}{Z(X)} \prod_C \psi_C(Y_C, X)$$

### Linear-Chain CRF

For sequence labeling (e.g., NER, POS tagging):

$$P(Y \mid X) = \frac{1}{Z(X)} \prod_{t=1}^{T} \psi(y_{t-1}, y_t, X, t)$$

Features:
- Transition features: $f(y_{t-1}, y_t)$
- Emission features: $g(y_t, X, t)$

## Inference in MRFs

### Exact Inference

- **Variable elimination**: Same as for Bayesian networks
- **Junction tree**: Works for both directed and undirected models

### Approximate Inference

Due to the partition function problem, approximate methods are essential:

1. **Loopy Belief Propagation**: Message passing on graphs with cycles
2. **MCMC (Gibbs Sampling)**: Sample from the distribution
3. **Variational Methods**: Approximate with tractable distribution

## Converting Between Representations

### Bayesian Network → MRF (Moralization)

1. Add edges between all parents of each node (marry the parents)
2. Drop edge directions
3. Each CPT becomes a factor

### MRF → Bayesian Network

Not always possible! MRFs can represent some distributions that BNs cannot (and vice versa).

```python
def moralize_bayesian_network(bn) -> MarkovRandomField:
    """
    Convert a Bayesian Network to a Markov Random Field.
    
    1. Marry parents (add edges between co-parents)
    2. Drop directions
    3. Convert CPTs to factors
    """
    mrf = MarkovRandomField()
    
    # Add variables
    for var in bn.variables:
        mrf.add_variable(var, bn.cardinalities[var])
    
    # Convert each CPT to a factor
    for var in bn.variables:
        cpt = bn.cpts[var]
        variables = cpt.parents + [cpt.variable]
        mrf.add_factor(variables, cpt.values)
    
    return mrf
```

## Summary

| Concept | Description |
|---------|-------------|
| **MRF** | Undirected graphical model |
| **Potential** | Non-negative function over clique variables |
| **Gibbs Distribution** | $P(X) = (1/Z) \prod_C \psi_C(X_C)$ |
| **Partition Function** | Normalizing constant $Z$ |
| **Hammersley-Clifford** | Factorization ↔ Markov properties |
| **CRF** | Discriminative MRF: $P(Y\mid X)$ |

## Key Takeaways

1. **Symmetric relationships**: MRFs model undirected dependencies naturally
2. **Partition function**: The main computational challenge
3. **Log-linear models**: Convenient parameterization with features
4. **CRFs**: Powerful for structured prediction (sequences, grids)
5. **Inference**: Same algorithms as BNs, but partition function adds complexity
