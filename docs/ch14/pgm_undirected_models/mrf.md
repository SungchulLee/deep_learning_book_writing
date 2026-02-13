# Markov Random Fields

## Overview

**Markov Random Fields (MRFs)**, also called undirected graphical models, use undirected graphs to represent probability distributions. Unlike Bayesian networks, MRFs model symmetric relationships without directional semantics, making them natural for spatial data, pairwise constraints, and energy-based formulations.

## Key Differences from Bayesian Networks

| Aspect | Bayesian Network | Markov Random Field |
|--------|------------------|---------------------|
| Graph type | Directed (DAG) | Undirected |
| Edge meaning | Causal / generative | Symmetric constraint |
| Factorization | CPTs: $P(X_i \mid \text{Pa})$ | Potentials: $\psi_C(X_C)$ |
| Partition function | Not needed (normalized by construction) | Required: $Z = \sum_x \prod_C \psi_C$ |
| Independence test | D-separation | Graph separation |

## The Gibbs Distribution

MRFs define distributions using the **Gibbs distribution**:

$$P(X) = \frac{1}{Z} \prod_{C \in \mathcal{C}} \psi_C(X_C)$$

where $\mathcal{C}$ is the set of cliques in the graph, $\psi_C(X_C)$ is a **potential function** (non-negative) over clique $C$, and $Z = \sum_x \prod_{C \in \mathcal{C}} \psi_C(x_C)$ is the **partition function** ensuring normalization.

### The Partition Function Problem

Unlike Bayesian networks, MRFs require computing $Z$ to normalize the distribution. This sum over all configurations is exponential and often intractable:

$$Z = \sum_{x_1} \sum_{x_2} \cdots \sum_{x_n} \prod_{C \in \mathcal{C}} \psi_C(x_C)$$

Computing $Z$ is a fundamental computational challenge. For many models, exact computation is \#P-hard, motivating the approximate inference methods covered in Section 17.4.

## Potential Functions

Potentials are **not probabilities**—they can take any non-negative value. Common parametric forms include:

### Log-Linear (Exponential Family)

$$\psi_C(X_C) = \exp\!\left(\sum_k \theta_k f_k(X_C)\right)$$

where $f_k$ are **feature functions** and $\theta_k$ are learnable parameters. This form is particularly convenient because the log-probability becomes a linear function of features, enabling efficient gradient-based optimization.

### Pairwise Potentials

For edges $(i, j)$:

$$\psi_{ij}(X_i, X_j) = \exp(\theta_{ij}^{X_i, X_j})$$

Often represented as a matrix of compatibility scores. Positive entries for $(X_i, X_j)$ encourage those configurations; near-zero entries discourage them.

## PyTorch Implementation

```python
import torch
import numpy as np
from typing import Dict, List, Set, Optional
from itertools import product as cartesian_product


class UndirectedFactor:
    """
    A factor/potential function for undirected graphical models.
    Unlike CPTs, factors need not sum to 1.
    """
    
    def __init__(self, variables: List[str],
                 cardinalities: Dict[str, int],
                 values: Optional[torch.Tensor] = None):
        self.variables = variables
        self.cardinalities = cardinalities
        self.shape = tuple(cardinalities[v] for v in variables)
        
        if values is None:
            self.values = torch.ones(self.shape)
        else:
            self.values = values.float()
            assert (self.values >= 0).all(), "Potentials must be non-negative"
    
    def evaluate(self, assignment: Dict[str, int]) -> float:
        index = tuple(assignment[v] for v in self.variables)
        return self.values[index].item()


class MarkovRandomField:
    """
    Markov Random Field (Undirected Graphical Model).
    
    Defines: P(X) = (1/Z) prod_C psi_C(X_C)
    """
    
    def __init__(self):
        self.variables: List[str] = []
        self.cardinalities: Dict[str, int] = {}
        self.factors: List[UndirectedFactor] = []
        self._partition_function: Optional[float] = None
    
    def add_variable(self, name: str, cardinality: int):
        self.variables.append(name)
        self.cardinalities[name] = cardinality
        self._partition_function = None
    
    def add_factor(self, variables: List[str], values: torch.Tensor):
        factor = UndirectedFactor(variables, self.cardinalities, values)
        self.factors.append(factor)
        self._partition_function = None
    
    def unnormalized_probability(self, assignment: Dict[str, int]) -> float:
        """Compute prod_C psi_C(X_C)."""
        prob = 1.0
        for factor in self.factors:
            prob *= factor.evaluate(assignment)
        return prob
    
    def compute_partition_function(self) -> float:
        """Compute Z = sum_x prod_C psi_C(x_C). Exponential complexity!"""
        if self._partition_function is not None:
            return self._partition_function
        
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
        
        return marginal / marginal.sum()
```

## The Hammersley-Clifford Theorem

This fundamental theorem connects graph structure to factorization:

**Theorem (Hammersley-Clifford)**: A strictly positive distribution $P(X) > 0$ satisfies the **local Markov property** with respect to an undirected graph $G$ if and only if $P$ factorizes according to the cliques of $G$:

$$P(X) = \frac{1}{Z} \prod_{C \in \mathcal{C}} \psi_C(X_C)$$

### Markov Properties

MRFs satisfy three equivalent Markov properties (under positivity):

1. **Pairwise**: Non-adjacent nodes are conditionally independent given all others.
2. **Local**: A node is independent of all other nodes given its neighbors.
3. **Global**: Two sets of nodes are independent given any separating set.

The local Markov property states that the **Markov blanket** of a node in an MRF is simply its set of neighbors:

$$X_i \perp\!\!\!\perp X_{V \setminus \{i\} \setminus N_i} \mid X_{N_i}$$

This is simpler than the Bayesian network case, where the Markov blanket also includes parents of children.

## Classic MRF Examples

### The Ising Model

Binary variables on a grid with pairwise interactions:

$$P(X) = \frac{1}{Z} \exp\!\left(\sum_{i} h_i X_i + \sum_{(i,j) \in E} J_{ij} X_i X_j\right)$$

where $h_i$ is the external field and $J_{ij}$ is the coupling strength. Positive $J$ (ferromagnetic) favors aligned spins; negative $J$ (antiferromagnetic) favors alternating spins.

```python
def create_ising_model(n_rows: int, n_cols: int,
                       field: float = 0.0,
                       coupling: float = 1.0) -> MarkovRandomField:
    """
    Create an Ising model on a grid.
    
    Energy: E(X) = -h sum_i X_i - J sum_{i~j} X_i X_j
    P(X) proportional to exp(-E(X))
    """
    mrf = MarkovRandomField()
    
    # Add variables (binary spins)
    for i in range(n_rows):
        for j in range(n_cols):
            mrf.add_variable(f"X_{i}_{j}", 2)
    
    # Unary factors (external field)
    for i in range(n_rows):
        for j in range(n_cols):
            var = f"X_{i}_{j}"
            values = torch.tensor([
                np.exp(-field),   # X=0 -> spin=-1
                np.exp(field)     # X=1 -> spin=+1
            ])
            mrf.add_factor([var], values)
    
    # Pairwise factors (coupling)
    for i in range(n_rows):
        for j in range(n_cols):
            var1 = f"X_{i}_{j}"
            
            # Horizontal neighbor
            if j < n_cols - 1:
                var2 = f"X_{i}_{j+1}"
                values = torch.tensor([
                    [np.exp(coupling), np.exp(-coupling)],
                    [np.exp(-coupling), np.exp(coupling)]
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
    """Demonstrate the Ising model on a small grid."""
    mrf = create_ising_model(2, 2, field=0.0, coupling=1.0)
    
    print(f"2x2 Ising Model: {len(mrf.variables)} variables, {len(mrf.factors)} factors")
    Z = mrf.compute_partition_function()
    print(f"Partition function Z = {Z:.4f}")
    
    configs = [
        {'X_0_0': 0, 'X_0_1': 0, 'X_1_0': 0, 'X_1_1': 0},   # All -1
        {'X_0_0': 1, 'X_0_1': 1, 'X_1_0': 1, 'X_1_1': 1},   # All +1
        {'X_0_0': 0, 'X_0_1': 1, 'X_1_0': 1, 'X_1_1': 0},   # Alternating
    ]
    
    for config in configs:
        prob = mrf.probability(config)
        spins = [2 * v - 1 for v in config.values()]
        print(f"  Spins {spins}: P = {prob:.4f}")
    
    print("Same-spin configurations are more probable (ferromagnetic coupling).")


demonstrate_ising_model()
```

### The Potts Model

Generalization of the Ising model to $k$ states:

$$P(X) = \frac{1}{Z} \exp\!\left(\sum_{(i,j) \in E} \beta \cdot \mathbf{1}[X_i = X_j]\right)$$

The Potts model is widely used in image segmentation (neighboring pixels tend to share labels) and community detection (connected nodes tend to belong to the same community).

## Conditional Random Fields (CRFs)

**Conditional Random Fields** are discriminative MRFs that model $P(Y \mid X)$ directly, avoiding the need to model the input distribution $P(X)$:

$$P(Y \mid X) = \frac{1}{Z(X)} \prod_C \psi_C(Y_C, X)$$

### Linear-Chain CRF

For sequence labeling tasks (NER, POS tagging):

$$P(Y \mid X) = \frac{1}{Z(X)} \prod_{t=1}^{T} \psi(y_{t-1}, y_t, X, t)$$

with transition features $f(y_{t-1}, y_t)$ and emission features $g(y_t, X, t)$. CRFs are the discriminative counterpart of Hidden Markov Models and generally achieve better accuracy for structured prediction tasks because they can incorporate arbitrary overlapping features of the input.

## Converting Between Representations

### Bayesian Network → MRF (Moralization)

The moralization procedure converts a directed model to undirected:

1. **Marry parents**: Add edges between all parents of each node
2. **Drop directions**: Replace all directed edges with undirected edges
3. **Convert CPTs to factors**: Each CPT becomes a potential function

```python
def moralize_bayesian_network(bn) -> MarkovRandomField:
    """Convert a Bayesian Network to a Markov Random Field."""
    mrf = MarkovRandomField()
    
    for var in bn.variables:
        mrf.add_variable(var, bn.cardinalities[var])
    
    for var in bn.variables:
        cpt = bn.cpts[var]
        variables = cpt.parents + [cpt.variable]
        mrf.add_factor(variables, cpt.values)
    
    return mrf
```

Moralization can lose independence information: if $A \to C \leftarrow B$ and $A, B$ are not connected, moralization adds the edge $A - B$, destroying the marginal independence $A \perp\!\!\!\perp B$. This is the fundamental reason why the directed and undirected families have different expressive power.

### MRF → Bayesian Network

Not always possible without adding edges. Some distributions expressible as MRFs cannot be captured by any DAG (and vice versa). The intersection of both families corresponds to distributions that decompose according to **decomposable** (chordal) graphs.

## Summary

| Concept | Description |
|---------|-------------|
| **MRF** | Undirected graphical model with potential functions |
| **Potential** | Non-negative function $\psi_C(X_C)$ over clique variables |
| **Gibbs Distribution** | $P(X) = (1/Z) \prod_C \psi_C(X_C)$ |
| **Partition Function** | Normalizing constant $Z$, often intractable |
| **Hammersley-Clifford** | Factorization $\Leftrightarrow$ Markov property (under positivity) |
| **Ising Model** | Binary pairwise MRF on a grid |
| **CRF** | Discriminative MRF: $P(Y \mid X)$ |
| **Moralization** | Converting directed model to undirected |

## Quantitative Finance Application

MRFs are well-suited for modeling spatial and network dependencies in financial systems. A pairwise MRF over a network of banks can model systemic risk, where potential functions encode the tendency for connected institutions to experience correlated stress:

$$P(\text{Stress}_1, \ldots, \text{Stress}_n) = \frac{1}{Z} \prod_{(i,j) \in E} \exp\!\left(\beta_{ij} \cdot \mathbf{1}[\text{Stress}_i = \text{Stress}_j]\right)$$

The coupling parameters $\beta_{ij}$ capture contagion intensity between institutions, and the partition function (estimated via approximate inference) provides a measure of overall systemic fragility. The Ising model specifically has been applied to model market sentiment, where each trader's bullish/bearish position interacts with neighbors in an information network.
