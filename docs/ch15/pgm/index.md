# Probabilistic Graphical Models

## Overview

Probabilistic Graphical Models (PGMs) provide a powerful framework for representing and reasoning about complex probability distributions over many random variables. They combine graph theory with probability theory to create compact, interpretable representations of joint distributions that would otherwise be intractable to specify or compute.

## Why Probabilistic Graphical Models?

Consider a joint distribution over $n$ binary random variables. Naively, we would need $2^n - 1$ independent parameters to fully specify this distribution—an exponential explosion that quickly becomes infeasible. For just 30 variables, this exceeds one billion parameters!

PGMs exploit **conditional independence** relationships to achieve dramatic compression:

$$P(X_1, X_2, \ldots, X_n) = \prod_{i=1}^{n} P(X_i \mid \text{Parents}(X_i))$$

A Bayesian network over 30 variables, each with at most 3 parents, requires only $30 \times 2^3 = 240$ parameters—a reduction by a factor of over 4 million.

## The Two Families of Graphical Models

### Directed Graphical Models (Bayesian Networks)

Bayesian Networks use **directed acyclic graphs (DAGs)** where:

- Nodes represent random variables
- Directed edges represent direct probabilistic dependencies
- Each node stores $P(\text{Node} \mid \text{Parents})$

**Key characteristics:**

- Natural for modeling causal relationships
- Factorization follows ancestral ordering
- Conditional independencies via d-separation

### Undirected Graphical Models (Markov Random Fields)

Markov Random Fields use **undirected graphs** where:

- Nodes represent random variables
- Edges represent symmetric dependencies
- Factors (potential functions) defined over cliques

**Key characteristics:**

- Natural for symmetric relationships (spatial, pairwise constraints)
- Factorization via clique potentials
- Conditional independencies via graph separation

## Mathematical Framework

### Joint Distribution Factorization

**Bayesian Networks:**
$$P(X_1, \ldots, X_n) = \prod_{i=1}^{n} P(X_i \mid \text{Pa}(X_i))$$

**Markov Random Fields:**
$$P(X_1, \ldots, X_n) = \frac{1}{Z} \prod_{C \in \mathcal{C}} \psi_C(X_C)$$

where $Z = \sum_x \prod_{C \in \mathcal{C}} \psi_C(x_C)$ is the partition function.

### Conditional Independence

The graph structure encodes conditional independence relationships:

| Graph Type | Independence Criterion |
|------------|----------------------|
| Bayesian Network | D-separation |
| Markov Random Field | Graph separation |

If $X$ and $Y$ are separated by $Z$ in the appropriate sense, then:
$$X \perp\!\!\!\perp Y \mid Z$$

## Learning Objectives

By completing this chapter, you will:

1. **Master PGM fundamentals**: Understand graphical representations, independence, and factorization
2. **Build Bayesian Networks**: Construct directed models with CPTs and perform exact inference
3. **Work with Markov Random Fields**: Understand potential functions and the Hammersley-Clifford theorem
4. **Implement inference algorithms**: Variable elimination, belief propagation, sampling methods
5. **Learn parameters and structure**: MLE, Bayesian estimation, and structure discovery
6. **Apply to real problems**: Medical diagnosis, NLP, computer vision, finance

## Chapter Structure

### 15.2.1 Fundamentals
- Introduction to PGMs
- Independence and conditional independence
- D-separation
- Factorization theorems

### 15.2.2 Bayesian Networks
- DAG structure and semantics
- Conditional probability tables
- Building and querying networks
- The alarm network and other classics

### 15.2.3 Markov Random Fields
- Undirected models and potentials
- Gibbs distribution
- Hammersley-Clifford theorem
- Ising and Potts models

### 15.2.4 Factor Graphs
- Unified representation
- Message passing framework
- Conversion between representations

### 15.2.5 Exact Inference
- Inference by enumeration
- Variable elimination algorithm
- Junction tree algorithm

### 15.2.6 Approximate Inference
- Loopy belief propagation
- Sampling methods (rejection, importance, MCMC)
- Variational inference basics

### 15.2.7 Parameter Learning
- Maximum likelihood estimation
- Bayesian parameter learning
- EM algorithm for missing data

### 15.2.8 Structure Learning
- Constraint-based methods (PC algorithm)
- Score-based methods (BIC, hill climbing)
- Causal discovery

### 15.2.9 Applications
- Medical diagnosis systems
- Natural language processing
- Computer vision
- Financial modeling

## Prerequisites

- **Probability theory**: Joint, marginal, conditional distributions; Bayes' theorem
- **Linear algebra**: Matrix operations, eigendecomposition
- **Basic graph theory**: Nodes, edges, paths, cycles
- **Python and PyTorch**: Implementation skills

## Key Connections

PGMs form the foundation for many modern deep learning techniques:

| PGM Concept | Deep Learning Connection |
|-------------|-------------------------|
| Latent variable models | VAEs, deep generative models |
| Message passing | Graph neural networks |
| Variational inference | VAE training, amortized inference |
| Energy-based models | EBMs, contrastive learning |
| Sequential models | RNNs, transformers |

## PyTorch Implementation Focus

Throughout this chapter, we implement PGMs from scratch in PyTorch, emphasizing:

- Efficient tensor operations for factor manipulation
- Automatic differentiation for parameter learning
- GPU acceleration for large-scale inference
- Integration with neural network components

```python
import torch
import torch.nn as nn
from typing import Dict, List, Optional

class Factor:
    """A factor over discrete random variables."""
    
    def __init__(self, variables: List[str], values: torch.Tensor):
        self.variables = variables
        self.values = values
    
    def marginalize(self, variables_to_sum: List[str]) -> 'Factor':
        """Sum out specified variables."""
        axes = [self.variables.index(v) for v in variables_to_sum]
        new_values = self.values.sum(dim=axes)
        new_vars = [v for v in self.variables if v not in variables_to_sum]
        return Factor(new_vars, new_values)
```

## References

### Textbooks
1. Koller & Friedman - *Probabilistic Graphical Models: Principles and Techniques*
2. Bishop - *Pattern Recognition and Machine Learning*, Chapter 8
3. Murphy - *Machine Learning: A Probabilistic Perspective*, Chapters 10, 19-22

### Foundational Papers
1. Pearl (1988) - *Probabilistic Reasoning in Intelligent Systems*
2. Lauritzen & Spiegelhalter (1988) - *Local Computations with Probabilities*
3. Jordan et al. (1999) - *An Introduction to Variational Methods*
