# Chapter 17: Probabilistic Graphical Models

## Overview

Probabilistic Graphical Models (PGMs) provide a rigorous framework for representing and reasoning about complex probability distributions over many random variables. They combine graph theory with probability theory to create compact, interpretable representations of joint distributions that would otherwise be intractable to specify or compute.

PGMs occupy a unique position in the deep learning landscape: they predate neural networks' modern resurgence yet remain foundational to many state-of-the-art architectures. Variational autoencoders, graph neural networks, energy-based models, and diffusion models all trace their theoretical roots to PGMs.

## Motivation: The Curse of Dimensionality in Probability

Consider a joint distribution over $n$ discrete random variables, each taking $k$ values. A full specification requires $k^n - 1$ independent parameters—an exponential explosion that quickly becomes infeasible. For just 20 binary variables, this exceeds one million parameters; for 50, it exceeds $10^{15}$.

PGMs exploit **conditional independence** to achieve dramatic compression. A Bayesian network over 30 variables, each with at most 3 parents, requires only $30 \times 2^3 = 240$ parameters—a reduction by a factor of over four million compared to the naive $2^{30} - 1 \approx 10^9$.

## The Two Families of Graphical Models

### Directed Models (Bayesian Networks)

Bayesian Networks use **directed acyclic graphs (DAGs)** where nodes represent random variables and directed edges encode direct probabilistic dependencies. Each node stores a conditional probability distribution $P(X_i \mid \text{Parents}(X_i))$, and the joint distribution factors as:

$$P(X_1, \ldots, X_n) = \prod_{i=1}^{n} P(X_i \mid \text{Pa}(X_i))$$

Bayesian networks are natural for modeling causal or generative processes, and conditional independencies are read from the graph via d-separation.

### Undirected Models (Markov Random Fields)

Markov Random Fields use **undirected graphs** where edges represent symmetric dependencies. Factors (potential functions) are defined over cliques and the joint distribution takes the form:

$$P(X_1, \ldots, X_n) = \frac{1}{Z} \prod_{C \in \mathcal{C}} \psi_C(X_C)$$

where $Z = \sum_x \prod_{C \in \mathcal{C}} \psi_C(x_C)$ is the partition function. Undirected models are natural for symmetric relationships such as spatial correlations, pairwise constraints, and energy-based formulations.

### Comparison

| Aspect | Bayesian Network | Markov Random Field |
|--------|------------------|---------------------|
| Graph type | Directed (DAG) | Undirected |
| Edge semantics | Causal / generative | Symmetric constraint |
| Factorization | CPTs: $P(X_i \mid \text{Pa})$ | Potentials: $\psi_C(X_C)$ |
| Normalization | Automatic (product of conditional distributions) | Requires partition function $Z$ |
| Independence criterion | D-separation | Graph separation |

## Conditional Independence

The graph structure encodes conditional independence relationships. If $X$ and $Y$ are separated by $Z$ in the graph-appropriate sense, then:

$$X \perp\!\!\!\perp Y \mid Z$$

This is the core principle that allows PGMs to compactly represent high-dimensional distributions and enables efficient inference algorithms.

## Chapter Roadmap

### 17.1 Foundations

We begin with the mathematical foundations of PGMs, covering conditional independence, the three fundamental graph structures (chain, fork, collider), and the factorization theorems that connect graph structure to probability distributions. We then develop d-separation—the graphical criterion for reading conditional independencies from directed graphs—along with the Markov blanket concept.

### 17.2 Directed Models

This section develops Bayesian networks in depth: DAG structure and semantics, conditional probability tables (CPTs), classic networks (alarm, weather, student), parameter counting, forward sampling, and inference by enumeration.

### 17.3 Undirected Models

We cover Markov Random Fields with their potential functions, the Gibbs distribution, the Hammersley-Clifford theorem, and classic models (Ising, Potts). We also introduce Conditional Random Fields (CRFs) for discriminative structured prediction, and factor graphs as a unified representation bridging directed and undirected models.

### 17.4 Inference

The inference section progresses from exact to approximate methods. Variable elimination introduces the key idea of pushing sums inside products. Belief propagation formalizes message passing on trees. The junction tree algorithm extends exact inference to arbitrary graphs by clustering nodes. Loopy belief propagation addresses approximate inference on graphs with cycles.

### 17.5 Learning

We cover both parameter learning (MLE, Bayesian estimation, EM for missing data) and structure learning (the PC constraint-based algorithm, score-based hill climbing with BIC, and the challenges of Markov equivalence and combinatorial explosion).

## Connections to Deep Learning

PGMs form the theoretical foundation for many modern deep learning techniques:

| PGM Concept | Deep Learning Connection |
|-------------|-------------------------|
| Latent variable models | VAEs, deep generative models |
| Message passing | Graph neural networks |
| Variational inference | VAE training, amortized inference |
| Energy-based models | EBMs, contrastive learning |
| Structured prediction (CRFs) | Sequence labeling, NER |
| Factor graphs | Belief propagation neural networks |

## Application to Quantitative Finance

PGMs have natural applications in quantitative finance:

- **Credit risk modeling**: Bayesian networks for dependent default events, where industry sectors, macroeconomic factors, and firm-specific variables form a causal hierarchy
- **Market regime detection**: Hidden Markov models (a special case of dynamic Bayesian networks) for identifying bull/bear regimes from price data
- **Portfolio risk**: Markov Random Fields for modeling spatial correlations in multi-asset portfolios, where pairwise potentials capture sector-level dependencies
- **Causal discovery**: Structure learning algorithms applied to financial time series to identify lead-lag relationships and common drivers
- **Stress testing**: Bayesian networks encoding regulatory scenarios, propagating shocks through conditional probability tables to estimate portfolio-level impact

## Prerequisites

- **Probability theory**: Joint, marginal, and conditional distributions; Bayes' theorem; independence
- **Basic graph theory**: Nodes, edges, paths, cycles, directed acyclic graphs
- **Python and PyTorch**: Implementation skills for tensor operations and optimization

## References

### Textbooks

1. Koller & Friedman — *Probabilistic Graphical Models: Principles and Techniques* (2009)
2. Bishop — *Pattern Recognition and Machine Learning*, Chapter 8 (2006)
3. Murphy — *Machine Learning: A Probabilistic Perspective*, Chapters 10, 19–22 (2012)

### Foundational Papers

1. Pearl (1988) — *Probabilistic Reasoning in Intelligent Systems*
2. Lauritzen & Spiegelhalter (1988) — "Local Computations with Probabilities on Graphical Structures"
3. Jordan et al. (1999) — "An Introduction to Variational Methods for Graphical Models"
