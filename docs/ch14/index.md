# Chapter 14: Probabilistic Graphical Models

This chapter provides a comprehensive treatment of probabilistic graphical models (PGMs), which use graph structures to compactly represent high-dimensional joint distributions. We cover directed models (Bayesian networks), undirected models (Markov random fields), exact and approximate inference algorithms, and methods for learning both parameters and structure from data.

---

## PGM Foundations

- PGM Overview -- Complete educational overview of graphical models from foundational concepts to advanced applications
- [Fundamentals](pgm_foundations/fundamentals.md) -- The curse of dimensionality in probability and how conditional independence enables tractable factorizations
- Conditional Independence -- Formal definition, relation to graphical models, factorization, and statistical testing methods
- D-Separation -- Graphical criterion for determining conditional independence in Bayesian networks without computing probabilities
- Plate Notation -- Compact graphical representation for models with repeated structure, including nested and crossed plates

## Directed Models

- Bayesian Networks -- Directed acyclic graphs representing joint distributions via conditional probability tables
- Naive Bayes as a PGM -- The naive conditional independence assumption, parameter savings, and why it works despite the assumption
- Dynamic Bayesian Networks -- Temporal extensions of Bayesian networks including the relationship to HMMs and Kalman filters
- Causal Inference -- Distinguishing correlation from causation through do-calculus, interventions, and the adjustment formula

## Undirected Models

- Markov Random Fields -- Undirected graphical models using potential functions and the Gibbs distribution for symmetric relationships
- Factor Graphs -- Unified bipartite graph representation subsuming both Bayesian networks and MRFs
- Ising Model -- The canonical undirected model from statistical physics with pairwise interactions on a lattice
- Conditional Random Fields -- Discriminative undirected models for sequence labeling that model conditional distributions directly

## Inference

- Variable Elimination -- Exploiting factored structure by pushing sums inside products to eliminate variables one at a time
- Belief Propagation -- Message passing algorithm for computing exact marginals on tree-structured graphical models
- Loopy Belief Propagation -- Applying BP message passing to graphs with cycles as an approximate inference method
- Junction Tree Algorithm -- Exact inference on arbitrary graphs by transforming the original graph into a tree of clusters

## Learning

- Parameter Learning -- MLE and Bayesian estimation of conditional distributions given a fixed graph structure
- Structure Learning -- Discovering graph structure from data through score-based search and constraint-based testing
- [EM for PGMs](pgm_learning/em_for_pgms.md) -- Expectation-Maximization for learning parameters of graphical models with latent variables
- Score-Based vs Constraint-Based Learning -- Comparing the two main paradigms for structure discovery with their respective scoring functions and tests
