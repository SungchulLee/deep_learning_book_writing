# Markov Chains

## Overview

Markov chains are fundamental stochastic models that describe systems transitioning between discrete states according to probabilistic rules. Named after the Russian mathematician Andrey Markov, these models are characterized by the **Markov property**: the future state depends only on the current state, not on the sequence of events that preceded it.

This property—often called "memorylessness"—makes Markov chains both mathematically tractable and widely applicable across diverse fields including finance, physics, biology, computer science, and operations research.

## Mathematical Foundation

### The Markov Property

A stochastic process $\{X_n\}_{n \geq 0}$ is a **Markov chain** if for all states $i, j$ and all $n \geq 0$:

$$P(X_{n+1} = j \mid X_n = i, X_{n-1} = i_{n-1}, \ldots, X_0 = i_0) = P(X_{n+1} = j \mid X_n = i)$$

This equation states that given the present state $X_n = i$, the probability of transitioning to state $j$ is independent of all past states.

### Transition Matrix

The dynamics of a Markov chain are fully characterized by its **transition matrix** $P$, where:

$$P_{ij} = P(X_{n+1} = j \mid X_n = i)$$

The transition matrix must satisfy:

1. **Non-negativity**: $P_{ij} \geq 0$ for all $i, j$
2. **Row-stochastic**: $\sum_{j} P_{ij} = 1$ for all $i$

### Key Concepts

| Concept | Definition |
|---------|------------|
| **State Space** | The set $S$ of all possible states |
| **Transition Probability** | $P_{ij}$ = probability of moving from $i$ to $j$ |
| **n-Step Transition** | $P^{(n)}_{ij} = P(X_{n+m} = j \mid X_m = i)$ |
| **Stationary Distribution** | Distribution $\pi$ satisfying $\pi = \pi P$ |
| **Irreducibility** | All states communicate with each other |
| **Aperiodicity** | $\gcd\{n : P^{(n)}_{ii} > 0\} = 1$ |
| **Ergodicity** | Irreducible and aperiodic |

## Chapter Structure

This chapter covers Markov chains through the following sections:

### Fundamentals
- **Markov Property and Basic Definitions**: Core concepts and the memoryless property
- **Transition Matrices**: Matrix representations and operations
- **State Classification**: Communicating classes, periodicity, and recurrence
- **Random Walks**: Special case of Markov chains with important applications

### Analysis Methods
- **Stationary Distributions**: Computing long-run behavior via multiple methods
- **Absorbing Chains**: Analysis of chains with absorbing states
- **Convergence Theorems**: Ergodic theory and mixing times

### Applications
- **Weather Modeling**: Estimation from data and prediction
- **PageRank Algorithm**: Google's web page ranking system
- **Text Generation**: N-gram models for language processing
- **Continuous-Time Markov Chains**: Extension to continuous time

### Finance Applications
- **Credit Rating Transitions**: Modeling rating migrations
- **Regime-Switching Models**: Market state dynamics
- **Option Pricing**: Lattice methods for derivatives

## Learning Objectives

Upon completing this chapter, you will be able to:

1. **Understand** the Markov property and its implications
2. **Construct** transition matrices from real-world data
3. **Compute** stationary distributions using multiple methods
4. **Analyze** chain properties (irreducibility, aperiodicity, ergodicity)
5. **Apply** Markov chains to practical problems in finance and beyond
6. **Implement** Markov chain algorithms in PyTorch

## Prerequisites

- Linear algebra (matrices, eigenvalues, eigenvectors)
- Probability theory (conditional probability, distributions)
- Basic Python/PyTorch programming

## Connections to Deep Learning

Markov chains serve as foundational building blocks for several advanced topics in this curriculum:

| Topic | Connection |
|-------|------------|
| **MCMC Methods** | Metropolis-Hastings, Gibbs Sampling use Markov chains for sampling |
| **Hidden Markov Models** | Latent state sequences modeled as Markov chains |
| **Reinforcement Learning** | MDPs extend Markov chains with actions and rewards |
| **Diffusion Models** | Forward/reverse diffusion processes are continuous-time Markov chains |
| **Normalizing Flows** | Discrete flows can be viewed through a Markov lens |

## References

1. Lawler, G.F. *Introduction to Stochastic Processes* (2nd ed.). Chapman & Hall/CRC, 2006.
2. Levin, D.A., Peres, Y., & Wilmer, E.L. *Markov Chains and Mixing Times* (2nd ed.). AMS, 2017.
3. Grimmett, G. & Stirzaker, D. *Probability and Random Processes* (4th ed.). Oxford University Press, 2020.
4. Kemeny, J.G. & Snell, J.L. *Finite Markov Chains*. Springer-Verlag, 1976.
5. Norris, J.R. *Markov Chains*. Cambridge University Press, 1997.
