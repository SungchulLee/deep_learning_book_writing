# Ising Model

## Overview

The Ising model is the canonical undirected graphical model, originally from statistical physics. It models binary variables on a lattice with pairwise interactions and serves as the foundation for understanding MRFs.

## Definition

For binary variables $x_i \in \{-1, +1\}$ on a graph $G = (V, E)$:

$$P(x) = \frac{1}{Z} \exp\left(\sum_{(i,j) \in E} J_{ij} x_i x_j + \sum_i h_i x_i\right)$$

where $J_{ij}$ are coupling strengths, $h_i$ are external fields, and $Z$ is the partition function.

## Parameters

- **Ferromagnetic** ($J_{ij} > 0$): neighboring variables prefer the same state
- **Antiferromagnetic** ($J_{ij} < 0$): neighboring variables prefer opposite states
- **External field** ($h_i$): biases individual variables toward $+1$ or $-1$

## Phase Transitions

The Ising model exhibits a phase transition at a critical temperature $T_c$:

- $T > T_c$: disordered phase, variables are approximately independent
- $T < T_c$: ordered phase, variables align (spontaneous magnetization)

This phase transition is foundational to understanding computational hardness: inference and sampling are easy in the high-temperature regime but become hard near the critical temperature.

## Connection to PGMs

The Ising model is a pairwise MRF with log-linear potentials. Every pairwise MRF with binary variables can be written as an Ising model. The Boltzmann machine (a key model in deep learning history) is a generalized Ising model with hidden variables and learned couplings.

## Inference

Exact inference is tractable on trees (belief propagation) but NP-hard on general graphs. Approximate methods include mean-field variational inference, loopy belief propagation, and MCMC (Gibbs sampling).
