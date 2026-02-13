# Plate Notation

## Overview

Compact graphical representation for PGMs with repeated structure. A plate (rectangle) indicates enclosed variables are replicated $N$ times, avoiding drawing hundreds of identical nodes.

## Shared Parameters

Variables outside the plate connected to inside variables are shared across all replications. Parameters $\theta$ outside a plate of $N$ observations encode the assumption of i.i.d. data.

## Nested Plates

Represent hierarchical structure: outer plate for groups ($J$), inner plate for observations within groups ($N_j$). Example: hierarchical Bayesian model with group-specific parameters drawn from a shared prior.

## Common Patterns

i.i.d. data: single plate with shared parameters. Mixture models: plate with latent discrete variable. Time series: plate with chain structure. Hierarchical: nested plates. Topic models: crossed plates (documents $\times$ topics).

Standard representation in probabilistic programming languages (PyMC, Stan, Pyro).
