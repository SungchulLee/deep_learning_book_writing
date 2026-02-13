# Ordering Strategies

## Overview

The chain rule decomposition requires choosing an ordering of the variables. While the joint distribution $p(x)$ is independent of ordering, the complexity of each conditional depends heavily on the chosen order.

## Natural Orderings

For structured data, natural orderings exist:

- **Text**: left-to-right (temporal order)
- **Images**: raster scan (top-left to bottom-right)
- **Audio**: temporal order
- **Time series**: chronological order

## Raster Scan Order

For images with height $H$ and width $W$:

$$p(x) = \prod_{i=1}^{H} \prod_{j=1}^{W} p(x_{i,j} \mid x_{<i,j})$$

where $x_{<i,j}$ denotes all pixels above row $i$ and to the left of position $j$ in row $i$.

## Order Matters for Efficiency

Different orderings lead to different conditional complexities. For a checkerboard pattern, raster scan order requires long-range dependencies, while a hierarchical (multi-scale) ordering can exploit local structure more efficiently.

## Order-Agnostic Models

Some approaches avoid fixing a single ordering:

### NADE with Random Orderings
Train on multiple random orderings, making the model robust to order choice.

### XLNet
Uses permutation language modeling: sample a random permutation and factorize according to that order during training.

### Masked Autoregressive Flows
Use random masks to implicitly define orderings, with the constraint that the mask must maintain autoregressive structure.

## Multi-Scale Ordering

Hierarchical generation proceeds from coarse to fine:

1. Generate a low-resolution version
2. Conditionally generate higher-resolution details
3. Repeat until full resolution

This reduces the effective sequence length and allows the model to capture global structure before local details.
