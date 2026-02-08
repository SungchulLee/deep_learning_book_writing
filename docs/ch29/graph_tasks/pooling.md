# 29.5.3 Graph Pooling

## Overview
Graph pooling compresses node embeddings into a fixed-size graph representation. Flat pooling methods: sum, mean, max, and attention-weighted. Choice affects preserved information: sum captures size, mean captures distribution, max captures extremes.

## Attention Pooling
$$h_G = \sum_v \alpha_v h_v, \quad \alpha_v = \frac{\exp(f(h_v))}{\sum_u \exp(f(h_u))}$$
where $f$ is a learnable scoring function.

## Multi-Pool
Concatenate multiple pooling strategies: $h_G = [\text{sum}(h) \| \text{mean}(h) \| \text{max}(h)]$
