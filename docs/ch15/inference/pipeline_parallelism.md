# Pipeline Parallelism

## Learning Objectives

- Understand pipeline parallelism for LLM inference and training
- Analyze pipeline bubble overhead
- Compare with tensor parallelism

## Core Concept

**Pipeline parallelism** assigns different layers to different GPUs. For a model with $L$ layers and $P$ GPUs, each GPU handles $L/P$ consecutive layers:

```
GPU 0: Layers 0-19    [==========]
GPU 1: Layers 20-39              [==========]
GPU 2: Layers 40-59                         [==========]
GPU 3: Layers 60-79                                    [==========]
```

## Pipeline Bubble

The main inefficiency is the **pipeline bubble**: during startup and teardown, some GPUs are idle.

For $P$ pipeline stages and $M$ microbatches:

$$\text{Bubble fraction} = \frac{P - 1}{M + P - 1}$$

| Microbatches ($M$) | GPUs ($P$) | Bubble |
|-------------------|-----------|--------|
| 4 | 4 | 43% |
| 8 | 4 | 27% |
| 16 | 4 | 16% |
| 32 | 4 | 9% |

## Interleaved Pipeline

Assign non-contiguous layers to reduce bubble size:

```
Standard:     GPU 0: [L0-L19]  GPU 1: [L20-L39]
Interleaved:  GPU 0: [L0-L9, L20-L29]  GPU 1: [L10-L19, L30-L39]
```

Interleaving reduces bubble by factor of $v$ (number of virtual stages per GPU) but increases communication.

## Pipeline vs. Tensor Parallelism

| Aspect | Tensor Parallel | Pipeline Parallel |
|--------|----------------|-------------------|
| Communication | All-reduce (high bandwidth) | Point-to-point (lower) |
| Latency | Adds sync per layer | Adds pipeline depth |
| Memory per GPU | Model/P + full KV cache | Full layers + partial KV |
| Best for | Latency-sensitive serving | Throughput, training |
| Typical scale | 2-8 GPUs | 2-16+ GPUs |

## Combined Approach

Production systems often combine both:

```
8 GPUs total:
- 4-way Tensor Parallel within a node
- 2-way Pipeline Parallel across nodes

GPU Layout:
  Node 0: [TP0, TP1, TP2, TP3] → Layers 0-39
  Node 1: [TP0, TP1, TP2, TP3] → Layers 40-79
```

## References

1. Huang, Y., et al. (2019). "GPipe: Efficient Training of Giant Neural Networks Using Pipeline Parallelism." *NeurIPS*.
2. Narayanan, D., et al. (2021). "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM." *SC*.
