# Tensor Parallelism

## Learning Objectives

- Understand column and row parallelism for linear layers
- Apply tensor parallelism to attention layers
- Analyze communication costs

## Core Idea

**Tensor parallelism** splits individual weight matrices across GPUs, allowing each GPU to compute a portion of each layer's output.

## Column Parallelism

Split the weight matrix along the **column dimension**:

$$W = [W_1 | W_2 | \cdots | W_P]$$

Each GPU $i$ computes $Y_i = XW_i$, producing a partial output. For a linear layer $Y = XW + b$:

```
GPU 0: Y_0 = X @ W_0     → partial output (first N/P columns)
GPU 1: Y_1 = X @ W_1     → partial output (next N/P columns)
Result: Y = [Y_0 | Y_1]  → concatenate
```

## Row Parallelism

Split along the **row dimension**:

$$W = \begin{bmatrix} W_1 \\ W_2 \\ \vdots \\ W_P \end{bmatrix}$$

Each GPU $i$ computes $Y_i = X_i W_i$ where $X_i$ is a partition of the input. Results are **summed** via all-reduce:

```
GPU 0: Y_0 = X_0 @ W_0   → partial sum
GPU 1: Y_1 = X_1 @ W_1   → partial sum
Result: Y = Y_0 + Y_1    → all-reduce
```

## Attention Parallelism

Multi-head attention is naturally parallelizable—split heads across GPUs:

```
GPU 0: heads 0-15  → partial attention output
GPU 1: heads 16-31 → partial attention output
All-reduce → combined output
```

For **Grouped Query Attention (GQA)**, KV heads are replicated or distributed across GPUs.

## Communication Cost

Per transformer layer with $P$ GPUs:

$$C_{\text{comm}} = 2 \times \text{AllReduce}(B \times S \times d_{\text{model}})$$

Two all-reduce operations per layer: one after the attention output projection, one after the FFN.

$$\text{Total per token} = 2L \times 2 \times \frac{P-1}{P} \times B \times S \times d \times \text{dtype\_size}$$

## Trade-off

| GPUs | Latency Reduction | Communication Overhead |
|------|-------------------|----------------------|
| 2 | ~1.8x | Low |
| 4 | ~3.2x | Moderate |
| 8 | ~5.5x | Significant |

Diminishing returns beyond 8 GPUs for tensor parallelism due to communication costs.

## References

1. Shoeybi, M., et al. (2020). "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism."
