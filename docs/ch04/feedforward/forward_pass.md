# Forward Pass

## Learning Objectives

!!! abstract "What You Will Learn"
    - Derive the forward propagation equations for single samples and mini-batches
    - Trace data flow through a concrete network with explicit numerical values
    - Understand the computational graph perspective and why caching intermediate values is essential
    - Implement forward propagation manually and compare with `nn.Module`
    - Analyze time and memory complexity of the forward pass
    - Apply numerical stability techniques for softmax and cross-entropy

## Prerequisites

| Topic | Why It Matters |
|-------|---------------|
| MLP Architecture (§4.2.1) | Defines the layer computation being executed |
| Matrix multiplication | Forward pass is a sequence of matrix-vector products |
| Activation functions (Ch 4.1) | Applied element-wise after each linear transformation |

---

## Overview

**Forward propagation** (or forward pass) is the process of computing the output of a neural network given an input. Data flows sequentially from the input layer through each hidden layer to the output layer, with every layer applying a linear transformation followed by a nonlinear activation.

The forward pass serves two purposes: during **inference**, it produces predictions; during **training**, it also builds a **computational graph** and caches intermediate values needed for backpropagation (§4.2.5).

---

## Mathematical Formulation

### Single Sample

For a network with $L$ layers and parameters $\boldsymbol{\theta} = \{(\mathbf{W}^{[l]}, \mathbf{b}^{[l]})\}_{l=1}^L$:

**Input assignment:**

$$
\mathbf{a}^{[0]} = \mathbf{x} \in \mathbb{R}^{n^{[0]}}
$$

**Layer-wise computation** for $l = 1, 2, \ldots, L$:

$$
\mathbf{z}^{[l]} = \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]} \in \mathbb{R}^{n^{[l]}}
$$

$$
\mathbf{a}^{[l]} = \sigma^{[l]}\!\left(\mathbf{z}^{[l]}\right) \in \mathbb{R}^{n^{[l]}}
$$

**Output:**

$$
\hat{\mathbf{y}} = \mathbf{a}^{[L]}
$$

Each layer thus performs two operations: (1) an **affine transformation** that projects the previous activation into a new space, and (2) a **pointwise nonlinearity** that introduces nonlinear capacity.

### Mini-Batch Processing

For a mini-batch of $B$ samples, each vector computation becomes a matrix computation. Using PyTorch's row-major convention where each row is a sample:

**Input:** $\mathbf{A}^{[0]} = \mathbf{X} \in \mathbb{R}^{B \times n^{[0]}}$

**Layer computation** for $l = 1, \ldots, L$:

$$
\mathbf{Z}^{[l]} = \mathbf{A}^{[l-1]} (\mathbf{W}^{[l]})^\top + \mathbf{1}_B \, (\mathbf{b}^{[l]})^\top \in \mathbb{R}^{B \times n^{[l]}}
$$

$$
\mathbf{A}^{[l]} = \sigma^{[l]}\!\left(\mathbf{Z}^{[l]}\right) \in \mathbb{R}^{B \times n^{[l]}}
$$

where the bias term is broadcast across all $B$ rows. In PyTorch, `nn.Linear` handles this as `output = input @ weight.T + bias` with automatic broadcasting.

!!! tip "Why Batch Processing Is Efficient"
    Matrix multiplication $\mathbf{A}^{[l-1]} (\mathbf{W}^{[l]})^\top$ computes all $B$ samples simultaneously using highly optimized BLAS/cuBLAS routines. A loop over individual samples would be orders of magnitude slower.

---

## Step-by-Step Numerical Example

Consider a 2-layer network for binary classification:

- Input: $\mathbf{x} \in \mathbb{R}^2$
- Hidden layer: 3 neurons with ReLU
- Output: 1 neuron with sigmoid

### Layer Dimensions

| Layer $l$ | $n^{[l-1]} \to n^{[l]}$ | $\mathbf{W}^{[l]}$ shape | $\mathbf{b}^{[l]}$ shape |
|-----------|--------------------------|---------------------------|---------------------------|
| 1 (hidden) | $2 \to 3$ | $(3, 2)$ | $(3,)$ |
| 2 (output) | $3 \to 1$ | $(1, 3)$ | $(1,)$ |

### Concrete Values

Let $\mathbf{x} = \begin{bmatrix} 0.5 \\ 0.8 \end{bmatrix}$ and

$$
\mathbf{W}^{[1]} = \begin{bmatrix} 0.2 & -0.3 \\ 0.4 & 0.1 \\ -0.5 & 0.6 \end{bmatrix}, \quad
\mathbf{b}^{[1]} = \begin{bmatrix} 0.1 \\ -0.2 \\ 0.0 \end{bmatrix}
$$

$$
\mathbf{W}^{[2]} = \begin{bmatrix} 0.7 & -0.4 & 0.3 \end{bmatrix}, \quad
b^{[2]} = -0.1
$$

**Step 1 — Hidden layer pre-activation:**

$$
\mathbf{z}^{[1]} = \mathbf{W}^{[1]} \mathbf{x} + \mathbf{b}^{[1]}
= \begin{bmatrix} 0.2(0.5) + (-0.3)(0.8) + 0.1 \\ 0.4(0.5) + 0.1(0.8) + (-0.2) \\ -0.5(0.5) + 0.6(0.8) + 0.0 \end{bmatrix}
= \begin{bmatrix} -0.04 \\ 0.08 \\ 0.23 \end{bmatrix}
$$

**Step 2 — Hidden layer activation (ReLU):**

$$
\mathbf{a}^{[1]} = \text{ReLU}(\mathbf{z}^{[1]}) = \begin{bmatrix} \max(0, -0.04) \\ \max(0, 0.08) \\ \max(0, 0.23) \end{bmatrix} = \begin{bmatrix} 0.00 \\ 0.08 \\ 0.23 \end{bmatrix}
$$

Note: the first neuron is "dead" (output = 0) for this input.

**Step 3 — Output layer pre-activation:**

$$
z^{[2]} = \mathbf{W}^{[2]} \mathbf{a}^{[1]} + b^{[2]}
= 0.7(0.00) + (-0.4)(0.08) + 0.3(0.23) + (-0.1)
= -0.069
$$

**Step 4 — Output layer activation (sigmoid):**

$$
\hat{y} = \sigma(z^{[2]}) = \frac{1}{1 + e^{0.069}} = \frac{1}{1.0714} \approx 0.4828
$$

Since $\hat{y} < 0.5$, the prediction is class 0.

---

## Computational Graph

Forward propagation builds a **directed acyclic graph (DAG)** — the computational graph — that records every operation performed on the data:

```
x ──→ [z¹ = W¹x + b¹] ──→ [a¹ = ReLU(z¹)] ──→ [z² = W²a¹ + b²] ──→ [ŷ = σ(z²)]
       ↑                                          ↑
      W¹, b¹                                     W², b²
```

Each node in the graph stores:

1. **The operation** (matmul, add, ReLU, sigmoid, ...)
2. **Its inputs** (pointers to parent nodes)
3. **Its output value** (the computed tensor)
4. **A recipe for computing the local gradient** (used in backpropagation)

!!! info "Caching for Backpropagation"
    During training, the forward pass must cache all intermediate values $\{\mathbf{z}^{[l]}, \mathbf{a}^{[l]}\}_{l=1}^L$. These cached values are consumed during the backward pass to compute gradients. This is why training uses ~2–3× the memory of inference.
    
    During **inference only**, intermediate values are not needed and can be discarded immediately:
    ```python
    with torch.no_grad():      # disables gradient tracking
        output = model(input)   # no computational graph built
    ```

---

## Activation Functions in the Forward Pass

### ReLU

$$
\text{ReLU}(z) = \max(0, z) = \begin{cases} z & z > 0 \\ 0 & z \leq 0 \end{cases}
$$

Computation: a single comparison per element. The fastest activation; this is why ReLU is the default choice for hidden layers.

### Sigmoid

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

Computation: one exponentiation, one addition, one division per element. Output is bounded in $(0, 1)$, used for binary classification outputs.

### Softmax

$$
\text{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}, \quad i = 1, \ldots, K
$$

Computation: $K$ exponentiations plus a normalization. Outputs a valid probability distribution (non-negative, sums to 1).

---

## Numerical Stability

### Softmax Overflow

**Problem:** If $z_i$ is large (e.g., $z_i = 1000$), then $e^{z_i}$ overflows to `inf`.

**Solution — log-sum-exp trick:** Subtract $\max_j z_j$ before exponentiating:

$$
\text{softmax}(\mathbf{z})_i = \frac{e^{z_i - z_{\max}}}{\sum_{j} e^{z_j - z_{\max}}}
$$

This is mathematically identical (the constant cancels) but numerically stable because the largest exponent is $e^0 = 1$.

```python
def stable_softmax(z: torch.Tensor) -> torch.Tensor:
    z_shifted = z - z.max(dim=-1, keepdim=True).values
    exp_z = torch.exp(z_shifted)
    return exp_z / exp_z.sum(dim=-1, keepdim=True)
```

### Cross-Entropy with Logits

**Problem:** Computing `log(softmax(z))` involves `log(exp(...))` which can lose precision.

**Solution:** PyTorch's `nn.CrossEntropyLoss` fuses log-softmax and negative log-likelihood into a single numerically stable operation using the log-sum-exp identity:

$$
\log \text{softmax}(\mathbf{z})_i = z_i - \log \sum_{j} e^{z_j} = z_i - z_{\max} - \log \sum_{j} e^{z_j - z_{\max}}
$$

This is why the output layer should produce **raw logits** (no softmax) when using `nn.CrossEntropyLoss`.

### Sigmoid + BCE Stability

**Problem:** If $\hat{y} = \sigma(z)$ is very close to 0 or 1, then $\log(\hat{y})$ or $\log(1 - \hat{y})$ diverges.

**Solution:** Use `nn.BCEWithLogitsLoss`, which applies sigmoid internally with numerical safeguards:

$$
\text{BCE}(z, y) = \max(z, 0) - z \cdot y + \log(1 + e^{-|z|})
$$

```python
# ✗ Numerically fragile
loss = nn.BCELoss()(torch.sigmoid(logits), targets)

# ✓ Numerically stable (sigmoid is fused inside)
loss = nn.BCEWithLogitsLoss()(logits, targets)
```

---

## PyTorch Implementation

### Manual Forward Pass

```python
import torch
import torch.nn.functional as F


def forward_pass_manual(
    x: torch.Tensor,
    weights: list[torch.Tensor],
    biases: list[torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Manual forward propagation through an L-layer network.
    
    Hidden layers use ReLU; output layer uses sigmoid (binary classification).
    
    Args:
        x:       Input tensor, shape (B, n_in)
        weights: [W1, W2, ..., WL] where Wl has shape (n_l, n_{l-1})
        biases:  [b1, b2, ..., bL] where bl has shape (n_l,)
    
    Returns:
        output: Predictions, shape (B, n_out)
        cache:  Dict of all intermediate z and a values
    """
    L = len(weights)
    cache = {'a0': x}
    a = x
    
    for l in range(L):
        # Affine: z = a @ W^T + b   (PyTorch row-major convention)
        z = a @ weights[l].T + biases[l]
        cache[f'z{l+1}'] = z
        
        # Activation
        if l < L - 1:
            a = F.relu(z)          # hidden layers
        else:
            a = torch.sigmoid(z)   # output layer
        cache[f'a{l+1}'] = a
    
    return a, cache


# ── Example ──
torch.manual_seed(42)

W1 = torch.randn(3, 2) * 0.5     # hidden: 2 → 3
b1 = torch.zeros(3)
W2 = torch.randn(1, 3) * 0.5     # output: 3 → 1
b2 = torch.zeros(1)

x = torch.tensor([[0.5, 0.8],
                   [0.1, 0.2],
                   [0.9, 0.4]])    # batch of 3

output, cache = forward_pass_manual(x, [W1, W2], [b1, b2])

print("=== Forward Pass Trace ===")
for key in sorted(cache.keys()):
    print(f"  {key}: shape {cache[key].shape}")
    print(f"        {cache[key].detach()}")
print(f"\nPredictions: {output.detach().squeeze()}")
```

### Forward Pass with Intermediate Tracking (`nn.Module`)

```python
import torch
import torch.nn as nn


class TrackedMLP(nn.Module):
    """MLP that optionally returns all intermediate activations."""
    
    def __init__(self, layer_sizes: list[int]):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
    
    def forward(
        self, x: torch.Tensor, return_intermediates: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        cache = {'a0': x} if return_intermediates else None
        a = x
        
        for i, layer in enumerate(self.layers):
            z = layer(a)
            if return_intermediates:
                cache[f'z{i+1}'] = z
            
            # Activation: ReLU for hidden, identity for output
            a = torch.relu(z) if i < len(self.layers) - 1 else z
            
            if return_intermediates:
                cache[f'a{i+1}'] = a
        
        return (a, cache) if return_intermediates else a


# ── Usage ──
model = TrackedMLP([784, 256, 128, 10])
x = torch.randn(32, 784)

# Standard inference (no intermediates)
logits = model(x)
print(f"Output shape: {logits.shape}")

# Debug mode (with intermediates)
logits, cache = model(x, return_intermediates=True)
print("\nIntermediate shapes:")
for k, v in cache.items():
    print(f"  {k}: {v.shape}")
```

### Efficient Inference with `torch.no_grad()`

```python
model = TrackedMLP([784, 256, 128, 10])
model.eval()   # set dropout/batchnorm to eval mode

x = torch.randn(64, 784)

# ── Training mode: builds computational graph ──
logits_train = model(x)
print(f"Grad tracking: {logits_train.requires_grad}")  # True

# ── Inference mode: no graph, less memory ──
with torch.no_grad():
    logits_infer = model(x)
    print(f"Grad tracking: {logits_infer.requires_grad}")  # False

# Verify outputs are identical
print(f"Max difference: {(logits_train - logits_infer).abs().max().item():.1e}")  # 0.0
```

---

## Computational Complexity

### Time Complexity

For a single layer with $n_{\text{in}}$ inputs and $n_{\text{out}}$ outputs processing a batch of $B$ samples:

$$
T_{\text{layer}} = O(B \cdot n_{\text{in}} \cdot n_{\text{out}})
$$

This is the cost of the matrix multiplication $\mathbf{A}^{[l-1]} (\mathbf{W}^{[l]})^\top$. The activation function adds $O(B \cdot n_{\text{out}})$ which is dominated by the matmul.

For the full network:

$$
T_{\text{forward}} = O\!\left(B \sum_{l=1}^{L} n^{[l-1]} \cdot n^{[l]}\right) = O(B \cdot |\boldsymbol{\theta}|)
$$

The forward pass cost is **linear** in the number of parameters and the batch size.

### Memory Complexity

| Mode | What is stored | Memory |
|------|---------------|--------|
| **Training** | All $\mathbf{z}^{[l]}, \mathbf{a}^{[l]}$ for backprop | $O\!\left(B \sum_{l=0}^{L} n^{[l]}\right)$ |
| **Inference** | Only current layer's activation | $O\!\left(B \cdot \max_l n^{[l]}\right)$ |

Training memory is dominated by the cached activations, not the parameters themselves. For very deep or wide networks, this can be the bottleneck. Techniques like **gradient checkpointing** trade compute for memory by recomputing activations during the backward pass instead of caching them.

---

## Visualization: Activation Distributions

```python
import matplotlib.pyplot as plt

model = TrackedMLP([784, 512, 256, 128, 10])
x = torch.randn(500, 784)  # 500 random samples

logits, cache = model(x, return_intermediates=True)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for i, l in enumerate([1, 2, 3, 4]):
    vals = cache[f'a{l}'].detach().numpy().flatten()
    axes[i].hist(vals, bins=50, edgecolor='black', alpha=0.7, density=True)
    axes[i].set_title(f'Layer {l} activations\n'
                      f'mean={vals.mean():.3f}, std={vals.std():.3f}')
    axes[i].set_xlabel('Activation value')

plt.suptitle('Activation Distributions Through the Network', fontsize=14)
plt.tight_layout()
plt.savefig('forward_pass_activations.png', dpi=150, bbox_inches='tight')
plt.show()
```

Monitoring activation distributions reveals common issues: values collapsing to zero (vanishing activations), saturating at boundaries (sigmoid/tanh), or growing unboundedly (exploding activations). Healthy activations have moderate mean and standard deviation across all layers.

---

## Key Takeaways

!!! success "Summary"
    1. **Forward propagation** computes the network output by sequentially applying affine transformation + activation at each layer
    2. **Mini-batch processing** parallelizes across samples using matrix multiplication: $\mathbf{Z}^{[l]} = \mathbf{A}^{[l-1]} (\mathbf{W}^{[l]})^\top + \mathbf{b}^{[l]}$
    3. The forward pass builds a **computational graph** that records operations for backpropagation
    4. **Intermediate values must be cached** during training for the backward pass; inference can discard them
    5. **Numerical stability** requires the log-sum-exp trick for softmax and fused loss functions (`CrossEntropyLoss`, `BCEWithLogitsLoss`)
    6. **Time complexity** is $O(B \cdot |\boldsymbol{\theta}|)$; **memory** scales with $O(B \sum_l n^{[l]})$ during training

---

## References

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. Chapter 6.4.
- PyTorch Documentation: [`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)
- Griewank, A., & Walther, A. (2008). *Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation*. SIAM.
