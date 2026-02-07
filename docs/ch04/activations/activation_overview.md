# Activation Functions Overview

## Overview

Activation functions introduce **nonlinearity** into neural networks, enabling them to learn complex patterns that linear transformations alone cannot capture. Without activation functions, a deep neural network of any depth collapses mathematically to a single linear transformation, severely limiting its representational power.

This section provides a unified overview of activation functions: why they are necessary, how they work, and the landscape of available choices in modern deep learning.

## Learning Objectives

By the end of this section, you will understand:

1. Why linear transformations alone are insufficient for complex learning tasks
2. The mathematical proof that stacked linear layers collapse to a single layer
3. How nonlinearity enables the Universal Approximation Theorem
4. The role of activation functions in creating complex decision boundaries
5. The taxonomy of modern activation functions and their design principles

---

## The Limitation of Linear Networks

### Mathematical Foundation

A **linear transformation** maps an input $\mathbf{x}$ through weight matrix $\mathbf{W}$ and bias $\mathbf{b}$:

$$\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}$$

Consider stacking two linear layers without activation functions:

$$\begin{aligned}
\mathbf{h} &= \mathbf{W}_1\mathbf{x} + \mathbf{b}_1 \\
\mathbf{y} &= \mathbf{W}_2\mathbf{h} + \mathbf{b}_2
\end{aligned}$$

Substituting the first equation into the second:

$$\begin{aligned}
\mathbf{y} &= \mathbf{W}_2(\mathbf{W}_1\mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2 \\
&= \mathbf{W}_2\mathbf{W}_1\mathbf{x} + \mathbf{W}_2\mathbf{b}_1 + \mathbf{b}_2 \\
&= \mathbf{W}'\mathbf{x} + \mathbf{b}'
\end{aligned}$$

where $\mathbf{W}' = \mathbf{W}_2\mathbf{W}_1$ and $\mathbf{b}' = \mathbf{W}_2\mathbf{b}_1 + \mathbf{b}_2$.

**Induction to $L$ layers.** The argument extends immediately by induction. Suppose a stack of $L$ linear layers $\mathbf{y} = \mathbf{W}_L(\cdots(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1)\cdots) + \mathbf{b}_L$ collapses to $\mathbf{y} = \mathbf{W}'\mathbf{x} + \mathbf{b}'$ where

$$\mathbf{W}' = \prod_{l=L}^{1} \mathbf{W}_l, \quad \mathbf{b}' = \sum_{l=1}^{L}\left(\prod_{k=l+1}^{L}\mathbf{W}_k\right)\mathbf{b}_l$$

!!! warning "Critical Insight"
    **Any number of stacked linear layers without activation functions is mathematically equivalent to a single linear layer.** This means a 100-layer "deep" network without activations has the same representational capacity as a simple linear regression model.

### PyTorch Demonstration

```python
import torch
import torch.nn as nn

# Demonstrate the collapse of linear layers
x = torch.tensor([[1.0, 2.0]])  # Input: (1, 2)

# Two separate linear layers (without activation)
w1 = torch.tensor([[2.0], [2.0]])  # Shape: (2, 1)
w2 = torch.tensor([[3.0]])         # Shape: (1, 1)

# Sequential application
layer1_output = x @ w1
layer2_output = layer1_output @ w2
print(f"Sequential layers: {layer2_output.item()}")  # Output: 18.0

# Direct combination
combined_weight = w1 @ w2
direct_output = x @ combined_weight
print(f"Combined single layer: {direct_output.item()}")  # Output: 18.0

# They are identical!
assert torch.allclose(layer2_output, direct_output)
```

---

## How Activation Functions Break the Linearity

When we insert a nonlinear activation function $\sigma$ between layers:

$$\begin{aligned}
\mathbf{h} &= \sigma(\mathbf{W}_1\mathbf{x} + \mathbf{b}_1) \\
\mathbf{y} &= \mathbf{W}_2\mathbf{h} + \mathbf{b}_2
\end{aligned}$$

This composition **cannot be simplified** to a single linear transformation because of the nonlinear function $\sigma$ in between. Each layer can now learn increasingly abstract representations.

### PyTorch Demonstration with ReLU

```python
import torch
import torch.nn.functional as F

x = torch.tensor([[1.0, 2.0]])
w1 = torch.tensor([[2.0], [2.0]])
w2 = torch.tensor([[3.0]])

# With ReLU activation between layers
layer1_linear = x @ w1
layer1_activated = F.relu(layer1_linear)  # Nonlinearity!
output_with_activation = layer1_activated @ w2

print(f"With activation: {output_with_activation.item()}")

# This is NOT equivalent to any single linear transformation
# The network can now learn nonlinear patterns!
```

---

## The Universal Approximation Theorem

The **Universal Approximation Theorem** (Cybenko, 1989; Hornik, 1991) states that a feedforward network with:

- A single hidden layer
- Sufficient neurons
- A nonlinear activation function

can approximate **any continuous function** on a compact subset of $\mathbb{R}^n$ to arbitrary precision.

### Formal Statement

For any continuous function $f: [0,1]^n \to \mathbb{R}$, any $\epsilon > 0$, and any continuous, bounded, non-constant activation function $\sigma$, there exists a network:

$$F(\mathbf{x}) = \sum_{i=1}^{N} v_i \sigma(\mathbf{w}_i^T\mathbf{x} + b_i) + c$$

such that:

$$|F(\mathbf{x}) - f(\mathbf{x})| < \epsilon \quad \forall \mathbf{x} \in [0,1]^n$$

!!! info "Practical Implications"
    While the theorem guarantees that such networks *exist*, it doesn't specify how to find them or how many neurons are needed. Deep networks often achieve better approximations with fewer parameters due to their hierarchical feature learning.

### Depth Separation Results

Subsequent work has shown that while width is sufficient in theory, **depth** provides exponential efficiency gains. Telgarsky (2016) proved that there exist functions expressible by $O(k^3)$-layer networks that require $O(2^k)$ neurons to approximate with $O(k)$-layer networks. This provides a formal justification for deep architectures beyond the single-hidden-layer guarantee.

---

## Geometric Interpretation

### Linear Networks: Hyperplane Decision Boundaries

A linear classifier can only create **hyperplane** decision boundaries:

- In 2D: straight lines
- In 3D: flat planes
- In $n$D: $(n-1)$-dimensional hyperplanes

This means a linear network **cannot** solve problems like:

- XOR classification
- Circular or spiral decision boundaries
- Any dataset that is not linearly separable

### Nonlinear Networks: Complex Decision Boundaries

With activation functions, neural networks can learn:

- Curved decision boundaries
- Multiple disconnected regions
- Arbitrary complex shapes

Each layer transforms the space, potentially making previously non-separable data linearly separable in the transformed space.

```python
import torch
import torch.nn as nn

class NonlinearClassifier(nn.Module):
    """Network that can solve XOR and other nonlinear problems."""
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 1)
        self.activation = nn.ReLU()  # Critical nonlinearity!
    
    def forward(self, x):
        x = self.activation(self.fc1(x))  # Transform space
        x = self.fc2(x)                    # Linear decision in new space
        return x

# XOR dataset (not linearly separable)
X_xor = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y_xor = torch.tensor([[0.], [1.], [1.], [0.]])

model = NonlinearClassifier()
# This network CAN learn XOR because of the nonlinearity
```

---

## Element-wise Application

Activation functions are applied **element-wise** (also called point-wise), meaning each element of the input tensor is transformed independently:

$$\sigma(\mathbf{z}) = [\sigma(z_1), \sigma(z_2), \ldots, \sigma(z_n)]^T$$

This property:

1. **Preserves tensor shape**: Input shape equals output shape
2. **Enables parallel computation**: Each element can be computed independently
3. **Allows different neurons to have different activation levels**: Creating rich representations

```python
import torch

z = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

# ReLU applied element-wise
relu_output = torch.relu(z)
print(f"Input:  {z.tolist()}")
print(f"Output: {relu_output.tolist()}")
# Output: [0.0, 0.0, 0.0, 1.0, 2.0]

# Each element is independently transformed
```

!!! note "Exception: Softmax"
    The **Softmax** function is a notable exception to the element-wise rule. It normalizes across a specified dimension so that outputs sum to 1, introducing dependencies between elements. See the [Softmax section](softmax.md) for details.

---

## Taxonomy of Activation Functions

Modern activation functions can be organized along several design axes:

### By Historical Era

| Era | Functions | Key Advance |
|-----|-----------|-------------|
| **Classical (1980s–2000s)** | Sigmoid, Tanh | Differentiable, bounded |
| **ReLU Revolution (2010s)** | ReLU, Leaky ReLU, PReLU, ELU | Solved vanishing gradients |
| **Smooth & Adaptive (2016+)** | GELU, Swish/SiLU, Mish | Smooth, non-monotonic, data-driven |
| **Gated Mechanisms (2017+)** | GLU, SwiGLU, GeGLU | Learned gating for transformers |

### By Design Properties

| Property | Functions | Benefit |
|----------|-----------|---------|
| **Bounded output** | Sigmoid, Tanh, Softmax | Stable, interpretable |
| **Unbounded positive** | ReLU, Leaky ReLU, PReLU | No vanishing gradient (positive side) |
| **Smooth** | GELU, Swish, Mish, ELU | Better optimization landscape |
| **Non-monotonic** | GELU, Swish, Mish | Richer gradient signal |
| **Self-normalizing** | SELU | No explicit normalization needed |
| **Gated** | GLU, SwiGLU, GeGLU | Learned feature selection |

### By Application Domain

| Domain | Default Choice | Alternative |
|--------|---------------|-------------|
| **CNN hidden layers** | ReLU | Leaky ReLU, Swish |
| **Transformer FFN** | GELU | SwiGLU (modern LLMs) |
| **RNN/LSTM gates** | Sigmoid + Tanh | (built-in, do not change) |
| **Binary classification output** | Sigmoid (via loss) | — |
| **Multiclass output** | Softmax (via loss) | — |
| **GAN discriminator** | Leaky ReLU | — |
| **Deep MLP** | ReLU + BatchNorm | SELU |
| **Mobile/edge deployment** | Hardswish | ReLU |

---

## Desirable Properties of Activation Functions

When evaluating or designing activation functions, consider these properties:

### Non-Saturation

An activation function **saturates** when its gradient approaches zero for large-magnitude inputs. Sigmoid and tanh saturate in both directions, causing the vanishing gradient problem. ReLU avoids this for positive inputs.

### Zero-Centered Output

When activations have a non-zero mean (e.g., sigmoid outputs are always positive), the gradients of subsequent weights are forced to have the same sign, leading to **zig-zag dynamics** during gradient descent. Tanh, ELU, and GELU produce approximately zero-centered outputs.

### Smoothness

Non-smooth activations like ReLU have undefined gradients at the origin. While this rarely causes problems in practice, smooth activations (GELU, Swish, ELU) can yield better optimization landscapes, particularly in attention-based architectures.

### Computational Efficiency

Simpler functions (ReLU: one comparison; Leaky ReLU: one comparison + one multiplication) are faster than those requiring exponentials (ELU, sigmoid) or error functions (GELU). For latency-critical applications, piecewise-linear approximations (Hardswish, approximate GELU) offer a practical compromise.

---

## Summary

| Aspect | Without Activation | With Activation |
|--------|-------------------|-----------------|
| **Multiple layers** | Collapse to single linear layer | Each layer learns distinct features |
| **Decision boundaries** | Only hyperplanes | Arbitrary complex shapes |
| **Function approximation** | Only linear functions | Any continuous function |
| **XOR problem** | Cannot solve | Can solve |
| **Deep learning** | Impossible | Enabled |

!!! tip "Key Takeaway"
    Activation functions are **not optional embellishments** but **fundamental requirements** for neural networks to learn complex patterns. Without them, deep learning as we know it would be impossible.

---

## Further Reading

- Cybenko, G. (1989). "Approximation by superpositions of a sigmoidal function"
- Hornik, K., Stinchcombe, M., & White, H. (1989). "Multilayer feedforward networks are universal approximators"
- Telgarsky, M. (2016). "Benefits of depth in neural networks"
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*, Chapter 6.3
