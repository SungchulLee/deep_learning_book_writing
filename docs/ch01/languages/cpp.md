# C++ in Deep Learning

C++ underpins the performance-critical layers of every major deep learning framework. Understanding where C++ fits helps you diagnose bottlenecks and extend frameworks when Python alone is insufficient.

## Definition

C++ is a statically typed, compiled language that provides direct memory management and zero-cost abstractions. In deep learning, C++ powers the backend execution engines of PyTorch (ATen/c10), TensorFlow (XLA), and ONNX Runtime. The PyTorch C++ frontend (LibTorch) exposes the same tensor API for inference in production environments where Python is unavailable.

## Explanation

Most deep learning practitioners never write C++ directly because Python APIs delegate computation to optimized C++ and CUDA backends. However, C++ becomes necessary in three scenarios:

- **Custom operators**: When a novel layer or loss function cannot be expressed efficiently using existing PyTorch primitives, you write a C++ (or CUDA) extension.
- **Production inference**: Deploying models on embedded devices, mobile platforms, or latency-sensitive servers often requires exporting to TorchScript or ONNX and running via a C++ runtime.
- **Framework internals**: Contributing to PyTorch core, writing autograd functions, or optimizing memory allocators requires understanding the C++ codebase.

Key C++ data structures used in framework internals:

$$
\begin{array}{lll}
\texttt{vector} & \text{Dynamic array} & O(1) \text{ amortized append} \\
\texttt{unordered\_map} & \text{Hash table} & O(1) \text{ average lookup} \\
\texttt{shared\_ptr} & \text{Reference-counted pointer} & \text{Automatic memory management}
\end{array}
$$

## Examples

While custom operators are written in C++, you invoke them from Python. Here is how PyTorch bridges the two worlds:

```python
import torch

# PyTorch operations dispatch to C++ (ATen) under the hood
x = torch.randn(3, 3)
y = torch.randn(3, 3)

# This single Python call triggers optimized C++ matrix multiply
z = torch.mm(x, y)
print(f"Result:\n{z}")

# TorchScript: compile Python model to C++-executable IR
@torch.jit.script
def relu_forward(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x > 0, x, torch.zeros_like(x))

out = relu_forward(torch.tensor([-1.0, 0.0, 2.0, -3.0]))
print(f"ReLU output: {out}")
```
