# Memory Management in PyTorch Tensors

## Learning Objectives

By the end of this section, you will be able to:

- Understand how PyTorch manages tensor memory allocation
- Distinguish between contiguous and non-contiguous tensors
- Apply memory-efficient operations in tensor manipulation
- Debug common memory-related issues in PyTorch workflows

---

## Overview

Memory management is a critical aspect of working with PyTorch tensors, especially when dealing with large datasets or complex neural network architectures. Understanding how PyTorch allocates, shares, and deallocates memory helps you write efficient code and avoid common pitfalls like memory leaks or unexpected tensor modifications.

---

## Memory Allocation Fundamentals

### Storage vs Tensor

In PyTorch, every tensor is a view into a **Storage** object that holds the actual data. Multiple tensors can share the same storage, which is key to understanding memory behavior:

```python
import torch

# Create a tensor
x = torch.tensor([[1, 2, 3], [4, 5, 6]])

# Access the underlying storage
print(x.storage())        # Flat 1D storage: [1, 2, 3, 4, 5, 6]
print(x.storage_offset()) # Starting position in storage: 0
print(x.stride())         # Steps to move along each dimension: (3, 1)
```

The relationship between tensor and storage can be expressed as:

$$
\text{tensor}[i, j] = \text{storage}[\text{offset} + i \cdot \text{stride}[0] + j \cdot \text{stride}[1]]
$$

### Memory Sharing Through Views

View operations create new tensors that share the same underlying storage:

```python
x = torch.arange(12).reshape(3, 4)
y = x[1:3, 1:3]  # Slice creates a view

# Both point to the same storage
print(x.data_ptr() == y.data_ptr())  # May differ due to offset
print(x.storage().data_ptr() == y.storage().data_ptr())  # True

# Modifying y affects x
y[0, 0] = 100
print(x[1, 1])  # 100
```

---

## Contiguous Memory Layout

### What is Contiguity?

A tensor is **contiguous** when its elements are stored in memory in the same order as they would be accessed by iterating through the tensor in row-major (C-style) order:

```python
x = torch.arange(12).reshape(3, 4)
print(x.is_contiguous())  # True

# Transpose creates a non-contiguous view
y = x.t()  # or x.T
print(y.is_contiguous())  # False

# Make contiguous by copying data
y_contig = y.contiguous()
print(y_contig.is_contiguous())  # True
```

### Why Contiguity Matters

1. **Performance**: Many operations require or are optimized for contiguous tensors
2. **Correctness**: Some operations (like `view()`) only work on contiguous tensors
3. **Memory Efficiency**: Non-contiguous tensors may require copying for certain operations

```python
x = torch.randn(3, 4)
y = x.t()

# view() requires contiguous tensor
try:
    z = y.view(12)
except RuntimeError as e:
    print(f"Error: {e}")

# Solutions:
z1 = y.contiguous().view(12)  # Make contiguous first
z2 = y.reshape(12)            # reshape() handles non-contiguous tensors
```

---

## Memory-Efficient Operations

### In-Place Operations

In-place operations modify tensors directly without allocating new memory:

```python
x = torch.randn(1000, 1000)

# Out-of-place (allocates new memory)
y = x + 1

# In-place (modifies x directly)
x.add_(1)  # Underscore suffix indicates in-place

# Common in-place operations
x.mul_(2)      # Multiply
x.zero_()      # Fill with zeros
x.fill_(3.14)  # Fill with value
x.clamp_(0, 1) # Clamp values
```

!!! warning "Caution with In-Place Operations"
    In-place operations can break gradient computation if the tensor requires gradients:
    
    ```python
    x = torch.randn(3, requires_grad=True)
    y = x ** 2
    y.add_(1)  # RuntimeError: in-place operation on leaf variable
    ```

### Pre-allocated Buffers

For repetitive operations, pre-allocate output tensors:

```python
# Without pre-allocation (creates new tensor each iteration)
for i in range(1000):
    result = torch.matmul(A, B)

# With pre-allocation (reuses memory)
result = torch.empty(A.shape[0], B.shape[1])
for i in range(1000):
    torch.matmul(A, B, out=result)
```

---

## GPU Memory Management

### Memory Allocation on GPU

```python
# Move tensor to GPU
x = torch.randn(1000, 1000, device='cuda')

# Check memory usage
print(f"Allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
print(f"Cached: {torch.cuda.memory_reserved() / 1e6:.2f} MB")
```

### Releasing GPU Memory

```python
# Delete tensor reference
del x

# Empty cache (releases cached memory back to GPU)
torch.cuda.empty_cache()

# Synchronize before measuring
torch.cuda.synchronize()
```

### Memory-Efficient Training Pattern

```python
def train_step(model, data, target, optimizer):
    optimizer.zero_grad()
    
    # Forward pass
    output = model(data)
    loss = criterion(output, target)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    # Return scalar loss (detached from computation graph)
    return loss.item()  # .item() converts to Python scalar
```

---

## Common Memory Issues and Solutions

### Issue 1: Accumulating Computation Graphs

```python
# WRONG: Accumulates graphs in memory
losses = []
for batch in dataloader:
    loss = model(batch)
    losses.append(loss)  # Keeps entire graph!

# CORRECT: Detach or convert to scalar
losses = []
for batch in dataloader:
    loss = model(batch)
    losses.append(loss.item())  # or loss.detach()
```

### Issue 2: Hidden References

```python
# WRONG: Slice keeps reference to original
x = torch.randn(10000, 10000)
y = x[0, :]  # y shares storage with x
del x        # Storage not freed!

# CORRECT: Clone if you need only the slice
y = x[0, :].clone()
del x  # Now storage can be freed
```

### Issue 3: Memory Fragmentation

```python
# Use memory snapshots for debugging
torch.cuda.memory._record_memory_history()

# Your code here...

# Get snapshot
snapshot = torch.cuda.memory._snapshot()
```

---

## Practical Memory Profiling

### Using PyTorch Profiler

```python
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    profile_memory=True
) as prof:
    # Your code here
    output = model(input)
    loss = criterion(output, target)
    loss.backward()

print(prof.key_averages().table(sort_by="cuda_memory_usage"))
```

### Memory Summary

```python
def print_memory_summary():
    if torch.cuda.is_available():
        print(torch.cuda.memory_summary())
```

---

## Best Practices Summary

| Practice | Benefit |
|----------|---------|
| Use in-place operations when safe | Reduces memory allocation |
| Pre-allocate output buffers | Avoids repeated allocations |
| Detach tensors from graphs when storing | Prevents graph accumulation |
| Clone slices if original is not needed | Allows garbage collection |
| Use `torch.no_grad()` during inference | Disables gradient tracking |
| Call `empty_cache()` judiciously | Releases unused cached memory |

---

## Connection to Other Topics

- **[Memory Layout and Strides](../tensors/memory_layout_strides.md)**: Deep dive into stride-based memory access
- **[Clone and Copy](clone_and_copy.md)**: When to copy vs share memory
- **[In-Place Operations](inplace_operations.md)**: Detailed guide to in-place tensor operations
- **[GPU and Performance](../performance/gpu.md)**: GPU-specific optimization techniques

---

## Exercises

1. **Memory Sharing Investigation**: Create a tensor and several views. Modify one view and observe the effects on others. Use `storage().data_ptr()` to verify memory sharing.

2. **Contiguity Challenge**: Create a non-contiguous tensor through transposition. Compare the performance of operations on contiguous vs non-contiguous versions.

3. **Memory Profiling**: Write a training loop and use the profiler to identify memory bottlenecks. Apply optimizations and measure improvements.

---

## Summary

Effective memory management in PyTorch requires understanding the relationship between tensors and their underlying storage. Key concepts include memory sharing through views, the importance of contiguity, and techniques for minimizing memory usage through in-place operations and pre-allocation. For GPU workloads, proper memory management is essential for training large models efficiently.
