# PyTorch Tensor Attributes and Methods: Complete Guide

A comprehensive collection of Python scripts demonstrating PyTorch tensor operations, attributes, and best practices.

## 📚 Table of Contents

### Core Concepts (Files 01-06)
1. **[01_autograd_related.py](01_autograd_related.py)** - Autograd & Gradient Computation
2. **[02_data_related.py](02_data_related.py)** - Data Conversion & Detachment
3. **[03_shape_dtype_device.py](03_shape_dtype_device.py)** - Basic Tensor Attributes
4. **[04_memory_and_storage.py](04_memory_and_storage.py)** - Memory Layout & Views
5. **[05_linear_weight_convention.py](05_linear_weight_convention.py)** - nn.Linear Weight Convention
6. **[06_cat_visualization.py](06_cat_visualization.py)** - Visual Example

### Extended Operations (Files 07-15)
7. **[07_inplace_operations.py](07_inplace_operations.py)** - In-place Ops & Performance
8. **[08_broadcasting_rules.py](08_broadcasting_rules.py)** - Broadcasting Mechanics
9. **[09_indexing_masking.py](09_indexing_masking.py)** - Advanced Indexing & Masking
10. **[10_creation_methods.py](10_creation_methods.py)** - Tensor Creation & Initialization
11. **[11_reductions_statistics.py](11_reductions_statistics.py)** - Reductions & Statistics
12. **[12_reshaping_squeezing.py](12_reshaping_squeezing.py)** - Reshaping & Dimension Manipulation
13. **[13_concatenation_splitting.py](13_concatenation_splitting.py)** - Concatenation & Splitting
14. **[14_matrix_operations.py](14_matrix_operations.py)** - Matrix Ops & Linear Algebra
15. **[15_comparison_clamping_sorting.py](15_comparison_clamping_sorting.py)** - Comparisons & Sorting

---

## 📖 Detailed File Descriptions

### 01. Autograd-Related Operations
**Key Topics:**
- `requires_grad`, `is_leaf`, `grad_fn` attributes
- `.backward()` for scalar vs non-scalar outputs
- Gradient accumulation and clearing
- `torch.no_grad()` for parameter updates
- `retain_graph=True` for multiple backward passes

**When to use:**
- Training neural networks
- Custom gradient computation
- Understanding the computational graph

---

### 02. Data-Related Operations
**Key Topics:**
- `.item()` and `.tolist()` for converting to Python
- `.detach()` vs `.detach().clone()`
- `.cpu().numpy()` for NumPy interop
- `.data` caveat (bypasses autograd)
- CUDA tensor conversions

**When to use:**
- Converting tensors to Python/NumPy
- Extracting values for logging
- Interfacing with non-PyTorch code

---

### 03. Shape, Dtype, Device
**Key Topics:**
- `.shape`, `.size()`, `.ndim` attributes
- `.dtype`, `.device`, `.layout`
- `.is_cuda`, `.is_contiguous()`, `.stride()`
- Transpose helpers: `.T`, `.mT`, `.H`

**When to use:**
- Debugging tensor dimensions
- Device management (CPU/GPU)
- Understanding memory layout

---

### 04. Memory and Storage
**Key Topics:**
- View vs copy: `.transpose()`, `.permute()`, `.contiguous()`
- `.view()` vs `.reshape()` vs `.clone()`
- `.expand()` vs `.repeat()`
- `.gather()` and `.scatter_()` operations

**When to use:**
- Optimizing memory usage
- Understanding when operations copy data
- Advanced indexing operations

---

### 05. nn.Linear Weight Convention
**Key Topics:**
- Weight shape: `(out_features, in_features)`
- Forward pass: `x @ W.T + b`
- Autograd with linear layers
- "Row = one output neuron" intuition

**When to use:**
- Building custom neural network layers
- Understanding weight matrices
- Debugging neural network shapes

---

### 06. Cat Visualization
**Key Topics:**
- Loading and displaying images as tensors
- Tensor attributes for image data

**When to use:**
- Working with image data
- Visualizing tensor contents

---

### 07. In-place Operations
**Key Topics:**
- Operations ending with `_`: `add_()`, `mul_()`, `clamp_()`
- Memory efficiency vs autograd restrictions
- When to use in-place operations
- Common patterns: `.fill_()`, `.zero_()`, `.copy_()`

**When to use:**
- Parameter updates inside `torch.no_grad()`
- Memory-critical situations
- Explicit tensor initialization

**⚠️ Avoid:**
- On leaf tensors with `requires_grad=True`
- On intermediate computation results

---

### 08. Broadcasting Rules
**Key Topics:**
- The three broadcasting rules
- Dimension alignment (trailing dimensions)
- Size-1 dimension expansion
- Common patterns: batch operations, channel operations
- `.broadcast_to()` for explicit broadcasting

**When to use:**
- Adding bias to batched data
- Per-channel operations in CNNs
- Avoiding explicit loops

**⚠️ Watch out for:**
- Unintended broadcasting (use `keepdim=True`)
- Incompatible shapes

---

### 09. Indexing and Masking
**Key Topics:**
- Basic slicing (views) vs fancy indexing (copies)
- Boolean masking for conditional selection
- Integer array indexing
- `torch.where()`, `masked_fill()`, `masked_select()`
- Ellipsis `...` for flexible indexing
- `None`/`newaxis` for adding dimensions

**When to use:**
- Data filtering and selection
- Attention masking
- Conditional operations
- Advanced data manipulation

---

### 10. Creation Methods
**Key Topics:**
- Basic: `torch.tensor()`, `torch.from_numpy()`, `torch.as_tensor()`
- Constants: `zeros()`, `ones()`, `full()`, `empty()`, `eye()`
- Random: `rand()`, `randn()`, `randint()`, `randperm()`
- Sequential: `arange()`, `linspace()`, `logspace()`
- `_like` constructors: `zeros_like()`, `ones_like()`, etc.

**When to use:**
- Initializing tensors
- Creating test data
- Setting up neural network parameters

---

### 11. Reductions and Statistics
**Key Topics:**
- Basic: `sum()`, `mean()`, `std()`, `var()`, `prod()`
- Extrema: `min()`, `max()`, `argmin()`, `argmax()`, `aminmax()`
- Quantiles: `median()`, `quantile()`
- Norms: `norm()`, `dist()`
- Logical: `all()`, `any()`, `count_nonzero()`
- Cumulative: `cumsum()`, `cumprod()`

**When to use:**
- Computing statistics
- Data normalization
- Loss computation
- Dimension-wise operations with `dim` and `keepdim`

---

### 12. Reshaping and Squeezing
**Key Topics:**
- Shape changes: `reshape()`, `view()`, `flatten()`
- Dimension manipulation: `squeeze()`, `unsqueeze()`
- Reordering: `transpose()`, `permute()`, `movedim()`
- `.contiguous()` for memory layout
- Using `-1` for auto-inference

**When to use:**
- Preparing data for different layers
- Batch processing
- Channel/dimension reordering
- Flattening for fully connected layers

**Key differences:**
- `view()` requires contiguous memory
- `reshape()` may copy if needed
- Most reshaping ops return views

---

### 13. Concatenation and Splitting
**Key Topics:**
- `torch.cat()`: concatenate along existing dimension
- `torch.stack()`: stack along NEW dimension
- `torch.split()`: split into specific sizes
- `torch.chunk()`: split into equal parts
- `torch.unbind()`: unpack along dimension
- Helpers: `hstack()`, `vstack()`, `dstack()`

**When to use:**
- Batching samples
- Multi-GPU gathering
- Feature concatenation
- Data splitting

**Key differences:**
- `cat`: requires matching dimensions (except cat dim)
- `stack`: requires ALL dimensions to match

---

### 14. Matrix Operations
**Key Topics:**
- Multiplication: `*` (element-wise), `@` (matrix), `mm()`, `bmm()`
- Vector ops: `dot()`, `cross()`, `outer()`
- `einsum()` for complex operations
- Linear algebra: `inv()`, `det()`, `solve()`, `eig()`
- Decompositions: `svd()`, `qr()`, `cholesky()`
- Practical: attention mechanisms, batch transformations

**When to use:**
- Neural network operations
- Linear algebra computations
- Attention mechanisms
- Solving linear systems

**Key differences:**
- `matmul()` broadcasts, `mm()` doesn't
- `@` is cleanest for matrix multiplication

---

### 15. Comparison, Clamping, Sorting
**Key Topics:**
- Comparisons: `==`, `!=`, `<`, `<=`, `>`, `>=`
- Logical: `&`, `|`, `~`
- Conditional: `torch.where()`, `masked_fill()`
- Clamping: `clamp()`, `clip()`
- Sorting: `sort()`, `argsort()`, `topk()`, `kthvalue()`
- Special values: `isnan()`, `isinf()`, `isfinite()`
- Equality: `equal()`, `allclose()`, `isclose()`

**When to use:**
- Data filtering
- Outlier detection
- Top-k accuracy
- Value range limiting
- NaN handling

---

## 🚀 Quick Start

```bash
# Run any script individually
python 01_autograd_related.py
python 08_broadcasting_rules.py
python 14_matrix_operations.py

# All scripts are standalone and self-contained
```

## 📊 Coverage Summary

| Category | Files | Key Operations |
|----------|-------|----------------|
| **Autograd** | 01 | backward, grad, requires_grad |
| **Data Conversion** | 02 | item, tolist, numpy, detach |
| **Attributes** | 03 | shape, dtype, device, layout |
| **Memory** | 04, 07 | view, clone, contiguous, in-place |
| **Creation** | 10 | zeros, ones, rand, arange |
| **Indexing** | 09 | slicing, masking, where |
| **Reshaping** | 12 | reshape, squeeze, permute |
| **Combining** | 13 | cat, stack, split, chunk |
| **Math** | 11, 14 | sum, mean, matmul, einsum |
| **Comparison** | 15 | ==, <, clamp, sort, topk |
| **Broadcasting** | 08 | automatic shape alignment |
| **Neural Nets** | 05 | Linear layer conventions |

## 🎯 Learning Path

### Beginner
1. Start with **03** (basic attributes)
2. Then **10** (creation methods)
3. Then **02** (data conversion)

### Intermediate
4. Study **08** (broadcasting) - crucial!
5. Then **11** (reductions)
6. Then **12** (reshaping)
7. Then **09** (indexing)

### Advanced
8. Master **01** (autograd)
9. Then **14** (matrix operations)
10. Then **04** (memory layout)
11. Then **07** (in-place ops)

### Practical Applications
12. **05** for neural network layer understanding
13. **13** for data batching
14. **15** for data preprocessing

## 💡 Common Patterns

### Batch Processing
```python
# Add batch dimension
sample = torch.randn(3, 224, 224)
batch = sample.unsqueeze(0)  # (1, 3, 224, 224)

# Remove batch dimension
result = batch.squeeze(0)  # (3, 224, 224)
```

### Data Normalization
```python
# Per-channel normalization
mean = data.mean(dim=(0, 2, 3), keepdim=True)
std = data.std(dim=(0, 2, 3), keepdim=True)
normalized = (data - mean) / (std + 1e-5)
```

### Attention Mechanism
```python
# Q @ K^T @ V
scores = Q @ K.transpose(-2, -1)
weights = F.softmax(scores, dim=-1)
output = weights @ V
```

### Data Filtering
```python
# Boolean masking
valid = data > threshold
filtered = data[valid]

# Conditional replacement
cleaned = torch.where(torch.isnan(data), mean_val, data)
```

## 🔍 Debugging Tips

1. **Shape mismatches**: Print `.shape` liberally
2. **Device errors**: Check `.device` and use `.to(device)`
3. **Gradient issues**: Check `.requires_grad` and `.is_leaf`
4. **Memory errors**: Use `torch.cuda.empty_cache()` and check views
5. **Broadcasting bugs**: Use `keepdim=True` in reductions
6. **NaN/Inf values**: Use `torch.isnan()` and `torch.isfinite()`

## 📚 Additional Resources

- [PyTorch Documentation](https://pytorch.org/docs/stable/torch.html)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Broadcasting Semantics](https://pytorch.org/docs/stable/notes/broadcasting.html)
- [Autograd Mechanics](https://pytorch.org/docs/stable/notes/autograd.html)

## 🤝 Usage

Each file is:
- **Standalone**: Run independently
- **Documented**: Inline comments and docstrings
- **Educational**: Progressive examples from simple to complex
- **Practical**: Real-world use cases included

Feel free to:
- Run scripts to see output
- Modify examples to experiment
- Copy patterns into your projects
- Use as a reference guide

## ⚡ Performance Notes

- **Views are fast**: `transpose()`, `view()`, `expand()` don't copy
- **Copies are necessary**: `clone()`, `contiguous()`, boolean indexing
- **In-place saves memory**: But breaks autograd on leaf tensors
- **Broadcasting is efficient**: No data duplication
- **Batch operations**: Always faster than loops

## 🎓 Key Takeaways

1. **Understand views vs copies** - Critical for memory efficiency
2. **Master broadcasting** - Eliminates most loops
3. **Use `keepdim=True`** - Prevents broadcasting bugs
4. **Know autograd rules** - Required for training
5. **Profile your code** - Premature optimization is real
6. **Read error messages** - PyTorch errors are informative
7. **Dimension first** - Always verify tensor shapes

---

**Happy Learning! 🔥**

Each script demonstrates production-ready patterns used in real PyTorch projects.
