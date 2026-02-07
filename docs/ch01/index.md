# Chapter 1: Introduction to Python for Deep Learning

This chapter provides the foundational setup and library overview needed before
diving into tensors, gradients, and model building.  After completing this
overview you should be able to:

- install and configure a Python environment for deep learning work,
- perform basic array and tensor operations in **NumPy** and **PyTorch**,
- understand how the two libraries relate and interoperate.

The detailed treatments—tensor creation, autograd mechanics, training loops,
etc.—follow in Sections 1.0–1.13.

---

## 1  Setting Up Python

### 1.1  Choosing a Python Distribution

For data analysis and deep learning, the **Anaconda** distribution is widely
recommended:

| Feature | Benefit |
|---|---|
| **Comprehensive** | Ships with 1 500+ data-science packages (NumPy, Pandas, Matplotlib, SciPy, …) |
| **User-friendly** | Includes Jupyter Notebook and the Spyder IDE out of the box |
| **Cross-platform** | Works on Windows, macOS, and Linux |

**Installation steps:**

1. Download the installer from
   [anaconda.com/products/distribution](https://www.anaconda.com/products/distribution).
2. Run the installer (optionally add Anaconda to your `PATH`).
3. Verify with `conda --version` in a terminal.

### 1.2  Installing Python Packages

Anaconda provides two package managers:

```bash
# conda (preferred for Anaconda-hosted packages)
conda install numpy

# pip (for packages outside the Anaconda repository)
pip install seaborn
```

**Best practice:** use `conda` when the package is available in the Anaconda
repository; fall back to `pip` otherwise.

### 1.3  Virtual Environments

A virtual environment isolates a project's dependencies so that packages from
different projects never conflict.

```bash
# create
conda create --name myenv

# activate
conda activate myenv

# install packages inside the environment
conda install numpy pandas matplotlib

# deactivate
conda deactivate
```

### 1.4  Integrated Development Environments (IDEs)

| IDE | Highlights |
|---|---|
| **Jupyter Notebook** | Interactive cell-based execution; inline plots; ideal for exploration |
| **Spyder** | Ships with Anaconda; MATLAB-like variable explorer; integrated debugger |
| **PyCharm** | Intelligent code completion; project navigation; strong library support |

Launch Jupyter from a terminal:

```bash
jupyter notebook
```

### 1.5  Basic Environment Configuration

After installation, a few housekeeping steps make day-to-day work smoother:

- organise notebooks and scripts in a clear directory structure,
- pre-install the libraries you will use most often,
- customise Jupyter settings (theme, default font, autosave interval).

```python
# install the essentials in one go
!pip install numpy pandas matplotlib seaborn
```

---

## 2  Overview of Key Libraries: NumPy and PyTorch

### 2.1  NumPy — Numerical Python

NumPy is the foundation of almost every scientific-computing stack in Python.
Its core data structure is the **ndarray**, an $N$-dimensional array that
supports fast, vectorised arithmetic.

#### 2.1.1  N-dimensional Arrays

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])
print(arr)           # [1 2 3 4 5]
print(arr.dtype)     # int64
```

#### 2.1.2  Element-wise Operations

```python
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
print(arr1 + arr2)   # [5 7 9]
```

#### 2.1.3  Broadcasting

NumPy stretches arrays of compatible shapes so that explicit replication is
unnecessary:

```python
arr = np.array([1, 2, 3])
print(arr + 10)      # [11 12 13]
```

#### 2.1.4  Linear Algebra

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(np.dot(A, B))  # [[19 22]
                      #  [43 50]]

# equivalently
print(A @ B)
```

#### 2.1.5  Data-Type Promotion

When arrays of different dtypes interact, NumPy promotes to the more general
type:

```python
x1 = np.array([1, 2, 3])          # int64
x2 = np.array([1., 2, 3])         # float64
x3 = x1 + x2                      # float64
```

#### 2.1.6  Reshaping

```python
x1 = np.array([1, 2, 3]).reshape((3, 1))          # (3,) → (3, 1)
x2 = np.array([1., 2, 3, 4, 5, 6]).reshape((3, 2))  # (6,) → (3, 2)
x3 = x1 + x2                                       # (3, 2) via broadcasting
```

#### 2.1.7  Random Number Generation

```python
random_array = np.random.rand(3, 3)   # uniform on [0, 1)
print(random_array)
```

#### 2.1.8  Applications

- **Data pre-processing** — reshape, normalise, and clean data before model
  training.
- **Numerical simulation** — run large-scale Monte-Carlo or finite-difference
  experiments.
- **Data storage** — save / load arrays efficiently with `np.save` / `np.load`.

---

### 2.2  PyTorch — Deep Learning Framework

PyTorch is an open-source library developed by Meta AI Research.  It provides a
NumPy-like tensor API with two critical additions: **automatic
differentiation** (autograd) and transparent **GPU acceleration**.

#### 2.2.1  Tensors

```python
import torch

tensor = torch.tensor([1, 2, 3, 4, 5])
print(tensor)
```

Tensors support the same `@` operator for matrix multiplication:

```python
A = torch.tensor([[1, 2], [3, 4]])
B = torch.tensor([[2, 3], [0, 1]])
C = A @ B
print(C)             # tensor([[ 2,  5],
                      #         [ 6, 13]])
```

#### 2.2.2  Data-Type Promotion

PyTorch follows analogous promotion rules, but also allows explicit dtype
selection:

```python
x1 = torch.tensor([1, 2, 3])                        # int64
x2 = torch.tensor([1., 2, 3], dtype=torch.float16)  # float16
x3 = x1 + x2                                        # float16
```

#### 2.2.3  Autograd

Autograd records operations on tensors to build a dynamic computational graph.
Calling `.backward()` then computes gradients via reverse-mode automatic
differentiation:

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
y.backward()
print(x.grad)        # tensor(4.)   ← dy/dx = 2x evaluated at x=2
```

#### 2.2.4  Dynamic Computational Graph

Unlike static-graph frameworks, PyTorch builds the graph on the fly.  This
means standard Python control flow (`if`, `for`, `while`) can vary the
architecture from one forward pass to the next:

```python
x = torch.randn(3, 3)
y = torch.randn(3, 3)
result = x * y       # graph constructed at execution time
```

#### 2.2.5  Neural Network Module (`torch.nn`)

`torch.nn` provides layers, loss functions, and containers for building models:

```python
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)
```

#### 2.2.6  Optimizers and Loss Functions

```python
import torch.optim as optim

model = SimpleNN()
optimizer = optim.SGD(model.parameters(), lr=0.01)
```

PyTorch ships with a wide selection of optimizers (SGD, Adam, AdamW, …) and
loss functions (MSE, Cross-Entropy, …).

#### 2.2.7  GPU Acceleration

Moving tensors to a GPU requires a single call:

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tensor = torch.tensor([1.0, 2.0, 3.0], device=device)
```

#### 2.2.8  Applications

- **Deep learning research** — flexible graph construction encourages rapid
  prototyping.
- **Computer vision** — image classification, object detection, segmentation.
- **Natural language processing** — Hugging Face Transformers, among many
  libraries, builds on PyTorch.
- **Production deployment** — TorchServe, ONNX export, TorchScript.

---

### 2.3  NumPy ↔ PyTorch Interoperability

PyTorch tensors are designed to interoperate seamlessly with NumPy arrays.
Conversion is zero-copy when the tensor lives on the CPU and shares the same
dtype:

```python
import numpy as np
import torch

# NumPy → PyTorch
np_array = np.array([1, 2, 3])
tensor   = torch.from_numpy(np_array)

# PyTorch → NumPy
np_back  = tensor.numpy()
```

!!! note
    Because the conversion is zero-copy, mutating the NumPy array also mutates
    the tensor (and vice-versa).  Use `.clone()` when an independent copy is
    needed.

---

### 2.4  Quick Visualisation with Matplotlib

Even a minimal plot can be useful for sanity-checking data before training:

```python
import matplotlib.pyplot as plt

plt.plot([0, 1, 2], [1, 2, 1])
plt.title("Sanity-check plot")
plt.show()
```

---

## What Comes Next

| Section | Topic |
|---|---|
| **1.0** | PyTorch quickstart (tensors, autograd, modules, training loops) |
| **1.1** | Detailed installation and GPU configuration |
| **1.2** | Tensor creation, dtypes, memory layout |
| **1.3** | Tensor attributes and methods (indexing, broadcasting, linalg, …) |
| **1.4** | Gradients and computational graphs |
| **1.5** | Gradient descent |
| **1.6–1.13** | MLE, linear / logistic / softmax regression, evaluation, datasets, dataloaders |
