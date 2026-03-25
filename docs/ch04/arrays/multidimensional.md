# Multi-Dimensional Arrays

Many computational tasks involve data organized along more than one axis: a grayscale image is a 2D grid of pixel intensities, a color image adds a channel dimension, and a training batch adds yet another. Multi-dimensional arrays store such data in a single contiguous block of memory, just like a 1D array, but provide access through multiple indices. The key design decision is how to flatten the multi-dimensional structure into a linear sequence of memory addresses, and this choice -- row-major vs. column-major order -- has significant consequences for performance.

## Row-Major Order (C Order)

In **row-major order**, elements of the same row are stored contiguously. For a 2D array with $m$ rows and $n$ columns, the element at position $(i, j)$ is stored at the linear offset

$$
\text{offset}(i, j) = i \cdot n + j
$$

and its memory address is $b + (i \cdot n + j) \cdot w$, where $b$ is the base address and $w$ is the element size in bytes.

??? example "Row-Major Layout for a 3 x 4 Matrix"

    Consider the matrix

    $$
    A = \begin{pmatrix} 1 & 2 & 3 & 4 \\ 5 & 6 & 7 & 8 \\ 9 & 10 & 11 & 12 \end{pmatrix}
    $$

    In row-major order, memory contains: `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]`.

    | Index $(i,j)$ | Offset $= i \cdot 4 + j$ | Value |
    |---------------|--------------------------|-------|
    | $(0, 0)$      | 0                        | 1     |
    | $(0, 3)$      | 3                        | 4     |
    | $(1, 0)$      | 4                        | 5     |
    | $(2, 2)$      | 10                       | 11    |

Row-major order is used by C, C++, Python (NumPy default), and PyTorch.

## Column-Major Order (Fortran Order)

In **column-major order**, elements of the same column are stored contiguously. The linear offset for an $m \times n$ array becomes

$$
\text{offset}(i, j) = j \cdot m + i
$$

The same matrix $A$ in column-major order is stored as: `[1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]`.

Column-major order is used by Fortran, MATLAB, R, and Julia.

## General d-Dimensional Address Computation

For a $d$-dimensional array with shape $(n_0, n_1, \ldots, n_{d-1})$, the element at index $(i_0, i_1, \ldots, i_{d-1})$ has a linear offset computed using **strides**. The stride $s_k$ for dimension $k$ gives the number of elements to skip when incrementing index $k$ by 1.

**Row-major strides:**

$$
s_k = \prod_{j=k+1}^{d-1} n_j \qquad \text{with } s_{d-1} = 1
$$

**Column-major strides:**

$$
s_k = \prod_{j=0}^{k-1} n_j \qquad \text{with } s_0 = 1
$$

The linear offset in either case is

$$
\text{offset}(i_0, i_1, \ldots, i_{d-1}) = \sum_{k=0}^{d-1} i_k \cdot s_k
$$

??? example "Strides for a 3D Tensor"

    A tensor of shape $(2, 3, 4)$ in row-major order has strides:

    - $s_2 = 1$
    - $s_1 = n_2 = 4$
    - $s_0 = n_1 \cdot n_2 = 3 \cdot 4 = 12$

    So element $(1, 2, 3)$ has offset $1 \cdot 12 + 2 \cdot 4 + 3 \cdot 1 = 23$.

## Performance Implications

The choice of memory layout determines which access patterns are cache-friendly. Traversing along the contiguous dimension accesses sequential memory addresses, which exploits spatial locality and results in few cache misses. Traversing along a non-contiguous dimension causes a cache miss every few elements.

!!! warning "Traversal Order Matters"

    For a row-major $m \times n$ matrix, iterating as `for i in range(m): for j in range(n)` (row by row) accesses memory sequentially. The reversed order `for j in range(n): for i in range(m)` jumps by $n$ elements at each step, which can be 10x slower for large arrays due to cache misses.

## NumPy and PyTorch Demonstration

```python
"""Demonstrate multi-dimensional array memory layout with NumPy."""

import numpy as np

# === Row-major (C order) vs Column-major (Fortran order) ===
a = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12]])

print(f"Shape: {a.shape}")
print(f"Strides (bytes): {a.strides}")
print(f"Strides (elements): {tuple(s // a.itemsize for s in a.strides)}")
print(f"Row-major flat: {a.ravel(order='C').tolist()}")
print(f"Col-major flat: {a.ravel(order='F').tolist()}")

# === 3D tensor strides ===
t = np.zeros((2, 3, 4), dtype=np.float32)
print(f"\n3D shape: {t.shape}")
print(f"3D strides (elements): {tuple(s // t.itemsize for s in t.strides)}")

# === Contiguous checks ===
print(f"\nC-contiguous: {a.flags['C_CONTIGUOUS']}")
col_a = np.asfortranarray(a)
print(f"F-contiguous: {col_a.flags['F_CONTIGUOUS']}")
print(f"F-order strides (elements): {tuple(s // col_a.itemsize for s in col_a.strides)}")
```

**Output:**
```
Shape: (3, 4)
Strides (bytes): (32, 8)
Strides (elements): (4, 1)
Row-major flat: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
Col-major flat: [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]

3D shape: (2, 3, 4)
3D strides (elements): (12, 4, 1)

C-contiguous: True
F-contiguous: True
F-order strides (elements): (1, 3, 9)
```

The strides output confirms the formulas: for the $(3, 4)$ array in C order, the row stride is 4 and the column stride is 1, meaning moving one row jumps over 4 elements while moving one column jumps by 1.

## Connection to Deep Learning Tensors

PyTorch tensors are multi-dimensional arrays with the stride-based addressing described above. Operations like `transpose`, `view`, and `permute` modify strides without copying data, creating different "views" of the same memory block. Understanding strides explains why some operations require a `contiguous()` call before reshaping and why certain memory layouts lead to faster GPU kernel execution. Chapter 2 covers tensor strides and memory layout in detail.

## Reference

- [Introduction to Algorithms (CLRS), Chapter 10](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
