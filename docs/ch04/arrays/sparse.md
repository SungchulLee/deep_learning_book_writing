# Sparse Arrays

Many real-world matrices contain mostly zeros. A word-document frequency matrix in NLP may have 100,000 rows and 50,000 columns (5 billion entries), but only a tiny fraction of word-document pairs are nonzero. Storing such a matrix as a dense 2D array wastes memory on billions of zeros. **Sparse arrays** exploit this structure by storing only the nonzero elements and their positions, reducing both memory usage and computation time from $O(mn)$ to $O(\text{nnz})$, where $\text{nnz}$ is the number of nonzero entries.

## Sparsity and When It Matters

The **sparsity** of a matrix $A \in \mathbb{R}^{m \times n}$ is the fraction of entries that are zero:

$$
\text{sparsity} = 1 - \frac{\text{nnz}}{m \cdot n}
$$

A matrix is considered sparse when $\text{nnz} \ll m \cdot n$. Common sources of sparsity include:

- **Graph adjacency matrices**: a graph with $n$ vertices and $e$ edges has $\text{nnz} = 2e$ (undirected) in an $n \times n$ matrix. Most real networks have $e = O(n)$, making the matrix 99.9%+ sparse for large $n$.
- **One-hot encodings**: a vocabulary of 50,000 tokens produces vectors with exactly 1 nonzero entry out of 50,000.
- **Feature matrices in NLP**: bag-of-words or TF-IDF representations are extremely sparse because each document uses only a small fraction of the vocabulary.

## Coordinate Format (COO)

The simplest sparse format stores each nonzero entry as a triplet $(i, j, v)$ where $i$ is the row index, $j$ is the column index, and $v$ is the value.

**Storage:** three arrays of length $\text{nnz}$:

- `row_indices`: $[i_1, i_2, \ldots, i_{\text{nnz}}]$
- `col_indices`: $[j_1, j_2, \ldots, j_{\text{nnz}}]$
- `values`: $[v_1, v_2, \ldots, v_{\text{nnz}}]$

**Space complexity:** $O(\text{nnz})$ -- specifically $3 \cdot \text{nnz}$ numbers.

??? example "COO Representation"

    The matrix

    $$
    A = \begin{pmatrix} 0 & 0 & 3 \\ 4 & 0 & 0 \\ 0 & 5 & 6 \end{pmatrix}
    $$

    has $\text{nnz} = 4$ and is stored as:

    | row | col | value |
    |-----|-----|-------|
    | 0   | 2   | 3     |
    | 1   | 0   | 4     |
    | 2   | 1   | 5     |
    | 2   | 2   | 6     |

**Advantages:** simple to construct, easy to add new entries. **Disadvantage:** accessing a specific element $(i, j)$ requires scanning all entries, giving $O(\text{nnz})$ lookup time.

## Compressed Sparse Row (CSR)

CSR is the most widely used format for sparse matrix arithmetic. It compresses the row indices by replacing them with a pointer array that marks where each row's entries begin.

**Storage:** three arrays:

- `values`: length $\text{nnz}$ -- the nonzero values, stored row by row.
- `col_indices`: length $\text{nnz}$ -- the column index of each nonzero value.
- `row_ptr`: length $m + 1$ -- `row_ptr[i]` is the index into `values` where row $i$ starts. The entries of row $i$ are `values[row_ptr[i] : row_ptr[i+1]]`.

**Space complexity:** $O(\text{nnz} + m)$ -- specifically $2 \cdot \text{nnz} + (m + 1)$ numbers.

??? example "CSR Representation"

    For the same matrix $A$:

    ```
    values     = [3, 4, 5, 6]
    col_indices = [2, 0, 1, 2]
    row_ptr    = [0, 1, 2, 4]
    ```

    - Row 0: `values[0:1]` = `[3]` at column `[2]`
    - Row 1: `values[1:2]` = `[4]` at column `[0]`
    - Row 2: `values[2:4]` = `[5, 6]` at columns `[1, 2]`

**Advantages:** efficient row slicing in $O(1)$, fast matrix-vector multiplication, compact storage. **Disadvantage:** inserting new nonzeros is expensive because it requires shifting arrays.

## Compressed Sparse Column (CSC)

CSC is the column-oriented analog of CSR. It stores entries column by column with a column pointer array.

**Storage:**

- `values`: length $\text{nnz}$
- `row_indices`: length $\text{nnz}$
- `col_ptr`: length $n + 1$

**Space complexity:** $O(\text{nnz} + n)$.

CSC is preferred when column slicing is frequent, such as in certain linear algebra solvers.

## Complexity Comparison

| Operation            | Dense          | COO               | CSR               |
|----------------------|----------------|-------------------|-------------------|
| Storage              | $O(mn)$        | $O(\text{nnz})$   | $O(\text{nnz}+m)$ |
| Access $(i,j)$       | $O(1)$         | $O(\text{nnz})$   | $O(\log d_i)$     |
| Row slice            | $O(n)$         | $O(\text{nnz})$   | $O(1)$ to locate  |
| Matrix-vector mult   | $O(mn)$        | $O(\text{nnz})$   | $O(\text{nnz})$   |
| Insert new nonzero   | $O(1)$         | $O(1)$ amortized  | $O(\text{nnz})$   |

Here $d_i$ is the number of nonzeros in row $i$ (binary search within the row's column indices).

!!! tip "When to Use Sparse Formats"

    Sparse formats save memory and computation only when $\text{nnz} \ll mn$. A rough rule of thumb: if more than 10-20% of entries are nonzero, dense storage is often faster because it avoids the overhead of index bookkeeping and benefits from optimized BLAS routines and cache locality.

## Python Demonstration

```python
"""Demonstrate sparse matrix formats using SciPy."""

import numpy as np
from scipy import sparse

# === Create a sparse matrix ===
dense = np.array([[0, 0, 3],
                  [4, 0, 0],
                  [0, 5, 6]])

# === COO format ===
coo = sparse.coo_matrix(dense)
print("COO format:")
print(f"  row:  {coo.row}")
print(f"  col:  {coo.col}")
print(f"  data: {coo.data}")
print(f"  nnz:  {coo.nnz}")

# === CSR format ===
csr = sparse.csr_matrix(dense)
print("\nCSR format:")
print(f"  data:    {csr.data}")
print(f"  indices: {csr.indices}")
print(f"  indptr:  {csr.indptr}")

# === Space savings ===
n = 10000
big_sparse = sparse.random(n, n, density=0.001, format='csr')
dense_bytes = n * n * 8  # float64
sparse_bytes = big_sparse.data.nbytes + big_sparse.indices.nbytes + big_sparse.indptr.nbytes
print(f"\n{n}x{n} matrix with 0.1% density:")
print(f"  Dense:  {dense_bytes / 1e6:.1f} MB")
print(f"  Sparse: {sparse_bytes / 1e6:.2f} MB")
print(f"  Ratio:  {dense_bytes / sparse_bytes:.0f}x")
```

**Output:**
```
COO format:
  row:  [0 1 2 2]
  col:  [2 0 1 2]
  data: [3 4 5 6]
  nnz:  4

CSR format:
  data:    [3 4 5 6]
  indices: [2 0 1 2]
  indptr:  [0 1 2 4]

10000x10000 matrix with 0.1% density:
  Dense:  800.0 MB
  Sparse: 1.28 MB
  Ratio:  625x
```

## Reference

- [Introduction to Algorithms (CLRS), Chapter 10](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
