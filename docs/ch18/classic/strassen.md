# Strassen's Matrix Multiplication

Multiplying two $n \times n$ matrices using the standard algorithm requires $O(n^3)$ scalar multiplications. In 1969, Volker Strassen showed that by cleverly combining seven recursive multiplications of $n/2 \times n/2$ submatrices -- instead of the eight required by the standard divide-and-conquer approach -- the complexity drops to $O(n^{\log_2 7}) \approx O(n^{2.807})$. Like [Karatsuba multiplication](karatsuba.md) for integers, the key insight is reducing the number of recursive multiplications at each level.

## Standard Matrix Multiplication

The product $C = A \cdot B$ of two $n \times n$ matrices is defined by

$$
C_{ij} = \sum_{k=1}^{n} A_{ik} \cdot B_{kj}
$$

Computing each of the $n^2$ entries requires $n$ multiplications and $n - 1$ additions, giving $\Theta(n^3)$ total work.

## Naive Divide and Conquer

Partition each $n \times n$ matrix into four $n/2 \times n/2$ submatrices:

$$
A = \begin{pmatrix} A_{11} & A_{12} \\ A_{21} & A_{22} \end{pmatrix}, \quad B = \begin{pmatrix} B_{11} & B_{12} \\ B_{21} & B_{22} \end{pmatrix}, \quad C = \begin{pmatrix} C_{11} & C_{12} \\ C_{21} & C_{22} \end{pmatrix}
$$

The block multiplication formulas are

$$
C_{11} = A_{11}B_{11} + A_{12}B_{21}
$$

$$
C_{12} = A_{11}B_{12} + A_{12}B_{22}
$$

$$
C_{21} = A_{21}B_{11} + A_{22}B_{21}
$$

$$
C_{22} = A_{21}B_{12} + A_{22}B_{22}
$$

This requires **8** multiplications of $n/2 \times n/2$ matrices plus **4** additions of $n/2 \times n/2$ matrices. The recurrence is

$$
T(n) = 8T\!\left(\frac{n}{2}\right) + \Theta(n^2)
$$

By the Master Theorem ($a = 8$, $b = 2$, $\log_2 8 = 3$, $f(n) = \Theta(n^2)$, case 1):

$$
T(n) = \Theta(n^3)
$$

No improvement over the standard algorithm.

## Strassen's Algorithm

Strassen reduces the number of recursive multiplications from 8 to **7** by defining the following seven products:

$$
M_1 = (A_{11} + A_{22})(B_{11} + B_{22})
$$

$$
M_2 = (A_{21} + A_{22}) B_{11}
$$

$$
M_3 = A_{11} (B_{12} - B_{22})
$$

$$
M_4 = A_{22} (B_{21} - B_{11})
$$

$$
M_5 = (A_{11} + A_{12}) B_{22}
$$

$$
M_6 = (A_{21} - A_{11})(B_{11} + B_{12})
$$

$$
M_7 = (A_{12} - A_{22})(B_{21} + B_{22})
$$

The result submatrices are then computed as

$$
C_{11} = M_1 + M_4 - M_5 + M_7
$$

$$
C_{12} = M_3 + M_5
$$

$$
C_{21} = M_2 + M_4
$$

$$
C_{22} = M_1 - M_2 + M_3 + M_6
$$

### Verification of Correctness

We verify $C_{11}$ as a representative example.

$$
C_{11} = M_1 + M_4 - M_5 + M_7
$$

Expanding:

$$
M_1 = A_{11}B_{11} + A_{11}B_{22} + A_{22}B_{11} + A_{22}B_{22}
$$

$$
M_4 = A_{22}B_{21} - A_{22}B_{11}
$$

$$
M_5 = A_{11}B_{22} + A_{12}B_{22}
$$

$$
M_7 = A_{12}B_{21} + A_{12}B_{22} - A_{22}B_{21} - A_{22}B_{22}
$$

Summing $M_1 + M_4 - M_5 + M_7$:

$$
= A_{11}B_{11} + \cancel{A_{11}B_{22}} + \cancel{A_{22}B_{11}} + \cancel{A_{22}B_{22}} + \cancel{A_{22}B_{21}} - \cancel{A_{22}B_{11}} - \cancel{A_{11}B_{22}} - \cancel{A_{12}B_{22}} + A_{12}B_{21} + \cancel{A_{12}B_{22}} - \cancel{A_{22}B_{21}} - \cancel{A_{22}B_{22}}
$$

$$
= A_{11}B_{11} + A_{12}B_{21}
$$

This matches the definition of $C_{11}$. The other three entries can be verified similarly. $\square$

## Complexity Analysis

### Recurrence

Strassen's algorithm performs 7 recursive multiplications on $n/2 \times n/2$ matrices, plus $O(n^2)$ work for the 18 matrix additions and subtractions:

$$
T(n) = 7T\!\left(\frac{n}{2}\right) + \Theta(n^2)
$$

### Solving via the Master Theorem

With $a = 7$, $b = 2$, $f(n) = \Theta(n^2)$:

$$
\log_b a = \log_2 7 \approx 2.807
$$

Since $f(n) = \Theta(n^2) = O(n^{\log_2 7 - \epsilon})$ for $\epsilon \approx 0.807$, this is case 1:

$$
T(n) = \Theta(n^{\log_2 7}) \approx \Theta(n^{2.807})
$$

### Comparison

| Algorithm | Multiplications | Additions | Time |
|---|---|---|---|
| Standard | $n^3$ | $n^3 - n^2$ | $\Theta(n^3)$ |
| Naive D&C | 8 recursive | 4 matrix adds | $\Theta(n^3)$ |
| Strassen | 7 recursive | 18 matrix adds | $\Theta(n^{2.807})$ |

For $n = 1024$, the standard method performs $\sim 10^9$ multiplications, while Strassen requires $\sim 10^{8.58} \approx 3.8 \times 10^8$ -- roughly a 2.8x speedup at this size.

## Practical Considerations

!!! tip "Crossover Point"
    Strassen's algorithm has larger constant factors than the standard algorithm due to the 18 additions (vs. 4) and the overhead of recursive partitioning. In practice, implementations switch to the standard algorithm when $n$ falls below a crossover point, typically around $n = 32$ to $n = 128$, depending on the hardware.

!!! warning "Numerical Stability"
    Strassen's algorithm is less numerically stable than the standard algorithm because it involves subtractions that can cause cancellation. For applications requiring high numerical precision, the standard $O(n^3)$ algorithm or algorithms with better stability properties may be preferred.

### Memory Overhead

The naive implementation of Strassen's algorithm creates many temporary matrices at each recursive level, leading to significant memory overhead. Careful implementation can reduce this to $O(n^2)$ additional space by reusing buffers.

## Beyond Strassen

Strassen's result sparked the search for faster matrix multiplication algorithms:

| Algorithm | Year | Exponent $\omega$ |
|---|---|---|
| Standard | -- | 3.000 |
| Strassen | 1969 | 2.807 |
| Coppersmith-Winograd | 1990 | 2.376 |
| Alman-Vassilevska Williams | 2021 | 2.373 |

The theoretical lower bound is $\omega \ge 2$ (since the output has $n^2$ entries). Whether $\omega = 2$ is achievable remains one of the major open problems in theoretical computer science.

## Summary

Strassen's algorithm reduces matrix multiplication from $\Theta(n^3)$ to $\Theta(n^{2.807})$ by replacing 8 recursive multiplications with 7, at the cost of more additions. The approach mirrors Karatsuba's strategy for integer multiplication: reducing the number of expensive recursive operations by one, even at the expense of more cheap operations (additions), yields an asymptotic improvement. The resulting recurrence $T(n) = 7T(n/2) + \Theta(n^2)$ is solved by the Master Theorem (case 1).

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 4. MIT Press.
- Strassen, V. (1969). Gaussian elimination is not optimal. *Numerische Mathematik*, 13(4), 354-356.
