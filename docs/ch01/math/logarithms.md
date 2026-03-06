# Logarithms

Logarithms appear frequently in algorithm analysis, especially in divide-and-conquer algorithms.

## Key Properties

$$

\begin{array}{ll}
\log_b(xy) = \log_b x + \log_b y \\
\log_b(x/y) = \log_b x - \log_b y \\
\log_b(x^n) = n \log_b x \\
\log_b x = \frac{\log_a x}{\log_a b} & \text{(change of base)}
\end{array}

$$

## Common Logarithms in CS

$$

\begin{array}{ll}
\lg n = \log_2 n & \text{Binary logarithm (most common in CS)} \\
\ln n = \log_e n & \text{Natural logarithm} \\
\lg \lg n = \log_2(\log_2 n) & \text{Iterated logarithm}
\end{array}

$$

## Why log2(n)?

Halving $n$ repeatedly until reaching 1 takes $\log_2 n$ steps — this is the depth of binary search, balanced BSTs, etc.

$$

n \rightarrow \frac{n}{2} \rightarrow \frac{n}{4} \rightarrow \cdots \rightarrow 1 \quad (\log_2 n \text{ steps})

$$

```python
import math

def main():
    for n in [2, 8, 64, 1024, 1_000_000]:
        print(f"n = {n:>10,}  log2 = {math.log2(n):>8.2f}  ln = {math.log(n):>8.2f}  log10 = {math.log10(n):>6.2f}")

if __name__ == "__main__":
    main()
```

**Output:**
```
n =          2  log2 =     1.00  ln =     0.69  log10 =  0.30
n =          8  log2 =     3.00  ln =     2.08  log10 =  0.90
n =         64  log2 =     6.00  ln =     4.16  log10 =  1.81
n =      1,024  log2 =    10.00  ln =     6.93  log10 =  3.01
n =  1,000,000  log2 =    19.93  ln =    13.82  log10 =  6.00
```

# Reference

[Introduction to Algorithms (CLRS), Section 3.2](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
