# Summation Formulas

Key summation formulas used throughout algorithm analysis:

## Arithmetic Series

$$

\sum_{i=1}^{n} i = \frac{n(n+1)}{2} = \Theta(n^2)

$$

## Geometric Series

$$

\sum_{i=0}^{n} r^i = \frac{r^{n+1}-1}{r-1} \quad (r \neq 1)

$$

## Harmonic Series

$$

H_n = \sum_{i=1}^{n} \frac{1}{i} = \ln n + \gamma + O\left(\frac{1}{n}\right) = \Theta(\log n)

$$

where $\gamma \approx 0.5772$ is the Euler-Mascheroni constant.

## Common Sums in Algorithms

$$

\begin{array}{ll}
\sum_{i=1}^{n} i^2 = \frac{n(n+1)(2n+1)}{6} & = \Theta(n^3) \\
\sum_{i=0}^{\log n} 2^i = 2n - 1 & = \Theta(n) \\
\sum_{i=1}^{n} i \cdot 2^i = (n-1) \cdot 2^{n+1} + 2 & = \Theta(n \cdot 2^n)
\end{array}

$$

```python
def verify_arithmetic_sum(n):
    actual = sum(range(1, n + 1))
    formula = n * (n + 1) // 2
    return actual, formula

def verify_geometric_sum(r, n):
    actual = sum(r**i for i in range(n + 1))
    formula = (r**(n + 1) - 1) / (r - 1) if r != 1 else n + 1
    return actual, formula

def main():
    for n in [10, 100, 1000]:
        a, f = verify_arithmetic_sum(n)
        print(f"Arithmetic sum(1..{n}): {a} = {f}")
    print()
    for r in [2, 3]:
        a, f = verify_geometric_sum(r, 10)
        print(f"Geometric sum(r={r}, n=10): {a} ≈ {f:.0f}")

if __name__ == "__main__":
    main()
```

**Output:**
```
Arithmetic sum(1..10): 55 = 55
Arithmetic sum(1..100): 5050 = 5050
Arithmetic sum(1..1000): 500500 = 500500

Geometric sum(r=2, n=10): 2047 ≈ 2047
Geometric sum(r=3, n=10): 88573 ≈ 88573
```

# Reference

[Introduction to Algorithms (CLRS), Appendix A](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
