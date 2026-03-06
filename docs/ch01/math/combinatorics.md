# Combinatorics

Combinatorics counts the number of ways to arrange or select objects — essential for analyzing algorithm complexity.

## Key Formulas

$$

\begin{array}{ll}
\text{Permutations:} & P(n, k) = \frac{n!}{(n-k)!} \\
\text{Combinations:} & \binom{n}{k} = \frac{n!}{k!(n-k)!} \\
\text{Arrangements with repetition:} & n^k \\
\text{Multiset coefficient:} & \binom{n+k-1}{k}
\end{array}

$$

## Binomial Theorem

$$

(x + y)^n = \sum_{k=0}^{n} \binom{n}{k} x^{n-k} y^k

$$

```python
from math import factorial, comb

def permutations(n, k):
    return factorial(n) // factorial(n - k)

def main():
    print("Combinations C(n, k):")
    for n in range(6):
        row = [comb(n, k) for k in range(n + 1)]
        print(f"  n={n}: {row}")
    print()
    print(f"P(5,3) = {permutations(5, 3)}")
    print(f"C(5,3) = {comb(5, 3)}")
    print(f"5^3 = {5**3}  (arrangements with repetition)")

if __name__ == "__main__":
    main()
```

**Output:**
```
Combinations C(n, k):
  n=0: [1]
  n=1: [1, 1]
  n=2: [1, 2, 1]
  n=3: [1, 3, 3, 1]
  n=4: [1, 4, 6, 4, 1]
  n=5: [1, 5, 10, 10, 5, 1]

P(5,3) = 60
C(5,3) = 10
5^3 = 125  (arrangements with repetition)
```

# Reference

[Concrete Mathematics (Graham, Knuth, Patashnik)](https://www.amazon.com/Concrete-Mathematics-Foundation-Computer-Science/dp/0201558025)
