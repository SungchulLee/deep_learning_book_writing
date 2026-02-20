# Induction

**Mathematical induction** proves that a property $P(n)$ holds for all natural numbers $n \geq n_0$.

## Steps

1. **Base case**: Prove $P(n_0)$
2. **Inductive hypothesis**: Assume $P(k)$ holds for some $k \geq n_0$
3. **Inductive step**: Prove $P(k+1)$ using the hypothesis

## Example

**Claim**: $\sum_{i=1}^{n} i = \frac{n(n+1)}{2}$

**Proof**:
- **Base**: $P(1)$: $1 = \frac{1 \cdot 2}{2} = 1$ ✓
- **Assume** $P(k)$: $\sum_{i=1}^{k} i = \frac{k(k+1)}{2}$
- **Prove** $P(k+1)$:

$$
\sum_{i=1}^{k+1} i = \frac{k(k+1)}{2} + (k+1) = \frac{k(k+1) + 2(k+1)}{2} = \frac{(k+1)(k+2)}{2} \quad \square
$$

```python
def sum_formula(n):
    return n * (n + 1) // 2


def verify_induction(max_n=20):
    # Base case
    assert sum_formula(1) == 1, "Base case failed"
    # Inductive verification
    for k in range(1, max_n):
        lhs = sum_formula(k) + (k + 1)
        rhs = sum_formula(k + 1)
        assert lhs == rhs, f"Inductive step failed at k={k}"
    print(f"Induction verified for n=1..{max_n}")


def main():
    verify_induction()


if __name__ == "__main__":
    main()
```

**Output:**
```
Induction verified for n=1..20
```


# Reference

[Introduction to Algorithms (CLRS), Appendix A](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
