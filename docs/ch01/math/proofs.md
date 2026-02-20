# Proof Techniques

Mathematical proofs establish the correctness and complexity of algorithms.

## Common Techniques

$$
\begin{array}{ll}
\text{Direct Proof} & \text{Assume premises, derive conclusion} \\
\text{Proof by Contradiction} & \text{Assume negation, derive contradiction} \\
\text{Proof by Induction} & \text{Base case + inductive step} \\
\text{Proof by Construction} & \text{Build an example} \\
\text{Proof by Contrapositive} & \text{Prove } \neg Q \Rightarrow \neg P \text{ instead of } P \Rightarrow Q
\end{array}
$$

## Example: Direct Proof

**Claim**: The sum of two even numbers is even.

**Proof**: Let $a = 2m$ and $b = 2n$. Then $a + b = 2m + 2n = 2(m+n)$, which is even. $\square$

```python
def is_even(n):
    return n % 2 == 0


def verify_sum_of_evens():
    """Verify: sum of two evens is always even."""
    for a in range(0, 20, 2):
        for b in range(0, 20, 2):
            assert is_even(a + b), f"Failed: {a} + {b} = {a+b}"
    print("Verified: sum of two even numbers is always even (tested 0-18)")


def main():
    verify_sum_of_evens()


if __name__ == "__main__":
    main()
```

**Output:**
```
Verified: sum of two even numbers is always even (tested 0-18)
```


# Reference

[How to Prove It (Velleman)](https://www.amazon.com/How-Prove-Structured-Approach-2nd/dp/0521675995)
