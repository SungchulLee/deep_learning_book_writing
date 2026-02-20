# Contradiction

**Proof by contradiction** assumes the negation of the claim and derives a logical impossibility.

## Structure

1. Assume $\neg P$ (the claim is false)
2. Derive a contradiction
3. Conclude $P$ must be true

## Classic Example: Irrationality of $\sqrt{2}$

**Claim**: $\sqrt{2}$ is irrational.

**Proof**: Assume $\sqrt{2} = p/q$ where $p, q$ are integers with no common factors. Then $2q^2 = p^2$, so $p^2$ is even, hence $p$ is even. Write $p = 2k$. Then $2q^2 = 4k^2$, so $q^2 = 2k^2$, meaning $q$ is also even. But this contradicts our assumption that $p, q$ have no common factors. $\square$

## Example in Algorithms: Infinite Primes

**Claim**: There are infinitely many primes.

```python
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True


def euclid_proof_demonstration(primes):
    """Show Euclid's proof idea: product of known primes + 1."""
    product = 1
    for p in primes:
        product *= p
    candidate = product + 1
    return candidate, is_prime(candidate)


def main():
    primes = [2, 3, 5, 7, 11, 13]
    candidate, prime = euclid_proof_demonstration(primes)
    print(f"Known primes: {primes}")
    print(f"Product + 1 = {candidate}")
    print(f"Is prime: {prime}")
    if not prime:
        # Find a prime factor not in our list
        for i in range(2, candidate):
            if candidate % i == 0 and is_prime(i):
                print(f"Has prime factor {i} not in original list: {i not in primes}")
                break


if __name__ == "__main__":
    main()
```

**Output:**
```
Known primes: [2, 3, 5, 7, 11, 13]
Product + 1 = 30031
Is prime: False
Has prime factor 59 not in original list: True
```


# Reference

[How to Prove It (Velleman)](https://www.amazon.com/How-Prove-Structured-Approach-2nd/dp/0521675995)
