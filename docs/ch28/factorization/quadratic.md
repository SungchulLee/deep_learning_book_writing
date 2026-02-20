# Quadratic Sieve

The quadratic sieve factors integers faster than trial division for large numbers.

```python
def sieve(n):
    is_prime = [True]*(n+1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5)+1):
        if is_prime[i]:
            for j in range(i*i, n+1, i): is_prime[j] = False
    return [i for i in range(n+1) if is_prime[i]]

print(f"Primes up to 50: {sieve(50)}")
```

**Output:**
```
Primes up to 50: [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
```


# Reference

[Introduction to Algorithms (CLRS), Chapter 31](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
