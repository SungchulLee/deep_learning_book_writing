# Pollard's Rho Algorithm

Trial division finds small factors quickly, but its $O(\sqrt{n})$ worst case
becomes impractical for large composites.  Pollard's rho algorithm exploits
the birthday paradox to find a non-trivial factor of $n$ in expected
$O(n^{1/4})$ arithmetic operations, making it one of the most efficient
methods for factoring numbers with moderate-sized factors.

## Core Idea

Let $p$ be an unknown prime factor of $n$.  If we generate a pseudo-random
sequence $x_0, x_1, x_2, \dots$ modulo $n$, the values modulo $p$ cycle much
sooner than the values modulo $n$.  By the birthday paradox, a collision
$x_i \equiv x_j \pmod{p}$ is expected after roughly $O(\sqrt{p})$ steps.
When this happens, $\gcd(x_i - x_j, n)$ reveals the factor $p$.

The name "rho" comes from the shape of the sequence when drawn as a graph:
a tail leading into a cycle, resembling the Greek letter $\rho$.

## The Iteration Function

We use a polynomial map $f(x) = x^2 + c \pmod{n}$ for a fixed constant
$c \notin \{0, -2\}$.  Starting from $x_0$, define

$$
x_{i+1} = f(x_i) = x_i^2 + c \pmod{n}
$$

The sequence is deterministic but behaves pseudo-randomly modulo any fixed
prime factor $p$ of $n$.

## Floyd's Cycle Detection

Rather than storing all previous values, Floyd's tortoise-and-hare method
uses two pointers:

- **Tortoise**: advances one step at a time, $x_i$.
- **Hare**: advances two steps at a time, $x_{2i}$.

At each step, compute $d = \gcd(|x_i - x_{2i}|, n)$.  If $1 < d < n$,
then $d$ is a non-trivial factor.

## Algorithm

```python
"""
Pollard's rho algorithm for integer factorization.

Expected time: O(n^{1/4}) arithmetic operations.
"""

import math
import random


# === Pollard's Rho ===
def pollard_rho(n: int) -> int:
    """Return a non-trivial factor of n, or n if n is prime."""
    if n % 2 == 0:
        return 2
    while True:
        x = random.randint(2, n - 1)
        y = x
        c = random.randint(1, n - 1)
        d = 1
        while d == 1:
            x = (x * x + c) % n          # tortoise
            y = (y * y + c) % n           # hare step 1
            y = (y * y + c) % n           # hare step 2
            d = math.gcd(abs(x - y), n)
        if d != n:
            return d
        # d == n means cycle without finding factor; retry


# === Full Factorization ===
def factorize(n: int) -> list[int]:
    """Return the complete prime factorization of n."""
    if n <= 1:
        return []
    factors = []
    stack = [n]
    while stack:
        k = stack.pop()
        if k == 1:
            continue
        if is_prime_miller_rabin(k):
            factors.append(k)
        else:
            d = pollard_rho(k)
            stack.append(d)
            stack.append(k // d)
    return sorted(factors)


def is_prime_miller_rabin(n: int, rounds: int = 20) -> bool:
    """Miller-Rabin primality test."""
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for _ in range(rounds):
        a = random.randint(2, n - 2)
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


# === Example ===
if __name__ == "__main__":
    n = 8051
    print(f"n = {n}")
    factor = pollard_rho(n)
    print(f"Non-trivial factor: {factor}")
    print(f"Full factorization: {factorize(n)}")
```

## Complexity Analysis

Let $p$ be the smallest prime factor of $n$.

- **Expected collisions.**  By the birthday paradox, a collision modulo $p$
  occurs after $O(\sqrt{p})$ iterations.
- **Since $p \le \sqrt{n}$,** the expected number of iterations is
  $O(n^{1/4})$.
- **Each iteration** performs a constant number of modular multiplications
  and one GCD computation, each costing $O(\log^2 n)$ with standard
  arithmetic.

Total expected time: $O(n^{1/4} \log^2 n)$ bit operations.

!!! tip "Brent's Improvement"
    Brent's variant replaces Floyd's cycle detection with a different
    advancement schedule.  It finds factors about 24% faster in practice
    by reducing the number of GCD computations.

## Failure and Retry

When $d = \gcd(|x_i - x_{2i}|, n) = n$, the tortoise and hare have
collided modulo $n$ itself, yielding a trivial factor.  The fix is simple:
restart with a different random $c$ or $x_0$.  The probability of needing
many restarts is low.

## Practical Considerations

| Aspect | Detail |
|---|---|
| Best for | Numbers up to ~60 digits with factors up to ~30 digits |
| Combines with | Miller-Rabin (test primality before factoring) |
| Not suitable for | Semiprimes with two large factors (use GNFS instead) |
| Parallelizable | Yes --- run independent instances with different $c$ values |

!!! warning "Choice of c"
    Avoid $c = 0$ (sequence degenerates to $x^{2^k}$) and $c = -2$ (the
    iteration has algebraic structure that prevents finding factors).

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction
  to Algorithms* (CLRS), Chapter 31.
- Pollard, J. M. "A Monte Carlo method for factorization." *BIT Numerical
  Mathematics*, 15(3), 1975.
