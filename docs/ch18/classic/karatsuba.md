# Karatsuba Multiplication

Multiplying two $n$-digit numbers using the grade-school algorithm requires $O(n^2)$ single-digit multiplications. In 1960, Anatoly Karatsuba discovered that a clever algebraic rearrangement reduces the number of recursive multiplications from four to three, yielding an $O(n^{\log_2 3}) \approx O(n^{1.585})$ algorithm. This was the first multiplication algorithm to break the $O(n^2)$ barrier and remains one of the most elegant examples of divide and conquer.

## The Grade-School Algorithm

To multiply two $n$-digit numbers $x$ and $y$, the standard method computes every pair of digits and sums the partial products. This requires $n^2$ single-digit multiplications and $O(n)$ additions per partial product, giving $O(n^2)$ total work.

## Karatsuba's Insight

Split each $n$-digit number into two halves of $n/2$ digits. Let $m = \lfloor n/2 \rfloor$, and write

$$
x = x_1 \cdot 10^m + x_0, \qquad y = y_1 \cdot 10^m + y_0
$$

where $x_1, x_0, y_1, y_0$ are at most $\lceil n/2 \rceil$-digit numbers.

### Naive Splitting

The product $x \cdot y$ expands as

$$
x \cdot y = x_1 y_1 \cdot 10^{2m} + (x_1 y_0 + x_0 y_1) \cdot 10^m + x_0 y_0
$$

This requires **four** multiplications of $n/2$-digit numbers: $x_1 y_1$, $x_1 y_0$, $x_0 y_1$, and $x_0 y_0$. The resulting recurrence is

$$
T(n) = 4T\!\left(\frac{n}{2}\right) + O(n)
$$

which solves to $T(n) = O(n^2)$ -- no improvement over grade school.

### The Key Trick

Karatsuba observed that the middle coefficient $x_1 y_0 + x_0 y_1$ can be computed using only **one** additional multiplication instead of two. Define three products:

$$
p_1 = x_1 \cdot y_1, \qquad p_2 = x_0 \cdot y_0, \qquad p_3 = (x_1 + x_0)(y_1 + y_0)
$$

Then

$$
x_1 y_0 + x_0 y_1 = p_3 - p_1 - p_2
$$

**Proof.** Expanding $p_3$:

$$
p_3 = (x_1 + x_0)(y_1 + y_0) = x_1 y_1 + x_1 y_0 + x_0 y_1 + x_0 y_0 = p_1 + (x_1 y_0 + x_0 y_1) + p_2
$$

Rearranging gives $x_1 y_0 + x_0 y_1 = p_3 - p_1 - p_2$. $\square$

The full product is therefore

$$
x \cdot y = p_1 \cdot 10^{2m} + (p_3 - p_1 - p_2) \cdot 10^m + p_2
$$

This requires only **three** multiplications ($p_1, p_2, p_3$) of roughly $n/2$-digit numbers, plus $O(n)$ additions and shifts.

## Algorithm

```
KARATSUBA(x, y, n):
    if n <= threshold:
        return x * y  (grade-school multiplication)

    m = floor(n / 2)
    x1 = x / 10^m        (high-order digits)
    x0 = x mod 10^m      (low-order digits)
    y1 = y / 10^m
    y0 = y mod 10^m

    p1 = KARATSUBA(x1, y1, ceil(n/2))
    p2 = KARATSUBA(x0, y0, ceil(n/2))
    p3 = KARATSUBA(x1 + x0, y1 + y0, ceil(n/2) + 1)

    return p1 * 10^(2m) + (p3 - p1 - p2) * 10^m + p2
```

### Python Implementation

```python
def karatsuba(x, y):
    """
    Multiply two non-negative integers using Karatsuba's algorithm.

    Parameters
    ----------
    x : int
        First non-negative integer.
    y : int
        Second non-negative integer.

    Returns
    -------
    int
        The product x * y.
    """
    # Base case: single-digit multiplication
    if x < 10 or y < 10:
        return x * y

    # Determine the number of digits
    n = max(len(str(x)), len(str(y)))
    m = n // 2

    # Split x and y
    power = 10 ** m
    x1, x0 = x // power, x % power
    y1, y0 = y // power, y % power

    # Three recursive multiplications
    p1 = karatsuba(x1, y1)
    p2 = karatsuba(x0, y0)
    p3 = karatsuba(x1 + x0, y1 + y0)

    # Combine
    return p1 * (10 ** (2 * m)) + (p3 - p1 - p2) * (10 ** m) + p2
```

## Complexity Analysis

### Recurrence

The algorithm makes 3 recursive calls on numbers of roughly $n/2$ digits, plus $O(n)$ work for addition, subtraction, and shifting:

$$
T(n) = 3T\!\left(\frac{n}{2}\right) + O(n)
$$

### Solving via the Master Theorem

With $a = 3$, $b = 2$, and $f(n) = O(n)$:

$$
\log_b a = \log_2 3 \approx 1.585
$$

Since $f(n) = O(n) = O(n^{\log_2 3 - \epsilon})$ for $\epsilon \approx 0.585$, this is case 1:

$$
T(n) = \Theta(n^{\log_2 3}) \approx \Theta(n^{1.585})
$$

### Improvement Over Grade School

| Method | Multiplications | Time |
|---|---|---|
| Grade school | $n^2$ | $O(n^2)$ |
| Karatsuba | $n^{\log_2 3}$ | $O(n^{1.585})$ |

For $n = 1000$ digits, grade school requires $\sim 10^6$ operations; Karatsuba requires $\sim 10^{4.76} \approx 57{,}500$ -- a 17x speedup.

## Correctness

Correctness follows from the algebraic identity

$$
x \cdot y = p_1 \cdot 10^{2m} + (p_3 - p_1 - p_2) \cdot 10^m + p_2
$$

and induction on $n$. The base case (direct multiplication for small $n$) is correct by definition. For the inductive step, assume Karatsuba correctly computes products of numbers with fewer than $n$ digits. Then $p_1$, $p_2$, and $p_3$ are correct, and the formula above gives the correct product. $\square$

## Practical Considerations

!!! tip "Base Case Threshold"
    In practice, Karatsuba is slower than grade-school multiplication for small numbers due to the overhead of additions, subtractions, and recursive calls. Implementations typically switch to the grade-school method when $n$ drops below a threshold (e.g., $n \le 64$ digits).

!!! note "Beyond Karatsuba"
    Karatsuba's approach generalizes: Toom-Cook splits numbers into 3 or more pieces, further reducing the exponent. The current asymptotically fastest known algorithm (Harvey-van der Hoeven, 2019) achieves $O(n \log n)$ for integer multiplication, though Karatsuba remains practical for moderately large numbers.

## Worked Example

Compute $1234 \times 5678$ using Karatsuba.

Set $m = 2$, so $10^m = 100$:

- $x_1 = 12$, $x_0 = 34$
- $y_1 = 56$, $y_0 = 78$

Three multiplications:

- $p_1 = 12 \times 56 = 672$
- $p_2 = 34 \times 78 = 2652$
- $p_3 = (12 + 34) \times (56 + 78) = 46 \times 134 = 6164$

Combine:

$$
1234 \times 5678 = 672 \times 10000 + (6164 - 672 - 2652) \times 100 + 2652
$$

$$
= 6720000 + 2840 \times 100 + 2652 = 6720000 + 284000 + 2652 = 7006652
$$

Verification: $1234 \times 5678 = 7006652$. Correct.

## Summary

Karatsuba multiplication reduces the number of recursive multiplications from four to three by computing the middle coefficient as $p_3 - p_1 - p_2$ instead of computing $x_1 y_0$ and $x_0 y_1$ separately. This simple algebraic trick improves the complexity from $O(n^2)$ to $O(n^{\log_2 3}) \approx O(n^{1.585})$, demonstrating how reducing the number of subproblems by even one can dramatically improve asymptotic performance.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 4. MIT Press.
- Karatsuba, A., & Ofman, Y. (1962). Multiplication of multidigit numbers on automata. *Soviet Physics Doklady*, 7, 595-596.
