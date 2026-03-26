# Fast Fourier Transform

Multiplying two polynomials of degree $n$ takes $O(n^2)$ using the standard coefficient-by-coefficient approach. The **Fast Fourier Transform** (FFT) reduces this to $O(n \log n)$ by exploiting the algebraic structure of the **roots of unity**. The core idea is to evaluate both polynomials at $n$ special points, multiply pointwise in $O(n)$ time, then interpolate back to coefficient form, with each transformation taking $O(n \log n)$.

## Discrete Fourier Transform

Given a sequence $a = (a_0, a_1, \dots, a_{n-1})$, the **Discrete Fourier Transform** (DFT) evaluates the polynomial $A(x) = \sum_{k=0}^{n-1} a_k x^k$ at the $n$-th roots of unity $\omega_n^0, \omega_n^1, \dots, \omega_n^{n-1}$, where:

$$
\omega_n = e^{2\pi i / n}
$$

The DFT produces:

$$
\hat{a}_j = A(\omega_n^j) = \sum_{k=0}^{n-1} a_k \, \omega_n^{jk} \quad \text{for } j = 0, 1, \dots, n-1
$$

Computing all $n$ values naively takes $O(n^2)$. The FFT computes the same result in $O(n \log n)$.

## Cooley-Tukey Algorithm

The FFT divides the polynomial into even-indexed and odd-indexed coefficients:

$$
A(x) = A_{\text{even}}(x^2) + x \cdot A_{\text{odd}}(x^2)
$$

where $A_{\text{even}}(y) = a_0 + a_2 y + a_4 y^2 + \cdots$ and $A_{\text{odd}}(y) = a_1 + a_3 y + a_5 y^2 + \cdots$.

Evaluating $A$ at all $n$-th roots of unity reduces to evaluating $A_{\text{even}}$ and $A_{\text{odd}}$ at the $(n/2)$-th roots of unity, because $(\omega_n^j)^2 = \omega_{n/2}^j$.

The **butterfly operation** combines the two half-size results:

$$
A(\omega_n^j) = A_{\text{even}}(\omega_{n/2}^j) + \omega_n^j \cdot A_{\text{odd}}(\omega_{n/2}^j)
$$

$$
A(\omega_n^{j + n/2}) = A_{\text{even}}(\omega_{n/2}^j) - \omega_n^j \cdot A_{\text{odd}}(\omega_{n/2}^j)
$$

This gives the recurrence $T(n) = 2T(n/2) + O(n) = O(n \log n)$.

## Inverse FFT

The inverse DFT recovers the coefficients from the point values:

$$
a_k = \frac{1}{n} \sum_{j=0}^{n-1} \hat{a}_j \, \omega_n^{-jk}
$$

This has the same structure as the forward DFT but with $\omega_n^{-1}$ instead of $\omega_n$ and a $1/n$ scaling factor. The same FFT algorithm computes the inverse with these two modifications.

## Polynomial Multiplication via FFT

To multiply polynomials $A(x)$ and $B(x)$ of degree $n$:

1. Pad coefficients to length $2n$ (power of 2).
2. Compute $\hat{A} = \text{FFT}(a)$ and $\hat{B} = \text{FFT}(b)$.
3. Multiply pointwise: $\hat{C}_j = \hat{A}_j \cdot \hat{B}_j$.
4. Compute $c = \text{IFFT}(\hat{C})$.

Total time: $O(n \log n)$.

## Implementation

```python
"""
Fast Fourier Transform (Cooley-Tukey radix-2 algorithm).

Computes the DFT and its inverse in O(n log n) time, enabling
fast polynomial multiplication.
"""

import cmath

# === FFT Core ===

def fft(a: list[complex], invert: bool = False) -> list[complex]:
    """Compute FFT (or inverse FFT) of a sequence.

    Args:
        a: Input sequence (length must be a power of 2).
        invert: If True, compute inverse FFT.

    Returns:
        Transformed sequence.
    """
    n = len(a)
    if n == 1:
        return a[:]

    a_even = fft(a[0::2], invert)
    a_odd = fft(a[1::2], invert)

    angle = 2 * cmath.pi / n * (-1 if invert else 1)
    w = complex(1, 0)
    wn = cmath.exp(complex(0, angle))

    result = [complex(0)] * n
    for j in range(n // 2):
        result[j] = a_even[j] + w * a_odd[j]
        result[j + n // 2] = a_even[j] - w * a_odd[j]
        if invert:
            result[j] /= 2
            result[j + n // 2] /= 2
        w *= wn

    return result


# === Polynomial Multiplication ===

def poly_multiply(a: list[float], b: list[float]) -> list[float]:
    """Multiply two polynomials using FFT.

    Args:
        a: Coefficients of first polynomial (a[i] = coefficient of x^i).
        b: Coefficients of second polynomial.

    Returns:
        Coefficients of the product polynomial.
    """
    result_len = len(a) + len(b) - 1
    n = 1
    while n < result_len:
        n *= 2

    fa = [complex(x) for x in a] + [complex(0)] * (n - len(a))
    fb = [complex(x) for x in b] + [complex(0)] * (n - len(b))

    fa = fft(fa)
    fb = fft(fb)

    fc = [fa[i] * fb[i] for i in range(n)]
    fc = fft(fc, invert=True)

    return [round(c.real) for c in fc[:result_len]]


# === Demonstration ===

if __name__ == "__main__":
    # Multiply (1 + 2x + 3x^2) * (4 + 5x)
    a = [1, 2, 3]   # 1 + 2x + 3x^2
    b = [4, 5]       # 4 + 5x
    result = poly_multiply(a, b)
    print(f"({a}) * ({b}) = {result}")

    # Verify: (1+2x+3x^2)(4+5x) = 4 + 13x + 22x^2 + 15x^3
    expected = [4, 13, 22, 15]
    print(f"Expected: {expected}")
    print(f"Match: {result == expected}")
```

**Output:**

```
([1, 2, 3]) * ([4, 5]) = [4, 13, 22, 15]
Expected: [4, 13, 22, 15]
Match: True
```

The FFT-based multiplication correctly computes $(1 + 2x + 3x^2)(4 + 5x) = 4 + 13x + 22x^2 + 15x^3$.

## Complexity

| Aspect | Cost |
|--------|:----:|
| Time   | $O(n \log n)$ |
| Space  | $O(n)$ |

The recursive FFT makes two calls of size $n/2$ and does $O(n)$ work at each level, giving $O(n \log n)$ total. An iterative (bottom-up) version avoids recursion overhead.

## Applications

- **Polynomial multiplication.** The primary application; also used for large integer multiplication.
- **Signal processing.** Converting between time-domain and frequency-domain representations.
- **Convolution.** Convolution of sequences reduces to pointwise multiplication in the frequency domain.
- **String matching.** Wildcard pattern matching can be formulated as polynomial multiplication.

## Reference

- Cooley, J. W., & Tukey, J. W. (1965). An algorithm for the machine calculation of complex Fourier series. *Mathematics of Computation*, 19(90), 297--301.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to Algorithms* (4th ed.), Chapter 30: Polynomials and the FFT.
