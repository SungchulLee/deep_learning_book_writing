# 빠른 푸리에 변환

Multiplying two polynomials of degree $n$ takes $O(n^2)$ using the standard coefficient-by-coefficient approach. The **Fast Fourier Transform** (FFT) reduces this to $O(n \log n)$ by exploiting the algebraic structure of the **roots of unity**. The core idea is to evaluate both polynomials at $n$ special points, multiply pointwise in $O(n)$ time, then interpolate back to coefficient form, with each transformation taking $O(n \log n)$.

## 띄엄띄엄 푸리에 변환

Given a sequence $a = (a_0, a_1, \dots, a_{n-1})$, the **Discrete Fourier Transform** (DFT) evaluates the polynomial $A(x) = \sum_{k=0}^{n-1} a_k x^k$ at the $n$-th roots of unity $\omega_n^0, \omega_n^1, \dots, \omega_n^{n-1}$, where:

$$
\omega_n = e^{2\pi i / n}
$$

띄엄띄엄 푸리에 변환은 다음을 내놓는다:

$$
\hat{a}_j = A(\omega_n^j) = \sum_{k=0}^{n-1} a_k \, \omega_n^{jk} \quad \text{for } j = 0, 1, \dots, n-1
$$

Computing all $n$ values naively takes $O(n^2)$. The FFT computes the same result in $O(n \log n)$.

## 쿨리-투키 알고리즘

빠른 푸리에 변환은 다항식을 짝수 번호 계수와 홀수 번호 계수로 나눈다:

$$
A(x) = A_{\text{even}}(x^2) + x \cdot A_{\text{odd}}(x^2)
$$

where $A_{\text{even}}(y) = a_0 + a_2 y + a_4 y^2 + \cdots$ and $A_{\text{odd}}(y) = a_1 + a_3 y + a_5 y^2 + \cdots$.

Evaluating $A$ at all $n$-th roots of unity reduces to evaluating $A_{\text{even}}$ and $A_{\text{odd}}$ at the $(n/2)$-th roots of unity, because $(\omega_n^j)^2 = \omega_{n/2}^j$.

**나비 연산**이 반 크기의 두 결과를 아우른다:

$$
A(\omega_n^j) = A_{\text{even}}(\omega_{n/2}^j) + \omega_n^j \cdot A_{\text{odd}}(\omega_{n/2}^j)
$$

$$
A(\omega_n^{j + n/2}) = A_{\text{even}}(\omega_{n/2}^j) - \omega_n^j \cdot A_{\text{odd}}(\omega_{n/2}^j)
$$

This gives the recurrence $T(n) = 2T(n/2) + O(n) = O(n \log n)$.

## 거꿀 빠른 푸리에 변환

거꿀 띄엄띄엄 푸리에 변환은 점 값에서 계수를 되찾는다:

$$
a_k = \frac{1}{n} \sum_{j=0}^{n-1} \hat{a}_j \, \omega_n^{-jk}
$$

This has the same structure as the forward DFT but with $\omega_n^{-1}$ instead of $\omega_n$ and a $1/n$ scaling factor. The same FFT algorithm computes the inverse with these two modifications.

## 빠른 푸리에 변환으로 하는 다항식 곱셈

차수 $n$인 다항식 $A(x)$과 $B(x)$을 곱하려면:

1. 계수를 길이 $2n$(2의 거듭제곱)까지 덧댄다.
2. Compute $\hat{A} = \text{FFT}(a)$ and $\hat{B} = \text{FFT}(b)$.
3. Multiply pointwise: $\hat{C}_j = \hat{A}_j \cdot \hat{B}_j$.
4. Compute $c = \text{IFFT}(\hat{C})$.

Total time: $O(n \log n)$.

## 구현

```python
"""
빠른 푸리에 변환(쿨리-투키 밑 2 알고리즘).

띄엄띄엄 푸리에 변환과 그 거꿀을 O(n log n) 시간에 셈해
빠른 다항식 곱셈을 가능하게 한다.
"""

import cmath

# === 빠른 푸리에 변환 고갱이 ===

def fft(a: list[complex], invert: bool = False) -> list[complex]:
    """차례의 빠른 푸리에 변환(또는 그 거꿀) 셈하기.

    인수:
        a: Input sequence (length must be a power of 2).
        invert: True이면 거꿀 빠른 푸리에 변환을 셈한다.

    반환값:
        바꾼 차례.
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


# === 다항식 곱셈 ===

def poly_multiply(a: list[float], b: list[float]) -> list[float]:
    """빠른 푸리에 변환으로 두 다항식 곱하기.

    인수:
        a: Coefficients of first polynomial (a[i] = coefficient of x^i).
        b: Coefficients of second polynomial.

    반환값:
        곱 다항식의 계수.
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


# === 시연 ===

if __name__ == "__main__":
    # (1 + 2x + 3x^2) * (4 + 5x) 곱하기
    a = [1, 2, 3]   # 1 + 2x + 3x^2
    b = [4, 5]       # 4 + 5x
    result = poly_multiply(a, b)
    print(f"({a}) * ({b}) = {result}")

    # 확인: (1+2x+3x^2)(4+5x) = 4 + 13x + 22x^2 + 15x^3
    expected = [4, 13, 22, 15]
    print(f"Expected: {expected}")
    print(f"Match: {result == expected}")
```

**출력:**

```
([1, 2, 3]) * ([4, 5]) = [4, 13, 22, 15]
Expected: [4, 13, 22, 15]
Match: True
```

The FFT-based multiplication correctly computes $(1 + 2x + 3x^2)(4 + 5x) = 4 + 13x + 22x^2 + 15x^3$.

## 복잡도

| 항목 | 비용 |
|--------|:----:|
| Time   | $O(n \log n)$ |
| 공간 | $O(n)$ |

The recursive FFT makes two calls of size $n/2$ and does $O(n)$ work at each level, giving $O(n \log n)$ total. An iterative (bottom-up) version avoids recursion overhead.

## 응용

- **다항식 곱셈.** 으뜸 쓰임새이며 큰 정수 곱셈에도 쓴다.
- **신호 다루기.** 때 영역과 잦기 영역 나타냄 사이를 오간다.
- **누비기.** 차례의 누비기는 잦기 영역에서 점별 곱셈으로 줄어든다.
- **글자열 짝짓기.** 아무거나 자리를 둔 무늬 짝짓기는 다항식 곱셈으로 세울 수 있다.

## 참고 문헌

- Cooley, J. W., & Tukey, J. W. (1965). An algorithm for the machine calculation of complex Fourier series. *Mathematics of Computation*, 19(90), 297--301.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to Algorithms* (4th ed.), 30장: Polynomials and the FFT.

## 연습문제

**연습문제 1.**
빠른 푸리에 변환의 핵심 생각과 그 시간 복잡도를 설명하여라.

??? success "연습문제 1 풀이"
    Fast Fourier Transform applies the divide-and-conquer paradigm: split the problem into smaller subproblems, solve them recursively, and combine the results. The time complexity is determined by the recurrence relation governing the subproblem sizes and the combination cost. The Master Theorem or recursion tree analysis typically gives the closed-form complexity. $\square$

---

**연습문제 2.**
빠른 푸리에 변환의 되돌이 관계식을 쓰고 마스터 정리로 풀어라.

??? success "연습문제 2 풀이"
    The recurrence depends on the specific algorithm's division strategy (number of subproblems $a$, size reduction factor $b$, and combination cost $f(n)$). Apply the Master Theorem: compare $f(n)$ with $n^{\log_b a}$ to determine which case applies. If $f(n) = \Theta(n^{\log_b a})$ (case 2), $T(n) = \Theta(n^{\log_b a} \log n)$. $\square$

---

**연습문제 3.**
빠른 푸리에 변환이 막무가내 방식보다 나은 장면을 설명하여라. 얼마나 빨라지는지 수로 나타내어라.

??? success "연습문제 3 풀이"
    The brute-force approach typically runs in $O(n^2)$ or worse. The divide-and-conquer approach achieves a lower complexity by reducing redundant computation through recursive decomposition. For input size $n = 10^6$, the difference between $O(n^2) = 10^{12}$ and $O(n \log n) = 2 \times 10^7$ operations is a factor of $50{,}000$. $\square$

---

**연습문제 4.**
빠른 푸리에 변환의 바탕 경우는 무엇인가? 그것이 알고리즘 전체의 옳음에 어떤 영향을 주는가?

??? success "연습문제 4 풀이"
    Base cases handle inputs too small to subdivide further (typically $n \leq 1$ or $n \leq 2$). They must return correct results directly. Without proper base cases, the recursion never terminates. Choosing a larger base case (e.g., $n \leq 10$) and switching to a simpler algorithm can improve practical performance by reducing recursion overhead while maintaining the same asymptotic complexity. $\square$
