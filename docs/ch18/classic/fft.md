# 빠른 푸리에 변환

차수가 $n$인 두 다항식을 여느 방식으로 계수마다 곱하면 $O(n^2)$이 든다. **빠른 푸리에 옮김**(FFT)은 **1의 거듭제곱근**이 지닌 대수 얼개를 써서 이를 $O(n \log n)$으로 줄인다. 고갱이 깨침은 두 다항식을 남다른 점 $n$개에서 값매김하고, $O(n)$ 때에 점마다 곱한 뒤, 다시 계수 꼴로 되돌리는 것이며, 옮김마다 $O(n \log n)$이 든다.

## 띄엄띄엄 푸리에 변환

이음 $a = (a_0, a_1, \dots, a_{n-1})$이 주어졌을 때 **따로 떨어진 푸리에 옮김**(DFT)은 다항식 $A(x) = \sum_{k=0}^{n-1} a_k x^k$을 1의 $n$제곱근 $\omega_n^0, \omega_n^1, \dots, \omega_n^{n-1}$에서 값매김한다. 여기서

$$
\omega_n = e^{2\pi i / n}
$$

띄엄띄엄 푸리에 변환은 다음을 내놓는다:

$$
\hat{a}_j = A(\omega_n^j) = \sum_{k=0}^{n-1} a_k \, \omega_n^{jk} \quad \text{for } j = 0, 1, \dots, n-1
$$

값 $n$개를 손쉽게 셈하면 $O(n^2)$이 든다. FFT은 같은 결과를 $O(n \log n)$에 셈한다.

## 쿨리-투키 알고리즘

빠른 푸리에 변환은 다항식을 짝수 번호 계수와 홀수 번호 계수로 나눈다:

$$
A(x) = A_{\text{even}}(x^2) + x \cdot A_{\text{odd}}(x^2)
$$

where $A_{\text{even}}(y) = a_0 + a_2 y + a_4 y^2 + \cdots$ and $A_{\text{odd}}(y) = a_1 + a_3 y + a_5 y^2 + \cdots$.

$(\omega_n^j)^2 = \omega_{n/2}^j$이므로, $A$을 1의 $n$제곱근 모두에서 값매김하는 일은 $A_{\text{even}}$과 $A_{\text{odd}}$을 1의 $(n/2)$제곱근에서 값매김하는 일로 줄어든다.

**나비 연산**이 반 크기의 두 결과를 아우른다:

$$
A(\omega_n^j) = A_{\text{even}}(\omega_{n/2}^j) + \omega_n^j \cdot A_{\text{odd}}(\omega_{n/2}^j)
$$

$$
A(\omega_n^{j + n/2}) = A_{\text{even}}(\omega_{n/2}^j) - \omega_n^j \cdot A_{\text{odd}}(\omega_{n/2}^j)
$$

그러면 되돌이 식 $T(n) = 2T(n/2) + O(n) = O(n \log n)$을 얻는다.

## 거꿀 빠른 푸리에 변환

거꿀 띄엄띄엄 푸리에 변환은 점 값에서 계수를 되찾는다:

$$
a_k = \frac{1}{n} \sum_{j=0}^{n-1} \hat{a}_j \, \omega_n^{-jk}
$$

이는 앞으로 가는 DFT과 얼개가 같되 $\omega_n$ 대신 $\omega_n^{-1}$을 쓰고 $1/n$을 곱한다. 이 두 가지만 고치면 같은 FFT 알고리즘으로 거꾸로 옮김도 셈할 수 있다.

## 빠른 푸리에 변환으로 하는 다항식 곱셈

차수 $n$인 다항식 $A(x)$과 $B(x)$을 곱하려면:

1. 계수를 길이 $2n$(2의 거듭제곱)까지 덧댄다.
2. Compute $\hat{A} = \text{FFT}(a)$ and $\hat{B} = \text{FFT}(b)$.
3. 점마다 곱한다: $\hat{C}_j = \hat{A}_j \cdot \hat{B}_j$.
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
        b: 둘째 다항식의 계수.

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

FFT 바탕 곱하기는 $(1 + 2x + 3x^2)(4 + 5x) = 4 + 13x + 22x^2 + 15x^3$을 옳게 셈한다.

## 복잡도

| 항목 | 비용 |
|--------|:----:|
| Time   | $O(n \log n)$ |
| 공간 | $O(n)$ |

되부르는 FFT은 크기 $n/2$인 부름을 둘 하고 켜마다 $O(n)$ 일감을 하므로 모두 $O(n \log n)$이다. 되돌이로(아래에서 위로) 짜면 되부름 덤을 피할 수 있다.

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
    빠른 푸리에 옮김은 나누어 다스리기 틀을 쓴다. 문제를 더 작은 잔문제로 쪼개고, 되부르며 풀고, 그 결과를 아우른다. 때 복잡도는 잔문제의 크기와 아우르는 값을 다스리는 되돌이 식이 정한다. 흔히 으뜸 정리나 되부름 나무 살피기로 닫힌 꼴의 복잡도를 얻는다. $\square$

---

**연습문제 2.**
빠른 푸리에 변환의 되돌이 관계식을 쓰고 마스터 정리로 풀어라.

??? success "연습문제 2 풀이"
    되돌이 식은 그 알고리즘이 어떻게 나누는지에 달려 있다(잔문제의 수 $a$, 크기를 줄이는 값 $b$, 아우르는 값 $f(n)$). 으뜸 정리를 쓴다. $f(n)$을 $n^{\log_b a}$과 견주어 어느 갈래인지 가린다. $f(n) = \Theta(n^{\log_b a})$이면(둘째 갈래) $T(n) = \Theta(n^{\log_b a} \log n)$이다. $\square$

---

**연습문제 3.**
빠른 푸리에 변환이 막무가내 방식보다 나은 장면을 설명하여라. 얼마나 빨라지는지 수로 나타내어라.

??? success "연습문제 3 풀이"
    막무가내로 하는 길은 흔히 $O(n^2)$ 이상이 든다. 나누어 다스리는 길은 되부르며 쪼개어 군더더기 셈을 줄이므로 복잡도가 더 낮다. 들임 크기가 $n = 10^6$이면 $O(n^2) = 10^{12}$과 $O(n \log n) = 2 \times 10^7$의 차이는 $50{,}000$ 곱절이다. $\square$

---

**연습문제 4.**
빠른 푸리에 변환의 바탕 경우는 무엇인가? 그것이 알고리즘 전체의 옳음에 어떤 영향을 주는가?

??? success "연습문제 4 풀이"
    밑 자리는 더 나눌 수 없을 만큼 작은 들임을 다룬다(흔히 $n \leq 1$이나 $n \leq 2$). 이때는 옳은 결과를 곧바로 돌려주어야 한다. 밑 자리가 제대로 없으면 되부름이 끝나지 않는다. 밑 자리를 더 크게 잡고($n \leq 10$ 따위) 더 단순한 알고리즘으로 갈아타면 같은 점근 복잡도를 지키면서 되부름 덤을 줄여 참으로 더 빠르게 할 수 있다. $\square$
