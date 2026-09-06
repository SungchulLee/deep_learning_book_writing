# 카라추바 곱셈

Multiplying two $n$-digit numbers using the grade-school algorithm requires $O(n^2)$ single-digit multiplications. In 1960, Anatoly Karatsuba discovered that a clever algebraic rearrangement reduces the number of recursive multiplications from four to three, yielding an $O(n^{\log_2 3}) \approx O(n^{1.585})$ algorithm. This was the first multiplication algorithm to break the $O(n^2)$ barrier and remains one of the most elegant examples of divide and conquer.

## 초등학교 곱셈법

To multiply two $n$-digit numbers $x$ and $y$, the standard method computes every pair of digits and sums the partial products. This requires $n^2$ single-digit multiplications and $O(n)$ additions per partial product, giving $O(n^2)$ total work.

## 카라추바의 눈썰미

Split each $n$-digit number into two halves of $n/2$ digits. Let $m = \lfloor n/2 \rfloor$, and write

$$
x = x_1 \cdot 10^m + x_0, \qquad y = y_1 \cdot 10^m + y_0
$$

where $x_1, x_0, y_1, y_0$ are at most $\lceil n/2 \rceil$-digit numbers.

### 막무가내 쪼개기

The product $x \cdot y$ expands as

$$
x \cdot y = x_1 y_1 \cdot 10^{2m} + (x_1 y_0 + x_0 y_1) \cdot 10^m + x_0 y_0
$$

여기에는 $n/2$자리 수의 곱셈이 **네** 번 필요하다. 곧 $x_1 y_1$, $x_1 y_0$, $x_0 y_1$, $x_0 y_0$이다. 그 되돌이 관계식은 다음과 같다

$$
T(n) = 4T\!\left(\frac{n}{2}\right) + O(n)
$$

which solves to $T(n) = O(n^2)$ -- no improvement over grade school.

### 핵심 재주

카라추바는 가운데 계수 $x_1 y_0 + x_0 y_1$을 곱셈 두 번이 아니라 **한** 번만 더해 셈할 수 있음을 알아챘다. 곱 셋을 정한다:

$$
p_1 = x_1 \cdot y_1, \qquad p_2 = x_0 \cdot y_0, \qquad p_3 = (x_1 + x_0)(y_1 + y_0)
$$

그러면

$$
x_1 y_0 + x_0 y_1 = p_3 - p_1 - p_2
$$

**증명.** $p_3$을 펼치면:

$$
p_3 = (x_1 + x_0)(y_1 + y_0) = x_1 y_1 + x_1 y_0 + x_0 y_1 + x_0 y_0 = p_1 + (x_1 y_0 + x_0 y_1) + p_2
$$

Rearranging gives $x_1 y_0 + x_0 y_1 = p_3 - p_1 - p_2$. $\square$

따라서 온전한 곱은 다음과 같다

$$
x \cdot y = p_1 \cdot 10^{2m} + (p_3 - p_1 - p_2) \cdot 10^m + p_2
$$

여기에는 대략 $n/2$자리 수의 곱셈이 **세** 번($p_1, p_2, p_3$)과 $O(n)$의 덧셈과 자리 옮김만 필요하다.

## 알고리즘

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

### 파이썬 구현

```python
def karatsuba(x, y):
    """
    카라추바 알고리즘으로 음이 아닌 두 정수 곱하기.

    매개변수
    ----------
    x : int
        첫 번째 음이 아닌 정수.
    y : int
        두 번째 음이 아닌 정수.

    반환값
    -------
    int
        곱 x * y.
    """
    # 바탕 경우: 한 자리 곱셈
    if x < 10 or y < 10:
        return x * y

    # 자릿수 정하기
    n = max(len(str(x)), len(str(y)))
    m = n // 2

    # x와 y 쪼개기
    power = 10 ** m
    x1, x0 = x // power, x % power
    y1, y0 = y // power, y % power

    # 되돌이 곱셈 세 번
    p1 = karatsuba(x1, y1)
    p2 = karatsuba(x0, y0)
    p3 = karatsuba(x1 + x0, y1 + y0)

    # 합친다
    return p1 * (10 ** (2 * m)) + (p3 - p1 - p2) * (10 ** m) + p2
```

## 복잡도 분석

### 점화식

이 알고리즘은 대략 $n/2$자리 수에 되돌이 부름을 3번 하고, 덧셈·뺄셈·자리 옮김에 $O(n)$의 품을 쓴다:

$$
T(n) = 3T\!\left(\frac{n}{2}\right) + O(n)
$$

### 마스터 정리로 풀기

$a = 3$, $b = 2$, $f(n) = O(n)$이므로:

$$
\log_b a = \log_2 3 \approx 1.585
$$

Since $f(n) = O(n) = O(n^{\log_2 3 - \epsilon})$ for $\epsilon \approx 0.585$, this is case 1:

$$
T(n) = \Theta(n^{\log_2 3}) \approx \Theta(n^{1.585})
$$

### 초등학교 곱셈법보다 나아진 점

| 방법 | 곱셈 횟수 | 시간 |
|---|---|---|
| Grade school | $n^2$ | $O(n^2)$ |
| Karatsuba | $n^{\log_2 3}$ | $O(n^{1.585})$ |

For $n = 1000$ digits, grade school requires $\sim 10^6$ operations; Karatsuba requires $\sim 10^{4.76} \approx 57{,}500$ -- a 17x speedup.

## 올바름

옳음은 다음 대수 항등식에서 따라 나온다

$$
x \cdot y = p_1 \cdot 10^{2m} + (p_3 - p_1 - p_2) \cdot 10^m + p_2
$$

and induction on $n$. The base case (direct multiplication for small $n$) is correct by definition. For the inductive step, assume Karatsuba correctly computes products of numbers with fewer than $n$ digits. Then $p_1$, $p_2$, and $p_3$ are correct, and the formula above gives the correct product. $\square$

## 실용적인 고려

!!! tip "바탕 경우의 문턱값"
    In practice, Karatsuba is slower than grade-school multiplication for small numbers due to the overhead of additions, subtractions, and recursive calls. Implementations typically switch to the grade-school method when $n$ drops below a threshold (e.g., $n \le 64$ digits).

!!! note "카라추바를 넘어"
    Karatsuba's approach generalizes: Toom-Cook splits numbers into 3 or more pieces, further reducing the exponent. The current asymptotically fastest known algorithm (Harvey-van der Hoeven, 2019) achieves $O(n \log n)$ for integer multiplication, though Karatsuba remains practical for moderately large numbers.

## 풀이 예제

Compute $1234 \times 5678$ using Karatsuba.

Set $m = 2$, so $10^m = 100$:

- $x_1 = 12$, $x_0 = 34$
- $y_1 = 56$, $y_0 = 78$

곱셈 세 번:

- $p_1 = 12 \times 56 = 672$
- $p_2 = 34 \times 78 = 2652$
- $p_3 = (12 + 34) \times (56 + 78) = 46 \times 134 = 6164$

아우르기:

$$
1234 \times 5678 = 672 \times 10000 + (6164 - 672 - 2652) \times 100 + 2652
$$

$$
= 6720000 + 2840 \times 100 + 2652 = 6720000 + 284000 + 2652 = 7006652
$$

Verification: $1234 \times 5678 = 7006652$. Correct.

## 요약

Karatsuba multiplication reduces the number of recursive multiplications from four to three by computing the middle coefficient as $p_3 - p_1 - p_2$ instead of computing $x_1 y_0$ and $x_0 y_1$ separately. This simple algebraic trick improves the complexity from $O(n^2)$ to $O(n^{\log_2 3}) \approx O(n^{1.585})$, demonstrating how reducing the number of subproblems by even one can dramatically improve asymptotic performance.

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), 4장. MIT Press.
- Karatsuba, A., & Ofman, Y. (1962). Multiplication of multidigit numbers on automata. *Soviet Physics Doklady*, 7, 595-596.

## 연습문제

**연습문제 1.**
카라추바 곱셈의 핵심 생각과 그 시간 복잡도를 설명하여라.

??? success "연습문제 1 풀이"
    Karatsuba Multiplication applies the divide-and-conquer paradigm: split the problem into smaller subproblems, solve them recursively, and combine the results. The time complexity is determined by the recurrence relation governing the subproblem sizes and the combination cost. The Master Theorem or recursion tree analysis typically gives the closed-form complexity. $\square$

---

**연습문제 2.**
카라추바 곱셈의 되돌이 관계식을 쓰고 마스터 정리로 풀어라.

??? success "연습문제 2 풀이"
    The recurrence depends on the specific algorithm's division strategy (number of subproblems $a$, size reduction factor $b$, and combination cost $f(n)$). Apply the Master Theorem: compare $f(n)$ with $n^{\log_b a}$ to determine which case applies. If $f(n) = \Theta(n^{\log_b a})$ (case 2), $T(n) = \Theta(n^{\log_b a} \log n)$. $\square$

---

**연습문제 3.**
카라추바 곱셈이 막무가내 방식보다 나은 장면을 설명하여라. 얼마나 빨라지는지 수로 나타내어라.

??? success "연습문제 3 풀이"
    The brute-force approach typically runs in $O(n^2)$ or worse. The divide-and-conquer approach achieves a lower complexity by reducing redundant computation through recursive decomposition. For input size $n = 10^6$, the difference between $O(n^2) = 10^{12}$ and $O(n \log n) = 2 \times 10^7$ operations is a factor of $50{,}000$. $\square$

---

**연습문제 4.**
카라추바 곱셈의 바탕 경우는 무엇인가? 그것이 알고리즘 전체의 옳음에 어떤 영향을 주는가?

??? success "연습문제 4 풀이"
    Base cases handle inputs too small to subdivide further (typically $n \leq 1$ or $n \leq 2$). They must return correct results directly. Without proper base cases, the recursion never terminates. Choosing a larger base case (e.g., $n \leq 10$) and switching to a simpler algorithm can improve practical performance by reducing recursion overhead while maintaining the same asymptotic complexity. $\square$
