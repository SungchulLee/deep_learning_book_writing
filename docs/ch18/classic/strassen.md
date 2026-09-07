# 슈트라센 행렬 곱셈

Multiplying two $n \times n$ matrices using the standard algorithm requires $O(n^3)$ scalar multiplications. In 1969, Volker Strassen showed that by cleverly combining seven recursive multiplications of $n/2 \times n/2$ submatrices -- instead of the eight required by the standard divide-and-conquer approach -- the complexity drops to $O(n^{\log_2 7}) \approx O(n^{2.807})$. Like [Karatsuba multiplication](karatsuba.md) for integers, the key insight is reducing the number of recursive multiplications at each level.

## 보통의 행렬 곱셈

The product $C = A \cdot B$ of two $n \times n$ matrices is defined by

$$
C_{ij} = \sum_{k=1}^{n} A_{ik} \cdot B_{kj}
$$

Computing each of the $n^2$ entries requires $n$ multiplications and $n - 1$ additions, giving $\Theta(n^3)$ total work.

## 막무가내 나누어 이기기

Partition each $n \times n$ matrix into four $n/2 \times n/2$ submatrices:

$$
A = \begin{pmatrix} A_{11} & A_{12} \\ A_{21} & A_{22} \end{pmatrix}, \quad B = \begin{pmatrix} B_{11} & B_{12} \\ B_{21} & B_{22} \end{pmatrix}, \quad C = \begin{pmatrix} C_{11} & C_{12} \\ C_{21} & C_{22} \end{pmatrix}
$$

덩이 곱셈 식은 다음과 같다

$$
C_{11} = A_{11}B_{11} + A_{12}B_{21}
$$

$$
C_{12} = A_{11}B_{12} + A_{12}B_{22}
$$

$$
C_{21} = A_{21}B_{11} + A_{22}B_{21}
$$

$$
C_{22} = A_{21}B_{12} + A_{22}B_{22}
$$

This requires **8** multiplications of $n/2 \times n/2$ matrices plus **4** additions of $n/2 \times n/2$ matrices. The recurrence is

$$
T(n) = 8T\!\left(\frac{n}{2}\right) + \Theta(n^2)
$$

By the Master Theorem ($a = 8$, $b = 2$, $\log_2 8 = 3$, $f(n) = \Theta(n^2)$, case 1):

$$
T(n) = \Theta(n^3)
$$

보통의 알고리즘보다 나아진 것이 없다.

## 슈트라센 알고리즘

슈트라센은 다음 일곱 곱을 정해 되돌이 곱셈 횟수를 8에서 **7**로 줄인다:

$$
M_1 = (A_{11} + A_{22})(B_{11} + B_{22})
$$

$$
M_2 = (A_{21} + A_{22}) B_{11}
$$

$$
M_3 = A_{11} (B_{12} - B_{22})
$$

$$
M_4 = A_{22} (B_{21} - B_{11})
$$

$$
M_5 = (A_{11} + A_{12}) B_{22}
$$

$$
M_6 = (A_{21} - A_{11})(B_{11} + B_{12})
$$

$$
M_7 = (A_{12} - A_{22})(B_{21} + B_{22})
$$

그다음 결과 부분 행렬을 다음과 같이 셈한다

$$
C_{11} = M_1 + M_4 - M_5 + M_7
$$

$$
C_{12} = M_3 + M_5
$$

$$
C_{21} = M_2 + M_4
$$

$$
C_{22} = M_1 - M_2 + M_3 + M_6
$$

### 옳음 확인

We verify $C_{11}$ as a representative example.

$$
C_{11} = M_1 + M_4 - M_5 + M_7
$$

펼치면 다음과 같다.

$$
M_1 = A_{11}B_{11} + A_{11}B_{22} + A_{22}B_{11} + A_{22}B_{22}
$$

$$
M_4 = A_{22}B_{21} - A_{22}B_{11}
$$

$$
M_5 = A_{11}B_{22} + A_{12}B_{22}
$$

$$
M_7 = A_{12}B_{21} + A_{12}B_{22} - A_{22}B_{21} - A_{22}B_{22}
$$

$M_1 + M_4 - M_5 + M_7$을 더하면:

$$
= A_{11}B_{11} + \cancel{A_{11}B_{22}} + \cancel{A_{22}B_{11}} + \cancel{A_{22}B_{22}} + \cancel{A_{22}B_{21}} - \cancel{A_{22}B_{11}} - \cancel{A_{11}B_{22}} - \cancel{A_{12}B_{22}} + A_{12}B_{21} + \cancel{A_{12}B_{22}} - \cancel{A_{22}B_{21}} - \cancel{A_{22}B_{22}}
$$

$$
= A_{11}B_{11} + A_{12}B_{21}
$$

This matches the definition of $C_{11}$. The other three entries can be verified similarly. $\square$

## 복잡도 분석

### 점화식

Strassen's algorithm performs 7 recursive multiplications on $n/2 \times n/2$ matrices, plus $O(n^2)$ work for the 18 matrix additions and subtractions:

$$
T(n) = 7T\!\left(\frac{n}{2}\right) + \Theta(n^2)
$$

### 마스터 정리로 풀기

With $a = 7$, $b = 2$, $f(n) = \Theta(n^2)$:

$$
\log_b a = \log_2 7 \approx 2.807
$$

Since $f(n) = \Theta(n^2) = O(n^{\log_2 7 - \epsilon})$ for $\epsilon \approx 0.807$, this is case 1:

$$
T(n) = \Theta(n^{\log_2 7}) \approx \Theta(n^{2.807})
$$

### 견줌

| 알고리즘 | 곱셈 횟수 | 덧셈 횟수 | 시간 |
|---|---|---|---|
| Standard | $n^3$ | $n^3 - n^2$ | $\Theta(n^3)$ |
| Naive D&C | 8 recursive | 4 matrix adds | $\Theta(n^3)$ |
| Strassen | 7 recursive | 18 matrix adds | $\Theta(n^{2.807})$ |

For $n = 1024$, the standard method performs $\sim 10^9$ multiplications, while Strassen requires $\sim 10^{8.58} \approx 3.8 \times 10^8$ -- roughly a 2.8x speedup at this size.

## 실용적인 고려

!!! tip "갈리는 지점"
    슈트라센 알고리즘은 (4번이 아니라) 18번의 덧셈과 되돌이로 쪼개는 덧짐 때문에 보통의 알고리즘보다 상수 인자가 크다. 실전에서는 $n$이 갈리는 지점 아래로 내려가면 보통의 알고리즘으로 바꾸는데, 그 지점은 하드웨어에 따라 대개 $n = 32$에서 $n = 128$쯤이다.

!!! warning "수치의 든든함"
    Strassen's algorithm is less numerically stable than the standard algorithm because it involves subtractions that can cause cancellation. For applications requiring high numerical precision, the standard $O(n^3)$ algorithm or algorithms with better stability properties may be preferred.

### 기억 공간 덧짐

The naive implementation of Strassen's algorithm creates many temporary matrices at each recursive level, leading to significant memory overhead. Careful implementation can reduce this to $O(n^2)$ additional space by reusing buffers.

## 슈트라센을 넘어

슈트라센의 결과는 더 빠른 행렬 곱셈 알고리즘을 찾는 흐름에 불을 붙였다:

| Algorithm | Year | Exponent $\omega$ |
|---|---|---|
| 보통 | -- | 3.000 |
| 슈트라센 | 1969 | 2.807 |
| 코퍼스미스-위노그라드 | 1990 | 2.376 |
| 알만-바실레프스카 윌리엄스 | 2021 | 2.373 |

The theoretical lower bound is $\omega \ge 2$ (since the output has $n^2$ entries). Whether $\omega = 2$ is achievable remains one of the major open problems in theoretical computer science.

## 요약

Strassen's algorithm reduces matrix multiplication from $\Theta(n^3)$ to $\Theta(n^{2.807})$ by replacing 8 recursive multiplications with 7, at the cost of more additions. The approach mirrors Karatsuba's strategy for integer multiplication: reducing the number of expensive recursive operations by one, even at the expense of more cheap operations (additions), yields an asymptotic improvement. The resulting recurrence $T(n) = 7T(n/2) + \Theta(n^2)$ is solved by the Master Theorem (case 1).

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), 4장. MIT Press.
- Strassen, V. (1969). Gaussian elimination is not optimal. *Numerische Mathematik*, 13(4), 354-356.

## 연습문제

**연습문제 1.**
슈트라센 행렬 곱셈의 핵심 생각과 그 시간 복잡도를 설명하여라.

??? success "연습문제 1 풀이"
    Strassen's Matrix Multiplication applies the divide-and-conquer paradigm: split the problem into smaller subproblems, solve them recursively, and combine the results. The time complexity is determined by the recurrence relation governing the subproblem sizes and the combination cost. The Master Theorem or recursion tree analysis typically gives the closed-form complexity. $\square$

---

**연습문제 2.**
슈트라센 행렬 곱셈의 되돌이 관계식을 쓰고 마스터 정리로 풀어라.

??? success "연습문제 2 풀이"
    되돌이 식은 그 알고리즘이 어떻게 나누는지에 달려 있다(잔문제의 수 $a$, 크기를 줄이는 값 $b$, 아우르는 값 $f(n)$). 으뜸 정리를 쓴다. $f(n)$을 $n^{\log_b a}$과 견주어 어느 갈래인지 가린다. $f(n) = \Theta(n^{\log_b a})$이면(둘째 갈래) $T(n) = \Theta(n^{\log_b a} \log n)$이다. $\square$

---

**연습문제 3.**
슈트라센 행렬 곱셈이 막무가내 방식보다 나은 장면을 설명하여라. 얼마나 빨라지는지 수로 나타내어라.

??? success "연습문제 3 풀이"
    막무가내로 하는 길은 흔히 $O(n^2)$ 이상이 든다. 나누어 다스리는 길은 되부르며 쪼개어 군더더기 셈을 줄이므로 복잡도가 더 낮다. 들임 크기가 $n = 10^6$이면 $O(n^2) = 10^{12}$과 $O(n \log n) = 2 \times 10^7$의 차이는 $50{,}000$ 곱절이다. $\square$

---

**연습문제 4.**
슈트라센 행렬 곱셈의 바탕 경우는 무엇인가? 그것이 알고리즘 전체의 옳음에 어떤 영향을 주는가?

??? success "연습문제 4 풀이"
    밑 자리는 더 나눌 수 없을 만큼 작은 들임을 다룬다(흔히 $n \leq 1$이나 $n \leq 2$). 이때는 옳은 결과를 곧바로 돌려주어야 한다. 밑 자리가 제대로 없으면 되부름이 끝나지 않는다. 밑 자리를 더 크게 잡고($n \leq 10$ 따위) 더 단순한 알고리즘으로 갈아타면 같은 점근 복잡도를 지키면서 되부름 덤을 줄여 참으로 더 빠르게 할 수 있다. $\square$
