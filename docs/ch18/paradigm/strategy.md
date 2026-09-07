# 나누어 이기기 전략

여러 셈 문제는 자연스러운 되돌이 짜임을 지닌다. 곧 큰 보기를 같은 갈래의 작은 보기로 쪼개어 저마다 따로 풀고, 그 부분 결과를 모아 본디 문제의 풀이로 만든다. **나누어 이기기**라 하는 이 생각은 알고리즘 꾸미기에서 가장 힘세고 널리 쓰이는 틀 가운데 하나이다. 어울러 정렬, 이분 찾기, 빠른 푸리에 변환처럼 갖가지 알고리즘이 이에 바탕한다.

이 쪽에서는 나누어 이기기 전략을 큰 틀에서 소개하고, 왜 그것이 효율적인 알고리즘으로 이어지는지 밝히며, 이 장 나머지에서 쓸 말을 세운다.

## 핵심 생각

나누어 이기기 알고리즘은 크기 $n$인 문제를 세 단계로 풀어낸다:

1. **Divide** the problem into $a \ge 1$ subproblems, each of size roughly $n / b$ for some $b > 1$.
2. 작은 문제마다 되돌이로 **이긴다**. 작은 문제가 충분히 작아지면 **바탕 경우**로 곧바로 푼다.
3. 작은 문제의 풀이를 본디 문제의 풀이로 **아우른다**.

핵심 눈썰미는, 나누고 아우르는 데 드는 전체 품이 되돌이 켜마다 문제 크기를 지수적으로 줄여 아끼는 품보다 훨씬 적은 경우가 많다는 것이다.

## 나누어 이기기가 되는 까닭

Consider a problem of size $n$ that we split into $a$ subproblems each of size $n/b$. If solving the full problem directly takes $\Theta(n^c)$ work for some constant $c$, then the recursive approach replaces a single $\Theta(n^c)$ computation with $a$ computations of size $(n/b)^c = n^c / b^c$, plus the overhead $D(n)$ of dividing and $C(n)$ of combining. The total work at the top level is therefore

$$
a \cdot \left(\frac{n}{b}\right)^c + D(n) + C(n)
$$

When $a < b^c$, the subproblem work shrinks geometrically at each level, and the algorithm is faster than the brute-force approach. When $a > b^c$, the work grows at each level but the depth is only $\log_b n$, so the total is still bounded. The precise trade-off is captured by the **Master Theorem**, analyzed in detail on the [Recurrence Analysis](recurrence.md) page.

## 엄밀한 얼거리

$T(n)$을 크기 $n$인 들임에 대한 나누어 이기기 알고리즘의 도는 시간이라 하자. 일반 되돌이 관계식은 다음과 같다

$$
T(n) = a \, T\!\left(\frac{n}{b}\right) + f(n)
$$

여기서 각 기호는 다음과 같다.

- $a$은 되돌이 부름마다 생기는 작은 문제의 개수,
- $b$은 문제 크기가 줄어드는 비율,
- $f(n)$은 나누고 아우르는 값을 담는다.

The **base case** is $T(n) = \Theta(1)$ for $n \le n_0$, where $n_0$ is a small constant. Choosing the right base case is important for practical efficiency: switching to an $O(n^2)$ algorithm when $n$ drops below a threshold (e.g., insertion sort for small arrays inside merge sort) can significantly reduce constant factors.

## 나누어 이기기 알고리즘 꾸미기

나누어 이기기 풀이를 만들려면 네 물음에 답해야 한다:

1. **How to divide?** Choose a splitting strategy that produces balanced subproblems. Unbalanced splits (e.g., $n - 1$ and $1$) lead to $O(n)$ recursion depth and often $O(n^2)$ total work.
2. **작은 문제는 몇 개인가?** $a$을 줄이는 것이 알고리즘을 빠르게 하는 가장 곧바른 길이다. 카라추바 곱셈은 곱셈 4번을 3번으로, 슈트라센 알고리즘은 8번을 7번으로 줄인다.
3. **How to combine?** The combine step must run in low-order time (typically $O(n)$ or $O(n \log n)$) to keep the overall complexity favorable.
4. **바탕 경우는 무엇인가?** 바탕 경우가 너무 크면 품을 낭비하고, 너무 작으면 되돌이 덧짐이 지나치게 든다.

!!! tip "고르게 쪼개면 깊이가 가장 좋아진다"
    Splitting the problem into subproblems of roughly equal size ensures the recursion tree has depth $\Theta(\log n)$. This logarithmic depth is the fundamental source of efficiency in divide-and-conquer algorithms.

## 다른 틀과의 견줌

나누어 이기기는 주요 알고리즘 꾸미기 틀 가운데 하나이다. 다른 틀과의 관계를 알면 언제 써야 할지가 또렷해진다.

| 틀 | 핵심 성질 | 작은 문제의 겹침 |
|---|---|---|
| **나누어 이기기** | 작은 문제끼리 안 얽힌다 | 겹치지 않음 |
| **동적 계획** | 작은 문제가 겹치고 풀이를 나눠 갖는다 | 많이 겹침 |
| **욕심쟁이** | 그때그때 가장 좋은 것을 고른다 | 작은 문제 없음 |
| **되돌아가기** | 찾을 자리를 살펴보고 쳐 낸다 | 경우마다 다름 |

나누어 이기기 알고리즘은 서로 **얽히지 않는** 작은 문제를 낳는다. 곧 한 작은 문제의 풀이가 다른 것의 풀이에 달려 있지 않다. 작은 문제가 겹칠 때, 곧 서로 다른 되돌이 가지가 같은 작은 문제를 여러 번 풀 때는 보통 동적 계획이 더 알맞다.

## 대표 보기

다음 표는 이 장에서 다루는 나누어 이기기 알고리즘 몇 가지와 그 되돌이 관계식, 그리고 나오는 복잡도를 보여 준다.

| 알고리즘 | $a$ | $b$ | $f(n)$ | $T(n)$ |
|---|---|---|---|---|
| Binary search | $1$ | $2$ | $O(1)$ | $O(\log n)$ |
| Merge sort | $2$ | $2$ | $O(n)$ | $O(n \log n)$ |
| Karatsuba multiplication | $3$ | $2$ | $O(n)$ | $O(n^{\log_2 3}) \approx O(n^{1.585})$ |
| Strassen's matrix multiply | $7$ | $2$ | $O(n^2)$ | $O(n^{\log_2 7}) \approx O(n^{2.807})$ |
| Closest pair of points | $2$ | $2$ | $O(n)$ | $O(n \log n)$ |
| FFT | $2$ | $2$ | $O(n)$ | $O(n \log n)$ |

이 알고리즘은 저마다 [고전 나누어 이기기](../classic/binary_search.md) 절에서 자세히 다룬다.

## 요약

나누어 이기기는 어려운 문제를 자기 자신의 더 작은 판으로 바꾸고, 저마다 되돌이로 풀어 결과를 아우른다. 그 힘은 세 성질에서 나온다. (1) 고르게 쪼개면 되돌이 깊이가 로그로 유지되고, (2) 작은 문제끼리 안 얽혀 셈을 되풀이하지 않으며, (3) 아우르기가 효율적이면 켜마다의 품이 잡힌다. 그 결과 도는 시간은 되돌이 관계식 $T(n) = aT(n/b) + f(n)$이 다스리며, 그 풀이는 $a$, $b$, $f(n)$ 사이 관계에 달렸다.

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), 4장. MIT Press.

## 연습문제

**연습문제 1.**
나누어 이기기 전략의 핵심 생각과 그 시간 복잡도를 설명하여라.

??? success "연습문제 1 풀이"
    Divide and Conquer Strategy applies the divide-and-conquer paradigm: split the problem into smaller subproblems, solve them recursively, and combine the results. The time complexity is determined by the recurrence relation governing the subproblem sizes and the combination cost. The Master Theorem or recursion tree analysis typically gives the closed-form complexity. $\square$

---

**연습문제 2.**
나누어 이기기 전략의 되돌이 관계식을 쓰고 마스터 정리로 풀어라.

??? success "연습문제 2 풀이"
    되돌이 식은 그 알고리즘이 어떻게 나누는지에 달려 있다(잔문제의 수 $a$, 크기를 줄이는 값 $b$, 아우르는 값 $f(n)$). 으뜸 정리를 쓴다. $f(n)$을 $n^{\log_b a}$과 견주어 어느 갈래인지 가린다. $f(n) = \Theta(n^{\log_b a})$이면(둘째 갈래) $T(n) = \Theta(n^{\log_b a} \log n)$이다. $\square$

---

**연습문제 3.**
나누어 이기기 전략이 막무가내 방식보다 나은 장면을 설명하여라. 얼마나 빨라지는지 수로 나타내어라.

??? success "연습문제 3 풀이"
    막무가내로 하는 길은 흔히 $O(n^2)$ 이상이 든다. 나누어 다스리는 길은 되부르며 쪼개어 군더더기 셈을 줄이므로 복잡도가 더 낮다. 들임 크기가 $n = 10^6$이면 $O(n^2) = 10^{12}$과 $O(n \log n) = 2 \times 10^7$의 차이는 $50{,}000$ 곱절이다. $\square$

---

**연습문제 4.**
나누어 이기기 전략의 바탕 경우는 무엇인가? 그것이 알고리즘 전체의 옳음에 어떤 영향을 주는가?

??? success "연습문제 4 풀이"
    밑 자리는 더 나눌 수 없을 만큼 작은 들임을 다룬다(흔히 $n \leq 1$이나 $n \leq 2$). 이때는 옳은 결과를 곧바로 돌려주어야 한다. 밑 자리가 제대로 없으면 되부름이 끝나지 않는다. 밑 자리를 더 크게 잡고($n \leq 10$ 따위) 더 단순한 알고리즘으로 갈아타면 같은 점근 복잡도를 지키면서 되부름 덤을 줄여 참으로 더 빠르게 할 수 있다. $\square$
