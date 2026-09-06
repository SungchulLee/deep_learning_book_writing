# 반씩 자르기 방법

이분 찾기는 띄엄띄엄한 정렬 배열에서 값을 찾는다. **반씩 자르기 방법**은 이 생각을 이어진 함수로 넓힌다. 곧 구간 $[a, b]$에서 부호가 바뀌는 이어진 함수 $f$이 주어질 때 구간을 거듭 반으로 잘라 뿌리를 찾는다. 반씩 자르기 방법은 이분 찾기의 이어진 판이며 수치 살피기에서 가장 믿음직한 뿌리 찾기 알고리즘 가운데 하나이다.

## 문제 서술

Given a continuous function $f: [a, b] \to \mathbb{R}$ with $f(a) \cdot f(b) < 0$ (i.e., $f$ has opposite signs at the endpoints), find a value $c \in [a, b]$ such that $f(c) = 0$.

The existence of such a root is guaranteed by the **Intermediate Value Theorem**: if $f$ is continuous on $[a, b]$ and $f(a) \cdot f(b) < 0$, then there exists at least one $c \in (a, b)$ with $f(c) = 0$.

## 알고리즘

단계마다 반씩 자르기 방법은 가운뎃점 $m = (a + b) / 2$을 셈하고 $f(m)$을 값매김한다:

- $f(m) = 0$이면 $m$이 뿌리이다.
- If $f(a) \cdot f(m) < 0$, the root lies in $[a, m]$, so set $b = m$.
- If $f(m) \cdot f(b) < 0$, the root lies in $[m, b]$, so set $a = m$.

The process repeats until the interval width $b - a$ falls below a specified tolerance $\epsilon$.

### 파이썬 구현

```python
def bisection(f, a, b, tol=1e-10, max_iter=100):
    """
    반씩 자르기 방법으로 [a, b]에서 f의 뿌리 찾기.

    매개변수
    ----------
    f : callable
        f(a) * f(b) < 0인 이어진 함수.
    a : float
        구간의 왼쪽 끝점.
    b : float
        구간의 오른쪽 끝점.
    tol : float
        구간 너비에 대한 모임 너그러움.
    max_iter : int
        최대 바퀴 수.

    반환값
    -------
    float
        f의 어림 뿌리.

    일으키는 예외
    ------
    ValueError
        f(a)와 f(b)의 부호가 같을 때.
    """
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs")

    for _ in range(max_iter):
        mid = (a + b) / 2.0
        fm = f(mid)

        if fm == 0.0 or (b - a) / 2.0 < tol:
            return mid

        if fa * fm < 0:
            b = mid
            fb = fm
        else:
            a = mid
            fa = fm

    return (a + b) / 2.0
```

## 모임 분석

### 어긋남의 한계

$k$바퀴 뒤 구간의 너비는 다음과 같다

$$
b_k - a_k = \frac{b_0 - a_0}{2^k}
$$

여기서 $[a_0, b_0]$은 처음 구간이다. 가운뎃점 $m_k = (a_k + b_k) / 2$은 다음을 채운다

$$
|m_k - c^*| \le \frac{b_0 - a_0}{2^{k+1}}
$$

where $c^*$ is the true root.

### 너그러움에 이르는 바퀴 수

To achieve $|m_k - c^*| < \epsilon$, we need

$$
\frac{b_0 - a_0}{2^{k+1}} < \epsilon
$$

$k$에 대해 풀면:

$$
k > \log_2\!\left(\frac{b_0 - a_0}{\epsilon}\right) - 1
$$

따라서 필요한 바퀴 수는 다음과 같다

$$
k = \left\lceil \log_2\!\left(\frac{b_0 - a_0}{\epsilon}\right) \right\rceil
$$

### 수렴 속도

반씩 자르기 방법은 비율 $1/2$으로 **한 줄로** 모인다. 바퀴마다 결과에 대략 한 비트의 정확도가 더해진다. 이는 뉴턴 방법(제곱으로 모임)보다 느리지만, 반씩 자르기는 모임이 보장되는 반면 뉴턴 방법은 시작점을 잘못 고르면 흩어질 수 있다.

!!! note "모임이 보장됨"
    뉴턴 방법이나 할선 방법과 달리 반씩 자르기는 결코 어그러지지 않는다. 곧 부호 바뀜 조건을 채우는 어떤 이어진 함수에도 모인다. 이 튼튼함 덕분에 더 빠른 방법이 흔들릴 때 믿고 기댈 수 있다.

## 풀이 예제

Find a root of $f(x) = x^3 - x - 2$ on $[1, 2]$.

$f(1) = 1 - 1 - 2 = -2 < 0$이고 $f(2) = 8 - 2 - 2 = 4 > 0$이므로 $[1, 2]$에 뿌리가 있다.

| 바퀴 | $a$ | $b$ | $m$ | $f(m)$ | 하는 일 |
|---|---|---|---|---|---|
| 1 | 1.000 | 2.000 | 1.500 | $-0.125$ | $f(a) \cdot f(m) > 0$, set $a = 1.5$ |
| 2 | 1.500 | 2.000 | 1.750 | $1.609$ | $f(a) \cdot f(m) < 0$, set $b = 1.75$ |
| 3 | 1.500 | 1.750 | 1.625 | $0.666$ | $f(a) \cdot f(m) < 0$, set $b = 1.625$ |
| 4 | 1.500 | 1.625 | 1.5625 | $0.252$ | $f(a) \cdot f(m) < 0$, set $b = 1.5625$ |
| 5 | 1.500 | 1.5625 | 1.5313 | $0.059$ | $f(a) \cdot f(m) < 0$, set $b = 1.5313$ |

After 5 iterations, the root is bracketed in $[1.500, 1.531]$, an interval of width $0.031$, consistent with $1.0 / 2^5 = 0.03125$.

The exact root is $c^* \approx 1.5214$, and the approximation after 5 iterations is $m_5 \approx 1.5156$, with error $\approx 0.006$.

## 이분 찾기와의 이음

반씩 자르기 방법은 짜임으로 보아 이분 찾기와 같다:

| 갈래 | 이분 찾기 | 반씩 자르기 |
|---|---|---|
| 자리 | 띄엄띄엄한 정렬 배열 | 이어진 구간 |
| 조건 | 찾는 값과의 견줌 | 함숫값의 부호 |
| 쪼개기 | 가운뎃점 번호 | 구간의 가운뎃점 |
| Convergence | Exact in $O(\log n)$ steps | Approximate: error halves each step |
| 보장 | 찾는 값이 있으면 옳다 | $f$이 이어져 있고 부호가 바뀌면 모인다 |

두 알고리즘 모두 단조성을 써먹어 단계마다 찾을 자리를 반으로 줄인다. [틀 쪽](binary_search_template.md)의 이분 찾기 틀은 반씩 자르기의 띄엄띄엄한 판으로 볼 수 있다.

## 한계

- **뿌리 하나만 찾는다**: $[a, b]$에서 $f$의 뿌리가 여럿이면 반씩 자르기는 하나만 찾는다.
- **Requires sign change**: if $f$ touches zero without crossing (e.g., $f(x) = x^2$ at $x = 0$), bisection cannot detect the root.
- **한 줄로 모임**: 높은 정밀도를 얻으려면 바퀴를 많이 돌아야 한다. 실전에서는 좋은 첫 구간을 잡는 데 쓰고 그 뒤 뉴턴 같은 더 빠른 방법으로 다듬는 경우가 많다.

## 요약

The bisection method applies the divide-and-conquer strategy to continuous root finding. By repeatedly halving an interval where a sign change occurs, it converges to a root at a rate of one bit of accuracy per iteration. The method requires $\lceil \log_2((b-a)/\epsilon) \rceil$ iterations to achieve tolerance $\epsilon$ and is guaranteed to converge for any continuous function satisfying the sign-change condition.

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), 4장. MIT Press.
- Burden, R. L., & Faires, J. D. (2011). *Numerical Analysis* (9th ed.), 2장. Cengage Learning.

## 연습문제

**연습문제 1.**
반씩 자르기 방법의 핵심 생각과 그 시간 복잡도를 설명하여라.

??? success "연습문제 1 풀이"
    Bisection Method applies the divide-and-conquer paradigm: split the problem into smaller subproblems, solve them recursively, and combine the results. The time complexity is determined by the recurrence relation governing the subproblem sizes and the combination cost. The Master Theorem or recursion tree analysis typically gives the closed-form complexity. $\square$

---

**연습문제 2.**
반씩 자르기 방법의 되돌이 관계식을 쓰고 마스터 정리로 풀어라.

??? success "연습문제 2 풀이"
    The recurrence depends on the specific algorithm's division strategy (number of subproblems $a$, size reduction factor $b$, and combination cost $f(n)$). Apply the Master Theorem: compare $f(n)$ with $n^{\log_b a}$ to determine which case applies. If $f(n) = \Theta(n^{\log_b a})$ (case 2), $T(n) = \Theta(n^{\log_b a} \log n)$. $\square$

---

**연습문제 3.**
반씩 자르기 방법이 막무가내 방식보다 나은 장면을 설명하여라. 얼마나 빨라지는지 수로 나타내어라.

??? success "연습문제 3 풀이"
    The brute-force approach typically runs in $O(n^2)$ or worse. The divide-and-conquer approach achieves a lower complexity by reducing redundant computation through recursive decomposition. For input size $n = 10^6$, the difference between $O(n^2) = 10^{12}$ and $O(n \log n) = 2 \times 10^7$ operations is a factor of $50{,}000$. $\square$

---

**연습문제 4.**
반씩 자르기 방법의 바탕 경우는 무엇인가? 그것이 알고리즘 전체의 옳음에 어떤 영향을 주는가?

??? success "연습문제 4 풀이"
    Base cases handle inputs too small to subdivide further (typically $n \leq 1$ or $n \leq 2$). They must return correct results directly. Without proper base cases, the recursion never terminates. Choosing a larger base case (e.g., $n \leq 10$) and switching to a simpler algorithm can improve practical performance by reducing recursion overhead while maintaining the same asymptotic complexity. $\square$
