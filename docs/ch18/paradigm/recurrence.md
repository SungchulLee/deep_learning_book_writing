# 되돌이 관계 살피기

나누어 이기기 알고리즘은 문제를 작은 문제로 쪼개고 저마다 되돌이로 풀어 그 결과를 아울러 문제를 푼다. 알고리즘이 더 작은 들임에 스스로를 부르므로 그 도는 시간은 **되돌이 관계식**, 곧 $T(n)$을 더 작은 인자에서의 $T$으로 나타내는 식을 채운다. 이 되돌이 관계식을 풀면 알고리즘의 점근 복잡도가 나온다.

이 쪽에서는 나누어 이기기 알고리즘이 어떻게 되돌이 관계식을 낳는지 보이고, 그것을 푸는 주된 재주 셋, 곧 되돌이 나무 방법, 대입 방법, 마스터 정리를 살펴본다.

## 알고리즘에서 되돌이 관계식으로

다음과 같은 나누어 이기기 알고리즘을 보자:

- 크기 $n$인 문제를 크기 $n/b$인 작은 문제 $a$개로 나누고,
- 나누는 데 $D(n)$, 아우르는 데 $C(n)$의 시간을 쓴다.

그 도는 시간은 다음을 채운다

$$
T(n) = \begin{cases} \Theta(1) & \text{if } n \le n_0 \\ a \, T\!\left(\dfrac{n}{b}\right) + f(n) & \text{if } n > n_0 \end{cases}
$$

여기서 $f(n) = D(n) + C(n)$은 켜마다 되돌이가 아닌 전체 품이다. 매개변수 $a$, $b$, $f(n)$이 $T(n)$의 점근 몸짓을 온전히 정한다.

### 보기: 어울러 정렬

어울러 정렬은 배열을 반으로 쪼개고($a = 2$, $b = 2$) 나누는 데 $O(1)$, 어울리는 데 $O(n)$을 쓴다. 그 되돌이 관계식은 다음과 같다

$$
T(n) = 2T\!\left(\frac{n}{2}\right) + \Theta(n)
$$

### 보기: 이분 찾기

이분 찾기는 배열의 한쪽 반만 살피고($a = 1$, $b = 2$) 부름마다 $O(1)$을 쓴다:

$$
T(n) = T\!\left(\frac{n}{2}\right) + \Theta(1)
$$

## 되돌이 나무 방법

**되돌이 나무** 방법은 되돌이 관계식을, 마디마다 작은 문제 하나의 값을 나타내는 나무로 바꾼다. 켜마다의 값을 모두 더하면 전체 도는 시간이 나온다.

되돌이 관계식 $T(n) = aT(n/b) + f(n)$에 대해:

- **켜 0**(뿌리): 크기 $n$인 문제 하나가 $f(n)$을 보탠다.
- **Level 1**: $a$ problems of size $n/b$ each contribute $f(n/b)$, total $a \cdot f(n/b)$.
- **Level $k$**: $a^k$ problems of size $n/b^k$ each contribute $f(n/b^k)$, total $a^k \cdot f(n/b^k)$.
- **Depth**: the recursion bottoms out when $n/b^k \le n_0$, giving depth $k = \log_b n$ (ignoring constant $n_0$).

전체 값은 다음과 같다

$$
T(n) = \sum_{k=0}^{\log_b n} a^k \cdot f\!\left(\frac{n}{b^k}\right)
$$

### 풀어 본 보기: 어울러 정렬

$T(n) = 2T(n/2) + cn$에 대해:

| 켜 | 마디 수 | 마디마다의 크기 | 마디마다의 값 | 켜의 값 |
|---|---|---|---|---|
| $0$ | $1$ | $n$ | $cn$ | $cn$ |
| $1$ | $2$ | $n/2$ | $cn/2$ | $cn$ |
| $2$ | $4$ | $n/4$ | $cn/4$ | $cn$ |
| $k$ | $2^k$ | $n/2^k$ | $cn/2^k$ | $cn$ |

Every level costs $cn$, and there are $\log_2 n$ levels, so

$$
T(n) = cn \cdot \log_2 n = \Theta(n \log n)
$$

## 대입 방법

**대입 방법**은 두 단계로 이루어진다:

1. 풀이의 꼴을 **어림잡는다**(흔히 되돌이 나무에서 실마리를 얻는다).
2. 그 어림이 옳음을 수학적 귀납법으로 **증명한다**.

### 보기: 어울러 정렬이 O(n log n)임을 증명하기

**Claim.** $T(n) \le cn \log n$ for some constant $c > 0$ and all $n \ge 2$.

**Inductive step.** Assume $T(k) \le ck \log k$ for all $k < n$. Then

$$
T(n) = 2T\!\left(\frac{n}{2}\right) + \Theta(n) \le 2 \cdot c \cdot \frac{n}{2} \cdot \log \frac{n}{2} + dn
$$

$$
= cn(\log n - 1) + dn = cn \log n - cn + dn \le cn \log n
$$

provided $c \ge d$. $\square$

!!! warning "대입에서 흔히 빠지는 함정"
    A frequent mistake is guessing $T(n) \le cn$ for merge sort. The inductive step yields $T(n) \le cn + dn$, which does not prove $T(n) \le cn$ because the extra $dn$ term cannot be absorbed. The guess must match the asymptotic form exactly, including logarithmic factors.

## 마스터 정리

**마스터 정리**는 다음 꼴의 되돌이 관계식에 대해 곧바른 식을 준다

$$
T(n) = aT\!\left(\frac{n}{b}\right) + f(n)
$$

where $a \ge 1$ and $b > 1$. The key quantity is the **critical exponent** $\log_b a$, which represents the growth rate of the number of subproblems.

### 세 가지 경우

**Case 1.** If $f(n) = O(n^{\log_b a - \epsilon})$ for some $\epsilon > 0$, then the leaf work dominates:

$$
T(n) = \Theta(n^{\log_b a})
$$

**Case 2.** If $f(n) = \Theta(n^{\log_b a})$, then work is evenly distributed across levels:

$$
T(n) = \Theta(n^{\log_b a} \log n)
$$

**Case 3.** If $f(n) = \Omega(n^{\log_b a + \epsilon})$ for some $\epsilon > 0$, and $af(n/b) \le cf(n)$ for some $c < 1$ (regularity condition), then the root work dominates:

$$
T(n) = \Theta(f(n))
$$

### 마스터 정리 쓰기

| Algorithm | Recurrence | $a$ | $b$ | $\log_b a$ | Case | $T(n)$ |
|---|---|---|---|---|---|---|
| Binary search | $T(n) = T(n/2) + O(1)$ | $1$ | $2$ | $0$ | 2 | $\Theta(\log n)$ |
| Merge sort | $T(n) = 2T(n/2) + O(n)$ | $2$ | $2$ | $1$ | 2 | $\Theta(n \log n)$ |
| Karatsuba | $T(n) = 3T(n/2) + O(n)$ | $3$ | $2$ | $1.585$ | 1 | $\Theta(n^{1.585})$ |
| Strassen | $T(n) = 7T(n/2) + O(n^2)$ | $7$ | $2$ | $2.807$ | 1 | $\Theta(n^{2.807})$ |

### 마스터 정리를 쓸 수 없을 때

The Master Theorem requires $f(n)$ to be **polynomially smaller or larger** than $n^{\log_b a}$. It does not cover cases where $f(n)$ differs by a logarithmic factor. For example, the recurrence

$$
T(n) = 2T\!\left(\frac{n}{2}\right) + n \log n
$$

은 2번과 3번 경우 사이의 틈에 놓인다. **넓힌 마스터 정리**(아크라-바지 방법)가 그런 경우를 다룬다. [되돌이 관계식을 자세히 다룬 장](../../ch02/recurrences/akra_bazzi.md)을 보라.

## 실용적인 고려

### 바닥과 천장

Real algorithms split arrays at $\lfloor n/2 \rfloor$ and $\lceil n/2 \rceil$, not exactly $n/2$. The standard approach is to solve the recurrence assuming exact division and then verify that floors and ceilings do not change the asymptotic result. For the Master Theorem, this assumption is provably safe.

### 바탕 경우의 상수 인자

The base case $T(n_0) = \Theta(1)$ absorbs implementation-dependent constants. Changing the base case threshold (e.g., switching to insertion sort for $n \le 32$) does not alter the asymptotic solution but can significantly affect practical performance.

## 요약

나누어 이기기 알고리즘마다 되돌이 관계식 $T(n) = aT(n/b) + f(n)$을 낳는다. 그런 되돌이 관계식을 푸는 방법이 셋 있다:

1. **되돌이 나무**: 켜마다의 값을 그려 보고 모든 켜에 걸쳐 더한다.
2. **대입**: 답을 어림잡고 귀납법으로 증명한다.
3. **Master Theorem**: compare $f(n)$ to $n^{\log_b a}$ and read off the answer.

마스터 정리는 쓸 수 있을 때 가장 빠르지만, 되돌이 나무 방법은 마스터 정리가 주지 못하는 직관을 주고, 대입 방법은 마스터 정리가 놓치는 경우를 다룬다.

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), 4장. MIT Press.

## 연습문제

**연습문제 1.**
되돌이 관계 살피기의 핵심 생각과 그 시간 복잡도를 설명하여라.

??? success "연습문제 1 풀이"
    Recurrence Analysis applies the divide-and-conquer paradigm: split the problem into smaller subproblems, solve them recursively, and combine the results. The time complexity is determined by the recurrence relation governing the subproblem sizes and the combination cost. The Master Theorem or recursion tree analysis typically gives the closed-form complexity. $\square$

---

**연습문제 2.**
되돌이 관계 살피기의 되돌이 관계식을 쓰고 마스터 정리로 풀어라.

??? success "연습문제 2 풀이"
    The recurrence depends on the specific algorithm's division strategy (number of subproblems $a$, size reduction factor $b$, and combination cost $f(n)$). Apply the Master Theorem: compare $f(n)$ with $n^{\log_b a}$ to determine which case applies. If $f(n) = \Theta(n^{\log_b a})$ (case 2), $T(n) = \Theta(n^{\log_b a} \log n)$. $\square$

---

**연습문제 3.**
되돌이 관계 살피기이 막무가내 방식보다 나은 장면을 설명하여라. 얼마나 빨라지는지 수로 나타내어라.

??? success "연습문제 3 풀이"
    The brute-force approach typically runs in $O(n^2)$ or worse. The divide-and-conquer approach achieves a lower complexity by reducing redundant computation through recursive decomposition. For input size $n = 10^6$, the difference between $O(n^2) = 10^{12}$ and $O(n \log n) = 2 \times 10^7$ operations is a factor of $50{,}000$. $\square$

---

**연습문제 4.**
되돌이 관계 살피기의 바탕 경우는 무엇인가? 그것이 알고리즘 전체의 옳음에 어떤 영향을 주는가?

??? success "연습문제 4 풀이"
    Base cases handle inputs too small to subdivide further (typically $n \leq 1$ or $n \leq 2$). They must return correct results directly. Without proper base cases, the recursion never terminates. Choosing a larger base case (e.g., $n \leq 10$) and switching to a simpler algorithm can improve practical performance by reducing recursion overhead while maintaining the same asymptotic complexity. $\square$
