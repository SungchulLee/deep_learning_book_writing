# 이분 찾기 틀

보통의 이분 찾기는 정렬된 배열에서 특정한 값을 찾는다. 그러나 여러 문제는 조건이 거짓에서 참으로(또는 그 반대로) 바뀌는 **경계**를 찾아야 한다. 넓힌 이분 찾기 틀은 그런 문제를 모두 한결같이 다룬다. 곧 단조 술어가 주어질 때 그것을 채우는 가장 작은(또는 가장 큰) 번호를 찾는다.

이 틀은 이분 찾기를 올바로 짜기 어렵게 만드는, 어긋나기 쉬운 테두리 다루기의 세부, 곧 하나 차이 어긋남, 끝나지 않는 되풀이, 잘못된 가운뎃점 반올림을 없애 준다.

## 넓힌 문제

Suppose we have a search space $\{0, 1, \ldots, n-1\}$ and a boolean predicate $\text{condition}(m)$ that is **monotone**: there exists a threshold $k$ such that

$$
\text{condition}(m) = \begin{cases} \text{false} & \text{if } m < k \\ \text{true} & \text{if } m \ge k \end{cases}
$$

The goal is to find the smallest $m$ for which $\text{condition}(m)$ is true. This is the **leftmost true** problem.

## 틀

```python
def binary_search_template(lo, hi, condition):
    """
    [lo, hi]에서 조건을 채우는 가장 작은 값 찾기.

    매개변수
    ----------
    lo : int
        찾을 자리의 아래 테두리(포함).
    hi : int
        찾을 자리의 위 테두리(포함).
    condition : callable
        단조 술어. 곧 문턱값 아래에서는 거짓,
        문턱값 위와 그 자리에서는 참이다.

    반환값
    -------
    int
        [lo, hi]에서 다음을 채우는 가장 작은 값 m,
        condition(m)이 참인 값. 그런 값이 없으면 hi + 1.
    """
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if condition(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo
```

### 핵심 설계 선택

1. **되풀이 조건 `lo < hi`**(`lo <= hi`가 아님): `lo == hi`일 때 되풀이가 끝나고 그때 답은 `lo`이다.
2. **`hi = mid`**(`hi = mid - 1`이 아님): 조건이 참이면 `mid`이 답일 수 있으므로 찾을 자리에 남겨 둔다.
3. **`lo = mid + 1`**: 조건이 거짓이면 `mid`은 결코 답이 아니므로 뺀다.
4. **`mid = lo + (hi - lo) // 2`**: 내림하므로 `lo < hi`일 때 `mid < hi`가 되어 끝나지 않는 되풀이를 막는다.

!!! warning "끝나지 않는 되풀이 덫"
    `mid = lo + (hi - lo) // 2`에 `hi = mid - 1`을 함께 쓰면 되풀이가 답을 놓칠 수 있다. 같은 가운뎃점 식에 `lo = mid`을 쓰면 `hi - lo == 1`일 때 되풀이가 끝나지 않는다. 위의 틀은 두 함정을 모두 피한다.

## 옳음의 증명

**Loop invariant.** At the start of each iteration, the answer (the smallest $m$ with $\text{condition}(m)$ true) lies in $[\text{lo}, \text{hi}]$.

**첫자리매김.** 처음 범위가 찾을 자리 전체를 덮으므로 불변량이 성립한다.

**Maintenance.** Let $\text{mid} = \lfloor (\text{lo} + \text{hi}) / 2 \rfloor$.

- If $\text{condition}(\text{mid})$ is true, then the answer is at most $\text{mid}$, so setting $\text{hi} = \text{mid}$ preserves the invariant.
- If $\text{condition}(\text{mid})$ is false, then the answer is at least $\text{mid} + 1$, so setting $\text{lo} = \text{mid} + 1$ preserves the invariant.

**Termination.** The quantity $\text{hi} - \text{lo}$ is a non-negative integer that strictly decreases at each iteration (because $\text{mid} < \text{hi}$ when $\text{lo} < \text{hi}$). When $\text{lo} = \text{hi}$, the loop terminates, and the invariant guarantees `lo` is the answer. $\square$

**Time complexity.** The search space halves at each iteration, so the template performs $O(\log(\text{hi} - \text{lo}))$ iterations. If each call to `condition` takes $O(C)$ time, the total is $O(C \log(\text{hi} - \text{lo}))$.

## 가장 오른쪽 거짓 변종

To find the **largest** $m$ for which $\text{condition}(m)$ is false, use a mirror template:

```python
def binary_search_rightmost_false(lo, hi, condition):
    """
    [lo, hi]에서 조건이 거짓인 가장 큰 값 찾기.

    모든 값에 대해 조건이 참이면 lo - 1을 돌려준다.
    """
    while lo < hi:
        mid = lo + (hi - lo + 1) // 2  # 올림
        if condition(mid):
            hi = mid - 1
        else:
            lo = mid
    return lo
```

`lo = mid`일 때 끝나지 않는 되풀이를 막으려 가운뎃점을 **올림**한다(`(hi - lo + 1) // 2`)는 점에 유의하라.

## 응용

### 넣을 자리 찾기

정렬된 배열에서 정렬을 지키려면 `target`을 어느 번호에 넣어야 하는지 찾아라.

```python
def search_insert(nums, target):
    """정렬된 배열에서 찾는 값을 넣을 자리 찾기."""
    return binary_search_template(
        0, len(nums),
        lambda mid: mid == len(nums) or nums[mid] >= target
    )
```

### 정수 제곱근

Find the largest integer $k$ such that $k^2 \le x$.

```python
def integer_sqrt(x):
    """이분 찾기로 floor(sqrt(x)) 셈하기."""
    if x < 0:
        raise ValueError("Square root of negative number")
    if x == 0:
        return 0
    # (k+1)^2 > x인 가장 작은 k를 찾아 k 돌려주기
    return binary_search_template(
        1, x,
        lambda mid: mid * mid > x
    ) - 1
```

### 처음 나쁜 판

$1$부터 $n$까지 번호가 매겨진 판 $n$개와, 단조인(처음 나쁜 판 뒤의 모든 판도 나쁜) 함수 `is_bad(v)`가 주어질 때 처음 나쁜 판을 찾아라.

```python
def first_bad_version(n, is_bad):
    """1..n번 판 가운데 처음 나쁜 판 찾기."""
    return binary_search_template(1, n, is_bad)
```

### 짐을 실어 나를 담이

$d$일 안에 모든 짐을 나르는 데 필요한 최소 배 담이를 찾아라. "담이 $c$으로 $d$일 안에 모든 짐을 나를 수 있다"라는 술어는 $c$에 대해 단조이다.

```python
def ship_within_days(weights, days):
    """주어진 날 안에 모든 무게를 나를 최소 담이 찾기."""
    def can_ship(capacity):
        day_count, current_load = 1, 0
        for w in weights:
            if current_load + w > capacity:
                day_count += 1
                current_load = 0
            current_load += w
        return day_count <= days

    return binary_search_template(
        max(weights), sum(weights), can_ship
    )
```

## 이 틀을 쓸 때

문제가 다음 성질을 가질 때면 언제나 이 틀을 쓸 수 있다:

1. **단조 술어**: 찾을 자리 전체에서 조건이 거짓에서 참으로 정확히 한 번 바뀐다.
2. **Bounded search space**: the range $[\text{lo}, \text{hi}]$ is known in advance.
3. **효율적인 값매김**: `condition(mid)`을 살피는 데 다항 시간이 든다.

!!! tip "이분 찾기 문제 알아보기"
    문제가 "X를 채우는 최솟값"이나 "Y를 넘지 않는 최댓값"을 묻고 X나 Y가 답에 대해 단조라면, 답을 두고 이분 찾기를 하는 것이 알맞을 가능성이 높다.

## 요약

The generalized binary search template reduces all binary search variants to a single pattern: define a monotone predicate, set the search bounds, and let the template find the transition point. The correctness proof relies on a loop invariant showing that the answer always lies within the current bounds, and termination follows from the strictly decreasing search space. The template runs in $O(\log n)$ iterations, each calling the predicate once.

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), 2장. MIT Press.

## 연습문제

**연습문제 1.**
이분 찾기 틀의 핵심 생각과 그 시간 복잡도를 설명하여라.

??? success "연습문제 1 풀이"
    Binary Search Template applies the divide-and-conquer paradigm: split the problem into smaller subproblems, solve them recursively, and combine the results. The time complexity is determined by the recurrence relation governing the subproblem sizes and the combination cost. The Master Theorem or recursion tree analysis typically gives the closed-form complexity. $\square$

---

**연습문제 2.**
이분 찾기 틀의 되돌이 관계식을 쓰고 마스터 정리로 풀어라.

??? success "연습문제 2 풀이"
    The recurrence depends on the specific algorithm's division strategy (number of subproblems $a$, size reduction factor $b$, and combination cost $f(n)$). Apply the Master Theorem: compare $f(n)$ with $n^{\log_b a}$ to determine which case applies. If $f(n) = \Theta(n^{\log_b a})$ (case 2), $T(n) = \Theta(n^{\log_b a} \log n)$. $\square$

---

**연습문제 3.**
이분 찾기 틀이 막무가내 방식보다 나은 장면을 설명하여라. 얼마나 빨라지는지 수로 나타내어라.

??? success "연습문제 3 풀이"
    The brute-force approach typically runs in $O(n^2)$ or worse. The divide-and-conquer approach achieves a lower complexity by reducing redundant computation through recursive decomposition. For input size $n = 10^6$, the difference between $O(n^2) = 10^{12}$ and $O(n \log n) = 2 \times 10^7$ operations is a factor of $50{,}000$. $\square$

---

**연습문제 4.**
이분 찾기 틀의 바탕 경우는 무엇인가? 그것이 알고리즘 전체의 옳음에 어떤 영향을 주는가?

??? success "연습문제 4 풀이"
    Base cases handle inputs too small to subdivide further (typically $n \leq 1$ or $n \leq 2$). They must return correct results directly. Without proper base cases, the recursion never terminates. Choosing a larger base case (e.g., $n \leq 10$) and switching to a simpler algorithm can improve practical performance by reducing recursion overhead while maintaining the same asymptotic complexity. $\square$
