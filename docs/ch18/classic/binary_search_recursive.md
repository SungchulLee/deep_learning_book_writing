# 이분 찾기 — 되돌이

[되풀이 이분 찾기](binary_search.md)는 `while` 되풀이로 찾을 자리를 좁힌다. 이와 같은 **되돌이** 세우기는 같은 논리를, 더 작은 부분 배열에 스스로를 부르는 함수로 나타낸다. 되돌이 판은 이분 찾기의 나누어 이기기 짜임을 드러낸다. 곧 부름마다 배열을 반으로 나누고, 한쪽 반에 되돌이해 이기며, 결과를 그대로 돌려주어 아우른다.

이 쪽에서는 되돌이 세우기를 내놓고, 짜임 귀납법으로 옳음을 증명하며, 시간과 공간 복잡도를 살피고, 되풀이 판과 견준다.

## 되돌이로 세우기

되돌이 이분 찾기는 배열 $A$, 찾는 값 $x$, 지금의 테두리 $l$과 $r$을 매개변수로 받는다.

### 의사 코드

```
RECURSIVE-BINARY-SEARCH(A, x, l, r):
    if l > r:
        return NOT-FOUND
    m = floor((l + r) / 2)
    if A[m] == x:
        return m
    else if A[m] < x:
        return RECURSIVE-BINARY-SEARCH(A, x, m + 1, r)
    else:
        return RECURSIVE-BINARY-SEARCH(A, x, l, m - 1)
```

첫 부름은 `RECURSIVE-BINARY-SEARCH(A, x, 0, n - 1)`이다.

### 파이썬 구현

```python
def binary_search_recursive(arr, target, left=None, right=None):
    """
    정렬된 배열에서 되돌이로 값 찾기.

    매개변수
    ----------
    arr : list
        견줄 수 있는 원소를 정렬한 목록.
    target : comparable
        찾을 값.
    left : int, optional
        찾을 범위의 왼쪽 테두리(붙박이: 0).
    right : int, optional
        찾을 범위의 오른쪽 테두리(붙박이: len(arr) - 1).

    반환값
    -------
    int or None
        찾으면 그 번호, 없으면 None.
    """
    if left is None:
        left = 0
    if right is None:
        right = len(arr) - 1

    if left > right:
        return None

    mid = left + (right - left) // 2

    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right)
    else:
        return binary_search_recursive(arr, target, left, mid - 1)
```

## 짜임 귀납법으로 보는 옳음

찾을 자리의 크기 $s = r - l + 1$에 대한 강한 귀납법으로 옳음을 증명한다.

**Base case** ($s \le 0$). When $l > r$, the search space is empty. If $x$ were in $A[l \,..\, r]$, this subarray would be non-empty, so returning `NOT-FOUND` is correct.

**Inductive step.** Assume the algorithm is correct for all search spaces of size less than $s$. Consider a call with search space of size $s = r - l + 1 > 0$. Compute $m = \lfloor (l + r) / 2 \rfloor$.

- $A[m] = x$이면 $m$을 돌려주는 것이 옳다.
- If $A[m] < x$: because $A$ is sorted, $x \notin A[l \,..\, m]$. The recursive call on $A[m+1 \,..\, r]$ has search space of size $r - m \le s - 1 < s$. By the inductive hypothesis, this call returns the correct answer.
- If $A[m] > x$: symmetrically, the recursive call on $A[l \,..\, m-1]$ has search space of size $m - l \le s - 1 < s$, and is correct by the inductive hypothesis.

In all cases, the algorithm returns the correct result. $\square$

## 복잡도 분석

### 시간 복잡도

되돌이 이분 찾기는 다음 되돌이 관계식을 채운다

$$
T(n) = T\!\left(\frac{n}{2}\right) + O(1)
$$

with base case $T(0) = O(1)$. By the Master Theorem ($a = 1$, $b = 2$, $f(n) = O(1) = O(n^0)$, case 2):

$$
T(n) = O(\log n)
$$

이는 되풀이 판과 정확히 같다.

### 공간 복잡도

Each recursive call adds a frame to the call stack. Because the recursion depth is $O(\log n)$ (the search space halves at each call), the space complexity is

$$
S(n) = O(\log n)
$$

This is the key difference from the iterative version, which uses $O(1)$ auxiliary space. In practice, the $O(\log n)$ stack depth is small (e.g., $\log_2 10^9 \approx 30$ frames), so the overhead is rarely a concern.

!!! note "꼬리 부름 다듬기"
    The recursive call is in **tail position** -- it is the last operation before the function returns. Languages that support tail call optimization (TCO), such as Scheme or certain C compilers with optimization flags, can transform the recursion into a loop, eliminating the stack overhead entirely. Python does not support TCO, so the $O(\log n)$ stack usage applies.

## 견줌: 되풀이와 되돌이

| 성질 | 되풀이 | 되돌이 |
|---|---|---|
| Time complexity | $O(\log n)$ | $O(\log n)$ |
| Space complexity | $O(1)$ | $O(\log n)$ |
| 나누어 이기기 짜임 | 숨어 있음 | 드러남 |
| 꼬리 부름 가능 | 해당 없음 | 가능 |
| Stack overflow risk | None | Theoretical (depth $\approx 30$ for $n = 10^9$) |

두 판 모두 옳고 시간 복잡도가 같다. 실전 코드에서는 공간을 $O(1)$만 쓰는 되풀이 판을 대개 더 낫게 여긴다. 되돌이 판은 나누어 이기기 짜임을 이해하는 데 값지며 더 복잡한 되돌이 알고리즘의 틀이 된다.

## 풀이 예제

$A = [3, 7, 12, 19, 25, 31, 42]$($n = 7$)에서 $x = 12$을 찾는다:

| 부름 | $l$ | $r$ | $m$ | $A[m]$ | 하는 일 |
|---|---|---|---|---|---|
| 1 | 0 | 6 | 3 | 19 | $19 > 12$, $[0, 2]$에 되돌이 |
| 2 | 0 | 2 | 1 | 7 | $7 < 12$, $[2, 2]$에 되돌이 |
| 3 | 2 | 2 | 2 | 12 | $12 = 12$, $2$을 돌려줌 |

찾는 값을 부름 3번 만에 번호 2에서 찾는다. 되돌이가 풀리며 결과 $2$이 틀마다 거슬러 올라간다.

## 요약

Recursive binary search makes the divide-and-conquer structure explicit: each call divides the search space in half, recurses on one half, and returns the result directly. It has the same $O(\log n)$ time complexity as the iterative version but uses $O(\log n)$ stack space. The correctness proof proceeds by strong induction on the search space size.

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), 2장. MIT Press.

## 연습문제

**연습문제 1.**
되돌이 이분 찾기의 되돌이 관계식을 쓰고 풀어라.

??? success "연습문제 1 풀이"
    $T(n) = T(n/2) + O(1)$ with base case $T(1) = O(1)$. By the Master Theorem (case 2 with $a=1, b=2, k=0$): $T(n) = O(\log n)$. Alternatively, unroll: $T(n) = T(n/2) + c = T(n/4) + 2c = \cdots = T(1) + c\log_2 n = O(\log n)$. $\square$

---

**연습문제 2.**
What is the maximum recursion depth for recursive binary search on an array of $10^6$ elements?

??? success "연습문제 2 풀이"
    Recursion depth $= \lceil \log_2(10^6) \rceil = \lceil 19.93 \rceil = 20$. This is well within Python's default recursion limit of 1000. Even for $10^9$ elements, the depth is only $\lceil \log_2(10^9) \rceil = 30$. Recursive binary search is safe from stack overflow for any practical input size. $\square$

---

**연습문제 3.**
공간 복잡도와 실전 성능으로 되풀이 이분 찾기와 되돌이 이분 찾기를 견주어라.

??? success "연습문제 3 풀이"
    **Iterative**: $O(1)$ extra space (only loop variables). No function call overhead. **Recursive**: $O(\log n)$ stack space for recursion frames. Each frame adds function call overhead (parameter passing, return address). In practice, iterative is slightly faster due to avoiding call overhead. Both are $O(\log n)$ time. Iterative is generally preferred in production code; recursive is clearer for teaching. $\square$

---

**연습문제 4.**
찾는 값이 없으면 가장 가까운 원소의 번호를 돌려주도록 되돌이 이분 찾기를 고쳐라.

??? success "연습문제 4 풀이"
    바탕 경우 `low > high` 뒤에 (쓸 수 있으면) `arr[low]`과 `arr[high]`을 찾는 값과 견주어 더 가까운 쪽의 번호를 돌려준다:

    ```python
    def closest_search(arr, target, low, high):
        if low > high:
            if low >= len(arr): return high
            if high < 0: return low
            return low if abs(arr[low] - target) <= abs(arr[high] - target) else high
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            return closest_search(arr, target, mid + 1, high)
        else:
            return closest_search(arr, target, low, mid - 1)
    ```
    $\square$
