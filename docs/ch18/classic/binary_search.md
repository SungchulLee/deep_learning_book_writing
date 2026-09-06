# 이분 찾기

Searching for a specific value in an unsorted array requires examining every element, taking $O(n)$ time in the worst case. When the array is **sorted**, however, we can exploit the ordering to eliminate half of the remaining elements at each step. This idea -- binary search -- is one of the simplest and most powerful applications of the divide-and-conquer paradigm, reducing the search time from $O(n)$ to $O(\log n)$.

## 문제 서술

Given a sorted array $A[0 \,..\, n-1]$ in non-decreasing order and a target value $x$, determine whether $x$ is present in $A$. If so, return an index $i$ such that $A[i] = x$; otherwise, indicate that $x$ is absent.

## 알고리즘

Binary search maintains two pointers, $l$ (left) and $r$ (right), that bound the subarray where $x$ could reside. At each step, it computes the midpoint $m = \lfloor (l + r) / 2 \rfloor$ and compares $A[m]$ with $x$:

- $A[m] = x$이면 찾기가 성공한다.
- If $A[m] < x$, then $x$ can only be in $A[m+1 \,..\, r]$, so set $l = m + 1$.
- If $A[m] > x$, then $x$ can only be in $A[l \,..\, m-1]$, so set $r = m - 1$.

$l > r$이면 찾을 자리가 비었다는 뜻이고 $x$이 배열에 없으므로 찾기가 끝난다.

### 의사 코드

```
BINARY-SEARCH(A, x):
    l = 0
    r = n - 1
    while l <= r:
        m = floor((l + r) / 2)
        if A[m] == x:
            return m
        else if A[m] < x:
            l = m + 1
        else:
            r = m - 1
    return NOT-FOUND
```

### 파이썬 구현

```python
def binary_search(arr, target):
    """
    정렬된 배열에서 값 찾기.

    매개변수
    ----------
    arr : list
        견줄 수 있는 원소를 정렬한 목록.
    target : comparable
        찾을 값.

    반환값
    -------
    int or None
        찾으면 그 번호, 없으면 None.
    """
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = left + (right - left) // 2  # 정수 넘침을 피한다
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return None
```

!!! tip "정수 넘침 피하기"
    가운뎃점을 `(left + right) // 2` 대신 `left + (right - left) // 2`으로 셈하면 자릿수가 정해진 정수를 쓰는 말(보기로 C, 자바)에서 정수 넘침을 막는다. 파이썬은 정수의 자릿수에 제한이 없어 두 식 모두 맞지만, 더 안전한 꼴을 버릇 들이는 것이 좋다.

## 올바름

**되풀이 불변량**으로 옳음을 증명한다.

**Loop invariant.** At the start of each iteration of the `while` loop, if $x$ is in $A$, then $x \in A[l \,..\, r]$.

**Initialization.** Before the first iteration, $l = 0$ and $r = n - 1$, so the invariant holds trivially: if $x$ is in $A$, it is in $A[0 \,..\, n-1]$.

**Maintenance.** Suppose the invariant holds at the start of an iteration. We compute $m = \lfloor (l + r) / 2 \rfloor$.

- $A[m] = x$이면 $m$을 돌려주며, 이는 옳다.
- If $A[m] < x$, then because $A$ is sorted, $x \notin A[l \,..\, m]$. Setting $l = m + 1$ preserves the invariant.
- If $A[m] > x$, then $x \notin A[m \,..\, r]$. Setting $r = m - 1$ preserves the invariant.

**Termination.** The loop terminates when $l > r$. By the invariant, if $x$ were in $A$, it would be in $A[l \,..\, r]$. But $l > r$ means this subarray is empty, so $x \notin A$. Returning `NOT-FOUND` is correct. $\square$

## 복잡도 분석

### 시간 복잡도

Each iteration halves the search space. After $k$ iterations, the remaining search space has size at most $\lfloor n / 2^k \rfloor$. The loop terminates when this size drops to zero, which happens when $2^k > n$, i.e., after $k = \lfloor \log_2 n \rfloor + 1$ iterations.

바퀴마다 $O(1)$의 품(견줌 한 번, 가운뎃점 셈 한 번, 가리개 고침 한 번)이 드므로 전체 시간은 다음과 같다

$$
T(n) = O(\log n)
$$

되돌이 관계식으로 보면, 이분 찾기는 크기 $n/2$인 작은 문제 하나를 $O(1)$의 덧짐으로 푼다:

$$
T(n) = T\!\left(\frac{n}{2}\right) + O(1)
$$

By the Master Theorem (case 2, with $a = 1$, $b = 2$, $\log_b a = 0$, and $f(n) = O(1) = O(n^0)$):

$$
T(n) = O(\log n)
$$

### 공간 복잡도

The iterative implementation uses $O(1)$ auxiliary space. The recursive version (covered on the [Binary Search - Recursive](binary_search_recursive.md) page) uses $O(\log n)$ stack space.

### 아래 한계

Any comparison-based search algorithm on a sorted array must make at least $\lceil \log_2(n + 1) \rceil$ comparisons in the worst case, because each comparison yields at most one bit of information, and there are $n + 1$ possible outcomes (found at one of $n$ positions, or not found). Binary search matches this lower bound and is therefore **optimal** among comparison-based search algorithms.

## 풀이 예제

$n = 10$인 정렬된 배열 $A = [2, 5, 8, 12, 16, 23, 38, 56, 72, 91]$에서 $x = 23$을 찾는다고 하자.

| 바퀴 | $l$ | $r$ | $m$ | $A[m]$ | 하는 일 |
|---|---|---|---|---|---|
| 1 | 0 | 9 | 4 | 16 | $16 < 23$, $l = 5$으로 |
| 2 | 5 | 9 | 7 | 56 | $56 > 23$, $r = 6$으로 |
| 3 | 5 | 6 | 5 | 23 | $23 = 23$, $5$을 돌려줌 |

The search finds $x = 23$ at index 5 in 3 iterations, consistent with $\lfloor \log_2 10 \rfloor + 1 = 4$ maximum iterations.

## 나누어 이기기 관점

이분 찾기는 짜임이 오그라든 나누어 이기기 알고리즘이다:

- **나누기**: 가운뎃점을 $O(1)$에 셈한다.
- **이기기**: 작은 문제 정확히 하나(왼쪽 반이나 오른쪽 반)에 되돌이한다.
- **아우르기**: 할 일이 없다. 곧 작은 문제의 답이 본디 문제의 답이다.

Because only one subproblem is solved at each level ($a = 1$), the total work is proportional to the depth of the recursion, which is $O(\log n)$.

## 요약

Binary search exploits the sorted order of an array to halve the search space at each step, achieving $O(\log n)$ time complexity. Its correctness follows from a simple loop invariant, and its efficiency matches the information-theoretic lower bound for comparison-based search.

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), 2장. MIT Press.

## 연습문제

**연습문제 1.**
정렬된 배열 $[2, 5, 8, 12, 16, 23, 38, 56, 72, 91]$에서 23을 찾는 이분 찾기를 좇아라. 견줌은 몇 번 필요한가?

??? success "연습문제 1 풀이"
    Low=0, high=9, mid=4 (value 16 < 23). Low=5, high=9, mid=7 (value 56 > 23). Low=5, high=6, mid=5 (value 23 = 23). Found at index 5 in 3 comparisons. Binary search halves the search space each time, so at most $\lceil \log_2 10 \rceil = 4$ comparisons are needed for 10 elements. $\square$

---

**연습문제 2.**
Prove that binary search on a sorted array of $n$ elements runs in $O(\log n)$ time.

??? success "연습문제 2 풀이"
    At each step, the search range $[\text{low}, \text{high}]$ is halved. Starting with $n$ elements, after $k$ steps the range has at most $n/2^k$ elements. The algorithm terminates when the range is empty or the target is found. The range becomes empty when $n/2^k < 1$, i.e., $k > \log_2 n$. Therefore at most $\lceil \log_2 n \rceil + 1$ iterations are needed: $O(\log n)$. $\square$

---

**연습문제 3.**
같은 값이 있는 정렬된 배열에서 찾는 값이 처음 나오는 자리를 찾도록 이분 찾기를 고쳐라.

??? success "연습문제 3 풀이"
    `arr[mid] == target`일 때 바로 돌려주지 않는다. 그 대신 `result = mid`으로 두고 더 앞의 것을 찾으러 왼쪽으로 이어 간다(`high = mid - 1`). 되풀이가 끝나면 `result`에 가장 왼쪽 번호가 들어 있다.

    ```python
    def first_occurrence(arr, target):
        low, high, result = 0, len(arr) - 1, -1
        while low <= high:
            mid = (low + high) // 2
            if arr[mid] == target:
                result = mid
                high = mid - 1
            elif arr[mid] < target:
                low = mid + 1
            else:
                high = mid - 1
        return result
    ```
    $\square$

---

**연습문제 4.**
Prove that binary search is optimal: no comparison-based search on a sorted array can do better than $\Omega(\log n)$ in the worst case.

??? success "연습문제 4 풀이"
    Any comparison-based algorithm can be modeled as a binary decision tree. Each internal node represents a comparison with two outcomes. To distinguish among $n$ possible positions, the tree must have at least $n$ leaves. A binary tree with $n$ leaves has height at least $\lceil \log_2 n \rceil$. The worst case follows the longest root-to-leaf path, requiring at least $\lceil \log_2 n \rceil$ comparisons. Therefore $\Omega(\log n)$ comparisons are necessary. $\square$
