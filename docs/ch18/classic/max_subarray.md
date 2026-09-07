# 최대 부분 배열

음수가 들어 있을 수 있는 수의 배열이 주어질 때 **최대 부분 배열 문제**는 합이 가장 큰 잇닿은 부분 배열을 묻는다. 이 문제는 금융 살피기(가장 이득이 큰 거래 기간 찾기), 신호 다루기(가장 센 신호 토막 찾기), 유전체학(생물학적으로 뜻있는 자리 가려내기)에서 자연스레 나타난다.

While Kadane's algorithm solves the problem in $O(n)$ time using dynamic programming, the divide-and-conquer approach provides an elegant $O(n \log n)$ solution that illustrates the paradigm's mechanics clearly, especially the combine step.

## 문제 서술

Given an array $A[0 \,..\, n-1]$ of real numbers, find indices $i$ and $j$ with $0 \le i \le j \le n-1$ that maximize

$$
\sum_{k=i}^{j} A[k]
$$

원소가 모두 음수이면 최대 부분 배열은 값이 가장 큰 원소 하나이다.

## 나누어 이기기 방식

The key insight is that a maximum subarray of $A[\text{lo} \,..\, \text{hi}]$ must lie in exactly one of three positions relative to the midpoint $\text{mid} = \lfloor (\text{lo} + \text{hi}) / 2 \rfloor$:

1. **Entirely in the left half**: $A[\text{lo} \,..\, \text{mid}]$
2. **Entirely in the right half**: $A[\text{mid}+1 \,..\, \text{hi}]$
3. **Crossing the midpoint**: some $A[i \,..\, j]$ with $i \le \text{mid} < j$

1번과 2번 경우는 같은 꼴의 작은 문제이다(되돌이로 푼다). 3번 경우는 따로 **아우르기** 단계가 필요하다.

### 최대 걸침 부분 배열 찾기

A crossing subarray includes $A[\text{mid}]$ and extends left to some index $i$ and right to some index $j$. We find the best leftward extension and the best rightward extension independently, then combine them.

**Left extension.** Starting from $\text{mid}$, scan leftward, tracking the maximum suffix sum:

$$
\text{left\_sum} = \max_{i \le \text{mid}} \sum_{k=i}^{\text{mid}} A[k]
$$

**Right extension.** Starting from $\text{mid}+1$, scan rightward, tracking the maximum prefix sum:

$$
\text{right\_sum} = \max_{j \ge \text{mid}+1} \sum_{k=\text{mid}+1}^{j} A[k]
$$

The maximum crossing sum is $\text{left\_sum} + \text{right\_sum}$, computed in $O(n)$ time with a single pass in each direction.

### 알고리즘

```
MAX-CROSSING-SUBARRAY(A, lo, mid, hi):
    left_sum = -infinity
    sum = 0
    for i = mid downto lo:
        sum = sum + A[i]
        if sum > left_sum:
            left_sum = sum
            max_left = i

    right_sum = -infinity
    sum = 0
    for j = mid + 1 to hi:
        sum = sum + A[j]
        if sum > right_sum:
            right_sum = sum
            max_right = j

    return (max_left, max_right, left_sum + right_sum)
```

```
MAX-SUBARRAY(A, lo, hi):
    if lo == hi:
        return (lo, hi, A[lo])

    mid = floor((lo + hi) / 2)
    (l1, r1, s1) = MAX-SUBARRAY(A, lo, mid)
    (l2, r2, s2) = MAX-SUBARRAY(A, mid + 1, hi)
    (l3, r3, s3) = MAX-CROSSING-SUBARRAY(A, lo, mid, hi)

    return the triple (li, ri, si) with the largest si
```

### 파이썬 구현

```python
def max_crossing_subarray(arr, lo, mid, hi):
    """
    가운뎃점을 걸치는 최대 부분 배열 찾기.

    매개변수
    ----------
    arr : list
        들임 배열.
    lo, mid, hi : int
        lo <= mid < hi인 부분 배열의 테두리.

    반환값
    -------
    tuple
        (left_index, right_index, max_sum)
    """
    # 가운데에서 왼쪽으로 뻗기
    left_sum = float('-inf')
    total = 0
    max_left = mid
    for i in range(mid, lo - 1, -1):
        total += arr[i]
        if total > left_sum:
            left_sum = total
            max_left = i

    # 가운데 + 1에서 오른쪽으로 뻗기
    right_sum = float('-inf')
    total = 0
    max_right = mid + 1
    for j in range(mid + 1, hi + 1):
        total += arr[j]
        if total > right_sum:
            right_sum = total
            max_right = j

    return max_left, max_right, left_sum + right_sum


def max_subarray_dc(arr, lo=None, hi=None):
    """
    나누어 이기기로 최대 부분 배열 찾기.

    매개변수
    ----------
    arr : list
        수의 들임 배열.
    lo : int, optional
        왼쪽 테두리(붙박이: 0).
    hi : int, optional
        오른쪽 테두리(붙박이: len(arr) - 1).

    반환값
    -------
    tuple
        (left_index, right_index, max_sum)
    """
    if lo is None:
        lo = 0
    if hi is None:
        hi = len(arr) - 1

    # 바탕 경우: 원소 하나
    if lo == hi:
        return lo, hi, arr[lo]

    mid = (lo + hi) // 2

    # 이기기: 왼쪽과 오른쪽의 작은 문제 풀기
    l1, r1, s1 = max_subarray_dc(arr, lo, mid)
    l2, r2, s2 = max_subarray_dc(arr, mid + 1, hi)

    # 아우르기: 최대 걸침 부분 배열 찾기
    l3, r3, s3 = max_crossing_subarray(arr, lo, mid, hi)

    # 셋 가운데 가장 좋은 것 돌려주기
    if s1 >= s2 and s1 >= s3:
        return l1, r1, s1
    elif s2 >= s1 and s2 >= s3:
        return l2, r2, s2
    else:
        return l3, r3, s3
```

## 올바름

The algorithm is correct because it exhausts all possibilities for where the maximum subarray can lie. Every contiguous subarray of $A[\text{lo} \,..\, \text{hi}]$ either lies entirely in the left half, entirely in the right half, or crosses the midpoint. The recursive calls correctly handle the first two cases (by induction), and `MAX-CROSSING-SUBARRAY` correctly handles the third by independently optimizing the left and right extensions.

## 복잡도 분석

### 점화식

$T(n)$을 크기 $n$인 배열에서의 도는 시간이라 하자. 이 알고리즘은:

- $O(1)$ 시간에 나눈다(가운뎃점 셈하기).
- 크기 $n/2$인 작은 문제 둘을 풀어 이긴다.
- $O(n)$ 시간에 아우른다(걸침 부분 배열 셈하기).

되돌이 관계식은 다음과 같다

$$
T(n) = 2T\!\left(\frac{n}{2}\right) + \Theta(n)
$$

### 되돌이 관계식 풀기

By the Master Theorem with $a = 2$, $b = 2$, and $f(n) = \Theta(n)$:

$$
\log_b a = \log_2 2 = 1
$$

Since $f(n) = \Theta(n^1)$, this is case 2:

$$
T(n) = \Theta(n \log n)
$$

### 공간 복잡도

The recursion depth is $O(\log n)$, and each level uses $O(1)$ auxiliary space (besides the recursive calls), giving $O(\log n)$ total space.

## 풀이 예제

$n = 7$인 $A = [2, -4, 3, -1, 2, -5, 4]$을 보자.

**Top-level call** on $A[0..6]$, $\text{mid} = 3$:

- **왼쪽** $A[0..3] = [2, -4, 3, -1]$: 되돌이 부름이 합이 $3$인 부분 배열 $[3]$을 돌려준다.
- **오른쪽** $A[4..6] = [2, -5, 4]$: 되돌이 부름이 합이 $4$인 부분 배열 $[4]$을 돌려준다.
- **걸침**: 번호 3에서 왼쪽으로 가장 좋게 뻗은 것은 합이 $2$인 $A[2..3]$이고, 번호 4에서 오른쪽으로 가장 좋게 뻗은 것은 합이 $2$인 $A[4]$이다. 걸침 합 = $4$.

Maximum of $\{3, 4, 4\} = 4$, achieved by either the right subarray $[4]$ or the crossing subarray $A[2..4] = [3, -1, 2]$.

## 카데인 알고리즘과의 견줌

| 성질 | 나누어 이기기 | 카데인 알고리즘 |
|---|---|---|
| Time complexity | $O(n \log n)$ | $O(n)$ |
| Space complexity | $O(\log n)$ | $O(1)$ |
| 틀 | 나누어 이기기 | 동적 계획 |
| 나란히 하기 | 가능(왼쪽과 오른쪽 반) | 불가(차례대로 훑음) |
| 배움의 값 | 나누어 이기기의 아우르기 단계를 보여 준다 | 동적 계획과 욕심쟁이를 보여 준다 |

차례대로 돌릴 때는 카데인 알고리즘이 엄밀히 더 빠르지만, 나누어 이기기 방식은 나란히 하기가 더 자연스럽고 이 틀을 가르치는 데 아주 좋은 보기가 된다.

## 요약

The divide-and-conquer solution to the maximum subarray problem splits the array at the midpoint, recursively finds the maximum subarrays in each half, and combines by finding the maximum crossing subarray in $O(n)$ time. The resulting $O(n \log n)$ algorithm is slower than Kadane's $O(n)$ solution but provides a clean illustration of all three divide-and-conquer steps, especially the combine step that handles the crossing case.

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), 4장. MIT Press.

## 연습문제

**연습문제 1.**
최대 부분 배열의 핵심 생각과 그 시간 복잡도를 설명하여라.

??? success "연습문제 1 풀이"
    Maximum Subarray은 나누어 다스리기 틀을 쓴다. 문제를 더 작은 잔문제로 쪼개고, 되부르며 풀고, 그 결과를 아우른다. 때 복잡도는 잔문제의 크기와 아우르는 값을 다스리는 되돌이 식이 정한다. 흔히 으뜸 정리나 되부름 나무 살피기로 닫힌 꼴의 복잡도를 얻는다. $\square$

---

**연습문제 2.**
최대 부분 배열의 되돌이 관계식을 쓰고 마스터 정리로 풀어라.

??? success "연습문제 2 풀이"
    되돌이 식은 그 알고리즘이 어떻게 나누는지에 달려 있다(잔문제의 수 $a$, 크기를 줄이는 값 $b$, 아우르는 값 $f(n)$). 으뜸 정리를 쓴다. $f(n)$을 $n^{\log_b a}$과 견주어 어느 갈래인지 가린다. $f(n) = \Theta(n^{\log_b a})$이면(둘째 갈래) $T(n) = \Theta(n^{\log_b a} \log n)$이다. $\square$

---

**연습문제 3.**
최대 부분 배열이 막무가내 방식보다 나은 장면을 설명하여라. 얼마나 빨라지는지 수로 나타내어라.

??? success "연습문제 3 풀이"
    막무가내로 하는 길은 흔히 $O(n^2)$ 이상이 든다. 나누어 다스리는 길은 되부르며 쪼개어 군더더기 셈을 줄이므로 복잡도가 더 낮다. 들임 크기가 $n = 10^6$이면 $O(n^2) = 10^{12}$과 $O(n \log n) = 2 \times 10^7$의 차이는 $50{,}000$ 곱절이다. $\square$

---

**연습문제 4.**
최대 부분 배열의 바탕 경우는 무엇인가? 그것이 알고리즘 전체의 옳음에 어떤 영향을 주는가?

??? success "연습문제 4 풀이"
    밑 자리는 더 나눌 수 없을 만큼 작은 들임을 다룬다(흔히 $n \leq 1$이나 $n \leq 2$). 이때는 옳은 결과를 곧바로 돌려주어야 한다. 밑 자리가 제대로 없으면 되부름이 끝나지 않는다. 밑 자리를 더 크게 잡고($n \leq 10$ 따위) 더 단순한 알고리즘으로 갈아타면 같은 점근 복잡도를 지키면서 되부름 덤을 줄여 참으로 더 빠르게 할 수 있다. $\square$
