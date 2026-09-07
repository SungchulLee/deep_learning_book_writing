# 최대 부분 배열

음수가 들어 있을 수 있는 수의 배열이 주어질 때 **최대 부분 배열 문제**는 합이 가장 큰 잇닿은 부분 배열을 묻는다. 이 문제는 금융 살피기(가장 이득이 큰 거래 기간 찾기), 신호 다루기(가장 센 신호 토막 찾기), 유전체학(생물학적으로 뜻있는 자리 가려내기)에서 자연스레 나타난다.

카데인 알고리즘은 갈피 다지기로 이 문제를 $O(n)$ 때에 풀지만, 나누어 다스리는 길은 $O(n \log n)$의 멋진 풀이를 주며 이 틀이 어떻게 도는지를, 무엇보다 아우르는 걸음을 또렷이 보여 준다.

## 문제 서술

실수 배열 $A[0 \,..\, n-1]$이 주어졌을 때 다음을 가장 크게 하는 자리 번호 $i$과 $j$($0 \le i \le j \le n-1$)을 찾아라.

$$
\sum_{k=i}^{j} A[k]
$$

원소가 모두 음수이면 최대 부분 배열은 값이 가장 큰 원소 하나이다.

## 나누어 이기기 방식

고갱이 깨침은 $A[\text{lo} \,..\, \text{hi}]$의 가장 큰 잔배열이 가운데 자리 $\text{mid} = \lfloor (\text{lo} + \text{hi}) / 2 \rfloor$을 두고 다음 세 자리 가운데 꼭 하나에 있다는 것이다.

1. **온통 왼쪽 반에**: $A[\text{lo} \,..\, \text{mid}]$
2. **온통 오른쪽 반에**: $A[\text{mid}+1 \,..\, \text{hi}]$
3. **가운데 자리를 가로질러**: $i \le \text{mid} < j$인 어떤 $A[i \,..\, j]$

1번과 2번 경우는 같은 꼴의 작은 문제이다(되돌이로 푼다). 3번 경우는 따로 **아우르기** 단계가 필요하다.

### 최대 걸침 부분 배열 찾기

가로지르는 잔배열은 $A[\text{mid}]$을 담고 왼쪽으로 어떤 자리 번호 $i$까지, 오른쪽으로 어떤 자리 번호 $j$까지 뻗는다. 왼쪽으로 가장 좋게 뻗는 것과 오른쪽으로 가장 좋게 뻗는 것을 따로 찾은 뒤 아우른다.

**왼쪽으로 뻗기.** $\text{mid}$에서 비롯해 왼쪽으로 훑으며 가장 큰 꼬리 합을 좇는다.

$$
\text{left\_sum} = \max_{i \le \text{mid}} \sum_{k=i}^{\text{mid}} A[k]
$$

**오른쪽으로 뻗기.** $\text{mid}+1$에서 비롯해 오른쪽으로 훑으며 가장 큰 머리 합을 좇는다.

$$
\text{right\_sum} = \max_{j \ge \text{mid}+1} \sum_{k=\text{mid}+1}^{j} A[k]
$$

가로지르는 가장 큰 합은 $\text{left\_sum} + \text{right\_sum}$이며, 방향마다 한 번씩 훑어 $O(n)$ 때에 셈한다.

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
        (왼쪽 번호, 오른쪽 번호, 가장 큰 합)
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
        (왼쪽 번호, 오른쪽 번호, 가장 큰 합)
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

이 알고리즘은 가장 큰 잔배열이 있을 수 있는 자리를 모두 훑으므로 옳다. $A[\text{lo} \,..\, \text{hi}]$의 이어진 잔배열은 모두 온통 왼쪽 반에 있거나, 온통 오른쪽 반에 있거나, 가운데 자리를 가로지른다. 되부름이 앞의 두 자리를 옳게 다루고(미루어 나아가기에 따라), `MAX-CROSSING-SUBARRAY`이 왼쪽과 오른쪽으로 뻗는 것을 따로 가장 좋게 하여 셋째를 옳게 다룬다.

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

$a = 2$, $b = 2$, $f(n) = \Theta(n)$인 으뜸 정리에 따라

$$
\log_b a = \log_2 2 = 1
$$

$f(n) = \Theta(n^1)$이므로 둘째 갈래다.

$$
T(n) = \Theta(n \log n)
$$

### 공간 복잡도

되부름 깊이가 $O(\log n)$이고 켜마다 (되부름 말고는) 덧붙은 자리를 $O(1)$만큼 쓰므로 온 자리는 $O(\log n)$이다.

## 풀이 예제

$n = 7$인 $A = [2, -4, 3, -1, 2, -5, 4]$을 보자.

$A[0..6]$에 대한 **맨 위 부름**, $\text{mid} = 3$:

- **왼쪽** $A[0..3] = [2, -4, 3, -1]$: 되돌이 부름이 합이 $3$인 부분 배열 $[3]$을 돌려준다.
- **오른쪽** $A[4..6] = [2, -5, 4]$: 되돌이 부름이 합이 $4$인 부분 배열 $[4]$을 돌려준다.
- **걸침**: 번호 3에서 왼쪽으로 가장 좋게 뻗은 것은 합이 $2$인 $A[2..3]$이고, 번호 4에서 오른쪽으로 가장 좋게 뻗은 것은 합이 $2$인 $A[4]$이다. 걸침 합 = $4$.

$\{3, 4, 4\} = 4$ 가운데 가장 큰 값이며, 오른쪽 잔배열 $[4]$이나 가로지르는 잔배열 $A[2..4] = [3, -1, 2]$에서 이루어진다.

## 카데인 알고리즘과의 견줌

| 성질 | 나누어 이기기 | 카데인 알고리즘 |
|---|---|---|
| 때 복잡도 | $O(n \log n)$ | $O(n)$ |
| 자리 복잡도 | $O(\log n)$ | $O(1)$ |
| 틀 | 나누어 이기기 | 동적 계획 |
| 나란히 하기 | 가능(왼쪽과 오른쪽 반) | 불가(차례대로 훑음) |
| 배움의 값 | 나누어 이기기의 아우르기 단계를 보여 준다 | 동적 계획과 욕심쟁이를 보여 준다 |

차례대로 돌릴 때는 카데인 알고리즘이 엄밀히 더 빠르지만, 나누어 이기기 방식은 나란히 하기가 더 자연스럽고 이 틀을 가르치는 데 아주 좋은 보기가 된다.

## 요약

가장 큰 잔배열 문제를 나누어 다스리는 풀이는 배열을 가운데에서 쪼개고, 반쪽마다 가장 큰 잔배열을 되부르며 찾은 뒤, 가로지르는 가장 큰 잔배열을 $O(n)$ 때에 찾아 아우른다. 그렇게 나온 $O(n \log n)$ 알고리즘은 카데인의 $O(n)$ 풀이보다 느리지만, 나누어 다스리기의 세 걸음을, 무엇보다 가로지름을 다루는 아우르는 걸음을 깔끔하게 보여 준다.

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
