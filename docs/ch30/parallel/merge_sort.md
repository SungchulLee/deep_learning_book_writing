# 나란한 합침 정렬

합침 정렬은 가르기 걸음이 배열을 서로 독립인 반 둘로 나누므로 나란히 하기에 자연스러운 후보다. 그러나 되돌이 부름만 어수룩하게 나란히 하면 뻗음이 $O(n)$인 차례 합침 걸음이 남아 온 나란함이 가둬진다. 뻗음 $O(\log^2 n)$을 이루는 열쇠는 두 갈래 찾기로 줄 세운 반 둘을 뻗음 $O(\log n)$에 합치는 **나란한 합침** 절차다.

## 차례 밑그림

여느 합침 정렬은 일이 $O(n \log n)$이고 뻗음도 $O(n \log n)$이다(온전히 차례). 그 점화식:

$$
T_1(n) = 2 \cdot T_1(n/2) + O(n) = O(n \log n)
$$

차례로 돌릴 때 $O(n)$인 합침 걸음이 뻗음을 지배한다.

## 되돌이 부름만 나란히

가장 단순한 나란히 하기는 되돌이 부름 둘을 갈라 놓는다:

- **일**: $T_1(n) = 2 \cdot T_1(n/2) + O(n) = O(n \log n)$.
- **뻗음**: 합침이 아직 차례이므로 $T_\infty(n) = T_\infty(n/2) + O(n) = O(n)$.
- **나란함**: $O(\log n)$ -- 매우 낮다.

이 길은 차례 돌림보다 거의 나아지지 않는다.

## 두 갈래 찾기로 하는 나란한 합침

합침 걸음의 뻗음을 줄이려 나누어 정복하기 합침을 쓴다. 줄 세운 배열 $L$과 $R$이 주어질 때:

1. 더 큰 배열($L$이라 하자)의 가운뎃값 원소 $m$을 고른다.
2. $R$에서 $m$을 두 갈래로 찾아 그 매김 $j$을 얻는다.
3. $L[0 \ldots |L|/2 - 1]$과 $R[0 \ldots j - 1]$의 원소는 모두 $m$ 앞에, 나머지는 뒤에 놓인다.
4. "앞" 반 둘과 "뒤" 반 둘을 되돌이로 나란히 합친다.

합침 점화식은 다음이 된다:

$$
M_\infty(n) = M_\infty(3n/4) + O(\log n) = O(\log^2 n)
$$

$3n/4$이 나오는 까닭은 가장 나쁜 가르기가 온 원소를 많아야 $3n/4$개 무리로 나누기 때문이다.

## 온전한 나란한 합침 정렬 살피기

나란한 합침을 쓰면 온 점화식은 다음이 된다:

**일**:

$$
T_1(n) = 2 \cdot T_1(n/2) + O(n) = O(n \log n)
$$

**뻗음**:

$$
T_\infty(n) = T_\infty(n/2) + O(\log^2 n) = O(\log^3 n)
$$

!!! note "뻗음 O(log^2 n) 이루기"
    (매김을 쓰는) 더 정교한 나란한 합침으로 뻗음 $O(\log n)$을 이루면 온 뻗음이 $T_\infty(n) = T_\infty(n/2) + O(\log n) = O(\log^2 n)$이 된다. 여기 밝힌 두 갈래 찾기 바탕 합침은 $O(\log^3 n)$을 준다.

**나란함**: $P = O(n \log n / \log^3 n) = O(n / \log^2 n)$이며 $n$이 크면 상당하다.

## 구현

```python
"""
나란한 합침을 흉내 낸 나란한 합침 정렬.

나란한 합침은 두 갈래 찾기에 바탕한 나누어 정복하기를 쓴다.
참으로 나란한 시스템에서는 merge_sort과 merge의 되돌이 부름이
한꺼번에 돈다.
"""

from bisect import bisect_left

# ===================================================================
# 나란한 합침
# ===================================================================

def parallel_merge(left, right):
    """나누어 정복하기로 줄 세운 배열 둘을 합친다.

    두 갈래 찾기로 합침을 서로 독립인 부분 문제로 갈라
    나란히 돌릴 수 있게 한다.

    인수:
        left: 줄 세운 배열
        right: 줄 세운 배열

    반환값:
        합쳐 줄 세운 배열
    """
    if not left:
        return list(right)
    if not right:
        return list(left)
    if len(left) + len(right) <= 4:
        return _sequential_merge(left, right)

    # left이 더 큰 배열이 되게 한다
    if len(left) < len(right):
        left, right = right, left

    mid = len(left) // 2
    pivot = left[mid]
    j = bisect_left(right, pivot)

    # 갈라짐: 서로 독립인 반 둘을 합친다
    lower = parallel_merge(left[:mid], right[:j])
    upper = parallel_merge(left[mid + 1:], right[j:])

    return lower + [pivot] + upper

# ===================================================================
# 나란한 합침 정렬
# ===================================================================

def parallel_merge_sort(arr):
    """나란한 합침 정렬로 배열을 줄 세운다.

    인수:
        arr: 들임 배열

    반환값:
        줄 세운 배열
    """
    if len(arr) <= 1:
        return list(arr)

    mid = len(arr) // 2

    # 갈라짐: 반 둘을 줄 세운다(실제 시스템에서는 나란히)
    left = parallel_merge_sort(arr[:mid])
    right = parallel_merge_sort(arr[mid:])

    # 합침: 나란한 합침
    return parallel_merge(left, right)


def _sequential_merge(left, right):
    """작은 들임을 위한 여느 차례 합침."""
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

# ===================================================================
# 메인
# ===================================================================

if __name__ == "__main__":
    import math

    arr = [38, 27, 43, 3, 9, 82, 10, 15, 72, 4]
    sorted_arr = parallel_merge_sort(arr)

    print(f"Input:  {arr}")
    print(f"Sorted: {sorted_arr}")
    print(f"Correct: {sorted_arr == sorted(arr)}")

    n = len(arr)
    work = n * math.ceil(math.log2(n))
    span = math.ceil(math.log2(n)) ** 2
    print(f"\nn = {n}")
    print(f"Work O(n log n) ~ {work}")
    print(f"Span O(log^2 n) ~ {span}")
    print(f"Parallelism ~ {work / span:.1f}")
```

**출력:**
```
Input:  [38, 27, 43, 3, 9, 82, 10, 15, 72, 4]
Sorted: [3, 4, 9, 10, 15, 27, 38, 43, 72, 82]
올바름: True

n = 10
일 O(n log n) ~ 40
뻗음 O(log^2 n) ~ 16
나란함 ~ 2.5
```

## 복잡도 요약

| 변형 | 일 $T_1$ | 뻗음 $T_\infty$ | 나란함 |
|---|---|---|---|
| 차례 합침 정렬 | $O(n \log n)$ | $O(n \log n)$ | $O(1)$ |
| 되돌이만 나란히 | $O(n \log n)$ | $O(n)$ | $O(\log n)$ |
| 나란한 합침(두 갈래 찾기) | $O(n \log n)$ | $O(\log^3 n)$ | $O(n / \log^2 n)$ |
| 나란한 합침(매김) | $O(n \log n)$ | $O(\log^2 n)$ | $O(n / \log n)$ |

## 참고 문헌

- Cormen, T. H. et al. *Introduction to Algorithms*, 27장(여러 실 알고리즘).
- Cole, R. (1988). "Parallel merge sort." *SIAM Journal on Computing*, 17(4), 770--785.


## 연습문제

**연습문제 1.**
나란한 합침 정렬 알고리즘을 밝히고 그 일과 뻗음을 살펴라.

??? success "연습문제 1 풀이"
    나란한 합침 정렬: (1) 배열을 반으로 가르고, (2) 반씩 되돌이로 나란히 줄 세우고(갈라짐), (3) 줄 세운 반 둘을 합친다. 일: $W(n) = 2W(n/2) + O(n) = O(n \log n)$. 합침 걸음도 나란히 할 수 있다. 두 갈래 찾기로 원소마다 매김을 찾아 곧바로 놓는다. 나란한 합침: 일 $O(n)$, 뻗음 $O(\log n)$. 온 뻗음: $D(n) = D(n/2) + O(\log n) = O(\log^2 n)$. 나란함: $O(n/\log n)$.

---

**연습문제 2.**
일, 뻗음, 실제 성능으로 나란한 합침 정렬과 나란한 빠른 정렬을 견주어라.

??? success "연습문제 2 풀이"
    합침 정렬: $W = O(n\log n)$으로 정해져 있고 $D = O(\log^2 n)$이다. 빠른 정렬: $W = O(n\log n)$은 기댓값이고 $D = O(\log^2 n)$도 기댓값이며 가장 나쁜 경우 $O(n)$이다. 합침 정렬이 더 헤아리기 쉽고 두름에 친하다(합칠 때 차례로 닿는다). 빠른 정렬은 실제 상수가 낫지만(제자리, 평균 견줌이 적다) 가장 나쁜 경우 뻗음이 나쁘다. GPU 줄 세우기에서는 기수 정렬이 둘 다보다 나을 때가 많다. 너비가 붙박인 열쇠에 일 $O(n)$, 뻗음 $O(\log n)$이고 기억 닿기 무늬가 뛰어나다.

---

**연습문제 3.**
두 갈래 찾기를 쓰는 나란한 합침 연산을 밝히고 왜 뻗음 $O(\log n)$을 이루는지 말하여라.

??? success "연습문제 3 풀이"
    줄 세운 배열 $A[1..n]$과 $B[1..n]$을 $C[1..2n]$으로 합친다. 원소 $A[i]$마다 $B$에서 두 갈래로 찾아 매김 $r_i$($A[i]$ 이하인 $B$ 원소의 개수)을 얻는다. $A[i]$을 $C$의 자리 $i + r_i$에 놓는다. $B[j]$마다도 마찬가지다. 두 갈래 찾기가 모두 나란히 돌며 저마다 $O(\log n)$ 때가 든다. 온 일: $O(n \log n)$(더 정교한 알고리즘으로 $O(n)$까지 좋게 할 수 있다). 뻗음: $O(\log n)$(찾기가 모두 독립이다).

---

**연습문제 4.**
줄 세우기는 깊은 배움 익히기 흐름에 어떻게 쓰이는가?

??? success "연습문제 4 풀이"
    줄 세우기는 다음에 나온다. (1) 차례 짓기의 빔 찾기용 위 k개 고르기(부분 줄 세우기), (2) 물체 찾기의 최대 아닌 것 누르기(믿음 점수로 줄 세우기), (3) 익히기 표본을 어려움(차례 배우기)이나 차례 길이(효율 좋은 묶기를 위한 통 담기)로 줄 세우기, (4) 매김 손실 셈하기(보기로 견줌 배우기는 닮음으로 줄 세워야 한다), (5) 양자화 눈금 맞추기의 분위 셈하기. GPU에 맞춘 줄 세우기(기수 정렬, 바이토닉 정렬)가 실시간 헤아림에 결정적이다.