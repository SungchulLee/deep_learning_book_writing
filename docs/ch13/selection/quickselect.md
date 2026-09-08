# 빠른 고르기

$k$번째로 작은 원소를 찾으려고 배열을 정렬하면 $O(n \log n)$이 드는데, 필요보다 훨씬 많다. 1961년 토니 호어가 지어낸 **빠른 고르기**는 빠른 정렬의 나눔을 손질해 고르기 문제를 기대 시간 $O(n)$에 푼다. 핵심 통찰은 나눈 뒤 축의 어느 쪽에 $k$번째 원소가 있는지 알게 되므로 양쪽이 아니라 **한쪽**에서만 되돌이하면 된다는 것이다. 걸음마다 (평균으로) 일이 반으로 줄어 등비급수를 이루고 그 합이 $O(n)$이 된다.

---

## 1. 알고리즘

배열 $A[lo..hi]$과 목표 순위 $k$(이 부분 배열 안에서 0부터 세는)이 주어지면 다음과 같이 한다.

1. $lo = hi$이면 $A[lo]$을 되돌린다.
2. 축을 고른다(기대 성능이 가장 좋으려면 무작위로 고른다).
3. 축을 기준으로 $A[lo..hi]$을 나눈다. 축이 자리 $p$에 떨어진다고 하자.
4. $k = p$이면 $A[p]$을 되돌린다(축이 답이다).
5. $k < p$이면 $A[lo..p-1]$에서 되돌이한다.
6. $k > p$이면 $A[p+1..hi]$에서 되돌이한다.

**양쪽**에서 되돌이하는 빠른 정렬과 달리 빠른 고르기는 한쪽에서만 되돌이한다. 이것이 기대 전체 일의 양을 $O(n \log n)$에서 $O(n)$으로 줄인다.

---

## 2. 기대 시간 분석

무작위 축을 쓰면 나눔이 배열을 대체로 반으로 쪼갠다. 기대 전체 일의 양은 다음과 같다.

$$
E[T(n)] = n + \frac{1}{n} \sum_{q=0}^{n-1} E\!\left[T\!\left(\max(q, n - 1 - q)\right)\right]
$$

최악의 되돌이 부름은 더 큰 쪽을 다룬다. 무작위 축이 확률 $1/2$으로 가운데 절반에 떨어진다는 사실로 위 한계를 잡으면 다음이 나온다.

$$
E[T(n)] \leq n + \frac{3n}{4} + \frac{9n}{16} + \cdots = n \sum_{i=0}^{\infty} \left(\frac{3}{4}\right)^i = 4n
$$

그러므로 $E[T(n)] = O(n)$이다. 더 빈틈없이 뜯어보면 $E[T(n)] \leq 3.39\, n + o(n)$이다.

---

## 3. 최악의 경우

최악의 경우는 축마다 가장 작거나 가장 큰 원소일 때 일어난다.

$$
T(n) = n + (n-1) + (n-2) + \cdots + 1 = \frac{n(n+1)}{2} = O(n^2)
$$

무작위 축에서는 이런 일이 많아야 확률 $O(1/n!)$으로 일어나므로 실전에서는 하찮다.

---

## 4. 구현

```python
"""
빠른 고르기: 기대 시간 O(n)의 나눔 기반 고르기.

무작위 축을 기준으로 나누고 목표 순위를 담은 쪽에서만 되돌이하여
k번째로 작은 원소를 찾는다.
"""

import random

# === 나눔 ===

def partition(arr: list, lo: int, hi: int) -> int:
    """무작위 축을 쓴 로무토 나눔. 축의 첨자를 되돌린다."""
    pivot_idx = random.randint(lo, hi)
    arr[pivot_idx], arr[hi] = arr[hi], arr[pivot_idx]
    pivot = arr[hi]

    i = lo
    for j in range(lo, hi):
        if arr[j] <= pivot:
            arr[i], arr[j] = arr[j], arr[i]
            i += 1
    arr[i], arr[hi] = arr[hi], arr[i]
    return i

# === 빠른 고르기 ===

def quickselect(arr: list, k: int):
    """k번째로 작은 원소(1부터 세는)를 찾는다.

    정렬된 배열에서 첨자 k-1에 있을 원소를 되돌린다.
    원본을 지키려고 베낀 것에서 굴린다.
    """
    if k < 1 or k > len(arr):
        raise ValueError(f"k={k} out of range for array of size {len(arr)}")

    data = arr.copy()
    return _quickselect(data, 0, len(data) - 1, k - 1)

def _quickselect(arr: list, lo: int, hi: int, k: int):
    """순위 k에 대한 arr[lo..hi]의 되돌이 빠른 고르기."""
    if lo == hi:
        return arr[lo]

    pivot_pos = partition(arr, lo, hi)

    if k == pivot_pos:
        return arr[k]
    elif k < pivot_pos:
        return _quickselect(arr, lo, pivot_pos - 1, k)
    else:
        return _quickselect(arr, pivot_pos + 1, hi, k)

# === 되풀이 판 ===

def quickselect_iterative(arr: list, k: int):
    """꼬리 부름 없애기를 쓴 되풀이 빠른 고르기."""
    data = arr.copy()
    lo, hi = 0, len(data) - 1
    k -= 1  # 0부터 세는 첨자로 바꾸기

    while lo < hi:
        pivot_pos = partition(data, lo, hi)
        if k == pivot_pos:
            return data[k]
        elif k < pivot_pos:
            hi = pivot_pos - 1
        else:
            lo = pivot_pos + 1

    return data[lo]

# === 시연 ===

if __name__ == "__main__":
    random.seed(42)

    data = [7, 10, 4, 3, 20, 15, 8, 1, 12, 5]
    print(f"Array:  {data}")
    print(f"Sorted: {sorted(data)}")
    print()

    for k in [1, 3, 5, 8, 10]:
        result_rec = quickselect(data, k)
        result_iter = quickselect_iterative(data, k)
        print(f"k={k:2d}: recursive={result_rec:3d}, "
              f"iterative={result_iter:3d}")

    print()
    print("Finding median:")
    n = len(data)
    median = quickselect(data, (n + 1) // 2)
    print(f"  Array size: {n}, median (k={( n + 1) // 2}): {median}")
```

**출력:**
```
Array:  [7, 10, 4, 3, 20, 15, 8, 1, 12, 5]
Sorted: [1, 3, 4, 5, 7, 8, 10, 12, 15, 20]

k= 1: recursive=  1, iterative=  1
k= 3: recursive=  4, iterative=  4
k= 5: recursive=  7, iterative=  7
k= 8: recursive= 12, iterative= 12
k=10: recursive= 20, iterative= 20

Finding median:
  Array size: 10, median (k=5): 7
```

---

## 5. 복잡도

| 경우 | 시간 | 공간 |
|------|------|-------|
| 최선 | $O(n)$ | 되풀이 $O(1)$ / 되돌이 $O(\log n)$ |
| 기대 | $O(n)$ | 되풀이 $O(1)$ / 되돌이 $O(\log n)$ |
| 최악 | $O(n^2)$ | 되돌이 $O(n)$ |

!!! warning "적수 입력"
    축을 정해 놓고 고르면(이를테면 늘 첫 원소나 마지막 원소) 적수가 $O(n^2)$을 강제할 수 있다. 늘 무작위로 축을 고르거나, 되돌이 깊이가 문턱값을 넘으면 중앙값의 중앙값으로 물러서라.

---

## 연습문제

**연습문제 1.**
빠른 고르기의 핵심 생각과 그 시간·공간 복잡도를 밝혀라.

??? success "연습문제 1 풀이"
    이 알고리즘은 견줌 너머의 성질(정수 열쇠, 바깥 저장 장치, 병렬 하드웨어)을 살려 써서 비교 기반 정렬이 따라올 수 없는 성능을 이룬다. 구체적인 복잡도 한계는 이 쪽에서 뜯어본다.

---

**연습문제 2.**
작은 입력에서 빠른 고르기를 따라가라. 훑기나 단계마다 보여라.

??? success "연습문제 2 풀이"
    원소 6~8개에 알고리즘을 적용하며 훑을 때마다의 상태를 보여라. 이 따라가기가 알고리즘의 얼개를 드러내고 옳음을 눈에 보이게 한다.

---

**연습문제 3.**
어떤 조건에서 비교 기반 정렬보다 빠른 고르기가 나은가?

??? success "연습문제 3 풀이"
    다음일 때 낫다. 입력의 정수 범위가 묶여 있을 때(세기 정렬과 기수 정렬), 데이터가 램을 넘칠 때(바깥 정렬), 병렬 하드웨어를 쓸 수 있을 때(병렬 정렬). 이런 조건에서는 알고리즘이 비교 기반의 아래 한계 $\Omega(n\log n)$을 비껴갈 수 있다.

---

**연습문제 4.**
빠른 고르기가 실전에서 이득을 주는 깊은 학습 응용을 서술하라.

??? success "연습문제 4 풀이"
    응용: 어휘를 찾기 위한 토큰 번호 정렬($O(n + V)$의 세기 정렬), GPU 기억을 넘치는 데이터셋의 바깥 정렬, GPU에서 배치 연산을 위한 병렬 정렬. 특정 조건(정수 열쇠, 큰 데이터, 병렬성)이 갖추어질 때 이득이 가장 크다.

## 정리하며

이 마당은 알고리즘、기대 시간 분석、최악의 경우、구현을 차례로 짚었다.

**참고 문헌**

- Hoare, C. A. R. (1961). Algorithm 65: Find. *Communications of the ACM*, 4(7), 321-322.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), 9장. MIT Press.
