# 인트로 정렬

빠른 정렬은 평균으로 빠르지만 적수 입력에서는 $O(n^2)$으로 무너진다. 힙 정렬은 최악의 경우 $O(n \log n)$을 보장하지만 캐시 지역성이 나쁘다. 끼워넣기 정렬은 아주 작은 배열에 가장 좋다. 1997년 데이비드 머서가 들여온 **인트로 정렬**(들여다보는 정렬)은 이 셋을 모두 아우른다. 빠른 정렬로 시작해 되돌이 깊이를 살피다가 깊이가 $2 \lfloor \log_2 n \rfloor$을 넘으면 힙 정렬로 갈아탄다. 부분 문제가 작으면 끼워넣기 정렬로 물러선다. 이 섞은 전략은 빠른 정렬의 실전 속도를 지니면서 최악의 경우 $O(n \log n)$ 시간을 이룬다. 인트로 정렬은 C++ `std::sort` 뒤에 있는 알고리즘이다.

---

## 1. 전략

핵심 통찰은 빠른 정렬의 최악의 경우가 치우친 나눔이 낳는 깊은 되돌이에서 온다는 것이다. 되돌이 깊이를 세다가 문턱값을 넘으면 빠른 정렬을 그만두어 인트로 정렬은 $O(n^2)$ 덫을 피한다.

이 알고리즘은 세 단계로 나아간다.

1. **빠른 정렬 단계**: 보통의 축 고르기(대개 셋의 중앙값)로 배열을 나눈다. 깊이 세개를 하나씩 줄이며 양쪽 반쪽에서 되돌이한다.
2. **힙 정렬로 물러서기**: 깊이 세개가 0이 되면 되돌이를 멈추고 지금 부분 배열을 힙 정렬로 정렬한다. 나눔이 늘 가장 치우쳐도 $O(n \log n)$이 보장된다.
3. **끼워넣기 정렬로 마무리**: 부분 배열의 원소가 16개보다 적으면 끼워넣기 정렬로 정렬한다. 아주 작은 배열에서 끼워넣기 정렬의 $O(n^2)$ 비용은 하찮고, 짐이 가벼워 작은 $n$에서는 빠른 정렬보다 빠르다.

---

## 2. 깊이 한계

깊이 한계는 대개 다음으로 둔다.

$$
d_{\max} = 2 \lfloor \log_2 n \rfloor
$$

고르게 나뉘는 빠른 정렬이 깊이 $\log_2 n$에 다다르므로 이 값을 쓴다. 2배라는 여유가 조금 치우친 나눔은 눈감아 주면서도 병적인 경우는 일찍 잡아낸다.

---

## 3. 복잡도

| 경우 | 시간 | 공간 |
|------|------|-------|
| 최선 | $O(n \log n)$ | $O(\log n)$ |
| 평균 | $O(n \log n)$ | $O(\log n)$ |
| 최악 | $O(n \log n)$ | $O(\log n)$ |

최악의 경우 보장은 힙 정렬로 물러서기에서 온다. 무작위 입력에서는 깊이 한계에 거의 닿지 않으므로 평균의 경우는 보통의 빠른 정렬과 같다.

---

## 4. 구현

```python
"""
인트로 정렬: 빠른 정렬, 힙 정렬, 끼워넣기 정렬을 섞은 것.

되돌이 깊이를 살피다가 깊이가 2 * floor(log2(n))을 넘으면 힙 정렬로
물러서서 최악의 경우 O(n log n)을 이룬다.
작은 부분 배열은 끼워넣기 정렬로 마무리한다.
"""

import math

# === 끼워넣기 정렬(작은 부분 배열용) ===

def insertion_sort(arr: list, lo: int, hi: int) -> None:
    """끼워넣기 정렬로 arr[lo..hi]을 제자리에서 정렬한다."""
    for i in range(lo + 1, hi + 1):
        key = arr[i]
        j = i - 1
        while j >= lo and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key

# === 힙 정렬(깊은 되돌이에서 물러설 곳) ===

def heapsort(arr: list, lo: int, hi: int) -> None:
    """힙 정렬로 arr[lo..hi]을 제자리에서 정렬한다."""
    n = hi - lo + 1

    def sift_down(start: int, end: int) -> None:
        root = start
        while True:
            child = 2 * root + 1
            if child > end:
                break
            if child + 1 <= end and arr[lo + child] < arr[lo + child + 1]:
                child += 1
            if arr[lo + root] < arr[lo + child]:
                arr[lo + root], arr[lo + child] = arr[lo + child], arr[lo + root]
                root = child
            else:
                break

    # 최대 힙을 쌓는다
    for i in range(n // 2 - 1, -1, -1):
        sift_down(i, n - 1)

    # 원소를 꺼낸다
    for i in range(n - 1, 0, -1):
        arr[lo], arr[lo + i] = arr[lo + i], arr[lo]
        sift_down(0, i - 1)

# === 셋의 중앙값 축 ===

def median_of_three(arr: list, lo: int, hi: int) -> int:
    """arr[lo], arr[mid], arr[hi]의 중앙값 첨자를 되돌린다."""
    mid = (lo + hi) // 2
    if arr[lo] > arr[mid]:
        arr[lo], arr[mid] = arr[mid], arr[lo]
    if arr[lo] > arr[hi]:
        arr[lo], arr[hi] = arr[hi], arr[lo]
    if arr[mid] > arr[hi]:
        arr[mid], arr[hi] = arr[hi], arr[mid]
    return mid

# === 인트로 정렬 ===

def introsort(arr: list) -> None:
    """인트로 정렬로 arr을 제자리에서 정렬한다."""
    if len(arr) <= 1:
        return
    max_depth = 2 * math.floor(math.log2(len(arr)))
    _introsort_impl(arr, 0, len(arr) - 1, max_depth)

SIZE_THRESHOLD = 16

def _introsort_impl(arr: list, lo: int, hi: int, depth_limit: int) -> None:
    """깊이를 좇는 되돌이 인트로 정렬."""
    while hi - lo + 1 > SIZE_THRESHOLD:
        if depth_limit == 0:
            heapsort(arr, lo, hi)
            return

        depth_limit -= 1
        pivot_idx = median_of_three(arr, lo, hi)
        arr[pivot_idx], arr[hi] = arr[hi], arr[pivot_idx]

        # 로무토식 나눔
        pivot = arr[hi]
        i = lo
        for j in range(lo, hi):
            if arr[j] <= pivot:
                arr[i], arr[j] = arr[j], arr[i]
                i += 1
        arr[i], arr[hi] = arr[hi], arr[i]

        # 작은 쪽은 되돌이하고 큰 쪽은 되풀이한다
        if i - lo < hi - i:
            _introsort_impl(arr, lo, i - 1, depth_limit)
            lo = i + 1
        else:
            _introsort_impl(arr, i + 1, hi, depth_limit)
            hi = i - 1

    insertion_sort(arr, lo, hi)

# === 시연 ===

if __name__ == "__main__":
    data = [38, 27, 43, 3, 9, 82, 10, 55, 1, 72, 64, 29]
    print(f"Before: {data}")
    introsort(data)
    print(f"After:  {data}")
    print()

    # 소박한 빠른 정렬의 최악의 경우 — 이미 정렬됨
    worst = list(range(20, 0, -1))
    print(f"Reverse-sorted input: {worst}")
    introsort(worst)
    print(f"After introsort:      {worst}")
```

**출력:**
```
Before: [38, 27, 43, 3, 9, 82, 10, 55, 1, 72, 64, 29]
After:  [1, 3, 9, 10, 27, 29, 38, 43, 55, 64, 72, 82]

Reverse-sorted input: [20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
After introsort:      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
```

---

## 5. 그냥 힙 정렬만 쓰면 안 될까?

힙 정렬이 이미 $O(n \log n)$을 보장한다면 그냥 그것만 쓰면 되지 않을까? 답은 **캐시 효율**이다. 빠른 정렬은 기억을 차례대로 훑어 공간 지역성의 덕을 보지만, 힙 정렬은 힙 배열에서 어버이와 자식 사이를 뛰어다녀 캐시가 자주 빗나간다. 무작위 데이터에서 인트로 정렬은 거의 모든 시간을 빠른 정렬 단계에서 보내고, 적수 입력에서만 힙 정렬로 물러선다.

!!! tip "실전의 문턱값"
    끼워넣기 정렬의 문턱값 16은 실험으로 고른 값이다. 8에서 32 사이면 잘 굴러간다. 8보다 작으면 함수 부름의 짐이 커지고, 32보다 크면 끼워넣기 정렬의 이차 비용이 눈에 띈다.

---

## 연습문제

**연습문제 1.**
$[38, 27, 43, 3, 9, 82, 10]$에서 인트로 정렬을 따라가며 큰 걸음마다의 상태를 보여라.

??? success "연습문제 1 풀이"
    알고리즘을 한 걸음씩 밟아라. 나누어 정복하는 정렬이라면 되돌이 나눔과 병합을 하나씩 보이고, 나눔 기반 정렬이라면 나눔마다와 축이 놓이는 자리를 보여라. 걸음마다 배열 전체의 상태를 내보여라.

---

**연습문제 2.**
인트로 정렬의 최선, 최악, 평균의 경우 시간 복잡도를 끌어내라.

??? success "연습문제 2 풀이"
    경우마다 점화식을 써라. 최선의 경우는 가장 좋은 나눔이거나 미리 정렬된 입력이다. 최악의 경우는 일을 가장 크게 만드는 적수 입력이다. 평균의 경우는 무작위 순열에 대한 기대 성능이다. 점화식을 각각 풀어라.

---

**연습문제 3.**
인트로 정렬은 안정적인가? 증명하거나 반례를 들어라.

??? success "연습문제 3 풀이"
    같은 원소가 본디 상대 차례를 지키면 그 정렬은 안정적이다. 모든 입력에서 이것이 성립함을 증명하거나, 정렬 도중 같은 원소 둘의 자리가 뒤바뀌는 구체적인 입력을 들어라.

---

**연습문제 4.**
float32 학습 손실 값 1000만 개를 정렬할 때 인트로 정렬을 다른 두 방법과 견주어라. 시간, 공간, 캐시 거동, GPU 어울림을 살펴라.

??? success "연습문제 4 풀이"
    $n = 10^7$에서는 실제 선택이 하드웨어에 달렸다. CPU에서는 캐시 친화적인 알고리즘(병합 정렬, 인트로 정렬)이 앞선다. GPU에서는 병렬에 어울리는 알고리즘(바이토닉 정렬, 기수 정렬)이 낫다. 기억이 빠듯하면 제자리 정렬이 $O(n)$ 공간을 아낀다. 이 쪽의 이론적 분석이 그 고름에 길잡이가 된다.

## 정리하며

이 마당은 전략、깊이 한계、복잡도、구현을 차례로 짚었다.

**참고 문헌**

- Musser, D. R. (1997). Introspective sorting and selection algorithms. *Software: Practice and Experience*, 27(8), 983-993.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press.
