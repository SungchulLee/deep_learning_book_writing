# 질주 모드

팀 정렬의 병합 단계에서는 정렬된 런 둘을 원소 하나씩 합친다. 한쪽 런이 한결같이 견줌에서 "이기면"(곧 그 원소가 더 작으면) 보통의 하나씩 병합은 한꺼번에 옮길 수 있는 원소에 견줌을 헛되이 쓴다. **질주 모드**(지수 찾기라고도 한다)는 이렇게 한쪽으로 쏠린 구간을 알아채고 지수 찾기로 끼울 자리를 찾은 다음 그 덩어리를 한 번에 옮긴다. 이 손질은 길이 $m$인 짧은 런을 길이 $n$인 긴 런에 병합할 때 견줌 횟수를 $O(n)$에서 $O(n + m \log(n/m))$으로 줄인다.

---

## 1. 질주가 켜질 때

팀 정렬은 `min_gallop`이라는 세개를 지닌다(처음에는 7). 보통의 병합 중에 같은 런이 잇달아 $\text{min\_gallop}$번 이기면 알고리즘이 질주 모드로 들어간다. 질주가 이롭다고 드러나면(옮길 큰 덩어리를 찾으면) `min_gallop`을 줄여 다음에 질주가 더 쉽게 켜지게 한다. 질주가 원소를 몇 개밖에 찾지 못하면 `min_gallop`을 늘려 문턱을 높이고 하나씩 병합을 편든다.

이 맞추어 가는 문턱값 덕분에 질주가 이로운, 한쪽으로 길게 쏠린 데이터에서만 질주를 쓰게 된다.

---

## 2. 질주 찾기 알고리즘

목표 값 $v$과 정렬된 배열 $B[0..n-1]$이 주어지면 질주 찾기는 $v$을 끼워야 할 자리를 찾는다.

1. **지수로 넓히기**: $k = 1$에서 시작한다. $B[2^k - 1] \geq v$인 자리를 찾거나 배열 끝에 닿을 때까지 자리 $0, 1, 3, 7, 15, \ldots$(곧 $2^k - 1$)을 살핀다.
2. **이진 찾기**: 범위 $[2^{k-1}, \min(2^k - 1, n-1)]$에서 이진 찾기를 한다.

지수 단계는 많아야 $\lceil \log_2(m+1) \rceil$번 견주며, $m$은 $B$에서 $v$보다 작은 원소의 개수이다. 이진 찾기가 $O(\log m)$번을 더한다. 모두 합쳐 $O(\log m)$이며, 죽 훑을 때의 $O(m)$과 견주어 보라.

---

## 3. 질주 병합의 복잡도

길이 $m$인 런을 길이 $n$인 런에 병합할 때($m \leq n$) 질주를 쓰면 견줌 횟수는 다음과 같다.

$$
O(m \log(n/m + 1))
$$

$m \ll n$일 때 보통의 $O(m + n)$ 병합보다 낫고, $m \approx n$일 때도 $O(m + n)$보다 나쁘지 않다.

---

## 4. 구현

```python
"""
팀 정렬의 병합 손질을 위한 질주(지수) 찾기.

병합 중에 한 런이 한결같이 이기면 질주 찾기가 O(m) 대신 O(log m)에
끼울 자리를 찾고, 이긴 원소를
한꺼번에 옮긴다.
"""

# === 질주 찾기 ===

def gallop_right(key, arr: list, lo: int, hi: int) -> int:
    """arr[lo..hi-1]에서 key를 끼울 수 있는 가장 오른쪽 자리를 찾는다.

    지수 찾기를 쓴다. 지나칠 때까지 걸음 크기를 두 배로 늘린 다음,
    찾아낸 범위에서 이진 찾기를 한다.

    arr[lo..i-1] <= key < arr[i..hi-1]을 만족하는 첨자 i을 되돌린다.
    """
    if lo >= hi:
        return lo

    # 지수로 넓히는 단계
    offset = 1
    last_offset = 0

    if key >= arr[lo]:
        # 오른쪽으로 질주한다: 열쇠가 적어도 arr[lo] 이상이다
        max_offset = hi - lo
        while offset < max_offset and key >= arr[lo + offset]:
            last_offset = offset
            offset = (offset << 1) + 1
            if offset <= 0:  # 넘침 막기
                offset = max_offset
        offset = min(offset, max_offset)

        # [lo + last_offset, lo + offset)에서 이진 찾기
        left = lo + last_offset
        right = lo + offset
    else:
        return lo

    # 이진 찾기 단계
    while left < right:
        mid = left + (right - left) // 2
        if key < arr[mid]:
            right = mid
        else:
            left = mid + 1

    return left

def gallop_left(key, arr: list, lo: int, hi: int) -> int:
    """arr[lo..hi-1]에서 key를 끼울 수 있는 가장 왼쪽 자리를 찾는다.

    arr[lo..i-1] < key <= arr[i..hi-1]을 만족하는 첨자 i을 되돌린다.
    """
    if lo >= hi:
        return lo

    offset = 1
    last_offset = 0

    if key > arr[lo]:
        max_offset = hi - lo
        while offset < max_offset and key > arr[lo + offset]:
            last_offset = offset
            offset = (offset << 1) + 1
            if offset <= 0:
                offset = max_offset
        offset = min(offset, max_offset)

        left = lo + last_offset
        right = lo + offset
    else:
        return lo

    while left < right:
        mid = left + (right - left) // 2
        if key <= arr[mid]:
            right = mid
        else:
            left = mid + 1

    return left

# === 시연 ===

if __name__ == "__main__":
    # 안에서 찾을 정렬된 런
    run = [2, 5, 8, 12, 16, 23, 38, 42, 55, 67, 72, 84, 91, 99]
    print(f"Sorted run: {run}")
    print()

    # 여러 열쇠의 끼울 자리를 질주로 찾는다
    for key in [10, 42, 1, 100, 55]:
        pos = gallop_right(key, run, 0, len(run))
        print(f"gallop_right({key:3d}): insert at index {pos}")

    print()

    # 죽 훑기의 횟수와 견준다
    for key in [10, 42, 84]:
        pos = gallop_right(key, run, 0, len(run))
        linear_steps = sum(1 for x in run if x <= key)
        print(f"key={key}: gallop found index {pos}, "
              f"linear scan would check {linear_steps} elements")
```

**출력:**
```
Sorted run: [2, 5, 8, 12, 16, 23, 38, 42, 55, 67, 72, 84, 91, 99]

gallop_right( 10): insert at index 3
gallop_right( 42): insert at index 8
gallop_right(  1): insert at index 0
gallop_right(100): insert at index 14
gallop_right( 55): insert at index 9

key=10: gallop found index 3, linear scan would check 3 elements
key=42: gallop found index 8, linear scan would check 8 elements
key=84: gallop found index 12, linear scan would check 12 elements
```

!!! warning "질주가 해로울 때"
    두 런이 엇갈려 있으면(번갈아 이기면) 질주는 옮길 큰 덩어리를 찾지 못한 채 지수로 넓히는 단계에 견줌을 헛되이 쓴다. 팀 정렬은 질주가 헛돌 때 `min_gallop` 문턱값을 높여, 엇갈린 데이터에서는 질주 모드에 들어가기 어렵게 하여 이를 다룬다.

---

## 연습문제

**연습문제 1.**
$[38, 27, 43, 3, 9, 82, 10]$에서 질주 모드를 따라가며 큰 걸음마다의 상태를 보여라.

??? success "연습문제 1 풀이"
    알고리즘을 한 걸음씩 밟아라. 나누어 정복하는 정렬이라면 되돌이 나눔과 병합을 하나씩 보이고, 나눔 기반 정렬이라면 나눔마다와 축이 놓이는 자리를 보여라. 걸음마다 배열 전체의 상태를 내보여라.

---

**연습문제 2.**
질주 모드의 최선, 최악, 평균의 경우 시간 복잡도를 끌어내라.

??? success "연습문제 2 풀이"
    경우마다 점화식을 써라. 최선의 경우는 가장 좋은 나눔이거나 미리 정렬된 입력이다. 최악의 경우는 일을 가장 크게 만드는 적수 입력이다. 평균의 경우는 무작위 순열에 대한 기대 성능이다. 점화식을 각각 풀어라.

---

**연습문제 3.**
질주 모드는 안정적인가? 증명하거나 반례를 들어라.

??? success "연습문제 3 풀이"
    같은 원소가 본디 상대 차례를 지키면 그 정렬은 안정적이다. 모든 입력에서 이것이 성립함을 증명하거나, 정렬 도중 같은 원소 둘의 자리가 뒤바뀌는 구체적인 입력을 들어라.

---

**연습문제 4.**
float32 학습 손실 값 1000만 개를 정렬할 때 질주 모드를 다른 두 방법과 견주어라. 시간, 공간, 캐시 거동, GPU 어울림을 살펴라.

??? success "연습문제 4 풀이"
    $n = 10^7$에서는 실제 선택이 하드웨어에 달렸다. CPU에서는 캐시 친화적인 알고리즘(병합 정렬, 인트로 정렬)이 앞선다. GPU에서는 병렬에 어울리는 알고리즘(바이토닉 정렬, 기수 정렬)이 낫다. 기억이 빠듯하면 제자리 정렬이 $O(n)$ 공간을 아낀다. 이 쪽의 이론적 분석이 그 고름에 길잡이가 된다.

## 정리하며

이 마당은 질주가 켜질 때、질주 찾기 알고리즘、질주 병합의 복잡도、구현을 차례로 짚었다.

**참고 문헌**

- Peters, T. (2002). *Timsort description*. [CPython 소스, `Objects/listsort.txt`](https://github.com/python/cpython/blob/main/Objects/listsort.txt).
- McIlroy, P. (1993). Optimistic sorting and information theoretic complexity. *Proceedings of SODA*, 467-474.
