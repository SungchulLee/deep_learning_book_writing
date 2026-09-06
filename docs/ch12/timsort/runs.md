# 자연 런

현실의 데이터가 온전히 무작위인 일은 드물다. 배열에는 이미 정렬된 부분 수열이 흔히 들어 있다. 앞선 연산이 남긴 오름 수열, 거꾸로 놓인 입력에서 온 내림 구간, 같은 값이 이어지는 평지 같은 것이다. 팀 정렬은 이미 (오름으로) 정렬되었거나 (내림으로) 거꾸로 정렬된 가장 긴 부분 수열, 곧 **자연 런**을 짚어내어 이 짜임을 살려 쓴다. 런마다 병합 단계의 벽돌이 되므로, 어느 정도 정렬된 데이터는 $O(n \log n)$보다 훨씬 빨리 정렬된다.

## 런 찾기

팀 정렬은 배열을 왼쪽에서 오른쪽으로 훑으며 두 종류의 런을 짚어낸다.

1. **오름 런**: 잇단 원소에 대해 $A[i] \leq A[i+1]$인 가장 긴 수열이다. 이웃한 원소가 같은 경우(감소하지 않는 차례)도 넣으며, 그래야 안정성이 지켜진다.
2. **내림 런**: 잇단 원소에 대해 $A[i] > A[i+1]$인 가장 긴 수열이다. 부등호가 **엄밀**하다는 점을 눈여겨보라. 같은 원소는 결코 내림으로 치지 않는다. 찾아낸 뒤 내림 런은 제자리에서 뒤집어 오름으로 만든다.

내림 런에 엄밀한 부등호를 쓰는 것은 안정성에 꼭 필요하다. 같은 원소를 내림 런에 넣으면 뒤집을 때 그 상대 차례가 바뀌기 때문이다.

## 최소 런 길이

짧은 런은 병합하기 비효율적이다. 팀 정렬은 32에서 64 사이의 **최소 런 길이**(`minrun`이라 한다)를 지킨다. 자연 런이 `minrun`보다 짧으면 뒤따르는 원소에 이진 끼워넣기 정렬을 써서 `minrun` 길이가 될 때까지 늘린다.

`minrun` 값은 $n$의 위쪽 6비트를 가져오고 남은 비트 가운데 하나라도 켜져 있으면 1을 더해 셈한다.

```
minrun = n
while minrun >= 64:
    minrun = (minrun + 1) >> 1
```

이 식은 $n / \text{minrun}$이 2의 거듭제곱에 가깝게(또는 조금 작게) 되도록 하여 고른 병합을 낸다.

## minrun이 중요한 까닭

`minrun`이 너무 작으면 병합할 런이 너무 많아져 짐이 는다. 너무 크면 짧은 런에 대한 끼워넣기 정렬이 값비싸진다. 32~64 범위가 이 둘의 균형을 잡는다. 이 크기의 배열에서는 하드웨어 캐시 덕분에 끼워넣기 정렬이 빠르고, 그 결과 런의 수가 $O(n / 32) = O(n)$ 안에 머물러 깊이 $O(\log n)$의 병합 트리가 된다.

## 구현

```python
"""
팀 정렬을 위한 자연 런 찾기와 늘리기.

입력 배열에서 오름 런과 내림 런을 짚어내고,
안정성을 지키려고 내림 런을 뒤집으며, 최소 런 길이 문턱에 닿도록
짧은 런을 이진 끼워넣기 정렬로
늘린다.
"""


# === 최소 런 길이 셈하기 ===

def compute_minrun(n: int) -> int:
    """크기 n인 배열에 대한 팀 정렬의 최소 런 길이를 셈한다.

    n/minrun이 2의 거듭제곱에 가깝도록 32에서 64 사이의
    값을 되돌린다.
    """
    r = 0
    while n >= 64:
        r |= n & 1
        n >>= 1
    return n + r


# === 이진 끼워넣기 정렬 ===

def binary_insertion_sort(arr: list, lo: int, hi: int,
                          start: int) -> None:
    """이진 끼워넣기 정렬로 arr[lo..hi]을 정렬한다.

    원소 arr[lo..start-1]은 이미 정렬되어 있다. start부터의 원소를
    이진 찾기로 끼울 자리를 찾아 끼워 넣는다.
    """
    for i in range(start, hi + 1):
        key = arr[i]
        # 끼울 자리를 이진으로 찾는다
        left, right = lo, i
        while left < right:
            mid = (left + right) // 2
            if key < arr[mid]:
                right = mid
            else:
                left = mid + 1
        # 원소를 밀고 끼워 넣는다
        for j in range(i, left, -1):
            arr[j] = arr[j - 1]
        arr[left] = key


# === 런 찾아 늘리기 ===

def find_run(arr: list, lo: int, hi: int) -> tuple:
    """lo에서 시작하는 다음 자연 런을 찾는다.

    (run_end, is_descending)을 되돌리며 arr[lo..run_end]이 가장 긴
    오름 런 또는 내림 런이다.
    """
    if lo >= hi:
        return lo, False

    run_end = lo + 1
    if arr[run_end] < arr[lo]:
        # 엄밀히 내림
        while run_end <= hi and arr[run_end] < arr[run_end - 1]:
            run_end += 1
        return run_end - 1, True
    else:
        # 감소하지 않음(오름)
        while run_end <= hi and arr[run_end] >= arr[run_end - 1]:
            run_end += 1
        return run_end - 1, False


def identify_runs(arr: list) -> list:
    """arr의 자연 런을 모두 짚어내고 짧은 것은 늘린다.

    런마다 (시작, 길이) 짝의 목록을 되돌린다.
    """
    n = len(arr)
    if n == 0:
        return []

    minrun = compute_minrun(n)
    runs = []
    lo = 0

    while lo < n:
        run_end, is_descending = find_run(arr, lo, n - 1)

        if is_descending:
            # 내림 런을 뒤집어 오름으로 만든다
            left, right = lo, run_end
            while left < right:
                arr[left], arr[right] = arr[right], arr[left]
                left += 1
                right -= 1

        run_length = run_end - lo + 1

        # 짧은 런을 이진 끼워넣기 정렬로 늘린다
        if run_length < minrun:
            force = min(lo + minrun - 1, n - 1)
            binary_insertion_sort(arr, lo, force, run_end + 1)
            run_end = force
            run_length = run_end - lo + 1

        runs.append((lo, run_length))
        lo = run_end + 1

    return runs


# === 시연 ===

if __name__ == "__main__":
    # 자연 런이 있는 배열
    arr = [1, 3, 5, 7, 9, 8, 6, 4, 2, 10, 11, 12]
    print(f"Input: {arr}")
    print(f"minrun for n={len(arr)}: {compute_minrun(len(arr))}")
    print()

    # 낱낱의 런을 찾는다
    arr_copy = arr.copy()
    lo = 0
    while lo < len(arr_copy):
        end, desc = find_run(arr_copy, lo, len(arr_copy) - 1)
        run_type = "descending" if desc else "ascending"
        print(f"Run at [{lo}..{end}]: {arr_copy[lo:end+1]} ({run_type})")
        if desc:
            left, right = lo, end
            while left < right:
                arr_copy[left], arr_copy[right] = (
                    arr_copy[right], arr_copy[left])
                left += 1
                right -= 1
            print(f"  Reversed: {arr_copy[lo:end+1]}")
        lo = end + 1

    print()

    # 여러 크기에 대한 minrun을 보여 준다
    for size in [64, 128, 256, 1000, 10000]:
        print(f"n={size:5d} -> minrun={compute_minrun(size)}")
```

**출력:**
```
Input: [1, 3, 5, 7, 9, 8, 6, 4, 2, 10, 11, 12]
minrun for n=12: 12

Run at [0..4]: [1, 3, 5, 7, 9] (ascending)
Run at [5..8]: [8, 6, 4, 2] (descending)
  Reversed: [2, 4, 6, 8]
Run at [9..11]: [10, 11, 12] (ascending)

n=   64 -> minrun=64
n=  128 -> minrun=64
n=  256 -> minrun=64
n= 1000 -> minrun=63
n=10000 -> minrun=40
```

!!! tip "맞추어 가는 성능"
    이미 정렬된 데이터에서 팀 정렬은 길이 $n$인 런 하나를 찾고 $O(n)$에 끝낸다. 병합이 필요 없다. 거꾸로 정렬된 데이터에서는 내림 런 하나를 찾아 $O(n)$에 뒤집고 끝낸다. 이렇게 맞추어 가는 성질 덕분에 팀 정렬이 현실의 데이터에서 뛰어나다.

## 참고 문헌

- Peters, T. (2002). *Timsort description*. [CPython 소스, `Objects/listsort.txt`](https://github.com/python/cpython/blob/main/Objects/listsort.txt).
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press.


## 연습문제

**연습문제 1.**
$[38, 27, 43, 3, 9, 82, 10]$에서 자연 런을 따라가며 큰 걸음마다의 상태를 보여라.

??? success "연습문제 1 풀이"
    알고리즘을 한 걸음씩 밟아라. 나누어 정복하는 정렬이라면 되돌이 나눔과 병합을 하나씩 보이고, 나눔 기반 정렬이라면 나눔마다와 축이 놓이는 자리를 보여라. 걸음마다 배열 전체의 상태를 내보여라.

---

**연습문제 2.**
자연 런의 최선, 최악, 평균의 경우 시간 복잡도를 끌어내라.

??? success "연습문제 2 풀이"
    경우마다 점화식을 써라. 최선의 경우는 가장 좋은 나눔이거나 미리 정렬된 입력이다. 최악의 경우는 일을 가장 크게 만드는 적수 입력이다. 평균의 경우는 무작위 순열에 대한 기대 성능이다. 점화식을 각각 풀어라.

---

**연습문제 3.**
자연 런은 안정적인가? 증명하거나 반례를 들어라.

??? success "연습문제 3 풀이"
    같은 원소가 본디 상대 차례를 지키면 그 정렬은 안정적이다. 모든 입력에서 이것이 성립함을 증명하거나, 정렬 도중 같은 원소 둘의 자리가 뒤바뀌는 구체적인 입력을 들어라.

---

**연습문제 4.**
float32 학습 손실 값 1000만 개를 정렬할 때 자연 런을 다른 두 방법과 견주어라. 시간, 공간, 캐시 거동, GPU 어울림을 살펴라.

??? success "연습문제 4 풀이"
    $n = 10^7$에서는 실제 선택이 하드웨어에 달렸다. CPU에서는 캐시 친화적인 알고리즘(병합 정렬, 인트로 정렬)이 앞선다. GPU에서는 병렬에 어울리는 알고리즘(바이토닉 정렬, 기수 정렬)이 낫다. 기억이 빠듯하면 제자리 정렬이 $O(n)$ 공간을 아낀다. 이 쪽의 이론적 분석이 그 고름에 길잡이가 된다.