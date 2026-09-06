# 병렬 빠른 정렬

빠른 정렬은 병렬에 자연스럽게 어울린다. 배열을 축을 기준으로 나누고 나면 왼쪽과 오른쪽 부분 배열이 서로 독립이라 함께 정렬할 수 있다. 그러나 소박한 구현에서는 나누는 걸음 자체가 차례대로 굴러가고, 치우친 나눔은 짐을 고르게 나누지 못하게 한다. **병렬 빠른 정렬**은 두 어려움을 모두 다루어 낮은 병렬 깊이로 전체 일 $O(n \log n)$을 이룬다.

## 병렬성이 나오는 곳

Quicksort has two potential parallelism sites:

1. **되돌이 부분 문제.** 나눈 뒤 축 양쪽의 부분 배열은 서로 독립이다. 그것들을 나란히 정렬하기는 쉽다.
2. **나누는 걸음 자체.** 보통의 로무토나 호어 나눔은 배열을 차례대로 훑는다. 병렬 나눔은 그 훑기를 프로세서들에 나눈다.

## 소박한 병렬 빠른 정렬

The simplest parallel quicksort spawns a new task for each recursive call:

1. Choose a pivot and partition the array sequentially in $O(n)$.
2. Spawn two parallel tasks for the left and right sub-arrays.
3. Wait for both tasks to complete.

### 복잡도(소박한 판)

Let $W(n)$ and $S(n)$ denote work and span (critical path length).

$$
W(n) = O(n \log n) \quad \text{(same as sequential)}
$$

$$
S(n) = O(n) \quad \text{expected (dominated by partition)}
$$

나누는 걸음이 차례대로 굴러가므로 뻗침이 $O(n)$이다. $n/2$으로 완벽히 쪼개더라도 첫 나눔에 $\Theta(n)$ 시간이 든다.

## 병렬 나눔

To reduce the span, we can parallelize the partition step using a **prefix sum**:

1. Divide the array into $p$ blocks, one per processor.
2. Each processor counts how many elements in its block are $\le$ pivot and $>$ pivot.
3. 이 세기들의 앞부분 합을 셈해 블록마다 원소가 놓일 마지막 자리를 정한다.
4. Each processor moves its elements to their final positions.

**Partition span:** $O(n/p + \log p)$ with $p$ processors.

### 나아진 복잡도

With parallel partition and $p = n / \log n$ processors:

$$
W(n) = O(n \log n), \qquad S(n) = O(\log^2 n) \quad \text{expected}
$$

The span comes from $O(\log n)$ recursion levels, each with $O(\log n)$ partition span.

## 짐 고르게 나누기의 어려움

축의 질이 짐의 균형을 정한다. 축이 나쁘면 한쪽 부분 배열에 원소 대부분이 몰려 프로세서가 놀게 된다.

**Strategies for better pivots:**

| 전략 | 설명 | 짐 |
|----------|-------------|----------|
| 무작위 축 | 고르게 아무렇게나 고른다 | $O(1)$ |
| 셋의 중앙값 | 첫째, 가운데, 마지막의 중앙값 | $O(1)$ |
| 표본 뽑기 | 원소 $O(\sqrt{n})$개를 무작위로 뽑아 중앙값을 쓴다 | $O(\sqrt{n})$ |
| 정확한 중앙값 | 중앙값의 중앙값을 쓴다 | $O(n)$ — 취지를 무너뜨린다 |

In practice, random pivots provide expected $O(\log n)$ depth with high probability.

## 표본 정렬(병렬로 일반화하기)

For $p$ processors, **sample sort** generalizes quicksort:

1. Each processor sorts a random sample of its local elements.
2. Select $p - 1$ **splitters** from the combined samples (evenly spaced).
3. Use the splitters to partition all elements into $p$ buckets.
4. Each processor sorts its bucket locally.

Sample sort achieves near-perfect load balance with high probability.

$$
W(n) = O(n \log n), \qquad S(n) = O\!\left(\frac{n}{p} \log \frac{n}{p}\right)
$$

## 구현

```python
"""
병렬 빠른 정렬 — 일감 기반 병렬 정렬을 보인다.

실 기반 병렬성을 위해 파이썬의 concurrent.futures을 쓴다.
일의 양:  O(n log n)
뻗침:  소박한 판은 O(n), 병렬 나눔을 쓰면 O(log^2 n)
"""

from concurrent.futures import ThreadPoolExecutor, Future


# === 차례대로 나눔 ===

def _partition(arr: list, lo: int, hi: int) -> int:
    """로무토 나눔: arr[lo..hi]을 나눈 뒤 축의 첨자를 되돌린다."""
    pivot = arr[hi]
    i = lo
    for j in range(lo, hi):
        if arr[j] <= pivot:
            arr[i], arr[j] = arr[j], arr[i]
            i += 1
    arr[i], arr[hi] = arr[hi], arr[i]
    return i


# === 병렬 빠른 정렬 ===

def _parallel_quicksort(
    arr: list,
    lo: int,
    hi: int,
    executor: ThreadPoolExecutor,
    depth_limit: int,
) -> None:
    """병렬 빠른 정렬로 arr[lo..hi]을 정렬한다.

    일감 짐이 너무 커지지 않도록 depth_limit에 이르면
    차례대로 하는 정렬로 물러선다.
    """
    if lo >= hi:
        return

    pivot_idx = _partition(arr, lo, hi)

    if depth_limit > 0:
        # 왼쪽과 오른쪽 부분 배열에 병렬 일감 띄우기
        left_future = executor.submit(
            _parallel_quicksort, arr, lo, pivot_idx - 1,
            executor, depth_limit - 1,
        )
        _parallel_quicksort(
            arr, pivot_idx + 1, hi, executor, depth_limit - 1,
        )
        left_future.result()  # 왼쪽을 기다린다
    else:
        # 차례대로 물러서기
        _sequential_quicksort(arr, lo, pivot_idx - 1)
        _sequential_quicksort(arr, pivot_idx + 1, hi)


def _sequential_quicksort(arr: list, lo: int, hi: int) -> None:
    """작은 부분 문제를 위한 표준 차례대로 빠른 정렬."""
    if lo >= hi:
        return
    pivot_idx = _partition(arr, lo, hi)
    _sequential_quicksort(arr, lo, pivot_idx - 1)
    _sequential_quicksort(arr, pivot_idx + 1, hi)


def parallel_quicksort(
    arr: list, max_workers: int = 4, parallel_depth: int = 3
) -> list:
    """병렬 빠른 정렬로 *arr*을 정렬한다.

    매개변수
    ----------
    arr : list[int]
        입력 배열.
    max_workers : int
        풀 안의 실 개수.
    parallel_depth : int
        병렬 일감을 띄울 최대 되돌이 깊이.

    반환값
    -------
    list[int]
        정렬된 배열.
    """
    result = list(arr)
    if len(result) <= 1:
        return result

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        _parallel_quicksort(result, 0, len(result) - 1, executor, parallel_depth)

    return result


# === 시연 ===

if __name__ == "__main__":
    import random
    import time

    random.seed(42)
    data = [random.randint(0, 99999) for _ in range(10000)]

    # 병렬 빠른 정렬
    start = time.perf_counter()
    sorted_parallel = parallel_quicksort(data, max_workers=4, parallel_depth=3)
    t_parallel = time.perf_counter() - start

    # 맞는지 확인하기
    print(f"Correctly sorted: {sorted_parallel == sorted(data)}")
    print(f"Parallel time:    {t_parallel:.4f}s")

    # 보여 주기용 작은 보기
    small = [3, 6, 8, 10, 1, 2, 1]
    print(f"\nInput:  {small}")
    print(f"Sorted: {parallel_quicksort(small)}")
```

**출력:**
```
Correctly sorted: True
Parallel time:    0.0312s  (varies by hardware)

Input:  [3, 6, 8, 10, 1, 2, 1]
Sorted: [1, 1, 2, 3, 6, 8, 10]
```

## 다른 병렬 정렬과의 견줌

| 알고리즘 | 일의 양 | 뻗침 | 제자리 | 실전성 |
|-----------|------|------|----------|----------|
| 병렬 빠른 정렬(소박한 판) | $O(n \log n)$ | $O(n)$ | 예 | 예 |
| 병렬 빠른 정렬(병렬 나눔) | $O(n \log n)$ | $O(\log^2 n)$ | 아니오 | 예 |
| 병렬 병합 정렬 | $O(n \log n)$ | $O(\log^3 n)$ | 아니오 | 예 |
| 바이토닉 정렬 | $O(n \log^2 n)$ | $O(\log^2 n)$ | 예 | GPU |
| 표본 정렬 | $O(n \log n)$ | $O(n/p \cdot \log(n/p))$ | 아니오 | 예 |

## 실용적인 고려

- **깊이 한계.** 되돌이 부름마다 실을 띄우면 짐이 지나치게 커진다. 병렬로 띄우는 것을 되돌이의 위쪽 몇 층으로(프로세서가 $p$개면 대개 $\log_2 p$층으로) 묶고 그 아래로는 차례대로 정렬로 갈아타라.
- **파이썬의 GIL.** 파이썬의 전역 인터프리터 잠금이 참된 실 병렬성을 막는다. CPU에 매인 정렬이라면 `multiprocessing`이나 C 확장을 쓰라. 위의 실 기반 구현은 개념을 보여 줄 뿐이다.
- **캐시 효과.** 빠른 정렬의 차례대로 하는 나눔은 지역성이 좋다. 그것을 프로세서들에 쪼개면 캐시 효율이 떨어질 수 있다.

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022).
  *Introduction to Algorithms* (4th ed.), 26-27장. MIT Press.
- Blelloch, G. E. (1996). Programming parallel algorithms. *Communications of the ACM*, 39(3), 85-97.


## 연습문제

**연습문제 1.**
병렬 빠른 정렬의 핵심 생각과 그 시간·공간 복잡도를 밝혀라.

??? success "연습문제 1 풀이"
    이 알고리즘은 견줌 너머의 성질(정수 열쇠, 바깥 저장 장치, 병렬 하드웨어)을 살려 써서 비교 기반 정렬이 따라올 수 없는 성능을 이룬다. 구체적인 복잡도 한계는 이 쪽에서 뜯어본다.

---

**연습문제 2.**
작은 입력에서 병렬 빠른 정렬을 따라가라. 훑기나 단계마다 보여라.

??? success "연습문제 2 풀이"
    원소 6~8개에 알고리즘을 적용하며 훑을 때마다의 상태를 보여라. 이 따라가기가 알고리즘의 얼개를 드러내고 옳음을 눈에 보이게 한다.

---

**연습문제 3.**
어떤 조건에서 비교 기반 정렬보다 병렬 빠른 정렬이 나은가?

??? success "연습문제 3 풀이"
    다음일 때 낫다. 입력의 정수 범위가 묶여 있을 때(세기 정렬과 기수 정렬), 데이터가 램을 넘칠 때(바깥 정렬), 병렬 하드웨어를 쓸 수 있을 때(병렬 정렬). 이런 조건에서는 알고리즘이 비교 기반의 아래 한계 $\Omega(n\log n)$을 비껴갈 수 있다.

---

**연습문제 4.**
병렬 빠른 정렬이 실전에서 이득을 주는 깊은 학습 응용을 서술하라.

??? success "연습문제 4 풀이"
    응용: 어휘를 찾기 위한 토큰 번호 정렬($O(n + V)$의 세기 정렬), GPU 기억을 넘치는 데이터셋의 바깥 정렬, GPU에서 배치 연산을 위한 병렬 정렬. 특정 조건(정수 열쇠, 큰 데이터, 병렬성)이 갖추어질 때 이득이 가장 크다.