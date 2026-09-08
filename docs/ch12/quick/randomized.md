# 무작위 빠른 정렬

축 규칙이 붙박인(이를테면 늘 첫 원소를 고르는) 정해진 빠른 정렬은, 가장 치우친 나눔이 나오도록 입력을 지어내는 적수에게 $O(n^2)$으로 몰릴 수 있다. **무작위 빠른 정렬**은 부분 배열에서 축을 고르게 무작위로 골라 이 약점을 없앤다. 축이 무작위이므로 어떤 붙박이 입력도 최악의 거동을 한결같이 끌어내지 못한다. 기대 도는 시간은 "무작위 입력에 대한 평균"이 아니라 모든 입력에서 $O(n \log n)$이 된다.

---

## 1. 핵심 생각

정해진 빠른 정렬과 달라지는 것은 축 고르기 걸음뿐이다. $A[lo..hi]$을 나누기 전에 알고리즘은 고르게 무작위인 첨자 $r \in [lo, hi]$을 골라 $A[r]$을 $A[hi]$(나눔 방식에 따라 $A[lo]$)과 맞바꾼다. 그다음 나눔은 로무토나 호어와 똑같이 나아간다.

축이 어느 원소든 될 가능성이 같으므로 되돌이 트리의 기대 깊이는 $O(\log n)$이고 기대 견줌 총 횟수는 다음과 같다.

$$
E[C(n)] = 2n \ln n + O(n) \approx 1.386\, n \log_2 n
$$

이 기댓값은 입력 분포가 아니라 알고리즘의 무작위 선택에 대한 것이다. 다시 말해 이 보장은 모든 입력에서 성립한다.

---

## 2. 기대 견줌 횟수 분석

원소를 정렬한 차례를 $z_1 < z_2 < \cdots < z_n$이라 하자. 정렬 도중 $z_i$과 $z_j$이 한 번이라도 견주어지면 $X_{ij} = 1$인 지시 변수를 정의한다. 견줌은 많아야 한 번 일어나므로($z_i, z_j$ 가운데 하나가 축으로 뽑힐 때) 견줌의 총 횟수는 다음과 같다.

$$
C(n) = \sum_{i=1}^{n-1} \sum_{j=i+1}^{n} X_{ij}
$$

원소 $z_i$과 $z_j$은 $\{z_{i+1}, \ldots, z_{j-1}\}$의 어떤 원소보다 먼저 둘 가운데 하나가 축으로 뽑힐 때에만 견주어진다. $\{z_i, z_{i+1}, \ldots, z_j\}$의 원소 $j - i + 1$개가 이 모임에서 축으로 처음 뽑힐 가능성이 같으므로 다음과 같다.

$$
P(X_{ij} = 1) = \frac{2}{j - i + 1}
$$

기댓값을 취해 더하면 다음과 같다.

$$
E[C(n)] = \sum_{i=1}^{n-1} \sum_{j=i+1}^{n} \frac{2}{j - i + 1} = 2n H_n - O(n) = 2n \ln n + O(n)
$$

여기서 $H_n = \sum_{k=1}^{n} 1/k$은 $n$번째 조화수이다.

---

## 3. 복잡도

| 경우 | 시간 | 공간 |
|------|------|-------|
| 최선 | $O(n \log n)$ | $O(\log n)$ |
| 기대값(어떤 입력이든) | $O(n \log n)$ | 기대값 $O(\log n)$ |
| 최악 | $O(n^2)$ | $O(n)$ |

최악의 경우는 $O(n^2)$이지만 그 확률이 많아야 $O(1/n!)$이라 실전에서는 하찮다.

---

## 4. 구현

```python
"""
제자리 로무토 나눔을 쓰는 무작위 빠른 정렬.

축을 고르게 무작위로 고르므로 기대 도는 시간이 모든 입력에서
O(n log n)이 되어 적수의 최악 경우가 사라진다.
"""

import random

# === 무작위 나눔 ===

def randomized_partition(arr: list, lo: int, hi: int) -> int:
    """아무렇게나 고른 축을 기준으로 arr[lo..hi]을 나눈다.

    축 원소의 마지막 첨자를 되돌린다.
    로무토 나눔 방식을 쓴다.
    """
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

# === 무작위 빠른 정렬 ===

def randomized_quicksort(arr: list, lo: int, hi: int) -> None:
    """무작위 빠른 정렬로 arr[lo..hi]을 제자리에서 정렬한다."""
    if lo < hi:
        pivot_pos = randomized_partition(arr, lo, hi)
        randomized_quicksort(arr, lo, pivot_pos - 1)
        randomized_quicksort(arr, pivot_pos + 1, hi)

# === 시연 ===

if __name__ == "__main__":
    random.seed(42)

    data = [3, 6, 8, 10, 1, 2, 1]
    print(f"Before: {data}")
    randomized_quicksort(data, 0, len(data) - 1)
    print(f"After:  {data}")
    print()

    # 정렬된 입력 — 더는 적수가 아니다
    sorted_input = list(range(1, 16))
    print(f"Sorted input: {sorted_input}")
    randomized_quicksort(sorted_input, 0, len(sorted_input) - 1)
    print(f"After:        {sorted_input}")
    print()

    # 거꾸로 정렬된 입력
    reverse_input = list(range(15, 0, -1))
    print(f"Reverse input: {reverse_input}")
    randomized_quicksort(reverse_input, 0, len(reverse_input) - 1)
    print(f"After:         {reverse_input}")
```

**출력:**
```
Before: [3, 6, 8, 10, 1, 2, 1]
After:  [1, 1, 2, 3, 6, 8, 10]

Sorted input: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
After:        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

Reverse input: [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
After:         [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
```

!!! warning "무작위 보장과 정해진 보장"
    무작위 빠른 정렬은 모든 입력에서 **기대** 시간 $O(n \log n)$을 보장하지만 최악의 경우는 여전히 $O(n^2)$이다. 최악의 경우도 엄밀히 $O(n \log n)$이어야 한다면, 되돌이 깊이가 문턱값을 넘으면 힙 정렬로 물러서는 인트로 정렬을 쓰라.

---

## 연습문제

**연습문제 1.**
$[38, 27, 43, 3, 9, 82, 10]$에서 무작위 빠른 정렬을 따라가며 큰 걸음마다의 상태를 보여라.

??? success "연습문제 1 풀이"
    알고리즘을 한 걸음씩 밟아라. 나누어 정복하는 정렬이라면 되돌이 나눔과 병합을 하나씩 보이고, 나눔 기반 정렬이라면 나눔마다와 축이 놓이는 자리를 보여라. 걸음마다 배열 전체의 상태를 내보여라.

---

**연습문제 2.**
무작위 빠른 정렬의 최선, 최악, 평균의 경우 시간 복잡도를 끌어내라.

??? success "연습문제 2 풀이"
    경우마다 점화식을 써라. 최선의 경우는 가장 좋은 나눔이거나 미리 정렬된 입력이다. 최악의 경우는 일을 가장 크게 만드는 적수 입력이다. 평균의 경우는 무작위 순열에 대한 기대 성능이다. 점화식을 각각 풀어라.

---

**연습문제 3.**
무작위 빠른 정렬은 안정적인가? 증명하거나 반례를 들어라.

??? success "연습문제 3 풀이"
    같은 원소가 본디 상대 차례를 지키면 그 정렬은 안정적이다. 모든 입력에서 이것이 성립함을 증명하거나, 정렬 도중 같은 원소 둘의 자리가 뒤바뀌는 구체적인 입력을 들어라.

---

**연습문제 4.**
float32 학습 손실 값 1000만 개를 정렬할 때 무작위 빠른 정렬을 다른 두 방법과 견주어라. 시간, 공간, 캐시 거동, GPU 어울림을 살펴라.

??? success "연습문제 4 풀이"
    $n = 10^7$에서는 실제 선택이 하드웨어에 달렸다. CPU에서는 캐시 친화적인 알고리즘(병합 정렬, 인트로 정렬)이 앞선다. GPU에서는 병렬에 어울리는 알고리즘(바이토닉 정렬, 기수 정렬)이 낫다. 기억이 빠듯하면 제자리 정렬이 $O(n)$ 공간을 아낀다. 이 쪽의 이론적 분석이 그 고름에 길잡이가 된다.

## 정리하며

이 마당은 핵심 생각、기대 견줌 횟수 분석、복잡도、구현을 차례로 짚었다.

**참고 문헌**

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), 7장. MIT Press.
