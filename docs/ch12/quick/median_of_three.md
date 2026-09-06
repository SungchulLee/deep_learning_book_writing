# 셋의 중앙값 축 고르기

빠른 정렬의 성능은 축의 질에 크게 달렸다. 첫 원소나 마지막 원소를 축으로 고르면 정렬되었거나 거의 정렬된 입력에서 $O(n^2)$ 거동이 나오는데, 하필 실전에서 가장 자주 나타나는 입력이 그렇다. **셋의 중앙값** 축 고르기는 원소 셋(대개 첫째, 가운데, 마지막)을 살펴 그 중앙값을 축으로 쓴다. 이 단순한 어림법은 정렬된 데이터에서 최악의 경우를 피하고 평균으로 더 고른 나눔을 낸다.

## 왜 필요한가

이미 오름차순으로 정렬된 배열을 생각해 보자. 늘 $A[lo]$을 축으로 고르면 나눌 때마다 빈 부분 배열 하나와 크기 $n - 1$인 부분 배열 하나가 나와 $T(n) = T(n-1) + \Theta(n) = \Theta(n^2)$이 된다. 셋의 중앙값 어림법은 $\{A[lo],\, A[\lfloor(lo+hi)/2\rfloor],\, A[hi]\}$의 중앙값을 고른다. 정렬된 배열에서 이 중앙값은 늘 가운데 원소여서 깊이 $O(\log n)$의 완벽하게 고른 나눔이 나온다.

## 알고리즘

첨자 $lo$과 $hi$이 주어지면 $mid = \lfloor(lo + hi) / 2\rfloor$이라 하자. 셋의 중앙값 절차는 다음과 같다.

1. $A[lo]$, $A[mid]$, $A[hi]$을 견준다.
2. 이 셋 가운데 중앙값을 지닌 원소를 짚어낸다.
3. 그 중앙값 원소를 축 자리로 맞바꾼다(로무토는 $A[hi]$, 호어는 $A[lo]$).
4. 보통의 나눔을 이어 간다.

이 세 번의 견줌은 곁다리로 그 세 원소를 어느 정도 정렬해 주기도 하는데, 그러면 부분 배열의 양 끝에 파수병이 놓여 호어 나눔의 안쪽 되돌이에서 범위를 살필 필요가 없어진다.

## 분석

고르게 무작위인 순열에서 뽑은 원소 셋의 중앙값은 기대 순위가 $n/2$이라 고른 나눔을 낸다. 더 정확히는 축이 배열의 가운데 3분의 1에 떨어질 확률이 다음과 같다.

$$
P\!\left(\frac{n}{3} \leq \text{순위} \leq \frac{2n}{3}\right) = \frac{11}{27} \approx 0.407
$$

무작위 원소 하나일 때의 $1/3 \approx 0.333$과 견주어 보라. 기대 견줌 횟수는 $\approx 1.386\, n \ln n$(무작위 축)에서 $\approx 1.188\, n \ln n$(셋의 중앙값)으로 떨어진다.

!!! tip "아홉수(셋의 중앙값의 중앙값)"
    아주 큰 배열에서는 **아홉수**를 쓰는 구현도 있다. 원소 셋씩 세 무리를 잡아 무리마다 중앙값을 찾고, 그 세 중앙값의 중앙값을 쓴다. 나눔마다 견줌이 늘어나는 대신 축을 더 잘 어림하며, 벤틀리-매킬로이의 빠른 정렬 다듬기에 쓰인다.

## 구현

```python
"""
빠른 정렬을 위한 셋의 중앙값 축 고르기.

첫째, 가운데, 마지막 원소의 중앙값을 고르면 정렬된 입력에서
최악의 거동을 피하고 평균으로 더 고른 나눔이 나옴을
보여 준다.
"""


# === 셋의 중앙값 고르기 ===

def median_of_three(arr: list, lo: int, hi: int) -> int:
    """arr[lo], arr[mid], arr[hi]의 중앙값 첨자를 되돌린다.

    곁다리로 이 세 원소를 어느 정도 정렬하여
    arr[lo] <= arr[mid] <= arr[hi]이 되게 한다.
    """
    mid = (lo + hi) // 2
    if arr[lo] > arr[mid]:
        arr[lo], arr[mid] = arr[mid], arr[lo]
    if arr[lo] > arr[hi]:
        arr[lo], arr[hi] = arr[hi], arr[lo]
    if arr[mid] > arr[hi]:
        arr[mid], arr[hi] = arr[hi], arr[mid]
    return mid


# === 셋의 중앙값을 쓰는 빠른 정렬 ===

def quicksort_mot(arr: list, lo: int, hi: int) -> None:
    """셋의 중앙값 축 고르기를 쓰는 빠른 정렬."""
    if lo >= hi:
        return

    # 축을 골라 arr[hi - 1]로 옮긴다
    pivot_idx = median_of_three(arr, lo, hi)
    arr[pivot_idx], arr[hi - 1] = arr[hi - 1], arr[pivot_idx]
    pivot = arr[hi - 1]

    # 호어와 비슷한 나눔(arr[lo]와 arr[hi]가 파수병이다)
    i = lo
    j = hi - 1
    while True:
        i += 1
        while arr[i] < pivot:
            i += 1
        j -= 1
        while arr[j] > pivot:
            j -= 1
        if i >= j:
            break
        arr[i], arr[j] = arr[j], arr[i]

    # 축을 마지막 자리에 놓는다
    arr[i], arr[hi - 1] = arr[hi - 1], arr[i]

    quicksort_mot(arr, lo, i - 1)
    quicksort_mot(arr, i + 1, hi)


# === 시연 ===

if __name__ == "__main__":
    # 무작위 입력
    data = [38, 27, 43, 3, 9, 82, 10, 55, 1, 72]
    print(f"Before: {data}")
    quicksort_mot(data, 0, len(data) - 1)
    print(f"After:  {data}")
    print()

    # 정렬된 입력(소박한 축의 최악의 경우)
    sorted_data = list(range(1, 11))
    print(f"Sorted input: {sorted_data}")
    quicksort_mot(sorted_data, 0, len(sorted_data) - 1)
    print(f"After:        {sorted_data}")
    print()

    # 축 고르기를 보여 준다
    example = [50, 10, 30, 90, 70]
    print(f"Array: {example}")
    idx = median_of_three(example, 0, len(example) - 1)
    print(f"Median-of-three index: {idx}, value: {example[idx]}")
    print(f"After partial sort: {example}")
```

**출력:**
```
Before: [38, 27, 43, 3, 9, 82, 10, 55, 1, 72]
After:  [1, 3, 9, 10, 27, 38, 43, 55, 72, 82]

Sorted input: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
After:        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

Array: [50, 10, 30, 90, 70]
Median-of-three index: 2, value: 50
After partial sort: [30, 10, 50, 90, 70]
```

## 복잡도

| 축 전략 | 평균 견줌 | 최악의 경우 | 정렬된 입력 |
|----------------|-----------------|------------|--------------|
| 첫 원소 | $\approx 1.386\, n \ln n$ | $O(n^2)$ | $O(n^2)$ |
| 무작위 원소 | $\approx 1.386\, n \ln n$ | $O(n^2)$, 기대값 $O(n \log n)$ | 기대값 $O(n \log n)$ |
| 셋의 중앙값 | $\approx 1.188\, n \ln n$ | $O(n^2)$ | $O(n \log n)$ |

셋의 중앙값이 $O(n^2)$ 최악의 경우를 아예 없애지는 못한다. 적수는 여전히 이를 깨뜨리는 입력을 지어낼 수 있다. 그러나 그런 입력이 저절로 생기지는 않으며, 셋의 중앙값에 인트로 정렬의 깊이 한계를 곁들이면 최악의 경우 $O(n \log n)$이 된다.

## 참고 문헌

- Sedgewick, R. (1978). Implementing Quicksort programs. *Communications of the ACM*, 21(10), 847-857.
- Bentley, J. L., & McIlroy, M. D. (1993). Engineering a sort function. *Software: Practice and Experience*, 23(11), 1249-1265.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press.


## 연습문제

**연습문제 1.**
$[38, 27, 43, 3, 9, 82, 10]$에서 셋의 중앙값 축 고르기를 따라가며 큰 걸음마다의 상태를 보여라.

??? success "연습문제 1 풀이"
    알고리즘을 한 걸음씩 밟아라. 나누어 정복하는 정렬이라면 되돌이 나눔과 병합을 하나씩 보이고, 나눔 기반 정렬이라면 나눔마다와 축이 놓이는 자리를 보여라. 걸음마다 배열 전체의 상태를 내보여라.

---

**연습문제 2.**
셋의 중앙값 축 고르기의 최선, 최악, 평균의 경우 시간 복잡도를 끌어내라.

??? success "연습문제 2 풀이"
    경우마다 점화식을 써라. 최선의 경우는 가장 좋은 나눔이거나 미리 정렬된 입력이다. 최악의 경우는 일을 가장 크게 만드는 적수 입력이다. 평균의 경우는 무작위 순열에 대한 기대 성능이다. 점화식을 각각 풀어라.

---

**연습문제 3.**
셋의 중앙값 축 고르기는 안정적인가? 증명하거나 반례를 들어라.

??? success "연습문제 3 풀이"
    같은 원소가 본디 상대 차례를 지키면 그 정렬은 안정적이다. 모든 입력에서 이것이 성립함을 증명하거나, 정렬 도중 같은 원소 둘의 자리가 뒤바뀌는 구체적인 입력을 들어라.

---

**연습문제 4.**
float32 학습 손실 값 1000만 개를 정렬할 때 셋의 중앙값 축 고르기를 다른 두 방법과 견주어라. 시간, 공간, 캐시 거동, GPU 어울림을 살펴라.

??? success "연습문제 4 풀이"
    $n = 10^7$에서는 실제 선택이 하드웨어에 달렸다. CPU에서는 캐시 친화적인 알고리즘(병합 정렬, 인트로 정렬)이 앞선다. GPU에서는 병렬에 어울리는 알고리즘(바이토닉 정렬, 기수 정렬)이 낫다. 기억이 빠듯하면 제자리 정렬이 $O(n)$ 공간을 아낀다. 이 쪽의 이론적 분석이 그 고름에 길잡이가 된다.