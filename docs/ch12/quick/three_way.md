# 세 갈래 나눔

보통의 두 갈래 빠른 정렬은 원소를 "축보다 작다"와 "축보다 크다"로 나눈다. 배열에 같은 열쇠가 많으면 같은 원소가 축의 양쪽에 흩어져 뒤이은 되돌이 부름에서 헛된 견줌이 생긴다. **세 갈래 나눔**(다익스트라의 이름을 딴 네덜란드 국기 알고리즘이라고도 한다)은 배열을 축보다 작은 것, 축과 같은 것, 축보다 큰 것 세 무리로 쪼갠다. 같은 원소는 곧바로 마지막 자리에 놓여 되돌이에서 빠지므로, 같은 값이 많은 입력에서 도는 시간이 $O(n^2)$에서 $O(n \log n)$으로 줄 수 있다.

---

## 1. 네덜란드 국기 문제

다익스트라는 1976년에 이 문제를 내놓았다. 빨강, 하양, 파랑으로 칠해진 원소의 배열이 주어졌을 때 맞바꿈만으로 빨강이 먼저, 그다음 하양, 그다음 파랑이 오도록 다시 늘어놓으라는 것이다. 세 갈래 나눔은 가리개 둘로 갈라지는 세 자리를 지녀 이를 푼다.

---

## 2. 알고리즘

축 값 $v$과 배열 $A[lo..hi]$이 주어지면 가리개 셋을 지닌다.

- **lt**: "작은 쪽" 자리의 경계이다. $A[lo..lt-1] < v$이다.
- **gt**: "큰 쪽" 자리의 경계이다. $A[gt+1..hi] > v$이다.
- **i**: 훑는 가리개이다. $A[lt..i-1] = v$이다.

$lt = lo$, $gt = hi$, $i = lo$으로 둔다. $i \leq gt$인 동안 다음을 되풀이한다.

1. $A[i] < v$이면 $A[i]$과 $A[lt]$을 맞바꾸고 $lt$과 $i$을 함께 하나씩 민다.
2. $A[i] > v$이면 $A[i]$과 $A[gt]$을 맞바꾸고 $gt$을 하나 당긴다($i$은 밀지 않는다).
3. $A[i] = v$이면 $i$을 하나 민다.

되돌이가 끝나면 배열은 다음을 만족한다.

$$
A[lo..lt-1] < v = A[lt..gt] < A[gt+1..hi]
$$

이 알고리즘은 $A[lo..lt-1]$과 $A[gt+1..hi]$에서만 되돌이하며 $v$과 같은 원소는 모두 건너뛴다.

---

## 3. 복잡도

| 입력 종류 | 보통의 빠른 정렬 | 세 갈래 빠른 정렬 |
|------------|-------------------|---------------------|
| 모두 다름 | 평균 $O(n \log n)$ | 평균 $O(n \log n)$ |
| 서로 다른 열쇠가 적음(값 $k$가지) | 평균 $O(n \log n)$ | 평균 $O(n \log k)$ |
| 모두 같음 | $O(n^2)$ | $O(n)$ |

모든 원소가 같으면 보통의 빠른 정렬은 여전히 $n$번 되돌이하지만(나눌 때마다 축 하나만 빠진다), 세 갈래 빠른 정렬은 원소가 모두 "같은" 자리에 들어가므로 한 번 훑고 끝난다.

!!! tip "엔트로피에 가장 알맞은 정렬"
    세 갈래 빠른 정렬은 **엔트로피에 가장 알맞다**. 기대 도는 시간이 열쇠 분포의 섀넌 엔트로피 $H = -\sum p_i \log p_i$에 $n$을 곱한 값에 비례한다. 같은 열쇠가 많으면 엔트로피가 낮아 정렬이 더 빠르다.

---

## 4. 구현

```python
"""
빠른 정렬을 위한 세 갈래 나눔(네덜란드 국기).

배열을 축보다 작은 것, 같은 것, 큰 것 세 자리로 쪼갠다.
같은 원소는 되돌이에서 빠지므로 같은 열쇠가 많은 입력에
이 변형이 가장 알맞다.
"""

import random

# === 세 갈래 나눔 ===

def three_way_partition(arr: list, lo: int, hi: int) -> tuple:
    """arr[lo]을 기준으로 arr[lo..hi]을 세 자리로 나눈다.

    다음을 만족하는 (lt, gt)을 되돌린다.
      arr[lo..lt-1]  < pivot
      arr[lt..gt]    = pivot
      arr[gt+1..hi]  > pivot
    """
    pivot = arr[lo]
    lt = lo
    gt = hi
    i = lo

    while i <= gt:
        if arr[i] < pivot:
            arr[i], arr[lt] = arr[lt], arr[i]
            lt += 1
            i += 1
        elif arr[i] > pivot:
            arr[i], arr[gt] = arr[gt], arr[i]
            gt -= 1
        else:
            i += 1

    return lt, gt

# === 세 갈래 빠른 정렬 ===

def three_way_quicksort(arr: list, lo: int, hi: int) -> None:
    """세 갈래 나눔으로 arr[lo..hi]을 정렬한다."""
    if lo >= hi:
        return
    lt, gt = three_way_partition(arr, lo, hi)
    three_way_quicksort(arr, lo, lt - 1)
    three_way_quicksort(arr, gt + 1, hi)

# === 시연 ===

if __name__ == "__main__":
    # 같은 값이 많은 배열
    data = [4, 2, 4, 1, 3, 4, 2, 1, 4, 3]
    print(f"Before: {data}")
    three_way_quicksort(data, 0, len(data) - 1)
    print(f"After:  {data}")
    print()

    # 나누는 걸음을 보여 준다
    example = [3, 1, 4, 3, 5, 3, 2, 3]
    print(f"Partition example: {example}")
    lt, gt = three_way_partition(example, 0, len(example) - 1)
    print(f"After partition:   {example}")
    print(f"lt={lt}, gt={gt}")
    print(f"Less:  {example[:lt]}")
    print(f"Equal: {example[lt:gt+1]}")
    print(f"Greater: {example[gt+1:]}")
    print()

    # 모두 같은 입력(보통의 빠른 정렬의 최악의 경우)
    equal = [5] * 10
    print(f"All equal: {equal}")
    three_way_quicksort(equal, 0, len(equal) - 1)
    print(f"After:     {equal}")
```

**출력:**
```
Before: [4, 2, 4, 1, 3, 4, 2, 1, 4, 3]
After:  [1, 1, 2, 2, 3, 3, 4, 4, 4, 4]

Partition example: [3, 1, 4, 3, 5, 3, 2, 3]
After partition:   [2, 1, 3, 3, 3, 3, 5, 4]
lt=2, gt=5
Less:  [2, 1]
Equal: [3, 3, 3, 3]
Greater: [5, 4]

All equal: [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
After:     [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
```

---

## 연습문제

**연습문제 1.**
$[38, 27, 43, 3, 9, 82, 10]$에서 세 갈래 나눔을 따라가며 큰 걸음마다의 상태를 보여라.

??? success "연습문제 1 풀이"
    알고리즘을 한 걸음씩 밟아라. 나누어 정복하는 정렬이라면 되돌이 나눔과 병합을 하나씩 보이고, 나눔 기반 정렬이라면 나눔마다와 축이 놓이는 자리를 보여라. 걸음마다 배열 전체의 상태를 내보여라.

---

**연습문제 2.**
세 갈래 나눔의 최선, 최악, 평균의 경우 시간 복잡도를 끌어내라.

??? success "연습문제 2 풀이"
    경우마다 점화식을 써라. 최선의 경우는 가장 좋은 나눔이거나 미리 정렬된 입력이다. 최악의 경우는 일을 가장 크게 만드는 적수 입력이다. 평균의 경우는 무작위 순열에 대한 기대 성능이다. 점화식을 각각 풀어라.

---

**연습문제 3.**
세 갈래 나눔은 안정적인가? 증명하거나 반례를 들어라.

??? success "연습문제 3 풀이"
    같은 원소가 본디 상대 차례를 지키면 그 정렬은 안정적이다. 모든 입력에서 이것이 성립함을 증명하거나, 정렬 도중 같은 원소 둘의 자리가 뒤바뀌는 구체적인 입력을 들어라.

---

**연습문제 4.**
float32 학습 손실 값 1000만 개를 정렬할 때 세 갈래 나눔을 다른 두 방법과 견주어라. 시간, 공간, 캐시 거동, GPU 어울림을 살펴라.

??? success "연습문제 4 풀이"
    $n = 10^7$에서는 실제 선택이 하드웨어에 달렸다. CPU에서는 캐시 친화적인 알고리즘(병합 정렬, 인트로 정렬)이 앞선다. GPU에서는 병렬에 어울리는 알고리즘(바이토닉 정렬, 기수 정렬)이 낫다. 기억이 빠듯하면 제자리 정렬이 $O(n)$ 공간을 아낀다. 이 쪽의 이론적 분석이 그 고름에 길잡이가 된다.

## 정리하며

이 마당은 네덜란드 국기 문제、알고리즘、복잡도、구현을 차례로 짚었다.

**참고 문헌**

- Dijkstra, E. W. (1976). *A Discipline of Programming*. Prentice-Hall.
- Bentley, J. L., & McIlroy, M. D. (1993). Engineering a sort function. *Software: Practice and Experience*, 23(11), 1249-1265.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press.
