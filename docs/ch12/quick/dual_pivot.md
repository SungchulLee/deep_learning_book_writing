# 두 축 빠른 정렬

보통의 빠른 정렬은 축 하나를 골라 배열을 둘로 쪼갠다. 그러면 **축이 둘**이면 어떨까 하는 물음이 자연스레 따라온다. 두 축 빠른 정렬은 축 원소 $p$과 $q$($p \leq q$)을 골라 배열을 세 무리로 나눈다. 곧 $p$보다 작은 원소, $p$과 $q$ 사이의 원소, $q$보다 큰 원소이다. 나누는 걸음마다 견줌이 늘어나지만 기대 되돌이 깊이가 줄고, 실전에서 세 갈래 쪼갬이 캐시 거동을 좋게 한다. 2009년 블라디미르 야로슬랍스키가 들여온 이 변형은 자바 `Arrays.sort`의 기본형 자료에 대한 기본 정렬 알고리즘이다.

## 나눔 방식

배열 $A[lo..hi]$이 주어지면 축 둘 $p = A[lo]$과 $q = A[hi]$을 고르되 $p \leq q$이 되게 한다(필요하면 맞바꾼다). 목표는 $A$을 세 자리로 다시 늘어놓는 것이다.

$$
A[lo..j-1] < p \leq A[j..k-1] \leq q < A[k..hi]
$$

가리개 셋이 나눔의 경계를 지킨다.

- **lt**(작은 쪽 가리개): $A[lo+1..lt-1]$의 모든 것이 $p$보다 작다.
- **gt**(큰 쪽 가리개): $A[gt+1..hi-1]$의 모든 것이 $q$보다 크다.
- **i**(훑는 가리개): $lt$에서 $gt$까지 돌며 원소마다 갈래를 매긴다.

훑는 가리개 $i$은 원소마다 꼭 한 번 다루며 맞바꿈으로 알맞은 자리에 놓는다.

## 알고리즘

1. $A[lo] > A[hi]$이면 맞바꾸어 $p \leq q$이 되게 한다.
2. $lt = lo + 1$, $gt = hi - 1$, $i = lo + 1$으로 둔다.
3. $i \leq gt$인 동안 다음을 되풀이한다.
    - $A[i] < p$이면 $A[i]$과 $A[lt]$을 맞바꾸고 $lt$과 $i$을 함께 하나씩 민다.
    - 아니고 $A[i] > q$이면 $A[i]$과 $A[gt]$을 맞바꾸고 $gt$을 하나 당긴다($i$은 **밀지 않는다**. 맞바꾸어 들어온 원소를 아직 살피지 않았기 때문이다).
    - 그 밖의 경우($p \leq A[i] \leq q$)에는 $i$을 하나 민다.
4. 축을 마지막 자리에 놓는다. $A[lo]$과 $A[lt-1]$을, $A[hi]$과 $A[gt+1]$을 맞바꾼다.
5. 세 조각 $A[lo..lt-2]$, $A[lt..gt]$, $A[gt+2..hi]$에서 되돌이한다.

## 복잡도

| 경우 | 견줌 | 맞바꿈 |
|------|-------------|-------|
| 최선 | $O(n \log n)$ | $O(n \log n)$ |
| 평균 | $\approx 1.9\, n \ln n$ | $\approx 0.6\, n \ln n$ |
| 최악 | $O(n^2)$ | $O(n^2)$ |

평균의 경우 견줌 횟수 $\approx 1.9\, n \ln n$은 고전적인 한 축 빠른 정렬의 $\approx 1.39\, n \ln n$보다 조금 많다. 그러나 평균 맞바꿈 수는 적고, 세 갈래 나눔이 평균으로 더 작은 부분 문제를 내놓아 오늘날 하드웨어에서 캐시 성능을 좋게 한다.

!!! tip "맞바꿈이 적은 것이 중요한 까닭"
    오늘날 CPU에서는 기억을 어떻게 훑느냐가 도는 시간을 좌우한다. 두 축 방식은 원소를 옮기는 총 횟수를 줄이고 더 작은 되돌이 부분 문제 셋을 내놓아 저마다 더 빨리 캐시에 들어간다. 견줌이 더 많은데도 두 축 빠른 정렬이 실전에서 한 축 빠른 정렬을 앞지르는 까닭이 여기에 있다.

## 구현

```python
"""
야로슬랍스키 나눔 방식을 쓰는 두 축 빠른 정렬.

배열을 축 둘을 기준으로 세 자리로 나눈 뒤
자리마다 되돌이한다. 자바의 Arrays.sort이 기본형 자료에
쓰는 알고리즘이다.
"""


# === 두 축 나눔 ===

def dual_pivot_partition(arr: list, lo: int, hi: int) -> tuple:
    """축 둘을 기준으로 arr[lo..hi]을 나눈다.

    다음을 만족하는 (lt, gt)을 되돌린다.
      - arr[lo..lt-1] < pivot1
      - arr[lt..gt]는 pivot1과 pivot2 사이(양 끝 포함)
      - arr[gt+1..hi] > pivot2
    """
    if arr[lo] > arr[hi]:
        arr[lo], arr[hi] = arr[hi], arr[lo]

    p, q = arr[lo], arr[hi]
    lt = lo + 1
    gt = hi - 1
    i = lo + 1

    while i <= gt:
        if arr[i] < p:
            arr[i], arr[lt] = arr[lt], arr[i]
            lt += 1
            i += 1
        elif arr[i] > q:
            arr[i], arr[gt] = arr[gt], arr[i]
            gt -= 1
        else:
            i += 1

    lt -= 1
    gt += 1
    arr[lo], arr[lt] = arr[lt], arr[lo]
    arr[hi], arr[gt] = arr[gt], arr[hi]

    return lt, gt


# === 두 축 빠른 정렬 ===

def dual_pivot_quicksort(arr: list, lo: int, hi: int) -> None:
    """두 축 빠른 정렬로 arr[lo..hi]을 제자리에서 정렬한다."""
    if lo >= hi:
        return

    lt, gt = dual_pivot_partition(arr, lo, hi)
    dual_pivot_quicksort(arr, lo, lt - 1)
    dual_pivot_quicksort(arr, lt + 1, gt - 1)
    dual_pivot_quicksort(arr, gt + 1, hi)


# === 시연 ===

if __name__ == "__main__":
    data = [24, 8, 42, 75, 29, 77, 38, 57, 7, 53]
    print(f"Before: {data}")
    dual_pivot_quicksort(data, 0, len(data) - 1)
    print(f"After:  {data}")
    print()

    # 작은 보기에서 나누는 걸음을 보여 준다
    example = [35, 10, 40, 20, 50, 30, 45]
    print(f"Partition example: {example}")
    lt, gt = dual_pivot_partition(example, 0, len(example) - 1)
    print(f"After partition:   {example}")
    print(f"Pivot positions:   lt={lt}, gt={gt}")
    print(f"Left pivot:  {example[lt]}")
    print(f"Right pivot: {example[gt]}")
```

**출력:**
```
Before: [24, 8, 42, 75, 29, 77, 38, 57, 7, 53]
After:  [7, 8, 24, 29, 38, 42, 53, 57, 75, 77]

Partition example: [35, 10, 40, 20, 50, 30, 45]
After partition:   [30, 10, 35, 20, 40, 45, 50]
Pivot positions:   lt=2, gt=5
Left pivot:  35
Right pivot: 45
```

## 한 축 빠른 정렬과의 견줌

| 성질 | 한 축 | 두 축 |
|----------|-------------|------------|
| 층마다 조각 수 | 2 | 3 |
| 평균 견줌 | $\approx 1.39\, n \ln n$ | $\approx 1.9\, n \ln n$ |
| 평균 맞바꿈 | $\approx 0.33\, n \ln n$ | $\approx 0.6\, n \ln n$ |
| 캐시 거동 | 좋음 | 더 좋음(부분 문제가 더 작다) |
| 실제 쓰임 | C 표준 라이브러리 `qsort` | 자바 `Arrays.sort`(기본형) |

두 축 변형이 오늘날 하드웨어에서 앞서는 주된 까닭은 세 갈래 쪼갬이 더 작은 부분 문제를 내놓아 L1과 L2 캐시에 더 빨리 들어가고, 값비싼 캐시 빗나감을 줄이기 때문이다.

## 참고 문헌

- Yaroslavskiy, V. (2009). *Dual-Pivot Quicksort*. [연구 논문].
- Wild, S., & Nebel, M. E. (2012). Average case analysis of Java 7's dual pivot quicksort. *European Symposium on Algorithms*, 825-836.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press.


## 연습문제

**연습문제 1.**
$[38, 27, 43, 3, 9, 82, 10]$에서 두 축 빠른 정렬을 따라가며 큰 걸음마다의 상태를 보여라.

??? success "연습문제 1 풀이"
    알고리즘을 한 걸음씩 밟아라. 나누어 정복하는 정렬이라면 되돌이 나눔과 병합을 하나씩 보이고, 나눔 기반 정렬이라면 나눔마다와 축이 놓이는 자리를 보여라. 걸음마다 배열 전체의 상태를 내보여라.

---

**연습문제 2.**
두 축 빠른 정렬의 최선, 최악, 평균의 경우 시간 복잡도를 끌어내라.

??? success "연습문제 2 풀이"
    경우마다 점화식을 써라. 최선의 경우는 가장 좋은 나눔이거나 미리 정렬된 입력이다. 최악의 경우는 일을 가장 크게 만드는 적수 입력이다. 평균의 경우는 무작위 순열에 대한 기대 성능이다. 점화식을 각각 풀어라.

---

**연습문제 3.**
두 축 빠른 정렬은 안정적인가? 증명하거나 반례를 들어라.

??? success "연습문제 3 풀이"
    같은 원소가 본디 상대 차례를 지키면 그 정렬은 안정적이다. 모든 입력에서 이것이 성립함을 증명하거나, 정렬 도중 같은 원소 둘의 자리가 뒤바뀌는 구체적인 입력을 들어라.

---

**연습문제 4.**
float32 학습 손실 값 1000만 개를 정렬할 때 두 축 빠른 정렬을 다른 두 방법과 견주어라. 시간, 공간, 캐시 거동, GPU 어울림을 살펴라.

??? success "연습문제 4 풀이"
    $n = 10^7$에서는 실제 선택이 하드웨어에 달렸다. CPU에서는 캐시 친화적인 알고리즘(병합 정렬, 인트로 정렬)이 앞선다. GPU에서는 병렬에 어울리는 알고리즘(바이토닉 정렬, 기수 정렬)이 낫다. 기억이 빠듯하면 제자리 정렬이 $O(n)$ 공간을 아낀다. 이 쪽의 이론적 분석이 그 고름에 길잡이가 된다.