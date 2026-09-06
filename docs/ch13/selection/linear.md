# 선형 시간 고르기

빠른 고르기는 $k$번째로 작은 원소를 기대 시간 $O(n)$에 찾지만, 축이 한결같이 치우친 나눔을 낼 때는 최악의 경우가 $O(n^2)$이다. **선형 시간 고르기**(Blum, Floyd, Pratt, Rivest, Tarjan의 이름을 딴 1973년 BFPRT 알고리즘)는 걸음마다 원소의 일정 비율을 반드시 걸러 내는 축을 골라 최악의 경우 $O(n)$ 시간을 보장한다. 핵심 기법은 **중앙값의 중앙값**이다. 곧 배열을 다섯씩 무리로 나누고 무리마다 중앙값을 찾은 뒤, 그 중앙값들의 중앙값을 되돌이로 골라 축으로 삼는다.

## 알고리즘 훑어보기

SELECT 알고리즘은 $A[1..n]$에서 $k$번째로 작은 원소를 찾는다.

1. 원소 $n$개를 5개씩 $\lceil n/5 \rceil$개 무리로 **나눈다**(마지막 무리는 더 적을 수 있다).
2. 무리를 정렬해 무리마다 **중앙값을 찾는다**(무리마다 많아야 6번 견준다).
3. $\lceil n/5 \rceil$개 무리 중앙값의 중앙값 $m$을 **되돌이로 고른다**.
4. $m$을 기준으로 배열을 **나눈다**. $m$이 자리 $q$에 놓인다고 하자.
5. $k = q$이면 $m$을 되돌린다. $k < q$이면 왼쪽에서, $k > q$이면 오른쪽에서 되돌이한다.

## 왜 다섯씩 묶는가

5는 점화식이 $O(n)$으로 풀리게 하는 가장 작은 홀수 무리 크기이다. 중앙값의 중앙값 $m$은 대략 원소 $3n/10$개보다 크고 대략 $3n/10$개보다 작다. 곧 최악의 경우 되돌이 부름이 많아야 원소 $7n/10$개를 다룬다는 뜻이다.

점화식은 다음과 같다.

$$
T(n) \leq T(\lceil n/5 \rceil) + T(7n/10) + O(n)
$$

$n/5 + 7n/10 = 9n/10 < n$이므로 이는 $T(n) = O(n)$으로 풀린다.

!!! tip "왜 셋씩은 안 되는가?"
    셋씩 묶으면 중앙값의 중앙값이 걸음마다 원소 $n/4$개만 걸러 내도록 보장하여 점화식이 $T(n) \leq T(n/3) + T(3n/4) + O(n)$이 된다. $1/3 + 3/4 = 13/12 > 1$이므로 이는 $O(n)$으로 풀리지 않는다.

## 보장 분석

$\lceil n/5 \rceil$개 무리 중앙값 가운데 적어도 절반은 $m$ 이하이고 적어도 절반은 $m$ 이상이다. 그런 무리 중앙값마다 (다섯의 중앙값이므로) 제 무리에 자기보다 작은 원소가 2개 있다. 그러므로 $m$ 이하임이 보장되는 원소의 개수는 적어도 다음과 같다.

$$
3 \cdot \left\lfloor \frac{1}{2} \cdot \lceil n/5 \rceil \right\rfloor \geq \frac{3n}{10} - 6
$$

대칭으로 적어도 원소 $3n/10 - 6$개가 $m$ 이상이다. 더 큰 조각에 대한 되돌이 부름에는 많아야 원소 $7n/10 + 6$개가 든다.

## 구현

```python
"""
최악의 경우 선형 시간 고르기(BFPRT / 중앙값의 중앙값).

중앙값의 중앙값 기법으로 축을 골라 걸음마다 적어도 원소의 30%이
걸러지도록 하여 k번째로 작은 원소를 찾는 데 O(n) 시간을
보장한다.
"""


# === 작은 무리를 위한 끼워넣기 정렬 ===

def insertion_sort(arr: list, lo: int, hi: int) -> None:
    """끼워넣기 정렬로 arr[lo..hi]을 제자리에서 정렬한다."""
    for i in range(lo + 1, hi + 1):
        key = arr[i]
        j = i - 1
        while j >= lo and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key


# === 중앙값의 중앙값 ===

def median_of_medians(arr: list, lo: int, hi: int) -> int:
    """중앙값의 중앙값 기법으로 좋은 축을 찾는다.

    축의 값(첨자가 아니다)을 되돌린다.
    """
    n = hi - lo + 1
    if n <= 5:
        insertion_sort(arr, lo, hi)
        return arr[lo + n // 2]

    # 5개씩 무리로 나누고 무리마다 중앙값 찾기
    medians = []
    for i in range(lo, hi + 1, 5):
        group_end = min(i + 4, hi)
        insertion_sort(arr, i, group_end)
        medians.append(arr[i + (group_end - i) // 2])

    # 무리 중앙값의 중앙값을 되돌이로 찾기
    return select(medians, len(medians) // 2)


# === 선형 시간 고르기 ===

def select(arr: list, k: int):
    """최악의 경우 O(n)에 k번째로 작은 원소(0부터 세는)를 찾는다."""
    if len(arr) <= 5:
        return sorted(arr)[k]

    pivot = median_of_medians(arr, 0, len(arr) - 1)

    # 축을 기준으로 세 갈래 나눔
    less = [x for x in arr if x < pivot]
    equal = [x for x in arr if x == pivot]
    greater = [x for x in arr if x > pivot]

    if k < len(less):
        return select(less, k)
    elif k < len(less) + len(equal):
        return pivot
    else:
        return select(greater, k - len(less) - len(equal))


# === 시연 ===

if __name__ == "__main__":
    data = [12, 3, 5, 7, 4, 19, 26, 1, 8, 15, 20, 11, 9, 2, 6]
    print(f"Array:  {data}")
    print(f"Sorted: {sorted(data)}")
    print()

    for k in range(len(data)):
        result = select(data.copy(), k)
        print(f"k={k:2d} (rank {k+1:2d}): {result}")
```

**출력:**
```
Array:  [12, 3, 5, 7, 4, 19, 26, 1, 8, 15, 20, 11, 9, 2, 6]
Sorted: [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 15, 19, 20, 26]

k= 0 (rank  1): 1
k= 1 (rank  2): 2
k= 2 (rank  3): 3
k= 3 (rank  4): 4
k= 4 (rank  5): 5
k= 5 (rank  6): 6
k= 6 (rank  7): 7
k= 7 (rank  8): 8
k= 8 (rank  9): 9
k= 9 (rank 10): 11
k=10 (rank 11): 12
k=11 (rank 12): 15
k=12 (rank 13): 19
k=13 (rank 14): 20
k=14 (rank 15): 26
```

## 복잡도

| 성질 | 값 |
|----------|-------|
| 시간(최악의 경우) | $O(n)$ |
| 시간(평균) | $O(n)$ |
| 공간 | $O(\log n)$(되돌이 더미) |
| 견줌 | $\leq 5.43\, n + o(n)$ |

!!! warning "실전 성능"
    최악의 경우 $O(n)$이지만 중앙값의 중앙값은 상수 인자가 크다(빠른 고르기에 견주어 대략 5배). 실전에서는 무작위 빠른 고르기가 평균으로 더 빠르며, 최악의 경우 보장이 꼭 필요하지 않다면 그쪽이 낫다. 실제 구현 가운데 많은 것이 빠른 고르기를 쓰다가 되돌이 깊이가 문턱값을 넘을 때만 중앙값의 중앙값으로 물러선다.

## 참고 문헌

- Blum, M., Floyd, R. W., Pratt, V. R., Rivest, R. L., & Tarjan, R. E. (1973). Time bounds for selection. *Journal of Computer and System Sciences*, 7(4), 448-461.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), 9장. MIT Press.


## 연습문제

**연습문제 1.**
선형 시간 고르기의 핵심 생각과 그 시간·공간 복잡도를 밝혀라.

??? success "연습문제 1 풀이"
    이 알고리즘은 견줌 너머의 성질(정수 열쇠, 바깥 저장 장치, 병렬 하드웨어)을 살려 써서 비교 기반 정렬이 따라올 수 없는 성능을 이룬다. 구체적인 복잡도 한계는 이 쪽에서 뜯어본다.

---

**연습문제 2.**
작은 입력에서 선형 시간 고르기를 따라가라. 훑기나 단계마다 보여라.

??? success "연습문제 2 풀이"
    원소 6~8개에 알고리즘을 적용하며 훑을 때마다의 상태를 보여라. 이 따라가기가 알고리즘의 얼개를 드러내고 옳음을 눈에 보이게 한다.

---

**연습문제 3.**
어떤 조건에서 비교 기반 정렬보다 선형 시간 고르기가 나은가?

??? success "연습문제 3 풀이"
    다음일 때 낫다. 입력의 정수 범위가 묶여 있을 때(세기 정렬과 기수 정렬), 데이터가 램을 넘칠 때(바깥 정렬), 병렬 하드웨어를 쓸 수 있을 때(병렬 정렬). 이런 조건에서는 알고리즘이 비교 기반의 아래 한계 $\Omega(n\log n)$을 비껴갈 수 있다.

---

**연습문제 4.**
선형 시간 고르기가 실전에서 이득을 주는 깊은 학습 응용을 서술하라.

??? success "연습문제 4 풀이"
    응용: 어휘를 찾기 위한 토큰 번호 정렬($O(n + V)$의 세기 정렬), GPU 기억을 넘치는 데이터셋의 바깥 정렬, GPU에서 배치 연산을 위한 병렬 정렬. 특정 조건(정수 열쇠, 큰 데이터, 병렬성)이 갖추어질 때 이득이 가장 크다.