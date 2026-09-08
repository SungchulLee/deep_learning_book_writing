# 중앙값의 중앙값

빠른 고르기의 최악의 경우 $O(n^2)$은 축을 잘못 고르는 데서 온다. 축이 늘 극단적인 순위에 떨어지면 원소가 거의 다 다음 되돌이 부름까지 살아남는다. **중앙값의 중앙값**은 고른 축의 순위가 $3n/10$과 $7n/10$ 사이임을 보장하는 축 고르기 기법이다. 그러면 되돌이 부름마다 적어도 원소의 $30\%$이 걸러져 최악의 경우 $O(n)$ 고르기가 된다. 이 기법은 1973년 Blum, Floyd, Pratt, Rivest, Tarjan이 들여왔다.

---

## 1. 기법

중앙값의 중앙값 절차는 원소 $n$개의 배열 $A$에서 축을 고른다.

1. **묶기**: $A$을 원소 5개씩 $\lceil n/5 \rceil$개 무리로 나눈다(마지막 무리는 더 작을 수 있다).
2. **무리마다 정렬**: 끼워넣기 정렬을 쓴다(원소 5개에 많아야 6번 견준다).
3. **중앙값 뽑기**: 정렬한 무리마다 중앙값(가운데 원소)을 가져온다. 그러면 중앙값이 $\lceil n/5 \rceil$개 나온다.
4. **되돌이**: 고르기 알고리즘을 되돌이로 써서 이 $\lceil n/5 \rceil$개 중앙값의 중앙값을 찾는다. 이것이 축이다.

---

## 2. 축의 질 보장

중앙값의 중앙값으로 고른 축 $m$은 좋은 가름자임이 보장된다. $\lceil n/5 \rceil$개 무리 중앙값 가운데 적어도 절반이 $m$ 이하이다. 그 무리 중앙값마다 (다섯의 중앙값이므로) 제 무리에 자기보다 작은 원소가 적어도 2개 있다. 그러므로 다음과 같다.

$$
\text{elements} \leq m \geq 3 \cdot \left\lceil \frac{1}{2} \cdot \lceil n/5 \rceil \right\rceil \geq \frac{3n}{10} - 6
$$

대칭으로 적어도 원소 $3n/10 - 6$개가 $m$ 이상이다. $m$을 기준으로 나누면 더 큰 쪽에 많아야 원소 $7n/10 + 6$개가 든다.

---

## 3. 점화식

전체 일의 양은 다음을 만족한다.

$$
T(n) \leq T(\lceil n/5 \rceil) + T(7n/10 + 6) + O(n)
$$

- $T(\lceil n/5 \rceil)$: 무리 중앙값의 중앙값을 되돌이로 찾기.
- $T(7n/10 + 6)$: 더 큰 조각에서의 되돌이 고르기.
- $O(n)$: 묶기, 무리 정렬하기, 나누기.

$n/5 + 7n/10 = 9n/10 < n$이므로, 넉넉히 큰 상수 $c$에 대해 $T(n) \leq cn$임을 대입으로 증명할 수 있고 $T(n) = O(n)$이 된다.

!!! note "대입 증명 얼개"
    더 작은 모든 $n$에 대해 $T(n) \leq cn$이라고 놓자. 그러면 $T(n) \leq c \cdot n/5 + c(7n/10 + 6) + an = cn(9/10) + 6c + an = cn - cn/10 + 6c + an$이다. $c \geq 10a$이고 $n \geq 60$이면 $T(n) \leq cn$이 된다.

---

## 4. 왜 다섯씩 묶는가

무리 크기 5는 점화식이 $O(n)$으로 풀리게 하는 가장 작은 홀수이다.

| 무리 크기 $g$ | 걸러지는 비율 | 되돌이 비율 | 합 |
|---|---|---|---|
| 3 | $\geq n/4$ | $n/3 + 3n/4$ | $13/12 > 1$ |
| 5 | $\geq 3n/10$ | $n/5 + 7n/10$ | $9/10 < 1$ |
| 7 | $\geq 2n/7$ | $n/7 + 5n/7$ | $6/7 < 1$ |

셋씩 묶으면 되돌이 부분 문제 둘의 합이 $n$을 넘어 무너진다. 일곱 이상으로 묶으면 굴러가기는 하지만 점근 한계는 나아지지 않고 상수 인자만 커진다.

---

## 5. 구현

```python
"""
최악의 경우 선형 고르기를 위한 중앙값의 중앙값 축 고르기.

배열을 5개씩 무리로 나누고 무리마다 중앙값을 찾은 뒤,
그 중앙값들의 중앙값을 되돌이로 고른다. 그러면 적어도 원소의 30%을
걸러 내는 축이 보장된다.
"""

# === 작은 무리 정렬 ===

def sort5(arr: list, lo: int, hi: int) -> None:
    """끼워넣기 정렬로 arr[lo..hi]을 정렬한다(5개 이하 무리용)."""
    for i in range(lo + 1, hi + 1):
        key = arr[i]
        j = i - 1
        while j >= lo and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key

# === 중앙값의 중앙값 ===

def median_of_medians(arr: list, lo: int, hi: int) -> int:
    """중앙값의 중앙값으로 축을 고른다. 축의 값을 되돌린다."""
    n = hi - lo + 1
    if n <= 5:
        sort5(arr, lo, hi)
        return arr[lo + n // 2]

    # 5개짜리 무리마다 중앙값 찾기
    num_groups = (n + 4) // 5
    for i in range(num_groups):
        group_lo = lo + i * 5
        group_hi = min(group_lo + 4, hi)
        sort5(arr, group_lo, group_hi)
        # 무리 중앙값을 배열 앞으로 옮기기
        median_idx = group_lo + (group_hi - group_lo) // 2
        arr[lo + i], arr[median_idx] = arr[median_idx], arr[lo + i]

    # 무리 중앙값의 중앙값을 되돌이로 찾기
    return median_of_medians(arr, lo, lo + num_groups - 1)

def select_mom(arr: list, k: int):
    """최악의 경우 O(n)에 k번째로 작은 것(0부터 세는)을 찾는다."""
    data = arr.copy()
    return _select(data, 0, len(data) - 1, k)

def _select(arr: list, lo: int, hi: int, k: int):
    """중앙값의 중앙값 축을 쓴 되돌이 고르기."""
    if lo == hi:
        return arr[lo]

    pivot = median_of_medians(arr, lo, hi)

    # 세 갈래 나눔
    lt, gt = lo, hi
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

    if k < lt:
        return _select(arr, lo, lt - 1, k)
    elif k > gt:
        return _select(arr, gt + 1, hi, k)
    else:
        return arr[k]

# === 시연 ===

if __name__ == "__main__":
    data = [31, 12, 5, 23, 7, 19, 42, 3, 15, 8, 27, 35, 1, 10, 20]
    print(f"Array:  {data}")
    print(f"Sorted: {sorted(data)}")
    print()

    for k in [0, 4, 7, 11, 14]:
        result = select_mom(data, k)
        expected = sorted(data)[k]
        status = "OK" if result == expected else "MISMATCH"
        print(f"k={k:2d}: got {result:3d}, expected {expected:3d} [{status}]")
```

**출력:**
```
Array:  [31, 12, 5, 23, 7, 19, 42, 3, 15, 8, 27, 35, 1, 10, 20]
Sorted: [1, 3, 5, 7, 8, 10, 12, 15, 19, 20, 23, 27, 31, 35, 42]

k= 0: got   1, expected   1 [OK]
k= 4: got   8, expected   8 [OK]
k= 7: got  15, expected  15 [OK]
k=11: got  27, expected  27 [OK]
k=14: got  42, expected  42 [OK]
```

---

## 연습문제

**연습문제 1.**
중앙값의 중앙값의 핵심 생각과 그 시간·공간 복잡도를 밝혀라.

??? success "연습문제 1 풀이"
    이 알고리즘은 견줌 너머의 성질(정수 열쇠, 바깥 저장 장치, 병렬 하드웨어)을 살려 써서 비교 기반 정렬이 따라올 수 없는 성능을 이룬다. 구체적인 복잡도 한계는 이 쪽에서 뜯어본다.

---

**연습문제 2.**
작은 입력에서 중앙값의 중앙값을 따라가라. 훑기나 단계마다 보여라.

??? success "연습문제 2 풀이"
    원소 6~8개에 알고리즘을 적용하며 훑을 때마다의 상태를 보여라. 이 따라가기가 알고리즘의 얼개를 드러내고 옳음을 눈에 보이게 한다.

---

**연습문제 3.**
어떤 조건에서 비교 기반 정렬보다 중앙값의 중앙값이 나은가?

??? success "연습문제 3 풀이"
    다음일 때 낫다. 입력의 정수 범위가 묶여 있을 때(세기 정렬과 기수 정렬), 데이터가 램을 넘칠 때(바깥 정렬), 병렬 하드웨어를 쓸 수 있을 때(병렬 정렬). 이런 조건에서는 알고리즘이 비교 기반의 아래 한계 $\Omega(n\log n)$을 비껴갈 수 있다.

---

**연습문제 4.**
중앙값의 중앙값이 실전에서 이득을 주는 깊은 학습 응용을 서술하라.

??? success "연습문제 4 풀이"
    응용: 어휘를 찾기 위한 토큰 번호 정렬($O(n + V)$의 세기 정렬), GPU 기억을 넘치는 데이터셋의 바깥 정렬, GPU에서 배치 연산을 위한 병렬 정렬. 특정 조건(정수 열쇠, 큰 데이터, 병렬성)이 갖추어질 때 이득이 가장 크다.

## 정리하며

이 마당은 기법、축의 질 보장、점화식、왜 다섯씩 묶는가을 차례로 짚었다.

**참고 문헌**

- Blum, M., Floyd, R. W., Pratt, V. R., Rivest, R. L., & Tarjan, R. E. (1973). Time bounds for selection. *Journal of Computer and System Sciences*, 7(4), 448-461.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), 9장. MIT Press.
