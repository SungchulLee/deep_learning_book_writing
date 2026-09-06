# k번째로 작은 원소

정렬되지 않은 배열에서 $k$번째로 작은 원소를 찾는 일은 컴퓨터 과학의 근본 문제 가운데 하나이다. 소박하게 다가가면 배열을 먼저 정렬하고 자리 $k$을 읽는데 $O(n \log n)$이 든다. 그러나 정렬은 필요보다 훨씬 많은 일을 한다. 우리는 정렬된 차례 전체가 아니라 원소 하나만 있으면 된다. **고르기 문제**는 이렇게 묻는다. $k$번째로 작은 것을 $O(n)$ 시간에 찾을 수 있는가? 답은 그렇다이며, 걸음마다 축의 한쪽으로만 찾기를 좁히는 나눔 기반 알고리즘을 쓴다.

## 문제 서술

원소 $n$개의 정렬되지 않은 배열 $A$과 $1 \leq k \leq n$인 정수 $k$이 주어졌을 때, 배열을 감소하지 않는 차례로 정렬했다면 자리 $k$에 놓였을 원소를 찾아라.

잘 알려진 이름이 붙은 특별한 경우는 다음과 같다.

- $k = 1$: **가장 작은 값**(죽 훑으면 시시하게 $O(n)$이다)
- $k = n$: **가장 큰 값**(역시 $O(n)$이다)
- $k = \lfloor (n+1)/2 \rfloor$: **중앙값**(가장 어려운 특별한 경우)

## 다가가는 길

### 정렬 기반

배열을 정렬하고 $A[k-1]$을 되돌린다. 시간: $O(n \log n)$. 단순하지만 물음 하나에는 아깝다.

### 힙 기반

$O(n)$에 최소 힙을 쌓고 가장 작은 것을 $k$번 꺼낸다. 시간: $O(n + k \log n)$. $k$이 작을 때(이를테면 다섯째로 작은 것) 효율적이지만 $k = n/2$이면 $O(n \log n)$으로 무너진다.

### 나눔 기반(빠른 고르기)

축을 기준으로 배열을 나눈다. 축이 자리 $k$에 떨어지면 그것을 되돌린다. 아니면 자리 $k$을 담은 쪽에서 되돌이한다. 기대 시간: $O(n)$, 최악의 경우 $O(n^2)$.

### 정해진 선형 시간(중앙값의 중앙값)

최악의 경우 $O(n)$을 보장하도록 중앙값의 중앙값 알고리즘으로 축을 고른다. 빠른 고르기와 중앙값의 중앙값 쪽에서 자세히 다룬다.

## 아래 한계

$k$번째로 작은 원소를 찾는 비교 기반 알고리즘은 적어도 다음이 필요하다.

$$
n - 1 \text{ comparisons for } k = 1 \text{ or } k = n
$$

일반의 경우 정보 이론의 아래 한계는 $\Omega(n)$이다. 원소마다 적어도 한 번은 살펴야 하기 때문이다(보지 않은 원소가 답일 수도 있다). 나눔 기반 알고리즘이 이 한계를 이룬다.

## 구현

```python
"""
나눔 기반으로 k번째로 작은 원소 고르기.

소박한 길(정렬 기반)과 빠른 고르기 길을 모두 보인다.
빠른 고르기 판은 나눔의 한쪽에서만 되돌이하여
기대 시간 O(n)을 이룬다.
"""

import random


# === 소박한 고르기(정렬 기반) ===

def kth_smallest_sort(arr: list, k: int):
    """정렬해서 k번째로 작은 것을 찾는다. O(n log n)."""
    return sorted(arr)[k - 1]


# === 빠른 고르기 ===

def kth_smallest(arr: list, k: int):
    """무작위 빠른 고르기로 k번째로 작은 것을 찾는다.

    원본을 건드리지 않으려고 베낀 것에서 굴린다.
    기대 시간 O(n).
    """
    if k < 1 or k > len(arr):
        raise ValueError(f"k={k} out of range for array of size {len(arr)}")

    data = arr.copy()
    return _quickselect(data, 0, len(data) - 1, k - 1)


def _quickselect(arr: list, lo: int, hi: int, k: int):
    """arr[lo..hi]을 정렬했다면 첨자 k에 있을 원소를 되돌린다."""
    if lo == hi:
        return arr[lo]

    pivot_idx = _partition(arr, lo, hi)

    if k == pivot_idx:
        return arr[k]
    elif k < pivot_idx:
        return _quickselect(arr, lo, pivot_idx - 1, k)
    else:
        return _quickselect(arr, pivot_idx + 1, hi, k)


def _partition(arr: list, lo: int, hi: int) -> int:
    """무작위 축을 쓴 로무토 나눔."""
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


# === 시연 ===

if __name__ == "__main__":
    random.seed(42)

    data = [7, 10, 4, 3, 20, 15, 8, 1, 12, 5]
    print(f"Array: {data}")
    print(f"Sorted: {sorted(data)}")
    print()

    for k in [1, 3, 5, 7, 10]:
        result = kth_smallest(data, k)
        print(f"k={k:2d}: {result}")

    print()
    print(f"Minimum (k=1):  {kth_smallest(data, 1)}")
    print(f"Maximum (k=10): {kth_smallest(data, 10)}")
    print(f"Median (k=5):   {kth_smallest(data, 5)}")
```

**출력:**
```
Array: [7, 10, 4, 3, 20, 15, 8, 1, 12, 5]
Sorted: [1, 3, 4, 5, 7, 8, 10, 12, 15, 20]

k= 1: 1
k= 3: 4
k= 5: 7
k= 7: 10
k=10: 20

Minimum (k=1):  1
Maximum (k=10): 20
Median (k=5):   7
```

## 복잡도 비교

| 방법 | 시간(기대) | 시간(최악) | 공간 |
|--------|----------------|--------------|-------|
| 정렬 후 첨자 | $O(n \log n)$ | $O(n \log n)$ | $O(n)$ |
| 최소 힙과 꺼내기 | $O(n + k \log n)$ | $O(n + k \log n)$ | $O(n)$ |
| 빠른 고르기 | $O(n)$ | $O(n^2)$ | 기대 $O(\log n)$ |
| 중앙값의 중앙값 | $O(n)$ | $O(n)$ | $O(\log n)$ |

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), 9장. MIT Press.
- Blum, M., Floyd, R. W., Pratt, V. R., Rivest, R. L., & Tarjan, R. E. (1973). Time bounds for selection. *Journal of Computer and System Sciences*, 7(4), 448-461.


## 연습문제

**연습문제 1.**
k번째로 작은 원소의 핵심 생각과 그 시간·공간 복잡도를 밝혀라.

??? success "연습문제 1 풀이"
    이 알고리즘은 견줌 너머의 성질(정수 열쇠, 바깥 저장 장치, 병렬 하드웨어)을 살려 써서 비교 기반 정렬이 따라올 수 없는 성능을 이룬다. 구체적인 복잡도 한계는 이 쪽에서 뜯어본다.

---

**연습문제 2.**
작은 입력에서 k번째로 작은 원소를 따라가라. 훑기나 단계마다 보여라.

??? success "연습문제 2 풀이"
    원소 6~8개에 알고리즘을 적용하며 훑을 때마다의 상태를 보여라. 이 따라가기가 알고리즘의 얼개를 드러내고 옳음을 눈에 보이게 한다.

---

**연습문제 3.**
어떤 조건에서 비교 기반 정렬보다 k번째로 작은 원소가 나은가?

??? success "연습문제 3 풀이"
    다음일 때 낫다. 입력의 정수 범위가 묶여 있을 때(세기 정렬과 기수 정렬), 데이터가 램을 넘칠 때(바깥 정렬), 병렬 하드웨어를 쓸 수 있을 때(병렬 정렬). 이런 조건에서는 알고리즘이 비교 기반의 아래 한계 $\Omega(n\log n)$을 비껴갈 수 있다.

---

**연습문제 4.**
k번째로 작은 원소가 실전에서 이득을 주는 깊은 학습 응용을 서술하라.

??? success "연습문제 4 풀이"
    응용: 어휘를 찾기 위한 토큰 번호 정렬($O(n + V)$의 세기 정렬), GPU 기억을 넘치는 데이터셋의 바깥 정렬, GPU에서 배치 연산을 위한 병렬 정렬. 특정 조건(정수 열쇠, 큰 데이터, 병렬성)이 갖추어질 때 이득이 가장 크다.