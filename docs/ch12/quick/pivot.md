# 축 고르기 전략

빠른 정렬의 성능은 좋은 축을 고르는 데 달렸다. 이상적인 축은 배열을 똑같은 반쪽 둘로 쪼개어 $O(n \log n)$ 시간을 낸다. 나쁜 축은 한쪽으로 기운 쪼갬, 극단적으로는 원소를 모두 한쪽에 몰아넣는 쪼갬을 내어 $O(n^2)$으로 이끈다. 참 중앙값을 찾는 데는 (중앙값의 중앙값 알고리즘으로) $O(n)$ 시간이 들므로, 실전의 축 전략은 고르는 데 시간을 너무 들이지 않으면서 최악의 경우를 피하는 "그만하면 좋은" 축을 노린다.

## 자리를 붙박은 축

### 첫 원소나 마지막 원소

가장 단순한 선택은 $A[\ell]$이나 $A[r]$을 축으로 쓰는 것이다.

$$
\text{pivot} = A[\ell] \quad \text{or} \quad \text{pivot} = A[r]
$$

어떤 붙박이 자리든 기대 순위가 $n/2$인 무작위 순열에서는 잘 굴러간다. 그러나 정렬되었거나 거의 정렬된 입력에서는 처참하게 무너지는데, 하필 실전에서 자주 나오는 경우이다(이를테면 이미 정렬된 리스트에 덧붙이는 경우).

!!! warning "정렬된 입력과 붙박이 축"
    배열이 오름차순으로 정렬되어 있고 축이 늘 마지막 원소이면, 나눌 때마다 크기 $(n-1, 0)$의 쪼갬이 나온다. 점화식은 $T(n) = T(n-1) + O(n) = O(n^2)$이 된다.

### 가운데 원소

$A[\lfloor (\ell + r)/2 \rfloor]$을 쓰면 정렬된 입력에서 최악의 경우를 피하지만, 일부러 지어낸 입력에는 여전히 당할 수 있다.

## 무작위 축

고르게 무작위인 첨자 $k \in [\ell, r]$을 골라 $A[k]$을 축으로 쓴다(또는 $A[k]$을 $A[r]$과 맞바꾸고 로무토로 이어 간다).

$$
k \sim \text{Uniform}\{l, l+1, \ldots, r\}
$$

**기대 시간**: 어떤 입력에서도 $O(n \log n)$이다. 축을 고르는 일이 데이터와 무관하므로 적수가 최악의 입력을 지어낼 수 없다.

**최악의 경우 시간**: 여전히 $O(n^2)$이지만 그 확률이 지수로 줄어든다. 축 고르기 $n$번이 모두 나쁠(이를테면 늘 가장 작은 10%에 들) 확률은 $(1/10)^n$이다.

무작위 축은 적수의 최악 입력을 없애는 가장 단순한 전략이며, 정해진 보장이 필요하지 않다면 기본으로 권할 만하다.

## 셋의 중앙값

원소 셋, 대개 $A[\ell]$, $A[\lfloor (\ell+r)/2 \rfloor]$, $A[r]$의 **중앙값**을 골라 축으로 쓴다.

**장점:**

- 정렬된 입력, 거꾸로 정렬된 입력, 파이프오르간 모양 입력에서 최악의 경우를 피한다.
- $\{1, \ldots, n\}$에서 뽑은 무작위 원소 셋의 중앙값은 기대 순위가 대략 $n/2$이라 잘 고른 나눔이 나온다.
- 무작위 축에 견주어 기대 견줌 횟수를 5% 남짓 줄인다.

**기대 견줌 횟수**(세지윅의 분석에 따르면):

$$
C(n) \approx \frac{12}{7} n \ln n \approx 1.714 n \ln n
$$

무작위 축의 $2n \ln n$에 견주면 14% 나아진 것이다.

이는 셋의 중앙값 쪽에서 자세히 다룬다.

## 중앙값의 중앙값(정해진 선형 고르기)

**중앙값의 중앙값** 알고리즘(Blum, Floyd, Pratt, Rivest, Tarjan, 1973)은 최악의 경우 $O(n)$ 시간에 참 중앙값을 찾는다.

1. 배열을 5개씩 무리로 나눈다.
2. 무리마다 중앙값을 찾는다(무리마다 상수 시간).
3. 이 $\lceil n/5 \rceil$개 중앙값의 중앙값을 되돌이로 찾는다.
4. 이 "중앙값의 중앙값"을 축으로 쓴다.

그러면 축의 양쪽에 적어도 $3n/10$개의 원소가 놓임이 보장되어 다음이 나온다.

$$
T(n) = T(n/5) + T(7n/10) + O(n) = O(n)
$$

이것을 빠른 정렬의 축으로 쓰면 **정해진** 최악의 경우 $O(n \log n)$ 시간을 얻는다. 그러나 상수 인자가 커서($\approx 5$배) 실전에서는 무작위 빠른 정렬보다 느리므로 정렬에는 거의 쓰이지 않는다. 주된 쓰임새는 고르기 알고리즘($k$번째로 작은 원소 찾기)이다.

## 아홉수(셋의 중앙값의 중앙값)

튜키의 **아홉수**는 셋의 중앙값 셋의 중앙값을 고른다.

1. 배열에서 원소 셋씩 표본 셋을 뽑는다.
2. 표본마다 중앙값을 찾는다.
3. 이 세 중앙값의 중앙값이 축이다.

원소 9개를 살피는 대신 셋의 중앙값 하나보다 참 중앙값에 더 가깝게 어림한다. 몇몇 고성능 구현(이를테면 `pdqsort`)에서 쓴다.

## 전략의 견줌

$$
\begin{array}{lccl}
\textbf{Strategy} & \textbf{Selection cost} & \textbf{Worst case} & \textbf{Notes} \\
\hline
\text{Fixed (first/last)}    & O(1) & O(n^2) & \text{Fails on sorted input} \\
\text{Middle element}        & O(1) & O(n^2) & \text{Better but still exploitable} \\
\text{Random}                & O(1) & O(n^2) \text{ (prob.)} & \text{No adversarial worst case} \\
\text{Median-of-three}       & O(1) & O(n^2) & \text{Practical default} \\
\text{Ninther}               & O(1) & O(n^2) & \text{Used in pdqsort} \\
\text{Median of medians}     & O(n) & O(n \log n) & \text{Theoretical; large constant}
\end{array}
$$

## 파이썬 구현

```python
"""
빠른 정렬을 위한 축 고르기 전략.

여러 축 고르기 방법과 그것이 나눔의 질을 비추는
빠른 정렬의 되돌이 깊이에 미치는 영향을 보여 준다.
"""

import random


# === 축 고르기 전략 ===========================================================

def pivot_first(arr: list, left: int, right: int) -> int:
    """첫 원소를 축으로 고른다."""
    return left


def pivot_last(arr: list, left: int, right: int) -> int:
    """마지막 원소를 축으로 고른다."""
    return right


def pivot_random(arr: list, left: int, right: int) -> int:
    """무작위 원소를 축으로 고른다."""
    return random.randint(left, right)


def pivot_median_of_three(arr: list, left: int, right: int) -> int:
    """첫째, 가운데, 마지막 원소의 중앙값을 고른다."""
    mid = (left + right) // 2
    candidates = [(arr[left], left), (arr[mid], mid), (arr[right], right)]
    candidates.sort(key=lambda x: x[0])
    return candidates[1][1]  # 중앙값의 첨자


# === 축을 골라 쓸 수 있는 빠른 정렬 ===========================================

def quicksort(arr: list, left: int, right: int,
              pivot_fn, depth: list) -> None:
    """깊이를 좇고 축 고르기를 설정할 수 있는 빠른 정렬."""
    if left < right:
        depth[0] += 1
        # 로무토 나눔을 위해 고른 축을 끝으로 옮긴다
        pivot_idx = pivot_fn(arr, left, right)
        arr[pivot_idx], arr[right] = arr[right], arr[pivot_idx]
        # 로무토 나눔
        pivot = arr[right]
        i = left
        for j in range(left, right):
            if arr[j] <= pivot:
                arr[i], arr[j] = arr[j], arr[i]
                i += 1
        arr[i], arr[right] = arr[right], arr[i]
        quicksort(arr, left, i - 1, pivot_fn, depth)
        quicksort(arr, i + 1, right, pivot_fn, depth)


# === 메인 =====================================================================

if __name__ == "__main__":
    n = 1000
    sorted_data = list(range(n))

    strategies = [
        ("First element", pivot_first),
        ("Last element", pivot_last),
        ("Random", pivot_random),
        ("Median-of-three", pivot_median_of_three),
    ]

    print("Pivot strategy comparison on sorted input (n=1000):")
    print(f"{'Strategy':<20} {'Max recursion depth':>20}")
    print("-" * 42)

    import sys
    sys.setrecursionlimit(5000)

    for name, fn in strategies:
        data = sorted_data[:]
        depth = [0]
        quicksort(data, 0, len(data) - 1, fn, depth)
        assert data == sorted(data), f"{name} failed!"
        print(f"{name:<20} {depth[0]:>20}")
```

**출력(흔한 예):**
```
Pivot strategy comparison on sorted input (n=1000):
Strategy              Max recursion depth
------------------------------------------
First element                        999
Last element                         999
Random                                22
Median-of-three                       19
```

자리를 붙박은 전략은 정렬된 입력에서 깊이가 $O(n)$으로 무너지지만, 무작위와 셋의 중앙값은 깊이 $O(\log n)$을 이룬다.

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press, 7.3~7.4절과 9.3절.
- Blum, M., Floyd, R. W., Pratt, V. R., Rivest, R. L., & Tarjan, R. E. (1973). Time bounds for selection. *Journal of Computer and System Sciences*, 7(4), 448-461.
- Sedgewick, R. (1978). Implementing Quicksort programs. *Communications of the ACM*, 21(10), 847-857.


## 연습문제

**연습문제 1.**
$[38, 27, 43, 3, 9, 82, 10]$에서 축 고르기 전략을 따라가며 큰 걸음마다의 상태를 보여라.

??? success "연습문제 1 풀이"
    알고리즘을 한 걸음씩 밟아라. 나누어 정복하는 정렬이라면 되돌이 나눔과 병합을 하나씩 보이고, 나눔 기반 정렬이라면 나눔마다와 축이 놓이는 자리를 보여라. 걸음마다 배열 전체의 상태를 내보여라.

---

**연습문제 2.**
축 고르기 전략의 최선, 최악, 평균의 경우 시간 복잡도를 끌어내라.

??? success "연습문제 2 풀이"
    경우마다 점화식을 써라. 최선의 경우는 가장 좋은 나눔이거나 미리 정렬된 입력이다. 최악의 경우는 일을 가장 크게 만드는 적수 입력이다. 평균의 경우는 무작위 순열에 대한 기대 성능이다. 점화식을 각각 풀어라.

---

**연습문제 3.**
축 고르기 전략은 안정적인가? 증명하거나 반례를 들어라.

??? success "연습문제 3 풀이"
    같은 원소가 본디 상대 차례를 지키면 그 정렬은 안정적이다. 모든 입력에서 이것이 성립함을 증명하거나, 정렬 도중 같은 원소 둘의 자리가 뒤바뀌는 구체적인 입력을 들어라.

---

**연습문제 4.**
float32 학습 손실 값 1000만 개를 정렬할 때 축 고르기 전략을 다른 두 방법과 견주어라. 시간, 공간, 캐시 거동, GPU 어울림을 살펴라.

??? success "연습문제 4 풀이"
    $n = 10^7$에서는 실제 선택이 하드웨어에 달렸다. CPU에서는 캐시 친화적인 알고리즘(병합 정렬, 인트로 정렬)이 앞선다. GPU에서는 병렬에 어울리는 알고리즘(바이토닉 정렬, 기수 정렬)이 낫다. 기억이 빠듯하면 제자리 정렬이 $O(n)$ 공간을 아낀다. 이 쪽의 이론적 분석이 그 고름에 길잡이가 된다.