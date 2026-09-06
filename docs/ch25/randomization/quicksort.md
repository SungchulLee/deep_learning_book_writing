# 무작위 빠른 정렬

정해진 빠른 정렬은 짓궂은 들임에 약하다. 곧 축 고르기 규칙을 내다볼 수 있으면 짓궂은 이가 $O(n^2)$번 견주게 하는 들임을 만들 수 있다. **마구잡이 빠른 정렬**은 축을 고르게 아무렇게나 골라 이 약점을 없애고, 어떤 들임도 가장 나쁜 움직임을 한결같이 일으킬 수 없게 한다. 기댓값 견줌 횟수는 모든 들임에서 $O(n \log n)$이며, 표시 아무 변수를 쓴 살피기는 기댓값 선형성의 가장 아름다운 쓰임새 가운데 하나이다.

## 알고리즘

배열 $A[1 \ldots n]$이 주어질 때 마구잡이 빠른 정렬은 다음과 같이 나아간다.

1. $n \leq 1$이면 돌아간다.
2. $\{1, 2, \ldots, n\}$에서 축 어깨수 $q$을 고르게 아무렇게나 고른다.
3. $A[q]$을 가운데 두고 $A$을 가른다. 곧 $A[q]$보다 작은 낱개는 왼쪽으로, 큰 낱개는 오른쪽으로 간다.
4. 왼쪽과 오른쪽 아래 배열에 되돌이한다.

마구잡이는 걸음 2에만 있다. 이 알고리즘은 늘 제대로 정렬한 배열을 낸다(라스베이거스 알고리즘이다).

## 기댓값 견줌 횟수

$z_1 < z_2 < \cdots < z_n$을 정렬한 차례의 $A$의 낱개라 하자. 표시 아무 변수를 다음과 같이 둔다.

$$
X_{ij} = \begin{cases} 1 & \text{if } z_i \text{ and } z_j \text{ are compared during execution} \\ 0 & \text{otherwise} \end{cases}
$$

온 견줌 횟수는 다음과 같다.

$$
X = \sum_{i=1}^{n-1} \sum_{j=i+1}^{n} X_{ij}
$$

기댓값의 선형성에 따라,

$$
E[X] = \sum_{i=1}^{n-1} \sum_{j=i+1}^{n} \Pr[z_i \text{ and } z_j \text{ are compared}]
$$

**핵심 살핌.** 낱개 $z_i$과 $z_j$이 견주어지는 것과 둘 가운데 하나가 모임 $\{z_i, z_{i+1}, \ldots, z_j\}$에서 처음으로 축이 되는 것은 서로 같다. $i < k < j$인 낱개 $z_k$이 먼저 축이 되면 $z_i$과 $z_j$은 다른 아래 배열로 갈려 결코 견주어지지 않는다.

모임 $\{z_i, z_{i+1}, \ldots, z_j\}$에는 낱개가 $j - i + 1$개 있고 저마다 이 모임에서 처음 축이 될 가능성이 같다. $z_i$이나 $z_j$이 먼저 골릴 확률은 다음과 같다.

$$
\Pr[X_{ij} = 1] = \frac{2}{j - i + 1}
$$

따라서,

$$
E[X] = \sum_{i=1}^{n-1} \sum_{j=i+1}^{n} \frac{2}{j - i + 1}
$$

$k = j - i$을 넣으면,

$$
E[X] = \sum_{i=1}^{n-1} \sum_{k=1}^{n-i} \frac{2}{k + 1} < \sum_{i=1}^{n-1} \sum_{k=1}^{n} \frac{2}{k} = 2(n-1) H_n
$$

여기서 $H_n = \sum_{k=1}^{n} 1/k = \ln n + O(1)$은 $n$번째 조화수이다. 따라서

$$
E[X] = 2n \ln n + O(n) = O(n \log n)
$$

!!! tip "이 살피기가 통하는 까닭"
    표시 변수 재주는 되돌이 관계식을 아예 풀지 않아도 되게 한다. 아무 가르기와 되돌이 아래 문제 크기를 따지는 대신 낱개 짝마다 견주어질 확률을 곧바로 센다. 기댓값의 선형성이 모든 매임을 저절로 다룬다.

## 평균 둘레의 모임

기댓값 견줌 횟수는 $\Theta(n \log n)$인데 크게 벗어날 가능성은 얼마나 될까? 마팅게일 논증이나 꼼꼼한 흩어짐 살피기로 다음을 보일 수 있다.

$$
\Pr[X > c \cdot n \log n] \leq n^{-\alpha}
$$

알맞은 상수 $c$과 $\alpha$에 대해 그렇다. 도는 시간이 기댓값 둘레에 날카롭게 모이므로 마구잡이 빠른 정렬은 실제로 믿을 만하다.

## 구현

```python
"""
아무 축을 고르는 마구잡이 빠른 정렬.

라스베이거스의 보장을 보인다. 곧 늘 옳다.
with O(n log n) expected comparisons.
"""

import random

# === 나눔 ===

def partition(arr, lo, hi):
    """Lomuto partition around arr[hi]."""
    pivot = arr[hi]
    i = lo
    for j in range(lo, hi):
        if arr[j] <= pivot:
            arr[i], arr[j] = arr[j], arr[i]
            i += 1
    arr[i], arr[hi] = arr[hi], arr[i]
    return i

# === 무작위 빠른 정렬 ===

def randomized_quicksort(arr, lo, hi):
    """Sort arr[lo..hi] using a uniformly random pivot."""
    if lo < hi:
        pivot_idx = random.randint(lo, hi)
        arr[pivot_idx], arr[hi] = arr[hi], arr[pivot_idx]
        mid = partition(arr, lo, hi)
        randomized_quicksort(arr, lo, mid - 1)
        randomized_quicksort(arr, mid + 1, hi)

# === 메인 ===

if __name__ == "__main__":
    data = [3, 6, 8, 10, 1, 2, 1]
    randomized_quicksort(data, 0, len(data) - 1)
    print(data)
```

**출력:**
```
[1, 1, 2, 3, 6, 8, 10]
```

## 가장 나쁜 경우의 확률

가장 나쁜 경우는 $O(n^2)$이지만 그에 가까워질 확률은 아주 작다. 마구잡이 빠른 정렬이 $cn \log n$번보다 많이 견줄 확률은 $c$에 따라 지수로 줄어들어 어떤 들임에서도 $O(n^2)$ 움직임은 사실상 일어나지 않는다.

## 참고 문헌

- Motwani, R. & Raghavan, P. *Randomized Algorithms*. Cambridge University Press, 1995.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L. & Stein, C. *Introduction to Algorithms*. MIT Press, 2022.

## 연습문제

**연습문제 1.**
마구잡이 빠른 정렬의 기댓값 도는 시간이 $O(n \log n)$임을 밝혀라.

??? success "연습문제 1 풀이"
    기댓값 견줌 횟수는 $\sum_{i=1}^{n} \sum_{j=i+1}^{n} P(z_i \text{과 } z_j \text{이 견주어짐}) = \sum_{i=1}^{n} \sum_{j=i+1}^{n} \frac{2}{j - i + 1}$이다. $z_i$과 $z_j$은 둘 가운데 하나가 $\{z_i, z_{i+1}, \ldots, z_j\}$에서 처음 축이 될 때만 견주어지고 그 확률이 $2/(j-i+1)$이기 때문이다. 이중 합은 $\sum_{i=1}^{n} \sum_{k=2}^{n-i+1} \frac{2}{k} \leq \sum_{i=1}^{n} 2H_n = 2nH_n = O(n \log n)$이 되며 $H_n$은 조화수이다. $\square$

---

**연습문제 2.**
마구잡이 빠른 정렬의 가장 나쁜 경우 도는 시간은 얼마인가? 그럴 가능성은 얼마인가?

??? success "연습문제 2 풀이"
    가장 나쁜 경우는 $O(n^2)$이며 축이 늘 가장 작거나 가장 큰 낱개일 때 일어난다. 그럴 확률은 $2/n \cdot 2/(n-1) \cdots = O(1/n!)$으로 천문학처럼 낮다. 체르노프 같은 모임 한계에 따라 마구잡이 빠른 정렬이 (알맞은 상수 $c$에 대해) $cn \log n$번보다 많이 견줄 확률은 지수로 줄어든다. 실제로 $O(n^2)$ 움직임은 사실상 일어나지 않는다. $\square$

---

**연습문제 3.**
마구잡이 빠른 정렬을 셋 가운데값으로 축을 고르는 정해진 빠른 정렬과 견주어라.

??? success "연습문제 3 풀이"
    **마구잡이**: 들임과 상관없이 기댓값 $O(n \log n)$이다. 축이 아무렇게나 골리므로 짓궂은 들임도 가장 나쁜 움직임을 일으킬 수 없다. 짜기 쉽다. **셋 가운데값**: 아무 들임에서는 평균으로 정해진 $O(n \log n)$이지만 짓궂은 들임이 $O(n^2)$을 일으킬 수 있다. 상수 갑절이 조금 낫다(축이 평균으로 가운데값에 더 가깝다). 실제로는 흔한 자료에서 둘의 솜씨가 비슷하며, 짓궂은 들임이 있을 수 있으면 마구잡이를 더 낫게 여긴다. $\square$

---

**연습문제 4.**
가장 나쁜 경우의 복잡도가 더 나쁜데도 실제로 빠른 정렬을 합치기 정렬보다 더 낫게 여기는 까닭은 무엇인가?

??? success "연습문제 4 풀이"
    빠른 정렬에는 실제의 장점이 여럿 있다. (1) **제자리에서**: 합치기 정렬의 $O(n)$ 곁 배열과 달리 $O(\log n)$ 쌓기 자리만 쓴다. (2) **저장턱에 친함**: 차례로 훑는 결이 CPU 저장턱과 잘 맞는다. (3) **작은 상수 갑절**: 견줌마다 자료를 덜 옮긴다. (4) **맞추어 감**: 마구잡이 빠른 정렬은 들임 짜임에 맞추어 간다. 합치기 정렬은 가장 나쁜 경우 $O(n \log n)$을 보장하고 안정되므로 이음 목록과 안정이 중요한 곳에서 더 낫다. 대개의 표준 꾸러미 정렬은 둘을 아우른 섞은 방식(예컨대 Timsort, introsort)을 쓴다. $\square$
