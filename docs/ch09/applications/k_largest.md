# 가장 큰 k개 원소

항목 $n$개의 모음에서 가장 큰(또는 가장 작은) $k$개를 찾는 일은 데이터 처리와 순위 매기기와 분석에서 흔한 문제이다. 배열 전체를 정렬하면 $O(n \log n)$이 들지만, 힙을 쓰는 방법은 $O(k)$의 공간만으로 $O(n \log k)$ 시간에 푼다. $k \ll n$이면 크게 나아지며, 원소가 하나씩 흘러드는 상황에도 자연스레 들어맞는다.

## 최소 힙을 쓰는 방법

이 방법은 지금까지 본 가장 큰 $k$개를 담는 "창"으로 **크기 $k$의 최소 힙**을 쓴다. 힙의 최솟값이 문턱값 노릇을 한다. 이 문턱값보다 큰 새 원소는 상위 $k$개에 든다.

### 알고리즘

1. 앞의 $k$개 원소를 최소 힙에 넣는다.
2. 남은 원소 $x$마다 다음과 같이 한다.
    - $x$이 힙의 최솟값보다 크면 최솟값을 $x$으로 바꾸고 아래로 내린다.
    - 그렇지 않으면 $x$을 건너뛴다.
3. 힙이 가장 큰 $k$개 원소를 담는다.

### 왜 최소 힙인가

(최대 힙이 아니라) 최소 힙을 쓰는 것이 핵심 통찰이다. 최소 힙의 뿌리는 **가장 큰 $k$개 가운데 가장 작은 것**이며, 이것이 바로 새 후보마다 견주고 싶은 원소이다. 새 원소가 이 문턱값을 넘으면 지금의 최솟값을 밀어내고 아래로 내리기가 힙 성질을 되살린다.

## 복잡도 분석

| 단계 | 비용 |
|------|------|
| 크기 $k$의 첫 힙 세우기 | $O(k)$ |
| 남은 $n - k$개 원소 처리하기 | 많아야 $(n - k) \cdot O(\log k)$ |
| **모두** | $O(n \log k)$ |

$$
T(n, k) = O(k) + O((n - k) \log k) = O(n \log k)
$$

**공간**: 힙에 $O(k)$.

$k$이 상수이면(이를테면 "상위 10개") 시간이 $O(n)$으로 간단해진다.

## 여러 방식 견주기

| 방법 | 시간 | 공간 | 흘려보내기 가능한가 |
|----------|------|-------|-----------|
| 정렬한 뒤 뒤의 $k$개 가져오기 | $O(n \log n)$ | 제자리 정렬이면 $O(1)$ | 아니다 |
| 크기 $k$의 최소 힙 | $O(n \log k)$ | $O(k)$ | 그렇다 |
| 퀵셀렉트와 나누기 | 기댓값 $O(n)$ | $O(1)$ | 아니다 |
| 크기 $n$의 최대 힙과 $k$번 꺼내기 | $O(n + k \log n)$ | $O(n)$ | 아니다 |

흘려보내기가 필요하거나 $k \ll n$일 때는 최소 힙 방법이 최적이다. 데이터가 메모리에 들어가고 한 번만 셈하면 될 때는 퀵셀렉트가 빠르다.

## 구현

```python
"""
최소 힙으로 가장 큰 k개 원소 찾기.

흘려보내는 데이터에 알맞은, O(n log k) 시간과 O(k) 공간에 도는
힙 기반 방법을 보인다.
"""

import heapq


# === 맨바닥부터 구현하기 ===

def k_largest_manual(arr, k):
    """손수 다루는 최소 힙으로 가장 큰 k개 원소를 찾는다."""
    if k <= 0:
        return []
    if k >= len(arr):
        return sorted(arr, reverse=True)

    # 앞의 k개 원소로 최소 힙을 세운다
    heap = arr[:k]
    # 힙을 O(k)에 세운다
    for i in range(k // 2 - 1, -1, -1):
        _sift_down(heap, i, k)

    # 남은 원소를 처리한다
    for x in arr[k:]:
        if x > heap[0]:
            heap[0] = x
            _sift_down(heap, 0, k)

    # 정렬된 순서로 꺼낸다 (큰 것부터)
    result = []
    while heap:
        # 뿌리를 마지막과 맞바꾸고 줄인 뒤 아래로 내린다
        heap[0], heap[-1] = heap[-1], heap[0]
        result.append(heap.pop())
        if heap:
            _sift_down(heap, 0, len(heap))
    return result


def _sift_down(arr, i, n):
    """최소 힙에서 아래로 내리기."""
    while True:
        smallest = i
        left = 2 * i + 1
        right = 2 * i + 2
        if left < n and arr[left] < arr[smallest]:
            smallest = left
        if right < n and arr[right] < arr[smallest]:
            smallest = right
        if smallest == i:
            break
        arr[i], arr[smallest] = arr[smallest], arr[i]
        i = smallest


# === 파이썬 heapq 쓰기 ===

def k_largest_heapq(arr, k):
    """heapq.nlargest로 가장 큰 k개 원소를 찾는다."""
    return heapq.nlargest(k, arr)


def k_largest_heap_manual_heapq(arr, k):
    """힙을 손수 다루며 heapq로 가장 큰 k개를 찾는다."""
    if k <= 0:
        return []
    if k >= len(arr):
        return sorted(arr, reverse=True)

    # 크기 k의 최소 힙을 지킨다
    heap = arr[:k]
    heapq.heapify(heap)

    for x in arr[k:]:
        if x > heap[0]:
            heapq.heapreplace(heap, x)  # 한 연산으로 최솟값을 빼고 x를 넣는다

    return sorted(heap, reverse=True)


# === 시연 ===

if __name__ == "__main__":
    data = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9]
    k = 5

    print(f"Array: {data}")
    print(f"k = {k}\n")

    result1 = k_largest_manual(data, k)
    print(f"Manual heap:   {result1}")

    result2 = k_largest_heapq(data, k)
    print(f"heapq.nlargest: {result2}")

    result3 = k_largest_heap_manual_heapq(data, k)
    print(f"heapq manual:  {result3}")

    # 흘려보내기 보기
    print("\n--- Streaming Example ---")
    stream = [3, 7, 2, 8, 1, 9, 4, 6]
    k = 3
    heap = []
    for i, x in enumerate(stream):
        if len(heap) < k:
            heapq.heappush(heap, x)
        elif x > heap[0]:
            heapq.heapreplace(heap, x)
        print(f"  After seeing {x}: top-{k} = {sorted(heap, reverse=True)}")
```

**출력:**
```
Array: [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9]
k = 5

Manual heap:   [9, 9, 9, 8, 7]
heapq.nlargest: [9, 9, 9, 8, 7]
heapq manual:  [9, 9, 9, 8, 7]

--- Streaming Example ---
  After seeing 3: top-3 = [3]
  After seeing 7: top-3 = [7, 3]
  After seeing 2: top-3 = [7, 3, 2]
  After seeing 8: top-3 = [8, 7, 3]
  After seeing 1: top-3 = [8, 7, 3]
  After seeing 9: top-3 = [9, 8, 7]
  After seeing 4: top-3 = [9, 8, 7]
  After seeing 6: top-3 = [9, 8, 7]
```

## 가장 작은 k개 원소

쌍대 문제인 가장 작은 $k$개 찾기는 **크기 $k$의 최대 힙**을 쓴다. 뿌리가 가장 작은 $k$개 가운데 가장 큰 것을 지니고, 뿌리보다 작은 새 원소가 그것을 대신한다. (최소 힙만 주는) 파이썬의 `heapq`에서는 값의 부호를 뒤집는다.

```python
import heapq

def k_smallest(arr, k):
    """부호를 뒤집은 최대 힙으로 가장 작은 k개 원소를 찾는다."""
    heap = [-x for x in arr[:k]]
    heapq.heapify(heap)
    for x in arr[k:]:
        if x < -heap[0]:
            heapq.heapreplace(heap, -x)
    return sorted(-x for x in heap)
```

아니면 이를 속에서 처리해 주는 `heapq.nsmallest(k, arr)`를 쓴다.

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., and Stein, C. *Introduction to Algorithms* (4th ed.), Chapter 6 and 9. MIT Press.
- 파이썬 문서: [heapq.nlargest](https://docs.python.org/3/library/heapq.html#heapq.nlargest)


## 연습문제

**연습문제 1.**
가장 큰 k개 원소의 힙 성질을 밝히고 최솟값·최댓값 원소가 언제나 뿌리에 있음을 증명하라.

??? success "연습문제 1 풀이"
    힙 성질은 노드마다 열쇠가 자식보다 작거나 같거나(최소 힙) 크거나 같다(최대 힙)는 것이다. 뿌리에서 잎까지의 어떤 경로에서도 추이성이 성립하므로 뿌리가 모든 원소의 최솟값(또는 최댓값)이다.

---

**연습문제 2.**
배열 $[4, 7, 2, 9, 1, 5, 3]$에서 가장 큰 k개 원소를 따라가라. 단계마다와 그 결과로 나오는 힙을 보여라.

??? success "연습문제 2 풀이"
    이 쪽의 연산을 주어진 배열에 적용하라. 단계마다 배열과 그것이 나타내는 트리를 보여라. 비교와 자리바꿈을 짚어라.

---

**연습문제 3.**
가장 큰 k개 원소의 시간 복잡도를 증명하라. 그 한계는 빡빡한가?

??? success "연습문제 3 풀이"
    이 연산은 뿌리에서 잎까지 또는 잎에서 뿌리까지의 경로를 훑으며 층마다 $O(1)$의 일을 한다. 완전 이진 트리의 높이는 $\lfloor\log_2 n\rfloor$이므로 모두 $O(\log n)$이다. 이 한계는 빡빡하다. 높이 전체를 훑도록 강요하는 입력이 있다. $\square$

---

**연습문제 4.**
$k = 5$이고 어휘가 $n = 32{,}000$인 빔 탐색에서 힙으로 뽑기(원소 $n$개에서 상위 $k$개)와 정렬을 견주어라.

??? success "연습문제 4 풀이"
    정렬은 $O(n\log n) = O(32000 \times 15) \approx 48만$번의 연산이 든다. 힙(크기 $k$인 최소 힙)은 $O(n\log k) = O(32000 \times 2.3) \approx 7만 4천$번이다. 힙이 약 $6.5$배 빠르다. 아니면 `torch.topk`가 GPU에 맞추어 다듬은 부분 정렬 알고리즘을 쓴다.