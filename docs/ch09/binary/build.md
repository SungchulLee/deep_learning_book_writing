# 힙 세우기

정렬되지 않은 원소 $n$개의 배열이 주어지면, 꺼내기나 우선순위 큐 연산을 하기 전에 이를 올바른 힙으로 바꾸어야 할 때가 많다. 원소를 하나씩 넣는 순진한 방법은 $O(n \log n)$이 든다. **힙 세우기** 알고리즘은 아래에서 위로 가는 방법을 써서 같은 결과를 $O(n)$ 시간에 이룬다. 마지막 잎 아닌 노드에서 시작해 자리마다 아래로 내리기를 적용하며 뿌리 쪽으로 나아간다.

---

## 1. 순진한 방법: 되풀이 삽입

힙을 세우는 가장 곧은 길은 빈 힙에서 시작해 원소를 하나씩 넣는 것이다. 삽입마다 위로 올리기를 부르는데 최악의 경우 $O(\log n)$이 든다. 삽입 $n$번에 걸친 전체 비용은 다음과 같다.

$$
\sum_{i=1}^{n} O(\log i) = O(n \log n)
$$

이렇게 해도 되지만 최선은 아니다. 힙 세우기 알고리즘은 실제로 뜻있는 상수 배만큼, 그리고 이론상 점근 등급 하나만큼 더 낫다.

---

## 2. 아래에서 위로 가는 힙 세우기 알고리즘

핵심 통찰은 잎이 (어길 자식이 없으므로) 이미 힙 성질을 자명하게 만족한다는 것이다. 노드가 $n$개인 완전 이진 트리에서 잎은 (0부터 셀 때) 색인 $\lfloor n/2 \rfloor$부터 $n-1$까지를 차지한다. 힙 세우기는 마지막 잎 아닌 노드에서 시작해 색인마다 아래로 내리기를 적용하며 뿌리 쪽으로 거슬러 간다.

### 알고리즘

```
BUILD-MAX-HEAP(A):
    n = length(A)
    for i = floor(n/2) - 1 down to 0:
        MAX-HEAPIFY(A, i, n)
```

`MAX-HEAPIFY(A, i, n)`을 부를 때마다 색인 $i$을 뿌리로 하는 부분 트리가 올바른 최대 힙이 된다. 고리가 노드를 아래에서 위로 처리하므로 노드 $i$을 힙으로 만들 때에는 두 자식의 부분 트리가 이미 올바른 힙이다.

### 한 걸음씩 보는 보기

배열 `[4, 1, 3, 2, 16, 9, 10, 14, 8, 7]`으로 최대 힙을 세워 보자.

```
Initial array (n=10, last non-leaf at index 4):

          4
        /   \
      1       3
     / \     / \
    2   16  9   10
   / \  /
  14 8 7

Step 1: heapify index 4 (value 16) — children: 7. No swap needed.
Step 2: heapify index 3 (value 2)  — children: 14, 8. Swap 2 and 14.
Step 3: heapify index 2 (value 3)  — children: 9, 10. Swap 3 and 10.
Step 4: heapify index 1 (value 1)  — children: 14, 16. Swap 1 and 16, then 1 and 7.
Step 5: heapify index 0 (value 4)  — children: 16, 10. Swap 4 and 16, then 4 and 14, then 4 and 8.

Result:
          16
        /    \
      14      10
     / \     /  \
    8   7   9    3
   / \  /
  2  4 1
```

마지막 배열은 `[16, 14, 10, 8, 7, 9, 3, 2, 4, 1]`이다.

---

## 3. 아래에서 위로가 왜 O(n)인가

아래로 내리기마다 부분 트리의 높이를 $h$이라 할 때 $O(h)$까지 들 수 있으므로 $O(n)$ 복잡도가 뻔하지는 않다. 핵심은 노드 대부분이 부분 트리의 높이가 작은 트리 아래쪽에 있다는 것이다.

- 높이 0(잎): 노드 $\lceil n/2 \rceil$개, 저마다 일이 0
- 높이 1: 노드 $\lceil n/4 \rceil$개, 저마다 자리바꿈 많아야 1번
- 높이 $h$: 노드 $\lceil n/2^{h+1} \rceil$개, 저마다 자리바꿈 많아야 $h$번

전체 일은 다음과 같다.

$$
\sum_{h=0}^{\lfloor \log n \rfloor} \left\lceil \frac{n}{2^{h+1}} \right\rceil \cdot O(h) = O\!\left(n \sum_{h=0}^{\infty} \frac{h}{2^h}\right) = O(n)
$$

무한급수 $\sum_{h=0}^{\infty} h/2^h = 2$이 수렴하므로 힙 세우기 절차 전체가 $O(n)$ 시간에 돈다. 자세한 증명은 형제 쪽인 *힙 세우기의 O(n) 증명*에서 다룬다.

---

## 4. 구현

```python
"""
힙 세우기 알고리즘 구현.

아래에서 위로 내리기 방법으로 정렬되지 않은 배열에서
최대 힙을 O(n) 시간에 세운다.
"""

# === 아래로 내리기 (최대 힙으로 만들기) ===

def sift_down(arr, i, n):
    """색인 i를 뿌리로 하는 부분 트리의 최대 힙 성질을 되살린다.

    i의 두 부분 트리가 이미 올바른 최대 힙이라고 본다.
    arr[0:n]의 원소만 따진다.
    """
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2

    if left < n and arr[left] > arr[largest]:
        largest = left
    if right < n and arr[right] > arr[largest]:
        largest = right

    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        sift_down(arr, largest, n)

# === 최대 힙 세우기 ===

def build_max_heap(arr):
    """arr을 O(n) 시간에 최대 힙으로 바꾼다.

    마지막 잎 아닌 노드에서 뿌리까지 거슬러 가며
    자리마다 sift_down을 부른다.
    """
    n = len(arr)
    # 마지막 잎 아닌 노드는 색인 n//2 - 1에 있다
    for i in range(n // 2 - 1, -1, -1):
        sift_down(arr, i, n)

# === 최소 힙 세우기 ===

def sift_down_min(arr, i, n):
    """색인 i를 뿌리로 하는 부분 트리의 최소 힙 성질을 되살린다."""
    smallest = i
    left = 2 * i + 1
    right = 2 * i + 2

    if left < n and arr[left] < arr[smallest]:
        smallest = left
    if right < n and arr[right] < arr[smallest]:
        smallest = right

    if smallest != i:
        arr[i], arr[smallest] = arr[smallest], arr[i]
        sift_down_min(arr, smallest, n)

def build_min_heap(arr):
    """arr을 O(n) 시간에 최소 힙으로 바꾼다."""
    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        sift_down_min(arr, i, n)

# === 시연 ===

if __name__ == "__main__":
    # 최대 힙을 세운다
    data = [4, 1, 3, 2, 16, 9, 10, 14, 8, 7]
    print(f"Original array: {data}")

    build_max_heap(data)
    print(f"Max-heap:       {data}")

    # 최대 힙 성질을 확인한다
    for i in range(1, len(data)):
        parent = (i - 1) // 2
        assert data[parent] >= data[i], f"Heap violation at index {i}"
    print("Max-heap property verified.\n")

    # 최소 힙을 세운다
    data2 = [4, 1, 3, 2, 16, 9, 10, 14, 8, 7]
    build_min_heap(data2)
    print(f"Min-heap:       {data2}")

    # 파이썬의 heapq와 견준다
    import heapq
    data3 = [4, 1, 3, 2, 16, 9, 10, 14, 8, 7]
    heapq.heapify(data3)
    print(f"heapq result:   {data3}")
```

**출력:**
```
Original array: [4, 1, 3, 2, 16, 9, 10, 14, 8, 7]
Max-heap:       [16, 14, 10, 8, 7, 9, 3, 2, 4, 1]
Max-heap property verified.

Min-heap:       [1, 2, 3, 4, 7, 9, 10, 14, 8, 16]
heapq result:   [1, 2, 3, 4, 7, 9, 10, 14, 8, 16]
```

---

## 5. 세우는 방법 견주기

| 방법 | 방식 | 시간 | 공간 |
|----------|----------|------|-------|
| 되풀이 삽입 | 하나씩 넣고 저마다 위로 올린다 | $O(n \log n)$ | $O(1)$ |
| 아래에서 위로 힙 세우기 | 마지막 잎 아닌 노드에서 뿌리까지 아래로 내린다 | $O(n)$ | $O(1)$ |

두 방법 모두 올바른 힙을 내지만 아래에서 위로 가는 쪽이 엄밀히 더 빠르다. 파이썬의 `heapq.heapify`는 속에서 아래에서 위로 가는 $O(n)$ 알고리즘을 쓴다.

---

## 연습문제

**연습문제 1.**
힙 세우기의 힙 성질을 밝히고 최솟값·최댓값 원소가 언제나 뿌리에 있음을 증명하라.

??? success "연습문제 1 풀이"
    힙 성질은 노드마다 열쇠가 자식보다 작거나 같거나(최소 힙) 크거나 같다(최대 힙)는 것이다. 뿌리에서 잎까지의 어떤 경로에서도 추이성이 성립하므로 뿌리가 모든 원소의 최솟값(또는 최댓값)이다.

---

**연습문제 2.**
배열 $[4, 7, 2, 9, 1, 5, 3]$에서 힙 세우기를 따라가라. 단계마다와 그 결과로 나오는 힙을 보여라.

??? success "연습문제 2 풀이"
    이 쪽의 연산을 주어진 배열에 적용하라. 단계마다 배열과 그것이 나타내는 트리를 보여라. 비교와 자리바꿈을 짚어라.

---

**연습문제 3.**
힙 세우기의 시간 복잡도를 증명하라. 그 한계는 빡빡한가?

??? success "연습문제 3 풀이"
    이 연산은 뿌리에서 잎까지 또는 잎에서 뿌리까지의 경로를 훑으며 층마다 $O(1)$의 일을 한다. 완전 이진 트리의 높이는 $\lfloor\log_2 n\rfloor$이므로 모두 $O(\log n)$이다. 이 한계는 빡빡하다. 높이 전체를 훑도록 강요하는 입력이 있다. $\square$

---

**연습문제 4.**
$k = 5$이고 어휘가 $n = 32{,}000$인 빔 탐색에서 힙으로 뽑기(원소 $n$개에서 상위 $k$개)와 정렬을 견주어라.

??? success "연습문제 4 풀이"
    정렬은 $O(n\log n) = O(32000 \times 15) \approx 48만$번의 연산이 든다. 힙(크기 $k$인 최소 힙)은 $O(n\log k) = O(32000 \times 2.3) \approx 7만 4천$번이다. 힙이 약 $6.5$배 빠르다. 아니면 `torch.topk`가 GPU에 맞추어 다듬은 부분 정렬 알고리즘을 쓴다.

## 정리하며

이 마당은 순진한 방법: 되풀이 삽입、아래에서 위로 가는 힙 세우기 알고리즘、아래에서 위로가 왜 O(n)인가、구현을 차례로 짚었다.

**참고 문헌**

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., and Stein, C. *Introduction to Algorithms* (4th ed.), Chapter 6.3: Building a heap. MIT Press.
