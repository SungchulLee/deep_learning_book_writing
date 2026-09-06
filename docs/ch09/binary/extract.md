# 최솟값·최댓값 꺼내기

**꺼내기** 연산은 힙의 뿌리 원소, 곧 최소 힙에서는 최솟값을, 최대 힙에서는 최댓값을 없애고 돌려준다. 이것이 힙을 우선순위 큐로 쓸모 있게 만드는 중심 연산이다. 언제나 우선순위가 가장 높은 원소를 $O(\log n)$ 시간에 준다. 힙 정렬이 그저 되풀이되는 꺼내기이므로, 꺼내기를 이해하면 힙 정렬이 왜 통하는지도 또렷해진다.

## 알고리즘

뿌리를 그냥 꺼내면 색인 0에 빈틈이 생기고 완전 이진 트리의 모양이 깨질 수 있다. 표준 방법은 세 단계 절차로 이를 피한다.

1. 뿌리 값(돌려줄 원소)을 **저장한다**.
2. 배열의 마지막 원소를 뿌리 자리로 **옮기고** 힙의 크기를 하나 줄인다. 이는 완전 이진 트리의 모양을 지킨다.
3. 새 뿌리를 **아래로 내려** 힙 성질을 되살린다.

### 최댓값 꺼내기의 의사 코드

```
EXTRACT-MAX(A):
    if heap_size < 1:
        error "heap underflow"
    max_val = A[0]
    A[0] = A[heap_size - 1]
    heap_size = heap_size - 1
    MAX-HEAPIFY(A, 0, heap_size)
    return max_val
```

## 한 걸음씩 보는 예

최대 힙 `[16, 14, 10, 8, 7, 9, 3, 2, 4, 1]`에서 최댓값을 꺼내 보자.

```
Step 1: Save root value 16.

Step 2: Move last element (1) to root, reduce size to 9.
          1
        /    \
      14      10
     /  \    /  \
    8    7  9    3
   / \
  2   4

Step 3: Sift down 1.
  Compare 1 with children 14 and 10. Swap with 14 (larger child).
          14
        /    \
      1       10
     /  \    /  \
    8    7  9    3
   / \
  2   4

  Compare 1 with children 8 and 7. Swap with 8.
          14
        /    \
      8       10
     /  \    /  \
    1    7  9    3
   / \
  2   4

  Compare 1 with children 2 and 4. Swap with 4.
          14
        /    \
      8       10
     /  \    /  \
    4    7  9    3
   / \
  2   1

Result: returned 16, heap is [14, 8, 10, 4, 7, 9, 3, 2, 1].
```

## 복잡도 분석

아래로 내리기 절차는 뿌리에서 잎까지의 경로를 많아야 하나 훑는다. 원소가 $n$개인 힙의 높이가 $\lfloor \log_2 n \rfloor$이므로 아래로 내리기는 비교와 자리바꿈을 많아야 $\lfloor \log_2 n \rfloor$번 한다.

| 연산 | 시간 복잡도 |
|-----------|----------------|
| 뿌리 저장하기 | $O(1)$ |
| 마지막을 뿌리로 옮기기 | $O(1)$ |
| 아래로 내리기 | $O(\log n)$ |
| **합계** | $O(\log n)$ |

공간 복잡도는 되풀이 판이 $O(1)$이고, 재귀 판은 호출 스택 때문에 $O(\log n)$이다.

## 꺼내지 않고 엿보기

최솟값이나 최댓값을 없애지 않고 살펴야 할 때가 있다. 뿌리가 언제나 끝값 원소를 지니므로 엿보기는 간단한 $O(1)$ 배열 접근이다.

$$
\text{peek}(A) = A[0]
$$

## 구현

```python
"""
이진 힙의 최솟값·최댓값 꺼내기 연산.

아래로 내리기로 뿌리 원소를 없애고 힙 성질을
O(log n) 시간에 되살리는 것을 보인다.
"""


# === 최대 힙에서 꺼내기 ===

class MaxHeap:
    """삽입과 최댓값 꺼내기와 엿보기를 받쳐 주는 최대 힙."""

    def __init__(self, items=None):
        """선택으로 주어진 항목 리스트로 최대 힙을 세운다."""
        self.heap = list(items) if items else []
        # 아래에서 위로 내리기로 힙을 세운다
        for i in range(len(self.heap) // 2 - 1, -1, -1):
            self._sift_down(i)

    def _sift_down(self, i):
        """힙 성질을 되살리려고 색인 i의 원소를 아래로 옮긴다."""
        n = len(self.heap)
        while True:
            largest = i
            left = 2 * i + 1
            right = 2 * i + 2

            if left < n and self.heap[left] > self.heap[largest]:
                largest = left
            if right < n and self.heap[right] > self.heap[largest]:
                largest = right

            if largest == i:
                break
            self.heap[i], self.heap[largest] = self.heap[largest], self.heap[i]
            i = largest

    def _sift_up(self, i):
        """힙 성질을 되살리려고 색인 i의 원소를 위로 옮긴다."""
        while i > 0:
            parent = (i - 1) // 2
            if self.heap[i] > self.heap[parent]:
                self.heap[i], self.heap[parent] = self.heap[parent], self.heap[i]
                i = parent
            else:
                break

    def insert(self, val):
        """힙에 값을 넣는다. O(log n)."""
        self.heap.append(val)
        self._sift_up(len(self.heap) - 1)

    def peek(self):
        """최댓값을 없애지 않고 돌려준다. O(1)."""
        if not self.heap:
            raise IndexError("peek from empty heap")
        return self.heap[0]

    def extract_max(self):
        """최댓값 원소를 없애고 돌려준다. O(log n)."""
        if not self.heap:
            raise IndexError("extract from empty heap")

        max_val = self.heap[0]

        # 마지막 원소를 뿌리로 옮긴다
        last = self.heap.pop()
        if self.heap:
            self.heap[0] = last
            self._sift_down(0)

        return max_val

    def __len__(self):
        return len(self.heap)

    def __repr__(self):
        return f"MaxHeap({self.heap})"


# === 시연 ===

if __name__ == "__main__":
    # 최대 힙을 세운다
    h = MaxHeap([4, 1, 3, 2, 16, 9, 10, 14, 8, 7])
    print(f"Initial heap: {h.heap}")
    print(f"Peek: {h.peek()}")

    # 원소를 하나씩 꺼낸다 (내림차순으로 정렬된다)
    print("\nExtracting elements:")
    extracted = []
    while len(h) > 0:
        val = h.extract_max()
        extracted.append(val)
        print(f"  Extracted {val}, heap: {h.heap}")

    print(f"\nExtracted in order: {extracted}")

    # 삽입과 꺼내기를 번갈아 하는 것을 보인다
    print("\n--- Insert and Extract ---")
    h2 = MaxHeap()
    for val in [5, 3, 8]:
        h2.insert(val)
        print(f"  Inserted {val}: {h2.heap}")

    print(f"  Extract max: {h2.extract_max()}, heap: {h2.heap}")
    h2.insert(10)
    print(f"  Inserted 10: {h2.heap}")
    print(f"  Extract max: {h2.extract_max()}, heap: {h2.heap}")
```

**출력:**
```
Initial heap: [16, 14, 10, 8, 7, 9, 3, 2, 4, 1]
Peek: 16

Extracting elements:
  Extracted 16, heap: [14, 8, 10, 4, 7, 9, 3, 2, 1]
  Extracted 14, heap: [10, 8, 9, 4, 7, 1, 3, 2]
  Extracted 10, heap: [9, 8, 3, 4, 7, 1, 2]
  Extracted 9, heap: [8, 7, 3, 4, 2, 1]
  Extracted 8, heap: [7, 4, 3, 1, 2]
  Extracted 7, heap: [4, 2, 3, 1]
  Extracted 4, heap: [3, 2, 1]
  Extracted 3, heap: [2, 1]
  Extracted 2, heap: [1]
  Extracted 1, heap: []

Extracted in order: [16, 14, 10, 9, 8, 7, 4, 3, 2, 1]

--- Insert and Extract ---
  Inserted 5: [5]
  Inserted 3: [5, 3]
  Inserted 8: [8, 3, 5]
  Extract max: 8, heap: [5, 3]
  Inserted 10: [10, 3, 5]
  Extract max: 10, heap: [5, 3]
```

## 올바름의 논증

마지막 원소를 뿌리로 옮긴 뒤에도 뿌리의 두 부분 트리는 (건드리지 않았으므로) 여전히 올바른 힙이다. 어긋날 수 있는 곳은 뿌리뿐이다. 아래로 내리기는 그 원소가 두 자식 이상으로 큰 자리에 닿거나 잎이 될 때까지 뿌리를 가장 큰 자식과 되풀이해 맞바꾸어 힙 성질을 되살린다. 이것이 바로 `MAX-HEAPIFY`가 요구하는 조건이다. 두 부분 트리는 올바른 힙이고 뿌리만 성질을 어길 수 있다.

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., and Stein, C. *Introduction to Algorithms* (4th ed.), Chapter 6.2 and 6.5: Maintaining the heap property and Priority queues. MIT Press.


## 연습문제

**연습문제 1.**
최솟값·최댓값 꺼내기의 힙 성질을 밝히고 최솟값·최댓값 원소가 언제나 뿌리에 있음을 증명하라.

??? success "연습문제 1 풀이"
    힙 성질은 노드마다 열쇠가 자식보다 작거나 같거나(최소 힙) 크거나 같다(최대 힙)는 것이다. 뿌리에서 잎까지의 어떤 경로에서도 추이성이 성립하므로 뿌리가 모든 원소의 최솟값(또는 최댓값)이다.

---

**연습문제 2.**
배열 $[4, 7, 2, 9, 1, 5, 3]$에서 최솟값·최댓값 꺼내기를 따라가라. 단계마다와 그 결과로 나오는 힙을 보여라.

??? success "연습문제 2 풀이"
    이 쪽의 연산을 주어진 배열에 적용하라. 단계마다 배열과 그것이 나타내는 트리를 보여라. 비교와 자리바꿈을 짚어라.

---

**연습문제 3.**
최솟값·최댓값 꺼내기의 시간 복잡도를 증명하라. 그 한계는 빡빡한가?

??? success "연습문제 3 풀이"
    이 연산은 뿌리에서 잎까지 또는 잎에서 뿌리까지의 경로를 훑으며 층마다 $O(1)$의 일을 한다. 완전 이진 트리의 높이는 $\lfloor\log_2 n\rfloor$이므로 모두 $O(\log n)$이다. 이 한계는 빡빡하다. 높이 전체를 훑도록 강요하는 입력이 있다. $\square$

---

**연습문제 4.**
$k = 5$이고 어휘가 $n = 32{,}000$인 빔 탐색에서 힙으로 뽑기(원소 $n$개에서 상위 $k$개)와 정렬을 견주어라.

??? success "연습문제 4 풀이"
    정렬은 $O(n\log n) = O(32000 \times 15) \approx 48만$번의 연산이 든다. 힙(크기 $k$인 최소 힙)은 $O(n\log k) = O(32000 \times 2.3) \approx 7만 4천$번이다. 힙이 약 $6.5$배 빠르다. 아니면 `torch.topk`가 GPU에 맞추어 다듬은 부분 정렬 알고리즘을 쓴다.