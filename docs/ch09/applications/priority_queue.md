# 우선순위 큐

**우선순위 큐**는 원소마다 우선순위를 붙여 담고 우선순위가 가장 높은 원소를 효율적으로 꺼내게 해 주는 추상 자료형이다. 이 추상 자료형은 정렬된 배열이나 연결 리스트로도 구현할 수 있지만, 이진 힙이 연산 비용의 균형이 가장 좋다. 삽입과 꺼내기가 $O(\log n)$이고 엿보기가 $O(1)$이다. 우선순위 큐는 컴퓨터 과학 곳곳에 쓰인다. 데이크스트라의 최단 경로 알고리즘, 허프만 부호화, 사건 기반 모의 실험, 운영체제의 작업 일정 짜기가 그 보기이다.

---

## 1. 추상 자료형

우선순위 큐는 다음 핵심 연산을 받쳐 준다.

| 연산 | 설명 | 힙에서의 비용 |
|-----------|-------------|-----------|
| `insert(key, priority)` | 주어진 우선순위로 원소를 더한다 | $O(\log n)$ |
| `extract_min` / `extract_max` | 우선순위가 가장 높은 원소를 꺼내 돌려준다 | $O(\log n)$ |
| `peek` | 없애지 않고 우선순위가 가장 높은 원소를 돌려준다 | $O(1)$ |
| `decrease_key(i, new_key)` | 자리 $i$의 원소의 열쇠를 낮춘다 | $O(\log n)$ |
| `is_empty` | 큐에 원소가 있는지 살핀다 | $O(1)$ |

**최소 우선순위 큐**는 가장 작은 열쇠를 가장 높은 우선순위로 본다(데이크스트라 알고리즘에 쓴다). **최대 우선순위 큐**는 가장 큰 열쇠를 가장 높은 우선순위로 본다(마감으로 일정을 짤 때 쓴다).

---

## 2. 힙으로 하는 구현

이진 최소 힙이 최소 우선순위 큐를 곧바로 구현한다. 힙 성질이 가장 작은 원소를 언제나 뿌리에 두어 엿보기를 $O(1)$으로 만든다. 핵심 연산은 힙 연산에 이렇게 잇댄다.

- `insert` = 끝에 덧붙이고 위로 올리기
- `extract_min` = 뿌리를 저장하고 마지막을 뿌리로 옮긴 뒤 아래로 내리기
- `decrease_key` = 색인 $i$의 열쇠를 낮춘 뒤 위로 올리기

### 열쇠 낮추기

**열쇠 낮추기** 연산은 알려진 자리 $i$의 원소의 열쇠를 더 작은 새 값으로 낮춘다. 새 열쇠가 부모보다 작을 수 있으므로 자리 $i$에서 위로 올린다.

```
DECREASE-KEY(A, i, new_key):
    if new_key > A[i]:
        error "new key is larger than current key"
    A[i] = new_key
    while i > 0 and A[parent(i)] > A[i]:
        swap A[i] and A[parent(i)]
        i = parent(i)
```

이 연산은 변 늦추기가 이미 큐에 있는 꼭짓점의 우선순위를 고쳐야 하는 데이크스트라나 프림 같은 그래프 알고리즘에서 매우 중요하다.

---

## 3. 구현

```python
"""
이진 최소 힙으로 구현한 우선순위 큐.

삽입, 최솟값 꺼내기, 엿보기, 열쇠 낮추기를 받쳐 주며
실제로 쓰기 좋게 (우선순위, 값) 쌍을 다룬다.
"""

# === 최소 우선순위 큐 ===

class MinPriorityQueue:
    """이진 최소 힙에 바탕한 최소 우선순위 큐.

    원소를 (우선순위, 값) 쌍으로 담는다.
    우선순위가 가장 작은 원소를 먼저 꺼낸다.
    """

    def __init__(self):
        self.heap = []

    def _sift_up(self, i):
        """힙 성질을 되살리려고 색인 i의 원소를 위로 옮긴다."""
        while i > 0:
            parent = (i - 1) // 2
            if self.heap[i][0] < self.heap[parent][0]:
                self.heap[i], self.heap[parent] = self.heap[parent], self.heap[i]
                i = parent
            else:
                break

    def _sift_down(self, i):
        """힙 성질을 되살리려고 색인 i의 원소를 아래로 옮긴다."""
        n = len(self.heap)
        while True:
            smallest = i
            left = 2 * i + 1
            right = 2 * i + 2

            if left < n and self.heap[left][0] < self.heap[smallest][0]:
                smallest = left
            if right < n and self.heap[right][0] < self.heap[smallest][0]:
                smallest = right

            if smallest == i:
                break
            self.heap[i], self.heap[smallest] = self.heap[smallest], self.heap[i]
            i = smallest

    def insert(self, priority, value):
        """주어진 우선순위로 원소를 넣는다. O(log n)."""
        self.heap.append((priority, value))
        self._sift_up(len(self.heap) - 1)

    def peek(self):
        """우선순위가 가장 작은 원소를 없애지 않고 돌려준다. O(1)."""
        if not self.heap:
            raise IndexError("peek from empty priority queue")
        return self.heap[0]

    def extract_min(self):
        """우선순위가 가장 작은 원소를 없애고 돌려준다. O(log n)."""
        if not self.heap:
            raise IndexError("extract from empty priority queue")
        min_elem = self.heap[0]
        last = self.heap.pop()
        if self.heap:
            self.heap[0] = last
            self._sift_down(0)
        return min_elem

    def decrease_key(self, i, new_priority):
        """색인 i의 원소의 우선순위를 낮춘다. O(log n)."""
        if new_priority > self.heap[i][0]:
            raise ValueError("new priority is larger than current priority")
        self.heap[i] = (new_priority, self.heap[i][1])
        self._sift_up(i)

    def is_empty(self):
        """큐가 비었는지 살핀다. O(1)."""
        return len(self.heap) == 0

    def __len__(self):
        return len(self.heap)

    def __repr__(self):
        return f"MinPriorityQueue({self.heap})"

# === 시연 ===

if __name__ == "__main__":
    pq = MinPriorityQueue()

    # 작업 일정 짜기를 흉내 낸다
    tasks = [
        (3, "low-priority task"),
        (1, "urgent task"),
        (2, "medium task"),
        (5, "background task"),
        (1, "another urgent task"),
    ]

    print("Inserting tasks:")
    for priority, task in tasks:
        pq.insert(priority, task)
        print(f"  Inserted ({priority}, '{task}')")

    print(f"\nPeek: {pq.peek()}")

    print("\nExtracting tasks in priority order:")
    while not pq.is_empty():
        priority, task = pq.extract_min()
        print(f"  ({priority}) {task}")
```

**출력:**
```
Inserting tasks:
  Inserted (3, 'low-priority task')
  Inserted (1, 'urgent task')
  Inserted (2, 'medium task')
  Inserted (5, 'background task')
  Inserted (1, 'another urgent task')

Peek: (1, 'urgent task')

Extracting tasks in priority order:
  (1) urgent task
  (1) another urgent task
  (2) medium task
  (3) low-priority task
  (5) background task
```

---

## 4. 우선순위 큐로 쓰는 파이썬 heapq

파이썬의 `heapq` 모듈은 짝의 리스트에서 도는 최소 힙을 준다. 원소가 짝이면 파이썬이 첫 성분(우선순위)으로 견주므로 우선순위 큐로 쓰기 자연스럽다.

```python
"""
파이썬 heapq 모듈을 우선순위 큐로 쓰기.
"""

import heapq

# === heapq에 바탕한 우선순위 큐 ===

if __name__ == "__main__":
    pq = []

    # 우선순위를 붙여 작업을 넣는다
    heapq.heappush(pq, (3, "write report"))
    heapq.heappush(pq, (1, "fix critical bug"))
    heapq.heappush(pq, (2, "review PR"))
    heapq.heappush(pq, (1, "deploy hotfix"))

    print("Processing tasks:")
    while pq:
        priority, task = heapq.heappop(pq)
        print(f"  Priority {priority}: {task}")
```

**출력:**
```
Processing tasks:
  Priority 1: deploy hotfix
  Priority 1: fix critical bug
  Priority 2: review PR
  Priority 3: write report
```

??? warning "한계: 효율적인 열쇠 낮추기가 없다"
    파이썬의 `heapq`는 원소를 신원으로 $O(1)$에 짚을 길이 없어 열쇠 낮추기를 곧바로 받쳐 주지 않는다. 흔한 우회 방법은 다음과 같다.

    1. **게으른 삭제**: 항목을 못 쓰는 것으로 표시하고 고친 우선순위로 새 항목을 넣는다. 꺼낼 때 못 쓰는 항목은 건너뛴다.
    2. **색인 잇댐**: 값을 힙 색인으로 잇대는 사전을 지키고 자리를 바꿀 때마다 고친다.

---

## 5. 응용

힙으로 구현한 우선순위 큐는 수많은 알고리즘에 나온다.

| 응용 | 큐의 종류 | 핵심 연산 |
|------------|-----------|---------------|
| 데이크스트라 최단 경로 | 최소 우선순위 큐 | 늦춘 꼭짓점의 열쇠 낮추기 |
| 프림의 최소 신장 트리 | 최소 우선순위 큐 | 경계 꼭짓점의 열쇠 낮추기 |
| 허프만 부호화 | 최소 우선순위 큐 | 빈도가 가장 낮은 노드 둘 꺼내기 |
| 사건 기반 모의 실험 | 최소 우선순위 큐 | 시각 순으로 다음 사건 꺼내기 |
| 작업 일정 짜기 | 최대 우선순위 큐 | 우선순위가 가장 높은 작업 꺼내기 |
| 중앙값 유지하기 | 우선순위 큐 둘 | 최대 큐와 최소 큐의 균형 잡기 |

---

## 6. 복잡도 비교

| 구현 | 삽입 | 꺼내기 | 엿보기 | 열쇠 낮추기 |
|---------------|--------|---------|------|-------------|
| 정렬되지 않은 배열 | $O(1)$ | $O(n)$ | $O(n)$ | $O(1)$ |
| 정렬된 배열 | $O(n)$ | $O(1)$ | $O(1)$ | $O(n)$ |
| 이진 힙 | $O(\log n)$ | $O(\log n)$ | $O(1)$ | $O(\log n)$ |
| 피보나치 힙 | 분할 상환 $O(1)$ | 분할 상환 $O(\log n)$ | $O(1)$ | 분할 상환 $O(1)$ |

대부분의 응용에서 이진 힙이 실용적으로 가장 좋은 맞바꿈을 준다. 피보나치 힙은 열쇠 낮추기가 잦은 알고리즘에 더 좋은 이론적 한계를 주지만 상수가 커서 실제로는 좀처럼 쓰이지 않는다.

---

## 연습문제

**연습문제 1.**
우선순위 큐의 힙 성질을 밝히고 최솟값·최댓값 원소가 언제나 뿌리에 있음을 증명하라.

??? success "연습문제 1 풀이"
    힙 성질은 노드마다 열쇠가 자식보다 작거나 같거나(최소 힙) 크거나 같다(최대 힙)는 것이다. 뿌리에서 잎까지의 어떤 경로에서도 추이성이 성립하므로 뿌리가 모든 원소의 최솟값(또는 최댓값)이다.

---

**연습문제 2.**
배열 $[4, 7, 2, 9, 1, 5, 3]$에서 우선순위 큐를 따라가라. 단계마다와 그 결과로 나오는 힙을 보여라.

??? success "연습문제 2 풀이"
    이 쪽의 연산을 주어진 배열에 적용하라. 단계마다 배열과 그것이 나타내는 트리를 보여라. 비교와 자리바꿈을 짚어라.

---

**연습문제 3.**
우선순위 큐의 시간 복잡도를 증명하라. 그 한계는 빡빡한가?

??? success "연습문제 3 풀이"
    이 연산은 뿌리에서 잎까지 또는 잎에서 뿌리까지의 경로를 훑으며 층마다 $O(1)$의 일을 한다. 완전 이진 트리의 높이는 $\lfloor\log_2 n\rfloor$이므로 모두 $O(\log n)$이다. 이 한계는 빡빡하다. 높이 전체를 훑도록 강요하는 입력이 있다. $\square$

---

**연습문제 4.**
$k = 5$이고 어휘가 $n = 32{,}000$인 빔 탐색에서 힙으로 뽑기(원소 $n$개에서 상위 $k$개)와 정렬을 견주어라.

??? success "연습문제 4 풀이"
    정렬은 $O(n\log n) = O(32000 \times 15) \approx 48만$번의 연산이 든다. 힙(크기 $k$인 최소 힙)은 $O(n\log k) = O(32000 \times 2.3) \approx 7만 4천$번이다. 힙이 약 $6.5$배 빠르다. 아니면 `torch.topk`가 GPU에 맞추어 다듬은 부분 정렬 알고리즘을 쓴다.

## 정리하며

이 마당은 추상 자료형、힙으로 하는 구현、구현、우선순위 큐로 쓰는 파이썬 heapq을 차례로 짚었다.

**참고 문헌**

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., and Stein, C. *Introduction to Algorithms* (4th ed.), Chapter 6.5: Priority queues. MIT Press.
