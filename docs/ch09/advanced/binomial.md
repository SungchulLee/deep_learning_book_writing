# 이항 힙

이진 힙은 삽입과 최솟값 꺼내기를 $O(\log n)$에 하지만, 두 힙을 합치려면 힙을 맨바닥부터 다시 세워야 하므로 $O(n)$ 시간이 든다. **이항 힙**은 힙을 이항 트리의 모음으로 나타내어 이를 풀고, 합치기(따라서 삽입도)를 최악의 경우 $O(\log n)$, 삽입은 분할 상환 $O(1)$에 해낸다. 그래서 서로 다른 프로세서의 우선순위 큐를 합치는 병렬 알고리즘이나, 효율적인 열쇠 낮추기가 필요한 프림의 최소 신장 트리와 데이크스트라의 최단 경로 같은 그래프 알고리즘처럼 합치기가 잦을 때 이항 힙이 자연스러운 선택이 된다. 이항 힙은 또한 여기서 들여온 구조 제약을 느슨하게 하여 열쇠 낮추기를 분할 상환 $O(1)$으로 더 낫게 한 피보나치 힙으로 가는 개념적 디딤돌이기도 하다.

## 이항 트리

**이항 트리** $B_k$은 재귀적으로 정의된다.

- $B_0$은 노드 하나이다.
- $B_k$은 $B_{k-1}$ 둘을 이어 만든다. 하나가 다른 하나의 뿌리의 가장 왼쪽 자식이 된다.

### 이항 트리의 성질

이항 트리 $B_k$은 다음 성질을 가진다.

1. **높이**: $k$
2. **노드의 수**: $2^k$
3. **뿌리의 차수**: $k$
4. **뿌리의 자식**: 뿌리는 (어떤 순서로) 자식 $B_{k-1}, B_{k-2}, \ldots, B_0$을 가진다
5. **깊이 $d$의 노드 수**: $\binom{k}{d}$ (그래서 *이항* 트리라 부른다)

!!! example "처음 몇 개의 이항 트리"
    ```
    B_0:  o        B_1:  o        B_2:    o          B_3:        o
                          |              / |                    / | \
                          o            o   o                 o   o   o
                                       |                   / |   |
                                       o                 o   o   o
                                                         |
                                                         o
    Nodes:  1             2              4                    8
    Height: 0             1              2                    3
    ```

## 이항 힙의 짜임

**이항 힙**은 다음 두 성질을 만족하는 이항 트리의 모음(숲)이다.

1. **힙 순서**: 트리마다 최소 힙(또는 최대 힙) 성질을 만족한다. 노드마다 열쇠가 자식의 열쇠 이하이다.
2. **유일성**: 차수 $k$마다 모음 안에 이항 트리 $B_k$이 많아야 하나 있다.

유일성 성질은 이진 표현과의 직접적인 대응을 낳는다. $B_k$마다 노드가 꼭 $2^k$개이고 $B_k$이 많아야 하나 있으므로, 노드가 $n$개인 이항 힙은 $n$의 이진 표현에서 $k$번째 비트가 켜져 있을 때만 $B_k$을 담는다. 이를테면 $n = 13 = 1101_2$은 트리 $B_3, B_2, B_0$을 담아 노드가 모두 $8 + 4 + 1 = 13$개이다.

가장 작은 원소는 언제나 숲의 어느 트리의 뿌리이다. 노드가 $n$개인 힙에는 (그 수가 $n$의 비트 수로 한계 지어져) 트리가 많아야 $\lfloor \log_2 n \rfloor + 1$개이므로, 최솟값을 찾으려면 많아야 뿌리 $\lfloor \log_2 n \rfloor + 1$개를 살피면 된다.

## 합치기 연산

합치기는 이항 힙의 중심 연산이며 다른 모든 연산이 이것으로 귀착된다. 알고리즘은 이진 덧셈과 닮았다. 트리의 차수를 작은 쪽에서 큰 쪽으로 훑으며, 이진 산술에서 올림을 하듯 같은 차수의 트리를 합친다.

### 알고리즘

```
BINOMIAL-HEAP-MERGE(H1, H2):
    carry = None
    result = empty heap
    for k = 0, 1, 2, ...:
        trees at order k: t1 from H1, t2 from H2, tc from carry
        count = number of non-None trees among {t1, t2, tc}

        if count == 0: continue
        if count == 1: add the single tree to result at order k
        if count == 2: link the two trees to form B_{k+1}, set as carry
        if count == 3: add one tree to result at order k,
                       link the other two as carry
    return result
```

차수가 $k$인 두 트리를 **잇기**: 뿌리를 견준다. 뿌리가 큰 쪽이 다른 뿌리의 가장 왼쪽 자식이 되어 차수 $k+1$의 트리가 나온다. 이는 힙 순서 성질을 지킨다.

### 복잡도

합치기는 많아야 $O(\log n)$개의 차수를 훑으며 차수마다 일정한 일을 한다. 따라서 다음과 같다.

$$
T_{\text{merge}} = O(\log n)
$$

## 합치기로 하는 다른 연산

합치기가 있으면 나머지 우선순위 큐 연산이 자연스레 따라 나온다. 이항 힙의 우아함은 합치기를 중심에 둔 설계에 있다. 연산마다 합치기를 곧바로 부르거나 $O(\log n)$의 일을 한 뒤 합치기를 한다. 이 하나로 모으는 방식이 구현과 복잡도 분석을 모두 간단하게 한다.

| 연산 | 합치기를 어떻게 쓰는가 | 시간 |
|-----------|------------------|------|
| 삽입 | 노드 하나짜리 힙 $B_0$을 만들어 기존 힙과 합친다 | 최악 $O(\log n)$, 분할 상환 $O(1)$ |
| 최솟값 찾기 | 모든 트리의 뿌리를 살핀다 | $O(\log n)$ |
| 최솟값 꺼내기 | 최소 뿌리를 없애고 그 자식으로 새 힙을 만들어 합친다 | $O(\log n)$ |
| 열쇠 낮추기 | 이항 트리 안에서 위로 올린다 | $O(\log n)$ |
| 삭제 | 열쇠를 $-\infty$으로 낮춘 뒤 최솟값을 꺼낸다 | $O(\log n)$ |

!!! tip "분할 상환 O(1) 삽입"
    삽입 한 번이 (이진 올림처럼) $O(\log n)$번의 트리 합치기로 이어질 수 있지만, 처음에 빈 이항 힙에 삽입을 $n$번 하면 잇기 연산이 모두 많아야 $2n$번이다. 이진 계수기 분석과 닮은 논증으로 삽입당 분할 상환 비용은 $O(1)$이다.

## 구현

```python
"""
이항 힙 구현.

이항 힙은 합치기를 O(log n) 시간에 받쳐 주는 이항 트리의 숲이다.
모든 연산이 합치기로 귀착된다.
"""


# === 이항 트리 노드 ===

class BinomialNode:
    """이항 트리의 노드.

    노드마다 열쇠와 가장 왼쪽 자식을 가리키는 포인터,
    그리고 (숲의 연결 리스트를 위한) 다음 형제를 가리키는 포인터를 담는다.
    """

    def __init__(self, key):
        self.key = key
        self.order = 0          # 여기를 뿌리로 하는 이항 트리의 차수
        self.child = None       # 가장 왼쪽 자식
        self.sibling = None     # 숲에서의 다음 형제

    def __repr__(self):
        return f"BinomialNode(key={self.key}, order={self.order})"


# === 이항 힙 ===

class BinomialHeap:
    """이항 트리의 연결 리스트로 구현한 최소 이항 힙."""

    def __init__(self):
        self.head = None  # 트리 차수 순으로 정렬된 뿌리의 연결 리스트

    def _link(self, t1, t2):
        """같은 차수의 트리 둘을 잇는다.

        뿌리를 견주어 열쇠가 큰 뿌리를 열쇠가 작은 뿌리의 자식으로 만든다.
        (필요하면) 맞바꾼 뒤 t1이 언제나 이긴 쪽(작은 열쇠)이고
        t2가 그 자식이 된다.
        """
        if t1.key > t2.key:
            t1, t2 = t2, t1
        t2.sibling = t1.child
        t1.child = t2
        t1.order += 1
        return t1

    def merge(self, other):
        """다른 이항 힙을 이 힙에 합친다. O(log n)."""
        # 차수 순으로 정렬된 두 연결 리스트를 합친다
        merged = self._merge_lists(self.head, other.head)

        if merged is None:
            self.head = None
            return

        # 훑으며 같은 차수의 트리를 모은다
        prev = None
        curr = merged
        nxt = curr.sibling

        while nxt is not None:
            if curr.order != nxt.order or \
               (nxt.sibling is not None and nxt.sibling.order == curr.order):
                # 차수가 다르거나 같은 차수의 트리가 셋이면 다음으로 나아간다
                prev = curr
                curr = nxt
            else:
                # 같은 차수의 트리가 둘이면 잇는다
                linked = self._link(curr, nxt)
                linked.sibling = nxt.sibling
                if prev is None:
                    merged = linked
                else:
                    prev.sibling = linked
                curr = linked
            nxt = curr.sibling

        self.head = merged

    def _merge_lists(self, h1, h2):
        """차수로 정렬된 두 뿌리 목록을 정렬된 하나로 합친다."""
        if h1 is None:
            return h2
        if h2 is None:
            return h1

        if h1.order <= h2.order:
            head = h1
            h1 = h1.sibling
        else:
            head = h2
            h2 = h2.sibling

        tail = head
        while h1 is not None and h2 is not None:
            if h1.order <= h2.order:
                tail.sibling = h1
                h1 = h1.sibling
            else:
                tail.sibling = h2
                h2 = h2.sibling
            tail = tail.sibling

        tail.sibling = h1 if h1 is not None else h2
        return head

    def insert(self, key):
        """노드 하나짜리 힙을 만들어 합쳐서 열쇠를 넣는다. O(log n)."""
        node = BinomialNode(key)
        temp = BinomialHeap()
        temp.head = node
        self.merge(temp)

    def find_min(self):
        """가장 작은 열쇠를 돌려준다. O(log n)."""
        if self.head is None:
            raise IndexError("find_min from empty heap")
        min_key = self.head.key
        curr = self.head.sibling
        while curr is not None:
            if curr.key < min_key:
                min_key = curr.key
            curr = curr.sibling
        return min_key

    def extract_min(self):
        """가장 작은 열쇠를 없애고 돌려준다. O(log n)."""
        if self.head is None:
            raise IndexError("extract_min from empty heap")

        # 최소 뿌리와 그 앞의 것을 찾는다
        min_node = self.head
        min_prev = None
        prev = None
        curr = self.head
        while curr is not None:
            if curr.key < min_node.key:
                min_node = curr
                min_prev = prev
            prev = curr
            curr = curr.sibling

        # 뿌리 목록에서 min_node를 없앤다
        if min_prev is None:
            self.head = min_node.sibling
        else:
            min_prev.sibling = min_node.sibling

        # min_node의 자식을 뒤집어 새 힙을 만든다
        child_heap = BinomialHeap()
        child = min_node.child
        prev_child = None
        while child is not None:
            nxt = child.sibling
            child.sibling = prev_child
            prev_child = child
            child = nxt
        child_heap.head = prev_child

        # 자식을 도로 합친다
        self.merge(child_heap)
        return min_node.key

    def is_empty(self):
        """힙이 비었는지 살핀다."""
        return self.head is None

    def _collect_keys(self):
        """힙의 모든 열쇠를 모은다 (시험용)."""
        keys = []
        self._collect_from_node(self.head, keys)
        return keys

    def _collect_from_node(self, node, keys):
        while node is not None:
            keys.append(node.key)
            self._collect_from_node(node.child, keys)
            node = node.sibling


# === 시연 ===

if __name__ == "__main__":
    h = BinomialHeap()
    values = [7, 3, 8, 1, 5, 2, 9, 4, 6]

    print("Inserting values:")
    for v in values:
        h.insert(v)
        print(f"  Inserted {v}, min = {h.find_min()}")

    print(f"\nExtracting in order:")
    extracted = []
    while not h.is_empty():
        val = h.extract_min()
        extracted.append(val)
        print(f"  Extracted {val}")

    print(f"\nExtracted sequence: {extracted}")
    assert extracted == sorted(values), "Extraction order incorrect!"
    print("Correctness verified.")

    # 합치기를 보인다
    print("\n--- Merge Demo ---")
    h1 = BinomialHeap()
    for v in [5, 3, 7]:
        h1.insert(v)

    h2 = BinomialHeap()
    for v in [2, 8, 1]:
        h2.insert(v)

    print(f"H1 min: {h1.find_min()}, H2 min: {h2.find_min()}")
    h1.merge(h2)
    print(f"After merge, min: {h1.find_min()}")

    merged = []
    while not h1.is_empty():
        merged.append(h1.extract_min())
    print(f"Merged extraction: {merged}")
```

**출력:**
```
Inserting values:
  Inserted 7, min = 7
  Inserted 3, min = 3
  Inserted 8, min = 3
  Inserted 1, min = 1
  Inserted 5, min = 1
  Inserted 2, min = 1
  Inserted 9, min = 1
  Inserted 4, min = 1
  Inserted 6, min = 1

Extracting in order:
  Extracted 1
  Extracted 2
  Extracted 3
  Extracted 4
  Extracted 5
  Extracted 6
  Extracted 7
  Extracted 8
  Extracted 9

Extracted sequence: [1, 2, 3, 4, 5, 6, 7, 8, 9]
Correctness verified.

--- Merge Demo ---
H1 min: 3, H2 min: 1
After merge, min: 1
Merged extraction: [1, 2, 3, 5, 7, 8]
```

## 복잡도 요약

| 연산 | 이진 힙 | 이항 힙 (최악) | 이항 힙 (분할 상환) |
|-----------|:-----------:|:--------------------------:|:-------------------------:|
| 삽입 | $O(\log n)$ | $O(\log n)$ | $O(1)$ |
| 최솟값 찾기 | $O(1)$ | $O(\log n)$ | $O(\log n)$ |
| 최솟값 꺼내기 | $O(\log n)$ | $O(\log n)$ | $O(\log n)$ |
| 합치기 | $O(n)$ | $O(\log n)$ | $O(\log n)$ |
| 열쇠 낮추기 | $O(\log n)$ | $O(\log n)$ | $O(\log n)$ |

이진 힙에 견준 이항 힙의 핵심 이점은 $O(\log n)$의 합치기이다. 분할 상환 $O(1)$의 삽입은 이진 계수기 논증에서 나온다. 이진 계수기를 하나 올릴 때 분할 상환으로 $O(1)$개의 비트가 뒤집히듯, 이항 힙에 넣을 때 분할 상환으로 $O(1)$개의 트리가 이어진다. 이진 힙에 견준 대가는 구현이 조금 더 까다롭고 최솟값 찾기가 $O(1)$이 아니라 $O(\log n)$이라는 점이다.

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., and Stein, C. *Introduction to Algorithms* (4th ed.), Chapter 19: Binomial Heaps. MIT Press.


## 연습문제

**연습문제 1.**
이항 힙의 힙 성질을 밝히고 최솟값·최댓값 원소가 언제나 뿌리에 있음을 증명하라.

??? success "연습문제 1 풀이"
    힙 성질은 노드마다 열쇠가 자식보다 작거나 같거나(최소 힙) 크거나 같다(최대 힙)는 것이다. 뿌리에서 잎까지의 어떤 경로에서도 추이성이 성립하므로 뿌리가 모든 원소의 최솟값(또는 최댓값)이다.

---

**연습문제 2.**
배열 $[4, 7, 2, 9, 1, 5, 3]$에서 이항 힙을 따라가라. 단계마다와 그 결과로 나오는 힙을 보여라.

??? success "연습문제 2 풀이"
    이 쪽의 연산을 주어진 배열에 적용하라. 단계마다 배열과 그것이 나타내는 트리를 보여라. 비교와 자리바꿈을 짚어라.

---

**연습문제 3.**
이항 힙의 시간 복잡도를 증명하라. 그 한계는 빡빡한가?

??? success "연습문제 3 풀이"
    이 연산은 뿌리에서 잎까지 또는 잎에서 뿌리까지의 경로를 훑으며 층마다 $O(1)$의 일을 한다. 완전 이진 트리의 높이는 $\lfloor\log_2 n\rfloor$이므로 모두 $O(\log n)$이다. 이 한계는 빡빡하다. 높이 전체를 훑도록 강요하는 입력이 있다. $\square$

---

**연습문제 4.**
$k = 5$이고 어휘가 $n = 32{,}000$인 빔 탐색에서 힙으로 뽑기(원소 $n$개에서 상위 $k$개)와 정렬을 견주어라.

??? success "연습문제 4 풀이"
    정렬은 $O(n\log n) = O(32000 \times 15) \approx 48만$번의 연산이 든다. 힙(크기 $k$인 최소 힙)은 $O(n\log k) = O(32000 \times 2.3) \approx 7만 4천$번이다. 힙이 약 $6.5$배 빠르다. 아니면 `torch.topk`가 GPU에 맞추어 다듬은 부분 정렬 알고리즘을 쓴다.