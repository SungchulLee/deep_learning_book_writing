# 피보나치 힙

이항 힙은 모든 우선순위 큐 연산을 $O(\log n)$에 하지만 열쇠 낮추기가 여전히 $O(\log n)$이어서, 최솟값 꺼내기보다 열쇠 낮추기를 훨씬 자주 부르는 그래프 알고리즘에서 병목이 된다. **피보나치 힙**은 게으른 방식을 택해 열쇠 낮추기를 분할 상환 $O(1)$으로 낫게 한다. 구조 불변식을 곧바로 되살리는 대신 뒷정리를 다음 최솟값 꺼내기로 미룬다. 이 게으른 설계로 삽입, 합치기, 최솟값 찾기, 열쇠 낮추기가 분할 상환 $O(1)$이 되고 최솟값 꺼내기는 분할 상환 $O(\log n)$으로 남는다. 이 한계 덕분에 피보나치 힙은 데이크스트라 알고리즘($O(|V|\log|V| + |E|)$)과 프림의 최소 신장 트리 알고리즘에 이론상 최적이다.

## 구조

피보나치 힙은 힙 순서를 지키는 트리의 모음(숲)이지만, 이항 힙과 달리 트리의 모양이나 차수마다의 트리 수에 아무런 제약을 두지 않는다. 핵심 구조 요소는 다음과 같다.

- **뿌리 목록**: 트리 뿌리의 양방향 원형 연결 리스트.
- **최소 포인터**: 열쇠가 가장 작은 뿌리를 가리킨다.
- **노드의 칸**: 노드마다 열쇠, 차수(자식의 수), 부모를 가리키는 포인터, 자식의 양방향 원형 연결 리스트, **표시** 비트를 담는다.

표시 비트가 피보나치 힙 설계의 중심이다. 노드는 다른 노드의 자식이 된 뒤 첫 자식을 잃을 때 표시된다. 표시된 노드가 둘째 자식을 잃으면 부모에게서 잘려 뿌리 목록으로 옮겨진다. 이 **잇단 자르기** 얼개가 어떤 노드도 자식을 너무 많이 잃지 않게 막아 최대 차수를 한계 짓는다.

## 연산

### 삽입

삽입은 노드 하나짜리 트리를 새로 만들어 뿌리 목록에 더한다. 필요하면 최소 포인터를 고친다. 다지기는 하지 않는다. 이것이 "게으른" 철학이다.

**비용**: 최악의 경우와 분할 상환 모두 $O(1)$.

### 최솟값 찾기

최소 포인터가 곧바로 닿게 해 준다.

**비용**: 최악의 경우와 분할 상환 모두 $O(1)$.

### 합치기 (합집합)

두 피보나치 힙을 합치면 (원형 양방향 연결 리스트에서는 일정 시간에) 뿌리 목록을 이어 붙이고 최소 포인터를 고친다.

**비용**: 최악의 경우와 분할 상환 모두 $O(1)$.

### 최솟값 빼기

미뤄 둔 일을 여기서 한다. 알고리즘은 다음과 같다.

1. 뿌리 목록에서 최소 노드를 없앤다.
2. 최소 노드의 모든 자식을 뿌리 목록에 더한다(그리고 부모 포인터를 지운다).
3. **다지기**: 같은 차수의 뿌리가 둘 없을 때까지 같은 차수의 트리를 되풀이해 잇는다. 이항 힙의 합치기 단계와 닮았다.

다지기는 차수를 색인으로 하는 배열을 쓴다. 뿌리마다 같은 차수의 다른 뿌리가 배열에 있으면 둘을 잇는다(큰 뿌리가 작은 뿌리의 자식이 된다). 모든 차수가 서로 다를 때까지 이어 간다.

**비용**: 최악의 경우 $O(n)$, 분할 상환 $O(\log n)$.

### 열쇠 낮추기

노드 $x$의 열쇠를 낮추려면 다음과 같이 한다.

1. 새 열쇠가 $x$의 부모와의 힙 순서를 어기지 않으면 그냥 열쇠를 고치고 필요하면 최소 포인터를 고친다.
2. 힙 순서가 어긋나면 $x$을 부모에게서 **잘라** 뿌리 목록에 더한다.
3. **잇단 자르기**: $x$의 부모 $y$이 이미 표시되어 있으면 $y$도 그 부모에게서 자른다(그리고 $y$의 표시를 뗀다). 표시되지 않은 노드나 뿌리에 닿을 때까지 트리를 거슬러 되풀이한다. 자식을 잃은 첫 표시 없는 조상에 표시를 붙인다.

**비용**: (잇달음의 길이만큼) 최악의 경우 $O(\log n)$, 분할 상환 $O(1)$.

### 삭제

삭제는 열쇠 낮추기 뒤 최솟값 꺼내기로 귀착된다. 열쇠를 $-\infty$으로 두면 그 노드가 최소 자리로 가고, 그것을 꺼낸다.

**비용**: 최악의 경우 $O(n)$, 분할 상환 $O(\log n)$.

## 잇단 자르기 자세히 보기

잇단 자르기 얼개가 피보나치 힙을 이항 힙과 가르는 결정적인 특징이다. 다음 상황을 생각해 보자.

```
Before decrease-key(x):          After cuts:
      a (unmarked)                    a (now marked)
     / | \                           / \
    b  c  d                         b   d
   / \
  x   e (both unmarked)           c → root list (was marked)
                                  x → root list
```

$x$의 열쇠가 부모 $c$의 열쇠보다 낮아지면 $x$이 잘린다. $c$이 이미 표시되어 있었으면 $c$도 잘려 뿌리 목록에 더해진다. 노드 $a$은 자식 $c$을 잃었으므로 표시된다.

!!! warning "잇단 자르기가 왜 필요한가"
    잇단 자르기가 없으면 노드가 자식을 얼마든지 잃을 수 있다. 그러면 $O(\log n)$의 최솟값 꺼내기에 꼭 필요한 차수 한계 $D(n) = O(\log n)$이 깨진다. 잇단 자르기 규칙은 노드마다 제 자신이 잘리기 전에 자식을 많아야 하나 잃게 하여 피보나치 수에 바탕한 크기 보장을 지킨다.

## 복잡도 요약

| 연산 | 최악의 경우 | 분할 상환 |
|-----------|:----------:|:---------:|
| 삽입 | $O(1)$ | $O(1)$ |
| 최솟값 찾기 | $O(1)$ | $O(1)$ |
| 합치기 | $O(1)$ | $O(1)$ |
| 최솟값 꺼내기 | $O(n)$ | $O(\log n)$ |
| 열쇠 낮추기 | $O(\log n)$ | $O(1)$ |
| 삭제 | $O(n)$ | $O(\log n)$ |

## 구현

```python
"""
피보나치 힙 구현.

삽입, 합치기, 최솟값 찾기, 열쇠 낮추기를 분할 상환 O(1)에 받쳐 준다.
최솟값 꺼내기는 분할 상환 O(log n)이다. 모든 연산이 최솟값 꺼내기로
미룬 게으른 다지기를 쓴다.
"""

import math


# === 피보나치 힙 노드 ===

class FibNode:
    """피보나치 힙의 노드.

    속성:
        key: 우선순위 값
        degree: 자식의 수
        mark: 제 자신이 자식이 된 뒤로 자식을 잃었으면 True
        parent: 부모 노드를 가리키는 포인터 (뿌리 목록에 있으면 None)
        child: 자식 하나를 가리키는 포인터 (원형 자식 목록의 머리)
        left, right: 양방향 원형 연결 리스트의 형제
    """

    def __init__(self, key):
        self.key = key
        self.degree = 0
        self.mark = False
        self.parent = None
        self.child = None
        self.left = self
        self.right = self

    def __repr__(self):
        return f"FibNode(key={self.key}, deg={self.degree}, mark={self.mark})"


# === 피보나치 힙 ===

class FibonacciHeap:
    """게으른 다지기를 쓰는 최소 피보나치 힙."""

    def __init__(self):
        self.min_node = None
        self.n = 0

    def _add_to_root_list(self, node):
        """노드를 뿌리 목록에 더한다."""
        node.parent = None
        if self.min_node is None:
            node.left = node
            node.right = node
            self.min_node = node
        else:
            node.left = self.min_node
            node.right = self.min_node.right
            self.min_node.right.left = node
            self.min_node.right = node

    def _remove_from_list(self, node):
        """노드를 양방향 원형 연결 리스트에서 없앤다."""
        node.left.right = node.right
        node.right.left = node.left

    def insert(self, key):
        """힙에 열쇠를 넣는다. O(1)."""
        node = FibNode(key)
        self._add_to_root_list(node)
        if node.key < self.min_node.key:
            self.min_node = node
        self.n += 1
        return node

    def find_min(self):
        """가장 작은 열쇠를 돌려준다. O(1)."""
        if self.min_node is None:
            raise IndexError("find_min from empty heap")
        return self.min_node.key

    def merge(self, other):
        """다른 피보나치 힙을 이 힙에 합친다. O(1)."""
        if other.min_node is None:
            return
        if self.min_node is None:
            self.min_node = other.min_node
            self.n = other.n
            return
        # 뿌리 목록을 이어 붙인다
        self_right = self.min_node.right
        other_left = other.min_node.left
        self.min_node.right = other.min_node
        other.min_node.left = self.min_node
        self_right.left = other_left
        other_left.right = self_right
        # 최솟값을 고친다
        if other.min_node.key < self.min_node.key:
            self.min_node = other.min_node
        self.n += other.n

    def extract_min(self):
        """가장 작은 열쇠를 없애고 돌려준다. 분할 상환 O(log n)."""
        z = self.min_node
        if z is None:
            raise IndexError("extract_min from empty heap")

        # z의 모든 자식을 뿌리 목록에 더한다
        if z.child is not None:
            children = []
            c = z.child
            while True:
                children.append(c)
                c = c.right
                if c is z.child:
                    break
            for c in children:
                self._add_to_root_list(c)

        # 뿌리 목록에서 z를 없앤다
        self._remove_from_list(z)

        if z == z.right:
            # z가 유일한 뿌리였다
            self.min_node = None
        else:
            self.min_node = z.right
            self._consolidate()

        self.n -= 1
        return z.key

    def _consolidate(self):
        """같은 차수의 뿌리가 둘 없도록 트리를 다진다."""
        max_degree = int(math.log(self.n) / math.log(1.618)) + 2
        degree_table = [None] * (max_degree + 1)

        # 모든 뿌리를 모은다
        roots = []
        curr = self.min_node
        while True:
            roots.append(curr)
            curr = curr.right
            if curr is self.min_node:
                break

        for w in roots:
            x = w
            d = x.degree
            while d < len(degree_table) and degree_table[d] is not None:
                y = degree_table[d]
                if x.key > y.key:
                    x, y = y, x
                self._link(y, x)
                degree_table[d] = None
                d += 1
            if d >= len(degree_table):
                degree_table.extend([None] * (d - len(degree_table) + 1))
            degree_table[d] = x

        # degree_table로 뿌리 목록을 다시 세운다
        self.min_node = None
        for node in degree_table:
            if node is not None:
                node.left = node
                node.right = node
                self._add_to_root_list(node)
                if node.key < self.min_node.key:
                    self.min_node = node

    def _link(self, child, parent):
        """child를 parent의 자식으로 만든다."""
        self._remove_from_list(child)
        child.parent = parent
        if parent.child is None:
            parent.child = child
            child.left = child
            child.right = child
        else:
            child.left = parent.child
            child.right = parent.child.right
            parent.child.right.left = child
            parent.child.right = child
        parent.degree += 1
        child.mark = False

    def decrease_key(self, node, new_key):
        """노드의 열쇠를 낮춘다. 분할 상환 O(1)."""
        if new_key > node.key:
            raise ValueError("New key is greater than current key")
        node.key = new_key
        parent = node.parent
        if parent is not None and node.key < parent.key:
            self._cut(node, parent)
            self._cascading_cut(parent)
        if node.key < self.min_node.key:
            self.min_node = node

    def _cut(self, child, parent):
        """child를 parent에게서 잘라 뿌리 목록에 더한다."""
        if child.right == child:
            parent.child = None
        else:
            if parent.child == child:
                parent.child = child.right
            self._remove_from_list(child)
        parent.degree -= 1
        self._add_to_root_list(child)
        child.mark = False

    def _cascading_cut(self, node):
        """트리를 거슬러 잇단 자르기를 한다."""
        parent = node.parent
        if parent is not None:
            if not node.mark:
                node.mark = True
            else:
                self._cut(node, parent)
                self._cascading_cut(parent)

    def is_empty(self):
        """힙이 비었는지 살핀다."""
        return self.min_node is None


# === 시연 ===

if __name__ == "__main__":
    h = FibonacciHeap()
    values = [7, 3, 8, 1, 5, 2, 9, 4, 6]
    nodes = {}

    print("Inserting values:")
    for v in values:
        nodes[v] = h.insert(v)
        print(f"  Inserted {v}, min = {h.find_min()}")

    print(f"\nExtract min: {h.extract_min()}")
    print(f"New min: {h.find_min()}")

    # 열쇠 낮추기 보이기
    print(f"\nDecrease key 9 -> 0:")
    h.decrease_key(nodes[9], 0)
    print(f"New min: {h.find_min()}")

    print(f"\nExtracting all:")
    extracted = []
    while not h.is_empty():
        extracted.append(h.extract_min())
    print(f"Extracted: {extracted}")
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

Extract min: 1
New min: 2

Decrease key 9 -> 0:
New min: 0

Extracting all:
Extracted: [0, 2, 3, 4, 5, 6, 7, 8]
```

## 다른 힙과 견주기

| 연산 | 이진 힙 | 이항 힙 | 피보나치 힙 |
|-----------|:-----------:|:------------:|:--------------:|
| 삽입 | $O(\log n)$ | $O(\log n)$, 분할 상환 $O(1)$ | $O(1)$ |
| 최솟값 찾기 | $O(1)$ | $O(\log n)$ | $O(1)$ |
| 합치기 | $O(n)$ | $O(\log n)$ | $O(1)$ |
| 최솟값 꺼내기 | $O(\log n)$ | $O(\log n)$ | 분할 상환 $O(\log n)$ |
| 열쇠 낮추기 | $O(\log n)$ | $O(\log n)$ | 분할 상환 $O(1)$ |

!!! tip "언제 피보나치 힙을 쓰는가"
    피보나치 힙은 (데이크스트라나 프림처럼) 최솟값 꺼내기에 견주어 열쇠 낮추기를 많이 하는 알고리즘에 이론상 최적이다. 실제로는 상수 배와 포인터의 짐 때문에 어지간한 크기의 입력에서는 더 간단한 힙(이진 힙이나 $d$진 힙)이 빠른 경우가 많다. 피보나치 힙은 아주 큰 그래프이거나 점근적 최적성이 필요할 때 실용적인 이점이 된다.

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., and Stein, C. *Introduction to Algorithms* (4th ed.), Chapter 19: Fibonacci Heaps. MIT Press.
- Fredman, M. L. and Tarjan, R. E. "Fibonacci heaps and their uses in improved network optimization algorithms." *Journal of the ACM*, 34(3):596--615, 1987.


## 연습문제

**연습문제 1.**
피보나치 힙의 힙 성질을 밝히고 최솟값·최댓값 원소가 언제나 뿌리에 있음을 증명하라.

??? success "연습문제 1 풀이"
    힙 성질은 노드마다 열쇠가 자식보다 작거나 같거나(최소 힙) 크거나 같다(최대 힙)는 것이다. 뿌리에서 잎까지의 어떤 경로에서도 추이성이 성립하므로 뿌리가 모든 원소의 최솟값(또는 최댓값)이다.

---

**연습문제 2.**
배열 $[4, 7, 2, 9, 1, 5, 3]$에서 피보나치 힙을 따라가라. 단계마다와 그 결과로 나오는 힙을 보여라.

??? success "연습문제 2 풀이"
    이 쪽의 연산을 주어진 배열에 적용하라. 단계마다 배열과 그것이 나타내는 트리를 보여라. 비교와 자리바꿈을 짚어라.

---

**연습문제 3.**
피보나치 힙의 시간 복잡도를 증명하라. 그 한계는 빡빡한가?

??? success "연습문제 3 풀이"
    이 연산은 뿌리에서 잎까지 또는 잎에서 뿌리까지의 경로를 훑으며 층마다 $O(1)$의 일을 한다. 완전 이진 트리의 높이는 $\lfloor\log_2 n\rfloor$이므로 모두 $O(\log n)$이다. 이 한계는 빡빡하다. 높이 전체를 훑도록 강요하는 입력이 있다. $\square$

---

**연습문제 4.**
$k = 5$이고 어휘가 $n = 32{,}000$인 빔 탐색에서 힙으로 뽑기(원소 $n$개에서 상위 $k$개)와 정렬을 견주어라.

??? success "연습문제 4 풀이"
    정렬은 $O(n\log n) = O(32000 \times 15) \approx 48만$번의 연산이 든다. 힙(크기 $k$인 최소 힙)은 $O(n\log k) = O(32000 \times 2.3) \approx 7만 4천$번이다. 힙이 약 $6.5$배 빠르다. 아니면 `torch.topk`가 GPU에 맞추어 다듬은 부분 정렬 알고리즘을 쓴다.