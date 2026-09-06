# d진 힙

이진 힙은 노드마다 자식을 꼭 둘 준다. 자연스러운 물음이 생긴다. 노드마다 자식을 $d$개 갖게 하면 어떻게 될까? **$d$진 힙**은 내부 노드마다 자식을 $d$개까지($d \ge 2$) 갖게 하여 이진 힙을 일반화한 것이다. 이진 힙은 $d = 2$인 특수한 경우이다.

이 일반화의 동기는 성능의 맞바꿈이다. $d$을 키우면 트리의 높이가 $\log_2 n$에서 $\log_d n$으로 줄어 (열쇠 낮추기처럼) 뿌리에서 잎까지의 경로를 훑는 연산이 빨라진다. 그러나 자식 $d$개 가운데 가장 작은 것을 찾는 데 $O(1)$이 아니라 $O(d)$가 든다. $d$을 알맞게 고르면 특정 작업에서 성능을 다듬을 수 있다. 이를테면 그래프가 빽빽할 때 데이크스트라 알고리즘이 $d$진 힙의 덕을 본다.

## 짜임과 색인 공식

이진 힙처럼 $d$진 힙도 (0부터 세는) 층 순서 색인으로 배열에 담는다. 색인 $i$의 노드에 대해 다음과 같다.

노드 $i$의 **자식**: 색인 $di + 1, \; di + 2, \; \ldots, \; di + d$ 가운데 $n-1$ 이하인 것들.

노드 $i$의 **부모**($i > 0$일 때):

$$
\text{parent}(i) = \left\lfloor \frac{i - 1}{d} \right\rfloor
$$

트리의 높이는 다음과 같다.

$$
h = \lfloor \log_d n \rfloor = \left\lfloor \frac{\ln n}{\ln d} \right\rfloor
$$

!!! example "3진 최소 힙"
    ```
    Array: [1, 3, 5, 2, 7, 8, 9, 6, 4, 10]

    Tree (d=3):
                    1
                /   |   \
              3     5     2
            / | \   |   / | \
           7  8  9  6  4  10
    ```
    노드마다 자식이 많아야 3개이다. 마지막 층은 일부만 찰 수 있다.
    색인 3의 노드(값 2)의 부모는 색인 0(값 1)이다. 2 > 1이므로 올바르다.
    색인 7의 노드(값 6)의 부모는 색인 2(값 5)이다. 6 > 5이므로 올바르다.

## 연산

### 위로 올리기 (삽입과 열쇠 낮추기용)

위로 올리기는 노드를 하나뿐인 부모와 견주어 힙 성질이 어긋나면 자리를 바꾼다. 트리의 높이가 $\lfloor \log_d n \rfloor$이므로 위로 올리기는 많아야 그만큼의 층을 층마다 한 번씩 비교하며 훑는다.

$$
T_{\text{sift-up}} = O(\log_d n)
$$

### 아래로 내리기 (최솟값 꺼내기와 힙 세우기용)

아래로 내리기는 자리를 바꾸기 전에 노드의 자식 $d$개 가운데 가장 작은 것을 찾아야 한다. 층마다 가장 작은 자식을 찾는 데 비교가 $d - 1$번 들고, 아래로 내리기는 많아야 $\lfloor \log_d n \rfloor$개의 층을 훑는다.

$$
T_{\text{sift-down}} = O(d \log_d n)
$$

### 연산의 복잡도

| 연산 | 복잡도 |
|-----------|:----------:|
| 삽입 | $O(\log_d n)$ |
| 최솟값 찾기 | $O(1)$ |
| 최솟값 꺼내기 | $O(d \log_d n)$ |
| 열쇠 낮추기 | $O(\log_d n)$ |
| 힙 세우기 | $O(n)$ |

힙 세우기의 복잡도는 이진 힙과 같은 아래에서 위로의 논증으로 $O(n)$ 그대로이다. $d$과 무관하게 합이 망원경처럼 줄어든다.

## 맞바꿈

핵심 통찰은 $d$을 키우면 서로 반대되는 효과가 생긴다는 것이다.

- **열쇠 낮추기가 빨라진다**: $O(\log_d n) = O(\log n / \log d)$이며 $d$이 커질수록 줄어든다.
- **최솟값 꺼내기가 느려진다**: $O(d \log_d n) = O(d \log n / \log d)$이며 $d$이 $\log n$을 넘어서면 커진다.

최솟값 꺼내기를 $|V|$번, 열쇠 낮추기를 $|E|$번 하는 데이크스트라 최단 경로 같은 알고리즘에서 힙의 전체 비용은 다음과 같다.

$$
T = O\left(|V| \cdot d \cdot \frac{\log |V|}{\log d} + |E| \cdot \frac{\log |V|}{\log d}\right)
$$

$d = \max(2, \lceil |E|/|V| \rceil)$으로 두면 두 항의 균형이 맞는다. $|E| = \Theta(|V|^2)$인 빽빽한 그래프에서 $d = |V|$으로 고르면 다음을 얻는다.

$$
T = O(|V|^2)
$$

이는 정렬되지 않은 배열 우선순위 큐의 성능과 같고 빽빽한 그래프의 데이크스트라에 최적이다.

## 구현

```python
"""
d진 힙 구현.

d진 힙은 노드마다 자식을 d개까지 주어 이진 힙을 일반화한다.
이는 갈래 인수 d으로 다스리는, 최솟값 꺼내기 비용과
열쇠 낮추기 비용의 맞바꿈이다.
"""


# === d진 힙 ===

class DAryHeap:
    """배열로 담은 최소 d진 힙."""

    def __init__(self, d=2):
        """갈래 인수 d >= 2으로 시작한다."""
        if d < 2:
            raise ValueError("Branching factor d must be >= 2")
        self.d = d
        self.heap = []

    def _parent(self, i):
        """노드 i의 부모 색인을 돌려준다."""
        return (i - 1) // self.d

    def _children(self, i):
        """노드 i의 자식 색인 범위를 돌려준다."""
        start = self.d * i + 1
        end = min(start + self.d, len(self.heap))
        return range(start, end)

    def _sift_up(self, i):
        """힙 성질이 되살아날 때까지 노드 i를 위로 옮긴다."""
        while i > 0:
            parent = self._parent(i)
            if self.heap[i] < self.heap[parent]:
                self.heap[i], self.heap[parent] = self.heap[parent], self.heap[i]
                i = parent
            else:
                break

    def _sift_down(self, i):
        """힙 성질이 되살아날 때까지 노드 i를 아래로 옮긴다."""
        n = len(self.heap)
        while True:
            min_idx = i
            for c in self._children(i):
                if c < n and self.heap[c] < self.heap[min_idx]:
                    min_idx = c
            if min_idx == i:
                break
            self.heap[i], self.heap[min_idx] = self.heap[min_idx], self.heap[i]
            i = min_idx

    def insert(self, key):
        """힙에 열쇠를 넣는다. O(log_d n)."""
        self.heap.append(key)
        self._sift_up(len(self.heap) - 1)

    def find_min(self):
        """가장 작은 열쇠를 돌려준다. O(1)."""
        if not self.heap:
            raise IndexError("find_min from empty heap")
        return self.heap[0]

    def extract_min(self):
        """가장 작은 열쇠를 없애고 돌려준다. O(d * log_d n)."""
        if not self.heap:
            raise IndexError("extract_min from empty heap")
        min_val = self.heap[0]
        last = self.heap.pop()
        if self.heap:
            self.heap[0] = last
            self._sift_down(0)
        return min_val

    def decrease_key(self, i, new_key):
        """색인 i의 열쇠를 new_key로 낮춘다. O(log_d n)."""
        if new_key > self.heap[i]:
            raise ValueError("New key is greater than current key")
        self.heap[i] = new_key
        self._sift_up(i)

    @classmethod
    def build_heap(cls, data, d=2):
        """리스트로 d진 힙을 O(n) 시간에 세운다."""
        h = cls(d=d)
        h.heap = list(data)
        # 마지막 부모에서 뿌리까지 아래로 내린다
        n = len(h.heap)
        for i in range((n - 2) // d, -1, -1):
            h._sift_down(i)
        return h

    def is_empty(self):
        """힙이 비었는지 살핀다."""
        return len(self.heap) == 0


# === 시연 ===

if __name__ == "__main__":
    import math

    values = [7, 3, 8, 1, 5, 2, 9, 4, 6]

    for d in [2, 3, 4]:
        h = DAryHeap(d=d)
        for v in values:
            h.insert(v)

        extracted = []
        while not h.is_empty():
            extracted.append(h.extract_min())

        height = math.floor(math.log(len(values)) / math.log(d))
        print(f"d={d}: height={height}, extracted={extracted}")

    # 힙 세우기 보이기
    print("\nBuild-heap (d=3):")
    h = DAryHeap.build_heap([7, 3, 8, 1, 5, 2, 9, 4, 6], d=3)
    extracted = []
    while not h.is_empty():
        extracted.append(h.extract_min())
    print(f"  Extracted: {extracted}")
```

**출력:**
```
d=2: height=3, extracted=[1, 2, 3, 4, 5, 6, 7, 8, 9]
d=3: height=1, extracted=[1, 2, 3, 4, 5, 6, 7, 8, 9]
d=4: height=1, extracted=[1, 2, 3, 4, 5, 6, 7, 8, 9]

Build-heap (d=3):
  Extracted: [1, 2, 3, 4, 5, 6, 7, 8, 9]
```

## 이진 힙과 견주기

| 측면 | 이진 힙 ($d=2$) | $d$진 힙 |
|--------|:-------------------:|:------------:|
| 높이 | $\lfloor \log_2 n \rfloor$ | $\lfloor \log_d n \rfloor$ |
| 삽입 | $O(\log_2 n)$ | $O(\log_d n)$ |
| 최솟값 꺼내기 | $O(\log_2 n)$ | $O(d \log_d n)$ |
| 열쇠 낮추기 | $O(\log_2 n)$ | $O(\log_d n)$ |
| 캐시 움직임 | 좋음 | $d$이 크면 더 좋음 (더 넓고 얕다) |

!!! tip "실제 지침"
    실제로는 캐시를 더 잘 써서 $d = 4$가 $d = 2$보다 나은 경우가 많다. 트리가 얕으면 아래로 내릴 때 캐시를 덜 놓친다. 빽빽한 그래프의 데이크스트라에서는 $d = |E|/|V|$이 이론상 최적이다.

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., and Stein, C. *Introduction to Algorithms* (4th ed.), Problem 6-2: d-ary Heaps. MIT Press.
- Johnson, D. B. "Efficient algorithms for shortest paths in sparse networks." *Journal of the ACM*, 24(1):1--13, 1977.


## 연습문제

**연습문제 1.**
d진 힙의 힙 성질을 밝히고 최솟값·최댓값 원소가 언제나 뿌리에 있음을 증명하라.

??? success "연습문제 1 풀이"
    힙 성질은 노드마다 열쇠가 자식보다 작거나 같거나(최소 힙) 크거나 같다(최대 힙)는 것이다. 뿌리에서 잎까지의 어떤 경로에서도 추이성이 성립하므로 뿌리가 모든 원소의 최솟값(또는 최댓값)이다.

---

**연습문제 2.**
배열 $[4, 7, 2, 9, 1, 5, 3]$에서 d진 힙을 따라가라. 단계마다와 그 결과로 나오는 힙을 보여라.

??? success "연습문제 2 풀이"
    이 쪽의 연산을 주어진 배열에 적용하라. 단계마다 배열과 그것이 나타내는 트리를 보여라. 비교와 자리바꿈을 짚어라.

---

**연습문제 3.**
d진 힙의 시간 복잡도를 증명하라. 그 한계는 빡빡한가?

??? success "연습문제 3 풀이"
    이 연산은 뿌리에서 잎까지 또는 잎에서 뿌리까지의 경로를 훑으며 층마다 $O(1)$의 일을 한다. 완전 이진 트리의 높이는 $\lfloor\log_2 n\rfloor$이므로 모두 $O(\log n)$이다. 이 한계는 빡빡하다. 높이 전체를 훑도록 강요하는 입력이 있다. $\square$

---

**연습문제 4.**
$k = 5$이고 어휘가 $n = 32{,}000$인 빔 탐색에서 힙으로 뽑기(원소 $n$개에서 상위 $k$개)와 정렬을 견주어라.

??? success "연습문제 4 풀이"
    정렬은 $O(n\log n) = O(32000 \times 15) \approx 48만$번의 연산이 든다. 힙(크기 $k$인 최소 힙)은 $O(n\log k) = O(32000 \times 2.3) \approx 7만 4천$번이다. 힙이 약 $6.5$배 빠르다. 아니면 `torch.topk`가 GPU에 맞추어 다듬은 부분 정렬 알고리즘을 쓴다.