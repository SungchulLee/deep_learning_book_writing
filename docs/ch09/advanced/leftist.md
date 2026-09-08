# 좌편향 힙

이진 힙은 배열에 담기고 삽입과 최솟값 꺼내기를 효율적으로 하지만, 두 이진 힙을 합치는 데는 $O(n)$ 시간이 든다. **좌편향 힙**은 포인터에 바탕한, 힙 순서를 지키는 이진 트리로 합치기를 $O(\log n)$ 시간에 한다. 핵심 생각은 구조적인 치우침을 지키는 것이다. 트리의 오른쪽 등뼈가 언제나 짧다. 합치기 연산이 오른쪽 등뼈를 따라가므로 이것이 로그 시간의 합치기 비용을 보장한다. 다른 모든 연산(삽입, 최솟값 꺼내기, 삭제)이 합치기로 귀착되어 좌편향 힙은 합칠 수 있는 우선순위 큐 구현 가운데 가장 간단한 축에 든다.

---

## 1. s값 (널 경로 길이)

좌편향 성질은 노드마다의 **s값**(**널 경로 길이** 또는 **계수**라고도 한다)으로 정의한다.

**정의**: 노드 $x$의 s값 $s(x)$은 $x$에서 널(없는) 자손까지의 가장 짧은 경로의 길이이다.

$$
s(x) = \begin{cases} 0 & \text{if } x = \text{null} \\ 1 + \min(s(\text{left}(x)),\; s(\text{right}(x))) & \text{otherwise} \end{cases}
$$

!!! example "트리 안의 s값"
    ```
           4 (s=2)
          / \
      8 (s=2) 6 (s=1)
       / \      |
    10(s=1) 12(s=1)  7(s=1)
    ```
    노드 6은 자식이 하나(s=1인 7)이므로 $s(6) = \min(1, 0) + 1 = 1$이다. 노드 8은 자식이 둘(10과 12, 둘 다 s=1)이므로 $s(8) = \min(1,1)+1 = 2$이다. 노드 4는 $s(4) = \min(s(8), s(6)) + 1 = \min(2,1) + 1 = 2$이다. 노드마다 왼쪽 자식의 s값이 오른쪽 자식의 s값 이상이므로 좌편향 성질이 성립한다.

---

## 2. 좌편향 성질

힙 순서를 지키는 이진 트리가 모든 내부 노드 $x$에 대해 다음을 만족하면 **좌편향**이다.

$$
s(\text{left}(x)) \ge s(\text{right}(x))
$$

이 성질은 트리를 치우치게 하여 **오른쪽 등뼈**(뿌리에서 오른쪽 자식만 따라가는 경로)가 언제나 뿌리에서 널까지의 가장 짧은 경로가 되게 한다. 따라서 오른쪽 등뼈의 길이는 많아야 $\lfloor \log_2(n+1) \rfloor$이다.

!!! tip "오른쪽 등뼈 길이의 한계"
    오른쪽 등뼈의 길이가 $r$이면 그 꼭대기를 뿌리로 하는 부분 트리에는 (s값이 층마다 적어도 1씩 늘므로) 노드가 적어도 $2^r - 1$개 있다. 따라서 $n \ge 2^r - 1$이고 $r \le \lfloor \log_2(n+1) \rfloor$을 얻는다. 이 한계가 합치기를 효율적으로 만든다.

---

## 3. 합치기 연산

합치기가 근본 되는 연산이다. 좌편향 힙 $H_1$과 $H_2$이 주어졌을 때 다음과 같이 한다.

1. 뿌리를 견준다. 작은 쪽이 합친 힙의 뿌리가 된다.
2. 큰 뿌리의 힙을 이긴 쪽의 **오른쪽** 부분 트리와 재귀적으로 합친다.
3. 재귀 호출 뒤에 좌편향 성질이 어긋나면(오른쪽 자식의 s값이 왼쪽 자식보다 크면) 왼쪽과 오른쪽 자식을 맞바꾼다.
4. s값을 고친다: $s(\text{root}) = s(\text{right}) + 1$.

재귀 호출마다 두 힙 가운데 하나의 오른쪽 등뼈를 따라 내려가므로 전체 재귀 호출 횟수는 많아야 두 오른쪽 등뼈 길이의 합이고 다음을 얻는다.

$$
T_{\text{merge}} = O(\log n_1 + \log n_2) = O(\log n)
$$

여기서 $n = n_1 + n_2$이다.

---

## 4. 다른 연산

모든 연산이 합치기로 귀착된다.

| 연산 | 무엇으로 귀착되는가 | 시간 |
|-----------|----------|:----:|
| 삽입 | 노드 하나짜리 힙을 만들어 합친다 | $O(\log n)$ |
| 최솟값 찾기 | 뿌리의 열쇠를 돌려준다 | $O(1)$ |
| 최솟값 꺼내기 | 뿌리의 왼쪽 자식과 오른쪽 자식을 합친다 | $O(\log n)$ |
| 최솟값 지우기 | 최솟값 꺼내기와 같다 | $O(\log n)$ |

---

## 5. 구현

```python
"""
좌편향 힙 구현.

좌편향 힙은 왼쪽 자식의 s값(널 경로 길이)이 언제나
오른쪽 자식 이상인, 힙 순서를 지키는 이진 트리이다.
모든 연산이 O(log n)에 도는 합치기로 귀착된다.
"""

# === 좌편향 힙 노드 ===

class LeftistNode:
    """좌편향 힙의 노드.

    속성:
        key: 우선순위 값
        s: s값 (널 경로 길이)
        left: 왼쪽 자식
        right: 오른쪽 자식
    """

    def __init__(self, key):
        self.key = key
        self.s = 1  # 노드 하나의 s값은 1이다
        self.left = None
        self.right = None

    def __repr__(self):
        return f"LeftistNode(key={self.key}, s={self.s})"

# === 좌편향 힙 ===

class LeftistHeap:
    """모든 연산이 합치기로 귀착되는 최소 좌편향 힙."""

    def __init__(self):
        self.root = None
        self.size = 0

    @staticmethod
    def _s_value(node):
        """노드의 s값을 돌려준다 (널이면 0)."""
        return 0 if node is None else node.s

    @staticmethod
    def _merge_nodes(h1, h2):
        """좌편향 힙의 부분 트리 둘을 합친다. 새 뿌리를 돌려준다."""
        if h1 is None:
            return h2
        if h2 is None:
            return h1

        # h1이 더 작은 뿌리를 갖도록 한다
        if h1.key > h2.key:
            h1, h2 = h2, h1

        # h2를 h1의 오른쪽 부분 트리와 재귀적으로 합친다
        h1.right = LeftistHeap._merge_nodes(h1.right, h2)

        # 좌편향 성질을 되살린다: 왼쪽 s값 >= 오른쪽 s값
        if LeftistHeap._s_value(h1.left) < LeftistHeap._s_value(h1.right):
            h1.left, h1.right = h1.right, h1.left

        # s값을 고친다
        h1.s = LeftistHeap._s_value(h1.right) + 1
        return h1

    def merge(self, other):
        """다른 좌편향 힙을 이 힙에 합친다. O(log n)."""
        self.root = self._merge_nodes(self.root, other.root)
        self.size += other.size

    def insert(self, key):
        """노드 하나짜리 힙을 만들어 합쳐서 열쇠를 넣는다. O(log n)."""
        new_node = LeftistNode(key)
        self.root = self._merge_nodes(self.root, new_node)
        self.size += 1

    def find_min(self):
        """가장 작은 열쇠를 돌려준다. O(1)."""
        if self.root is None:
            raise IndexError("find_min from empty heap")
        return self.root.key

    def extract_min(self):
        """가장 작은 열쇠를 없애고 돌려준다. O(log n)."""
        if self.root is None:
            raise IndexError("extract_min from empty heap")
        min_key = self.root.key
        self.root = self._merge_nodes(self.root.left, self.root.right)
        self.size -= 1
        return min_key

    def is_empty(self):
        """힙이 비었는지 살핀다."""
        return self.root is None

    def _verify_leftist(self, node=None, check_root=True):
        """모든 노드에서 좌편향 성질이 성립하는지 확인한다."""
        if check_root:
            node = self.root
        if node is None:
            return True
        left_s = self._s_value(node.left)
        right_s = self._s_value(node.right)
        assert left_s >= right_s, \
            f"Leftist violated at key={node.key}: left_s={left_s}, right_s={right_s}"
        assert node.s == right_s + 1, \
            f"s-value wrong at key={node.key}"
        return (self._verify_leftist(node.left, False) and
                self._verify_leftist(node.right, False))

# === 시연 ===

if __name__ == "__main__":
    h = LeftistHeap()
    values = [7, 3, 8, 1, 5, 2, 9, 4, 6]

    print("Inserting values:")
    for v in values:
        h.insert(v)
        print(f"  Inserted {v}, min = {h.find_min()}, size = {h.size}")

    # 좌편향 성질을 확인한다
    h._verify_leftist()
    print("Leftist property verified.")

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
    h1 = LeftistHeap()
    for v in [5, 3, 7]:
        h1.insert(v)

    h2 = LeftistHeap()
    for v in [2, 8, 1]:
        h2.insert(v)

    print(f"H1 min: {h1.find_min()}, H2 min: {h2.find_min()}")
    h1.merge(h2)
    print(f"After merge, min: {h1.find_min()}, size: {h1.size}")

    h1._verify_leftist()
    print("Leftist property verified after merge.")

    merged = []
    while not h1.is_empty():
        merged.append(h1.extract_min())
    print(f"Merged extraction: {merged}")
```

**출력:**
```
Inserting values:
  Inserted 7, min = 7, size = 1
  Inserted 3, min = 3, size = 2
  Inserted 8, min = 3, size = 3
  Inserted 1, min = 1, size = 4
  Inserted 5, min = 1, size = 5
  Inserted 2, min = 1, size = 6
  Inserted 9, min = 1, size = 7
  Inserted 4, min = 1, size = 8
  Inserted 6, min = 1, size = 9
Leftist property verified.

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
After merge, min: 1, size: 6
Leftist property verified after merge.
Merged extraction: [1, 2, 3, 5, 7, 8]
```

---

## 6. 복잡도 요약

| 연산 | 좌편향 힙 | 이진 힙 |
|-----------|:------------:|:-----------:|
| 삽입 | $O(\log n)$ | $O(\log n)$ |
| 최솟값 찾기 | $O(1)$ | $O(1)$ |
| 최솟값 꺼내기 | $O(\log n)$ | $O(\log n)$ |
| 합치기 | $O(\log n)$ | $O(n)$ |

좌편향 힙은 모든 표준 연산에서 이진 힙과 맞먹으면서 $O(\log n)$의 합치기를 더한다. 그 대가는 포인터의 짐(노드마다 자식 포인터 둘과 s값을 담는다)과, 배열에 바탕한 이진 힙의 간결함 및 캐시 친화성을 맞바꾸는 것이다.

---

## 연습문제

**연습문제 1.**
좌편향 힙의 힙 성질을 밝히고 최솟값·최댓값 원소가 언제나 뿌리에 있음을 증명하라.

??? success "연습문제 1 풀이"
    힙 성질은 노드마다 열쇠가 자식보다 작거나 같거나(최소 힙) 크거나 같다(최대 힙)는 것이다. 뿌리에서 잎까지의 어떤 경로에서도 추이성이 성립하므로 뿌리가 모든 원소의 최솟값(또는 최댓값)이다.

---

**연습문제 2.**
배열 $[4, 7, 2, 9, 1, 5, 3]$에서 좌편향 힙을 따라가라. 단계마다와 그 결과로 나오는 힙을 보여라.

??? success "연습문제 2 풀이"
    이 쪽의 연산을 주어진 배열에 적용하라. 단계마다 배열과 그것이 나타내는 트리를 보여라. 비교와 자리바꿈을 짚어라.

---

**연습문제 3.**
좌편향 힙의 시간 복잡도를 증명하라. 그 한계는 빡빡한가?

??? success "연습문제 3 풀이"
    이 연산은 뿌리에서 잎까지 또는 잎에서 뿌리까지의 경로를 훑으며 층마다 $O(1)$의 일을 한다. 완전 이진 트리의 높이는 $\lfloor\log_2 n\rfloor$이므로 모두 $O(\log n)$이다. 이 한계는 빡빡하다. 높이 전체를 훑도록 강요하는 입력이 있다. $\square$

---

**연습문제 4.**
$k = 5$이고 어휘가 $n = 32{,}000$인 빔 탐색에서 힙으로 뽑기(원소 $n$개에서 상위 $k$개)와 정렬을 견주어라.

??? success "연습문제 4 풀이"
    정렬은 $O(n\log n) = O(32000 \times 15) \approx 48만$번의 연산이 든다. 힙(크기 $k$인 최소 힙)은 $O(n\log k) = O(32000 \times 2.3) \approx 7만 4천$번이다. 힙이 약 $6.5$배 빠르다. 아니면 `torch.topk`가 GPU에 맞추어 다듬은 부분 정렬 알고리즘을 쓴다.

## 정리하며

이 마당은 s값 (널 경로 길이)、좌편향 성질、합치기 연산、다른 연산을 차례로 짚었다.

**참고 문헌**

- Crane, C. A. "Linear lists and priority queues as balanced binary trees." Ph.D. thesis, Stanford University, 1972.
- Tarjan, R. E. *Data Structures and Network Algorithms*. SIAM, 1983.
