# 짝짓기 힙

피보나치 힙은 최적의 분할 상환 한계를 이루지만 구현이 지독히 까다롭기로 이름났고, 포인터를 다루고 표시를 관리하는 논리 때문에 상수 배가 크다. **짝짓기 힙**은 훨씬 간단한 대안을 준다. 같은 분할 상환 $O(1)$의 삽입과 합치기, 분할 상환 $O(\log n)$의 최솟값 꺼내기를 훨씬 간단한 짜임으로 이루며 실제로도 잘 돈다. 열쇠 낮추기의 한계는 분할 상환 $O(2^{O(\sqrt{\log \log n})})$으로 피보나치 힙의 $O(1)$에는 못 미치지만, 실제로 짝짓기 힙이 더 즐겨 쓰일 만큼 가깝다.

## 구조

짝짓기 힙은 힙 순서를 지키는 여러 갈래 트리이다. 노드마다 자식을 얼마든지 가질 수 있다. 표준 표현은 **왼쪽 자식, 오른쪽 형제** 방식을 쓴다.

- **child**: 가장 왼쪽 자식을 가리키는 포인터
- **sibling**: 다음 형제를 가리키는 포인터
- **prev**: 앞 형제를 가리키는 포인터 (가장 왼쪽 자식이면 부모)

구조 불변식은 **힙 순서 성질** 하나뿐이다. 노드마다 열쇠가 자식마다의 열쇠 이하이다. 균형 조건도, 차수 제약도, 표시도 없다. 이 간결함이 짝짓기 힙의 가장 큰 매력이다.

## 합치기 (짝짓기)

두 짝짓기 힙을 합치는 데는 일정 시간이 든다. 두 뿌리를 견주어 진 쪽을 이긴 쪽의 자식으로 삼는다.

```
MERGE(h1, h2):
    if h1 is None: return h2
    if h2 is None: return h1
    if h1.key <= h2.key:
        make h2 the leftmost child of h1
        return h1
    else:
        make h1 the leftmost child of h2
        return h2
```

**비용**: $O(1)$.

## 삽입

삽입은 노드 하나짜리 트리를 만들어 기존 힙과 합친다.

**비용**: $O(1)$ (합치기 한 번).

## 최솟값 찾기

뿌리가 최솟값을 지닌다.

**비용**: $O(1)$.

## 최솟값 꺼내기 (최솟값 지우기)

최솟값 꺼내기는 뿌리를 없애고 (뿌리의 자식인) 부분 트리의 모음을 남긴다. 핵심 물음은 이 부분 트리를 어떻게 다시 모을 것인가이다. **두 번 훑는 짝짓기** 방법이 가장 좋은 한계를 준다.

### 두 번 훑는 짝짓기

자식 $c_1, c_2, c_3, \ldots, c_k$(왼쪽에서 오른쪽)이 주어졌을 때 다음과 같이 한다.

1. **왼쪽에서 오른쪽으로 훑기**: 자식을 차례로 짝짓는다. $c_1$과 $c_2$을 합치고 $c_3$과 $c_4$을 합치는 식이다. $k$이 홀수이면 마지막 자식은 짝이 없다.
2. **오른쪽에서 왼쪽으로 훑기**: 그렇게 나온 트리들을 오른쪽에서 왼쪽으로 합쳐 트리 하나로 만든다.

```
EXTRACT-MIN(H):
    if H is empty: error
    min_key = H.root.key
    children = list of H.root's children [c1, c2, ..., ck]

    # 왼쪽에서 오른쪽으로 짝짓기
    paired = []
    for i in 0, 2, 4, ...:
        if i + 1 < k:
            paired.append(MERGE(children[i], children[i+1]))
        else:
            paired.append(children[i])

    # 오른쪽에서 왼쪽으로 모으기
    result = paired[-1]
    for i in len(paired) - 2 down to 0:
        result = MERGE(paired[i], result)

    H.root = result
    return min_key
```

**비용**: 최소 노드의 자식 수를 $k$이라 할 때 $O(k)$이다. 분할 상환은 $O(\log n)$이다.

!!! tip "왜 두 번 훑는가"
    (오른쪽에서 왼쪽으로 훑지 않고) 왼쪽에서 오른쪽으로만 합치면 병적으로 기운 트리가 생길 수 있다. 두 번 훑는 방법은 부분 트리마다 "무게"가 엇비슷한 부분 트리와 짝지어지게 하는데, 병합 정렬이 재귀적으로 반씩 나누어 균형을 얻는 것과 닮았다. 이것이 분할 상환 $O(\log n)$ 한계를 준다.

## 열쇠 낮추기

노드 $x$의 열쇠를 낮추려면 다음과 같이 한다.

1. $x$이 뿌리이면 그냥 열쇠를 고친다.
2. 그렇지 않으면 $x$을 부모에게서 자르고(형제 목록에서 떼어 내고) 그렇게 나온 부분 트리를 주 힙과 합친다.

**비용**: 연산 자체는 최악의 경우 $O(1)$이다. 분할 상환 한계는 $O(2^{O(\sqrt{\log \log n})})$으로 로그보다는 작지만 상수는 아니다.

## 구현

```python
"""
짝짓기 힙 구현.

짝짓기 힙은 O(1)의 합치기와 삽입, 두 번 훑는 짝짓기로
분할 상환 O(log n)의 최솟값 꺼내기를 받쳐 주는,
힙 순서를 지키는 간단한 여러 갈래 트리이다.
"""


# === 짝짓기 힙 노드 ===

class PairingNode:
    """왼쪽 자식, 오른쪽 형제 표현을 쓰는 짝짓기 힙의 노드.

    속성:
        key: 우선순위 값
        child: 가장 왼쪽 자식
        sibling: 다음 형제
        prev: 앞 형제나 부모 (자르기용)
    """

    def __init__(self, key):
        self.key = key
        self.child = None
        self.sibling = None
        self.prev = None

    def __repr__(self):
        return f"PairingNode(key={self.key})"


# === 짝짓기 힙 ===

class PairingHeap:
    """두 번 훑는 최솟값 지우기를 쓰는 최소 짝짓기 힙."""

    def __init__(self):
        self.root = None
        self.size = 0

    @staticmethod
    def _link(h1, h2):
        """트리 둘을 잇는다. 작은 뿌리가 부모가 된다."""
        if h1 is None:
            return h2
        if h2 is None:
            return h1
        if h1.key <= h2.key:
            # h2가 h1의 가장 왼쪽 자식이 된다
            h2.sibling = h1.child
            if h1.child is not None:
                h1.child.prev = h2
            h1.child = h2
            h2.prev = h1
            return h1
        else:
            # h1이 h2의 가장 왼쪽 자식이 된다
            h1.sibling = h2.child
            if h2.child is not None:
                h2.child.prev = h1
            h2.child = h1
            h1.prev = h2
            return h2

    def merge(self, other):
        """다른 짝짓기 힙을 이 힙에 합친다. O(1)."""
        self.root = self._link(self.root, other.root)
        self.size += other.size

    def insert(self, key):
        """열쇠를 넣는다. O(1)."""
        node = PairingNode(key)
        self.root = self._link(self.root, node)
        self.size += 1
        return node

    def find_min(self):
        """가장 작은 열쇠를 돌려준다. O(1)."""
        if self.root is None:
            raise IndexError("find_min from empty heap")
        return self.root.key

    def extract_min(self):
        """가장 작은 열쇠를 없애고 돌려준다. 분할 상환 O(log n)."""
        if self.root is None:
            raise IndexError("extract_min from empty heap")
        min_key = self.root.key

        # 모든 자식을 모은다
        children = []
        child = self.root.child
        while child is not None:
            nxt = child.sibling
            child.sibling = None
            child.prev = None
            children.append(child)
            child = nxt

        # 두 번 훑는 짝짓기
        self.root = self._two_pass_merge(children)
        self.size -= 1
        return min_key

    @staticmethod
    def _two_pass_merge(children):
        """두 번 훑는 짝짓기로 트리 목록을 합친다."""
        if not children:
            return None
        if len(children) == 1:
            return children[0]

        # 왼쪽에서 오른쪽으로 짝짓기
        paired = []
        i = 0
        while i + 1 < len(children):
            paired.append(PairingHeap._link(children[i], children[i + 1]))
            i += 2
        if i < len(children):
            paired.append(children[i])

        # 오른쪽에서 왼쪽으로 모으기
        result = paired[-1]
        for j in range(len(paired) - 2, -1, -1):
            result = PairingHeap._link(paired[j], result)

        return result

    def decrease_key(self, node, new_key):
        """노드의 열쇠를 낮춘다. 최악의 경우 O(1)."""
        if new_key > node.key:
            raise ValueError("New key is greater than current key")
        node.key = new_key
        if node is self.root:
            return
        # 노드를 부모에게서 자른다
        if node.prev is not None:
            if node.prev.child is node:
                node.prev.child = node.sibling
            else:
                node.prev.sibling = node.sibling
        if node.sibling is not None:
            node.sibling.prev = node.prev
        node.prev = None
        node.sibling = None
        self.root = self._link(self.root, node)

    def is_empty(self):
        """힙이 비었는지 살핀다."""
        return self.root is None


# === 시연 ===

if __name__ == "__main__":
    h = PairingHeap()
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
    print("Correctness verified." if extracted == sorted(extracted) else "ERROR!")
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
Correctness verified.
```

## 복잡도 비교

| 연산 | 이진 힙 | 피보나치 힙 | 짝짓기 힙 |
|-----------|:-----------:|:--------------:|:------------:|
| 삽입 | $O(\log n)$ | 분할 상환 $O(1)$ | $O(1)$ |
| 최솟값 찾기 | $O(1)$ | $O(1)$ | $O(1)$ |
| 합치기 | $O(n)$ | $O(1)$ | $O(1)$ |
| 최솟값 꺼내기 | $O(\log n)$ | 분할 상환 $O(\log n)$ | 분할 상환 $O(\log n)$ |
| 열쇠 낮추기 | $O(\log n)$ | 분할 상환 $O(1)$ | 분할 상환 $O(2^{O(\sqrt{\log \log n})})$ |

!!! tip "실제 성능"
    이론상 열쇠 낮추기 한계가 약한데도 성능 시험에서 짝짓기 힙이 피보나치 힙을 한결같이 앞선다. 포인터 짜임이 간단해 상수 배가 작고 캐시 움직임이 낫고 코드가 훨씬 적다. 실제 응용 대부분에서 합칠 수 있는 우선순위 큐로는 짝짓기 힙을 권한다.

## 참고 문헌

- Fredman, M. L., Sedgewick, R., Sleator, D. D., and Tarjan, R. E. "The pairing heap: a new form of self-adjusting heap." *Algorithmica*, 1(1):111--129, 1986.
- Iacono, J. "Improved upper bounds for pairing heaps." *Scandinavian Workshop on Algorithm Theory*, 2000.


## 연습문제

**연습문제 1.**
짝짓기 힙의 힙 성질을 밝히고 최솟값·최댓값 원소가 언제나 뿌리에 있음을 증명하라.

??? success "연습문제 1 풀이"
    힙 성질은 노드마다 열쇠가 자식보다 작거나 같거나(최소 힙) 크거나 같다(최대 힙)는 것이다. 뿌리에서 잎까지의 어떤 경로에서도 추이성이 성립하므로 뿌리가 모든 원소의 최솟값(또는 최댓값)이다.

---

**연습문제 2.**
배열 $[4, 7, 2, 9, 1, 5, 3]$에서 짝짓기 힙을 따라가라. 단계마다와 그 결과로 나오는 힙을 보여라.

??? success "연습문제 2 풀이"
    이 쪽의 연산을 주어진 배열에 적용하라. 단계마다 배열과 그것이 나타내는 트리를 보여라. 비교와 자리바꿈을 짚어라.

---

**연습문제 3.**
짝짓기 힙의 시간 복잡도를 증명하라. 그 한계는 빡빡한가?

??? success "연습문제 3 풀이"
    이 연산은 뿌리에서 잎까지 또는 잎에서 뿌리까지의 경로를 훑으며 층마다 $O(1)$의 일을 한다. 완전 이진 트리의 높이는 $\lfloor\log_2 n\rfloor$이므로 모두 $O(\log n)$이다. 이 한계는 빡빡하다. 높이 전체를 훑도록 강요하는 입력이 있다. $\square$

---

**연습문제 4.**
$k = 5$이고 어휘가 $n = 32{,}000$인 빔 탐색에서 힙으로 뽑기(원소 $n$개에서 상위 $k$개)와 정렬을 견주어라.

??? success "연습문제 4 풀이"
    정렬은 $O(n\log n) = O(32000 \times 15) \approx 48만$번의 연산이 든다. 힙(크기 $k$인 최소 힙)은 $O(n\log k) = O(32000 \times 2.3) \approx 7만 4천$번이다. 힙이 약 $6.5$배 빠르다. 아니면 `torch.topk`가 GPU에 맞추어 다듬은 부분 정렬 알고리즘을 쓴다.