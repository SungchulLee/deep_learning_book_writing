# 바깥 기억을 위한 B 나무

두 갈래 찾기 나무는 높이가 $O(\log_2 N)$이지만 켜마다 원반 닿기가 한 번씩 들어 원소 10억 개면 들고남이 30번쯤 든다. B 나무는 마디마다 열쇠를 수백에서 수천 개 채워 원반 덩이 하나가 마디 하나를 담게 해 이 문제를 푼다. 이로써 갈래 수가 2에서 $\Theta(B)$으로 늘어 나무 높이와 들고남 횟수가 $O(\log_B N)$으로 준다.

---

## 1. B-트리의 성질

최소 차수 $t \ge 2$의 B 나무는 다음 성질을 만족한다:

1. 마디마다 열쇠를 $t - 1$개에서 $2t - 1$개 담는다(뿌리는 1개만 가질 수도 있다).
2. 열쇠가 $k$개인 안쪽 마디마다 자식이 꼭 $k + 1$개다.
3. 모든 잎이 같은 깊이에 있다.
4. 마디 안의 열쇠는 오름차순으로 줄 세워져 있다.

바깥 기억에서는 마디마다 원반 덩이 하나에 들어가도록 $t$을 고른다. 열쇠 $2t - 1$개와 자식 가리개 $2t$개가 원소 $B$개 안에 들어가야 하므로:

$$
t = \Theta(B)
$$

이로써 안쪽 마디마다 갈래 수가 $\Theta(B)$이 되며 이것이 들고남 효율의 열쇠다.

---

## 2. 높이의 한계

최소 차수가 $t$이고 열쇠 $N$개를 담는 B 나무의 높이 $h$은 다음을 만족한다:

$$
h \le \log_t \frac{N + 1}{2}
$$

$t = \Theta(B)$이므로 다음을 얻는다:

$$
h = O(\log_B N)
$$

높이 $h$에서 나무는 적어도 열쇠 $2t^h - 1$개를 담으므로 높이가 밑 2가 아니라 밑 $B$의 로그로 늘어난다.

---

## 3. 연산의 들고남 복잡도

연산마다 뿌리에서 잎까지 길을 밟으며 켜마다 들고남 한 번을 한다:

| 연산 | 들고남 복잡도 | 밝힘 |
|---|---|---|
| 찾기 | $O(\log_B N)$ | 뿌리에서 잎까지 길 하나를 따라간다 |
| 넣기 | $O(\log_B N)$ | 찾기 + 마디가 갈라질 수 있음 |
| 지우기 | $O(\log_B N)$ | 찾기 + 합침이나 다시 나눔이 있을 수 있음 |
| 구간 묻기(결과 $K$개) | $O(\log_B N + K/B)$ | 찾기 + 잇단 잎 훑기 |

구간 묻기 가둠은 B 나무의 가까움을 비춘다. 시작 잎을 찾고 나면 잇단 열쇠가 이웃한 덩이에 채워져 있다.

---

## 4. 찾기

열쇠 $k$ 찾기는 뿌리에서 시작해 나무를 내려간다. 마디마다 열쇠 $O(B)$개를 두 갈래로 찾아 어느 자식으로 갈지 정한다. 마디 안의 두 갈래 찾기는 온전히 기억에서 하므로(마디 전체를 들고남 한 번으로 올렸다) 온 비용은 켜마다 들고남 한 번이다:

$$
\text{Search I/O} = O(\log_B N)
$$

---

## 5. 마디 갈라짐이 있는 넣기

열쇠를 넣으려면 먼저 알맞은 잎을 찾는다. 잎에 열쇠가 $2t - 1$개보다 적으면 곧바로 넣는다. 잎이 가득 차 있으면(열쇠가 $2t - 1$개) 열쇠 $t - 1$개짜리 마디 둘로 가르고 가운뎃값 열쇠를 어버이로 올린다. 갈라짐이 위로 줄줄이 번질 수 있지만 갈라짐마다 들고남 $O(1)$번이 든다(가득 찬 마디를 읽고 새 마디 둘을 적고 어버이를 고친다).

온 넣기 비용은 다음과 같다:

$$
\text{Insert I/O} = O(\log_B N)
$$

---

## 6. 다시 고르게 하는 지우기

지우기는 넣기를 그대로 뒤집는다. 열쇠를 빼서 마디의 열쇠가 $t - 1$개보다 적어지면 나무는 형제에게서 빌리거나 마디 둘을 합쳐 다시 고르게 한다. 다시 고르는 걸음마다 들고남 $O(1)$번이 들고 뿌리까지의 길에서 많아야 $O(\log_B N)$번 일어난다.

---

## 7. 보기: B 나무 마디와 찾기

```python
"""
바깥 기억을 위한 B 나무 마디 얼개와 찾기.

B 나무 마디가 열쇠 여럿을 덩이 하나에 채우는 방식과 찾기가
켜 O(log_B N)개를 밟는 모습을 보여 준다.
"""

import bisect
import math

# ===================================================================
# B 나무 마디
# ===================================================================

class BTreeNode:
    """최소 차수가 t인 B 나무 마디."""

    def __init__(self, t: int, leaf: bool = True):
        self.t = t
        self.leaf = leaf
        self.keys: list = []
        self.children: list[BTreeNode] = []

    def is_full(self) -> bool:
        return len(self.keys) == 2 * self.t - 1

# ===================================================================
# B 나무
# ===================================================================

class BTree:
    """최소 차수 t을 고를 수 있는 B 나무."""

    def __init__(self, t: int):
        self.t = t
        self.root = BTreeNode(t, leaf=True)
        self.io_count = 0  # 들고남 연산을 좇는다

    def search(self, node: BTreeNode, key: int) -> bool:
        """들고남을 세며 열쇠를 찾는다."""
        self.io_count += 1  # 마디 읽기 = 들고남 1번
        i = bisect.bisect_left(node.keys, key)
        if i < len(node.keys) and node.keys[i] == key:
            return True
        if node.leaf:
            return False
        return self.search(node.children[i], key)

    def _split_child(self, parent: BTreeNode, idx: int):
        """가득 찬 자식 마디를 가른다."""
        t = self.t
        child = parent.children[idx]
        new_node = BTreeNode(t, leaf=child.leaf)

        # 열쇠의 위 반을 새 마디로 옮긴다
        parent.keys.insert(idx, child.keys[t - 1])
        parent.children.insert(idx + 1, new_node)
        new_node.keys = child.keys[t:]
        child.keys = child.keys[:t - 1]

        if not child.leaf:
            new_node.children = child.children[t:]
            child.children = child.children[:t]

    def insert(self, key: int):
        """B 나무에 열쇠를 넣는다."""
        root = self.root
        if root.is_full():
            new_root = BTreeNode(self.t, leaf=False)
            new_root.children.append(self.root)
            self._split_child(new_root, 0)
            self.root = new_root
        self._insert_non_full(self.root, key)

    def _insert_non_full(self, node: BTreeNode, key: int):
        """가득 차지 않았음이 보장된 마디에 넣는다."""
        i = bisect.bisect_left(node.keys, key)
        if node.leaf:
            node.keys.insert(i, key)
        else:
            if node.children[i].is_full():
                self._split_child(node, i)
                if key > node.keys[i]:
                    i += 1
            self._insert_non_full(node.children[i], key)

    def height(self) -> int:
        """나무 높이를 셈한다."""
        h = 0
        node = self.root
        while not node.leaf:
            node = node.children[0]
            h += 1
        return h

# ===================================================================
# 메인
# ===================================================================

if __name__ == "__main__":
    import random

    t = 50  # 최소 차수(덩이 크기 B ~ 열쇠 100개를 흉내 냄)
    N = 100_000
    tree = BTree(t)

    # 열쇠 N개를 넣는다
    keys = list(range(N))
    random.shuffle(keys)
    for k in keys:
        tree.insert(k)

    # 찾기 들고남을 잰다
    tree.io_count = 0
    tree.search(tree.root, N // 2)
    actual_ios = tree.io_count

    theoretical = math.ceil(math.log(N) / math.log(t))
    print(f"B-tree with t={t}, N={N:,}")
    print(f"  Height:         {tree.height()}")
    print(f"  Search I/Os:    {actual_ios}")
    print(f"  Theoretical:    O(log_{t}({N})) = {theoretical}")
```

??? example "보기 내놓기"

    ```
    B-tree with t=50, N=100,000
      높이:         2
      찾기 들고남:    3
      Theoretical:    O(log_50(100000)) = 3
    ```

    갈래 수가 50이면 열쇠 10만 개의 B 나무는 높이가 2~3에 지나지 않아 찾기마다 들고남 3번이면 된다.

---

## 8. B 나무와 두 갈래 찾기 나무

| 성질 | 두 갈래 찾기 나무 | B 나무($t = \Theta(B)$) |
|---|---|---|
| 갈래 수 | 2 | $\Theta(B)$ |
| 높이 | $O(\log_2 N)$ | $O(\log_B N)$ |
| 찾기마다 들고남 | $O(\log_2 N)$ | $O(\log_B N)$ |
| 마디 크기 | 열쇠 1개 | 열쇠 $\Theta(B)$개 |
| 원반 덩이 쓰임 | 낮음 | 높음(50~100% 참) |

$B = 1000$이고 $N = 10^9$이면 두 갈래 찾기 나무는 찾기마다 들고남이 30번쯤 들지만 B 나무는 3번이면 된다.

---

## 연습문제

**연습문제 1.**
B 나무가 바깥 기억 찾기에 왜 가장 좋은지 밝혀라.

??? success "연습문제 1 풀이"
    갈래 수가 $B$인 B 나무의 높이는 $O(\log_B N)$이다. 마디마다 원반 덩이 하나에 들어간다. 찾기는 켜마다 덩이 하나를 읽어 들고남 $O(\log_B N)$번이 든다. 이는 가장 좋은 값이다. 열쇠 $N$개에 대한 어떤 견줌 바탕 찾기도 결과 $N+1$가지를 가려내야 한다. 들고남마다 열쇠 $B$개를 읽어 갈래가 $B+1$개다. 들고남 $T$번 뒤 $(B+1)^T \geq N$이므로 $T \geq \log_{B+1} N = \Omega(\log_B N)$이다.

---

**연습문제 2.**
B 나무 마디의 얼개와 넣기 연산을 밝혀라.

??? success "연습문제 2 풀이"
    안쪽 마디마다 열쇠 $\Theta(B)$개와 자식 가리개 $\Theta(B)$개를 담아 원반 덩이 하나를 채운다. 마디 안의 열쇠는 줄 세워져 있다. 넣기: 알맞은 잎을 찾는다(들고남 $O(\log_B N)$번). 잎에 자리가 있으면 열쇠를 넣는다. 가득 차면(넘침) 마디를 가른다. 가운뎃값 열쇠가 어버이로 올라가고 반쯤 찬 자식 둘이 생긴다. 어버이가 넘치면 갈라짐이 위로 퍼진다. 가장 나쁜 경우 갈라짐이 뿌리까지 줄줄이 번져 높이가 1 는다. 고루 나누면 넣기마다 들고남 $O(\log_B N)$번이다(대부분의 넣기는 번지지 않는다).

---

**연습문제 3.**
B 나무와 B+ 나무를 견주어라. 자료 바탕에서 B+ 나무를 왜 즐겨 쓰는가?

??? success "연습문제 3 풀이"
    B 나무는 모든 마디(안쪽과 잎)에 자료를 담는다. B+ 나무는 잎에만 자료를 담고 안쪽 마디에는 열쇠와 자식 가리개만 담는다. B+의 이점: (1) 안쪽 마디의 퍼짐이 크다(자료 짐이 없어 덩이마다 열쇠가 많다). 나무가 낮아진다, (2) 잎이 양쪽으로 이은 목록으로 이어져 효율 좋은 구간 묻기($O(K/B)$ 차례 훑기)를 할 수 있다, (3) 모든 찾기가 잎에 닿는다(닿기 때가 고르다), (4) 지우기가 더 단순하다. 대부분의 자료 바탕(MySQL InnoDB, PostgreSQL)이 B+ 나무를 쓴다.

---

**연습문제 4.**
B 나무 설계 원칙은 기계 배움 특징 곳간의 열쇠-값 곳간에 어떻게 쓰이는가?

??? success "연습문제 4 풀이"
    기계 배움 특징 곳간(Redis, RocksDB)은 낱것 번호를 열쇠로 하는 특징 박아 넣기 수십억 개를 다룬다. B 나무 원칙이 그대로 쓰인다. (1) 퍼짐이 크면 점 묻기(쓰는 이 하나의 특징 가져오기)의 찾기 깊이가 가장 얕아진다, (2) 줄 세워 담으면 구간 훑기(때 창의 특징 가져오기)를 할 수 있다, (3) 덩이에 줄 맞춘 마디가 SSD 처리량을 가장 크게 한다. (RocksDB이 쓰는) LSM 나무는 B 나무에 견주어 읽기 성능과 적기 성능을 맞바꾸며, 이는 적기가 많은 들이기와 읽기가 많은 서비스를 하는 기계 배움 특징 곳간에 알맞다.

## 정리하며

이 마당은 B-트리의 성질、높이의 한계、연산의 들고남 복잡도、찾기을 차례로 짚었다.

**참고 문헌**

- Bayer, R. & McCreight, E. "Organization and Maintenance of Large Ordered Indexes," *Acta Informatica*, 1(3), 1972.
- Cormen, T. et al. *Introduction to Algorithms*, 18장(B 나무), MIT Press, 2022.
- Vitter, J. S. *Algorithms and Data Structures for External Memory*, 2008.
