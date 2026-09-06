# 트리 용어

트리 알고리즘을 공부하기에 앞서 정확한 낱말이 필요하다. 트리는 파일 시스템과 구문 분석, 의사 결정, 탐색 등 컴퓨터 과학 곳곳에 나타나며, 모든 트리 알고리즘 설명이 여기서 정의하는 말에 기댄다. 이 쪽은 이 장의 나머지가 딛고 설 핵심 용어를 세운다.

## 재귀적 정의

**뿌리 있는 트리** $T$을 재귀적으로 정의한다.

- **빈 트리**에는 노드가 없다.
- **비어 있지 않은 트리**는 **뿌리**라 부르는 특별한 노드 $r$과, 비어 있지 않은 부분 트리 $T_1, T_2, \ldots, T_k$ 0개 이상으로 이루어지며, 각 부분 트리의 뿌리는 $r$과 변으로 이어져 있다.

이 재귀적인 짜임 덕분에 트리는 재귀 알고리즘과 잘 맞는다.

## 핵심 용어

다음 예제 트리를 생각해 보자.

```
            A             <- root
          / | \
         B  C  D          <- children of A
        / \    |
       E   F   G          <- E,F are children of B; G is child of D
          / \
         H   I            <- children of F
```

### 노드와 변

| 용어 | 정의 |
|------|-----------|
| **노드** (꼭짓점) | 데이터를 담는 트리의 기본 단위. 위의 $A$부터 $I$까지. |
| **변** | 부모와 자식을 잇는 연결. 노드가 $n$개인 트리에는 변이 꼭 $n - 1$개 있다. |
| **뿌리** | 부모가 없는 가장 위의 노드. 예에서는 노드 $A$. |

### 가족 관계

| 용어 | 정의 |
|------|-----------|
| **부모** | 어떤 노드 바로 위의 노드. $B$은 $E$과 $F$의 부모이다. |
| **자식** | 어떤 노드 바로 아래의 노드. $E$과 $F$은 $B$의 자식이다. |
| **형제** | 부모가 같은 노드들. $B$, $C$, $D$은 형제이다. |
| **조상** | 어떤 노드에서 뿌리까지의 경로 위에 있는 노드(자기 자신 포함). $H$의 조상은 $H$, $F$, $B$, $A$이다. |
| **자손** | 어떤 노드에서 변을 따라 아래로 내려가 닿을 수 있는 노드. $B$의 자손은 $E$, $F$, $H$, $I$이다. |

### 노드의 갈래

| 용어 | 정의 |
|------|-----------|
| **잎** (외부 노드) | 자식이 없는 노드. $E$, $H$, $I$, $C$, $G$이 잎이다. |
| **내부 노드** | 자식이 하나 이상인 노드. $A$, $B$, $D$, $F$이 내부 노드이다. |
| 노드의 **차수** | 그 노드의 자식 수. $\text{degree}(A) = 3$, $\text{degree}(B) = 2$, $\text{degree}(C) = 0$. |

### 구조에 관한 용어

| 용어 | 정의 |
|------|-----------|
| **부분 트리** | 어떤 노드와 그 모든 자손이 이루는 트리. $B$을 뿌리로 하는 부분 트리는 $\{B, E, F, H, I\}$이다. |
| **경로** | 잇따른 노드가 변으로 이어진 노드의 나열 $v_1, v_2, \ldots, v_k$. |
| **레벨** | 깊이가 같은 모든 노드의 모임. 레벨 0에는 뿌리만 있다. |
| **깊이** | 뿌리에서 어떤 노드까지의 변의 수. [높이와 깊이](height_depth.md)를 보라. |
| **높이** | 어떤 노드에서 잎까지 가장 긴 경로의 변의 수. [높이와 깊이](height_depth.md)를 보라. |

## 핵심 성질

재귀적 정의에서 근본적인 성질 몇 가지가 곧바로 따라 나온다.

1. **변의 수**: 노드가 $n$개인 트리에는 변이 꼭 $n - 1$개 있다.
2. **유일한 경로**: 트리 안 어떤 두 노드 사이에도 경로가 꼭 하나 있다.
3. **연결되어 있고 순환이 없음**: 트리는 순환이 없는 연결 그래프이다. 어떤 변을 없애도 트리가 끊어지고, 어떤 변을 더해도 순환이 생긴다.

!!! tip "이진 트리라는 특수한 경우"
    **이진 트리**에서는 노드마다 자식이 많아야 둘이고 각각 **왼쪽 자식**과 **오른쪽 자식**이라 부른다. 왼쪽과 오른쪽 부분 트리도 (비어 있을 수 있는) 이진 트리이다. 이 장은 대부분 이진 트리를 다루지만 위의 용어는 차수에 상관없이 모든 트리에 통한다.

## 트리의 차수

**트리의 차수**는 트리 안 노드의 차수 중 최댓값이다.

$$
\text{degree}(T) = \max_{v \in T} \text{degree}(v)
$$

이진 트리의 차수는 많아야 2이다. 삼진 트리의 차수는 많아야 3이다.

## 파이썬 예제

```python
"""
트리 용어 시연.

뿌리, 부모, 자식, 잎, 내부 노드, 깊이, 높이, 차수, 부분 트리의 크기,
조상 같은 핵심 개념을 보인다.
"""


# === 노드 정의 ===

class TreeNode:
    """일반 트리의 노드 (자식의 수에 제한이 없다)."""

    def __init__(self, key):
        self.key = key
        self.children = []
        self.parent = None

    def add_child(self, child_node):
        """자식을 더하고 그 부모 포인터를 지정한다."""
        child_node.parent = self
        self.children.append(child_node)

    def is_leaf(self):
        """잎에는 자식이 없다."""
        return len(self.children) == 0

    def is_root(self):
        """뿌리에는 부모가 없다."""
        return self.parent is None

    def degree(self):
        """자식의 수."""
        return len(self.children)

    def depth(self):
        """뿌리에서 이 노드까지의 변의 수."""
        d = 0
        current = self.parent
        while current is not None:
            d += 1
            current = current.parent
        return d

    def ancestors(self):
        """자기 자신에서 뿌리까지의 조상 목록을 돌려준다."""
        result = [self.key]
        current = self.parent
        while current is not None:
            result.append(current.key)
            current = current.parent
        return result

    def __repr__(self):
        return f"TreeNode({self.key})"


# === 트리 질의 ===

def subtree_size(node):
    """node를 뿌리로 하는 부분 트리의 노드 수를 돌려준다."""
    if node is None:
        return 0
    count = 1
    for child in node.children:
        count += subtree_size(child)
    return count


def tree_height(node):
    """node를 뿌리로 하는 부분 트리의 높이를 돌려준다."""
    if node is None or node.is_leaf():
        return 0
    return 1 + max(tree_height(c) for c in node.children)


# === 메인 ===

if __name__ == "__main__":
    # 예제 트리 만들기
    A = TreeNode("A")
    B, C, D = TreeNode("B"), TreeNode("C"), TreeNode("D")
    E, F, G = TreeNode("E"), TreeNode("F"), TreeNode("G")
    H, I = TreeNode("H"), TreeNode("I")

    A.add_child(B); A.add_child(C); A.add_child(D)
    B.add_child(E); B.add_child(F)
    D.add_child(G)
    F.add_child(H); F.add_child(I)

    nodes = [A, B, C, D, E, F, G, H, I]

    print(f"{'Node':<6} {'Depth':<6} {'Degree':<7} {'Leaf?':<6} {'Subtree size'}")
    print("-" * 45)
    for node in nodes:
        print(f"{node.key:<6} {node.depth():<6} {node.degree():<7} "
              f"{'yes' if node.is_leaf() else 'no':<6} {subtree_size(node)}")

    print(f"\nTree height: {tree_height(A)}")
    print(f"Ancestors of H: {H.ancestors()}")
    print(f"Edge count: {len(nodes) - 1}")
```

**출력:**
```
Node   Depth  Degree  Leaf?  Subtree size
---------------------------------------------
A      0      3       no     9
B      1      2       no     5
C      1      0       yes    1
D      1      1       no     2
E      2      0       yes    1
F      2      2       no     3
G      2      0       yes    1
H      3      0       yes    1
I      3      0       yes    1

Tree height: 3
Ancestors of H: ['H', 'F', 'B', 'A']
Edge count: 8
```

## 참고 문헌

- [Introduction to Algorithms (CLRS), 12장](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)


## 연습문제

**연습문제 1.**
트리 용어에 대해 트리의 높이 $h$으로 시간 복잡도를 밝혀라. $h = O(\log n)$이 되는 것은 언제인가?

??? success "연습문제 1 풀이"
    이 연산은 $O(h)$ 시간이 걸린다. 균형 잡힌 트리에서는 $h = O(\log n)$이다(무작위 삽입 순서에서는 기댓값이 $O(\log n)$이고, 균형 이진 탐색 트리 계열은 이를 보장한다). 한쪽으로 치우친 트리에서는 $h = O(n)$이다(정렬된 순서로 넣은 경우).

---

**연습문제 2.**
열쇠 5, 3, 7, 1, 4, 6, 8을 차례로 넣어 만든 이진 탐색 트리에 트리 용어을(를) 적용하라. 단계마다 보여라.

??? success "연습문제 2 풀이"
    차례로 넣어 트리를 만든 뒤 이 쪽의 연산을 적용한다. 단계마다 트리의 모습을 보이고 들르거나 고친 노드를 표시한다.

---

**연습문제 3.**
이진 탐색 트리의 높이를 $h$이라 할 때 트리 용어이(가) $O(h)$ 시간에 끝남을 증명하라.

??? success "연습문제 3 풀이"
    이 연산은 뿌리에서 잎으로 가는 경로(또는 그 일부)를 따라가며 노드마다 $O(1)$의 일을 한다. 경로의 길이는 많아야 $h$이다. 따라서 모두 $O(h)$이다. $\square$

---

**연습문제 4.**
이진 탐색 트리의 연산은 딥러닝에서 정렬된 데이터를 다루는 일(예를 들어 빔 탐색에서 상위 $k$개의 점수를 유지하는 일)과 어떤 관계가 있는가?

??? success "연습문제 4 풀이"
    빔 탐색은 가장 좋은 가설 $k$개를 유지하므로 새 후보를 넣고 가장 나쁜 것을 빼는 일을 효율적으로 해야 한다. 이진 탐색 트리는 둘 다 $O(\log k)$에 해낸다. 실제로는 더 간단하고 상수가 작은 힙을 즐겨 쓰지만, 이진 탐색 트리는 범위 검색이나 순위 같은 더 풍부한 질의를 지원한다.