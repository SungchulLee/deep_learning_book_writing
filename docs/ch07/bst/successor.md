# 후속자와 선행자

이진 탐색 트리의 응용에서는 특정 열쇠뿐 아니라 정렬된 순서에서 **다음**이나 **이전** 열쇠를 찾아야 할 때가 많다. 데이터베이스의 범위 질의, 반복자 구현, 순서 통계 연산 모두 어떤 노드의 중위 후속자와 선행자를 효율적으로 찾는 데 기댄다. 이진 탐색 트리의 중위 순회가 열쇠를 정렬된 순서로 내놓으므로, 후속자와 선행자는 전체를 훑지 않고도 트리의 구조에서 곧바로 나온다.

## 중위 후속자

노드 $x$의 **중위 후속자**는 $x.key$보다 큰 열쇠 가운데 가장 작은 열쇠를 가진 노드이다. 알고리즘은 $x$에 오른쪽 부분 트리가 있는지에 따라 두 경우로 나뉜다.

**경우 1: $x$에 오른쪽 자식이 있다.** 후속자는 $x$의 오른쪽 부분 트리에서 가장 왼쪽 노드, 곧 그 부분 트리의 최솟값이다.

**경우 2: $x$에 오른쪽 자식이 없다.** 후속자는 왼쪽 부분 트리가 $x$을 품는 조상 가운데 가장 낮은 것이다. $x$에서 위로 올라가며 부모의 왼쪽 자식인 노드를 찾는다. 그 부모가 후속자이다.

$$
\text{successor}(x) =
\begin{cases}
\text{minimum}(x.\text{right}) & \text{if } x.\text{right} \neq \text{nil} \\
\text{lowest ancestor } y \text{ such that } x \text{ is in } y.\text{left subtree} & \text{otherwise}
\end{cases}
$$

??? example "후속자 찾기"
    다음 이진 탐색 트리를 생각해 보자.

    ```
            15
           /  \
         10    20
        /  \   / \
       5   12 17  25
          /
         11
    ```

    - **12의 후속자:** 노드 12에는 오른쪽 자식이 없다. 위로 올라가 보면 12는 10의 오른쪽 자식이므로 계속 올라간다. 10은 15의 왼쪽 자식이다. 후속자는 **15**이다.
    - **10의 후속자:** 노드 10에는 오른쪽 자식(12)이 있다. 10의 오른쪽 부분 트리에서 가장 왼쪽 노드는 **11**이다.
    - **15의 후속자:** 노드 15에는 오른쪽 자식(20)이 있다. 15의 오른쪽 부분 트리에서 가장 왼쪽 노드는 **17**이다.
    - **25의 후속자:** 노드 25에는 오른쪽 자식이 없고, 왼쪽 부분 트리가 25를 품는 조상도 (지금까지 훑은 범위에는) 없다. 후속자는 **없음**이다. 25가 최댓값이다.

## 중위 선행자

노드 $x$의 **중위 선행자**는 $x.key$보다 작은 열쇠 가운데 가장 큰 열쇠를 가진 노드이다. 후속자의 논리를 그대로 뒤집으면 된다.

**경우 1: $x$에 왼쪽 자식이 있다.** 선행자는 $x$의 왼쪽 부분 트리에서 가장 오른쪽 노드, 곧 그 부분 트리의 최댓값이다.

**경우 2: $x$에 왼쪽 자식이 없다.** 선행자는 오른쪽 부분 트리가 $x$을 품는 조상 가운데 가장 낮은 것이다.

$$
\text{predecessor}(x) =
\begin{cases}
\text{maximum}(x.\text{left}) & \text{if } x.\text{left} \neq \text{nil} \\
\text{lowest ancestor } y \text{ such that } x \text{ is in } y.\text{right subtree} & \text{otherwise}
\end{cases}
$$

## 구현

구현을 두 가지 싣는다. 하나는 (교과서 알고리즘과 같이) 부모 포인터를 쓰고, 다른 하나는 부모 포인터 없이 위에서 아래로 찾는다.

### 부모 포인터가 있을 때

```python
"""부모 포인터를 쓰는 이진 탐색 트리의 후속자와 선행자."""


# === 노드 정의 ===

class Node:
    """부모 포인터가 있는 이진 탐색 트리 노드."""

    def __init__(self, key: int):
        self.key = key
        self.left: Node | None = None
        self.right: Node | None = None
        self.parent: Node | None = None


# === 후속자와 선행자 ===

def tree_minimum(x: Node) -> Node:
    """x를 뿌리로 하는 부분 트리에서 열쇠가 가장 작은 노드를 돌려준다."""
    while x.left is not None:
        x = x.left
    return x


def tree_maximum(x: Node) -> Node:
    """x를 뿌리로 하는 부분 트리에서 열쇠가 가장 큰 노드를 돌려준다."""
    while x.right is not None:
        x = x.right
    return x


def successor(x: Node) -> Node | None:
    """노드 x의 중위 후속자를 돌려주고, x가 최댓값이면 None을 돌려준다."""
    if x.right is not None:
        return tree_minimum(x.right)
    y = x.parent
    while y is not None and x == y.right:
        x = y
        y = y.parent
    return y


def predecessor(x: Node) -> Node | None:
    """노드 x의 중위 선행자를 돌려주고, x가 최솟값이면 None을 돌려준다."""
    if x.left is not None:
        return tree_maximum(x.left)
    y = x.parent
    while y is not None and x == y.left:
        x = y
        y = y.parent
    return y
```

### 부모 포인터가 없을 때

노드에 부모 포인터가 없으면 뿌리에서부터 찾아 후속자를 얻는다.

```python
"""부모 포인터 없이 찾는 이진 탐색 트리의 후속자 (위에서 아래로)."""


# === 위에서 아래로 찾는 후속자 ===

def successor_no_parent(root: Node | None, key: int) -> Node | None:
    """주어진 열쇠를 가진 노드의 중위 후속자를 찾는다."""
    successor_node = None
    current = root
    while current is not None:
        if key < current.key:
            successor_node = current  # 후속자 후보
            current = current.left
        elif key > current.key:
            current = current.right
        else:
            # 표적 열쇠를 가진 노드를 찾음
            if current.right is not None:
                return tree_minimum(current.right)
            return successor_node
    return None  # 열쇠를 못 찾음


# === 시연 ===

def insert(root: Node | None, key: int) -> Node:
    """부모 포인터를 지키며 이진 탐색 트리에 열쇠를 넣는다."""
    new_node = Node(key)
    if root is None:
        return new_node
    parent = None
    current = root
    while current is not None:
        parent = current
        if key < current.key:
            current = current.left
        else:
            current = current.right
    new_node.parent = parent
    if key < parent.key:
        parent.left = new_node
    else:
        parent.right = new_node
    return root


if __name__ == "__main__":
    root = None
    for k in [15, 10, 20, 5, 12, 17, 25, 11]:
        root = insert(root, k)

    # 부모 포인터로 후속자 찾기
    def find_node(root, key):
        while root and root.key != key:
            root = root.left if key < root.key else root.right
        return root

    for key in [5, 10, 11, 12, 15, 17, 20, 25]:
        node = find_node(root, key)
        succ = successor(node)
        pred = predecessor(node)
        succ_key = succ.key if succ else None
        pred_key = pred.key if pred else None
        print(f"key={key:2d}  successor={succ_key}  predecessor={pred_key}")
```

## 복잡도

후속자와 선행자 모두 트리의 높이를 $h$이라 할 때 $O(h)$ 시간에 끝난다. 최악의 경우(치우친 트리)에는 $h = n - 1$이므로 $O(n)$이다. 균형 잡힌 이진 탐색 트리에서는 $h = O(\log n)$이다.

스택 프레임 말고는 공간을 더 쓰지 않는다(여기 실은 반복 판본에서는 공간이 $O(1)$이다).

| 연산 | 시간 | 공간 |
|---|---|---|
| 후속자 (부모 포인터 있음) | $O(h)$ | $O(1)$ |
| 선행자 (부모 포인터 있음) | $O(h)$ | $O(1)$ |
| 후속자 (위에서 아래로) | $O(h)$ | $O(1)$ |

!!! note "모든 후속자를 훑을 때의 분할 상환 비용"
    `successor`을 $n$번 불러 이진 탐색 트리 전체를 정렬된 순서로 훑는 데 걸리는 시간은 $O(nh)$이 아니라 모두 $O(n)$이다. 트리의 변마다 많아야 두 번(내려갈 때 한 번, 올라올 때 한 번) 지나므로 후속자 호출마다 분할 상환 비용이 $O(1)$이다.

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 12. MIT Press.


## 연습문제

**연습문제 1.**
후속자와 선행자에 대해 트리의 높이 $h$으로 시간 복잡도를 밝혀라. $h = O(\log n)$이 되는 것은 언제인가?

??? success "연습문제 1 풀이"
    이 연산은 $O(h)$ 시간이 걸린다. 균형 잡힌 트리에서는 $h = O(\log n)$이다(무작위 삽입 순서에서는 기댓값이 $O(\log n)$이고, 균형 이진 탐색 트리 계열은 이를 보장한다). 한쪽으로 치우친 트리에서는 $h = O(n)$이다(정렬된 순서로 넣은 경우).

---

**연습문제 2.**
열쇠 5, 3, 7, 1, 4, 6, 8을 차례로 넣어 만든 이진 탐색 트리에 후속자와 선행자을(를) 적용하라. 단계마다 보여라.

??? success "연습문제 2 풀이"
    차례로 넣어 트리를 만든 뒤 이 쪽의 연산을 적용한다. 단계마다 트리의 모습을 보이고 들르거나 고친 노드를 표시한다.

---

**연습문제 3.**
이진 탐색 트리의 높이를 $h$이라 할 때 후속자와 선행자이(가) $O(h)$ 시간에 끝남을 증명하라.

??? success "연습문제 3 풀이"
    이 연산은 뿌리에서 잎으로 가는 경로(또는 그 일부)를 따라가며 노드마다 $O(1)$의 일을 한다. 경로의 길이는 많아야 $h$이다. 따라서 모두 $O(h)$이다. $\square$

---

**연습문제 4.**
이진 탐색 트리의 연산은 딥러닝에서 정렬된 데이터를 다루는 일(예를 들어 빔 탐색에서 상위 $k$개의 점수를 유지하는 일)과 어떤 관계가 있는가?

??? success "연습문제 4 풀이"
    빔 탐색은 가장 좋은 가설 $k$개를 유지하므로 새 후보를 넣고 가장 나쁜 것을 빼는 일을 효율적으로 해야 한다. 이진 탐색 트리는 둘 다 $O(\log k)$에 해낸다. 실제로는 더 간단하고 상수가 작은 힙을 즐겨 쓰지만, 이진 탐색 트리는 범위 검색이나 순위 같은 더 풍부한 질의를 지원한다.