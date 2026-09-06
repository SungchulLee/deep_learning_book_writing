# 연결 표현

연결 표현은 실제로 이진 트리를 구현하는 가장 흔한 방법이다. 트리가 완전해야 하는 [배열 표현](array.md)과 달리, 연결된 노드는 균형 잡힌 것이든 치우친 것이든 그 사이 어떤 모양이든 나타낼 수 있다. 노드마다 메모리에 따로 있는 객체이며 참조로 자식과(그리고 원한다면 부모와) 이어진다. 이 유연함 덕분에 연결 표현이 이진 탐색 트리와 식 트리, 이 책에서 다루는 대부분의 트리 알고리즘에서 기본 선택이 된다.

## 노드 구조

연결 표현에서 이진 트리의 노드는 적어도 다음 세 칸을 담는다.

- **열쇠**(또는 데이터): 이 노드가 담은 값
- **left**: 왼쪽 자식에 대한 참조 (없으면 널)
- **right**: 오른쪽 자식에 대한 참조 (없으면 널)

어떤 구현은 넷째 칸을 더한다.

- **parent**: 부모 노드에 대한 참조 (뿌리에서는 널)

부모 포인터는 있어도 없어도 되지만, 노드의 후속자를 찾거나 잎에서 뿌리로 거슬러 올라가는 일을 간단하게 해 준다.

!!! note "이진 트리와 일반 트리"
    **이진 트리**에서는 노드마다 자식이 많아야 둘이므로 `left`과 `right` 포인터면 충분하다. 자식의 수에 제한이 없는 **일반 트리**에서는 **왼쪽 자식, 오른쪽 형제** 표현을 흔히 쓴다. `left`은 첫 자식을 가리키고 `right`은 다음 형제를 가리킨다. 이렇게 하면 노드마다 포인터 두 개만으로 뿌리 있는 어떤 트리든 나타낼 수 있다.

## 공간 분석

연결 표현에서 노드마다 다음이 필요하다.

$$
\text{Space per node} = \text{size(key)} + 2 \times \text{size(pointer)}
$$

부모 포인터를 두면 $\text{size(key)} + 3 \times \text{size(pointer)}$이 된다. 노드가 $n$개인 트리에서 전체 공간은 트리의 모양과 상관없이 $\Theta(n)$이다. 반면 배열 표현이 $\Theta(n)$의 공간을 쓰는 것은 완전 트리일 때뿐이다. 높이가 $n - 1$인 치우친 트리는 크기가 $2^n - 1$인 배열이 필요하여 지수적으로 공간을 낭비한다.

| 트리의 모양 | 연결 표현의 공간 | 배열 표현의 공간 |
|---|---|---|
| 완전 ($h = \lfloor \log_2 n \rfloor$) | $\Theta(n)$ | $\Theta(n)$ |
| 치우침 ($h = n - 1$) | $\Theta(n)$ | $\Theta(2^n)$ |

## 트리 만들기

노드를 하나씩 만들어 `left`과 `right` 칸으로 이으면 트리가 된다. 트리에는 뿌리 노드에 대한 참조로 닿는다.

```python
"""
이진 트리의 연결 표현.

노드 만들기, 트리 세우기, 연결(포인터 기반) 표현으로 하는
기본적인 순회를 보인다.
"""


# === 노드 정의 ===

class Node:
    """연결 이진 트리의 노드.

    속성:
        key: 이 노드에 담긴 값.
        left: 왼쪽 자식에 대한 참조 (없으면 None).
        right: 오른쪽 자식에 대한 참조 (없으면 None).
        parent: 부모 노드에 대한 참조 (뿌리에서는 None).
    """

    def __init__(self, key, left=None, right=None):
        self.key = key
        self.left = left
        self.right = right
        self.parent = None

    def __repr__(self):
        return f"Node({self.key})"


# === 트리 만들기 도우미 ===

def build_tree(key, left=None, right=None):
    """노드를 만들고 자식의 부모 포인터를 지정한다."""
    node = Node(key, left, right)
    if left is not None:
        left.parent = node
    if right is not None:
        right.parent = node
    return node


def tree_size(node):
    """node를 뿌리로 하는 부분 트리의 노드 수를 돌려준다."""
    if node is None:
        return 0
    return 1 + tree_size(node.left) + tree_size(node.right)


def tree_height(node):
    """부분 트리의 높이를 돌려준다 (변의 수)."""
    if node is None:
        return -1
    return 1 + max(tree_height(node.left), tree_height(node.right))


# === 보이기 ===

def print_tree(node, level=0, prefix="Root: "):
    """들여쓰기로 트리의 짜임을 출력한다."""
    if node is not None:
        print(" " * (level * 4) + prefix + str(node.key))
        if node.left is not None or node.right is not None:
            print_tree(node.left, level + 1, "L--- ")
            print_tree(node.right, level + 1, "R--- ")


# === 메인 ===

if __name__ == "__main__":
    # 예제 트리 만들기:
    #        10
    #       /  \
    #      5    15
    #     / \     \
    #    3   7    20
    tree = build_tree(10,
        build_tree(5,
            build_tree(3),
            build_tree(7)),
        build_tree(15,
            None,
            build_tree(20)))

    print_tree(tree)
    print()
    print(f"Size:   {tree_size(tree)} nodes")
    print(f"Height: {tree_height(tree)} edges")
    print()

    # 부모 포인터 보이기
    node_7 = tree.left.right
    print(f"Node {node_7.key}'s parent: {node_7.parent}")
    print(f"Node {node_7.parent.key}'s parent: {node_7.parent.parent}")
```

**출력:**
```
Root: 10
    L--- 5
        L--- 3
        R--- 7
    R--- 15
        L--- None
        R--- 20

Size:   6 nodes
Height: 2 edges

Node 7's parent: Node(5)
Node 5's parent: Node(10)
```

## 왼쪽 자식 오른쪽 형제 표현

일반(이진이 아닌) 트리에서 **왼쪽 자식 오른쪽 형제** 부호화는 노드마다 포인터 두 개만으로 차수에 제한이 없는 트리를 담는다.

- **left_child**: 노드의 첫(가장 왼쪽) 자식을 가리킨다
- **right_sibling**: 노드의 다음 형제를 가리킨다

이렇게 하면 뿌리 있는 어떤 숲도 이진 트리로 바뀐다. 노드가 자식을 $k$개 가질 수 있는 일반 트리에서 `left_child`을 따라 내려가면 첫 자식에 닿고, 이어 `right_sibling`을 따라가면 모든 형제를 훑을 수 있다.

```
General tree:          Left-child right-sibling:
      A                     A
    / | \                  /
   B  C  D               B --> C --> D
  / \                    /
 E   F                  E --> F
```

## 참고 문헌

- [Introduction to Algorithms (CLRS), 12장](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)


## 연습문제

**연습문제 1.**
연결 표현에 대해 트리의 높이 $h$으로 시간 복잡도를 밝혀라. $h = O(\log n)$이 되는 것은 언제인가?

??? success "연습문제 1 풀이"
    이 연산은 $O(h)$ 시간이 걸린다. 균형 잡힌 트리에서는 $h = O(\log n)$이다(무작위 삽입 순서에서는 기댓값이 $O(\log n)$이고, 균형 이진 탐색 트리 계열은 이를 보장한다). 한쪽으로 치우친 트리에서는 $h = O(n)$이다(정렬된 순서로 넣은 경우).

---

**연습문제 2.**
열쇠 5, 3, 7, 1, 4, 6, 8을 차례로 넣어 만든 이진 탐색 트리에 연결 표현을(를) 적용하라. 단계마다 보여라.

??? success "연습문제 2 풀이"
    차례로 넣어 트리를 만든 뒤 이 쪽의 연산을 적용한다. 단계마다 트리의 모습을 보이고 들르거나 고친 노드를 표시한다.

---

**연습문제 3.**
이진 탐색 트리의 높이를 $h$이라 할 때 연결 표현이(가) $O(h)$ 시간에 끝남을 증명하라.

??? success "연습문제 3 풀이"
    이 연산은 뿌리에서 잎으로 가는 경로(또는 그 일부)를 따라가며 노드마다 $O(1)$의 일을 한다. 경로의 길이는 많아야 $h$이다. 따라서 모두 $O(h)$이다. $\square$

---

**연습문제 4.**
이진 탐색 트리의 연산은 딥러닝에서 정렬된 데이터를 다루는 일(예를 들어 빔 탐색에서 상위 $k$개의 점수를 유지하는 일)과 어떤 관계가 있는가?

??? success "연습문제 4 풀이"
    빔 탐색은 가장 좋은 가설 $k$개를 유지하므로 새 후보를 넣고 가장 나쁜 것을 빼는 일을 효율적으로 해야 한다. 이진 탐색 트리는 둘 다 $O(\log k)$에 해낸다. 실제로는 더 간단하고 상수가 작은 힙을 즐겨 쓰지만, 이진 탐색 트리는 범위 검색이나 순위 같은 더 풍부한 질의를 지원한다.