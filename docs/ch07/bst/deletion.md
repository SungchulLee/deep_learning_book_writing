# 삭제

삭제는 이진 탐색 트리 연산 가운데 가장 손이 많이 간다. 노드를 없애면서도 트리 전체의 [이진 탐색 트리 성질](property.md)을 지켜야 하기 때문이다. 언제나 잎을 더하는 [삽입](insertion.md)과 달리, 삭제는 대상 노드에 자식이 있으면 구조를 다시 짜야 할 수도 있다. 알고리즘은 지울 노드의 자식 수에 따라 세 경우를 다룬다.

## 세 가지 경우

지울 노드를 $z$이라 하자.

### 경우 1: 잎 노드 (자식 없음)

$z$에 자식이 없으면 부모의 해당 자식 포인터를 널로 두어 그냥 없앤다.

```
Delete 4:
     5              5
    / \            / \
   3   7    ->    3   7
  /
 4
```

### 경우 2: 자식 하나

$z$에 자식이 꼭 하나 있으면 $z$을 그 자식으로 바꾼다. 자식의 부분 트리가 "올라와" $z$의 자리를 차지한다.

```
Delete 3 (has only left child 1):
     5              5
    / \            / \
   3   7    ->    1   7
  /
 1
```

### 경우 3: 자식 둘

$z$에 자식이 둘이면 그냥 없앴다가는 트리가 끊어진다. 대신 $z$의 **중위 후속자** $y$, 곧 $z$의 오른쪽 부분 트리에서 열쇠가 가장 작은 노드를 찾아 $z$을 그것으로 바꾼다.

중위 후속자 $y$은 $z$의 오른쪽 자식으로 간 뒤 왼쪽 포인터를 따라가 왼쪽 자식이 없는 노드에 닿으면 찾을 수 있다. $y$에는 왼쪽 자식이 없으므로 $y$을 그 자리에서 없애는 일은 경우 1이나 경우 2에 해당한다.

절차는 다음과 같다.

1. $z$의 [후속자](successor.md) $y$을 찾는다($z$의 오른쪽 부분 트리에서 가장 작은 노드).
2. $z$의 열쇠를 $y$의 열쇠로 바꾼다.
3. $y$을 원래 자리에서 지운다(경우 1이나 경우 2이다).

```
Delete 5 (has two children):
     5              6
    / \            / \
   3   8    ->    3   8
  / \ /         / \
 1  4 6        1   4
      \              \
       7              7
```

여기서 5의 중위 후속자는 6이다. 6을 뿌리 자리에 옮겨 적은 뒤 원래의 노드 6(자식이 하나, 7이다)을 지운다.

!!! note "후속자와 선행자"
    중위 후속자 대신 **중위 선행자**(왼쪽 부분 트리에서 가장 큰 노드)를 써도 된다. 두 방법 모두 이진 탐색 트리의 성질을 지킨다. 어떤 구현은 평균적으로 트리를 더 균형 있게 두려고 둘을 번갈아 쓴다.

## transplant 도우미

CLRS의 방법은 한 부분 트리를 다른 부분 트리로 바꾸는 `transplant` 부프로그램을 쓴다. `transplant(T, u, v)`은 $u$의 부모가 $v$을 가리키도록 고쳐 $u$을 뿌리로 하는 부분 트리를 $v$을 뿌리로 하는 부분 트리로 바꾼다.

## 복잡도

삭제는 뿌리에서 잎까지의 경로를 많아야 두 번 훑으므로(노드를 찾는 데 한 번, 후속자를 찾는 데 한 번) 트리의 높이를 $h$이라 할 때 $O(h)$ 시간에 끝난다. 균형 잡힌 이진 탐색 트리에서는 $O(\log n)$, 치우친 트리에서는 $O(n)$이다.

## 구현

```python
"""
세 경우를 모두 다루는 이진 탐색 트리의 삭제.

표준적인 이진 탐색 트리 삭제 알고리즘을 구현한다. 잎 없애기,
자식 하나인 노드 갈아 끼우기, 그리고 중위 후속자로 자식 둘인 노드
갈아 끼우기이다.
"""


# === 노드 정의 ===

class Node:
    """이진 탐색 트리의 노드."""

    def __init__(self, key, left=None, right=None):
        self.key = key
        self.left = left
        self.right = right

    def __repr__(self):
        return f"Node({self.key})"


# === 이진 탐색 트리 연산 ===

def insert(root, key):
    """이진 탐색 트리에 열쇠를 넣는다."""
    if root is None:
        return Node(key)
    if key <= root.key:
        root.left = insert(root.left, key)
    else:
        root.right = insert(root.right, key)
    return root


def find_min(node):
    """부분 트리에서 열쇠가 가장 작은 노드를 찾는다."""
    while node.left is not None:
        node = node.left
    return node


def delete(root, key):
    """주어진 열쇠를 가진 노드를 이진 탐색 트리에서 지운다.

    세 경우를 다룬다.
      1. 잎 노드: 곧바로 없앤다.
      2. 자식 하나: 자식으로 갈아 끼운다.
      3. 자식 둘: 중위 후속자로 갈아 끼운다.
    """
    if root is None:
        return None

    if key < root.key:
        root.left = delete(root.left, key)
    elif key > root.key:
        root.right = delete(root.right, key)
    else:
        # 지울 노드를 찾음
        # 경우 1과 2: 자식이 없거나 하나
        if root.left is None:
            return root.right
        if root.right is None:
            return root.left

        # 경우 3: 자식이 둘
        # 중위 후속자 찾기 (오른쪽 부분 트리의 최솟값)
        successor = find_min(root.right)
        root.key = successor.key
        root.right = delete(root.right, successor.key)

    return root


# === 보이기 ===

def inorder(node):
    """열쇠를 정렬된 순서로 내놓는다."""
    if node is not None:
        yield from inorder(node.left)
        yield node.key
        yield from inorder(node.right)


def print_tree(node, level=0, prefix="Root: "):
    """트리의 짜임을 출력한다."""
    if node is not None:
        print(" " * (level * 4) + prefix + str(node.key))
        if node.left is not None or node.right is not None:
            print_tree(node.left, level + 1, "L--- ")
            print_tree(node.right, level + 1, "R--- ")


# === 메인 ===

if __name__ == "__main__":
    root = None
    for k in [5, 3, 8, 1, 4, 6, 9, 7]:
        root = insert(root, k)

    print("Original tree:")
    print_tree(root)
    print(f"Inorder: {list(inorder(root))}")
    print()

    # 경우 1: 잎 지우기
    root = delete(root, 4)
    print("After deleting 4 (leaf):")
    print(f"Inorder: {list(inorder(root))}")
    print()

    # 경우 2: 자식이 하나인 노드 지우기
    root = delete(root, 6)
    print("After deleting 6 (one child: 7):")
    print(f"Inorder: {list(inorder(root))}")
    print()

    # 경우 3: 자식이 둘인 노드 지우기
    root = delete(root, 5)
    print("After deleting 5 (two children, successor=7):")
    print_tree(root)
    print(f"Inorder: {list(inorder(root))}")
```

**출력:**
```
Original tree:
Root: 5
    L--- 3
        L--- 1
        R--- 4
    R--- 8
        L--- 6
            R--- 7
        R--- 9
Inorder: [1, 3, 4, 5, 6, 7, 8, 9]

After deleting 4 (leaf):
Inorder: [1, 3, 5, 6, 7, 8, 9]

After deleting 6 (one child: 7):
Inorder: [1, 3, 5, 7, 8, 9]

After deleting 5 (two children, successor=7):
Root: 7
    L--- 3
        L--- 1
    R--- 8
        R--- 9
Inorder: [1, 3, 7, 8, 9]
```

## 참고 문헌

- [Introduction to Algorithms (CLRS), 12.3절](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)


## 연습문제

**연습문제 1.**
삭제에 대해 트리의 높이 $h$으로 시간 복잡도를 밝혀라. $h = O(\log n)$이 되는 것은 언제인가?

??? success "연습문제 1 풀이"
    이 연산은 $O(h)$ 시간이 걸린다. 균형 잡힌 트리에서는 $h = O(\log n)$이다(무작위 삽입 순서에서는 기댓값이 $O(\log n)$이고, 균형 이진 탐색 트리 계열은 이를 보장한다). 한쪽으로 치우친 트리에서는 $h = O(n)$이다(정렬된 순서로 넣은 경우).

---

**연습문제 2.**
열쇠 5, 3, 7, 1, 4, 6, 8을 차례로 넣어 만든 이진 탐색 트리에 삭제을(를) 적용하라. 단계마다 보여라.

??? success "연습문제 2 풀이"
    차례로 넣어 트리를 만든 뒤 이 쪽의 연산을 적용한다. 단계마다 트리의 모습을 보이고 들르거나 고친 노드를 표시한다.

---

**연습문제 3.**
이진 탐색 트리의 높이를 $h$이라 할 때 삭제이(가) $O(h)$ 시간에 끝남을 증명하라.

??? success "연습문제 3 풀이"
    이 연산은 뿌리에서 잎으로 가는 경로(또는 그 일부)를 따라가며 노드마다 $O(1)$의 일을 한다. 경로의 길이는 많아야 $h$이다. 따라서 모두 $O(h)$이다. $\square$

---

**연습문제 4.**
이진 탐색 트리의 연산은 딥러닝에서 정렬된 데이터를 다루는 일(예를 들어 빔 탐색에서 상위 $k$개의 점수를 유지하는 일)과 어떤 관계가 있는가?

??? success "연습문제 4 풀이"
    빔 탐색은 가장 좋은 가설 $k$개를 유지하므로 새 후보를 넣고 가장 나쁜 것을 빼는 일을 효율적으로 해야 한다. 이진 탐색 트리는 둘 다 $O(\log k)$에 해낸다. 실제로는 더 간단하고 상수가 작은 힙을 즐겨 쓰지만, 이진 탐색 트리는 범위 검색이나 순위 같은 더 풍부한 질의를 지원한다.