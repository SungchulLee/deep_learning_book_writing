# 삽입

삽입은 이진 탐색 트리가 자라는 방식이다. 빈 트리에서 시작하여 열쇠를 하나씩 넣어 전체 구조를 세운다. 삽입할 때마다 뿌리에서 내려가며 노드마다 새 열쇠를 견주어 왼쪽으로 갈지 오른쪽으로 갈지 정하고, 빈 자리를 찾으면 그곳에서 새 노드가 잎이 된다. 새 노드는 언제나 잎 자리에 놓이므로 삽입은 기존 노드의 구조를 바꾸지 않고 트리를 아래로 늘리기만 한다.

## 알고리즘

$r$을 뿌리로 하는 이진 탐색 트리에 열쇠 $k$을 넣으려면 다음과 같이 한다.

1. 트리가 비어 있으면 열쇠가 $k$인 새 노드를 만들어 뿌리로 돌려준다.
2. $k$을 $r.\text{key}$과 견준다.
    - $k \leq r.\text{key}$이면 왼쪽 부분 트리에 재귀적으로 넣는다.
    - $k > r.\text{key}$이면 오른쪽 부분 트리에 재귀적으로 넣는다.
3. (바뀌었을 수도 있는) 뿌리를 돌려준다.

재귀가 널 자리에 닿을 때까지 이어지므로 새 노드는 언제나 잎이 된다.

## 그림으로 따라가기

이 이진 탐색 트리에 열쇠 5를 넣어 보자.

```
Step 1: Start at root (8).        Step 2: 5 < 8, go left to 3.
         8                                  8
        / \                                / \
       3   10                             3   10
      / \                                / \
     1   6                              1   6

Step 3: 5 > 3, go right to 6.     Step 4: 5 < 6, go left (null).
         8                                  8
        / \                                / \
       3   10                             3   10
      / \                                / \
     1   6                              1   6
                                           /
                                          5   <- new leaf
```

## 이진 탐색 트리 성질 지키기

!!! note "올바름"
    삽입은 [이진 탐색 트리 성질](property.md)을 지킨다. 단계마다 알고리즘은 순서 불변식을 지키는 부분 트리를 고른다. 지금 노드보다 작거나 같은 열쇠는 왼쪽으로, 큰 열쇠는 오른쪽으로 간다. 새 잎이 놓인 자리 덕분에 모든 조상의 이진 탐색 트리 성질이 그대로 남는다.

## 복잡도

삽입은 뿌리에서 잎까지의 경로 하나를 따라가며 노드마다 $O(1)$의 일을 한다. 트리의 높이를 $h$이라 할 때 시간 복잡도는 $O(h)$이다.

- **균형 잡힌 트리**: $O(\log n)$
- **치우친 트리**: $O(n)$

삽입 순서가 트리의 높이에 어떻게 영향을 주는지는 [복잡도](complexity.md)에서 자세히 다룬다.

## 재귀 구현과 반복 구현

재귀 방식은 우아하고 알고리즘 설명을 그대로 옮긴 것이다. 반복 방식은 스택 부담을 없애 주며, 트리가 아주 깊을 수 있을 때 (스택 넘침을 막으려고) 즐겨 쓴다.

```python
"""
이진 탐색 트리의 삽입: 재귀 구현과 반복 구현.

삽입이 이진 탐색 트리의 성질을 지키면서 언제나 새 잎을 만드는 방식을
보이고, 순차열로 트리를 만드는 과정을 시연한다.
"""


# === 노드 정의 ===

class Node:
    """이진 탐색 트리의 노드."""

    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None

    def __repr__(self):
        return f"Node({self.key})"


# === 재귀 삽입 ===

def insert_recursive(root, key):
    """재귀로 이진 탐색 트리에 열쇠를 넣는다.

    고친 트리의 뿌리를 돌려준다.
    새 노드는 언제나 잎으로 만들어진다.
    """
    if root is None:
        return Node(key)
    if key <= root.key:
        root.left = insert_recursive(root.left, key)
    else:
        root.right = insert_recursive(root.right, key)
    return root


# === 반복 삽입 ===

def insert_iterative(root, key):
    """반복으로 이진 탐색 트리에 열쇠를 넣는다.

    트리를 내려가며 알맞은 널 자리를 찾은 뒤
    새 노드를 붙인다.
    """
    new_node = Node(key)
    if root is None:
        return new_node

    parent = None
    current = root
    while current is not None:
        parent = current
        if key <= current.key:
            current = current.left
        else:
            current = current.right

    if key <= parent.key:
        parent.left = new_node
    else:
        parent.right = new_node

    return root


# === 보이기 도우미 ===

def inorder(node):
    """열쇠를 정렬된 순서로 내놓는다."""
    if node is not None:
        yield from inorder(node.left)
        yield node.key
        yield from inorder(node.right)


def print_tree(node, level=0, prefix="Root: "):
    """들여쓰기로 트리의 짜임을 출력한다."""
    if node is not None:
        print(" " * (level * 4) + prefix + str(node.key))
        if node.left is not None or node.right is not None:
            print_tree(node.left, level + 1, "L--- ")
            print_tree(node.right, level + 1, "R--- ")


# === 메인 ===

if __name__ == "__main__":
    # 열쇠를 하나씩 넣어 이진 탐색 트리 만들기
    keys = [8, 3, 10, 1, 6, 14, 4, 7, 13]

    root = None
    for k in keys:
        root = insert_recursive(root, k)
        print(f"Insert {k}: inorder = {list(inorder(root))}")

    print()
    print("Final tree structure:")
    print_tree(root)

    # 반복 판본도 같은 결과를 내는지 확인
    root2 = None
    for k in keys:
        root2 = insert_iterative(root2, k)
    print()
    print(f"Iterative inorder: {list(inorder(root2))}")
    print(f"Results match: {list(inorder(root)) == list(inorder(root2))}")
```

**출력:**
```
Insert 8: inorder = [8]
Insert 3: inorder = [3, 8]
Insert 10: inorder = [3, 8, 10]
Insert 1: inorder = [1, 3, 8, 10]
Insert 6: inorder = [1, 3, 6, 8, 10]
Insert 14: inorder = [1, 3, 6, 8, 10, 14]
Insert 4: inorder = [1, 3, 4, 6, 8, 10, 14]
Insert 7: inorder = [1, 3, 4, 6, 7, 8, 10, 14]
Insert 13: inorder = [1, 3, 4, 6, 7, 8, 10, 13, 14]

Final tree structure:
Root: 8
    L--- 3
        L--- 1
        R--- 6
            L--- 4
            R--- 7
    R--- 10
        R--- 14
            L--- 13

Iterative inorder: [1, 3, 4, 6, 7, 8, 10, 13, 14]
Results match: True
```

## 참고 문헌

- [Introduction to Algorithms (CLRS), 12.3절](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)


## 연습문제

**연습문제 1.**
삽입에 대해 트리의 높이 $h$으로 시간 복잡도를 밝혀라. $h = O(\log n)$이 되는 것은 언제인가?

??? success "연습문제 1 풀이"
    이 연산은 $O(h)$ 시간이 걸린다. 균형 잡힌 트리에서는 $h = O(\log n)$이다(무작위 삽입 순서에서는 기댓값이 $O(\log n)$이고, 균형 이진 탐색 트리 계열은 이를 보장한다). 한쪽으로 치우친 트리에서는 $h = O(n)$이다(정렬된 순서로 넣은 경우).

---

**연습문제 2.**
열쇠 5, 3, 7, 1, 4, 6, 8을 차례로 넣어 만든 이진 탐색 트리에 삽입을(를) 적용하라. 단계마다 보여라.

??? success "연습문제 2 풀이"
    차례로 넣어 트리를 만든 뒤 이 쪽의 연산을 적용한다. 단계마다 트리의 모습을 보이고 들르거나 고친 노드를 표시한다.

---

**연습문제 3.**
이진 탐색 트리의 높이를 $h$이라 할 때 삽입이(가) $O(h)$ 시간에 끝남을 증명하라.

??? success "연습문제 3 풀이"
    이 연산은 뿌리에서 잎으로 가는 경로(또는 그 일부)를 따라가며 노드마다 $O(1)$의 일을 한다. 경로의 길이는 많아야 $h$이다. 따라서 모두 $O(h)$이다. $\square$

---

**연습문제 4.**
이진 탐색 트리의 연산은 딥러닝에서 정렬된 데이터를 다루는 일(예를 들어 빔 탐색에서 상위 $k$개의 점수를 유지하는 일)과 어떤 관계가 있는가?

??? success "연습문제 4 풀이"
    빔 탐색은 가장 좋은 가설 $k$개를 유지하므로 새 후보를 넣고 가장 나쁜 것을 빼는 일을 효율적으로 해야 한다. 이진 탐색 트리는 둘 다 $O(\log k)$에 해낸다. 실제로는 더 간단하고 상수가 작은 힙을 즐겨 쓰지만, 이진 탐색 트리는 범위 검색이나 순위 같은 더 풍부한 질의를 지원한다.