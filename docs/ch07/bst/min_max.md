# 최솟값과 최댓값

[이진 탐색 트리 성질](property.md) 덕분에 이진 탐색 트리에서 가장 작은 열쇠와 가장 큰 열쇠를 찾는 일은 간단하다. 왼쪽 부분 트리의 열쇠는 모두 뿌리보다 작거나 같고 오른쪽 부분 트리의 열쇠는 모두 크므로, 최솟값은 가장 왼쪽 경로의 끝에, 최댓값은 가장 오른쪽 경로의 끝에 있다. 이 연산들은 (오른쪽 부분 트리의 최솟값이 필요한) [삭제](deletion.md)와 [후속자·선행자](successor.md) 질의의 바탕이 되는 구성 블록이다.

---

## 1. 최솟값 찾기

가장 작은 열쇠를 찾으려면 뿌리에서 시작하여 왼쪽 자식 포인터를 따라가다 왼쪽 자식이 없는 노드에 닿으면 된다. 그 노드가 최솟값을 담고 있다.

```
         8
        / \
       3   10
      / \    \
     1   6   14
    ^
    minimum (follow left pointers)
```

**왜 통하는가**: 이진 탐색 트리 성질에 따라 왼쪽 부분 트리의 모든 노드는 열쇠가 부모의 열쇠보다 작거나 같다. 왼쪽 포인터를 거듭 따라가면 트리에서 가장 작은 열쇠에 이른다. 어떤 노드에 왼쪽 자식이 없으면 그 부분 트리에 그보다 작은 열쇠가 없으므로 그것이 최솟값이다.

알고리즘은 층마다 많아야 노드 하나를 들르므로 트리의 높이를 $h$이라 할 때 $O(h)$ 시간에 끝난다.

---

## 2. 최댓값 찾기

대칭적으로, 가장 큰 열쇠는 뿌리에서 시작하여 오른쪽 자식 포인터를 따라가다 오른쪽 자식이 없는 노드에 닿으면 찾을 수 있다.

```
         8
        / \
       3   10
      / \    \
     1   6   14
               ^
               maximum (follow right pointers)
```

이것도 $O(h)$ 시간에 끝난다.

---

## 3. 엄밀한 서술

!!! note "정리"
    노드가 $n \geq 1$개이고 높이가 $h$인 이진 탐색 트리에서, 뿌리부터 왼쪽 자식 포인터를 따라가면 가장 작은 열쇠를 $O(h)$ 시간에 찾을 수 있고, 오른쪽 자식 포인터를 따라가면 가장 큰 열쇠를 $O(h)$ 시간에 찾을 수 있다.

**증명**: 최솟값을 보자. $x_0 = \text{root}, x_1 = x_0.\text{left}, x_2 = x_1.\text{left}, \ldots, x_k$이라 하고 $x_k.\text{left} = \text{null}$이라 하자. 이진 탐색 트리 성질에 따라 모든 $i$에 대해 $x_i.\text{key} \geq x_{i+1}.\text{key}$이므로 이 경로 위의 노드 가운데 $x_k$의 열쇠가 가장 작다. 게다가 이 경로 위에 없는 노드 $y$은 어떤 $x_i$의 오른쪽 부분 트리에 있으므로 $y.\text{key} > x_i.\text{key} \geq x_k.\text{key}$이다. 따라서 $x_k.\text{key}$이 전체 최솟값이다. 경로의 변은 많아야 $h$개이므로 알고리즘은 $O(h)$ 시간이 걸린다. 최댓값의 논증은 대칭이다. $\square$

---

## 4. 구현

```python
"""
이진 탐색 트리에서 최솟값과 최댓값 찾기.

이진 탐색 트리의 성질을 써서 가장 작은 열쇠와 가장 큰 열쇠를 찾는
재귀 방법과 반복 방법을 모두 보인다.
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

# === 최솟값 ===

def find_min_iterative(node):
    """왼쪽 포인터를 따라가며 가장 작은 열쇠를 찾는다."""
    if node is None:
        return None
    while node.left is not None:
        node = node.left
    return node

def find_min_recursive(node):
    """재귀로 가장 작은 열쇠를 찾는다."""
    if node is None:
        return None
    if node.left is None:
        return node
    return find_min_recursive(node.left)

# === 최댓값 ===

def find_max_iterative(node):
    """오른쪽 포인터를 따라가며 가장 큰 열쇠를 찾는다."""
    if node is None:
        return None
    while node.right is not None:
        node = node.right
    return node

def find_max_recursive(node):
    """재귀로 가장 큰 열쇠를 찾는다."""
    if node is None:
        return None
    if node.right is None:
        return node
    return find_max_recursive(node.right)

# === 이진 탐색 트리 만들기 ===

def insert(root, key):
    """이진 탐색 트리에 열쇠를 넣는다."""
    if root is None:
        return Node(key)
    if key <= root.key:
        root.left = insert(root.left, key)
    else:
        root.right = insert(root.right, key)
    return root

def inorder(node):
    """열쇠를 정렬된 순서로 내놓는다."""
    if node is not None:
        yield from inorder(node.left)
        yield node.key
        yield from inorder(node.right)

# === 메인 ===

if __name__ == "__main__":
    root = None
    keys = [8, 3, 10, 1, 6, 14, 4, 7, 13]
    for k in keys:
        root = insert(root, k)

    print(f"BST keys (inorder): {list(inorder(root))}")
    print()

    min_node = find_min_iterative(root)
    max_node = find_max_iterative(root)
    print(f"Minimum (iterative): {min_node.key}")
    print(f"Maximum (iterative): {max_node.key}")
    print()

    min_node_r = find_min_recursive(root)
    max_node_r = find_max_recursive(root)
    print(f"Minimum (recursive): {min_node_r.key}")
    print(f"Maximum (recursive): {max_node_r.key}")
    print()

    # 부분 트리의 최솟값과 최댓값
    right_subtree_min = find_min_iterative(root.right)
    left_subtree_max = find_max_iterative(root.left)
    print(f"Min of right subtree (rooted at {root.right.key}): {right_subtree_min.key}")
    print(f"Max of left subtree (rooted at {root.left.key}): {left_subtree_max.key}")
```

**출력:**
```
BST keys (inorder): [1, 3, 4, 6, 7, 8, 10, 13, 14]

Minimum (iterative): 1
Maximum (iterative): 14

Minimum (recursive): 1
Maximum (recursive): 14

Min of right subtree (rooted at 10): 10
Max of left subtree (rooted at 3): 7
```

---

## 연습문제

**연습문제 1.**
최솟값과 최댓값에 대해 트리의 높이 $h$으로 시간 복잡도를 밝혀라. $h = O(\log n)$이 되는 것은 언제인가?

??? success "연습문제 1 풀이"
    이 연산은 $O(h)$ 시간이 걸린다. 균형 잡힌 트리에서는 $h = O(\log n)$이다(무작위 삽입 순서에서는 기댓값이 $O(\log n)$이고, 균형 이진 탐색 트리 계열은 이를 보장한다). 한쪽으로 치우친 트리에서는 $h = O(n)$이다(정렬된 순서로 넣은 경우).

---

**연습문제 2.**
열쇠 5, 3, 7, 1, 4, 6, 8을 차례로 넣어 만든 이진 탐색 트리에 최솟값과 최댓값을(를) 적용하라. 단계마다 보여라.

??? success "연습문제 2 풀이"
    차례로 넣어 트리를 만든 뒤 이 쪽의 연산을 적용한다. 단계마다 트리의 모습을 보이고 들르거나 고친 노드를 표시한다.

---

**연습문제 3.**
이진 탐색 트리의 높이를 $h$이라 할 때 최솟값과 최댓값이(가) $O(h)$ 시간에 끝남을 증명하라.

??? success "연습문제 3 풀이"
    이 연산은 뿌리에서 잎으로 가는 경로(또는 그 일부)를 따라가며 노드마다 $O(1)$의 일을 한다. 경로의 길이는 많아야 $h$이다. 따라서 모두 $O(h)$이다. $\square$

---

**연습문제 4.**
이진 탐색 트리의 연산은 딥러닝에서 정렬된 데이터를 다루는 일(예를 들어 빔 탐색에서 상위 $k$개의 점수를 유지하는 일)과 어떤 관계가 있는가?

??? success "연습문제 4 풀이"
    빔 탐색은 가장 좋은 가설 $k$개를 유지하므로 새 후보를 넣고 가장 나쁜 것을 빼는 일을 효율적으로 해야 한다. 이진 탐색 트리는 둘 다 $O(\log k)$에 해낸다. 실제로는 더 간단하고 상수가 작은 힙을 즐겨 쓰지만, 이진 탐색 트리는 범위 검색이나 순위 같은 더 풍부한 질의를 지원한다.

## 정리하며

이 마당은 최솟값 찾기、최댓값 찾기、엄밀한 서술、구현을 차례로 짚었다.

**참고 문헌**

- [Introduction to Algorithms (CLRS), 12.2절](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
