# 증강 이진 탐색 트리

보통의 이진 탐색 트리는 열쇠 값에 따른 찾기와 삽입과 삭제를 지원한다. 그런데 "다섯째로 작은 원소는 무엇인가?"나 "구간 $[a, b]$에 드는 원소는 몇 개인가?" 같은, 그냥 이진 탐색 트리로는 효율적으로 답할 수 없는 질의를 필요로 하는 응용이 많다. **증강 이진 탐색 트리**는 노드마다 추가 정보를 두고 삽입과 삭제 중에 이를 유지하여, 트리의 높이를 $h$이라 할 때 이런 풍부한 질의를 $O(h)$ 시간에 답한다.

## 증강하는 방법

이진 탐색 트리를 증강하는 일반적인 전략은 네 단계이다.

1. 노드마다 담을 **추가 정보를 고른다**.
2. 삽입과 삭제와 회전 중에 그 정보를 점근적 비용을 늘리지 않고 유지할 수 있는지 **확인한다**.
3. 추가 정보를 쓰는 **새 연산을 설계한다**.
4. 유지 절차의 **올바름을 증명한다**.

!!! tip "증강 정리 (CLRS)"
    노드마다 담은 추가 정보를 그 노드 자신의 데이터와 두 자식의 증강 정보로 계산할 수 있다면, 그 정보는 삽입과 삭제와 회전 중에 점근적 부담 없이 $O(h)$ 시간에 유지할 수 있다.

## 순서 통계 트리

가장 흔한 증강은 노드마다 **부분 트리의 크기**를 담아 **순서 통계 트리**를 만드는 것이다. 노드 $x$마다 다음을 담는다.

$$
x.\text{size} = 1 + x.\text{left}.\text{size} + x.\text{right}.\text{size}
$$

널 노드의 크기는 0으로 둔다. 이 한 칸만 더해도 강력한 연산 두 가지가 가능해진다.

### 선택: $k$번째로 작은 원소 찾기

순위 $k$이 주어지면 `select(root, k)`은 $k$번째로 작은 열쇠를 가진 노드를 돌려준다. 알고리즘은 $k$을 왼쪽 부분 트리의 크기와 견주며 나아간다.

- $r = x.\text{left}.\text{size} + 1$이라 하자($x$이 자기 부분 트리 안에서 갖는 순위).
- $k = r$이면 $x$을 돌려준다.
- $k < r$이면 왼쪽 부분 트리로 내려간다.
- $k > r$이면 순위 $k - r$으로 오른쪽 부분 트리에 내려간다.

이 연산은 $O(h)$ 시간에 끝난다.

### 순위: 주어진 열쇠의 순위 찾기

`rank(root, key)`은 트리 안에서 주어진 열쇠보다 작거나 같은 열쇠의 개수를 돌려준다. 뿌리에서 시작하여 내려가며 다음을 한다.

- 왼쪽으로 가면 순위는 그대로이다.
- 노드 $x$을 들르거나 지나 오른쪽으로 가면 지금까지의 순위에 $x.\text{left}.\text{size} + 1$을 더한다.

이것도 $O(h)$ 시간에 끝난다.

## 부분 트리 크기 유지하기

**삽입**할 때는 삽입 경로 위의 모든 조상의 크기를 1씩 늘린다. **삭제**할 때는 지운 노드에서 뿌리까지의 경로 위에서 크기를 1씩 줄인다.

(AVL이나 레드-블랙 트리 같은 균형 이진 탐색 트리에서 쓰는) **회전** 중에는 노드 두 개만 부분 트리 관계가 바뀐다.

```
Right rotation at y:          Left rotation at x:
      y            x              x            y
     / \          / \            / \          / \
    x   C  -->  A   y          A   y   -->  x   C
   / \             / \            / \      / \
  A   B           B   C         B   C    A   B
```

$y$에서 오른쪽 회전을 한 뒤에는 다음과 같다.

$$
x.\text{size} = y.\text{size} \quad (\text{x now roots the same subtree})
$$

$$
y.\text{size} = y.\text{left}.\text{size} + y.\text{right}.\text{size} + 1
$$

회전에 관여한 두 노드의 크기만 고치면 되므로 회전은 여전히 $O(1)$이다.

## 그 밖의 증강

부분 트리 크기 증강이 가장 흔하지만, 쓸모 있는 증강에는 다음도 있다.

| 증강 | 노드마다 담는 것 | 가능해지는 일 |
|---|---|---|
| 부분 트리 크기 | $1 + \text{left.size} + \text{right.size}$ | 선택, 순위 |
| 부분 트리 최소/최대 | $\min(\text{key}, \text{left.min}, \text{right.min})$ | 구간 최솟값 질의 |
| 부분 트리 합 | $\text{key} + \text{left.sum} + \text{right.sum}$ | 구간 합 질의 |
| 구간의 최대 끝점 | $\max(\text{high}, \text{left.max}, \text{right.max})$ | 구간 겹침 질의 |

## 예

```python
"""
순서 통계 트리: 부분 트리의 크기로 증강한 이진 탐색 트리.

선택(k번째로 작은 것 찾기)과 순위(어떤 열쇠 이하인 원소 세기)를
$O(h)$ 시간에 지원한다.
"""


# === 노드 정의 ===

class Node:
    """부분 트리의 크기로 증강한 이진 탐색 트리 노드."""

    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
        self.size = 1  # 이 노드와 모든 자손을 센다


# === 크기 도우미 ===

def size(node):
    """부분 트리의 크기를 돌려준다 (널이면 0)."""
    return node.size if node else 0


# === 크기를 유지하는 삽입 ===

def insert(root, key):
    """열쇠를 넣고 경로 위의 부분 트리 크기를 갱신한다."""
    if root is None:
        return Node(key)
    if key <= root.key:
        root.left = insert(root.left, key)
    else:
        root.right = insert(root.right, key)
    root.size = 1 + size(root.left) + size(root.right)
    return root


# === 선택: k번째로 작은 것 찾기 (1부터 셈) ===

def select(node, k):
    """k번째로 작은 열쇠를 가진 노드를 돌려준다.

    k는 1부터 센다. select(root, 1)은 최솟값을 돌려준다.
    """
    if node is None:
        return None
    left_size = size(node.left)
    rank_of_node = left_size + 1
    if k == rank_of_node:
        return node
    elif k < rank_of_node:
        return select(node.left, k)
    else:
        return select(node.right, k - rank_of_node)


# === 순위: 주어진 열쇠 이하인 열쇠의 수 ===

def rank(node, key):
    """key보다 작거나 같은 열쇠의 개수를 돌려준다."""
    if node is None:
        return 0
    if key < node.key:
        return rank(node.left, key)
    elif key > node.key:
        return size(node.left) + 1 + rank(node.right, key)
    else:
        return size(node.left) + 1


# === 중위 순회 ===

def inorder(node):
    """(열쇠, 크기) 쌍을 정렬된 순서로 내놓는다."""
    if node is not None:
        yield from inorder(node.left)
        yield (node.key, node.size)
        yield from inorder(node.right)


# === 메인 ===

if __name__ == "__main__":
    root = None
    keys = [15, 6, 18, 3, 7, 17, 20, 2, 4, 13, 9]
    for k in keys:
        root = insert(root, k)

    print("Inorder (key, subtree_size):")
    print(f"  {list(inorder(root))}")
    print()

    for k in [1, 3, 5, 7, 11]:
        result = select(root, k)
        print(f"  Select(k={k}): {result.key if result else None}")

    print()
    for key in [1, 6, 13, 18, 25]:
        print(f"  Rank(key={key}): {rank(root, key)}")
```

**출력:**
```
Inorder (key, subtree_size):
  [(2, 1), (3, 2), (4, 1), (6, 4), (7, 1), (9, 1), (13, 3), (15, 11), (17, 1), (18, 3), (20, 1)]

  Select(k=1): 2
  Select(k=3): 4
  Select(k=5): 7
  Select(k=7): 13
  Select(k=11): 20

  Rank(key=1): 0
  Rank(key=6): 4
  Rank(key=13): 7
  Rank(key=18): 10
  Rank(key=25): 11
```

## 참고 문헌

- [Introduction to Algorithms (CLRS), 14장 — 자료 구조 증강하기](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)


## 연습문제

**연습문제 1.**
증강 이진 탐색 트리에 대해 트리의 높이 $h$으로 시간 복잡도를 밝혀라. $h = O(\log n)$이 되는 것은 언제인가?

??? success "연습문제 1 풀이"
    이 연산은 $O(h)$ 시간이 걸린다. 균형 잡힌 트리에서는 $h = O(\log n)$이다(무작위 삽입 순서에서는 기댓값이 $O(\log n)$이고, 균형 이진 탐색 트리 계열은 이를 보장한다). 한쪽으로 치우친 트리에서는 $h = O(n)$이다(정렬된 순서로 넣은 경우).

---

**연습문제 2.**
열쇠 5, 3, 7, 1, 4, 6, 8을 차례로 넣어 만든 이진 탐색 트리에 증강 이진 탐색 트리을(를) 적용하라. 단계마다 보여라.

??? success "연습문제 2 풀이"
    차례로 넣어 트리를 만든 뒤 이 쪽의 연산을 적용한다. 단계마다 트리의 모습을 보이고 들르거나 고친 노드를 표시한다.

---

**연습문제 3.**
이진 탐색 트리의 높이를 $h$이라 할 때 증강 이진 탐색 트리이(가) $O(h)$ 시간에 끝남을 증명하라.

??? success "연습문제 3 풀이"
    이 연산은 뿌리에서 잎으로 가는 경로(또는 그 일부)를 따라가며 노드마다 $O(1)$의 일을 한다. 경로의 길이는 많아야 $h$이다. 따라서 모두 $O(h)$이다. $\square$

---

**연습문제 4.**
이진 탐색 트리의 연산은 딥러닝에서 정렬된 데이터를 다루는 일(예를 들어 빔 탐색에서 상위 $k$개의 점수를 유지하는 일)과 어떤 관계가 있는가?

??? success "연습문제 4 풀이"
    빔 탐색은 가장 좋은 가설 $k$개를 유지하므로 새 후보를 넣고 가장 나쁜 것을 빼는 일을 효율적으로 해야 한다. 이진 탐색 트리는 둘 다 $O(\log k)$에 해낸다. 실제로는 더 간단하고 상수가 작은 힙을 즐겨 쓰지만, 이진 탐색 트리는 범위 검색이나 순위 같은 더 풍부한 질의를 지원한다.