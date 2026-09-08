# 이진 탐색 트리의 성질

이진 탐색 트리의 성질은 평범한 이진 트리를 강력한 탐색 구조로 바꾸는 단 하나의 불변식이다. 정렬된 배열의 이진 탐색이 비교할 때마다 후보의 절반을 걸러 내듯이, 이 성질 덕분에 찾기 도중 노드마다 부분 트리 하나를 통째로 걸러 낼 수 있다. 정렬된 순서와 트리 구조 사이의 이 관계가 [찾기](search.md), [삽입](insertion.md), [삭제](deletion.md), [후속자·선행자](successor.md) 질의를 비롯한 모든 연산의 바탕이다.

---

## 1. 엄밀한 정의

이진 트리가 모든 노드 $x$에 대해 다음을 만족하면 **이진 탐색 트리 성질**을 만족한다고 한다.

$$
\text{For all nodes } y \text{ in the left subtree of } x: \quad y.\text{key} \leq x.\text{key}
$$

$$
\text{For all nodes } z \text{ in the right subtree of } x: \quad z.\text{key} > x.\text{key}
$$

이 성질이 바로 아래 자식뿐 아니라 각 부분 트리의 **모든** 노드에 적용된다는 점에 주의하라. 더 깊은 자손은 살피지 않고 바로 아래 자식만 확인하는 것이 흔한 실수이다.

!!! warning "중복된 열쇠"
    중복된 열쇠를 다루는 방식은 교과서마다 다르다. 이 책에서는 중복을 **왼쪽 부분 트리**에 둔다(왼쪽에 $\leq$, 오른쪽에 $>$을 쓴다). 어떤 구현은 왼쪽에 $<$, 오른쪽에 $\geq$을 쓰거나 중복을 아예 금지한다. 한결같이 쓰기만 하면 어느 쪽이든 올바름에는 영향이 없다.

---

## 2. 그림으로 보는 예

```
         8
        / \
       3   10
      / \    \
     1   6   14
        / \  /
       4  7 13
```

노드 8을 보자. 왼쪽 부분 트리의 열쇠($\{1, 3, 4, 6, 7\}$)는 모두 8보다 작거나 같고, 오른쪽 부분 트리의 열쇠($\{10, 13, 14\}$)는 모두 8보다 크다. 이 성질은 트리의 모든 노드에서 재귀적으로 성립한다.

다음 트리는 노드마다 바로 아래 자식은 조건을 만족하지만 이진 탐색 트리 성질을 어긴다.

```
         5
        / \
       3   8
      / \
     1   7      <- 7 > 5, so 7 is in the wrong subtree!
```

노드 3의 오른쪽 자식은 7이고 $7 > 3$을 만족한다. 그런데 7은 5의 왼쪽 부분 트리에도 있고 $7 > 5$이므로 뿌리에서 이진 탐색 트리 성질이 깨진다.

---

## 3. 중위 순회는 정렬된 순서를 낸다

이진 탐색 트리 성질에서 나오는 가장 중요한 결과는 **중위 순회**가 열쇠의 오름차순(같은 값 허용)으로 노드를 들른다는 것이다.

!!! note "정리"
    $T$이 이진 탐색 트리이면 $T$의 중위 순회는 열쇠를 정렬된(감소하지 않는) 순서로 들른다.

**증명 개요**: 노드 수에 대한 귀납법으로 보인다. 노드가 하나이면 자명하다. 뿌리가 $x$이고 왼쪽 부분 트리가 $L$, 오른쪽 부분 트리가 $R$인 트리를 보자. 이진 탐색 트리 성질에 따라 $L$의 열쇠는 모두 $x.\text{key}$ 이하이고 $R$의 열쇠는 모두 $x.\text{key}$보다 크다. 귀납 가정에 따라 $L$의 중위 순회와 $R$의 중위 순회는 각각 정렬된 결과를 낸다. 따라서 중위 순서 $[L, x, R]$은 $x.\text{key}$ 이하 값들의 정렬된 나열 뒤에 $x.\text{key}$, 그 뒤에 $x.\text{key}$보다 큰 값들의 정렬된 나열을 이어 붙인 것이다. 결과는 정렬되어 있다. $\square$

---

## 4. 이진 탐색 트리 성질 검증하기

올바른 검증은 노드를 바로 아래 자식과 견주는 데 그치지 않고, 노드의 열쇠가 조상에게서 물려받은 유효 범위 안에 있는지 확인해야 한다.

```python
"""
이진 탐색 트리의 성질: 정의와 검증과 확인.

이진 탐색 트리의 불변식을 보이고, 중위 순회가 정렬된 결과를 내놓음을
보이며, 범위 확인으로 올바르게 검증하는 방법을 준다.
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

# === 이진 탐색 트리 검증 ===

def is_valid_bst(node, min_key=float("-inf"), max_key=float("inf")):
    """node를 뿌리로 하는 트리가 이진 탐색 트리의 성질을 만족하는지 확인한다.

    재귀 호출마다 노드의 열쇠가 놓여야 하는 유효 범위 [min_key, max_key]를
    좁혀 간다.
    """
    if node is None:
        return True
    if node.key <= min_key or node.key > max_key:
        return False
    return (is_valid_bst(node.left, min_key, node.key) and
            is_valid_bst(node.right, node.key, max_key))

# === 중위 순회 ===

def inorder(node):
    """열쇠를 중위(정렬된) 순서로 내놓는다."""
    if node is not None:
        yield from inorder(node.left)
        yield node.key
        yield from inorder(node.right)

# === 메인 ===

if __name__ == "__main__":
    # 올바른 이진 탐색 트리
    valid_tree = Node(8,
        Node(3,
            Node(1),
            Node(6, Node(4), Node(7))),
        Node(10,
            None,
            Node(14, Node(13))))

    print("Valid BST:")
    print(f"  Inorder traversal: {list(inorder(valid_tree))}")
    print(f"  Is valid BST:      {is_valid_bst(valid_tree)}")
    print()

    # 잘못된 이진 탐색 트리 (5의 왼쪽 부분 트리에 있는 7이 성질을 어긴다)
    invalid_tree = Node(5,
        Node(3, Node(1), Node(7)),
        Node(8))

    print("Invalid BST (7 in wrong subtree):")
    print(f"  Inorder traversal: {list(inorder(invalid_tree))}")
    print(f"  Is valid BST:      {is_valid_bst(invalid_tree)}")
```

**출력:**
```
Valid BST:
  Inorder traversal: [1, 3, 4, 6, 7, 8, 10, 13, 14]
  Is valid BST:      True

Invalid BST (7 in wrong subtree):
  Inorder traversal: [1, 3, 7, 5, 8]
  Is valid BST:      False
```

---

## 연습문제

**연습문제 1.**
이진 탐색 트리의 성질에 대해 트리의 높이 $h$으로 시간 복잡도를 밝혀라. $h = O(\log n)$이 되는 것은 언제인가?

??? success "연습문제 1 풀이"
    이 연산은 $O(h)$ 시간이 걸린다. 균형 잡힌 트리에서는 $h = O(\log n)$이다(무작위 삽입 순서에서는 기댓값이 $O(\log n)$이고, 균형 이진 탐색 트리 계열은 이를 보장한다). 한쪽으로 치우친 트리에서는 $h = O(n)$이다(정렬된 순서로 넣은 경우).

---

**연습문제 2.**
열쇠 5, 3, 7, 1, 4, 6, 8을 차례로 넣어 만든 이진 탐색 트리에 이진 탐색 트리의 성질을(를) 적용하라. 단계마다 보여라.

??? success "연습문제 2 풀이"
    차례로 넣어 트리를 만든 뒤 이 쪽의 연산을 적용한다. 단계마다 트리의 모습을 보이고 들르거나 고친 노드를 표시한다.

---

**연습문제 3.**
이진 탐색 트리의 높이를 $h$이라 할 때 이진 탐색 트리의 성질이(가) $O(h)$ 시간에 끝남을 증명하라.

??? success "연습문제 3 풀이"
    이 연산은 뿌리에서 잎으로 가는 경로(또는 그 일부)를 따라가며 노드마다 $O(1)$의 일을 한다. 경로의 길이는 많아야 $h$이다. 따라서 모두 $O(h)$이다. $\square$

---

**연습문제 4.**
이진 탐색 트리의 연산은 딥러닝에서 정렬된 데이터를 다루는 일(예를 들어 빔 탐색에서 상위 $k$개의 점수를 유지하는 일)과 어떤 관계가 있는가?

??? success "연습문제 4 풀이"
    빔 탐색은 가장 좋은 가설 $k$개를 유지하므로 새 후보를 넣고 가장 나쁜 것을 빼는 일을 효율적으로 해야 한다. 이진 탐색 트리는 둘 다 $O(\log k)$에 해낸다. 실제로는 더 간단하고 상수가 작은 힙을 즐겨 쓰지만, 이진 탐색 트리는 범위 검색이나 순위 같은 더 풍부한 질의를 지원한다.

## 정리하며

이 마당은 엄밀한 정의、그림으로 보는 예、중위 순회는 정렬된 순서를 낸다、이진 탐색 트리 성질 검증하기을 차례로 짚었다.

**참고 문헌**

- [Introduction to Algorithms (CLRS), 12장](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
