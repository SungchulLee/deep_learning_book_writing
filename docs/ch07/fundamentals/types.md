# 포화·완전·완벽 트리

이진 트리에는 구조에 따라 몇 가지 갈래가 있고, 그 구별은 알고리즘 설계에서 중요하다. 완벽 이진 트리는 로그 높이를 보장하고, 완전 이진 트리는 배열에 효율적으로 담을 수 있으며, 포화 이진 트리는 내부 노드와 잎 사이의 관계를 묶는다. 알고리즘이 어떤 갈래의 트리를 요구하는지(또는 만들어 내는지) 알아보면 시간과 공간 복잡도를 가늠하기 쉽다.

---

## 1. 포화 이진 트리

**포화 이진 트리**(**진 이진 트리**라고도 한다)는 노드마다 자식이 0개이거나 2개인 이진 트리이다. 자식이 하나뿐인 노드는 없다.

```
     Full:              Not full:
       1                   1
      / \                 / \
     2   3               2   3
    / \                  /
   4   5                4
```

**핵심 성질**: 내부 노드가 $n$개인 포화 이진 트리에는 잎이 꼭 $n + 1$개 있다.

??? note "증명"
    $L$을 잎의 수, $I$을 내부 노드의 수라 하자. 내부 노드마다 (자식마다 하나씩) 변을 꼭 2개 내보내므로 변은 모두 $2I$개이다. 전체 노드가 $n$개인 트리에는 변이 $n - 1$개 있고 $n = I + L$이므로 $2I = I + L - 1$이고, 따라서 $L = I + 1$이다.

포화 이진 트리의 전체 노드 수는 다음과 같다.

$$
n = 2I + 1 = 2L - 1
$$

여기서 $I$은 내부 노드의 수, $L$은 잎의 수이다.

---

## 2. 완전 이진 트리

**완전 이진 트리**는 마지막 층만 왼쪽부터 차 있고 나머지 층은 모두 꽉 차 있는 이진 트리이다.

```
     Complete:          Not complete:
         1                  1
        / \                / \
       2   3              2   3
      / \ /              / \   \
     4  5 6             4   5   7
```

**핵심 성질**:

- 노드가 $n$개인 완전 이진 트리의 높이는 $h = \lfloor \log_2 n \rfloor$이다.
- 공간을 낭비하지 않고 [배열](array.md)에 효율적으로 담을 수 있다.
- 노드 수는 $2^h \leq n \leq 2^{h+1} - 1$을 만족한다.

완전 이진 트리는 **이진 힙**이 쓰는 모양이다. 완전하다는 성질 덕분에 힙의 높이가 $\Theta(\log n)$으로 유지되어 삽입과 추출이 효율적이다.

---

## 3. 완벽 이진 트리

**완벽 이진 트리**는 모든 내부 노드의 자식이 꼭 둘이고 모든 잎이 같은 깊이에 있는 이진 트리이다.

```
     Perfect (h=2):
         1
        / \
       2   3
      / \ / \
     4  5 6  7
```

완벽 이진 트리는 포화이면서 완전하다. 주어진 높이에서 노드 수가 최대이다.

$$
n = 2^{h+1} - 1
$$

여기서 $h$은 트리의 높이이다. 같은 말로, 노드가 $n$개인 완벽 이진 트리의 높이는 다음과 같다.

$$
h = \log_2(n + 1) - 1
$$

깊이 $d$마다 노드 수는 꼭 $2^d$개이고, 잎의 수는 다음과 같다.

$$
L = 2^h = \frac{n + 1}{2}
$$

곧 완벽 이진 트리에서 전체 노드의 약 절반이 잎이라는 뜻이다.

---

## 4. 비교

| 성질 | 포화 | 완전 | 완벽 |
|---|---|---|---|
| 노드마다 자식이 0개나 2개 | 그렇다 | 꼭 그렇지는 않다 | 그렇다 |
| 마지막 층만 빼고 모두 꽉 참 | 꼭 그렇지는 않다 | 그렇다 | 그렇다 |
| 모든 잎이 같은 깊이 | 꼭 그렇지는 않다 | 꼭 그렇지는 않다 | 그렇다 |
| 배열에 효율적으로 담을 수 있음 | 아니다 | 그렇다 | 그렇다 |
| 높이 보장 | $\Theta(\log n)$에서 $\Theta(n)$까지 | $\Theta(\log n)$ | $\Theta(\log n)$ |

!!! warning "교과서마다 용어가 다르다"
    어떤 저자는 우리가 "완벽"이라 부르는 것을 "완전"이라 부른다. 또 어떤 이는 "포화"를 "완전"의 뜻으로 쓴다. 이 책은 CLRS와 대부분의 알고리즘 교과서를 따른다. **포화** = 노드마다 자식이 0개나 2개, **완전** = 마지막 층만 빼고 모두 꽉 참(마지막 층은 왼쪽부터), **완벽** = 모든 층이 꽉 참.

---

## 5. 갈래 사이의 관계

세 갈래는 위계를 이룬다.

- 모든 **완벽** 이진 트리는 **완전**이면서 **포화**이다.
- **완전** 이진 트리가 반드시 포화는 아니다(마지막 층의 경계에 왼쪽 자식만 있는 노드가 있을 수 있다).
- **포화** 이진 트리가 반드시 완전은 아니다(잎이 서로 다른 깊이에 있을 수 있다).

```python
"""
이진 트리를 포화·완전·완벽으로 가르기.

이진 트리가 각 구조적 성질을 만족하는지 시험하는 함수를 주고,
갈래마다 예를 든다.
"""

# === 노드 정의 ===

class Node:
    """이진 트리의 노드."""

    def __init__(self, key, left=None, right=None):
        self.key = key
        self.left = left
        self.right = right

# === 갈래를 가리는 함수 ===

def is_full(node):
    """트리가 포화 이진 트리인지 확인한다.

    노드마다 자식이 0개이거나 2개이다.
    """
    if node is None:
        return True
    if node.left is None and node.right is None:
        return True
    if node.left is not None and node.right is not None:
        return is_full(node.left) and is_full(node.right)
    return False

def _count_nodes(node):
    """전체 노드 수를 돌려준다."""
    if node is None:
        return 0
    return 1 + _count_nodes(node.left) + _count_nodes(node.right)

def is_complete(node, index=0, node_count=None):
    """트리가 완전 이진 트리인지 확인한다.

    배열 색인의 성질을 쓴다. 노드가 n개인 완전 트리에서는 (레벨 순서로 매긴)
    노드의 색인이 모두 n보다 작아야 한다.
    """
    if node_count is None:
        node_count = _count_nodes(node)
    if node is None:
        return True
    if index >= node_count:
        return False
    return (is_complete(node.left, 2 * index + 1, node_count) and
            is_complete(node.right, 2 * index + 2, node_count))

def _height(node):
    """트리의 높이를 돌려준다."""
    if node is None:
        return -1
    return 1 + max(_height(node.left), _height(node.right))

def is_perfect(node, depth=0, target_depth=None):
    """트리가 완벽 이진 트리인지 확인한다.

    모든 잎이 같은 깊이에 있어야 하고 내부 노드마다 자식이 꼭 둘이어야
    한다.
    """
    if target_depth is None:
        target_depth = _height(node)
    if node is None:
        return True
    if node.left is None and node.right is None:
        return depth == target_depth
    if node.left is None or node.right is None:
        return False
    return (is_perfect(node.left, depth + 1, target_depth) and
            is_perfect(node.right, depth + 1, target_depth))

# === 메인 ===

if __name__ == "__main__":
    # 완벽 트리 (포화이면서 완전하기도 하다)
    perfect = Node(1,
        Node(2, Node(4), Node(5)),
        Node(3, Node(6), Node(7)))

    # 포화이지만 완전하지 않음 (잎이 서로 다른 깊이에 있다)
    full_only = Node(1,
        Node(2, Node(4), Node(5)),
        Node(3))

    # 완전하지만 포화가 아님 (경계의 노드 3에 왼쪽 자식만 있다)
    complete_only = Node(1,
        Node(2, Node(4), Node(5)),
        Node(3, Node(6)))

    trees = [
        ("Perfect tree", perfect),
        ("Full-only tree", full_only),
        ("Complete-only tree", complete_only),
    ]

    for name, tree in trees:
        print(f"{name}:")
        print(f"  Full:     {is_full(tree)}")
        print(f"  Complete: {is_complete(tree)}")
        print(f"  Perfect:  {is_perfect(tree)}")
        print()
```

**출력:**
```
Perfect tree:
  Full:     True
  Complete: True
  Perfect:  True

Full-only tree:
  Full:     True
  Complete: False
  Perfect:  False

Complete-only tree:
  Full:     False
  Complete: True
  Perfect:  False
```

---

## 연습문제

**연습문제 1.**
포화·완전·완벽 트리에 대해 트리의 높이 $h$으로 시간 복잡도를 밝혀라. $h = O(\log n)$이 되는 것은 언제인가?

??? success "연습문제 1 풀이"
    이 연산은 $O(h)$ 시간이 걸린다. 균형 잡힌 트리에서는 $h = O(\log n)$이다(무작위 삽입 순서에서는 기댓값이 $O(\log n)$이고, 균형 이진 탐색 트리 계열은 이를 보장한다). 한쪽으로 치우친 트리에서는 $h = O(n)$이다(정렬된 순서로 넣은 경우).

---

**연습문제 2.**
열쇠 5, 3, 7, 1, 4, 6, 8을 차례로 넣어 만든 이진 탐색 트리에 포화·완전·완벽 트리을(를) 적용하라. 단계마다 보여라.

??? success "연습문제 2 풀이"
    차례로 넣어 트리를 만든 뒤 이 쪽의 연산을 적용한다. 단계마다 트리의 모습을 보이고 들르거나 고친 노드를 표시한다.

---

**연습문제 3.**
이진 탐색 트리의 높이를 $h$이라 할 때 포화·완전·완벽 트리이(가) $O(h)$ 시간에 끝남을 증명하라.

??? success "연습문제 3 풀이"
    이 연산은 뿌리에서 잎으로 가는 경로(또는 그 일부)를 따라가며 노드마다 $O(1)$의 일을 한다. 경로의 길이는 많아야 $h$이다. 따라서 모두 $O(h)$이다. $\square$

---

**연습문제 4.**
이진 탐색 트리의 연산은 딥러닝에서 정렬된 데이터를 다루는 일(예를 들어 빔 탐색에서 상위 $k$개의 점수를 유지하는 일)과 어떤 관계가 있는가?

??? success "연습문제 4 풀이"
    빔 탐색은 가장 좋은 가설 $k$개를 유지하므로 새 후보를 넣고 가장 나쁜 것을 빼는 일을 효율적으로 해야 한다. 이진 탐색 트리는 둘 다 $O(\log k)$에 해낸다. 실제로는 더 간단하고 상수가 작은 힙을 즐겨 쓰지만, 이진 탐색 트리는 범위 검색이나 순위 같은 더 풍부한 질의를 지원한다.

## 정리하며

이 마당은 포화 이진 트리、완전 이진 트리、완벽 이진 트리、비교을 차례로 짚었다.

**참고 문헌**

- [Introduction to Algorithms (CLRS), 12장](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
