# 높이와 깊이

트리의 높이와 노드마다의 깊이를 이해하는 일은 트리 연산의 시간 복잡도를 분석하는 데 꼭 필요하다. 이진 트리에서 찾기와 삽입과 삭제 알고리즘은 대부분 높이에 비례하는 시간이 걸리므로, 높이를 계산하고 그 한계를 아는 것이 알고리즘의 효율을 곧바로 가른다.

## 노드의 깊이

노드 $v$의 **깊이**는 뿌리에서 $v$까지 가는 경로의 변의 수이다. 같은 말로, 깊이는 $v$의 조상 수(자기 자신은 빼고)를 센 것이다.

$$
\text{depth}(v) = \begin{cases} 0 & \text{if } v \text{ is the root} \\ 1 + \text{depth}(\text{parent}(v)) & \text{otherwise} \end{cases}
$$

뿌리의 깊이는 언제나 0이다. 뿌리의 자식은 깊이 1, 손자는 깊이 2, 이런 식으로 이어진다.

## 노드의 높이

노드 $v$의 **높이**는 $v$에서 잎까지 내려가는 가장 긴 경로의 변의 수이다. 가장 깊은 자손이 얼마나 아래에 있는지를 나타낸다.

$$
\text{height}(v) = \begin{cases} 0 & \text{if } v \text{ is a leaf} \\ 1 + \max(\text{height}(\text{left}(v)),\; \text{height}(\text{right}(v))) & \text{otherwise} \end{cases}
$$

빈 부분 트리에서도 재귀적 정의가 깔끔하게 통하도록 널(빈) 부분 트리의 높이를 $-1$으로 두기로 한다.

$$
\text{height}(\text{null}) = -1
$$

이렇게 하면 널 자식 둘을 가진 잎 노드의 높이가 $1 + \max(-1, -1) = 0$이 되어 앞뒤가 맞는다.

## 트리의 높이

**트리의 높이**는 뿌리 노드의 높이이며, 같은 말로 트리 안 어떤 노드의 깊이 중 최댓값이다.

$$
h(T) = \text{height}(\text{root}) = \max_{v \in T} \text{depth}(v)
$$

## 높이의 한계

노드가 $n \geq 1$개이고 높이가 $h$인 이진 트리에 대해 다음이 성립한다.

- **최소 높이** (완전 이진 트리): $h = \lfloor \log_2 n \rfloor$
- **최대 높이** (한쪽으로 치우친 트리): $h = n - 1$

따라서 다음과 같은 한계가 나온다.

$$
\lfloor \log_2 n \rfloor \leq h \leq n - 1
$$

균형 잡힌 이진 트리는 $h = \Theta(\log n)$을 지켜 연산이 효율적이다. 내부 노드마다 자식이 꼭 하나뿐인 치우친 트리는 $h = n - 1$인 연결 리스트로 무너져 연산이 $\Theta(n)$이 된다.

!!! note "변을 세는 방식과 노드를 세는 방식"
    어떤 책은 높이를 뿌리에서 잎까지 가장 긴 경로의 **변**의 수가 아니라 **노드**의 수로 정의한다. 그 방식에서는 높이 = (변으로 센 높이) + 1이고, 노드가 하나뿐인 트리의 높이가 0이 아니라 1이다. 이 책은 CLRS를 따라 변을 세는 방식을 쓴다.

## 예

다음 이진 트리를 생각해 보자.

```
         A          depth 0, height 3
        / \
       B   C        depth 1, heights 1 and 2
      /   / \
     D   E   F      depth 2, heights 0, 1, and 0
            /
           G        depth 3, height 0
```

| 노드 | 깊이 | 높이 |
|------|-------|--------|
| A    | 0     | 3      |
| B    | 1     | 1      |
| C    | 1     | 2      |
| D    | 2     | 0      |
| E    | 2     | 1      |
| F    | 2     | 0      |
| G    | 3     | 0      |

이 트리의 높이는 $\text{height}(A) = 3$이고, 최대 깊이도 3이다(노드 G).

```python
"""
이진 트리의 높이와 깊이 계산.

노드의 깊이와 높이, 트리의 높이를 재귀로 계산하는 방법과 이들 사이의
관계를 보인다.
"""


# === 노드 정의 ===

class Node:
    """이진 트리의 노드."""

    def __init__(self, key, left=None, right=None):
        self.key = key
        self.left = left
        self.right = right


# === 깊이 계산 ===

def depth(root, target, current_depth=0):
    """표적 노드의 깊이를 돌려주고, 없으면 -1을 돌려준다."""
    if root is None:
        return -1
    if root.key == target:
        return current_depth
    left_result = depth(root.left, target, current_depth + 1)
    if left_result != -1:
        return left_result
    return depth(root.right, target, current_depth + 1)


# === 높이 계산 ===

def height(node):
    """node를 뿌리로 하는 부분 트리의 높이를 돌려준다.

    height(null) = -1이라는 규약을 쓰므로 잎의 높이는 0이다.
    """
    if node is None:
        return -1
    return 1 + max(height(node.left), height(node.right))


# === 메인 ===

if __name__ == "__main__":
    # 예제 트리 만들기:
    #          A
    #         / \
    #        B   C
    #       /   / \
    #      D   E   F
    #             /
    #            G
    tree = Node("A",
        Node("B", Node("D")),
        Node("C",
            Node("E", None, Node("G")),
            Node("F")))

    print("Tree height:", height(tree))
    print()

    for label in ["A", "B", "C", "D", "E", "F", "G"]:
        d = depth(tree, label)
        print(f"  Node {label}: depth = {d}")
```

**출력:**
```
Tree height: 3

  Node A: depth = 0
  Node B: depth = 1
  Node C: depth = 1
  Node D: depth = 2
  Node E: depth = 2
  Node F: depth = 2
  Node G: depth = 3
```

## 참고 문헌

- [Introduction to Algorithms (CLRS), 12장](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)


## 연습문제

**연습문제 1.**
높이와 깊이에 대해 트리의 높이 $h$으로 시간 복잡도를 밝혀라. $h = O(\log n)$이 되는 것은 언제인가?

??? success "연습문제 1 풀이"
    이 연산은 $O(h)$ 시간이 걸린다. 균형 잡힌 트리에서는 $h = O(\log n)$이다(무작위 삽입 순서에서는 기댓값이 $O(\log n)$이고, 균형 이진 탐색 트리 계열은 이를 보장한다). 한쪽으로 치우친 트리에서는 $h = O(n)$이다(정렬된 순서로 넣은 경우).

---

**연습문제 2.**
열쇠 5, 3, 7, 1, 4, 6, 8을 차례로 넣어 만든 이진 탐색 트리에 높이와 깊이을(를) 적용하라. 단계마다 보여라.

??? success "연습문제 2 풀이"
    차례로 넣어 트리를 만든 뒤 이 쪽의 연산을 적용한다. 단계마다 트리의 모습을 보이고 들르거나 고친 노드를 표시한다.

---

**연습문제 3.**
이진 탐색 트리의 높이를 $h$이라 할 때 높이와 깊이이(가) $O(h)$ 시간에 끝남을 증명하라.

??? success "연습문제 3 풀이"
    이 연산은 뿌리에서 잎으로 가는 경로(또는 그 일부)를 따라가며 노드마다 $O(1)$의 일을 한다. 경로의 길이는 많아야 $h$이다. 따라서 모두 $O(h)$이다. $\square$

---

**연습문제 4.**
이진 탐색 트리의 연산은 딥러닝에서 정렬된 데이터를 다루는 일(예를 들어 빔 탐색에서 상위 $k$개의 점수를 유지하는 일)과 어떤 관계가 있는가?

??? success "연습문제 4 풀이"
    빔 탐색은 가장 좋은 가설 $k$개를 유지하므로 새 후보를 넣고 가장 나쁜 것을 빼는 일을 효율적으로 해야 한다. 이진 탐색 트리는 둘 다 $O(\log k)$에 해낸다. 실제로는 더 간단하고 상수가 작은 힙을 즐겨 쓰지만, 이진 탐색 트리는 범위 검색이나 순위 같은 더 풍부한 질의를 지원한다.