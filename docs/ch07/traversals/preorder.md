# 전위 순회

전위 순회는 **뿌리를 먼저** 처리한 뒤 왼쪽 부분 트리를 재귀적으로 들르고 마지막에 오른쪽 부분 트리를 들러 이진 트리의 모든 노드를 훑는다. 이 "자식보다 뿌리 먼저" 순서 덕분에 부모를 자손보다 먼저 다루어야 할 때, 이를테면 트리를 베끼거나 구조를 직렬화하거나 식 트리에서 전위 식을 만들 때 전위 순회가 자연스러운 선택이 된다.

## 정의

뿌리가 $r$인 이진 트리에서 **전위 순회**는 다음 순서로 노드를 들른다.

$$
\text{visit}(r), \quad \text{preorder}(r.\text{left}), \quad \text{preorder}(r.\text{right})
$$

바탕 경우는 빈 부분 트리(널 포인터)이며 아무것도 내놓지 않는다. 노드가 $n$개인 트리에서 순회는 꼭 $n$번의 들름으로 이루어진 나열을 낸다.

## 재귀 알고리즘

재귀 형태는 정의를 그대로 옮긴 것이다. 호출마다 지금 노드를 처리한 뒤 왼쪽과 오른쪽 자식에게 넘긴다.

```python
"""이진 트리의 전위 순회: 재귀 방법과 반복 방법."""

from __future__ import annotations


# === 노드 정의 ===

class TreeNode:
    """정수 값을 갖는 이진 트리 노드."""

    def __init__(self, val: int = 0, left: TreeNode | None = None,
                 right: TreeNode | None = None):
        self.val = val
        self.left = left
        self.right = right


# === 재귀 전위 순회 ===

def preorder_recursive(root: TreeNode | None) -> list[int]:
    """*root*를 뿌리로 하는 트리의 전위 순회를 돌려준다."""
    if root is None:
        return []
    return [root.val] + preorder_recursive(root.left) + preorder_recursive(root.right)
```

재귀 판본은 간결하지만 트리의 높이를 $h$이라 할 때 호출 스택에 $O(h)$의 공간을 쓴다. 치우친 트리에서는 $h = n - 1$이므로 최악의 공간은 $O(n)$이다.

## 반복 알고리즘

명시적인 스택이 호출 스택을 대신한다. 핵심은 왼쪽 자식이 먼저 꺼내져(들러져) 나오도록 오른쪽 자식을 왼쪽 자식보다 **먼저** 밀어 넣어야 한다는 것이다.

```python
# === 반복적 전위 순회 ===

def preorder_iterative(root: TreeNode | None) -> list[int]:
    """명시적인 스택을 쓰는 반복적 전위 순회."""
    if root is None:
        return []
    stack: list[TreeNode] = [root]
    result: list[int] = []
    while stack:
        node = stack.pop()
        result.append(node.val)
        # 왼쪽이 먼저 처리되도록 오른쪽을 먼저 밀어 넣는다
        if node.right is not None:
            stack.append(node.right)
        if node.left is not None:
            stack.append(node.left)
    return result
```

??? example "반복적 전위 순회 따라가기"
    다음 트리를 생각해 보자.

    ```
          1
         / \
        2   3
       / \
      4   5
    ```

    | 단계 | 스택 (맨 위 → 오른쪽) | 꺼냄 | 지금까지의 출력 |
    |------|-------------------|-----|---------------|
    | 0 | [1] | — | [] |
    | 1 | [3, 2] | 1 | [1] |
    | 2 | [3, 5, 4] | 2 | [1, 2] |
    | 3 | [3, 5] | 4 | [1, 2, 4] |
    | 4 | [3] | 5 | [1, 2, 4, 5] |
    | 5 | [] | 3 | [1, 2, 4, 5, 3] |

    결과: **1, 2, 4, 5, 3**

## 복잡도

| 항목 | 재귀 | 반복 |
|--------|-----------|-----------|
| 시간 | $O(n)$ | $O(n)$ |
| 공간 | 호출 스택 $O(h)$ | 명시적 스택 $O(h)$ |

두 방법 모두 노드마다 꼭 한 번씩 들르므로 시간은 $O(n)$이다. 공간은 둘 다 $O(h)$이며, 균형 잡힌 트리에서는 $O(\log n)$, 최악의 경우에는 $O(n)$이다.

!!! tip "재귀와 반복 중에 고르기"
    재귀 판본이 더 간단하고 대부분의 쓰임에 충분하다. 트리가 아주 깊을 수 있어 스택이 넘칠 위험이 있거나 순회 상태를 세밀하게 다스려야 할 때는 반복 판본을 쓰라.

## 응용

전위 순회는 실전에서 여러 곳에 나타난다.

- **트리 베끼기:** 자식보다 뿌리를 먼저 처리하므로 자식을 붙일 때 부모 노드가 이미 있다.
- **직렬화와 역직렬화:** 전위 순회에 널 표시를 함께 쓰면 이진 트리를 유일하게 담을 수 있다.
- **전위 식 계산:** 식 트리를 전위로 훑으면 전위(폴란드) 표기법이 나온다.
- **디렉터리 목록:** 들여쓰기로 출력하는 파일 시스템 트리는 전위 순서를 쓴다. 디렉터리 이름이 그 내용보다 먼저 나온다.

## 시연

```python
# === 시연 ===

if __name__ == "__main__":
    #       1
    #      / \
    #     2   3
    #    / \
    #   4   5
    tree = TreeNode(1,
        TreeNode(2, TreeNode(4), TreeNode(5)),
        TreeNode(3))

    print(f"Recursive preorder: {preorder_recursive(tree)}")  # [1, 2, 4, 5, 3]
    print(f"Iterative preorder: {preorder_iterative(tree)}")  # [1, 2, 4, 5, 3]
```

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 12. MIT Press.
- Knuth, D. E. (1997). *The Art of Computer Programming*, Volume 1, Section 2.3.1. Addison-Wesley.


## 연습문제

**연습문제 1.**
전위 순회에 대해 트리의 높이 $h$으로 시간 복잡도를 밝혀라. $h = O(\log n)$이 되는 것은 언제인가?

??? success "연습문제 1 풀이"
    이 연산은 $O(h)$ 시간이 걸린다. 균형 잡힌 트리에서는 $h = O(\log n)$이다(무작위 삽입 순서에서는 기댓값이 $O(\log n)$이고, 균형 이진 탐색 트리 계열은 이를 보장한다). 한쪽으로 치우친 트리에서는 $h = O(n)$이다(정렬된 순서로 넣은 경우).

---

**연습문제 2.**
열쇠 5, 3, 7, 1, 4, 6, 8을 차례로 넣어 만든 이진 탐색 트리에 전위 순회을(를) 적용하라. 단계마다 보여라.

??? success "연습문제 2 풀이"
    차례로 넣어 트리를 만든 뒤 이 쪽의 연산을 적용한다. 단계마다 트리의 모습을 보이고 들르거나 고친 노드를 표시한다.

---

**연습문제 3.**
이진 탐색 트리의 높이를 $h$이라 할 때 전위 순회이(가) $O(h)$ 시간에 끝남을 증명하라.

??? success "연습문제 3 풀이"
    이 연산은 뿌리에서 잎으로 가는 경로(또는 그 일부)를 따라가며 노드마다 $O(1)$의 일을 한다. 경로의 길이는 많아야 $h$이다. 따라서 모두 $O(h)$이다. $\square$

---

**연습문제 4.**
이진 탐색 트리의 연산은 딥러닝에서 정렬된 데이터를 다루는 일(예를 들어 빔 탐색에서 상위 $k$개의 점수를 유지하는 일)과 어떤 관계가 있는가?

??? success "연습문제 4 풀이"
    빔 탐색은 가장 좋은 가설 $k$개를 유지하므로 새 후보를 넣고 가장 나쁜 것을 빼는 일을 효율적으로 해야 한다. 이진 탐색 트리는 둘 다 $O(\log k)$에 해낸다. 실제로는 더 간단하고 상수가 작은 힙을 즐겨 쓰지만, 이진 탐색 트리는 범위 검색이나 순위 같은 더 풍부한 질의를 지원한다.