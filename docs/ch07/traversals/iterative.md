# 반복적 순회

재귀로 하는 트리 순회는 우아하지만 트리의 높이를 $h$이라 할 때 스택 프레임을 $O(h)$개 암묵적으로 쓴다. 아주 깊거나 균형이 나쁜 트리에서는 스택이 넘칠 위험이 있다. **반복적 순회**는 호출 스택을 명시적인 스택 자료 구조로 바꾸어 같은 순서로 들르면서도 메모리 사용을 다스릴 수 있고, 함수 호출 부담이 줄어 상수 배 성능도 나을 때가 많다.

## 반복적 전위 순회

전위 순회는 자식보다 **먼저** 지금 노드를 들른다. 뿌리, 왼쪽 부분 트리, 오른쪽 부분 트리 순이다.

핵심은 오른쪽 자식을 먼저 밀어 넣어 왼쪽 자식이 먼저 꺼내지도록(그래서 먼저 처리되도록) 하는 것이다.

```python
"""명시적인 스택을 쓰는 반복적 트리 순회."""

from __future__ import annotations


# === 노드 정의 ===

class TreeNode:
    """이진 트리 노드."""

    def __init__(self, val: int = 0, left: TreeNode | None = None,
                 right: TreeNode | None = None):
        self.val = val
        self.left = left
        self.right = right


# === 반복적 전위 순회 ===

def preorder(root: TreeNode | None) -> list[int]:
    """전위 순회: 뿌리 -> 왼쪽 -> 오른쪽."""
    if root is None:
        return []
    result: list[int] = []
    stack = [root]
    while stack:
        node = stack.pop()
        result.append(node.val)
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
    return result
```

!!! tip "왜 오른쪽을 먼저 밀어 넣는가?"
    스택은 나중에 넣은 것이 먼저 나온다. 오른쪽을 먼저, 왼쪽을 나중에 넣으면 왼쪽 자식이 다음에 꺼내져 전위 순서(뿌리, 왼쪽, 오른쪽)가 지켜진다.

## 반복적 중위 순회

중위 순회는 왼쪽 부분 트리, 지금 노드, 오른쪽 부분 트리 순으로 들른다. 반복 판본은 "갈 수 있는 데까지 왼쪽으로" 가는 단계를 좇는 포인터를 쓴다.

```python
# === 반복적 중위 순회 ===

def inorder(root: TreeNode | None) -> list[int]:
    """중위 순회: 왼쪽 -> 뿌리 -> 오른쪽."""
    result: list[int] = []
    stack: list[TreeNode] = []
    current = root
    while current or stack:
        # 갈 수 있는 데까지 왼쪽으로
        while current:
            stack.append(current)
            current = current.left
        # 노드 들르기
        current = stack.pop()
        result.append(current.val)
        # 오른쪽 부분 트리로 옮기기
        current = current.right
    return result
```

알고리즘은 불변식 두 개를 지킨다.

1. 스택 위의 노드는 아직 들르지 않았지만 그 왼쪽 부분 트리는 살피는 중이거나 이미 살폈다.
2. `current` 포인터는 들르기 전에 왼쪽 부분 트리를 모두 살펴야 하는 다음 노드를 가리킨다.

## 반복적 후위 순회

후위 순회는 왼쪽 부분 트리, 오른쪽 부분 트리, 그다음 지금 노드 순으로 들른다. 뿌리를 **마지막에** 들러야 하므로 반복으로 구현하기가 가장 까다롭다.

### 스택 두 개를 쓰는 방법

깔끔한 방법은 스택 두 개를 쓰는 것이다. 첫 스택으로 변형된 전위 순회(뿌리, 오른쪽, 왼쪽)를 돌리고, 둘째 스택으로 그 결과를 뒤집어 후위 순서(왼쪽, 오른쪽, 뿌리)를 얻는다.

```python
# === 반복적 후위 순회 (스택 둘) ===

def postorder_two_stacks(root: TreeNode | None) -> list[int]:
    """스택 두 개를 쓰는 후위 순회."""
    if root is None:
        return []
    stack1 = [root]
    stack2: list[int] = []
    while stack1:
        node = stack1.pop()
        stack2.append(node.val)
        if node.left:
            stack1.append(node.left)
        if node.right:
            stack1.append(node.right)
    return stack2[::-1]
```

### 스택 하나를 쓰는 방법

스택 하나짜리 판본은 바로 앞에 들른 노드를 좇아 왼쪽 부분 트리에서 돌아온 것인지 오른쪽에서 돌아온 것인지 가린다.

```python
# === 반복적 후위 순회 (스택 하나) ===

def postorder_one_stack(root: TreeNode | None) -> list[int]:
    """스택 하나를 쓰는 후위 순회."""
    if root is None:
        return []
    result: list[int] = []
    stack: list[TreeNode] = []
    current = root
    last_visited: TreeNode | None = None
    while current or stack:
        while current:
            stack.append(current)
            current = current.left
        peek = stack[-1]
        if peek.right and peek.right != last_visited:
            current = peek.right
        else:
            result.append(peek.val)
            last_visited = stack.pop()
    return result
```

!!! note "`last_visited` 요령"
    어떤 노드의 왼쪽 부분 트리를 들른 뒤 스택 맨 위를 엿본다. 아직 들르지 않은 오른쪽 자식이 있으면 그리로 간다. 없으면 그 노드 자신을 들르고 `last_visited`으로 표시하여 오른쪽 부분 트리에 다시 들어가지 않게 한다.

## 비교

| 순회 | 재귀 스택의 깊이 | 반복 스택의 크기 | 구현의 까다로움 |
|---|---|---|---|
| 전위 | $O(h)$ | $O(h)$ | 간단 |
| 중위 | $O(h)$ | $O(h)$ | 보통 |
| 후위 | $O(h)$ | $O(h)$ (스택 하나) 또는 $O(n)$ (스택 둘) | 까다로움 |

스택 하나를 쓰면 세 반복적 순회 모두 재귀 판본과 같이 $O(h)$의 공간을 쓴다. 스택 둘을 쓰는 후위 순회는 `stack2`이 노드 값 $n$개를 모두 담으므로 $O(n)$의 공간을 쓴다.

## 시연

```python
# === 시연 ===

if __name__ == "__main__":
    # 트리 만들기:      1
    #                 / \
    #                2   3
    #               / \
    #              4   5
    tree = TreeNode(1,
        TreeNode(2, TreeNode(4), TreeNode(5)),
        TreeNode(3))

    print(f"Preorder:  {preorder(tree)}")           # [1, 2, 4, 5, 3]
    print(f"Inorder:   {inorder(tree)}")            # [4, 2, 5, 1, 3]
    print(f"Postorder: {postorder_two_stacks(tree)}") # [4, 5, 2, 3, 1]
    print(f"Postorder: {postorder_one_stack(tree)}")  # [4, 5, 2, 3, 1]
```

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 12. MIT Press.


## 연습문제

**연습문제 1.**
반복적 순회에 대해 트리의 높이 $h$으로 시간 복잡도를 밝혀라. $h = O(\log n)$이 되는 것은 언제인가?

??? success "연습문제 1 풀이"
    이 연산은 $O(h)$ 시간이 걸린다. 균형 잡힌 트리에서는 $h = O(\log n)$이다(무작위 삽입 순서에서는 기댓값이 $O(\log n)$이고, 균형 이진 탐색 트리 계열은 이를 보장한다). 한쪽으로 치우친 트리에서는 $h = O(n)$이다(정렬된 순서로 넣은 경우).

---

**연습문제 2.**
열쇠 5, 3, 7, 1, 4, 6, 8을 차례로 넣어 만든 이진 탐색 트리에 반복적 순회을(를) 적용하라. 단계마다 보여라.

??? success "연습문제 2 풀이"
    차례로 넣어 트리를 만든 뒤 이 쪽의 연산을 적용한다. 단계마다 트리의 모습을 보이고 들르거나 고친 노드를 표시한다.

---

**연습문제 3.**
이진 탐색 트리의 높이를 $h$이라 할 때 반복적 순회이(가) $O(h)$ 시간에 끝남을 증명하라.

??? success "연습문제 3 풀이"
    이 연산은 뿌리에서 잎으로 가는 경로(또는 그 일부)를 따라가며 노드마다 $O(1)$의 일을 한다. 경로의 길이는 많아야 $h$이다. 따라서 모두 $O(h)$이다. $\square$

---

**연습문제 4.**
이진 탐색 트리의 연산은 딥러닝에서 정렬된 데이터를 다루는 일(예를 들어 빔 탐색에서 상위 $k$개의 점수를 유지하는 일)과 어떤 관계가 있는가?

??? success "연습문제 4 풀이"
    빔 탐색은 가장 좋은 가설 $k$개를 유지하므로 새 후보를 넣고 가장 나쁜 것을 빼는 일을 효율적으로 해야 한다. 이진 탐색 트리는 둘 다 $O(\log k)$에 해낸다. 실제로는 더 간단하고 상수가 작은 힙을 즐겨 쓰지만, 이진 탐색 트리는 범위 검색이나 순위 같은 더 풍부한 질의를 지원한다.