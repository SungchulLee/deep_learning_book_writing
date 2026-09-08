# 레벨 순서 순회

전위·중위·후위 순회가 트리를 깊이 우선으로 살피는 데 반해 **레벨 순서 순회**는 너비 우선으로 노드를 들른다. 깊이 0의 노드(뿌리)를 모두 들르고, 이어 깊이 1, 그다음 깊이 2, 이렇게 이어 간다. 이는 큐로 발견한 순서대로 노드를 처리하는 트리 너비 우선 탐색과 같다.

---

## 1. 알고리즘

레벨 순서 순회는 먼저 넣은 것이 먼저 나오는 큐를 쓴다.

1. 뿌리를 큐에 넣는다.
2. 큐가 비지 않은 동안 다음을 되풀이한다.
    - 큐에서 노드를 꺼내 들른다.
    - 왼쪽 자식이 있으면 큐에 넣는다.
    - 오른쪽 자식이 있으면 큐에 넣는다.

큐 덕분에 깊이 $d$의 모든 노드가 깊이 $d+1$의 어떤 노드보다 먼저 처리된다. 깊이 $d$의 노드가 (깊이 $d+1$인) 자식을 넣을 때는 다른 깊이 $d$의 노드가 이미 모두 큐에 들어 있기 때문이다.

??? example "레벨 순서 순회"
    다음 트리를 생각해 보자.

    ```
            1
           / \
          2   3
         / \   \
        4   5   6
    ```

    | 큐의 상태 | 꺼냄 | 넣음 |
    |---|---|---|
    | [1] | 1 | 2, 3 |
    | [2, 3] | 2 | 4, 5 |
    | [3, 4, 5] | 3 | 6 |
    | [4, 5, 6] | 4 | -- |
    | [5, 6] | 5 | -- |
    | [6] | 6 | -- |

    들르는 순서: **1, 2, 3, 4, 5, 6** (층마다 왼쪽에서 오른쪽으로).

---

## 2. 구현

### 기본 레벨 순서 순회

```python
"""이진 트리의 레벨 순서(너비 우선) 순회."""

from __future__ import annotations
from collections import deque

# === 노드 정의 ===

class TreeNode:
    """이진 트리 노드."""

    def __init__(self, val: int = 0, left: TreeNode | None = None,
                 right: TreeNode | None = None):
        self.val = val
        self.left = left
        self.right = right

# === 기본 레벨 순서 순회 ===

def level_order(root: TreeNode | None) -> list[int]:
    """노드 값을 레벨 순서로 돌려준다."""
    if root is None:
        return []
    result: list[int] = []
    queue: deque[TreeNode] = deque([root])
    while queue:
        node = queue.popleft()
        result.append(node.val)
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    return result
```

### 층별로 묶기

흔한 변형은 노드를 층별로 묶어 리스트의 리스트를 돌려주는 것이다. 다음 층으로 넘어가기 전에 지금 층의 노드를 한 묶음으로 처리하면 된다.

```python
# === 층별로 묶기 ===

def level_order_grouped(root: TreeNode | None) -> list[list[int]]:
    """노드 값을 층별로 묶어 돌려준다."""
    if root is None:
        return []
    result: list[list[int]] = []
    queue: deque[TreeNode] = deque([root])
    while queue:
        level_size = len(queue)
        level: list[int] = []
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result.append(level)
    return result
```

핵심 요령은 층이 시작될 때 `len(queue)`을 붙잡아 두는 것이다. 그 값이 지금 층에 속한 노드 수를 정확히 알려 주고, 그 뒤에 넣는 노드는 모두 다음 층에 속한다.

### 지그재그(나선) 순서

지그재그 순회는 층마다 방향을 바꾼다. 짝수 깊이에서는 왼쪽에서 오른쪽으로, 홀수 깊이에서는 오른쪽에서 왼쪽으로 간다.

```python
# === 지그재그 레벨 순서 ===

def zigzag_level_order(root: TreeNode | None) -> list[list[int]]:
    """노드 값을 지그재그(나선) 레벨 순서로 돌려준다."""
    if root is None:
        return []
    result: list[list[int]] = []
    queue: deque[TreeNode] = deque([root])
    left_to_right = True
    while queue:
        level_size = len(queue)
        level: list[int] = []
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        if not left_to_right:
            level.reverse()
        result.append(level)
        left_to_right = not left_to_right
    return result
```

---

## 3. 복잡도

| 항목 | 복잡도 |
|---|---|
| 시간 | $O(n)$ — 노드마다 꼭 한 번씩 넣고 꺼낸다 |
| 공간 | $O(w)$, 여기서 $w$은 트리의 최대 너비 |

최대 너비 $w$은 가장 넓은 층에서 나온다. 높이가 $h$인 완전 이진 트리에서는 마지막 층에 노드가 $2^h$개 있으므로 최악의 경우 $w = O(n)$이다. 치우친 트리에서는 $w = O(1)$이다.

!!! note "깊이 우선 순회와의 공간 비교"
    깊이 우선 순회(전위·중위·후위)는 높이를 $h$이라 할 때 $O(h)$의 공간을 쓴다. 레벨 순서 순회는 최대 너비를 $w$이라 할 때 $O(w)$의 공간을 쓴다. 균형 잡힌 트리에서는 $h = O(\log n)$인데 $w = O(n)$이므로 깊이 우선이 공간을 덜 쓴다. 치우친 트리에서는 $h = O(n)$인데 $w = O(1)$이므로 레벨 순서가 공간을 덜 쓴다.

---

## 4. 응용

- 벌레잡이나 시각화를 위해 **트리를 층별로 출력하기**.
- 트리의 **최소 깊이 찾기**(너비 우선 탐색이 처음 만나는 잎이 최소 깊이에 있다).
- 이진 트리의 **직렬화와 역직렬화**(레벨 순서 부호화를 흔히 쓴다).
- **같은 층의 노드 잇기**(예를 들어 "오른쪽 다음" 포인터 채우기).
- 최대 너비 질의를 위해 층마다 **너비 계산하기**.

---

## 5. 시연

```python
# === 시연 ===

if __name__ == "__main__":
    tree = TreeNode(1,
        TreeNode(2, TreeNode(4), TreeNode(5)),
        TreeNode(3, None, TreeNode(6)))

    print(f"Level-order:  {level_order(tree)}")
    print(f"By level:     {level_order_grouped(tree)}")
    print(f"Zigzag:       {zigzag_level_order(tree)}")
```

---

## 연습문제

**연습문제 1.**
레벨 순서 순회에 대해 트리의 높이 $h$으로 시간 복잡도를 밝혀라. $h = O(\log n)$이 되는 것은 언제인가?

??? success "연습문제 1 풀이"
    이 연산은 $O(h)$ 시간이 걸린다. 균형 잡힌 트리에서는 $h = O(\log n)$이다(무작위 삽입 순서에서는 기댓값이 $O(\log n)$이고, 균형 이진 탐색 트리 계열은 이를 보장한다). 한쪽으로 치우친 트리에서는 $h = O(n)$이다(정렬된 순서로 넣은 경우).

---

**연습문제 2.**
열쇠 5, 3, 7, 1, 4, 6, 8을 차례로 넣어 만든 이진 탐색 트리에 레벨 순서 순회을(를) 적용하라. 단계마다 보여라.

??? success "연습문제 2 풀이"
    차례로 넣어 트리를 만든 뒤 이 쪽의 연산을 적용한다. 단계마다 트리의 모습을 보이고 들르거나 고친 노드를 표시한다.

---

**연습문제 3.**
이진 탐색 트리의 높이를 $h$이라 할 때 레벨 순서 순회이(가) $O(h)$ 시간에 끝남을 증명하라.

??? success "연습문제 3 풀이"
    이 연산은 뿌리에서 잎으로 가는 경로(또는 그 일부)를 따라가며 노드마다 $O(1)$의 일을 한다. 경로의 길이는 많아야 $h$이다. 따라서 모두 $O(h)$이다. $\square$

---

**연습문제 4.**
이진 탐색 트리의 연산은 딥러닝에서 정렬된 데이터를 다루는 일(예를 들어 빔 탐색에서 상위 $k$개의 점수를 유지하는 일)과 어떤 관계가 있는가?

??? success "연습문제 4 풀이"
    빔 탐색은 가장 좋은 가설 $k$개를 유지하므로 새 후보를 넣고 가장 나쁜 것을 빼는 일을 효율적으로 해야 한다. 이진 탐색 트리는 둘 다 $O(\log k)$에 해낸다. 실제로는 더 간단하고 상수가 작은 힙을 즐겨 쓰지만, 이진 탐색 트리는 범위 검색이나 순위 같은 더 풍부한 질의를 지원한다.

## 정리하며

이 마당은 알고리즘、구현、복잡도、응용을 차례로 짚었다.

**참고 문헌**

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 12. MIT Press.
