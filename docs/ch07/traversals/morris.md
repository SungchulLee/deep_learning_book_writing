# 모리스 순회

재귀 순회든 [반복 순회](iterative.md)든 호출 스택이나 명시적인 스택으로 $O(h)$의 보조 공간을 쓴다. **모리스 순회**(Morris, 1979)는 이 공간 부담을 아예 없앤다. 트리의 구조를 잠시 고쳐 왼쪽 부분 트리마다 가장 오른쪽 노드에서 그 중위 후속자로 가는 **실 연결**을 만들어 $O(1)$의 보조 공간으로 중위 순회를 해낸다. 이 실 덕분에 스택 없이도 조상 노드로 돌아갈 수 있다. 순회가 끝나면 고친 것을 모두 되돌려 트리가 원래 모습으로 남는다.

---

## 1. 핵심 착상: 실 달린 이진 트리

보통의 이진 트리에는 `nil`인 오른쪽 자식 포인터가 많다. 모리스 순회는 이 널 포인터를 잠시 중위 후속자를 가리키게 하여 활용한다. 구체적으로, 왼쪽 부분 트리가 있는 노드 $x$에 대해 다음과 같이 둔다.

$$
\text{rightmost node of } x.\text{left} \;\rightarrow\; x
$$

이 실 덕분에 $x$의 왼쪽 부분 트리를 끝낸 뒤 스택 없이 $x$으로 돌아올 수 있다.

---

## 2. 알고리즘

알고리즘은 뿌리에서 시작하는 포인터 `current` 하나만 둔다. 단계마다 다음을 한다.

**경우 1: `current.left`이 `nil`이다.** `current`을 들르고 `current.right`으로 옮긴다.

**경우 2: `current.left`이 `nil`이 아니다.** `current`의 중위 선행자, 곧 `current`의 왼쪽 부분 트리에서 가장 오른쪽 노드를 찾는다.

- **작은 경우 2a:** 선행자의 오른쪽 자식이 `nil`이다. 왼쪽 부분 트리를 아직 훑지 않았다는 뜻이다. 실을 만든다: `predecessor.right = current`으로 둔다. `current.left`으로 옮긴다.
- **작은 경우 2b:** 선행자의 오른쪽 자식이 `current`이다. 왼쪽 부분 트리를 다 훑고 실을 타고 돌아왔다는 뜻이다. 실을 없앤다: `predecessor.right = nil`으로 둔다. `current`을 들르고 `current.right`으로 옮긴다.

`current`이 `nil`이 되면 순회가 끝난다.

??? example "모리스 순회 한 단계씩 따라가기"
    다음 트리를 생각해 보자.

    ```
          4
         / \
        2   5
       / \
      1   3
    ```

    | 단계 | current | 하는 일 | 만든/없앤 실 |
    |------|---------|--------|------------------------|
    | 1 | 4 | 왼쪽이 있다. 4의 선행자는 3. 3.right은 nil | 실 만들기: 3 -> 4. 2로 옮김 |
    | 2 | 2 | 왼쪽이 있다. 2의 선행자는 1. 1.right은 nil | 실 만들기: 1 -> 2. 1로 옮김 |
    | 3 | 1 | 왼쪽 자식 없음 | **1 들름**. 1.right = 2(실)로 옮김 |
    | 4 | 2 | 왼쪽이 있다. 2의 선행자는 1. 1.right = 2 (실 발견) | 실 없앰. **2 들름**. 3으로 옮김 |
    | 5 | 3 | 왼쪽 자식 없음 | **3 들름**. 3.right = 4(실)로 옮김 |
    | 6 | 4 | 왼쪽이 있다. 4의 선행자는 3. 3.right = 4 (실 발견) | 실 없앰. **4 들름**. 5로 옮김 |
    | 7 | 5 | 왼쪽 자식 없음 | **5 들름**. nil로 옮김 |

    중위 순회 결과: **1, 2, 3, 4, 5**

---

## 3. 구현

```python
"""모리스 순회: 공간 O(1)로 하는 이진 트리의 중위 순회."""

from __future__ import annotations

# === 노드 정의 ===

class TreeNode:
    """이진 트리 노드."""

    def __init__(self, val: int = 0, left: TreeNode | None = None,
                 right: TreeNode | None = None):
        self.val = val
        self.left = left
        self.right = right

# === 모리스 중위 순회 ===

def morris_inorder(root: TreeNode | None) -> list[int]:
    """보조 공간 O(1)로 하는 중위 순회."""
    result: list[int] = []
    current = root
    while current is not None:
        if current.left is None:
            # 경우 1: 왼쪽 부분 트리 없음 — 들르고 오른쪽으로
            result.append(current.val)
            current = current.right
        else:
            # 중위 선행자 찾기
            predecessor = current.left
            while predecessor.right is not None and predecessor.right is not current:
                predecessor = predecessor.right

            if predecessor.right is None:
                # 경우 2a: 실을 만들고 왼쪽으로
                predecessor.right = current
                current = current.left
            else:
                # 경우 2b: 실을 없애고 들른 뒤 오른쪽으로
                predecessor.right = None
                result.append(current.val)
                current = current.right
    return result

# === 모리스 전위 순회 ===

def morris_preorder(root: TreeNode | None) -> list[int]:
    """보조 공간 O(1)로 하는 전위 순회."""
    result: list[int] = []
    current = root
    while current is not None:
        if current.left is None:
            result.append(current.val)
            current = current.right
        else:
            predecessor = current.left
            while predecessor.right is not None and predecessor.right is not current:
                predecessor = predecessor.right

            if predecessor.right is None:
                # 왼쪽으로 가기 전에 지금 노드를 들른다 (전위)
                result.append(current.val)
                predecessor.right = current
                current = current.left
            else:
                predecessor.right = None
                current = current.right
    return result

# === 시연 ===

if __name__ == "__main__":
    tree = TreeNode(4,
        TreeNode(2, TreeNode(1), TreeNode(3)),
        TreeNode(5))

    print(f"Morris inorder:  {morris_inorder(tree)}")   # [1, 2, 3, 4, 5]
    print(f"Morris preorder: {morris_preorder(tree)}")  # [4, 2, 1, 3, 5]
```

**출력:**

```
Morris inorder:  [1, 2, 3, 4, 5]
Morris preorder: [4, 2, 1, 3, 5]
```

---

## 4. 전체 일의 양이 $O(n)$인 까닭

알고리즘이 선행자를 거듭 찾기는 하지만 분할 상환으로 따져 보면 전체 일의 양은 선형이다. 노드마다 `current`이 되는 것은 많아야 두 번, 곧 선행자에서 오는 실을 만들 때 한 번과 그 실을 없앨 때 한 번이다. 선행자를 찾는 동안에도 노드를 건드리지만, 그런 걸음은 오른쪽 자식 포인터의 사슬을 따라가고, 모든 선행자 찾기를 통틀어 트리의 변마다 많아야 두 번(실을 만들 때 한 번, 알아채고 없앨 때 한 번) 지난다.

트리의 변이 $n - 1$개이므로 모든 선행자 찾기의 총비용은 $O(n)$이다.

---

## 5. 복잡도

이로부터 다음과 같은 복잡도가 나온다.

| 항목 | 복잡도 |
|---|---|
| 시간 | $O(n)$ |
| 보조 공간 | $O(1)$ |

$O(1)$의 공간이 모리스 순회의 결정적인 장점이다. 그 대가로 트리를 잠시 고치지만 순회가 끝나기 전에 온전히 되돌린다.

!!! warning "스레드 안전성"
    모리스 순회는 실행 중에 트리를 고친다. 같은 트리에 대한 다른 연산과 동시에 돌리면 안전하지 않다. 스레드 안전성이 필요하면 스택을 쓰는 반복적 순회를 쓰라.

---

## 6. 모리스 전위 순회와 모리스 중위 순회

모리스 전위 순회와 중위 순회의 유일한 차이는 노드를 **언제** 들르느냐이다.

| 순회 | 들르는 시점 |
|---|---|
| 모리스 중위 | 실을 **없앨** 때 들른다 (경우 2b) |
| 모리스 전위 | 실을 **만들** 때 들른다 (경우 2a) |

모리스 후위 순회는 더 까다롭다. 후위 순회는 부모보다 자식을 먼저 들러야 하는데 이는 실이 놓이는 자연스러운 방향과 어긋난다. 표준적인 방법은 실을 없애는 단계에서 왼쪽 부분 트리마다 오른쪽 등뼈를 뒤집는 것인데, 실제로 이 기법이 필요한 일은 드물다.

---

## 연습문제

**연습문제 1.**
모리스 순회에 대해 트리의 높이 $h$으로 시간 복잡도를 밝혀라. $h = O(\log n)$이 되는 것은 언제인가?

??? success "연습문제 1 풀이"
    이 연산은 $O(h)$ 시간이 걸린다. 균형 잡힌 트리에서는 $h = O(\log n)$이다(무작위 삽입 순서에서는 기댓값이 $O(\log n)$이고, 균형 이진 탐색 트리 계열은 이를 보장한다). 한쪽으로 치우친 트리에서는 $h = O(n)$이다(정렬된 순서로 넣은 경우).

---

**연습문제 2.**
열쇠 5, 3, 7, 1, 4, 6, 8을 차례로 넣어 만든 이진 탐색 트리에 모리스 순회을(를) 적용하라. 단계마다 보여라.

??? success "연습문제 2 풀이"
    차례로 넣어 트리를 만든 뒤 이 쪽의 연산을 적용한다. 단계마다 트리의 모습을 보이고 들르거나 고친 노드를 표시한다.

---

**연습문제 3.**
이진 탐색 트리의 높이를 $h$이라 할 때 모리스 순회이(가) $O(h)$ 시간에 끝남을 증명하라.

??? success "연습문제 3 풀이"
    이 연산은 뿌리에서 잎으로 가는 경로(또는 그 일부)를 따라가며 노드마다 $O(1)$의 일을 한다. 경로의 길이는 많아야 $h$이다. 따라서 모두 $O(h)$이다. $\square$

---

**연습문제 4.**
이진 탐색 트리의 연산은 딥러닝에서 정렬된 데이터를 다루는 일(예를 들어 빔 탐색에서 상위 $k$개의 점수를 유지하는 일)과 어떤 관계가 있는가?

??? success "연습문제 4 풀이"
    빔 탐색은 가장 좋은 가설 $k$개를 유지하므로 새 후보를 넣고 가장 나쁜 것을 빼는 일을 효율적으로 해야 한다. 이진 탐색 트리는 둘 다 $O(\log k)$에 해낸다. 실제로는 더 간단하고 상수가 작은 힙을 즐겨 쓰지만, 이진 탐색 트리는 범위 검색이나 순위 같은 더 풍부한 질의를 지원한다.

## 정리하며

이 마당은 핵심 착상: 실 달린 이진 트리、알고리즘、구현、전체 일의 양이 $O(n)$인 까닭을 차례로 짚었다.

**참고 문헌**

- Morris, J. H. (1979). Traversing binary trees simply and cheaply. *Information Processing Letters*, 9(5), 197–200.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 12. MIT Press.
