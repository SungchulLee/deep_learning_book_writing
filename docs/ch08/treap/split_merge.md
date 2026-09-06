# 트립의 쪼개기와 합치기

표준 이진 탐색 트리 연산(찾기, 삽입, 삭제)은 회전으로 트립에서도 되지만, **쪼개기**와 **합치기**라는 기본 연산이 더 깔끔한 길을 준다. 쪼개기는 열쇠 문턱값을 기준으로 트립을 둘로 나누고 합치기는 트립 둘을 하나로 모은다. 둘을 합치면 강력한 연장이 된다. 삽입은 쪼개기 한 번과 합치기 두 번으로, 삭제는 쪼개기 두 번과 합치기 한 번으로 줄어든다. 트립의 [높이](priorities.md)가 기댓값 $O(\log n)$이므로 두 연산 모두 기댓값 $O(\log n)$ 시간에 돈다.

## 합치기

`merge(L, R)`은 **$L$의 모든 열쇠가 $R$의 모든 열쇠보다 작은** 트립 $L$과 $R$을 받아 둘의 모든 원소를 담은 트립 하나를 돌려준다.

알고리즘은 우선순위의 힙 순서를 이용한다.

1. $L$이 비어 있으면 $R$을 돌려준다. $R$이 비어 있으면 $L$을 돌려준다.
2. (최대 힙 관례로) $L.\text{priority} > R.\text{priority}$이면 $L$의 뿌리가 합친 트리의 뿌리가 되어야 한다. $L.\text{right}$과 $R$을 재귀적으로 합쳐 그 결과를 $L$의 오른쪽 자식으로 붙인다.
3. 그렇지 않으면 $R$의 뿌리가 뿌리가 되어야 한다. $L$과 $R.\text{left}$을 재귀적으로 합쳐 그 결과를 $R$의 왼쪽 자식으로 붙인다.

$$
\text{merge}(L, R) = \begin{cases}
R & \text{if } L = \text{nil} \\
L & \text{if } R = \text{nil} \\
(L.\text{key},\; L.\text{left},\; \text{merge}(L.\text{right}, R)) & \text{if } L.\text{pri} > R.\text{pri} \\
(R.\text{key},\; \text{merge}(L, R.\text{left}),\; R.\text{right}) & \text{otherwise}
\end{cases}
$$

재귀 호출마다 $L$이나 $R$에서 한 층 내려가므로 전체 일은 기댓값 $O(h_L + h_R) = O(\log n)$이다.

## 쪼개기

`split(T, k)`은 트립 $T$을 $L$이 $k$ 이하의 열쇠를 모두, $R$이 $k$보다 큰 열쇠를 모두 담도록 트립 둘 $(L, R)$으로 나눈다.

1. $T$이 비어 있으면 $(\text{nil}, \text{nil})$을 돌려준다.
2. $T.\text{key} \le k$이면 $T$과 그 왼쪽 부분 트리가 $L$에 든다. $T.\text{right}$을 $k$으로 재귀적으로 쪼개 $(L', R)$을 얻는다. $T.\text{right} = L'$으로 두고 $(T, R)$을 돌려준다.
3. $T.\text{key} > k$이면 $T$과 그 오른쪽 부분 트리가 $R$에 든다. $T.\text{left}$을 $k$으로 재귀적으로 쪼개 $(L, R')$을 얻는다. $T.\text{left} = R'$으로 두고 $(L, T)$을 돌려준다.

재귀 호출마다 한 층 내려가므로 쪼개기는 기댓값 $O(\log n)$ 시간에 돈다.

## 구현

```python
"""트립의 쪼개기와 합치기 연산."""

from __future__ import annotations

import random


# === 노드 정의 ===

class TreapNode:
    """열쇠와 무작위 우선순위를 지닌 트립 노드."""

    def __init__(self, key: int):
        self.key = key
        self.priority = random.random()
        self.left: TreapNode | None = None
        self.right: TreapNode | None = None


# === 합치기 ===

def merge(left: TreapNode | None, right: TreapNode | None) -> TreapNode | None:
    """*left*의 모든 열쇠가 *right*의 모든 열쇠보다 작은 두 트립을 합친다."""
    if left is None:
        return right
    if right is None:
        return left
    if left.priority > right.priority:
        left.right = merge(left.right, right)
        return left
    else:
        right.left = merge(left, right.left)
        return right


# === 쪼개기 ===

def split(node: TreapNode | None, key: int
          ) -> tuple[TreapNode | None, TreapNode | None]:
    """트립을 (L, R)으로 쪼갠다. L은 key 이하, R은 key보다 큰 열쇠를 가진다."""
    if node is None:
        return None, None
    if node.key <= key:
        left, right = split(node.right, key)
        node.right = left
        return node, right
    else:
        left, right = split(node.left, key)
        node.left = right
        return left, node


# === 쪼개기와 합치기로 하는 삽입 ===

def insert(root: TreapNode | None, key: int) -> TreapNode:
    """쪼개기와 합치기로 트립에 열쇠를 넣는다."""
    left, right = split(root, key)
    new_node = TreapNode(key)
    return merge(merge(left, new_node), right)


# === 쪼개기와 합치기로 하는 삭제 ===

def delete(root: TreapNode | None, key: int) -> TreapNode | None:
    """쪼개기와 합치기로 트립에서 열쇠를 지운다."""
    left, right = split(root, key)
    left_without, _ = split(left, key - 1)
    return merge(left_without, right)


# === 중위 순회 ===

def inorder(node: TreapNode | None) -> list[int]:
    """열쇠를 정렬된 순서로 모은다."""
    if node is None:
        return []
    return inorder(node.left) + [node.key] + inorder(node.right)


# === 시연 ===

if __name__ == "__main__":
    root: TreapNode | None = None
    for k in [5, 3, 8, 1, 4, 7, 9]:
        root = insert(root, k)
    print(f"After inserts: {inorder(root)}")  # [1, 3, 4, 5, 7, 8, 9]

    root = delete(root, 5)
    print(f"After delete 5: {inorder(root)}")  # [1, 3, 4, 7, 8, 9]
```

## 쪼개기·합치기로 하는 삽입과 삭제

쪼개기와 합치기를 쓰면 삽입과 삭제가 간단한 짜맞춤이 된다.

**Insert(T, k):**

1. $T$을 $k$에서 $(L, R)$으로 쪼갠다.
2. 열쇠가 $k$인 노드 하나짜리 트립 $N$을 새로 만든다.
3. $\text{merge}(\text{merge}(L, N), R)$을 돌려준다.

**Delete(T, k):**

1. $T$을 $k$에서 $(L, R)$으로 쪼갠다($k$ 이하의 열쇠는 $L$에, $k$보다 큰 열쇠는 $R$에).
2. $L$을 $k - 1$에서 $(L', M)$으로 쪼갠다(열쇠가 $k$인 노드가 $M$의 뿌리이다).
3. $\text{merge}(L', R)$을 돌려준다.

## 복잡도

| 연산 | 기대 시간 |
|-----------|---------------|
| 합치기 | $O(\log n)$ |
| 쪼개기 | $O(\log n)$ |
| 삽입 (쪼개기·합치기로) | $O(\log n)$ |
| 삭제 (쪼개기·합치기로) | $O(\log n)$ |

!!! tip "회전에 견준 쪼개기·합치기의 이점"
    쪼개기·합치기 방식은 드러난 회전 논리를 피하고 (열쇠가 암묵적인 배열 색인인) **암묵 트립**으로 자연스럽게 넓어져, 뒤집기나 범위 질의 같은 수열 연산을 $O(\log n)$ 시간에 효율적으로 하게 해 준다.

## 참고 문헌

- Aragon, C. R., & Seidel, R. (1989). Randomized search trees. *30th IEEE Symposium on Foundations of Computer Science*, 540–545.
- Blelloch, G. E., & Reid-Miller, M. (1998). Fast set operations using treaps. *10th ACM Symposium on Parallel Algorithms and Architectures*, 16–26.


## 연습문제

**연습문제 1.**
트립의 쪼개기와 합치기의 균형 불변식을 밝히고 그것이 높이 $O(\log n)$을 보장함을 증명하라.

??? success "연습문제 1 풀이"
    각 구조의 불변식(균형 인수, 색의 성질, 차수 제약)이 경로 길이의 치우침을 묶는다. 높이의 한계는 그 불변식에서 따라 나온다. 트리의 층마다 (불변식이 정하는) 최소한의 노드가 있어야 하므로 전체 노드 수 $n$이 높이에 따라 지수적으로 늘고, 따라서 $h = O(\log n)$이다.

---

**연습문제 2.**
구조를 다시 짜야 하는(회전, 색 바꾸기, 쪼개기·합치기) 트리에서 트립의 쪼개기와 합치기를 따라가라. 앞뒤의 상태를 보여라.

??? success "연습문제 2 풀이"
    이 쪽에서 설명한 재구성 상황을 일으키는 트리를 하나 만들어라. 어긋난 곳을 보이고, 어느 경우에 해당하는지 가리고, 고친 뒤, 불변식이 되살아났는지 확인하라.

---

**연습문제 3.**
트립의 쪼개기와 합치기이(가) 구조를 다시 짜는 연산을 많아야 $O(\log n)$번 필요로 함을 증명하라.

??? success "연습문제 3 풀이"
    구조를 다시 짤 때마다 어긋난 곳이 뿌리에 한 층 가까워지거나 해소된다. 트리의 층이 $O(\log n)$개이므로 재구성은 많아야 $O(\log n)$번 필요하다. 레드-블랙 삽입 같은 연산에서는 회전 2번과 색 바꾸기 $O(\log n)$번이면 충분하다. $\square$

---

**연습문제 4.**
최악의 높이, 연산마다의 회전 횟수, 구현의 까다로움 면에서 트립의 쪼개기와 합치기를 다른 균형 트리 구조와 견주어라.

??? success "연습문제 4 풀이"
    AVL은 높이가 $1.44\log n$ 이하이고 삭제마다 회전이 $O(\log n)$번까지 든다. 레드-블랙은 높이가 $2\log n$ 이하이고 연산마다 회전이 많아야 3번이다. B-트리는 높이가 $O(\log_B n)$이며 디스크 입출력에 맞추어져 있다. 스플레이 트리는 분할 상환으로 $O(\log n)$이지만 최악은 $O(n)$이다.