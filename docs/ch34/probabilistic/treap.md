# 트립

AVL이나 붉은-검은 나무 같은 고른 이진 찾기 나무는 정해진 불변량과 돌리기로 고름을 지킨다. **트립**(나무 + 더미)은 그 대신 마구잡이로 어림 $O(\log n)$ 고름을 이룬다. 마디마다 아무 우선값을 받고, 나무는 열쇠에 대해 이진 찾기 나무 성질을, 우선값에 대해 더미 성질을 함께 채운다. 이 어우름이 나무 꼴을 하나로 정하며, 원소를 아무 차례로 넣어 만든 마구잡이 이진 찾기 나무와 같은 것을 내놓는다.

## 정의

트립은 마디마다 (열쇠, 우선값) 짝을 갈무리하는 이진 나무로 다음을 채운다.

1. **이진 찾기 나무 성질**: 마디 $x$마다 왼쪽 밑나무의 온 열쇠가 $x.\text{열쇠}$보다 작고 오른쪽 밑나무의 온 열쇠가 더 크다.
2. **더미 성질**: 마디 $x$마다 $x.\text{우선값} \ge$ 두 자식의 우선값이다(우선값에 대한 가장 큰 더미).

우선값이 모두 다르면 주어진 (열쇠, 우선값) 짝 모임에 대해 트립 얼개가 하나뿐이다.

## 어림 높이

우선값을 서로 매이지 않고 고르게 아무렇게나 뽑으면, 그 트립은 열쇠를 고르게 아무 차례로 넣어 지은 **마구잡이 이진 찾기 나무**와 같은 분포를 지닌다. 아무 마디의 어림 깊이는 다음과 같다.

$$
E[\text{순위 } k \text{ 마디의 깊이}] = H_k + H_{n-k+1} - 1
$$

여기서 $H_k = \sum_{i=1}^{k} 1/i$은 $k$번째 조화수다. 온 마디에 걸친 어림 깊이의 최댓값은 다음과 같다.

$$
E[h] = O(\log n)
$$

## 돌리기

넣은 뒤 더미 성질이 깨지면 나무 돌리기가 이진 찾기 나무 성질을 깨지 않고 그것을 되살린다.

- 마디 $x$에서 **오른쪽 돌리기**: 왼쪽 자식 $y$이 어버이가 되고 $x$이 $y$의 오른쪽 자식이 된다.
- 마디 $x$에서 **왼쪽 돌리기**: 오른쪽 자식 $y$이 어버이가 되고 $x$이 $y$의 왼쪽 자식이 된다.

## 넣기

(열쇠 $k$, 우선값 $p$)을 넣으려면:

1. $k$에 따라 여느 이진 찾기 나무 넣기를 벌여 새 마디를 잎으로 둔다.
2. 새 마디의 우선값이 어버이의 우선값을 넘는 동안 새 마디를 위로 돌린다(왼쪽 자식이면 오른쪽 돌리기, 오른쪽 자식이면 왼쪽 돌리기).

돌리기 횟수는 처음 넣은 곳의 깊이와 같으므로 다음과 같다.

$$
E[T_{\text{넣기}}] = O(\log n)
$$

## 지우기

열쇠 $k$을 지우려면:

1. 열쇠가 $k$인 마디 $x$을 찾는다.
2. $x$이 잎이 될 때까지 (우선값이 더 높은 자식 쪽으로) $x$을 아래로 돌린다.
3. 그 잎을 없앤다.

$$
E[T_{\text{지우기}}] = O(\log n)
$$

## 쪼개기와 아우르기

트립은 좋은 쪼개기와 아우르기 연산을 받쳐 준다.

**쪼개기(뿌리, 열쇠)**: 트립을 트립 둘 $L$과 $R$으로 쪼갠다. $L$의 온 열쇠는 그 열쇠 $\le$이고 $R$의 온 열쇠는 $>$이다. 어림 때 $O(\log n)$.

**아우르기($L$, $R$)**: $L$의 온 열쇠가 $R$의 온 열쇠보다 작은 트립 둘을 아우른다. 뿌리 우선값을 견주고 되돌아 들어가며 아우른다. 어림 때 $O(\log n)$.

## 구현

```python
"""
트립 -- 더미 차례를 지닌 우선값을 쓰는 마구잡이 이진 찾기 나무.

아무 우선값을 매기고 더미 성질을 지켜 찾기, 넣기, 지우기에
어림 O(log n) 때를 이룬다.
"""

from __future__ import annotations
import random
from dataclasses import dataclass, field


# === 트립 마디 ================================================================

@dataclass
class TreapNode:
    """열쇠와 아무 우선값을 갈무리하는 마디."""
    key: int
    priority: float = field(default_factory=random.random)
    left: TreapNode | None = None
    right: TreapNode | None = None


# === 돌리기 ===================================================================

def rotate_right(node: TreapNode) -> TreapNode:
    """오른쪽 돌리기: 왼쪽 자식이 뿌리가 된다."""
    new_root = node.left
    node.left = new_root.right
    new_root.right = node
    return new_root


def rotate_left(node: TreapNode) -> TreapNode:
    """왼쪽 돌리기: 오른쪽 자식이 뿌리가 된다."""
    new_root = node.right
    node.right = new_root.left
    new_root.left = node
    return new_root


# === 트립 연산 ================================================================

def insert(root: TreapNode | None, key: int) -> TreapNode:
    """아무 우선값과 함께 *key*을 넣고 두 성질을 모두 지킨다."""
    if root is None:
        return TreapNode(key)
    if key < root.key:
        root.left = insert(root.left, key)
        if root.left.priority > root.priority:
            root = rotate_right(root)
    elif key > root.key:
        root.right = insert(root.right, key)
        if root.right.priority > root.priority:
            root = rotate_left(root)
    return root  # 겹치는 열쇠: 바뀌지 않는다


def search(root: TreapNode | None, key: int) -> bool:
    """여느 이진 찾기 나무 찾기로 *key*을 찾는다."""
    if root is None:
        return False
    if key == root.key:
        return True
    elif key < root.key:
        return search(root.left, key)
    else:
        return search(root.right, key)


def delete(root: TreapNode | None, key: int) -> TreapNode | None:
    """*key*을 잎까지 아래로 돌려 지운다."""
    if root is None:
        return None
    if key < root.key:
        root.left = delete(root.left, key)
    elif key > root.key:
        root.right = delete(root.right, key)
    else:
        # 지울 마디를 찾았다
        if root.left is None:
            return root.right
        elif root.right is None:
            return root.left
        elif root.left.priority > root.right.priority:
            root = rotate_right(root)
            root.right = delete(root.right, key)
        else:
            root = rotate_left(root)
            root.left = delete(root.left, key)
    return root


def inorder(root: TreapNode | None) -> list[int]:
    """매긴 열쇠를 돌려주는 가운데 먼저 훑기."""
    if root is None:
        return []
    return inorder(root.left) + [root.key] + inorder(root.right)


def height(root: TreapNode | None) -> int:
    """트립의 높이를 셈한다."""
    if root is None:
        return -1
    return 1 + max(height(root.left), height(root.right))


# === 메인 =====================================================================

if __name__ == "__main__":
    random.seed(42)
    root = None
    keys = [5, 3, 7, 1, 4, 6, 8, 2, 9]
    for k in keys:
        root = insert(root, k)

    print(f"Sorted: {inorder(root)}")
    print(f"Height: {height(root)}")
    print(f"Search 4: {search(root, 4)}")
    print(f"Search 10: {search(root, 10)}")

    root = delete(root, 5)
    print(f"After deleting 5: {inorder(root)}")
```

**출력:**

```
Sorted: [1, 2, 3, 4, 5, 6, 7, 8, 9]
Height: 4
Search 4: True
Search 10: False
After deleting 5: [1, 2, 3, 4, 6, 7, 8, 9]
```

가운데 먼저 훑기가 이진 찾기 나무 성질을 알려 주고, 높이는 $\log_2 9 \approx 3.2$에 가까우며(마구잡이가 나무를 고르게 지킨다), 지우기가 차례를 지키면서 열쇠를 옳게 없앤다.

## 참고 문헌

- Seidel, R. and Aragon, C.R. "Randomized Search Trees." *Algorithmica*, 1996
- [Advanced Data Structures (Brass)](https://www.cambridge.org/core/books/advanced-data-structures/D56E2269D7CEE969A3B8105D3541F601)

## 연습문제

**연습문제 1.**
열쇠 5, 3, 7, 1, 4을 저마다 아무 우선값 10, 30, 20, 5, 25으로 트립에 넣어라. 그 나무를 그리고 이진 찾기 나무 성질과 더미 성질을 모두 살펴라.

??? success "연습문제 1 풀이"
    우선값: 5$\to$10, 3$\to$30, 7$\to$20, 1$\to$5, 4$\to$25. 더미 성질은 우선값이 높은 마디가 뿌리에 가까워야 한다고 바란다(우선값에 대한 가장 큰 더미). 우선값 내림차순으로 매기면 3(30), 4(25), 7(20), 5(10), 1(5)이다. 뿌리는 (우선값이 가장 높은) 3이다. 이진 찾기 나무 성질: 왼쪽 밑나무의 열쇠는 $< 3$이고 오른쪽은 $> 3$이다. 3의 왼쪽 밑나무: 우선값 5의 열쇠 1뿐이다. 3의 오른쪽 밑나무: 열쇠 4, 5, 7. 이 가운데 4의 우선값이 가장 높으므로(25) 4이 3의 오른쪽 자식이다. 4의 왼쪽: 없다(3과 4 사이의 열쇠가 없다). 4의 오른쪽: 열쇠 5, 7. 이 가운데 7의 우선값 20이 5의 우선값 10보다 크므로 7이 4의 오른쪽 자식이고 5은 7의 왼쪽 자식이다. 마지막 나무: 3(30)의 왼쪽=1(5), 오른쪽=4(25). 4(25)의 오른쪽=7(20). 7(20)의 왼쪽=5(10). 이진 찾기 나무 성질과 가장 큰 더미 성질이 모두 성립한다. $\square$

---

**연습문제 2.**
열쇠 $n$개와 아무 우선값을 지닌 트립이 마구잡이 이진 찾기 나무(열쇠를 아무 차례로 넣어 지은 이진 찾기 나무)와 같은 어림 얼개를 지님을 증명하여라.

??? success "연습문제 2 풀이"
    마구잡이 이진 찾기 나무에서 뿌리는 처음 넣은 열쇠이며, 이는 열쇠 $n$개 가운데 어느 것이든 똑같이 될 낌새를 지닌다. 트립에서 뿌리는 아무 우선값이 가장 높은 열쇠다. 우선값이 이어진 분포에서 서로 매이지 않고 같은 분포로 뽑히므로 열쇠마다 가장 큰 우선값을 지닐 낌새가 같다. 따라서 두 모형 모두 열쇠 $n$개에서 뿌리를 고르게 아무렇게나 고른다. 뿌리 $r$이 주어지면 이진 찾기 나무 성질이 나머지 열쇠를 $< r$(왼쪽 밑나무)과 $> r$(오른쪽 밑나무)으로 가른다. 마구잡이 이진 찾기 나무에서 각 갈래 안의 넣는 차례는 아무 차례이다. 트립에서 각 갈래 안의 우선값은 서로 매이지 않고 같은 분포이므로 같은 따짐으로 각 갈래가 마구잡이 얼개를 지닌다. $n$에 대한 귀납법으로 나무 꼴에 대한 두 분포가 같다. 따라서 마구잡이 이진 찾기 나무의 온 어림 성질(어림 깊이 $O(\log n)$, 어림 높이 $O(\log n)$)이 트립에도 그대로 듣는다. $\square$

---

**연습문제 3.**
트립의 쪼개기와 아우르기 연산을 밝혀라. 어림 때 복잡도는 얼마이며 왜 쓸모 있는가?

??? success "연습문제 3 풀이"
    **쪼개기($T$, $k$)**: 트립 $T$을 트립 둘 $L$과 $R$으로 쪼갠다. $L$은 $\le k$인 온 열쇠를, $R$은 $> k$인 온 열쇠를 담는다. 알고리즘: $T$이 비었으면 (빈 것, 빈 것)을 돌려준다. 뿌리의 열쇠가 $\le k$이면 오른쪽 밑나무를 되돌아 들어가며 쪼갠다. 뿌리와 왼쪽 밑나무는 $L$으로 가고 쪼갠 오른쪽 몫은 $R$으로 간다. 아니면 왼쪽 밑나무를 되돌아 들어가며 쪼개고 뿌리와 오른쪽 밑나무는 $R$으로 간다. **아우르기($L$, $R$)**: $L$의 온 열쇠가 $R$의 온 열쇠보다 작은 트립 둘을 아우른다. 어느 하나가 비었으면 다른 것을 돌려준다. $L$의 뿌리의 우선값이 더 높으면 $L$의 뿌리가 새 뿌리가 되고 그 왼쪽 밑나무는 그대로, 오른쪽 밑나무는 아우르기($L$.right, $R$)이다. 아니면 $R$의 뿌리로 맞바꿔 똑같이 한다. 어림 때: 둘 다 (높이에 견주어) $O(\log n)$이다. 이 연산들이 좋은 넣기(쪼개고 아우르기), 지우기(쪼개고 아우르기), 구간 연산(두 곳에서 쪼개고 다룬 뒤 다시 아우르기)을 이루게 한다. $\square$

---

**연습문제 4.**
쪼개기와 아우르기를 받쳐 주는 움직이는 열을 지니는 데 트립을 쓴다. 범위 합 물음에 $O(\log n)$ 때에 답하도록 덧붙이는 길을 풀어라.

??? success "연습문제 4 풀이"
    마디마다 그 밑나무의 온 값 합을 갈무리하는 `sum` 밭을 덧붙인다. 곧 `node.sum = node.value + node.left.sum + node.right.sum`이다. 쪼개고 아우르는 동안 `sum`을 고친다(되돌아 들어가는 부름마다 고친 마디의 `sum`을 $O(1)$에 고친다). $[a, b]$ 안 열쇠의 범위 합 물음에서는 트립을 $a-1$에서 쪼개어 $(L, R)$을 얻고, 이어 $R$을 $b$에서 쪼개어 $(M, R')$을 얻는다. 답은 $M.\text{뿌리.sum}$이다. $M$과 $R'$을 아우르고 다시 $L$과 아울러 트립을 되살린다. 온통 쪼개기 세 번과 아우르기 두 번이며 저마다 $O(\log n)$이다. `sum`을 알맞은 연산으로 갈음하면 결합 법칙이 성립하는 아무 모으기(최댓값, 최솟값, 최대공약수)로도 넓힐 수 있다. $\square$

---

**연습문제 5.**
영속(함수형) 자료 얼개로 쓸 때 트립과 붉은-검은 나무를 견주어라. 어느 쪽이 영속으로 만들기 쉬우며 그 까닭은 무엇인가?

??? success "연습문제 5 풀이"
    **트립**이 영속으로 만들기 쉽다. 쪼개기와 아우르기가 절로 위에서 아래로 흐르고 길 하나를 따라 새 마디를 만들므로, 길 베끼기로 연산마다 새 마디 $O(\log n)$개를 낳는다. 넣기와 지우기를 쪼개기와 아우르기의 어우름으로 나타내므로 같은 영속 거동을 물려받는다. 돌리기가 어림할 수 없게 번지지 않고, 얼개가 바뀌지 않는 우선값으로 정해진다. **붉은-검은 나무**는 여러 켜의 여러 마디에 미칠 수 있는 돌리기와 다시 칠하기가 있어야 한다. 이를 영속으로 만들려면 넣기 길뿐 아니라 돌리기에 걸리는 마디(동기, 삼촌)까지 베껴야 한다. 점근 비용은 같지만(새 마디 $O(\log n)$개) 돌리기 경우를 영속으로 다루어야 하므로 만들기가 훨씬 얽힌다. 트립은 쉬움 덕분에(쪼개기와 아우르기라는 한가운데 연산 둘) 겨루기 짜기에서 영속 매긴 그릇으로 즐겨 쓴다. $\square$
