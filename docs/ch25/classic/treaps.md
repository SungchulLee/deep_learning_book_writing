# 마구잡이 트립

트립(나무 + 무지개탑)은 열쇠에는 이진 찾기 나무 차례를, 마구잡이로 매긴 우선값에는 무지개탑 차례를 지키는 마구잡이 이진 찾기 나무이다. 우선값을 고르게 아무렇게나 고르면 트립의 짜임은 마구잡이 이진 찾기 나무와 같아져, AVL이나 붉은-검은 나무의 복잡한 다시 고르기 논리 없이도 기댓값 높이 $O(\log n)$과 찾기, 넣기, 지우기의 기댓값 시간 $O(\log n)$을 준다.

## 정의

**트립**은 마디마다 열쇠-우선값 짝 $(k, p)$을 담고 두 성질을 함께 채우는 이진 나무이다.

1. **열쇠의 이진 찾기 나무 성질:** 마디마다 왼쪽 아래 나무의 열쇠는 모두 작고 오른쪽 아래 나무의 열쇠는 모두 크다.
2. **우선값의 최소 무지개탑 성질:** 마디마다 우선값이 그 자식의 우선값보다 작거나 같다.

!!! note "하나뿐인 짜임"
    서로 다른 열쇠 $n$개와 서로 다른 우선값 $n$개가 주어지면 두 성질을 모두 채우는 트립은 꼭 하나뿐이다. 짜임은 우선값의 차례로 하나뿐이게 정해진다.

## 아무 우선값이 통하는 까닭

열쇠마다 고르게 아무 우선값을 매기면 짜임이 (분포로) **마구잡이 이진 찾기 나무**, 곧 열쇠를 아무 차례로 넣어 세운 이진 찾기 나무와 똑같은 트립이 나온다. 마구잡이 이진 찾기 나무의 기댓값 깊이는 $O(\log n)$이고 기댓값 높이는 $\Theta(\log n)$이다.

**핵심 통찰:** 우선값이 가장 작은 마디가 뿌리가 된다. 그 열쇠가 남은 마디를 왼쪽과 오른쪽 아래 나무로 가르고 이 과정이 되돌이된다. 이는 아무 넣기 차례를 고르는 것과 똑같다.

## 기대 높이

**정리.** 낱개 $n$개인 트립에서 어느 마디든 기댓값 깊이는 $O(\log n)$이다.

차례 $i$인 마디($i$번째로 작은 열쇠)의 기댓값 깊이는 다음과 같다.

$$
E[\text{depth}(i)] = H_i + H_{n - i + 1} - 1
$$

여기서 $H_k = \sum_{j=1}^{k} 1/j$은 $k$번째 조화수이다. $H_k = O(\log k)$이므로 기댓값 깊이는 $O(\log n)$이다.

## 돌리기

트립 셈은 이진 찾기 나무의 넣기와 지우기 뒤 무지개탑 성질을 되살리려 **돌리기**를 쓴다.

- 마디 $x$에서 **오른쪽 돌리기**: $x$의 왼쪽 자식을 $x$의 자리로 올린다.
- 마디 $x$에서 **왼쪽 돌리기**: $x$의 오른쪽 자식을 $x$의 자리로 올린다.

돌리기는 이진 찾기 나무 성질을 지키면서 우선값이 더 높은 마디를 위로 올릴 수 있게 한다.

## 연산

### 찾기

찾기는 여느 이진 찾기 나무와 똑같이 된다. 곧 우선값을 무시하고 열쇠 차례를 따른다. 기댓값 시간: $O(\log n)$.

### 삽입

1. 여느 이진 찾기 나무 넣기로 새 마디를 잎으로 넣는다.
2. 아무 우선값을 매긴다.
3. 마디의 우선값이 어버이보다 작은 동안 위로 돌린다(왼쪽 자식이면 오른쪽 돌리기, 오른쪽 자식이면 왼쪽 돌리기).

### 지우기

1. 지울 마디를 찾는다.
2. 잎이 될 때까지 (우선값이 더 작은 자식 쪽으로) 아래로 돌린다.
3. 그 잎을 없앤다.

아니면 마디의 우선값을 $\infty$으로 두고 돌리기로 잎 자리까지 가라앉게 한다.

## 구현

```python
"""
마구잡이 트립: 아무 우선값을 갖춘 이진 찾기 나무 + 무지개탑.

Supports search, insert, and delete in O(log n) expected time
돌리기로 우선값의 무지개탑 성질을 지킨다.
"""

import random


# === 트립 마디 ===

class TreapNode:
    """트립의 마디."""

    def __init__(self, key, priority=None):
        self.key = key
        self.priority = priority if priority is not None else random.random()
        self.left = None
        self.right = None


# === 회전 ===

def rotate_right(node):
    """오른쪽 돌리기: 마디의 왼쪽 자식을 올린다."""
    new_root = node.left
    node.left = new_root.right
    new_root.right = node
    return new_root


def rotate_left(node):
    """왼쪽 돌리기: 마디의 오른쪽 자식을 올린다."""
    new_root = node.right
    node.right = new_root.left
    new_root.left = node
    return new_root


# === 삽입 ===

def insert(root, key):
    """트립에 열쇠를 넣는다.

    부분 트리의 새 뿌리를 돌려준다.
    """
    if root is None:
        return TreapNode(key)

    if key < root.key:
        root.left = insert(root.left, key)
        if root.left.priority < root.priority:
            root = rotate_right(root)
    elif key > root.key:
        root.right = insert(root.right, key)
        if root.right.priority < root.priority:
            root = rotate_left(root)

    return root


# === 지우기 ===

def delete(root, key):
    """트립에서 열쇠를 지운다.

    부분 트리의 새 뿌리를 돌려준다.
    """
    if root is None:
        return None

    if key < root.key:
        root.left = delete(root.left, key)
    elif key > root.key:
        root.right = delete(root.right, key)
    else:
        # 지울 노드를 찾음
        if root.left is None:
            return root.right
        elif root.right is None:
            return root.left
        else:
            # 우선값이 더 작은 자식 쪽으로 돌린다
            if root.left.priority < root.right.priority:
                root = rotate_right(root)
                root.right = delete(root.right, key)
            else:
                root = rotate_left(root)
                root.left = delete(root.left, key)

    return root


# === 찾기 ===

def search(root, key):
    """트립에서 열쇠를 찾는다."""
    if root is None:
        return False
    if key == root.key:
        return True
    elif key < root.key:
        return search(root.left, key)
    else:
        return search(root.right, key)


# === 중위 순회 ===

def inorder(root):
    """열쇠를 정렬한 차례로 돌려준다."""
    if root is None:
        return []
    return inorder(root.left) + [root.key] + inorder(root.right)


# === 나무 높이 ===

def height(root):
    """트립의 높이를 셈한다."""
    if root is None:
        return -1
    return 1 + max(height(root.left), height(root.right))


# === 메인 ===

if __name__ == "__main__":
    random.seed(42)
    root = None

    keys = [5, 2, 8, 1, 4, 7, 9, 3, 6]
    for k in keys:
        root = insert(root, k)

    print(f"Inserted: {keys}")
    print(f"Inorder:  {inorder(root)}")
    print(f"Height:   {height(root)}")
    print(f"Root:     key={root.key}, priority={root.priority:.4f}")

    # 탐색
    for k in [4, 10]:
        print(f"Search {k}: {search(root, k)}")

    # 지우기
    root = delete(root, 5)
    print(f"\nAfter deleting 5:")
    print(f"Inorder:  {inorder(root)}")
    print(f"Height:   {height(root)}")

    # 여러 트립에 걸친 평균 높이
    total_height = 0
    n = 1000
    trials = 100
    for _ in range(trials):
        r = None
        for k in range(n):
            r = insert(r, k)
        total_height += height(r)
    avg_h = total_height / trials
    print(f"\nAverage height of treap with {n} keys: {avg_h:.1f}")
    print(f"Expected O(log n) = {2 * 13.8:.1f} (2 * ln 1000)")
```

## 가르기와 합치기

트립은 효율 좋은 **가르기**와 **합치기** 셈을 받쳐 주므로 차례와 구간 셈을 짜는 데 쓸모 있다.

**Split(T, k):** 트립 $T$을 $T_1$(열쇠 $\le k$)과 $T_2$(열쇠 $> k$)으로 기댓값 $O(\log n)$ 시간에 가른다.

**Merge(T_1, T_2):** $T_1$의 열쇠가 모두 $T_2$의 열쇠보다 작은 두 트립을 기댓값 $O(\log n)$ 시간에 합친다.

## 복잡도 요약

| 셈 | 기댓값 | 가장 나쁜 경우 |
|---|---|---|
| 찾기 | $O(\log n)$ | $O(n)$ |
| 삽입 | $O(\log n)$ | $O(n)$ |
| 삭제 | $O(\log n)$ | $O(n)$ |
| 쪼개기 | $O(\log n)$ | $O(n)$ |
| 합치기 | $O(\log n)$ | $O(n)$ |
| 자리 | $O(n)$ | $O(n)$ |

## 참고 문헌

- Aragon, C. R. & Seidel, R. "Randomized Search Trees." *Algorithmica*, 1996.
- Motwani, R. & Raghavan, P. *Randomized Algorithms*. Cambridge University Press.

## 연습문제

**연습문제 1.**
마구잡이 트립의 핵심 마구잡이 재주와 그것이 정해진 방식보다 나은 까닭을 설명하라.

??? success "연습문제 1 풀이"
    마구잡이 트립은 마구잡이를 써서 정해진 알고리즘이 마주칠 수 있는 가장 나쁜 들임을 피한다. 아무렇게나 고르므로 알고리즘의 솜씨가 들임의 짜임이 아니라 제 동전 던지기에 달린다. 그래서 모든 들임에 대해 참인 센 기댓값 시간이나 높은 확률의 보장을 흔히 얻으며, 짓궂거나 병리적인 경우를 걱정할 까닭이 없어진다. $\square$

---

**연습문제 2.**
마구잡이 트립의 기댓값 시간 복잡도는 얼마인가? 가장 나쁜 경우의 복잡도는 얼마인가?

??? success "연습문제 2 풀이"
    기댓값 시간 복잡도는 흔히 $O(n)$이나 $O(n \log n)$이며 높은 확률로 이룬다. 가장 나쁜 경우는 다항식만큼 더 나쁠 수 있지만(예컨대 $O(n^2)$) 그럴 확률은 무시할 만큼 작다. 기댓값과 가장 나쁜 경우의 틈이 마구잡이의 값이며, 가장 나쁜 움직임이 일어날 확률은 들임 크기에 따라 지수로 줄어든다. $\square$

---

**연습문제 3.**
마구잡이 트립은 라스베이거스 알고리즘인가 몬테카를로 알고리즘인가? 그 차이를 설명하라.

??? success "연습문제 3 풀이"
    **라스베이거스**: 늘 옳은 결과를 내며 도는 시간이 아무 변수이다(기댓값이 다항식). **몬테카를로**: 늘 다항식 시간에 돌지만 결과가 어떤 가둔 확률로 틀릴 수 있다. 마구잡이 트립은 옳음을 보장하느냐 도는 시간을 보장하느냐에 따라 이 가운데 하나에 든다. 이 가름이 어긋날 확률을 어떻게 다룰지 정한다. $\square$

---

**연습문제 4.**
마구잡이 트립에서 마구잡이를 없애거나 솜씨가 나쁠 확률을 줄이는 법을 설명하라.

??? success "연습문제 4 풀이"
    방책은 다음과 같다. (1) **거듭 해 보기**: 알고리즘을 여러 번 돌려 가장 좋거나 많은 쪽 결과를 택하면 어긋날 확률이 지수로 줄어든다. (2) **마구잡이 없애기**: 조건부 기댓값이나 흩는 함수 무리로 아무 고르기를 정해진 고르기로 바꾼다. (3) **키우기**: 몬테카를로 알고리즘에서는 $k$번 되풀이해 어긋남을 $2^{-k}$으로 줄인다. (4) **비슷 마구잡이 만들개**: 알고리즘이 보기에 "마구잡이처럼 보이는" 정해진 차례를 쓴다. $\square$
