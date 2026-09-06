# 재균형을 하는 삭제

삽입은 부분 트리의 높이를 많아야 1만큼 늘리므로, AVL 삽입은 균형을 되살리는 데 회전이 많아야 한 번(단일이든 이중이든) 필요하다. 그런데 삭제는 부분 트리의 높이를 줄일 수 있고, 그 결과로 하는 회전이 다시 그 부분 트리의 높이를 줄여 불균형이 위로 퍼질 수 있다. 그래서 삭제 한 번에 최악의 경우 회전이 $O(\log n)$번 필요하며, 두 연산 가운데 더 까다롭다.

## 표준 이진 탐색 트리 삭제

균형을 다루기에 앞서, 이진 탐색 트리에서 노드 $z$을 지우는 데 세 경우가 있음을 떠올리자.

1. **잎 노드** ($z$에 자식이 없음): $z$을 곧바로 없앤다.
2. **자식 하나**: $z$을 그 하나뿐인 자식으로 바꾼다.
3. **자식 둘**: $z$의 중위 후속자 $y$($z$의 오른쪽 부분 트리에서 가장 작은 노드)을 찾아 $y$의 열쇠를 $z$에 옮겨 적고, 오른쪽 부분 트리에서 $y$을 지운다. $y$에는 왼쪽 자식이 없으므로 $y$을 지우는 일은 경우 1이나 2가 된다.

구조적으로 지운 뒤에는 지운 자리에서 뿌리까지의 경로 위 조상들의 높이가 달라질 수 있다.

## 삭제 뒤의 재균형

실제로 없앤 노드의 부모에서 시작하여 뿌리까지 올라간다. 조상 $x$마다 다음을 한다.

1. $x$의 높이를 갱신한다.
2. 균형 인수 $\text{BF}(x) = h(\text{left}(x)) - h(\text{right}(x))$을 계산한다.
3. $|\text{BF}(x)| \leq 1$이면 다음 조상으로 넘어간다.
4. $|\text{BF}(x)| = 2$이면 $x$에서 알맞은 회전을 한다.

어떤 회전을 할지는 삽입과 같은 표를 따른다.

| $\text{BF}(x)$ | $\text{BF}(\text{무거운 자식})$ | 회전 |
|:-:|:-:|:--|
| $+2$ | $\geq 0$ | $x$에서 오른쪽 회전 |
| $+2$ | $-1$ | 왼쪽-오른쪽 이중 회전 |
| $-2$ | $\leq 0$ | $x$에서 왼쪽 회전 |
| $-2$ | $+1$ | 오른쪽-왼쪽 이중 회전 |

!!! warning "삭제는 연쇄될 수 있다"
    회전 한 번으로 모든 균형 인수가 되살아나는 삽입과 달리, 삭제에서의 회전은 고친 부분 트리의 높이를 줄일 수 있다. 그 높이 감소가 할아버지 노드의 균형을 깨뜨려 또 회전이 필요해질 수 있다. 최악의 경우 회전이 뿌리까지 퍼져 삭제 한 번에 $O(\log n)$번의 회전이 든다.

## 회전이 여러 번 일어날 수 있는 까닭

$\text{BF}(x) = +2$이고 $\text{BF}(\text{left}(x)) = 0$인 노드 $x$에서 오른쪽 회전을 한다고 하자. 회전 전에 $x$을 뿌리로 하는 부분 트리의 높이는 $h$이다. 회전 뒤 새 뿌리의 $\text{BF} = -1$이 되고 부분 트리의 높이는 $h - 1$으로 줄어든다. 바로 이 높이 감소가 $x$의 부모의 균형을 깨뜨릴 수 있는 상황이다.

반면 삽입 중에 $\text{BF}(\text{left}(x)) = +1$이면 회전이 $\text{BF} = 0$인 새 뿌리를 만들고 부분 트리의 높이가 삽입 전 값으로 돌아가므로 더는 퍼지지 않는다.

## 삭제 알고리즘

```python
"""
균형을 되잡는 AVL 트리 삭제.

이진 탐색 트리 삭제의 세 경우와 그에 이어
회전이 O(log n)번 필요할 수 있는 아래에서 위로의 균형 되잡기 걸음을 보인다.
"""


# === AVL 노드 ===

class AVLNode:
    """열쇠와 왼쪽·오른쪽 자식과 담아 둔 높이를 지닌 노드."""

    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
        self.height = 0


# === 높이와 균형 도구 ===

def height(node):
    """노드의 높이를 돌려준다. 널이면 -1이다."""
    return node.height if node else -1


def update_height(node):
    """자식에서 높이를 다시 셈한다."""
    node.height = 1 + max(height(node.left), height(node.right))


def balance_factor(node):
    """균형 인수 = h(왼쪽) - h(오른쪽)을 셈한다."""
    return height(node.left) - height(node.right)


# === 회전 ===

def rotate_right(y):
    """y에서 오른쪽으로 회전하고 새 뿌리를 돌려준다."""
    x = y.left
    t = x.right
    x.right = y
    y.left = t
    update_height(y)
    update_height(x)
    return x


def rotate_left(x):
    """x에서 왼쪽으로 회전하고 새 뿌리를 돌려준다."""
    y = x.right
    t = y.left
    y.left = x
    x.right = t
    update_height(x)
    update_height(y)
    return y


# === 균형 되잡기 ===

def rebalance(node):
    """|균형 인수| >= 2이면 회전하고 새 부분 트리의 뿌리를 돌려준다."""
    bf = balance_factor(node)
    if bf > 1:
        if balance_factor(node.left) < 0:
            node.left = rotate_left(node.left)
        return rotate_right(node)
    if bf < -1:
        if balance_factor(node.right) > 0:
            node.right = rotate_right(node.right)
        return rotate_left(node)
    return node


# === 삽입 (트리를 세우려고) ===

def insert(node, key):
    """열쇠를 넣고 균형을 되잡는다."""
    if node is None:
        return AVLNode(key)
    if key < node.key:
        node.left = insert(node.left, key)
    elif key > node.key:
        node.right = insert(node.right, key)
    else:
        return node
    update_height(node)
    return rebalance(node)


# === 삭제 ===

def find_min(node):
    """부분 트리에서 열쇠가 가장 작은 노드를 찾는다."""
    while node.left is not None:
        node = node.left
    return node


def delete(node, key):
    """AVL 트리에서 열쇠를 지우고 모든 조상의 균형을 되잡는다."""
    if node is None:
        return None

    if key < node.key:
        node.left = delete(node.left, key)
    elif key > node.key:
        node.right = delete(node.right, key)
    else:
        # 지울 노드를 찾음
        if node.left is None:
            return node.right
        elif node.right is None:
            return node.left
        else:
            # 자식이 둘: 중위 후속자로 바꾼다
            successor = find_min(node.right)
            node.key = successor.key
            node.right = delete(node.right, successor.key)

    update_height(node)
    return rebalance(node)


# === 보이기 ===

def print_tree(node, level=0):
    """균형 인수와 함께 트리를 옆으로 찍는다."""
    if node is None:
        return
    print_tree(node.right, level + 1)
    bf = balance_factor(node)
    print(f"{'    ' * level}{node.key} [BF={bf:+d}]")
    print_tree(node.left, level + 1)


if __name__ == "__main__":
    # AVL 트리를 세운다
    root = None
    for key in [50, 30, 70, 20, 40, 60, 80, 10, 25]:
        root = insert(root, key)

    print("Before deletion:")
    print_tree(root)
    print()

    # 80(잎)을 지우고 70(자식 하나), 60을 차례로 지운다
    for key in [80, 70, 60]:
        root = delete(root, key)
        print(f"After deleting {key}:")
        print_tree(root)
        print()
```

**출력:**
```
Before deletion:
        80 [BF=+0]
    70 [BF=+0]
        60 [BF=+0]
50 [BF=+0]
        40 [BF=+0]
    30 [BF=+0]
            25 [BF=+0]
        20 [BF=-1]
            10 [BF=+0]

After deleting 80:
    70 [BF=+1]
        60 [BF=+0]
50 [BF=+0]
        40 [BF=+0]
    30 [BF=+0]
            25 [BF=+0]
        20 [BF=-1]
            10 [BF=+0]

After deleting 70:
    60 [BF=+0]
50 [BF=+1]
        40 [BF=+0]
    30 [BF=+0]
            25 [BF=+0]
        20 [BF=-1]
            10 [BF=+0]

After deleting 60:
    50 [BF=+0]
        40 [BF=+0]
30 [BF=+0]
        25 [BF=+0]
    20 [BF=+0]
        10 [BF=+0]
```

60을 지우고 나면 노드 50이 오른쪽 자식이 되고 노드 30이 새 뿌리가 되며 트리가 다시 균형을 잡는다.

## 복잡도

| 연산 | 시간 | 회전 |
|:--|:-:|:-:|
| 이진 탐색 트리 삭제 단계 | $O(\log n)$ | 0 |
| 재균형을 위한 거슬러 오르기 | $O(\log n)$ | 최악의 경우 $O(\log n)$ |
| **합계** | $O(\log n)$ | $O(\log n)$ |

회전 하나하나는 $O(1)$ 시간이 걸리지만 삭제 한 번에 회전이 $O(\log n)$번까지 일어날 수 있다. 그래도 회전마다 트리의 서로 다른 층에서 일어나므로 전체 일의 양은 $O(\log n)$에 머문다.

## 삽입과 견주기

| 성질 | 삽입 | 삭제 |
|:--|:-:|:-:|
| 최대 회전 횟수 | 1 (단일 또는 이중) | $O(\log n)$ |
| 회전 뒤의 높이 변화 | 삽입 전으로 되돌아감 | 1만큼 줄 수 있음 |
| 회전 뒤의 전파 | 멈춘다 | 위로 이어질 수 있다 |

이 비대칭은 삽입이 높이를 더하고 삭제가 높이를 빼기 때문에 생긴다. 삽입 뒤의 회전은 원래 높이를 되살려 전파를 멈춘다. 삭제 뒤의 회전은 부분 트리의 높이를 삭제 전보다 낮출 수 있어 부모의 균형을 깨뜨릴 수 있다.

## 참고 문헌

- [Introduction to Algorithms (CLRS), 13~14장](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)


## 연습문제

**연습문제 1.**
재균형을 하는 삭제의 균형 불변식을 밝히고 그것이 높이 $O(\log n)$을 보장함을 증명하라.

??? success "연습문제 1 풀이"
    각 구조의 불변식(균형 인수, 색의 성질, 차수 제약)이 경로 길이의 치우침을 묶는다. 높이의 한계는 그 불변식에서 따라 나온다. 트리의 층마다 (불변식이 정하는) 최소한의 노드가 있어야 하므로 전체 노드 수 $n$이 높이에 따라 지수적으로 늘고, 따라서 $h = O(\log n)$이다.

---

**연습문제 2.**
구조를 다시 짜야 하는(회전, 색 바꾸기, 쪼개기·합치기) 트리에서 재균형을 하는 삭제를 따라가라. 앞뒤의 상태를 보여라.

??? success "연습문제 2 풀이"
    이 쪽에서 설명한 재구성 상황을 일으키는 트리를 하나 만들어라. 어긋난 곳을 보이고, 어느 경우에 해당하는지 가리고, 고친 뒤, 불변식이 되살아났는지 확인하라.

---

**연습문제 3.**
재균형을 하는 삭제이(가) 구조를 다시 짜는 연산을 많아야 $O(\log n)$번 필요로 함을 증명하라.

??? success "연습문제 3 풀이"
    구조를 다시 짤 때마다 어긋난 곳이 뿌리에 한 층 가까워지거나 해소된다. 트리의 층이 $O(\log n)$개이므로 재구성은 많아야 $O(\log n)$번 필요하다. 레드-블랙 삽입 같은 연산에서는 회전 2번과 색 바꾸기 $O(\log n)$번이면 충분하다. $\square$

---

**연습문제 4.**
최악의 높이, 연산마다의 회전 횟수, 구현의 까다로움 면에서 재균형을 하는 삭제를 다른 균형 트리 구조와 견주어라.

??? success "연습문제 4 풀이"
    AVL은 높이가 $1.44\log n$ 이하이고 삭제마다 회전이 $O(\log n)$번까지 든다. 레드-블랙은 높이가 $2\log n$ 이하이고 연산마다 회전이 많아야 3번이다. B-트리는 높이가 $O(\log_B n)$이며 디스크 입출력에 맞추어져 있다. 스플레이 트리는 분할 상환으로 $O(\log n)$이지만 최악은 $O(n)$이다.