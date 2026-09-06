# 재균형을 하는 삽입

보통의 이진 탐색 트리에 새 열쇠를 넣는 일은 간단하다. 뿌리에서 내려가며 열쇠가 작으면 왼쪽으로, 크면 오른쪽으로 가서 새 노드를 잎으로 붙인다. 문제는 그러면 부분 트리의 높이가 늘어 AVL 균형 조건이 깨질 수 있다는 것이다. AVL 삽입은 삽입 경로를 거슬러 올라가며 균형 인수를 살피고 많아야 **회전 한 번**(단일이든 이중이든)으로 불변식을 되살려 이를 푼다.

## 삽입 알고리즘

AVL 삽입은 두 단계로 나아간다.

1. **이진 탐색 트리 삽입**: 표준 이진 탐색 트리에서와 똑같이 열쇠를 잎으로 넣는다.
2. **손질하며 거슬러 오르기**: 새 잎에서 뿌리까지의 경로를 되짚으며 높이를 갱신하고, 균형 인수가 $\pm 2$에 이른 첫 노드에서 다시 균형을 잡는다.

손질하며 거슬러 오르는 일이 알고리즘의 핵심이다. 경로 위의 노드 $x$마다 다음을 한다.

1. $h(x) = 1 + \max(h(\text{left}(x)),\, h(\text{right}(x)))$으로 갱신한다.
2. $\text{BF}(x) = h(\text{left}(x)) - h(\text{right}(x))$을 계산한다.
3. $|\text{BF}(x)| \leq 1$이면 부모로 옮긴다.
4. $|\text{BF}(x)| = 2$이면 어느 회전에 해당하는지 가려 적용한다.

## 회전의 경우

어떤 회전을 할지는 $x$과 그 무거운 자식의 균형 인수에 달려 있다.

| $\text{BF}(x)$ | 무거운 자식의 방향 | $\text{BF}(\text{자식})$ | 경우 | 고치는 법 |
|:-:|:-:|:-:|:-:|:--|
| $+2$ | 왼쪽 자식 $y$ | $+1$ | 왼쪽-왼쪽 | $x$에서 오른쪽 회전 |
| $+2$ | 왼쪽 자식 $y$ | $-1$ | 왼쪽-오른쪽 | $y$에서 왼쪽 회전 뒤 $x$에서 오른쪽 회전 |
| $-2$ | 오른쪽 자식 $y$ | $-1$ | 오른쪽-오른쪽 | $x$에서 왼쪽 회전 |
| $-2$ | 오른쪽 자식 $y$ | $+1$ | 오른쪽-왼쪽 | $y$에서 오른쪽 회전 뒤 $x$에서 왼쪽 회전 |

!!! info "삽입마다 회전은 많아야 한 번"
    균형이 깨진 가장 낮은 조상에서 회전을 하고 나면 그 부분 트리의 높이가 **삽입 전** 값으로 돌아간다. 곧 회전한 자리보다 위의 어떤 조상도 균형이 깨질 수 없으므로 손질이 바로 끝난다. 회전이 $O(\log n)$번 필요할 수 있는 삭제와의 근본적인 차이이다.

## 회전 한 번으로 충분한 까닭

삽입 전에 $x$을 뿌리로 하는 부분 트리의 높이가 $h$이라 하자. 삽입이 $x$의 한쪽 부분 트리의 높이를 $h-1$에서 $h$으로 늘려 $\text{BF}(x) = +2$(또는 $-2$)이 된다. 회전 뒤 이 부분 트리의 새 뿌리는 높이가 $h$으로 삽입 전과 같다. $x$의 부모가 보기에 $x$ 자리의 높이가 바뀌지 않았으므로 더 균형을 잡을 필요가 없다.

## 한 걸음씩 보는 예

빈 AVL 트리에 열쇠 10, 20, 30을 넣어 보자.

**10 넣기**: 트리에 $\text{BF} = 0$인 노드 하나가 있다.

```
10 [BF=0]
```

**20 넣기**: 노드 20이 10의 오른쪽으로 간다. 이제 노드 10의 $\text{BF} = -1$이다.

```
10 [BF=-1]
  \
   20 [BF=0]
```

**30 넣기**: 노드 30이 20의 오른쪽으로 간다. 이제 $\text{BF}(20) = -1$이고 $\text{BF}(10) = -2$이다. 오른쪽-오른쪽 경우이므로 노드 10에서 왼쪽 회전을 한다.

```
Before rotation:        After rotation:
10 [BF=-2]                 20 [BF=0]
  \                       /  \
   20 [BF=-1]           10    30
     \                [BF=0] [BF=0]
      30 [BF=0]
```

이제 트리가 균형 잡혔다. 옛 뿌리(10)의 자리에서 높이가 (회전 전) 2에서 (회전 뒤) 1로 돌아와 삽입 전 높이와 같아졌다.

## 구현

```python
"""
균형을 되잡는 AVL 트리 삽입.

두 단계 방식을 보인다. 표준 이진 탐색 트리 삽입에 이어
회전이 많아야 한 번인 아래에서 위로의 바로잡기 걸음을 한다.
"""


# === AVL 노드 ===

class AVLNode:
    """열쇠와 자식과 담아 둔 높이를 지닌 AVL 트리 노드."""

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
    """y에서 오른쪽으로 회전."""
    x = y.left
    y.left = x.right
    x.right = y
    update_height(y)
    update_height(x)
    return x


def rotate_left(x):
    """x에서 왼쪽으로 회전."""
    y = x.right
    x.right = y.left
    y.left = x
    update_height(x)
    update_height(y)
    return y


# === 균형을 되잡는 삽입 ===

def insert(node, key):
    """AVL 트리에 열쇠를 넣고 필요하면 균형을 되잡는다.

    부분 트리의 새 뿌리를 돌려준다.
    회전(한 번이든 두 번이든)은 많아야 한 차례 한다.
    """
    # 1단계: 이진 탐색 트리 삽입
    if node is None:
        return AVLNode(key)

    if key < node.key:
        node.left = insert(node.left, key)
    elif key > node.key:
        node.right = insert(node.right, key)
    else:
        return node  # 겹치는 열쇠라 넣지 않는다

    # 2단계: 바로잡기
    update_height(node)
    bf = balance_factor(node)

    # 왼쪽-왼쪽 경우
    if bf > 1 and balance_factor(node.left) >= 0:
        return rotate_right(node)

    # 왼쪽-오른쪽 경우
    if bf > 1 and balance_factor(node.left) < 0:
        node.left = rotate_left(node.left)
        return rotate_right(node)

    # 오른쪽-오른쪽 경우
    if bf < -1 and balance_factor(node.right) <= 0:
        return rotate_left(node)

    # 오른쪽-왼쪽 경우
    if bf < -1 and balance_factor(node.right) > 0:
        node.right = rotate_right(node.right)
        return rotate_left(node)

    return node


# === 보이기 ===

def print_tree(node, level=0):
    """균형 인수와 함께 트리를 옆으로 찍는다."""
    if node is None:
        return
    print_tree(node.right, level + 1)
    bf = balance_factor(node)
    print(f"{'    ' * level}{node.key} [BF={bf:+d}]")
    print_tree(node.left, level + 1)


# === 중위 순회 ===

def inorder(node):
    """정렬된 열쇠 목록을 돌려준다."""
    if node is None:
        return []
    return inorder(node.left) + [node.key] + inorder(node.right)


if __name__ == "__main__":
    # 회전의 네 경우를 모두 보인다
    print("=== Right-Right case: insert 10, 20, 30 ===")
    root = None
    for key in [10, 20, 30]:
        root = insert(root, key)
    print_tree(root)
    print(f"Inorder: {inorder(root)}")
    print()

    print("=== Left-Left case: insert 30, 20, 10 ===")
    root = None
    for key in [30, 20, 10]:
        root = insert(root, key)
    print_tree(root)
    print(f"Inorder: {inorder(root)}")
    print()

    print("=== Left-Right case: insert 30, 10, 20 ===")
    root = None
    for key in [30, 10, 20]:
        root = insert(root, key)
    print_tree(root)
    print(f"Inorder: {inorder(root)}")
    print()

    print("=== Right-Left case: insert 10, 30, 20 ===")
    root = None
    for key in [10, 30, 20]:
        root = insert(root, key)
    print_tree(root)
    print(f"Inorder: {inorder(root)}")
    print()

    # 더 큰 보기
    print("=== Larger example: insert 50,30,70,20,40,60,80,10,25,35 ===")
    root = None
    for key in [50, 30, 70, 20, 40, 60, 80, 10, 25, 35]:
        root = insert(root, key)
    print_tree(root)
    print(f"Inorder: {inorder(root)}")
```

**출력:**
```
=== Right-Right case: insert 10, 20, 30 ===
    30 [BF=+0]
20 [BF=+0]
    10 [BF=+0]
Inorder: [10, 20, 30]

=== Left-Left case: insert 30, 20, 10 ===
    30 [BF=+0]
20 [BF=+0]
    10 [BF=+0]
Inorder: [10, 20, 30]

=== Left-Right case: insert 30, 10, 20 ===
    30 [BF=+0]
20 [BF=+0]
    10 [BF=+0]
Inorder: [10, 20, 30]

=== Right-Left case: insert 10, 30, 20 ===
    30 [BF=+0]
20 [BF=+0]
    10 [BF=+0]
Inorder: [10, 20, 30]

=== Larger example: insert 50,30,70,20,40,60,80,10,25,35 ===
        80 [BF=+0]
    70 [BF=+1]
        60 [BF=+0]
50 [BF=+0]
            40 [BF=+1]
                35 [BF=+0]
        30 [BF=+0]
            25 [BF=+0]
    20 [BF=-1]
        10 [BF=+0]
Inorder: [10, 20, 25, 30, 35, 40, 50, 60, 70, 80]
```

## 복잡도

| 항목 | 비용 |
|:--|:-:|
| 잎까지 내려가는 이진 탐색 트리 걸음 | $O(\log n)$ |
| 손질하며 거슬러 오르기 (높이 갱신) | $O(\log n)$ |
| 회전 | $O(1)$ (단일이든 이중이든 많아야 한 번) |
| **전체 삽입 시간** | $O(\log n)$ |

손질하며 거슬러 오를 때 많아야 조상 $O(\log n)$개를 건드리지만 그 가운데 회전이 필요한 것은 하나뿐이다. 그 회전 뒤 부분 트리의 높이가 삽입 전 값으로 돌아가므로 더 균형을 잡을 필요가 없다.

## 참고 문헌

- [Introduction to Algorithms (CLRS), 13~14장](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)


## 연습문제

**연습문제 1.**
재균형을 하는 삽입의 균형 불변식을 밝히고 그것이 높이 $O(\log n)$을 보장함을 증명하라.

??? success "연습문제 1 풀이"
    각 구조의 불변식(균형 인수, 색의 성질, 차수 제약)이 경로 길이의 치우침을 묶는다. 높이의 한계는 그 불변식에서 따라 나온다. 트리의 층마다 (불변식이 정하는) 최소한의 노드가 있어야 하므로 전체 노드 수 $n$이 높이에 따라 지수적으로 늘고, 따라서 $h = O(\log n)$이다.

---

**연습문제 2.**
구조를 다시 짜야 하는(회전, 색 바꾸기, 쪼개기·합치기) 트리에서 재균형을 하는 삽입을 따라가라. 앞뒤의 상태를 보여라.

??? success "연습문제 2 풀이"
    이 쪽에서 설명한 재구성 상황을 일으키는 트리를 하나 만들어라. 어긋난 곳을 보이고, 어느 경우에 해당하는지 가리고, 고친 뒤, 불변식이 되살아났는지 확인하라.

---

**연습문제 3.**
재균형을 하는 삽입이(가) 구조를 다시 짜는 연산을 많아야 $O(\log n)$번 필요로 함을 증명하라.

??? success "연습문제 3 풀이"
    구조를 다시 짤 때마다 어긋난 곳이 뿌리에 한 층 가까워지거나 해소된다. 트리의 층이 $O(\log n)$개이므로 재구성은 많아야 $O(\log n)$번 필요하다. 레드-블랙 삽입 같은 연산에서는 회전 2번과 색 바꾸기 $O(\log n)$번이면 충분하다. $\square$

---

**연습문제 4.**
최악의 높이, 연산마다의 회전 횟수, 구현의 까다로움 면에서 재균형을 하는 삽입을 다른 균형 트리 구조와 견주어라.

??? success "연습문제 4 풀이"
    AVL은 높이가 $1.44\log n$ 이하이고 삭제마다 회전이 $O(\log n)$번까지 든다. 레드-블랙은 높이가 $2\log n$ 이하이고 연산마다 회전이 많아야 3번이다. B-트리는 높이가 $O(\log_B n)$이며 디스크 입출력에 맞추어져 있다. 스플레이 트리는 분할 상환으로 $O(\log n)$이지만 최악은 $O(n)$이다.