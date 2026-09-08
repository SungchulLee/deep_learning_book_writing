# 이중 회전

무거운 부분 트리가 부모와 같은 쪽으로 기울어 있을 때, 곧 왼쪽-왼쪽이거나 오른쪽-오른쪽일 때는 단일 회전으로 불균형이 바로잡힌다. 그런데 불균형이 **지그재그** 모양(왼쪽-오른쪽이나 오른쪽-왼쪽)이면 단일 회전으로는 균형이 되살아나지 않는다. 그럴 때 AVL 트리는 **이중 회전**을 쓴다. 단일 회전 둘을 하나로 엮어, 지그재그를 곧게 편 뒤 높이를 바로잡는 연산이다.

---

## 1. 단일 회전이 통하지 않을 때

$\text{BF}(z) = +2$인 노드 $z$을 생각해 보자. 왼쪽 부분 트리가 너무 높다는 뜻이다. 무거운 경로가 왼쪽으로 갔다가 **오른쪽**으로 가면($z$의 왼쪽 자식 $y$을 거쳐 $y$의 오른쪽 자식 $x$으로), $z$에서 오른쪽 회전을 한 번 해도 문제가 풀리지 않는다. $y$의 높은 오른쪽 부분 트리를 반대쪽으로 옮길 뿐 전체 높이는 줄지 않는다.

핵심은 불균형을 일으킨 노드 $x$의 값이 $y$과 $z$ "사이"에 있으므로 그것이 이 부분 트리의 새 뿌리가 되어야 한다는 것이다. 이중 회전은 먼저 $x$을 $y$의 자리로 올리고 이어 $z$의 자리로 올려 이를 이룬다.

---

## 2. 왼쪽-오른쪽 이중 회전

$y$이 $z$의 왼쪽 자식일 때 $\text{BF}(z) = +2$이고 $\text{BF}(y) = -1$인 경우를 다룬다.

**1단계: $y$에서 왼쪽 회전** ($y$은 $z$의 왼쪽 자식)

```
      z                z
     / \              / \
    y   D    →       x   D
   / \              / \
  A   x            y   C
     / \          / \
    B   C        A   B
```

**2단계: $z$에서 오른쪽 회전**

```
      z                x
     / \              / \
    x   D    →       y   z
   / \              / \ / \
  y   C            A  B C  D
 / \
A   B
```

이중 회전 뒤에는 $x$이 뿌리에 앉고 $y$이 왼쪽 자식, $z$이 오른쪽 자식이 된다. 네 부분 트리 $A, B, C, D$이 알맞게 나뉘어 이진 탐색 트리의 순서가 지켜지고 모든 균형 인수가 $\{-1, 0, +1\}$으로 돌아온다.

### 엄밀한 높이 분석

부분 트리 $A, B, C, D$의 높이를 각각 $a, b, c, d$이라 하자. 이중 회전 전에는 다음과 같다.

- $\text{BF}(y) = -1$이므로 $h(y) = 1 + \max(a, 1 + \max(b, c)) = 1 + (1 + \max(b, c))$
- $\text{BF}(z) = +2$일 때 $h(z) = 1 + \max(h(y), d)$

이중 회전 뒤 노드 $x$은 다음과 같다.

$$
h(\text{left of } x) = 1 + \max(a, b)
$$

$$
h(\text{right of } x) = 1 + \max(c, d)
$$

균형 조건에서 $\max(b, c) = a = d$이므로 $x$의 두 자식의 높이가 같아 $\text{BF}(x) = 0$이거나 $|\text{BF}(x)| \leq 1$이다.

---

## 3. 오른쪽-왼쪽 이중 회전

$y$이 $z$의 오른쪽 자식일 때 $\text{BF}(z) = -2$이고 $\text{BF}(y) = +1$인 대칭적인 경우이다.

**1단계: $y$에서 오른쪽 회전** ($y$은 $z$의 오른쪽 자식)

```
    z                z
   / \              / \
  A   y    →       A   x
     / \              / \
    x   D            B   y
   / \                  / \
  B   C                C   D
```

**2단계: $z$에서 왼쪽 회전**

```
    z                  x
   / \                / \
  A   x      →      z   y
     / \            / \ / \
    B   y          A  B C  D
       / \
      C   D
```

움직임은 왼쪽-오른쪽 경우를 좌우로 완전히 뒤집은 것이다.

---

## 4. 구현

```python
"""
AVL의 두 번 회전: 왼쪽-오른쪽과 오른쪽-왼쪽.

한 번 회전으로는 풀 수 없는 지그재그 불균형을 한 번 회전 둘을
이어 붙여 어떻게 바로잡는지 보인다.
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

# === 한 번 회전 ===

def rotate_left(x):
    """x에서 왼쪽으로 회전."""
    y = x.right
    x.right = y.left
    y.left = x
    update_height(x)
    update_height(y)
    return y

def rotate_right(y):
    """y에서 오른쪽으로 회전."""
    x = y.left
    y.left = x.right
    x.right = y
    update_height(y)
    update_height(x)
    return x

# === 두 번 회전 ===

def left_right_rotate(z):
    """z에서의 왼쪽-오른쪽 두 번 회전.

    먼저 z의 왼쪽 자식을 왼쪽으로 회전한 뒤 z를 오른쪽으로 회전한다.
    BF(z) = +2이고 BF(z.left) = -1일 때 쓴다.
    """
    z.left = rotate_left(z.left)
    return rotate_right(z)

def right_left_rotate(z):
    """z에서의 오른쪽-왼쪽 두 번 회전.

    먼저 z의 오른쪽 자식을 오른쪽으로 회전한 뒤 z를 왼쪽으로 회전한다.
    BF(z) = -2이고 BF(z.right) = +1일 때 쓴다.
    """
    z.right = rotate_right(z.right)
    return rotate_left(z)

# === 균형 되잡기 (하나로 모음) ===

def rebalance(node):
    """필요에 따라 한 번 또는 두 번 회전한다."""
    bf = balance_factor(node)
    if bf > 1:
        if balance_factor(node.left) < 0:
            return left_right_rotate(node)  # 두 번
        return rotate_right(node)           # 한 번
    if bf < -1:
        if balance_factor(node.right) > 0:
            return right_left_rotate(node)  # 두 번
        return rotate_left(node)            # 한 번
    return node

# === 삽입 ===

def insert(node, key):
    """AVL 트리에 열쇠를 넣는다."""
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
    # 왼쪽-오른쪽 두 번 회전을 보인다
    # 30, 10, 20을 넣으면 30에서 LR 회전이 일어난다
    print("=== Left-Right Double Rotation ===")
    root = None
    for key in [30, 10, 20]:
        root = insert(root, key)
        print(f"After inserting {key}:")
        print_tree(root)
        print()

    # 오른쪽-왼쪽 두 번 회전을 보인다
    # 10, 30, 20을 넣으면 10에서 RL 회전이 일어난다
    print("=== Right-Left Double Rotation ===")
    root = None
    for key in [10, 30, 20]:
        root = insert(root, key)
        print(f"After inserting {key}:")
        print_tree(root)
        print()
```

**출력:**
```
=== Left-Right Double Rotation ===
After inserting 30:
30 [BF=+0]

After inserting 10:
30 [BF=+1]
    10 [BF=+0]

After inserting 20:
    30 [BF=+0]
20 [BF=+0]
    10 [BF=+0]

=== Right-Left Double Rotation ===
After inserting 10:
10 [BF=+0]

After inserting 30:
    30 [BF=+0]
10 [BF=-1]

After inserting 20:
    30 [BF=+0]
20 [BF=+0]
    10 [BF=+0]
```

두 경우 모두 지그재그 모양(30-10-20이나 10-30-20)에서 노드 20이 균형 잡힌 뿌리가 되는데, 이는 단일 회전으로는 이룰 수 없다.

---

## 5. 회전의 경우 정리

| 불균형의 모양 | $\text{BF}(z)$ | $\text{BF}(\text{자식})$ | 고치는 법 |
|:--|:-:|:-:|:--|
| 왼쪽-왼쪽 (곧음) | $+2$ | $+1$ 또는 $0$ | 단일 오른쪽 회전 |
| 왼쪽-오른쪽 (지그재그) | $+2$ | $-1$ | **왼쪽-오른쪽 이중 회전** |
| 오른쪽-오른쪽 (곧음) | $-2$ | $-1$ 또는 $0$ | 단일 왼쪽 회전 |
| 오른쪽-왼쪽 (지그재그) | $-2$ | $+1$ | **오른쪽-왼쪽 이중 회전** |

이중 회전은 단일 회전보다 포인터를 두 번 더 고치지만 여전히 $O(1)$ 시간에 끝나므로 AVL 삽입과 삭제의 $O(\log n)$ 복잡도가 지켜진다.

---

## 연습문제

**연습문제 1.**
이중 회전의 균형 불변식을 밝히고 그것이 높이 $O(\log n)$을 보장함을 증명하라.

??? success "연습문제 1 풀이"
    각 구조의 불변식(균형 인수, 색의 성질, 차수 제약)이 경로 길이의 치우침을 묶는다. 높이의 한계는 그 불변식에서 따라 나온다. 트리의 층마다 (불변식이 정하는) 최소한의 노드가 있어야 하므로 전체 노드 수 $n$이 높이에 따라 지수적으로 늘고, 따라서 $h = O(\log n)$이다.

---

**연습문제 2.**
구조를 다시 짜야 하는(회전, 색 바꾸기, 쪼개기·합치기) 트리에서 이중 회전을 따라가라. 앞뒤의 상태를 보여라.

??? success "연습문제 2 풀이"
    이 쪽에서 설명한 재구성 상황을 일으키는 트리를 하나 만들어라. 어긋난 곳을 보이고, 어느 경우에 해당하는지 가리고, 고친 뒤, 불변식이 되살아났는지 확인하라.

---

**연습문제 3.**
이중 회전이(가) 구조를 다시 짜는 연산을 많아야 $O(\log n)$번 필요로 함을 증명하라.

??? success "연습문제 3 풀이"
    구조를 다시 짤 때마다 어긋난 곳이 뿌리에 한 층 가까워지거나 해소된다. 트리의 층이 $O(\log n)$개이므로 재구성은 많아야 $O(\log n)$번 필요하다. 레드-블랙 삽입 같은 연산에서는 회전 2번과 색 바꾸기 $O(\log n)$번이면 충분하다. $\square$

---

**연습문제 4.**
최악의 높이, 연산마다의 회전 횟수, 구현의 까다로움 면에서 이중 회전을 다른 균형 트리 구조와 견주어라.

??? success "연습문제 4 풀이"
    AVL은 높이가 $1.44\log n$ 이하이고 삭제마다 회전이 $O(\log n)$번까지 든다. 레드-블랙은 높이가 $2\log n$ 이하이고 연산마다 회전이 많아야 3번이다. B-트리는 높이가 $O(\log_B n)$이며 디스크 입출력에 맞추어져 있다. 스플레이 트리는 분할 상환으로 $O(\log n)$이지만 최악은 $O(n)$이다.

## 정리하며

이 마당은 단일 회전이 통하지 않을 때、왼쪽-오른쪽 이중 회전、오른쪽-왼쪽 이중 회전、구현을 차례로 짚었다.

**참고 문헌**

- [Introduction to Algorithms (CLRS), 13~14장](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
