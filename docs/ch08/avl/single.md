# 단일 회전

AVL 트리의 노드에서 균형이 깨졌는데 무거운 부분 트리가 불균형과 **같은 쪽**으로 기울어 있으면, 곧 왼쪽 부분 트리의 왼쪽 자식(왼쪽-왼쪽)이거나 오른쪽 부분 트리의 오른쪽 자식(오른쪽-오른쪽)이면, 단일 회전으로 균형이 되살아난다. 단일 회전은 AVL 트리에서 가장 간단한 재균형 도구이다. 포인터를 한 번 고쳐 무거운 자식을 균형이 깨진 부모의 자리로 올리면 부분 트리의 높이가 꼭 1만큼 줄어든다.

---

## 1. 오른쪽 회전 (왼쪽-왼쪽 불균형용)

노드 $z$의 $\text{BF}(z) = +2$이고 왼쪽 자식 $y$의 $\text{BF}(y) \geq 0$이면 무거운 경로가 왼쪽으로 곧게 내려간다. $z$에서 **오른쪽 회전**을 하면 $y$이 새 뿌리가 된다.

```
Before:           After:
    z                y
   / \              / \
  y   C    →       A   z
 / \                  / \
A   B                B   C
```

이 연산은 포인터 세 개를 고친다.

1. $z.\text{left} \leftarrow y.\text{right}$ (부분 트리 $B$이 $z$의 왼쪽 자식이 된다)
2. $y.\text{right} \leftarrow z$ (옛 뿌리 $z$이 $y$의 오른쪽 자식이 된다)
3. $z$의 부모가 대신 $y$을 가리키도록 고친다

### 올바름

이진 탐색 트리의 성질이 지켜지는 까닭은 다음과 같다.

- $A$의 모든 열쇠가 $y.\text{key}$보다 작다 (그대로이다).
- $y.\text{key} < z.\text{key}$이다 (원래 트리의 이진 탐색 트리 성질).
- $B$의 모든 열쇠가 $y.\text{key} < B.\text{key} < z.\text{key}$을 만족한다. $B$을 $y$의 오른쪽에서 $z$의 왼쪽으로 옮겨도 이것이 지켜진다.
- $C$의 모든 열쇠가 $z.\text{key}$보다 크다 (그대로이다).

### 높이 분석

$h(A) = a$, $h(B) = b$, $h(C) = c$이라 하자. 회전 전에는 다음과 같다.

- $h(y) = 1 + \max(a, b)$
- $\text{BF}(z) = +2$이므로 $h(z) = 1 + \max(h(y), c) = 1 + h(y)$

회전 뒤에는 다음과 같다.

$$
h(z_{\text{new}}) = 1 + \max(b, c)
$$

$$
h(y_{\text{new}}) = 1 + \max(a, h(z_{\text{new}})) = 1 + \max(a, 1 + \max(b, c))
$$

$\text{BF}(y) = +1$일 때(삽입의 경우) $a = b + 1$이고 $c = a - 1 = b$이다. 그러면 $h(z_{\text{new}}) = 1 + b$이고 $h(y_{\text{new}}) = 1 + a = 2 + b$이므로 $\text{BF}(y_{\text{new}}) = 0$이다.

---

## 2. 왼쪽 회전 (오른쪽-오른쪽 불균형용)

노드 $z$의 $\text{BF}(z) = -2$이고 오른쪽 자식 $y$의 $\text{BF}(y) \leq 0$이면 $z$에서의 **왼쪽 회전**이 거울상이 된다.

```
Before:           After:
  z                  y
 / \                / \
A   y      →       z   C
   / \            / \
  B   C          A   B
```

포인터를 고치는 일도 오른쪽 회전의 거울상이다.

1. $z.\text{right} \leftarrow y.\text{left}$ (부분 트리 $B$이 $z$의 오른쪽 자식이 된다)
2. $y.\text{left} \leftarrow z$ ($z$이 $y$의 왼쪽 자식이 된다)
3. $z$의 부모가 $y$을 가리키도록 고친다

---

## 3. 구현

```python
"""
AVL의 한 번 회전: 왼쪽과 오른쪽.

왼쪽-왼쪽과 오른쪽-오른쪽 불균형을 O(1) 시간에 바로잡는
근본 되는 포인터 연산을 보인다.
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

def rotate_right(z):
    """z에서 오른쪽으로 회전(왼쪽-왼쪽 불균형을 바로잡는다).

    Before:     z          After:     y
               / \\                  / \\
              y   C                A   z
             / \\                    / \\
            A   B                  B   C
    """
    y = z.left
    b = y.right

    # 회전을 한다
    y.right = z
    z.left = b

    # 높이를 고친다 (y가 z에 매이므로 z를 먼저)
    update_height(z)
    update_height(y)

    return y  # 새 뿌리

def rotate_left(z):
    """z에서 왼쪽으로 회전(오른쪽-오른쪽 불균형을 바로잡는다).

    Before:   z            After:     y
             / \\                    / \\
            A   y                  z   C
               / \\              / \\
              B   C            A   B
    """
    y = z.right
    b = y.left

    # 회전을 한다
    y.left = z
    z.right = b

    # 높이를 고친다 (y가 z에 매이므로 z를 먼저)
    update_height(z)
    update_height(y)

    return y  # 새 뿌리

# === 한 번 회전으로 균형을 되잡는 삽입 ===

def insert(node, key):
    """열쇠를 넣고 LL/RR 경우에 한 번 회전을 적용한다."""
    if node is None:
        return AVLNode(key)

    if key < node.key:
        node.left = insert(node.left, key)
    elif key > node.key:
        node.right = insert(node.right, key)
    else:
        return node

    update_height(node)
    bf = balance_factor(node)

    # 왼쪽-왼쪽 경우: 오른쪽으로 한 번 회전
    if bf > 1 and balance_factor(node.left) >= 0:
        return rotate_right(node)

    # 오른쪽-오른쪽 경우: 왼쪽으로 한 번 회전
    if bf < -1 and balance_factor(node.right) <= 0:
        return rotate_left(node)

    # 왼쪽-오른쪽과 오른쪽-왼쪽 경우는 두 번 회전으로 다룬다
    # (두 번 회전 쪽에서 다룬다)
    if bf > 1 and balance_factor(node.left) < 0:
        node.left = rotate_left(node.left)
        return rotate_right(node)
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

if __name__ == "__main__":
    # 왼쪽-왼쪽 경우: 30, 20, 10을 넣는다
    print("=== Left-Left Case (Right Rotation) ===")
    print("Inserting 30, 20, 10:")
    root = None
    for key in [30, 20, 10]:
        root = insert(root, key)
    print_tree(root)
    print()

    # 오른쪽-오른쪽 경우: 10, 20, 30을 넣는다
    print("=== Right-Right Case (Left Rotation) ===")
    print("Inserting 10, 20, 30:")
    root = None
    for key in [10, 20, 30]:
        root = insert(root, key)
    print_tree(root)
    print()

    # 한 번 회전이 여러 번 일어나는 더 긴 차례
    print("=== Sorted Insertion (multiple rotations) ===")
    print("Inserting 1, 2, 3, 4, 5, 6, 7:")
    root = None
    for key in range(1, 8):
        root = insert(root, key)
    print_tree(root)
```

**출력:**
```
=== Left-Left Case (Right Rotation) ===
Inserting 30, 20, 10:
    30 [BF=+0]
20 [BF=+0]
    10 [BF=+0]

=== Right-Right Case (Left Rotation) ===
Inserting 10, 20, 30:
    30 [BF=+0]
20 [BF=+0]
    10 [BF=+0]

=== Sorted Insertion (multiple rotations) ===
Inserting 1, 2, 3, 4, 5, 6, 7:
        7 [BF=+0]
    6 [BF=+0]
        5 [BF=+0]
4 [BF=+0]
        3 [BF=+0]
    2 [BF=+0]
        1 [BF=+0]
```

열쇠를 정렬된 순서로 넣으면 보통의 이진 탐색 트리에서는 치우친 사슬이 된다. AVL 트리는 걸음마다 왼쪽 회전을 하여 높이가 2인 완벽하게 균형 잡힌 트리를 만든다.

---

## 4. 복잡도

단일 회전은 일정한 횟수의 포인터 고치기와 높이 갱신 두 번을 한다.

| 연산 | 비용 |
|:--|:-:|
| 포인터 고치기 | $O(1)$ |
| 높이 갱신 | $O(1)$ |
| **회전당 합계** | $O(1)$ |

회전 자체는 $O(1)$이다. 전체 삽입 비용이 $O(\log n)$인 것은 회전 때문이 아니라 이진 탐색 트리를 내려가는 걸음과 손질 경로 때문이다.

---

## 연습문제

**연습문제 1.**
단일 회전의 균형 불변식을 밝히고 그것이 높이 $O(\log n)$을 보장함을 증명하라.

??? success "연습문제 1 풀이"
    각 구조의 불변식(균형 인수, 색의 성질, 차수 제약)이 경로 길이의 치우침을 묶는다. 높이의 한계는 그 불변식에서 따라 나온다. 트리의 층마다 (불변식이 정하는) 최소한의 노드가 있어야 하므로 전체 노드 수 $n$이 높이에 따라 지수적으로 늘고, 따라서 $h = O(\log n)$이다.

---

**연습문제 2.**
구조를 다시 짜야 하는(회전, 색 바꾸기, 쪼개기·합치기) 트리에서 단일 회전을 따라가라. 앞뒤의 상태를 보여라.

??? success "연습문제 2 풀이"
    이 쪽에서 설명한 재구성 상황을 일으키는 트리를 하나 만들어라. 어긋난 곳을 보이고, 어느 경우에 해당하는지 가리고, 고친 뒤, 불변식이 되살아났는지 확인하라.

---

**연습문제 3.**
단일 회전이(가) 구조를 다시 짜는 연산을 많아야 $O(\log n)$번 필요로 함을 증명하라.

??? success "연습문제 3 풀이"
    구조를 다시 짤 때마다 어긋난 곳이 뿌리에 한 층 가까워지거나 해소된다. 트리의 층이 $O(\log n)$개이므로 재구성은 많아야 $O(\log n)$번 필요하다. 레드-블랙 삽입 같은 연산에서는 회전 2번과 색 바꾸기 $O(\log n)$번이면 충분하다. $\square$

---

**연습문제 4.**
최악의 높이, 연산마다의 회전 횟수, 구현의 까다로움 면에서 단일 회전을 다른 균형 트리 구조와 견주어라.

??? success "연습문제 4 풀이"
    AVL은 높이가 $1.44\log n$ 이하이고 삭제마다 회전이 $O(\log n)$번까지 든다. 레드-블랙은 높이가 $2\log n$ 이하이고 연산마다 회전이 많아야 3번이다. B-트리는 높이가 $O(\log_B n)$이며 디스크 입출력에 맞추어져 있다. 스플레이 트리는 분할 상환으로 $O(\log n)$이지만 최악은 $O(n)$이다.

## 정리하며

이 마당은 오른쪽 회전 (왼쪽-왼쪽 불균형용)、왼쪽 회전 (오른쪽-오른쪽 불균형용)、구현、복잡도을 차례로 짚었다.

**참고 문헌**

- [Introduction to Algorithms (CLRS), 13~14장](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
