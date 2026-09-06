# 높이의 한계

레드-블랙 트리의 실용적인 값어치는 높이가 $O(\log n)$이라는 보장에 달려 있다. 다섯 가지 레드-블랙 성질은 높이를 대놓고 말하지 않지만 트리의 모양을 충분히 단단히 묶어 로그 높이를 보장한다. 이 절은 앞 절에서 세운 검은 높이 보조정리로 한계 $h \leq 2\log_2(n+1)$을 증명한다.

## 진술

!!! info "정리: 레드-블랙 트리의 높이 한계"
    내부 노드가 $n$개인 레드-블랙 트리의 높이는 많아야 $2\log_2(n+1)$이다.

## 증명

증명은 두 사실을 엮는다.

**사실 1.** 뿌리의 검은 높이는 $\text{bh}(\text{root}) \geq h/2$을 만족한다.

성질 4(붉은 노드가 잇따를 수 없다)에 따라 뿌리에서 잎까지의 어떤 경로에서도 노드의 많아야 절반이 붉다. 그 경로의 변이 $h$개(곧 뿌리 아래 노드가 $h$개)이므로 그 가운데 적어도 $h/2$개가 검다. 따라서 $\text{bh}(\text{root}) \geq h/2$이다.

**사실 2.** 어떤 노드 $x$을 뿌리로 하는 부분 트리에도 내부 노드가 적어도 $2^{\text{bh}(x)} - 1$개 있다(앞 절의 검은 높이 보조정리).

사실 2를 뿌리에 적용하면 다음과 같다.

$$
n \geq 2^{\text{bh}(\text{root})} - 1 \geq 2^{h/2} - 1
$$

$h$에 대해 풀면 다음과 같다.

$$
n + 1 \geq 2^{h/2}
$$

$$
\log_2(n + 1) \geq h/2
$$

$$
h \leq 2\log_2(n + 1)
$$

$\square$

## 해석

한계 $h \leq 2\log_2(n+1)$은 다음을 뜻한다.

- 노드가 $n = 10^6$개인 레드-블랙 트리의 높이는 많아야 $2 \times 20 = 40$이다.
- 완벽하게 균형 잡힌 이진 트리라면 높이가 약 $20$일 것이다.
- AVL 트리라면 높이가 많아야 약 $29$일 것이다.

레드-블랙 트리는 최악의 경우 완벽한 트리보다 대략 두 배 높지만, 이 2라는 인수는 점근 복잡도에 영향을 주지 않는 작은 상수이다.

## 한계가 빈틈없는가

낮은 차수의 항을 빼면 이 한계는 빈틈없다. 뿌리에서 잎까지의 모든 경로가 (뿌리의 검정에서 시작하여) 빨강과 검정을 번갈아 오가는 트리를 생각해 보자. 길이가 $h = 2b$인 그런 경로에는 검은 노드가 $b$개 있고 트리는 $h = 2 \cdot \text{bh}(\text{root})$에 이른다.

다만 $n$이 클 때 무작위 삽입으로 만든 레드-블랙 트리의 실제 높이는 $\log_2 n$에 가까운 편이며 최악값 $2\log_2 n$보다 훨씬 낮다.

## AVL 트리와 견주기

| 트리의 종류 | 높이의 한계 | 완벽한 트리에 대한 비 |
|:--|:--|:-:|
| 완벽 이진 트리 | $\lfloor \log_2 n \rfloor$ | 1.0 |
| AVL 트리 | $1.44 \log_2 n$ | 1.44 |
| 레드-블랙 트리 | $2 \log_2(n+1)$ | 2.0 |

AVL 트리는 높이의 한계가 더 빡빡해서 찾기가 빠르다. 레드-블랙 트리는 재균형이 더 간단하여(고칠 때마다 회전이 적다) 그것을 메운다. 찾기가 주를 이루는 응용에는 AVL 트리가 나을 수 있다. 삽입과 삭제가 잦은 응용에서는 대체로 레드-블랙 트리가 실제로 더 낫다.

## 확인

```python
"""
적흑 트리의 높이 한계 h <= 2 * log2(n + 1)을 확인한다.

여러 크기의 적흑 트리를 세우고 실제 높이가 이론상의 한계를
결코 넘지 않음을 살핀다.
"""

import math
import random


# === 상수 ===

RED = "R"
BLACK = "B"


# === 적흑 트리 구현 ===

class RBNode:
    """적흑 트리의 노드."""

    def __init__(self, key, color=RED):
        self.key = key
        self.color = color
        self.left = None
        self.right = None
        self.parent = None


# 파수
NIL = RBNode(key=None, color=BLACK)


def left_rotate(tree, x):
    """x에서 왼쪽으로 회전."""
    y = x.right
    x.right = y.left
    if y.left is not NIL:
        y.left.parent = x
    y.parent = x.parent
    if x.parent is None:
        tree["root"] = y
    elif x is x.parent.left:
        x.parent.left = y
    else:
        x.parent.right = y
    y.left = x
    x.parent = y


def right_rotate(tree, y):
    """y에서 오른쪽으로 회전."""
    x = y.left
    y.left = x.right
    if x.right is not NIL:
        x.right.parent = y
    x.parent = y.parent
    if y.parent is None:
        tree["root"] = x
    elif y is y.parent.left:
        y.parent.left = x
    else:
        y.parent.right = x
    x.right = y
    y.parent = x


def rb_insert(tree, key):
    """적흑 트리에 열쇠를 넣고 바로잡는다."""
    z = RBNode(key, RED)
    z.left = NIL
    z.right = NIL

    y_node = None
    x_node = tree["root"]

    while x_node is not NIL:
        y_node = x_node
        if z.key < x_node.key:
            x_node = x_node.left
        else:
            x_node = x_node.right

    z.parent = y_node
    if y_node is None:
        tree["root"] = z
    elif z.key < y_node.key:
        y_node.left = z
    else:
        y_node.right = z

    rb_insert_fixup(tree, z)


def rb_insert_fixup(tree, z):
    """삽입 뒤의 적흑 위반을 바로잡는다."""
    while z.parent is not None and z.parent.color == RED:
        if z.parent is z.parent.parent.left:
            uncle = z.parent.parent.right
            if uncle.color == RED:
                z.parent.color = BLACK
                uncle.color = BLACK
                z.parent.parent.color = RED
                z = z.parent.parent
            else:
                if z is z.parent.right:
                    z = z.parent
                    left_rotate(tree, z)
                z.parent.color = BLACK
                z.parent.parent.color = RED
                right_rotate(tree, z.parent.parent)
        else:
            uncle = z.parent.parent.left
            if uncle.color == RED:
                z.parent.color = BLACK
                uncle.color = BLACK
                z.parent.parent.color = RED
                z = z.parent.parent
            else:
                if z is z.parent.left:
                    z = z.parent
                    right_rotate(tree, z)
                z.parent.color = BLACK
                z.parent.parent.color = RED
                left_rotate(tree, z.parent.parent)
    tree["root"].color = BLACK


# === 재기 ===

def tree_height(node):
    """트리의 높이를 셈한다."""
    if node is NIL:
        return -1
    return 1 + max(tree_height(node.left), tree_height(node.right))


def count_nodes(node):
    """내부 노드의 수를 센다."""
    if node is NIL:
        return 0
    return 1 + count_nodes(node.left) + count_nodes(node.right)


if __name__ == "__main__":
    print(f"{'n':>8} | {'h':>4} | {'bound':>8} | {'ok':>4}")
    print("-" * 32)

    for n in [10, 50, 100, 500, 1000, 5000, 10000]:
        tree = {"root": NIL}
        keys = list(range(n))
        random.seed(42)
        random.shuffle(keys)
        for k in keys:
            rb_insert(tree, k)

        h = tree_height(tree["root"])
        bound = 2 * math.log2(n + 1)
        ok = h <= bound
        print(f"{n:8d} | {h:4d} | {bound:8.2f} | {'yes' if ok else 'NO':>4}")
```

**출력:**
```
       n |    h |    bound |   ok
--------------------------------
      10 |    4 |     6.91 |  yes
      50 |    8 |    11.33 |  yes
     100 |   10 |    13.32 |  yes
     500 |   14 |    17.93 |  yes
    1000 |   16 |    19.93 |  yes
    5000 |   20 |    24.58 |  yes
   10000 |   22 |    26.58 |  yes
```

모든 경우에 실제 높이가 이론적 한계보다 훨씬 낮다.

## 참고 문헌

- [Introduction to Algorithms (CLRS), 13장](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)


## 연습문제

**연습문제 1.**
높이의 한계의 균형 불변식을 밝히고 그것이 높이 $O(\log n)$을 보장함을 증명하라.

??? success "연습문제 1 풀이"
    각 구조의 불변식(균형 인수, 색의 성질, 차수 제약)이 경로 길이의 치우침을 묶는다. 높이의 한계는 그 불변식에서 따라 나온다. 트리의 층마다 (불변식이 정하는) 최소한의 노드가 있어야 하므로 전체 노드 수 $n$이 높이에 따라 지수적으로 늘고, 따라서 $h = O(\log n)$이다.

---

**연습문제 2.**
구조를 다시 짜야 하는(회전, 색 바꾸기, 쪼개기·합치기) 트리에서 높이의 한계를 따라가라. 앞뒤의 상태를 보여라.

??? success "연습문제 2 풀이"
    이 쪽에서 설명한 재구성 상황을 일으키는 트리를 하나 만들어라. 어긋난 곳을 보이고, 어느 경우에 해당하는지 가리고, 고친 뒤, 불변식이 되살아났는지 확인하라.

---

**연습문제 3.**
높이의 한계이(가) 구조를 다시 짜는 연산을 많아야 $O(\log n)$번 필요로 함을 증명하라.

??? success "연습문제 3 풀이"
    구조를 다시 짤 때마다 어긋난 곳이 뿌리에 한 층 가까워지거나 해소된다. 트리의 층이 $O(\log n)$개이므로 재구성은 많아야 $O(\log n)$번 필요하다. 레드-블랙 삽입 같은 연산에서는 회전 2번과 색 바꾸기 $O(\log n)$번이면 충분하다. $\square$

---

**연습문제 4.**
최악의 높이, 연산마다의 회전 횟수, 구현의 까다로움 면에서 높이의 한계를 다른 균형 트리 구조와 견주어라.

??? success "연습문제 4 풀이"
    AVL은 높이가 $1.44\log n$ 이하이고 삭제마다 회전이 $O(\log n)$번까지 든다. 레드-블랙은 높이가 $2\log n$ 이하이고 연산마다 회전이 많아야 3번이다. B-트리는 높이가 $O(\log_B n)$이며 디스크 입출력에 맞추어져 있다. 스플레이 트리는 분할 상환으로 $O(\log n)$이지만 최악은 $O(n)$이다.