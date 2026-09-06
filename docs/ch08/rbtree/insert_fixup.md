# 삽입 뒤 손질

레드-블랙 트리에 붉은 노드 $z$을 넣은 뒤 깨질 수 있는 성질은 성질 4뿐이다. $z$과 그 부모 $p$이 둘 다 붉을 수 있는 것이다. 삽입 뒤 손질 절차는 트리를 거슬러 오르며 걸음마다 세 경우 가운데 하나를 적용하여 이 빨강-빨강 충돌을 푼다. 이 절차는 많아야 색 바꾸기 $O(\log n)$번과 **회전 두 번** 뒤에 끝나며 다섯 가지 레드-블랙 성질을 모두 되살린다.

## 설정

손질 반복문은 $z$의 부모가 붉은 동안 돈다(부모가 검으면 어긋남이 없다). 되풀이마다 $z$은 고치고 있는 노드, $p = z.\text{parent}$은 붉은 부모, $g = p.\text{parent}$은 $p$의 부모이다($p$이 붉고 삽입 전에 트리가 성질 4를 만족했으므로 $g$은 검어야 한다).

세 경우는 $z$의 **삼촌** $u$($p$의 형제)의 색에 달려 있다.

## 경우 1: 삼촌이 붉다 (색 바꾸기)

삼촌 $u$이 붉으면 $p$과 $u$ 둘 다 검은 할아버지 $g$의 붉은 자식이다.

**고치는 법**: $p$과 $u$을 검게, $g$을 붉게 칠한다. 그러면 성질 5가 지켜지지만($g$을 지나는 검은 높이가 그대로이다) $g$과 $g$의 부모 사이에 새 빨강-빨강 위반이 생길 수 있다.

```
Before:              After:
    g(B)                g(R)    <- may violate Prop 4
   / \                 / \
  p(R) u(R)          p(B) u(B)
 /                   /
z(R)                z(R)
```

색을 바꾼 뒤 $z \leftarrow g$으로 두고 반복문을 되풀이한다. 어긋남이 두 층 위로 올라가므로 이 경우는 많아야 $O(\log n)$번 되풀이된다.

## 경우 2: 삼촌이 검고 $z$이 안쪽 자식이다 (회전 뒤 이어짐)

삼촌 $u$이 검고 $z$이 "안쪽" 자식이면(왼쪽 부모의 오른쪽 자식이거나 오른쪽 부모의 왼쪽 자식이면) 단일 회전으로 곧바로 고칠 수 없다.

**고치는 법**: $z$을 $p$의 자리로 올려 $p$을 바깥쪽 자식으로 만든다. 그러면 경우 2가 경우 3으로 바뀐다.

$p$이 $g$의 왼쪽 자식이고 $z$이 $p$의 오른쪽 자식인 경우에는 다음과 같다.

```
Before:              After left-rotate at p:
    g(B)                 g(B)
   / \                  / \
  p(R) u(B)           z(R) u(B)
    \                 /
    z(R)             p(R)
```

$z \leftarrow p$으로 두고(옛 부모가 이제 바깥쪽 자식이다) 경우 3으로 넘어간다.

## 경우 3: 삼촌이 검고 $z$이 바깥쪽 자식이다 (회전과 색 바꾸기)

삼촌 $u$이 검고 $z$이 "바깥쪽" 자식이면(왼쪽 부모의 왼쪽 자식이거나 오른쪽 부모의 오른쪽 자식이면) $g$에서 회전 한 번과 색 바꾸기로 어긋남이 완전히 풀린다.

**고치는 법**: $p$을 검게, $g$을 붉게 칠한 뒤 $g$을 반대 방향으로 회전한다.

$p$이 $g$의 왼쪽 자식인 경우에는 다음과 같다.

```
Before:              After right-rotate at g:
    g(B)                p(B)
   / \                 / \
  p(R) u(B)          z(R) g(R)
 /                          \
z(R)                        u(B)
```

경우 3을 마치면 어긋남이 풀린다. (이제 이 부분 트리의 뿌리인) $p$이 검고 어느 자식도 빨강-빨강 충돌을 만들지 않는다. 반복문이 끝난다.

## 경우 정리

| 경우 | 삼촌 | $z$의 자리 | 하는 일 | 이어지는가 |
|:-:|:-:|:-:|:--|:-:|
| 1 | 빨강 | 어느 쪽이든 | $p$, $u$, $g$의 색 바꾸기 | 그렇다 (위로 올라감) |
| 2 | 검정 | 안쪽 자식 | $p$에서 회전 | 경우 3으로 넘어감 |
| 3 | 검정 | 바깥쪽 자식 | $g$에서 회전과 색 바꾸기 | 아니다 (끝난다) |

대칭적인 경우($p$이 $g$의 오른쪽 자식일 때)는 위의 것을 좌우로 뒤집은 것이다.

## 끝남과 복잡도

- **경우 1**은 $z$을 두 층 위로 올리므로 많아야 $h/2 = O(\log n)$번 실행된다.
- **경우 2**는 회전 한 번으로 경우 3이 된다.
- **경우 3**은 회전 한 번으로 반복문을 끝낸다.

따라서 회전은 모두 합쳐 많아야 **2번**(경우 2에서 한 번, 경우 3에서 한 번)이고 색 바꾸기는 많아야 $O(\log n)$번이다.

반복문이 끝나면 뿌리를 검게 칠한다(경우 1의 색 바꾸기로 붉어졌다면 성질 2를 만족시키기 위해서이다).

## 구현

```python
"""
온전한 바로잡기를 갖춘 적흑 트리 삽입.

CLRS를 따라 INSERT-FIXUP의 세 경우를 모두 구현하며,
회전이 많아야 두 번임을 보인다.
"""


# === 상수 ===

RED = "R"
BLACK = "B"


# === 적흑 노드 ===

class RBNode:
    """적흑 트리의 노드."""

    def __init__(self, key, color=RED):
        self.key = key
        self.color = color
        self.left = None
        self.right = None
        self.parent = None

    def __repr__(self):
        return f"{self.key}({self.color})"


# === 파수 ===

NIL = RBNode(key=None, color=BLACK)
NIL.left = NIL
NIL.right = NIL


# === 회전 ===

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


# === 삽입 바로잡기 ===

def insert_fixup(tree, z):
    """빨간 노드 z를 넣은 뒤의 적흑 위반을 바로잡는다.

    회전은 많아야 두 번, 다시 칠하기는 O(log n)번이다.
    """
    rotations = 0

    while z.parent is not None and z.parent.color == RED:
        if z.parent is z.parent.parent.left:
            uncle = z.parent.parent.right

            if uncle.color == RED:
                # 경우 1: 삼촌이 빨갛다
                z.parent.color = BLACK
                uncle.color = BLACK
                z.parent.parent.color = RED
                z = z.parent.parent
            else:
                if z is z.parent.right:
                    # 경우 2: 삼촌이 검고 z가 안쪽 자식이다
                    z = z.parent
                    left_rotate(tree, z)
                    rotations += 1
                # 경우 3: 삼촌이 검고 z가 바깥쪽 자식이다
                z.parent.color = BLACK
                z.parent.parent.color = RED
                right_rotate(tree, z.parent.parent)
                rotations += 1
        else:
            # 대칭: 부모가 조부모의 오른쪽 자식이다
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
                    rotations += 1
                z.parent.color = BLACK
                z.parent.parent.color = RED
                left_rotate(tree, z.parent.parent)
                rotations += 1

    tree["root"].color = BLACK
    return rotations


# === 삽입 ===

def rb_insert(tree, key):
    """바로잡기와 함께 적흑 트리에 열쇠를 넣는다."""
    z = RBNode(key, RED)
    z.left = NIL
    z.right = NIL

    y = None
    x = tree["root"]

    while x is not NIL:
        y = x
        if z.key < x.key:
            x = x.left
        else:
            x = x.right

    z.parent = y
    if y is None:
        tree["root"] = z
    elif z.key < y.key:
        y.left = z
    else:
        y.right = z

    rots = insert_fixup(tree, z)
    return rots


# === 보이기 ===

def print_tree(node, level=0):
    """색과 함께 트리를 옆으로 찍는다."""
    if node is NIL:
        return
    print_tree(node.right, level + 1)
    print(f"{'    ' * level}{node.key}({node.color})")
    print_tree(node.left, level + 1)


if __name__ == "__main__":
    tree = {"root": NIL}

    keys = [10, 20, 30, 15, 25, 5, 1]
    for key in keys:
        rots = rb_insert(tree, key)
        print(f"Insert {key}: {rots} rotation(s)")
        print_tree(tree["root"])
        print()
```

**출력:**
```
Insert 10: 0 rotation(s)
10(B)

Insert 20: 0 rotation(s)
    20(R)
10(B)

Insert 30: 2 rotation(s)
    30(R)
20(B)
    10(R)

Insert 15: 0 rotation(s)
    30(B)
20(B)
        15(R)
    10(B)

Insert 25: 2 rotation(s)
    30(B)
        25(R)
20(B)
        15(R)
    10(B)

Insert 5: 0 rotation(s)
    30(B)
        25(R)
20(B)
        15(R)
    10(B)
        5(R)

Insert 1: 2 rotation(s)
    30(B)
        25(R)
20(B)
        15(B)
    5(B)
        10(R)
            1(R)
```

삽입마다 회전이 많아야 2번 쓰여 이론적 보장이 확인된다.

## 참고 문헌

- [Introduction to Algorithms (CLRS), 13장](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)


## 연습문제

**연습문제 1.**
삽입 뒤 손질의 균형 불변식을 밝히고 그것이 높이 $O(\log n)$을 보장함을 증명하라.

??? success "연습문제 1 풀이"
    각 구조의 불변식(균형 인수, 색의 성질, 차수 제약)이 경로 길이의 치우침을 묶는다. 높이의 한계는 그 불변식에서 따라 나온다. 트리의 층마다 (불변식이 정하는) 최소한의 노드가 있어야 하므로 전체 노드 수 $n$이 높이에 따라 지수적으로 늘고, 따라서 $h = O(\log n)$이다.

---

**연습문제 2.**
구조를 다시 짜야 하는(회전, 색 바꾸기, 쪼개기·합치기) 트리에서 삽입 뒤 손질을 따라가라. 앞뒤의 상태를 보여라.

??? success "연습문제 2 풀이"
    이 쪽에서 설명한 재구성 상황을 일으키는 트리를 하나 만들어라. 어긋난 곳을 보이고, 어느 경우에 해당하는지 가리고, 고친 뒤, 불변식이 되살아났는지 확인하라.

---

**연습문제 3.**
삽입 뒤 손질이(가) 구조를 다시 짜는 연산을 많아야 $O(\log n)$번 필요로 함을 증명하라.

??? success "연습문제 3 풀이"
    구조를 다시 짤 때마다 어긋난 곳이 뿌리에 한 층 가까워지거나 해소된다. 트리의 층이 $O(\log n)$개이므로 재구성은 많아야 $O(\log n)$번 필요하다. 레드-블랙 삽입 같은 연산에서는 회전 2번과 색 바꾸기 $O(\log n)$번이면 충분하다. $\square$

---

**연습문제 4.**
최악의 높이, 연산마다의 회전 횟수, 구현의 까다로움 면에서 삽입 뒤 손질을 다른 균형 트리 구조와 견주어라.

??? success "연습문제 4 풀이"
    AVL은 높이가 $1.44\log n$ 이하이고 삭제마다 회전이 $O(\log n)$번까지 든다. 레드-블랙은 높이가 $2\log n$ 이하이고 연산마다 회전이 많아야 3번이다. B-트리는 높이가 $O(\log_B n)$이며 디스크 입출력에 맞추어져 있다. 스플레이 트리는 분할 상환으로 $O(\log n)$이지만 최악은 $O(n)$이다.