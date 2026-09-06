# 삽입

레드-블랙 트리의 삽입은 표준 이진 탐색 트리 삽입과 똑같이 시작한다. 뿌리에서 내려가 새 노드를 잎으로 붙인다. 새 노드는 언제나 **붉게** 칠하는데, 붉은 노드를 더해도 어떤 경로의 검은 높이도 바뀌지 않기 때문이다(성질 5가 지켜진다). 그런데 새 노드의 부모도 붉으면 성질 4(붉은 노드가 잇따를 수 없다)가 깨진다. 다음 절에서 다루는 **삽입 뒤 손질** 절차가 색 바꾸기와 많아야 두 번의 회전으로 모든 성질을 되살린다.

## 왜 새 노드를 붉게 칠하는가

새 노드를 검게 칠하면 그것을 지나는 모든 경로의 검은 높이가 1씩 늘어 모든 조상에서 성질 5가 깨진다. 성질 5를 되살리는 일은 경로 전체의 개수에 영향을 주므로 비싸다.

새 노드를 붉게 칠하면 성질 5는 지켜지지만 (부모가 붉으면) 성질 4가 깨질 수 있다. 성질 4를 되살리는 일은 국소적이다. 빨강-빨강 충돌 하나만 고치면 되고, 삽입 경로 위에 머무는 색 바꾸기와 회전으로 풀 수 있다.

## 삽입 절차

삽입 절차는 CLRS의 형식을 따른다.

**1단계.** 표준 이진 탐색 트리 삽입을 한다. 뿌리에서 내려가며 새 열쇠를 노드마다 견준다. NIL 보초에 닿으면 그것을 새 노드로 바꾼다.

**2단계.** 새 노드를 붉게 칠한다.

**3단계.** 새 노드의 자식을 NIL 보초로 둔다.

**4단계.** 레드-블랙 성질을 되살리려고 `INSERT-FIXUP`을 부른다.

### 의사 코드

```
RB-INSERT(T, z):
    y = T.nil
    x = T.root
    while x != T.nil:
        y = x
        if z.key < x.key:
            x = x.left
        else:
            x = x.right
    z.parent = y
    if y == T.nil:
        T.root = z
    elif z.key < y.key:
        y.left = z
    else:
        y.right = z
    z.left = T.nil
    z.right = T.nil
    z.color = RED
    RB-INSERT-FIXUP(T, z)
```

## 무엇이 잘못될 수 있는가

붉은 노드 $z$을 넣은 뒤에는 다음과 같다.

- **성질 1** (노드는 붉거나 검다): 만족한다. $z$은 붉다.
- **성질 2** (뿌리는 검다): $z$이 뿌리일 때만(트리가 비어 있었을 때만) 깨진다. 고치는 법: 뿌리를 검게 칠한다.
- **성질 3** (잎은 검다): 만족한다. $z$의 자식은 NIL 보초(검정)이다.
- **성질 4** (빨강-빨강 없음): $z$의 부모가 붉으면 깨진다.
- **성질 5** (검은 높이가 한결같다): 만족한다. $z$이 붉고, 검은 NIL 하나를 검은 NIL 둘을 가진 붉은 노드로 바꾸었기 때문이다.

따라서 깨질 수 있는 것은 성질 2(고치기 쉽다)와 성질 4(손질이 처리한다)뿐이다.

## 구현

```python
"""
적흑 트리 삽입(바로잡기는 다음 절에 있으므로 여기서는 뺀다).

이진 탐색 트리 삽입 단계와 바로잡기가 필요해지게 만드는
처음의 빨간 칠하기를 보인다.
"""


# === 상수 ===

RED = "R"
BLACK = "B"


# === 적흑 노드 ===

class RBNode:
    """열쇠와 색과 자식과 부모를 지닌 적흑 트리 노드."""

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
NIL.parent = NIL


# === 이진 탐색 트리 삽입 단계 ===

def bst_insert(tree, z):
    """표준 이진 탐색 트리 삽입으로 노드 z를 트리에 넣는다.

    z를 빨갛게 칠하고 자식을 NIL으로 둔다.
    적흑 위반을 바로잡지는 **않는다**.
    """
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

    z.left = NIL
    z.right = NIL
    z.color = RED


# === 보이기 ===

def print_tree(node, level=0):
    """색과 함께 트리를 옆으로 찍는다."""
    if node is NIL:
        return
    print_tree(node.right, level + 1)
    indent = "    " * level
    print(f"{indent}{node.key}({node.color})")
    print_tree(node.left, level + 1)


def check_property_4(node):
    """어디선가 성질 4가 어겨졌는지 살핀다."""
    if node is NIL:
        return True
    if node.color == RED:
        if node.left.color == RED or node.right.color == RED:
            print(f"  Property 4 VIOLATED: {node} has red child")
            return False
    return check_property_4(node.left) and check_property_4(node.right)


if __name__ == "__main__":
    # 노드를 넣고 바로잡기 전의 빨강-빨강 위반을 보인다
    tree = {"root": NIL}

    # 뿌리를 넣는다 (빨갛게 되므로 검게 다시 칠해야 한다)
    node10 = RBNode(10)
    bst_insert(tree, node10)
    tree["root"].color = BLACK  # 성질 2를 바로잡는다
    print("After inserting 10 (root, colored black):")
    print_tree(tree["root"])
    print()

    # 5를 넣는다 (빨갛고 부모가 검으므로 위반이 없다)
    node5 = RBNode(5)
    bst_insert(tree, node5)
    print("After inserting 5 (red, parent black -> OK):")
    print_tree(tree["root"])
    check_property_4(tree["root"])
    print()

    # 3을 넣는다 (빨갛고 부모 5도 빨가므로 위반이다)
    node3 = RBNode(3)
    bst_insert(tree, node3)
    print("After inserting 3 (red, parent red -> VIOLATION):")
    print_tree(tree["root"])
    check_property_4(tree["root"])
    print("  -> INSERT-FIXUP needed to resolve this violation")
```

**출력:**
```
After inserting 10 (root, colored black):
10(B)

After inserting 5 (red, parent black -> OK):
10(B)
    5(R)

After inserting 3 (red, parent red -> VIOLATION):
10(B)
    5(R)
        3(R)
  Property 4 VIOLATED: 5(R) has red child
  -> INSERT-FIXUP needed to resolve this violation
```

노드 5의 어긋남(붉은 자식을 가진 붉은 노드)이 바로 `INSERT-FIXUP`이 푸는 상황이며 다음 절에서 자세히 다룬다.

## 복잡도

이진 탐색 트리 삽입 단계는 (트리를 내려가는 데) $O(\log n)$ 시간이 걸린다. (다음 절의) 손질도 회전 두 번과 함께 $O(\log n)$ 시간이 걸린다. 따라서 전체 삽입 시간은 $O(\log n)$이다.

## 참고 문헌

- [Introduction to Algorithms (CLRS), 13장](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)


## 연습문제

**연습문제 1.**
삽입의 균형 불변식을 밝히고 그것이 높이 $O(\log n)$을 보장함을 증명하라.

??? success "연습문제 1 풀이"
    각 구조의 불변식(균형 인수, 색의 성질, 차수 제약)이 경로 길이의 치우침을 묶는다. 높이의 한계는 그 불변식에서 따라 나온다. 트리의 층마다 (불변식이 정하는) 최소한의 노드가 있어야 하므로 전체 노드 수 $n$이 높이에 따라 지수적으로 늘고, 따라서 $h = O(\log n)$이다.

---

**연습문제 2.**
구조를 다시 짜야 하는(회전, 색 바꾸기, 쪼개기·합치기) 트리에서 삽입을 따라가라. 앞뒤의 상태를 보여라.

??? success "연습문제 2 풀이"
    이 쪽에서 설명한 재구성 상황을 일으키는 트리를 하나 만들어라. 어긋난 곳을 보이고, 어느 경우에 해당하는지 가리고, 고친 뒤, 불변식이 되살아났는지 확인하라.

---

**연습문제 3.**
삽입이(가) 구조를 다시 짜는 연산을 많아야 $O(\log n)$번 필요로 함을 증명하라.

??? success "연습문제 3 풀이"
    구조를 다시 짤 때마다 어긋난 곳이 뿌리에 한 층 가까워지거나 해소된다. 트리의 층이 $O(\log n)$개이므로 재구성은 많아야 $O(\log n)$번 필요하다. 레드-블랙 삽입 같은 연산에서는 회전 2번과 색 바꾸기 $O(\log n)$번이면 충분하다. $\square$

---

**연습문제 4.**
최악의 높이, 연산마다의 회전 횟수, 구현의 까다로움 면에서 삽입을 다른 균형 트리 구조와 견주어라.

??? success "연습문제 4 풀이"
    AVL은 높이가 $1.44\log n$ 이하이고 삭제마다 회전이 $O(\log n)$번까지 든다. 레드-블랙은 높이가 $2\log n$ 이하이고 연산마다 회전이 많아야 3번이다. B-트리는 높이가 $O(\log_B n)$이며 디스크 입출력에 맞추어져 있다. 스플레이 트리는 분할 상환으로 $O(\log n)$이지만 최악은 $O(n)$이다.