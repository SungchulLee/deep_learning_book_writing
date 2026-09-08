# 검은 높이

레드-블랙 성질은 뿌리에서 잎까지의 경로를 따라 색이 어떻게 놓이는지를 묶는다. 성질 5는 어떤 노드에서 자손 잎까지 가는 모든 경로에 검은 노드가 똑같이 있어야 한다고 요구한다. 이 개수, 곧 **검은 높이**가 색칠 규칙과 로그 높이 보장을 잇는 핵심 양이다. 레드-블랙 트리의 높이 한계를 증명하기에 앞서 검은 높이를 이해해야 한다.

---

## 1. 정의

노드 $x$의 **검은 높이** $\text{bh}(x)$은 $x$에서 잎(NIL 보초)까지 내려가는 아무 단순 경로에 있는 검은 노드의 수이며, **$x$ 자신은 세지 않는다**.

성질 5에 따라 어느 자손 잎으로 가든 이 개수가 같으므로 $\text{bh}(x)$은 잘 정의된다.

NIL 보초 노드(잎)의 검은 높이는 0이다.

$$
\text{bh}(\text{NIL}) = 0
$$

---

## 2. 검은 높이 계산하기

자식이 $l$과 $r$인 내부 노드 $x$에 대해 다음과 같다.

$$
\text{bh}(x) = \begin{cases} \text{bh}(l) & \text{if } l \text{ is black, then } \text{bh}(l) = \text{bh}(r) \\ \text{bh}(l) + 1 & \text{wait — let us be more careful} \end{cases}
$$

사실 이 관계는 자식의 색에 달려 있다.

- 자식 $c$이 **검으면** $\text{bh}(x) = \text{bh}(c) + 1$이다($x$에서 $c$을 지나는 경로가 $c$에서 검은 노드를 하나 더 얻는다).
- 자식 $c$이 **붉으면** $\text{bh}(x) = \text{bh}(c)$이다(붉은 노드는 세지 않는다).

성질 5가 두 자식 모두 같은 $\text{bh}(x)$을 준다고 보장하므로 어느 자식을 써도 된다.

$$
\text{bh}(x) = \begin{cases} \text{bh}(\text{child}) + 1 & \text{if child is black} \\ \text{bh}(\text{child}) & \text{if child is red} \end{cases}
$$

---

## 3. 핵심 보조정리

!!! info "보조정리: 부분 트리의 최소 크기"
    노드 $x$을 뿌리로 하는 부분 트리에는 내부 노드가 적어도 $2^{\text{bh}(x)} - 1$개 있다.

**$x$의 높이에 대한 귀납법으로 증명한다.**

*바탕 경우*: $x$이 잎(NIL)이면 $\text{bh}(x) = 0$이고 부분 트리의 내부 노드는 $0 = 2^0 - 1$개이다.

*귀납 단계*: $x$을 자식이 $l$과 $r$인 내부 노드라 하자. 자식마다 검은 높이가 적어도 $\text{bh}(x) - 1$이다(자식이 검으면 정확히 $\text{bh}(x) - 1$, 붉으면 $\text{bh}(x)$이다). 귀납 가정에 따라 자식의 부분 트리마다 내부 노드가 적어도 $2^{\text{bh}(x)-1} - 1$개 있다. 따라서 다음이 성립한다.

$$
n(x) \geq 1 + 2\bigl(2^{\text{bh}(x)-1} - 1\bigr) = 2^{\text{bh}(x)} - 1
$$

$\square$

---

## 4. 예

레드-블랙 트리를 생각해 보자 (B = 검정, R = 빨강).

```
          10(B)          bh = 2
         /     \
       5(R)    15(B)     bh(5)=2, bh(15)=1
      /   \    /   \
    3(B) 7(B) 13(R) 20(B)   bh(3)=1, bh(7)=1, bh(13)=1, bh(20)=0
   / \  / \   / \   / \
  N  N N  N 11(B) N  N  N
             / \
            N   N
```

- NIL 노드: $\text{bh} = 0$
- 노드 20 (검정, 잎에 가까움): $\text{bh}(20) = 0$ (경로가 곧바로 NIL로 간다)
- 노드 3 (검정): $\text{bh}(3) = 0 + 1 = 1$ (아래의 NIL을 세면 그렇다. 그런데 NIL은 검지만 $\text{bh}$은 NIL을 세지 않는다). 다시 정리해 보자.

$\text{bh}(x)$이 $x$ **아래**의 검은 노드를 센다는 규약을 쓰면($x$과 NIL은 빼고) 다음과 같다.

- 노드 20 (검정, 자식이 NIL): $\text{bh}(20) = 0$
- 노드 3 (검정, 자식이 NIL): $\text{bh}(3) = 0$
- 노드 7 (검정, 자식이 NIL): $\text{bh}(7) = 0$
- 노드 11 (검정, 자식이 NIL): $\text{bh}(11) = 0$
- 노드 13 (빨강, 자식은 11(B)과 NIL): $\text{bh}(13) = 0 + 1 = 1$
- 노드 15 (검정, 자식은 13(R)과 20(B)): $\text{bh}(15) = \text{bh}(13) = 1$이다(13이 붉으므로 더 세지 않는다). 확인해 보면 $\text{bh}(20) + 1 = 1$이다(20은 검다). 둘 다 1이다. 맞다.
- 노드 5 (빨강, 자식은 3(B)과 7(B)): $\text{bh}(5) = \text{bh}(3) + 1 = 1$
- 노드 10 (검정, 자식은 5(R)과 15(B)): $\text{bh}(10) = \text{bh}(5) = 1$이고(5가 붉다) $\text{bh}(15) + 1 = 2$이다…

이 어긋남은 예제 트리에 문제가 있음을 드러낸다. 성질 5를 만족하지 않는 것이다. 올바른 예를 대신 쓰자.

**바로잡은 예:**

```
          10(B)          bh = 2
         /     \
       5(B)    15(B)     bh = 1
      /   \    /   \
    3(R) 7(R) 13(R) 20(R)   bh = 1
```

- 노드 3, 7, 13, 20 (빨강, 자식이 NIL): $\text{bh} = 0 + 1 = 1$이다. 잠깐, NIL은 검으므로 붉은 노드에서 그 NIL 자식으로 가는 경로에는 검은 노드가 1개(NIL) 있다. 그런데 CLRS의 규약에서 $\text{bh}$은 경로 위의 검은 노드를 세되 $x$ 자신은 **빼고** NIL은 **넣는다**. 이 규약에서는 $\text{bh}(\text{NIL}) = 0$이고, NIL을 세면 $\text{bh}(3) = 0 + 1 = 1$이다…

**CLRS의 규약**을 정확히 따르자. $\text{bh}(x)$은 $x$에서 잎까지 가는 아무 경로 위의 검은 노드 수이며 **$x$은 세지 않는다**. NIL 보초는 $\text{bh} = 0$인 잎이다.

- 노드 3 (빨강, 자식이 NIL(B)): 3에서 잎까지의 경로는 {NIL}이고 검은 노드 수는 1이다. 따라서 $\text{bh}(3) = 1$이다.
- 노드 5 (검정, 왼쪽 자식 3(R)): 5에서 3을 지나 잎까지의 경로는 {3, NIL}이다. 검은 노드 수는 1(NIL뿐)이다. 따라서 $\text{bh}(5) = 1$이다.
- 노드 10 (검정, 왼쪽 자식 5(B)): 5를 지나 3을 지나 NIL까지의 경로는 {5, 3, NIL}이다. 검은 노드 수는 2(5와 NIL)이다. 따라서 $\text{bh}(10) = 2$이다.

보조정리를 확인해 보자. 노드 10의 부분 트리에는 내부 노드가 7개 있고 $2^{\text{bh}(10)} - 1 = 2^2 - 1 = 3$이다. 과연 $7 \geq 3$이다.

---

## 5. 구현

```python
"""
적흑 트리의 검은 높이 셈하기.

검은 높이의 정의를 보이고 노드에서 잎까지의 모든 경로에서
검은 노드의 수가 같음을 확인한다.
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

    def __repr__(self):
        return f"{self.key}({self.color})"

# 파수 NIL 노드
NIL = RBNode(key=None, color=BLACK)
NIL.left = NIL
NIL.right = NIL

# === 검은 높이 셈하기 ===

def black_height(node):
    """노드의 검은 높이를 셈한다(CLRS 관례).

    노드 자신은 세지 않고, 노드에서 자손 잎까지의 아무 경로에나 있는
    검은 노드의 수를 돌려준다.
    트리가 성질 5를 어기면 -1을 돌려준다.
    """
    if node is NIL:
        return 0

    left_bh = black_height(node.left)
    right_bh = black_height(node.right)

    if left_bh == -1 or right_bh == -1:
        return -1  # 부분 트리에 위반이 있다

    # 자식의 색에 맞추어 고친다
    left_count = left_bh + (1 if node.left.color == BLACK else 0)
    right_count = right_bh + (1 if node.right.color == BLACK else 0)

    if left_count != right_count:
        print(f"  Property 5 violation at {node}: "
              f"left bh={left_count}, right bh={right_count}")
        return -1

    return left_count

# === 트리 세우개 (손수) ===

def build_example_tree():
    """보여 주기 위해 올바른 적흑 트리를 세운다."""
    root = RBNode(10, BLACK)
    root.left = RBNode(5, BLACK)
    root.right = RBNode(15, BLACK)
    root.left.left = RBNode(3, RED)
    root.left.right = RBNode(7, RED)
    root.right.left = RBNode(13, RED)
    root.right.right = RBNode(20, RED)

    # NIL 자식을 둔다
    for node in [root.left.left, root.left.right,
                 root.right.left, root.right.right]:
        node.left = NIL
        node.right = NIL
    root.left.left.left = NIL
    root.left.left.right = NIL

    return root

# === 보이기 ===

def print_tree(node, level=0):
    """색과 검은 높이와 함께 트리를 옆으로 찍는다."""
    if node is NIL:
        return
    print_tree(node.right, level + 1)
    bh = black_height(node)
    indent = "    " * level
    print(f"{indent}{node.key}({node.color}) bh={bh}")
    print_tree(node.left, level + 1)

if __name__ == "__main__":
    root = build_example_tree()
    print("Red-Black Tree with black-heights:")
    print_tree(root)
    print()
    print(f"Root black-height: {black_height(root)}")
    n = 7  # 내부 노드
    bh = black_height(root)
    print(f"Internal nodes: {n}")
    print(f"Lemma check: 2^bh - 1 = {2**bh - 1} <= {n}: {2**bh - 1 <= n}")
```

**출력:**
```
Red-Black Tree with black-heights:
        20(R) bh=1
    15(B) bh=1
        13(R) bh=1
10(B) bh=2
        7(R) bh=1
    5(B) bh=1
        3(R) bh=1

Root black-height: 2
Internal nodes: 7
Lemma check: 2^bh - 1 = 3 <= 7: True
```

---

## 6. 무엇이 중요한가

검은 높이는 결정적인 두 구실을 한다.

1. **높이 한계의 증명**: 보조정리 $n \geq 2^{\text{bh}(x)} - 1$과 $\text{bh}(\text{root}) \geq h/2$을 함께 쓰면 $h \leq 2\log_2(n+1)$이 나온다. 이 증명은 높이의 한계 절에서 온전히 펼친다.

2. **알고리즘의 올바름**: 삽입과 삭제의 손질 과정에서 알고리즘은 검은 높이의 불변식을 지킨다. 손질 절차의 모든 경우 분석이 색을 바꾸고 회전한 뒤에도 검은 높이가 어긋나지 않는지 확인한다.

---

## 연습문제

**연습문제 1.**
검은 높이의 균형 불변식을 밝히고 그것이 높이 $O(\log n)$을 보장함을 증명하라.

??? success "연습문제 1 풀이"
    각 구조의 불변식(균형 인수, 색의 성질, 차수 제약)이 경로 길이의 치우침을 묶는다. 높이의 한계는 그 불변식에서 따라 나온다. 트리의 층마다 (불변식이 정하는) 최소한의 노드가 있어야 하므로 전체 노드 수 $n$이 높이에 따라 지수적으로 늘고, 따라서 $h = O(\log n)$이다.

---

**연습문제 2.**
구조를 다시 짜야 하는(회전, 색 바꾸기, 쪼개기·합치기) 트리에서 검은 높이를 따라가라. 앞뒤의 상태를 보여라.

??? success "연습문제 2 풀이"
    이 쪽에서 설명한 재구성 상황을 일으키는 트리를 하나 만들어라. 어긋난 곳을 보이고, 어느 경우에 해당하는지 가리고, 고친 뒤, 불변식이 되살아났는지 확인하라.

---

**연습문제 3.**
검은 높이이(가) 구조를 다시 짜는 연산을 많아야 $O(\log n)$번 필요로 함을 증명하라.

??? success "연습문제 3 풀이"
    구조를 다시 짤 때마다 어긋난 곳이 뿌리에 한 층 가까워지거나 해소된다. 트리의 층이 $O(\log n)$개이므로 재구성은 많아야 $O(\log n)$번 필요하다. 레드-블랙 삽입 같은 연산에서는 회전 2번과 색 바꾸기 $O(\log n)$번이면 충분하다. $\square$

---

**연습문제 4.**
최악의 높이, 연산마다의 회전 횟수, 구현의 까다로움 면에서 검은 높이를 다른 균형 트리 구조와 견주어라.

??? success "연습문제 4 풀이"
    AVL은 높이가 $1.44\log n$ 이하이고 삭제마다 회전이 $O(\log n)$번까지 든다. 레드-블랙은 높이가 $2\log n$ 이하이고 연산마다 회전이 많아야 3번이다. B-트리는 높이가 $O(\log_B n)$이며 디스크 입출력에 맞추어져 있다. 스플레이 트리는 분할 상환으로 $O(\log n)$이지만 최악은 $O(n)$이다.

## 정리하며

이 마당은 정의、검은 높이 계산하기、핵심 보조정리、예을 차례로 짚었다.

**참고 문헌**

- [Introduction to Algorithms (CLRS), 13장](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
