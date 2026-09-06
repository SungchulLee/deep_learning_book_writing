# 균형 인수

보통의 이진 탐색 트리에 열쇠를 정렬된 순서로 넣으면 찾기에 $O(n)$이 걸리는 치우친 사슬이 된다. AVL 트리는 노드마다 얼마나 "한쪽으로 기울었는지"를 좇고 그 기욺이 문턱을 넘으면 다시 균형을 잡아 이런 퇴화를 막는다. **균형 인수**는 이 기욺을 나타내는 정수량이며 AVL 트리의 모든 회전을 일으키는 방아쇠이다.

## 정의

이진 트리의 노드 $x$에 대해 $h(x)$을 $x$을 뿌리로 하는 부분 트리의 높이라 하고, 뿌리에서 잎까지 가장 긴 경로의 길이로 정의한다. 빈 부분 트리(널 자식)의 높이는 관례로 $-1$이다.

노드 $x$의 **균형 인수**는 다음과 같다.

$$
\text{BF}(x) = h(\text{left}(x)) - h(\text{right}(x))
$$

여기서 $\text{left}(x)$과 $\text{right}(x)$은 각각 $x$의 왼쪽 자식과 오른쪽 자식이다. 균형 인수는 그저 왼쪽 부분 트리의 높이에서 오른쪽 부분 트리의 높이를 뺀 것이다.

## AVL 불변식

AVL 트리는 모든 노드가 **AVL 균형 조건**을 만족하는 이진 탐색 트리이다.

$$
\text{BF}(x) \in \{-1, 0, 1\}
$$

트리의 모든 노드 $x$에 대해 그러하다. 같은 말로, 어떤 노드든 두 자식 부분 트리의 높이 차이가 많아야 1이다.

삽입이나 삭제 때문에 어떤 노드의 균형 인수가 $\{-1, 0, 1\}$을 벗어나면, 곧 $|\text{BF}(x)| \geq 2$이 되면, 트리는 그 노드에서 회전을 한두 번 하여 불변식을 되살린다.

## 값의 뜻

허용되는 세 값은 저마다 노드의 다른 모양을 나타낸다.

| 균형 인수 | 뜻 |
|:-:|:--|
| $+1$ | 왼쪽 부분 트리가 오른쪽보다 한 층 높다 (왼쪽으로 무겁다) |
| $0$ | 두 부분 트리의 높이가 같다 (이 노드에서 완벽히 균형 잡혔다) |
| $-1$ | 오른쪽 부분 트리가 왼쪽보다 한 층 높다 (오른쪽으로 무겁다) |

균형 인수가 $+2$이면 왼쪽 부분 트리가 너무 높다는 뜻이며 오른쪽 회전(또는 왼쪽-오른쪽 이중 회전)이 필요하다. $-2$이면 오른쪽 부분 트리가 너무 높다는 뜻이며 왼쪽 회전(또는 오른쪽-왼쪽 이중 회전)이 필요하다.

## 균형 인수 계산하기

노드마다 제 높이(또는 같은 뜻으로 균형 인수)를 정수 칸에 담는다. 삽입이나 삭제를 할 때마다 알고리즘은 고친 잎에서 뿌리까지 거슬러 올라가며 높이를 갱신하고 균형 인수를 살핀다.

자식이 $l$과 $r$인 노드 $x$에 대해 다음과 같다.

$$
h(x) = 1 + \max\bigl(h(l),\, h(r)\bigr)
$$

$$
\text{BF}(x) = h(l) - h(r)
$$

$|\text{BF}(x)| \leq 1$이면 그 노드는 균형 잡혀 있으므로 계속 위로 올라간다. $|\text{BF}(x)| = 2$이면 $x$에서 알맞은 회전을 한다.

## 예

노드마다 균형 인수를 적어 둔 다음 AVL 트리를 보자.

```
        30 [+1]
       /  \
     20 [0]  40 [-1]
    /  \       \
  10 [0] 25 [0]  50 [0]
```

- 노드 30은 왼쪽 부분 트리의 높이가 2이고 오른쪽도 2인데, 노드 20을 지나는 왼쪽이 높이 2이고 40을 지나는 오른쪽도 높이 2이다. 잠깐, 다시 세어 보자. 노드 10의 높이가 0, 노드 25의 높이가 0이므로 노드 20의 높이는 1이다. 노드 50의 높이가 0이므로 노드 40의 높이는 1이다. 노드 30은 양쪽 다 높이가 1이므로 $\text{BF}(30) = 1 - 1 = 0$이다.

바로잡아 적은 트리는 다음과 같다.

```
        30 [0]
       /  \
     20 [0]  40 [-1]
    /  \       \
  10 [0] 25 [0]  50 [0]
```

- 노드 10: 자식 없음, $\text{BF} = (-1) - (-1) = 0$
- 노드 25: 자식 없음, $\text{BF} = (-1) - (-1) = 0$
- 노드 20: $h(\text{left}) = 0$, $h(\text{right}) = 0$이므로 $\text{BF} = 0$
- 노드 50: 자식 없음, $\text{BF} = 0$
- 노드 40: $h(\text{left}) = -1$, $h(\text{right}) = 0$이므로 $\text{BF} = -1$
- 노드 30: $h(\text{left}) = 1$, $h(\text{right}) = 1$이므로 $\text{BF} = 0$

이제 5를 넣는다고 하자. 노드 10의 왼쪽으로 간다.

```
          30 [+1]
         /  \
       20 [+1]  40 [-1]
      /  \       \
   10 [+1] 25 [0]  50 [0]
   /
  5 [0]
```

모든 균형 인수가 $\{-1, 0, +1\}$에 남아 있으므로 회전이 필요 없다. 그런데 3을 더 넣으면 다음과 같다.

```
            30 [+2]       <-- violation!
           /  \
         20 [+2]  40 [-1]  <-- violation!
        /  \       \
     10 [+2] 25 [0]  50 [0]  <-- violation!
     /
    5 [+1]
   /
  3 [0]
```

이제 노드 10의 $\text{BF} = +2$이 되어 AVL 조건이 깨진다. 노드 10에서 오른쪽 회전을 하면 균형이 되살아난다. 실제로 알고리즘은 균형이 깨진 가장 낮은 조상에서 이를 알아채고 거기서 회전한다. 노드 10에서 고친 결과가 위로 퍼지며 모든 조상에 충분할 수 있다.

## 구현

```python
"""
균형 인수를 셈하는 AVL 트리 노드.

AVL 트리의 균형 되잡기의 바탕이 되는 높이 추적과
균형 인수 셈하기를 보인다.
"""


# === AVL 노드 정의 ===

class AVLNode:
    """제 높이를 스스로 좇는 AVL 트리의 노드."""

    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
        self.height = 0  # 잎의 높이는 0이다

    def __repr__(self):
        return f"AVLNode({self.key})"


# === 높이와 균형 인수 도구 ===

def height(node):
    """노드의 높이를 돌려준다. 널이면 -1을 돌려준다."""
    if node is None:
        return -1
    return node.height


def update_height(node):
    """자식에서 노드의 높이를 다시 셈한다."""
    node.height = 1 + max(height(node.left), height(node.right))


def balance_factor(node):
    """균형 인수 = 왼쪽 높이 - 오른쪽 높이를 셈한다."""
    return height(node.left) - height(node.right)


# === 시연 ===

def insert_bst(node, key):
    """그냥 이진 탐색 트리에 열쇠를 넣는다(균형을 되잡지 않는다)."""
    if node is None:
        return AVLNode(key)
    if key < node.key:
        node.left = insert_bst(node.left, key)
    elif key > node.key:
        node.right = insert_bst(node.right, key)
    update_height(node)
    return node


def print_balance_factors(node, level=0):
    """균형 인수와 함께 트리를 찍는다."""
    if node is None:
        return
    print_balance_factors(node.right, level + 1)
    indent = "    " * level
    bf = balance_factor(node)
    print(f"{indent}{node.key} [BF={bf:+d}]")
    print_balance_factors(node.left, level + 1)


if __name__ == "__main__":
    root = None
    for key in [30, 20, 40, 10, 25, 50]:
        root = insert_bst(root, key)

    print("AVL tree with balance factors:")
    print_balance_factors(root)
    print()

    # 5와 3을 넣어 불균형을 만든다
    root = insert_bst(root, 5)
    root = insert_bst(root, 3)
    print("After inserting 5 and 3 (imbalanced):")
    print_balance_factors(root)
```

**출력:**
```
AVL tree with balance factors:
        50 [BF=+0]
    40 [BF=-1]
30 [BF=+0]
        25 [BF=+0]
    20 [BF=+0]
        10 [BF=+0]

After inserting 5 and 3 (imbalanced):
        50 [BF=+0]
    40 [BF=-1]
30 [BF=+2]
        25 [BF=+0]
    20 [BF=+2]
            5 [BF=+1]
                3 [BF=+0]
        10 [BF=+2]
```

노드 10의 $\text{BF} = +2$은 오른쪽 회전을 일으킬 AVL 위반을 확인해 준다.

## 회전과의 관계

균형 인수가 어떤 회전을 할지 정한다.

| $\text{BF}(x)$ | $\text{BF}(\text{자식})$ | 회전 |
|:-:|:-:|:--|
| $+2$ | $+1$ 또는 $0$ | $x$에서 단일 오른쪽 회전 |
| $+2$ | $-1$ | $x$에서 왼쪽-오른쪽 이중 회전 |
| $-2$ | $-1$ 또는 $0$ | $x$에서 단일 왼쪽 회전 |
| $-2$ | $+1$ | $x$에서 오른쪽-왼쪽 이중 회전 |

자식의 균형 인수가 무거운 부분 트리가 "같은 쪽"(단일 회전)에 있는지 "반대쪽"(이중 회전)에 있는지를 정한다. 단일 회전과 이중 회전은 다음 절에서 다룬다.

## 참고 문헌

- [10.1 AVL Tree - Insertion and Rotations](https://www.youtube.com/watch?v=jDM6_TnYIqE&list=PLDN4rrl48XKpZkf03iYFl-O29szjTrs_O&index=76)
- [Introduction to Algorithms (CLRS), 13장](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)


## 연습문제

**연습문제 1.**
균형 인수의 균형 불변식을 밝히고 그것이 높이 $O(\log n)$을 보장함을 증명하라.

??? success "연습문제 1 풀이"
    각 구조의 불변식(균형 인수, 색의 성질, 차수 제약)이 경로 길이의 치우침을 묶는다. 높이의 한계는 그 불변식에서 따라 나온다. 트리의 층마다 (불변식이 정하는) 최소한의 노드가 있어야 하므로 전체 노드 수 $n$이 높이에 따라 지수적으로 늘고, 따라서 $h = O(\log n)$이다.

---

**연습문제 2.**
구조를 다시 짜야 하는(회전, 색 바꾸기, 쪼개기·합치기) 트리에서 균형 인수를 따라가라. 앞뒤의 상태를 보여라.

??? success "연습문제 2 풀이"
    이 쪽에서 설명한 재구성 상황을 일으키는 트리를 하나 만들어라. 어긋난 곳을 보이고, 어느 경우에 해당하는지 가리고, 고친 뒤, 불변식이 되살아났는지 확인하라.

---

**연습문제 3.**
균형 인수이(가) 구조를 다시 짜는 연산을 많아야 $O(\log n)$번 필요로 함을 증명하라.

??? success "연습문제 3 풀이"
    구조를 다시 짤 때마다 어긋난 곳이 뿌리에 한 층 가까워지거나 해소된다. 트리의 층이 $O(\log n)$개이므로 재구성은 많아야 $O(\log n)$번 필요하다. 레드-블랙 삽입 같은 연산에서는 회전 2번과 색 바꾸기 $O(\log n)$번이면 충분하다. $\square$

---

**연습문제 4.**
최악의 높이, 연산마다의 회전 횟수, 구현의 까다로움 면에서 균형 인수를 다른 균형 트리 구조와 견주어라.

??? success "연습문제 4 풀이"
    AVL은 높이가 $1.44\log n$ 이하이고 삭제마다 회전이 $O(\log n)$번까지 든다. 레드-블랙은 높이가 $2\log n$ 이하이고 연산마다 회전이 많아야 3번이다. B-트리는 높이가 $O(\log_B n)$이며 디스크 입출력에 맞추어져 있다. 스플레이 트리는 분할 상환으로 $O(\log n)$이지만 최악은 $O(n)$이다.