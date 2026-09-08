# 왼쪽으로 기운 레드-블랙 트리

표준 [레드-블랙 트리](properties.md)는 노드의 양쪽 어디에나 붉은 이음을 허용하므로 [삽입](insert_fixup.md)과 [삭제](delete_fixup.md)에서 다룰 경우가 많아진다. 세지윅의 **왼쪽으로 기운 레드-블랙(LLRB) 트리**는 붉은 이음이 왼쪽으로 기운다는 불변식 하나를 더해 경우의 수를 절반쯤 줄인다. 그러면서도 최악의 경우 $O(\log n)$이라는 보장은 그대로이고 코드는 훨씬 간단해진다.

---

## 1. 왼쪽으로 기운다는 불변식

LLRB 트리는 규칙 하나가 더 있는 레드-블랙 트리이다.

- **붉은 왼쪽 자식 없이 붉은 오른쪽 자식만 갖는 노드는 없다.**

같은 말로, 어떤 노드에 붉은 자식이 꼭 하나 있다면 그것은 왼쪽 자식이어야 한다. 그러면 표준 레드-블랙 트리의 연산을 복잡하게 만드는 대칭적인 경우가 사라진다.

---

## 2. 2-3 트리와의 대응

LLRB 트리는 **2-3 트리**를 이진으로 나타낸 것이다. 2-3 트리의 노드 종류마다 정해진 LLRB 모양에 대응한다.

| 2-3 트리의 노드 | LLRB에서의 모습 |
|---------------|---------------------|
| 2-노드 (열쇠 1개, 자식 2개) | 검은 자식 둘을 가진 검은 노드 |
| 3-노드 (열쇠 2개, 자식 3개) | 붉은 왼쪽 자식을 가진 검은 노드 |

3-노드의 둘째 열쇠는 언제나 첫 열쇠의 붉은 왼쪽 자식이 되므로 구조가 하나로 정해진다. 2-3 트리마다 LLRB 트리가 꼭 하나씩 있다.

!!! note "왼쪽으로 기울면 코드가 간단해지는 까닭"
    표준 레드-블랙 트리는 2-3-4 트리를 나타내며, 4-노드는 붉은 자식을 둘 가질 수 있다. LLRB 트리는 2-3 트리만 나타내므로 4-노드가 아예 없다. 여기에 왼쪽으로 기운다는 제약이 더해져 연산마다 다룰 경우가 줄어든다.

---

## 3. 핵심 연산

### 회전과 색 뒤집기

국소적인 변환 세 가지가 LLRB의 불변식을 지킨다.

**왼쪽 회전:** 오른쪽으로 기운 붉은 이음을 왼쪽으로 기울게 바꾼다.

**오른쪽 회전:** 오른쪽으로 기운 붉은 이음을 잠시 만든다(삽입 중에 왼쪽으로 기운 붉은 이음이 잇따르는 것을 고칠 때 쓴다).

**색 뒤집기:** 두 자식이 모두 붉으면(잠시 생긴 4-노드) 세 색을 모두 뒤집는다. 부모가 붉어지고 두 자식이 검어진다. 그러면 4-노드가 쪼개지고 가운데 열쇠가 위로 올라간다.

### 삽입

LLRB 삽입은 간단한 재귀 방식을 따른다.

1. 새 열쇠를 잎 층에 **붉은** 노드로 넣는다(표준 이진 탐색 트리 삽입).
2. 재귀에서 되돌아 올라오며 다음을 한다.
      - 오른쪽 자식이 붉고 왼쪽 자식이 검으면 **왼쪽 회전**을 한다.
      - 왼쪽 자식이 붉고 그 왼쪽 자식도 붉으면 **오른쪽 회전**을 한다.
      - 두 자식이 모두 붉으면 **색을 뒤집는다**.

재귀 호출마다 이 세 가지를 차례로 확인하면 모든 LLRB 불변식이 되살아난다.

```python
"""왼쪽으로 기운 적흑 트리 삽입."""

from __future__ import annotations

# === 상수 ===

RED = True
BLACK = False

# === 노드 정의 ===

class Node:
    """색 비트를 지닌 LLRB 트리 노드."""

    def __init__(self, key: int, color: bool = RED):
        self.key = key
        self.left: Node | None = None
        self.right: Node | None = None
        self.color = color

# === 도우미 함수 ===

def is_red(node: Node | None) -> bool:
    """노드가 있고 빨가면 True를 돌려준다."""
    return node is not None and node.color == RED

def rotate_left(h: Node) -> Node:
    """오른쪽으로 기운 빨간 이음을 왼쪽으로 기울게 회전한다."""
    x = h.right
    h.right = x.left
    x.left = h
    x.color = h.color
    h.color = RED
    return x

def rotate_right(h: Node) -> Node:
    """왼쪽으로 기운 빨간 이음을 오른쪽으로 기울게 회전한다(잠깐 동안)."""
    x = h.left
    h.left = x.right
    x.right = h
    x.color = h.color
    h.color = RED
    return x

def flip_colors(h: Node) -> None:
    """색을 뒤집어 임시 4-노드를 쪼갠다."""
    h.color = RED
    h.left.color = BLACK
    h.right.color = BLACK

# === 삽입 ===

def insert(node: Node | None, key: int) -> Node:
    """*node*를 뿌리로 하는 LLRB 부분 트리에 열쇠를 넣는다."""
    if node is None:
        return Node(key, RED)

    if key < node.key:
        node.left = insert(node.left, key)
    elif key > node.key:
        node.right = insert(node.right, key)
    # 겹치는 열쇠는 무시한다

    # 거슬러 올라가며 바로잡는다
    if is_red(node.right) and not is_red(node.left):
        node = rotate_left(node)
    if is_red(node.left) and is_red(node.left.left):
        node = rotate_right(node)
    if is_red(node.left) and is_red(node.right):
        flip_colors(node)

    return node

# === 시연 ===

if __name__ == "__main__":
    root: Node | None = None
    for key in [7, 3, 18, 10, 22, 8, 11, 26]:
        root = insert(root, key)
        root.color = BLACK  # 뿌리는 언제나 검다

    def inorder(node: Node | None) -> list[int]:
        """열쇠를 정렬된 순서로 모은다."""
        if node is None:
            return []
        return inorder(node.left) + [node.key] + inorder(node.right)

    print(f"Inorder: {inorder(root)}")
    # [3, 7, 8, 10, 11, 18, 22, 26]
```

---

## 4. 복잡도

| 연산 | 시간 |
|-----------|------|
| 찾기 | $O(\log n)$ |
| 삽입 | $O(\log n)$ |
| 삭제 | $O(\log n)$ |

LLRB 트리의 높이는 많아야 $2 \log_2(n + 1)$으로 표준 레드-블랙 트리와 같은 한계이다. 실제 상수 배도 비슷하지만 코드는 훨씬 짧다.

---

## 5. LLRB와 표준 레드-블랙 트리

| 항목 | 표준 레드-블랙 | LLRB |
|--------|-------------|------|
| 바탕이 되는 트리 | 2-3-4 트리 | 2-3 트리 |
| 붉은 이음의 방향 | 어느 쪽이든 | 왼쪽만 |
| 삽입의 경우 | 3개와 대칭 3개 | 3개 (대칭 경우 없음) |
| 삭제의 복잡함 | 4개와 대칭 4개 | 더 적은 경우 |
| 구현의 크기 | 약 100줄 | 약 40줄 |

---

## 연습문제

**연습문제 1.**
왼쪽으로 기운 레드-블랙 트리의 균형 불변식을 밝히고 그것이 높이 $O(\log n)$을 보장함을 증명하라.

??? success "연습문제 1 풀이"
    각 구조의 불변식(균형 인수, 색의 성질, 차수 제약)이 경로 길이의 치우침을 묶는다. 높이의 한계는 그 불변식에서 따라 나온다. 트리의 층마다 (불변식이 정하는) 최소한의 노드가 있어야 하므로 전체 노드 수 $n$이 높이에 따라 지수적으로 늘고, 따라서 $h = O(\log n)$이다.

---

**연습문제 2.**
구조를 다시 짜야 하는(회전, 색 바꾸기, 쪼개기·합치기) 트리에서 왼쪽으로 기운 레드-블랙 트리를 따라가라. 앞뒤의 상태를 보여라.

??? success "연습문제 2 풀이"
    이 쪽에서 설명한 재구성 상황을 일으키는 트리를 하나 만들어라. 어긋난 곳을 보이고, 어느 경우에 해당하는지 가리고, 고친 뒤, 불변식이 되살아났는지 확인하라.

---

**연습문제 3.**
왼쪽으로 기운 레드-블랙 트리이(가) 구조를 다시 짜는 연산을 많아야 $O(\log n)$번 필요로 함을 증명하라.

??? success "연습문제 3 풀이"
    구조를 다시 짤 때마다 어긋난 곳이 뿌리에 한 층 가까워지거나 해소된다. 트리의 층이 $O(\log n)$개이므로 재구성은 많아야 $O(\log n)$번 필요하다. 레드-블랙 삽입 같은 연산에서는 회전 2번과 색 바꾸기 $O(\log n)$번이면 충분하다. $\square$

---

**연습문제 4.**
최악의 높이, 연산마다의 회전 횟수, 구현의 까다로움 면에서 왼쪽으로 기운 레드-블랙 트리를 다른 균형 트리 구조와 견주어라.

??? success "연습문제 4 풀이"
    AVL은 높이가 $1.44\log n$ 이하이고 삭제마다 회전이 $O(\log n)$번까지 든다. 레드-블랙은 높이가 $2\log n$ 이하이고 연산마다 회전이 많아야 3번이다. B-트리는 높이가 $O(\log_B n)$이며 디스크 입출력에 맞추어져 있다. 스플레이 트리는 분할 상환으로 $O(\log n)$이지만 최악은 $O(n)$이다.

## 정리하며

이 마당은 왼쪽으로 기운다는 불변식、2-3 트리와의 대응、핵심 연산、복잡도을 차례로 짚었다.

**참고 문헌**

- Sedgewick, R. (2008). Left-leaning red-black trees. *Dagstuhl Workshop on Data Structures*.
- Sedgewick, R., & Wayne, K. (2011). *Algorithms* (4th ed.), Section 3.3. Addison-Wesley.
