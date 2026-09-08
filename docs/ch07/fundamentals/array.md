# 배열 표현

이진 트리가 완전하거나 거의 완전할 때는 노드를 단순한 배열에 담으면 자식 포인터의 부담을 아예 없앨 수 있다. 참조로 이어진 노드 객체를 따로 잡는 대신, 트리를 층별로 이어진 메모리에 늘어놓는다. 이 방식은 이진 힙 자료 구조의 바탕이며, 부모와 자식 사이를 오가는 일이 배열 색인의 산술로 줄어들어 캐시 성능도 뛰어나다.

---

## 1. 색인 공식

배열 표현의 핵심 착상은 트리의 자리와 배열의 색인 사이의 일대일 대응이다. 길이가 $n$인 배열 $A$에 담긴 이진 트리를 생각해 보자.

### 0부터 세는 색인

뿌리가 색인 $0$에 있을 때 색인 $i$인 노드의 관계는 다음과 같다.

$$
\text{left child}(i) = 2i + 1
$$

$$
\text{right child}(i) = 2i + 2
$$

$$
\text{parent}(i) = \left\lfloor \frac{i - 1}{2} \right\rfloor \quad \text{for } i > 0
$$

자식의 색인은 배열의 전체 원소 수인 $n$보다 작을 때만 쓸 수 있다.

### 1부터 세는 색인

뿌리가 색인 $1$에 있을 때(교과서와 힙 구현에서 흔한 방식) 공식이 간단해진다.

$$
\text{left child}(i) = 2i
$$

$$
\text{right child}(i) = 2i + 1
$$

$$
\text{parent}(i) = \left\lfloor \frac{i}{2} \right\rfloor \quad \text{for } i > 1
$$

1부터 세는 방식은 비트 이동으로 보는 해석을 뚜렷하게 해 준다. 2를 곱하는 것은 왼쪽 이동이고 2로 나누는 것은 오른쪽 이동이다.

---

## 2. 왜 통하는가

위의 공식은 간단히 세어 보면 나온다. 깊이 $d$에서 완전 이진 트리는 노드를 꼭 $2^d$개 갖는다. 깊이 $d$의 노드는 배열의 자리 $2^d - 1$부터 $2^{d+1} - 2$까지를 차지한다(0부터 셀 때). 층 안에서 깊이 $d$의 $k$번째 노드의 왼쪽 자식은 깊이 $d+1$의 $(2k)$번째 노드이다. 이 층 안의 자리를 배열 전체의 색인으로 옮기면 위의 공식이 나온다.

!!! tip "배열 표현을 쓸 때"
    배열 배치는 트리가 **완전**하거나 **거의 완전**할 때, 곧 마지막 층만 왼쪽부터 차 있고 나머지 층이 모두 꽉 차 있을 때 잘 통한다. 성기거나 크게 치우친 트리에서는 없는 노드도 색인 자리를 차지하므로 공간이 낭비된다. 그럴 때는 [연결 표현](linked.md)이 낫다.

---

## 3. 공간 견주기

| 표현 | 노드당 공간 | 이동 비용 |
|---|---|---|
| 연결 (포인터) | 데이터와 포인터 2개 | 포인터 따라가기 |
| 배열 | 데이터만 | 색인 산술 |

노드가 $n$개인 완전 이진 트리에서 배열 표현은 포인터 부담 없이 $\Theta(n)$의 공간을 쓰지만, 연결 표현은 $\Theta(n)$의 공간에 포인터 $2n$개가 더 든다.

---

## 4. 예

값 `[1, 2, 3, 4, 5, 6, 7]`을 층별로 담은 노드 7개짜리 완전 이진 트리를 생각해 보자.

```
         1          depth 0, index 0
        / \
       2   3        depth 1, indices 1-2
      / \ / \
     4  5 6  7      depth 2, indices 3-6
```

배열은 이를 `A = [1, 2, 3, 4, 5, 6, 7]`으로 담는다. 색인 1(값 2)인 노드의 자식을 찾으려면 왼쪽 자식은 $2(1)+1 = 3$(값 4)에, 오른쪽 자식은 $2(1)+2 = 4$(값 5)에 있다. 색인 5(값 6)인 노드의 부모는 $\lfloor(5-1)/2\rfloor = 2$(값 3)에 있다.

```python
"""
이진 트리의 배열 표현.

완전 이진 트리가 평평한 배열에 어떻게 대응하는지, 그리고 부모와 자식의
관계가 어떻게 색인 산술로 줄어드는지 보인다.
"""

# === 색인으로 오가기 (0부터 셈) ===

def left_child(i: int) -> int:
    """노드 i의 왼쪽 자식의 색인을 돌려준다."""
    return 2 * i + 1

def right_child(i: int) -> int:
    """노드 i의 오른쪽 자식의 색인을 돌려준다."""
    return 2 * i + 2

def parent(i: int) -> int:
    """노드 i의 부모의 색인을 돌려준다 (뿌리에서는 정의되지 않는다)."""
    return (i - 1) // 2

# === 배열 위에서의 트리 연산 ===

def get_children(tree: list, i: int) -> list:
    """노드 i에 실제로 있는 자식들의 값을 돌려준다."""
    children = []
    l, r = left_child(i), right_child(i)
    if l < len(tree):
        children.append(tree[l])
    if r < len(tree):
        children.append(tree[r])
    return children

def print_tree_levels(tree: list) -> None:
    """트리를 층별로 출력한다."""
    if not tree:
        print("Empty tree")
        return
    level = 0
    i = 0
    while i < len(tree):
        level_size = 2 ** level
        level_nodes = tree[i : i + level_size]
        print(f"  Depth {level}: {level_nodes}")
        i += level_size
        level += 1

# === 메인 ===

if __name__ == "__main__":
    tree = [1, 2, 3, 4, 5, 6, 7]

    print("Array representation:", tree)
    print()
    print_tree_levels(tree)
    print()

    for idx in range(len(tree)):
        children = get_children(tree, idx)
        p = parent(idx) if idx > 0 else None
        print(
            f"  Node {tree[idx]} (index {idx}): "
            f"parent={'root' if p is None else tree[p]}, "
            f"children={children}"
        )
```

**출력:**
```
Array representation: [1, 2, 3, 4, 5, 6, 7]

  Depth 0: [1]
  Depth 1: [2, 3]
  Depth 2: [4, 5, 6, 7]

  Node 1 (index 0): parent=root, children=[2, 3]
  Node 2 (index 1): parent=1, children=[4, 5]
  Node 3 (index 2): parent=1, children=[6, 7]
  Node 4 (index 3): parent=2, children=[]
  Node 5 (index 4): parent=2, children=[]
  Node 6 (index 5): parent=3, children=[]
  Node 7 (index 6): parent=3, children=[]
```

---

## 연습문제

**연습문제 1.**
배열 표현에 대해 트리의 높이 $h$으로 시간 복잡도를 밝혀라. $h = O(\log n)$이 되는 것은 언제인가?

??? success "연습문제 1 풀이"
    이 연산은 $O(h)$ 시간이 걸린다. 균형 잡힌 트리에서는 $h = O(\log n)$이다(무작위 삽입 순서에서는 기댓값이 $O(\log n)$이고, 균형 이진 탐색 트리 계열은 이를 보장한다). 한쪽으로 치우친 트리에서는 $h = O(n)$이다(정렬된 순서로 넣은 경우).

---

**연습문제 2.**
열쇠 5, 3, 7, 1, 4, 6, 8을 차례로 넣어 만든 이진 탐색 트리에 배열 표현을(를) 적용하라. 단계마다 보여라.

??? success "연습문제 2 풀이"
    차례로 넣어 트리를 만든 뒤 이 쪽의 연산을 적용한다. 단계마다 트리의 모습을 보이고 들르거나 고친 노드를 표시한다.

---

**연습문제 3.**
이진 탐색 트리의 높이를 $h$이라 할 때 배열 표현이(가) $O(h)$ 시간에 끝남을 증명하라.

??? success "연습문제 3 풀이"
    이 연산은 뿌리에서 잎으로 가는 경로(또는 그 일부)를 따라가며 노드마다 $O(1)$의 일을 한다. 경로의 길이는 많아야 $h$이다. 따라서 모두 $O(h)$이다. $\square$

---

**연습문제 4.**
이진 탐색 트리의 연산은 딥러닝에서 정렬된 데이터를 다루는 일(예를 들어 빔 탐색에서 상위 $k$개의 점수를 유지하는 일)과 어떤 관계가 있는가?

??? success "연습문제 4 풀이"
    빔 탐색은 가장 좋은 가설 $k$개를 유지하므로 새 후보를 넣고 가장 나쁜 것을 빼는 일을 효율적으로 해야 한다. 이진 탐색 트리는 둘 다 $O(\log k)$에 해낸다. 실제로는 더 간단하고 상수가 작은 힙을 즐겨 쓰지만, 이진 탐색 트리는 범위 검색이나 순위 같은 더 풍부한 질의를 지원한다.

## 정리하며

이 마당은 색인 공식、왜 통하는가、공간 견주기、예을 차례로 짚었다.

**참고 문헌**

- [Introduction to Algorithms (CLRS), 6장 — 힙 정렬](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
