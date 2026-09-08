# 오일러 투어

표준 트리 순회인 전위·중위·후위 순회는 저마다 모든 노드를 꼭 한 번씩 들른다. **오일러 투어**는 다르게 접근한다. 트리의 윤곽을 따라 그리며 부분 트리에 들어가고 나올 때마다 노드를 여러 번 들른다. 그러면 트리 전체의 구조를 담은 선형 나열이 나오고, 최소 공통 조상이나 부분 트리 합 같은 트리 질의를 배열의 구간 질의로 바꿀 수 있다. 그런 질의는 $O(n)$의 전처리 뒤에 $O(1)$ 시간에 답할 수 있다.

---

## 1. 정의

노드가 $n$개인 뿌리 있는 트리의 오일러 투어는 깊이 우선 탐색을 하면서 노드마다 **들어갈** 때와 (모든 자식에서 돌아와) **나올** 때를 모두 적어 길이가 $2n$인 나열을 만든다.

엄밀히는 뿌리에 다음 재귀 절차를 적용하여 $\text{ET}[0 \ldots 2n-1]$을 정의한다.

$$
\text{EulerTour}(v): \quad \text{record } v \;\text{(enter)}, \quad \text{recurse on each child of } v, \quad \text{record } v \;\text{(exit)}
$$

이렇게 만든 나열에는 노드마다 꼭 두 번씩 나온다. 한 번은 **들어간 시각** $\text{tin}(v)$에, 한 번은 **나온 시각** $\text{tout}(v)$에 나온다.

??? example "예제 트리의 오일러 투어"
    다음 트리를 생각해 보자.

    ```
           A
          / \
         B   C
        / \   \
       D   E   F
    ```

    오일러 투어는 다음 순서로 들른다: **A** B D D E E B **C** F F C **A**

    | 단계 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
    |------|---|---|---|---|---|---|---|---|---|---|----|----|
    | 노드 | A | B | D | D | E | E | B | C | F | F | C  | A  |
    | 종류 | 들어감| 들어감| 들어감|나옴| 들어감|나옴|나옴| 들어감| 들어감|나옴|나옴 |나옴 |

    들어간/나온 시각: $\text{tin}(A)=0, \text{tout}(A)=11$; $\text{tin}(B)=1, \text{tout}(B)=6$; $\text{tin}(D)=2, \text{tout}(D)=3$.

---

## 2. 핵심 성질

들어간 시각과 나온 시각은 트리의 구조를 쓸모 있게 담고 있다.

**조상 판정.** 노드 $u$이 노드 $v$의 조상인 것은 다음과 동치이다.

$$
\text{tin}(u) \leq \text{tin}(v) \leq \text{tout}(v) \leq \text{tout}(u)
$$

투어를 만들고 나면 $O(1)$에 확인할 수 있다.

**부분 트리 구간.** $v$을 뿌리로 하는 부분 트리는 오일러 투어에서 이어진 구간 $[\text{tin}(v), \text{tout}(v)]$에 대응한다. 합, 최솟값, 최댓값, 개수 같은 부분 트리 질의는 모두 이 구간의 구간 질의로 바뀐다.

**최소 공통 조상으로의 환원.** 노드 $u$과 $v$의 최소 공통 조상은 오일러 투어에서 자리 $\text{tin}(u)$과 $\text{tin}(v)$ 사이에 있는 노드 가운데 깊이가 가장 작은 것이다. 이로써 최소 공통 조상 문제가 구간 최솟값 질의(RMQ)로 바뀌고, 성긴 표로 $O(n)$의 전처리를 하면 질의마다 $O(1)$에 풀 수 있다.

---

## 3. 변형

### 온전한 오일러 투어 (항목 2n개)

노드마다 꼭 두 번씩 나온다. $v$의 부분 트리를 구간 $[\text{tin}(v), \text{tout}(v)]$에 대응시켜 부분 트리 질의에 쓴다.

### 최소 공통 조상용 오일러 투어 (항목 2n - 1개)

변을 지날 때마다(내려갈 때와 올라올 때 모두) 노드를 적고, 자식에서 돌아올 때 부모도 적는다. 그러면 항목이 $2n - 1$개가 되고, $u$과 $v$의 최소 공통 조상은 둘이 처음 나온 자리 사이에서 가장 얕은 노드이다.

### 납작한 오일러 투어 (항목 n개, 들어갈 때만)

노드를 들어갈 때만 적는다. 그러면 나열이 전위 순회가 된다. 부분 트리의 크기를 함께 쓰면 $v$의 부분 트리를 구간 $[\text{tin}(v), \text{tin}(v) + \text{size}(v) - 1]$에 대응시켜 여전히 부분 트리 질의에 답할 수 있다.

---

## 4. 구현

```python
"""들어간 시각과 나온 시각이 있는 뿌리 있는 트리의 오일러 투어."""

# === 트리 노드 ===

class TreeNode:
    """뿌리 있는 트리의 노드."""

    def __init__(self, val: int):
        self.val = val
        self.children: list["TreeNode"] = []

# === 오일러 투어 ===

def euler_tour(root: TreeNode) -> tuple[list[int], dict[int, int], dict[int, int]]:
    """
    뿌리 있는 트리의 오일러 투어를 계산한다.

    반환값:
        tour: 오일러 투어 순서의 노드 값 목록 (길이 2n)
        tin:  노드 값을 들어간 시각으로 잇는 사전
        tout: 노드 값을 나온 시각으로 잇는 사전
    """
    tour: list[int] = []
    tin: dict[int, int] = {}
    tout: dict[int, int] = {}
    timer = [0]

    def dfs(node: TreeNode) -> None:
        tin[node.val] = timer[0]
        tour.append(node.val)
        timer[0] += 1
        for child in node.children:
            dfs(child)
        tout[node.val] = timer[0]
        tour.append(node.val)
        timer[0] += 1

    dfs(root)
    return tour, tin, tout

def is_ancestor(u: int, v: int, tin: dict[int, int], tout: dict[int, int]) -> bool:
    """들어간 시각과 나온 시각으로 u가 v의 조상인지 O(1)에 확인한다."""
    return tin[u] <= tin[v] and tout[v] <= tout[u]

# === 시연 ===

if __name__ == "__main__":
    # 예제 트리 만들기:  A(0) -> B(1), C(2);  B -> D(3), E(4);  C -> F(5)
    nodes = [TreeNode(i) for i in range(6)]
    nodes[0].children = [nodes[1], nodes[2]]  # A -> B, C
    nodes[1].children = [nodes[3], nodes[4]]  # B -> D, E
    nodes[2].children = [nodes[5]]            # C -> F

    tour, tin, tout = euler_tour(nodes[0])
    labels = "ABCDEF"

    print("Euler tour:", " ".join(labels[v] for v in tour))
    print()
    for i, label in enumerate(labels):
        print(f"  {label}: tin={tin[i]}, tout={tout[i]}")

    print()
    print(f"Is A ancestor of D? {is_ancestor(0, 3, tin, tout)}")  # True
    print(f"Is B ancestor of F? {is_ancestor(1, 5, tin, tout)}")  # False
    print(f"Is C ancestor of F? {is_ancestor(2, 5, tin, tout)}")  # True
```

---

## 5. 응용

| 응용 | 환원 | 질의 시간 |
|---|---|---|
| 부분 트리 합/최솟값/최댓값 | $[\text{tin}(v), \text{tout}(v)]$의 구간 질의 | 성긴 표로 $O(1)$ |
| 부분 트리 갱신 | $[\text{tin}(v), \text{tout}(v)]$의 구간 갱신 | 펜윅 트리로 $O(\log n)$ |
| 최소 공통 조상 | 깊이에 대한 구간 최솟값 질의 | $O(n)$ 전처리 뒤 $O(1)$ |
| 조상 판정 | 들어간/나온 시각 견주기 | $O(1)$ |
| 부분 트리 크기 | $\text{tout}(v) - \text{tin}(v) + 1)/2$ 또는 직접 세기 | $O(1)$ |

---

## 6. 복잡도

오일러 투어를 만들려면 깊이 우선 탐색을 한 번 하면 된다.

- **시간:** $O(n)$
- **공간:** 투어 배열과 시각 표시에 $O(n)$

---

## 연습문제

**연습문제 1.**
오일러 투어에 대해 트리의 높이 $h$으로 시간 복잡도를 밝혀라. $h = O(\log n)$이 되는 것은 언제인가?

??? success "연습문제 1 풀이"
    이 연산은 $O(h)$ 시간이 걸린다. 균형 잡힌 트리에서는 $h = O(\log n)$이다(무작위 삽입 순서에서는 기댓값이 $O(\log n)$이고, 균형 이진 탐색 트리 계열은 이를 보장한다). 한쪽으로 치우친 트리에서는 $h = O(n)$이다(정렬된 순서로 넣은 경우).

---

**연습문제 2.**
열쇠 5, 3, 7, 1, 4, 6, 8을 차례로 넣어 만든 이진 탐색 트리에 오일러 투어을(를) 적용하라. 단계마다 보여라.

??? success "연습문제 2 풀이"
    차례로 넣어 트리를 만든 뒤 이 쪽의 연산을 적용한다. 단계마다 트리의 모습을 보이고 들르거나 고친 노드를 표시한다.

---

**연습문제 3.**
이진 탐색 트리의 높이를 $h$이라 할 때 오일러 투어이(가) $O(h)$ 시간에 끝남을 증명하라.

??? success "연습문제 3 풀이"
    이 연산은 뿌리에서 잎으로 가는 경로(또는 그 일부)를 따라가며 노드마다 $O(1)$의 일을 한다. 경로의 길이는 많아야 $h$이다. 따라서 모두 $O(h)$이다. $\square$

---

**연습문제 4.**
이진 탐색 트리의 연산은 딥러닝에서 정렬된 데이터를 다루는 일(예를 들어 빔 탐색에서 상위 $k$개의 점수를 유지하는 일)과 어떤 관계가 있는가?

??? success "연습문제 4 풀이"
    빔 탐색은 가장 좋은 가설 $k$개를 유지하므로 새 후보를 넣고 가장 나쁜 것을 빼는 일을 효율적으로 해야 한다. 이진 탐색 트리는 둘 다 $O(\log k)$에 해낸다. 실제로는 더 간단하고 상수가 작은 힙을 즐겨 쓰지만, 이진 탐색 트리는 범위 검색이나 순위 같은 더 풍부한 질의를 지원한다.

## 정리하며

이 마당은 정의、핵심 성질、변형、구현을 차례로 짚었다.

**참고 문헌**

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 12. MIT Press.
- Bender, M. A., & Farach-Colton, M. (2000). The LCA problem revisited. *Proceedings of LATIN 2000*, 88--94.
