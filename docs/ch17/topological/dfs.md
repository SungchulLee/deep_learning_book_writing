# 깊이 우선 돌아보기 바탕 위상 정렬

깊이 우선 돌아보기는 유향 비순환 그래프에서 꼭짓점의 자연스러운 차례를 드러낸다. 돌아보기가 한 꼭짓점 다루기를 마치면, 곧 그 자손을 모두 살펴보고 나면, 그 꼭짓점은 자기가 달린 모든 꼭짓점 뒤에 안전하게 올 수 있다. 마침 시각의 거꿀 차례로 꼭짓점을 적어 두면 들어오는 차수를 따로 세지 않고도 올바른 위상 차례를 얻는다.

## 핵심 생각

깊이 우선 돌아보기 바탕 위상 정렬은 유향 비순환 그래프 위 돌아보기의 핵심 성질을 써먹는다. 곧 변 $(u, v)$이 있으면 $v$이 $u$보다 먼저 끝난다. 돌아보기가 $u$으로 돌아오기 앞서 $v$과 그 자손을 모두 살펴보기 때문이다. 그러므로 마침 시각 차례를 뒤집으면 $u$이 $v$ 앞에 놓여 위상 차례 제약을 채운다.

!!! tip "마침 시각 성질"
    In a DFS of a DAG $G = (V, E)$, for every edge $(u, v) \in E$, vertex $u$ has a later finish time than $v$. Sorting vertices by decreasing finish time yields a topological ordering.

## 알고리즘

이 알고리즘은 보통의 깊이 우선 돌아보기를 하면서 꼭짓점이 끝날 때마다(이웃을 모두 살펴보고 나면) 그것을 목록에 덧붙인다. 마지막 목록을 뒤집으면 위상 차례가 된다.

**단계:**

1. 모든 꼭짓점을 들르지 않음으로 첫자리매김한다.
2. 들르지 않은 꼭짓점 $u$마다 $u$에서 깊이 우선 돌아보기를 한다.
3. 돌아보기에서 $u$의 이웃을 모두 살펴본 뒤 $u$을 쌓기(또는 목록)에 덧붙인다.
4. 이 목록을 뒤집은 것이 위상 차례이다.

## 올바름

!!! note "거꿀 마침 차례가 되는 까닭"
    유향 비순환 그래프의 아무 변 $(u, v)$을 보자. 돌아보기가 변 $(u, v)$을 다룰 때 꼭짓점 $v$은 다음 가운데 하나이다:

    - **들르지 않음:** 돌아보기가 $v$으로 되돌이해 들어가므로 $v$이 $u$보다 먼저 끝난다.
    - **다 다룸:** $v$이 이미 끝났으므로 $v$의 마침 시각이 $u$의 것보다 이르다.
    - **지금 다루는 중(회색):** 이는 $v$이 $u$의 조상이라는 뜻이어서 뒤로 가는 변, 곧 순환을 만든다. 그러나 $G$이 유향 비순환 그래프이므로 이 경우는 생길 수 없다.

    가능한 모든 경우에 $v$이 $u$보다 먼저 끝나므로 거꿀 마침 시각 차례에서 $u$이 $v$ 앞에 옴이 보장된다.

## 복잡도

이 알고리즘은 꼭짓점마다 정확히 한 번 들르고 변마다 정확히 한 번 지나므로:

$$
T(V, E) = O(V + E)
$$

공간 복잡도는 되돌이 쌓기와 색/들름 배열 때문에 $O(V)$이다.

## 구현

```python
"""
깊이 우선 돌아보기 바탕 위상 정렬.

거꿀 마침 시각 차례로 꼭짓점을 적어 올바른
유향 비순환 그래프의 위상 차례를 내놓는다.
"""


# === 깊이 우선 돌아보기 위상 정렬 ===
def topo_sort_dfs(graph, n):
    """
    깊이 우선 돌아보기로 위상 차례를 셈한다.

    매개변수
    ----------
    graph : dict[int, list[int]]
        꼭짓점 이름이 0부터 n-1인 유향 비순환 그래프의 이웃 목록.
    n : int
        꼭짓점의 개수.

    반환값
    -------
    list[int]
        위상 차례의 꼭짓점들. 순환이 있으면 빈 목록
        알아낸다.
    """
    WHITE, GRAY, BLACK = 0, 1, 2
    color = [WHITE] * n
    order = []
    has_cycle = False

    def dfs(u):
        nonlocal has_cycle
        color[u] = GRAY
        for v in graph.get(u, []):
            if color[v] == GRAY:
                has_cycle = True
                return
            if color[v] == WHITE:
                dfs(v)
                if has_cycle:
                    return
        color[u] = BLACK
        order.append(u)  # 마침 시각 적기

    for u in range(n):
        if color[u] == WHITE:
            dfs(u)
            if has_cycle:
                return []

    order.reverse()
    return order


# === 메인 ===
if __name__ == "__main__":
    # 유향 비순환 그래프: 0 -> 1 -> 3, 0 -> 2 -> 3 -> 4
    dag = {0: [1, 2], 1: [3], 2: [3], 3: [4], 4: []}
    result = topo_sort_dfs(dag, 5)
    print(f"Topological order: {result}")

    # 확인: 변 (u, v)마다 u가 v 앞에 오는지
    pos = {v: i for i, v in enumerate(result)}
    valid = all(pos[u] < pos[v] for u in dag for v in dag[u])
    print(f"Valid topological order: {valid}")
```

**출력:**
```
Topological order: [0, 2, 1, 3, 4]
Valid topological order: True
```

세 가지 색칠 방식(흰색, 회색, 검은색)은 순환도 한꺼번에 찾아낸다. 회색 꼭짓점은 지금 되돌이 쌓기에 있는 것이며, 그것을 다시 만나면 뒤로 가는 변이 있다는 뜻이고 곧 순환이 있다는 뜻이다. 그래서 이 알고리즘은 위상 정렬기이자 [유향 비순환 그래프 확인기](dag.md) 노릇을 함께 한다.

## 되풀이 변종

되돌이 깊이가 부름 쌓기 한계를 넘을 수 있는 그래프에서는 드러난 쌓기를 쓰는 되풀이 판이 쌓기 넘침을 막는다:

```python
"""
되풀이 깊이 우선 돌아보기 바탕 위상 정렬.

큰 그래프에서 되돌이 깊이 한계를 피하려 드러난 쌓기를 쓴다.
"""


# === 되풀이 깊이 우선 돌아보기 위상 정렬 ===
def topo_sort_dfs_iterative(graph, n):
    """
    되풀이 깊이 우선 돌아보기로 위상 차례를 셈한다.

    매개변수
    ----------
    graph : dict[int, list[int]]
        꼭짓점이 0부터 n-1인 유향 비순환 그래프의 이웃 목록.
    n : int
        꼭짓점의 개수.

    반환값
    -------
    list[int]
        위상 차례로 늘어놓은 꼭짓점.
    """
    visited = [False] * n
    order = []

    for start in range(n):
        if visited[start]:
            continue
        stack = [(start, 0)]
        visited[start] = True
        while stack:
            u, idx = stack.pop()
            neighbors = graph.get(u, [])
            if idx < len(neighbors):
                stack.append((u, idx + 1))
                v = neighbors[idx]
                if not visited[v]:
                    visited[v] = True
                    stack.append((v, 0))
            else:
                order.append(u)

    order.reverse()
    return order


# === 메인 ===
if __name__ == "__main__":
    dag = {0: [1, 2], 1: [3], 2: [3], 3: [4], 4: []}
    print(f"Iterative topological order: {topo_sort_dfs_iterative(dag, 5)}")
```

**출력:**
```
Iterative topological order: [0, 2, 1, 3, 4]
```

## 칸 알고리즘과의 견줌

깊이 우선 돌아보기 바탕 정렬과 [칸 알고리즘](kahn.md)은 둘 다 $O(V + E)$ 시간에 돌지만 방식이 다르다:

| 성질 | 깊이 우선 돌아보기 바탕 | 칸 알고리즘 |
|---|---|---|
| 전략 | 거꿀 마침 시각 차례 | 근원을 되풀이해 없애기 |
| 자료 짜임새 | 되돌이 쌓기(또는 드러난 쌓기) | 들어오는 차수가 0인 꼭짓점의 줄서기 |
| 순환 찾기 | 돌아보기 도중의 뒤로 가는 변 | 다루지 못한 꼭짓점이 남음 |
| 내놓는 차례 | 흔히 넣은 차례의 거꿀 | 너비 우선 돌아보기 비슷한 차례를 따르는 편 |
| 나란함 | 나란히 하기 더 어렵다 | 근원을 나란히 다룰 수 있다 |

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. 20장.

## 연습문제

**연습문제 1.**
깊이 우선 돌아보기 바탕 위상 정렬 알고리즘을 설명하고 그 옳음을 증명하여라.

??? success "연습문제 1 풀이"
    Run DFS on the entire graph. When a vertex finishes (all descendants explored), prepend it to the output list. The result is a reverse post-order. **Correctness**: for any edge $(u, v)$ in a DAG, $v$ finishes before $u$ (since $v$ is either a descendant of $u$, or was already finished when $(u,v)$ was examined). Therefore $u$ appears before $v$ in the output list, satisfying the topological order condition. Time: $O(V + E)$. $\square$

---

**연습문제 2.**
순환이 있는 그래프에서 깊이 우선 돌아보기 바탕 위상 정렬을 돌리면 어떻게 되는가?

??? success "연습문제 2 풀이"
    The algorithm still produces an ordering, but it is not a valid topological sort (since no valid ordering exists for a graph with cycles). To detect this, check for back edges during DFS: if an edge $(u, v)$ is found where $v$ is gray (on the current recursion stack), a cycle exists. The algorithm should report an error. Without this check, the output may appear valid but will violate the ordering requirement for at least one edge. $\square$

---

**연습문제 3.**
위상 정렬에서 깊이 우선 돌아보기 바탕 방식과 칸 알고리즘을 견주어라.

??? success "연습문제 3 풀이"
    **깊이 우선 돌아보기 바탕**: 순환(뒤로 가는 변)을 자연스레 찾아낸다. 거꿀 후위 차례를 내놓는다. 되돌이 쌓기를 쓴다($O(V)$ 공간). 이미 있는 돌아보기 얼거리로 짜기 쉽다.

    **칸**: 들어오는 차수가 0인 꼭짓점을 되풀이해 없앤다. 모든 꼭짓점을 다루지 못하면 순환이 있음을 알아챈다. 드러난 줄서기/목록과 들어오는 차수 배열을 쓴다. "켜별" 짜임을 이해하기에 더 직관적이다.

    Both run in $O(V + E)$. DFS is preferred when DFS is already being used for other purposes. Kahn's is preferred for level-by-level processing. $\square$

---

**연습문제 4.**
깊이 우선 돌아보기 바탕 위상 정렬은 시작 꼭짓점과 이웃을 다루는 차례에 따라 다른 차례를 내놓을 수 있는가?

??? success "연습문제 4 풀이"
    Yes. The DFS-based sort depends on: (1) the order in which source vertices are chosen, and (2) the order in which neighbors are explored. Different choices lead to different valid topological orderings. For example, in a DAG with $A \to C, B \to C$: starting DFS from $A$ first gives $[A, B, C]$ or $[B, A, C]$ depending on when $B$ is processed. All outputs are valid topological orderings, but they may differ. $\square$
