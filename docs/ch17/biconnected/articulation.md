# 이음매 점 찾기

주고받기 그물에서 어떤 마디는 결정적이다. 곧 그것이 무너지면 그물이 끊어진 조각으로 갈라진다. 이런 무른 마디를 **이음매 점**(또는 **자름 꼭짓점**)이라 한다. 그것을 가려내는 일은 그물의 튼튼함을 재고, 탈이 나도 견디는 체계를 꾸미고, [두 겹 이음 조각](components.md)의 짜임을 이해하는 데 꼭 필요하다. 깊이 우선 돌아보기 한 번으로 모든 이음매 점을 선형 시간에 찾을 수 있다.

## 정의

!!! note "엄밀한 정의"
    이어진 방향 없는 그래프 $G = (V, E)$의 꼭짓점 $v$을 ($v$에 닿는 모든 변과 함께) 없앴을 때 $G$이 끊어지면 $v$을 **이음매 점**이라 한다. 같은 말로, 그래프 $G - v$의 이어진 조각 수가 $G$의 것보다 많다.

이음매 점이 없는 그래프를 **두 겹 이어졌다**고 한다. 곧 꼭짓점 짝마다 그 사이에 꼭짓점이 겹치지 않는 경로가 적어도 둘 있다.

## 깊이 우선 돌아보기 바탕 알고리즘

이 알고리즘은 [타잔의 강한 이음 조각 알고리즘](../scc/tarjan.md)과 비슷하게 돌아보기의 찾은 시각과 낮은 이음 값을 쓰되 방향 없는 그래프에 쓴다.

꼭짓점 $u$마다 다음을 정한다:

- $\text{disc}[u]$: the discovery time of $u$ in the DFS.
- $\text{low}[u]$: the minimum discovery time reachable from $u$ through the DFS subtree of $u$, including back edges.

$$
\text{low}[u] = \min\!\Big(\text{disc}[u],\ \min_{\substack{v \text{ child of } u}} \text{low}[v],\ \min_{\substack{(u,w) \text{ back edge}}} \text{disc}[w]\Big)
$$

꼭짓점 $u$이 이음매 점인 것은 다음 조건 가운데 하나가 참일 때 그리고 오직 그때뿐이다:

1. **뿌리 조건:** $u$이 돌아보기 나무의 뿌리이고 자식이 둘 이상이다.
2. **Non-root condition:** $u$ is not the root and has a child $v$ such that $\text{low}[v] \geq \text{disc}[u]$.

!!! tip "조건 뒤에 있는 직관"
    뿌리가 아닐 때의 조건은 이렇다. 곧 $v$의 아래나무에 있는 어느 꼭짓점도 뒤로 가는 변으로 $u$의 조상에 닿을 수 없다면, $u$을 없앴을 때 $v$의 아래나무가 나머지 그래프에서 끊어진다. 뿌리 조건은 뿌리에 부모가 없는 특수한 경우를 다룬다. 곧 뿌리는 서로 얽히지 않은 아래나무가 여럿일 때만 이음매 점이다.

## 복잡도

이 알고리즘은 깊이 우선 돌아보기를 한 번만 한다:

$$
T(V, E) = O(V + E)
$$

공간 복잡도는 찾은 시각, 낮은 이음 값, 부모 좇기 때문에 $O(V)$이다.

## 구현

```python
"""
방향 없는 그래프에서 이음매 점(자름 꼭짓점) 찾기.

찾은 시각과 낮은 이음 값을 쓰는 한 번의 깊이 우선 돌아보기로
없애면 그래프가 끊어지는 꼭짓점을 가려낸다.
"""


# === 이음매 점 찾기 ===
def find_articulation_points(graph, n):
    """
    방향 없는 그래프의 이음매 점을 모두 찾는다.

    매개변수
    ----------
    graph : dict[int, list[int]]
        꼭짓점이 0부터 n-1인 방향 없는 그래프의 이웃 목록.
    n : int
        꼭짓점의 개수.

    반환값
    -------
    list[int]
        이음매 점 꼭짓점의 목록.
    """
    disc = [-1] * n
    low = [0] * n
    parent = [-1] * n
    is_ap = [False] * n
    timer = [0]

    def dfs(u):
        disc[u] = low[u] = timer[0]
        timer[0] += 1
        children = 0

        for v in graph.get(u, []):
            if disc[v] == -1:
                children += 1
                parent[v] = u
                dfs(v)
                low[u] = min(low[u], low[v])

                # 자식이 2개 이상인 뿌리
                if parent[u] == -1 and children > 1:
                    is_ap[u] = True

                # 뿌리가 아닐 때: v의 아래나무가 u 위로 닿지 못한다
                if parent[u] != -1 and low[v] >= disc[u]:
                    is_ap[u] = True

            elif v != parent[u]:
                low[u] = min(low[u], disc[v])

    for u in range(n):
        if disc[u] == -1:
            dfs(u)

    return [u for u in range(n) if is_ap[u]]


# === 메인 ===
if __name__ == "__main__":
    # 그래프: 0-1-2-3-4에 1-3 지름길과 따로 떨어진 5-6 다리
    graph = {
        0: [1],
        1: [0, 2, 3],
        2: [1, 3],
        3: [1, 2, 4],
        4: [3, 5],
        5: [4, 6],
        6: [5],
    }
    aps = find_articulation_points(graph, 7)
    print(f"Articulation points: {aps}")
```

**출력:**
```
Articulation points: [3, 4, 5]
```

꼭짓점 3은 없애면 꼭짓점 4(와 그 너머)가 나머지에서 끊어지므로 이음매 점이다. 꼭짓점 4와 5는 [다리](bridges.md)의 사슬을 이루므로 이음매 점이다. 곧 어느 쪽을 없애도 그래프가 갈라진다.

## 풀이 예제

Consider the graph with edges: $\{0\text{-}1,\ 1\text{-}2,\ 2\text{-}3,\ 3\text{-}1,\ 3\text{-}4,\ 4\text{-}5,\ 5\text{-}6\}$.

| 꼭짓점 | disc | low | 부모 | 자식 | 이음매 점? |
|---|---|---|---|---|---|
| 0 | 0 | 0 | 없음 | 1 | 아니다(자식 1개인 뿌리) |
| 1 | 1 | 1 | 0 | 2 | 아니다(low[2]=1, low[3]=1, 둘 다 1에 닿음) |
| 2 | 2 | 1 | 1 | 1 | 아니다(low[3]=1 < disc[2]=2) |
| 3 | 3 | 1 | 2 | 1 | 그렇다(low[4]=4 >= disc[3]=3) |
| 4 | 4 | 4 | 3 | 1 | 그렇다(low[5]=5 >= disc[4]=4) |
| 5 | 5 | 5 | 4 | 1 | 그렇다(low[6]=6 >= disc[5]=5) |
| 6 | 6 | 6 | 5 | 0 | 아니다(잎) |

Articulation points: $\{3, 4, 5\}$.

## 다리 및 두 겹 이음 조각과의 관계

- 두 끝점이 모두 이음매 점인 변은 흔히 (늘 그렇지는 않지만) [다리](bridges.md)이다.
- 다리마다 적어도 한 끝점이 이음매 점이다(그 다리가 외딴 꼭짓점 둘을 잇는 경우는 빼고).
- 이음매 점을 모두 없애면 그래프가 [두 겹 이음 조각](components.md)으로 쪼개진다.
- [덩이-자름 나무](block_cut.md)는 이음매 점이 두 겹 이음 조각을 어떻게 잇는지 나무로 나타낸다.

## 참고 문헌

- Hopcroft, J. E., & Tarjan, R. E. (1973). Algorithm 447: efficient algorithms for graph manipulation. *Communications of the ACM*, 16(6), 372-378.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. 20장.

## 연습문제

**연습문제 1.**
이음매 점을 정의하고 그물 믿음성에서 갖는 뜻을 설명하여라.

??? success "연습문제 1 풀이"
    An **articulation point** (cut vertex) is a vertex whose removal disconnects the graph. In a network, articulation points are single points of failure: if a router, server, or link at that point fails, parts of the network become unreachable. Identifying articulation points helps design redundant networks. They can be found in $O(V + E)$ using DFS with low-link values. $\square$

---

**연습문제 2.**
이음매 점을 찾는 깊이 우선 돌아보기 바탕 알고리즘을 설명하여라. 그것을 가려내는 두 조건은 무엇인가?

??? success "연습문제 2 풀이"
    Run DFS, computing discovery time $\text{disc}[v]$ and low value $\text{low}[v]$ for each vertex. Vertex $u$ is an articulation point if: (1) $u$ is the DFS root and has $\geq 2$ children in the DFS tree, or (2) $u$ is not the root and has a child $v$ with $\text{low}[v] \geq \text{disc}[u]$ (no vertex in $v$'s subtree can reach above $u$ via a back edge). Condition (2) means removing $u$ would disconnect $v$'s subtree from the rest of the graph. $\square$

---

**연습문제 3.**
깊이 우선 돌아보기 나무의 뿌리가 이음매 점인 것은 자식이 둘 이상일 때 그리고 오직 그때뿐임을 증명하여라.

??? success "연습문제 3 풀이"
    $(\Rightarrow)$ If the root has $\geq 2$ children, removing it disconnects their subtrees (in the DFS tree, all edges between subtrees pass through the root, and there are no cross edges in undirected DFS). $(\Leftarrow)$ If the root has exactly one child, removing it leaves one connected subtree (all vertices are in that subtree, connected via tree and back edges). With zero children, the root is isolated, and its removal doesn't disconnect anything. $\square$

---

**연습문제 4.**
그래프에 꼭짓점이 8개이고 2-꼭짓점-이어짐이다(이음매 점이 없다). 변은 적어도 몇 개여야 하는가?

??? success "연습문제 4 풀이"
    A 2-vertex-connected graph on $n$ vertices must have at least $n$ edges. This is because it must contain a cycle through every vertex (otherwise, a leaf would be an articulation point of its neighbor). With $n = 8$, a single Hamiltonian cycle provides exactly 8 edges and is 2-connected. Therefore the minimum is 8 edges. $\square$
