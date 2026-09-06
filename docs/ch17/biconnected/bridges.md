# 다리 찾기

[이음매 점](articulation.md)이 없애면 그래프가 끊어지는 꼭짓점이라면, **다리**는 같은 성질을 갖는 변이다. 다리는 결정적인 이음줄 하나를 나타낸다. 곧 그것이 무너지면 그물의 두 부분 사이 주고받기가 끊긴다. 다리 찾기는 이음매 점 찾기와 같은 돌아보기 얼거리를 쓰되 낮은 이음 값에 조금 더 빡빡한 조건을 건다.

## 정의

!!! note "엄밀한 정의"
    이어진 방향 없는 그래프 $G = (V, E)$의 변 $(u, v)$을 없앴을 때 $G$이 끊어지면 그 변을 **다리**라 한다. 같은 말로, $(u, v)$이 다리인 것은 그것이 어떤 순환에도 놓이지 않을 때 그리고 오직 그때뿐이다.

순환 위의 변은 순환이 다른 길을 주므로 없애도 그래프가 끊어지지 않는다. 거꾸로 어떤 순환에도 없는 변은 그 끝점이 딸린 조각 사이의 유일한 길이므로 다리가 된다.

## 깊이 우선 돌아보기 바탕 찾기

다리 찾기 알고리즘은 [이음매 점 알고리즘](articulation.md)을 그대로 본뜨되 부등호를 빡빡하게 쓴다.

돌아보기 나무의 나무 변 $(u, v)$($u$이 $v$의 부모)에 대해:

- $(u, v)$ is a bridge if and only if $\text{low}[v] > \text{disc}[u]$.

!!! tip "빡빡한 부등호와 느슨한 부등호"
    For articulation points, the condition is $\text{low}[v] \geq \text{disc}[u]$ (non-strict). For bridges, it is $\text{low}[v] > \text{disc}[u]$ (strict). The difference arises because a back edge from $v$'s subtree to $u$ itself saves $u$ from being a bridge endpoint (the cycle through $u$ provides an alternate path), but it does not save $u$ from being an articulation point.

## 복잡도

깊이 우선 돌아보기 한 번으로 모든 다리를 찾는다:

$$
T(V, E) = O(V + E)
$$

공간 복잡도는 찾은 시각, 낮은 이음 값, 부모 배열 때문에 $O(V)$이다.

## 구현

```python
"""
방향 없는 그래프에서 다리(자름 변) 찾기.

찾은 시각과 낮은 이음 값을 쓰는 깊이 우선 돌아보기를 쓴다. 변 (u, v)은
v의 아래나무에 있는 어느 꼭짓점도 뒤로 가는 변으로 u나 u의 조상에
닿을 수 없으면 다리이다.
"""


# === 다리 찾기 ===
def find_bridges(graph, n):
    """
    방향 없는 그래프의 다리를 모두 찾는다.

    매개변수
    ----------
    graph : dict[int, list[int]]
        꼭짓점이 0부터 n-1인 방향 없는 그래프의 이웃 목록.
    n : int
        꼭짓점의 개수.

    반환값
    -------
    list[tuple[int, int]]
        다리 변의 목록.
    """
    disc = [-1] * n
    low = [0] * n
    parent = [-1] * n
    bridges = []
    timer = [0]

    def dfs(u):
        disc[u] = low[u] = timer[0]
        timer[0] += 1

        for v in graph.get(u, []):
            if disc[v] == -1:
                parent[v] = u
                dfs(v)
                low[u] = min(low[u], low[v])

                # 빡빡한 부등호: 다리 조건
                if low[v] > disc[u]:
                    bridges.append((u, v))

            elif v != parent[u]:
                low[u] = min(low[u], disc[v])

    for u in range(n):
        if disc[u] == -1:
            dfs(u)

    return bridges


# === 메인 ===
if __name__ == "__main__":
    # 다리가 있는 그래프: 3-4, 4-5, 5-6
    graph = {
        0: [1],
        1: [0, 2, 3],
        2: [1, 3],
        3: [1, 2, 4],
        4: [3, 5],
        5: [4, 6],
        6: [5],
    }
    bridges = find_bridges(graph, 7)
    print(f"Bridges: {bridges}")

    # 삼각형 그래프(다리 없음)
    triangle = {0: [1, 2], 1: [0, 2], 2: [0, 1]}
    print(f"Triangle bridges: {find_bridges(triangle, 3)}")
```

**출력:**
```
Bridges: [(3, 4), (4, 5), (5, 6)]
Triangle bridges: []
```

변 3-4, 4-5, 5-6은 다른 길이 없는 사슬을 이루므로 다리이다. 삼각형에는 변마다 순환 위에 있으므로 다리가 없다.

## 이음매 점과의 관계

다리마다 적어도 한 끝점이 이음매 점이다:

- $(u, v)$이 다리이고 $u$과 $v$의 차수가 모두 1보다 크면 $u$과 $v$ 모두 이음매 점이다.
- $(u, v)$이 다리이고 $v$이 잎(차수 1)이면 $u$이 이음매 점이다($u$도 차수가 1이면 그래프에 꼭짓점 둘과 변 하나뿐이므로 예외이다).

그러나 이음매 점에 반드시 다리가 닿아 있는 것은 아니다. 보기로 삼각형 둘이 꼭짓점 1을 함께 갖는 꼴의 그래프에서 꼭짓점 1을 보라. 그것은 이음매 점이지만 거기 닿는 어느 변도 다리가 아니다.

## 다리 나무

[두 겹 이음 조각](components.md)(2-변-이음 조각)마다 꼭짓점 하나로 오그리면 **다리 나무**(또는 다리로만 제한한 **덩이-자름 나무**)가 나온다. 이 나무는 그래프의 층진 다리 짜임을 담아내며 다음에 쓸모 있다:

- 두 꼭짓점이 다리로 갈라지는지 묻는 물음에 답하기.
- 모든 다리를 없애려고 더해야 하는 변의 최소 개수를 셈하기.
- Finding the number of bridges on any path in $O(\log V)$ time with LCA queries.

## 응용

- **그물 믿음성:** 다리는 주고받기 그물에서 혼자 무너지면 끝나는 곳을 나타낸다.
- **나름:** 길 그물의 다리 변은 막히면 지역이 끊어지는 길이다.
- **생물학:** 단백질 주고받기 그물에서 다리 주고받기는 없어서는 안 될 조절 통로를 가리킬 수 있다.

## 참고 문헌

- Tarjan, R. E. (1974). A note on finding the bridges of a graph. *Information Processing Letters*, 2(6), 160-161.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. 20장.

## 연습문제

**연습문제 1.**
그래프의 다리를 정의하고 깊이 우선 돌아보기로 다리를 찾는 조건을 설명하여라.

??? success "연습문제 1 풀이"
    A **bridge** is an edge whose removal disconnects the graph. Edge $(u, v)$ (where $u$ is the parent of $v$ in the DFS tree) is a bridge if and only if $\text{low}[v] > \text{disc}[u]$. This means no vertex in $v$'s subtree has a back edge to $u$ or any ancestor of $u$, so removing $(u, v)$ disconnects $v$'s subtree from the rest. The algorithm runs in $O(V + E)$. $\square$

---

**연습문제 2.**
다리가 어떤 순환에도 들 수 없음을 증명하여라.

??? success "연습문제 2 풀이"
    Suppose edge $e = (u, v)$ is both a bridge and part of cycle $C$. Removing $e$ disconnects the graph by assumption. But in cycle $C$, the remaining edges still provide a path from $u$ to $v$ (going around the cycle the other way). So $u$ and $v$ remain connected, and since all other vertices were connected to either $u$ or $v$ before removal, the graph stays connected — contradicting $e$ being a bridge. $\square$

---

**연습문제 3.**
이어진 그래프에 다리가 없다. 그 변 이어짐에 대해 무엇을 말할 수 있는가?

??? success "연습문제 3 풀이"
    If a connected graph has no bridges, its edge connectivity is at least 2: removing any single edge keeps the graph connected. This means the graph is **2-edge-connected**. Equivalently, every edge lies on at least one cycle. Such graphs have at least $V$ edges (they contain a spanning cycle through every edge). $\square$

---

**연습문제 4.**
다리와 두 겹 이음 조각 사이의 관계를 설명하여라.

??? success "연습문제 4 풀이"
    Each bridge forms its own biconnected component containing just that single edge (and its two endpoints). Non-bridge edges are grouped into biconnected components with $\geq 2$ edges. The number of biconnected components equals the number of bridges plus the number of maximal 2-edge-connected subgraphs. In the block-cut tree, bridge-blocks are leaf-like structures connecting different parts of the graph. $\square$
