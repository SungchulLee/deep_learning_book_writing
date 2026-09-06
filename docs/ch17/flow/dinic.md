# 디닉 알고리즘

While the Edmonds-Karp algorithm improves Ford-Fulkerson by always choosing shortest augmenting paths, it still sends flow along only one path per iteration. Dinic's algorithm (also spelled Dinitz) pushes this idea further: it builds a **level graph** capturing all shortest paths from source to sink, then sends as much flow as possible through that structure in a single phase. This yields a worst-case complexity of $O(V^2 E)$, and the algorithm performs especially well on unit-capacity networks where it runs in $O(E\sqrt{V})$ time.

## 켜 그래프

**켜 그래프**(또는 **층 그래프**)는 근원에서의 거리에 따라 남은 그래프를 켜로 갈라 놓는다.

Given a flow network $G = (V, E)$ with source $s$ and sink $t$, and the current residual graph $G_f$, define the **level** of each vertex $v$ as $\text{level}(v) = d(s, v)$, the shortest-path distance (in number of edges) from $s$ to $v$ in $G_f$.

The level graph $G_L = (V_L, E_L)$ contains only edges $(u, v)$ from $G_f$ that satisfy $\text{level}(v) = \text{level}(u) + 1$. In other words, $G_L$ keeps only those residual edges that move strictly one level closer to the sink.

**Construction via BFS.** Run BFS from $s$ in $G_f$, recording the level of each reachable vertex. Then filter edges: keep $(u, v) \in G_f$ only if $\text{level}(v) = \text{level}(u) + 1$ and $v$ is reachable. This takes $O(V + E)$ time.

너비 우선 돌아보기로 $t$에 닿지 못하면 늘림 경로가 없으므로 알고리즘이 끝난다. 지금 흐름이 최대이다.

## 막는 흐름

켜 그래프 $G_L$의 **막는 흐름**은 $G_L$에서 $s$에서 $t$으로 가는 모든 경로가 꽉 찬 변(흐름이 남은 담이와 같은 변)을 적어도 하나 갖게 하는 흐름 $f'$이다. 다시 말해 막는 흐름을 더하고 나면 $G_L$에 $s$-$t$ 경로가 남지 않는다.

막는 흐름이 $G_L$에서 꼭 최대 흐름인 것은 아니다. 그저 모든 $s$-$t$ 경로를 막을 뿐이다. 막는 흐름 찾기는 최대 흐름 찾기보다 값이 싸며, 이것이 디닉 알고리즘을 효율적으로 만드는 핵심 눈썰미이다.

**깊이 우선 돌아보기로 막는 흐름 찾기.** $G_L$에서 $s$부터 깊이 우선 돌아보기를 한다. $t$으로 가는 경로를 찾으면 그 경로를 따라 남은 담이의 최솟값만큼 흘려 보낸다(적어도 변 하나가 꽉 찬다). 꽉 찬 변을 없앤다. 어떤 꼭짓점에서의 돌아보기가 막다른 곳($G_L$에 나가는 변이 없음)에 이르면 그 꼭짓점을 지우고 되돌아간다. $s$에서 $t$으로 가는 경로가 없어질 때까지 이어 간다.

가리개를 잘 다루면(다시 시작하지 않고 가리개를 앞으로 밀면) 막는 흐름을 찾는 전체 품은 $O(VE)$이다.

## 알고리즘

디닉 알고리즘은 켜 그래프 세우기와 막는 흐름 찾기, 이 두 단계를 늘림 경로가 없어질 때까지 되풀이한다.

```
DINIC(G, s, t):
    Initialize flow f = 0
    while True:
        Build level graph G_L via BFS from s in G_f
        if t is not reachable:
            return f
        Find a blocking flow f' in G_L
        f = f + f'
        Update residual graph G_f
```

바깥 되풀이의 한 바퀴를 **단계**라 한다. 단계마다 남은 그래프에서 $s$부터 $t$까지의 거리가 적어도 1 늘어난다. 이 거리가 $|V| - 1$ 아래로 묶이므로 단계는 많아야 $|V| - 1$개이다.

## 복잡도 분석

**Lemma (distance increase).** Let $d_f(s, t)$ denote the shortest-path distance from $s$ to $t$ in $G_f$. After adding a blocking flow to the level graph, $d_{f'}(s, t) > d_f(s, t)$.

*Proof sketch.* The blocking flow saturates at least one edge on every shortest path. When the residual graph is updated, any new augmenting path must use at least one edge that goes "backward" (from a higher level to a lower level). Such backward edges force the new shortest path to be strictly longer than the previous one. $\square$

**Theorem.** Dinic's algorithm runs in $O(V^2 E)$ time.

*증명.* 단계는 많아야 $O(V)$개이다(단계마다 거리가 적어도 1 늘고 $|V| - 1$ 아래로 묶이기 때문이다). 각 단계는 다음으로 이루어진다:

- 너비 우선 돌아보기로 켜 그래프 세우기: $O(E)$
- 막는 흐름 찾기: $O(VE)$

The total time is $O(V) \cdot O(VE) = O(V^2 E)$. $\square$

## 담이가 1인 그물

On networks where every edge has capacity 1, Dinic's algorithm achieves $O(E\sqrt{V})$ time.

The key observation is that in a unit-capacity network, each blocking flow phase takes $O(E)$ time (each augmenting path saturates an edge, removing it, so the total work across all paths in one phase is $O(E)$). Furthermore, after $\sqrt{V}$ phases, the maximum remaining augmentable flow is at most $\sqrt{V}$ (by a counting argument on the level structure). The remaining flow can be found in at most $\sqrt{V}$ additional phases, each costing $O(E)$.

Total: $O(\sqrt{V}) \cdot O(E) = O(E\sqrt{V})$.

This makes Dinic's algorithm the method of choice for bipartite matching, where the underlying network is unit-capacity and $O(E\sqrt{V})$ matches the best known bound for maximum bipartite matching.

## 풀이 예제

Consider a network with vertices $\{s, a, b, c, t\}$ and edges:

| 변 | 담이 |
|------|----------|
| $(s, a)$ | 10 |
| $(s, b)$ | 10 |
| $(a, b)$ | 2 |
| $(a, c)$ | 8 |
| $(b, c)$ | 6 |
| $(c, t)$ | 14 |
| $(a, t)$ | 4 |

**Phase 1.** BFS from $s$ gives levels: $\text{level}(s) = 0$, $\text{level}(a) = 1$, $\text{level}(b) = 1$, $\text{level}(c) = 2$, $\text{level}(t) = 2$. The level graph keeps edges $(s,a)$, $(s,b)$, $(a,c)$, $(a,t)$, $(b,c)$, $(c,t)$ (edge $(a,b)$ is excluded since both are at level 1).

막는 흐름은 $G_L$의 모든 $s$-$t$ 경로가 막힐 때까지 경로를 찾아 흐름을 흘려 보낸다. 보기로:

- Path $s \to a \to t$: push 4 (saturates $(a,t)$)
- Path $s \to a \to c \to t$: push 6 (saturates remaining capacity on $(a,c)$ after considering available flow)
- Path $s \to b \to c \to t$: push 4 (limited by remaining capacity on $(c,t)$)

1단계 뒤 전체 흐름: 14.

**2단계.** 고쳐진 남은 그래프 위에 켜 그래프를 다시 세운다. $s$에서 $t$까지의 거리가 늘었다. 너비 우선 돌아보기가 $t$에 닿지 못할 때까지 이어 가며, 그때 흐름이 최대이다.

## 파이썬 구현

```python
"""
최대 흐름을 위한 디닉 알고리즘.

너비 우선 돌아보기로 켜 그래프를 세우고 깊이 우선 돌아보기로 막는 흐름을 찾아
최대 흐름을 O(V^2 E) 시간에 셈한다.
"""

from collections import deque

# === 변 나타내기 ===

class Edge:
    """담이와 흐름을 좇는 방향 변."""

    __slots__ = ['to', 'cap', 'rev']

    def __init__(self, to: int, cap: int, rev: int):
        self.to = to
        self.cap = cap
        self.rev = rev  # graph[to]에 있는 거꿀 변의 번호


# === 그래프 세우기 ===

def add_edge(graph: list, u: int, v: int, cap: int) -> None:
    """담이가 주어진 변 u -> v과 담이 0인 거꿀 변을 더한다."""
    graph[u].append(Edge(v, cap, len(graph[v])))
    graph[v].append(Edge(u, 0, len(graph[u]) - 1))


# === 켜 그래프를 위한 너비 우선 돌아보기 ===

def bfs(graph: list, s: int, t: int, level: list) -> bool:
    """너비 우선 돌아보기로 켜 그래프를 세운다. t에 닿을 수 있으면 True를 돌려준다."""
    for i in range(len(level)):
        level[i] = -1
    level[s] = 0
    queue = deque([s])
    while queue:
        u = queue.popleft()
        for e in graph[u]:
            if e.cap > 0 and level[e.to] < 0:
                level[e.to] = level[u] + 1
                queue.append(e.to)
    return level[t] >= 0


# === 막는 흐름을 위한 깊이 우선 돌아보기 ===

def dfs(graph: list, u: int, t: int, f: int,
        level: list, iter_: list) -> int:
    """가리개를 앞으로 미는 깊이 우선 돌아보기로 막는 흐름을 찾는다."""
    if u == t:
        return f
    while iter_[u] < len(graph[u]):
        e = graph[u][iter_[u]]
        if e.cap > 0 and level[e.to] == level[u] + 1:
            d = dfs(graph, e.to, t, min(f, e.cap), level, iter_)
            if d > 0:
                e.cap -= d
                graph[e.to][e.rev].cap += d
                return d
        iter_[u] += 1
    return 0


# === 주된 알고리즘 ===

def dinic(n: int, edges: list, s: int, t: int) -> int:
    """
    디닉 알고리즘으로 최대 흐름을 셈한다.

    매개변수
    ----------
    n : int
        꼭짓점의 개수(0부터 n-1까지 이름 붙임).
    edges : (u, v, cap) 튜플의 목록
        담이를 갖는 방향 변.
    s : int
        근원 꼭짓점.
    t : int
        바닥 꼭짓점.

    반환값
    -------
    int
        s에서 t까지의 최대 흐름 값.
    """
    graph = [[] for _ in range(n)]
    for u, v, cap in edges:
        add_edge(graph, u, v, cap)

    level = [0] * n
    flow = 0

    while bfs(graph, s, t, level):
        iter_ = [0] * n
        while True:
            f = dfs(graph, s, t, float('inf'), level, iter_)
            if f == 0:
                break
            flow += f

    return flow


# === 보기 ===

if __name__ == "__main__":
    # s=0, a=1, b=2, c=3, t=4
    edges = [
        (0, 1, 10),  # s -> a
        (0, 2, 10),  # s -> b
        (1, 2, 2),   # a -> b
        (1, 3, 8),   # a -> c
        (2, 3, 6),   # b -> c
        (3, 4, 14),  # c -> t
        (1, 4, 4),   # a -> t
    ]
    result = dinic(5, edges, 0, 4)
    print(f"Maximum flow: {result}")
```

## 에드먼즈-카프와의 견줌

| 성질 | 에드먼즈-카프 | 디닉 |
|----------|-------------|---------|
| 경로 전략 | 최단 경로 하나 | 모든 최단 경로(막는 흐름) |
| Time complexity | $O(VE^2)$ | $O(V^2 E)$ |
| Unit-capacity networks | $O(E \cdot \min(E^{1/2}, V^{2/3}))$ | $O(E\sqrt{V})$ |
| 짜기 | 더 단순하다(너비 우선만) | 더 손이 간다(너비 우선 + 깊이 우선) |

Dinic's algorithm dominates Edmonds-Karp when $V < E$, which is the common case in dense networks. For sparse graphs where $E = O(V)$, both yield $O(V^3)$.

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. 24, 26장.
- Dinitz, Y. (1970). Algorithm for solution of a problem of maximum flow in networks with power estimation. *Doklady Akademii Nauk SSSR*, 194(4).

## 연습문제

**연습문제 1.**
디닉 알고리즘을 설명하고 담이가 1인 그물에서 왜 에드먼즈-카프보다 빠른지 설명하여라.

??? success "연습문제 1 풀이"
    Dinic's algorithm builds a level graph using BFS, then finds blocking flows using DFS. A blocking flow saturates at least one edge on every $s$-$t$ path in the level graph. After each blocking flow, the shortest $s$-$t$ path length increases by at least 1, so at most $O(V)$ phases are needed. Each phase takes $O(VE)$ for general graphs, giving $O(V^2 E)$ total. For unit-capacity networks, each phase takes $O(E)$ and there are $O(\sqrt{E})$ phases, giving $O(E\sqrt{E}) = O(E^{1.5})$, faster than Edmonds-Karp's $O(VE^2)$. $\square$

---

**연습문제 2.**
막는 흐름이란 무엇인가? 최대 흐름과 어떻게 다른가?

??? success "연습문제 2 풀이"
    A **blocking flow** saturates at least one edge on every path from $s$ to $t$ in the level graph, meaning no more flow can be pushed along shortest paths. A **maximum flow** has no augmenting path at all (in the full residual graph, not just the level graph). A blocking flow may not be maximum because augmenting paths of greater length may still exist. Dinic's algorithm finds the maximum flow by iterating: each blocking flow eliminates shortest paths, and the overall process converges when no $s$-$t$ path remains. $\square$

---

**연습문제 3.**
디닉 알고리즘이 많아야 $V - 1$단계 뒤에 끝남을 증명하여라.

??? success "연습문제 3 풀이"
    Each phase computes a blocking flow in the level graph. After a blocking flow, the shortest $s$-$t$ path in the residual graph is strictly longer than before (every shortest path in the current level graph has been blocked). The shortest path length starts at $\geq 1$ and can be at most $V - 1$ (a simple path visits at most $V$ vertices). After $V - 1$ phases, the shortest path would need $\geq V$ edges, which is impossible for simple paths. Therefore no augmenting path exists, and the algorithm terminates with the maximum flow. $\square$

---

**연습문제 4.**
포드-풀커슨, 에드먼즈-카프, 디닉 알고리즘의 시간 복잡도를 견주어라.

??? success "연습문제 4 풀이"
    | 알고리즘 | 시간 복잡도 | 비고 |
    |---|---|---|
    | Ford-Fulkerson | $O(E \cdot f^*)$ | $f^*$ = max flow value; may not terminate with irrational capacities |
    | Edmonds-Karp | $O(VE^2)$ | BFS for shortest augmenting path; always terminates |
    | Dinic | $O(V^2 E)$ | Level graph + blocking flow; $O(E\sqrt{V})$ for unit capacity |

    Dinic is fastest for most practical cases, especially sparse graphs and unit-capacity networks. $\square$
