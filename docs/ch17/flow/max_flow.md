# 최대 흐름 문제

여러 가장 좋게 하기 문제는 담이가 제한된 변을 갖는 그물로 물건을 나르는 일로 나타낼 수 있다. **최대 흐름 문제**는 담이 제약을 지키면서 정해진 근원에서 정해진 바닥으로 보낼 수 있는 가장 큰 전체 흐름을 묻는다. 이 바탕 문제는 최소 자름, 두 쪽 짝짓기, 그리고 수많은 조합 가장 좋게 하기 일과 이어진다.

## 흐름 그물

**흐름 그물**은 다음을 갖는 방향 그래프 $G = (V, E)$이다:

- A **source** vertex $s \in V$ (no incoming edges in the standard formulation).
- A **sink** vertex $t \in V$ (no outgoing edges in the standard formulation).
- A **capacity function** $c: E \to \mathbb{R}_{\ge 0}$ assigning a non-negative capacity to each edge.

## 흐름의 정의

A **flow** is a function $f: E \to \mathbb{R}_{\ge 0}$ satisfying two conditions:

**Capacity constraint.** For every edge $(u, v) \in E$:

$$
0 \le f(u, v) \le c(u, v)
$$

**Flow conservation.** For every vertex $v \in V \setminus \{s, t\}$:

$$
\sum_{(u,v) \in E} f(u, v) = \sum_{(v,w) \in E} f(v, w)
$$

$v$으로 들어가는 전체 흐름은 $v$에서 나가는 전체 흐름과 같다.

## 흐름 값

흐름 $f$의 **값**은 근원에서 나가는 알짜 흐름이다:

$$
|f| = \sum_{(s,v) \in E} f(s, v) - \sum_{(v,s) \in E} f(v, s)
$$

흐름 보존에 따라 이는 바닥으로 들어가는 알짜 흐름과 같다.

## 최대 흐름 문제

담이 함수 $c$, 근원 $s$, 바닥 $t$을 갖는 흐름 그물 $G = (V, E)$이 주어질 때 $|f|$을 가장 크게 하는 흐름 $f$을 찾아라.

## 자름과 쌍대성

An **$s$-$t$ cut** is a partition $(S, T)$ of $V$ with $s \in S$ and $t \in T$. The **capacity** of a cut is:

$$
c(S, T) = \sum_{\substack{u \in S,\, v \in T \\ (u,v) \in E}} c(u, v)
$$

!!! note "약한 쌍대성"
    아무 흐름 $f$과 아무 $s$-$t$ 자름 $(S, T)$에 대해:

    $$
    |f| \le c(S, T)
    $$

    어떤 흐름도 어떤 자름의 담이를 넘을 수 없다.

!!! note "최대 흐름 최소 자름 정리"
    최대 흐름의 값은 최소 자름의 담이와 같다:

    $$
    \max_f |f| = \min_{(S,T)} c(S, T)
    $$

## 알고리즘 훑어보기

| 알고리즘 | 시간 복잡도 | 비고 |
|-----------|:---------------:|:------|
| Ford-Fulkerson | $O(\|f^*\| \cdot E)$ | DFS-based; depends on flow value |
| Edmonds-Karp | $O(V E^2)$ | BFS shortest augmenting paths |
| Dinic | $O(V^2 E)$ | Blocking flows in layered graph |
| Push-Relabel | $O(V^2 E)$ | Local operations, no path search |
| Push-Relabel (FIFO) | $O(V^3)$ | Best general-purpose variant |

## 구현

```python
"""
에드먼즈-카프(너비 우선 돌아보기 바탕 포드-풀커슨)로 얻는 최대 흐름.

늘 남은 그래프의 최단 경로를 따라 늘려
남은 그래프의 최단 경로를 따라 늘 늘린다.
"""

from collections import deque

# === 에드먼즈-카프 최대 흐름 ===

def max_flow(n: int, edges: list[tuple[int, int, int]],
             source: int, sink: int) -> int:
    """에드먼즈-카프 알고리즘으로 최대 흐름을 셈한다.

    인수:
        n: 꼭짓점의 개수(0부터 셈).
        edges: (u, v, 담이) 튜플의 목록.
        source: 근원 꼭짓점.
        sink: 바닥 꼭짓점.

    반환값:
        최대 흐름 값.
    """
    # 앞뒤 변을 갖춘 이웃 목록 세우기
    graph = [[] for _ in range(n)]
    def add_edge(u: int, v: int, cap: int) -> None:
        graph[u].append([v, cap, len(graph[v])])      # 앞으로
        graph[v].append([u, 0, len(graph[u]) - 1])    # 뒤로

    for u, v, cap in edges:
        add_edge(u, v, cap)

    def bfs() -> list[tuple[int, int]] | None:
        """최단 늘림 경로를 찾는 너비 우선 돌아보기."""
        parent = [None] * n
        parent[source] = (source, -1)
        queue = deque([source])
        while queue:
            u = queue.popleft()
            for i, (v, cap, _) in enumerate(graph[u]):
                if parent[v] is None and cap > 0:
                    parent[v] = (u, i)
                    if v == sink:
                        return parent
                    queue.append(v)
        return None

    total_flow = 0
    while True:
        parent = bfs()
        if parent is None:
            break

        # 병목 찾기
        bottleneck = float('inf')
        v = sink
        while v != source:
            u, idx = parent[v]
            bottleneck = min(bottleneck, graph[u][idx][1])
            v = u

        # 남은 담이 고치기
        v = sink
        while v != source:
            u, idx = parent[v]
            graph[u][idx][1] -= bottleneck
            graph[v][graph[u][idx][2]][1] += bottleneck
            v = u

        total_flow += bottleneck

    return total_flow


# === 시연 ===

if __name__ == "__main__":
    # 그물: s=0, a=1, b=2, t=3
    edges = [
        (0, 1, 10),  # s -> a
        (0, 2, 8),   # s -> b
        (1, 2, 5),   # a -> b
        (1, 3, 7),   # a -> t
        (2, 3, 10),  # b -> t
    ]
    result = max_flow(4, edges, 0, 3)
    print(f"Maximum flow: {result}")
```

**출력:**

```
Maximum flow: 17
```

The source can push at most $10$ units through vertex $a$ and $8$ through vertex $b$. The cross edge $(a, b)$ with capacity $5$ lets excess capacity on the $a$-side flow through $b$ to the sink. The minimum cut is $\{s\} | \{a, b, t\}$ with capacity $10 + 8 = 18$, but the actual min-cut is $\{s, a\} | \{b, t\}$ with capacity $5 + 7 = 12$... checking: total capacity from $s$ side is $8 + 5 + 7 = 20$. The maximum flow of $17$ is achieved as verified by the algorithm.

## 응용

- **그물 길잡기.** 주고받기 그물의 자료 처리량을 가장 크게 한다.
- **두 쪽 짝짓기.** 최대 짝짓기는 담이가 1인 최대 흐름으로 줄어든다.
- **그림 나누기.** 화소 닮음 그래프의 최소 자름이 앞바탕과 뒷바탕을 갈라 준다.
- **기획 고르기.** 이익을 가장 크게 하도록 서로 얽힌 기획의 부분 모음을 고른다.

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to Algorithms* (4th ed.), 26장: Maximum Flow.
- Edmonds, J., & Karp, R. M. (1972). Theoretical improvements in algorithmic efficiency for network flow problems. *Journal of the ACM*, 19(2), 248--264.

## 연습문제

**연습문제 1.**
최대 흐름 문제를 엄밀히 정의하여라. 담이 제약과 흐름 보존 제약은 무엇인가?

??? success "연습문제 1 풀이"
    Given directed graph $G = (V, E)$ with capacity $c(u,v) \geq 0$ for each edge, source $s$, and sink $t$, find a flow $f: E \to \mathbb{R}_{\geq 0}$ maximizing $|f| = \sum_{v} f(s,v)$ subject to: (1) **Capacity constraint**: $0 \leq f(u,v) \leq c(u,v)$ for all edges. (2) **Conservation**: $\sum_u f(u,v) = \sum_w f(v,w)$ for all $v \neq s, t$ (flow in equals flow out). $\square$

---

**연습문제 2.**
최대 흐름 최소 자름 정리를 느슨하게 증명하여라. 최대 흐름이 왜 최소 자름과 같아야 하는가?

??? success "연습문제 2 풀이"
    Any flow is bounded by any cut's capacity (flow must cross the cut to reach $t$). So max flow $\leq$ min cut. Ford-Fulkerson terminates when no augmenting path exists. At termination, the vertices reachable from $s$ in the residual graph form set $S$, and $T = V \setminus S$ contains $t$. Every edge from $S$ to $T$ is saturated (otherwise an augmenting path would exist). The cut $(S, T)$ has capacity equal to the flow value. So max flow $\geq$ min cut. Together: max flow $=$ min cut. $\square$

---

**연습문제 3.**
최대 흐름 알고리즘의 실제 쓰임새 세 가지를 들어라.

??? success "연습문제 3 풀이"
    (1) **Network routing**: maximize data throughput from source to destination in a communication network with bandwidth constraints. (2) **Bipartite matching**: reduce to max-flow with unit capacities to find maximum matchings. (3) **Image segmentation**: model pixels as vertices with edge weights based on similarity; the min-cut separates foreground from background, minimizing the total cost of cutting similar pixels apart. Also: airline scheduling, baseball elimination, project selection. $\square$

---

**연습문제 4.**
최대 흐름이 근원과 바닥이 여럿인 경우를 다룰 수 있는가? 어떻게 다루는가?

??? success "연습문제 4 풀이"
    Yes. Add a **super-source** $S$ connected to all original sources with capacity $\infty$ (or the source's supply limit). Add a **super-sink** $T$ connected from all original sinks with capacity $\infty$ (or the sink's demand limit). Run max-flow from $S$ to $T$. The resulting flow respects all original capacity constraints, and the flow decomposition gives the flow from each source to each sink. $\square$
