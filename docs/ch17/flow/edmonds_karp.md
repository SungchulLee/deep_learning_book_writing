# 에드먼즈-카프 알고리즘

The Ford-Fulkerson method leaves the choice of augmenting path unspecified, which can lead to poor performance or even non-termination with irrational capacities. The Edmonds-Karp algorithm resolves this by always choosing the **shortest augmenting path** (fewest edges) via BFS. This simple rule guarantees termination in $O(VE)$ augmentations and yields an overall $O(VE^2)$ time complexity.

## 포드-풀커슨에서 에드먼즈-카프로

포드-풀커슨은 남은 그래프에서 늘림 경로를 거듭 찾아 그 길로 흐름을 흘려 보낸다. 이 방법은 어떤 경로 고르기 전략에도 옳지만, 늘림 횟수는 어떤 경로를 고르느냐에 달렸다.

에드먼즈-카프는 늘림 경로를 찾을 때 근원 $s$에서 **너비 우선 돌아보기**를 쓰도록 포드-풀커슨을 좁힌 것이다. 너비 우선 돌아보기는 거리가 늘어나는 차례로 꼭짓점을 살펴보므로, 찾은 경로는 늘 남은 그래프의 모든 $s$-$t$ 경로 가운데 변의 수가 가장 적다.

## 알고리즘

```
EDMONDS-KARP(G, s, t):
    Initialize flow f(u,v) = 0 for all (u,v)
    while BFS finds a path P from s to t in G_f:
        c_f(P) = min { c_f(u,v) : (u,v) in P }
        for each edge (u,v) in P:
            f(u,v) = f(u,v) + c_f(P)
            f(v,u) = f(v,u) - c_f(P)
    return f
```

너비 우선 돌아보기마다 $O(V + E)$ 시간이 든다. 핵심 물음은 이것이다. 늘림이 몇 번이나 일어날 수 있는가?

## 최단 경로 거리의 단조성

The crucial property underlying the $O(VE^2)$ bound is that shortest-path distances in the residual graph never decrease.

**Lemma.** Let $\delta_f(s, v)$ denote the shortest-path distance from $s$ to $v$ in residual graph $G_f$. After augmenting along a shortest path, for every vertex $v \in V$:

$$
\delta_{f'}(s, v) \geq \delta_f(s, v)
$$

여기서 $f'$은 늘린 뒤의 흐름이다.

*Proof sketch.* Suppose for contradiction that $\delta_{f'}(s, v) < \delta_f(s, v)$ for some vertex $v$. Pick $v$ with the smallest $\delta_{f'}(s, v)$ among all such vertices. Let $u$ be the predecessor of $v$ on a shortest path from $s$ to $v$ in $G_{f'}$. Then $\delta_{f'}(s, v) = \delta_{f'}(s, u) + 1$.

By our choice of $v$, we have $\delta_{f'}(s, u) \geq \delta_f(s, u)$. Edge $(u, v)$ must exist in $G_{f'}$. If $(u, v)$ also exists in $G_f$, then $\delta_f(s, v) \leq \delta_f(s, u) + 1 \leq \delta_{f'}(s, u) + 1 = \delta_{f'}(s, v)$, contradicting our assumption.

If $(u, v)$ does not exist in $G_f$, it was created by augmenting along $(v, u)$. Since augmentation uses a shortest path, $(v, u)$ lies on a shortest $s$-$t$ path in $G_f$, so $\delta_f(s, u) = \delta_f(s, v) + 1$. Then $\delta_{f'}(s, v) = \delta_{f'}(s, u) + 1 \geq \delta_f(s, u) + 1 = \delta_f(s, v) + 2$, again contradicting $\delta_{f'}(s, v) < \delta_f(s, v)$. $\square$

## 늘림 횟수의 한계

**정리.** 에드먼즈-카프 알고리즘은 많아야 $O(VE)$번 늘린다.

*증명 얼개.* 남은 그래프의 변 $(u, v)$이 늘림 경로에서 남은 담이가 가장 작으면(곧 꽉 차게 되면) 그 변을 그 경로 위의 **결정적인** 변이라 하자. 늘림마다 적어도 변 하나가 꽉 찬다.

When edge $(u, v)$ is critical, $\delta_f(s, v) = \delta_f(s, u) + 1$. For $(u, v)$ to appear again in the residual graph, flow must be pushed along $(v, u)$ in some later augmentation. At that point, $\delta_{f'}(s, u) = \delta_{f'}(s, v) + 1$. By the monotonicity lemma:

$$
\delta_{f'}(s, u) = \delta_{f'}(s, v) + 1 \geq \delta_f(s, v) + 1 = \delta_f(s, u) + 2
$$

So $\delta_{f'}(s, u)$ increases by at least 2 between consecutive times $(u, v)$ is critical. Since distances are bounded by $|V| - 1$, each edge can be critical at most $O(V)$ times. With $O(E)$ edges, there are at most $O(VE)$ total augmentations. $\square$

## 전체 복잡도

**Theorem.** The Edmonds-Karp algorithm runs in $O(VE^2)$ time.

*Proof.* Each augmentation requires a BFS taking $O(E)$ time (since $E \geq V - 1$ in a connected graph). With $O(VE)$ augmentations, the total is $O(VE) \cdot O(E) = O(VE^2)$. $\square$

## 풀이 예제

Consider a network with vertices $\{s, a, b, t\}$:

| 변 | 담이 |
|------|----------|
| $(s, a)$ | 4 |
| $(s, b)$ | 3 |
| $(a, b)$ | 2 |
| $(a, t)$ | 3 |
| $(b, t)$ | 5 |

**Iteration 1.** BFS from $s$ finds shortest path $s \to a \to t$ (2 edges). Bottleneck capacity: $\min(4, 3) = 3$. Push flow 3.

남은 그래프 고침: $(s, a)$의 남은 담이는 1, $(a, t)$은 꽉 참, 거꿀 변 $(a, s)$과 $(t, a)$이 담이 3으로 생긴다.

**Iteration 2.** BFS finds $s \to b \to t$ (2 edges). Bottleneck: $\min(3, 5) = 3$. Push flow 3.

**Iteration 3.** BFS finds $s \to a \to b \to t$ (3 edges). Bottleneck: $\min(1, 2, 2) = 1$. Push flow 1.

**4번째 바퀴.** 너비 우선 돌아보기가 $s$에서 $t$에 닿지 못한다. 알고리즘이 최대 흐름 = 7로 끝난다.

$s$에서 $t$까지의 최단 경로 거리가 1번째와 2번째 바퀴에서 2였다가 3번째 바퀴에서 3으로 늘어난 것을 보라. 단조성을 확인해 준다.

## 파이썬 구현

```python
"""
최대 흐름을 위한 에드먼즈-카프 알고리즘.

너비 우선 돌아보기로 최단 늘림 경로를 찾아
시간 복잡도 O(VE^2).
"""

from collections import deque

# === 늘림 경로를 찾는 너비 우선 돌아보기 ===

def bfs(capacity: list, source: int, sink: int, parent: list) -> int:
    """
    너비 우선 돌아보기로 최단 늘림 경로를 찾는다.

    경로의 병목 담이를 돌려준다. 경로가 없으면 0.
    """
    n = len(capacity)
    visited = [False] * n
    visited[source] = True
    queue = deque([(source, float('inf'))])

    while queue:
        u, flow = queue.popleft()
        for v in range(n):
            if not visited[v] and capacity[u][v] > 0:
                visited[v] = True
                parent[v] = u
                new_flow = min(flow, capacity[u][v])
                if v == sink:
                    return new_flow
                queue.append((v, new_flow))
    return 0


# === 주된 알고리즘 ===

def edmonds_karp(n: int, edges: list, source: int, sink: int) -> int:
    """
    에드먼즈-카프로 최대 흐름을 셈한다.

    매개변수
    ----------
    n : int
        꼭짓점의 개수(0부터 n-1까지 이름 붙임).
    edges : (u, v, cap) 튜플의 목록
        담이를 갖는 방향 변.
    source : int
        근원 꼭짓점.
    sink : int
        바닥 꼭짓점.

    반환값
    -------
    int
        최대 흐름 값.
    """
    capacity = [[0] * n for _ in range(n)]
    for u, v, cap in edges:
        capacity[u][v] += cap

    parent = [-1] * n
    max_flow = 0

    while True:
        augment = bfs(capacity, source, sink, parent)
        if augment == 0:
            break
        max_flow += augment
        v = sink
        while v != source:
            u = parent[v]
            capacity[u][v] -= augment
            capacity[v][u] += augment
            v = u

    return max_flow


# === 보기 ===

if __name__ == "__main__":
    # s=0, a=1, b=2, t=3
    edges = [
        (0, 1, 4),  # s -> a
        (0, 2, 3),  # s -> b
        (1, 2, 2),  # a -> b
        (1, 3, 3),  # a -> t
        (2, 3, 5),  # b -> t
    ]
    result = edmonds_karp(4, edges, 0, 3)
    print(f"Maximum flow: {result}")  # 기대값: 7
```

!!! tip "에드먼즈-카프를 쓸 때"
    Edmonds-Karp is a good default choice when you need a straightforward max-flow implementation. Its $O(VE^2)$ bound is polynomial and the code is simple (just BFS). For better performance on dense graphs, consider Dinic's algorithm with its $O(V^2E)$ bound.

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. 26장.
- Edmonds, J., & Karp, R. M. (1972). Theoretical improvements in algorithmic efficiency for network flow problems. *Journal of the ACM*, 19(2), 248-264.

## 연습문제

**연습문제 1.**
에드먼즈-카프는 포드-풀커슨을 어떻게 낫게 하는가? 어떤 찾기 전략을 쓰는가?

??? success "연습문제 1 풀이"
    Edmonds-Karp uses BFS (instead of arbitrary path search) to find the shortest augmenting path in the residual graph. This guarantees that the shortest path length never decreases between iterations and increases after at most $O(E)$ augmentations at the same distance. Total augmentations: $O(VE)$. Each BFS takes $O(E)$. Total: $O(VE^2)$, which is polynomial — unlike Ford-Fulkerson's $O(Ef^*)$ that depends on the flow value. $\square$

---

**연습문제 2.**
에드먼즈-카프에서 최단 늘림 경로의 길이가 결코 줄지 않음을 증명하여라.

??? success "연습문제 2 풀이"
    After augmenting along a shortest $s$-$t$ path $P$, some edges in $P$ are saturated (removed from residual) and their reverse edges are added. The reverse edges can only appear in longer paths (they go backward). Any new $s$-$t$ path in the residual graph either uses only original residual edges (same or longer) or uses a reverse edge (longer by at least 2). Therefore the shortest path length is non-decreasing. $\square$

---

**연습문제 3.**
에드먼즈-카프에서 늘림 경로 바퀴는 많아야 몇 번인가? 까닭을 대어라.

??? success "연습문제 3 풀이"
    At most $O(VE)$. For each distance level $d$ (shortest path length), at most $O(E)$ augmentations occur before the distance increases (each augmentation saturates at least one edge at distance $d$, and an edge can be saturated at this distance at most $O(V)$ times). Since $d$ ranges from 1 to at most $V - 1$, total augmentations $\leq O(V \cdot E) = O(VE)$. Each augmentation uses BFS in $O(E)$, giving $O(VE^2)$ total. $\square$

---

**연습문제 4.**
에드먼즈-카프를 짜고, 근원 $s = 0$, 바닥 $t = 3$, 담이가 $(0,1,10), (0,2,8), (1,2,5), (1,3,5), (2,3,10)$인 변을 갖는 그래프에서 최대 흐름을 찾아라.

??? success "연습문제 4 풀이"
    BFS finds path $0 \to 1 \to 3$, push 5. Then $0 \to 2 \to 3$, push 8. Then $0 \to 1 \to 2 \to 3$, push 2 (limited by residual $1 \to 2$: capacity 5, but also limited by $2 \to 3$: residual $10 - 8 = 2$). No more augmenting paths. Max flow $= 5 + 8 + 2 = 15$. $\square$
