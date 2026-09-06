# 포드-풀커슨 방법

포드-풀커슨 방법은 그물의 최대 흐름을 셈하는 바탕 방식이다. 이것은 알고리즘 하나가 아니라 **방법**(또는 얼거리)이다. 곧 남은 그래프에서 늘림 경로를 찾아 그 길로 흐름을 흘려 보내고, 늘림 경로가 없어질 때까지 되풀이한다. 늘림 경로를 어떻게 찾을지는 정해 두지 않았다. 어떻게 고르느냐에 따라 성능 보장이 다른 여러 알고리즘(에드먼즈-카프 같은)이 나온다.

## 남은 그래프

이 방법을 말하기에 앞서 남은 그래프라는 생각이 필요한데, 이는 흐름을 더 보낼 수 있는 남은 담이를 담아낸다.

담이 함수가 $c$인 흐름 그물 $G = (V, E)$과 흐름 $f$이 주어질 때 변 $(u, v)$의 **남은 담이**는 다음과 같다:

$$
c_f(u, v) = c(u, v) - f(u, v)
$$

**남은 그래프** $G_f = (V, E_f)$은 남은 담이가 양수인 변을 담는다:

$$
E_f = \{(u, v) \in V \times V : c_f(u, v) > 0\}
$$

여기에는 앞으로 가는 변(담이가 남은 것)과 뒤로 가는 변(흐름을 물릴 수 있는 것)이 모두 든다. 변 $(u, v)$이 흐름 $f(u, v) > 0$을 나르면 남은 그래프에는 남은 담이가 $f(u, v)$인 거꿀 변 $(v, u)$이 들어 있으며, 이는 앞서 보낸 흐름을 "물릴" 수 있음을 나타낸다.

## 늘림 경로

**늘림 경로**는 남은 그래프 $G_f$에서 근원 $s$부터 바닥 $t$까지의 단순 경로이다. **병목 담이**(또는 경로의 남은 담이)는 그 경로를 따라 남은 담이의 최솟값이다:

$$
c_f(p) = \min_{(u,v) \in p} c_f(u, v)
$$

경로 $p$을 따라 늘린다는 것은 $p$의 앞으로 가는 변마다 흐름을 $c_f(p)$만큼 늘리고 뒤로 가는 변마다 $c_f(p)$만큼 줄이는 것이다.

## 그 방법

```text
FORD-FULKERSON(G, s, t):
    Initialize f(u,v) = 0 for all (u,v) in E
    while there exists an augmenting path p in G_f:
        c_f(p) = min { c_f(u,v) : (u,v) in p }
        for each edge (u,v) in p:
            if (u,v) is a forward edge:
                f(u,v) = f(u,v) + c_f(p)
            else:  // (u,v) is a backward edge
                f(v,u) = f(v,u) - c_f(p)
    return f
```

남은 그래프에 늘림 경로가 없으면 이 방법이 끝난다. 그때 흐름이 최대이다.

## 올바름

포드-풀커슨의 옳음은 **최대 흐름 최소 자름 정리**에서 따라 나온다. 곧 최대 흐름의 값은 최소 자름의 담이와 같다.

**정리(최대 흐름 최소 자름).** 다음은 서로 같다:

1. $f$은 $G$의 최대 흐름이다.
2. 남은 그래프 $G_f$에 늘림 경로가 없다.
3. $G$의 어떤 자름 $(S, T)$에 대해 $|f| = c(S, T)$이다.

*Proof sketch.* $(1 \Rightarrow 2)$: If an augmenting path existed, we could increase the flow, contradicting maximality. $(2 \Rightarrow 3)$: Define $S = \{v \in V : v \text{ is reachable from } s \text{ in } G_f\}$ and $T = V \setminus S$. Since $t \notin S$ (no augmenting path), $(S, T)$ is a cut. Every edge from $S$ to $T$ must be saturated, and every edge from $T$ to $S$ must carry zero flow, so $|f| = c(S, T)$. $(3 \Rightarrow 1)$: Since $|f| \leq c(S, T)$ for any cut (the weak duality bound), $|f| = c(S, T)$ implies $f$ is maximum. $\square$

## 복잡도

With integer capacities, the Ford-Fulkerson method terminates in at most $|f^*|$ augmentations, where $f^*$ is the maximum flow value, since each augmentation increases the flow by at least 1. Each augmentation requires $O(E)$ time to find the path (e.g., via DFS) and update the flow.

**Total time:** $O(|f^*| \cdot E)$

이 한계는 그래프 크기뿐 아니라 흐름 값에도 달렸다. 담이가 크면 아주 느릴 수 있다.

!!! warning "무리수 담이에서 끝나지 않음"
    변의 담이가 무리수이면 포드-풀커슨 방법이 끝나지 않을 수 있고 최대 흐름보다 작은 값으로 다가갈 수도 있다. 이런 병리적인 몸짓 때문에 너비 우선 돌아보기(에드먼즈-카프)나 다른 짜임새 있는 경로 고르기 전략을 쓰게 된다.

## 풀이 예제

Consider a network with vertices $\{s, a, b, t\}$:

| 변 | 담이 |
|------|----------|
| $(s, a)$ | 10 |
| $(s, b)$ | 8 |
| $(a, b)$ | 5 |
| $(a, t)$ | 7 |
| $(b, t)$ | 10 |

**Iteration 1.** Find augmenting path $s \to a \to t$. Bottleneck: $\min(10, 7) = 7$. Push flow 7.

1번째 바퀴 뒤: $f(s,a) = 7$, $f(a,t) = 7$. 남은 담이: $(s,a): 3$, $(a,t): 0$(꽉 참), $(a,s): 7$, $(t,a): 7$.

**Iteration 2.** Find path $s \to b \to t$. Bottleneck: $\min(8, 10) = 8$. Push flow 8.

2번째 바퀴 뒤: 여기에 더해 $f(s,b) = 8$, $f(b,t) = 8$. 남은 담이: $(s,b): 0$, $(b,t): 2$.

**Iteration 3.** Find path $s \to a \to b \to t$. Bottleneck: $\min(3, 5, 2) = 2$. Push flow 2.

3번째 바퀴 뒤: $f(s,a) = 9$, $f(a,b) = 2$, $f(b,t) = 10$.

**4번째 바퀴.** 늘림 경로가 없다. 최대 흐름 = $7 + 8 + 2 = 17$.

## 파이썬 구현

```python
"""
깊이 우선 돌아보기를 쓴 최대 흐름의 포드-풀커슨 방법.

깊이 우선 돌아보기로 남은 그래프에서 늘림 경로를 거듭 찾아
근원에서 바닥으로 가는 경로가 없어질 때까지 흐름을 흘려 보낸다.
"""

# === 늘림 경로를 찾는 깊이 우선 돌아보기 ===

def dfs(capacity: list, source: int, sink: int, visited: list,
        u: int, bottleneck: int) -> int:
    """
    깊이 우선 돌아보기로 늘림 경로를 찾아 병목 담이를 돌려준다.

    u에서 바닥으로 가는 늘림 경로가 없으면 0을 돌려준다.
    """
    if u == sink:
        return bottleneck
    visited[u] = True
    for v in range(len(capacity)):
        if not visited[v] and capacity[u][v] > 0:
            result = dfs(
                capacity, source, sink, visited,
                v, min(bottleneck, capacity[u][v])
            )
            if result > 0:
                capacity[u][v] -= result
                capacity[v][u] += result
                return result
    return 0


# === 주된 알고리즘 ===

def ford_fulkerson(n: int, edges: list, source: int, sink: int) -> int:
    """
    깊이 우선 돌아보기를 쓴 포드-풀커슨으로 최대 흐름을 셈한다.

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

    max_flow = 0
    while True:
        visited = [False] * n
        augment = dfs(capacity, source, sink, visited,
                      source, float('inf'))
        if augment == 0:
            break
        max_flow += augment

    return max_flow


# === 보기 ===

if __name__ == "__main__":
    # s=0, a=1, b=2, t=3
    edges = [
        (0, 1, 10),  # s -> a
        (0, 2, 8),   # s -> b
        (1, 2, 5),   # a -> b
        (1, 3, 7),   # a -> t
        (2, 3, 10),  # b -> t
    ]
    result = ford_fulkerson(4, edges, 0, 3)
    print(f"Maximum flow: {result}")
```

**출력:**

```
Maximum flow: 17
```

The algorithm finds the maximum flow of 17, matching the hand-traced example above. The $O(V^2)$ adjacency matrix representation used here is convenient for small dense graphs; for sparse graphs, an adjacency list with explicit edge objects is more space-efficient.

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. 26장.
- Ford, L. R., & Fulkerson, D. R. (1956). Maximal flow through a network. *Canadian Journal of Mathematics*, 8, 399-404.

## 연습문제

**연습문제 1.**
포드-풀커슨 방법을 설명하여라. 왜 "알고리즘"이 아니라 "방법"이라 부르는가?

??? success "연습문제 1 풀이"
    Ford-Fulkerson repeatedly finds augmenting paths from source to sink in the residual graph and pushes flow along them until no augmenting path exists. It is called a "method" because it does not specify how to find augmenting paths — any path-finding strategy works (DFS, BFS, random). Different strategies yield different algorithms (e.g., Edmonds-Karp uses BFS). The choice affects termination guarantees and time complexity. $\square$

---

**연습문제 2.**
남은 그래프와 남은 담이를 정의하여라. 포드-풀커슨에 왜 꼭 필요한가?

??? success "연습문제 2 풀이"
    The **residual graph** $G_f$ for flow $f$ has the same vertices as $G$. For each edge $(u,v)$ with capacity $c$ and flow $f(u,v)$: if $f(u,v) < c$, add forward edge $(u,v)$ with residual capacity $c - f(u,v)$. If $f(u,v) > 0$, add backward edge $(v,u)$ with residual capacity $f(u,v)$. Forward edges allow pushing more flow; backward edges allow canceling existing flow. The residual graph is essential because augmenting paths in $G_f$ correspond to valid flow increases in $G$. $\square$

---

**연습문제 3.**
담이가 모두 정수일 때 포드-풀커슨이 끝남을 증명하여라. 가장 나쁜 경우의 도는 시간은 무엇인가?

??? success "연습문제 3 풀이"
    Each augmentation pushes at least 1 unit of flow (since the bottleneck capacity is a positive integer). The maximum flow is at most $f^* \leq \sum_{(s,v)} c(s,v)$, which is finite. After at most $f^*$ augmentations, the maximum flow is reached. Each augmentation takes $O(E)$ (DFS or BFS). Total: $O(Ef^*)$. This can be large if $f^*$ is exponential in the input size (e.g., $f^* = 2^n$), which is why Edmonds-Karp's $O(VE^2)$ bound is preferred. $\square$

---

**연습문제 4.**
Construct an example where Ford-Fulkerson with DFS takes $O(Ef^*)$ time while Edmonds-Karp would be much faster.

??? success "연습문제 4 풀이"
    Consider: source $s$, sink $t$, two middle vertices $a, b$. Edges: $(s,a,1000), (s,b,1000), (a,t,1000), (b,t,1000), (a,b,1)$. DFS might alternately find paths $s \to a \to b \to t$ and $s \to b \to a \to t$, each pushing 1 unit through the bottleneck edge $(a,b)$. This requires 2000 iterations. Edmonds-Karp uses BFS, finding $s \to a \to t$ (push 1000) then $s \to b \to t$ (push 1000) in just 2 iterations, total flow 2000. $\square$
