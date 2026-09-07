# 존슨의 알고리즘

플로이드-워셜은 그래프의 빽빽함과 상관없이 $\Theta(V^3)$에 돈다. $E \ll V^2$인 **성긴** 그래프에서는 꼭짓점마다 데이크스트라를 돌리는 편이 빠를 텐데, 데이크스트라는 무게가 음이 아니어야 한다. 존슨의 알고리즘은 변의 무게를 **다시 매겨** 음의 무게를 없앤 뒤 꼭짓점마다 데이크스트라를 돌려 이 틈을 잇는다. 그 결과는 $O(V^2 \log V + VE)$에 도는 모든 짝 최단 경로 알고리즘으로, 성긴 그래프에서 플로이드-워셜보다 빠르다.

## 무게 다시 매기기 기법

핵심 통찰은 변의 무게마다 잘 고른 값을 더하면 어느 길이 최단인지는 그대로 두고 무게를 모두 음이 아니게 만들 수 있다는 것이다.

무게 함수 $w$과 **퍼텐셜 함수** $h: V \to \mathbb{R}$이 주어졌을 때 다시 매긴 변의 무게를 다음과 같이 정한다:

$$
\hat{w}(u, v) = w(u, v) + h(u) - h(v)
$$

이 다시 매기기는 최단 경로를 지킨다. 아무 길 $p = \langle v_0, v_1, \dots, v_k \rangle$에 대해 다음이 성립하기 때문이다:

$$
\hat{w}(p) = \sum_{i=0}^{k-1} \hat{w}(v_i, v_{i+1}) = \sum_{i=0}^{k-1} \left[w(v_i, v_{i+1}) + h(v_i) - h(v_{i+1})\right] = w(p) + h(v_0) - h(v_k)
$$

망원경처럼 접히는 합은 다시 매긴 길 무게가 원래 것과 끝점에만 기대는 상수 $h(v_0) - h(v_k)$만큼 다르다는 뜻이다. 그러므로 $w$ 아래의 최단 경로는 $\hat{w}$ 아래에서도 최단이다.

## 퍼텐셜 함수 고르기

다시 매긴 변을 모두 음이 아니게 하려고 존슨의 알고리즘은 $h(v) = \delta(s', v)$으로 놓는다. 여기서 $s'$은 무게 0인 변으로 기존 꼭짓점 모두에 이어진 새 꼭짓점이다:

1. 그래프에 새 꼭짓점 $s'$을 더한다.
2. 모든 $v \in V$에 대해 짐이 $0$인 변 $(s', v)$을 더한다.
3. $s'$에서 벨먼-포드를 돌려 $h(v) = \delta(s', v)$을 셈한다.

삼각 부등식에 따라 변 $(u, v)$마다 $\delta(s', v) \le \delta(s', u) + w(u, v)$이다. 옮겨 쓰면 다음과 같다:

$$
\hat{w}(u, v) = w(u, v) + h(u) - h(v) = w(u, v) + \delta(s', u) - \delta(s', v) \ge 0
$$

벨먼-포드가 음수 순환을 알아내면 알고리즘은 이를 알리고 멈춘다.

## 알고리즘의 걸음

```
JOHNSON(G, w):
    1. Add vertex s' and edges (s', v, 0) for all v in V
    2. Run BELLMAN-FORD(G', w, s')
       - If negative cycle detected: return "negative cycle"
       - Otherwise: h(v) = delta(s', v)
    3. For each edge (u, v) in E:
       w_hat(u, v) = w(u, v) + h(u) - h(v)
    4. For each vertex u in V:
       Run DIJKSTRA(G, w_hat, u) to get d_hat(u, v) for all v
       For each vertex v in V:
           d(u, v) = d_hat(u, v) - h(u) + h(v)
    5. Return distance matrix d
```

## 복잡도

| 걸음 | 시간 |
|---|---|
| $s'$에서의 벨먼-포드 | $O(VE)$ |
| 모든 변의 무게 다시 매기기 | $O(E)$ |
| 데이크스트라 $V$번 돌리기(이진 힙) | $O(V(V+E)\log V)$ |
| 거리 되돌리기 | $O(V^2)$ |
| **합계** | $O(V^2 \log V + VE)$ |

성긴 그래프($E = O(V)$)에서는 $O(V^2 \log V)$이 되어 플로이드-워셜의 $\Theta(V^3)$보다 훨씬 낫다. 빽빽한 그래프($E = O(V^2)$)에서는 두 알고리즘 모두 $\Theta(V^3)$이며, 플로이드-워셜이 더 단순하고 상수 인자가 작다.

## 어느 것을 언제 쓰나

| 잣대 | 플로이드-워셜 | 존슨 |
|---|---|---|
| 그래프의 빽빽함 | 빽빽함($E \approx V^2$) | 성김($E \ll V^2$) |
| 음의 변 | 받쳐 줌 | 받쳐 줌 |
| 시간 | $\Theta(V^3)$ | $O(V^2\log V + VE)$ |
| 구현 | 더 단순함 | 더 복잡함 |

## 풀이 예제

무게가 음인 변을 아우르는 꼭짓점 4개를 생각하자:

| 변 | 무게 |
|---|---|
| $(0, 1)$ | 1 |
| $(0, 2)$ | 4 |
| $(1, 2)$ | -3 |
| $(2, 3)$ | 2 |

**걸음 1:** 변 $(s', 0) = 0$, $(s', 1) = 0$, $(s', 2) = 0$, $(s', 3) = 0$과 함께 $s'$을 더한다.

**걸음 2:** $s'$에서 벨먼-포드: $h(0) = 0$, $h(1) = 0$, $h(2) = -3$, $h(3) = -1$.

**걸음 3:** 무게를 다시 매긴다.
$\hat{w}(0,1) = 1 + 0 - 0 = 1$.
$\hat{w}(0,2) = 4 + 0 - (-3) = 7$.
$\hat{w}(1,2) = -3 + 0 - (-3) = 0$.
$\hat{w}(2,3) = 2 + (-3) - (-1) = 0$.

다시 매긴 변은 모두 음이 아니다.

**걸음 4:** 다시 매긴 변으로 꼭짓점마다 데이크스트라를 돌린 뒤 되돌린다. 곧 $d(u, v) = \hat{d}(u, v) - h(u) + h(v)$.

## 구현

```python
"""
모든 짝 최단 경로를 위한 존슨 알고리즘.

벨먼-포드의 무게 다시 매기기와 데이크스트라를 어우러지게 하여
O(V^2 log V + VE) 시간이 걸리며, 성긴 그래프에서 플로이드-워셜을 이긴다.
"""

import heapq
from math import inf


# === 퍼텐셜 셈하기를 위한 벨먼-포드 ==========================================

def bellman_ford(vertices: list, edges: list, source) -> tuple[dict, bool]:
    """벨먼-포드를 돌려 거리와 순환 상태 돌려주기.

    음의 순환이 없으면 (dist, True)을, 아니면 (dist, False)을 돌려준다.
    """
    dist = {v: inf for v in vertices}
    dist[source] = 0

    for _ in range(len(vertices) - 1):
        for u, v, w in edges:
            if dist[u] != inf and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w

    # 음의 순환 살피기
    for u, v, w in edges:
        if dist[u] != inf and dist[u] + w < dist[v]:
            return dist, False

    return dist, True


# === 데이크스트라 ============================================================

def dijkstra(graph: dict, source) -> dict:
    """근원에서 데이크스트라를 돌려 최단 거리 돌려주기."""
    dist = {v: inf for v in graph}
    dist[source] = 0
    pq = [(0, source)]

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(pq, (dist[v], v))

    return dist


# === 존슨 알고리즘 ===========================================================

def johnson(vertices: list, edges: list) -> tuple[dict, bool]:
    """존슨 알고리즘으로 모든 짝의 최단 경로 셈하기.

    매개변수
    ----------
    vertices : list
        모든 꼭짓점 이름.
    edges : list of (u, v, w)
        무게 있는 방향 변.

    반환값
    -------
    dist : dict of dict
        dist[u][v] = u에서 v까지 최단 경로의 무게.
    no_negative_cycle : bool
        음의 순환이 없으면 True.
    """
    # 걸음 1: 모든 꼭짓점으로 무게 0인 변을 갖는 가상 근원 s' 더하기
    s_prime = "__s_prime__"
    aug_vertices = vertices + [s_prime]
    aug_edges = edges + [(s_prime, v, 0) for v in vertices]

    # 걸음 2: s'에서 벨먼-포드를 돌려 퍼텐셜 h 얻기
    h, ok = bellman_ford(aug_vertices, aug_edges, s_prime)
    if not ok:
        return {}, False

    # 걸음 3: 변의 무게 다시 매기기
    reweighted_graph = {v: [] for v in vertices}
    for u, v, w in edges:
        w_hat = w + h[u] - h[v]
        reweighted_graph[u].append((v, w_hat))

    # 걸음 4: 꼭짓점마다 데이크스트라를 돌리고 무게 되돌리기
    dist = {}
    for u in vertices:
        d_hat = dijkstra(reweighted_graph, u)
        dist[u] = {}
        for v in vertices:
            if d_hat[v] == inf:
                dist[u][v] = inf
            else:
                dist[u][v] = d_hat[v] - h[u] + h[v]

    return dist, True


# === 보임 ====================================================================

if __name__ == "__main__":
    vertices = [0, 1, 2, 3]
    edges = [
        (0, 1, 1), (0, 2, 4),
        (1, 2, -3),
        (2, 3, 2),
    ]

    dist, ok = johnson(vertices, edges)
    print(f"No negative cycle: {ok}")
    print("\nAll-pairs shortest distances:")
    for u in vertices:
        row = {v: dist[u][v] if dist[u][v] != inf else "inf" for v in vertices}
        print(f"  From {u}: {row}")
```

**출력:**

```
No negative cycle: True

All-pairs shortest distances:
  From 0: {0: 0, 1: 1, 2: -2, 3: 0}
  From 1: {0: 'inf', 1: 0, 2: -3, 3: -1}
  From 2: {0: 'inf', 1: 'inf', 2: 0, 3: 2}
  From 3: {0: 'inf', 1: 'inf', 2: 'inf', 3: 0}
```

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to
  Algorithms* (4th ed.), Chapter 25.3: Johnson's Algorithm for Sparse Graphs.
- Johnson, D. B. (1977). Efficient algorithms for shortest paths in sparse networks. *Journal of the ACM*, 24(1), 1-13.

## 연습문제

**연습문제 1.**
존슨 알고리즘의 무게 다시 매기기 걸음을 설명하여라. 퍼텐셜 함수 $h(v)$을 더하면 왜 최단 경로가 지켜지는가?

??? success "연습문제 1 풀이"
    존슨의 알고리즘은 새 무게를 $\hat{w}(u,v) = w(u,v) + h(u) - h(v)$으로 정한다. 여기서 $h(v) = \delta(s, v)$은 (벨먼-포드로 셈한) 보조 샘 $s$에서 $v$까지의 최단 경로 거리이다. $u$에서 $v$까지의 아무 길 $p$에 대해 새 무게는 $\hat{w}(p) = w(p) + h(u) - h(v)$이다. $h(u) - h(v)$은 모든 $u$-$v$ 길에 대해 상수이므로 $w(p)$이 가장 작은 길이 $\hat{w}(p)$도 가장 작다. 삼각 부등식 $h(v) \leq h(u) + w(u,v)$이 $\hat{w}(u,v) \geq 0$을 보장하여 데이크스트라를 쓸 수 있게 한다. $\square$

---

**연습문제 2.**
존슨 알고리즘의 시간 복잡도는 얼마인가? 걸음별로 나누어 보여라.

??? success "연습문제 2 풀이"
    (1) 보조 꼭짓점과 무게 0인 변 더하기: $O(V)$. (2) 보조 꼭짓점에서의 벨먼-포드: $O(VE)$. (3) 모든 변의 무게 다시 매기기: $O(E)$. (4) 이진 힙으로 꼭짓점마다 데이크스트라 돌리기: $O(V(V + E)\log V)$. (5) 거리 바로잡기: $O(V^2)$. 합계: $O(VE + V(V+E)\log V)$. 성긴 그래프($E = O(V)$)에서는 $O(V^2 \log V)$으로 플로이드-워셜의 $O(V^3)$보다 훨씬 낫다. $\square$

---

**연습문제 3.**
모든 변의 무게에 큰 상수를 더해 음이 아니게 만든 뒤 데이크스트라를 돌리면 왜 안 되는가?

??? success "연습문제 3 풀이"
    변마다 상수 $C$을 더하면 길의 무게가 $C \times (\text{길 안 변의 개수})$만큼 바뀐다. 변이 많은 길일수록 더 크게 벌을 받아 어느 길이 최단인지가 바뀔 수 있다. 이를테면 무게 1인 변 5개의 길(합 5)과 무게 4인 변 2개의 길(합 8)에 $C = 10$을 더하면 합이 $55$과 $28$이 되어 차례가 뒤집힌다. 존슨의 무게 다시 매기기는 어느 길에서나 서로 지워지는 꼭짓점 퍼텐셜을 써서 이를 피한다. $\square$

---

**연습문제 4.**
그래프에 무게가 음인 고리가 있으면 존슨의 알고리즘은 무너진다. 이를 어떻게 알아내며 알고리즘은 무엇을 알려야 하는가?

??? success "연습문제 4 풀이"
    벨먼-포드 걸음(걸음 2)이 무게가 음인 고리를 알아낸다. $V - 1$바퀴 늦춘 뒤에도 늦출 수 있는 변이 있으면 음의 고리가 있다. 벨먼-포드가 `False`을 되돌리고, 존슨의 알고리즘은 (음의 고리를 지나는 길은 얼마든지 짧게 만들 수 있으므로) 최단 경로가 정해지지 않는다고 알려야 한다. 알고리즘은 데이크스트라 단계로 가지 않고 바로 멈춘다. $\square$
