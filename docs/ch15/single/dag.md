# DAG 최단 경로

그래프에 고리가 없으면 최단 경로를 셈하기가 훨씬 단순해진다. 방향 비순환 그래프(DAG)는 꼭짓점의 위상 차례를 가지며, 그 차례로 꼭짓점을 다루면 꼭짓점 $v$에 이르기 전에 $v$의 앞선 꼭짓점이 모두 확정되어 있음이 보장된다. 그러면 데이크스트라의 우선순위 줄도 벨먼-포드의 되풀이 훑기도 필요 없어져, 무게가 음인 변도 어렵지 않게 다루는 깔끔한 $O(V + E)$ 알고리즘이 나온다.

## 위상 차례가 왜 되는가

DAG에서 위상 정렬은 변 $(v_i, v_j)$마다 $i < j$이 되도록 모든 꼭짓점을 줄 세운 차례 $v_1, v_2, \dots, v_n$을 낸다. 곧 꼭짓점 $v_j$을 다룰 때 최단 경로 위에서 $v_j$의 앞선 꼭짓점이 될 수 있는 꼭짓점은 모두 이미 다뤄졌고 그 거리가 확정되었다는 뜻이다.

**모임 성질**에 따라 변 $(u, v)$을 늦출 때 $d[u] = \delta(s, u)$이면 그 뒤에 $d[v] = \delta(s, v)$이다. 위상 차례가 변마다 바로 이 앞선 조건을 보장한다.

## 알고리즘

```
DAG-SHORTEST-PATHS(G, w, s):
    topological sort G
    INITIALIZE-SINGLE-SOURCE(G, s)
    for each vertex u in topological order:
        for each edge (u, v) in Adj[u]:
            RELAX(u, v, w)
```

Each vertex is processed exactly once, and each edge is relaxed exactly once.

## 올바름

!!! note "맞음 정리"
    `DAG-SHORTEST-PATHS`이 멈춘 뒤 모든 $v \in V$에 대해 $d[v] = \delta(s, v)$이다.

**증명.** $p = \langle v_0, v_1, \dots, v_k \rangle$을 $s = v_0$에서 어떤 꼭짓점 $v_k$까지의 최단 경로라 하자. 위상 차례에서 $v_0$은 $v_1$보다 앞에, $v_1$은 $v_2$보다 앞에 나오고 이런 식으로 이어진다. 그러므로 변 $(v_0, v_1)$이 $(v_1, v_2)$보다 먼저, $(v_1, v_2)$이 $(v_2, v_3)$보다 먼저 늦춰지고 이런 식으로 이어진다.

길 늦추기 성질에 따라 $p$의 변을 모두 늦춘 뒤 $d[v_k] = \delta(s, v_k)$이다. $\square$

## 복잡도

- **시간:** 위상 정렬에 $O(V + E)$이 든다. 주된 되풀이가 꼭짓점마다 한 번, 변마다 한 번 다루므로 역시 $O(V + E)$이다. 합계: $O(V + E)$.
- **공간:** 거리 배열과 앞선 것 배열에 $O(V)$, 그래프 표현에 $O(V + E)$.

이는 점근으로 가장 좋다. 어떤 알고리즘이든 변마다 적어도 한 번은 살펴야 하기 때문이다.

## 다른 알고리즘과의 견줌

| 알고리즘 | 음의 무게를 다루나 | DAG이 필요한가 | 시간 |
|---|---|---|---|
| DAG 최단 경로 | 예 | 예 | $O(V + E)$ |
| 데이크스트라 | 아니오 | 아니오 | $O((V+E)\log V)$ |
| 벨먼-포드 | 예 | 아니오 | $O(VE)$ |

The DAG algorithm is the fastest but applies only to acyclic graphs.

## DAG에서 가장 긴 길

쓸모 있는 변형 하나. DAG에서 **가장 긴 길**을 찾으려면 변의 무게를 모두 음으로 뒤집고 DAG 최단 경로를 돌린다. 아니면 늦추기 조건을 $d[v] < d[u] + w(u, v)$으로 바꾸고 거리를 $-\infty$으로 첫걸음 잡는다. 이는 일감 일정 짜기(PERT/CPM)의 임계 경로 분석에 쓸모 있다.

## 풀이 예제

Consider the DAG with vertices in topological order $\langle s, a, b, c, d, e \rangle$:

| 변 | 무게 |
|---|---|
| $(s, a)$ | 5 |
| $(s, b)$ | 3 |
| $(a, b)$ | 2 |
| $(a, c)$ | 6 |
| $(b, c)$ | 7 |
| $(b, d)$ | 4 |
| $(c, d)$ | -1 |
| $(c, e)$ | 1 |
| $(d, e)$ | -2 |

**Processing $s$:** Relax $(s, a)$: $d[a] = 5$.  Relax $(s, b)$: $d[b] = 3$.

**$a$ 다루기:** $(a, b)$ 늦추기: $d[b] = \min(3, 5+2) = 3$(바뀜 없음). $(a, c)$ 늦추기: $d[c] = 11$.

**$b$ 다루기:** $(b, c)$ 늦추기: $d[c] = \min(11, 3+7) = 10$. $(b, d)$ 늦추기: $d[d] = 7$.

**$c$ 다루기:** $(c, d)$ 늦추기: $d[d] = \min(7, 10-1) = 7$(바뀜 없음). $(c, e)$ 늦추기: $d[e] = 11$.

**Processing $d$:** Relax $(d, e)$: $d[e] = \min(11, 7-2) = 5$.

**Final distances:** $d[s]=0, d[a]=5, d[b]=3, d[c]=10, d[d]=7, d[e]=5$.

## 구현

```python
"""
위상 정렬을 쓰는 유향 비순환 그래프의 최단 경로.

방향 비순환 그래프에서 단일 근원 최단 경로를 O(V + E) 시간에 셈하며,
음의 무게 변을 올바로 다룬다.
"""

from math import inf
from collections import defaultdict, deque


# === 위상 정렬(칸 알고리즘) ==================================================

def topological_sort(graph: dict, vertices: list) -> list:
    """칸 알고리즘으로 위상 차례의 꼭짓점 돌려주기.

    매개변수
    ----------
    graph : dict
        꼭짓점 -> (이웃, 무게) 목록으로 잇는 이웃 목록.
    vertices : list
        모든 꼭짓점 이름.

    반환값
    -------
    list
        위상 차례로 늘어놓은 꼭짓점.

    일으키는 예외
    ------
    ValueError
        그래프에 순환이 있으면.
    """
    in_degree = defaultdict(int)
    for v in vertices:
        in_degree[v]  # 꼭짓점이 모두 나오도록 하기
    for u in graph:
        for v, _ in graph[u]:
            in_degree[v] += 1

    queue = deque(v for v in vertices if in_degree[v] == 0)
    order = []
    while queue:
        u = queue.popleft()
        order.append(u)
        for v, _ in graph.get(u, []):
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    if len(order) != len(vertices):
        raise ValueError("Graph contains a cycle")
    return order


# === 유향 비순환 그래프의 최단 경로 ==========================================

def dag_shortest_paths(graph: dict, vertices: list, source) -> tuple[dict, dict]:
    """유향 비순환 그래프에서 근원으로부터의 최단 경로 셈하기.

    매개변수
    ----------
    graph : dict
        꼭짓점 -> (이웃, 무게) 목록으로 잇는 이웃 목록.
    vertices : list
        모든 꼭짓점 이름.
    source : hashable
        근원 꼭짓점.

    반환값
    -------
    dist : dict
        근원에서의 최단 거리.
    pred : dict
        경로를 되짚기 위한 앞선 꼭짓점 가리개.
    """
    order = topological_sort(graph, vertices)

    # 초기화한다
    dist = {v: inf for v in vertices}
    dist[source] = 0
    pred = {v: None for v in vertices}

    # 위상 차례로 변 늦추기
    for u in order:
        if dist[u] == inf:
            continue  # u에 닿을 수 없으므로 건너뜀
        for v, w in graph.get(u, []):
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                pred[v] = u

    return dist, pred


# === 경로 되짚기 =============================================================

def get_path(pred: dict, source, target) -> list:
    """근원에서 과녁까지의 최단 경로 되짚기."""
    path = []
    v = target
    while v is not None:
        path.append(v)
        v = pred[v]
    path.reverse()
    return path if path and path[0] == source else []


# === 보임 ====================================================================

if __name__ == "__main__":
    vertices = ["s", "a", "b", "c", "d", "e"]
    graph = {
        "s": [("a", 5), ("b", 3)],
        "a": [("b", 2), ("c", 6)],
        "b": [("c", 7), ("d", 4)],
        "c": [("d", -1), ("e", 1)],
        "d": [("e", -2)],
        "e": [],
    }

    dist, pred = dag_shortest_paths(graph, vertices, "s")
    print(f"Distances: {dist}")
    print(f"Path s->e: {get_path(pred, 's', 'e')}")
    print(f"Path s->c: {get_path(pred, 's', 'c')}")
    print(f"Path s->d: {get_path(pred, 's', 'd')}")
```

**출력:**

```
Distances: {'s': 0, 'a': 5, 'b': 3, 'c': 10, 'd': 7, 'e': 5}
Path s->e: ['s', 'b', 'd', 'e']
Path s->c: ['s', 'b', 'c']
Path s->d: ['s', 'b', 'd']
```

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to
  Algorithms* (4th ed.), Chapter 24.2: Single-Source Shortest Paths in DAGs.

## 연습문제

**연습문제 1.**
DAG 최단 경로 알고리즘이 왜 벨먼-포드와 데이크스트라보다 빠른가? DAG의 어떤 성질이 이를 가능하게 하는가?

??? success "연습문제 1 풀이"
    DAG 알고리즘은 $O(V + E)$, 곧 선형 시간에 돈다. DAG에 위상 차례가 있기에 가능하다. 곧 그 차례로 꼭짓점을 다루면 꼭짓점 $v$을 다룰 때 $v$의 앞선 꼭짓점이 모두 이미 확정되어 있음이 보장된다. 변마다 꼭 한 번 늦춰진다. 벨먼-포드는 위상 차례가 없어 모든 변을 되풀이해 늦춰야 하므로 $O(VE)$이 든다. 데이크스트라는 다음 꼭짓점을 고르는 데 우선순위 줄이 필요해 $O((V+E)\log V)$이 든다. DAG에 고리가 없다는 점이 이 두 필요를 모두 없앤다. $\square$

---

**연습문제 2.**
DAG 최단 경로 알고리즘이 음의 변 무게를 다룰 수 있는가? 왜 그런가?

??? success "연습문제 2 풀이"
    다룰 수 있다. DAG 알고리즘은 최소 거리가 아니라 위상 차례로 꼭짓점을 다루므로 음의 무게를 맞게 다룬다. 꼭짓점 $v$을 다룰 때 들어오는 변이 모두 이미 늦춰졌으므로, 변의 무게가 양이든 음이든 $d[v]$이 맞다. 이는 음의 무게에서 무너지는 데이크스트라 알고리즘에 견준 핵심 이점이다. $\square$

---

**연습문제 3.**
DAG에서 가장 긴 길을 찾는 법을 밝혀라. 이 문제가 일반 그래프에서는 왜 NP-어려움이고 DAG에서는 왜 다항 시간인가?

??? success "연습문제 3 풀이"
    변의 무게를 모두 음으로 뒤집고 DAG 최단 경로 알고리즘을 돌린 뒤 결과를 다시 뒤집는다. 아니면 늦추기를 $\min$ 대신 $\max$을 쓰도록 고친다. 곧 $d[v] = \max(d[v], d[u] + w(u,v))$이다. $d[s] = 0$으로, $v \neq s$이면 $d[v] = -\infty$으로 첫걸음 잡는다. 이는 $O(V + E)$에 돈다. 일반 그래프에서 가장 긴 길은 (가장 긴 단순 길 찾기로 줄어드는) 해밀턴 길을 담을 수 있어 NP-어려움이다. DAG은 위상 차례가 꼭짓점을 다시 들르는 것을 막아 "단순 길" 제약이 저절로 지켜지므로 이를 피한다. $\square$

---

**연습문제 4.**
어떤 일감에 기댐이 있는 과제들이 있다(DAG). 과제마다 걸리는 시간이 있다. DAG 최단 경로 알고리즘으로 임계 경로(시작에서 끝까지 가장 긴 길)를 찾는 법을 밝혀라.

??? success "연습문제 4 풀이"
    과제마다 걸리는 시간을 무게로 갖는 꼭짓점으로 본뜬다. 기댐마다 변을 더한다. 앞서 할 것이 없는 과제 모두에 이어진 가상 샘(걸리는 시간 0)과, 뒤따를 것이 없는 과제 모두에서 이어지는 가상 웅덩이를 더한다. 무게를 음으로 뒤집은 DAG 최단 경로 알고리즘(또는 최대 늦추기 판)으로 샘에서 웅덩이까지 가장 긴 길을 찾는다. 임계 경로가 일감을 마치는 최소 시간을 정한다. 이 길 위의 과제는 늦추면 일감 전체가 늦어진다. 시간 복잡도: $O(V + E)$. $\square$
