# k개의 최단 경로

여러 쓰임새에서 최단 경로 하나만 아는 것으로는 모자라다. 길 안내 얼개는 막힘을 피할 다른 길을 내놓을 수 있다. 망은 고장에 견디려고 예비 길이 필요할 수 있다. **$k$개의 최단 경로 문제**는 샘 $s$에서 과녁 $t$까지 값의 합으로 차례 매긴 $k$개의 최단 경로를 구한다. 판에 따라 길이 변을 함께 쓸 수도 있고(고리 있는 판), 변이 겹치지 않거나 꼭짓점이 되풀이되지 않아야 할 수도 있다.

## 문제의 여러 판

- **$k$개의 최단 단순 길**: 길마다 꼭짓점을 두 번 넘게 들르지 않는다. 일반으로 NP-어려움이지만 $k$이 알맞으면 옌의 알고리즘으로 감당할 만하다.
- **$k$개의 최단 걸음**: 꼭짓점과 변을 되풀이할 수 있다. 엡스타인의 알고리즘으로 다항 시간에 풀린다.

실전 알고리즘 대부분은 단순 길 판에 초점을 맞춘다.

## 옌의 알고리즘

옌의 알고리즘(1971)은 $s$에서 $t$까지 $k$개의 최단 단순(고리 없는) 길을 찾는다:

1. 데이크스트라로 최단 경로 $P_1$을 찾는다.
2. $i = 2, 3, \ldots, k$에 대해:
    - $P_{i-1}$의 꼭짓점 $v_j$(갈림 마디)마다, $v_j$까지 앞부분이 같은, 앞서 찾은 길이 쓴 변을 잠시 지운다.
    - 고친 그래프에서 $v_j$에서 $t$까지의 최단 경로(갈림 길)를 찾는다.
    - 뿌리 길($s$에서 $v_j$까지)과 갈림 길을 이어 붙여 후보를 얻는다.
    - 후보를 모두 최소 힙에 넣는다.
    - 값이 가장 작은 후보를 꺼내 $P_i$으로 삼는다.

## 옌 알고리즘의 복잡도

$k$번의 되풀이마다 데이크스트라를 많아야 $|P|$번 돌린다($|P|$은 길의 길이):

$$
O(kn(m + n \log n))
$$

여기서 $n = |V|$이고 $m = |E|$이다. 이는 피보나치 힙을 쓴 데이크스트라를 놓고 한 것이다.

## 엡스타인의 알고리즘

엡스타인의 알고리즘(1998)은 (꼭짓점 되풀이를 허락하는) $k$개의 최단 걸음을 다음 시간에 찾는다:

$$
O(m + n \log n + k \log k)
$$

**길 그래프**를 써서 모든 길을 간결한 속뜻 표현으로 쌓은 뒤 차례대로 꺼낸다. 옌의 알고리즘보다 훨씬 빠르지만 단순 길을 보장하지는 않는다.

## 구현

```python
"""
옌 알고리즘으로 구하는 k개의 짧은 단순 경로.

근원에서 과녁까지 고리 없는 짧은 경로 k개를 찾되,
앞서 찾은 경로에서 되풀이해 벗어나며 찾는다.
단일 근원 최단 경로에 데이크스트라 알고리즘을 쓴다.
"""

import heapq
from collections import defaultdict


# === 데이크스트라 알고리즘 ===

def dijkstra(graph: dict, source: int, target: int,
             blocked_edges: set = None,
             blocked_nodes: set = None) -> tuple:
    """근원에서 과녁까지의 최단 경로 찾기.

    (값, 경로)을 돌려주거나, 경로가 없으면 (float('inf'), [])을 돌려준다.
    """
    if blocked_edges is None:
        blocked_edges = set()
    if blocked_nodes is None:
        blocked_nodes = set()

    dist = {source: 0}
    prev = {source: None}
    heap = [(0, source)]

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist.get(u, float('inf')):
            continue
        if u == target:
            break
        for v, w in graph.get(u, []):
            if v in blocked_nodes or (u, v) in blocked_edges:
                continue
            new_dist = d + w
            if new_dist < dist.get(v, float('inf')):
                dist[v] = new_dist
                prev[v] = u
                heapq.heappush(heap, (new_dist, v))

    if target not in dist:
        return float('inf'), []

    path = []
    node = target
    while node is not None:
        path.append(node)
        node = prev[node]
    return dist[target], path[::-1]


# === 옌 알고리즘 ===

def yen_k_shortest(graph: dict, source: int, target: int,
                   k: int) -> list:
    """옌 알고리즘으로 k개의 짧은 단순 경로 찾기.

    값으로 정렬한 (값, 경로) 짝의 목록을 돌려준다.
    """
    # 첫 최단 경로
    cost, path = dijkstra(graph, source, target)
    if not path:
        return []

    a_paths = [(cost, path)]
    candidates = []
    candidate_set = set()

    for i in range(1, k):
        prev_path = a_paths[i - 1][1]

        for j in range(len(prev_path) - 1):
            spur_node = prev_path[j]
            root_path = prev_path[:j + 1]
            root_cost = 0
            for idx in range(len(root_path) - 1):
                u, v = root_path[idx], root_path[idx + 1]
                for nb, w in graph.get(u, []):
                    if nb == v:
                        root_cost += w
                        break

            # 같은 뿌리를 나눠 갖는 경로의 변 막기
            blocked_edges = set()
            for _, p in a_paths:
                if p[:j + 1] == root_path and j + 1 < len(p):
                    blocked_edges.add((p[j], p[j + 1]))

            blocked_nodes = set(root_path[:-1])

            spur_cost, spur_path = dijkstra(
                graph, spur_node, target,
                blocked_edges, blocked_nodes
            )

            if spur_path:
                total_path = root_path[:-1] + spur_path
                total_cost = root_cost + spur_cost
                path_tuple = tuple(total_path)
                if path_tuple not in candidate_set:
                    candidate_set.add(path_tuple)
                    heapq.heappush(candidates, (total_cost, total_path))

        if not candidates:
            break

        next_cost, next_path = heapq.heappop(candidates)
        a_paths.append((next_cost, next_path))

    return a_paths


# === 시연 ===

if __name__ == "__main__":
    # 작은 무게 그래프 세우기(이웃 목록)
    graph = defaultdict(list)
    edges = [
        (0, 1, 1), (0, 2, 5), (1, 2, 2), (1, 3, 6),
        (2, 3, 2), (2, 4, 7), (3, 4, 1), (0, 3, 8)
    ]
    for u, v, w in edges:
        graph[u].append((v, w))

    print("Graph edges:", edges)
    print()

    k = 4
    paths = yen_k_shortest(graph, 0, 4, k)
    print(f"Top {k} shortest paths from 0 to 4:")
    for i, (cost, path) in enumerate(paths, 1):
        print(f"  #{i}: cost={cost}, path={path}")
```

**출력:**
```
Graph edges: [(0, 1, 1), (0, 2, 5), (1, 2, 2), (1, 3, 6), (2, 3, 2), (2, 4, 7), (3, 4, 1), (0, 3, 8)]

Top 4 shortest paths from 0 to 4:
  #1: 값=6, 경로=[0, 1, 2, 3, 4]
  #2: 값=8, 경로=[0, 1, 3, 4]
  #3: 값=8, 경로=[0, 2, 3, 4]
  #4: 값=9, 경로=[0, 3, 4]
```

## 비교

| 알고리즘 | 찾는 길 | 시간 | 단순 길인가? |
|-----------|-------------|------|---------------|
| 옌(1971) | $k$개의 최단 | $O(kn(m + n \log n))$ | 예 |
| 엡스타인(1998) | $k$개의 최단 | $O(m + n \log n + k \log k)$ | 아니오(걸음) |
| 롤러(1972) | $k$개의 최단 | $O(kn(m + n \log n))$ | 예 |

## 참고 문헌

- Yen, J. Y. (1971). Finding the $K$ shortest loopless paths in a network. *Management Science*, 17(11), 712-716.
- Eppstein, D. (1998). Finding the $k$ shortest paths. *SIAM Journal on Computing*, 28(2), 652-673.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press.

## 연습문제

**연습문제 1.**
$K$개의 최단 고리 없는 길을 찾는 옌의 알고리즘을 설명하여라. 시간 복잡도는 얼마인가?

??? success "연습문제 1 풀이"
    옌의 알고리즘은 다음과 같이 $k$번째 최단 경로를 되풀이해 찾는다. (1) $(k-1)$번째 최단 경로 위의 변마다 그 변과 해당 갈림 마디의 앞선 것들을 잠시 지운 뒤, 갈림 마디에서 과녁까지 데이크스트라를 돌린다. (2) 후보 갈림 길을 모두 모아 가장 짧은 것을 고른다. (3) $k = 2, \ldots, K$에 대해 되풀이한다. 시간 복잡도는 $O(KV(V + E) \log V)$이다. $K$번의 되풀이마다 데이크스트라를 $V$번까지 돌리기 때문이다. "고리 없음" 제약이 $K$개의 최단 걸음을 찾는 것보다 문제를 어렵게 만든다. $\square$

---

**연습문제 2.**
어떤 쓰임새가 최단 경로 하나가 아니라 $K$개의 최단 경로를 왜 필요로 하겠는가?

??? success "연습문제 2 풀이"
    실전에서의 까닭은 다음과 같다. (1) **다른 길 제안**: GPS 길 안내는 여러 길(가장 빠른, 가장 짧은, 경치 좋은)을 준다. (2) **튼튼함**: 최단 경로가 막히면 둘째로 짧은 길이 물러설 자리가 된다. (3) **여러 갈래**: 망 라우팅에서 여러 길로 통행을 나누면 짐이 고르게 퍼진다. (4) **어림**: 말소리 알아듣기나 자연어 처리에서 $K$개의 가장 좋은 가설을 더 비싼 모형으로 다시 차례 매긴다. (5) **뜯어보기**: 길의 성격(거리, 도는 횟수, 통행료)을 견주면 이용자가 알고 고를 수 있다. $\square$

---

**연습문제 3.**
$K$개의 최단 단순 길과 $K$개의 최단 걸음을 갈라 설명하여라. 어느 쪽이 셈하기 더 어려운가?

??? success "연습문제 3 풀이"
    **단순 길**은 꼭짓점마다 많아야 한 번 들른다. **걸음**은 꼭짓점을 다시 들를 수 있다. $K$개의 최단 걸음은 엡스타인의 알고리즘으로 $O(E + KV \log V)$에 찾을 수 있는데, 걸음이 앞부분을 함께 쓸 수 있어 효율적이다. $K$개의 최단 단순 길은 더 어렵다. 옌의 알고리즘은 $O(KV(V+E)\log V)$이 들고, $K$이 크면 이 문제는 NP-어려움 문제와 이어진다. 실전에서는 고리 있는 걸음이 뜻이 닿는 일이 드물어 $K$개의 최단 단순 길이 더 쓸모 있다. $\square$

---

**연습문제 4.**
$K$개의 최단 경로를 찾는 데 A*을 쓸 수 있는가? 그 길을 밝혀라.

??? success "연습문제 4 풀이"
    쓸 수 있다. A*을 돌리되 목표를 처음 꺼냈을 때 멈추지 말고 목표를 $K$번 꺼낼 때까지 이어 간다. $k$번째로 꺼낸 것이 $k$번째 최단 경로이다. 이렇게 하면 $K$개의 최단 걸음(반드시 단순 길은 아니다)을 찾는다. 맞음을 보장하려면 꼭짓점을 여러 번 넓힐 수 있게 해야 한다. 시간 복잡도는 $O(KE + KV\log(KV))$이다. 단순 길을 원하면 고리를 피하려고 장부를 더 적어야 해서 문제가 어려워진다. $\square$
