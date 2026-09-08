# 데이크스트라 알고리즘의 맞음

데이크스트라 알고리즘은 잠정 거리가 가장 작은 꼭짓점을 욕심껏 꺼내 그 나가는 변을 늦추며 최단 경로를 셈한다. 이 욕심 전략이 맞다는 것은 뻔하지 않다. 어림값이 가장 작은 꼭짓점이 왜 실제로 올바른 최단 경로 거리를 갖는가? 이 쪽은 변 무게가 음이 아니라는 점과 늦추기의 위 한계 성질에 기댄 엄밀한 증명을 보인다.

---

## 1. 알고리즘 되짚기

데이크스트라 알고리즘은 최단 경로 거리가 확정된 꼭짓점의 묶음 $S$을 지킨다. 걸음마다 $d[u]$이 가장 작은 꼭짓점 $u \in V \setminus S$을 꺼내 $S$에 넣고 $u$에서 나가는 변을 모두 늦춘다.

```
DIJKSTRA(G, w, s):
    INITIALIZE-SINGLE-SOURCE(G, s)
    S = {}
    Q = V            // min-priority queue keyed by d[v]
    while Q is not empty:
        u = EXTRACT-MIN(Q)
        S = S ∪ {u}
        for each edge (u, v) in Adj[u]:
            RELAX(u, v, w)
```

---

## 2. 맞음 정리

!!! note "정리: 데이크스트라의 맞음"
    변의 무게가 모두 음이 아니면(모든 $(u, v) \in E$에 대해 $w(u, v) \ge 0$이면), 꼭짓점 $u$을 우선순위 줄에서 꺼낼 때마다 $d[u] = \delta(s, u)$이다.

---

## 3. 어긋냄으로 증명

**증명.** 어긋냄을 위해 $d[u] \ne \delta(s, u)$인, $S$에 들어간 **첫 꼭짓점**을 $u$이라 하자.

**걸음 1: $u \ne s$.** 샘 $s$이 $d[s] = 0 = \delta(s, s)$으로 먼저 들어가므로 $u$은 다른 꼭짓점이다.

**걸음 2: $u$에 닿을 수 있다.** $u$이 $d[u] < \infty$으로 줄에서 꺼내지므로(아니라면 조건 $d[u] \ne \delta(s, u)$이 $\delta(s, u) < \infty$을 뜻해 길이 있다는 말이 된다) $s$에서 $u$까지의 최단 경로가 있다.

**걸음 3: 결정적인 변 찾기.** $p$을 $s$에서 $u$까지의 최단 경로라 하자. $u$이 막 들어가려는 그때 $x \in S$이고 $y \notin S$인 $p$ 위의 첫 변 $(x, y)$을 생각하자. $s \in S$이고 $u \notin S$이므로 그런 변이 반드시 있다.

$$
s \xrightarrow{p_1} x \to y \xrightarrow{p_2} u
$$

**걸음 4: $d[y] = \delta(s, y)$임을 보이기.** $x$은 $u$보다 먼저 $S$에 들어갔고 $u$이 어림값이 틀린 *첫* 꼭짓점이므로 $d[x] = \delta(s, x)$이다. $x$이 $S$에 들어갈 때 변 $(x, y)$이 늦춰졌다. 모임 성질에 따라 다음이 성립한다:

$$
d[y] \le d[x] + w(x, y) = \delta(s, x) + w(x, y) = \delta(s, y)
$$

$p$이 최단 경로이고 $s \leadsto x \to y$이 그 부분 길이므로(가장 좋은 밑짜임) 마지막 등호가 성립한다. 위 한계 성질 $d[y] \ge \delta(s, y)$과 합치면 $d[y] = \delta(s, y)$을 얻는다.

**걸음 5: 어긋남 이끌어 내기.** 변의 무게가 모두 음이 아니므로 부분 길 $y \leadsto u$의 무게도 음이 아니다. 따라서 다음과 같다:

$$
\delta(s, y) \le \delta(s, u)
$$

따라서 다음이 성립한다.

$$
d[y] = \delta(s, y) \le \delta(s, u) \le d[u]
$$

그런데 $u$은 `EXTRACT-MIN`이 골랐으므로 $d[u] \le d[y]$이다.  아우르면

$$
d[u] \le d[y] = \delta(s, y) \le \delta(s, u) \le d[u]
$$

그러면 $d[u] = \delta(s, u)$이 되어 우리 가정과 어긋난다. $\square$

---

## 4. 음이 아닌 무게가 왜 꼭 필요한가

증명은 **걸음 5**에서 무너진다. 변이 음의 무게를 가질 수 있으면 부분 길 $y \leadsto u$의 무게 합이 음일 수 있어 $\delta(s, u) < \delta(s, y)$이 될 수 있다. 그러면 $y$보다 $u$을 먼저 꺼낸다고 $d[u] = \delta(s, u)$이 보장되지 않는다.

??? example "음의 무게를 쓴 어긋냄 보기"
    꼭짓점 $\{s, a, b\}$과 변 $(s, a, 3)$, $(s, b, 5)$, $(b, a, -4)$을 생각하자. 데이크스트라는 $d[a] = 3$으로 $a$을 먼저 꺼내지만, 참된 최단 경로 $s \to b \to a$의 무게는 $5 + (-4) = 1 < 3$이다. 이 알고리즘은 $a$을 너무 일찍 확정해 틀린 답을 낸다.

---

## 5. 복잡도 분석

시간 복잡도는 우선순위 줄을 어떻게 짜느냐에 달렸다.

| 우선순위 줄 | `EXTRACT-MIN` | `DECREASE-KEY` | 합계 |
|---|---|---|---|
| 배열(정렬 안 함) | $O(V)$ | $O(1)$ | $O(V^2)$ |
| 이진 힙 | $O(\log V)$ | $O(\log V)$ | $O((V+E)\log V)$ |
| 피보나치 힙 | 고르게 나눠 $O(\log V)$ | 고르게 나눠 $O(1)$ | $O(V\log V + E)$ |

성긴 그래프($E = O(V)$)에서는 이진 힙이 $O(V \log V)$을 준다. 빽빽한 그래프($E = O(V^2)$)에서는 단순 배열이 $O(V^2)$을 주며 이것이 가장 좋다.

---

## 6. 구현

```python
"""
옳음을 확인하는 데이크스트라 알고리즘.

알고리즘을 보이고 욕심쟁이 꺼내기 차례가
올바른 최단 경로 거리를 내는지 확인한다.
"""

import heapq
from math import inf

# === 데이크스트라 알고리즘 ===================================================

def dijkstra(graph: dict, source) -> tuple[dict, dict]:
    """데이크스트라 알고리즘으로 근원으로부터의 최단 경로 셈하기.

    매개변수
    ----------
    graph : dict
        꼭짓점 -> (이웃, 무게) 목록으로 잇는 이웃 목록.
        무게가 모두 음이 아니어야 한다.
    source : hashable
        근원 꼭짓점.

    반환값
    -------
    dist : dict
        근원에서의 최단 거리.
    pred : dict
        경로를 되짚기 위한 앞선 꼭짓점 가리개.
    """
    dist = {v: inf for v in graph}
    dist[source] = 0
    pred = {v: None for v in graph}
    pq = [(0, source)]

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue  # 묵은 항목
        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                pred[v] = u
                heapq.heappush(pq, (dist[v], v))

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
    graph = {
        0: [(1, 4), (2, 1)],
        1: [(3, 1)],
        2: [(1, 2), (3, 5)],
        3: [],
    }

    dist, pred = dijkstra(graph, 0)
    print(f"Distances: {dist}")
    print(f"Path 0->3: {get_path(pred, 0, 3)}")
    print(f"Path 0->1: {get_path(pred, 0, 1)}")

    # 욕심쟁이의 옳음 확인하기: 꺼내는 차례
    print("\n--- Extraction order verification ---")
    finalized = {}
    dist2 = {v: inf for v in graph}
    dist2[0] = 0
    pq = [(0, 0)]
    step = 0
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist2[u]:
            continue
        finalized[u] = dist2[u]
        step += 1
        print(f"Step {step}: extract vertex {u} with d[{u}] = {dist2[u]}")
        for v, w in graph[u]:
            if dist2[u] + w < dist2[v]:
                dist2[v] = dist2[u] + w
                heapq.heappush(pq, (dist2[v], v))
    print(f"Final: {finalized}")
```

**출력:**

```
Distances: {0: 0, 1: 3, 2: 1, 3: 4}
Path 0->3: [0, 2, 1, 3]
Path 0->1: [0, 2, 1]

--- Extraction order verification ---
Step 1: extract vertex 0 with d[0] = 0
Step 2: extract vertex 2 with d[2] = 1
Step 3: extract vertex 1 with d[1] = 3
Step 4: extract vertex 3 with d[3] = 4
Final: {0: 0, 2: 1, 1: 3, 3: 4}
```

---

## 연습문제

**연습문제 1.**
데이크스트라의 맞음 증명에서 우리는 변의 무게가 음이 아니라고 놓는다. 음의 변 무게 때문에 데이크스트라가 틀린 결과를 내는 작은 보기(꼭짓점 3개)를 지어라.

??? success "연습문제 1 풀이"
    꼭짓점 $\{s, a, b\}$과 변 $(s, a, 1)$, $(s, b, 3)$, $(a, b, -5)$. $s$에서 데이크스트라: $s$을 꺼내고($d=0$) 늦춰 $d[a]=1, d[b]=3$을 얻는다. $a$을 꺼내고($d=1$) $(a,b)$을 늦추면 $d[b] = \min(3, 1+(-5)) = -4$이다. 그런데 ($b$을 먼저 꺼냈다면) $a$을 꺼내기 전에 $b$이 이미 $d=3$으로 확정되었을 것이다. 최소 힙을 쓴 데이크스트라에서는 $a$(거리 1)을 $b$(거리 3)보다 먼저 꺼내므로 $d[b]$이 $-4$으로 새로 고쳐진다. 그러나 $(s, a, 5)$, $(s, b, 2)$, $(a, b, -4)$을 생각해 보자. 데이크스트라는 $d = 2$인 $b$을 먼저 꺼내 확정한다. 그다음 $d = 5$인 $a$을 꺼내 $d[a] + w(a,b) = 5 - 4 = 1 < 2 = d[b]$을 셈하지만 $b$은 이미 확정되었다. 데이크스트라는 올바른 $d[b] = 1$ 대신 $d[b] = 2$을 알린다. $\square$

---

**연습문제 2.**
데이크스트라의 맞음 증명에 쓰이는 핵심 되풀이 불변량을 밝히고 증명하여라.

??? success "연습문제 2 풀이"
    **불변량**: 되풀이가 시작될 때마다 확정 묶음 $S$의 꼭짓점 $v$마다 $d[v] = \delta(s, v)$(참된 최단 경로 거리)이다.

    **귀납으로 증명**: 바탕 경우: $d[s] = 0 = \delta(s, s)$인 $S = \{s\}$. 귀납 걸음: $|S| = k$에서 불변량이 성립한다고 놓자. 다음에 꺼내는 꼭짓점을 $u$이라 하자(확정되지 않은 꼭짓점 가운데 $d[u]$이 가장 작다). 어긋냄을 위해 $d[u] > \delta(s, u)$이라고 하자. 그러면 $s$에서 $u$까지의 참된 최단 경로가 $x \in S, y \notin S$인 어떤 변 $(x, y)$에서 $S$을 벗어나야 한다. 불변량에 따라 $d[x] = \delta(s, x)$이고, $(x, y)$을 늦춘 뒤 (그 부분 길이 최단 경로의 일부이므로) $d[y] \leq \delta(s, x) + w(x, y) \leq \delta(s, u)$이다. 그런데 $u$은 $d[u]$이 가장 작은 것으로 골랐으므로 $d[u] \leq d[y] \leq \delta(s, u) \leq d[u]$이고 $d[u] = \delta(s, u)$이다. 어긋남이 풀렸다. $\square$

---

**연습문제 3.**
데이크스트라의 맞음이 우선순위 줄을 어떻게 구현하느냐에 달렸는가? 왜 그런가?

??? success "연습문제 3 풀이"
    맞음은 우선순위 줄의 구현에 달려 있지 않다. 최소 꺼내기와 열쇠 낮추기를 받쳐 주는 올바른 최소 우선순위 줄이면 무엇이든 된다. 증명은 꺼낸 꼭짓점이 확정되지 않은 꼭짓점 가운데 잠정 거리가 가장 작다는 성질에만 기댄다. 우선순위 줄을 무엇으로 고르느냐(이진 힙, 피보나치 힙, 배열)는 시간 복잡도에 영향을 주지만 맞음에는 영향을 주지 않는다. 이진 힙은 $O((V+E)\log V)$, 피보나치 힙은 $O(VE \log V + V^2 \log V)$, 단순 배열은 $O(V^2)$을 준다. $\square$

---

**연습문제 4.**
데이크스트라 알고리즘이 최단 경로 거리가 줄지 않는 차례로 꼭짓점을 다룸을 증명하여라.

??? success "연습문제 4 풀이"
    $u_1, u_2, \ldots, u_n$을 꼭짓점을 꺼낸 차례라 하자. $\delta(s, u_i) \leq \delta(s, u_{i+1})$임을 보인다. $u_{i+1}$을 꺼낼 때 맞음 증명에 따라 $d[u_{i+1}] = \delta(s, u_{i+1})$이다. $u_i$을 꺼낼 때 $u_{i+1}$은 확정되지 않았고 ($u_i$이 $d$ 값이 가장 작았으므로) $d[u_{i+1}] \geq d[u_i]$이었다. $u_i$을 꺼내고 변을 늦춘 뒤 $d[u_{i+1}}]$은 줄기만 한다. 그런데 $d[u_i] = \delta(s, u_i)$이고, 음이 아닌 무게의 늦추기로는 꺼낼 때 $d[u_{i+1}] < d[u_i]$이 될 수 없다($d[u_{i+1}]$이 이미 $\geq d[u_i]$이었고 $u_i$을 거친 늦추기는 음이 아닌 무게를 더하기 때문이다). 그러므로 $\delta(s, u_{i+1}) \geq \delta(s, u_i)$이다. $\square$

## 정리하며

이 마당은 알고리즘 되짚기、맞음 정리、어긋냄으로 증명、음이 아닌 무게가 왜 꼭 필요한가을 차례로 짚었다.

**참고 문헌**

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to
  Algorithms* (4th ed.), Chapter 24.3: Dijkstra's Algorithm.
- Dijkstra, E. W. (1959). A note on two problems in connexion with graphs. *Numerische Mathematik*, 1, 269-271.
