# 최소 값 최대 흐름

보통의 최대 흐름 문제는 변으로 흐름을 보내는 값을 헤아리지 않는다. 여러 쓰임새에서 변마다 담이와 함께 낱개당 값이 있고, 우리는 정해진 양의 흐름(또는 가능한 최대 흐름)을 **가장 적은 전체 값**으로 보내고자 한다. 최소 값 최대 흐름 문제는 두 목표를 아우른다. 곧 모든 최대 흐름 가운데 값이 가장 작은 것을 찾는다.

## 문제 정식화

**값 흐름 그물**은 흐름 그물에 값 함수를 더해 넓힌 것이다. 다음을 갖는 방향 그래프 $G = (V, E)$이 주어질 때:

- 담는 힘 함수 $c: E \to \mathbb{R}_{\ge 0}$.
- 값 함수 $w: E \to \mathbb{R}$(이음마다 흐름 한 단위에 드는 값).
- 근원 $s$과 바닥 $t$.

흐름 $f$의 **값**은 다음과 같다:

$$
\text{cost}(f) = \sum_{(u,v) \in E} w(u, v) \cdot f(u, v)
$$

**값이 가장 작은 가장 큰 흐름** 문제는 다음을 채우는 흐름 $f^*$을 찾는다.

$$
|f^*| = \max_f |f| \quad \text{and} \quad \text{cost}(f^*) = \min \{\text{cost}(f) : |f| = |f^*|\}
$$

## 알고리즘: 잇단 최단 경로

가장 직관적인 방식은 남은 그래프에서 늘 **최단(값이 가장 작은) 경로**를 따라 늘리도록 포드-풀커슨을 고쳐 쓰는 것이다. 남은 그래프에서 앞으로 가는 변의 값은 $w(u,v)$, 뒤로 가는 변의 값은 $-w(u,v)$이다.

**1단계.** $f = 0$으로 첫자리매김한다.

**2단계.** (값이 음수인 뒤로 가는 변이 있을 수 있으므로) 벨먼-포드나 SPFA를 써서 남은 그래프에서 $s$부터 $t$까지 값이 가장 작은 경로를 찾는다.

**3단계.** 이 경로를 따라 병목 담이만큼 흐름을 늘린다.

**4단계.** 남은 그래프에 $s$-$t$ 경로가 없어질 때까지 되풀이한다.

늘릴 때마다 쓸 수 있는 가장 싼 길로 흐름을 보내므로 전체 값이 될 수 있는 한 천천히 늘어난다. 늘림 경로가 더 없으면 흐름은 최대이고 값은 모든 최대 흐름 가운데 가장 작다.

## 존슨의 퍼텐셜 재주

벨먼-포드는 가장 짧은 길 물음마다 $O(VE)$이 든다. **잠재값**(줄인 값)을 써서 음수 이음을 없애고 데이크스트라로 갈아탈 수 있다(물음마다 $O(E \log V)$).

잠재값 $h: V \to \mathbb{R}$과 줄인 값을 다음과 같이 매긴다.

$$
w_h(u, v) = w(u, v) + h(u) - h(v)
$$

첫 벨먼-포드 훑기가 $h$을 $s$에서의 최단 경로 거리로 첫자리매김하고 나면 줄인 값이 모두 음이 아니다. 늘릴 때마다 새 거리로 퍼텐셜을 고친다. 곧 $h'(v) = h(v) + d(v)$이며 여기서 $d(v)$은 줄인 값으로 잰 최단 거리이다.

## 구현

```python
"""
데이크스트라를 쓴 잇단 최단 경로로 하는 최소 값 최대 흐름.

첫 벨먼-포드 훑기 뒤 음수 변을 피하려 존슨의 퍼텐셜 재주를 써
쓴다.
여기서 F는 최대 흐름 값.
"""

import heapq
from collections import defaultdict

# === 최소 값 최대 흐름 ===

def min_cost_max_flow(
    n: int, edges: list[tuple[int, int, int, int]], source: int, sink: int
) -> tuple[int, int]:
    """최소 값 최대 흐름을 셈한다.

    인수:
        n: 꼭짓점의 개수(0부터 셈).
        edges: (u, v, 담이, 값) 튜플의 목록.
        source: 근원 꼭짓점.
        sink: 바닥 꼭짓점.

    반환값:
        튜플 (최대 흐름 값, 최소 값).
    """
    graph = [[] for _ in range(n)]

    def add_edge(u: int, v: int, cap: int, cost: int) -> None:
        graph[u].append([v, cap, cost, len(graph[v])])
        graph[v].append([u, 0, -cost, len(graph[u]) - 1])

    for u, v, cap, cost in edges:
        add_edge(u, v, cap, cost)

    total_flow = 0
    total_cost = 0
    potential = [0] * n  # 존슨의 퍼텐셜

    while True:
        # 퍼텐셜을 쓴 데이크스트라
        dist = [float('inf')] * n
        dist[source] = 0
        prev_node = [-1] * n
        prev_edge = [-1] * n
        pq = [(0, source)]

        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            for i, (v, cap, cost, _) in enumerate(graph[u]):
                if cap > 0:
                    new_dist = d + cost + potential[u] - potential[v]
                    if new_dist < dist[v]:
                        dist[v] = new_dist
                        prev_node[v] = u
                        prev_edge[v] = i
                        heapq.heappush(pq, (new_dist, v))

        if dist[sink] == float('inf'):
            break

        # 퍼텐셜 고치기
        for v in range(n):
            if dist[v] < float('inf'):
                potential[v] += dist[v]

        # 최단 경로를 따라 병목 찾기
        bottleneck = float('inf')
        v = sink
        while v != source:
            u = prev_node[v]
            idx = prev_edge[v]
            bottleneck = min(bottleneck, graph[u][idx][1])
            v = u

        # 흐름 늘리기
        v = sink
        while v != source:
            u = prev_node[v]
            idx = prev_edge[v]
            graph[u][idx][1] -= bottleneck
            graph[v][graph[u][idx][3]][1] += bottleneck
            v = u

        total_flow += bottleneck
        total_cost += bottleneck * potential[sink]

    return total_flow, total_cost


# === 시연 ===

if __name__ == "__main__":
    # 그물: s=0, a=1, b=2, t=3
    # (u, v, 담이, 낱개당 값)
    edges = [
        (0, 1, 4, 1),   # s->a: cap 4, cost 1
        (0, 2, 3, 2),   # s->b: cap 3, cost 2
        (1, 2, 2, 1),   # a->b: cap 2, cost 1
        (1, 3, 3, 3),   # a->t: cap 3, cost 3
        (2, 3, 5, 2),   # b->t: cap 5, cost 2
    ]
    flow, cost = min_cost_max_flow(4, edges, 0, 3)
    print(f"Max flow: {flow}")
    print(f"Min cost: {cost}")
```

**출력:**

```
Max flow: 7
Min cost: 27
```

이 알고리즘은 전체 값 $27$으로 $7$낱의 최대 흐름을 찾는다. 값이 싼 경로로 먼저 흐름을 보내고, 전체 흐름을 가장 크게 하는 데 필요할 때만 값이 비싼 경로를 쓴다.

## 복잡도

| 항목 | 비용 |
|--------|:----:|
| 잇따른 가장 짧은 길(벨먼-포드) | $O(V E \cdot |f^*|)$ |
| 데이크스트라 + 잠재값 | $O(|f^*| \cdot E \log V)$ |
| 공간 | $O(V + E)$ |

최대 흐름 값이 작은 그물에서는 잇단 최단 경로 방식이 쓸 만하다. 흐름 값이 크면 순환 없애기나 그물 심플렉스 알고리즘이 더 효율적일 수 있다.

## 응용

- **나름.** 수요를 채우면서 가장 적은 실어 나름 값으로 공장에서 창고로 물건을 보낸다.
- **배정 문제.** 헝가리 알고리즘은 두 쪽 그물에서의 최소 값 흐름의 특수한 경우이다.
- **그물 꾸미기.** 늦음이나 값을 가장 작게 하며 주고받기 그물로 오감을 흘려 보낸다.

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to Algorithms* (4th ed.), 26장: Maximum Flow.
- Ahuja, R. K., Magnanti, T. L., & Orlin, J. B. (1993). *Network Flows: Theory, Algorithms, and Applications*. Prentice Hall.

## 연습문제

**연습문제 1.**
최소 값 최대 흐름 문제를 정의하고 보통의 최대 흐름 문제를 어떻게 넓히는지 설명하여라.

??? success "연습문제 1 풀이"
    여느 가장 큰 흐름은 이음의 값을 헤아리지 않고 흐름만 가장 크게 한다. 값이 가장 작은 가장 큰 흐름은 이음마다 값 $w(u,v)$을 더 매기고, $\sum_{(u,v)} w(u,v) \cdot f(u,v)$(온 값)을 가장 작게 하는 가장 큰 흐름을 찾는다. 흐름 키우기와 값 줄이기를 함께 다루는 셈이다. 실어 나르기(바라는 만큼 대면서 나르는 값을 가장 작게), 맡기기(맡기는 값의 합을 가장 작게) 따위에 쓴다. $\square$

---

**연습문제 2.**
최소 값 최대 흐름을 푸는 잇단 최단 경로 알고리즘을 설명하여라.

??? success "연습문제 2 풀이"
    나머지 그래프에서 $s$부터 $t$까지 값이 가장 작은 늘리는 길을 거듭 찾는다(나머지 이음의 값이 음수일 수 있으므로 벨먼-포드나 SPFA을 쓴다). 그 길로 흐름을 밀어 넣는다. 늘리는 길이 없어질 때까지 되풀이한다. 늘릴 때마다 쓸 수 있는 가장 싼 길로 흐름을 보내며, 잇따른 가장 짧은 길의 결이 가장 좋음을 보장한다. 때는 $O(f^* \cdot VE)$이며 $f^*$은 가장 큰 흐름 값이다. $\square$

---

**연습문제 3.**
최소 값 흐름에서 남은 변의 값이 왜 음수일 수 있는가?

??? success "연습문제 3 풀이"
    값이 $w(u,v)$인 이음 $(u,v)$으로 흐름 $f(u,v) > 0$을 보내면 나머지 그래프에 값이 $-w(u,v)$인 뒤로 가는 이음 $(v,u)$이 생긴다. 이 음수 값은 이미 흘린 것을 물릴 때 아끼는 값을 나타낸다. 뒤로 가는 이음으로 흐름을 보내면 앞서 보낸 흐름이 사실상 물려 단위마다 값이 $w(u,v)$만큼 준다. 이 얼개 덕에 알고리즘이 앞서 잘못 고른 흐름을 바로잡을 수 있다. $\square$

---

**연습문제 4.**
배정 문제에서 최소 값 최대 흐름과 헝가리 알고리즘을 견주어라.

??? success "연습문제 4 풀이"
    둘 다 맡기기 문제를 가장 좋게 푼다. 헝가리 알고리즘은 값이 붙은 두 쪽 짝짓기에 맞추어 만든 것으로 $O(n^3)$ 때에 돈다. 값이 가장 작은 가장 큰 흐름은 더 두루 쓴다. 두 쪽이 아닌 그물, 꼭짓점마다 여러 번 맡기기(담는 힘 매임과 함께), 실어 나르기 문제까지 다룬다. 여느 맡기기 문제에서는 헝가리가 대체로 더 빠르다($O(n^3)$ 대 흐름 바탕 방법의 $O(n^3 \log n)$ 이상). 넓힌 맡기기나 실어 나르기 문제에는 값이 가장 작은 가장 큰 흐름이 알맞다. $\square$
