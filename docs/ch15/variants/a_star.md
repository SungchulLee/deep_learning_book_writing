# A* 찾기

데이크스트라 알고리즘은 모든 방향으로 고르게 꼭짓점을 살펴보는데, 특정한 과녁에 닿는 것이 목표라면 이는 헤플 수 있다. A* 찾기는 과녁까지 남은 거리를 어림하는 **어림짐작 함수**를 넣어 데이크스트라를 낫게 만든다. 그러면 찾기가 목표 쪽으로 이끌리고 살펴보는 꼭짓점이 훨씬 적어지곤 한다. A*은 놀이, 로봇, 길 찾기에서 표준 알고리즘이다.

---

## 1. A*의 값 매기기 함수

A*은 값 매기기 함수로 차례 매긴 우선순위 줄을 지킨다

$$
f(n) = g(n) + h(n)
$$

여기서 $g(n)$은 샘 $s$에서 꼭짓점 $n$까지 알려진 가장 싼 길의 실제 값이고, $h(n)$은 $n$에서 과녁 $t$까지 값의 어림짐작이다. 이 알고리즘은 늘 $f$ 값이 가장 작은 꼭짓점을 넓힌다.

- 모든 $n$에 대해 $h(n) = 0$이면 A*은 데이크스트라 알고리즘으로 주저앉는다.
- $h(n)$이 남은 참값을 정확히 비추면 A*은 과녁으로 곧장 나아간다.

---

## 2. 받아들일 만함과 가장 좋음

어림짐작 $h$이 목표까지의 참값을 결코 넘겨 어림하지 않으면 **받아들일 만하다**고 한다:

$$
h(n) \leq h^*(n) \quad \text{for all } n
$$

여기서 $h^*(n)$은 $n$에서 $t$까지의 참된 최단 경로 값이다.

!!! tip "받아들일 만하면 가장 좋음이 보장된다"
    $h$이 받아들일 만하면 A*은 $s$에서 $t$까지 가장 좋은(최단) 길을 찾음이 보장된다. $t$까지의 덜 좋은 길은 모두 $f > f^*$(가장 좋은 값)이므로 A*이 가장 좋은 길을 먼저 넓히기 때문이다.

---

## 3. 한결같음(단조성)

값이 $w(u, v)$인 변 $(u, v)$마다 다음이 성립하면 어림짐작 $h$이 **한결같다**(단조라고도 한다)고 한다:

$$
h(u) \leq w(u, v) + h(v)
$$

한결같으면 받아들일 만하다(거꾸로는 아니다). 한결같은 어림짐작을 쓰면 어느 길에서나 $f$ 값이 줄지 않고, A*은 닫힌 꼭짓점을 다시 열 필요가 없다. 그래서 실전에서 더 효율적이다.

---

## 4. 알고리즘

1. 첫걸음: $g(s) = 0$, $f(s) = h(s)$으로 놓고 $s$을 우선순위 줄에 넣는다.
2. 우선순위 줄이 비지 않은 동안:
    - $f(n)$이 가장 작은 꼭짓점 $n$을 꺼낸다.
    - $n = t$이면 길을 되살려 되돌린다.
    - $n$의 이웃 $v$마다: $g(n) + w(n, v) < g(v)$이면 $g(v)$과 $f(v) = g(v) + h(v)$을 새로 고치고 $v$을 줄에 넣는다.
3. $t$에 닿지 못한 채 줄이 비면 길이 없다.

---

## 5. 복잡도

한결같은 어림짐작을 쓰면 A*은 꼭짓점마다 많아야 한 번 넓힌다. 최악의 경우(알려 주는 바 없는 어림짐작) A*은 꼭짓점 $V$개를 모두 넓히고 변 $E$개를 모두 살펴, 이진 힙을 쓴 데이크스트라와 같은 $O((V + E) \log V)$ 복잡도가 된다. 실전에서 좋은 어림짐작은 넓히는 꼭짓점의 개수를 크게 줄인다.

---

## 6. 구현

```python
"""
무게 그래프를 위한 A* 찾기 알고리즘.

어림짐작 함수로 찾기를 과녁 쪽으로 이끌어,
데이크스트라보다 살펴보는 꼭짓점 수를 줄인다.
"""

import heapq

# === A* 찾기 ==================================================================

def a_star(graph, source, target, heuristic):
    """A* 찾기로 최단 경로 찾기.

    매개변수
    ----------
    graph : dict[int, list[tuple[int, float]]]
        (이웃, 무게) 짝을 담은 이웃 목록.
    source : int
        시작 꼭짓점.
    target : int
        목표 꼭짓점.
    heuristic : dict[int, float]
        꼭짓점마다 과녁까지의 어림 값.

    반환값
    -------
    tuple[list[int], float] | tuple[None, float]
        찾으면 (경로, 값), 아니면 (None, inf).
    """
    g_cost = {source: 0.0}
    f_cost = {source: heuristic.get(source, 0.0)}
    predecessor = {source: None}
    open_set = [(f_cost[source], source)]
    closed = set()

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == target:
            path = []
            node = target
            while node is not None:
                path.append(node)
                node = predecessor[node]
            return path[::-1], g_cost[target]

        if current in closed:
            continue
        closed.add(current)

        for neighbor, weight in graph[current]:
            if neighbor in closed:
                continue
            tentative_g = g_cost[current] + weight
            if tentative_g < g_cost.get(neighbor, float("inf")):
                g_cost[neighbor] = tentative_g
                f_cost[neighbor] = tentative_g + heuristic.get(neighbor, 0.0)
                predecessor[neighbor] = current
                heapq.heappush(open_set, (f_cost[neighbor], neighbor))

    return None, float("inf")

# === 메인 =====================================================================

if __name__ == "__main__":
    # 무게 그래프: 꼭짓점 -> [(이웃, 무게), ...]
    graph = {
        0: [(1, 1.0), (2, 4.0)],
        1: [(2, 2.0), (3, 5.0)],
        2: [(3, 1.0)],
        3: [],
    }

    # 어림짐작: 꼭짓점마다 과녁(꼭짓점 3)까지의 어림 거리
    h = {0: 3.0, 1: 2.0, 2: 1.0, 3: 0.0}

    path, cost = a_star(graph, 0, 3, h)
    print(f"A* path: {path}")
    print(f"Total cost: {cost}")
```

**출력:**
```
A* path: [0, 1, 2, 3]
Total cost: 4.0
```

어림짐작이 A*을 이끌어 비싼 곧바른 변 $0 \to 2$을 살펴보지 않고도 값 4.0의 가장 좋은 길 $0 \to 1 \to 2 \to 3$을 찾게 한다.

---

## 연습문제

**연습문제 1.**
어림짐작 함수 $h$의 받아들일 만함과 한결같음을 정의하여라. 한결같으면 받아들일 만함을 증명하여라.

??? success "연습문제 1 풀이"
    모든 $v$에 대해 $h(v) \leq \delta(v, t)$이면(목표 $t$까지의 참값을 결코 넘겨 어림하지 않으면) 어림짐작 $h$이 **받아들일 만하다**. 변 $(u,v)$마다 $h(u) \leq w(u,v) + h(v)$이고 $h(t) = 0$이면 어림짐작이 **한결같다**(단조이다). 한결같으면 받아들일 만함을 증명하자. $p = v_0, v_1, \ldots, v_k = t$을 $v_0$에서 $t$까지의 최단 경로라 하자. 한결같음에 따라 $h(v_i) \leq w(v_i, v_{i+1}) + h(v_{i+1})$이다. 망원경처럼 접으면 $h(v_0) \leq \sum_{i=0}^{k-1} w(v_i, v_{i+1}) + h(t) = \delta(v_0, t) + 0 = \delta(v_0, t)$이다. $\square$

---

**연습문제 2.**
값이 고른 격자 그래프에서 맨해튼 거리 어림짐작을 쓸 때 A*, 데이크스트라, BFS이 넓히는 마디의 개수를 견주어라.

??? success "연습문제 2 풀이"
    값이 고른 격자에서는 (값이 모두 같으므로) BFS과 데이크스트라가 같은 마디를 넓히며 거의 둥근 앞자락을 살펴본다. 거리 $d$의 과녁이면 마디 $O(d^2)$개이다. 맨해튼 거리 어림짐작을 쓴 A*은 찾기를 목표 쪽으로 모아, 샘과 과녁 사이의 마름모꼴 구역쯤을 넓히므로 마디가 훨씬 적다. 트인 격자에서 A*은 넓히는 마디가 $O(d)$에 가까워질 수 있다. 걸림돌이 없을 때 나아짐이 가장 극적이고, 걸림돌이 많으면 A*이 데이크스트라의 개수에 가까워질 수 있다. $\square$

---

**연습문제 3.**
어림짐작이 참된 거리를 넘겨 어림하면 어떻게 되는가? 그래도 A*이 가장 좋은 길을 찾음이 보장되는가?

??? success "연습문제 3 풀이"
    $h$이 넘겨 어림하면(받아들일 만하지 않으면) A*이 덜 좋은 길을 찾을 수 있다. 어떤 가운데 꼭짓점에서 남은 값을 어림짐작이 낮게 매겨 $f = g + h$이 작아 보이는 덜 좋은 길로 목표 마디를 넓힐 수 있다. 특히 가장 좋은 길이 $h$이 너무 높은 꼭짓점을 지나면 A*이 그것을 뒤로 미룬다. 무게 준 A*은 일부러 $\epsilon > 1$인 $f = g + \epsilon \cdot h$을 써서 가장 좋음을 내주고 빠르기를 얻으며, 가장 좋은 값의 $\epsilon$배 안에 드는 길을 보장한다. $\square$

---

**연습문제 4.**
한결같은 어림짐작을 쓴 A*이 닫힌(확정된) 마디를 결코 다시 열지 않음을 증명하여라.

??? success "연습문제 4 풀이"
    꼭짓점 $u$이 $f(u) = g(u) + h(u)$으로 확정되었다고 하자. 나중에 변 $(v, u)$과 함께 찾은 아무 꼭짓점 $v$에 대해 $g'(u) = g(v) + w(v, u)$이다. 한결같음에 따라 $h(v) \leq w(v, u) + h(u)$이므로 $f(v) = g(v) + h(v) \leq g(v) + w(v, u) + h(u) = g'(u) + h(u) = f'(u)$이다. $v$은 $u$보다 나중에 다뤄졌으므로 $f(v) \geq f(u)$이고, 따라서 $f'(u) \geq f(v) \geq f(u)$이며 $g'(u) + h(u) \geq g(u) + h(u)$, 곧 $g'(u) \geq g(u)$이다. $v$을 거치는 새 길이 더 짧지 않으므로 다시 열 필요가 없다. $\square$

## 정리하며

이 마당은 A*의 값 매기기 함수、받아들일 만함과 가장 좋음、한결같음(단조성)、알고리즘을 차례로 짚었다.

**참고 문헌**

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), 24-25장. MIT Press.
- Hart, P. E., Nilsson, N. J., & Raphael, B. (1968). A formal basis for the heuristic determination of minimum cost paths. *IEEE Transactions on Systems Science and Cybernetics*, 4(2), 100-107.
