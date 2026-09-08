# 더 빠른 최단 경로 알고리즘

표준 벨먼-포드 알고리즘은 $|V| - 1$번 훑을 때마다 *모든* 변을 늦춘다. 지난 훑기 이후 꼬리 꼭짓점이 바뀌지 않은 변까지 늦춘다. **더 빠른 최단 경로 알고리즘**(SPFA)은 거리가 최근에 줄어든 꼭짓점의 줄을 지켜 이 낭비를 피한다. 그 꼭짓점에서 나가는 변만 늦추면 되므로, 무게가 음인 변을 다루는 힘은 그대로 두면서 실전에서 도는 시간이 크게 줄곤 한다.

---

## 1. 핵심 생각

SPFA은 벨먼-포드를 줄로 다듬은 것이다. 훑을 때마다 모든 변을 훑는 대신 "살아 있는" 꼭짓점, 곧 거리 어림값이 방금 나아진 꼭짓점을 담은 FIFO 줄을 지킨다. 꼭짓점 $u$을 꺼내면 그 나가는 변을 늦춘다. 변 $(u, v)$을 늦춰 $d[v]$이 줄고 $v$이 아직 줄에 없으면 나중에 다루려고 $v$을 줄에 넣는다.

이 길은 어림값이 바뀌지 않은 꼭짓점에서 나가는 변을 늦추지 않는데, 그것이 표준 벨먼-포드에서 군더더기 일의 주된 원천이다.

---

## 2. 알고리즘

```
SPFA(G, w, s):
    INITIALIZE-SINGLE-SOURCE(G, s)
    queue = {s}
    in_queue = {s: TRUE, all others: FALSE}
    while queue is not empty:
        u = DEQUEUE(queue)
        in_queue[u] = FALSE
        for each edge (u, v) in Adj[u]:
            if d[u] + w(u,v) < d[v]:
                d[v] = d[u] + w(u,v)
                pred[v] = u
                if not in_queue[v]:
                    ENQUEUE(queue, v)
                    in_queue[v] = TRUE
```

`in_queue` 깃발이 겹치는 항목을 막아 언제든 꼭짓점마다 줄에 많아야 한 번 나타나게 한다.

---

## 3. 복잡도

- **최악의 경우:** 벨먼-포드와 같은 $O(VE)$이다. 적수가 짠 그래프는 꼭짓점마다 $O(V)$번 줄에 들게 만들 수 있다.
- **평균의 경우:** 겪어 보면 훨씬 빠르며 무작위 그래프에서는 흔히 $O(E)$에 가깝다. 그러나 더 빈틈없는 최악의 경우 한계는 알려져 있지 않다.
- **공간:** 줄과 도움 배열에 $O(V)$.

!!! warning "최악의 경우 성능"
    평균으로는 잘 굴러가지만 SPFA은 잘 짜인 그래프에서 $O(VE)$으로 무너질 수 있다. 그래서 적수 입력이 있을 수 있는 겨루기 자리에서는 (무게가 음이 아닐 때) 데이크스트라 알고리즘이나 ($O(VE)$이 보장된) 표준 벨먼-포드를 흔히 더 낫게 여긴다.

---

## 4. 음의 고리 알아내기

SPFA은 꼭짓점마다 줄에 몇 번 들어갔는지 세어 음의 고리를 알아낼 수 있다. 어떤 꼭짓점이 $|V| - 1$번을 넘게 줄에 들면 그것은 음의 고리 위에 있거나 그것에서 닿을 수 있다.

음의 고리가 없는 그래프에서는 꼭짓점마다 거리가 많아야 $|V| - 1$번 줄 수 있으므로(있을 수 있는 최단 경로 길이마다 한 번) 이렇게 된다.

---

## 5. 벨먼-포드와의 견줌

| 결 | 벨먼-포드 | SPFA |
|---|---|---|
| 변 다루기 | 모든 변을 $\lvert V\rvert - 1$번 | 최근에 나아진 꼭짓점에서 나가는 변만 |
| 최악의 경우 시간 | $O(VE)$ | $O(VE)$ |
| 실전 성능 | 한결같음 | 흔히 훨씬 빠르나 들쭉날쭉함 |
| 구현 | 더 단순함 | 살짝 더 복잡함(줄 다루기) |
| 음의 고리 알아내기 | $\lvert V\rvert - 1$ 뒤 덧훑기 | 줄에 넣은 횟수 세기 |

---

## 6. 풀이 예제

변이 다음과 같은 꼭짓점 $\{s, a, b, c, d\}$을 보자.

| 변 | 무게 |
|---|---|
| $(s, a)$ | 1 |
| $(s, b)$ | 4 |
| $(a, b)$ | 2 |
| $(a, c)$ | 6 |
| $(b, c)$ | 3 |
| $(b, d)$ | 1 |
| $(c, d)$ | -2 |

**걸음 1:** $d[s]=0$으로 두고 나머지는 모두 $\infty$으로 둔다.  $s$을 줄에 넣는다.

**걸음 2:** $s$을 꺼낸다. $(s,a)$ 늦추기: $d[a]=1$, $a$을 줄에 넣는다. $(s,b)$ 늦추기: $d[b]=4$, $b$을 줄에 넣는다. 줄: $[a, b]$.

**걸음 3:** $a$을 꺼낸다. $(a,b)$ 늦추기: $d[b]=\min(4, 1+2)=3$, $b$은 이미 줄에 있다. $(a,c)$ 늦추기: $d[c]=7$, $c$을 줄에 넣는다. 줄: $[b, c]$.

**걸음 4:** $b$을 꺼낸다. $(b,c)$ 늦추기: $d[c]=\min(7, 3+3)=6$, $c$은 이미 줄에 있다. $(b,d)$ 늦추기: $d[d]=4$, $d$을 줄에 넣는다. 줄: $[c, d]$.

**걸음 5:** $c$을 꺼낸다. $(c,d)$ 늦추기: $d[d]=\min(4, 6-2)=4$(바뀜 없음). 줄: $[d]$.

**걸음 6:** $d$을 줄에서 꺼낸다.  나가는 변이 없다.  줄이 비었다.

**마지막 거리:** $d[s]=0, d[a]=1, d[b]=3, d[c]=6, d[d]=4$.

---

## 7. 구현

```python
"""
더 빠른 최단 경로 알고리즘(SPFA).

벨먼-포드를 줄 기반으로 최적화하여 쓸데없는
변 늦추기를 피하려고 최근에 나아진 꼭짓점만 다룬다.
"""

from collections import deque
from math import inf

# === SPFA 알고리즘 ===========================================================

def spfa(graph: dict, source) -> tuple[dict, dict, bool]:
    """주어진 근원 꼭짓점에서 SPFA 돌리기.

    매개변수
    ----------
    graph : dict
        꼭짓점 -> (이웃, 무게) 목록으로 잇는 이웃 목록.
    source : hashable
        근원 꼭짓점.

    반환값
    -------
    dist : dict
        근원에서의 최단 거리.
    pred : dict
        경로를 되짚기 위한 앞선 꼭짓점 가리개.
    no_negative_cycle : bool
        근원에서 음의 순환에 닿을 수 없으면 True.
    """
    n = len(graph)
    dist = {v: inf for v in graph}
    dist[source] = 0
    pred = {v: None for v in graph}
    in_queue = {v: False for v in graph}
    count = {v: 0 for v in graph}  # 순환을 알아내려는 줄 넣기 횟수

    queue = deque([source])
    in_queue[source] = True
    count[source] = 1

    while queue:
        u = queue.popleft()
        in_queue[u] = False

        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                pred[v] = u
                if not in_queue[v]:
                    queue.append(v)
                    in_queue[v] = True
                    count[v] += 1
                    if count[v] >= n:
                        return dist, pred, False  # 음의 순환

    return dist, pred, True

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
        "s": [("a", 1), ("b", 4)],
        "a": [("b", 2), ("c", 6)],
        "b": [("c", 3), ("d", 1)],
        "c": [("d", -2)],
        "d": [],
    }

    dist, pred, ok = spfa(graph, "s")
    print(f"No negative cycle: {ok}")
    print(f"Distances: {dist}")
    print(f"Path s->d: {get_path(pred, 's', 'd')}")
    print(f"Path s->c: {get_path(pred, 's', 'c')}")

    # 음의 순환이 있는 그래프
    print("\n--- Graph with negative cycle ---")
    graph_neg = {
        "s": [("a", 1)],
        "a": [("b", -3)],
        "b": [("c", 1)],
        "c": [("a", -1)],  # 순환: a->b->c->a = -3+1+(-1) = -3
    }
    dist2, pred2, ok2 = spfa(graph_neg, "s")
    print(f"No negative cycle: {ok2}")
```

**출력:**

```
No negative cycle: True
Distances: {'s': 0, 'a': 1, 'b': 3, 'c': 6, 'd': 4}
Path s->d: ['s', 'a', 'b', 'd']
Path s->c: ['s', 'a', 'b', 'c']

--- Graph with negative cycle ---
No negative cycle: False
```

---

## 연습문제

**연습문제 1.**
SPFA(더 빠른 최단 경로 알고리즘)를 밝히고 벨먼-포드보다 어떻게 나은지 설명하여라.

??? success "연습문제 1 풀이"
    SPFA은 거리가 최근에 줄어든 꼭짓점의 줄을 지킨다. 처음에는 샘만 줄에 있다. 꼭짓점 $u$을 꺼내면 변 $(u, v)$을 모두 늦춘다. $d[v]$이 줄고 $v$이 아직 줄에 없으면 $v$을 줄에 넣는다. 그러면 (바퀴마다 모든 변을 늦추는 벨먼-포드와 달리) 거리가 바뀌지 않은 꼭짓점에서 나가는 변을 늦추지 않는다. 평균의 경우 복잡도는 $O(E)$이지만 최악의 경우는 여전히 $O(VE)$이다. $\square$

---

**연습문제 2.**
SPFA이 음의 고리를 어떻게 알아낼 수 있는가?

??? success "연습문제 2 풀이"
    꼭짓점마다 줄에 들어간 횟수를 좇는다. 어떤 꼭짓점이 $V$번 이상 줄에 들면 음의 고리가 있다. 음의 고리가 없는 그래프에서는 꼭짓점마다 많아야 $V - 1$번 늦춰지므로(있을 수 있는 최단 경로 길이마다 한 번) 이렇게 된다. 아니면 늦추기의 총 횟수를 세어 $V \cdot E$을 넘으면 음의 고리가 있는 것이다. $\square$

---

**연습문제 3.**
우선순위 줄을 쓴 데이크스트라 알고리즘과 SPFA을 견주어라. 어느 것을 언제 고르겠는가?

??? success "연습문제 3 풀이"
    **SPFA**: 음의 무게를 다루며 평균은 $O(E)$이지만 최악의 경우는 $O(VE)$이다. 단순한 FIFO 줄을 쓴다. 음의 고리를 알아낼 수 있다. 평균 성능으로 넉넉하다면 음의 무게가 있는 그래프에 가장 알맞다.

    **데이크스트라**: 무게가 음이 아니어야 하며 이진 힙으로 $O((V+E)\log V)$이 보장된다. 우선순위 줄을 쓴다. 음의 고리를 알아내지 못한다. 최악의 경우 보장이 필요한, 무게가 음이 아닌 그래프에 가장 알맞다.

    음의 무게가 있는 성긴 그래프에서는 SPFA이 더 단순하고 평균으로 더 빠르다. 데이크스트라는 더 미리 알 수 있어, 적수 입력이 SPFA의 최악의 경우를 끌어낼 수 있는 겨루기 자리에서 더 낫다. $\square$

---

**연습문제 4.**
SPFA의 SLF(가장 짧은 이름표 먼저) 다듬기는 꼭짓점의 거리가 줄 맨 앞 원소의 거리보다 작으면 그 꼭짓점을 줄의 앞에 넣는다. 이것이 왜 성능을 낫게 할 수 있는지 설명하여라.

??? success "연습문제 4 풀이"
    SLF은 잠정 거리가 작은 꼭짓점을 먼저 다루므로 SPFA이 데이크스트라처럼 움직이게 한다. 거리가 작은 꼭짓점을 먼저 다루면 늦추기가 더 잘 들어(참 최단 거리에 더 빨리 닿는다) 꼭짓점이 줄에 다시 들어오는 횟수가 준다. 가장 나쁠 때의 복잡도는 그대로지만 평균 성능은 흔히 크게 나아지며, 특히 양수와 음수 변 짐이 섞인 그래프에서 그렇다. $\square$

## 정리하며

이 마당은 핵심 생각、알고리즘、복잡도、음의 고리 알아내기을 차례로 짚었다.

**참고 문헌**

- Duan, F. (1994). About the Shortest Path Faster Algorithm. *Journal of Southwest Jiaotong University*, 29(2), 207-212.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to
  Algorithms* (4th ed.), Chapter 24: Single-Source Shortest Paths.
