# 음의 고리 알아내기

무게가 음인 고리는 변 무게의 합이 음인 방향 고리이다. 어떤 꼭짓점 $v$으로 가는 길 위에서 샘으로부터 그런 고리에 닿을 수 있으면 최단 경로 무게가 $\delta(s, v) = -\infty$이 된다. 고리를 되풀이해 돌면 길 무게가 끝없이 줄기 때문이다. 이런 고리를 알아내는 일은 꼭 필요하다. 알아내지 못하면 최단 경로 알고리즘이 끝없이 맴돌거나 뜻 없는 결과를 되돌릴 수 있다.

## 음의 고리란 무엇인가

무게 있는 방향 그래프의 고리 $c = \langle v_0, v_1, \dots, v_k = v_0 \rangle$이 다음을 만족하면 **무게가 음인 고리**이다:

$$
w(c) = \sum_{i=0}^{k-1} w(v_i, v_{i+1}) < 0
$$

그런 고리에서 닿을 수 있는 꼭짓점 $v$마다 최단 경로가 없다. 어떤 길이든 고리를 한 바퀴 더 돌면 "나아지기" 때문이다.

## 벨먼-포드로 알아내기

벨먼-포드 알고리즘은 음의 고리를 자연스럽게 알아낸다. 늦추기를 $|V| - 1$번 훑고 나면 음의 고리가 없는 그래프에서는 모든 최단 경로가 맞게 셈해진다(단순 최단 경로의 변이 많아야 $|V| - 1$개이므로). $|V|$번째 훑기가 아직 늦출 수 있는 변이 있는지 살핀다:

```
NEGATIVE-CYCLE-CHECK(G, w, s):
    Run BELLMAN-FORD for |V| - 1 passes
    for each edge (u, v) in E:
        if d[v] > d[u] + w(u, v):
            return TRUE   // negative cycle exists
    return FALSE
```

### 올바름

!!! note "알아내기 정리"
    벨먼-포드를 $|V| - 1$번 훑은 뒤, $d[v] > d[u] + w(u, v)$인 변 $(u, v)$이 있을 때 그리고 그때만 $s$에서 닿을 수 있는, 무게가 음인 고리가 있다.

**증명($\Rightarrow$).** $s$에서 음의 고리 $c = \langle v_0, v_1, \dots, v_k = v_0 \rangle$에 닿을 수 있는데, 어긋냄을 위해 $c$의 모든 변에 대해 $d[v_i] \le d[v_{i-1}] + w(v_{i-1}, v_i)$이라고 하자. 고리를 따라 합하면 다음과 같다:

$$
\sum_{i=1}^{k} d[v_i] \le \sum_{i=1}^{k} d[v_{i-1}] + \sum_{i=1}^{k} w(v_{i-1}, v_i)
$$

$v_k = v_0$이므로 왼쪽과 오른쪽의 $d$ 값 합이 같아 다음이 된다:

$$
0 \le \sum_{i=1}^{k} w(v_{i-1}, v_i) = w(c) < 0
$$

이는 어긋남이므로 고리 안 변 가운데 적어도 하나는 아직 늦출 수 있어야 한다.

**증명($\Leftarrow$).** 닿을 수 있는 음의 고리가 없으면 벨먼-포드의 맞음이 $|V| - 1$번 훑은 뒤 모든 $v$에 대해 $d[v] = \delta(s, v)$임을 보장한다. 삼각 부등식에 따라 $\delta(s, v) \le \delta(s, u) + w(u, v)$이므로 더 늦출 수 있는 변이 없다. $\square$

## 고리 뽑아내기

음의 고리를 알아내는 것도 쓸모 있지만, 흔히 그 고리 자체를 찾아내야 한다. $|V|$번째 훑기가 아직 늦출 수 있는 변 $(u, v)$을 찾으면 꼭짓점 $v$은 음의 고리 위에 있거나 그것에서 닿을 수 있다. 고리를 뽑아내려면 다음과 같이 한다:

1. $v$에서 앞선 것 가리개를 $|V|$번 따라가 (고리로 가는 길 위가 아니라) 고리 안에 있음을 보장한다.
2. 그 꼭짓점에서 어떤 꼭짓점을 다시 만날 때까지 앞선 것을 따라가면 고리가 나온다.

## 응용

Negative cycles arise in several practical contexts:

- **환차익 거래:** 변의 무게가 $-\log(\text{환율})$인 외환 그래프에서 음의 고리는 이익을 내는 거래의 늘어놓음에 해당한다.
- **밑천 최적화:** 일정 짜기나 길 정하기 문제에서 음의 고리는 짜임을 바꿔 전체 값을 줄일 수 있는 틈을 알릴 수 있다.
- **확인:** (최단 경로와 차이 제약의 이음을 거쳐) 제약 얼개에 쓸 만한 풀이가 없음을 증명하기.

## 풀이 예제

Consider vertices $\{s, a, b, c\}$ with edges:

| 변 | 무게 |
|---|---|
| $(s, a)$ | 4 |
| $(a, b)$ | -2 |
| $(b, c)$ | 3 |
| $(c, a)$ | -5 |

The cycle $a \to b \to c \to a$ has weight $(-2) + 3 + (-5) = -4 < 0$.

**After 3 passes of Bellman-Ford** ($|V|-1 = 3$):

- $d[s] = 0$, $d[a] = 4$, but the cycle keeps reducing $d[a]$.

**훑기 4(알아내기 훑기):** 변 $(c, a)$이 $d[a] > d[c] + w(c, a)$을 만족하여 음의 고리를 확인해 준다.

## 구현

```python
"""
벨먼-포드로 음의 순환 알아내고 뽑아내기.

근원에서 음의 무게 순환에 닿을 수 있는지 알아내고,
있으면 그 순환의 꼭짓점을 뽑아낸다.
"""

from math import inf


# === 음의 순환 알아내기를 붙인 벨먼-포드 =====================================

def detect_negative_cycle(
    vertices: list, edges: list, source
) -> tuple[bool, list]:
    """근원에서 닿을 수 있는 음의 순환을 알아내고 뽑아내기.

    매개변수
    ----------
    vertices : list
        모든 꼭짓점 이름.
    edges : list of (u, v, w)
        무게 있는 방향 변.
    source : hashable
        근원 꼭짓점.

    반환값
    -------
    has_cycle : bool
        근원에서 음의 순환에 닿을 수 있으면 True.
    cycle : list
        음의 순환을 이루는 꼭짓점들(없으면 비어 있다).
    """
    n = len(vertices)
    dist = {v: inf for v in vertices}
    dist[source] = 0
    pred = {v: None for v in vertices}

    # 표준 |V| - 1번의 늦추기 훑기
    for _ in range(n - 1):
        for u, v, w in edges:
            if dist[u] != inf and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                pred[v] = u

    # |V|번째 훑기: 음의 순환 살피기
    cycle_vertex = None
    for u, v, w in edges:
        if dist[u] != inf and dist[u] + w < dist[v]:
            dist[v] = dist[u] + w
            pred[v] = u
            cycle_vertex = v
            break

    if cycle_vertex is None:
        return False, []

    # 순환 안에 있음을 보장하려고 |V|걸음 되짚기
    v = cycle_vertex
    for _ in range(n):
        v = pred[v]

    # 순환 뽑아내기
    cycle = []
    u = v
    while True:
        cycle.append(u)
        u = pred[u]
        if u == v:
            cycle.append(u)
            break
    cycle.reverse()
    return True, cycle


# === 순환 무게 셈하기 ========================================================

def cycle_weight(cycle: list, edge_weights: dict) -> float:
    """순환의 전체 무게 셈하기.

    매개변수
    ----------
    cycle : list
        cycle[0] == cycle[-1]인 순환의 꼭짓점들.
    edge_weights : dict
        (u, v) -> 무게로 잇기.
    """
    total = 0
    for i in range(len(cycle) - 1):
        total += edge_weights[(cycle[i], cycle[i + 1])]
    return total


# === 보임 ====================================================================

if __name__ == "__main__":
    # 음의 순환이 있는 그래프: a -> b -> c -> a의 무게가 -4
    vertices = ["s", "a", "b", "c"]
    edges = [
        ("s", "a", 4),
        ("a", "b", -2),
        ("b", "c", 3),
        ("c", "a", -5),
    ]
    edge_weights = {(u, v): w for u, v, w in edges}

    has_cycle, cycle = detect_negative_cycle(vertices, edges, "s")
    print(f"Negative cycle detected: {has_cycle}")
    if has_cycle:
        print(f"Cycle: {cycle}")
        print(f"Cycle weight: {cycle_weight(cycle, edge_weights)}")

    # 음의 순환이 없는 그래프
    print("\n--- Graph without negative cycle ---")
    edges_ok = [
        ("s", "a", 4),
        ("a", "b", -2),
        ("b", "c", 3),
        ("c", "a", 1),  # 순환 무게 = -2 + 3 + 1 = 2 > 0
    ]
    has_cycle2, cycle2 = detect_negative_cycle(vertices, edges_ok, "s")
    print(f"Negative cycle detected: {has_cycle2}")
```

**출력:**

```
Negative cycle detected: True
Cycle: ['a', 'b', 'c', 'a']
Cycle weight: -4

--- Graph without negative cycle ---
Negative cycle detected: False
```

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to
  Algorithms* (4th ed.), Chapter 24.1: The Bellman-Ford Algorithm.

## 연습문제

**연습문제 1.**
샘에서 닿을 수 있는, 무게가 음인 고리가 있으면 왜 최단 경로가 정해지지 않는지 설명하여라.

??? success "연습문제 1 풀이"
    샘 $s$에서 무게가 음인 고리에 닿을 수 있고 그 고리에서 어떤 꼭짓점 $v$에 닿을 수 있으면, $s$에서 $v$으로 가는 길에 그 고리를 얼마든지 돌 수 있다. 한 바퀴 돌 때마다 길 무게의 합이 줄어들므로 길 무게에 끝이 있는 아래 한계가 없다. 곧 $\delta(s, v) = -\infty$이다. 최단 경로는 끝이 있는 가장 좋은 값을 필요로 하는데, 길에 음의 고리가 있으면 그런 값이 없다. $\square$

---

**연습문제 2.**
벨먼-포드가 음의 고리를 알아낸 뒤 그것에 영향받는 꼭짓점(곧 최단 경로 거리가 $-\infty$인 꼭짓점)을 모두 가려내는 법을 밝혀라.

??? success "연습문제 2 풀이"
    표준 벨먼-포드를 $V - 1$바퀴 돌린다. 그다음 한 바퀴 더 돌린다. 거리가 줄어든 꼭짓점 $v$은 모두 음의 고리 위에 있거나 그것에서 닿을 수 있다. 영향받는 꼭짓점을 모두 찾으려면 $V$번째 바퀴에서 새로 고쳐진 꼭짓점을 모두 모아 저마다에서 BFS/DFS을 돌린다. 이 새로 고쳐진 꼭짓점에서 닿을 수 있는 꼭짓점은 모두 $\delta(s, v) = -\infty$이다. 이 덧걸음에 $O(V + E)$ 시간이 든다. $\square$

---

**연습문제 3.**
데이크스트라 알고리즘이 음의 고리를 알아낼 수 있는가? 왜 그런가?

??? success "연습문제 3 풀이"
    알아낼 수 없다. 데이크스트라 알고리즘은 변의 무게가 음이 아니라고 놓으며 음의 고리를 알아내는 장치가 없다. 꼭짓점마다 꼭 한 번 확정하므로(우선순위 줄에서 꺼내며) 음의 고리를 거치는 더 짧은 길이 있음을 알아내려고 꼭짓점을 다시 들르는 일이 없다. 이 알고리즘은 잘못되었다는 낌새 없이 그냥 틀린 최단 경로 거리를 내놓는다. 음의 고리가 있을 수 있으면 벨먼-포드나 SPFA을 써야 한다. $\square$

---

**연습문제 4.**
어떤 그래프에 무게가 음인 고리는 없지만 무게가 음인 변은 얼마쯤 있다. 최단 경로가 그래도 있고 변이 많아야 $V - 1$개임을 증명하여라.

??? success "연습문제 4 풀이"
    음의 고리가 없으면 어떤 고리든 무게의 합이 음이 아니다. $s$에서 $v$까지의 최단 경로가 꼭짓점 $u$을 두 번 들렀다면, $u$을 두 번 들르는 사이의 조각이 무게가 음이 아닌 고리이다. 이 고리를 지우면 더 길지 않으면서 $u$을 한 번만 들르는 길이 된다. 되풀이되는 꼭짓점마다 이를 되풀이하면 단순 길(꼭짓점이 되풀이되지 않는 길)이 나온다. 단순 길은 꼭짓점을 많아야 $V$개 들르므로 변이 많아야 $V - 1$개이다. 단순 최단 경로가 있으므로 벨먼-포드가 $V - 1$바퀴에 모인다. $\square$
