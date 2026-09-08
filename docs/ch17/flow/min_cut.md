# 최대 흐름 최소 자름 정리

최대 흐름 최소 자름 정리는 조합 가장 좋게 하기에서 가장 중요한 결과 가운데 하나이다. 이는 깊은 쌍대성을 세운다. 곧 그물로 밀어 보낼 수 있는 최대 흐름의 양은 근원과 바닥을 끊는 데 없애야 하는 최소 담이와 같다. 이 정리는 흐름 알고리즘의 옳음 증명이자 최소 자름을 찾는 실전 방법이 된다.

---

## 1. 자름의 정의

$G = (V, E)$을 근원 $s$과 바닥 $t$을 갖는 흐름 그물이라 하자.

**$s$-$t$ 자름.** $s \in S$이고 $t \in T$인 $V$의 가름 $(S, T)$.

**자름의 담이.** $S$에서 $T$으로 건너가는 변의 담이의 합:

$$
c(S, T) = \sum_{\substack{u \in S,\, v \in T \\ (u,v) \in E}} c(u, v)
$$

$S$에서 $T$으로 가는 변만 센다는 데 유의하라. $T$에서 $S$으로 가는 변은 자름 담이에 보태지지 않는다.

**자름을 건너는 알짜 흐름.** 흐름 $f$에 대해 자름 $(S, T)$을 건너는 알짜 흐름은 다음과 같다:

$$
f(S, T) = \sum_{\substack{u \in S,\, v \in T \\ (u,v) \in E}} f(u, v) - \sum_{\substack{v \in T,\, u \in S \\ (v,u) \in E}} f(v, u)
$$

---

## 2. 약한 쌍대성

!!! note "약한 쌍대성 보조정리"
    아무 흐름 $f$과 아무 $s$-$t$ 자름 $(S, T)$에 대해:

    $$
    |f| = f(S, T) \le c(S, T)
    $$

첫 등식은 흐름 보존에서 따라 나온다. 곧 흐름의 값은 어느 자름을 건너는 알짜 흐름과도 같다. 부등식은 흐름 값마다 그 변의 담이 아래로 묶이므로 따라 나온다.

이로부터 곧바로 $\max_f |f| \le \min_{(S,T)} c(S, T)$이 따라 나온다.

---

## 3. 그 정리

!!! note "최대 흐름 최소 자름 정리(포드와 풀커슨, 1956)"
    흐름 그물에서 다음 세 조건은 서로 같다:

    1. $f$은 최대 흐름이다.
    2. 남은 그래프 $G_f$에 $s$에서 $t$으로 가는 늘림 경로가 없다.
    3. $|f| = c(S, T)$인 $s$-$t$ 자름 $(S, T)$이 있다.

??? example "증명"
    **(1) $\Rightarrow$ (2):** $G_f$에 늘리는 길이 있었다면 흐름을 더 키울 수 있으니 가장 크다는 것과 어긋난다.

    **(2) $\Rightarrow$ (3):** $S = \{v \in V : v \text{ is reachable from } s \text{ in } G_f\}$이라 하고 $T = V \setminus S$이라 하자. 늘리는 길이 없으므로 $t \notin S$이고 $(S, T)$은 옳은 $s$-$t$ 자름이다. $u \in S$이고 $v \in T$인 이음 $(u, v)$마다 나머지 담는 힘이 0이어야 한다(안 그러면 $v$에 닿을 수 있다). 따라서 $f(u, v) = c(u, v)$이다. $v \in T$이고 $u \in S$인 이음 $(v, u)$마다 $f(v, u) = 0$이어야 한다(안 그러면 뒤집은 이음 때문에 $v$에 닿을 수 있다). 그러므로 $|f| = f(S, T) = c(S, T)$이다.

    **(3) $\Rightarrow$ (1):** 여린 짝 성질에 따라 모든 자름 $(S', T')$에 대해 $|f| \le c(S', T')$이다. 어떤 자름에서 $|f| = c(S, T)$이면 $f$은 위 테두리에 닿았으므로 가장 클 수밖에 없다. $\square$

---

## 4. 최소 자름 찾기

어떤 가장 큰 흐름 알고리즘으로든 가장 큰 흐름 $f^*$을 셈한 뒤에

1. 나머지 그래프 $G_{f^*}$을 짓는다.
2. $G_{f^*}$에서 $s$부터 너비 먼저 찾기나 깊이 먼저 찾기를 돌려 닿을 수 있는 꼭짓점 $S$을 모두 찾는다.
3. Set $T = V \setminus S$.
4. 본디 그래프에서 $S$에서 $T$으로 가는 변이 최소 자름을 이룬다.

---

## 5. 구현

```python
"""
최대 흐름을 셈한 뒤 뽑아내어 최소 s-t 자름을 찾는다
남은 그래프에서 닿을 수 있는 꼭짓점을 뽑아낸다.
"""

from collections import deque

# === 에드먼즈-카프 + 최소 자름 뽑기 ===

def min_cut(n: int, edges: list[tuple[int, int, int]],
            source: int, sink: int) -> tuple[int, set, set]:
    """최대 흐름을 셈하고 최소 자름을 뽑아낸다.

    인수:
        n: 꼭짓점의 개수(0부터 셈).
        edges: (u, v, 담이) 튜플의 목록.
        source: 근원 꼭짓점.
        sink: 바닥 꼭짓점.

    반환값:
        튜플 (최대 흐름 값, S, T). 여기서 (S, T)은 최소 자름.
    """
    graph = [[] for _ in range(n)]

    def add_edge(u: int, v: int, cap: int) -> None:
        graph[u].append([v, cap, len(graph[v])])
        graph[v].append([u, 0, len(graph[u]) - 1])

    for u, v, cap in edges:
        add_edge(u, v, cap)

    # 에드먼즈-카프 최대 흐름
    total_flow = 0
    while True:
        parent = [None] * n
        parent[source] = (source, -1)
        queue = deque([source])
        while queue:
            u = queue.popleft()
            for i, (v, cap, _) in enumerate(graph[u]):
                if parent[v] is None and cap > 0:
                    parent[v] = (u, i)
                    if v == sink:
                        break
                    queue.append(v)
            else:
                continue
            break

        if parent[sink] is None:
            break

        bottleneck = float('inf')
        v = sink
        while v != source:
            u, idx = parent[v]
            bottleneck = min(bottleneck, graph[u][idx][1])
            v = u

        v = sink
        while v != source:
            u, idx = parent[v]
            graph[u][idx][1] -= bottleneck
            graph[v][graph[u][idx][2]][1] += bottleneck
            v = u

        total_flow += bottleneck

    # 최소 자름 뽑아내기: 근원에서 남은 그래프를 너비 우선 돌아보기
    visited = set()
    queue = deque([source])
    visited.add(source)
    while queue:
        u = queue.popleft()
        for v, cap, _ in graph[u]:
            if v not in visited and cap > 0:
                visited.add(v)
                queue.append(v)

    s_side = visited
    t_side = set(range(n)) - visited
    return total_flow, s_side, t_side

# === 시연 ===

if __name__ == "__main__":
    edges = [
        (0, 1, 3),  # s -> a
        (0, 2, 2),  # s -> b
        (1, 2, 1),  # a -> b
        (1, 3, 2),  # a -> t
        (2, 3, 3),  # b -> t
    ]
    flow_val, S, T = min_cut(4, edges, 0, 3)
    print(f"Max flow = Min cut capacity = {flow_val}")
    print(f"S = {sorted(S)}")
    print(f"T = {sorted(T)}")

    # 자름 변 보이기
    for u, v, cap in edges:
        if u in S and v in T:
            print(f"  Cut edge: ({u}, {v}), capacity {cap}")
```

**출력:**

```
Max flow = Min cut capacity = 5
S = [0]
T = [1, 2, 3]
Cut edges: (0, 1), capacity 3
Cut edges: (0, 2), capacity 2
```

최소 자름은 근원을 나머지 모두에서 갈라내며 전체 담이는 $3 + 2 = 5$으로, 최대 흐름 최소 자름 정리를 확인해 준다.

---

## 6. 응용

- **그물 믿음성.** 최소 자름은 그물에서 가장 무른 병목을 가려낸다.
- **그림 나누기.** 근원/바닥 이음을 붙인 화소 그래프에서 최소 자름으로 앞바탕과 뒷바탕을 가를 수 있다.
- **이어짐.** 두 꼭짓점을 끊는 데 필요한 변의 최소 개수는 그 둘 사이에 변이 겹치지 않는 경로의 최대 개수와 같다(멩거 정리).

---

## 연습문제

**연습문제 1.**
최대 흐름 최소 자름 정리를 또렷이 말하여라.

??? success "연습문제 1 풀이"
    **정리**: 어떤 흐름 그물에서도 $s$-$t$ 흐름의 가장 큰 값은 $s$-$t$ 자름의 가장 작은 담는 힘과 같다. $s$-$t$ 자름 $(S, T)$은 $V$을 $S \ni s$과 $T \ni t$으로 가르며 담는 힘은 $c(S, T) = \sum_{u \in S, v \in T} c(u, v)$이다. 이 정리는 흐름과 자름 사이의 센 짝 성질을 세워 짜맞춤 가장 좋게 하기와 선형 계획의 짝 성질을 잇는다. $\square$

---

**연습문제 2.**
최대 흐름 풀이가 주어질 때 그에 딸린 최소 자름을 찾는 방법을 설명하여라.

??? success "연습문제 2 풀이"
    가장 큰 흐름을 셈한 뒤 나머지 그래프를 짓는다. 나머지 그래프에서 $s$부터 너비 먼저 찾기나 깊이 먼저 찾기를 돌린다. 닿을 수 있는 꼭짓점 묶음을 $S$이라 하고 $T = V \setminus S$이라 하자. 자름 $(S, T)$이 가장 작은 자름이다. 그 담는 힘은 가장 큰 흐름 값과 같다. 가장 작은 자름의 이음은 본디 그래프에서 $u \in S$이고 $v \in T$인 이음 $(u, v)$ 모두이며, 이 이음들은 가득 차 있다. 가장 큰 흐름을 셈한 뒤 $O(V + E)$이 든다. $\square$

---

**연습문제 3.**
어떤 흐름 값도 어떤 자름의 담이를 넘지 않음을 증명하여라.

??? success "연습문제 3 풀이"
    $f$을 아무 흐름, $(S, T)$을 아무 자름이라 하자. 흐름 값은 $|f| = \sum_{u \in S, v \in T} f(u,v) - \sum_{u \in T, v \in S} f(u,v)$이다(자름을 가로지르는 알짜 흐름). $f(u,v) \leq c(u,v)$이고 $f(u,v) \geq 0$이므로 $|f| \leq \sum_{u \in S, v \in T} c(u,v) - 0 = c(S, T)$이다. 어떤 흐름과 어떤 자름에도 이것이 이루어지므로 가장 큰 흐름 $\leq$ 가장 작은 자름이다. 가장 큰 흐름-가장 작은 자름 정리는 여기서 같음이 이루어짐을 보인다. $\square$

---

**연습문제 4.**
그림 나누기에서 최소 자름의 쓰임새를 설명하여라.

??? success "연습문제 4 풀이"
    그림을 그래프로 그린다. 낱그림점마다 꼭짓점 하나다. 샘 $s$(앞바탕)과 웅덩이 $t$(뒷바탕)을 더한다. 이웃한 낱그림점 사이의 이음 짐은 닮음을 나타낸다(짐이 크면 닮았고 자르는 값이 비싸다). $s$에서 낱그림점으로 가는 이음은 앞바탕일 그럴듯함을, 낱그림점에서 $t$으로 가는 이음은 뒷바탕일 그럴듯함을 나타낸다. 가장 작은 자름이 앞바탕과 뒷바탕을 갈라, 닮지 않은 짝을 자르는 값과 잘못 가른 낱그림점의 값을 합쳐 가장 작게 한다. 주고받는 나누기(GrabCut)나 의료 영상에 쓴다. $\square$

## 정리하며

이 마당은 자름의 정의、약한 쌍대성、그 정리、최소 자름 찾기을 차례로 짚었다.

**참고 문헌**

- Ford, L. R., & Fulkerson, D. R. (1956). Maximal flow through a network. *Canadian Journal of Mathematics*, 8, 399--404.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to Algorithms* (4th ed.), 26장: Maximum Flow.
