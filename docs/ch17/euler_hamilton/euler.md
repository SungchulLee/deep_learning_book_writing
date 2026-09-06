# 오일러 경로와 회로

쾨니히스베르크의 일곱 다리 수수께끼는 다리마다 정확히 한 번씩 건너 도시를 걸어 돌고 처음 자리로 돌아올 수 있는지 물었다. 오일러는 1736년에 그런 걸음이 없음을 증명하며 그래프 이론이라는 분야를 열었다. 핵심 눈썰미는 그런 돌아보기가 있는지가 오로지 꼭짓점 **차수의 홀짝**에 달렸다는 것이다.

## 정의

**오일러 회로.** 변마다 정확히 한 번씩 들르고 처음 꼭짓점으로 돌아오는, 그래프의 닫힌 걸음.

**오일러 경로.** 변마다 정확히 한 번씩 들르되 시작과 끝 꼭짓점이 다를 수 있는 걸음.

**오일러 그래프.** 오일러 회로가 있는 그래프. (회로는 없고) 오일러 경로만 있는 그래프를 *반오일러*라 한다.

## 있음 정리

### 방향 없는 그래프

!!! note "오일러 정리(방향 없음)"
    $G = (V, E)$을 (외딴 꼭짓점을 뺀) 이어진 방향 없는 그래프라 하자. 그러면:

    - $G$에 **오일러 회로**가 있는 것은 꼭짓점마다 차수가 짝수일 때 그리고 오직 그때뿐이다.
    - $G$에 (회로는 없고) **오일러 경로**가 있는 것은 차수가 홀수인 꼭짓점이 정확히 둘일 때 그리고 오직 그때뿐이다. 그 경로는 홀수 차수 꼭짓점 하나에서 시작해 다른 하나에서 끝나야 한다.

??? example "회로인 경우의 증명 얼개"
    **필요함.** 오일러 회로가 있으면 걸음이 어떤 변으로 꼭짓점에 들어갈 때마다 다른 변으로 나와야 한다. 따라서 꼭짓점마다 들어가고 나온 횟수가 같으므로 그 차수는 짝수이다.

    **Sufficiency.** Start at any vertex and follow unused edges until returning to the start (this must happen since every vertex has even degree). If some edges remain unused, there exists a vertex $v$ on the current walk that is incident to an unused edge (by connectivity). Start a new walk from $v$ using only unused edges, splice it into the first walk, and repeat until all edges are covered. $\square$

### 방향 그래프

!!! note "오일러 정리(방향)"
    $G = (V, E)$을 차수가 0이 아닌 꼭짓점이 모두 같은 강한 이음 조각에 드는 방향 그래프라 하자. 그러면:

    - $G$ has an **Eulerian circuit** if and only if $\text{in-deg}(v) = \text{out-deg}(v)$ for every vertex $v$.
    - $G$ has an **Eulerian path** if and only if there is exactly one vertex with $\text{out-deg} - \text{in-deg} = 1$ (the start) and one vertex with $\text{in-deg} - \text{out-deg} = 1$ (the end), with all other vertices balanced.

## 있는지 살피기

```python
"""
방향 없는 그래프에 오일러 경로나 회로가 있는지 살핀다.

차수 홀짝 정리를 쓴다. 곧 홀수 차수 꼭짓점을 세어
오일러 회로가 있는지, 오일러 경로가 있는지, 둘 다 없는지 정한다.
"""

from collections import defaultdict

# === 오일러가 있는지 살피기 ===

def euler_type(n: int, edges: list[tuple[int, int]]) -> str:
    """그래프에 오일러 회로가 있는지, 경로가 있는지, 둘 다 없는지 정한다.

    인수:
        n: 꼭짓점의 개수(0부터 셈).
        edges: 방향 없는 변 (u, v)의 목록.

    반환값:
        'circuit', 'path', 또는 'none'.
    """
    degree = [0] * n
    adj = defaultdict(set)

    for u, v in edges:
        degree[u] += 1
        degree[v] += 1
        adj[u].add(v)
        adj[v].add(u)

    # 차수가 0이 아닌 꼭짓점끼리 이어졌는지 살피기
    nonzero = [v for v in range(n) if degree[v] > 0]
    if not nonzero:
        return "circuit"  # 아무것도 아님: 변 없음

    visited = set()
    stack = [nonzero[0]]
    while stack:
        v = stack.pop()
        if v in visited:
            continue
        visited.add(v)
        for w in adj[v]:
            if w not in visited:
                stack.append(w)

    if visited != set(nonzero):
        return "none"  # 이어져 있지 않음

    odd_count = sum(1 for v in range(n) if degree[v] % 2 == 1)
    if odd_count == 0:
        return "circuit"
    elif odd_count == 2:
        return "path"
    else:
        return "none"


# === 시연 ===

if __name__ == "__main__":
    # 삼각형: 0-1-2-0(차수가 모두 짝수)
    print(euler_type(3, [(0,1),(1,2),(2,0)]))  # 회로

    # 경로 그래프: 0-1-2(꼭짓점 0,2의 차수가 홀수)
    print(euler_type(3, [(0,1),(1,2)]))  # 경로

    # 별: 가운데 0, 잎 1,2,3(꼭짓점 0의 차수는 3)
    print(euler_type(4, [(0,1),(0,2),(0,3)]))  # 없음
```

**출력:**

```
circuit
path
none
```

삼각형은 차수가 모두 짝수이므로 오일러 회로를 허락한다. 경로 그래프 $0{-}1{-}2$은 홀수 차수 꼭짓점이 정확히 둘($0$과 $2$)이므로 오일러 경로를 허락한다. 별 그래프는 꼭짓점 $0$의 차수가 $3$이고 차수 $1$인 꼭짓점이 셋이어서 홀수 차수 꼭짓점이 넷이므로 오일러 경로도 회로도 없다.

## 복잡도

| 갈래 | 있는지 살피기 | 회로/경로 찾기 |
|--------|:---------------:|:-----------------------:|
| 시간 | $O(V + E)$ | $O(V + E)$(히어홀처) |
| 공간 | $O(V + E)$ | $O(E)$ |

있는지 살피는 데는 차수를 셈하고 이어짐을 확인하는 한 번의 훑기면 된다. 실제 오일러 회로나 경로를 세우는 일은 따로 다루는 히어홀처 알고리즘이 맡는다.

## 차수 홀짝 간추림

| 홀수 차수 꼭짓점 | 방향 없는 그래프에 있는 것 |
|:-------------------:|:--------------------:|
| $0$ | 오일러 회로 |
| $2$ | 오일러 경로만 |
| 그 밖의 수 | 둘 다 없음 |

## 참고 문헌

- Euler, L. (1736). Solutio problematis ad geometriam situs pertinentis. *Commentarii academiae scientiarum Petropolitanae*, 8, 128--140.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to Algorithms* (4th ed.), 22장: Elementary Graph Algorithms.

## 연습문제

**연습문제 1.**
방향 없는 그래프에서 오일러 회로와 오일러 경로가 있을 필요충분조건을 말하여라.

??? success "연습문제 1 풀이"
    **Eulerian circuit** (visits every edge exactly once and returns to start): The graph must be connected (ignoring isolated vertices) and every vertex must have even degree. **Eulerian path** (visits every edge exactly once, different start and end): The graph must be connected and have exactly 0 or 2 vertices of odd degree. If 0 odd-degree vertices, an Eulerian circuit exists (which is also a path). If 2 odd-degree vertices, the path starts at one and ends at the other. $\square$

---

**연습문제 2.**
이어진 그래프의 꼭짓점마다 차수가 짝수이면 그 그래프에 오일러 회로가 있음을 증명하여라.

??? success "연습문제 2 풀이"
    Start a walk from any vertex. At each step, traverse an unused edge. Since every vertex has even degree, whenever we enter a vertex, there is always an unused edge to leave (the entering edge uses one of the even number of edges). The walk must return to the start (we cannot get stuck elsewhere). If the walk misses some edges, they form a subgraph where every vertex still has even degree. Start a new walk from a vertex shared with the main walk and splice it in. Repeat until all edges are covered. $\square$

---

**연습문제 3.**
그 조건을 방향 그래프로 넓혀라. 방향 그래프에 오일러 회로가 있는 때는 언제인가?

??? success "연습문제 3 풀이"
    A directed graph has an **Eulerian circuit** if and only if: (1) the graph is strongly connected (ignoring isolated vertices), and (2) every vertex has equal in-degree and out-degree ($\deg^-(v) = \deg^+(v)$ for all $v$). For an **Eulerian path**: at most one vertex has $\deg^+(v) - \deg^-(v) = 1$ (start) and at most one has $\deg^-(v) - \deg^+(v) = 1$ (end), with all others balanced, and the underlying graph is weakly connected. $\square$

---

**연습문제 4.**
A graph has 7 vertices and the following degrees: $\{4, 4, 4, 2, 2, 2, 2\}$. Does it have an Eulerian circuit? An Eulerian path?

??? success "연습문제 4 풀이"
    All degrees are even, so if the graph is connected, it has an Eulerian circuit (and hence also an Eulerian path). The sum of degrees is $4+4+4+2+2+2+2 = 20$, giving 10 edges. If the graph is connected (which is possible with these degrees), an Eulerian circuit exists. Without connectivity information, we cannot be certain — the graph might be disconnected with separate even-degree components. $\square$
