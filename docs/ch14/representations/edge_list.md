# 변 목록

변 목록은 가장 곧바른 그래프 표현이다. 곧 변마다 튜플로 저장한다. 행렬이나 이웃 목록처럼 이웃함을 빠르게 묻지는 못하지만, 단순하고 간결하게 저장하므로 변을 차례로 다루는 알고리즘에 딱 맞다. 크러스컬의 최소 뻗음 나무 알고리즘, 벨먼-포드 최단 경로, 그리고 여러 입출력 꼴이 그렇다. 변 목록이 빛나는 때(와 모자란 때)를 알면 그래프 표현 사이의 주고받음이 또렷해진다.

---

## 1. 정의

**변 목록**은 그래프 $G = (V, E)$을 변의 모음으로 나타내며, 변마다 튜플로 저장한다:

- **무게 없음:** 변마다 $(u, v)$.
- **무게 있음:** $(u, v, w)$이며 여기서 $w = w(u, v)$은 변의 무게이다.

무방향 그래프에서는 변마다 한 번만 나타난다($(u, v)$이나 $(v, u)$ 가운데 하나이며 둘 다는 아니다). 방향 그래프에서는 방향 변 $(u, v)$마다 방향과 함께 저장한다.

---

## 2. 복잡도 분석

| 연산 | 시간 | 비고 |
|---|---|---|
| 공간 | $O(E)$ | 변마다 항목 하나 |
| 변 $(u,v)$이 있는지 살피기 | $O(E)$ | 죽 훑어야 한다 |
| $u$의 모든 이웃 찾기 | $O(E)$ | $u$을 찾아 모든 변을 훑는다 |
| 모든 변 훑기 | $O(E)$ | 자연스럽고 차례대로 닿는다 |
| 변 더하기 | $O(1)$ | 목록 끝에 붙인다 |
| 변 지우기 | $O(E)$ | 찾아서 지운다 |
| 무게로 정렬 | $O(E \log E)$ | 크러스컬에 필요하다 |

이웃함을 묻는 데 드는 $O(E)$이 가장 큰 흠이다. "$(u, v)$이 변인가?"를 되풀이해 살피는 알고리즘에는 이웃 목록이나 행렬이 훨씬 효율적이다.

---

## 3. 변 목록을 언제 쓰나

변 목록은 다음 몇 가지 상황에서 고를 만한 표현이다:

1. **변 다루기 알고리즘.** 크러스컬의 MST 알고리즘은 변을 무게로 정렬해 차례로 다룬다. 벨먼-포드는 되풀이마다 모든 변을 늦춘다. 둘 다 변 목록을 자연스럽게 쓴다.

2. **입력 뜯어 읽기.** 그래프 문제는 흔히 변의 목록으로 주어진다. 그대로 저장하면 필요하지도 않은 이웃 짜임을 쌓는 짐을 피할 수 있다.

3. **아주 성긴 그래프.** $|E| \ll |V|$이면 빈 목록 $|V|$개를 가진 이웃 목록은 공간을 헤프게 쓴다. 변 목록은 $O(E)$ 공간만 쓴다.

4. **바뀌지 않는 그래프.** 쌓은 뒤 그래프가 바뀌지 않고 알고리즘이 변을 훑기만 한다면, 변 목록이 캐시에 가장 상냥한 차례 짜임이다.

---

## 4. 구현

```python
"""
그래프의 변 목록 표현.

변 목록 쌓기, 기본 연산, 그리고 이웃 물음이 필요한
알고리즘을 위해 이웃 목록으로 바꾸기를
보인다.
"""

# === 변 목록 클래스 ===

class EdgeListGraph:
    """변의 목록으로 나타낸 그래프."""

    def __init__(self, n_vertices, directed=False):
        self.n = n_vertices
        self.directed = directed
        self.edges = []

    def add_edge(self, u, v, weight=None):
        """그래프에 변을 더한다."""
        if weight is not None:
            self.edges.append((u, v, weight))
        else:
            self.edges.append((u, v))

    def has_edge(self, u, v):
        """변 (u, v)이 있는지 살핀다. $O(E)$ 시간."""
        for edge in self.edges:
            eu, ev = edge[0], edge[1]
            if eu == u and ev == v:
                return True
            if not self.directed and eu == v and ev == u:
                return True
        return False

    def neighbors(self, u):
        """꼭짓점 u의 이웃을 모두 찾는다. $O(E)$ 시간."""
        result = []
        for edge in self.edges:
            eu, ev = edge[0], edge[1]
            if eu == u:
                result.append(ev)
            elif not self.directed and ev == u:
                result.append(eu)
        return result

    def to_adjacency_list(self):
        """이웃 목록 표현으로 바꾼다. $O(E)$ 시간."""
        adj = [[] for _ in range(self.n)]
        for edge in self.edges:
            u, v = edge[0], edge[1]
            w = edge[2] if len(edge) == 3 else None
            adj[u].append((v, w) if w is not None else v)
            if not self.directed:
                adj[v].append((u, w) if w is not None else u)
        return adj

    def sorted_by_weight(self):
        """무게로 정렬한 변을 되돌린다(크러스컬용)."""
        return sorted(self.edges, key=lambda e: e[2])

# === 메인 ===

if __name__ == "__main__":
    # 무게 있는 무방향 그래프
    g = EdgeListGraph(5, directed=False)
    g.add_edge(0, 1, 4)
    g.add_edge(0, 2, 1)
    g.add_edge(1, 2, 2)
    g.add_edge(1, 3, 5)
    g.add_edge(2, 4, 3)

    print("Edge list:")
    for e in g.edges:
        print(f"  {e}")

    print(f"\nEdge (0,1) exists: {g.has_edge(0, 1)}")
    print(f"Edge (3,4) exists: {g.has_edge(3, 4)}")
    print(f"Neighbors of 2: {g.neighbors(2)}")

    print("\nEdges sorted by weight (for Kruskal's):")
    for e in g.sorted_by_weight():
        print(f"  {e[0]}-{e[1]} weight={e[2]}")

    print("\nConverted to adjacency list:")
    adj = g.to_adjacency_list()
    for v in range(5):
        print(f"  {v}: {adj[v]}")
```

**출력:**
```
Edge list:
  (0, 1, 4)
  (0, 2, 1)
  (1, 2, 2)
  (1, 3, 5)
  (2, 4, 3)
Edge (0,1) exists: True
Edge (3,4) exists: False
Neighbors of 2: [0, 1, 4]
Edges sorted by weight (for Kruskal's):
  0-2 weight=1
  1-2 weight=2
  2-4 weight=3
  0-1 weight=4
  1-3 weight=5
Converted to adjacency list:
  0: [(1, 4), (2, 1)]
  1: [(0, 4), (2, 2), (3, 5)]
  2: [(0, 1), (1, 2), (4, 3)]
  3: [(1, 5)]
  4: [(2, 3)]
```

---

## 5. 다른 표현과의 견줌

모든 연산에 걸친 자세한 주고받음 분석은 [표현 방식의 견줌](comparison.md)을 보아라. 핵심 통찰은 변 목록이 묻는 속도를 내주고 저장의 단순함과 차례로 닿는 효율을 얻는다는 것이다.

---

## 연습문제

**연습문제 1.**
변 $m$개의 변 목록이 주어졌을 때 모든 꼭짓점의 차수를 $O(m)$ 시간에 셈하는 법을 밝혀라.

??? success "연습문제 1 풀이"
    크기 $n$(꼭짓점의 개수)의 차수 배열을 모두 0으로 첫걸음 잡는다. 변 목록을 훑으며 변 $(u, v)$마다 $\text{degree}[u]$과 $\text{degree}[v]$을 1씩 올린다(무방향 그래프). 방향 그래프에서 나가는 차수라면 $\text{degree}[u]$만 올린다. 변 $m$개를 한 번 훑으므로 $O(m)$ 시간이 든다. $\square$

---

**연습문제 2.**
크러스컬의 MST 알고리즘은 변을 무게 차례로 다룬다. 변 목록 표현이 왜 이 알고리즘에 특히 잘 맞는가?

??? success "연습문제 2 풀이"
    크러스컬 알고리즘은 모든 변을 무게로 정렬한 뒤 차례로 다루며 고리를 만들지 않는 변을 더한다. 변 목록은 모든 변을 배열 하나에 저장하므로 $O(E \log E)$ 시간에 곧바로 정렬할 수 있다. 바꿀 필요가 없다. 이웃 목록이나 행렬이라면 정렬하기 전에 모든 변을 목록으로 뽑아내는 걸음이 더 든다. 변 목록은 기억 공간도 가장 적게($O(E)$) 쓰므로 이 쓰임에 가장 좋다. $\square$

---

**연습문제 3.**
변 목록에서 변 $(u, v)$이 있는지 어떻게 살피겠는가? 시간 복잡도는 얼마인가? 되풀이해 찾는 일을 빠르게 할 방법을 내놓아라.

??? success "연습문제 3 풀이"
    변 목록을 죽 훑으며 항목마다 살피면 물음마다 $O(E)$이 든다. 되풀이해 찾는 일을 빠르게 하려면 변 목록을 사전 차례로 정렬하고 이분 찾기를 써서 물음마다 $O(\log E)$으로 줄인다. 아니면 변 튜플의 해시 집합을 쌓아 기대 $O(1)$에 찾되 공간을 $O(E)$ 더 쓴다. $\square$

---

**연습문제 4.**
꼭짓점 5개의 그래프에 대해 다음 변 목록을 이웃 목록으로 바꿔라: $\{(0,1), (0,3), (1,2), (2,3), (2,4)\}$. 나온 이웃 목록을 적어라.

??? success "연습문제 4 풀이"
    변마다 다루며 양쪽 방향을 모두 더한다(무방향 그래프):

    - 꼭짓점 0: $[1, 3]$
    - 꼭짓점 1: $[0, 2]$
    - 꼭짓점 2: $[1, 3, 4]$
    - 꼭짓점 3: $[0, 2]$
    - 꼭짓점 4: $[2]$

    바꾸는 데 $O(V + E)$ 시간이 든다. 곧 빈 목록 $V$개를 첫걸음 잡고 변 $E$개를 훑으며 양끝 목록에 더한다. $\square$

## 정리하며

이 마당은 정의、복잡도 분석、변 목록을 언제 쓰나、구현을 차례로 짚었다.

**참고 문헌**

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. 22장.
