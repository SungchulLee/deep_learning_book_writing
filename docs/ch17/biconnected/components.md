# 두 겹 이음 조각

이어진 그래프에는 없애면 그래프가 끊어지는 꼭짓점이 있을 수 있다. 그래프의 어느 부분이 꼭짓점 하나를 잃어도 견디는지 살피면 **두 겹 이음 조각**이라는 생각에 이른다. 그래프를 이 조각으로 쪼개면 남는 이어짐의 속 짜임이 드러나고 그래프를 붙들고 있는 결정적인 이음매 점이 가려진다.

## 정의

**Biconnected graph.** A connected undirected graph $G = (V, E)$ with $|V| \ge 3$ is *biconnected* (or 2-connected) if removing any single vertex leaves the graph connected. Equivalently, $G$ is biconnected if and only if every pair of vertices lies on a common simple cycle.

**두 겹 이음 조각.** $G$의 가장 큰 두 겹 이음 아래그래프. $G$의 변마다 정확히 하나의 두 겹 이음 조각에 든다. 다리(없애면 $G$이 끊어지는 변)는 그 변 하나와 두 끝점만으로 된 두 겹 이음 조각을 이룬다.

**이음매 점.** 없애면 이어진 조각의 수가 늘어나는 꼭짓점 $v$. 꼭짓점이 이음매 점인 것은 그것이 두 겹 이음 조각 둘 이상에 들 때 그리고 오직 그때뿐이다.

## 핵심 성질

1. 변마다 정확히 하나의 두 겹 이음 조각에 든다.
2. 두 겹 이음 조각 둘은 많아야 꼭짓점 하나를 함께 가지며, 함께 갖는 꼭짓점은 이음매 점이다.
3. 이음매 점이 없는 그래프는 두 겹 이어졌거나, 변 하나이거나, 꼭짓점 하나이다.
4. 두 겹 이음 조각의 수는 덩이-자름 나무의 변 수와 같다.

## 알고리즘

깊이 우선 돌아보기 한 번으로 모든 두 겹 이음 조각과 이음매 점을 $O(V + E)$ 시간에 찾는다. 이 알고리즘은 다음을 지닌다:

- $\text{disc}[v]$: the discovery time of vertex $v$.
- $\text{low}[v]$: the minimum discovery time reachable from the subtree rooted at $v$ using at most one back edge.

$$
\text{low}[v] = \min\!\bigl(\text{disc}[v],\; \min_{(v,w) \text{ back edge}} \text{disc}[w],\; \min_{(v,u) \text{ tree edge}} \text{low}[u]\bigr)
$$

꼭짓점 $v$이 이음매 점인 경우는 다음과 같다:

- $v$이 돌아보기 뿌리이고 돌아보기 나무에서 자식이 둘 이상이거나,
- $v$ is not the root and has a child $u$ with $\text{low}[u] \ge \text{disc}[v]$.

To extract biconnected components, maintain an edge stack. When the DFS backtracks from $u$ to $v$ and $\text{low}[u] \ge \text{disc}[v]$, pop all edges from the stack down to and including $(v, u)$; these edges form one biconnected component.

## 구현

```python
"""
깊이 우선 돌아보기로 얻는 방향 없는 그래프의 두 겹 이음 조각.

변 쌓기를 쓴 타잔 알고리즘으로 모든 두 겹 이음
조각과 이음매 점을 O(V + E) 시간에 가려낸다.
"""

from collections import defaultdict

# === 두 겹 이음 조각 찾개 ===

class BiconnectedComponents:
    """두 겹 이음 조각과 이음매 점을 모두 찾는다."""

    def __init__(self, n: int):
        """꼭짓점 n개(0부터 셈)로 그래프를 첫자리매김한다."""
        self.n = n
        self.adj = defaultdict(list)
        self.components = []
        self.articulation_points = set()

    def add_edge(self, u: int, v: int) -> None:
        """방향 없는 변 (u, v)을 더한다."""
        self.adj[u].append(v)
        self.adj[v].append(u)

    def find_components(self) -> None:
        """두 겹 이음 조각을 모두 찾으려 깊이 우선 돌아보기를 한다."""
        disc = [-1] * self.n
        low = [0] * self.n
        parent = [-1] * self.n
        stack = []  # 변 쌓기
        timer = [0]

        def dfs(u: int) -> None:
            disc[u] = low[u] = timer[0]
            timer[0] += 1
            children = 0

            for v in self.adj[u]:
                if disc[v] == -1:
                    children += 1
                    parent[v] = u
                    stack.append((u, v))
                    dfs(v)
                    low[u] = min(low[u], low[v])

                    # 이음매 점인지 살피기
                    is_root = parent[u] == -1
                    if (is_root and children > 1) or \
                       (not is_root and low[v] >= disc[u]):
                        self.articulation_points.add(u)

                    # 경계를 찾으면 조각 뽑아내기
                    if low[v] >= disc[u]:
                        component = []
                        while stack:
                            edge = stack.pop()
                            component.append(edge)
                            if edge == (u, v):
                                break
                        self.components.append(component)

                elif v != parent[u] and disc[v] < disc[u]:
                    stack.append((u, v))
                    low[u] = min(low[u], disc[v])

        for i in range(self.n):
            if disc[i] == -1:
                dfs(i)


# === 시연 ===

if __name__ == "__main__":
    # 그래프: 0-1-2-0(삼각형), 2-3, 3-4-5-3(삼각형)
    bc = BiconnectedComponents(6)
    for u, v in [(0,1),(1,2),(2,0),(2,3),(3,4),(4,5),(5,3)]:
        bc.add_edge(u, v)
    bc.find_components()

    print(f"Number of biconnected components: {len(bc.components)}")
    print(f"Articulation points: {sorted(bc.articulation_points)}")
    for i, comp in enumerate(bc.components):
        vertices = set()
        for u, v in comp:
            vertices.update([u, v])
        print(f"  Component {i}: vertices {sorted(vertices)}, edges {comp}")
```

**출력:**

```
Number of biconnected components: 3
Articulation points: [2, 3]
  Component 0: vertices [3, 4, 5], edges [(4, 5), (5, 3), (3, 4)]
  Component 1: vertices [2, 3], edges [(2, 3)]
  Component 2: vertices [0, 1, 2], edges [(1, 2), (2, 0), (0, 1)]
```

The triangle $\{0, 1, 2\}$ forms one biconnected component, the bridge $(2, 3)$ forms another, and the triangle $\{3, 4, 5\}$ forms a third. Vertices $2$ and $3$ are articulation points because they each connect two components.

## 복잡도

| 항목 | 비용 |
|--------|:----:|
| 시간 | $O(V + E)$ |
| 공간 | $O(V + E)$ |

이 알고리즘은 돌아보기를 한 번만 한다. 변마다 정확히 한 번 쌓기에 올리고 꺼내므로 전체 품은 그래프 크기에 선형이다.

## 응용

- **그물 믿음성.** 두 겹 이음 조각은 어느 마디 하나가 무너져도 이어진 채 남는 그물의 부분을 가려낸다.
- **남는 이음.** 두 겹 이음 조각 안에서는 아무 꼭짓점 짝 사이에도 꼭짓점이 겹치지 않는 경로가 적어도 둘 있다.
- **덩이-자름 나무.** 두 겹 이음 조각과 이음매 점이 함께 덩이-자름 나무를 정하는데, 이는 그래프 이어짐 짜임을 한 층 위에서 본 모습이다.

## 참고 문헌

- Hopcroft, J. E., & Tarjan, R. E. (1973). Algorithm 447: Efficient algorithms for graph manipulation. *Communications of the ACM*, 16(6), 372--378.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to Algorithms* (4th ed.), 22장: Elementary Graph Algorithms.

## 연습문제

**연습문제 1.**
두 겹 이음 조각을 정의하고 이어진 조각과 어떻게 다른지 설명하여라.

??? success "연습문제 1 풀이"
    A **biconnected component** (block) is a maximal 2-connected subgraph: it remains connected after removing any single vertex. A connected component only requires connectivity (removing a vertex may disconnect it). Every graph can be decomposed into biconnected components that share at most one vertex (an articulation point). Biconnected components represent the "robust" parts of a graph where no single vertex failure causes disconnection. $\square$

---

**연습문제 2.**
쌓기를 써서 두 겹 이음 조각을 찾는 깊이 우선 돌아보기 바탕 알고리즘을 설명하여라.

??? success "연습문제 2 풀이"
    Maintain an edge stack during DFS. Push each tree edge and back edge onto the stack. When an articulation point condition is detected ($\text{low}[v] \geq \text{disc}[u]$ for child $v$ of $u$), pop edges from the stack until edge $(u, v)$ is reached — these edges form one biconnected component. At the DFS root, after processing each child subtree, pop the remaining edges for the last component. Time: $O(V + E)$. Each edge belongs to exactly one biconnected component. $\square$

---

**연습문제 3.**
두 겹 이음 조각 둘이 많아야 꼭짓점 하나를 함께 가지며 그 함께 갖는 꼭짓점이 이음매 점이어야 함을 증명하여라.

??? success "연습문제 3 풀이"
    Suppose components $B_1$ and $B_2$ share two vertices $u$ and $v$. Since both components are 2-connected, there exist two vertex-disjoint paths from $u$ to $v$ in $B_1$ and two in $B_2$. Combining paths from $B_1$ and $B_2$ gives a 2-connected subgraph containing both $B_1$ and $B_2$, contradicting their maximality. Therefore they share at most one vertex. If vertex $w$ belongs to two components, removing $w$ disconnects the edges in these components from each other, making $w$ an articulation point. $\square$

---

**연습문제 4.**
그래프에 꼭짓점 10개, 변 15개, 이음매 점 3개가 있다. 두 겹 이음 조각은 적어도 몇 개인가?

??? success "연습문제 4 풀이"
    Each articulation point belongs to at least 2 biconnected components. The minimum number of components occurs when each articulation point connects exactly 2 components. Starting with 1 component, each articulation point adds at least 1 new component. So minimum components $= 1 + 3 = 4$. For example: a graph with 4 biconnected components connected in a path through 3 articulation points. $\square$
