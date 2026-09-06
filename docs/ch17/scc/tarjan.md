# 타잔 알고리즘

[코사라주 알고리즘](kosaraju.md)은 깊이 우선 돌아보기를 두 번 하고 뒤집은 그래프를 세워야 강하게 이어진 조각을 찾는다. 타잔 알고리즘은 돌아보기 한 번으로 뒤집은 그래프 없이 같은 결과를 얻으므로 실전에서 공간을 더 아낀다. 핵심 생각은 꼭짓점마다 뒤로 가는 변으로 닿을 수 있는 가장 이른 조상을 좇고, 그 앎을 써서 돌아보기 도중에 조각마다 "뿌리"를 가려내는 것이다.

## 낮은 이음 값

타잔 알고리즘은 꼭짓점 $u$마다 두 값을 매긴다:

- $\text{disc}[u]$: the discovery time of $u$ in the DFS (the order in which $u$ is first visited).
- $\text{low}[u]$: the smallest discovery time reachable from $u$ through the DFS subtree of $u$, including back edges.

낮은 이음 값은 되돌이로 정의한다:

$$
\text{low}[u] = \min\!\Big(\text{disc}[u],\ \min_{(u,v) \in E} \text{low}[v],\ \min_{\substack{(u,v) \in E \\ v \text{ on stack}}} \text{disc}[v]\Big)
$$

A vertex $u$ is the **root** of an SCC if $\text{low}[u] = \text{disc}[u]$. This means $u$ cannot reach any vertex discovered earlier than itself, so $u$ and all vertices above it on the stack form a maximal strongly connected set.

## 알고리즘

1. Maintain a global timer, a stack, and arrays for $\text{disc}$, $\text{low}$, and whether each vertex is on the stack.
2. 들르지 않은 꼭짓점 $u$마다 깊이 우선 돌아보기를 한다:
    - Set $\text{disc}[u] = \text{low}[u] = \text{timer}$; increment timer.
    - $u$을 쌓기에 올린다.
    - $u$의 각 이웃 $v$에 대해:
        - If $v$ is unvisited, recurse on $v$ and set $\text{low}[u] = \min(\text{low}[u], \text{low}[v])$.
        - If $v$ is on the stack, set $\text{low}[u] = \min(\text{low}[u], \text{disc}[v])$.
    - After processing all neighbors, if $\text{low}[u] = \text{disc}[u]$, pop vertices from the stack until $u$ is popped. These vertices form one SCC.

## 올바름

!!! note "뿌리 찾기가 되는 까닭"
    A vertex $u$ with $\text{low}[u] = \text{disc}[u]$ is the first-discovered vertex in its SCC. All other vertices $v$ in the same SCC have $\text{low}[v] < \text{disc}[v]$ because they can reach $u$ (or an earlier vertex) through back edges. When the DFS backtracks to $u$ and finds $\text{low}[u] = \text{disc}[u]$, every vertex of $u$'s SCC is on the stack above $u$, so popping until $u$ extracts exactly one SCC.

**핵심 불변량:** 언제나 쌓기에는 아직 조각이 온전히 가려지지 않은 꼭짓점이 들어 있다. 꼭짓점은 자기 조각의 뿌리를 찾을 때까지 쌓기에 남는다.

## 복잡도

돌아보기 한 번으로 꼭짓점과 변을 정확히 한 번씩 들른다:

$$
T(V, E) = O(V + E)
$$

공간 복잡도는 쌓기, 찾은 시각, 낮은 이음 값 때문에 $O(V)$이며, 뒤집은 그래프는 필요 없다.

## 구현

```python
"""
강하게 이어진 조각을 찾는 타잔 알고리즘.

낮은 이음 값을 쓰는 한 번의 깊이 우선 돌아보기로 강한 이음 조각의 뿌리를 가려내고
쌓기에서 조각을 뽑아낸다.
"""


# === 타잔의 강한 이음 조각 알고리즘 ===
def tarjan_scc(graph, n):
    """
    타잔 알고리즘으로 강하게 이어진 조각을 모두 찾는다.

    매개변수
    ----------
    graph : dict[int, list[int]]
        꼭짓점이 0부터 n-1인 방향 그래프의 이웃 목록.
    n : int
        꼭짓점의 개수.

    반환값
    -------
    list[list[int]]
        강한 이음 조각의 목록. 저마다 꼭짓점 이름의 목록이다.
    """
    disc = [-1] * n
    low = [0] * n
    on_stack = [False] * n
    stack = []
    timer = [0]
    sccs = []

    def dfs(u):
        disc[u] = low[u] = timer[0]
        timer[0] += 1
        stack.append(u)
        on_stack[u] = True

        for v in graph.get(u, []):
            if disc[v] == -1:
                dfs(v)
                low[u] = min(low[u], low[v])
            elif on_stack[v]:
                low[u] = min(low[u], disc[v])

        # u가 강한 이음 조각의 뿌리이면
        if low[u] == disc[u]:
            component = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                component.append(w)
                if w == u:
                    break
            sccs.append(component)

    for u in range(n):
        if disc[u] == -1:
            dfs(u)

    return sccs


# === 메인 ===
if __name__ == "__main__":
    graph = {
        0: [1], 1: [2, 3], 2: [0], 3: [4],
        4: [5], 5: [3], 6: [5, 7], 7: [],
    }
    sccs = tarjan_scc(graph, 8)
    print("Strongly connected components:")
    for i, scc in enumerate(sccs):
        print(f"  C{i+1} = {sorted(scc)}")
```

**출력:**
```
Strongly connected components:
  C1 = [3, 4, 5]
  C2 = [0, 1, 2]
  C3 = [7]
  C4 = [6]
```

Note that Tarjan's algorithm outputs SCCs in reverse topological order of the [condensation graph](condensation.md). The SCC containing vertices $\{3, 4, 5\}$ appears first because it is a sink SCC -- no edges leave it to other SCCs.

## 한 걸음씩 따라가기

꼭짓점 0에서 깊이 우선 돌아보기를 시작하면:

| 단계 | 꼭짓점 | disc | low | 쌓기 | 하는 일 |
|---|---|---|---|---|---|
| 1 | 0 | 0 | 0 | [0] | 0에 들름 |
| 2 | 1 | 1 | 1 | [0,1] | 1에 들름 |
| 3 | 2 | 2 | 2 | [0,1,2] | 2에 들름 |
| 4 | 2→0 | - | low[2]=0 | [0,1,2] | 뒤로 가는 변, low 고침 |
| 5 | 1←2 | - | low[1]=0 | [0,1,2] | 되돌아옴, low[1]=min(1,0) |
| 6 | 3 | 3 | 3 | [0,1,2,3] | 3에 들름 |
| 7 | 4 | 4 | 4 | [0,1,2,3,4] | 4에 들름 |
| 8 | 5 | 5 | 5 | [0,1,2,3,4,5] | 5에 들름 |
| 9 | 5→3 | - | low[5]=3 | [0,1,2,3,4,5] | 뒤로 가는 변 |
| 10 | 4←5 | - | low[4]=3 | [0,1,2,3,4,5] | 되돌아옴 |
| 11 | 3←4 | - | low[3]=3 | [0,1,2,3,4,5] | low[3]==disc[3], 조각 꺼냄: {3,4,5} |
| 12 | 1←3 | - | low[1]=0 | [0,1,2] | 되돌아옴 |
| 13 | 0←1 | - | low[0]=0 | [0,1,2] | low[0]==disc[0], 조각 꺼냄: {0,1,2} |

## 참고 문헌

- Tarjan, R. E. (1972). Depth-first search and linear graph algorithms. *SIAM Journal on Computing*, 1(2), 146-160.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. 20장.

## 연습문제

**연습문제 1.**
타잔 알고리즘에서 낮은 이음 값이 하는 몫을 설명하여라.

??? success "연습문제 1 풀이"
    The **low-link value** $\text{low}[v]$ is the smallest discovery time reachable from $v$'s subtree (including via back edges and cross edges within the current DFS stack). When $\text{low}[v] = \text{disc}[v]$ for a vertex $v$, it means $v$ is the root of its SCC — no vertex in $v$'s subtree can reach an ancestor of $v$ in the DFS stack. At this point, all vertices on the stack from $v$ to the top form one SCC. $\square$

---

**연습문제 2.**
Trace Tarjan's algorithm on the graph: edges $\{(0,1),(1,2),(2,0),(1,3),(3,4),(4,3)\}$.

??? success "연습문제 2 풀이"
    DFS from 0: disc[0]=0, low[0]=0. Explore 1: disc[1]=1, low[1]=1. Explore 2: disc[2]=2, low[2]=2. Edge (2,0): 0 is on stack, low[2]=min(2,0)=0. Back to 1: low[1]=min(1,0)=0. Explore 3: disc[3]=3, low[3]=3. Explore 4: disc[4]=4, low[4]=4. Edge (4,3): 3 on stack, low[4]=min(4,3)=3. Back to 3: low[3]=min(3,3)=3. low[3]==disc[3], so SCC={3,4}. Back to 1: low[1]=0. Back to 0: low[0]=min(0,0)=0. low[0]==disc[0], so SCC={0,1,2}. Result: two SCCs: $\{0,1,2\}$ and $\{3,4\}$. $\square$

---

**연습문제 3.**
타잔 알고리즘은 왜 쌓기를 쓰는가? 없으면 무엇이 잘못되는가?

??? success "연습문제 3 풀이"
    The stack tracks vertices in the current DFS path that have not yet been assigned to an SCC. When an SCC root is found ($\text{low}[v] = \text{disc}[v]$), all vertices above $v$ on the stack belong to $v$'s SCC. Without the stack, we could not distinguish between vertices in the current SCC and vertices already assigned to other SCCs. Cross edges to finished SCCs would incorrectly lower low-link values if we did not check stack membership. $\square$

---

**연습문제 4.**
타잔 알고리즘이 $O(V + E)$ 시간에 도는 것을 증명하여라.

??? success "연습문제 4 풀이"
    Each vertex is visited exactly once by DFS: $O(V)$. Each edge is examined exactly once: $O(E)$. Each vertex is pushed and popped from the stack exactly once: $O(V)$. The low-link update is $O(1)$ per edge. SCC identification (popping until root) processes each vertex once across all SCCs. Total: $O(V + E)$. $\square$
