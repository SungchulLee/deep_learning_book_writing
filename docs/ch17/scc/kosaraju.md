# 코사라주 알고리즘

[강하게 이어진 조각](definition.md)을 효율적으로 찾으려면 재치 있는 눈썰미가 필요하다. 곧 "바닥" 조각을 먼저 들르는 차례로 꼭짓점을 다룰 수 있다면, 뒤집은 그래프에서 깊이 우선 돌아보기를 한 번만 해도 조각마다 따로 떼어 낼 수 있다. 코사라주 알고리즘은 깊이 우선 돌아보기 두 번으로 이를 이룬다. 첫 번째로 다루는 차례를 정하고 두 번째로 조각을 뽑아낸다. 그 결과는 짜기도 쉽고 옳음을 보이기도 쉬운 우아한 $O(V + E)$ 알고리즘이다.

## 알고리즘 훑어보기

코사라주 알고리즘은 세 단계를 밟는다:

1. **$G$ 위의 첫 깊이 우선 돌아보기.** 본디 그래프에서 깊이 우선 돌아보기를 하고 마침 시각이 줄어드는 차례로 꼭짓점을 적어 둔다(곧 꼭짓점의 돌아보기 부름이 끝날 때 그 꼭짓점을 쌓기에 올린다).

2. **Transpose the graph.** Construct the transpose graph $G^T = (V, E^T)$ where every edge $(u, v) \in E$ is reversed to $(v, u) \in E^T$.

3. **Second DFS pass on $G^T$.** Process vertices in the order from step 1 (popping from the stack). Each DFS tree in this pass forms one strongly connected component.

## 왜 통하는가

핵심 눈썰미는 마침 시각과 조각 짜임 사이의 관계이다.

!!! note "강한 이음 조각의 마침 시각 성질"
    If $C_1$ and $C_2$ are two different SCCs and there is an edge from $C_1$ to $C_2$ in the [condensation graph](condensation.md), then the vertex with the latest finish time in $C_1 \cup C_2$ belongs to $C_1$.

**Proof sketch.** If DFS first enters $C_1$, it explores all of $C_1$ and then reaches $C_2$ via the inter-component edge. All vertices in $C_2$ finish before the DFS returns to $C_1$, so $C_1$'s vertices finish later. If DFS first enters $C_2$, it cannot reach $C_1$ (no edge from $C_2$ to $C_1$, since the condensation is a DAG). After finishing $C_2$, DFS eventually starts on a vertex in $C_1$, which finishes later. $\square$

**Consequence.** In the second pass on $G^T$, the vertex with the latest finish time starts DFS in a "source" SCC of the condensation DAG. In $G^T$, inter-component edges are reversed, so this SCC has no outgoing edges in $G^T$. The DFS therefore stays within this SCC, correctly identifying it. Once processed, we move to the next unvisited vertex with the highest finish time, which is in another source SCC of the remaining condensation -- and the process repeats.

## 복잡도

Both DFS passes visit each vertex and edge exactly once. Constructing $G^T$ takes $O(V + E)$. Therefore:

$$
T(V, E) = O(V + E)
$$

Space complexity is $O(V + E)$ for storing $G^T$ and the finish-time stack.

## 구현

```python
"""
강하게 이어진 조각을 찾는 코사라주 알고리즘.

깊이 우선 돌아보기를 두 번 한다. 하나는 마침 차례를 정하려 본디 그래프에서,
다른 하나는 강한 이음 조각을 뽑으려 뒤집은 그래프에서 한다.
"""


# === 코사라주 알고리즘 ===
def kosaraju_scc(graph, n):
    """
    코사라주 알고리즘으로 강하게 이어진 조각을 모두 찾는다.

    매개변수
    ----------
    graph : dict[int, list[int]]
        꼭짓점이 0부터 n-1인 방향 그래프의 이웃 목록.
    n : int
        꼭짓점의 개수.

    반환값
    -------
    list[list[int]]
        강한 이음 조각의 목록. 저마다 꼭짓점 이름의 목록으로 나타낸다.
    """
    # 1차: 본디 그래프에서 깊이 우선 돌아보기, 마침 차례 적기
    visited = [False] * n
    finish_stack = []

    def dfs1(u):
        visited[u] = True
        for v in graph.get(u, []):
            if not visited[v]:
                dfs1(v)
        finish_stack.append(u)

    for u in range(n):
        if not visited[u]:
            dfs1(u)

    # 뒤집은 그래프 세우기
    transpose = {i: [] for i in range(n)}
    for u in range(n):
        for v in graph.get(u, []):
            transpose[v].append(u)

    # 2차: 거꿀 마침 차례로 뒤집은 그래프 돌아보기
    visited = [False] * n
    sccs = []

    def dfs2(u, component):
        visited[u] = True
        component.append(u)
        for v in transpose.get(u, []):
            if not visited[v]:
                dfs2(v, component)

    while finish_stack:
        u = finish_stack.pop()
        if not visited[u]:
            component = []
            dfs2(u, component)
            sccs.append(component)

    return sccs


# === 메인 ===
if __name__ == "__main__":
    graph = {
        0: [1], 1: [2, 3], 2: [0], 3: [4],
        4: [5], 5: [3], 6: [5, 7], 7: [],
    }
    sccs = kosaraju_scc(graph, 8)
    print("Strongly connected components:")
    for i, scc in enumerate(sccs):
        print(f"  C{i+1} = {sorted(scc)}")
```

**출력:**
```
Strongly connected components:
  C1 = [0, 1, 2]
  C2 = [3, 4, 5]
  C3 = [6]
  C4 = [7]
```

## 한 걸음씩 따라가기

위의 보기 그래프를 쓰면:

**Pass 1 (DFS on $G$):** Starting from vertex 0, the DFS explores $0 \to 1 \to 2$ (back to 0, already visited), then $1 \to 3 \to 4 \to 5$ (back to 3, already visited). Finish order accumulates as vertices complete their exploration. After processing all vertices starting from 0 and then 6, the finish stack (bottom to top) might be: $[7, 5, 4, 3, 2, 0, 1, 6]$.

**Transpose $G^T$:** Reverse all edges. The edge $0 \to 1$ becomes $1 \to 0$, and so on.

**Pass 2 (DFS on $G^T$):** Pop vertex 6 from the stack. DFS on $G^T$ from 6 finds only 6 (no incoming edges in $G$). Pop vertex 1; DFS explores $\{1, 0, 2\}$ -- these form one SCC. Pop vertex 3; DFS explores $\{3, 5, 4\}$ -- another SCC. Pop vertex 7; it is alone.

## 타잔 알고리즘과의 견줌

| 성질 | 코사라주 | [타잔](tarjan.md) |
|---|---|---|
| DFS passes | Two (on $G$ and $G^T$) | One |
| Extra storage | Transpose graph $G^T$ | Low-link values and stack |
| 짜기 | 이해하기 더 쉽다 | 조금 더 복잡하다 |
| 시간 복잡도 | $O(V + E)$ | $O(V + E)$ |
| Space complexity | $O(V + E)$ for $G^T$ | $O(V)$ extra |

코사라주 알고리즘은 생각이 또렷해서 가르칠 때 즐겨 쓰이고, 타잔 알고리즘은 기억 공간이 빠듯한 실전에서 즐겨 쓰인다.

## 참고 문헌

- Sharir, M. (1981). A strong-connectivity algorithm and its applications in data flow analysis. *Computers & Mathematics with Applications*, 7(1), 67-72.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. 20장.

## 연습문제

**연습문제 1.**
코사라주 알고리즘의 두 번 도는 깊이 우선 돌아보기 방식을 설명하여라.

??? success "연습문제 1 풀이"
    **Pass 1**: Run DFS on the original graph $G$, recording finish times. Push vertices onto a stack in order of completion. **Pass 2**: Transpose the graph ($G^T$ — reverse all edges). Pop vertices from the stack and run DFS on $G^T$. Each DFS from a new root discovers one SCC. The stack order ensures we process "source" SCCs in the condensation first, preventing them from reaching into other SCCs in $G^T$. Total time: $O(V + E)$. $\square$

---

**연습문제 2.**
코사라주 알고리즘에 왜 뒤집은 그래프가 필요한가? 변 뒤집기는 어떤 몫을 하는가?

??? success "연습문제 2 풀이"
    Edge reversal ensures that DFS in pass 2 stays within a single SCC. In $G^T$, the SCCs are the same as in $G$ (mutual reachability is symmetric under transposition). The finish-time ordering from pass 1 processes SCCs in reverse topological order of the condensation. In $G^T$, edges between SCCs are reversed, so DFS from a "source" SCC in the condensation (processed first) cannot cross into other SCCs. This confines each DFS to exactly one SCC. $\square$

---

**연습문제 3.**
코사라주 알고리즘이 강한 이음 조각을 올바로 가려냄을 증명하여라.

??? success "연습문제 3 풀이"
    After pass 1, vertices are ordered by decreasing finish time. In the condensation DAG of $G$, the SCC with the latest finish time is a "source" (no incoming edges from other SCCs). In $G^T$, this SCC becomes a "sink," so DFS from it in $G^T$ stays within the SCC. After discovering this SCC, we remove it and repeat. The next SCC in the stack is now a source in the remaining condensation. By induction, each DFS in pass 2 discovers exactly one SCC. $\square$

---

**연습문제 4.**
강한 이음 조각을 찾는 코사라주 알고리즘과 타잔 알고리즘을 견주어라.

??? success "연습문제 4 풀이"
    Both run in $O(V + E)$. **Kosaraju**: two DFS passes (one on $G$, one on $G^T$), simpler to understand, requires $O(V + E)$ extra space for the transpose graph. **Tarjan**: single DFS pass, uses a stack and low-link values, more space-efficient (no transpose needed), slightly more complex to implement. Tarjan produces SCCs in reverse topological order of the condensation DAG. In practice, both have similar performance; Tarjan is preferred when memory is a concern. $\square$
