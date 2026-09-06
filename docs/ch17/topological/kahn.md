# 칸 알고리즘

[깊이 우선 돌아보기 바탕 방식](dfs.md)이 마침 시각으로 위상 차례를 찾아내는 데 견주어, 칸 알고리즘은 더 곧바른 길을 간다. 곧 들어오는 변이 없는 꼭짓점을 거듭 가려내 내놓고 그 나가는 변을 없앤다. 이 "근원을 벗겨 내는" 전략은 우리가 달림을 두고 자연스레 생각하는 방식을 그대로 옮긴 것이다. 곧 앞선 조건이 없는 일부터 시작해 그것을 끝내고 다음 켜를 연다.

## 직관

강의 선행 조건을 나타내는 유향 비순환 그래프를 보자. 어떤 강의는 선행 조건이 아예 없는데 그것이 근원이다. 학생이 근원 강의를 모두 마치면 새 강의를 들을 수 있게 된다(그 선행 조건이 이제 채워졌다). 칸 알고리즘은 이 켜별 과정을 다듬어 나타낸 것이다.

이 알고리즘은 꼭짓점마다 **들어오는 차수**, 곧 들어오는 변의 수를 지닌다. 들어오는 차수가 0인 꼭짓점은 위상 차례에서 다음에 놓을 수 있는 근원이다. 근원을 "없앨" 때 그 이웃의 들어오는 차수를 하나씩 줄이는데, 그러면서 새 근원이 생길 수 있다.

## 알고리즘

**들임:** $|V| = n$, $|E| = m$인 방향 그래프 $G = (V, E)$.

**내놓음:** $V$의 위상 차례, 또는 $G$에 순환이 있다는 알림.

1. 꼭짓점마다 들어오는 차수를 셈한다.
2. 들어오는 차수가 0인 꼭짓점을 모두 넣어 줄서기 $Q$을 첫자리매김한다.
3. $Q$이 비어 있지 않은 동안:
    - 꼭짓점 $u$을 줄서기에서 빼내 내놓는 목록에 덧붙인다.
    - For each neighbor $v$ of $u$, decrement $\text{in-degree}(v)$ by 1.
    - If $\text{in-degree}(v)$ becomes zero, enqueue $v$.
4. 내놓는 목록에 꼭짓점 $n$개가 모두 들어 있으면 그것을 위상 차례로 돌려준다. 그렇지 않으면 $G$에 순환이 있다.

## 올바름

!!! note "칸 알고리즘이 올바른 위상 차례를 내놓는 까닭"
    **주장:** $G$이 유향 비순환 그래프이면 칸 알고리즘은 올바른 위상 차례를 내놓는다.

    **Proof.** We show that for every edge $(u, v) \in E$, vertex $u$ appears before $v$ in the output. When $u$ is dequeued in step 3, the algorithm decrements $\text{in-degree}(v)$. Before this point, $v$ cannot have in-degree zero (since the edge from $u$ contributes to $v$'s in-degree), so $v$ has not yet been dequeued. Therefore $u$ precedes $v$ in the output. $\square$

!!! note "순환 찾기"
    **주장:** $G$에 순환이 있으면 칸 알고리즘은 $n$개보다 적은 꼭짓점을 다룬다.

    **Proof.** Every vertex in a cycle always has at least one predecessor that is also in the cycle. Since no vertex in the cycle ever reaches in-degree zero, none of them are enqueued. The output list therefore omits at least the vertices in the cycle. $\square$

## 복잡도

꼭짓점마다 정확히 한 번 줄서기에 넣고 빼며, 변마다 정확히 한 번 살핀다(그 근원을 줄서기에서 뺄 때). 따라서:

$$
T(V, E) = O(V + E)
$$

공간 복잡도는 들어오는 차수 배열과 줄서기 때문에 $O(V)$이다.

## 구현

```python
"""
위상 정렬을 위한 칸 알고리즘.

근원을 되풀이해 없애(들어오는 차수가 0인 꼭짓점을 너비 우선으로 훑어)
유향 비순환 그래프의 위상 차례를 내놓으며 순환 찾기도 갖추었다
알아내기.
"""

from collections import deque


# === 칸의 위상 정렬 ===
def kahn_topo_sort(graph, n):
    """
    칸 알고리즘으로 위상 차례를 셈한다.

    매개변수
    ----------
    graph : dict[int, list[int]]
        꼭짓점이 0부터 n-1인 방향 그래프의 이웃 목록.
    n : int
        꼭짓점의 개수.

    반환값
    -------
    list[int]
        위상 차례의 꼭짓점들. 순환이 있으면 빈 목록.
    """
    in_degree = [0] * n
    for u in range(n):
        for v in graph.get(u, []):
            in_degree[v] += 1

    queue = deque(v for v in range(n) if in_degree[v] == 0)
    order = []

    while queue:
        u = queue.popleft()
        order.append(u)
        for v in graph.get(u, []):
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    if len(order) != n:
        return []  # 고리 찾음
    return order


# === 메인 ===
if __name__ == "__main__":
    # 유향 비순환 그래프: 0 -> 1 -> 3, 0 -> 2 -> 3 -> 4
    dag = {0: [1, 2], 1: [3], 2: [3], 3: [4], 4: []}
    result = kahn_topo_sort(dag, 5)
    print(f"Topological order: {result}")

    # 올바른지 확인
    pos = {v: i for i, v in enumerate(result)}
    valid = all(pos[u] < pos[v] for u in dag for v in dag[u])
    print(f"Valid topological order: {valid}")

    # 순환이 있는 그래프: 0 -> 1 -> 2 -> 0
    cyclic = {0: [1], 1: [2], 2: [0]}
    result_cyclic = kahn_topo_sort(cyclic, 3)
    print(f"Cyclic graph result: {result_cyclic}")
```

**출력:**
```
Topological order: [0, 1, 2, 3, 4]
Valid topological order: True
Cyclic graph result: []
```

## 사전 차례로 가장 작은 위상 차례

쓸모 있는 변종은 줄서기를 최소 힙(우선순위 줄서기)으로 바꾼다. 그러면 다음 근원 꼭짓점으로 고를 수 있는 것들 가운데 늘 이름표가 가장 작은 것을 고르게 된다. 그 결과가 **사전 차례로 가장 작은** 위상 차례이다.

```python
"""
사전 차례로 가장 작은 위상 차례를 내놓는 칸 알고리즘 변종
보통의 줄서기 대신 최소 힙을 써서 만든다.
"""

import heapq


# === 사전 차례 칸 정렬 ===
def kahn_lex_smallest(graph, n):
    """
    사전 차례로 가장 작은 위상 차례를 셈한다.

    매개변수
    ----------
    graph : dict[int, list[int]]
        꼭짓점이 0부터 n-1인 유향 비순환 그래프의 이웃 목록.
    n : int
        꼭짓점의 개수.

    반환값
    -------
    list[int]
        사전 차례로 가장 작은 위상 차례.
    """
    in_degree = [0] * n
    for u in range(n):
        for v in graph.get(u, []):
            in_degree[v] += 1

    heap = [v for v in range(n) if in_degree[v] == 0]
    heapq.heapify(heap)
    order = []

    while heap:
        u = heapq.heappop(heap)
        order.append(u)
        for v in graph.get(u, []):
            in_degree[v] -= 1
            if in_degree[v] == 0:
                heapq.heappush(heap, v)

    return order


# === 메인 ===
if __name__ == "__main__":
    dag = {0: [2, 1], 1: [3], 2: [3], 3: [4], 4: []}
    print(f"Lex smallest order: {kahn_lex_smallest(dag, 5)}")
```

**출력:**
```
Lex smallest order: [0, 1, 2, 3, 4]
```

The heap variant runs in $O((V + E) \log V)$ due to the heap operations.

## 깊이 우선 돌아보기 바탕 정렬과의 견줌

| 성질 | 칸 알고리즘 | 깊이 우선 돌아보기 바탕 |
|---|---|---|
| 전략 | 근원을 되풀이해 없애기 | 거꿀 마침 시각 차례 |
| 자료 짜임새 | 줄서기(또는 힙) | 되돌이 쌓기 |
| 순환 찾기 | 덜 내놓음(꼭짓점 $n$개 미만) | 돌아보기 도중의 뒤로 가는 변 |
| 사전 차례 | 최소 힙으로 쉽다 | 품이 더 든다 |
| 나란함 | 같은 "켜"의 근원끼리 서로 안 얽힌다 | 본디 차례차례이다 |

올바른 고름 가운데 특정한 차례(사전 차례로 가장 작은 것 같은)가 필요한 쓰임새에는 우선순위 줄서기를 쓴 칸 알고리즘이 표준 방식이다. 일반 위상 정렬에는 두 방법 모두 $O(V + E)$으로 똑같이 효율적이다.

## 참고 문헌

- Kahn, A. B. (1962). Topological sorting of large networks. *Communications of the ACM*, 5(11), 558-562.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. 20장.

## 연습문제

**연습문제 1.**
칸 알고리즘을 한 단계씩 설명하여라.

??? success "연습문제 1 풀이"
    (1) Compute in-degree for all vertices. (2) Add all vertices with in-degree 0 to a queue. (3) While the queue is non-empty: dequeue vertex $v$, add $v$ to the output. For each neighbor $u$ of $v$, decrement $\text{in\_degree}[u]$. If $\text{in\_degree}[u]$ becomes 0, enqueue $u$. (4) If all vertices are in the output, it is a valid topological sort. If some vertices remain (in-degree never reached 0), a cycle exists. Time: $O(V + E)$. $\square$

---

**연습문제 2.**
칸 알고리즘이 순환을 찾아냄을 증명하여라.

??? success "연습문제 2 풀이"
    If the graph has a cycle $v_1 \to v_2 \to \cdots \to v_k \to v_1$, then every vertex in the cycle always has in-degree $\geq 1$ (from the predecessor in the cycle). No vertex in the cycle ever has in-degree 0, so none is ever enqueued. After the algorithm terminates, these vertices remain unprocessed. The count of processed vertices is less than $V$, signaling a cycle. $\square$

---

**연습문제 3.**
순환에 든 꼭짓점을 모두 찾도록 칸 알고리즘을 어떻게 고칠 수 있는가?

??? success "연습문제 3 풀이"
    After running Kahn's algorithm, any vertex not in the output list is involved in a cycle (its in-degree never reached 0). Collect all unprocessed vertices — they form one or more cycles in the graph. To find the specific cycles, run DFS on the subgraph induced by these vertices. This gives the cycle structure in $O(V + E)$ total time. $\square$

---

**연습문제 4.**
칸 알고리즘에서 보통 줄서기 대신 우선순위 줄서기(최소 힙)를 쓰면 어떤 차례가 나오는가?

??? success "연습문제 4 풀이"
    Using a min-heap produces the **lexicographically smallest** topological ordering. At each step, among all vertices with in-degree 0, the smallest-numbered vertex is chosen. This is useful when a canonical or deterministic ordering is needed (e.g., for testing or comparison). The time complexity increases to $O((V + E) \log V)$ due to heap operations (versus $O(V + E)$ with a regular queue). $\square$
