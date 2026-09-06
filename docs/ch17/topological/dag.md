# 유향 비순환 그래프 성질

실제 세상의 여러 과정에는 자연스러운 차례가 있다. 대학 강의에는 선행 조건이 있고, 소프트웨어 빌드는 라이브러리에 달려 있으며, 일 일정은 마감을 지켜야 한다. 이 모든 장면은 순환이 없는 방향 그래프라는 공통된 짜임을 나눠 가진다. 방향 그래프가 언제 순환이 없는지 아는 것이 위상 정렬의 바탕이다. 위상 차례가 있는 것은 그 그래프가 유향 비순환 그래프일 때 그리고 오직 그때뿐이기 때문이다.

## 정의

A **directed acyclic graph (DAG)** is a directed graph $G = (V, E)$ that contains no directed cycle. Equivalently, there is no vertex $v \in V$ such that a directed path from $v$ leads back to $v$.

More precisely, $G$ is a DAG if and only if there is no sequence of vertices $v_0, v_1, \ldots, v_k$ with $k \geq 1$ such that $(v_i, v_{i+1}) \in E$ for all $0 \leq i < k$ and $v_k = v_0$.

## 핵심 정리

유향 비순환 그래프와 위상 정렬을 잇는 고갱이 결과는 다음과 같다:

!!! tip "유향 비순환 그래프와 위상 차례의 같음"
    방향 그래프 $G$에 위상 차례가 있는 것은 $G$이 유향 비순환 그래프일 때 그리고 오직 그때뿐이다.

**Proof sketch (forward direction).** Suppose $G$ has a topological ordering $v_1, v_2, \ldots, v_n$ where every edge $(v_i, v_j)$ satisfies $i < j$. If $G$ contained a directed cycle $v_{a_1} \to v_{a_2} \to \cdots \to v_{a_k} \to v_{a_1}$, then we would need $a_1 < a_2 < \cdots < a_k < a_1$, which is a contradiction. Therefore $G$ must be acyclic. $\square$

**Proof sketch (reverse direction).** Suppose $G$ is a DAG. Every DAG has at least one vertex with in-degree zero (otherwise, following predecessors indefinitely in a finite graph would produce a cycle). Remove such a vertex, add it to the ordering, and repeat on the remaining graph (which is still a DAG). This process produces a valid topological ordering of all vertices. $\square$

## 유향 비순환 그래프의 성질

유향 비순환 그래프에는 알고리즘이 써먹는 중요한 짜임 성질이 여럿 있다.

**근원 꼭짓점과 바닥 꼭짓점.** 비어 있지 않은 유향 비순환 그래프에는 근원(들어오는 차수가 0인 꼭짓점)이 적어도 하나, 바닥(나가는 차수가 0인 꼭짓점)이 적어도 하나 있다. 근원이 없다면 앞선 것을 끝없이 좇아 올라가다 끝내 어떤 꼭짓점에 다시 이르러 순환을 이루게 되어, 순환이 없다는 가정과 어긋난다.

**최장 경로.** 유향 비순환 그래프의 최장 경로는 위상 정렬 뒤에 동적 계획을 써서 $O(V + E)$ 시간에 셈할 수 있다. 최장 경로 찾기가 NP-어려움인 일반 방향 그래프와 견주어 보라.

**Number of topological orderings.** A DAG may have many valid topological orderings. The number of distinct orderings depends on the graph structure. A path graph $v_1 \to v_2 \to \cdots \to v_n$ has exactly one topological order, while a graph with no edges on $n$ vertices has $n!$ orderings.

## 고리 알아내기

방향 그래프에 위상 차례가 있는 것은 그것이 유향 비순환 그래프일 때뿐이므로, 어떤 위상 정렬 알고리즘이든 순환 찾개 노릇도 한다. 널리 쓰이는 방식이 둘 있다:

1. **깊이 우선 돌아보기 바탕 찾기.** 깊이 우선 돌아보기 도중 뒤로 가는 변(지금 되돌이 쌓기에 있는 꼭짓점으로 가는 변)을 만나면 그 그래프에 순환이 있다. 자세한 것은 [깊이 우선 돌아보기 바탕 위상 정렬](dfs.md) 쪽을 보라.

2. **칸 알고리즘.** 칸 알고리즘이 모든 꼭짓점을 다루지 못한 채 끝나면 남은 꼭짓점이 하나 이상의 순환을 이룬다. 자세한 것은 [칸 알고리즘](kahn.md) 쪽을 보라.

```python
"""
DFS으로 방향 그래프에서 고리 알아내기.

깊이 우선 돌아보기의 뒤로 가는 변이 방향 순환을 어떻게 드러내는지 보여 주며,
그 그래프가 유향 비순환 그래프가 아님을 확인해 준다.
"""


# === 깊이 우선 돌아보기로 순환 찾기 ===
def has_cycle(graph, n):
    """
    방향 그래프에 순환이 있는지 정한다.

    매개변수
    ----------
    graph : dict[int, list[int]]
        방향 그래프를 이웃 목록으로 나타낸 것.
    n : int
        꼭짓점의 개수(이름은 0부터 n-1).

    반환값
    -------
    bool
        그래프에 방향 순환이 있으면 True, 아니면 False.
    """
    WHITE, GRAY, BLACK = 0, 1, 2
    color = [WHITE] * n

    def dfs(u):
        color[u] = GRAY
        for v in graph.get(u, []):
            if color[v] == GRAY:
                return True  # 뒤로 가는 변을 찾음 => 순환
            if color[v] == WHITE and dfs(v):
                return True
        color[u] = BLACK
        return False

    return any(color[u] == WHITE and dfs(u) for u in range(n))


# === 메인 ===
if __name__ == "__main__":
    # 유향 비순환 그래프: 0 -> 1 -> 3, 0 -> 2 -> 3 -> 4
    dag = {0: [1, 2], 1: [3], 2: [3], 3: [4], 4: []}
    print(f"DAG has cycle: {has_cycle(dag, 5)}")

    # 순환이 있는 그래프: 0 -> 1 -> 2 -> 0
    cyclic = {0: [1], 1: [2], 2: [0]}
    print(f"Cyclic graph has cycle: {has_cycle(cyclic, 3)}")
```

**출력:**
```
DAG has cycle: False
Cyclic graph has cycle: True
```

위의 세 가지 색칠 방식은 꼭짓점마다 흰색(들르지 않음), 회색(지금 되돌이 쌓기에 있음), 검은색(다 다룸)으로 표시한다. 회색에서 회색으로 가는 변이 뒤로 가는 변이며, 이는 순환이 있음을 보여 준다.

## 위상 정렬과의 관계

유향 비순환 그래프라는 성질이 모든 위상 정렬 알고리즘으로 들어가는 문이다. 그래프가 유향 비순환 그래프임을 확인하고 나면, 모든 방향 변 $(u, v)$에 대해 꼭짓점 $u$이 $v$ 앞에 오도록 꼭짓점을 한 줄로 늘어놓을 수 있다. 주된 방식은 둘이다:

- [**칸 알고리즘**](kahn.md) — 줄서기를 써서 근원 꼭짓점을 되풀이해 없앤다
- [**깊이 우선 돌아보기 바탕 정렬**](dfs.md) — 깊이 우선 돌아보기의 마침 시각을 쓴다

둘 다 $O(V + E)$ 시간에 돌며 유향 비순환 그래프인지도 한꺼번에 확인할 수 있다.

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. 20장.

## 연습문제

**연습문제 1.**
방향 그래프에 위상 차례가 있는 것은 그것이 유향 비순환 그래프일 때 그리고 오직 그때뿐임을 증명하여라.

??? success "연습문제 1 풀이"
    $(\Rightarrow)$ If $G$ has a topological ordering $v_1, v_2, \ldots, v_n$, then every edge $(v_i, v_j)$ has $i < j$. A directed cycle $v_{i_1} \to v_{i_2} \to \cdots \to v_{i_k} \to v_{i_1}$ would require $i_1 < i_2 < \cdots < i_k < i_1$, a contradiction. So $G$ is acyclic.

    $(\Leftarrow)$ If $G$ is a DAG, it has at least one vertex with in-degree 0 (otherwise, following incoming edges would create a cycle). Remove this vertex and repeat. The removal order is a valid topological sort. $\square$

---

**연습문제 2.**
유향 비순환 그래프에 위상 차례가 여럿 있을 수 있는가? 차례가 하나뿐인 때는 언제인가?

??? success "연습문제 2 풀이"
    Yes. A DAG has a unique topological ordering if and only if there is a Hamiltonian path in the DAG (a directed path visiting every vertex). Otherwise, at some step, multiple vertices have in-degree 0, and choosing differently gives different orderings. Example: $A \to C, B \to C$ has orderings $[A, B, C]$ and $[B, A, C]$. The chain $A \to B \to C$ has only $[A, B, C]$. $\square$

---

**연습문제 3.**
유향 비순환 그래프마다 들어오는 차수가 0인 꼭짓점과 나가는 차수가 0인 꼭짓점이 적어도 하나씩 있음을 증명하여라.

??? success "연습문제 3 풀이"
    **In-degree 0**: Suppose every vertex has in-degree $\geq 1$. Start at any vertex and follow incoming edges backward: $v_0 \leftarrow v_1 \leftarrow v_2 \leftarrow \cdots$. Since there are finitely many vertices, some vertex must repeat, creating a directed cycle. This contradicts the DAG property. **Out-degree 0**: Analogously, if every vertex has out-degree $\geq 1$, following outgoing edges forward creates a cycle. $\square$

---

**연습문제 4.**
유향 비순환 그래프의 최장 경로란 무엇인가? 모든 일의 일정을 짜는 데 필요한 최소 켜 수와 어떻게 이어지는가?

??? success "연습문제 4 풀이"
    The longest path in a DAG (also called the critical path) has length equal to the number of edges on the longest directed path. It determines the minimum number of sequential steps needed: tasks on the critical path cannot be parallelized with each other. The minimum number of levels (parallel scheduling depth) equals the critical path length plus 1. This is computed in $O(V + E)$ by processing vertices in topological order and tracking maximum distances. $\square$
