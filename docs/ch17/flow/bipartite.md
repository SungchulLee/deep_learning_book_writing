# 두 쪽 짝짓기

실제 세상의 여러 배정 문제가 두 쪽 그래프의 짝짓기로 줄어든다. 곧 일꾼을 일에, 학생을 기획에, 서버를 요청에 맞추는 일이다. 두 쪽 그래프가 주어질 때 **짝짓기**는 끝점을 함께 갖지 않는 변의 부분 모음을 고른다. 근본이 되는 알고리즘 물음은 크기가 가장 큰 짝짓기를 찾는 것이며, 이는 최대 흐름 문제로 줄여 효율적으로 풀 수 있다.

## 정의

**Bipartite graph.** A graph $G = (L \cup R, E)$ whose vertices partition into two disjoint sets $L$ and $R$ such that every edge connects a vertex in $L$ to a vertex in $R$.

**Matching.** A subset $M \subseteq E$ such that no two edges in $M$ share an endpoint. A vertex is *matched* if it is an endpoint of some edge in $M$; otherwise it is *free* (or *unmatched*).

**최대 짝짓기.** 변의 개수가 가능한 한 가장 많은 짝짓기.

**완전 짝짓기.** 꼭짓점을 모두 덮는 짝짓기($|L| = |R|$이고 변이 넉넉할 때만 가능하다).

## 최대 흐름으로 줄이기

최대 두 쪽 짝짓기 문제는 특별히 세운 그물에서의 최대 흐름으로 줄어든다:

1. Add a **source** $s$ with edges $(s, \ell)$ of capacity $1$ for every $\ell \in L$.
2. Add a **sink** $t$ with edges $(r, t)$ of capacity $1$ for every $r \in R$.
3. For each original edge $(\ell, r) \in E$, add an edge of capacity $1$.

A maximum integer flow in this network corresponds to a maximum matching: each unit of flow through an edge $(\ell, r)$ indicates that $\ell$ is matched to $r$. Since all capacities are $1$ and the graph is bipartite, the max flow is always integral.

$$
|M^*| = \max\text{-flow}(s, t)
$$

## 쾨니히 정리

!!! note "쾨니히 정리"
    어떤 두 쪽 그래프에서도 최대 짝짓기의 크기는 최소 꼭짓점 덮개의 크기와 같다.

$$
|M^*| = |\text{min vertex cover}|
$$

이는 두 쪽 그래프에만 있는 놀라운 쌍대 결과이다. 일반 그래프에서는 최소 꼭짓점 덮개가 최대 짝짓기보다 훨씬 클 수 있다.

## 늘림 경로

짝짓기 $M$에 대한 **늘림 경로**는 다음과 같은 경로이다:

- $L$의 짝 없는 꼭짓점에서 시작한다.
- 짝지어지지 않은 변과 짝지어진 변을 번갈아 지난다.
- $R$의 짝 없는 꼭짓점에서 끝난다.

늘림 경로 위의 변마다 짝지어짐/짝지어지지 않음을 뒤집으면 짝짓기 크기가 하나 늘어난다. **베르주 보조정리**는 짝짓기가 최대인 것은 늘림 경로가 없을 때 그리고 오직 그때뿐이라고 말한다.

## 구현

```python
"""
늘림 경로로 하는 최대 두 쪽 짝짓기(홉크로프트-카프 방식 깊이 우선 돌아보기).

늘림 경로를 거듭 찾아
두 쪽 그래프에서 최대 짝짓기를 찾는다.
"""

# === 최대 두 쪽 짝짓기 ===

def max_bipartite_matching(
    n_left: int, n_right: int, edges: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """두 쪽 그래프에서 최대 짝짓기를 찾는다.

    인수:
        n_left: 왼쪽 꼭짓점의 개수(0부터 셈).
        n_right: 오른쪽 꼭짓점의 개수(0부터 셈).
        edges: 변 (l, r)의 목록. l은 [0, n_left), r은 [0, n_right).

    반환값:
        짝지어진 짝 (l, r)의 목록.
    """
    adj = [[] for _ in range(n_left)]
    for l, r in edges:
        adj[l].append(r)

    match_right = [-1] * n_right  # match_right[r] = 짝지어진 왼쪽 꼭짓점

    def dfs(u: int, visited: set) -> bool:
        """왼쪽 꼭짓점 u에서 늘림 경로를 찾아본다."""
        for v in adj[u]:
            if v not in visited:
                visited.add(v)
                if match_right[v] == -1 or dfs(match_right[v], visited):
                    match_right[v] = u
                    return True
        return False

    matching_size = 0
    for u in range(n_left):
        visited = set()
        if dfs(u, visited):
            matching_size += 1

    result = []
    for r in range(n_right):
        if match_right[r] != -1:
            result.append((match_right[r], r))
    return result


# === 시연 ===

if __name__ == "__main__":
    # 일꾼 L={0,1,2}, 일 R={0,1,2}
    # 일꾼 0은 일 0,1을 할 수 있다
    # 일꾼 1은 일 0,2를 할 수 있다
    # 일꾼 2는 일 1을 할 수 있다
    edges = [(0,0),(0,1),(1,0),(1,2),(2,1)]
    matching = max_bipartite_matching(3, 3, edges)
    print(f"Maximum matching size: {len(matching)}")
    for l, r in matching:
        print(f"  Worker {l} -> Job {r}")
```

**출력:**

```
Maximum matching size: 3
  Worker 1 -> Job 0
  Worker 2 -> Job 1
  Worker 0 -> Job 2
```

일꾼 셋이 모두 서로 다른 일에 짝지어져 완전 짝짓기를 이룬다. 일꾼 $0$은 일 $0$이나 $1$을 할 수 있지만, 알고리즘은 전체 짝짓기를 크게 하려고 (일꾼 $1$을 거쳐) 그를 일 $2$에 배정한다.

## 복잡도

| 알고리즘 | 시간 |
|-----------|:----:|
| Augmenting path DFS (above) | $O(V \cdot E)$ |
| Hopcroft-Karp | $O(E \sqrt{V})$ |
| Max flow reduction (Ford-Fulkerson) | $O(V \cdot E)$ |

단순한 깊이 우선 돌아보기 방식은 늘림 경로 찾기를 $O(V)$번 하며 한 번에 $O(E)$ 시간이 든다. 홉크로프트-카프 알고리즘은 너비 우선 돌아보기 단계를 써서 늘림 경로 여럿을 한꺼번에 찾아 이를 낫게 한다.

## 홀 정리

!!! note "홀의 혼인 정리"
    A bipartite graph $G = (L \cup R, E)$ has a matching that covers every vertex in $L$ if and only if for every subset $S \subseteq L$:

    $$
    |N(S)| \ge |S|
    $$

    여기서 $N(S)$은 $R$에 있는 $S$의 이웃 모음을 나타낸다.

This theorem provides a necessary and sufficient condition for the existence of a perfect matching on the $L$ side, though checking it directly requires examining all $2^{|L|}$ subsets.

## 참고 문헌

- Hopcroft, J. E., & Karp, R. M. (1973). An $n^{5/2}$ algorithm for maximum matchings in bipartite graphs. *SIAM Journal on Computing*, 2(4), 225--231.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to Algorithms* (4th ed.), 26장: Maximum Flow.

## 연습문제

**연습문제 1.**
두 쪽 짝짓기를 최대 흐름 문제로 줄이는 방법을 보여라. 근원, 바닥, 담이는 무엇인가?

??? success "연습문제 1 풀이"
    Given bipartite graph $(L, R, E)$: add source $s$ with edges of capacity 1 to all vertices in $L$. Add sink $t$ with edges of capacity 1 from all vertices in $R$. Set capacity 1 for all original edges $(u, v)$ where $u \in L, v \in R$. The max flow from $s$ to $t$ equals the maximum matching size. Each unit of flow through an $L$-to-$R$ edge represents a matched pair. The capacity constraints ensure each vertex is matched at most once. $\square$

---

**연습문제 2.**
두 쪽 그래프에 대한 홀 정리(혼인 정리)를 말하여라.

??? success "연습문제 2 풀이"
    **Hall's theorem**: A bipartite graph $(L, R, E)$ has a matching that saturates all vertices in $L$ if and only if for every subset $S \subseteq L$, $|N(S)| \geq |S|$, where $N(S)$ is the set of neighbors of $S$ in $R$. In other words, every subset of $L$ has at least as many neighbors as its own size. This is a necessary and sufficient condition for a perfect matching on the $L$ side. $\square$

---

**연습문제 3.**
홉크로프트-카프로 최대 두 쪽 짝짓기를 찾을 때 시간 복잡도는 무엇인가? 흐름 바탕 방식과 견주면 어떠한가?

??? success "연습문제 3 풀이"
    Hopcroft-Karp runs in $O(E\sqrt{V})$ time. The flow-based approach using Edmonds-Karp gives $O(VE)$ for bipartite matching (since max flow $\leq V/2$, and each augmenting path takes $O(E)$). Hopcroft-Karp is faster because it finds multiple augmenting paths simultaneously using BFS phases, with $O(\sqrt{V})$ phases each processing $O(E)$ edges. For dense bipartite graphs, both are $O(V^{2.5})$, but Hopcroft-Karp has better constants. $\square$

---

**연습문제 4.**
두 쪽 짝짓기의 실제 쓰임새를 하나 들고 그것을 두 쪽 그래프로 나타내어라.

??? success "연습문제 4 풀이"
    **Job assignment**: $n$ workers and $m$ jobs, where worker $i$ can perform job $j$ if they have the required skills. Model as bipartite graph: $L$ = workers, $R$ = jobs, edge $(i, j)$ if worker $i$ can do job $j$. Maximum matching assigns as many workers to jobs as possible, each worker to one job and each job to one worker. This is used in scheduling, resource allocation, and organ donation matching. $\square$
