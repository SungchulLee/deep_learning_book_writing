# 해밀턴 경로와 순환

오일러 회로가 **변**마다 정확히 한 번씩 들르는 데 견주어, 해밀턴 순환은 **꼭짓점**마다 정확히 한 번씩 들른다. 문제 진술의 이 사소해 보이는 바뀜이 엄청난 결과를 낳는다. 곧 그래프에 해밀턴 순환이 있는지 정하는 것은 NP-완전이며, 이는 효율적인 일반 알고리즘이 알려져 있지 않다는 뜻이다. 그럼에도 있음을 보장하는 충분조건이 여럿 있고, 실전에서 쓰이는 되돌아가기 알고리즘이 알맞은 크기의 보기를 푼다.

## 정의

**해밀턴 경로.** 꼭짓점마다 정확히 한 번씩 들르는 $G = (V, E)$의 단순 경로.

**Hamiltonian cycle.** A simple cycle that visits every vertex exactly once and returns to the starting vertex. Equivalently, a Hamiltonian path from $v$ to $w$ where $(w, v) \in E$.

**해밀턴 그래프.** 해밀턴 순환이 있는 그래프.

## 오일러와의 견줌

| 성질 | 오일러 | 해밀턴 |
|----------|:--------:|:-----------:|
| 무엇마다 들르나 | 변 | 꼭짓점 |
| 있는지 살피기 | 차수 홀짝으로 $O(V + E)$ | NP-완전 |
| 효율적인 특징지음 | 있다(차수 정리) | 알려진 특징지음 없음 |

## 충분조건

해밀턴성에 대한 간단한 필요충분조건은 알려져 있지 않지만, 고전적인 정리 몇이 충분조건을 준다.

!!! note "디랙 정리(1952)"
    If $G$ is a simple graph on $n \ge 3$ vertices and every vertex satisfies $\deg(v) \ge n/2$, then $G$ is Hamiltonian.

!!! note "오레 정리(1960)"
    If $G$ is a simple graph on $n \ge 3$ vertices and for every pair of non-adjacent vertices $u, v$ we have $\deg(u) + \deg(v) \ge n$, then $G$ is Hamiltonian.

Ore's theorem generalizes Dirac's theorem: if every vertex has degree at least $n/2$, then any pair of vertices satisfies $\deg(u) + \deg(v) \ge n$.

## 복잡도

해밀턴 순환 문제는 카프가 처음 내놓은 21가지 NP-완전 문제(1972) 가운데 하나이다. 다음으로 제한해도 여전히 NP-완전이다:

- 최대 차수가 3인 평면 그래프.
- 두 쪽 그래프.
- 격자 그래프.

The best known exact algorithms run in $O^*(2^n)$ time using dynamic programming over subsets (the Held-Karp algorithm), improving on the $O(n!)$ naive approach.

## 되돌아가기 알고리즘

되돌아가기 방식은 꼭짓점을 하나씩 붙여 경로를 세우며, 올바른 해밀턴 경로나 순환으로 이어질 수 없는 가지를 쳐 낸다. 가장 나쁜 경우 지수 시간이지만 작거나 알맞은 크기의 그래프에는 쓸 만하다.

```python
"""
되돌아가기로 해밀턴 순환 찾기.

꼭짓점을 하나씩 붙여 경로를 세우고 올바르지 않은 뻗음을 쳐 내며
해밀턴 순환이 있는지 살핀다.
"""

# === 되돌아가기 해밀턴 순환 ===

def hamiltonian_cycle(n: int, edges: list[tuple[int, int]]) -> list[int] | None:
    """있으면 해밀턴 순환을 찾는다.

    인수:
        n: 꼭짓점의 개수(0부터 셈).
        edges: 방향 없는 변의 목록.

    반환값:
        순환을 이루는 꼭짓점의 목록. 순환이 없으면 None.
    """
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    path = [0]
    visited = {0}

    def backtrack() -> bool:
        if len(path) == n:
            # 시작점으로 돌아올 수 있는지 살피기
            return 0 in adj[path[-1]]

        last = path[-1]
        for neighbor in sorted(adj[last]):
            if neighbor not in visited:
                visited.add(neighbor)
                path.append(neighbor)
                if backtrack():
                    return True
                path.pop()
                visited.remove(neighbor)
        return False

    if backtrack():
        return path + [path[0]]
    return None


# === 시연 ===

if __name__ == "__main__":
    # 완전 그래프 K4
    k4_edges = [(i, j) for i in range(4) for j in range(i+1, 4)]
    result = hamiltonian_cycle(4, k4_edges)
    print(f"K4 Hamiltonian cycle: {result}")

    # 경로 그래프: 0-1-2-3(해밀턴 순환 없음)
    path_edges = [(0,1),(1,2),(2,3)]
    result = hamiltonian_cycle(4, path_edges)
    print(f"Path graph cycle: {result}")

    # 피터슨 그래프(해밀턴 경로는 있으나 순환은 없다)
    petersen = [
        (0,1),(1,2),(2,3),(3,4),(4,0),  # 바깥 순환
        (0,5),(1,6),(2,7),(3,8),(4,9),  # 바큇살
        (5,7),(7,9),(9,6),(6,8),(8,5),  # 안쪽 오각별
    ]
    result = hamiltonian_cycle(10, petersen)
    print(f"Petersen graph cycle: {result}")
```

**출력:**

```
K4 Hamiltonian cycle: [0, 1, 2, 3, 0]
Path graph cycle: None
Petersen graph cycle: None
```

The complete graph $K_4$ satisfies Dirac's condition ($\deg(v) = 3 \ge 4/2$) and indeed has a Hamiltonian cycle. The path graph on 4 vertices has no cycle at all. The Petersen graph famously has Hamiltonian paths but no Hamiltonian cycle.

## 동적 계획 방식

The Held-Karp algorithm uses bitmask DP to find a Hamiltonian path in $O(n^2 \cdot 2^n)$ time and $O(n \cdot 2^n)$ space. Define:

$$
\text{dp}[S][v] = \text{True if there is a path visiting exactly the vertices in } S \text{ and ending at } v
$$

점화식은 다음과 같다.

$$
\text{dp}[S][v] = \bigvee_{u \in S \setminus \{v\},\; (u,v) \in E} \text{dp}[S \setminus \{v\}][u]
$$

A Hamiltonian cycle exists if $\text{dp}[\{0, 1, \dots, n{-}1\}][v] = \text{True}$ for some $v$ adjacent to the starting vertex.

## 참고 문헌

- Karp, R. M. (1972). Reducibility among combinatorial problems. In *Complexity of Computer Computations*, pp. 85--103.
- Ore, O. (1960). Note on Hamilton circuits. *The American Mathematical Monthly*, 67(1), 55.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to Algorithms* (4th ed.), 34장: NP-Completeness.

## 연습문제

**연습문제 1.**
오일러 회로를 살피는 것은 다항 시간인데 해밀턴 순환이 있는지 정하는 것은 왜 NP-완전인지 설명하여라.

??? success "연습문제 1 풀이"
    Eulerian circuits have a simple characterization (all even degrees + connectivity) that can be checked in $O(V + E)$. No such simple characterization exists for Hamiltonian cycles. The Hamiltonian cycle problem is NP-complete (proven by reduction from 3-SAT or Vertex Cover). This means no polynomial-time algorithm is known, and all known approaches require exponential time in the worst case (e.g., $O(n! \cdot n)$ brute force, $O(2^n \cdot n^2)$ dynamic programming). $\square$

---

**연습문제 2.**
디랙 정리를 말하고 그것으로 $K_6$에 해밀턴 순환이 있는지 정하여라.

??? success "연습문제 2 풀이"
    **Dirac's theorem**: If $G$ is a simple graph on $n \geq 3$ vertices where every vertex has degree $\geq n/2$, then $G$ has a Hamiltonian cycle. In $K_6$, every vertex has degree 5. Since $5 \geq 6/2 = 3$, Dirac's condition is satisfied, and $K_6$ has a Hamiltonian cycle. (In fact, $K_n$ has a Hamiltonian cycle for all $n \geq 3$.) $\square$

---

**연습문제 3.**
Describe the $O(2^n \cdot n^2)$ dynamic programming algorithm for the Hamiltonian cycle problem.

??? success "연습문제 3 풀이"
    Use bitmask DP. Let $dp[S][v]$ = True if there is a path visiting exactly the vertices in set $S$ and ending at vertex $v$. Base: $dp[\{0\}][0] = \text{True}$ (start at vertex 0). Transition: $dp[S \cup \{v\}][v] = \text{True}$ if $dp[S][u] = \text{True}$ and $(u, v)$ is an edge, for some $u \in S$. Answer: $dp[\{0,\ldots,n-1\}][v] \land (v, 0) \in E$ for some $v$. There are $2^n$ subsets, $n$ choices for $v$, and $n$ choices for $u$, giving $O(2^n \cdot n^2)$. $\square$

---

**연습문제 4.**
떠돌이 장사꾼 문제(TSP)는 무게가 가장 작은 해밀턴 순환을 묻는다. 이것은 해밀턴 순환 문제와 어떻게 이어지는가?

??? success "연습문제 4 풀이"
    TSP is an optimization version: find a Hamiltonian cycle of minimum total weight. The decision version of TSP ("is there a Hamiltonian cycle of weight $\leq k$?") is NP-complete. If we could solve TSP in polynomial time, we could solve the Hamiltonian cycle problem by checking if the minimum TSP tour has finite weight in a graph with edge weight 1 for existing edges and $\infty$ for non-edges. TSP is at least as hard as the Hamiltonian cycle problem. $\square$
