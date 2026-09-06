# 2-충족 가능성

충족 가능성 문제(SAT)는 불 식을 참으로 만드는 변수 배정이 있는지 묻는다. 일반적으로 SAT은 NP-완전이지만, 마디마다 글자가 정확히 둘인 특수한 경우인 **2-SAT**은 [강하게 이어진 조각](definition.md)을 거쳐 우아한 다항 시간 풀이를 허락한다. 논리에서 그래프 이론으로 옮기는 이 줄임은 조각 쪼갬의 가장 아름다운 쓰임새 가운데 하나이다.

## 문제 정식화

A **2-SAT** instance consists of $n$ Boolean variables $x_1, x_2, \ldots, x_n$ and $m$ clauses, where each clause is a disjunction of exactly two literals:

$$
(l_{1,1} \lor l_{1,2}) \land (l_{2,1} \lor l_{2,2}) \land \cdots \land (l_{m,1} \lor l_{m,2})
$$

Each literal $l_{i,j}$ is either a variable $x_k$ or its negation $\neg x_k$. The goal is to determine whether there exists an assignment of truth values to the variables that satisfies all clauses simultaneously.

## 함의 그래프

The key insight is that each clause $(a \lor b)$ is logically equivalent to two implications:

$$
(\neg a \Rightarrow b) \quad \text{and} \quad (\neg b \Rightarrow a)
$$

**함의 그래프** $G = (V, E)$을 다음과 같이 세운다:

- **Vertices:** For each variable $x_i$, create two vertices: $x_i$ and $\neg x_i$. So $|V| = 2n$.
- **Edges:** For each clause $(a \lor b)$, add directed edges $\neg a \to b$ and $\neg b \to a$.

!!! tip "왜 함의인가?"
    The clause $(a \lor b)$ says "at least one of $a$, $b$ must be true." If $a$ is false, then $b$ must be true -- hence $\neg a \Rightarrow b$. Symmetrically, if $b$ is false, then $a$ must be true -- hence $\neg b \Rightarrow a$.

## 강한 이음 조각에 바탕한 풀이

The formula is satisfiable if and only if no variable $x_i$ and its negation $\neg x_i$ belong to the same SCC in the implication graph.

!!! note "2-SAT 충족 가능성 정리"
    A 2-SAT formula is satisfiable if and only if for every variable $x_i$, the vertices $x_i$ and $\neg x_i$ are in different SCCs of the implication graph.

**Proof sketch (forward).** If $x_i$ and $\neg x_i$ are in the same SCC, there is a path from $x_i$ to $\neg x_i$ and from $\neg x_i$ to $x_i$ in the implication graph. This means setting $x_i = \text{true}$ forces $x_i = \text{false}$ (and vice versa), making the formula unsatisfiable. $\square$

**Proof sketch (reverse).** If no variable shares an SCC with its negation, we can assign truth values consistently by processing the [condensation DAG](condensation.md) in reverse topological order: for each unassigned variable, set the literal whose SCC appears later in the topological order to true. $\square$

## 배정 뽑아내기

충족 가능함을 확인하고 나면 배정을 다음과 같이 뽑아낸다:

1. ([타잔](tarjan.md)이나 [코사라주](kosaraju.md)를 써서) 함의 그래프의 강한 이음 조각을 셈한다.
2. For each variable $x_i$, compare the topological order of the SCC containing $x_i$ with the SCC containing $\neg x_i$.
3. Set $x_i = \text{true}$ if the SCC of $x_i$ appears later (higher topological order) than the SCC of $\neg x_i$.

직관으로 보면, 오그린 유향 비순환 그래프에서 뒤쪽 조각이 함의 사슬의 앞쪽 조각을 "덮어쓴다".

## 복잡도

함의 그래프를 세우는 데 $O(n + m)$이 든다. 강한 이음 조각을 찾는 데 $O(V + E) = O(n + m)$이 든다. 따라서:

$$
T(n, m) = O(n + m)
$$

## 구현

```python
"""
강하게 이어진 조각을 쓰는 2-SAT 풀개.

함의 그래프를 세우고 타잔 알고리즘으로 강한 이음 조각을 찾아,
어떤 변수와 그 부정이 같은 조각에 드는지 살펴
충족 가능한지 정한다.
"""


# === 타잔의 강한 이음 조각(도우미) ===
def tarjan_scc(graph, n):
    """강한 이음 조각을 찾아 꼭짓점-조각 번호 대응을 돌려준다."""
    disc = [-1] * n
    low = [0] * n
    on_stack = [False] * n
    stack = []
    timer = [0]
    scc_id = [-1] * n
    scc_count = [0]

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

        if low[u] == disc[u]:
            while True:
                w = stack.pop()
                on_stack[w] = False
                scc_id[w] = scc_count[0]
                if w == u:
                    break
            scc_count[0] += 1

    for u in range(n):
        if disc[u] == -1:
            dfs(u)

    return scc_id


# === 2-SAT 풀개 ===
def solve_2sat(num_vars, clauses):
    """
    2-SAT 보기를 푼다.

    매개변수
    ----------
    num_vars : int
        불 변수의 개수(x_0, x_1, ..., x_{num_vars-1}).
    clauses : list[tuple[int, int]]
        마디 (a, b)마다 논리합이다. 변수는 0부터 센다.
        x_i에는 양의 정수를, x_i 아님에는 음수를 쓴다.
        보기로 (1, -2)은 (x_0 또는 x_1 아님)을 뜻한다.

    반환값
    -------
    list[bool] or None
        충족 가능하면 배정, 아니면 None.
    """
    n = num_vars

    def var_to_node(literal):
        """글자를 함의 그래프 마디 번호에 대응시킨다."""
        if literal > 0:
            return 2 * (literal - 1)      # x_i
        else:
            return 2 * (-literal - 1) + 1  # x_i 아님

    def negate_node(node):
        """부정을 나타내는 마디를 돌려준다."""
        return node ^ 1

    # 함의 그래프 세우기
    total_nodes = 2 * n
    graph = {i: [] for i in range(total_nodes)}

    for a, b in clauses:
        na = var_to_node(a)
        nb = var_to_node(b)
        # (a 또는 b) => (a 아님 -> b) 그리고 (b 아님 -> a)
        graph[negate_node(na)].append(nb)
        graph[negate_node(nb)].append(na)

    # 강한 이음 조각 찾기
    scc_id = tarjan_scc(graph, total_nodes)

    # 충족 가능한지 살피기
    for i in range(n):
        if scc_id[2 * i] == scc_id[2 * i + 1]:
            return None  # x_i과 x_i 아님이 같은 강한 이음 조각에 있음

    # 배정 뽑아내기: SCC(x_i) > SCC(x_i 아님)이면 x_i은 참
    # 타잔은 조각을 거꿀 위상 차례로 내놓으므로 더 큰 쪽이
    # 조각 번호 = 위상 차례에서 더 앞
    assignment = [False] * n
    for i in range(n):
        assignment[i] = scc_id[2 * i] < scc_id[2 * i + 1]

    return assignment


# === 메인 ===
if __name__ == "__main__":
    # 보기: (x1 또는 x2) 그리고 (x1 아님 또는 x3) 그리고 (x2 아님 또는 x3 아님)
    clauses = [(1, 2), (-1, 3), (-2, -3)]
    result = solve_2sat(3, clauses)
    if result is not None:
        print(f"Satisfiable: {result}")
        names = [f"x{i+1}={'T' if v else 'F'}" for i, v in enumerate(result)]
        print(f"Assignment: {', '.join(names)}")
    else:
        print("Unsatisfiable")

    # 충족할 수 없음: (x1 또는 x1) 그리고 (x1 아님 또는 x1 아님)
    clauses2 = [(1, 1), (-1, -1)]
    result2 = solve_2sat(1, clauses2)
    print(f"\nSecond formula: {'Satisfiable' if result2 else 'Unsatisfiable'}")
```

**출력:**
```
Satisfiable: [False, True, False]
Assignment: x1=F, x2=T, x3=F

Second formula: Unsatisfiable
```

## 흔한 2-SAT 무늬

여러 제약 문제가 2-SAT으로 줄어든다:

| 제약 | 마디 |
|---|---|
| "At least one of $a$, $b$ is true" | $(a \lor b)$ |
| "At most one of $a$, $b$ is true" | $(\neg a \lor \neg b)$ |
| "Exactly one of $a$, $b$ is true" | $(a \lor b) \land (\neg a \lor \neg b)$ |
| "$a$ implies $b$" | $(\neg a \lor b)$ |
| "$a$ must be true" | $(a \lor a)$ |
| "$a$ must be false" | $(\neg a \lor \neg a)$ |

## 참고 문헌

- Aspvall, B., Plass, M. F., & Tarjan, R. E. (1979). A linear-time algorithm for testing the truth of certain quantified Boolean formulas. *Information Processing Letters*, 8(3), 121-123.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. 20장.

## 연습문제

**연습문제 1.**
2-SAT 문제를 정의하고 강한 이음 조각과 어떻게 이어지는지 설명하여라.

??? success "연습문제 1 풀이"
    **2-SAT**: given a Boolean formula in CNF where each clause has exactly 2 literals, determine if a satisfying assignment exists. Each clause $(a \lor b)$ is equivalent to implications $(\neg a \Rightarrow b)$ and $(\neg b \Rightarrow a)$. Build an implication graph with vertices for each literal and its negation, and edges for implications. The formula is satisfiable if and only if no variable $x$ has $x$ and $\neg x$ in the same SCC. Time: $O(V + E)$ using Tarjan's or Kosaraju's. $\square$

---

**연습문제 2.**
Construct the implication graph for: $(x_1 \lor x_2) \land (\neg x_1 \lor x_3) \land (\neg x_2 \lor \neg x_3)$.

??? success "연습문제 2 풀이"
    Clause $(x_1 \lor x_2)$: add $\neg x_1 \to x_2$ and $\neg x_2 \to x_1$. Clause $(\neg x_1 \lor x_3)$: add $x_1 \to x_3$ and $\neg x_3 \to \neg x_1$. Clause $(\neg x_2 \lor \neg x_3)$: add $x_2 \to \neg x_3$ and $x_3 \to \neg x_2$. The implication graph has 6 vertices ($x_1, \neg x_1, x_2, \neg x_2, x_3, \neg x_3$) and 6 edges. SCCs: check that no $x_i$ and $\neg x_i$ are in the same SCC. If not, the formula is satisfiable. $\square$

---

**연습문제 3.**
3-SAT은 NP-완전인데 2-SAT은 왜 다항 시간인가?

??? success "연습문제 3 풀이"
    2-SAT clauses have exactly 2 literals, creating implications that form a directed graph. SCC analysis on this graph solves 2-SAT in $O(V + E)$. 3-SAT clauses have 3 literals, and no similar implication graph structure exists — the interaction between 3 literals cannot be reduced to binary implications. The reduction from any NP problem to 3-SAT (Cook-Levin theorem) shows 3-SAT is NP-complete. Adding just one more literal per clause makes the problem fundamentally harder. $\square$

---

**연습문제 4.**
함의 그래프에서 강한 이음 조각을 찾은 뒤 실제 변수 배정을 어떻게 뽑아내는가?

??? success "연습문제 4 풀이"
    Process SCCs in reverse topological order (from sinks to sources in the condensation). For each variable $x_i$: if $x_i$'s SCC comes after $\neg x_i$'s SCC in topological order, set $x_i = \text{True}$; otherwise, set $x_i = \text{False}$. This works because if $\neg x_i$ is in an earlier SCC, the implication chain forces $x_i$ to be true. Tarjan's algorithm conveniently produces SCCs in reverse topological order, so assignments can be read directly. $\square$
