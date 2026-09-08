# 2-충족 가능성

충족 가능성 문제(SAT)는 불 식을 참으로 만드는 변수 배정이 있는지 묻는다. 일반적으로 SAT은 NP-완전이지만, 마디마다 글자가 정확히 둘인 특수한 경우인 **2-SAT**은 [강하게 이어진 조각](definition.md)을 거쳐 우아한 다항 시간 풀이를 허락한다. 논리에서 그래프 이론으로 옮기는 이 줄임은 조각 쪼갬의 가장 아름다운 쓰임새 가운데 하나이다.

---

## 1. 문제 정식화

**2-SAT** 사례는 불 변수 $n$개 $x_1, x_2, \ldots, x_n$과 마디 $m$개로 이루어지며, 마디마다 정확히 글자 둘의 논리합이다.

$$
(l_{1,1} \lor l_{1,2}) \land (l_{2,1} \lor l_{2,2}) \land \cdots \land (l_{m,1} \lor l_{m,2})
$$

글자 $l_{i,j}$은 저마다 변수 $x_k$이거나 그 부정 $\neg x_k$이다. 목표는 모든 절을 한꺼번에 참으로 만드는 진리값 배정이 있는지 가리는 것이다.

---

## 2. 함의 그래프

고갱이 눈썰미는 마디 $(a \lor b)$이 논리적으로 함의 둘과 같다는 것이다.

$$
(\neg a \Rightarrow b) \quad \text{and} \quad (\neg b \Rightarrow a)
$$

**함의 그래프** $G = (V, E)$을 다음과 같이 세운다:

- **꼭짓점:** 변수 $x_i$마다 꼭짓점 둘, 곧 $x_i$과 $\neg x_i$을 만든다. 그러므로 $|V| = 2n$이다.
- **변:** 마디 $(a \lor b)$마다 방향 있는 변 $\neg a \to b$과 $\neg b \to a$을 더한다.

!!! tip "왜 함의인가?"
    마디 $(a \lor b)$은 "$a$과 $b$ 가운데 적어도 하나는 참이어야 한다"는 뜻이다. $a$이 거짓이면 $b$이 참이어야 하므로 $\neg a \Rightarrow b$이다. 대칭으로 $b$이 거짓이면 $a$이 참이어야 하므로 $\neg b \Rightarrow a$이다.

---

## 3. 강한 이음 조각에 바탕한 풀이

논리식이 충족 가능한 것은 함의 그래프에서 어떤 변수 $x_i$과 그 부정 $\neg x_i$도 같은 강한 이음 조각에 들지 않는 것과 같은 뜻이다.

!!! note "2-SAT 충족 가능성 정리"
    2-SAT 논리식이 충족 가능한 것은 모든 변수 $x_i$에 대해 꼭짓점 $x_i$과 $\neg x_i$이 함의 그래프의 서로 다른 강한 이음 조각에 있는 것과 같은 뜻이다.

**증명 얼개(정방향).** $x_i$과 $\neg x_i$이 같은 조각에 있으면 함의 그래프에 $x_i$에서 $\neg x_i$으로, 또 $\neg x_i$에서 $x_i$으로 가는 경로가 있다. 이는 $x_i = \text{true}$으로 두면 $x_i = \text{false}$이 되고 거꾸로도 그렇다는 뜻이므로 논리식이 충족 불가능하다. $\square$

**증명 얼개(역방향).** 어떤 변수도 그 부정과 조각을 함께 쓰지 않으면 [오그린 그래프](condensation.md)를 위상 차례의 거꾸로로 다루며 진리값을 앞뒤 맞게 배정할 수 있다. 아직 배정하지 않은 변수마다 위상 차례에서 더 뒤에 오는 조각의 글자을 참으로 둔다. $\square$

---

## 4. 배정 뽑아내기

충족 가능함을 확인하고 나면 배정을 다음과 같이 뽑아낸다:

1. ([타잔](tarjan.md)이나 [코사라주](kosaraju.md)를 써서) 함의 그래프의 강한 이음 조각을 셈한다.
2. 변수 $x_i$마다 $x_i$을 담은 조각과 $\neg x_i$을 담은 조각의 위상 차례를 견준다.
3. $x_i$의 조각이 $\neg x_i$의 조각보다 뒤에 오면(위상 차례가 더 높으면) $x_i = \text{true}$으로 둔다.

직관으로 보면, 오그린 유향 비순환 그래프에서 뒤쪽 조각이 함의 사슬의 앞쪽 조각을 "덮어쓴다".

---

## 5. 복잡도

함의 그래프를 세우는 데 $O(n + m)$이 든다. 강한 이음 조각을 찾는 데 $O(V + E) = O(n + m)$이 든다. 따라서:

$$
T(n, m) = O(n + m)
$$

---

## 6. 구현

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

---

## 7. 흔한 2-SAT 무늬

여러 제약 문제가 2-SAT으로 줄어든다:

| 제약 | 마디 |
|---|---|
| "$a$과 $b$ 가운데 적어도 하나가 참" | $(a \lor b)$ |
| "$a$과 $b$ 가운데 많아야 하나가 참" | $(\neg a \lor \neg b)$ |
| "$a$과 $b$ 가운데 꼭 하나가 참" | $(a \lor b) \land (\neg a \lor \neg b)$ |
| "$a$ implies $b$" | $(\neg a \lor b)$ |
| "$a$ must be true" | $(a \lor a)$ |
| "$a$ must be false" | $(\neg a \lor \neg a)$ |

---

## 연습문제

**연습문제 1.**
2-SAT 문제를 정의하고 강한 이음 조각과 어떻게 이어지는지 설명하여라.

??? success "연습문제 1 풀이"
    **2-SAT**: 마디마다 글자가 정확히 둘인 논리곱 표준형 불 식이 주어졌을 때 충족하는 배정이 있는지 가린다. 마디 $(a \lor b)$은 함의 $(\neg a \Rightarrow b)$과 $(\neg b \Rightarrow a)$과 같다. 글자와 그 부정마다 꼭짓점을 두고 함의마다 변을 두어 함의 그래프를 세운다. 이 식이 충족 가능한 것은 어떤 변수 $x$도 $x$과 $\neg x$이 같은 강한 이음 조각에 들지 않는 것과 같은 뜻이다. 타잔이나 코사라주를 쓰면 시간은 $O(V + E)$이다. $\square$

---

**연습문제 2.**
$(x_1 \lor x_2) \land (\neg x_1 \lor x_3) \land (\neg x_2 \lor \neg x_3)$의 함의 그래프를 세워라.

??? success "연습문제 2 풀이"
    마디 $(x_1 \lor x_2)$: $\neg x_1 \to x_2$과 $\neg x_2 \to x_1$을 더한다. 마디 $(\neg x_1 \lor x_3)$: $x_1 \to x_3$과 $\neg x_3 \to \neg x_1$을 더한다. 마디 $(\neg x_2 \lor \neg x_3)$: $x_2 \to \neg x_3$과 $x_3 \to \neg x_2$을 더한다. 함의 그래프에는 꼭짓점 6개($x_1, \neg x_1, x_2, \neg x_2, x_3, \neg x_3$)와 변 6개가 있다. 강한 이음 조각을 보아 어떤 $x_i$과 $\neg x_i$도 같은 조각에 없는지 살핀다. 그러면 이 식은 충족 가능하다. $\square$

---

**연습문제 3.**
3-SAT은 NP-완전인데 2-SAT은 왜 다항 시간인가?

??? success "연습문제 3 풀이"
    2-SAT의 마디에는 글자가 정확히 둘 있어 방향 그래프를 이루는 함의가 생긴다. 이 그래프에서 강한 이음 조각을 살피면 2-SAT을 $O(V + E)$에 푼다. 3-SAT의 마디에는 글자가 셋이라 비슷한 함의 그래프 얼개가 없다. 글자 셋 사이의 주고받음은 둘씩의 함의로 줄일 수 없기 때문이다. 어떤 NP 문제든 3-SAT으로 되돌릴 수 있다는 것(쿡-레빈 정리)이 3-SAT이 NP-완전임을 보인다. 마디마다 글자를 하나만 더해도 문제가 밑바탕부터 어려워진다. $\square$

---

**연습문제 4.**
함의 그래프에서 강한 이음 조각을 찾은 뒤 실제 변수 배정을 어떻게 뽑아내는가?

??? success "연습문제 4 풀이"
    강한 이음 조각을 위상 차례의 거꾸로로 다룬다(오그린 그래프의 바닥에서 근원 쪽으로). 변수 $x_i$마다 $x_i$의 조각이 위상 차례에서 $\neg x_i$의 조각보다 뒤에 오면 $x_i = \text{True}$으로 두고 아니면 $x_i = \text{False}$으로 둔다. $\neg x_i$이 더 이른 조각에 있으면 함의 사슬이 $x_i$을 참이 되게 하므로 이렇게 하면 된다. 타잔 알고리즘이 마침 위상 차례의 거꾸로로 조각을 내놓으므로 배정을 곧바로 읽을 수 있다. $\square$

## 정리하며

이 마당은 문제 정식화、함의 그래프、강한 이음 조각에 바탕한 풀이、배정 뽑아내기을 차례로 짚었다.

**참고 문헌**

- Aspvall, B., Plass, M. F., & Tarjan, R. E. (1979). A linear-time algorithm for testing the truth of certain quantified Boolean formulas. *Information Processing Letters*, 8(3), 121-123.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. 20장.
