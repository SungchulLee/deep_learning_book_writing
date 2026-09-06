# 가지 뻗어 묶기로 푸는 떠돌이 장수 문제

떠돌이 장수 문제는 도시를 꼭 한 번씩 들르고 출발점으로 돌아오는 가장 짧은 여행길을 묻는다. 가능한 여행길이 $n!$개라 $n$이 웬만해도 막무가내로는 안 된다. 가지 뻗어 묶기는 어중간한 여행길을 짜임새 있게 살피면서 아래 묶음이 여태 가장 좋은 온전한 여행길을 넘는 것을 쳐 내며, 도시 수십 개 문제도 흔히 효율 좋게 푼다.

## 문제 정식화

도시 $n$개와 거리 행렬 $d[i][j]$(도시 $i$에서 $j$로 가는 값)이 주어질 때 다음을 가장 작게 하는 $\{0, 1, \ldots, n-1\}$의 자리 바꿈 $\pi$을 찾아라:

$$
\text{cost}(\pi) = \sum_{k=0}^{n-2} d[\pi(k)][\pi(k+1)] + d[\pi(n-1)][\pi(0)]
$$

## 가로줄-세로줄 줄이기로 얻는 아래 묶음

떠돌이 장수 문제의 흔한 묶기 재주는 값 행렬을 줄인다:

1. **가로줄 줄이기**: 가로줄마다 최소 원소를 모든 칸에서 뺀다. 뺀 값의 합이 아래 묶음에 보태진다.
2. **세로줄 줄이기**: 가로줄을 줄인 뒤 세로줄마다 최소를 뺀다. 이것도 아래 묶음에 더한다.

그렇게 나온 **줄인 값 행렬**은 가로줄과 세로줄마다 적어도 0이 하나 있다. 줄인 총량이 어느 마디에서든 여행길 값의 아래 묶음을 준다:

$$
\text{LB} = \text{여태의 값} + \text{남은 행렬을 줄인 값}
$$

!!! tip "줄이기가 통하는 까닭"
    여행길마다 가로줄과 세로줄에서 꼭 한 원소씩 써야 한다. 모든 칸에서 가로줄 최소를 빼도 어느 여행길이 가장 좋은지는 바뀌지 않고 바탕 값만 옮겨진다. 쌓인 옮김이 옳은 아래 묶음이다.

## 알고리즘

1. 온전히 줄인 값 행렬과 함께 출발 도시를 뜻하는 뿌리 마디에서 시작한다.
2. 아직 들르지 않은 도시마다 자식 마디를 만든다:
    - 그 도시로 가는 변을 고정하고 남은 아래 문제의 줄인 값 행렬을 셈한다.
    - 자식의 아래 묶음을 셈한다.
3. 아래 묶음이 여태 가장 좋은 여행길을 넘는 자식은 쳐 낸다.
4. 가장 그럴듯한 자식을 살피거나(가장 좋은 것 먼저) 깊이 먼저를 쓴다.
5. 잎 마디(온전한 여행길)에서 나아졌으면 가장 좋은 여행길을 새로 고친다.

## 복잡도

| 갈래 | 값 |
|---|---|
| 최악의 경우 시간 | $O(n! \cdot n^2)$ |
| 실전 시간 | 좋은 묶음이면 훨씬 적다 |
| 공간 | 줄인 행렬에 마디마다 $O(n^2)$ |

## 파이썬 구현

```python
"""
떠돌이 장수 — 행렬 줄이기를 쓴 가지 뻗어 묶기.

가로줄과 세로줄 줄이기로 아래 묶음을 셈하고 깊이 먼저로 살피며
그럴듯하지 않은 갈래를 쳐 낸다.
"""

import math


# === 행렬 줄이기 ===

def reduce_matrix(matrix: list[list[float]]) -> tuple[list[list[float]], float]:
    """가로줄과 세로줄의 최소로 값 행렬을 줄인다.

    (줄인 행렬, 줄인 값)을 돌려준다.
    """
    n = len(matrix)
    reduced = [row[:] for row in matrix]
    cost = 0.0

    # 가로줄 줄이기
    for i in range(n):
        finite_vals = [reduced[i][j] for j in range(n) if reduced[i][j] < math.inf]
        if finite_vals:
            row_min = min(finite_vals)
            if row_min > 0:
                cost += row_min
                for j in range(n):
                    if reduced[i][j] < math.inf:
                        reduced[i][j] -= row_min

    # 세로줄 줄이기
    for j in range(n):
        finite_vals = [reduced[i][j] for i in range(n) if reduced[i][j] < math.inf]
        if finite_vals:
            col_min = min(finite_vals)
            if col_min > 0:
                cost += col_min
                for i in range(n):
                    if reduced[i][j] < math.inf:
                        reduced[i][j] -= col_min

    return reduced, cost


# === 가지 뻗어 묶기 떠돌이 장수 ===

def tsp_branch_bound(dist: list[list[float]]) -> tuple[float, list[int]]:
    """행렬 줄이기를 쓴 가지 뻗어 묶기로 떠돌이 장수 문제를 푼다.

    인수:
        dist: n x n 거리 행렬.

    반환값:
        (가장 적은 값, 여행길). 여행길은 도시 번호의 목록이다.
    """
    n = len(dist)
    INF = math.inf

    # 첫 줄이기
    matrix, root_cost = reduce_matrix(dist)
    best_cost = INF
    best_tour: list[int] = []

    def dfs(
        matrix: list[list[float]], cost: float,
        path: list[int], visited: set[int]
    ) -> None:
        nonlocal best_cost, best_tour

        if len(path) == n:
            total = cost + matrix[path[-1]][path[0]]
            if total < best_cost:
                best_cost = total
                best_tour = path[:]
            return

        last = path[-1]
        for city in range(n):
            if city in visited:
                continue

            # 마지막 도시에서 그 도시로 가는 변의 값
            edge_cost = matrix[last][city]
            if edge_cost >= INF:
                continue

            # 자식 마디의 줄인 행렬을 만든다
            child_matrix = [row[:] for row in matrix]
            # 마지막 도시의 가로줄과 다음 도시의 세로줄을 막는다
            for j in range(n):
                child_matrix[last][j] = INF
            for i in range(n):
                child_matrix[i][city] = INF
            child_matrix[city][path[0]] = INF  # 너무 일찍 돌아가는 것을 막는다

            child_matrix, reduction = reduce_matrix(child_matrix)
            child_cost = cost + edge_cost + reduction

            if child_cost < best_cost:
                path.append(city)
                visited.add(city)
                dfs(child_matrix, child_cost, path, visited)
                path.pop()
                visited.discard(city)

    dfs(matrix, root_cost, [0], {0})
    return best_cost, best_tour


# === 메인 ===

if __name__ == "__main__":
    dist = [
        [math.inf, 10, 15, 20],
        [10, math.inf, 35, 25],
        [15, 35, math.inf, 30],
        [20, 25, 30, math.inf],
    ]

    cost, tour = tsp_branch_bound(dist)
    tour_str = " -> ".join(str(c) for c in tour) + f" -> {tour[0]}"
    print(f"Minimum cost: {cost}")
    print(f"Tour: {tour_str}")
    # 내임:
    # 가장 적은 값: 80
    # 여행길: 0 -> 1 -> 3 -> 2 -> 0
```

## 풀이 예제

위의 도시 4개 거리 행렬에 대해:

1. **뿌리 줄이기**: 가로줄 최소 = $[10, 10, 15, 20]$, 가로줄을 줄인 뒤 세로줄 최소 = $[0, 0, 0, 0]$. 뿌리 아래 묶음 = $55$.
2. **도시 0에서 가지 뻗기**: 변 $0 \to 1$, $0 \to 2$, $0 \to 3$을 시험한다.
3. **변 $0 \to 1$**(줄인 행렬에서 값 0): 새 아래 묶음 = $55 + 0 + \text{줄인 값} = 55$. 그럴듯하다.
4. 온전한 여행길을 찾을 때까지 가지를 계속 뻗는다. 가장 좋은 여행길: $0 \to 1 \to 3 \to 2 \to 0$이고 값은 80이다.

## 참고 문헌

- Little, J. D. C., Murty, K. G., Sweeney, D. W., & Karel, C. (1963). An algorithm for the traveling salesman problem. *Operations Research*, 11(6), 972-989.
- Skiena, S. S. (2020). *The Algorithm Design Manual* (3rd ed.), Chapter 9. Springer.

## 연습문제

**연습문제 1.**
가지 뻗어 묶기로 푸는 떠돌이 장수 문제의 고갱이 생각과 그것이 풀이 공간을 어떻게 짜임새 있게 살피는지 설명하라.

??? success "연습문제 1 풀이"
    가지 뻗어 묶기로 푸는 떠돌이 장수 문제은 풀이 공간을 나무로 보고 살피며 마디마다 어중간한 풀이를 뜻한다. 마디마다 알고리즘은 어중간한 풀이를 넓히고 될 수 있는지 제약을 살핀다. 어중간한 풀이가 제약을 어기거나 (가장 좋거나 옳은 온전한 풀이로 이어질 수 없음이 밝혀지면) 알고리즘은 **가지를 쳐**(되짚어) 그 아래 나무 전체를 살피지 않는다. 가지치기가 찾기 공간의 큰 몫을 없애므로 막무가내보다 효율이 좋다. $\square$

---

**연습문제 2.**
가지 뻗어 묶기로 푸는 떠돌이 장수 문제의 최악의 경우 시간 복잡도는 무엇인가? 가지치기는 언제 찾기 공간을 크게 줄이는가?

??? success "연습문제 2 풀이"
    최악의 경우(가지치기가 없으면) 알고리즘이 풀이 공간 전체를 살피며 이는 흔히 지수나 계승이다. 곧 갈래 수가 $b$이고 깊이가 $d$이면 $O(b^d)$, 자리 바꿈 문제이면 $O(n!)$이다. 가지치기는 다음일 때 찾기를 크게 줄인다. (1) 제약이 빡빡해 될 수 없는 갈래가 많을 때, (2) 좋은 묶음이 갈래를 일찍 없앨 때, (3) 차례를 매기는 어림짐작이 그럴듯한 갈래를 먼저 살필 때이다. 실전에서 가지치기는 도는 시간을 자릿수만큼 줄일 수 있다. $\square$

---

**연습문제 3.**
가지 뻗어 묶기로 푸는 떠돌이 장수 문제의 가지치기 조건을 적어라. 무엇이 좋은 가지치기 잣대를 만드는가?

??? success "연습문제 3 풀이"
    가지치기 잣대는 어중간한 풀이를 언제 버릴지 정한다. 좋은 잣대는 다음과 같다. (1) **될 수 있음**: 어중간한 풀이가 이미 제약을 어긴다. (2) **묶음**: 어중간한 풀이를 가장 좋게 마무리해도 여태 가장 좋은 풀이보다 나을 수 없다. (3) **누름**: 다른 어중간한 풀이가 적어도 그만큼 좋음이 밝혀진다. 잘 듣는 가지치기 잣대는 따지기 값싸고 큰 아래 나무를 없앤다. $\square$

---

**연습문제 4.**
작은 경우에 가지 뻗어 묶기로 푸는 떠돌이 장수 문제을 짜고 살핀 마디의 수를 전체 찾기 공간의 크기와 견주어 세어라.

??? success "연습문제 4 풀이"
    작은 경우(예컨대 N-여왕에서 $n = 8$, 배낭에서 담이 20)에는 전체 찾기 공간에 마디가 수백만 개일 수 있지만 가지치기가 잘 들면 수천 개만 살핀다. (살핀 수 / 전체) 비가 가지치기가 얼마나 잘 드는지 값으로 나타낸다. 제약이 잘 걸린 문제에서는 이 비가 1% 아래일 수 있어 되짚기가 막무가내보다 힘이 셈을 보여 준다. $\square$
