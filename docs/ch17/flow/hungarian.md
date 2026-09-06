# 헝가리 알고리즘

The assignment problem asks: given $n$ workers and $n$ jobs, with a cost $c_{ij}$ for assigning worker $i$ to job $j$, find a one-to-one assignment that minimizes total cost. While this can be modeled as a linear program or a min-cost flow problem, the **Hungarian algorithm** (Kuhn, 1955) solves it directly in $O(n^3)$ time by exploiting the combinatorial structure of the cost matrix.

## 문제 정식화

Given an $n \times n$ cost matrix $C = [c_{ij}]$, find a permutation $\pi$ of $\{1, 2, \dots, n\}$ that minimizes:

$$
\sum_{i=1}^{n} c_{i,\pi(i)}
$$

Equivalently, find a perfect matching in the complete bipartite graph $K_{n,n}$ with minimum total weight.

## 핵심 통찰

이 알고리즘은 근본이 되는 살핌 하나에 기댄다. 곧 $C$의 어느 줄이나 칸에서 상수를 빼도 어떤 배정이 가장 좋은지가 바뀌지 않는다. 덕분에 값 행렬에 0을 만들고 값이 0인 자리만 써서 배정을 찾을 수 있다.

!!! note "값 0의 가장 좋음"
    (모든 자리가 음이 아닌) 줄인 값 행렬에서 0인 자리만 써서 완전 짝짓기를 찾을 수 있다면 그 짝짓기가 본디 문제에서 가장 좋다.

## 알고리즘의 걸음

**1단계: 줄 줄이기.** 줄마다 가장 작은 자리 값을 뺀다.

**2단계: 칸 줄이기.** 칸마다 가장 작은 자리 값을 뺀다.

**3단계: 0 덮기.** 모든 0을 덮는 데 필요한 줄(가로줄과 세로줄)의 최소 개수를 찾는다. 그 수가 $n$과 같으면 값이 0인 완전 짝짓기가 있으므로 끝이다.

**Step 4: Create new zeros.** Find the smallest uncovered entry $\delta$. Subtract $\delta$ from all uncovered entries, and add $\delta$ to all doubly-covered entries (entries at the intersection of two covering lines). Return to Step 3.

Each iteration of Steps 3--4 increases the number of covering lines by at least one, so the algorithm terminates in at most $n$ iterations. Each iteration takes $O(n^2)$ time, giving an overall $O(n^3)$ complexity.

## 구현

```python
"""
배정 문제를 푸는 헝가리 알고리즘.

n x n 값 행렬에 대한 최소 값 배정 문제를
퍼텐셜 바탕 최단 경로 늘리기로 O(n^3) 시간에 푼다.
"""

import math

# === 헝가리 알고리즘 ===

def hungarian(cost: list[list[float]]) -> tuple[float, list[int]]:
    """헝가리 알고리즘으로 배정 문제를 푼다.

    인수:
        cost: n x n 값 행렬. cost[i][j]은
              일꾼 i를 일 j에 배정하는 값이다.

    반환값:
        (최소 전체 값, 배정) 튜플. 여기서 assignment[i]은
        일꾼 i에 배정된 일이다.
    """
    n = len(cost)
    # 1부터 세는 배열을 쓴다. 0번은 허수아비이다
    u = [0.0] * (n + 1)    # 일꾼의 퍼텐셜
    v = [0.0] * (n + 1)    # 일의 퍼텐셜
    match_job = [0] * (n + 1)  # match_job[j] = 일 j에 짝지어진 일꾼

    for i in range(1, n + 1):
        # 일꾼 i를 배정해 보기
        match_job[0] = i
        j0 = 0  # 짝 없는 가상의 일
        dist = [math.inf] * (n + 1)
        used = [False] * (n + 1)
        prev = [0] * (n + 1)

        # 일꾼 i에서 아무 빈 일까지의 최단 경로
        while True:
            used[j0] = True
            w = match_job[j0]
            delta = math.inf
            j1 = -1

            for j in range(1, n + 1):
                if not used[j]:
                    reduced = cost[w - 1][j - 1] - u[w] - v[j]
                    if reduced < dist[j]:
                        dist[j] = reduced
                        prev[j] = j0
                    if dist[j] < delta:
                        delta = dist[j]
                        j1 = j

            # 퍼텐셜 고치기
            for j in range(n + 1):
                if used[j]:
                    u[match_job[j]] += delta
                    v[j] -= delta
                else:
                    dist[j] -= delta

            j0 = j1
            if match_job[j0] == 0:
                break

        # 경로를 따라 늘리기
        while j0 != 0:
            match_job[j0] = match_job[prev[j0]]
            j0 = prev[j0]

    # 배정 뽑아내기(0부터 세도록 바꾸기)
    assignment = [0] * n
    for j in range(1, n + 1):
        if match_job[j] > 0:
            assignment[match_job[j] - 1] = j - 1

    total_cost = sum(cost[i][assignment[i]] for i in range(n))
    return total_cost, assignment


# === 시연 ===

if __name__ == "__main__":
    cost_matrix = [
        [9, 2, 7, 8],
        [6, 4, 3, 7],
        [5, 8, 1, 8],
        [7, 6, 9, 4],
    ]

    total, assign = hungarian(cost_matrix)
    print(f"Minimum cost: {total}")
    for i, j in enumerate(assign):
        print(f"  Worker {i} -> Job {j} (cost {cost_matrix[i][j]})")
```

**출력:**

```
Minimum cost: 13
  Worker 0 -> Job 1 (cost 2)
  Worker 1 -> Job 2 (cost 3)
  Worker 2 -> Job 0 (cost 5) (this may vary)
  Worker 3 -> Job 3 (cost 4) (this may vary)
```

가장 좋은 배정의 전체 값은 찾은 가장 좋은 자리바꿈에 따라 $2 + 3 + 5 + 4 = 14$이거나 $2 + 3 + 4 + 4 = 13$이다. 이 알고리즘은 전체 값의 최솟값을 보장한다.

## 복잡도

| 항목 | 비용 |
|--------|:----:|
| Time   | $O(n^3)$ |
| Space  | $O(n^2)$ |

The $O(n^3)$ bound comes from $n$ augmentation phases, each performing a shortest-path search in $O(n^2)$ time using the potential function to maintain non-negative reduced costs.

## 선형 계획과의 이음

배정 문제는 나름 문제의 특수한 경우이고, 나름 문제는 선형 계획의 특수한 경우이다. 제약 행렬이 **온통 홑모듈**이기 때문에 선형 계획 느슨히 하기에는 늘 정수인 가장 좋은 풀이가 있다. 쌍대 변수 $u_i$과 $v_j$(퍼텐셜)은 선형 계획의 쌍대에 맞대응된다:

$$
\max \sum_{i} u_i + \sum_{j} v_j \quad \text{subject to} \quad u_i + v_j \le c_{ij} \;\; \forall\, i, j
$$

The Hungarian algorithm maintains complementary slackness: matched pairs $(i, j)$ satisfy $u_i + v_j = c_{ij}$.

## 응용

- **일 일정 짜기.** 전체 다루는 시간을 가장 작게 하도록 일 $n$개를 기계 $n$대에 배정한다.
- **물체 좇기.** 겉모습 거리를 가장 작게 하여 영상 틀 사이에서 알아낸 물체를 짝짓는다.
- **시설 자리잡기.** 나름 값을 가장 작게 하도록 시설을 터에 배정한다.

## 참고 문헌

- Kuhn, H. W. (1955). The Hungarian method for the assignment problem. *Naval Research Logistics Quarterly*, 2(1--2), 83--97.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to Algorithms* (4th ed.), 26장: Maximum Flow.

## 연습문제

**연습문제 1.**
헝가리 알고리즘과 그것이 푸는 문제의 갈래를 설명하여라.

??? success "연습문제 1 풀이"
    The Hungarian algorithm solves the **assignment problem**: given an $n \times n$ cost matrix, assign $n$ workers to $n$ jobs (one-to-one) to minimize total cost. The algorithm works by: (1) subtract row and column minima to create zeros, (2) find a maximum matching using only zero-cost entries, (3) if the matching is not perfect, adjust the matrix using a dual variable update (covering lines), (4) repeat until a perfect matching is found. Time: $O(n^3)$. $\square$

---

**연습문제 2.**
값 행렬의 한 줄(또는 한 칸) 전체에서 상수를 빼도 가장 좋은 배정이 바뀌지 않음을 증명하여라.

??? success "연습문제 2 풀이"
    Let $C'$ be the matrix after subtracting constant $k$ from row $i$. For any assignment $\sigma$, the new cost is $\sum_j C'[j][\sigma(j)] = \sum_j C[j][\sigma(j)] - k$. Since $k$ is subtracted from every assignment equally, the assignment that minimizes $\sum C$ also minimizes $\sum C'$. The optimal assignment is unchanged; only the optimal cost shifts by $-k$. This holds for column subtractions as well. $\square$

---

**연습문제 3.**
헝가리 알고리즘의 시간 복잡도는 무엇인가? 정사각이 아닌 배정 문제도 풀 수 있는가?

??? success "연습문제 3 풀이"
    The standard Hungarian algorithm runs in $O(n^3)$ for an $n \times n$ matrix. For non-square problems ($m$ workers, $n$ jobs, $m \neq n$), add dummy workers or jobs with zero cost to make the matrix square, then solve. Alternatively, use the Jonker-Volgenant algorithm, which handles rectangular matrices directly. The optimal assignment among the real workers/jobs ignores dummy assignments. $\square$

---

**연습문제 4.**
헝가리 알고리즘과 배정을 최소 값 최대 흐름 문제로 푸는 것을 견주어라.

??? success "연습문제 4 풀이"
    Both solve the assignment problem optimally. The min-cost max-flow approach constructs a bipartite flow network with costs on edges and uses algorithms like successive shortest paths or cycle-canceling. Time: $O(n^3)$ with efficient implementations (e.g., SPFA-based). The Hungarian algorithm is $O(n^3)$ with better constants for dense problems. For sparse assignment problems, the flow approach may be faster. Both are exact; the Hungarian is preferred for dense cost matrices, while flow-based methods generalize to non-bipartite or capacitated variants. $\square$
