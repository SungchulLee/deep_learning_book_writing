# 가장 좋은 이진 찾기 나무

이진 찾기 나무에서 열쇠를 찾는 값은 그 깊이에 달렸다. 어떤 열쇠를 다른 것보다 훨씬 자주 찾는다면 자주 찾는 열쇠를 뿌리 가까이 두어 기대 찾기 값을 줄일 수 있다. **가장 좋은 이진 찾기 나무** 문제는 다가감 잦기를 안다고 할 때 이 기대 값을 가장 작게 하는 나무 짜임을 동적 계획으로 찾는다.

## 문제 서술

Given $n$ keys $k_1 < k_2 < \cdots < k_n$ with search probabilities $p_1, p_2, \dots, p_n$ and $n + 1$ dummy keys $d_0, d_1, \dots, d_n$ representing unsuccessful searches with probabilities $q_0, q_1, \dots, q_n$, where:

$$
\sum_{i=1}^{n} p_i + \sum_{j=0}^{n} q_j = 1
$$

이진 찾기 나무 $T$의 **기대 찾기 값**은 다음과 같다:

$$
E[\text{cost}] = \sum_{i=1}^{n} p_i \cdot (\text{depth}_T(k_i) + 1) + \sum_{j=0}^{n} q_j \cdot (\text{depth}_T(d_j) + 1)
$$

목표는 이 기대 값을 가장 작게 하는 이진 찾기 나무를 찾는 것이다.

## 가장 좋은 밑짜임

If an optimal BST has $k_r$ as its root, then the left subtree (containing $k_1, \dots, k_{r-1}$) must be an optimal BST for those keys, and similarly for the right subtree. This optimal substructure enables a DP solution.

Define $e[i, j]$ as the expected cost of an optimal BST for keys $k_i, \dots, k_j$ (with dummy keys $d_{i-1}, \dots, d_j$). The weight of this subproblem is:

$$
w[i, j] = \sum_{\ell=i}^{j} p_\ell + \sum_{\ell=i-1}^{j} q_\ell
$$

## 점화식

When $k_r$ is chosen as the root of the subtree for keys $k_i, \dots, k_j$, the cost increases by $w[i, j]$ (each node's depth increases by 1 when it becomes a child of $k_r$):

$$
e[i, j] = \min_{i \le r \le j} \bigl\{e[i, r{-}1] + e[r{+}1, j] + w[i, j]\bigr\}
$$

Base case: $e[i, i{-}1] = q_{i-1}$ (a subtree containing only the dummy key $d_{i-1}$).

## 구현

```python
"""
동적 계획으로 얻는 가장 좋은 이진 찾기 나무.

열쇠의 다가감 확률이 주어질 때 기대 찾기 값을 가장 작게 하는
이진 찾기 나무 짜임을 O(n^3) 시간에 찾는다.
"""

# === 가장 좋은 이진 찾기 나무 ===

def optimal_bst(
    p: list[float], q: list[float]
) -> tuple[float, list[list[int]]]:
    """가장 좋은 이진 찾기 나무의 값과 뿌리 표 셈하기.

    인수:
        p: 열쇠 k_1..k_n의 찾기 확률(1부터 셈, p[0]은 쓰지 않음).
        q: 허수아비 열쇠 d_0..d_n의 찾기 확률.

    반환값:
        (최소 기대 값, 뿌리 표) 튜플. 여기서 root[i][j]은
        열쇠 k_i..k_j의 가장 좋은 뿌리 번호이다.
    """
    n = len(p) - 1  # p은 1부터 센다

    # e[i][j] = 열쇠 k_i..k_j의 기대 값
    # w[i][j] = 열쇠 k_i..k_j의 전체 확률 무게
    e = [[0.0] * (n + 2) for _ in range(n + 2)]
    w = [[0.0] * (n + 2) for _ in range(n + 2)]
    root = [[0] * (n + 1) for _ in range(n + 1)]

    # 바탕 경우: e[i][i-1] = q[i-1]
    for i in range(1, n + 2):
        e[i][i - 1] = q[i - 1]
        w[i][i - 1] = q[i - 1]

    # 사슬 길이를 늘려 가며 표 채우기
    for length in range(1, n + 1):
        for i in range(1, n - length + 2):
            j = i + length - 1
            e[i][j] = float('inf')
            w[i][j] = w[i][j - 1] + p[j] + q[j]

            for r in range(i, j + 1):
                cost = e[i][r - 1] + e[r + 1][j] + w[i][j]
                if cost < e[i][j]:
                    e[i][j] = cost
                    root[i][j] = r

    return e[1][n], root


def print_optimal_bst(root: list[list[int]], i: int, j: int,
                      parent: str = "root") -> None:
    """가장 좋은 이진 찾기 나무의 짜임 찍기."""
    if i > j:
        print(f"  d_{j} is {parent}")
        return
    r = root[i][j]
    print(f"  k_{r} is {parent}")
    print_optimal_bst(root, i, r - 1, f"left child of k_{r}")
    print_optimal_bst(root, r + 1, j, f"right child of k_{r}")


# === 시연 ===

if __name__ == "__main__":
    # CLRS의 보기: 확률이 주어진 열쇠 5개
    p = [0, 0.15, 0.10, 0.05, 0.10, 0.20]  # 1부터 센다
    q = [0.05, 0.10, 0.05, 0.05, 0.05, 0.10]

    cost, root = optimal_bst(p, q)
    print(f"Minimum expected search cost: {cost:.2f}")
    print("Optimal BST structure:")
    print_optimal_bst(root, 1, 5)
```

**출력:**

```
Minimum expected search cost: 2.75
Optimal BST structure:
  k_2 is root
  k_1 is left child of k_2
  d_0 is left child of k_1
  d_1 is right child of k_1
  k_5 is right child of k_2
  k_4 is left child of k_5
  k_3 is left child of k_4
  d_2 is left child of k_3
  d_3 is right child of k_3
  d_4 is right child of k_4
  d_5 is right child of k_5
```

뿌리에 있는 열쇠 $k_2$이 다가감 잦기의 균형을 잡는다. 가장 자주 찾는 열쇠 $k_5$($p_5 = 0.20$)이 깊이 1에 있어 기대 값에 보태는 몫을 가장 작게 한다.

## 복잡도

| 항목 | 비용 |
|--------|:----:|
| Time   | $O(n^3)$ |
| Space  | $O(n^2)$ |

The three nested loops (length, start position, root choice) give $O(n^3)$ time. Knuth's optimization reduces this to $O(n^2)$ by observing that $\text{root}[i, j-1] \le \text{root}[i, j] \le \text{root}[i+1, j]$, which limits the search range for $r$.

## 균형 이진 탐색 트리와의 비교

| 전략 | 기대 값 | 보장 |
|----------|:------------:|:---------:|
| 가장 좋은 이진 찾기 나무 | 가능한 최솟값 | 잦기를 알아야 한다 |
| Balanced BST | $O(\log n)$ per search | No frequency knowledge needed |
| Splay tree | $O(\log n)$ amortized | Adapts to access patterns |

가장 좋은 이진 찾기 나무는 붙박이 짜임이다. 다가감 잦기가 때에 따라 바뀌면 스플레이 나무 같은 스스로 고치는 나무가 움직이는 대안이 된다.

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to Algorithms* (4th ed.), 15장: Dynamic Programming.
- Knuth, D. E. (1971). Optimum binary search trees. *Acta Informatica*, 1(1), 14--25.

## 연습문제

**연습문제 1.**
가장 좋은 이진 찾기 나무의 핵심 생각과 그 시간 복잡도를 설명하여라.

??? success "연습문제 1 풀이"
    Optimal Binary Search Tree applies the divide-and-conquer paradigm: split the problem into smaller subproblems, solve them recursively, and combine the results. The time complexity is determined by the recurrence relation governing the subproblem sizes and the combination cost. The Master Theorem or recursion tree analysis typically gives the closed-form complexity. $\square$

---

**연습문제 2.**
가장 좋은 이진 찾기 나무의 되돌이 관계식을 쓰고 마스터 정리로 풀어라.

??? success "연습문제 2 풀이"
    The recurrence depends on the specific algorithm's division strategy (number of subproblems $a$, size reduction factor $b$, and combination cost $f(n)$). Apply the Master Theorem: compare $f(n)$ with $n^{\log_b a}$ to determine which case applies. If $f(n) = \Theta(n^{\log_b a})$ (case 2), $T(n) = \Theta(n^{\log_b a} \log n)$. $\square$

---

**연습문제 3.**
가장 좋은 이진 찾기 나무이 막무가내 방식보다 나은 장면을 설명하여라. 얼마나 빨라지는지 수로 나타내어라.

??? success "연습문제 3 풀이"
    The brute-force approach typically runs in $O(n^2)$ or worse. The divide-and-conquer approach achieves a lower complexity by reducing redundant computation through recursive decomposition. For input size $n = 10^6$, the difference between $O(n^2) = 10^{12}$ and $O(n \log n) = 2 \times 10^7$ operations is a factor of $50{,}000$. $\square$

---

**연습문제 4.**
가장 좋은 이진 찾기 나무의 바탕 경우는 무엇인가? 그것이 알고리즘 전체의 옳음에 어떤 영향을 주는가?

??? success "연습문제 4 풀이"
    Base cases handle inputs too small to subdivide further (typically $n \leq 1$ or $n \leq 2$). They must return correct results directly. Without proper base cases, the recursion never terminates. Choosing a larger base case (e.g., $n \leq 10$) and switching to a simpler algorithm can improve practical performance by reducing recursion overhead while maintaining the same asymptotic complexity. $\square$
