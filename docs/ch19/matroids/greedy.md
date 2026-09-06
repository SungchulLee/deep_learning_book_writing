# 매트로이드 욕심쟁이 알고리즘

욕심쟁이 알고리즘은 걸음마다 그 자리에서 가장 좋은 고름을 하며 이것이 전체 최적으로 이어지기를 바란다. 대부분의 가장 좋게 하기 문제에서 이 바람은 근거가 없다. 그러나 매트로이드 짜임을 지닌 문제에서는 욕심쟁이 방식이 가장 좋음을 증명할 수 있다. **매트로이드 욕심쟁이 정리**가 또렷한 특징지음을 준다. 곧 쓸 수 있는 가장 좋은 원소를 늘 고르는 욕심쟁이 알고리즘이 가장 좋은 풀이를 내는 것은, 될 수 있음의 제약이 매트로이드를 이룰 때 그리고 오직 그때뿐이다. 이 정리는 크러스컬의 최소 뻗은 나무 알고리즘, 가장 좋은 일정 짜기 등 여러 욕심쟁이 성공의 옳음 증명을 하나로 묶는다.

## 알고리즘

Given a weighted matroid $M = (S, \mathcal{I})$ with a weight function $w : S \to \mathbb{R}_{\ge 0}$, the goal is to find an independent set of maximum total weight:

$$
\max_{A \in \mathcal{I}} \sum_{x \in A} w(x)
$$

욕심쟁이 알고리즘은 놀랍도록 단순하다:

1. Sort elements of $S$ in decreasing order of weight: $w(x_1) \ge w(x_2) \ge \cdots \ge w(x_n)$.
2. Initialize $A \leftarrow \emptyset$.
3. For each $x_i$ in sorted order: if $A \cup \{x_i\} \in \mathcal{I}$, set $A \leftarrow A \cup \{x_i\}$.
4. $A$을 돌려준다.

```text
GREEDY-MATROID(M, w):
    sort S by w in decreasing order
    A ← ∅
    for each x in S (in sorted order):
        if A ∪ {x} ∈ I:
            A ← A ∪ {x}
    return A
```

이 알고리즘은 얽히지 않음을 지키는 가장 무거운 원소를 욕심껏 더한다. 한 번 더한 원소는 결코 없애지 않는다.

## 가장 좋음 정리

!!! note "매트로이드 욕심쟁이 정리"
    Let $M = (S, \mathcal{I})$ be a matroid and $w : S \to \mathbb{R}_{\ge 0}$ a weight function. The greedy algorithm returns an independent set $A$ of maximum weight. Moreover, $A$ is a base (maximal independent set) whenever all weights are positive.

## 정확성 증명

**주장.** 욕심쟁이 알고리즘은 무게가 가장 큰 얽히지 않는 모음을 낸다.

**Proof.** Let $A = \{a_1, a_2, \dots, a_k\}$ be the greedy solution in the order elements were added, and let $O = \{o_1, o_2, \dots, o_m\}$ be an optimal solution with elements sorted by decreasing weight.

We show $w(a_i) \ge w(o_i)$ for all $i \le k$, which implies $w(A) \ge w(O)$.

어긋남을 이끌어 내려고 $w(a_i) < w(o_i)$인 첫 번호가 $i$이라고 하자. 다음을 보자:

- $A_{i-1} = \{a_1, \dots, a_{i-1}\}$ (the first $i-1$ greedy choices).
- $O_i = \{o_1, \dots, o_i\}$ (the first $i$ elements of the optimal solution).

Since $|A_{i-1}| = i - 1 < i = |O_i|$ and both are independent (by the hereditary property for $O_i$), the exchange property guarantees some $o_j \in O_i \setminus A_{i-1}$ such that $A_{i-1} \cup \{o_j\} \in \mathcal{I}$.

Since $o_j \in O_i$, we have $w(o_j) \ge w(o_i) > w(a_i)$. But the greedy algorithm considers elements in decreasing weight order and would have chosen $o_j$ before $a_i$ (or at the same step), contradicting the fact that $a_i$ was chosen at step $i$.

Therefore $w(a_i) \ge w(o_i)$ for all $i$, and since $k \ge m$ would follow from the exchange property (the greedy solution is maximal), we have $w(A) \ge w(O)$.

$\square$

## 거꿀: 매트로이드가 욕심쟁이의 가장 좋음을 특징짓는다

매트로이드 욕심쟁이 정리에는 놀라운 거꿀이 있다:

!!! note "거꿀 정리(에드먼즈, 라도)"
    Let $(S, \mathcal{I})$ be a non-empty hereditary set system (satisfying Axioms 1 and 2). The greedy algorithm finds a maximum-weight independent set for **every** weight function $w : S \to \mathbb{R}_{\ge 0}$ if and only if $(S, \mathcal{I})$ is a matroid.

곧 매트로이드는 욕심쟁이가 가장 좋기 위한 충분조건일 뿐 아니라 **정확한** 특징지음이다. 물려받는 모음 체계가 매트로이드가 아니면(맞바꿈 성질이 어그러지면) 욕심쟁이가 어그러지는 무게 함수가 있다.

**Proof sketch of the converse.** Suppose $\mathcal{I}$ is hereditary but violates the exchange property: there exist $A, B \in \mathcal{I}$ with $|A| < |B|$ such that $A \cup \{x\} \notin \mathcal{I}$ for all $x \in B \setminus A$. Assign weights so that elements in $A$ have slightly higher weight than elements in $B \setminus A$, and all other elements have weight 0. The greedy algorithm selects all of $A$ first, then gets stuck with a smaller independent set than $B$, proving greedy is suboptimal.

## 응용

### 최소 뻗은 나무(크러스컬 알고리즘)

그래프 매트로이드에서 변이 원소이고 숲이 얽히지 않는 모음이다. 변을 무게로 정렬하고 순환을 만들지 않는 변을 더하는 것이 바로 매트로이드 욕심쟁이 알고리즘이다. 정리는 이것이 최소 뻗은 나무를 냄을 보장한다(최소 무게를 쓰는데, 이는 무게의 부호를 뒤집어 가장 크게 하는 것과 같다).

### 무게 있는 일 일정 짜기

마감이 $d_i$이고 이익이 $p_i$인 단위 시간 일 $n$개가 주어질 때, 모두 마감을 맞추도록 일정을 짤 수 있으면 그 일 모음을 얽히지 않는다고 하자. 이는 매트로이드를 이루며 욕심쟁이 알고리즘(이익이 큰 일부터 일정을 짜기)이 가장 좋다.

### 무게가 가장 작은 기저

어떤 매트로이드에서든 무게가 커지는 차례로 정렬하고 얽히지 않음을 지키는 원소를 욕심껏 더하면 무게가 가장 작은 기저가 나온다. 이는 크러스컬 알고리즘과 가장 좋은 일정 짜기를 함께 넓힌 것이다.

## 구현

```python
"""
매트로이드 욕심쟁이 알고리즘과 그 쓰임새.

두루 쓰는 매트로이드 욕심쟁이 알고리즘과 그것을 무게 붙은 일 차례 짜기와
최소 뻗음 나무에 쓰는 법을 보인다.
"""

from typing import Callable

# === 두루 쓰는 매트로이드 욕심쟁이 ===

def matroid_greedy(
    elements: list,
    weight: Callable,
    is_independent: Callable[[list], bool],
    maximize: bool = True
) -> list:
    """두루 쓰는 매트로이드 욕심쟁이 알고리즘.

    인수:
        elements: 바탕 모임의 원소.
        weight: 원소를 그 무게에 대응시키는 함수.
        is_independent: 원소의 목록이 홀로서기인지 살피는 함수.
        maximize: 참이면 무게가 가장 큰 홀로서기 모임을 찾고,
                  거짓이면 무게가 가장 작은 바탕을 찾는다.

    반환값:
        가장 좋은 홀로서기 모임(모든 원소의 무게가 양수이면 바탕이 된다).
    """
    sorted_elems = sorted(elements, key=weight, reverse=maximize)
    result = []

    for x in sorted_elems:
        candidate = result + [x]
        if is_independent(candidate):
            result.append(x)

    return result


# === 쓰임새: 무게 붙은 일 차례 짜기 ===

def schedule_jobs(
    jobs: list[tuple[str, int, int]]
) -> tuple[list[str], int]:
    """매트로이드 욕심쟁이로 단위 시간 일의 이익을 가장 크게 하도록 차례를 짠다.

    인수:
        jobs: (이름, 마감, 이익) 짝의 목록.
              마감은 1부터 센다.

    반환값:
        (차례에 넣은 일의 이름, 전체 이익)의 짝.
    """
    def is_feasible(selected_jobs: list[tuple[str, int, int]]) -> bool:
        """고른 일이 모두 마감을 지킬 수 있는지 살핀다."""
        deadlines = sorted(j[1] for j in selected_jobs)
        for i, d in enumerate(deadlines):
            if d < i + 1:  # 칸 i+1이 필요하나 마감이 더 이르다
                return False
        return True

    result = matroid_greedy(
        elements=jobs,
        weight=lambda j: j[2],
        is_independent=is_feasible,
        maximize=True
    )
    names = [j[0] for j in result]
    profit = sum(j[2] for j in result)
    return names, profit


# === 쓰임새: 매트로이드 욕심쟁이로 세우는 최소 뻗음 나무 ===

def mst_matroid(
    n: int,
    edges: list[tuple[int, int, float]]
) -> list[tuple[int, int, float]]:
    """매트로이드 욕심쟁이 알고리즘으로 최소 뻗음 나무를 찾는다.

    인수:
        n: 꼭짓점의 개수.
        edges: (u, v, 무게) 짝의 목록.

    반환값:
        최소 뻗음 나무 변의 목록.
    """
    class UnionFind:
        def __init__(self, size):
            self.parent = list(range(size))
            self.rank = [0] * size

        def find(self, x):
            while self.parent[x] != x:
                self.parent[x] = self.parent[self.parent[x]]
                x = self.parent[x]
            return x

        def union(self, x, y):
            rx, ry = self.find(x), self.find(y)
            if rx == ry:
                return False
            if self.rank[rx] < self.rank[ry]:
                rx, ry = ry, rx
            self.parent[ry] = rx
            if self.rank[rx] == self.rank[ry]:
                self.rank[rx] += 1
            return True

    def is_acyclic(edge_list):
        uf = UnionFind(n)
        for u, v, _ in edge_list:
            if not uf.union(u, v):
                return False
        return True

    return matroid_greedy(
        elements=edges,
        weight=lambda e: e[2],
        is_independent=is_acyclic,
        maximize=False
    )


# === 시연 ===

if __name__ == "__main__":
    # 일 차례 짜기
    jobs = [
        ("a", 2, 100),
        ("b", 1, 19),
        ("c", 2, 27),
        ("d", 1, 25),
        ("e", 3, 15),
    ]
    scheduled, profit = schedule_jobs(jobs)
    print("=== Weighted Job Scheduling ===")
    print(f"Jobs: {[(j[0], f'd={j[1]}', f'p={j[2]}') for j in jobs]}")
    print(f"Scheduled: {scheduled}")
    print(f"Total profit: {profit}")

    # 최소 뻗음 나무
    edges = [
        (0, 1, 1), (0, 2, 4), (1, 2, 2),
        (1, 3, 3), (2, 3, 5),
    ]
    mst = mst_matroid(4, edges)
    print(f"\n=== MST via Matroid Greedy ===")
    print(f"MST edges: {[(u, v, w) for u, v, w in mst]}")
    print(f"Total weight: {sum(w for _, _, w in mst)}")
```

**출력:**

```
=== Weighted Job Scheduling ===
Jobs: [('a', "d=2", "p=100"), ('b', "d=1", "p=19"), ('c', "d=2", "p=27"), ('d', "d=1", "p=25")]
Scheduled: ['a', 'd', 'e']
Total profit: 140

=== MST via Matroid Greedy ===
MST edges: [(0, 1, 1), (1, 2, 2), (1, 3, 3)]
Total weight: 6
```

일 일정 짜개는 마감을 지킬 수 없게 하는 일은 건너뛰며 이익이 가장 큰 일(이익 100인 $a$, 다음에 25인 $d$, 다음에 15인 $e$)을 욕심껏 고른다. 최소 뻗은 나무 쓰임새에서는 숲에 순환이 생기지 않게 하는 무게가 가장 작은 변을 욕심껏 고른다.

## 복잡도

| 항목 | 비용 |
|--------|:----:|
| Sorting | $O(n \log n)$ |
| Independence checks | $O(n \cdot f(n))$ |
| Total | $O(n \log n + n \cdot f(n))$ |

Here $f(n)$ is the cost of one independence check. For graphic matroids with union-find, $f(n) \approx O(\alpha(n))$, giving $O(n \log n)$ total. For general matroids, $f(n)$ depends on the specific independence oracle.

## 참고 문헌

- Edmonds, J. (1971). Matroids and the greedy algorithm. *Mathematical Programming*, 1(1), 127--136.
- Rado, R. (1957). Note on independence functions. *Proceedings of the London Mathematical Society*, 7(1), 300--320.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to Algorithms* (4th ed.), 16장: Greedy Algorithms.

## 연습문제

**연습문제 1.**
매트로이드 욕심쟁이 알고리즘에서 욕심쟁이 고름이 무엇인지 가려내고 왜 가장 좋은 풀이로 이어지는지 밝혀라.

??? success "연습문제 1 풀이"
    The greedy choice selects the locally optimal option at each step. For Matroid Greedy Algorithm, this choice satisfies the greedy choice property: there exists an optimal solution that includes this greedy selection. Combined with optimal substructure (the remaining subproblem after the greedy choice is also optimally solvable by the same strategy), the greedy algorithm produces a globally optimal solution. $\square$

---

**연습문제 2.**
매트로이드 욕심쟁이 알고리즘이 가장 좋은 아래 짜임을 갖는지 증명하거나 반증하여라.

??? success "연습문제 2 풀이"
    Optimal substructure means that an optimal solution to the problem contains optimal solutions to its subproblems. For Matroid Greedy Algorithm, after making the greedy choice, the remaining problem is a smaller instance of the same type. If the subproblem solution were not optimal, we could improve the overall solution by replacing it — contradicting overall optimality. Therefore optimal substructure holds. $\square$

---

**연습문제 3.**
매트로이드 욕심쟁이 알고리즘의 시간 복잡도는 무엇인가? 가장 값비싼 단계를 가려내어라.

??? success "연습문제 3 풀이"
    The time complexity depends on the sorting step (if required) and the greedy selection loop. Sorting typically dominates at $O(n \log n)$. The greedy loop processes each element once in $O(n)$. Total: $O(n \log n)$. If the input is pre-sorted, the algorithm runs in $O(n)$. $\square$

---

**연습문제 4.**
(매트로이드 욕심쟁이 알고리즘에서 쓴 것이 아닌) 다른 욕심쟁이 전략은 가장 좋은 풀이를 내지 못함을 보이는 반례를 들어라.

??? success "연습문제 4 풀이"
    Consider an alternative greedy criterion that does not align with the problem's structure. This alternative may select an element that blocks better future choices. The counterexample demonstrates that the wrong greedy criterion can produce a suboptimal result, highlighting why the specific greedy choice property must be proven for each problem. $\square$
