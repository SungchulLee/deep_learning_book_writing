# 욕심쟁이 고름 성질

풀이를 조금씩 세우는 알고리즘을 꾸밀 때 자연스러운 물음이 떠오른다. 곧 결정을 할 때마다 다시는 되돌아보지 않고 못 박아도 될까? **욕심쟁이 고름 성질**이 이 전략이 통한다는 이론의 보장을 준다. 문제가 이 성질을 채우면 걸음마다 전체에서 가장 좋은 풀이로 뻗을 수 있는, 그 자리에서 가장 좋은 고름이 있다. 곧 되돌아가기도 샅샅이 찾기도 필요 없다.

## 느슨한 직관

무대 하나에서 겹치지 않는 행사의 수를 가장 크게 하도록 활동을 고르는 일을 보자. 욕심쟁이 방식은 가장 일찍 끝나는 활동을 고른 뒤 남은 어울리는 활동에 대해 되풀이한다. 욕심쟁이 고름 성질은 가장 일찍 끝나는 활동을 넣는 것이 늘 안전하다고 말한다. 곧 어떤 가장 좋은 풀이는 반드시 그것을 담고 있으므로 바로 못 박아도 잃을 것이 없다.

핵심 눈썰미는 가능한 모든 부분 모음을 살펴볼 필요가 없다는 것이다. 그 대신 돌이킬 수 없는 고름 하나를 하고, 문제를 줄이고, 되돌이한다.

## 엄밀한 정의

Let $\mathcal{P}$ be an optimization problem in which a solution is built by making a sequence of choices $c_1, c_2, \ldots, c_k$. The problem satisfies the **greedy choice property** if the following holds:

!!! note "욕심쟁이 고름 성질"
    For every instance of $\mathcal{P}$, there exists an optimal solution that includes the greedy (locally optimal) first choice. That is, making the locally optimal choice at the current step does not preclude reaching a globally optimal solution.

More precisely, let $S^*$ be any optimal solution. If the greedy choice is $g$, then there exists an optimal solution $S'$ such that $g \in S'$.

## 가장 좋은 아래 짜임과의 관계

욕심쟁이 고름 성질만으로는 욕심쟁이의 옳음을 보장하지 못한다. **가장 좋은 아래 짜임**과 짝을 이뤄야 한다. 곧 욕심쟁이 고름을 한 뒤 남은 작은 문제에도, 그 고름과 아울러 본디 문제의 가장 좋은 풀이를 내는 가장 좋은 풀이가 있어야 한다.

이 두 성질이 함께 욕심쟁이 틀을 뒷받침한다:

1. **욕심쟁이 고름 성질** — 그 자리에서 가장 좋은 고름이 전체로도 안전하다.
2. **가장 좋은 아래 짜임** — 욕심쟁이 고름 뒤에 남은 문제도 가장 좋게 하기 문제이며, 그 가장 좋은 풀이가 욕심쟁이 고름과 아울러 전체에서 가장 좋은 것을 이룬다.

$$
\text{OPT}(\mathcal{P}) = g \cup \text{OPT}(\mathcal{P}')
$$

where $g$ is the greedy choice and $\mathcal{P}'$ is the subproblem remaining after committing to $g$.

## 증명 틀

어떤 문제에 욕심쟁이 고름 성질을 세우는 표준 방식은 **"오려 붙이기"** 또는 **맞바꿈 논증**이다:

1. **Assume** an optimal solution $S^*$ exists.
2. **If** $S^*$ already contains the greedy choice $g$, we are done.
3. **If not**, construct a new solution $S'$ by replacing some element of $S^*$ with $g$.
4. $S'$이 될 수 있음(모든 제약을 채움)을 **보인다**.
5. **Show** that $S'$ is at least as good as $S^*$ (the objective value does not worsen).
6. $S'$이 $g$을 담은 가장 좋은 풀이라고 **매듭짓는다**.

??? example "활동 고르기: 욕심쟁이 고름 증명 얼개"
    **주장.** 가장 일찍 끝나는 활동 $a_1$을 담은 가장 좋은 풀이가 있다.

    **Proof sketch.** Let $S^* = \{a_{j_1}, a_{j_2}, \ldots, a_{j_k}\}$ be an optimal set of non-overlapping activities sorted by finish time. If $a_{j_1} = a_1$, we are done. Otherwise, $a_1$ finishes no later than $a_{j_1}$, so replacing $a_{j_1}$ with $a_1$ preserves non-overlap with $a_{j_2}, \ldots, a_{j_k}$. The resulting set $S' = \{a_1, a_{j_2}, \ldots, a_{j_k}\}$ has the same size $k$, so it is also optimal. $\square$

## 욕심쟁이 고름 성질이 어그러질 때

모든 가장 좋게 하기 문제가 욕심쟁이 고름 성질을 갖지는 않는다. 고전적인 반례는 **0-1 배낭 문제**이다. 곧 물건을 쪼갤 수 없으므로 값 대 무게 비가 가장 큰 물건을 먼저 고르면 가장 좋지 않은 풀이가 될 수 있다.

??? warning "반례: 0-1 배낭"
    배낭의 담이가 $W = 50$이고 물건이 셋 있다고 하자:

    | 물건 | 무게 | 값 | 비 |
    |------|--------|-------|-------|
    | A    | 10     | 60    | 6.0   |
    | B    | 20     | 100   | 5.0   |
    | C    | 30     | 120   | 4.0   |

    (비로 고르는) 욕심쟁이 전략은 A를 먼저, 다음에 B를 골라 무게 30에 전체 값 160을 얻는다. 그러나 가장 좋은 풀이는 B + C으로 전체 값 220에 무게 50이다. 욕심쟁이의 첫 고름(물건 A)은 가장 좋은 풀이에 들지 않는다.

## 동적 계획과의 견줌

욕심쟁이 알고리즘과 동적 계획 모두 가장 좋은 아래 짜임을 써먹는다. 결정적인 차이는 욕심쟁이 고름 성질이 성립하느냐이다:

| 성질 | 욕심쟁이 | 동적 계획 |
|----------|--------|---------------------|
| 가장 좋은 아래 짜임 | 필요함 | 필요함 |
| 욕심쟁이 고름 성질 | 필요함 | 필요 없음 |
| 겹치는 작은 문제 | 필요 없음 | 써먹음 |
| 고름을 다시 헤아림 | 전혀 안 함 | 함(작은 문제를 모두 푼다) |
| Time complexity | Often $O(n \log n)$ | Often $O(n^2)$ or $O(nW)$ |

욕심쟁이 고름 성질이 성립하면 겹치는 작은 문제를 푸는 덧짐을 피하므로 욕심쟁이 알고리즘이 낫다.

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), 16장. MIT Press.
- Kleinberg, J. & Tardos, E. (2006). *Algorithm Design*, 4장. Pearson.

## 연습문제

**연습문제 1.**
욕심쟁이 고름 성질에서 욕심쟁이 고름이 무엇인지 가려내고 왜 가장 좋은 풀이로 이어지는지 밝혀라.

??? success "연습문제 1 풀이"
    The greedy choice selects the locally optimal option at each step. For Greedy Choice Property, this choice satisfies the greedy choice property: there exists an optimal solution that includes this greedy selection. Combined with optimal substructure (the remaining subproblem after the greedy choice is also optimally solvable by the same strategy), the greedy algorithm produces a globally optimal solution. $\square$

---

**연습문제 2.**
욕심쟁이 고름 성질이 가장 좋은 아래 짜임을 갖는지 증명하거나 반증하여라.

??? success "연습문제 2 풀이"
    Optimal substructure means that an optimal solution to the problem contains optimal solutions to its subproblems. For Greedy Choice Property, after making the greedy choice, the remaining problem is a smaller instance of the same type. If the subproblem solution were not optimal, we could improve the overall solution by replacing it — contradicting overall optimality. Therefore optimal substructure holds. $\square$

---

**연습문제 3.**
욕심쟁이 고름 성질의 시간 복잡도는 무엇인가? 가장 값비싼 단계를 가려내어라.

??? success "연습문제 3 풀이"
    The time complexity depends on the sorting step (if required) and the greedy selection loop. Sorting typically dominates at $O(n \log n)$. The greedy loop processes each element once in $O(n)$. Total: $O(n \log n)$. If the input is pre-sorted, the algorithm runs in $O(n)$. $\square$

---

**연습문제 4.**
(욕심쟁이 고름 성질에서 쓴 것이 아닌) 다른 욕심쟁이 전략은 가장 좋은 풀이를 내지 못함을 보이는 반례를 들어라.

??? success "연습문제 4 풀이"
    Consider an alternative greedy criterion that does not align with the problem's structure. This alternative may select an element that blocks better future choices. The counterexample demonstrates that the wrong greedy criterion can produce a suboptimal result, highlighting why the specific greedy choice property must be proven for each problem. $\square$
