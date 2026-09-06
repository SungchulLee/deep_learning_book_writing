# 욕심쟁이가 앞선다

맞바꿈 논증은 아무 가장 좋은 풀이를 한 번에 하나씩 맞바꾸어 욕심쟁이 풀이로 바꾼다. **욕심쟁이가 앞선다** 재주는 다른 관점을 취한다. 곧 가장 좋은 풀이를 고치는 대신, 욕심쟁이 풀이를 다른 아무 될 수 있는 풀이와 걸음마다 곧바로 견주어, 고름마다 욕심쟁이가 어떤 경쟁자만큼은 나아가 있음을 보인다. 이 귀납 불변량이 욕심쟁이 풀이가 전체에서 가장 좋다는 깔끔한 증명으로 이어진다.

## 직관

나란한 트랙에서 같은 때에 출발한 달리기 선수 둘을 그려 보라. 검문소마다 선수 G(욕심쟁이)가 늘 선수 O(다른 아무 전략)만큼은 앞서 있다면 G이 적어도 그만큼 일찍 들어온다. 가장 크게 하기 문제라면 적어도 그만큼의 값을 쌓는다. "앞선다" 불변량이 바로 이것을 담아낸다. 곧 욕심쟁이는 결코 뒤처지지 않는다.

이 재주는 고름에 자연스러운 차례가 있고 걸음마다 견줄 수 있는 나아감 잣대가 있을 때 가장 잘 듣는다.

## 엄밀한 틀

!!! note "욕심쟁이가 앞선다 틀"
    **Setup.** Let $G = (g_1, g_2, \ldots, g_k)$ be the greedy solution and $O = (o_1, o_2, \ldots, o_m)$ be any feasible solution, both sorted in the order the algorithm processes them.

    **Define a measure.** Choose a function $\mu$ that captures "progress" after $i$ steps. For activity selection, $\mu(i) = f(g_i)$ (finish time of the $i$-th selected activity).

    **Stays-ahead invariant.** For all $i \leq \min(k, m)$:

    $$
    \mu_G(i) \leq \mu_O(i) \quad \text{(for earliest-finish-time problems)}
    $$

    또는 그 문제의 목표에 알맞은 부등식.

    **$i$에 대한 귀납법으로 증명한다**:

    - *바탕 경우*($i = 1$): 욕심쟁이 규칙에서 따라 나온다.
    - *Inductive step*: assume $\mu_G(i-1) \leq \mu_O(i-1)$; show $\mu_G(i) \leq \mu_O(i)$.

    **Conclude:** Since $G$ stays ahead at every step, $G$ is at least as good as $O$ overall: $k \geq m$ (for maximization of count) or $\text{cost}(G) \leq \text{cost}(O)$ (for minimization).

## 보기 1: 활동 고르기

**Problem.** Select the maximum number of mutually compatible activities from $\{a_1, \ldots, a_n\}$ with start times $s_i$ and finish times $f_i$.

**욕심쟁이 규칙.** 늘 아직 안 고른 어울리는 활동 가운데 가장 일찍 끝나는 것을 고른다.

**정리.** 욕심쟁이 알고리즘은 크기가 가장 큰 어울리는 모음을 낸다.

**증명.**

Let $G = (g_1, g_2, \ldots, g_k)$ be the greedy solution, sorted by finish time. Let $O = (o_1, o_2, \ldots, o_m)$ be any maximum-size compatible set, also sorted by finish time.

**Stays-ahead invariant:** for all $1 \leq i \leq \min(k, m)$, we have $f(g_i) \leq f(o_i)$.

*Base case* ($i = 1$): The greedy algorithm picks the activity with the globally earliest finish time, so $f(g_1) \leq f(o_1)$.

*Inductive step:* Assume $f(g_{i-1}) \leq f(o_{i-1})$ for some $i \geq 2$. Activity $o_i$ is compatible with $o_{i-1}$, so:

$$
s(o_i) \geq f(o_{i-1}) \geq f(g_{i-1})
$$

This means $o_i$ is available (compatible with $g_{i-1}$) when the greedy algorithm makes its $i$-th choice. The greedy algorithm picks the available activity with the smallest finish time, so:

$$
f(g_i) \leq f(o_i)
$$

**Conclusion.** Suppose for contradiction that $k < m$. Then $o_{k+1}$ exists and satisfies $s(o_{k+1}) \geq f(o_k) \geq f(g_k)$, so $o_{k+1}$ is compatible with $g_k$. But then the greedy algorithm would have selected at least one more activity after $g_k$, contradicting $|G| = k$. Therefore $k \geq m$, and since $O$ is a maximum-size set, $k = m$. $\square$

## 보기 2: 최대 늦음 가장 작게 하기

**Problem.** Given $n$ jobs with processing times $p_i$ and deadlines $d_i$, schedule all jobs on a single machine (no idle time) to minimize the maximum lateness $L_{\max} = \max_i (C_i - d_i)$, where $C_i$ is the completion time of job $i$.

**Greedy rule.** Schedule jobs in order of increasing deadline: $d_1 \leq d_2 \leq \cdots \leq d_n$ (Earliest Deadline First, EDF).

**Theorem.** EDF minimizes $L_{\max}$.

??? example "욕심쟁이가 앞선다에 의한 증명"
    Let $G$ be the EDF schedule and $O$ be any other schedule. We show $L_{\max}(G) \leq L_{\max}(O)$.

    **살핌.** $G$에는 노는 때가 없으므로 $j$번째 일의 마침 시각은 다음과 같다:

    $$
    C_j^G = \sum_{i=1}^{j} p_{\sigma_G(i)}
    $$

    where $\sigma_G$ is the EDF permutation. The same formula holds for $O$ with permutation $\sigma_O$.

    **핵심 눈썰미.** 두 일정 모두 노는 때 없이 같은 일 모음을 다루므로, 어떤 일정에서든 $j$번째 일의 마침 시각은 그 일정의 자리바꿈에서 처음 $j$개 다루는 시간의 합과 같다.

    Since EDF processes jobs in deadline order, the job completing at position $j$ has the smallest possible deadline among jobs in positions $1, \ldots, n$. This means the lateness $C_j^G - d_{\sigma_G(j)}$ is minimized for the "worst-positioned" job.

    More precisely, any schedule with an inversion (job $a$ before job $b$ with $d_a > d_b$) can be improved by swapping $a$ and $b$, reducing the maximum lateness. Since EDF has no inversions, it is optimal. $\square$

## 앞선다가 통하는 까닭

이 재주의 힘은 **귀납을 더 세게 만드는 데** 있다. 곧 마지막 결과가 가장 좋다는 것만이 아니라 욕심쟁이 풀이가 가운데 걸음마다 앞선다는 것을 증명한다. 이 더 센 주장 덕분에 귀납법이 깔끔하게 통한다.

The measure $\mu$ must be chosen carefully. Good measures satisfy:

1. **Monotonicity**: $\mu_G(i) \leq \mu_G(i+1)$ (progress always advances).
2. **Comparability**: $\mu_G(i)$ and $\mu_O(i)$ measure the same quantity for the same step index.
3. **Terminal implication**: the stays-ahead invariant at $i = \min(k, m)$ implies $G$ is at least as good as $O$.

## 맞바꿈 논증과의 견줌

| 갈래 | 욕심쟁이가 앞선다 | 맞바꿈 논증 |
|--------|--------------------|--------------------|
| Proof structure | Induction on step $i$ | Transform optimal $\to$ greedy |
| 무엇을 견주나 | 욕심쟁이와 아무 풀이, 걸음마다 | 맞바꿈 지점에서 두 풀이 |
| 가장 센 때 | 자연스러운 나아감 잣대가 있을 때 | 맞바꿈 한 번으로 될 수 있음이 지켜질 때 |
| 보기 | 활동 고르기, 일정 짜기 | 허프먼, 쪼갤 수 있는 배낭 |
| Typical invariant | $f(g_i) \leq f(o_i)$ | $\|S'\| \geq \|S^*\|$ after swap |

두 재주는 욕심쟁이의 옳음을 증명하는 데 논리로는 같지만, 욕심쟁이가 정렬된 차례로 물건을 다루는 문제에서는 앞선다 쪽이 흔히 더 우아한 증명을 낸다.

## 흔히 빠지는 함정

!!! warning "앞선다 증명에서의 잘못"

    1. **잘못된 잣대.** 걸음마다의 마침 시각 대신 전체 값을 쓰거나 그 반대로 하는 것. 잣대는 걸음마다 견줄 수 있어야 한다.
    2. **어긋남 이끌기에서의 하나 차이.** 귀납법 끝에서 $k < m$인 경우를 따로 다루는 것을 잊는 것.
    3. **Assuming $k = m$.** The proof must establish this, not assume it. In activity selection, the stays-ahead invariant implies $k \geq m$ and feasibility gives $k \leq m$.

## 참고 문헌

- Kleinberg, J. & Tardos, E. (2006). *Algorithm Design*, 4.1절. Pearson.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), 16장. MIT Press.

## 연습문제

**연습문제 1.**
욕심쟁이가 앞선다에서 욕심쟁이 고름이 무엇인지 가려내고 왜 가장 좋은 풀이로 이어지는지 밝혀라.

??? success "연습문제 1 풀이"
    The greedy choice selects the locally optimal option at each step. For Greedy Stays Ahead, this choice satisfies the greedy choice property: there exists an optimal solution that includes this greedy selection. Combined with optimal substructure (the remaining subproblem after the greedy choice is also optimally solvable by the same strategy), the greedy algorithm produces a globally optimal solution. $\square$

---

**연습문제 2.**
욕심쟁이가 앞선다이 가장 좋은 아래 짜임을 갖는지 증명하거나 반증하여라.

??? success "연습문제 2 풀이"
    Optimal substructure means that an optimal solution to the problem contains optimal solutions to its subproblems. For Greedy Stays Ahead, after making the greedy choice, the remaining problem is a smaller instance of the same type. If the subproblem solution were not optimal, we could improve the overall solution by replacing it — contradicting overall optimality. Therefore optimal substructure holds. $\square$

---

**연습문제 3.**
욕심쟁이가 앞선다의 시간 복잡도는 무엇인가? 가장 값비싼 단계를 가려내어라.

??? success "연습문제 3 풀이"
    The time complexity depends on the sorting step (if required) and the greedy selection loop. Sorting typically dominates at $O(n \log n)$. The greedy loop processes each element once in $O(n)$. Total: $O(n \log n)$. If the input is pre-sorted, the algorithm runs in $O(n)$. $\square$

---

**연습문제 4.**
(욕심쟁이가 앞선다에서 쓴 것이 아닌) 다른 욕심쟁이 전략은 가장 좋은 풀이를 내지 못함을 보이는 반례를 들어라.

??? success "연습문제 4 풀이"
    Consider an alternative greedy criterion that does not align with the problem's structure. This alternative may select an element that blocks better future choices. The counterexample demonstrates that the wrong greedy criterion can produce a suboptimal result, highlighting why the specific greedy choice property must be proven for each problem. $\square$
