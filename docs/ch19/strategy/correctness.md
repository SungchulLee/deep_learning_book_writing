# 옳음 증명하기

욕심쟁이 알고리즘은 꾸미기가 속임수처럼 단순하다. 곧 걸음마다 가장 좋아 보이는 것을 고른다. 참된 어려움은 이 코앞만 보는 전략이 정말로 전체에서 가장 좋은 풀이를 내는지 증명하는 데 있다. 벨먼 식과 작은 문제 크기에 대한 귀납법에서 옳음이 자연스레 따라 나오는 동적 계획과 달리, 욕심쟁이 알고리즘은 걸음마다의 돌이킬 수 없는 고름이 결코 막다른 곳으로 이어지지 않음을 보이는 따로 된 논증이 필요하다. 이 쪽에서는 그 일반 얼거리와 주된 증명 재주 둘을 내놓는다.

## 욕심쟁이의 옳음이 어려운 까닭

어려움은 근본적인 어긋남에서 온다. 곧 욕심쟁이 알고리즘은 다른 길을 살펴보지 않고 고름을 못 박는다. 증명은 알고리즘이 그 다른 길을 결코 헤아리지 않는데도, 어떤 다른 고름도 더 나은 결과로 이어질 수 없었음을 보여야 한다.

활동 고르기에 쓸 수 있는 욕심쟁이 규칙 셋을 보자:

1. **가장 일찍 시작하는** 활동을 고른다.
2. **가장 짧은** 활동을 고른다.
3. **가장 일찍 끝나는** 활동을 고른다.

1번과 2번 규칙은 직관으로는 그럴듯하지만 틀렸음을 증명할 수 있다. 3번 규칙만이 가장 좋은 풀이를 낸다. 엄밀한 증명 없이는 옳은 욕심쟁이 전략과 그렇지 않은 것을 가려낼 수 없다.

## 두 부분 얼거리

욕심쟁이의 옳음 증명은 모두 성질 둘을 세운다:

!!! note "욕심쟁이가 옳으려면"

    1. **욕심쟁이 고름 성질**: 욕심쟁이 알고리즘의 첫 고름을 담은 가장 좋은 풀이가 있다.
    2. **가장 좋은 아래 짜임**: 욕심쟁이 고름을 한 뒤 남은 작은 문제에는, 그 고름과 아울러 본디 문제의 가장 좋은 풀이를 이루는 가장 좋은 풀이가 있다.

두 성질이 있으면 고름의 개수에 대한 귀납법으로 옳음이 따라 나온다:

- **바탕 경우**: 빈 문제에는 빈 풀이가 뻔히 가장 좋다.
- **귀납 단계**: 크기가 $< k$인 아무 작은 문제에도 욕심쟁이 알고리즘이 가장 좋은 풀이를 낸다고 하자. 크기 $k$인 문제에서 욕심쟁이 고름 성질이 첫 고름이 안전함을 보장하고, 가장 좋은 아래 짜임이 남은 작은 문제(크기 $< k$)가 귀납 가정으로 가장 좋게 풀림을 보장한다.

$$
\text{OPT}(\mathcal{P}) = \{g\} \cup \text{OPT}(\mathcal{P}')
$$

where $g$ is the greedy choice and $\mathcal{P}'$ is the residual subproblem.

## 증명 재주 1: 맞바꿈 논증

**맞바꿈 논증**은 어떤 가장 좋은 풀이든 목표를 나쁘게 하지 않고 욕심쟁이 고름과 맞는 것으로 바꿀 수 있음을 보여 욕심쟁이 고름 성질을 증명한다.

**일반 틀:**

1. Let $S^*$ be an arbitrary optimal solution.
2. If $S^*$ already includes the greedy choice $g$, we are done.
3. Otherwise, identify an element $x \in S^*$ that can be replaced by $g$.
4. Show that the modified solution $S' = (S^* \setminus \{x\}) \cup \{g\}$ is feasible.
5. Show that $\text{cost}(S') \leq \text{cost}(S^*)$ (for minimization) or $\text{value}(S') \geq \text{value}(S^*)$ (for maximization).
6. $S'$이 가장 좋고 $g$을 담고 있다고 매듭짓는다.

맞바꿈 논증은 가장 널리 쓰이는 재주이다. 욕심쟁이 고름과 그렇지 않은 고름 사이에, 될 수 있음을 지키는 자연스러운 "맞바꿈"이 있을 때 잘 듣는다.

??? example "맞바꿈 논증: 활동 고르기"
    **욕심쟁이 규칙**: 늘 가장 일찍 끝나는 활동을 고른다.

    Let $S^* = \{a_{j_1}, \ldots, a_{j_k}\}$ be optimal, sorted by finish time. Let $a_1$ be the activity with the globally earliest finish time. If $a_{j_1} = a_1$, done. Otherwise, $f_1 \leq f_{j_1}$, so replacing $a_{j_1}$ with $a_1$ preserves compatibility with $a_{j_2}, \ldots, a_{j_k}$. The resulting set has the same cardinality $k$, so it is optimal and contains $a_1$. $\square$

## 증명 재주 2: 욕심쟁이가 앞선다

**욕심쟁이가 앞선다** 재주는 알고리즘의 걸음마다 욕심쟁이 풀이가 다른 어떤 풀이의 그에 맞는 앞부분만큼은 좋음을 보인다.

**일반 틀:**

1. Let $G = (g_1, g_2, \ldots, g_k)$ be the greedy solution and $O = (o_1, o_2, \ldots, o_m)$ be any feasible solution, both ordered by the algorithm's processing order.
2. 나아감을 재는 잣대를 정한다(보기로 마침 시각, 부분 값).
3. 이 잣대로 볼 때 욕심쟁이 풀이의 $i$번째 고름이 $O$의 $i$번째 고름만큼은 좋음을 $i$에 대한 귀납법으로 증명한다.
4. Conclude that $k \geq m$ (for maximization of count) or that the greedy cost is no worse.

??? example "욕심쟁이가 앞선다: 활동 고르기"
    Let $G = (g_1, \ldots, g_k)$ be the greedy solution (sorted by finish time) and $O = (o_1, \ldots, o_m)$ be any optimal solution (sorted by finish time).

    **Claim**: for all $i \leq \min(k, m)$, the finish time of $g_i$ is at most the finish time of $o_i$: $f(g_i) \leq f(o_i)$.

    **$i$에 대한 귀납법 증명**:

    - *Base case* ($i = 1$): the greedy algorithm picks the activity with the earliest finish time, so $f(g_1) \leq f(o_1)$.
    - *Inductive step*: assume $f(g_{i-1}) \leq f(o_{i-1})$. Since $o_i$ starts after $o_{i-1}$ finishes, we have $s(o_i) \geq f(o_{i-1}) \geq f(g_{i-1})$. So $o_i$ is available when the greedy algorithm makes its $i$-th choice. The greedy algorithm picks the available activity with the earliest finish time, so $f(g_i) \leq f(o_i)$.

    Since $f(g_i) \leq f(o_i)$ for all $i$, the greedy solution is at least as long as any other: $k \geq m$. $\square$

## 알맞은 재주 고르기

| 잣대 | 맞바꿈 논증 | 욕심쟁이가 앞선다 |
|-----------|-------------------|--------------------|
| 알맞은 곳 | 맞바꿈 한 번이면 되는 문제 | 자연스러운 차례가 있는 문제 |
| 증명의 짜임 | 아무 가장 좋은 풀이를 고친다 | 걸음 번호에 대한 귀납법 |
| 흔한 곳 | 허프먼 부호, 쪼갤 수 있는 배낭 | 활동 고르기, 일정 짜기 |
| 어려운 점 | 알맞은 맞바꿈 찾기 | "앞섬" 잣대 정하기 |

두 재주는 힘이 같다. 곧 한쪽으로 한 옳음 증명은 다른 쪽으로도 다시 쓸 수 있다. 어느 쪽을 고르느냐는 보통 그 문제에서 어느 쪽이 더 깔끔한 논증이 되느냐의 문제이다.

## 흔히 빠지는 함정

!!! warning "피해야 할 잘못"

    1. **보기만 보고 옳다고 여기기.** 시험 경우 몇 개에 욕심쟁이 알고리즘을 돌려 맞는 답이 나오는 것을 본다고 증명이 되지는 않는다.
    2. **욕심쟁이와 어림짐작을 헷갈리기.** 욕심쟁이 어림짐작은 좋지만 가장 좋지는 않은 풀이를 낼 수 있다. 옳음 증명이 있는 것만이 엄밀한 뜻에서 참된 "욕심쟁이 알고리즘"이다.
    3. **가장 좋은 아래 짜임을 잊기.** 욕심쟁이 고름 성질만으로는 모자란다. 남은 작은 문제가 알맞은 짜임을 갖는지도 확인해야 한다.
    4. **잘못된 욕심쟁이 잣대.** 활동 고르기를 시작 시각, 길이, 부딪힘 수로 정렬하면 모두 어그러진다. 가장 이른 마침 시각만이 되며, 증명이 그 까닭을 드러낸다.

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), 16장. MIT Press.
- Kleinberg, J. & Tardos, E. (2006). *Algorithm Design*, 4장. Pearson.

## 연습문제

**연습문제 1.**
옳음 증명하기에서 욕심쟁이 고름이 무엇인지 가려내고 왜 가장 좋은 풀이로 이어지는지 밝혀라.

??? success "연습문제 1 풀이"
    The greedy choice selects the locally optimal option at each step. For Proving Correctness, this choice satisfies the greedy choice property: there exists an optimal solution that includes this greedy selection. Combined with optimal substructure (the remaining subproblem after the greedy choice is also optimally solvable by the same strategy), the greedy algorithm produces a globally optimal solution. $\square$

---

**연습문제 2.**
옳음 증명하기이 가장 좋은 아래 짜임을 갖는지 증명하거나 반증하여라.

??? success "연습문제 2 풀이"
    Optimal substructure means that an optimal solution to the problem contains optimal solutions to its subproblems. For Proving Correctness, after making the greedy choice, the remaining problem is a smaller instance of the same type. If the subproblem solution were not optimal, we could improve the overall solution by replacing it — contradicting overall optimality. Therefore optimal substructure holds. $\square$

---

**연습문제 3.**
옳음 증명하기의 시간 복잡도는 무엇인가? 가장 값비싼 단계를 가려내어라.

??? success "연습문제 3 풀이"
    The time complexity depends on the sorting step (if required) and the greedy selection loop. Sorting typically dominates at $O(n \log n)$. The greedy loop processes each element once in $O(n)$. Total: $O(n \log n)$. If the input is pre-sorted, the algorithm runs in $O(n)$. $\square$

---

**연습문제 4.**
(옳음 증명하기에서 쓴 것이 아닌) 다른 욕심쟁이 전략은 가장 좋은 풀이를 내지 못함을 보이는 반례를 들어라.

??? success "연습문제 4 풀이"
    Consider an alternative greedy criterion that does not align with the problem's structure. This alternative may select an element that blocks better future choices. The counterexample demonstrates that the wrong greedy criterion can produce a suboptimal result, highlighting why the specific greedy choice property must be proven for each problem. $\square$
