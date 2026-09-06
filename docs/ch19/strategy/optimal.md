# 가장 좋은 아래 짜임

욕심쟁이 알고리즘은 그 자리에서 가장 좋은 고름을 하나 하고 같은 문제의 더 작은 판을 푸는 식으로 돌아간다. 이 되돌이 쪼갬이 전체에서 가장 좋은 풀이를 내려면 문제가 **가장 좋은 아래 짜임**을 지녀야 한다. 곧 본디 문제의 가장 좋은 풀이가 그 안에 작은 문제의 가장 좋은 풀이를 담고 있어야 한다. 이 성질이 없으면 아무리 영리한 욕심쟁이 고름이라도, 그 고름과 아울렀을 때 전체 최적에 못 미치는 작은 문제를 남길 수 있다.

## 직관

가장 좋은 아래 짜임은 "쌓아 올리는" 성질이다. 곧 전체의 훌륭함이 부분의 훌륭함에서 지어진다고 말한다. 작은 문제마다 가장 좋게 풀고 그 결과를 올바로 아우르면 전체에서 가장 좋은 풀이를 얻는다.

강의실에 활동 일정을 짜는 일을 보자. 첫 활동 $a_1$(가장 일찍 끝나는 것)을 고른 뒤 남은 작은 문제는, $a_1$이 끝난 뒤 시작하며 서로 겹치지 않는 활동을 가장 많이 고르는 것이다. 전체에서 가장 좋은 풀이가 $a_1$을 담고 있다면 그 풀이의 남은 활동도 남은 작은 문제의 가장 좋은 풀이여야 한다. 그렇지 않다면 더 나은 모음을 끼워 넣어 온 풀이를 낫게 할 수 있어 가장 좋음에 어긋난다.

## 엄밀한 정의

!!! note "욕심쟁이 문제의 가장 좋은 아래 짜임"
    A problem $\mathcal{P}$ exhibits **optimal substructure** if an optimal solution to $\mathcal{P}$ can be constructed from the greedy choice $g$ combined with an optimal solution to the subproblem $\mathcal{P}'$ that remains after making choice $g$:

    $$
    \text{OPT}(\mathcal{P}) = \{g\} \cup \text{OPT}(\mathcal{P}')
    $$

이 되돌이 쪼갬 덕분에 욕심쟁이 알고리즘은 남은 문제를 가장 좋게 풀면 전체에서도 가장 좋은 결과가 나오리라 믿고 한 번에 하나씩 고를 수 있다.

## 가장 좋은 아래 짜임 증명하기

표준 증명 재주는 **오려 붙이기 논증**이다:

1. **Assume** $S^*$ is an optimal solution to $\mathcal{P}$ that includes the greedy choice $g$.
2. **Define** $S^* \setminus \{g\}$ as a candidate solution to the subproblem $\mathcal{P}'$.
3. **Suppose for contradiction** that $S^* \setminus \{g\}$ is not optimal for $\mathcal{P}'$. Then there exists a strictly better solution $T'$ for $\mathcal{P}'$.
4. **Paste**: form $S' = \{g\} \cup T'$. Since $T'$ is better than $S^* \setminus \{g\}$ for $\mathcal{P}'$, the combined solution $S'$ is better than $S^*$ for $\mathcal{P}$.
5. **Contradiction**: this contradicts the optimality of $S^*$.

Therefore, $S^* \setminus \{g\}$ must be optimal for $\mathcal{P}'$.

## 보기: 활동 고르기

**문제.** 시작 시각 $s_i$과 마침 시각 $f_i$을 갖는 활동 $n$개가 주어질 때 서로 어울리는(겹치지 않는) 활동을 가장 많이 골라라.

**욕심쟁이 고름.** 마침 시각 $f_1$이 가장 작은 활동 $a_1$을 고른다.

**Subproblem.** Let $\mathcal{P}' = \{a_i : s_i \geq f_1\}$ be the set of activities that start after $a_1$ finishes.

**Optimal substructure claim.** If $S^* = \{a_1\} \cup R$ is an optimal solution containing $a_1$, then $R$ is an optimal solution to $\mathcal{P}'$.

??? example "오려 붙이기 증명"
    **Proof.** Suppose $R$ is not optimal for $\mathcal{P}'$. Then there exists a compatible set $R'$ for $\mathcal{P}'$ with $|R'| > |R|$. Since every activity in $R'$ starts after $f_1$, the set $\{a_1\} \cup R'$ is a compatible set for $\mathcal{P}$ with $|\{a_1\} \cup R'| = 1 + |R'| > 1 + |R| = |S^*|$. This contradicts the optimality of $S^*$, so $R$ must be optimal for $\mathcal{P}'$. $\square$

## 보기: 쪼갤 수 있는 배낭

**문제.** 무게가 $w_i$이고 값이 $v_i$인 물건 $n$개와 담이 $W$의 배낭이 주어질 때, 물건을 쪼개어 담아 전체 값을 가장 크게 하여라.

**욕심쟁이 고름.** 값 대 무게 비 $v_i / w_i$가 가장 큰 물건을 될 수 있는 한 많이 담는다.

**Optimal substructure.** After filling the knapsack with the greedy choice (fully or partially taking the best-ratio item), the remaining capacity $W' = W - \min(w_1, W)$ defines a subproblem. An optimal solution to the original problem restricted to the greedy choice decomposes as:

$$
\text{OPT}(W) = v_{\text{greedy}} + \text{OPT}(W')
$$

where $v_{\text{greedy}}$ is the value gained from the greedy choice.

## 동적 계획과의 대비

욕심쟁이 알고리즘과 동적 계획 모두 가장 좋은 아래 짜임에 기대지만, 작은 문제를 살펴보는 방식이 다르다:

| 갈래 | 욕심쟁이 | 동적 계획 |
|--------|--------|---------------------|
| 헤아리는 작은 문제 수 | 하나(욕심쟁이 고름 뒤) | 가능한 모든 고름 |
| 작은 문제의 얽힘 | 사슬 하나 | 겹치는 유향 비순환 그래프 |
| 옳음 증명 | 욕심쟁이 고름 + 가장 좋은 아래 짜임 | 벨먼 식 + 귀납법 |

동적 계획에서는 알고리즘이 가능한 첫 고름을 모두 헤아리고 가장 좋은 것을 골라야 한다. 욕심쟁이 알고리즘에서는 욕심쟁이 고름 성질이 고름 하나만 헤아려도 됨을 보장한다.

## 흔한 함정

흔한 잘못은 가장 좋은 아래 짜임만으로 욕심쟁이 방식이 뒷받침된다고 여기는 것이다. 그렇지 않다. **욕심쟁이 고름 성질**도 있어야 한다. 그것이 그 자리에서 가장 좋은 고름이 어떤 전체 최적 풀이의 한 몫임을 보장한다. 0-1 배낭 문제는 가장 좋은 아래 짜임을 갖지만(동적 계획으로 풀 수 있다) 욕심쟁이 고름 성질이 없어 욕심쟁이가 어그러진다.

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), 16장. MIT Press.
- Kleinberg, J. & Tardos, E. (2006). *Algorithm Design*, 4장. Pearson.

## 연습문제

**연습문제 1.**
가장 좋은 아래 짜임에서 욕심쟁이 고름이 무엇인지 가려내고 왜 가장 좋은 풀이로 이어지는지 밝혀라.

??? success "연습문제 1 풀이"
    The greedy choice selects the locally optimal option at each step. For Optimal Substructure, this choice satisfies the greedy choice property: there exists an optimal solution that includes this greedy selection. Combined with optimal substructure (the remaining subproblem after the greedy choice is also optimally solvable by the same strategy), the greedy algorithm produces a globally optimal solution. $\square$

---

**연습문제 2.**
가장 좋은 아래 짜임이 가장 좋은 아래 짜임을 갖는지 증명하거나 반증하여라.

??? success "연습문제 2 풀이"
    Optimal substructure means that an optimal solution to the problem contains optimal solutions to its subproblems. For Optimal Substructure, after making the greedy choice, the remaining problem is a smaller instance of the same type. If the subproblem solution were not optimal, we could improve the overall solution by replacing it — contradicting overall optimality. Therefore optimal substructure holds. $\square$

---

**연습문제 3.**
가장 좋은 아래 짜임의 시간 복잡도는 무엇인가? 가장 값비싼 단계를 가려내어라.

??? success "연습문제 3 풀이"
    The time complexity depends on the sorting step (if required) and the greedy selection loop. Sorting typically dominates at $O(n \log n)$. The greedy loop processes each element once in $O(n)$. Total: $O(n \log n)$. If the input is pre-sorted, the algorithm runs in $O(n)$. $\square$

---

**연습문제 4.**
(가장 좋은 아래 짜임에서 쓴 것이 아닌) 다른 욕심쟁이 전략은 가장 좋은 풀이를 내지 못함을 보이는 반례를 들어라.

??? success "연습문제 4 풀이"
    Consider an alternative greedy criterion that does not align with the problem's structure. This alternative may select an element that blocks better future choices. The counterexample demonstrates that the wrong greedy criterion can produce a suboptimal result, highlighting why the specific greedy choice property must be proven for each problem. $\square$
