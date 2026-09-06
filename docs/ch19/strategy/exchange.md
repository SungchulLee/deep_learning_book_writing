# 맞바꿈 논증

고름마다 돌이킬 수 없이 못 박는 욕심쟁이 알고리즘이 가장 좋은 풀이를 냄을 어떻게 증명할까? **맞바꿈 논증**이 가장 널리 쓰이는 재주이다. 생각은 이렇다. 아무 가장 좋은 풀이를 가져와 한 번에 하나씩 맞바꾸며 결코 나빠지지 않게 욕심쟁이 풀이로 차츰 바꾼다. 모든 가장 좋은 풀이를 욕심쟁이의 내놓음에 맞게 빚어낼 수 있다면 그 내놓음도 가장 좋아야 한다.

## 핵심 생각

맞바꿈 논증은 어긋남 이끌기와 세우기를 아울러 돌아간다. 욕심쟁이 풀이가 하나뿐임을 증명하는 것이 아니라, 욕심쟁이 고름과 맞는 가장 좋은 풀이가 있음을 증명한다:

1. Start with any optimal solution $S^*$.
2. Find the first point where $S^*$ and the greedy solution $G$ differ.
3. Swap one element of $S^*$ to match $G$ at that point.
4. 그 맞바꿈이 될 수 있음을 지키고 목표를 나쁘게 하지 않음을 보인다.
5. Repeat until $S^* = G$.

맞바꿈마다 여전히 가장 좋은(적어도 나쁘지는 않은) 풀이가 나오면 $G$이 가장 좋다.

## 엄밀한 틀

!!! note "맞바꿈 논증 틀"
    **목표**: 욕심쟁이 알고리즘 $G$이 가장 좋은 풀이를 냄을 증명한다.

    **Step 1 (Setup).** Let $S^* = (s_1^*, s_2^*, \ldots, s_m^*)$ be any optimal solution. Let $G = (g_1, g_2, \ldots, g_k)$ be the greedy solution. Both are ordered by the algorithm's selection order.

    **Step 2 (Find first difference).** Let $i$ be the smallest index where $s_i^* \neq g_i$.

    **Step 3 (Exchange).** Construct $S' = (S^* \setminus \{s_i^*\}) \cup \{g_i\}$ (swap $s_i^*$ for $g_i$).

    **4단계(될 수 있음).** $S'$이 문제의 모든 제약을 채움을 보인다.

    **Step 5 (Quality).** Show $\text{cost}(S') \leq \text{cost}(S^*)$ (minimization) or $\text{value}(S') \geq \text{value}(S^*)$ (maximization).

    **6단계(매듭짓기).** $S'$은 가장 좋고 $G$과 고름 하나를 더 맞춘다. 온전히 맞을 때까지 되풀이한다.

## 보기 1: 활동 고르기

**Problem.** Select the maximum number of mutually compatible (non-overlapping) activities from a set $\{a_1, \ldots, a_n\}$, where each activity $a_i$ has start time $s_i$ and finish time $f_i$.

**욕심쟁이 규칙.** 늘 가장 일찍 끝나는 활동을 고른다.

**정리.** 욕심쟁이 알고리즘은 가장 좋은 풀이를 낸다.

??? example "맞바꿈 논증에 의한 증명"
    Let $S^* = \{a_{j_1}, a_{j_2}, \ldots, a_{j_k}\}$ be an optimal set of compatible activities, sorted by finish time. Let $a_1$ be the activity with the globally earliest finish time (the greedy first choice).

    **Case 1:** $a_{j_1} = a_1$. The optimal solution already includes the greedy choice.

    **Case 2:** $a_{j_1} \neq a_1$. Since $a_1$ has the earliest finish time among all activities, $f_1 \leq f_{j_1}$. Construct:

    $$
    S' = (S^* \setminus \{a_{j_1}\}) \cup \{a_1\}
    $$

    **Feasibility:** Activity $a_1$ finishes at time $f_1 \leq f_{j_1}$, so $a_1$ is compatible with $a_{j_2}$ (which starts at $s_{j_2} \geq f_{j_1} \geq f_1$). All other pairwise compatibilities in $S^*$ are unchanged.

    **Quality:** $|S'| = |S^*| = k$, so $S'$ is also a maximum-size compatible set.

    Therefore, $S'$ is optimal and includes $a_1$. By optimal substructure, the subproblem $\{a_i : s_i \geq f_1\}$ has an optimal solution that combines with $a_1$ to give an optimal overall solution. The greedy algorithm solves this subproblem recursively, so by induction it is correct. $\square$

## 보기 2: 쪼갤 수 있는 배낭

**문제.** 무게가 $w_i$이고 값이 $v_i$인 물건이 주어질 때, 물건을 쪼개어 담을 수 있는 담이 $W$의 배낭에서 전체 값을 가장 크게 하여라.

**욕심쟁이 규칙.** 물건을 값 대 무게 비 $r_i = v_i / w_i$의 내림차순으로 정렬한다. 비가 가장 큰 물건부터 욕심껏 배낭을 채운다.

**정리.** 욕심쟁이 알고리즘은 가장 좋은 풀이를 낸다.

??? example "맞바꿈 논증에 의한 증명"
    Without loss of generality, assume $r_1 \geq r_2 \geq \cdots \geq r_n$. Let $S^* = (x_1^*, x_2^*, \ldots, x_n^*)$ be any optimal solution, where $x_i^* \in [0, 1]$ is the fraction of item $i$ taken.

    Let the greedy solution be $G = (x_1^G, x_2^G, \ldots, x_n^G)$. In $G$, items are taken greedily: $x_1^G = \min(1, W/w_1)$, then fill remaining capacity with item 2, and so on.

    Suppose $S^* \neq G$. Let $j$ be the first index where $x_j^* \neq x_j^G$. Since the greedy algorithm takes as much of item $j$ as possible, $x_j^G > x_j^*$. Let $\delta = x_j^G - x_j^*$. The greedy solution takes $\delta$ more of item $j$.

    In $S^*$, the capacity freed by reducing item $j$ must be allocated to items $k > j$ (which have lower ratios). Construct $S'$ by increasing $x_j$ by some amount and decreasing later items by the same weight. Since $r_j \geq r_k$ for all $k > j$:

    $$
    \text{value}(S') - \text{value}(S^*) = \delta \cdot w_j \cdot r_j - \sum_{k>j} \Delta_k \cdot w_k \cdot r_k \geq 0
    $$

    because we replace lower-ratio capacity usage with higher-ratio usage. So $S'$ is at least as good and agrees with $G$ on one more item. Repeat. $\square$

## 보기 3: 허프먼 부호

**Problem.** Construct a prefix-free binary code minimizing the weighted path length $\sum_i f_i \cdot d_i$, where $f_i$ is the frequency of character $i$ and $d_i$ is its code length.

**욕심쟁이 규칙.** 잦기가 가장 낮은 글자 둘을 거듭 어울린다.

허프먼 부호의 맞바꿈 논증은, 어떤 가장 좋은 나무에서도 잦기가 가장 낮은 두 글자를 값을 늘리지 않고 가장 깊은 곳의 형제로 만들 수 있음을 보인다. 모음의 원소가 아니라 나무의 자리를 맞바꾸므로 더 손이 가는 맞바꿈이다.

## 짜임의 무늬

맞바꿈 논증 증명에는 흔한 무늬가 여럿 나타난다:

### 한 번 맞바꿈
원소 하나를 갈음하고 풀이가 나아지거나 그대로임을 보인다. 활동 고르기와 쪼갤 수 있는 배낭에 쓴다.

### 이웃 맞바꿈
차례에서 이웃한 원소 둘을 맞바꾸고 나아짐을 보인다. 일정 짜기 문제에 흔하다.

??? example "이웃 맞바꿈: 무게를 준 마침 시각 가장 작게 하기"
    Given jobs with processing times $p_i$ and weights $w_i$, schedule to minimize $\sum w_i C_i$ where $C_i$ is the completion time. The greedy rule is to sort by $w_i / p_i$ in decreasing order.

    어떤 일정에서 이웃한 일 $j$과 $k$을 보자. $w_j / p_j < w_k / p_k$이면 $k$이 먼저 오도록 맞바꾸면 무게를 준 전체 마침 시각이 줄어든다:

    $$
    w_j(p_j + p_k) + w_k \cdot p_k > w_k(p_k + p_j) + w_j \cdot p_j
    $$

    simplifies to $w_j \cdot p_k > w_k \cdot p_j$, i.e., $w_j / p_j < w_k / p_k$, which is our assumption. So any inversion increases cost, and the greedy order is optimal. $\square$

### 나무 맞바꿈
나무 짜임에서 자리를 맞바꾼다. 허프먼 부호 증명에 쓴다.

## 맞바꿈 논증이 어그러질 때

맞바꿈 논증은 맞바꿈마다 될 수 있음이 지켜져야 한다. 원소 하나를 맞바꾸는 것이 다른 제약에 줄줄이 영향을 주는 문제에서는 논증이 어렵거나 아예 안 된다:

- **0-1 배낭**: 물건을 맞바꾸면 담이 제약을 어길 수 있고, 맞바꿈 한 번으로 풀이를 기울 길이 없다.
- **그래프 색칠하기**: 꼭짓점 하나의 빛깔을 바꾸면 그래프 전체가 바뀌어야 할 수 있다.
- **떠돌이 장사꾼**: 여행에서 도시 하나를 빼고 다른 것을 넣으면 아예 다른 여행 짜임이 된다.

이런 경우에는 욕심쟁이 고름 성질이 성립하지 않고, 맞바꿈 논증도 마땅히 통하지 않는다.

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), 16장. MIT Press.
- Kleinberg, J. & Tardos, E. (2006). *Algorithm Design*, 4.1절. Pearson.

## 연습문제

**연습문제 1.**
욕심쟁이 알고리즘의 맞바꿈 논증 증명 재주를 설명하여라.

??? success "연습문제 1 풀이"
    The exchange argument proves a greedy algorithm is optimal by showing: (1) Start with any optimal solution $O$. (2) If $O$ differs from the greedy solution $G$, find a specific element where they differ. (3) Modify $O$ by "exchanging" that element to match $G$, creating solution $O'$. (4) Show $O'$ is still feasible and no worse than $O$. (5) Repeat until $O$ matches $G$, proving $G$ is optimal. The key is constructing the exchange that maintains or improves the objective. $\square$

---

**연습문제 2.**
일정 짜기 문제에서 일을 마감으로 정렬하면 최대 늦음이 가장 작아짐을 맞바꿈 논증으로 증명하여라.

??? success "연습문제 2 풀이"
    Suppose optimal schedule $O$ has an inversion: jobs $i, j$ where $d_i < d_j$ but $j$ is scheduled before $i$. Swap $i$ and $j$ in $O$. Job $j$ now finishes earlier (less lateness). Job $i$ now finishes later, but since $d_i < d_j$, $i$'s lateness with $j$'s old completion time is at most $j$'s old lateness. The maximum lateness does not increase. Repeat to eliminate all inversions, arriving at the sorted-by-deadline schedule. Therefore sorting by deadline is optimal. $\square$

---

**연습문제 3.**
맞바꿈 논증은 왜 맞바꿈마다 풀이가 나빠지지 않음을 보여야 하는가?

??? success "연습문제 3 풀이"
    If an exchange could worsen the solution, we cannot guarantee convergence to the greedy solution through a sequence of non-worsening swaps starting from an optimal solution. The exchange argument builds a chain: $O = O_0 \to O_1 \to \cdots \to G$ where each $O_i$ is at least as good as $O_{i-1}$. If any step worsened the solution, we could not conclude $G \geq O$ (in quality). The non-worsening property is what bridges the gap between "greedy is feasible" and "greedy is optimal." $\square$

---

**연습문제 4.**
맞바꿈 논증과 "욕심쟁이가 앞선다" 재주를 견주어라. 저마다 언제 쓰는 것이 더 자연스러운가?

??? success "연습문제 4 풀이"
    **Exchange argument**: Works by transforming any optimal solution into the greedy solution through local swaps. Natural for scheduling and assignment problems where solutions are permutations. **Greedy stays ahead**: Shows the greedy solution is at least as good as any other solution at every step (by induction). Natural for selection problems (like activity selection) where we build a solution incrementally. Both are valid for any greedy proof; the choice depends on which is easier to formalize for the specific problem. $\square$
