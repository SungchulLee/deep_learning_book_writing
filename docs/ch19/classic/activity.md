# 활동 고르기

활동 고르기 문제는 욕심쟁이 알고리즘 꾸미기의 대표 보기이다. 함께 쓰는 자원(강의실, 회의실, 셈틀 머리)을 저마다 홀로 써야 하는 활동 모음이 주어질 때, 겹치지 않는 활동을 될 수 있는 한 많이 고르는 것이 목표이다. 이 문제는 일정 짜기, 자원 나누기, 컴파일러 다듬기에 나타나며, 가장 이른 마침 시각으로 고르는 욕심쟁이 풀이는 단순하면서도 가장 좋음을 증명할 수 있다.

## 문제 서술

**Input.** A set $S = \{a_1, a_2, \ldots, a_n\}$ of $n$ activities. Each activity $a_i$ has a start time $s_i$ and a finish time $f_i$, with $s_i < f_i$.

**Compatibility.** Two activities $a_i$ and $a_j$ are **compatible** if their intervals do not overlap: $f_i \leq s_j$ or $f_j \leq s_i$.

**Goal.** Find a maximum-size subset $A \subseteq S$ of mutually compatible activities.

## 왜 가장 이른 마침 시각인가?

자연스러운 욕심쟁이 전략이 여럿 있다:

| 전략 | 욕심쟁이 규칙 | 가장 좋은가? |
|----------|-------------|----------|
| Earliest start time | Pick $\min(s_i)$ | No |
| Shortest duration | Pick $\min(f_i - s_i)$ | No |
| 부딪힘이 가장 적은 것 | 다른 것과 가장 적게 겹치는 활동을 고른다 | 아니다 |
| **Earliest finish time** | **Pick $\min(f_i)$** | **Yes** |

가장 이른 마침 시각 전략이 통하는 것은 뒤이을 활동에 남는 시간을 가장 많이 남기기 때문이다. 될 수 있는 한 일찍 끝냄으로써 그 뒤에 고를 수 있는 어울리는 활동의 수를 가장 크게 한다.

## 알고리즘

!!! note "욕심쟁이 활동 고르기"

    1. Sort activities by finish time: $f_1 \leq f_2 \leq \cdots \leq f_n$.
    2. $a_1$(가장 일찍 끝나는 활동)을 고른다.
    3. For $i = 2, \ldots, n$: if $s_i \geq f_{\text{last}}$ (where $f_{\text{last}}$ is the finish time of the most recently selected activity), select $a_i$.

## 풀이 예제

활동 여섯을 보자:

| 활동 | $s_i$ | $f_i$ |
|----------|-------|-------|
| $a_1$    | 1     | 4     |
| $a_2$    | 3     | 5     |
| $a_3$    | 0     | 6     |
| $a_4$    | 5     | 7     |
| $a_5$    | 3     | 9     |
| $a_6$    | 5     | 9     |

**마침 시각으로 정렬:** $a_1, a_2, a_3, a_4, a_5, a_6$.

**욕심쟁이 실행:**

1. Select $a_1$ (finishes at 4). Set $f_{\text{last}} = 4$.
2. $a_2$: $s_2 = 3 < 4$. 건너뜀(겹침).
3. $a_3$: $s_3 = 0 < 4$. 건너뜀(겹침).
4. $a_4$: $s_4 = 5 \geq 4$. Select. Set $f_{\text{last}} = 7$.
5. $a_5$: $s_5 = 3 < 7$. 건너뜀.
6. $a_6$: $s_6 = 5 < 7$. 건너뜀.

**Result:** $\{a_1, a_4\}$ with 2 activities. This is maximum --- no set of 3 mutually compatible activities exists.

## 옳음의 증명

**정리.** 욕심쟁이 알고리즘은 서로 어울리는 활동의 가장 큰 모음을 고른다.

증명은 욕심쟁이 고름 성질과 가장 좋은 아래 짜임을 아우른다.

**욕심쟁이 고름 성질.** $a_1$(가장 일찍 끝나는 활동)을 담은 가장 좋은 풀이가 있다.

*Proof.* Let $S^* = \{a_{j_1}, a_{j_2}, \ldots, a_{j_k}\}$ be an optimal solution sorted by finish time. If $a_{j_1} = a_1$, done. Otherwise, $f_1 \leq f_{j_1}$, so replacing $a_{j_1}$ with $a_1$ preserves compatibility with all subsequent activities. The resulting set has the same cardinality $k$. $\square$

**Optimal substructure.** If an optimal solution contains $a_1$, then the remaining activities $\{a_1\} \cup R$ have $R$ optimal for $S' = \{a_i \in S : s_i \geq f_1\}$.

*Proof.* Cut-and-paste: if $R$ is not optimal for $S'$, a better $R'$ would yield $\{a_1\} \cup R'$ with $|R'| > |R|$, contradicting $|S^*| = k$. $\square$

**매듭.** 활동의 개수에 대한 귀납법으로, 욕심쟁이 알고리즘은 가장 큰 모음을 고른다.

## 파이썬 구현

```python
"""
가장 일찍 마치는 것을 고르는 욕심쟁이 전략을 쓴 일 고르기.

시작과 마침의 때가 주어진 일에서 서로 어울리는(겹치지 않는)
일을 가장 많이 고른다.
"""


# === 욕심쟁이 일 고르기 ===

def activity_selection(activities):
    """겹치지 않는 일을 가장 많이 고른다.

    인수:
        activities: (시작, 마침) 짝의 목록

    반환값:
        고른 (시작, 마침) 짝의 목록
    """
    # 마치는 때로 정렬한다
    sorted_acts = sorted(activities, key=lambda x: x[1])

    selected = [sorted_acts[0]]
    last_finish = sorted_acts[0][1]

    for start, finish in sorted_acts[1:]:
        if start >= last_finish:
            selected.append((start, finish))
            last_finish = finish

    return selected


# === 되돌이 판 ===

def activity_selection_recursive(activities, k=0):
    """되돌이 욕심쟁이 일 고르기.

    인수:
        activities: 마치는 때로 정렬한 (시작, 마침) 짝의 목록
        k: 마지막으로 고른 일이 마치는 때

    반환값:
        고른 (시작, 마침) 짝의 목록
    """
    # 처음으로 어울리는 일을 찾는다
    for i, (start, finish) in enumerate(activities):
        if start >= k:
            return [(start, finish)] + activity_selection_recursive(
                activities[i + 1:], finish
            )
    return []


if __name__ == "__main__":
    # 위에서 풀어 본 보기
    activities = [(1, 4), (3, 5), (0, 6), (5, 7), (3, 9), (5, 9)]

    result = activity_selection(activities)
    print(f"Activities: {activities}")
    print(f"Selected:   {result}")
    print(f"Count:      {len(result)}")

    # 되돌이 판
    sorted_acts = sorted(activities, key=lambda x: x[1])
    result_rec = activity_selection_recursive(sorted_acts)
    print(f"Recursive:  {result_rec}")
```

**출력:**
```
Activities: [(1, 4), (3, 5), (0, 6), (5, 7), (3, 9), (5, 9)]
Selected:   [(1, 4), (5, 7)]
Count:      2
Recursive:  [(1, 4), (5, 7)]
```

## 복잡도 분석

**시간 복잡도.**

- Sorting: $O(n \log n)$.
- 한 번 훑기: $O(n)$.
- **Total:** $O(n \log n)$.

활동이 마침 시각으로 미리 정렬돼 있으면 알고리즘은 $O(n)$에 돈다.

**공간 복잡도.** 내놓음에 $O(n)$(고른 활동의 수만 세면 딸린 공간은 $O(1)$).

## 무게 있는 활동 고르기

활동 $a_i$마다 무게(값) $v_i$이 딸리면 목표가 개수가 아니라 전체 무게를 가장 크게 하는 것으로 바뀐다. 욕심쟁이 방식은 더는 통하지 않고 이 변종에는 동적 계획이 필요하다:

$$
\text{OPT}(j) = \max\bigl(\text{OPT}(j-1),\; v_j + \text{OPT}(p(j))\bigr)
$$

여기서 $p(j)$은 $a_i$이 $a_j$과 어울리는, $i < j$인 가장 큰 번호이다.

이는 중요한 가르침을 보여 준다. 곧 목표가 조금(개수에서 무게 합으로) 바뀌면 욕심쟁이 고름 성질이 무너질 수 있다.

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), 16.1절. MIT Press.
- Kleinberg, J. & Tardos, E. (2006). *Algorithm Design*, 4.1절. Pearson.

## 연습문제

**연습문제 1.**
구간이 $(1,4), (3,5), (0,6), (5,7), (3,9), (5,9), (6,10), (8,11), (8,12), (2,14), (12,16)$인 활동에 활동 고르기 알고리즘을 써라. 어떤 활동이 골라지는가?

??? success "연습문제 1 풀이"
    Sort by finish time: $(1,4), (3,5), (0,6), (5,7), (3,9), (5,9), (6,10), (8,11), (8,12), (2,14), (12,16)$. Select $(1,4)$ (first to finish). Skip $(3,5)$ and $(0,6)$ (overlap). Select $(5,7)$. Skip $(3,9), (5,9), (6,10)$. Select $(8,11)$. Skip $(8,12), (2,14)$. Select $(12,16)$. Selected: $\{(1,4), (5,7), (8,11), (12,16)\}$ — 4 activities. $\square$

---

**연습문제 2.**
가장 일찍 끝나는 활동을 고르는 것이 가장 좋은 풀이로 이어지는 욕심쟁이 고름임을 증명하여라.

??? success "연습문제 2 풀이"
    Let $a_1$ be the activity with the earliest finish time. Suppose optimal solution $O$ does not include $a_1$. Let $a_k$ be the first activity in $O$ (earliest finish time in $O$). Since $a_1$ finishes no later than $a_k$, replacing $a_k$ with $a_1$ in $O$ yields a solution $O'$ that is still feasible (no new overlaps, since $a_1$ finishes earlier) and has the same size. Therefore an optimal solution including $a_1$ exists. By induction on the remaining subproblem (activities starting after $a_1$ finishes), the greedy strategy is optimal. $\square$

---

**연습문제 3.**
길이가 가장 짧은 활동을 고르면 왜 늘 가장 좋은 풀이가 되지 않는가?

??? success "연습문제 3 풀이"
    Counterexample: activities $(0, 3), (2, 5), (4, 7)$. The shortest duration is $(2,5)$ with length 3. Selecting it conflicts with both others, giving only 1 activity. But selecting $(0,3)$ and $(4,7)$ gives 2 non-overlapping activities. The shortest-duration heuristic fails because it does not account for how an activity's time slot blocks other activities. Earliest finish time correctly minimizes blocking. $\square$

---

**연습문제 4.**
활동 고르기 문제를 무게 있는 경우로 넓혀라. 곧 활동마다 값이 있고 전체 값을 가장 크게 하려 한다. 욕심쟁이 방식을 여전히 쓸 수 있는가?

??? success "연습문제 4 풀이"
    The greedy approach (by finish time) does not work for weighted activity selection. Counterexample: activities $(0,3, \text{value}=1)$ and $(1,2, \text{value}=100)$. Greedy selects the first (finishes earlier), getting value 1, but the optimal is the second with value 100. The weighted problem requires dynamic programming: sort by finish time, and for each activity, compute the best value by either including it (add its value to the best value of compatible earlier activities) or excluding it. This runs in $O(n \log n)$. $\square$
