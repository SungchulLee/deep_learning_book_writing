# 구간 일정 짜기

시작 시각과 마침 시각을 갖는 활동(또는 일) 모음이 주어질 때 **구간 일정 짜기** 문제는 겹치지 않는 활동의 최대 개수를 묻는다. 늘 가장 일찍 끝나는 활동을 고르는 욕심쟁이 전략이 가장 좋으며, 그 옳음은 다른 어떤 전략도 더 낫지 못함을 보이는 "맞바꿈 논증"으로 증명할 수 있다.

## 문제 서술

Given $n$ activities $\{a_1, a_2, \dots, a_n\}$ where activity $a_i$ has start time $s_i$ and finish time $f_i$, find the largest subset $S$ of mutually compatible activities. Two activities $a_i$ and $a_j$ are **compatible** if their intervals do not overlap: $f_i \le s_j$ or $f_j \le s_i$.

## 욕심쟁이 알고리즘

**전략.** 활동을 마침 시각으로 정렬한다. 마지막으로 고른 활동의 마침 시각 이후에 시작하는 활동을 욕심껏 고른다.

**왜 가장 이른 마침 시각인가?** 될 수 있는 한 일찍 끝냄으로써 뒤이을 활동에 자리를 가장 많이 남긴다. 다른 전략(가장 이른 시작 시각, 가장 짧은 길이)은 간단한 반례로 어그러짐을 보일 수 있다.

## 옳음의 증명

!!! note "정리"
    가장 이른 마침 시각 욕심쟁이 알고리즘은 서로 어울리는 활동의 가장 큰 모음을 낸다.

??? example "증명(맞바꿈 논증)"
    Let $G = \{g_1, g_2, \dots, g_k\}$ be the greedy solution (sorted by finish time) and $O = \{o_1, o_2, \dots, o_m\}$ be an optimal solution with $m \ge k$. We show $k = m$.

    **Claim.** For each $i \le k$, $f(g_i) \le f(o_i)$ (greedy finishes at least as early at every step).

    *Base case:* $f(g_1) \le f(o_1)$ because greedy picks the earliest finish time.

    *Inductive step:* Assume $f(g_i) \le f(o_i)$. Then $s(o_{i+1}) \ge f(o_i) \ge f(g_i)$, so $o_{i+1}$ is available to greedy at step $i+1$. Greedy picks $g_{i+1}$ with the earliest finish time among all available activities, so $f(g_{i+1}) \le f(o_{i+1})$.

    Since the greedy solution stays ahead at every step, if $m > k$, then $o_{k+1}$ would be compatible with $g_k$ (because $s(o_{k+1}) \ge f(o_k) \ge f(g_k)$), contradicting the fact that greedy stopped. Therefore $m = k$. $\square$

## 구현

```python
"""
가장 일찍 마치는 것을 고르는 욕심쟁이 알고리즘으로 하는 구간 차례 짜기.

늘 가장 일찍 마치는 일을 골라 겹치지 않는 구간을
가장 많이 고른다.
"""

# === 욕심쟁이 구간 차례 짜기 ===

def interval_scheduling(
    activities: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """겹치지 않는 일의 가장 큰 모임을 찾는다.

    인수:
        activities: (시작, 마침) 짝의 목록.

    반환값:
        서로 어울리는 일의 가장 큰 부분 모임.
    """
    # 마치는 때로 정렬한다
    sorted_acts = sorted(activities, key=lambda x: x[1])

    selected = []
    last_finish = -1

    for start, finish in sorted_acts:
        if start >= last_finish:
            selected.append((start, finish))
            last_finish = finish

    return selected


# === 시연 ===

if __name__ == "__main__":
    activities = [
        (1, 4), (3, 5), (0, 6), (5, 7),
        (3, 9), (5, 9), (6, 10), (8, 11),
        (8, 12), (2, 14), (12, 16),
    ]
    result = interval_scheduling(activities)
    print(f"Total activities: {len(activities)}")
    print(f"Maximum compatible set: {len(result)} activities")
    for s, f in result:
        print(f"  [{s}, {f})")
```

**출력:**

```
Total activities: 11
Maximum compatible set: 4 activities
  [1, 4)
  [5, 7)
  [8, 11)
  [12, 16)
```

활동 11개 가운데 욕심쟁이 알고리즘은 겹치지 않는 4개를 고른다. 고른 활동마다 앞 것이 끝난 뒤에 시작한다.

## 다른 전략이 어그러지는 까닭

| 전략 | 반례 |
|----------|---------------|
| 가장 이른 시작 시각 | 일찍 시작하는 긴 활동이 짧은 활동 여럿을 막는다 |
| 가장 짧은 길이 | 가운데의 짧은 활동이 겹치지 않는 둘과 겹친다 |
| 부딪힘이 가장 적은 것 | 부딪힘이 적은 활동도 가장 좋은 고름을 막을 수 있다 |

## 복잡도

| 항목 | 비용 |
|--------|:----:|
| Time   | $O(n \log n)$ |
| 공간 | $O(n)$ |

Sorting dominates at $O(n \log n)$. The selection pass is a single $O(n)$ scan.

## 무게 있는 변종

활동마다 무게(이익) $w_i$이 있으면 목표가 개수가 아니라 전체 무게를 가장 크게 하는 것이 된다. 욕심쟁이 방식은 더는 통하지 않고 동적 계획이 필요하다:

$$
\text{dp}[j] = \max\bigl(w_j + \text{dp}[p(j)],\; \text{dp}[j-1]\bigr)
$$

여기서 $p(j)$은 $j$과 어울리는 가장 늦은 활동이다(마침 시각으로 정렬한 뒤 이분 찾기로 찾는다).

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to Algorithms* (4th ed.), 16장: Greedy Algorithms.
- Kleinberg, J., & Tardos, E. (2006). *Algorithm Design*. Pearson. 4장: Greedy Algorithms.

## 연습문제

**연습문제 1.**
구간 일정 짜기에서 욕심쟁이 고름이 무엇인지 가려내고 왜 가장 좋은 풀이로 이어지는지 밝혀라.

??? success "연습문제 1 풀이"
    욕심쟁이 고름은 걸음마다 그 자리에서 가장 좋은 것을 고른다. Interval Scheduling에서는 이 고름이 욕심쟁이 고름 성질을 채운다. 곧 이 욕심쟁이 고름을 담은 가장 좋은 풀이가 있다. 여기에 가장 좋은 아래 짜임(욕심쟁이 고름 뒤 남은 아래 문제도 같은 전략으로 가장 좋게 풀린다)을 더하면 욕심쟁이 알고리즘이 두루 가장 좋은 풀이를 내놓는다. $\square$

---

**연습문제 2.**
구간 일정 짜기이 가장 좋은 아래 짜임을 갖는지 증명하거나 반증하여라.

??? success "연습문제 2 풀이"
    가장 좋은 아래 짜임이란 문제의 가장 좋은 풀이가 그 아래 문제의 가장 좋은 풀이를 담는다는 뜻이다. Interval Scheduling에서는 욕심쟁이 고름을 하고 나면 남은 문제가 같은 갈래의 더 작은 사례가 된다. 아래 문제의 풀이가 가장 좋지 않다면 그것을 갈음해 온 풀이를 더 낫게 할 수 있는데, 이는 온 풀이가 가장 좋다는 것과 어긋난다. 그러므로 가장 좋은 아래 짜임이 이루어진다. $\square$

---

**연습문제 3.**
구간 일정 짜기의 시간 복잡도는 무엇인가? 가장 값비싼 단계를 가려내어라.

??? success "연습문제 3 풀이"
    시간 복잡도는 (필요하다면) 줄 세우는 걸음과 욕심쟁이 고름 되돌이에 달렸다. 흔히 줄 세우기가 $O(n \log n)$으로 가장 크다. 욕심쟁이 되돌이는 원소마다 한 번씩 다루어 $O(n)$이다. 모두 $O(n \log n)$이다. 들임이 미리 줄 세워져 있으면 알고리즘은 $O(n)$에 돈다. $\square$

---

**연습문제 4.**
(구간 일정 짜기에서 쓴 것이 아닌) 다른 욕심쟁이 전략은 가장 좋은 풀이를 내지 못함을 보이는 반례를 들어라.

??? success "연습문제 4 풀이"
    문제의 짜임과 맞지 않는 다른 욕심쟁이 잣대를 생각해 보자. 이 잣대는 뒤에 더 좋은 고름을 막는 원소를 고를 수 있다. 어긋나는 보기는 잘못된 욕심쟁이 잣대가 가장 좋지 않은 결과를 낼 수 있음을 보이며, 문제마다 그에 맞는 욕심쟁이 고름 성질을 반드시 증명해야 하는 까닭을 드러낸다. $\square$
