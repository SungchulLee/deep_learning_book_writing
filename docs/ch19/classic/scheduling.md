# 일 일정 짜기

공장, 운영 체제, 기획 살림 같은 실제 여러 자리에서, 저마다 마감이 있는 일 모음을 기계 하나로 다뤄야 한다. 마감을 놓치면 벌이 따르므로 일정 짜개는 가장 나쁜 경우의 늦음을 가장 작게 하도록 일을 늘어놓고자 한다. 욕심쟁이 풀이인 **가장 이른 마감 먼저(EDF)**는 그저 마감 차례로 일을 다룬다. 단순한데도 EDF는 최대 늦음을 가장 작게 하는 데 가장 좋음을 증명할 수 있고, 그 증명이 맞바꿈 논증을 아름답게 보여 준다.

## 문제 진술: 최대 늦음 가장 작게 하기

**Input.** A set of $n$ jobs $\{1, 2, \ldots, n\}$. Job $i$ has:

- 다루는 시간 $p_i > 0$(끝내는 데 드는 시간).
- 마감 $d_i$(이때까지 끝나면 좋은 시각).

**제약.** 기계 하나가 한 번에 일 하나를 다룬다. 모든 일은 때 0에 쓸 수 있다. 가로채기가 없다(한 번 시작한 일은 끝까지 돈다). 노는 때가 없다.

**Schedule.** A permutation $\sigma$ of $\{1, 2, \ldots, n\}$. Job $\sigma(j)$ is the $j$-th job processed. Its completion time is:

$$
C_{\sigma(j)} = \sum_{k=1}^{j} p_{\sigma(k)}
$$

**늦음.** 일 $i$의 늦음은 $L_i = C_i - d_i$이다. 양수이면 그 일이 마감 뒤에 끝난다는 뜻이다.

**목표.** **최대 늦음**을 가장 작게 한다:

$$
L_{\max} = \max_{1 \leq i \leq n} (C_i - d_i)
$$

## 욕심쟁이 알고리즘: 가장 이른 마감 먼저

!!! note "EDF 일정 짜기"

    1. Sort jobs by deadline: $d_1 \leq d_2 \leq \cdots \leq d_n$.
    2. 노는 때 없이 이 차례로 일을 다룬다.
    3. The $j$-th job completes at time $C_j = \sum_{k=1}^{j} p_k$.

이 알고리즘은 차례를 정할 때 다루는 시간을 아예 헤아리지 않는다. 오직 마감만 본다.

## 풀이 예제

일 넷을 보자:

| 일 | $p_i$ | $d_i$ |
|-----|--------|--------|
| 1   | 3      | 6      |
| 2   | 2      | 8      |
| 3   | 1      | 9      |
| 4   | 4      | 9      |

**EDF 차례**(마감으로 정렬): $1, 2, 3, 4$.

| 자리 | 일 | $C_i$ | $d_i$ | $L_i = C_i - d_i$ |
|----------|-----|--------|--------|--------------------|
| 1        | 1   | 3      | 6      | $-3$               |
| 2        | 2   | 5      | 8      | $-3$               |
| 3        | 3   | 6      | 9      | $-3$               |
| 4        | 4   | 10     | 9      | $1$                |

$$
L_{\max} = 1
$$

**다른 차례** $2, 1, 3, 4$:

| 자리 | 일 | $C_i$ | $d_i$ | $L_i$ |
|----------|-----|--------|--------|--------|
| 1        | 2   | 2      | 8      | $-6$   |
| 2        | 1   | 5      | 6      | $-1$   |
| 3        | 3   | 6      | 9      | $-3$   |
| 4        | 4   | 10     | 9      | $1$    |

This also gives $L_{\max} = 1$, which matches EDF. But no schedule achieves $L_{\max} < 1$, since the total processing time is 10 and the latest deadline is 9.

## 옳음의 증명

**Theorem.** EDF minimizes the maximum lateness $L_{\max}$.

The proof uses the exchange argument, showing that any **inversion** in the schedule can be removed without increasing $L_{\max}$.

**정의.** 일정에서 **뒤바뀜**이란 일 $i$이 일 $j$보다 앞에 놓였는데 $d_i > d_j$인 이웃한 일 짝 $(i, j)$이다.

**주장 1.** 노는 때가 없는 가장 좋은 일정이 있다.

*Proof.* Removing idle time shifts jobs earlier, which can only decrease lateness. $\square$

**주장 2.** 뒤바뀜이 없는 가장 좋은 일정이 있다.

??? example "맞바꿈에 의한 증명"
    Suppose schedule $\sigma$ has an inversion: job $i$ immediately precedes job $j$ with $d_i > d_j$. Let $\sigma'$ be the schedule obtained by swapping $i$ and $j$.

    맞바꾸기 앞에 두 일 모두 같은 때 $t$에 시작한다:

    - In $\sigma$: $C_i = t + p_i$, $C_j = t + p_i + p_j$
    - In $\sigma'$: $C_j' = t + p_j$, $C_i' = t + p_j + p_i$

    다른 모든 일의 마침 시각은 그대로이다.

    **일 $j$이 나아진다:** $C_j' = t + p_j < t + p_i + p_j = C_j$이므로 $L_j' < L_j$이다.

    **일 $i$의 새 늦음:** $L_i' = t + p_i + p_j - d_i$.

    **핵심 견줌:** 맞바꾸기 앞에는 $L_j = t + p_i + p_j - d_j$이다. 맞바꾼 뒤에는 $L_i' = t + p_i + p_j - d_i$이다. $d_i > d_j$이므로 $L_i' < L_j$이다.

    따라서 다음이 성립한다.

    $$
    L_{\max}(\sigma') = \max(L_j', L_i', \ldots) \leq \max(L_j, \ldots) = L_{\max}(\sigma)
    $$

    Swapping the inversion does not increase $L_{\max}$. $\square$

**주장 3.** 뒤바뀜이 없는 일정은 일을 EDF 차례로 다룬다.

**Conclusion.** Since inversions can be eliminated without increasing $L_{\max}$, and a schedule with no inversions is the EDF schedule, EDF is optimal.

## 파이썬 구현

```python
"""
마감 이른 것 먼저로 최대 늦음을 가장 작게 하는 일 차례 짜기.

가로채기 없는 기계 하나에서 마감으로 일을 정렬하면 최악의
늦음이 가장 작아짐을 보인다.
"""


# === 마감 이른 것 먼저 차례 짜기 ===

def edf_schedule(jobs):
    """마감 이른 것 먼저로 일의 차례를 짠다.

    인수:
        jobs: (처리 시간, 마감) 짝의 목록

    반환값:
        (차례, 최대 늦음)의 짝. 여기서 차례는
        (일 번호, 마친 때, 늦음) 짝의 목록이다
    """
    # 마감으로 정렬하되 본디 번호를 지닌다
    indexed_jobs = sorted(enumerate(jobs), key=lambda x: x[1][1])

    schedule = []
    current_time = 0

    for original_idx, (proc_time, deadline) in indexed_jobs:
        current_time += proc_time
        lateness = current_time - deadline
        schedule.append((original_idx, current_time, lateness))

    max_lateness = max(entry[2] for entry in schedule)
    return schedule, max_lateness


if __name__ == "__main__":
    # 보기: (처리 시간, 마감)
    jobs = [(3, 6), (2, 8), (1, 9), (4, 9)]

    schedule, max_lateness = edf_schedule(jobs)

    print("EDF Schedule:")
    print(f"{'Job':>4} {'C_i':>6} {'d_i':>6} {'L_i':>6}")
    print("-" * 24)
    for job_idx, completion, lateness in schedule:
        proc, deadline = jobs[job_idx]
        print(f"{job_idx + 1:>4} {completion:>6} {deadline:>6} {lateness:>6}")
    print(f"\nMaximum lateness: {max_lateness}")
```

**출력:**
```
EDF Schedule:
 Job    C_i    d_i    L_i
------------------------
   1      3      6     -3
   2      5      8     -3
   3      6      9     -3
   4     10      9      1

Maximum lateness: 1
```

## 복잡도 분석

- **Sorting:** $O(n \log n)$.
- **일정 짜기:** 한 번 훑기로 $O(n)$.
- **Total:** $O(n \log n)$.

## 변종: 무게를 준 마침 시각 가장 작게 하기

A related problem minimizes the **total weighted completion time** $\sum_{i=1}^{n} w_i C_i$, where $w_i$ is the weight (priority) of job $i$.

**욕심쟁이 규칙.** 일을 $w_i / p_i$(무게 대 다루는 시간 비)의 내림차순으로 다룬다.

이 변종도 욕심쟁이 알고리즘으로 풀 수 있으며, 옳음은 이웃 맞바꿈 논증으로 증명한다. 곧 $w_i/p_i < w_j/p_j$인 이웃한 일 $i$과 $j$을 맞바꾸면 목표가 엄밀히 나아진다.

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), 16장. MIT Press.
- Kleinberg, J. & Tardos, E. (2006). *Algorithm Design*, 4.2절. Pearson.

## 연습문제

**연습문제 1.**
일 일정 짜기에서 욕심쟁이 고름이 무엇인지 가려내고 왜 가장 좋은 풀이로 이어지는지 밝혀라.

??? success "연습문제 1 풀이"
    욕심쟁이 고름은 걸음마다 그 자리에서 가장 좋은 것을 고른다. Job Scheduling에서는 이 고름이 욕심쟁이 고름 성질을 채운다. 곧 이 욕심쟁이 고름을 담은 가장 좋은 풀이가 있다. 여기에 가장 좋은 아래 짜임(욕심쟁이 고름 뒤 남은 아래 문제도 같은 전략으로 가장 좋게 풀린다)을 더하면 욕심쟁이 알고리즘이 두루 가장 좋은 풀이를 내놓는다. $\square$

---

**연습문제 2.**
일 일정 짜기이 가장 좋은 아래 짜임을 갖는지 증명하거나 반증하여라.

??? success "연습문제 2 풀이"
    가장 좋은 아래 짜임이란 문제의 가장 좋은 풀이가 그 아래 문제의 가장 좋은 풀이를 담는다는 뜻이다. Job Scheduling에서는 욕심쟁이 고름을 하고 나면 남은 문제가 같은 갈래의 더 작은 사례가 된다. 아래 문제의 풀이가 가장 좋지 않다면 그것을 갈음해 온 풀이를 더 낫게 할 수 있는데, 이는 온 풀이가 가장 좋다는 것과 어긋난다. 그러므로 가장 좋은 아래 짜임이 이루어진다. $\square$

---

**연습문제 3.**
일 일정 짜기의 시간 복잡도는 무엇인가? 가장 값비싼 단계를 가려내어라.

??? success "연습문제 3 풀이"
    시간 복잡도는 (필요하다면) 줄 세우는 걸음과 욕심쟁이 고름 되돌이에 달렸다. 흔히 줄 세우기가 $O(n \log n)$으로 가장 크다. 욕심쟁이 되돌이는 원소마다 한 번씩 다루어 $O(n)$이다. 모두 $O(n \log n)$이다. 들임이 미리 줄 세워져 있으면 알고리즘은 $O(n)$에 돈다. $\square$

---

**연습문제 4.**
(일 일정 짜기에서 쓴 것이 아닌) 다른 욕심쟁이 전략은 가장 좋은 풀이를 내지 못함을 보이는 반례를 들어라.

??? success "연습문제 4 풀이"
    문제의 짜임과 맞지 않는 다른 욕심쟁이 잣대를 생각해 보자. 이 잣대는 뒤에 더 좋은 고름을 막는 원소를 고를 수 있다. 어긋나는 보기는 잘못된 욕심쟁이 잣대가 가장 좋지 않은 결과를 낼 수 있음을 보이며, 문제마다 그에 맞는 욕심쟁이 고름 성질을 반드시 증명해야 하는 까닭을 드러낸다. $\square$
