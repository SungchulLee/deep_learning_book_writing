# 짐 고르게 나누기 어림

똑같은 기계 $m$대와 다루는 시간이 $p_1, p_2, \dots, p_n$인 일 $n$개가 있다고 하자. 모든 일을 기계 하나씩에 매겨 **마침 때**, 곧 어느 기계든 최대 온 짐을 가장 작게 하려 한다. 이것이 **최소 마침 때 차례 잡기**(또는 짐 고르게 나누기) 문제이며 $m = 2$에서도 NP-어려움이다. 욕심쟁이 알고리즘이 놀랍도록 좋은 어림을 준다.

---

## 1. 문제의 정의

기계 $m$대와 다루는 시간이 $p_j > 0$인 일 $n$개가 주어질 때 마침 때를 가장 작게 하는 매김 $\sigma: \{1, \dots, n\} \to \{1, \dots, m\}$을 찾아라.

$$
C_{\max} = \max_{i=1}^{m} \sum_{\substack{j : \sigma(j) = i}} p_j
$$

$\text{OPT}$을 가장 좋은 마침 때라 하자. 단순한 아래 한계 둘이 참이다.

$$
\text{OPT} \ge \frac{1}{m} \sum_{j=1}^n p_j
\qquad \text{and} \qquad
\text{OPT} \ge \max_j\, p_j
$$

첫째는 평균 짐이 $\sum p_j / m$이고 최댓값이 적어도 평균 이상이므로 따라 나온다. 둘째는 어떤 기계든 가장 큰 일을 다루어야 하므로 참이다.

---

## 2. 목록 차례 잡기(그레이엄 알고리즘)

**직관.** 일을 아무 차례로 다룬다. 일마다 지금 짐이 가장 적은 기계에 매긴다. 이 욕심쟁이 규칙은 일이 남아 있는 동안 기계가 놀게 두지 않는다.

!!! tip "정리(Graham, 1966)"
    목록 차례 잡기는 마침 때가 많아야 $(2 - 1/m) \cdot \text{OPT}$이다.

**밝힘.** 기계 $i^*$이 마침 때 $C_{\max}$을 이루고 $j^*$을 $i^*$에 마지막으로 매긴 일이라 하자. $j^*$을 매길 때 $i^*$의 짐이 모든 기계 가운데 가장 적었으므로,

$$
C_{\max} - p_{j^*} \le \frac{1}{m} \sum_{j=1}^n p_j
$$

두 아래 한계를 쓰면:

$$
C_{\max} = (C_{\max} - p_{j^*}) + p_{j^*}
\le \frac{1}{m} \sum_{j=1}^n p_j + \max_j\, p_j
\le \text{OPT} + \text{OPT} - \frac{\text{OPT}}{m}
$$

더 자세히 따져 보자. (평균 한계에서) $C_{\max} - p_{j^*} \le \text{OPT}$이고 (최대 일 한계에서) $p_{j^*} \le \text{OPT}$이지만 더 빡빡하게 아울러야 한다. 왜냐하면

$$
C_{\max} - p_{j^*} \le \frac{1}{m}\sum p_j \le \text{OPT}
$$

$C_{\max} \le \text{OPT} + p_{j^*} \le \text{OPT} + \text{OPT} = 2 \cdot \text{OPT}$을 얻는다.

더 다듬은 $(2 - 1/m)$ 한계에서는 $j^*$ 앞의 $i^*$ 짐이 많아야 *모든* 기계 짐의 평균이었고 $p_{j^*} \le \text{OPT}$이므로,

$$
C_{\max} \le \frac{1}{m}\sum_{j=1}^{n} p_j + p_{j^*}

- \frac{p_{j^*}}{m}
= \frac{1}{m}\sum p_j + \left(1 - \frac{1}{m}\right)p_{j^*}
\le \text{OPT} + \left(1 - \frac{1}{m}\right)\text{OPT}
= \left(2 - \frac{1}{m}\right)\text{OPT} \qquad \square
$$

---

## 3. 가장 긴 일 먼저(LPT)

**직관.** 목록 차례 잡기 앞에 일을 내림 차례로 정렬하면 큰 일이 뒤늦게 이미 짐이 많은 기계에 놓이는 것을 막는다.

!!! tip "정리(Graham, 1969)"
    가장 긴 일 먼저 차례 잡기는 마침 때가 많아야 $(4/3 - 1/(3m)) \cdot \text{OPT}$이다.

나아진 비율은 (마지막에 놓은 일) $p_{j^*}$이 $p_{j^*} \le \text{OPT}/3$을 채우면(정렬한 차례에서 남은 가장 작은 일이고 일이 적어도 $m + 1$개 있으므로) 한계가 $4/3$으로 빡빡해진다는 데서 온다.

---

## 4. 구현

```python
"""
짐 고르게 나누기: 목록 차례 잡기와 가장 긴 일 먼저 어림 알고리즘.
"""

import heapq

# === 목록 차례 잡기 ==========================================================

def list_scheduling(jobs, m):
    """
    욕심쟁이 목록 차례 잡기: 일마다 짐이 가장 적은 기계에 매긴다.

    (마침 때, 매김)을 돌려준다.
    어림 비율: 2 - 1/m.
    """
    # 최소 무지개탑: (지금 짐, 기계 아이디)
    machines = [(0, i) for i in range(m)]
    assignment = [0] * len(jobs)

    for j, p in enumerate(jobs):
        load, mid = heapq.heappop(machines)
        assignment[j] = mid
        heapq.heappush(machines, (load + p, mid))

    makespan = max(load for load, _ in machines)
    return makespan, assignment

# === 가장 긴 일 먼저 차례 잡기 ================================================

def lpt_scheduling(jobs, m):
    """
    가장 긴 일 먼저 차례 잡기.

    (마침 때, 본디 어깨수를 쓴 매김)을 돌려준다.
    어림 비율: 4/3 - 1/(3m).
    """
    indexed = sorted(enumerate(jobs), key=lambda x: -x[1])
    machines = [(0, i) for i in range(m)]
    assignment = [0] * len(jobs)

    for orig_idx, p in indexed:
        load, mid = heapq.heappop(machines)
        assignment[orig_idx] = mid
        heapq.heappush(machines, (load + p, mid))

    makespan = max(load for load, _ in machines)
    return makespan, assignment

# === 보여 주기 ===============================================================

if __name__ == "__main__":
    jobs = [6, 3, 8, 5, 2, 7, 4, 1]
    m = 3

    ms_list, assign_list = list_scheduling(jobs, m)
    print(f"List Scheduling: makespan={ms_list}")
    for i in range(m):
        task_indices = [j for j in range(len(jobs)) if assign_list[j] == i]
        load = sum(jobs[j] for j in task_indices)
        print(f"  Machine {i}: jobs={task_indices}, load={load}")

    print()
    ms_lpt, assign_lpt = lpt_scheduling(jobs, m)
    print(f"LPT Scheduling:  makespan={ms_lpt}")
    for i in range(m):
        task_indices = [j for j in range(len(jobs)) if assign_lpt[j] == i]
        load = sum(jobs[j] for j in task_indices)
        print(f"  Machine {i}: jobs={task_indices}, load={load}")

    lb = sum(jobs) / m
    print(f"\nLower bound (avg): {lb:.1f}")
    print(f"Lower bound (max): {max(jobs)}")
```

**출력:**

```
List Scheduling: makespan=15
  Machine 0: jobs=[0, 4, 5], load=15
  Machine 1: jobs=[1, 3, 6], load=12
  Machine 2: jobs=[2, 7], load=9

LPT Scheduling:  makespan=13
  Machine 0: jobs=[1, 2, 4], load=13
  Machine 1: jobs=[5, 6, 7], load=12
  Machine 2: jobs=[0, 3], load=11

Lower bound (avg): 12.0
Lower bound (max): 8
```

---

## 연습문제

**연습문제 1.**
짐 고르게 나누기 어림의 어림 알고리즘을 설명하고 그 어림 보장을 밝혀라.

??? success "연습문제 1 풀이"
    이 알고리즘은 다항식 시간에 돌며 가장 좋은 값의 밝힐 수 있는 갑절 안에 드는 풀이를 낸다. 어림 비율은 알고리즘이 내놓은 것을 가장 좋은 값의 아래 한계(가장 작게 하기)나 위 한계(가장 크게 하기), 곧 선형 계획 느슨하게 하기 값이나 조합 한계, 문제의 짜임 성질과 이어 밝힌다. $\square$

---

**연습문제 2.**
짐 고르게 나누기 어림의 어림 비율을 밝히는 데 어떤 아래 한계 재주를 쓰는가?

??? success "연습문제 2 풀이"
    밝힘은 흔히 알고리즘의 풀이를 느슨하게 한 한계(선형 계획 느슨하게 하기, 분수 풀이, 조합 아래 한계)와 견준다. 가장 작게 하기에서는 $ALG \leq \rho \cdot LP^* \leq \rho \cdot OPT$이다. 가장 크게 하기에서는 $ALG \geq OPT / \rho$이다. 아래 한계는 효율 좋게 셈할 수 있고 쓸모 있는 비율을 줄 만큼 빡빡해야 한다. $\square$

---

**연습문제 3.**
짐 고르게 나누기 어림의 어림 비율을 더 좋게 할 수 있는가? 알려진 어려움 결과는 무엇인가?

??? success "연습문제 3 풀이"
    어림 비율이 얼마나 빡빡한지는 복잡도 이론의 가정(P $\neq$ NP, 하나뿐인 놀이 추측 등)에 달렸다. 어떤 문제에서는 단순한 욕심쟁이나 반올림 알고리즘이 여느 가정 아래 이미 가장 좋다. 다른 문제에서는 가장 좋은 알고리즘과 가장 센 어려움 결과 사이에 틈이 있어 아직 풀리지 않은 연구 문제로 남아 있다. $\square$

---

**연습문제 4.**
짐 고르게 나누기 어림을 구체적인 보기에 써서 어림 비율이 참임을 확인하라.

??? success "연습문제 4 풀이"
    작은 보기(예컨대 꼭짓점이나 물건 5~6개)를 고른다. 어림 알고리즘을 한 걸음씩 돌린다. 알고리즘이 내놓은 것을 (작은 보기에서 막무가내로 찾은) 가장 좋은 풀이와 견준다. 비율 $ALG/OPT$(또는 $OPT/ALG$)이 밝힌 한계 안에 드는지 확인한다. 그러면 구체적인 보기에서 이론이 굳어진다. $\square$

## 정리하며

| 알고리즘 | 비율 | 시간 |
|---|---|---|
| 목록 차례 잡기 | $2 - 1/m$ | $O(n \log m)$ |
| 가장 긴 일 먼저 | $4/3 - 1/(3m)$ | $O(n \log n)$ |
| PTAS(호크바움-시모이스) | $1 + \epsilon$ | $O(n \cdot (n/\epsilon)^{O(m)})$ |

$m$이 붙박이면 다항식 시간 어림 얼개가 있지만(Hochbaum과 Shmoys, 1987), $m$이 바뀌면 문제가 강하게 NP-어려움이므로 P = NP가 아니라면 FPTAS은 있을 수 없다.

**참고 문헌**

- Graham, R. L. "Bounds on Multiprocessing Timing Anomalies." *SIAM J. Appl. Math.*, 1969.
- Hochbaum, D. S. and Shmoys, D. B. "Using Dual Approximation Algorithms for Scheduling Problems." *JACM*, 1987.
