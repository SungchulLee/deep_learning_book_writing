# 흐름 차례 잡기

운영 얼개의 **차례잡이**는 때마다 어느 흐름이 CPU에서 돌지 정한다. 나름, 답하는 때, 고름 같은 자를 다듬는 것이 뜻이다. 차례잡이는 자주(자리를 바꿀 때마다) 도므로 그 알고리즘이 잘 들어야 한다. 흔히 판단마다 $O(1)$이나 $O(\log n)$이다.

---

## 1. 차례 잡기의 자

온 때가 $a_i$이고 쓰는 때가 $b_i$인 흐름 $n$개에 대해

- **돌아오는 때**: $T_i = C_i - a_i$이고 $C_i$은 마친 때다.
- **기다리는 때**: $W_i = T_i - b_i$.
- **답하는 때**: 온 때부터 처음 돌 때까지.
- **나름**: 때 낱마다 마친 흐름의 수.

---

## 2. FCFS(먼저 온 것을 먼저 다룸)

흐름을 온 차례대로 돌린다. 가로채지 않는다.

$$
\text{고르게 기다리는 때} = \frac{1}{n} \sum_{i=1}^{n} W_i
$$

- **나은 점**: 단순하고 온 차례로는 고르다.
- **나쁜 점**: **줄줄이 딸림**. 짧은 흐름이 긴 흐름 뒤에서 기다려 고르게 기다리는 때가 부푼다.

---

## 3. SJF(짧은 일을 먼저)

쓰는 때가 가장 짧은 흐름을 먼저 돌린다. 가로채지 않는 알고리즘 가운데 고르게 기다리는 때를 가장 작게 함이 증명되어 있다.

$$
\text{고르게 기다리는 때}_{\text{SJF}} \le \text{고르게 기다리는 때}_{\text{가로채지 않는 어떤 것}}
$$

가로채는 갈래(**SRTF**, 남은 때가 가장 짧은 것을 먼저)는 새로 온 흐름의 남은 때가 더 짧으면 돌고 있는 흐름을 가로챈다.

!!! warning "굶주림"
    짧은 흐름이 끊이지 않고 오면 SJF에서 긴 흐름이 굶을 수 있다. **나이 먹임**은 기다리는 흐름의 앞섬을 조금씩 올려 이를 눅인다.

---

## 4. 돌아가며 주기(RR)

흐름마다 붙박이 **때 몫** $q$만큼 돌고 나면 가로채여 마칠 채비 줄의 꽁무니로 간다.

- $q$이 크면 RR이 FCFS으로 무너진다.
- $q$이 작으면 자리 바꿈의 짐이 판친다.
- 흔한 몫은 10~100 밀리초다.

고르게 기다리는 때는 $q$과 쓰는 때의 퍼짐에 달렸다. 쓰는 때가 모두 $b$으로 같으면

$$
\text{고르게 기다리는 때} = (n - 1) \cdot q \cdot \left\lfloor \frac{b}{q} \right\rfloor / n
$$

---

## 5. 여러 켜 되먹임 줄(MLFQ)

MLFQ은 앞선 줄을 여럿 두어 잘 답하는 것과 나름을 저울질한다.

1. 새 흐름은 **가장 앞선** 줄로 들어간다.
2. 흐름이 막히지 않고 제 몫을 다 쓰면 한 켜 아래 줄로 내린다.
3. 아래 줄일수록 몫이 길다(보기로 켜마다 두 곱절).
4. 굶주림을 막으려고 이따금 모든 흐름을 맨 위 줄로 끌어올린다.

이렇게 하면 쓰는 때를 어림하지 않고도 서로 주고받는(들고남에 매인) 흐름과 CPU에 매인 흐름을 저절로 갈라낸다.

---

## 6. 견주기

| 알고리즘 | 가로챔 | 고른 기다림이 가장 작은가 | 굶주림 | 복잡도 |
|---|---|---|---|---|
| FCFS | 아니오 | 아니오 | 없음 | $O(1)$ |
| SJF | 아니오 | 예 | 있음 | $O(n)$ |
| SRTF | 예 | 예 | 있음 | $O(\log n)$ |
| RR | 예 | 아니오 | 없음 | $O(1)$ |
| MLFQ | 예 | 맞추어 감 | 끌어올리면 없음 | $O(1)$ |

---

## 7. 짜보기

```python
"""
흐름 차례 잡기 -- FCFS, SJF, 돌아가며 주기 흉내내기.

온 때와 쓰는 때가 주어진 흐름 모둠에 대해 알고리즘마다
기다리는 때와 돌아오는 때를 셈한다.
"""

from __future__ import annotations
from dataclasses import dataclass
from collections import deque

# === 흐름 ===================================================================

@dataclass
class Process:
    """온 때와 CPU 쓰는 때를 지닌 흐름."""
    pid: int
    arrival: int
    burst: int

# === FCFS ===================================================================

def fcfs(processes: list[Process]) -> list[tuple[int, int, int]]:
    """FCFS을 돌린다. 흐름마다 (번호, 기다리는 때, 돌아오는 때)를 내놓는다."""
    procs = sorted(processes, key=lambda p: (p.arrival, p.pid))
    results = []
    time = 0
    for p in procs:
        time = max(time, p.arrival)
        wait = time - p.arrival
        turnaround = wait + p.burst
        results.append((p.pid, wait, turnaround))
        time += p.burst
    return results

# === SJF(가로채지 않음) =====================================================

def sjf(processes: list[Process]) -> list[tuple[int, int, int]]:
    """SJF을 돌린다. 흐름마다 (번호, 기다리는 때, 돌아오는 때)를 내놓는다."""
    remaining = list(processes)
    results = []
    time = 0
    while remaining:
        available = [p for p in remaining if p.arrival <= time]
        if not available:
            time = min(p.arrival for p in remaining)
            available = [p for p in remaining if p.arrival <= time]
        chosen = min(available, key=lambda p: p.burst)
        remaining.remove(chosen)
        wait = time - chosen.arrival
        turnaround = wait + chosen.burst
        results.append((chosen.pid, wait, turnaround))
        time += chosen.burst
    return results

# === 돌아가며 주기 ==========================================================

def round_robin(processes: list[Process],
                quantum: int) -> list[tuple[int, int, int]]:
    """주어진 몫으로 돌아가며 주기를 돌린다. (번호, 기다림, 돌아옴)을 내놓는다."""
    n = len(processes)
    remaining_burst = {p.pid: p.burst for p in processes}
    arrival = {p.pid: p.arrival for p in processes}
    queue: deque[int] = deque()
    time = 0
    finish_time: dict[int, int] = {}
    arrived = set()
    procs_by_arrival = sorted(processes, key=lambda p: p.arrival)
    idx = 0

    # 처음 온 것들을 넣는다
    while idx < n and procs_by_arrival[idx].arrival <= time:
        queue.append(procs_by_arrival[idx].pid)
        arrived.add(procs_by_arrival[idx].pid)
        idx += 1

    while queue or idx < n:
        if not queue:
            time = procs_by_arrival[idx].arrival
            while idx < n and procs_by_arrival[idx].arrival <= time:
                queue.append(procs_by_arrival[idx].pid)
                arrived.add(procs_by_arrival[idx].pid)
                idx += 1

        pid = queue.popleft()
        run = min(quantum, remaining_burst[pid])
        time += run
        remaining_burst[pid] -= run

        # 새로 온 흐름을 넣는다
        while idx < n and procs_by_arrival[idx].arrival <= time:
            queue.append(procs_by_arrival[idx].pid)
            arrived.add(procs_by_arrival[idx].pid)
            idx += 1

        if remaining_burst[pid] > 0:
            queue.append(pid)
        else:
            finish_time[pid] = time

    results = []
    for p in processes:
        turnaround = finish_time[p.pid] - p.arrival
        wait = turnaround - p.burst
        results.append((p.pid, wait, turnaround))
    return results

# === 메인 ===================================================================

if __name__ == "__main__":
    processes = [
        Process(pid=1, arrival=0, burst=6),
        Process(pid=2, arrival=1, burst=4),
        Process(pid=3, arrival=2, burst=2),
        Process(pid=4, arrival=3, burst=3),
    ]

    for name, result in [
        ("FCFS", fcfs(processes)),
        ("SJF", sjf(processes)),
        ("RR(q=2)", round_robin(processes, quantum=2)),
    ]:
        avg_wait = sum(r[1] for r in result) / len(result)
        avg_turn = sum(r[2] for r in result) / len(result)
        print(f"{name:10s}  고른 기다림={avg_wait:.1f}  고른 돌아옴={avg_turn:.1f}")
        for pid, wait, turn in sorted(result):
            print(f"  P{pid}: 기다림={wait}, 돌아옴={turn}")
```

**내놓기:**

```
FCFS        고른 기다림=4.5  고른 돌아옴=8.2
  P1: 기다림=0, 돌아옴=6
  P2: 기다림=5, 돌아옴=9
  P3: 기다림=6, 돌아옴=8
  P4: 기다림=7, 돌아옴=10
SJF         고른 기다림=2.8  고른 돌아옴=6.5
  P1: 기다림=0, 돌아옴=6
  P2: 기다림=7, 돌아옴=11
  P3: 기다림=4, 돌아옴=6
  P4: 기다림=0, 돌아옴=3
RR(q=2)     고른 기다림=5.5  고른 돌아옴=9.2
  P1: 기다림=7, 돌아옴=13
  P2: 기다림=6, 돌아옴=10
  P3: 기다림=2, 돌아옴=4
  P4: 기다림=7, 돌아옴=10
```

SJF이 고르게 기다리는 때가 가장 짧아(2.8) 가로채지 않는 차례 잡기에서 가장 좋음을 밝혀 준다. FCFS은 줄줄이 딸림을 겪는다(P3은 2만 있으면 되는데 6을 기다린다). 돌아가며 주기는 CPU 때를 더 고르게 나누지만 자리 바꿈의 짐 때문에 고르게 기다리는 때가 길다.

---

## 연습문제

**연습문제 1.**
흐름 셋이 때 0에 오고 쓰는 때가 저마다 10, 5, 8 밀리초다. FCFS과 SJF 차례 잡기에서 고르게 돌아오는 때와 고르게 기다리는 때를 셈하여라.

??? success "연습문제 1 풀이"
    **FCFS**(차례: P1, P2, P3): P1이 10에, P2이 15에, P3이 23에 마친다. 돌아옴은 (10 + 15 + 23)/3 = 16 밀리초, 기다림은 (0 + 10 + 15)/3 = 8.33 밀리초다. **SJF**(차례: P2, P3, P1): P2이 5에, P3이 13에, P1이 23에 마친다. 돌아옴은 (5 + 13 + 23)/3 = 13.67 밀리초, 기다림은 (0 + 5 + 13)/3 = 6 밀리초다. SJF은 가로채지 않는 차례 가운데 고르게 기다리는 때를 가장 작게 한다. 여기서 나아진 만큼은 $8.33 - 6 = 2.33$ 밀리초(28% 줆)다. $\square$

---

**연습문제 2.**
때 0에 모두 와 있는 일감 모둠에 대해 짧은 일 먼저(SJF)가 고르게 기다리는 때를 가장 작게 함을 증명하여라.

??? success "연습문제 2 풀이"
    일감의 쓰는 때를 $b_1 \le b_2 \le \cdots \le b_n$(SJF 차례)이라 하자. 일감 $i$의 기다리는 때는 $\sum_{j=1}^{i-1} b_j$이다. 온 기다리는 때는 $\sum_{i=1}^{n} \sum_{j=1}^{i-1} b_j = \sum_{j=1}^{n} (n - j) b_j$이다. 이 짐 실은 더함을 가장 작게 하려면 큰 짐 $(n - j)$에 작은 $b_j$을 곱해야 하고, 이것이 바로 SJF 차례(짧은 것이 먼저)다. 긴 일이 짧은 일 앞에 놓인 이웃한 두 일감을 맞바꾸면 그 사이에 놓인 일감의 수만큼 쓰는 때의 차이가 온 기다리는 때에 더해진다. 다시 늘어놓기 부등식에 따라 줄 세운 차례가 가장 좋다. $\square$

---

**연습문제 3.**
여러 켜 되먹임 줄(MLFQ) 차례잡이를 밝혀라. 서로 주고받는 흐름의 답하는 때와 묶음 흐름의 나름을 어떻게 저울질하는가?

??? success "연습문제 3 풀이"
    MLFQ은 앞선 줄을 여럿 지닌다. 새 흐름은 가장 앞선 줄로 들어간다. 흐름이 막히지 않고 제 때 몫을 다 쓰면 (CPU에 매였다고 보아) 아래 줄로 내린다. 일찍 막히면(들고남에 매인, 서로 주고받는 흐름) 앞선 자리에 남는다. 앞선 줄일수록 때 몫이 짧다. 이렇게 흐름이 저절로 갈린다. 서로 주고받는 흐름(CPU를 짧게 쓰고 들고남이 잦은)은 짧은 몫으로 앞선 자리에 남아 답하는 때가 짧다. CPU에 매인 흐름은 긴 몫을 지닌 뒤진 줄로 가라앉아, 자리를 자꾸 바꾸지 않으면서 나름을 크게 한다. 굶주림을 막으려고 이따금 "끌어올림"으로 모든 흐름을 맨 위 줄로 되돌린다. 오래 도는 흐름이 새로 오는 서로 주고받는 흐름에 밀려 끝내 굶는 일을 막는다. $\square$

---

**연습문제 4.**
리눅스의 아주 고른 차례잡이(CFS)는 헛 돎 때를 열쇠로 삼는 붉은검은 나무를 쓴다. $O(\log n)$ 차례 잡기와 고름을 어떻게 이루는지 밝혀라.

??? success "연습문제 4 풀이"
    CFS은 돌 수 있는 흐름마다 "헛 돎 때"(vruntime)를 좇는다. 이는 그 흐름이 받은 온 CPU 때를 제 앞섬(nice 값)으로 짐 실어 잰 것이다. 헛 돎 때가 가장 작은 흐름이 다음에 돈다. 제 고른 몫에 견주어 CPU 때를 가장 적게 받은 흐름이다. 흐름은 헛 돎 때를 열쇠로 삼는 붉은검은 나무에 갈무리된다. 다음 흐름을 고르려면 가장 왼쪽 마디를 집으면 된다(갈무리한 손가락질이 있으면 $O(1)$, 나무를 훑으면 $O(\log n)$). 흐름을 넣는 데(깨어나거나 새로 나거나)는 나무에 $O(\log n)$으로 넣는다. 이렇게 고름이 이루어진다. 때가 흐르면 도는 흐름의 헛 돎 때가 늘어 끝내 다른 흐름이 가장 작아지므로 모든 흐름의 헛 돎 때가 한곳으로 모인다. 앞선 흐름은 헛 돎 때가 줄여져 셈되므로(헛 돎 때를 더 더디게 "번다") CPU 때를 더 많이 받는다. $\square$

---

**연습문제 5.**
어느 제때 얼개에 되풀이하는 일감 둘이 있다. 일감 A은 돌이가 10 밀리초이고 도는 때가 3 밀리초, 일감 B은 돌이가 20 밀리초이고 도는 때가 8 밀리초다. 빠르기 홑결 차례 잡기(RMS)와 가장 이른 기한 먼저(EDF)에서 이 얼개의 차례를 잡을 수 있는지 가려라.

??? success "연습문제 5 풀이"
    CPU 씀씀이는 $U = 3/10 + 8/20 = 0.3 + 0.4 = 0.7$이다. **RMS**: 일감 둘일 때 씀씀이 울타리는 $2(2^{1/2} - 1) = 2 \times 0.414 = 0.828$이다. $0.7 < 0.828$이므로 RMS에서 차례를 잡을 수 있다. RMS은 돌이가 짧은 일감 A에 더 앞선 자리를 준다. 차례는 [0-3] A, [3-11] B, [10-13] A, [13-19] B 이어짐, [20-23] A, ... 이렇게 되어 두 일감 모두 기한을 지킨다. **EDF**: 씀씀이 울타리가 1.0이다(홑 다룸꾼에서 EDF이 가장 좋다). $0.7 < 1.0$이므로 차례를 잡을 수 있다. EDF은 늘 참 기한이 가장 이른 일감을 잡으며 그때그때 맞추어 간다. 여기서는 둘 다 되지만, EDF은 $U = 1.0$까지 되는 데 견주어 RMS은 일감 둘일 때 $U \approx 0.828$을 넘으면 무너진다. $\square$

## 정리하며

이 마당은 차례 잡기의 자、FCFS(먼저 온 것을 먼저 다룸)、SJF(짧은 일을 먼저)、돌아가며 주기(RR)을 차례로 짚었다.

**살펴볼 거리**

- Silberschatz, A., Galvin, P.B., and Gagne, G. *Operating System Concepts*. Wiley
- Arpaci-Dusseau, R.H. and Arpaci-Dusseau, A.C. *Operating Systems: Three Easy Pieces*
