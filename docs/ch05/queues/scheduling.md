# 작업 스케줄링

운영체제는 기다리는 프로세스 가운데 어느 것에 CPU를 줄지 정해야 한다. 가장 단순한 정책인 **선착순(FCFS)**은 큐를 쓴다. 프로세스를 도착한 순서대로 처리하며, 각 프로세스가 끝나야 다음 프로세스가 시작된다. 더 대화형에 가까운 정책인 **라운드 로빈(RR)**도 큐를 쓰되, 프로세스마다 정해진 시간 조각(퀀텀)을 준 뒤 큐의 뒤로 보낸다. 두 알고리즘 모두 공평함을 위해 큐의 선입선출 성질에 기댄다. 이 쪽은 선착순과 라운드 로빈 스케줄링을 설명하고, 주요 성능 지표를 정의하며, 두 방식을 견주어 본다.

---

## 1. 스케줄링 지표

스케줄링 알고리즘을 평가하는 데 세 가지 지표를 쓴다.

**반환 시간**은 프로세스가 도착한 때부터 끝날 때까지의 전체 시간이다.

$$
T_{\text{turnaround}} = T_{\text{completion}} - T_{\text{arrival}}
$$

**대기 시간**은 프로세스가 실행되지 않고 준비 큐에서 보내는 전체 시간이다.

$$
T_{\text{waiting}} = T_{\text{turnaround}} - T_{\text{burst}}
$$

여기서 $T_{\text{burst}}$은 프로세스가 실제로 필요로 하는 CPU 시간이다.

**평균 반환 시간**과 **평균 대기 시간**은 모든 프로세스에 걸쳐 계산하며 전체적인 성능 척도가 된다.

---

## 2. 선착순 스케줄링

선착순(선입선출 스케줄링이라고도 한다)에서는 프로세스를 도착 순서대로 처리한다. 각 프로세스가 끝나야 다음 프로세스가 시작된다.

**장점**: 구현이 간단하고 도착 순서라는 뜻에서 공평하다.

**단점**: **호송 효과**가 있다. 큐 앞의 긴 프로세스가 뒤따르는 모든 프로세스를 늦추어 평균 대기 시간을 부풀린다.

---

## 3. 라운드 로빈 스케줄링

라운드 로빈은 프로세스마다 정해진 **시간 퀀텀** $q$을 준다. 퀀텀이 다하면 프로세스를 멈추고 큐의 뒤로 보낸다. 퀀텀 안에 끝나면 큐에서 빠진다.

라운드 로빈은 한 프로세스가 CPU를 독차지하는 시간을 제한하여 호송 효과를 줄인다. 퀀텀을 어떻게 정하느냐가 매우 중요하다.

- **너무 크면** ($q \to \infty$): 선착순으로 되돌아간다
- **너무 작으면** ($q \to 0$): 문맥 전환의 부담이 지나치게 커진다

```python
"""
작업 스케줄링 — 큐를 쓰는 선착순과 라운드 로빈 스케줄링.

큐의 선입선출 성질이 공정한 프로세스 배정을 뒷받침하는 모습을 보이고,
반환 시간과 대기 시간도 계산한다.
"""
from collections import deque

# === 프로세스의 표현 ===================================================

class Process:
    """도착 시각과 필요 시간을 갖는 프로세스를 나타낸다."""

    def __init__(self, name, arrival, burst):
        self.name = name
        self.arrival = arrival
        self.burst = burst
        self.remaining = burst
        self.completion = 0

    def __repr__(self):
        return f"{self.name}(arr={self.arrival}, burst={self.burst})"

# === 선착순 스케줄링 ==========================================================

def fcfs_schedule(processes):
    """선착순 스케줄링.

    프로세스를 도착 시각 순으로 정렬하여 그 차례대로 처리한다.
    각 프로세스가 끝나야 다음 프로세스가 시작된다.

    시간:  정렬에 O(n log n), 배정에 O(n).
    공간: O(n).
    """
    procs = sorted(processes, key=lambda p: p.arrival)
    clock = 0
    results = []

    for p in procs:
        if clock < p.arrival:
            clock = p.arrival  # 프로세스가 올 때까지 CPU가 논다
        clock += p.burst
        p.completion = clock
        turnaround = p.completion - p.arrival
        waiting = turnaround - p.burst
        results.append((p.name, p.arrival, p.burst, p.completion, turnaround, waiting))

    return results

# === 라운드 로빈 스케줄링 ===================================================

def round_robin_schedule(processes, quantum):
    """고정 시간 할당량을 쓰는 라운드 로빈 스케줄링.

    각 프로세스는 많아야 `quantum` 시간 단위만큼 돌다가 멈추어
    큐의 뒤로 간다.

    시간:  최악의 경우 O(n * max_burst / quantum).
    공간: O(n).
    """
    # 원본을 바꾸지 않으려고 복사본 만들기
    procs = [Process(p.name, p.arrival, p.burst) for p in processes]
    procs.sort(key=lambda p: p.arrival)

    queue = deque()
    clock = 0
    idx = 0  # 다음에 도착할 프로세스
    results = {}
    completed_order = []

    # 첫 프로세스로 시작
    if procs:
        clock = procs[0].arrival
        queue.append(procs[0])
        idx = 1

    while queue:
        p = queue.popleft()
        run_time = min(quantum, p.remaining)
        clock += run_time
        p.remaining -= run_time

        # 이 시간 조각 동안 새로 도착한 것이 있는지 확인
        while idx < len(procs) and procs[idx].arrival <= clock:
            queue.append(procs[idx])
            idx += 1

        if p.remaining > 0:
            queue.append(p)  # 끝나지 않았으므로 다시 넣는다
        else:
            p.completion = clock
            turnaround = p.completion - p.arrival
            waiting = turnaround - p.burst
            results[p.name] = (p.name, p.arrival, p.burst, p.completion, turnaround, waiting)
            completed_order.append(p.name)

        # 큐는 비었는데 프로세스가 남았으면 다음 도착 시각으로 건너뛴다
        if not queue and idx < len(procs):
            clock = procs[idx].arrival
            queue.append(procs[idx])
            idx += 1

    return [results[name] for name in completed_order]

# === 보이기 ==================================================================

def print_schedule(title, results):
    """배정 결과를 표로 정리해 출력한다."""
    print(f"\n{title}")
    print(f"  {'Process':<10s} {'Arrival':>8s} {'Burst':>6s} {'Finish':>7s} {'Turnaround':>11s} {'Waiting':>8s}")
    print(f"  {'-'*52}")
    total_ta, total_wt = 0, 0
    for name, arr, burst, comp, ta, wt in results:
        print(f"  {name:<10s} {arr:>8d} {burst:>6d} {comp:>7d} {ta:>11d} {wt:>8d}")
        total_ta += ta
        total_wt += wt
    n = len(results)
    print(f"  {'-'*52}")
    print(f"  {'Average':<10s} {'':>8s} {'':>6s} {'':>7s} {total_ta/n:>11.1f} {total_wt/n:>8.1f}")

# === 시연 ============================================================

if __name__ == "__main__":
    processes = [
        Process("P1", arrival=0, burst=8),
        Process("P2", arrival=1, burst=4),
        Process("P3", arrival=2, burst=2),
        Process("P4", arrival=3, burst=1),
    ]

    # 선착순
    fcfs_results = fcfs_schedule(processes)
    print_schedule("FCFS Scheduling:", fcfs_results)

    # 퀀텀이 3인 라운드 로빈
    rr_results = round_robin_schedule(processes, quantum=3)
    print_schedule("Round-Robin Scheduling (quantum=3):", rr_results)
```

**출력:**
```
FCFS Scheduling:
  Process     Arrival  Burst  Finish  Turnaround  Waiting
  ----------------------------------------------------
  P1                0      8        8           8        0
  P2                1      4       12          11        7
  P3                2      2       14          12       10
  P4                3      1       15          12       11
  ----------------------------------------------------
  Average                                    10.8      7.0

Round-Robin Scheduling (quantum=3):
  Process     Arrival  Burst  Finish  Turnaround  Waiting
  ----------------------------------------------------
  P3                2      2        8           6        4
  P4                3      1        9           6        5
  P2                1      4       13          12        8
  P1                0      8       15          15        7
  ----------------------------------------------------
  Average                                     9.8      6.0
```

선착순 일정은 호송 효과를 보여 준다. P1이 8단위 동안 돌면서 필요 시간이 짧은 P2, P3, P4를 기다리게 만든다. 퀀텀이 3인 라운드 로빈은 프로세스를 번갈아 돌려 평균 반환 시간을 10.8에서 9.8로 줄인다.

---

## 4. 비교

| 성질 | 선착순 | 라운드 로빈 |
|----------|------|-------------|
| 공평함 | 도착 순서 | CPU를 고르게 나눔 |
| 선점 | 없음 | 있음 (퀀텀 경계에서) |
| 굶주림 | 없음 | 없음 |
| 호송 효과 | 있음 | 누그러짐 |
| 부담 | 없음 | 퀀텀마다 문맥 전환 |
| 알맞은 곳 | 일괄 처리 | 대화형 시스템 |

!!! warning "문맥 전환의 부담"
    라운드 로빈은 프로세스를 멈출 때마다 문맥 전환 비용을 치른다. 퀀텀이 평균 필요 시간보다 훨씬 작으면 이 부담이 실제 계산을 압도할 수 있다. 요즘 시스템에서 퀀텀은 보통 10~100밀리초이다.

---

## 연습문제

**연습문제 1.**
작업 스케줄링의 추상 자료형이 지원하는 연산을 시간 복잡도와 함께 모두 열거하라. 어느 연산이 병목인가?

??? success "연습문제 1 풀이"
    추상 자료형은 구현과 무관하게 지원하는 연산을 정한다. 무엇이 병목인지는 쓰임새에 달렸다. 실시간 시스템에서는 최악의 복잡도가 중요하고, 일괄 처리에서는 상각 복잡도로 충분하다.

---

**연습문제 2.**
작업 스케줄링을(를) 서로 다른 두 자료 구조로 구현하라. 각각의 절충을 비교하라.

??? success "연습문제 2 풀이"
    구현 1: 배열 기반 — 접근은 상수 시간이지만 크기를 다시 잡아야 할 수 있다. 구현 2: 연결 리스트 기반 — 삽입과 삭제는 상수 시간이지만 접근은 $O(n)$이다. 어느 쪽을 고를지는 응용에서 예상되는 연산의 구성에 달렸다.

---

**연습문제 3.**
작업 스케줄링을(를) 쓰는 딥러닝 응용을 하나 설명하라(예: 그래프 신경망의 너비 우선 탐색, 기호 미분에서의 식 계산, 데이터 적재의 스케줄링).

??? success "연습문제 3 풀이"
    구체적인 응용은 그 추상 자료형의 순서 성질에 달렸다. 선입선출(큐)은 GNN의 너비 우선 그래프 순회에 쓰이고, 후입선출(스택)은 자동 미분 테이프 처리에 쓰이며, 우선순위 순서는 빔 탐색과 예정 표집에 쓰인다.

---

**연습문제 4.**
작업 스케줄링을(를) 원형 배열로 구현하면 모든 연산이 상각 $O(1)$ 시간임을 증명하라.

??? success "연습문제 4 풀이"
    원형 배열은 머리와 꼬리 인덱스를 용량으로 나눈 나머지로 관리한다. 넣기와 빼기는 인덱스를 $O(1)$에 조정한다. 배열이 가득 차면 용량을 두 배로 늘리는 데 $O(n)$이 들지만, 이는 값싼 연산 $O(n)$번 뒤에 한 번 일어나므로 동적 배열과 같은 논법으로 상각 $O(1)$이 된다. $\square$

## 정리하며

이 마당은 스케줄링 지표、선착순 스케줄링、라운드 로빈 스케줄링、비교을 차례로 짚었다.

**참고 문헌**

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 10. MIT Press.
- Silberschatz, A., Galvin, P. B., & Gagne, G. (2018). *Operating System Concepts* (10th ed.), Chapter 5. Wiley.
