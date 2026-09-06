# 일 훔치기

갈라짐-합침 나란함에서 일감은 그때그때 생기고 셈틀마다의 짐은 헤아릴 수 없다. 일감을 셈틀에 붙박아 맡기면 짐이 고르지 않다. 어떤 셈틀은 일찍 끝나 놀고 어떤 셈틀은 짐이 넘친다. **일 훔치기**는 단순한 규칙 하나로 이를 푼다. 노는 셈틀이 바쁜 셈틀의 줄에서 일감을 훔치는 것이며, 거의 가장 좋은 성능을 이룸이 밝혀져 있다.

## 일 훔치기 규약

셈틀마다 준비된 일감의 그 자리 **두 끝 줄**을 지닌다:

1. 셈틀이 새 부분 일감으로 **갈라지면** 그것을 제 줄의 아래에 넣는다.
2. 셈틀이 일감을 끝내면 제 줄의 아래에서 다음 일감을 꺼낸다(나중 것 먼저).
3. 셈틀의 줄이 비면 **도둑**이 된다. 아무 희생자 셈틀을 골라 그 줄 위쪽에서 일감을 훔친다(먼저 것 먼저).

!!! tip "왜 제 것은 나중 것 먼저, 훔치기는 먼저 것 먼저인가"
    제 일감은 아래에서 꺼낸다(나중 것 먼저). 최근에 생긴 일감이 대개 두름 가까움이 좋은 작은 잎 일감이기 때문이다. 훔치는 일감은 위에서 가져간다(먼저 것 먼저). 셈 나무 뿌리에 가까운 오래된 일감이 더 커서 훔치는 값을 더 많은 셈에 고루 나눌 수 있기 때문이다.

## 기대 돌림 때

다음 정리가 아무렇게 하는 일 훔치기의 이론 효율을 세운다.

**정리(블루모프-라이저슨, 1999).** 일이 $T_1$이고 뻗음이 $T_\infty$인 셈에서 셈틀 $p$개의 아무렇게 하는 일 훔치기 일정잡이는 다음 기대 돌림 때를 이룬다:

$$
\mathbb{E}[T_p] = O\!\left(\frac{T_1}{p} + T_\infty\right)
$$

이는 기댓값에서 브렌트 정리의 가둠과 맞으며, 일 훔치기가 점근으로 가장 좋은 욕심쟁이 일정잡이임을 뜻한다.

??? note "밝힘의 느낌"
    어느 때 걸음에서든 셈틀은 일감을 돌리거나(보람 있는 걸음) 훔치기를 시도한다(훔치기 시도). 모든 셈틀을 통틀은 보람 있는 걸음의 수는 꼭 $T_1$이다. 핵심 통찰은 온 훔치기 시도 수를 가두는 것이다. 훔치기마다 아무 희생자를 고르고 핵심 길 위 일감의 수가 모든 줄이 한꺼번에 빌 수 있는 잦음을 가두므로, 기대 온 훔치기 시도는 $O(p \cdot T_\infty)$이다. 셈틀 $p$개로 나누면 그 가둠이 나온다. $\square$

## 공간 가둠

일 훔치기는 공간 보장도 준다.

**정리.** 차례 쌓기 공간이 $S_1$인 셈을 셈틀 $p$개에서 일 훔치기로 돌리면 온 공간이 많아야 $O(p \cdot S_1)$이다.

셈틀마다 줄이 많아야 틀 $S_1$개(셈 나무의 최대 깊이)를 담고 셈틀이 $p$개다.

## 흉내 내기

```python
"""
일 훔치기 일정잡이 흉내내기.

여러 셈틀에서 아무 희생자 고르기로 갈라짐-합침 일감을 다루는
일 훔치기 일정잡이를 흉내 낸다.
"""

import random
from collections import deque

# ===================================================================
# 일감과 일 훔치기 일정잡이
# ===================================================================

class Task:
    """비용을 가진 일의 낱덩이."""

    def __init__(self, task_id, cost=1):
        self.task_id = task_id
        self.cost = cost

    def __repr__(self):
        return f"Task({self.task_id})"


class WorkStealingScheduler:
    """여러 셈틀에 걸친 일 훔치기를 흉내 낸다.

    인수:
        num_processors: 흉내 낼 셈틀 수
    """

    def __init__(self, num_processors):
        self.p = num_processors
        self.deques = [deque() for _ in range(self.p)]
        self.completed = [[] for _ in range(self.p)]
        self.steal_count = 0
        self.total_steps = 0

    def submit(self, tasks, processor=0):
        """Submit tasks to a processor's deque.

        인수:
            tasks: Task 대상의 목록
            processor: 과녁 셈틀 번호
        """
        for task in tasks:
            self.deques[processor].append(task)

    def run(self):
        """일 훔치기로 모든 일감을 돌린다."""
        while any(self.deques):
            self.total_steps += 1
            for pid in range(self.p):
                if self.deques[pid]:
                    # 아래에서 꺼낸다(나중 것 먼저)
                    task = self.deques[pid].pop()
                    self.completed[pid].append(task)
                else:
                    # 아무 희생자에게서 훔친다(먼저 것 먼저)
                    victims = [v for v in range(self.p)
                               if v != pid and self.deques[v]]
                    if victims:
                        victim = random.choice(victims)
                        task = self.deques[victim].popleft()
                        self.completed[pid].append(task)
                        self.steal_count += 1

    def report(self):
        """일정 잡기 통계를 찍는다."""
        total_tasks = sum(len(c) for c in self.completed)
        print(f"Processors:    {self.p}")
        print(f"Total tasks:   {total_tasks}")
        print(f"Time steps:    {self.total_steps}")
        print(f"Steal count:   {self.steal_count}")
        print(f"Tasks per processor:")
        for pid in range(self.p):
            print(f"  P{pid}: {len(self.completed[pid])} tasks")

# ===================================================================
# 메인
# ===================================================================

if __name__ == "__main__":
    random.seed(42)

    # 일감 20개를 만들어 모두 0번 셈틀에 맡긴다
    tasks = [Task(i) for i in range(20)]

    scheduler = WorkStealingScheduler(num_processors=4)
    scheduler.submit(tasks, processor=0)
    scheduler.run()
    scheduler.report()
```

**출력:**
```
셈틀:    4
온 일감:   20
때 걸음:    6
훔친 횟수:   15
Tasks per processor:
  P0: 일감 5개
  P1: 일감 5개
  P2: 일감 5개
  P3: 일감 5개
```

## 핵심 성질

| 성질 | 값 |
|---|---|
| 기대 때 | $O(T_1/p + T_\infty)$ |
| 공간 | $O(p \cdot S_1)$ |
| 주고받기 | 기대 훔치기 시도 $O(p \cdot T_\infty)$번 |
| 짐 고르기 | 높은 확률로 거의 가장 좋음 |

## 실용적인 고려

- **두 끝 줄 짜기**: 체이스-레브 두 끝 줄은 주인 쪽에서 자물쇠 없는 넣기와 꺼내기를, 도둑 쪽에서 견주고 바꾸기 바탕 훔치기를 주어 맞추기 덧짐을 가장 작게 한다.
- **훔치기 규칙**: 아무 희생자 고르기는 단순하며 이론과 실제 모두에서 좋은 가둠을 이룬다. 어떤 짜기는 두름 움직임을 좋게 하려 가까움을 살피는 훔치기를 쓴다.
- **알갱이 크기 다스리기**: 일감이 너무 잘면 줄 연산과 훔치기의 덧짐이 판을 친다. 흔한 다듬기는 자르는 크기 아래의 일감을 차례로 돌리는 것이다.

## 참고 문헌

- Blumofe, R. D. and Leiserson, C. E. (1999). "Scheduling multithreaded computations by work stealing." *JACM*, 46(5), 720--748.
- Chase, D. and Lev, Y. (2005). "Dynamic circular work-stealing deque." *SPAA*.


## 연습문제

**연습문제 1.**
일 훔치기 일정 잡기 알고리즘을 밝혀라.

??? success "연습문제 1 풀이"
    셈틀마다 그 자리 일감 두 끝 줄을 가진다. (갈라짐에서 나온) 새 일감을 아래에 넣는다. 셈틀은 제 줄 아래에서 일감을 꺼낸다(나중 것 먼저). 비면 아무 희생자의 위쪽에서 훔친다(먼저 것 먼저). 나중 것 먼저 돌림: 최근에 갈라진 일감이 두름에 있을 법하다(때의 가까움). 먼저 것 먼저 훔치기: 뿌리에 가까운 일감이 커서 훔치는 횟수가 준다. 아무렇게 훔치기가 짐 고르기를 보장한다.

---

**연습문제 2.**
일 훔치기 성능에 대한 블루모프-라이저슨 정리를 말하고 밝혀라.

??? success "연습문제 2 풀이"
    정리: 일이 $W$이고 뻗음이 $D$인 셈을 셈틀 $p$개에서 일 훔치기로 돌리면 기대 때 $T_p \leq W/p + O(D)$에 끝난다. 항 $W/p$은 온 일의 나란한 몫이다. 항 $O(D)$은 훔치기에 잃는 때를 가둔다(셈틀은 핵심 길 낱덩이마다 많아야 한 번 훔친다). 온 훔치기: 기댓값으로 $O(pD)$번. 나란함이 웬만하면 $D \leq W/p$이므로 $T_p \approx W/p$이 되어 거의 선형으로 빨라진다.

---

**연습문제 3.**
일 훔치기는 고르지 않은 나란함(일감 크기가 제각각)을 어떻게 다루는가?

??? success "연습문제 3 풀이"
    큰 일감이 훔쳐져 다시 갈라지므로 일 훔치기는 고르지 않은 나란함을 자연스럽게 다룬다. 셈틀이 작은 일감을 빨리 끝내면 다른 셈틀 줄의 위쪽에서 큰 일감을 훔친다. 그 큰 일감이 훔친 셈틀에서 다시 갈라져 나란함이 늘어난다. 이 저절로 되는 짐 고르기 덕에 부분 나무 크기가 들쭉날쭉한 나무 꼴 셈(구문 뜯기, 찾기, 그래프 알고리즘)에서 일 훔치기가 뛰어나다.

---

**연습문제 4.**
요즘 깊은 배움 틀에서 일 훔치기는 어떻게 짜여 있는가?

??? success "연습문제 4 풀이"
    PyTorch의 DataLoader은 일 훔치기처럼 도는 일감 줄과 함께 일꾼 여럿을 쓴다. 일꾼마다 앞손질할 다음 묶음을 집어 간다. TensorFlow의 실행기는 연산자 일정 잡기에 일 훔치기를 쓰는 실 못을 쓴다. 실이 연산을 끝내면 준비된 다음 연산을 훔친다. Intel oneDNN은 겹말기와 행렬 곱하기를 여러 CPU 코어에 나란히 하려 일 훔치기가 있는 TBB을 쓴다. 핵심 이점은 일감을 드러나게 맡기지 않고도 짐이 저절로 고르게 되는 것이다.