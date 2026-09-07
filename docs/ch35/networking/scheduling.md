# 꾸러미 차례 잡기

함께 쓰는 이음줄을 놓고 여러 꾸러미가 겨룰 때 **꾸러미 차례잡이**가 보내는 차례를 정한다. 어떤 차례 잡기 알고리즘을 고르느냐가 늦음, 고름, 그리고 서비스 됨됨이(QoS)를 가른다. FIFO 같은 단순한 꾀는 모든 오감을 똑같이 다루고, 더 촘촘한 알고리즘은 흐름의 앞섬이나 짐에 맞추어 너비를 나눈다.

## FIFO(먼저 든 것이 먼저 나감)

가장 단순한 차례잡이는 꾸러미를 온 차례대로 보낸다. 줄 하나가 들어오는 꾸러미를 모두 담고, 이음줄이 비면 줄 머리의 꾸러미를 보낸다.

- **늦음**: 고르게 기다리는 때는 오는 빠르기 $\lambda$과 다루는 빠르기 $\mu$에 달렸다. 푸아송 옴(M/M/1 줄)이면

$$
W = \frac{1}{\mu - \lambda}
$$

- **고름**: 없다. 한 흐름이 몰아치면 나머지가 모두 늦어진다.
- **복잡도**: 꾸러미마다 $O(1)$(줄에 넣고 빼기).

## 앞섬 줄 세우기

꾸러미를 앞섬 켜 $k$개로 가른다. 차례잡이는 늘 비지 않은 줄 가운데 가장 앞선 것을 먼저 다룬다.

- **나은 점**: 목숨이 걸린 오감(보기로 목소리, 다스림 꾸러미)에 늦음이 적은 서비스를 준다.
- **나쁜 점**: **굶주림**. 앞선 오감이 끊이지 않으면 뒤진 오감이 끝내 다뤄지지 않을 수 있다.

## 짐 실은 고른 줄 세우기(WFQ)

WFQ은 **두루 쓰는 다룸꾼 나눔**(GPS)에 가깝게 다가간다. GPS에서는 흐름 $i$마다 제 짐 $w_i$에 비례해 너비를 받는다.

$$
\text{빠르기}_i = \frac{w_i}{\sum_{j} w_j} \cdot C
$$

여기서 $C$은 이음줄이 담는 크기다. WFQ은 꾸러미마다 헛 마침 때를 셈하고 마침 때 차례로 다루어 이를 흉내낸다.

때 $a$에 온 흐름 $i$의 크기 $L_i$짜리 꾸러미에 대해

$$
F_i = \max(F_{i,\text{앞}}, V(a)) + \frac{L_i}{w_i}
$$

여기서 $V(a)$은 올 때의 헛 때이고 $F_{i,\text{앞}}$은 흐름 $i$의 앞 꾸러미의 마침 때다.

**복잡도**: 꾸러미마다 $O(\log N)$이고 $N$은 살아 있는 흐름의 수다(앞선 줄에 넣기).

## 모자람 돌아가며(DRR)

DRR은 꾸러미마다 $O(1)$의 복잡도로 고른 차례 잡기를 이룬다. 흐름 $i$마다 **몫** $Q_i$과 **모자람 세개** $D_i$을 지닌다.

1. 살아 있는 흐름을 돌아가며 들른다.
2. $D_i$에 $Q_i$을 더한다.
3. 줄 머리의 꾸러미 크기가 $D_i$ 아래인 동안 흐름 $i$의 꾸러미를 다루고, 꾸러미 크기를 $D_i$에서 뺀다.
4. $D_i$이 모자라거나 줄이 비면 다음 흐름으로 넘어간다.

때가 흐르면 흐름마다 제 몫 $Q_i$에 비례하는 너비를 받는다.

## 견주기

| 알고리즘 | 꾸러미마다의 때 | 고름 | 굶주림 없음 |
|---|---|---|---|
| FIFO | $O(1)$ | 없음 | 예 |
| 앞섬 | $O(1)$ | 갈래별 | 아니오 |
| WFQ | $O(\log N)$ | 짐대로 | 예 |
| DRR | $O(1)$ | 짐대로 | 예 |

## 짜보기

```python
"""
꾸러미 차례 잡기 -- FIFO, 앞섬, 모자람 돌아가며.

함께 쓰는 이음줄에서 꾸러미 차례 잡기 꾀 셋을 흉내내고
흐름마다의 나름과 차례를 견준다.
"""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass


# === 꾸러미 =================================================================

@dataclass
class Packet:
    """흐름 번호, 크기, 앞섬을 지닌 그물 꾸러미."""
    flow_id: int
    size: int
    priority: int = 0
    arrival: int = 0


# === FIFO 차례잡이 ==========================================================

class FIFOScheduler:
    """먼저 든 것이 먼저 나감: 온 차례대로 보낸다."""

    def __init__(self):
        self.queue: deque[Packet] = deque()

    def enqueue(self, pkt: Packet) -> None:
        self.queue.append(pkt)

    def dequeue(self) -> Packet | None:
        return self.queue.popleft() if self.queue else None


# === 앞섬 차례잡이 ==========================================================

class PriorityScheduler:
    """가장 앞선 꾸러미를 먼저 다룬다(번호가 작을수록 앞선다)."""

    def __init__(self, n_priorities: int = 3):
        self.queues = [deque() for _ in range(n_priorities)]

    def enqueue(self, pkt: Packet) -> None:
        self.queues[pkt.priority].append(pkt)

    def dequeue(self) -> Packet | None:
        for q in self.queues:
            if q:
                return q.popleft()
        return None


# === 모자람 돌아가며 ========================================================

class DRRScheduler:
    """꾸러미마다 O(1) 값으로 하는 고른 차례 잡기."""

    def __init__(self, n_flows: int, quantum: int = 500):
        self.n_flows = n_flows
        self.quantum = quantum
        self.queues: list[deque[Packet]] = [deque() for _ in range(n_flows)]
        self.deficit = [0] * n_flows

    def enqueue(self, pkt: Packet) -> None:
        self.queues[pkt.flow_id].append(pkt)

    def dequeue(self) -> Packet | None:
        """DRR 결에 따라 꾸러미 하나를 다룬다."""
        for _ in range(self.n_flows):
            for i in range(self.n_flows):
                self.deficit[i] += self.quantum
                while self.queues[i]:
                    pkt = self.queues[i][0]
                    if pkt.size <= self.deficit[i]:
                        self.deficit[i] -= pkt.size
                        return self.queues[i].popleft()
                    else:
                        break
                if not self.queues[i]:
                    self.deficit[i] = 0
        return None


# === 메인 ===================================================================

if __name__ == "__main__":
    # 흐름 셋에서 온 꾸러미를 짓는다
    packets = [
        Packet(flow_id=0, size=200, priority=0, arrival=0),
        Packet(flow_id=1, size=500, priority=1, arrival=1),
        Packet(flow_id=0, size=300, priority=0, arrival=2),
        Packet(flow_id=2, size=400, priority=2, arrival=3),
        Packet(flow_id=1, size=100, priority=1, arrival=4),
    ]

    for name, sched in [
        ("FIFO", FIFOScheduler()),
        ("앞섬", PriorityScheduler()),
        ("DRR", DRRScheduler(n_flows=3)),
    ]:
        for pkt in packets:
            sched.enqueue(pkt)
        order = []
        while True:
            pkt = sched.dequeue()
            if pkt is None:
                break
            order.append(f"F{pkt.flow_id}({pkt.size})")
        print(f"{name:10s}: {' -> '.join(order)}")
```

**내놓기:**

```
FIFO      : F0(200) -> F1(500) -> F0(300) -> F2(400) -> F1(100)
앞섬        : F0(200) -> F0(300) -> F1(500) -> F1(100) -> F2(400)
DRR       : F0(200) -> F0(300) -> F1(500) -> F2(400) -> F1(100)
```

FIFO은 온 차례를 지킨다. 앞섬은 흐름 0의 꾸러미를 모두 먼저(가장 앞서므로) 다루고 흐름 1, 흐름 2 차례로 넘어간다. DRR은 모자람을 셈하며 흐름을 돌아가므로 서비스를 더 고르게 나눈다.

## 살펴볼 거리

- Demers, A., Keshav, S., and Shenker, S. "Analysis and Simulation of a Fair Queueing Algorithm." *SIGCOMM*, 1989
- Shreedhar, M. and Varghese, G. "Efficient Fair Queuing Using Deficit Round-Robin." *IEEE/ACM Trans. Networking*, 1996

## 익힘 문제

**익힘 1.**
목소리, 그림, 자료 오감을 다루는 길잡이에 대해 FIFO, 앞섬 줄 세우기, 짐 실은 고른 줄 세우기(WFQ)를 견주어라. 어느 차례잡이가 가장 나은 서비스 됨됨이를 주는가?

??? success "익힘 1 풀이"
    **FIFO**: 모든 꾸러미가 줄 하나를 함께 쓴다. 자료 꾸러미가 몰아치면 목소리 꾸러미가 늦어져 떨림이 생긴다. 가름이 없다. **앞섬 줄 세우기**: 목소리가 가장 앞서고 그림이 가운데, 자료가 가장 뒤진다. 목소리와 그림은 늦음이 적지만, 목소리·그림 오감이 무거우면 자료가 굶을 수 있다(자료에 너비를 보장하지 않는다). **WFQ**: 흐름마다 제 짐에 비례하는 너비를 보장받는다. 목소리(짐 5), 그림(짐 3), 자료(짐 2)가 100 Mbps 이음줄을 50/30/20 Mbps로 나눈다. 굶는 흐름이 없고 목소리의 늦음이 한결같이 적다. 갈래마다 너비를 보장하면서도 남는 크기를 함께 쓰게 하므로 WFQ이 가장 나은 QoS를 준다. 앞섬 줄 세우기는 자료가 굶을 낌새가 있고, FIFO은 가름이 아예 없다. $\square$

---

**익힘 2.**
모자람 돌아가며(DRR) 알고리즘을 밝히고, 고른 줄 세우기에 다가가면서도 꾸러미마다 $O(1)$인 차례 잡기를 어떻게 이루는지 밝혀라.

??? success "익힘 2 풀이"
    DRR은 살아 있는 줄(흐름마다 하나)의 고리 목록을 지닌다. 줄마다 모자람 세개를 두고 0에서 시작한다. 돌이마다 이렇게 한다. (1) 줄의 모자람 세개에 몫 $Q$을 더한다. (2) 줄 머리 꾸러미의 크기가 모자람 세개 아래인 동안 꾸러미를 빼내고 그 크기를 모자람 세개에서 뺀다. (3) 줄이 비면 모자람을 0으로 되돌린다. (4) 다음 줄로 넘어간다. 꾸러미마다의 값은 $O(1)$이다(빼기 + 세개 고치기). 고름은 이렇다. 오래 두고 보면 흐름마다 제 몫에 비례하는 너비를 받는다. 고르게 꾸러미가 큰 흐름은 서비스가 조금 몰아칠 수 있으나 오래 두고 본 몫은 같다. DRR은 헛 마침 때를 셈하는 WFQ의 꾸러미마다 $O(\log n)$ 값 없이 WFQ에 가깝게 간다. 맞바꿈으로 DRR은 짧게 보면 덜 고르지만(떨림), $O(1)$이 꼭 있어야 하는 빠른 길잡이에서 쓸 만하다. $\square$

---

**익힘 3.**
길잡이가 10 Gbps로 꾸러미를 다루고 꾸러미가 고르게 500바이트다. 차례잡이가 초마다 몇 개의 꾸러미를 다뤄야 하며, 이 매임이 왜 $O(1)$ 알고리즘을 반기는가?

??? success "익힘 3 풀이"
    초마다의 꾸러미 수는 $10 \times 10^9 / (500 \times 8) = 2.5 \times 10^6$개다. 꾸러미마다 쓸 수 있는 때는 $1 / (2.5 \times 10^6) = 400$ 나노초다. 3 GHz 시계면 꾸러미마다 $400 \times 3 = 1200$ 시계 걸음이다. 흐름이 $n = 10{,}000$개인 $O(\log n)$ 차례잡이는 꾸러미마다 $\sim 14$번의 연산이 든다. 연산마다 갈무리를 빗맞으면($\sim 100$걸음) 모두 1400걸음이니 쓸 수 있는 때에 겨우 든다. DRR 같은 $O(1)$ 차례잡이는 $\sim 5$번의 연산($\sim 500$걸음)이면 되어 넉넉히 든다. 40 Gbps이나 100 Gbps에서는 꾸러미마다 쓸 수 있는 때가 100 나노초나 40 나노초로 줄어 $O(\log n)$조차 느려진다. 빠른 길잡이가 $O(1)$ 차례 잡기를 쓰는 까닭이다. $\square$

---

**익힘 4.**
표 두레박과 새는 두레박 오감 다듬이를 밝혀라. 저마다 몰아치는 오감을 어떻게 다스리는가?

??? success "익힘 4 풀이"
    **새는 두레박**: 꾸러미가 줄(두레박)에 들어가 붙박이 빠르기 $r$로 나간다. 몰아치면 줄이 차고 넘치는 꾸러미는 버려진다. 들임이 아무리 몰아쳐도 내놓기는 빠르기 $r$로 매끈하다. 받아 낼 수 있는 가장 큰 몰아침은 두레박 깊이인 꾸러미 $b$개다. **표 두레박**: 표가 빠르기 $r$로 쌓여 많아야 $b$개까지 모인다. 꾸러미마다 표 하나를 쓴다. 표가 있으면 꾸러미를 바로 보낸다(꾸러미 $b$개까지 몰아쳐도 된다). 표가 없으면 꾸러미가 기다린다. 내놓기는 몰아친다. 꾸러미 $b$개를 줄 빠르기로 몰아 보낸 다음 빠르기 $r$로 이어간다. 표 두레박은 다스린 몰아침을 받아 주고(웹 둘러보기처럼 몰아치는 쓰임에 이롭다), 새는 두레박은 매끈함을 굳게 지킨다(목소리 같은 붙박이 빠르기 흐름에 이롭다). 오래 두고 본 고른 빠르기를 지키면서도 마땅한 몰아침을 받아 주므로 표 두레박을 더 널리 쓴다. $\square$

---

**익힘 5.**
어느 금융 거래 그물은 저자 자료 꾸러미가 줄에서 기다리는 늦음이 10 마이크로초를 넘지 않기를 바란다. 이 울타리를 보장하는 차례 잡기 얼개를 꾸며라.

??? success "익힘 5 풀이"
    저자 자료를 가장 앞선 줄에, 나머지 오감을 뒤진 줄에 두는 **엄한 앞섬 줄 세우기**를 쓴다. 가장 앞선 줄이 줄에서 기다리는 가장 긴 늦음은 지금 보내고 있는 뒤진 꾸러미 가운데 가장 큰 것을 보내는 때로 매인다(가로채지 않는 차례 잡기). 가로채는 차례 잡기라면 0이다. 10 Gbps 이음줄에서 가장 큰 꾸러미(1500바이트)를 보내는 데 $1500 \times 8 / (10 \times 10^9) = 1.2$ 마이크로초가 든다. 가로채지 않는 엄한 앞섬이면 저자 자료의 가장 나쁜 늦음이 1.2 마이크로초이므로(보내고 있던 뒤진 꾸러미 하나를 기다린다) 10 마이크로초 울타리 안에 넉넉히 든다. 뒤진 오감이 굶지 않도록 저자 자료에 빠르기 한도를 건다(보기로 이음줄 크기의 50%로 잡는다). 저자 자료가 이를 넘으면 넘치는 꾸러미가 줄에 서서 늦음이 더 붙을 수 있으니 다스리는 이에게 알린다. $\square$
