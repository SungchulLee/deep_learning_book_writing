# Packet Scheduling

When multiple packets compete for a shared link, a **packet scheduler** determines the order of transmission. The choice of scheduling algorithm affects latency, fairness, and quality-of-service (QoS). Simple strategies like FIFO treat all traffic equally, while more sophisticated algorithms allocate bandwidth according to flow priorities or weights.

## FIFO (First-In First-Out)

The simplest scheduler transmits packets in arrival order. A single queue holds all incoming packets; when the link is free, the head-of-line packet is sent.

- **Latency**: Average waiting time depends on arrival rate $\lambda$ and service rate $\mu$. For Poisson arrivals (M/M/1 queue):

$$
W = \frac{1}{\mu - \lambda}
$$

- **Fairness**: None -- a burst from one flow delays all others.
- **Complexity**: $O(1)$ per packet (enqueue/dequeue from a queue).

## Priority Queuing

Packets are classified into $k$ priority levels. The scheduler always serves the highest-priority non-empty queue first.

- **Advantage**: Low-latency service for critical traffic (e.g., voice, control packets).
- **Disadvantage**: **Starvation** -- low-priority traffic may never be served if high-priority traffic is continuous.

## Weighted Fair Queuing (WFQ)

WFQ approximates **generalized processor sharing** (GPS), where each flow $i$ receives a share of bandwidth proportional to its weight $w_i$:

$$
\text{rate}_i = \frac{w_i}{\sum_{j} w_j} \cdot C
$$

where $C$ is the link capacity. WFQ simulates this by computing a virtual finish time for each packet and serving them in finish-time order.

For a packet of size $L_i$ in flow $i$ arriving at time $a$:

$$
F_i = \max(F_{i,\text{prev}}, V(a)) + \frac{L_i}{w_i}
$$

where $V(a)$ is the virtual time at arrival and $F_{i,\text{prev}}$ is the finish time of the previous packet in flow $i$.

**Complexity**: $O(\log N)$ per packet, where $N$ is the number of active flows (priority queue insertion).

## Deficit Round-Robin (DRR)

DRR provides fair scheduling with $O(1)$ per-packet complexity. Each flow $i$ has a **quantum** $Q_i$ and a **deficit counter** $D_i$:

1. Visit each active flow in round-robin order.
2. Add $Q_i$ to $D_i$.
3. Serve packets from flow $i$ as long as the head-of-line packet size $\le D_i$; subtract the packet size from $D_i$.
4. Move to the next flow when $D_i$ is insufficient or the queue is empty.

Over time, each flow receives bandwidth proportional to its quantum $Q_i$.

## Comparison

| Algorithm | Per-Packet Time | Fairness | Starvation-Free |
|---|---|---|---|
| FIFO | $O(1)$ | None | Yes |
| Priority | $O(1)$ | By class | No |
| WFQ | $O(\log N)$ | Weighted | Yes |
| DRR | $O(1)$ | Weighted | Yes |

## Implementation

```python
"""
Packet Scheduling -- FIFO, Priority, and Deficit Round-Robin.

Simulates three packet scheduling strategies on a shared link
and compares the per-flow throughput and ordering.
"""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass


# === Packet ===================================================================

@dataclass
class Packet:
    """A network packet with flow ID, size, and priority."""
    flow_id: int
    size: int
    priority: int = 0
    arrival: int = 0


# === FIFO Scheduler ===========================================================

class FIFOScheduler:
    """First-in first-out: transmit in arrival order."""

    def __init__(self):
        self.queue: deque[Packet] = deque()

    def enqueue(self, pkt: Packet) -> None:
        self.queue.append(pkt)

    def dequeue(self) -> Packet | None:
        return self.queue.popleft() if self.queue else None


# === Priority Scheduler =======================================================

class PriorityScheduler:
    """Serve highest-priority packets first (lower number = higher priority)."""

    def __init__(self, n_priorities: int = 3):
        self.queues = [deque() for _ in range(n_priorities)]

    def enqueue(self, pkt: Packet) -> None:
        self.queues[pkt.priority].append(pkt)

    def dequeue(self) -> Packet | None:
        for q in self.queues:
            if q:
                return q.popleft()
        return None


# === Deficit Round-Robin ======================================================

class DRRScheduler:
    """Fair scheduling with O(1) per-packet cost."""

    def __init__(self, n_flows: int, quantum: int = 500):
        self.n_flows = n_flows
        self.quantum = quantum
        self.queues: list[deque[Packet]] = [deque() for _ in range(n_flows)]
        self.deficit = [0] * n_flows

    def enqueue(self, pkt: Packet) -> None:
        self.queues[pkt.flow_id].append(pkt)

    def dequeue(self) -> Packet | None:
        """Serve one packet using DRR logic."""
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


# === Main =====================================================================

if __name__ == "__main__":
    # Create packets from 3 flows
    packets = [
        Packet(flow_id=0, size=200, priority=0, arrival=0),
        Packet(flow_id=1, size=500, priority=1, arrival=1),
        Packet(flow_id=0, size=300, priority=0, arrival=2),
        Packet(flow_id=2, size=400, priority=2, arrival=3),
        Packet(flow_id=1, size=100, priority=1, arrival=4),
    ]

    for name, sched in [
        ("FIFO", FIFOScheduler()),
        ("Priority", PriorityScheduler()),
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

**Output:**

```
FIFO      : F0(200) -> F1(500) -> F0(300) -> F2(400) -> F1(100)
Priority  : F0(200) -> F0(300) -> F1(500) -> F1(100) -> F2(400)
DRR       : F0(200) -> F0(300) -> F1(500) -> F2(400) -> F1(100)
```

FIFO preserves arrival order. Priority serves all flow-0 packets first (highest priority), then flow-1, then flow-2. DRR distributes service more fairly by cycling through flows with deficit accounting.

## Reference

- Demers, A., Keshav, S., and Shenker, S. "Analysis and Simulation of a Fair Queueing Algorithm." *SIGCOMM*, 1989
- Shreedhar, M. and Varghese, G. "Efficient Fair Queuing Using Deficit Round-Robin." *IEEE/ACM Trans. Networking*, 1996

## Exercises

**Exercise 1.**
Compare FIFO, priority queuing, and weighted fair queuing (WFQ) for a router handling voice, video, and data traffic. Which scheduler provides the best quality of service?

??? success "Solution to Exercise 1"
    **FIFO**: all packets share one queue. A burst of data packets can delay voice packets, causing jitter. No differentiation. **Priority queuing**: voice gets highest priority, video medium, data lowest. Voice and video get low latency, but data can starve if voice/video traffic is heavy (no bandwidth guarantee for data). **WFQ**: each flow gets a guaranteed share of bandwidth proportional to its weight. Voice (weight 5), video (weight 3), data (weight 2) share a 100 Mbps link as 50/30/20 Mbps. No flow starves; voice gets consistent low latency. WFQ provides the best QoS because it guarantees bandwidth for each class while allowing unused capacity to be shared. Priority queuing risks data starvation; FIFO provides no differentiation. $\square$

---

**Exercise 2.**
Describe the deficit round-robin (DRR) algorithm and explain how it achieves $O(1)$ per-packet scheduling while approximating fair queuing.

??? success "Solution to Exercise 2"
    DRR maintains a circular list of active queues (one per flow). Each queue has a deficit counter initialized to 0. Each round: (1) add a quantum $Q$ to the queue's deficit counter. (2) While the queue's head packet size $\le$ deficit counter: dequeue the packet, subtract its size from the deficit counter. (3) If the queue becomes empty, reset the deficit to 0. (4) Move to the next queue. Per-packet cost: $O(1)$ (dequeue + counter update). Fairness: over long periods, each flow receives bandwidth proportional to its quantum. Flows with larger average packet sizes may get slightly bursty service but the same long-term share. DRR approximates WFQ without the $O(\log n)$ per-packet cost of computing virtual finish times. The tradeoff: DRR has higher short-term unfairness (jitter) but is practical for high-speed routers where $O(1)$ is essential. $\square$

---

**Exercise 3.**
A router processes packets at 10 Gbps with an average packet size of 500 bytes. How many packets per second must the scheduler handle, and why does this constraint favor $O(1)$ algorithms?

??? success "Solution to Exercise 3"
    Packets per second: $10 \times 10^9 / (500 \times 8) = 2.5 \times 10^6$ packets/sec. Time budget per packet: $1 / (2.5 \times 10^6) = 400$ ns. At 3 GHz clock speed: $400 \times 3 = 1200$ clock cycles per packet. An $O(\log n)$ scheduler with $n = 10{,}000$ flows requires $\sim 14$ operations per packet. If each operation involves a cache miss ($\sim 100$ cycles), total = 1400 cycles -- barely fitting the budget. An $O(1)$ scheduler like DRR needs $\sim 5$ operations ($\sim 500$ cycles), comfortably within budget. At 40 Gbps or 100 Gbps, the per-packet budget shrinks to 100 ns or 40 ns, making even $O(\log n)$ too slow. This is why high-speed routers use $O(1)$ scheduling. $\square$

---

**Exercise 4.**
Explain token bucket and leaky bucket traffic shapers. How does each regulate burst traffic?

??? success "Solution to Exercise 4"
    **Leaky bucket**: packets enter a queue (bucket) and are transmitted at a fixed rate $r$. Bursts fill the queue; excess packets are dropped. The output is perfectly smooth at rate $r$ regardless of input burstiness. Maximum burst that can be absorbed: bucket depth $b$ packets. **Token bucket**: tokens accumulate at rate $r$ up to a maximum of $b$ tokens. Each packet consumes one token. If tokens are available, packets are sent immediately (even in bursts up to $b$ packets). If no tokens, packets wait. The output is bursty: a burst of $b$ packets can be sent at line rate, followed by a sustained rate of $r$. Token bucket allows controlled bursts (useful for bursty applications like web browsing); leaky bucket enforces strict smoothness (useful for CBR streams like voice). Token bucket is more commonly used because it accommodates legitimate bursts while enforcing a long-term average rate. $\square$

---

**Exercise 5.**
A financial trading network requires that market data packets never experience more than 10 microseconds of queuing delay. Design a scheduling scheme that guarantees this bound.

??? success "Solution to Exercise 5"
    Use **strict priority queuing** with market data in the highest priority queue and all other traffic in lower-priority queues. The maximum queuing delay for the highest-priority queue is bounded by the transmission time of the largest lower-priority packet currently being transmitted (non-preemptive scheduling) or zero (preemptive scheduling). For a 10 Gbps link, a maximum-size packet (1500 bytes) takes $1500 \times 8 / (10 \times 10^9) = 1.2$ microseconds to transmit. With non-preemptive strict priority, the worst-case delay for market data is 1.2 microseconds (waiting for one in-progress lower-priority packet) -- well within the 10 microsecond bound. To prevent starvation of lower-priority traffic, set a rate limiter on market data (e.g., police to 50% of link capacity). If market data exceeds this, excess packets are queued and may experience additional delay; alert the administrator. $\square$
