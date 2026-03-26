# Load Balancing

When a system receives more requests than a single server can handle, a **load balancer** distributes incoming traffic across multiple backend servers. The goal is to maximize throughput, minimize response time, and avoid overloading any single server. Load balancing algorithms range from simple static policies like round-robin to adaptive strategies that respond to real-time server health.

## Problem Formulation

Given $n$ servers with capacities $c_1, c_2, \ldots, c_n$ and a stream of requests with loads $w_1, w_2, \ldots$, assign each request to a server to minimize the **maximum load** (makespan):

$$
\min \max_{j=1}^{n} \sum_{i \in S_j} w_i
$$

where $S_j$ is the set of requests assigned to server $j$. This is equivalent to the multiprocessor scheduling problem, which is NP-hard in general, so practical load balancers use heuristic policies.

## Static Algorithms

### Round-Robin

Assign requests to servers in cyclic order: request $i$ goes to server $(i \bmod n)$.

- **Time complexity**: $O(1)$ per request.
- **Advantage**: Simple, no state required.
- **Disadvantage**: Ignores server load and request weight; unequal work leads to imbalance.

### Weighted Round-Robin

Assign servers weights $c_1, \ldots, c_n$ proportional to capacity. Server $j$ receives $c_j$ requests per cycle.

$$
\text{fraction of traffic to server } j = \frac{c_j}{\sum_{k=1}^{n} c_k}
$$

### Hash-Based

Hash the request key (e.g., client IP, session ID) to select a server:

$$
\text{server}(r) = h(r) \bmod n
$$

This ensures the same client always reaches the same server (session affinity), but adding or removing servers remaps nearly all keys.

!!! tip "Consistent hashing"
    Consistent hashing maps both servers and keys to a ring of size $2^{32}$, so adding or removing a server remaps only $O(K/n)$ keys on average (where $K$ is the total number of keys). This is the foundation of many distributed caches and CDNs.

## Dynamic Algorithms

### Least Connections

Route each new request to the server with the fewest active connections:

$$
\text{server}^* = \arg\min_{j=1}^{n} \text{active}_j
$$

This adapts to varying request durations and is widely used in HTTP load balancers.

### Weighted Least Connections

Combine capacity weights with connection counts:

$$
\text{server}^* = \arg\min_{j=1}^{n} \frac{\text{active}_j}{c_j}
$$

Servers with higher capacity weight absorb more connections before being considered "loaded."

### Least Response Time

Route to the server with the lowest recent average response time, directly optimizing user experience. Requires monitoring infrastructure to track per-server latency.

## Power of Two Choices

A celebrated result in randomized load balancing: instead of assigning each request to a single random server, pick **two servers at random** and send the request to the less loaded one.

With $n$ servers and $m = n$ requests (one per server on average), the maximum load under purely random assignment is:

$$
\Theta\!\left(\frac{\log n}{\log \log n}\right)
$$

With the two-choice strategy, the maximum load drops exponentially to:

$$
\frac{\log \log n}{\log 2} + O(1)
$$

The intuition is that under pure random assignment, unlucky collisions pile up on the most loaded server; with two choices, the algorithm avoids the worst-loaded option, preventing the pile-up from growing. This exponential improvement comes at negligible extra cost (one additional probe per request).

## Health Checks

Load balancers must detect and remove unhealthy servers:

- **Active checks**: Periodically send probe requests (HTTP GET, TCP SYN).
- **Passive checks**: Monitor response codes and latencies from real traffic.
- **Circuit breaker**: Remove a server after $k$ consecutive failures; re-add after recovery.

## Implementation

```python
"""
Load Balancing -- round-robin, least-connections, and power-of-two-choices.

Simulates different load balancing strategies and compares the
resulting load distributions across servers.
"""

import random


# === Round-Robin ==============================================================

class RoundRobinBalancer:
    """Cycle through servers in order."""

    def __init__(self, n_servers: int):
        self.n_servers = n_servers
        self.counter = 0

    def select(self) -> int:
        """Return the next server index."""
        server = self.counter % self.n_servers
        self.counter += 1
        return server


# === Least Connections ========================================================

class LeastConnectionsBalancer:
    """Route to the server with fewest active connections."""

    def __init__(self, n_servers: int):
        self.n_servers = n_servers
        self.active = [0] * n_servers

    def select(self) -> int:
        """Return server with minimum active connections."""
        server = min(range(self.n_servers), key=lambda j: self.active[j])
        self.active[server] += 1
        return server

    def release(self, server: int) -> None:
        """Mark a connection as completed."""
        self.active[server] -= 1


# === Power of Two Choices =====================================================

class TwoChoiceBalancer:
    """Pick two random servers, send to the less loaded one."""

    def __init__(self, n_servers: int):
        self.n_servers = n_servers
        self.load = [0] * n_servers

    def select(self) -> int:
        """Return the less loaded of two randomly chosen servers."""
        a, b = random.sample(range(self.n_servers), 2)
        server = a if self.load[a] <= self.load[b] else b
        self.load[server] += 1
        return server


# === Simulation ===============================================================

def simulate(balancer, n_requests: int, n_servers: int) -> list[int]:
    """Run a simulation and return per-server request counts."""
    counts = [0] * n_servers
    for _ in range(n_requests):
        server = balancer.select()
        counts[server] += 1
    return counts


# === Main =====================================================================

if __name__ == "__main__":
    random.seed(42)
    n_servers = 10
    n_requests = 10000

    for name, balancer in [
        ("Round-Robin", RoundRobinBalancer(n_servers)),
        ("Least-Connections", LeastConnectionsBalancer(n_servers)),
        ("Two-Choice", TwoChoiceBalancer(n_servers)),
    ]:
        counts = simulate(balancer, n_requests, n_servers)
        max_load = max(counts)
        min_load = min(counts)
        print(f"{name:20s}  max={max_load}  min={min_load}  "
              f"imbalance={max_load - min_load}")
```

## Reference

- Mitzenmacher, M. "The Power of Two Choices in Randomized Load Balancing." *IEEE TPDS*, 2001
- [Designing Data-Intensive Applications (Kleppmann)](https://dataintensive.net/)
- Karger, D. et al. "Consistent Hashing and Random Trees." *STOC*, 1997
