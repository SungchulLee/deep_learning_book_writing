# Leader Election

Many distributed algorithms require a single coordinator: a process that
initiates BFS, collects results, or breaks symmetry.  Leader election
designates exactly one process as the **leader** without centralized
control.  It is a fundamental building block in distributed computing, and
its message complexity depends heavily on the network topology.

## Problem Definition

Given $n$ processes, each with a unique identifier, the goal is to reach a
state where exactly one process outputs "leader" and all others output
"non-leader."

**Requirements:**

1. **Safety.**  At most one process is elected leader.
2. **Liveness.**  Eventually, exactly one process is elected.
3. **Symmetry breaking.**  Processes use their unique IDs to break
   symmetry (anonymous leader election is impossible in many models).

## Ring Topology

### LCR Algorithm (Le Lann, Chang, Roberts)

The simplest leader election algorithm operates on a **unidirectional ring**
of $n$ processes.

1. Each process sends its ID clockwise.
2. When a process receives an ID:
    - If the received ID is **greater** than its own, forward it.
    - If the received ID **equals** its own, declare itself leader.
    - If the received ID is **smaller**, discard it.

The process with the maximum ID receives its own message after $n$ hops
and becomes the leader.

**Complexity:**

| Metric | Bound |
|---|---|
| Rounds | $n$ |
| Messages (worst case) | $O(n^2)$ |
| Messages (best case) | $O(n)$ |

The worst case occurs when IDs are arranged in decreasing clockwise order:
each of the $n$ messages travels up to $n$ hops.

### Hirschberg-Sinclair Algorithm

This algorithm improves LCR to $O(n \log n)$ messages on a
**bidirectional ring** by using a doubling technique.

**Phase $i$:** Each surviving candidate sends its ID both clockwise and
counterclockwise for $2^i$ hops.

- If the ID reaches $2^i$ hops without encountering a larger ID, it
  "bounces back."
- If the candidate receives its own bounce-back from both directions, it
  survives to phase $i + 1$.
- Otherwise, it becomes a relay (non-candidate).

In each phase, at most half the candidates survive (any two surviving
candidates must be at least $2^i$ apart), so the number of phases is
$O(\log n)$.

**Complexity:**

| Metric | Bound |
|---|---|
| Phases | $O(\log n)$ |
| Messages per phase | $O(n)$ |
| Total messages | $O(n \log n)$ |

!!! note "Lower Bound"
    Any comparison-based leader election algorithm on a ring requires
    $\Omega(n \log n)$ messages, so Hirschberg-Sinclair is asymptotically
    optimal.

## General Network Topology

### Flood-Max Algorithm

On an arbitrary connected graph with diameter $D$:

1. Each process floods its ID to all neighbors for $D$ rounds.
2. After $D$ rounds, each process knows the maximum ID in the network.
3. The process with that ID declares itself leader.

**Complexity:** $O(D)$ rounds, $O(D \cdot m)$ messages where $m = |E|$.

## Simulation

```python
"""
Simulation of LCR leader election on a unidirectional ring.

Time : O(n) rounds
Messages: O(n^2) worst case
"""


# === LCR Leader Election ===
def lcr_leader_election(ids: list[int]) -> int:
    """Simulate LCR on a ring with the given process IDs. Return leader ID."""
    n = len(ids)
    # Each process has an outgoing message buffer
    messages = list(ids)  # Initially, each sends its own ID
    leader = -1

    for _ in range(n):
        new_messages = [0] * n
        for i in range(n):
            msg = messages[i]
            next_proc = (i + 1) % n

            if msg > ids[next_proc]:
                new_messages[next_proc] = msg  # forward
            elif msg == ids[next_proc]:
                leader = msg  # found leader
            # else: discard (smaller ID)
        messages = new_messages

    return leader


# === Hirschberg-Sinclair Simulation ===
def hs_leader_election(ids: list[int]) -> int:
    """Simulate Hirschberg-Sinclair on a bidirectional ring."""
    n = len(ids)
    active = [True] * n
    phase = 0

    while sum(active) > 1:
        dist = 2**phase
        survivors = []
        for i in range(n):
            if not active[i]:
                continue
            # Check if this candidate's ID is max within distance dist
            is_max = True
            for d in range(1, min(dist + 1, n)):
                left = (i - d) % n
                right = (i + d) % n
                if ids[left] > ids[i] or ids[right] > ids[i]:
                    is_max = False
                    break
            if is_max:
                survivors.append(i)

        new_active = [False] * n
        for s in survivors:
            new_active[s] = True
        active = new_active
        phase += 1

    for i in range(n):
        if active[i]:
            return ids[i]
    return -1


# === Example ===
if __name__ == "__main__":
    process_ids = [5, 3, 8, 1, 7, 2]
    print(f"Process IDs: {process_ids}")
    print(f"LCR leader: {lcr_leader_election(process_ids)}")
    print(f"H-S leader: {hs_leader_election(process_ids)}")
```

## Comparison

| Algorithm | Topology | Messages | Rounds |
|---|---|---|---|
| LCR | Unidirectional ring | $O(n^2)$ | $n$ |
| Hirschberg-Sinclair | Bidirectional ring | $O(n \log n)$ | $O(\log n)$ |
| Flood-Max | General graph | $O(D \cdot m)$ | $D$ |

## Reference

- Lynch, N. *Distributed Algorithms*. Morgan Kaufmann, 1996, Chapters 3--4.
- Peleg, D. *Distributed Computing: A Locality-Sensitive Approach*.
  SIAM, 2000.
