# Consensus

In a distributed system, processes must sometimes agree on a single value
despite communication delays and potential failures.  This is the
**consensus** problem, fundamental to replicated state machines, atomic
broadcast, and blockchain protocols.  Its impossibility in certain models
(the FLP result) is one of the most important theorems in distributed
computing.

## Problem Definition

A set of $n$ processes, each starting with an **input value** $v_i$, must
satisfy three properties:

1. **Agreement.**  All non-faulty processes decide the same value.
2. **Validity.**  The decided value is some process's input value.
3. **Termination.**  Every non-faulty process eventually decides.

## Failure Models

The difficulty of consensus depends on the failure model:

| Model | Description | Consensus solvable? |
|---|---|---|
| No failures | All processes are correct | Trivially yes |
| Crash failures | A process may stop permanently | Yes (synchronous), No (async, FLP) |
| Byzantine failures | A process may behave arbitrarily | Yes if $n > 3f$ |

Here $f$ denotes the maximum number of faulty processes.

## FLP Impossibility

Fischer, Lynch, and Paterson (1985) proved:

> **Theorem (FLP).**  In an asynchronous message-passing system, there is no
> deterministic consensus protocol that tolerates even a single crash failure.

### Proof Intuition

The proof shows that any deterministic protocol has a **bivalent** initial
configuration---one from which both $0$ and $1$ are still possible
decisions.  Because message delivery order is arbitrary in an asynchronous
system, an adversarial scheduler can always delay the critical message that
would tip the system to a decision, keeping it bivalent forever.

!!! warning "FLP Does Not Mean Consensus Is Impossible in Practice"
    FLP rules out *deterministic* protocols in *purely asynchronous* systems.
    Practical systems circumvent it via partial synchrony assumptions
    (Paxos, Raft), randomization, or failure detectors.

## Synchronous Consensus

In a synchronous system with at most $f$ crash failures, consensus can be
solved in $f + 1$ rounds.

### Algorithm (Flood-Set)

Each process maintains a set $W_i$ of values it has seen.

1. Initialize $W_i = \{v_i\}$.
2. For $f + 1$ rounds: broadcast $W_i$ to all processes, then set
   $W_i = W_i \cup \bigcup_{j} W_j$ (where $W_j$ are received sets).
3. Decide $\min(W_i)$.

After $f + 1$ rounds, at least one round had no failures (pigeonhole: $f$
failures across $f + 1$ rounds), so in that round all surviving processes
received identical sets.

**Complexity:**  $f + 1$ rounds, $O(n^2 f)$ messages.

## Paxos (Partial Synchrony)

Lamport's Paxos protocol achieves consensus under partial synchrony---the
system is eventually synchronous after some unknown time $T$.

### Roles

- **Proposers** suggest values.
- **Acceptors** vote on proposals.
- **Learners** learn the decided value.

### Two Phases

**Phase 1 (Prepare).**  A proposer selects a proposal number $n$ and sends
`PREPARE(n)` to a majority of acceptors.  Each acceptor replies with the
highest-numbered proposal it has accepted (if any) and promises not to
accept proposals with numbers less than $n$.

**Phase 2 (Accept).**  If the proposer receives promises from a majority,
it sends `ACCEPT(n, v)` where $v$ is the value from the highest-numbered
accepted proposal (or the proposer's own value if none).  Acceptors accept
the proposal if they have not promised a higher number.

A value is **decided** when a majority of acceptors accept it.

!!! note "Safety and Liveness"
    Paxos guarantees **safety** (agreement and validity) in all executions.
    **Liveness** (termination) requires eventual synchrony; competing
    proposers can livelock without a leader election mechanism.

## Byzantine Consensus

When up to $f$ processes may behave arbitrarily (send conflicting messages,
lie, etc.), consensus requires $n > 3f$ processes.

### Lower Bound

**Theorem.**  No protocol solves Byzantine consensus with $n \le 3f$.

**Proof sketch (for $n = 3, f = 1$).**  Consider three processes $A$, $B$,
$C$ where one is Byzantine.  If $A$ proposes $0$ and $C$ proposes $1$, the
Byzantine process $B$ can tell $A$ it proposes $0$ and tell $C$ it proposes
$1$.  Neither $A$ nor $C$ can distinguish this from a legitimate scenario,
making agreement impossible.

### Practical Byzantine Fault Tolerance (PBFT)

PBFT (Castro & Liskov, 1999) achieves Byzantine consensus with $n = 3f + 1$
in a partially synchronous network using three communication phases:
pre-prepare, prepare, and commit.

## Reference

- Fischer, M. J., Lynch, N. A., & Paterson, M. S. "Impossibility of
  Distributed Consensus with One Faulty Process." *JACM*, 1985.
- Lamport, L. "Paxos Made Simple." ACM SIGACT News, 2001.
- Lynch, N. *Distributed Algorithms*. Morgan Kaufmann, 1996.
