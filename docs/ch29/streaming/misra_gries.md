# Misra-Gries Algorithm

Suppose a stream of $n$ elements arrives one by one, and we want to find
items that appear more than $n/k$ times---the **heavy hitters**.  Storing
all counts requires memory proportional to the number of distinct elements,
which may be enormous.  The Misra-Gries algorithm solves this with only
$k - 1$ counters, using $O(k)$ space regardless of stream length.

## Problem Statement

Given a stream $a_1, a_2, \dots, a_n$ over a universe $\mathcal{U}$ and
a parameter $k \ge 2$, find all elements whose frequency exceeds $n/k$.

There are at most $k - 1$ such elements (since $k$ elements each exceeding
$n/k$ would require more than $n$ total occurrences).

## Algorithm

Maintain a set of at most $k - 1$ candidate-counter pairs
$D = \{(e_i, c_i)\}$.

For each arriving element $a$:

1. **If $a \in D$:** increment its counter, $c_a \leftarrow c_a + 1$.
2. **Else if $|D| < k - 1$:** insert $(a, 1)$ into $D$.
3. **Else:** decrement all counters by $1$ and remove any with count $0$.

After the stream ends, the candidates in $D$ are the potential heavy
hitters.  A second pass (or approximate acceptance) confirms which ones
truly exceed the threshold.

!!! note "Decrement Intuition"
    Each decrement operation conceptually "cancels" $k$ distinct elements
    (the new arrival plus the $k - 1$ candidates).  A true heavy hitter
    with frequency $> n/k$ survives because there are not enough other
    elements to cancel all its occurrences.

## Correctness Guarantee

**Theorem.**  Every element with frequency $f_e > n/k$ appears in the
final set $D$.

**Proof sketch.**  Each decrement operation reduces every counter by $1$
and removes the new element.  The total number of decrements across all
operations is at most $n/k$ (since each decrement consumes $k$ elements
from the stream).  An element with $f_e > n/k$ occurrences has its counter
decremented fewer than $f_e$ times, so its counter remains positive.
$\square$

The algorithm may also retain elements with $f_e \le n/k$ (false positives),
but it never misses a true heavy hitter (no false negatives).

## Error Bound

After processing the stream, the estimated count $\hat{f}_e$ for an
element $e$ in $D$ satisfies:

$$
f_e - \frac{n}{k} \le \hat{f}_e \le f_e
$$

The true count is never overestimated, and the underestimate is at most
$n/k$.

## Implementation

```python
"""
Misra-Gries algorithm for finding frequent items in a stream.

Space: O(k)
Time : O(n) amortized with hash map implementation
"""


# === Misra-Gries ===
class MisraGries:
    """Find elements with frequency > n/k using k-1 counters."""

    def __init__(self, k: int):
        self.k = k
        self.counters: dict[str, int] = {}

    def process(self, item: str) -> None:
        """Process one stream element."""
        if item in self.counters:
            self.counters[item] += 1
        elif len(self.counters) < self.k - 1:
            self.counters[item] = 1
        else:
            # Decrement all counters and remove zeros
            to_remove = []
            for key in self.counters:
                self.counters[key] -= 1
                if self.counters[key] == 0:
                    to_remove.append(key)
            for key in to_remove:
                del self.counters[key]

    def get_candidates(self) -> dict[str, int]:
        """Return candidate heavy hitters with their estimated counts."""
        return dict(self.counters)


# === Verification Pass ===
def verify_heavy_hitters(
    stream: list[str], candidates: dict[str, int], k: int
) -> dict[str, int]:
    """Second pass to get exact counts for candidates."""
    counts: dict[str, int] = {c: 0 for c in candidates}
    n = len(stream)
    for item in stream:
        if item in counts:
            counts[item] += 1
    return {item: count for item, count in counts.items() if count > n // k}


# === Example ===
if __name__ == "__main__":
    stream = ["a", "b", "a", "c", "a", "b", "a", "d", "a", "b"]
    k = 3  # Find elements with frequency > 10/3 ~ 3.33

    mg = MisraGries(k)
    for item in stream:
        mg.process(item)

    candidates = mg.get_candidates()
    print(f"Candidates: {candidates}")

    confirmed = verify_heavy_hitters(stream, candidates, k)
    print(f"Confirmed heavy hitters (freq > {len(stream)//k}): {confirmed}")
```

## Complexity

| Aspect | Bound |
|---|---|
| Space | $O(k)$ counters |
| Time per element | $O(1)$ amortized |
| Total time | $O(n)$ |
| False negatives | 0 (guaranteed) |
| False positives | At most $k - 1$ |

## Comparison with Related Algorithms

| Algorithm | Space | Counts | Deterministic |
|---|---|---|---|
| Misra-Gries | $O(k)$ | Approximate | Yes |
| Count-Min Sketch | $O(w \times d)$ | Approximate (overestimates) | No (hash-based) |
| Space-Saving | $O(k)$ | Approximate | Yes |

!!! tip "Space-Saving Extension"
    The Space-Saving algorithm (Metwally et al.) is equivalent to
    Misra-Gries but replaces the minimum-count candidate instead of
    decrementing all counters, providing tighter error bounds in practice.

## Reference

- Misra, J. & Gries, D. "Finding repeated elements." *Science of Computer
  Programming*, 2(2), 1982.
- Muthukrishnan, S. "Data Streams: Algorithms and Applications." Foundations
  and Trends in Theoretical Computer Science, 2005.
