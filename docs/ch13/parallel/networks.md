# Sorting Networks

Standard sorting algorithms make data-dependent decisions: which elements to compare
next depends on the results of previous comparisons.  A **sorting network** removes
this dependency by specifying a fixed sequence of comparisons at compile time.  The
same set of compare-and-swap operations is performed regardless of the input values.
This makes sorting networks the natural abstraction for parallel and hardware sorting,
where all comparators at the same "depth" can execute simultaneously.

## Comparators

The basic building block of a sorting network is the **comparator**, a device that
takes two inputs and outputs them in sorted order:

$$
\text{comparator}(a, b) = (\min(a, b),\; \max(a, b))
$$

A comparator connecting wires $i$ and $j$ (with $i < j$) swaps the values on those
wires if the value on wire $i$ is greater than the value on wire $j$.

## Sorting Network Definition

A sorting network for $n$ elements is a pair $(n, S)$ where $S$ is a sequence of
comparators $\{(i_1, j_1), (i_2, j_2), \dots, (i_m, j_m)\}$ with $0 \le i_k < j_k < n$.

Two key metrics characterize a sorting network:

| Metric | Definition |
|--------|-----------|
| **Size** | Total number of comparators $m$ |
| **Depth** | Number of parallel steps (comparators on disjoint wires can execute simultaneously) |

Depth determines the parallel time complexity, while size determines the total work.

## The Zero-One Principle

The **zero-one principle** is the fundamental tool for proving that a sorting network
is correct.

**Theorem.** If a comparator network sorts every input sequence of 0s and 1s correctly,
then it sorts every input sequence of arbitrary values correctly.

*Proof sketch.* Suppose the network fails on some input $a_0, \dots, a_{n-1}$.  Then
there exist indices $i < j$ such that $a_i > a_j$ in the output.  Define
$f(x) = \mathbf{1}[x \ge a_j]$.  Since $f$ is monotone, applying $f$ to every input
preserves the compare-and-swap behavior.  But in the $\{0, 1\}$ input
$f(a_0), \dots, f(a_{n-1})$, the output has a 1 before a 0, contradicting the
assumption that all $\{0,1\}$ inputs are sorted correctly.  $\square$

The zero-one principle reduces verification from infinitely many inputs to just $2^n$
binary inputs.

## Common Sorting Networks

### Insertion Sort Network

Apply comparators mimicking insertion sort: for each element $i$, compare it with
elements $i-1, i-2, \dots, 0$.

- **Depth:** $O(n)$
- **Size:** $O(n^2)$

### Odd-Even Transposition Sort

Alternate between comparing odd-even pairs $(1,2), (3,4), \dots$ and even-odd pairs
$(0,1), (2,3), \dots$ for $n$ rounds.

- **Depth:** $O(n)$
- **Size:** $O(n^2)$

### Bitonic Sort Network

Uses the bitonic merge structure.

- **Depth:** $O(\log^2 n)$
- **Size:** $O(n \log^2 n)$

### AKS Network (Optimal Depth)

Ajtai, Komlos, and Szemeredi proved the existence of sorting networks with:

- **Depth:** $O(\log n)$
- **Size:** $O(n \log n)$

This is asymptotically optimal, but the constant factors are enormous, making it
impractical.  It remains one of the great theoretical results in sorting.

## Lower Bounds

Any sorting network for $n$ elements must have:

$$
\text{depth} \ge \lceil \log_2 n \rceil
$$

This follows because each parallel step can at most double the number of distinct
orderings that can be distinguished.

$$
\text{size} \ge \lceil \log_2(n!) \rceil \approx n \log_2 n - 1.44n
$$

This follows from the information-theoretic lower bound: $n!$ possible permutations
require at least $\log_2(n!)$ binary decisions.

## Implementation

```python
"""
Sorting networks -- fixed comparison sequences for parallel sorting.

Implements several sorting networks and verifies them using the
zero-one principle.
Time:  Depends on network (see individual analyses)
Space: O(n) -- in-place via swaps
"""

# === Comparator =============================================================

def compare_and_swap(arr: list, i: int, j: int) -> None:
    """Swap arr[i] and arr[j] if arr[i] > arr[j]."""
    if arr[i] > arr[j]:
        arr[i], arr[j] = arr[j], arr[i]


# === Odd-Even Transposition Sort Network ====================================

def odd_even_sort_network(n: int) -> list[list[tuple[int, int]]]:
    """Generate the comparator sequence for odd-even transposition sort.

    Returns a list of parallel stages, each containing independent comparators.
    """
    stages: list[list[tuple[int, int]]] = []
    for phase in range(n):
        stage: list[tuple[int, int]] = []
        if phase % 2 == 0:
            # Even phase: compare (0,1), (2,3), ...
            for i in range(0, n - 1, 2):
                stage.append((i, i + 1))
        else:
            # Odd phase: compare (1,2), (3,4), ...
            for i in range(1, n - 1, 2):
                stage.append((i, i + 1))
        if stage:
            stages.append(stage)
    return stages


# === Apply a sorting network ================================================

def apply_network(
    arr: list[int], network: list[list[tuple[int, int]]]
) -> list[int]:
    """Apply a sorting network to *arr* and return the sorted result."""
    result = list(arr)
    for stage in network:
        for i, j in stage:
            compare_and_swap(result, i, j)
    return result


# === Verify using the zero-one principle ====================================

def verify_network(n: int, network: list[list[tuple[int, int]]]) -> bool:
    """Verify a sorting network using the zero-one principle.

    Tests all 2^n binary inputs.
    """
    for bits in range(1 << n):
        arr = [(bits >> i) & 1 for i in range(n)]
        result = apply_network(arr, network)
        if result != sorted(arr):
            return False
    return True


# === Demo ===================================================================

if __name__ == "__main__":
    n = 8
    network = odd_even_sort_network(n)
    print(f"Odd-even transposition sort for n={n}:")
    print(f"  Depth (parallel stages): {len(network)}")
    print(f"  Size (total comparators): {sum(len(s) for s in network)}")
    print(f"  Verified (0-1 principle): {verify_network(n, network)}")

    # Test on sample input
    data = [3, 7, 4, 8, 6, 2, 1, 5]
    sorted_data = apply_network(data, network)
    print(f"\n  Input:  {data}")
    print(f"  Output: {sorted_data}")

    # Smaller example for verification
    n_small = 6
    net_small = odd_even_sort_network(n_small)
    print(f"\nOdd-even for n={n_small}:")
    print(f"  Verified: {verify_network(n_small, net_small)}")
```

**Output:**
```
Odd-even transposition sort for n=8:
  Depth (parallel stages): 8
  Size (total comparators): 28
  Verified (0-1 principle): True

  Input:  [3, 7, 4, 8, 6, 2, 1, 5]
  Output: [1, 2, 3, 4, 5, 6, 7, 8]

Odd-even for n=6:
  Verified: True
```

## Summary of Known Sorting Networks

| Network | Depth | Size | Practical? |
|---------|-------|------|-----------|
| Insertion | $O(n)$ | $O(n^2)$ | Small $n$ only |
| Odd-even transposition | $O(n)$ | $O(n^2)$ | Simple, small $n$ |
| Bitonic (Batcher) | $O(\log^2 n)$ | $O(n \log^2 n)$ | Yes -- GPUs |
| Odd-even merge (Batcher) | $O(\log^2 n)$ | $O(n \log^2 n)$ | Yes |
| AKS | $O(\log n)$ | $O(n \log n)$ | No -- huge constants |
| Optimal (Goodrich) | $O(\log n)$ | $O(n \log n)$ | Theoretically |

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022).
  *Introduction to Algorithms* (4th ed.), Chapter 27. MIT Press.
- Knuth, D. E. (1998). *The Art of Computer Programming, Vol. 3: Sorting and
  Searching* (2nd ed.), Section 5.3.4. Addison-Wesley.
