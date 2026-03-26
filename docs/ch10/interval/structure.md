# Interval Tree Structure

Many applications involve collections of intervals: time windows in scheduling, genomic ranges in bioinformatics, or bounding boxes in computational geometry.  A common query is: "given a new interval, which existing intervals overlap with it?"  A naive approach checks every stored interval in $O(n)$ time.  An **interval tree** answers this query in $O(\log n)$ time by organizing intervals in an [augmented BST](augmented.md) where each node stores an interval and a max-endpoint field that enables efficient pruning during [overlap queries](overlap.md).

## Intervals

An interval $i$ is a pair $[i.low, i.high]$ with $i.low \le i.high$.  Two intervals $i$ and $j$ **overlap** if they share at least one point:

$$
i.low \le j.high \quad \text{and} \quad j.low \le i.high
$$

Intervals can be **closed** $[a, b]$, **open** $(a, b)$, or **half-open** $[a, b)$.  The interval tree structure works for all conventions; the overlap condition adjusts accordingly.

## Tree Organization

An interval tree is a balanced BST (typically a red-black tree) where:

- Each node $x$ stores an interval $[x.low, x.high]$.
- The BST key is the **low endpoint** $x.low$.
- Each node is augmented with a field $x.max$, the maximum high endpoint in $x$'s entire subtree.

$$
x.max = \max(x.high,\; x.left.max,\; x.right.max)
$$

The BST ordering on low endpoints combined with the max augmentation enables the overlap search algorithm to decide at each node whether to go left or right.

## Construction

Building an interval tree from $n$ intervals:

1. **Sequential insertion:** insert intervals one at a time into a balanced BST, updating the max field along the insertion path.  Total cost: $O(n \log n)$.
2. **Bulk construction:** sort the intervals by low endpoint and build a balanced BST bottom-up, computing max fields in a post-order traversal.  Total cost: $O(n \log n)$.

??? example "Building an interval tree"
    Given intervals: $[15, 20],\; [10, 30],\; [17, 19],\; [5, 20],\; [12, 15],\; [30, 40]$

    Sorted by low endpoint: $[5, 20],\; [10, 30],\; [12, 15],\; [15, 20],\; [17, 19],\; [30, 40]$

    A balanced interval tree (one possible configuration):

    ```
              [15, 20] max=40
             /              \
      [10, 30] max=30    [17, 19] max=40
      /        \              \
    [5,20]m=20 [12,15]m=15  [30,40]m=40
    ```

    Each node's max field equals the maximum high endpoint in its subtree.

## Properties

| Property | Description |
|----------|-------------|
| BST key | Low endpoint of the interval |
| Augmentation | Max high endpoint in subtree |
| Balance | Inherited from underlying BST (e.g., red-black) |
| Height | $O(\log n)$ |
| Space | $O(n)$ |

## Supported Operations

| Operation | Time | Description |
|-----------|------|-------------|
| Insert | $O(\log n)$ | Add a new interval |
| Delete | $O(\log n)$ | Remove an interval |
| [Overlap search](overlap.md) | $O(\log n)$ | Find one overlapping interval |
| Find all overlaps | $O(k \log n)$ | Find all $k$ overlapping intervals |

## Variants

Several interval tree variants exist for different use cases:

**Centered interval tree.** Stores all intervals containing a center point at the root, with left and right subtrees handling intervals entirely to the left or right.  This achieves $O(\log n + k)$ for reporting all $k$ overlaps but is more complex to implement.

**Augmented BST (CLRS).** The variant described here, using a balanced BST augmented with max endpoints.  Simpler to implement and sufficient for single-overlap queries.

**Segment tree.** A different structure that is better suited for stabbing queries (finding all intervals containing a given point) and supports lazy propagation for range updates.

## Applications

- **Scheduling:** find conflicts between meetings, room bookings, or resource allocations.
- **Computational geometry:** detect overlapping bounding boxes in collision detection.
- **Bioinformatics:** find genomic regions that overlap with a query region.
- **Database systems:** range-based index lookups with interval predicates.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Section 14.3. MIT Press.
- de Berg, M., Cheong, O., van Kreveld, M., & Overmars, M. (2008). *Computational Geometry: Algorithms and Applications* (3rd ed.), Chapter 10. Springer.
