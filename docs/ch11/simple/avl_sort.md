# AVL Sort

Balanced binary search trees guarantee $O(\log n)$ time for insertion, deletion, and search. A natural idea is to use this property for sorting: insert all $n$ elements into a balanced BST, then read them out in sorted order via an in-order traversal. When the balanced BST is an **AVL tree**, this approach is called **AVL sort**. It achieves $O(n \log n)$ worst-case time, matching the theoretical lower bound for comparison-based sorting. Although AVL sort is rarely used as a standalone sorting algorithm in practice, it illustrates the deep connection between balanced search trees and optimal sorting.

## Algorithm

AVL sort consists of two phases:

1. **Build phase.** Starting from an empty AVL tree, insert each of the $n$ input elements one at a time. Each insertion takes $O(\log n)$ time because the AVL tree maintains a height of $O(\log n)$ through rotations.

2. **Extract phase.** Perform an in-order traversal of the AVL tree. Since an in-order traversal of a BST visits nodes in sorted order, the traversal produces the elements in non-decreasing order.

```
AVL-Sort(A[1..n]):
    T ← empty AVL tree
    for i = 1 to n:
        AVL-Insert(T, A[i])
    In-Order-Traversal(T)  // outputs elements in sorted order
```

## Complexity Analysis

### Time Complexity

Each of the $n$ insertions into the AVL tree takes $O(\log n)$ time in the worst case. The in-order traversal visits each of the $n$ nodes exactly once, taking $\Theta(n)$ time. Therefore, the total time is

$$
T(n) = \underbrace{n \cdot O(\log n)}_{\text{insertions}} + \underbrace{\Theta(n)}_{\text{traversal}} = O(n \log n)
$$

This bound holds in all cases — best, average, and worst — because AVL trees guarantee $O(\log n)$ height regardless of insertion order. This is in contrast to sorting with an unbalanced BST, which degrades to $O(n^2)$ on sorted input.

### Space Complexity

The AVL tree stores all $n$ elements plus $O(1)$ additional data per node (height and pointers), requiring $\Theta(n)$ total space. This makes AVL sort an **out-of-place** algorithm, unlike heapsort which sorts in $O(1)$ auxiliary space.

## Comparison with Unbalanced BST Sort

Sorting by inserting into an **unbalanced** BST and then performing in-order traversal is sometimes called **tree sort**. The key difference is the worst-case guarantee:

| Property | BST Sort (unbalanced) | AVL Sort |
|----------|----------------------|----------|
| Best case | $O(n \log n)$ | $O(n \log n)$ |
| Average case | $O(n \log n)$ | $O(n \log n)$ |
| Worst case | $O(n^2)$ | $O(n \log n)$ |
| Extra space | $\Theta(n)$ | $\Theta(n)$ |
| Rotations needed | No | Yes |

The unbalanced BST degrades to $O(n^2)$ when the input is already sorted (or reverse sorted), because the tree becomes a linear chain of height $n$. The AVL tree avoids this by performing rotations to maintain balance after each insertion.

## Comparison with Other O(n log n) Sorts

| Property | AVL Sort | Merge Sort | Heapsort | Quicksort |
|----------|----------|------------|----------|-----------|
| Worst case | $O(n \log n)$ | $O(n \log n)$ | $O(n \log n)$ | $O(n^2)$ |
| Stable | No | Yes | No | No |
| In-place | No | No | Yes | Yes (relaxed) |
| Extra space | $\Theta(n)$ | $\Theta(n)$ | $O(1)$ | $O(\log n)$ |
| Cache friendly | No | Yes | No | Yes |

AVL sort uses $\Theta(n)$ extra space like merge sort, but merge sort has better cache locality due to sequential array access. Heapsort matches the $O(n \log n)$ worst case while using only $O(1)$ extra space. For these reasons, AVL sort is not competitive as a pure sorting algorithm.

## Why AVL Sort Matters

Despite its practical disadvantages, AVL sort is important for several reasons:

1. **Conceptual bridge.** It connects two fundamental areas — balanced search trees and sorting — showing that $O(n \log n)$ sorting follows directly from $O(\log n)$ balanced insertion.

2. **Online sorting.** AVL sort supports **online** operation: elements can arrive one at a time, and at any point the current sorted order is available via in-order traversal. Merge sort and quicksort require all elements to be present before sorting begins.

3. **Dynamic operations.** After sorting, the AVL tree supports additional insertions and deletions in $O(\log n)$ time while maintaining sorted order. Array-based sorts would require $O(n)$ time to insert into the sorted result.

4. **Foundation for augmented structures.** AVL trees can be augmented with order statistics (rank queries), interval data, or other metadata. The sorted order comes "for free" on top of these capabilities.

!!! tip "When to Use AVL Sort"
    Consider AVL sort when:

    - Elements arrive incrementally and you need sorted order at any time (online setting).
    - You need both sorted output and fast search/insert/delete afterwards.
    - You want a guaranteed $O(n \log n)$ worst case without randomization.

    For batch sorting of a fixed array, prefer merge sort, heapsort, or quicksort.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. Chapters 12--13.
- Adelson-Velsky, G. M., & Landis, E. M. (1962). An algorithm for the organization of information. *Doklady Akademii Nauk SSSR*, 146(2), 263--266.
