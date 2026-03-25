# Omega(n log n) Proof

Every comparison-based sorting algorithm we have seen — insertion sort at $O(n^2)$, merge sort and heapsort at $O(n \log n)$ — uses at least a certain number of comparisons in the worst case. But is $O(n \log n)$ the best we can do, or could a cleverer algorithm sort with fewer comparisons? The answer, proved rigorously in this section, is that $\Omega(n \log n)$ comparisons are necessary. No comparison-based sorting algorithm can do better. This result, established through the [decision tree model](decision_tree.md), is one of the landmark lower bounds in computer science.

## Theorem Statement

**Theorem.** Any comparison-based sorting algorithm requires

$$
\Omega(n \log n)
$$

comparisons in the worst case to sort $n$ distinct elements.

More precisely, any deterministic comparison-based sorting algorithm must make at least $\lceil \log_2(n!) \rceil$ comparisons on some input of size $n$.

## Proof Setup

We model any comparison-based sorting algorithm as a **decision tree** — a binary tree where each internal node represents a comparison $a_i \leq a_j$ and each leaf represents an output permutation.

For the algorithm to be correct, every possible permutation of the input must correspond to at least one reachable leaf. Since there are $n!$ permutations of $n$ distinct elements, the decision tree must have at least $n!$ leaves.

## Proof

**Step 1: Leaf count.** A correct sorting algorithm must produce a different output permutation for each of the $n!$ possible input orderings. Therefore, the decision tree has at least $n!$ leaves:

$$
\ell \geq n!
$$

**Step 2: Height bound.** A binary tree of height $h$ has at most $2^h$ leaves. Since $\ell \geq n!$:

$$
2^h \geq n!
$$

Taking $\log_2$ of both sides:

$$
h \geq \log_2(n!)
$$

**Step 3: Stirling's approximation.** We use Stirling's approximation to bound $\log_2(n!)$ from below:

$$
n! = \sqrt{2\pi n} \left(\frac{n}{e}\right)^n \left(1 + \Theta\left(\frac{1}{n}\right)\right)
$$

Taking logarithms:

$$
\log_2(n!) = n \log_2 n - n \log_2 e + \frac{1}{2}\log_2(2\pi n) + O\left(\frac{1}{n}\right)
$$

The dominant term is $n \log_2 n$, so:

$$
\log_2(n!) = \Theta(n \log n)
$$

**Step 4: Alternative direct bound.** We can also establish the bound without Stirling's approximation. Observe that:

$$
n! = n \cdot (n-1) \cdot (n-2) \cdots 2 \cdot 1 \geq \left(\frac{n}{2}\right)^{n/2}
$$

because the largest $n/2$ factors are each at least $n/2$. Taking $\log_2$:

$$
\log_2(n!) \geq \frac{n}{2} \log_2 \frac{n}{2} = \frac{n}{2} \log_2 n - \frac{n}{2} = \Omega(n \log n)
$$

**Conclusion.** Combining the steps:

$$
h \geq \log_2(n!) = \Omega(n \log n)
$$

Since $h$ is the worst-case number of comparisons for the algorithm, every comparison-based sorting algorithm requires $\Omega(n \log n)$ comparisons in the worst case. $\square$

## Tightness of the Bound

The lower bound is **tight**: merge sort achieves $O(n \log n)$ worst-case comparisons, matching the lower bound up to constant factors. More precisely:

- The lower bound gives $\log_2(n!) \approx n \log_2 n - 1.443n$ comparisons.
- Merge sort uses at most $n \lceil \log_2 n \rceil \approx n \log_2 n$ comparisons.

The gap between the lower bound and merge sort's upper bound is approximately $1.443n$ comparisons — a linear additive term. Closing this gap exactly (finding the sorting algorithm that minimizes the number of comparisons for every $n$) remains an area of ongoing research for small values of $n$.

## Average-Case Lower Bound

The $\Omega(n \log n)$ bound also holds for the **average case** when the input is a uniformly random permutation.

**Theorem.** Any comparison-based sorting algorithm makes at least

$$
\log_2(n!) - n \approx n \log_2 n - 2.443n
$$

comparisons on average over all $n!$ input permutations.

*Proof sketch.* Consider a decision tree with $\ell \geq n!$ leaves. The average path length from the root to a leaf (weighted by the probability of each permutation) is minimized when the tree is as balanced as possible. For a balanced binary tree with $\ell$ leaves, the average depth is at least $\log_2 \ell$. Since $\ell \geq n!$, the average number of comparisons is at least $\log_2(n!)$. A more careful analysis accounting for the uniform distribution over permutations gives the stated bound.

## Randomized Lower Bound

The lower bound extends to **randomized** comparison-based sorting algorithms. A randomized algorithm can be viewed as a probability distribution over deterministic algorithms (i.e., over decision trees). By Yao's minimax principle, the expected running time of the best randomized algorithm on the worst-case input is at least the average-case running time of the best deterministic algorithm on a random input. Since the average-case lower bound is $\Omega(n \log n)$, the randomized lower bound is also $\Omega(n \log n)$.

## What the Proof Does Not Show

It is important to understand the scope of this result:

1. **Only comparisons are counted.** The proof counts comparisons, not swaps, memory accesses, or total operations. An algorithm could make $\Theta(n \log n)$ comparisons but $\Theta(n^2)$ total operations (e.g., due to data movement).

2. **Only worst-case inputs.** The proof shows that every algorithm has at least one bad input. It does not say that *every* input requires $\Omega(n \log n)$ comparisons. Already-sorted inputs can be verified in $O(n)$ comparisons.

3. **Only comparison-based algorithms.** Counting sort, radix sort, and bucket sort are not comparison-based and can sort in $O(n)$ time under appropriate assumptions. The lower bound does not apply to them.

4. **Distinct elements assumed.** The proof assumes $n!$ possible permutations, which requires all elements to be distinct. With many duplicates, fewer permutations are possible and the effective lower bound is lower.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. Section 8.1.
- Knuth, D. E. (1997). *The Art of Computer Programming, Volume 3: Sorting and Searching* (2nd ed.). Addison-Wesley. Section 5.3.1.
