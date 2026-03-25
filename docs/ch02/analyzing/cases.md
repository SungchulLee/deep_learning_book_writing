# Best, Worst, and Average Case

An algorithm does not run in the same amount of time on every input of a given size.
Consider searching for a target value in an unsorted array: the target might be the
first element checked (fast) or the last (slow), and on average it falls somewhere
in between. Distinguishing these scenarios lets us make precise statements about an
algorithm's performance guarantees and expected behavior.

## Why Cases Matter

When we say an algorithm "takes $O(n)$ time," that statement hides which inputs we
are talking about. Two algorithms with identical worst-case bounds can differ
dramatically in practice if one has a much better average case. Conversely, an
algorithm with an excellent best case may be unusable if its worst case is
catastrophic. Analyzing all three cases gives a complete picture of performance.

## Best-Case Running Time

The **best-case running time** $T_{\text{best}}(n)$ is the minimum number of
operations the algorithm performs over all inputs of size $n$:

$$
T_{\text{best}}(n) = \min_{\text{input } I,\; |I|=n} T(I)
$$

Best-case analysis identifies the most favorable input configuration. While it rarely
drives algorithm selection, it reveals structural properties. For example, a best case
of $\Omega(n)$ for a sorting algorithm tells us that every comparison-based sort must
at least read all elements.

??? example "Linear Search Best Case"

    Searching for a target in an array of $n$ elements:

    ```
    LinearSearch(A, target):
        for i = 0 to n - 1:
            if A[i] == target:
                return i
        return -1
    ```

    The best case occurs when the target is the first element, giving
    $T_{\text{best}}(n) = \Theta(1)$.

## Worst-Case Running Time

The **worst-case running time** $T_{\text{worst}}(n)$ is the maximum number of
operations over all inputs of size $n$:

$$
T_{\text{worst}}(n) = \max_{\text{input } I,\; |I|=n} T(I)
$$

Worst-case analysis provides an **upper-bound guarantee**: for any input of size $n$,
the algorithm finishes in at most $T_{\text{worst}}(n)$ steps. This guarantee is
critical in real-time systems, security applications, and anywhere predictable
performance is required.

??? example "Linear Search Worst Case"

    The worst case for `LinearSearch` occurs when the target is the last element or
    is absent entirely, requiring all $n$ comparisons:

    $$
    T_{\text{worst}}(n) = \Theta(n)
    $$

## Average-Case Running Time

The **average-case running time** $T_{\text{avg}}(n)$ measures the expected number of
operations, averaged over some probability distribution on inputs of size $n$.
Formally, let $\mathcal{I}_n$ denote the set of all inputs of size $n$, and let
$\Pr[I]$ be the probability of input $I$. Then:

$$
T_{\text{avg}}(n) = \sum_{I \in \mathcal{I}_n} \Pr[I] \cdot T(I) = \mathbb{E}[T(I)]
$$

Average-case analysis requires an explicit assumption about the input distribution.
The most common assumption is the **uniform distribution**, where every input of size
$n$ is equally likely.

??? example "Linear Search Average Case"

    Assume the target is in the array and each position is equally likely. Position
    $i$ (0-indexed) requires $i + 1$ comparisons. Under the uniform distribution:

    $$
    T_{\text{avg}}(n) = \sum_{i=0}^{n-1} \frac{1}{n}(i + 1) = \frac{1}{n} \cdot \frac{n(n+1)}{2} = \frac{n+1}{2}
    $$

    On average, linear search examines about half the array, giving
    $T_{\text{avg}}(n) = \Theta(n)$.

## Comparing the Three Cases

The three cases satisfy the ordering:

$$
T_{\text{best}}(n) \leq T_{\text{avg}}(n) \leq T_{\text{worst}}(n)
$$

This inequality holds because the average cannot exceed the maximum or fall below the
minimum.

| Case | Linear Search | Insertion Sort | Binary Search |
|---|---|---|---|
| Best | $\Theta(1)$ | $\Theta(n)$ | $\Theta(1)$ |
| Average | $\Theta(n)$ | $\Theta(n^2)$ | $\Theta(\log n)$ |
| Worst | $\Theta(n)$ | $\Theta(n^2)$ | $\Theta(\log n)$ |

!!! tip "Which Case Matters Most?"

    In practice, **worst-case analysis** is the most commonly used because it provides
    a guarantee independent of the input. Average-case analysis is valuable when the
    input distribution is known or can be enforced (e.g., through randomization).
    Best-case analysis is the least informative for algorithm selection but useful for
    establishing lower bounds on specific problems.

## Connection to Asymptotic Notation

The three cases connect directly to asymptotic notation. Given an algorithm with
running time $T(n)$:

- $O$-notation naturally pairs with **worst-case** analysis: $T(n) = O(f(n))$ means
  the algorithm takes at most $f(n)$ steps on every input of size $n$.
- $\Omega$-notation naturally pairs with **best-case** analysis: $T(n) = \Omega(g(n))$
  means the algorithm takes at least $g(n)$ steps on some input of size $n$.
- $\Theta$-notation applies when the best and worst cases share the same growth rate,
  or when describing the exact growth of a specific case (e.g., "the worst case is
  $\Theta(n^2)$").

!!! warning "Common Misconception"

    $O$ does not mean "worst case" and $\Omega$ does not mean "best case." These are
    separate concepts. We can say "the best-case running time is $O(n)$" or "the
    worst-case running time is $\Omega(n^2)$." The asymptotic notation describes the
    growth rate of a function; the case specifies which function we are analyzing.

## Reference

- [Introduction to Algorithms (CLRS), Chapters 2-4](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
