# Common Growth Rates

Recognizing an algorithm's growth class is one of the most practical skills in algorithm analysis.  When you see that a loop runs $n^2$ times or that a recursive call halves the input, you can immediately classify the running time and compare it against alternatives.  This page catalogues the standard growth classes from fastest (smallest) to slowest (largest), with example algorithms and concrete numbers that build intuition for how these functions scale.

## Numerical Perspective

The table below shows function values for selected input sizes.  Even modest differences in growth class produce enormous differences at scale.

| $n$ | $\log_2 n$ | $\sqrt{n}$ | $n$ | $n \log_2 n$ | $n^2$ | $2^n$ |
|---|---|---|---|---|---|---|
| 10 | 3.3 | 3.2 | 10 | 33 | 100 | 1,024 |
| 100 | 6.6 | 10 | 100 | 664 | 10,000 | $1.3 \times 10^{30}$ |
| 1,000 | 10 | 31.6 | 1,000 | 9,966 | $10^6$ | $\gg 10^{300}$ |
| $10^6$ | 20 | 1,000 | $10^6$ | $2 \times 10^7$ | $10^{12}$ | -- |

The exponential column grows so fast that $2^{1000}$ already exceeds the number of atoms in the observable universe (roughly $10^{80}$).

## Constant -- $O(1)$

A constant-time operation takes the same number of steps regardless of input size.

- **Definition:** $T(n) = c$ for some constant $c > 0$.
- **Examples:** Array access by index, hash table lookup (amortized), pushing onto a stack.
- **Intuition:** Doubling the input size has no effect on running time.

## Logarithmic -- $O(\log n)$

Logarithmic time arises when each step eliminates a constant fraction of the remaining input.

- **Definition:** $T(n) = c \log_b n$ for some base $b > 1$.  The base does not affect the asymptotic class because $\log_b n = \Theta(\log n)$ for any fixed $b$.
- **Examples:** Binary search, balanced BST lookup, exponentiation by squaring.
- **Intuition:** Doubling the input adds only one extra step.

!!! tip "Base does not matter"

    Since $\log_2 n = \log_{10} n / \log_{10} 2$, changing the logarithm base only introduces a constant factor.  All logarithmic bases belong to the same $\Theta(\log n)$ class.

## Square Root -- $O(\sqrt{n})$

Some algorithms achieve sub-linear time by processing a number of elements proportional to $\sqrt{n}$.

- **Examples:** Trial division for primality testing (up to $\sqrt{n}$), Mo's algorithm for range queries.
- **Intuition:** Growing $n$ by a factor of 4 only doubles the running time.

## Linear -- $O(n)$

Linear time means the running time is directly proportional to the input size.

- **Definition:** $T(n) = cn$ for some constant $c$.
- **Examples:** Linear scan, counting sort (when range is $O(n)$), single-pass streaming algorithms.
- **Intuition:** Doubling the input exactly doubles the running time.  Every element is processed at most a constant number of times.

## Linearithmic -- $O(n \log n)$

The linearithmic class sits just above linear and appears in many optimal sorting and divide-and-conquer algorithms.

- **Definition:** $T(n) = cn \log n$.
- **Examples:** Merge sort, heapsort, FFT.
- **Intuition:** Each of the $n$ elements is processed $O(\log n)$ times, often once per level of a recursion tree with $\log n$ levels.

## Quadratic -- $O(n^2)$

Quadratic time typically arises from nested loops where each loop iterates over the entire input.

- **Definition:** $T(n) = cn^2$.
- **Examples:** Insertion sort (worst case), bubble sort, naive string matching.
- **Intuition:** Doubling the input quadruples the running time.  Feasible for $n$ up to roughly $10^4$.

## Cubic -- $O(n^3)$

Cubic time appears in algorithms with three nested loops or in naive matrix operations.

- **Definition:** $T(n) = cn^3$.
- **Examples:** Naive matrix multiplication, Floyd-Warshall shortest paths.
- **Intuition:** Doubling the input increases running time by a factor of 8.  Feasible for $n$ up to roughly $10^3$.

## Polynomial -- $O(n^k)$

Any algorithm with running time bounded by $n^k$ for a fixed constant $k$ is considered polynomial-time.  Polynomial algorithms are generally regarded as **efficient** in complexity theory (the class P).

!!! info "Practical note"

    While $O(n^{100})$ is technically polynomial, in practice only small exponents ($k \leq 3$ or so) yield feasible algorithms.  The theoretical distinction matters most when separating polynomial from exponential.

## Exponential -- $O(2^n)$

Exponential time grows so rapidly that even moderate input sizes become intractable.

- **Definition:** $T(n) = c \cdot 2^n$ (or more generally $c \cdot b^n$ for $b > 1$).
- **Examples:** Brute-force subset enumeration, naive recursive Fibonacci, exhaustive search over all binary strings.
- **Intuition:** Adding a single element to the input doubles the running time.

## Factorial -- $O(n!)$

Factorial growth arises from generating all permutations of the input.

- **Definition:** $T(n) = c \cdot n!$.
- **Examples:** Brute-force traveling salesman, generating all permutations.
- **Intuition:** $20! \approx 2.4 \times 10^{18}$, which already exceeds what any modern computer can enumerate in reasonable time.

## The Complete Hierarchy

Combining all classes into a single ordering:

$$
O(1) \subset O(\log n) \subset O(\sqrt{n}) \subset O(n) \subset O(n \log n) \subset O(n^2) \subset O(n^3) \subset O(2^n) \subset O(n!) \subset O(n^n)
$$

Each strict inclusion means there exist functions that belong to the larger class but not the smaller one.  For techniques to prove these relationships and compare arbitrary functions, see [Growth Rate Comparison](comparison.md).

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 3. MIT Press.
