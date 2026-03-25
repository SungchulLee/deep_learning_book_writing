# Master Theorem

Every divide-and-conquer algorithm that splits a problem of size $n$ into $a$ subproblems of size $n/b$ and does $f(n)$ non-recursive work produces a recurrence of the form $T(n) = aT(n/b) + f(n)$. Solving this recurrence from scratch each time would be tedious. The Master theorem provides a cookbook: compare the toll function $f(n)$ to the critical threshold $n^{\log_b a}$, and the answer falls into one of three cases. This makes it the single most-used tool for analyzing divide-and-conquer running times.

## Statement of the Master Theorem

!!! note "Master Theorem"
    Let $a \geq 1$ and $b > 1$ be constants, let $f(n)$ be a function, and let $T(n)$ be defined by the recurrence:

    $$
    T(n) = aT(n/b) + f(n)
    $$

    where $n/b$ is interpreted as $\lfloor n/b \rfloor$ or $\lceil n/b \rceil$. Then $T(n)$ has the following asymptotic bounds:

    **Case 1** (Leaf-heavy): If $f(n) = O(n^{\log_b a - \epsilon})$ for some constant $\epsilon > 0$, then

    $$
    T(n) = \Theta(n^{\log_b a})
    $$

    **Case 2** (Balanced): If $f(n) = \Theta(n^{\log_b a})$, then

    $$
    T(n) = \Theta(n^{\log_b a} \log n)
    $$

    **Case 3** (Root-heavy): If $f(n) = \Omega(n^{\log_b a + \epsilon})$ for some constant $\epsilon > 0$, and if $a f(n/b) \leq c f(n)$ for some constant $c < 1$ and all sufficiently large $n$ (the **regularity condition**), then

    $$
    T(n) = \Theta(f(n))
    $$

## The Critical Exponent

The quantity $\log_b a$ is the **critical exponent**. It represents the rate at which the number of subproblems grows relative to the rate at which subproblem size shrinks. Intuitively:

- $n^{\log_b a}$ is the total number of leaves in the recursion tree
- The cost per leaf is $\Theta(1)$ (base case work)
- So $\Theta(n^{\log_b a})$ is the total leaf-level cost

The three cases compare this leaf-level cost to the non-recursive work at the root and internal nodes.

## Intuition via the Recursion Tree

The [recursion tree](recursion_tree.md) provides the geometric intuition behind each case.

**Case 1** (Leaf-heavy): The work *increases* geometrically as we descend the tree. The leaves dominate, contributing $\Theta(n^{\log_b a})$, and the root's work $f(n)$ is negligible in comparison.

**Case 2** (Balanced): The work is roughly the *same* at every level of the tree. There are $\Theta(\log_b n)$ levels, each contributing $\Theta(n^{\log_b a})$ work, giving $\Theta(n^{\log_b a} \log n)$ in total.

**Case 3** (Root-heavy): The work *decreases* geometrically as we descend. The root dominates with cost $\Theta(f(n))$, and all other levels contribute a geometrically smaller amount.

## Worked Examples

### Example 1: Merge Sort (Case 2)

$$
T(n) = 2T(n/2) + \Theta(n)
$$

Here $a = 2$, $b = 2$, and $\log_b a = \log_2 2 = 1$. The toll function $f(n) = \Theta(n) = \Theta(n^1) = \Theta(n^{\log_b a})$.

This matches Case 2, so:

$$
T(n) = \Theta(n \log n)
$$

### Example 2: Binary Search (Case 2)

$$
T(n) = T(n/2) + \Theta(1)
$$

Here $a = 1$, $b = 2$, and $\log_b a = \log_2 1 = 0$. The toll function $f(n) = \Theta(1) = \Theta(n^0) = \Theta(n^{\log_b a})$.

Case 2 gives:

$$
T(n) = \Theta(n^0 \log n) = \Theta(\log n)
$$

### Example 3: Strassen's Algorithm (Case 1)

$$
T(n) = 7T(n/2) + \Theta(n^2)
$$

Here $a = 7$, $b = 2$, and $\log_b a = \log_2 7 \approx 2.807$. The toll function $f(n) = \Theta(n^2) = O(n^{2.807 - 0.807})$, so $\epsilon = 0.807 > 0$.

Case 1 gives:

$$
T(n) = \Theta(n^{\log_2 7}) \approx \Theta(n^{2.807})
$$

### Example 4: Root-Heavy Case (Case 3)

$$
T(n) = 2T(n/2) + n^2
$$

Here $a = 2$, $b = 2$, and $\log_b a = 1$. The toll function $f(n) = n^2 = \Omega(n^{1 + 1})$, so $\epsilon = 1 > 0$.

Check the regularity condition: $af(n/b) = 2(n/2)^2 = n^2/2 \leq (1/2) \cdot n^2 = cf(n)$ with $c = 1/2 < 1$.

Case 3 gives:

$$
T(n) = \Theta(n^2)
$$

### Example 5: Karatsuba Multiplication (Case 1)

$$
T(n) = 3T(n/2) + \Theta(n)
$$

Here $a = 3$, $b = 2$, and $\log_b a = \log_2 3 \approx 1.585$. The toll function $f(n) = \Theta(n) = O(n^{1.585 - 0.585})$, so $\epsilon = 0.585$.

Case 1 gives:

$$
T(n) = \Theta(n^{\log_2 3}) \approx \Theta(n^{1.585})
$$

## The Regularity Condition

Case 3 requires a regularity condition: $af(n/b) \leq cf(n)$ for some $c < 1$. This ensures that $f(n)$ does not oscillate in a way that would invalidate the conclusion. For most "well-behaved" functions -- polynomials, polynomials times logarithms, exponentials -- the regularity condition holds automatically.

!!! warning "When Regularity Fails"
    The function $f(n) = n^2 \sin^2(n\pi/2)$ satisfies $f(n) = \Omega(n^{1+\epsilon})$ on a dense subset of inputs but oscillates between $0$ and $n^2$. For the recurrence $T(n) = 2T(n/2) + f(n)$, the regularity condition fails because $af(n/b)$ can exceed $cf(n)$ when $f(n)$ happens to be near zero. Such pathological cases are rare in practice.

## Gap Between Cases

The three cases do not cover every possible $f(n)$. There is a gap between Cases 1 and 2 when $f(n)$ is smaller than $n^{\log_b a}$ but not polynomially smaller. For example:

$$
T(n) = 2T(n/2) + \frac{n}{\log n}
$$

Here $f(n) = n / \log n$, which is $o(n)$ but not $O(n^{1-\epsilon})$ for any $\epsilon > 0$. The Master theorem does not apply. The [Extended Master theorem](extended_master.md) and the [Akra-Bazzi method](akra_bazzi.md) handle such cases.

## Quick-Reference Table

| Recurrence | $a$ | $b$ | $\log_b a$ | Case | $T(n)$ |
|-----------|-----|-----|------------|------|--------|
| $T = 2T(n/2) + n$ | 2 | 2 | 1 | 2 | $\Theta(n \log n)$ |
| $T = T(n/2) + 1$ | 1 | 2 | 0 | 2 | $\Theta(\log n)$ |
| $T = 4T(n/2) + n$ | 4 | 2 | 2 | 1 | $\Theta(n^2)$ |
| $T = 7T(n/2) + n^2$ | 7 | 2 | 2.81 | 1 | $\Theta(n^{2.81})$ |
| $T = 3T(n/2) + n$ | 3 | 2 | 1.58 | 1 | $\Theta(n^{1.58})$ |
| $T = 2T(n/2) + n^2$ | 2 | 2 | 1 | 3 | $\Theta(n^2)$ |
| $T = 4T(n/2) + n^2$ | 4 | 2 | 2 | 2 | $\Theta(n^2 \log n)$ |
| $T = T(n/3) + 1$ | 1 | 3 | 0 | 2 | $\Theta(\log n)$ |
| $T = 9T(n/3) + n$ | 9 | 3 | 2 | 1 | $\Theta(n^2)$ |

## Connections to Other Topics

- **[Recurrence from Divide and Conquer](divide_conquer.md)**: How to derive the recurrences that the Master theorem solves
- **[Recursion Tree Method](recursion_tree.md)**: The visual intuition behind the three cases
- **[Extended Master Theorem](extended_master.md)**: Fills the gap for logarithmic factors
- **[Akra-Bazzi Method](akra_bazzi.md)**: The most general method, handling unequal splits
- **[Substitution Method](substitution.md)**: Can verify Master theorem results from first principles

## References

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 4. MIT Press.
- Bentley, J. L., Haken, D., & Saxe, J. B. (1980). A general method for solving divide-and-conquer recurrences. *SIGACT News*, 12(3), 36-44.
