# Akra-Bazzi Method

The Master theorem provides a clean recipe for recurrences of the form $T(n) = aT(n/b) + f(n)$, but it requires the subproblems to be exactly equal in size and the division ratio to be a constant. Many practical divide-and-conquer algorithms, such as median-of-medians selection or hybrid sorting routines, split the input into subproblems of *unequal* sizes. The Akra-Bazzi method generalizes the Master theorem to handle these asymmetric recurrences, making it one of the most powerful tools for solving divide-and-conquer recurrences.

## General Form

The Akra-Bazzi method applies to recurrences of the form:

$$
T(n) = \sum_{i=1}^{k} a_i \, T(b_i \, n) + g(n)
$$

where:

- $k \geq 1$ is the number of recursive subproblems
- $a_i > 0$ are the subproblem multipliers
- $0 < b_i < 1$ are the subproblem size ratios (each subproblem is a constant fraction of $n$)
- $g(n)$ is the non-recursive cost, which must satisfy a polynomial growth condition (described below)

Unlike the Master theorem, the $b_i$ values need not all be equal. This allows the method to handle recurrences where the input is split into parts of different sizes.

## Conditions on the Toll Function

The non-recursive cost $g(n)$ must satisfy the following **polynomial growth condition**: there exist constants $c_1, c_2 > 0$ such that for all $n$ sufficiently large,

$$
|g'(n)| \leq c_1 \, n^{c_2}
$$

In practice, this condition is satisfied by essentially every toll function encountered in algorithm analysis, including polynomials, polylogarithmic functions, and products of polynomials with logarithms.

## The Critical Exponent

The first step in applying the Akra-Bazzi method is to find the unique real number $p$ satisfying:

$$
\sum_{i=1}^{k} a_i \, b_i^{\,p} = 1
$$

This value $p$ is called the **critical exponent**. It always exists and is unique because the left-hand side is a strictly decreasing continuous function of $p$: when $p \to -\infty$ it tends to $+\infty$, and when $p \to +\infty$ it tends to $0$. By the intermediate value theorem, there is exactly one $p$ where the sum equals $1$.

For the special case $k = 1$ with $a_1 = a$ and $b_1 = 1/b$, the equation becomes $a \cdot (1/b)^p = 1$, which gives $p = \log_b a$. This recovers the critical exponent from the Master theorem.

## The Akra-Bazzi Theorem

!!! note "Akra-Bazzi Theorem"
    Given a recurrence $T(n) = \sum_{i=1}^{k} a_i \, T(b_i \, n) + g(n)$ where $a_i > 0$, $0 < b_i < 1$, and $g(n)$ satisfies the polynomial growth condition, the solution is:

    $$
    T(n) = \Theta\!\left( n^p \left( 1 + \int_1^n \frac{g(u)}{u^{p+1}} \, du \right) \right)
    $$

    where $p$ is the unique solution to $\sum_{i=1}^{k} a_i \, b_i^{\,p} = 1$.

The integral absorbs the contribution of the non-recursive work across all levels of the recursion. When $g(n)$ is a polynomial times a logarithm, this integral can be evaluated in closed form.

## Evaluating the Integral

The power of the Akra-Bazzi method lies in reducing the recurrence to a calculus problem. Here are common cases for the integral.

### Case 1: Polynomial Toll Function

If $g(n) = n^c$ for some constant $c$:

$$
\int_1^n \frac{u^c}{u^{p+1}} \, du = \int_1^n u^{c - p - 1} \, du = \begin{cases} \dfrac{n^{c-p} - 1}{c - p} & \text{if } c \neq p \\[6pt] \ln n & \text{if } c = p \end{cases}
$$

When $c > p$, the integral is $\Theta(n^{c-p})$, so $T(n) = \Theta(n^c)$. When $c < p$, the integral is $\Theta(1)$, so $T(n) = \Theta(n^p)$. When $c = p$, the integral is $\Theta(\ln n)$, so $T(n) = \Theta(n^p \log n)$.

### Case 2: Polynomial Times Logarithm

If $g(n) = n^c \log^d n$ for constants $c$ and $d \geq 0$:

$$
\int_1^n \frac{u^c \log^d u}{u^{p+1}} \, du = \int_1^n u^{c-p-1} \log^d u \, du
$$

- When $c > p$: the integral is $\Theta(n^{c-p} \log^d n)$, giving $T(n) = \Theta(n^c \log^d n)$
- When $c = p$: the integral is $\Theta(\log^{d+1} n)$, giving $T(n) = \Theta(n^p \log^{d+1} n)$
- When $c < p$: the integral is $\Theta(1)$, giving $T(n) = \Theta(n^p)$

## Worked Examples

### Example 1: Equal-Size Split (Recovering the Master Theorem)

Consider the merge sort recurrence:

$$
T(n) = 2T(n/2) + n
$$

This fits the Akra-Bazzi form with $k = 1$, $a_1 = 2$, $b_1 = 1/2$, and $g(n) = n$.

**Step 1**: Find $p$ from $2 \cdot (1/2)^p = 1$, which gives $p = 1$.

**Step 2**: Evaluate the integral:

$$
\int_1^n \frac{u}{u^{1+1}} \, du = \int_1^n \frac{1}{u} \, du = \ln n
$$

**Step 3**: Apply the theorem:

$$
T(n) = \Theta\!\left( n^1 \left(1 + \ln n\right) \right) = \Theta(n \log n)
$$

This matches the well-known merge sort complexity.

### Example 2: Unequal Split

Consider the recurrence arising in the median-of-medians algorithm:

$$
T(n) = T(n/5) + T(7n/10) + n
$$

Here $k = 2$, $a_1 = 1$, $b_1 = 1/5$, $a_2 = 1$, $b_2 = 7/10$, and $g(n) = n$.

**Step 1**: Find $p$ from $(1/5)^p + (7/10)^p = 1$.

Testing $p = 0$: $(1/5)^0 + (7/10)^0 = 1 + 1 = 2 > 1$.

Testing $p = 1$: $1/5 + 7/10 = 9/10 < 1$.

Since the left-hand side is continuous and strictly decreasing in $p$, the solution lies between $0$ and $1$. Numerically, $p \approx 0.8396$.

**Step 2**: Since $g(n) = n = n^1$ and $1 > p \approx 0.84$, the integral evaluates to $\Theta(n^{1-p})$.

**Step 3**: Apply the theorem:

$$
T(n) = \Theta\!\left( n^p \cdot n^{1-p} \right) = \Theta(n)
$$

This confirms that the median-of-medians algorithm runs in linear time.

### Example 3: Multiple Unequal Subproblems

Consider:

$$
T(n) = 3T(n/4) + 2T(n/3) + n^2
$$

Here $k = 2$, $a_1 = 3$, $b_1 = 1/4$, $a_2 = 2$, $b_2 = 1/3$, and $g(n) = n^2$.

**Step 1**: Find $p$ from $3 \cdot (1/4)^p + 2 \cdot (1/3)^p = 1$.

Testing $p = 1$: $3/4 + 2/3 = 17/12 > 1$.

Testing $p = 2$: $3/16 + 2/9 \approx 0.410 < 1$.

So $p$ lies between $1$ and $2$. Numerically, $p \approx 1.296$.

**Step 2**: Since $g(n) = n^2$ and $2 > p \approx 1.296$, the integral gives $\Theta(n^{2-p})$.

**Step 3**: Apply the theorem:

$$
T(n) = \Theta\!\left( n^p \cdot n^{2-p} \right) = \Theta(n^2)
$$

The quadratic toll function dominates the recursive work.

## Comparison with the Master Theorem

| Feature | Master Theorem | Akra-Bazzi Method |
|---------|---------------|-------------------|
| Subproblem sizes | Must be equal ($n/b$) | Can be unequal ($b_i n$) |
| Number of distinct sizes | One | Any finite number |
| Toll function restrictions | Polynomial regularity condition | Polynomial growth on derivative |
| Solution method | Case comparison | Integral evaluation |
| Scope | Subset of Akra-Bazzi | Most general for divide-and-conquer |

The Master theorem is simpler to apply when it applies, but the Akra-Bazzi method covers strictly more recurrences.

## Finding the Critical Exponent Numerically

When the equation $\sum a_i b_i^p = 1$ cannot be solved analytically, numerical root-finding methods such as bisection or Newton's method work well. The function $h(p) = \sum a_i b_i^p - 1$ is smooth and strictly decreasing, so convergence is reliable.

```python
"""Numerical computation of the Akra-Bazzi critical exponent."""


# ============================================================
# Critical exponent solver
# ============================================================
def find_critical_exponent(a_list, b_list, tol=1e-10):
    """Find p such that sum(a_i * b_i^p) = 1 using bisection.

    Parameters
    ----------
    a_list : list of float
        Subproblem multipliers (each > 0).
    b_list : list of float
        Subproblem size ratios (each in (0, 1)).
    tol : float
        Convergence tolerance.

    Returns
    -------
    float
        The critical exponent p.
    """
    def h(p):
        return sum(a * b ** p for a, b in zip(a_list, b_list)) - 1.0

    # Bracket the root
    lo, hi = -50.0, 50.0
    assert h(lo) > 0 and h(hi) < 0, "Root not bracketed"

    while hi - lo > tol:
        mid = (lo + hi) / 2.0
        if h(mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


# ============================================================
# Examples
# ============================================================
if __name__ == "__main__":
    # Example 1: Merge sort  T(n) = 2T(n/2) + n
    p = find_critical_exponent([2], [0.5])
    print(f"Merge sort: p = {p:.6f}")  # Expected: 1.0

    # Example 2: Median of medians  T(n) = T(n/5) + T(7n/10) + n
    p = find_critical_exponent([1, 1], [0.2, 0.7])
    print(f"Median of medians: p = {p:.6f}")  # Expected: ~0.8396

    # Example 3: T(n) = 3T(n/4) + 2T(n/3) + n^2
    p = find_critical_exponent([3, 2], [0.25, 1 / 3])
    print(f"Example 3: p = {p:.6f}")  # Expected: ~1.296
```

## Limitations

The Akra-Bazzi method is powerful but has boundaries:

- **Subproblem sizes must be constant fractions of $n$**: Recurrences like $T(n) = T(n - 1) + n$ (linear reduction) do not fit the framework.
- **The growth condition on $g$**: While broad, it excludes pathological toll functions like $g(n) = 2^n$.
- **Floor and ceiling effects**: The original theorem accounts for floors and ceilings in the arguments (e.g., $T(\lfloor n/5 \rfloor)$), but the integral formula applies to the continuous relaxation. This is valid because the error introduced by rounding is absorbed into the $\Theta$ notation.

## Connections to Other Topics

- **[Master Theorem](master.md)**: The special case where all subproblems have equal size
- **[Extended Master Theorem](extended_master.md)**: Handles logarithmic factors in the toll function within the Master theorem framework
- **[Recursion Tree Method](recursion_tree.md)**: Provides geometric intuition that the Akra-Bazzi integral formalizes
- **[Substitution Method](substitution.md)**: Can verify Akra-Bazzi results for specific recurrences

## References

- Akra, M., & Bazzi, L. (1998). On the solution of linear recurrence equations. *Computational Optimization and Applications*, 10(2), 195-210.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 4. MIT Press.
- Leighton, T. (1996). Notes on better master theorems for divide-and-conquer recurrences. MIT CSAIL.
