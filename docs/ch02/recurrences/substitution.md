# Substitution Method

The [recursion tree](recursion_tree.md) and [Master theorem](master.md) give us answers quickly, but neither constitutes a formal proof. The substitution method fills this gap: it is mathematical induction applied to recurrences. You guess a bound (often informed by a recursion tree), then prove by strong induction that the guess is correct. Because the method is general-purpose and relies only on induction, it can handle recurrences that fall outside the Master theorem's scope, and it is the standard way to *verify* results obtained by other methods.

## The Two-Step Process

The substitution method has exactly two steps:

1. **Guess** the form of the solution (e.g., $T(n) = O(n \log n)$)
2. **Prove** the guess by strong induction, substituting the inductive hypothesis into the recurrence

The guess must include an unspecified constant (e.g., $T(n) \leq cn \log n$ for some $c > 0$). The proof then determines conditions on $c$ that make the bound hold.

## Example 1: Merge Sort

### Guess

From the recursion tree or prior experience, we guess $T(n) = O(n \log n)$. More precisely, we claim there exists a constant $c > 0$ such that $T(n) \leq cn \log n$ for all $n \geq 2$.

### Inductive Step

The recurrence is $T(n) = 2T(n/2) + n$. Assume the bound holds for all values smaller than $n$:

$$
T(n/2) \leq c(n/2) \log(n/2)
$$

Substituting into the recurrence:

$$
T(n) = 2T(n/2) + n \leq 2 \cdot c(n/2) \log(n/2) + n = cn(\log n - 1) + n = cn \log n - cn + n
$$

This is at most $cn \log n$ whenever $cn \geq n$, i.e., $c \geq 1$.

### Base Case

We need $T(n_0) \leq cn_0 \log n_0$ for some base case $n_0$. Since $T(1) = \Theta(1)$ and $\log 1 = 0$, the bound $T(n) \leq cn \log n$ fails at $n = 1$. This is not a problem: we choose $n_0 = 2$ and verify that $T(2) \leq c \cdot 2 \cdot \log 2 = 2c$, which holds for $c$ large enough.

### Conclusion

For $c \geq 1$ and $n \geq 2$, $T(n) \leq cn \log n$, confirming $T(n) = O(n \log n)$.

## Example 2: Upper Bound with a Subtraction Trick

Consider $T(n) = 2T(n/2) + 1$. We guess $T(n) = O(n)$, meaning $T(n) \leq cn$ for some constant $c > 0$.

### Inductive Step

$$
T(n) = 2T(n/2) + 1 \leq 2c(n/2) + 1 = cn + 1
$$

This gives $cn + 1$, not $cn$. The induction *fails* because of the extra $+1$.

### Strengthening the Guess

The fix is to guess a tighter form: $T(n) \leq cn - d$ for constants $c, d > 0$.

$$
T(n) = 2T(n/2) + 1 \leq 2(c(n/2) - d) + 1 = cn - 2d + 1 \leq cn - d
$$

The last inequality holds when $d \geq 1$. Choosing $c$ large enough for the base case completes the proof.

!!! tip "The Subtraction Trick"
    When a naive guess fails by a lower-order additive term, subtracting a lower-order term from the hypothesis often resolves it. If you guess $T(n) \leq cn$ and the induction yields $cn + 1$, try $T(n) \leq cn - d$ instead.

## Example 3: Lower Bound

The substitution method also proves lower bounds. For $T(n) = 2T(n/2) + n$, we prove $T(n) = \Omega(n \log n)$ by showing $T(n) \geq cn \log n$ for some $c > 0$.

### Inductive Step

$$
T(n) = 2T(n/2) + n \geq 2c(n/2)\log(n/2) + n = cn(\log n - 1) + n = cn \log n + n(1 - c)
$$

This is at least $cn \log n$ whenever $c \leq 1$. Combined with the upper bound (Example 1), we get $T(n) = \Theta(n \log n)$.

## Example 4: A Non-Standard Recurrence

Consider $T(n) = T(n/3) + T(2n/3) + n$. This has unequal subproblem sizes, so the Master theorem does not apply directly.

### Guess

From a recursion tree analysis: every level sums to at most $n$, and the tree height is $\log_{3/2} n$. We guess $T(n) = O(n \log n)$, i.e., $T(n) \leq cn \log n$ for some $c$.

### Inductive Step

$$
T(n) \leq c(n/3)\log(n/3) + c(2n/3)\log(2n/3) + n
$$

$$
= \frac{cn}{3}(\log n - \log 3) + \frac{2cn}{3}(\log n - \log(3/2)) + n
$$

$$
= cn \log n - \frac{cn}{3}\log 3 - \frac{2cn}{3}\log(3/2) + n
$$

$$
= cn \log n - cn\left(\frac{\log 3}{3} + \frac{2\log(3/2)}{3}\right) + n
$$

The quantity $\frac{\log 3}{3} + \frac{2\log(3/2)}{3}$ is a positive constant, call it $\alpha$. So:

$$
T(n) \leq cn \log n - c\alpha n + n \leq cn \log n
$$

whenever $c \geq 1/\alpha$. This confirms $T(n) = O(n \log n)$.

## Common Pitfalls

### Pitfall 1: Wrong Inductive Form

!!! warning "Do Not Round the Asymptotic Class"
    A common mistake is to substitute the asymptotic class rather than the precise form. For example, if the guess is $T(n) = O(n)$, you must prove $T(n) \leq cn$ for a specific constant $c$. Writing $T(n) \leq O(n)$ in the inductive step is circular and proves nothing.

### Pitfall 2: Ignoring the Base Case

The inductive step may require $c$ to be large, and the base case imposes a lower bound on $c$. Both constraints must be satisfiable simultaneously. If $n_0 = 1$ causes $\log n_0 = 0$ to make the bound trivially false, choose a larger $n_0$.

### Pitfall 3: Changing the Constant

The constant $c$ must remain the *same* throughout the proof. If the inductive step works for $c = 5$ but the base case needs $c = 10$, then $c = 10$ must work for both.

### Pitfall 4: Adding Instead of Subtracting

When strengthening a guess, always *subtract* a lower-order term (e.g., $cn - d$), never *add* one (e.g., $cn + d$). Adding makes the bound weaker, not stronger.

## Proving Exact Bounds with Theta

To show $T(n) = \Theta(g(n))$, use the substitution method twice:

1. Prove $T(n) \leq c_1 g(n)$ for some $c_1 > 0$ (upper bound)
2. Prove $T(n) \geq c_2 g(n)$ for some $c_2 > 0$ (lower bound)

The constants $c_1$ and $c_2$ may differ. Together, they establish $c_2 g(n) \leq T(n) \leq c_1 g(n)$ for all sufficiently large $n$.

## Where Do Guesses Come From?

The substitution method requires a guess, but it does not tell you how to find one. Common sources include:

| Source | When to use |
|--------|-------------|
| [Recursion tree](recursion_tree.md) | Always a good first step for intuition |
| [Master theorem](master.md) | When the recurrence matches $T(n) = aT(n/b) + f(n)$ |
| Similar recurrences | If you know $T(n) = 2T(n/2) + n$ gives $\Theta(n \log n)$, try that form for $T(n) = 2T(n/2) + n\log n$ |
| Experimentation | Compute $T(n)$ for small $n$ and look for a pattern |

## Connections to Other Topics

- **[Recursion Tree Method](recursion_tree.md)**: The primary tool for generating guesses that the substitution method verifies
- **[Master Theorem](master.md)**: Provides answers that the substitution method can prove formally
- **[Recurrence from Divide and Conquer](divide_conquer.md)**: How to derive the recurrences that the substitution method solves

## References

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Section 4.3. MIT Press.
- Erickson, J. (2019). *Algorithms*, Chapter 1. Self-published.
