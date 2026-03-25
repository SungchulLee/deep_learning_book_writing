# Generating Functions

The methods covered in sibling pages -- substitution, recursion trees, and the Master theorem -- all work directly with the recurrence relation. Generating functions take a fundamentally different approach: they encode the entire sequence of values $T(0), T(1), T(2), \ldots$ as the coefficients of a formal power series, transform the recurrence into an algebraic equation, solve that equation, and then extract the coefficients to obtain a closed-form solution. While more heavyweight than the Master theorem for standard divide-and-conquer recurrences, generating functions can solve recurrences that no other method in this chapter can handle, including full-history recurrences and recurrences with non-constant coefficients.

## Ordinary Generating Functions

An **ordinary generating function** (OGF) for a sequence $\{a_n\}_{n=0}^{\infty}$ is the formal power series:

$$
A(x) = \sum_{n=0}^{\infty} a_n \, x^n = a_0 + a_1 x + a_2 x^2 + a_3 x^3 + \cdots
$$

The variable $x$ is a formal placeholder; convergence is not required for the algebraic manipulations to be valid. The key idea is that operations on the power series correspond to operations on the sequence:

| Operation on $A(x)$ | Effect on $\{a_n\}$ |
|---------------------|---------------------|
| $x \cdot A(x)$ | Shift: $\{0, a_0, a_1, a_2, \ldots\}$ |
| $A(x) - a_0$ | Remove first term |
| $\frac{A(x) - a_0}{x}$ | Shift left: $\{a_1, a_2, a_3, \ldots\}$ |
| $A(x) + B(x)$ | Termwise sum: $\{a_n + b_n\}$ |
| $A(x) \cdot B(x)$ | Convolution: $\{c_n = \sum_{k=0}^n a_k b_{n-k}\}$ |

## The Method in Four Steps

Solving a recurrence with generating functions follows a systematic procedure:

1. **Define**: Let $A(x) = \sum_{n \geq 0} a_n x^n$ where $a_n$ is the quantity defined by the recurrence
2. **Translate**: Multiply both sides of the recurrence by $x^n$ and sum over all valid $n$ to obtain an equation in $A(x)$
3. **Solve**: Manipulate the equation algebraically to isolate $A(x)$ as a closed-form expression in $x$
4. **Extract**: Expand $A(x)$ using partial fractions, the geometric series, or other identities to read off $a_n = [x^n] A(x)$

## Example 1: The Fibonacci Recurrence

The Fibonacci sequence satisfies $F_n = F_{n-1} + F_{n-2}$ for $n \geq 2$, with $F_0 = 0$ and $F_1 = 1$.

### Step 1: Define

$$
F(x) = \sum_{n=0}^{\infty} F_n \, x^n
$$

### Step 2: Translate

Multiply the recurrence $F_n = F_{n-1} + F_{n-2}$ by $x^n$ and sum for $n \geq 2$:

$$
\sum_{n=2}^{\infty} F_n x^n = \sum_{n=2}^{\infty} F_{n-1} x^n + \sum_{n=2}^{\infty} F_{n-2} x^n
$$

The left side is $F(x) - F_0 - F_1 x = F(x) - x$. The first sum on the right is $x(F(x) - F_0) = xF(x)$. The second sum is $x^2 F(x)$. So:

$$
F(x) - x = xF(x) + x^2 F(x)
$$

### Step 3: Solve

$$
F(x)(1 - x - x^2) = x
$$

$$
F(x) = \frac{x}{1 - x - x^2}
$$

### Step 4: Extract

Factor the denominator. The roots of $1 - x - x^2 = 0$ are $x = \frac{-1 \pm \sqrt{5}}{2}$, so the roots of the characteristic equation $t^2 - t - 1 = 0$ are $\phi = \frac{1+\sqrt{5}}{2}$ and $\hat{\phi} = \frac{1-\sqrt{5}}{2}$.

Using partial fractions:

$$
F(x) = \frac{1}{\sqrt{5}} \left( \frac{1}{1 - \phi x} - \frac{1}{1 - \hat{\phi} x} \right)
$$

Expanding each geometric series $\frac{1}{1 - rx} = \sum_{n \geq 0} r^n x^n$:

$$
F_n = \frac{1}{\sqrt{5}} \left( \phi^n - \hat{\phi}^n \right)
$$

This is **Binet's formula**, an exact closed form for the $n$-th Fibonacci number.

## Example 2: A Linear Recurrence with Constant Coefficients

Consider $a_n = 5a_{n-1} - 6a_{n-2}$ for $n \geq 2$, with $a_0 = 1$ and $a_1 = 4$.

### Step 1: Define

$$
A(x) = \sum_{n=0}^{\infty} a_n x^n
$$

### Step 2: Translate

$$
A(x) - a_0 - a_1 x = 5x(A(x) - a_0) + (-6x^2) A(x)
$$

$$
A(x) - 1 - 4x = 5xA(x) - 5x - 6x^2 A(x)
$$

### Step 3: Solve

$$
A(x)(1 - 5x + 6x^2) = 1 + 4x - 5x = 1 - x
$$

$$
A(x) = \frac{1 - x}{1 - 5x + 6x^2} = \frac{1 - x}{(1 - 2x)(1 - 3x)}
$$

### Step 4: Extract via Partial Fractions

Write $\frac{1 - x}{(1-2x)(1-3x)} = \frac{A}{1-2x} + \frac{B}{1-3x}$.

Setting $x = 1/2$: $A = \frac{1 - 1/2}{1 - 3/2} = \frac{1/2}{-1/2} = -1$.

Setting $x = 1/3$: $B = \frac{1 - 1/3}{1 - 2/3} = \frac{2/3}{1/3} = 2$.

So:

$$
A(x) = \frac{-1}{1-2x} + \frac{2}{1-3x}
$$

Expanding:

$$
a_n = -2^n + 2 \cdot 3^n
$$

Verification: $a_0 = -1 + 2 = 1$ and $a_1 = -2 + 6 = 4$, matching the initial conditions.

## Example 3: A Recurrence with Polynomial Toll

Consider $T(n) = 2T(n-1) + 1$ for $n \geq 1$, with $T(0) = 0$. This is a non-divide-and-conquer recurrence (subtractive rather than divisive), but generating functions handle it naturally.

### Steps 1-2: Define and Translate

$$
G(x) = \sum_{n=0}^{\infty} T(n) x^n
$$

$$
G(x) - T(0) = 2x \, G(x) + \sum_{n=1}^{\infty} x^n = 2x \, G(x) + \frac{x}{1-x}
$$

### Step 3: Solve

$$
G(x)(1 - 2x) = \frac{x}{1-x}
$$

$$
G(x) = \frac{x}{(1-x)(1-2x)}
$$

### Step 4: Extract

Partial fractions: $\frac{x}{(1-x)(1-2x)} = \frac{-1}{1-x} + \frac{1}{1-2x}$.

Verification: at $x = 1$, $\frac{-1}{0}$ diverges, so use cover-up more carefully. Setting $\frac{x}{(1-x)(1-2x)} = \frac{A}{1-x} + \frac{B}{1-2x}$:

$$
x = A(1-2x) + B(1-x)
$$

Setting $x = 1$: $1 = A(-1)$, so $A = -1$. Setting $x = 1/2$: $1/2 = B(1/2)$, so $B = 1$.

$$
T(n) = -1 + 2^n = 2^n - 1
$$

Verification: $T(1) = 2(0) + 1 = 1 = 2^1 - 1$ and $T(2) = 2(1) + 1 = 3 = 2^2 - 1$.

## Useful Generating Function Identities

The following closed-form generating functions appear frequently when extracting coefficients:

$$
\frac{1}{1 - rx} = \sum_{n=0}^{\infty} r^n x^n
$$

$$
\frac{1}{(1-x)^2} = \sum_{n=0}^{\infty} (n+1) x^n
$$

$$
\frac{1}{(1-x)^{k+1}} = \sum_{n=0}^{\infty} \binom{n+k}{k} x^n
$$

$$
e^x = \sum_{n=0}^{\infty} \frac{x^n}{n!}
$$

The last identity is relevant when using **exponential generating functions** (EGFs), where $\hat{A}(x) = \sum a_n x^n / n!$. EGFs are particularly natural for recurrences involving factorials or permutations.

## Scope and Limitations

Generating functions are the most general method in this chapter for solving recurrences, but they come with trade-offs:

- **Strengths**: Handle non-constant coefficients, full-history recurrences ($a_n = \sum_{k=0}^{n-1} a_k$), and recurrences that no theorem covers. They also yield exact solutions, not just asymptotic bounds.
- **Weaknesses**: The algebra can become intricate. Partial fraction decomposition for high-degree polynomials is tedious. For standard divide-and-conquer recurrences $T(n) = aT(n/b) + f(n)$, the [Master theorem](master.md) or [Akra-Bazzi method](akra_bazzi.md) is far simpler.

!!! tip "When to Use Generating Functions"
    Use generating functions when the recurrence has one or more of: non-constant coefficients, full-history dependence, a subtractive step size ($T(n-1)$ rather than $T(n/b)$), or when an exact closed-form solution (not just $\Theta$-notation) is needed.

## Connections to Other Topics

- **[Substitution Method](substitution.md)**: Once a generating function yields a closed form, substitution can verify it directly in the original recurrence
- **[Master Theorem](master.md)**: Handles divide-and-conquer recurrences more directly when applicable
- **[Akra-Bazzi Method](akra_bazzi.md)**: Another general method, but gives $\Theta$-bounds rather than exact solutions

## References

- Graham, R. L., Knuth, D. E., & Patashnik, O. (1994). *Concrete Mathematics* (2nd ed.), Chapter 7. Addison-Wesley.
- Wilf, H. S. (2006). *generatingfunctionology* (3rd ed.). A K Peters.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Appendix C. MIT Press.
