# Running Time

How long does an algorithm take? Measuring wall-clock time on a particular machine
gives a number that depends on the processor, the programming language, the compiler,
and even what other programs are running. To compare algorithms independently of these
factors, we define **running time** as a mathematical function of the input size,
counted in terms of elementary operations.

## Input Size

The **input size** $n$ is a measure of the amount of data the algorithm receives.
What constitutes "size" depends on the problem:

| Problem | Input Size $n$ |
|---|---|
| Sorting an array | Number of elements |
| Multiplying two integers | Number of digits (or bits) |
| Graph algorithms | Number of vertices $|V|$ and edges $|E|$ |
| Matrix operations | Dimension (e.g., $n \times n$) |

Some problems have multiple parameters. For graph algorithms, the running time is
often expressed as a function of both $|V|$ and $|E|$, such as $O(|V| + |E|)$.

## The RAM Model

To reason about running time without specifying hardware, we adopt the **Random Access
Machine (RAM) model**:

- The machine has an unbounded number of memory cells, each holding a single value.
- Each **primitive operation** takes exactly one time step:
    - Arithmetic: addition, subtraction, multiplication, division, modulus
    - Comparison: $<$, $\leq$, $=$, $\geq$, $>$
    - Data movement: assignment, array access by index
    - Control flow: branch, function call, return
- Memory access is uniform: reading cell $i$ costs the same as reading cell $j$.

!!! note "Limitations of the RAM Model"

    The RAM model ignores cache effects, memory hierarchy, and parallelism. Despite
    this simplification, it predicts real-world performance remarkably well for most
    algorithms. Cache-aware analysis and external memory models address the cases
    where it falls short.

## Running Time as a Function

The **running time** $T(n)$ of an algorithm is the number of primitive operations it
executes on an input of size $n$. Since different inputs of the same size may cause
different operation counts, $T(n)$ is often qualified as a best-case, worst-case, or
average-case quantity.

For a concrete algorithm, $T(n)$ is determined by:

1. **Counting operations** line by line.
2. **Summing** the counts over all executed lines.
3. **Expressing** the sum as a function of $n$.

??? example "Running Time of Linear Search"

    ```
    LinearSearch(A, n, target):
    1.  for i = 0 to n - 1:
    2.      if A[i] == target:
    3.          return i
    4.  return -1
    ```

    - Line 1 (loop test): executes up to $n + 1$ times.
    - Line 2 (comparison): executes up to $n$ times.
    - Line 3 (return): executes at most 1 time.
    - Line 4 (return): executes at most 1 time.

    In the **worst case** (target absent), lines 1-2 execute $n$ and $n$ times
    respectively, so $T(n) = 2n + 2 = \Theta(n)$.

## Why Not Measure Wall-Clock Time?

Empirical timing is useful for benchmarking implementations, but it cannot serve as a
general performance metric because:

- **Machine dependence:** The same algorithm runs faster on a newer CPU.
- **Input dependence:** A single timing measurement applies only to that specific
  input.
- **Implementation dependence:** Language choice, compiler optimizations, and data
  structure selection all affect timing without changing the algorithm itself.

The mathematical function $T(n)$ abstracts away all these factors, allowing us to
compare algorithms on equal footing.

## From Exact Counts to Asymptotic Notation

Exact operation counts like $T(n) = 3n^2 + 7n + 4$ are cumbersome to compute and
difficult to compare. Asymptotic notation ($O$, $\Omega$, $\Theta$) simplifies the
analysis by focusing on the growth rate as $n$ becomes large:

$$
T(n) = 3n^2 + 7n + 4 = \Theta(n^2)
$$

The constant factors $3$, $7$, and $4$ — which depend on the specific machine and
implementation — are absorbed. What remains is the essential information: doubling the
input size quadruples the running time.

!!! tip "The Analysis Pipeline"

    The full pipeline for analyzing an algorithm's running time is:

    1. **Define input size** $n$.
    2. **Choose a computational model** (typically RAM).
    3. **Count operations** to obtain $T(n)$.
    4. **Identify the case** (best, worst, or average).
    5. **Simplify** using asymptotic notation.

    The subsequent pages in this section cover steps 3-5 in detail.

## Reference

- [Introduction to Algorithms (CLRS), Chapters 2-4](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
