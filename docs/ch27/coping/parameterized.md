# Parameterized Algorithms

Some NP-hard problems become tractable when a natural **parameter** $k$ is small. A problem is **fixed-parameter tractable (FPT)** if it can be solved in time $f(k) \cdot n^{O(1)}$, confining the exponential blowup to $k$ alone. When $k$ is much smaller than $n$, FPT algorithms are practical even for large inputs. This page introduces parameterized complexity, the FPT class, kernelization, and the W-hierarchy.

## Fixed-Parameter Tractability

!!! tip "Definition: FPT"
    A parameterized problem $(L, k)$ is **fixed-parameter tractable** if there exists an algorithm solving it in time $f(k) \cdot n^{O(1)}$, where $f$ is a computable function depending only on $k$ and $n$ is the input size.

The key distinction from polynomial time is that $f(k)$ may be exponential or worse in $k$, but the dependence on $n$ is polynomial. For small $k$, this is dramatically faster than $O(n^k)$.

**Example:** Vertex Cover parameterized by solution size $k$.

- Brute force: $O(\binom{n}{k} \cdot m) = O(n^k \cdot m)$, not FPT.
- FPT algorithm: $O(2^k \cdot n)$ via bounded search tree.

At $k = 20$ and $n = 10^6$: the FPT algorithm does $\approx 10^{12}$ operations, while brute force does $\approx 10^{126}$.

## Bounded Search Tree

The simplest FPT technique for vertex cover:

1. Pick any edge $(u, v)$. At least one of $u, v$ must be in any vertex cover of size $\leq k$.
2. **Branch:** Either include $u$ (and recurse with $k - 1$) or include $v$ (and recurse with $k - 1$).
3. **Base case:** If $k = 0$ and edges remain, return "no." If no edges remain, return "yes."

The recursion tree has depth $k$ and branching factor 2, giving $2^k$ leaves. Each node does $O(n)$ work.

**Time:** $O(2^k \cdot n)$.

## Kernelization

**Kernelization** is a polynomial-time preprocessing step that reduces the instance to a smaller **kernel** whose size depends only on $k$.

!!! tip "Definition: Kernel"
    A kernelization for parameterized problem $(I, k)$ is a polynomial-time algorithm that produces an equivalent instance $(I', k')$ with $|I'| \leq g(k)$ and $k' \leq k$, for some computable $g$.

A problem is FPT if and only if it has a kernelization (possibly with exponential kernel size).

### Vertex Cover Kernel

**Crown reduction** and **Buss's rule** reduce $k$-Vertex Cover to a kernel of size $O(k^2)$:

1. **Remove isolated vertices** (they cannot be in a minimum cover).
2. **Buss's rule:** If any vertex $v$ has degree $> k$, it must be in the cover (otherwise $> k$ of its neighbors are needed). Include $v$, reduce $k$ by 1, remove $v$ and its edges.
3. **Crown reduction:** If more than $k^2$ edges remain after applying rule 2, the answer is "no" (a vertex cover of size $k$ can cover at most $k \cdot k = k^2$ edges via its endpoints).

The resulting kernel has at most $k^2$ edges and $2k^2$ vertices.

## The W-Hierarchy

Not all parameterized problems are FPT. The **W-hierarchy** classifies parameterized intractability:

$$
\text{FPT} \subseteq \text{W}[1] \subseteq \text{W}[2] \subseteq \cdots \subseteq \text{XP}
$$

where **XP** is the class solvable in $O(n^{f(k)})$ time (polynomial for each fixed $k$, but the degree depends on $k$).

### W[1]-Hard Problems

A problem is **W[1]-hard** if it is at least as hard as $k$-Clique under parameterized reductions. W[1]-hard problems are believed not to be FPT.

| Problem | Parameter | Complexity |
|---------|-----------|-----------|
| Vertex Cover | $k$ (cover size) | FPT |
| $k$-Path | $k$ (path length) | FPT (color-coding) |
| $k$-Clique | $k$ (clique size) | W[1]-complete |
| Independent Set | $k$ (set size) | W[1]-complete |
| Dominating Set | $k$ (set size) | W[2]-complete |
| Set Cover | $k$ (number of sets) | W[2]-complete |

### Parameterized Reductions

An FPT reduction from $(L_1, k_1)$ to $(L_2, k_2)$ maps instances in FPT time such that $k_2 = g(k_1)$. If $L_2$ is FPT, then $L_1$ is FPT. Contrapositive: if $L_1$ is W[1]-hard and reduces to $L_2$, then $L_2$ is W[1]-hard.

## Color-Coding

The **color-coding** technique (Alon, Yuster, Zwick, 1995) solves $k$-Path in $O(2^k \cdot m)$ expected time:

1. Randomly color each vertex with one of $k$ colors.
2. Use DP to find a path of length $k$ using all $k$ colors (**colorful** path).
3. Repeat $O(e^k)$ times to boost success probability.

A colorful path exists if and only if a $k$-path exists and the random coloring assigns distinct colors. The probability of a correct coloring is $k!/k^k \geq e^{-k}$.

## Important FPT Results

| Problem | Parameter | FPT Time | Technique |
|---------|-----------|----------|-----------|
| Vertex Cover | $k$ | $O(1.2738^k + kn)$ | Branching + reduction rules |
| Feedback Vertex Set | $k$ | $O(3.619^k \cdot n)$ | Iterative compression |
| $k$-Path | $k$ | $O(1.657^k \cdot m)$ | Narrow sieves |
| Treewidth | $k$ | $O(2^{O(k^3)} \cdot n)$ | FPT algorithm exists |
| Planar Dominating Set | $k$ | $O(2^{O(\sqrt{k})} \cdot n^{O(1)})$ | Bidimensionality |

??? example "Example: Bounded Search Tree for Vertex Cover"
    **Graph:** Edges $\{(a,b), (b,c), (c,d), (a,d)\}$, parameter $k = 2$.

    **Step 1:** Pick edge $(a,b)$. Branch:

    - **Include $a$:** Remove $a$ and edges $(a,b), (a,d)$. Remaining: $(b,c), (c,d)$. $k = 1$.
        - Pick edge $(b,c)$. Branch on $b$ or $c$ with $k = 0$.
        - Include $c$: removes $(b,c), (c,d)$. No edges left. **Success:** $\{a, c\}$.
    - **Include $b$:** Remove $b$ and edges $(a,b), (b,c)$. Remaining: $(c,d), (a,d)$. $k = 1$.
        - Pick $(c,d)$. Include $d$: removes $(c,d), (a,d)$. No edges left. **Success:** $\{b, d\}$.

    Both branches find valid covers of size 2.

## Reference

- Cygan, M., et al. (2015). *Parameterized Algorithms*. Springer.
- Downey, R. G., & Fellows, M. R. (2013). *Fundamentals of Parameterized Complexity*. Springer.
- Flum, J., & Grohe, M. (2006). *Parameterized Complexity Theory*. Springer.
