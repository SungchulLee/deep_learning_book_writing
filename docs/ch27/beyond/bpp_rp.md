# BPP and RP

Randomness appears to help computation: some problems have efficient randomized algorithms but no known deterministic polynomial-time solutions. The complexity classes **BPP** and **RP** formalize what randomized algorithms can achieve, with different error guarantees. Understanding these classes reveals the surprising fact that randomness may not provide essential computational power --- a conjecture captured by the belief that BPP = P.

## Probabilistic Turing Machines

A **probabilistic Turing machine (PTM)** is a Turing machine that, at each step, can flip a fair coin to decide its next move. Equivalently, it has two transition functions and randomly chooses between them at each step. The machine's output on a given input is a random variable.

## RP: One-Sided Error

!!! tip "Definition: RP (Randomized Polynomial Time)"
    A language $L$ is in **RP** if there exists a probabilistic polynomial-time Turing machine $M$ such that:

    - If $x \in L$: $\Pr[M \text{ accepts } x] \geq 1/2$
    - If $x \notin L$: $\Pr[M \text{ accepts } x] = 0$

RP has **one-sided error**: the machine never gives false positives. If it accepts, the input is definitely in $L$. If it rejects, the input might or might not be in $L$.

### co-RP

The complementary class **co-RP** flips the error side:

- If $x \in L$: $\Pr[M \text{ accepts } x] = 1$
- If $x \notin L$: $\Pr[M \text{ accepts } x] \leq 1/2$

A co-RP machine never gives false negatives.

### Error Amplification for RP

Running an RP machine $k$ times independently and accepting if any run accepts reduces the error probability exponentially:

$$
\Pr[\text{all } k \text{ runs reject} \mid x \in L] \leq \left(\frac{1}{2}\right)^k
$$

With $k = O(\log(1/\delta))$ repetitions, the error drops below any desired $\delta > 0$.

## BPP: Two-Sided Error

!!! tip "Definition: BPP (Bounded-Error Probabilistic Polynomial Time)"
    A language $L$ is in **BPP** if there exists a probabilistic polynomial-time Turing machine $M$ such that:

    - If $x \in L$: $\Pr[M \text{ accepts } x] \geq 2/3$
    - If $x \notin L$: $\Pr[M \text{ accepts } x] \leq 1/3$

BPP has **two-sided error**: the machine can err in both directions, but the error probability is bounded away from $1/2$.

### Error Amplification for BPP

The constants $2/3$ and $1/3$ are arbitrary --- any constants bounded away from $1/2$ yield the same class. Running $M$ independently $k$ times and taking a **majority vote**:

$$
\Pr[\text{majority wrong}] \leq \exp(-\Omega(k))
$$

by the Chernoff bound. With $k = O(\log(1/\delta))$ repetitions, the error drops below $\delta$.

### BPP Is Closed Under Complement

BPP = co-BPP: simply flip the accept/reject output. This symmetry distinguishes BPP from RP, where one-sided error creates an asymmetry.

## Class Relationships

The following containments hold:

$$
\text{P} \subseteq \text{RP} \subseteq \text{NP}
$$

$$
\text{P} \subseteq \text{co-RP} \subseteq \text{co-NP}
$$

$$
\text{RP} \cup \text{co-RP} \subseteq \text{BPP}
$$

$$
\text{RP} \cap \text{co-RP} = \text{ZPP}
$$

where **ZPP** (Zero-error Probabilistic Polynomial time) is the class of problems solvable by randomized algorithms that always give the correct answer but may vary in running time (expected polynomial).

### Relationship to NP

It is believed (but unproven) that:

$$
\text{BPP} \subseteq \text{NP} \cap \text{co-NP}
$$

If NP-complete problems were in BPP, then NP = RP, which would be a major breakthrough.

### The BPP = P Conjecture

!!! warning "Open Problem"
    It is widely conjectured that BPP = P, meaning every problem solvable by efficient randomized algorithms is also solvable by efficient deterministic algorithms. This conjecture is supported by the Impagliazzo-Wigderson theorem: if any problem in E = DTIME($2^{O(n)}$) requires exponential-size circuits, then BPP = P.

## Examples

| Problem | Class | Notes |
|---------|-------|-------|
| Polynomial Identity Testing | co-RP | Schwartz-Zippel lemma |
| Primality Testing | Was in co-RP | Now in P (AKS, 2002) |
| Perfect Matching (bipartite) | P | Deterministic via augmenting paths |
| Undirected Connectivity | L $\subseteq$ P | Random walks solve in randomized logspace |

### Polynomial Identity Testing

Given an arithmetic circuit computing a polynomial $p(x_1, \ldots, x_n)$, determine whether $p \equiv 0$. By the **Schwartz-Zippel lemma**, evaluating $p$ at a random point from a sufficiently large set detects non-zero polynomials with high probability. No efficient deterministic algorithm is known.

### Primality as Historical Example

Before AKS (2002), the best primality test was Miller-Rabin, which places primality in co-RP. The AKS deterministic polynomial-time algorithm moved primality into P, illustrating that randomness was not essential for this problem.

??? example "Example: Error Amplification"
    An RP algorithm for language $L$ accepts YES instances with probability $\geq 1/2$.

    **Goal:** Reduce error to $< 0.001$.

    **Method:** Run $k$ independent trials, accept if any trial accepts.

    **Required $k$:** $(1/2)^k < 0.001 \Rightarrow k > \log_2(1000) \approx 10$.

    With 10 repetitions, the probability of incorrectly rejecting a YES instance drops below $1/1024 < 0.001$. The total running time is $10 \cdot T(n)$, which remains polynomial.

## Reference

- Sipser, M. (2012). *Introduction to the Theory of Computation* (3rd ed.). Cengage Learning.
- Arora, S., & Barak, B. (2009). *Computational Complexity: A Modern Approach*. Cambridge University Press, Chapter 7.
