# Cook-Levin Theorem

How do we know that NP-complete problems exist at all?  Before 1971, it was conceivable that no single problem could be "hardest" in NP.  The **Cook-Levin theorem** settled this by proving that SAT is NP-complete -- the first problem ever shown to have this property.  The proof works by encoding the computation of *any* nondeterministic Turing machine as a Boolean formula, establishing that SAT is a universal target for polynomial-time reductions from all of NP.

## Theorem Statement

**Theorem (Cook, 1971; Levin, 1973).** The Boolean satisfiability problem (SAT) is NP-complete.

This asserts two things:

1. SAT $\in$ NP (easy: a satisfying assignment is a polynomial-length certificate).
2. Every language $L \in$ NP is polynomial-time many-one reducible to SAT.

The second claim is the substantial part: *any* NP computation can be expressed as a question about Boolean satisfiability.

## Proof Strategy

The proof constructs a polynomial-time reduction from an arbitrary NP language $L$ to SAT.  Since $L \in$ NP, there exists a nondeterministic Turing machine $N$ that decides $L$ in time $p(n)$ for some polynomial $p$.  The reduction encodes $N$'s computation on input $x$ as a Boolean formula $\phi_x$ such that:

$$
x \in L \iff \phi_x \text{ is satisfiable}
$$

## Proof Sketch

### Step 1: Computation Tableau

A nondeterministic TM $N$ running on input $x$ of length $n$ in time $T = p(n)$ can be described by a **tableau** -- a $T \times T$ grid where:

- Row $i$ represents the configuration at time step $i$.
- Column $j$ represents tape cell $j$.
- Each cell contains a symbol from a finite alphabet $\Gamma' = \Gamma \cup (Q \times \Gamma)$, where $Q$ is the state set and $\Gamma$ is the tape alphabet.  A cell $(i, j)$ containing $(q, a)$ means the head is at position $j$ in state $q$ reading symbol $a$ at time $i$.

### Step 2: Boolean Variables

Introduce Boolean variables to encode the tableau:

$$
\text{cell}[i, j, s] \quad \text{for } 0 \leq i, j \leq T-1, \; s \in \Gamma'
$$

Variable $\text{cell}[i, j, s] = 1$ means "cell $(i, j)$ contains symbol $s$."

### Step 3: Formula Construction

The formula $\phi_x$ is the conjunction of four types of constraints:

**Cell consistency.** Each cell contains exactly one symbol:

$$
\phi_{\text{cell}} = \bigwedge_{i,j} \left[ \left(\bigvee_{s \in \Gamma'} \text{cell}[i,j,s]\right) \;\wedge\; \bigwedge_{s \neq s'} \left(\neg \text{cell}[i,j,s] \vee \neg \text{cell}[i,j,s']\right) \right]
$$

**Initial configuration.** Row 0 encodes the starting configuration: $N$ is in start state $q_0$, input $x$ is on the tape, and remaining cells are blank.

**Acceptance.** At least one cell in the tableau contains an accepting state $q_{\text{acc}}$.

**Transition consistency.** Every $2 \times 3$ window of the tableau is consistent with $N$'s transition function.  For each pair of adjacent rows, every group of three consecutive cells in row $i$ must produce valid successor cells in row $i + 1$.  This is encoded as a disjunction over all valid local patterns (called **legal windows**).

### Step 4: Correctness

- **If $x \in L$**: some computation path of $N$ accepts, defining a valid tableau.  Setting variables according to this tableau satisfies $\phi_x$.
- **If $x \notin L$**: no computation path accepts, so no valid accepting tableau exists, and $\phi_x$ is unsatisfiable.

### Step 5: Polynomial Size

The tableau is $T \times T$ with $|\Gamma'|$ choices per cell, giving $O(T^2 \cdot |\Gamma'|)$ variables.  Since $|\Gamma'|$ is constant (determined by $N$) and $T = p(n)$, the number of variables is $O(p(n)^2)$, which is polynomial in $n$.  The formula size is also polynomial because each constraint involves a constant number of variables per window.

$\square$

## Key Observations

### Why This Proof Works

The brilliance of the Cook-Levin proof lies in its universality.  The encoding does not depend on the specific NP language $L$ -- it works for *any* NP language because it directly encodes TM computation.  The only problem-specific part is the choice of $N$ and its time bound $p(n)$.

### The Tableau Method

The tableau technique is reused throughout complexity theory:

- Proving PSPACE-completeness of QSAT.
- Showing undecidability results.
- Establishing completeness results for other complexity classes.

??? info "Cook vs. Levin"
    Stephen Cook presented his proof at STOC 1971 using Turing reductions.  Independently, Leonid Levin proved a similar result in the Soviet Union in 1973 using many-one reductions and a more general framework of "universal search problems."  The theorem is named for both.

## Consequences

The Cook-Levin theorem unlocked the entire theory of NP-completeness:

1. **First NP-complete problem**: SAT became the anchor from which all other NP-completeness proofs derive.
2. **Reduction cascade**: by reducing SAT to 3-SAT, then 3-SAT to CLIQUE, VERTEX COVER, etc., Karp (1972) established NP-completeness for 21 problems.
3. **Structural complexity**: the theorem shows that NP has a notion of "hardest problems," giving the class internal structure.
4. **P vs NP significance**: if any polynomial-time algorithm for SAT were found, $\mathbf{P} = \mathbf{NP}$ would follow.

## From SAT to 3-SAT

The Cook-Levin construction produces formulas that may have large clauses.  The standard next step is to reduce SAT to 3-SAT:

**Theorem.** 3-SAT is NP-complete.

The reduction replaces each clause $(l_1 \vee l_2 \vee \cdots \vee l_k)$ with $k > 3$ by introducing $k - 3$ fresh variables and producing $k - 2$ clauses of width 3.  This preserves satisfiability and runs in polynomial time.

3-SAT then serves as the standard starting point for NP-completeness reductions because its fixed clause width simplifies gadget constructions.

## Reference

- Cook, S. "The Complexity of Theorem-Proving Procedures." *STOC*, 1971.
- Levin, L. "Universal Sequential Search Problems." *Problems of Information Transmission*, 1973.
- Sipser, M. *Introduction to the Theory of Computation*. Cengage Learning.
- Arora, S. and Barak, B. *Computational Complexity: A Modern Approach*. Cambridge University Press.
