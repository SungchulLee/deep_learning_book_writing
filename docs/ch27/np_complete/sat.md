# SAT and 3-SAT

The **Boolean satisfiability problem** (SAT) asks whether a Boolean formula can be made true by some assignment of its variables.  SAT holds a unique position in complexity theory: it was the *first* problem proven NP-complete (Cook-Levin theorem, 1971), and it serves as the starting point for virtually all subsequent NP-completeness proofs.  Its restricted variant **3-SAT**, where each clause has at most three literals, remains NP-complete and is the most commonly used source for reductions.

## Boolean Formulas

A **Boolean formula** over variables $x_1, x_2, \ldots, x_n$ is built from variables, negation ($\neg$), conjunction ($\wedge$), and disjunction ($\vee$).

A **literal** is a variable $x_i$ or its negation $\neg x_i$ (also written $\overline{x_i}$).

A **clause** is a disjunction of literals: $(\ell_1 \vee \ell_2 \vee \cdots \vee \ell_k)$.

A formula is in **conjunctive normal form** (CNF) if it is a conjunction of clauses:

$$
\phi = C_1 \wedge C_2 \wedge \cdots \wedge C_m
$$

where each $C_j = (\ell_{j,1} \vee \ell_{j,2} \vee \cdots \vee \ell_{j,k_j})$.

An **assignment** $\alpha : \{x_1, \ldots, x_n\} \to \{0, 1\}$ **satisfies** $\phi$ if $\phi$ evaluates to 1 (true) under $\alpha$.

## Problem Definitions

### SAT

**Input:** A Boolean formula $\phi$.

**Question:** Does there exist an assignment $\alpha$ that satisfies $\phi$?

### CNF-SAT

**Input:** A Boolean formula $\phi$ in conjunctive normal form.

**Question:** Does there exist a satisfying assignment for $\phi$?

### k-SAT

**Input:** A CNF formula $\phi$ where every clause has *exactly* $k$ literals.

**Question:** Does there exist a satisfying assignment for $\phi$?

!!! tip "Why CNF?"
    Any Boolean formula can be converted to an equisatisfiable CNF formula in polynomial time (using the Tseitin transformation).  Therefore, CNF-SAT and general SAT are polynomial-time equivalent.

## NP-Completeness of SAT

**Theorem (Cook-Levin).** SAT is NP-complete.

**SAT is in NP.** Given a formula $\phi$ and an assignment $\alpha$, verify that $\phi(\alpha) = 1$ by evaluating the formula in $O(|\phi|)$ time.  The assignment $\alpha$ serves as the certificate.

**SAT is NP-hard.** The Cook-Levin theorem shows this by encoding the computation of any nondeterministic Turing machine as a Boolean formula.  The details are covered in the Cook-Levin theorem page.

## The Importance of 3-SAT

### 3-SAT is NP-Complete

**Theorem.** 3-SAT is NP-complete.

*Proof sketch.* 3-SAT is clearly in NP (same verifier as SAT).  To show NP-hardness, reduce SAT to 3-SAT.  Given a CNF formula, transform each clause with $k > 3$ literals into an equivalent set of 3-literal clauses using auxiliary variables:

Replace $(l_1 \vee l_2 \vee \cdots \vee l_k)$ with:

$$
(l_1 \vee l_2 \vee y_1) \wedge (\neg y_1 \vee l_3 \vee y_2) \wedge \cdots \wedge (\neg y_{k-3} \vee l_{k-1} \vee l_k)
$$

Each new auxiliary variable $y_i$ is fresh.  The resulting formula is satisfiable if and only if the original clause is satisfiable.  Clauses with fewer than 3 literals are padded by repeating a literal. $\square$

### 2-SAT is in P

In contrast to 3-SAT, **2-SAT** (every clause has exactly 2 literals) is solvable in polynomial time using implication graphs and strongly connected components:

$$
\text{2-SAT} \in \mathbf{P}
$$

This sharp transition from P to NP-complete between $k = 2$ and $k = 3$ is a fundamental phenomenon in complexity theory.

## Structure of SAT Instances

### Satisfiable vs. Unsatisfiable

A random $k$-SAT formula with $n$ variables and $m$ clauses undergoes a **phase transition** at a critical clause-to-variable ratio $r^* = m/n$:

- Below $r^*$: almost all formulas are satisfiable.
- Above $r^*$: almost all formulas are unsatisfiable.

For 3-SAT, experiments and theoretical results place $r^* \approx 4.267$.

### Horn-SAT

A **Horn clause** has at most one positive literal.  **Horn-SAT** is solvable in linear time using unit propagation, placing it firmly in P.  This is another example of a structural restriction making SAT tractable.

## SAT Solvers in Practice

Despite NP-completeness, modern SAT solvers handle industrial instances with millions of variables using:

- **Unit propagation**: if a clause has one unassigned literal, assign it to make the clause true.
- **Conflict-driven clause learning (CDCL)**: when a conflict arises, analyze the cause and add a learned clause to prevent repeating the same mistake.
- **Backjumping**: instead of chronological backtracking, jump back to the decision that caused the conflict.
- **Restarts**: periodically restart the search to escape unpromising regions.

??? example "Practical SAT solving"
    A CDCL solver maintains a partial assignment and alternates between **decision** (choose an unassigned variable and value), **propagation** (apply unit propagation), and **conflict analysis** (when a contradiction is found, derive a learned clause and backjump).  On structured industrial instances, this approach often runs in near-linear time despite worst-case exponential complexity.

## Variants and Extensions

| Variant | Complexity | Notes |
|---------|-----------|-------|
| 2-SAT | P | Implication graph algorithm |
| 3-SAT | NP-complete | Standard reduction source |
| Horn-SAT | P | Unit propagation |
| MAX-SAT | NP-hard | Maximize satisfied clauses |
| Weighted MAX-SAT | NP-hard | Weighted objective |
| #SAT | #P-complete | Count satisfying assignments |
| UNSAT | co-NP-complete | Complement of SAT |

## Reference

- Sipser, M. *Introduction to the Theory of Computation*. Cengage Learning.
- Arora, S. and Barak, B. *Computational Complexity: A Modern Approach*. Cambridge University Press.
- Cook, S. "The Complexity of Theorem-Proving Procedures." *STOC*, 1971.
- Biere, A. et al. *Handbook of Satisfiability*. IOS Press, 2009.
