# Huffman Optimality Proof

The Huffman coding page described the algorithm: repeatedly merge the two least-frequent characters. But why does this greedy strategy produce an optimal prefix-free code? The proof follows the standard two-part greedy framework --- establishing the **greedy choice property** (the two lowest-frequency characters can be made siblings at maximum depth) and **optimal substructure** (after the merge, the reduced problem has an optimal solution that extends to the original). This page presents both lemmas and the complete inductive proof.

## Setup and Notation

Let $C = \{c_1, c_2, \ldots, c_n\}$ be an alphabet of $n$ characters with frequencies $f_1, f_2, \ldots, f_n > 0$. A prefix-free code is represented by a full binary tree $T$ in which each leaf corresponds to a character. The cost is:

$$
B(T) = \sum_{i=1}^{n} f_i \cdot d_T(c_i)
$$

where $d_T(c_i)$ is the depth of character $c_i$ in tree $T$.

**Goal.** Show that Huffman's algorithm constructs a tree $T^*$ minimizing $B(T)$ over all full binary trees with $n$ leaves.

## Lemma 1: Greedy Choice Property

!!! note "Lemma 1 (Lowest-Frequency Siblings)"
    Let $x$ and $y$ be the two characters with the lowest frequencies in $C$ (breaking ties arbitrarily). Then there exists an optimal prefix-free code in which $x$ and $y$ are siblings at the maximum depth.

**Proof.** Let $T^*$ be an optimal tree. Let $a$ and $b$ be two sibling leaves at the maximum depth of $T^*$.

**Case 1:** $\{a, b\} = \{x, y\}$. Done.

**Case 2:** $\{a, b\} \neq \{x, y\}$. Without loss of generality, assume $f_x \leq f_y$. Since $x$ and $y$ have the two smallest frequencies, $f_x \leq f_a$ and $f_y \leq f_b$ (possibly after relabeling $a, b$).

Construct $T'$ by swapping $x$ with $a$ in $T^*$ (i.e., $x$ moves to $a$'s position and $a$ moves to $x$'s position). The change in cost is:

$$
B(T') - B(T^*) = f_x \cdot d_{T^*}(a) + f_a \cdot d_{T^*}(x) - f_x \cdot d_{T^*}(x) - f_a \cdot d_{T^*}(a)
$$

$$
= (f_a - f_x)(d_{T^*}(x) - d_{T^*}(a))
$$

Since $f_a \geq f_x$ and $d_{T^*}(a) \geq d_{T^*}(x)$ (because $a$ is at maximum depth), we have $d_{T^*}(x) - d_{T^*}(a) \leq 0$. Therefore $B(T') - B(T^*) \leq 0$, so $T'$ is at least as good as $T^*$.

Now swap $y$ with $b$ in $T'$ (where $b$ is the sibling of $x$ at maximum depth). By an analogous argument, $B(T'') \leq B(T')$. In $T''$, characters $x$ and $y$ are siblings at maximum depth, and $B(T'') \leq B(T^*)$. Since $T^*$ is optimal, $T''$ is also optimal. $\square$

## Lemma 2: Optimal Substructure

!!! note "Lemma 2 (Optimal Substructure)"
    Let $x$ and $y$ be sibling leaves in an optimal tree $T^*$ for alphabet $C$. Define a reduced alphabet $C' = (C \setminus \{x, y\}) \cup \{z\}$ where $z$ is a new character with frequency $f_z = f_x + f_y$. If $T'$ is an optimal tree for $C'$, then replacing the leaf $z$ in $T'$ with an internal node having children $x$ and $y$ produces an optimal tree for $C$.

**Proof.** Let $T$ be the tree obtained by expanding leaf $z$ in $T'$ into an internal node with children $x$ and $y$. The cost of $T$ relates to the cost of $T'$:

For every character $c \notin \{x, y, z\}$, $d_T(c) = d_{T'}(c)$.

For $x$ and $y$: $d_T(x) = d_T(y) = d_{T'}(z) + 1$.

Therefore:

$$
B(T) = \sum_{c \neq x,y} f_c \cdot d_T(c) + f_x \cdot d_T(x) + f_y \cdot d_T(y)
$$

$$
= \sum_{c \neq x,y} f_c \cdot d_{T'}(c) + (f_x + f_y)(d_{T'}(z) + 1)
$$

$$
= \sum_{c \neq z} f_c \cdot d_{T'}(c) + f_z \cdot d_{T'}(z) + f_x + f_y
$$

$$
= B(T') + f_x + f_y
$$

So $B(T) = B(T') + f_x + f_y$.

**Claim:** $T$ is optimal for $C$. Suppose not --- there exists a tree $\hat{T}$ for $C$ with $B(\hat{T}) < B(T)$. By Lemma 1, we may assume $x$ and $y$ are siblings in $\hat{T}$. Collapsing $x$ and $y$ into a single leaf $z$ with $f_z = f_x + f_y$ yields a tree $\hat{T}'$ for $C'$ with:

$$
B(\hat{T}') = B(\hat{T}) - f_x - f_y < B(T) - f_x - f_y = B(T')
$$

This contradicts the optimality of $T'$ for $C'$. $\square$

## Main Theorem

**Theorem.** Huffman's algorithm produces an optimal prefix-free code.

**Proof by strong induction on** $n = |C|$.

**Base case** ($n = 2$): There are only two characters. The only full binary tree has both characters at depth 1. This is trivially optimal, and the algorithm produces it in one merge.

**Inductive step:** Assume the algorithm is optimal for all alphabets of size $< n$. Consider an alphabet $C$ of size $n$.

1. The algorithm selects $x$ and $y$, the two least-frequent characters, and merges them into $z$ with $f_z = f_x + f_y$. This forms a reduced alphabet $C'$ with $|C'| = n - 1$.

2. By the inductive hypothesis, the algorithm produces an optimal tree $T'$ for $C'$.

3. By Lemma 2, expanding $z$ in $T'$ to an internal node with children $x$ and $y$ yields an optimal tree $T$ for $C$.

4. This expansion is exactly what the algorithm does --- it kept $x$ and $y$ as children of $z$ during the merge.

Therefore, the algorithm produces an optimal tree for $C$. $\square$

## Key Insight: Why the Proof Works

The proof hinges on the relationship $B(T) = B(T') + f_x + f_y$. The term $f_x + f_y$ is a constant that does not depend on the shape of the tree --- it comes solely from the fact that $x$ and $y$ are one level deeper than $z$. This means minimizing $B(T)$ over all trees for $C$ where $x$ and $y$ are siblings is equivalent to minimizing $B(T')$ over all trees for $C'$. The greedy merge does not lose any information about the remaining optimization.

## Uniqueness

The Huffman tree is not necessarily unique --- different tie-breaking rules when frequencies are equal produce different trees. However, all optimal trees have the same cost $B(T^*)$. Moreover, all optimal prefix-free codes for a given frequency distribution have the same set of codeword lengths (up to permutation of characters with equal frequency).

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 16.3. MIT Press.
- Huffman, D. A. (1952). A method for the construction of minimum-redundancy codes. *Proceedings of the IRE*, 40(9), 1098--1101.
