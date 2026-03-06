# Chapter Overview

This chapter covers the fundamental algorithms for string matching, one of the most important problems in computer science. Given a text $T$ of length $n$ and a pattern $P$ of length $m$, the goal is to find all occurrences of $P$ in $T$.

We organize the material into three major areas. **Exact single-pattern matching** includes the naive brute-force method, the Knuth-Morris-Pratt (KMP) algorithm with its failure function, the Boyer-Moore algorithm with its bad-character and good-suffix heuristics, the Rabin-Karp rolling-hash approach, and the Z-algorithm. **Multiple-pattern matching** covers the Aho-Corasick automaton together with its failure links and dictionary links. **Regular expression matching** introduces NFA construction via Thompson's algorithm, the subset construction for converting an NFA to a DFA, and DFA state minimization.

$$

\text{String Matching}
\left\{\begin{array}{lll}
\text{Exact Matching} & O(nm) \text{ naive}, \; O(n+m) \text{ KMP/BM/Z}\\
\\
\text{Multiple Pattern} & O(n + m + z) \text{ Aho-Corasick}\\
\\
\text{Regular Expression} & \text{Thompson's NFA} \to \text{DFA} \to \text{Minimized DFA}
\end{array}\right.

$$

Throughout this chapter, we use 0-based indexing for strings unless otherwise noted.

# Reference

[Introduction to Algorithms (CLRS), Chapters 32](https://mitpress.mit.edu/books/introduction-to-algorithms-fourth-edition/)

[Algorithms on Strings, Trees and Sequences - Dan Gusfield](https://www.cambridge.org/core/books/algorithms-on-strings-trees-and-sequences/F0B095049C8347C7F1D2DC5F1D74AC5D)
