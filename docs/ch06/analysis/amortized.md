# Amortized Cost

A single insertion into a hash table with dynamic resizing occasionally triggers a full rehash of all $n$ elements, costing $\Theta(n)$ time. This seems to contradict the claim that hash table operations run in $O(1)$ time. Amortized analysis resolves this apparent contradiction by showing that the expensive rehash operations occur so rarely that, averaged over any sequence of $n$ insertions, the cost per insertion is $O(1)$.

## Motivation

Consider building a hash table from scratch by inserting $n$ elements one at a time. The table starts with a small capacity and doubles whenever the load factor exceeds a threshold (typically $\alpha > 0.75$). Most insertions are cheap -- they compute a hash and append to a chain in $O(1)$ time. But each doubling copies all existing elements to the new table, costing $\Theta(n_i)$ time where $n_i$ is the number of elements at the time of the $i$-th resize. Despite these occasional expensive operations, we will prove that the total cost of $n$ insertions is $O(n)$, giving an amortized cost of $O(1)$ per insertion.

## Aggregate Method

The aggregate method computes the total cost of $n$ operations and divides by $n$.

**Setup.** Start with a table of size $m_0 = 1$. Double the table size whenever the table becomes full ($n = m$). After $n$ insertions, the table has been doubled $\lfloor \log_2 n \rfloor$ times.

**Total cost.** Each insertion has a basic cost of 1 (computing the hash and storing the element). The $i$-th doubling occurs after $2^i$ insertions and copies $2^i$ elements. The total cost of $n$ insertions is:

$$
T(n) = n + \sum_{i=0}^{\lfloor \log_2 n \rfloor} 2^i
$$

The first term $n$ counts the basic cost of each insertion. The summation counts the copying cost of all doublings. Evaluating the geometric series:

$$
\sum_{i=0}^{\lfloor \log_2 n \rfloor} 2^i = 2^{\lfloor \log_2 n \rfloor + 1} - 1 \leq 2n - 1
$$

Therefore:

$$
T(n) \leq n + 2n - 1 = 3n - 1
$$

The amortized cost per insertion is:

$$
\hat{c} = \frac{T(n)}{n} \leq \frac{3n - 1}{n} < 3 = O(1)
$$

## Potential Method

The potential method assigns a potential $\Phi$ to the data structure at each step. The amortized cost of an operation is its actual cost plus the change in potential:

$$
\hat{c}_i = c_i + \Phi_i - \Phi_{i-1}
$$

**Potential function.** Define the potential after $i$ insertions as:

$$
\Phi_i = 2n_i - m_i
$$

where $n_i$ is the number of elements and $m_i$ is the table size after the $i$-th operation. This potential tracks how close the table is to needing a resize.

**Initial condition.** $\Phi_0 = 2 \cdot 0 - m_0 = -m_0$. We can assume $m_0 = 1$ and $n_0 = 0$, giving $\Phi_0 = -1$. For the analysis, we want $\Phi_i \geq \Phi_0$ for all $i$, which holds since $n_i / m_i \leq 1$ implies $2n_i \geq m_i$ whenever $n_i \geq m_i / 2$.

**Case 1: Insertion without resize.** The actual cost is $c_i = 1$. The table size does not change ($m_i = m_{i-1}$), and the count increases by 1 ($n_i = n_{i-1} + 1$):

$$
\hat{c}_i = 1 + (2n_i - m_i) - (2n_{i-1} - m_{i-1}) = 1 + 2 = 3
$$

**Case 2: Insertion with resize.** The table is full ($n_{i-1} = m_{i-1}$), so the new table size is $m_i = 2m_{i-1}$ and the new count is $n_i = n_{i-1} + 1$. The actual cost is $c_i = 1 + n_{i-1}$ (insert plus copy all existing elements):

$$
\hat{c}_i = (1 + n_{i-1}) + (2n_i - m_i) - (2n_{i-1} - m_{i-1})
$$

Substituting $n_i = n_{i-1} + 1$ and $m_i = 2m_{i-1} = 2n_{i-1}$:

$$
\hat{c}_i = (1 + n_{i-1}) + (2(n_{i-1} + 1) - 2n_{i-1}) - (2n_{i-1} - n_{i-1})
$$

$$
= (1 + n_{i-1}) + 2 - n_{i-1} = 3
$$

In both cases, the amortized cost is exactly 3, confirming $O(1)$ amortized cost per insertion.

## Accounting Method

The accounting method assigns each operation a fixed "charge" (the amortized cost) and saves any excess as credit stored in the data structure.

**Charge each insertion \$3:**

- \$1 pays for the insertion itself.
- \$1 is saved as credit on the newly inserted element.
- \$1 is saved as credit on one element that was present at the last resize.

When a resize occurs, every element in the table has accumulated \$1 of credit. Since the table has $n$ elements and the resize costs $\Theta(n)$, the stored credits exactly pay for the copying.

**Credit invariant.** At all times, every element inserted since the last resize has \$1 of credit. When the table is full ($n = m$), the total credit is $n$, which pays for the $\Theta(n)$ resize cost.

??? example "Amortized Cost Trace"

    Track a table starting at size 1, doubling when full:

    | Operation | $n$ | $m$ | Actual cost | Credit change | Amortized cost |
    |---|---|---|---|---|---|
    | Insert 1 | 1 | 1 | 1 (insert) | +2 | 3 |
    | Insert 2 | 2 | 2 | 2 (resize + insert) | -1 + 2 = +1 | 3 |
    | Insert 3 | 3 | 4 | 3 (resize + insert) | -2 + 2 = 0 | 3 |
    | Insert 4 | 4 | 4 | 1 (insert) | +2 | 3 |
    | Insert 5 | 5 | 8 | 5 (resize + insert) | -4 + 2 = -2 | 3 |
    | Insert 6 | 6 | 8 | 1 (insert) | +2 | 3 |
    | Insert 7 | 7 | 8 | 1 (insert) | +2 | 3 |
    | Insert 8 | 8 | 8 | 1 (insert) | +2 | 3 |

    Total actual cost: $1 + 2 + 3 + 1 + 5 + 1 + 1 + 1 = 15$. Total amortized cost: $8 \times 3 = 24$. The amortized total always exceeds the actual total, confirming the analysis.

## Amortized Deletion

Deletion with table shrinking (halving when the load factor drops below $1/4$) also admits $O(1)$ amortized analysis using the same techniques. The asymmetric thresholds (double at $\alpha = 1$, halve at $\alpha = 1/4$) prevent thrashing -- a pathological pattern where alternating insertions and deletions near a boundary repeatedly trigger resizes.

The potential function for the combined insert/delete case is:

$$
\Phi_i =
\begin{cases}
2n_i - m_i & \text{if } n_i \geq m_i / 2 \\
m_i / 2 - n_i & \text{if } n_i < m_i / 2
\end{cases}
$$

This potential is zero when the table is half full and increases as the table approaches either full or quarter-full, accumulating enough credit to pay for the next resize in either direction.

## Summary

Amortized analysis of hash table resizing shows that each insertion costs $O(1)$ amortized time, despite occasional $\Theta(n)$ resize operations. The aggregate method directly computes the total cost as $O(n)$. The potential method confirms a constant amortized cost of 3 per insertion by tracking the gap between twice the element count and the table size. The accounting method provides an intuitive interpretation: each insertion saves enough credit to pay for its share of the next resize.

## Reference

- [Introduction to Algorithms (CLRS), Chapter 11](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
- [Introduction to Algorithms (CLRS), Chapter 16 — Amortized Analysis](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
