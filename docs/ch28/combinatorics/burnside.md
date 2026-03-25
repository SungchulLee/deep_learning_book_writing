# Burnside's Lemma

Burnside's lemma (also called the Cauchy-Frobenius lemma) answers a fundamental counting question: how many truly distinct objects exist when some objects are "the same" under symmetry? For instance, how many distinct necklaces can be made from colored beads when rotations are considered identical?

## Intuition

Naive counting over-counts objects that are related by symmetry. Burnside's lemma corrects this by averaging the number of objects fixed by each symmetry operation. The key insight is that the number of distinct objects equals the average number of fixed points across all symmetries in the group.

## Prerequisites

**Group action.** A group $G$ acts on a set $X$ if there is a mapping $G \times X \to X$, written $(g, x) \mapsto g \cdot x$, satisfying:

1. $e \cdot x = x$ for the identity $e \in G$ and all $x \in X$
2. $(gh) \cdot x = g \cdot (h \cdot x)$ for all $g, h \in G$ and $x \in X$

**Orbit.** The orbit of $x \in X$ under $G$ is $\text{Orb}(x) = \{g \cdot x : g \in G\}$. Two objects in the same orbit are considered equivalent.

**Fixed-point set.** For each $g \in G$, define:

$$
\text{Fix}(g) = \{x \in X : g \cdot x = x\}
$$

This is the set of objects left unchanged by the symmetry $g$.

## Statement

Let $G$ be a finite group acting on a finite set $X$. The number of distinct orbits is:

$$
|X / G| = \frac{1}{|G|} \sum_{g \in G} |\text{Fix}(g)|
$$

In words: the number of distinct objects under symmetry equals the average number of fixed points over all group elements.

## Proof

Define the set $S = \{(g, x) \in G \times X : g \cdot x = x\}$ and count $|S|$ in two ways.

**Counting by group elements:** For each $g \in G$, the elements $x$ with $g \cdot x = x$ form $\text{Fix}(g)$, so:

$$
|S| = \sum_{g \in G} |\text{Fix}(g)|
$$

**Counting by set elements:** For each $x \in X$, the elements $g$ with $g \cdot x = x$ form the stabilizer $\text{Stab}(x)$, so:

$$
|S| = \sum_{x \in X} |\text{Stab}(x)|
$$

By the orbit-stabilizer theorem, $|\text{Stab}(x)| = |G| / |\text{Orb}(x)|$. Substituting:

$$
|S| = \sum_{x \in X} \frac{|G|}{|\text{Orb}(x)|} = |G| \sum_{x \in X} \frac{1}{|\text{Orb}(x)|}
$$

Each orbit $O$ contributes $|O|$ terms of $1/|O|$, which sum to $1$. So the right side equals $|G| \cdot (\text{number of orbits})$. Combining:

$$
\sum_{g \in G} |\text{Fix}(g)| = |G| \cdot |X/G|
$$

$$
|X/G| = \frac{1}{|G|} \sum_{g \in G} |\text{Fix}(g)|
$$

## Example: Coloring a Square

**Problem.** How many distinct colorings of the vertices of a square exist using 2 colors, where two colorings are the same if one can be rotated into the other?

The rotation group of the square is $G = \{r_0, r_{90}, r_{180}, r_{270}\}$ with $|G| = 4$. The total set of colorings has $|X| = 2^4 = 16$ elements.

| Rotation $g$ | Description | $\lvert\text{Fix}(g)\rvert$ | Reason |
|---|---|---|---|
| $r_0$ (identity) | No rotation | 16 | Every coloring is fixed |
| $r_{90}$ | 90 degrees | 2 | All 4 vertices must match |
| $r_{180}$ | 180 degrees | 4 | Opposite pairs must match |
| $r_{270}$ | 270 degrees | 2 | All 4 vertices must match |

Applying Burnside's lemma:

$$
|X/G| = \frac{1}{4}(16 + 2 + 4 + 2) = \frac{24}{4} = 6
$$

There are **6** distinct colorings under rotation.

## Implementation

```python
from collections.abc import Callable


def burnside_count(
    elements: list,
    group: list[Callable],
) -> int:
    """Count distinct objects under a group action using Burnside's lemma.

    Args:
        elements: List of all objects in set X.
        group: List of functions, each mapping an element to its image
               under that group action.

    Returns:
        Number of distinct orbits.
    """
    total_fixed = 0
    for action in group:
        fixed = sum(1 for x in elements if action(x) == x)
        total_fixed += fixed
    return total_fixed // len(group)


if __name__ == "__main__":
    # === Square vertex colorings with 2 colors under rotation ===
    from itertools import product

    colorings = list(product(range(2), repeat=4))

    def rotate_0(c):
        return c

    def rotate_90(c):
        return (c[3], c[0], c[1], c[2])

    def rotate_180(c):
        return (c[2], c[3], c[0], c[1])

    def rotate_270(c):
        return (c[1], c[2], c[3], c[0])

    rotations = [rotate_0, rotate_90, rotate_180, rotate_270]
    distinct = burnside_count(colorings, rotations)
    print(f"Distinct square colorings (2 colors, rotations): {distinct}")
    # Output: 6
```

## Complexity

Computing Burnside's count requires evaluating $|\text{Fix}(g)|$ for each $g \in G$. If $|G| = m$ and $|X| = n$, the brute-force approach runs in $O(mn)$ time. In many applications (necklaces, grid colorings), the structure of $G$ allows computing $|\text{Fix}(g)|$ analytically, reducing the work to $O(m)$.

## Applications

| Problem | Group $G$ | Set $X$ |
|---|---|---|
| Necklace colorings | Cyclic group $C_n$ (rotations) | $k^n$ color assignments |
| Bracelet colorings | Dihedral group $D_n$ (rotations + reflections) | $k^n$ color assignments |
| Chemical isomer counting | Molecular symmetry group | Atom arrangements |
| Rubik's cube patterns | Cube rotation group | Face colorings |

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. Chapter 31.
- Burnside, W. (1897). *Theory of Groups of Finite Order*. Cambridge University Press.
