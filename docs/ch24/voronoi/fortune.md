# Fortune's Algorithm

Computing a Voronoi diagram by intersecting half-planes takes $O(n^2 \log n)$
time. Fortune's algorithm uses the sweep-line paradigm to construct the
Voronoi diagram of $n$ sites in $O(n \log n)$ time and $O(n)$ space,
matching the lower bound. The key innovation is the **beach line** — a
curve of parabolic arcs that tracks the frontier of the known diagram as
the sweep line advances.

## High-Level Idea

A vertical sweep line moves from left to right. At any position, the
portion of the Voronoi diagram to the left of the sweep line is fully
determined. The **beach line** is the boundary between the determined and
undetermined regions.

!!! note "Beach Line Definition"
    The beach line is the locus of points equidistant from a site to the
    left and the sweep line itself. For each site, this locus is a
    parabola, so the beach line is a sequence of parabolic arcs.

## Parabolas and the Beach Line

A parabola is the set of points equidistant from a point (the focus,
which is a site) and a line (the directrix, which is the sweep line).
If the sweep line is at $x = l$ and the site is at $(s_x, s_y)$, then
a point $(x, y)$ on the parabola satisfies:

$$
(x - s_x)^2 + (y - s_y)^2 = (x - l)^2
$$

Expanding and solving for $x$:

$$
x = \frac{(y - s_y)^2}{2(s_x - l)} + \frac{s_x + l}{2}
$$

The beach line at sweep position $l$ is the lower envelope of all such
parabolas (one per site to the left of $l$).

## Events

Fortune's algorithm processes two types of events:

### Site Events

When the sweep line reaches a new site $p$:

1. A new parabolic arc for $p$ appears on the beach line (initially
   a degenerate vertical ray).
2. The arc splits an existing arc into two pieces.
3. New **breakpoints** emerge between the new arc and the split arcs.
4. Check for potential circle events involving the new arc and its neighbors.

### Circle Events

A circle event occurs when three consecutive arcs on the beach line
converge to a single point — their three defining sites are cocircular,
and the center of that circle becomes a Voronoi vertex.

When a circle event fires:

1. The middle arc vanishes from the beach line.
2. A new Voronoi vertex is created at the circle center.
3. Two half-edges (Voronoi edges) are completed.
4. A new breakpoint emerges, and new potential circle events are checked.

!!! tip "Detecting Circle Events"
    Three consecutive arcs defined by sites $p_i, p_j, p_k$ generate a
    circle event only if $p_i, p_j, p_k$ are not collinear and the
    rightmost point of their circumcircle lies to the right of the
    sweep line.

## Data Structures

| Structure | Purpose | Operations |
|---|---|---|
| Event queue (priority queue) | Stores site and circle events, ordered by $x$ | Insert, delete-min: $O(\log n)$ |
| Beach line (balanced BST) | Stores parabolic arcs by $y$-position | Insert, delete, find-arc: $O(\log n)$ |
| DCEL (doubly connected edge list) | Stores the Voronoi diagram | Edge/face operations: $O(1)$ |

## Algorithm Pseudocode

1. Insert all site events into the event queue.
2. While the event queue is not empty:
    - If the next event is a **site event** for site $p$:
        - Find the arc on the beach line directly above $p$.
        - Split that arc and insert the new arc for $p$.
        - Start two new half-edges (Voronoi edges).
        - Check for circle events involving the new and adjacent arcs.
    - If the next event is a **circle event**:
        - Remove the vanishing arc from the beach line.
        - Add the Voronoi vertex to the diagram.
        - Complete half-edges and start new ones.
        - Check for new circle events among newly adjacent arcs.
3. Complete all remaining half-edges (they extend to infinity).

## Complexity Analysis

Each site causes one site event and at most one circle event creation per
neighbor check. The total number of events is $O(n)$.

- Site events: $n$, each processed in $O(\log n)$.
- Circle events: $O(n)$, each processed in $O(\log n)$.

$$
T(n) = O(n \log n)
$$

Space: $O(n)$ for the beach line, event queue, and output diagram.

## Worked Example

Three sites: $A = (1, 3)$, $B = (3, 1)$, $C = (5, 4)$.

1. **Site event at $x = 1$:** Arc for $A$ spans the entire beach line.
2. **Site event at $x = 3$:** Arc for $B$ splits $A$'s arc. Beach line:
   $A$-$B$-$A$. Two breakpoints track the emerging Voronoi edge between
   $V(A)$ and $V(B)$.
3. **Site event at $x = 5$:** Arc for $C$ splits one of $A$'s arcs.
   A circle event is created for the triple $(A, B, A)$ or $(B, A, C)$
   depending on the configuration.
4. **Circle event:** When processed, a Voronoi vertex is placed at the
   circumcenter of the three relevant sites.

## Implementation Notes

```python
"""
Fortune's algorithm: conceptual components.

Provides the parabola and circumcircle computations used in Fortune's
sweep-line algorithm for Voronoi diagrams. A full implementation requires
a balanced BST for the beach line and careful DCEL bookkeeping.
"""

import math


# === Parabola Computation ===

def parabola_x(site, sweep_x, y):
    """Compute the x-coordinate on the parabola at height y.

    The parabola has focus at site and directrix at x = sweep_x.
    """
    sx, sy = site
    if abs(sx - sweep_x) < 1e-10:
        return sx
    return ((y - sy) ** 2) / (2 * (sx - sweep_x)) + (sx + sweep_x) / 2


# === Breakpoint Computation ===

def breakpoint_y(s1, s2, sweep_x):
    """Compute the y-coordinate where two parabolas intersect.

    Returns the y-values of the intersection(s).
    """
    x1, y1 = s1
    x2, y2 = s2

    if abs(x1 - x2) < 1e-10:
        return [(y1 + y2) / 2]

    # Coefficients of quadratic in y
    a1 = 1 / (2 * (x1 - sweep_x))
    b1 = -y1 / (x1 - sweep_x)
    c1 = (y1 * y1 / (2 * (x1 - sweep_x))) + (x1 + sweep_x) / 2

    a2 = 1 / (2 * (x2 - sweep_x))
    b2 = -y2 / (x2 - sweep_x)
    c2 = (y2 * y2 / (2 * (x2 - sweep_x))) + (x2 + sweep_x) / 2

    a = a1 - a2
    b = b1 - b2
    c = c1 - c2

    if abs(a) < 1e-10:
        if abs(b) < 1e-10:
            return []
        return [-c / b]

    disc = b * b - 4 * a * c
    if disc < 0:
        return []

    sqrt_disc = math.sqrt(disc)
    return [(-b + sqrt_disc) / (2 * a), (-b - sqrt_disc) / (2 * a)]


# === Circle Event Detection ===

def circumcircle(a, b, c):
    """Compute the circumcircle of three points.

    Returns (center_x, center_y, radius) or None if collinear.
    """
    ax, ay = a
    bx, by = b
    cx, cy = c

    D = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(D) < 1e-10:
        return None

    ux = ((ax * ax + ay * ay) * (by - cy)
        + (bx * bx + by * by) * (cy - ay)
        + (cx * cx + cy * cy) * (ay - by)) / D
    uy = ((ax * ax + ay * ay) * (cx - bx)
        + (bx * bx + by * by) * (ax - cx)
        + (cx * cx + cy * cy) * (bx - ax)) / D
    r = math.hypot(ux - ax, uy - ay)

    return (ux, uy, r)


# === Main ===

if __name__ == "__main__":
    A, B, C = (1, 3), (3, 1), (5, 4)

    # Circumcircle (potential circle event)
    cc = circumcircle(A, B, C)
    print(f"Sites: A={A}, B={B}, C={C}")
    print(f"Circumcircle: center=({cc[0]:.3f}, {cc[1]:.3f}), r={cc[2]:.3f}")
    print(f"Circle event x = {cc[0] + cc[2]:.3f}")

    # Parabola values at sweep position x=4
    sweep = 4.0
    for y in [0, 1, 2, 3, 4, 5]:
        xA = parabola_x(A, sweep, y)
        xB = parabola_x(B, sweep, y)
        print(f"  y={y}: parab_A={xA:.2f}, parab_B={xB:.2f}")

    # Breakpoints
    bp = breakpoint_y(A, B, sweep)
    print(f"Breakpoints of A and B at sweep={sweep}: y={bp}")
```

## Reference

- Fortune, S. "A Sweepline Algorithm for Voronoi Diagrams." *Algorithmica*, 1987.
- de Berg, M., Cheong, O., van Kreveld, M., & Overmars, M. *Computational Geometry: Algorithms and Applications*. Springer, Chapter 7.
