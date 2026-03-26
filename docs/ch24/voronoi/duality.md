# Voronoi-Delaunay Duality

The Voronoi diagram and the Delaunay triangulation encode the same
geometric information in dual forms. Understanding this duality lets us
compute one from the other in $O(n)$ time and transfer properties freely
between the two structures. This connection is one of the most elegant
relationships in computational geometry.

## The Duality Correspondence

Given $n$ sites $P = \{p_1, \ldots, p_n\}$, the **Voronoi diagram**
$\text{Vor}(P)$ and the **Delaunay triangulation** $DT(P)$ are related
by the following mappings:

| Voronoi | Delaunay |
|---|---|
| Region $V(p_i)$ | Vertex $p_i$ |
| Edge between $V(p_i)$ and $V(p_j)$ | Edge $\overline{p_i p_j}$ |
| Vertex (meeting point of 3+ edges) | Triangle (circumscribed by 3+ sites) |

In short: faces become vertices, edges stay edges, and vertices become faces.

## Formal Statement

**Theorem.** Two sites $p_i$ and $p_j$ are connected by a Delaunay edge if
and only if their Voronoi regions share a boundary edge.

**Proof sketch.** If $V(p_i)$ and $V(p_j)$ share an edge $e$, then every
point on $e$ is equidistant from $p_i$ and $p_j$ and farther from all other
sites. The midpoint of $e$ (or any point on $e$) is the center of a circle
passing through $p_i$ and $p_j$ with no other site inside. This is exactly
the empty circumcircle condition for the Delaunay edge
$\overline{p_i p_j}$. $\square$

## Vertex-Triangle Correspondence

A Voronoi vertex $v$ where three regions $V(p_i)$, $V(p_j)$, $V(p_k)$ meet
is equidistant from $p_i$, $p_j$, $p_k$. Therefore $v$ is the circumcenter
of $\triangle p_i p_j p_k$, and this triangle appears in $DT(P)$.

$$
\|v - p_i\| = \|v - p_j\| = \|v - p_k\| = r
$$

where $r$ is the circumradius. The empty circumcircle property guarantees
that no other site lies inside the circle of radius $r$ centered at $v$.

## Constructing One from the Other

### Delaunay from Voronoi

Given $\text{Vor}(P)$:

1. For each Voronoi edge separating $V(p_i)$ and $V(p_j)$, add edge
   $\overline{p_i p_j}$ to $DT(P)$.

This takes $O(n)$ time since the Voronoi diagram has $O(n)$ edges.

### Voronoi from Delaunay

Given $DT(P)$:

1. For each Delaunay triangle, compute its circumcenter — this is a
   Voronoi vertex.
2. Connect circumcenters of adjacent triangles (those sharing a Delaunay
   edge) — these connections form Voronoi edges.

This also takes $O(n)$ time.

## Properties Transferred by Duality

!!! note "What Carries Over"
    Many properties have natural dual interpretations:

    - The **nearest neighbor** of $p_i$ (a Delaunay property) corresponds
      to the Voronoi region sharing the longest boundary with $V(p_i)$.
    - The **Euclidean MST** is a subgraph of $DT(P)$, so it can be found
      in $O(n \log n)$ time by computing $DT(P)$ first.
    - **Convex hull edges** of $P$ correspond to unbounded Voronoi edges.

## Worked Example

Consider four sites: $p_1 = (0, 0)$, $p_2 = (4, 0)$, $p_3 = (4, 4)$,
$p_4 = (0, 4)$.

**Delaunay triangulation** has two triangles: $\triangle p_1 p_2 p_3$
and $\triangle p_1 p_3 p_4$ (using diagonal $\overline{p_1 p_3}$).

**Voronoi vertices** are the circumcenters:
- $\triangle p_1 p_2 p_3$: circumcenter at $(2, 2)$
- $\triangle p_1 p_3 p_4$: circumcenter at $(2, 2)$

Since the four points are cocircular (all on a circle of radius $2\sqrt{2}$
centered at $(2, 2)$), both circumcenters coincide at $(2, 2)$, and this
single Voronoi vertex is where all four regions meet.

## Implementation

```python
"""
Voronoi-Delaunay duality: converting between the two structures.

Demonstrates the correspondence between Voronoi vertices and Delaunay
triangles via circumcenter computation.
"""

import math


# === Circumcenter ===

def circumcenter(a, b, c):
    """Compute the circumcenter of triangle abc.

    The circumcenter is the Voronoi vertex dual to this Delaunay triangle.
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
    return (ux, uy)


# === Duality Extraction ===

def delaunay_to_voronoi_vertices(triangles, points):
    """Extract Voronoi vertices from Delaunay triangles.

    Each Delaunay triangle corresponds to exactly one Voronoi vertex
    (its circumcenter).
    """
    vertices = []
    for tri in triangles:
        a, b, c = [points[i] for i in tri]
        cc = circumcenter(a, b, c)
        if cc is not None:
            vertices.append(cc)
    return vertices


def voronoi_edges_from_delaunay(triangles, points):
    """Extract Voronoi edges from adjacent Delaunay triangles.

    Two Delaunay triangles sharing an edge produce a Voronoi edge
    connecting their circumcenters.
    """
    # Build adjacency: edge -> list of triangle indices
    edge_to_tri = {}
    for idx, tri in enumerate(triangles):
        for i in range(3):
            edge = tuple(sorted([tri[i], tri[(i + 1) % 3]]))
            edge_to_tri.setdefault(edge, []).append(idx)

    voronoi_edges = []
    voronoi_vertices = delaunay_to_voronoi_vertices(triangles, points)

    for edge, tri_list in edge_to_tri.items():
        if len(tri_list) == 2:
            v1 = voronoi_vertices[tri_list[0]]
            v2 = voronoi_vertices[tri_list[1]]
            voronoi_edges.append((v1, v2))

    return voronoi_edges


# === Main ===

if __name__ == "__main__":
    # Five points
    points = [(0, 0), (4, 0), (4, 4), (0, 4), (2, 2)]

    # A valid Delaunay triangulation (indices into points)
    triangles = [
        (0, 1, 4),
        (1, 2, 4),
        (2, 3, 4),
        (3, 0, 4),
    ]

    print("Points:", points)
    print("Delaunay triangles:", triangles)

    # Voronoi vertices
    vv = delaunay_to_voronoi_vertices(triangles, points)
    print("\nVoronoi vertices (circumcenters):")
    for i, v in enumerate(vv):
        print(f"  Triangle {triangles[i]} -> ({v[0]:.3f}, {v[1]:.3f})")

    # Voronoi edges
    ve = voronoi_edges_from_delaunay(triangles, points)
    print(f"\nVoronoi edges ({len(ve)} edges):")
    for v1, v2 in ve:
        length = math.hypot(v2[0] - v1[0], v2[1] - v1[1])
        print(f"  ({v1[0]:.2f},{v1[1]:.2f}) -- ({v2[0]:.2f},{v2[1]:.2f})"
              f"  length={length:.3f}")
```

## Lifting to 3D

The duality has an elegant geometric interpretation via the **paraboloid
lifting**. Map each point $(x, y)$ to $(x, y, x^2 + y^2)$ on the paraboloid
$z = x^2 + y^2$. The lower convex hull of the lifted points projects down
to the Delaunay triangulation, while tangent planes to the paraboloid
correspond to Voronoi regions.

## Reference

- de Berg, M., Cheong, O., van Kreveld, M., & Overmars, M. *Computational Geometry: Algorithms and Applications*. Springer, Chapter 7 and 9.
- Edelsbrunner, H. & Seidel, R. "Voronoi Diagrams and Arrangements." *Discrete & Computational Geometry*, 1986.
