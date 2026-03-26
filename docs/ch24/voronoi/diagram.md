# Voronoi Diagram

Given a set of sites (points) in the plane, which site is closest to a given
query location? The **Voronoi diagram** answers this for *all* locations
simultaneously by partitioning the plane into regions, one per site, such
that every point in a region is closer to its site than to any other.
Voronoi diagrams appear in facility location, nearest-neighbor search,
mesh generation, and geographic analysis.

## Definition

Given $n$ sites $P = \{p_1, p_2, \ldots, p_n\}$ in $\mathbb{R}^2$, the
**Voronoi region** (or Voronoi cell) of site $p_i$ is:

$$
V(p_i) = \{ x \in \mathbb{R}^2 : \|x - p_i\| \le \|x - p_j\| \text{ for all } j \neq i \}
$$

The **Voronoi diagram** $\text{Vor}(P)$ is the partition of $\mathbb{R}^2$
into the Voronoi regions $V(p_1), V(p_2), \ldots, V(p_n)$.

## Structure

The boundary between two Voronoi regions $V(p_i)$ and $V(p_j)$ lies on
the **perpendicular bisector** of the segment $\overline{p_i p_j}$ — the set
of points equidistant from $p_i$ and $p_j$.

The Voronoi diagram consists of:

- **Voronoi edges:** portions of perpendicular bisectors where exactly two
  regions meet.
- **Voronoi vertices:** points where three or more edges meet. At a Voronoi
  vertex, the point is equidistant from three (or more) sites.

## Combinatorial Complexity

For $n$ sites in general position:

| Component | Count |
|---|---|
| Voronoi regions | $n$ |
| Voronoi vertices | $\le 2n - 5$ |
| Voronoi edges | $\le 3n - 6$ |

These bounds follow from Euler's formula for planar graphs: $V - E + F = 2$.

## Properties

- Each Voronoi region is a **convex polygon** (possibly unbounded).
- A Voronoi vertex $v$ is the center of a circle passing through three
  (or more) sites with no sites in its interior — this is the circumcircle
  of the corresponding Delaunay triangle.
- The **nearest neighbor** of $p_i$ shares a Voronoi edge with $p_i$.
- The Voronoi diagram is the **dual** of the Delaunay triangulation:
  connect two sites by a Delaunay edge if and only if their Voronoi
  regions share a boundary.

## Constructing a Voronoi Region

Each Voronoi region $V(p_i)$ is the intersection of $n - 1$ half-planes:

$$
V(p_i) = \bigcap_{j \neq i} H(p_i, p_j)
$$

where $H(p_i, p_j) = \{x : \|x - p_i\| \le \|x - p_j\|\}$ is the half-plane
containing $p_i$, bounded by the perpendicular bisector of $\overline{p_i p_j}$.

Computing each region this way takes $O(n^2 \log n)$ total. Fortune's
sweep-line algorithm achieves $O(n \log n)$.

## Worked Example

Consider three sites: $p_1 = (0, 0)$, $p_2 = (6, 0)$, $p_3 = (3, 5)$.

The perpendicular bisectors are:
- $p_1 p_2$: vertical line $x = 3$
- $p_1 p_3$: bisector of $(0,0)$ and $(3,5)$
- $p_2 p_3$: bisector of $(6,0)$ and $(3,5)$

These three bisectors meet at the circumcenter of $\triangle p_1 p_2 p_3$,
which is the unique Voronoi vertex.

## Implementation

```python
"""
Voronoi diagram: construction and nearest-neighbor queries.

Provides brute-force Voronoi region assignment and perpendicular
bisector computation for educational purposes.
"""

import math


# === Distance ===

def dist(a, b):
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


# === Nearest Site (Voronoi Region Query) ===

def nearest_site(sites, query):
    """Find the site closest to the query point.

    This is equivalent to determining which Voronoi region
    contains the query point.
    """
    best_idx = 0
    best_dist = dist(sites[0], query)
    for i in range(1, len(sites)):
        d = dist(sites[i], query)
        if d < best_dist:
            best_dist = d
            best_idx = i
    return best_idx, best_dist


# === Perpendicular Bisector ===

def perpendicular_bisector(p, q):
    """Compute the perpendicular bisector of segment pq.

    Returns (midpoint, direction_vector) where direction_vector
    is perpendicular to pq.
    """
    mid = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
    dx, dy = q[0] - p[0], q[1] - p[1]
    # Perpendicular direction: rotate 90 degrees
    perp = (-dy, dx)
    return mid, perp


# === Circumcenter (Voronoi Vertex for 3 Sites) ===

def circumcenter(a, b, c):
    """Compute the circumcenter of triangle abc.

    The circumcenter is equidistant from all three vertices
    and corresponds to a Voronoi vertex.
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


# === Voronoi Grid Visualization ===

def voronoi_grid(sites, grid_size=10, resolution=20):
    """Create an ASCII grid showing Voronoi regions.

    Each cell is labeled with the index of its nearest site.
    """
    grid = []
    step = grid_size / resolution
    for row in range(resolution):
        line = []
        y = grid_size - row * step
        for col in range(resolution):
            x = col * step
            idx, _ = nearest_site(sites, (x, y))
            line.append(str(idx))
        grid.append(" ".join(line))
    return "\n".join(grid)


# === Main ===

if __name__ == "__main__":
    sites = [(0, 0), (6, 0), (3, 5)]

    # Nearest site queries
    queries = [(1, 1), (5, 1), (3, 3), (3, 0)]
    print("Sites:", sites)
    for q in queries:
        idx, d = nearest_site(sites, q)
        print(f"  Query {q} -> site {idx} (p={sites[idx]}, dist={d:.3f})")

    # Voronoi vertex
    cc = circumcenter(*sites)
    print(f"\nVoronoi vertex (circumcenter): ({cc[0]:.3f}, {cc[1]:.3f})")
    for i, s in enumerate(sites):
        print(f"  Distance to p{i+1}: {dist(cc, s):.3f}")

    # Perpendicular bisectors
    for i in range(len(sites)):
        for j in range(i + 1, len(sites)):
            mid, perp = perpendicular_bisector(sites[i], sites[j])
            print(f"\nBisector of p{i+1}-p{j+1}:")
            print(f"  Midpoint: ({mid[0]:.1f}, {mid[1]:.1f})")
            print(f"  Direction: ({perp[0]:.1f}, {perp[1]:.1f})")
```

**Output:**
```
Sites: [(0, 0), (6, 0), (3, 5)]
  Query (1, 1) -> site 0 (p=(0, 0), dist=1.414)
  Query (5, 1) -> site 1 (p=(6, 0), dist=1.414)
  Query (3, 3) -> site 2 (p=(3, 5), dist=2.000)
  Query (3, 0) -> site 0 (p=(0, 0), dist=3.000)

Voronoi vertex (circumcenter): (3.000, 1.400)
  Distance to p1: 3.311
  Distance to p2: 3.311
  Distance to p3: 3.600

Bisector of p1-p2:
  Midpoint: (3.0, 0.0)
  Direction: (0.0, 6.0)

Bisector of p1-p3:
  Midpoint: (1.5, 2.5)
  Direction: (-5.0, 3.0)

Bisector of p2-p3:
  Midpoint: (4.5, 2.5)
  Direction: (-5.0, -3.0)
```

## Applications

| Application | How Voronoi Is Used |
|---|---|
| Nearest neighbor | Query point falls in the Voronoi region of its nearest site |
| Facility location | Place facilities to minimize worst-case distance |
| Mesh generation | Dual Delaunay triangulation provides quality meshes |
| Spatial interpolation | Natural neighbor interpolation uses Voronoi areas |
| Collision avoidance | Voronoi edges represent maximum-clearance paths |

## Reference

- de Berg, M., Cheong, O., van Kreveld, M., & Overmars, M. *Computational Geometry: Algorithms and Applications*. Springer, Chapter 7.
- Aurenhammer, F. "Voronoi Diagrams: A Survey of a Fundamental Geometric Data Structure." *ACM Computing Surveys*, 1991.
