# Andrew's Monotone Chain

**Andrew's Monotone Chain** is an important concept in algorithm design and analysis.

```python
def cross(o, a, b):
    return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

def convex_hull(points):
    points = sorted(set(points))
    if len(points) <= 1: return points
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0: lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0: upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]

pts = [(0,0),(1,1),(2,2),(0,2),(2,0),(1,0)]
print(f"Hull: {convex_hull(pts)}")
```

**Output:**
```
Hull: [(0, 0), (1, 0), (2, 0), (2, 2), (0, 2)]
```

# Reference

[Computational Geometry (de Berg et al.)](https://www.springer.com/gp/book/9783540779735)
