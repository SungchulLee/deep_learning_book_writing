# Circular Queue

A circular variant wraps around, connecting the end back to the beginning, enabling efficient use of fixed-size buffers.

```python
from collections import deque

q = deque()
for x in [1,2,3,4]: q.append(x)
while q:
    print(q.popleft(), end=" ")
print()
```

**Output:**
```
1 2 3 4
```

# Reference

[Introduction to Algorithms (CLRS), Chapter 10](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
