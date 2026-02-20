# Priority Queue ADT

A priority queue serves elements by priority rather than insertion order, typically implemented with a heap.

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
