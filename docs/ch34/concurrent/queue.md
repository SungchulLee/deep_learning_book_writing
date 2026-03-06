# Lock-Free Queue

Lock-free queue uses CAS operations for concurrent enqueue/dequeue.

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

[Advanced Data Structures (Brass)](https://www.cambridge.org/core/books/advanced-data-structures/D56E2269D7CEE969A3B8105D3541F601)
