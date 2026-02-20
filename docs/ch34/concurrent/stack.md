# Lock-Free Stack

Lock-free stack uses CAS (Compare-And-Swap) for thread-safe push/pop without locks.

```python
class Stack:
    def __init__(self):
        self.items = []
    def push(self, x): self.items.append(x)
    def pop(self): return self.items.pop()
    def peek(self): return self.items[-1]
    def is_empty(self): return len(self.items) == 0

s = Stack()
for x in [1,2,3]: s.push(x)
while not s.is_empty(): print(s.pop(), end=" ")
print()
```

**Output:**
```
3 2 1
```


# Reference

[Advanced Data Structures (Brass)](https://www.cambridge.org/core/books/advanced-data-structures/D56E2269D7CEE969A3B8105D3541F601)
