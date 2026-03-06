# Multipop Stack

Amortized analysis shows that even if individual operations are expensive, the average cost over a sequence of operations is low.

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

[Introduction to Algorithms (CLRS), Chapters 2-4](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
