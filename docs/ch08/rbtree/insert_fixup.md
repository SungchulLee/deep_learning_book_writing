# Insertion Fixup


!!! warning "Incomplete page"
    This page is missing the required five-section structure (Concept Definition, Explanation, Diagram / Example). Content needs to be reorganized and expanded.

B-trees are self-balancing trees optimized for systems that read/write large blocks of data.

```python
class Node:
    def __init__(self, val, nxt=None):
        self.val = val
        self.next = nxt

def print_list(head):
    vals = []
    while head:
        vals.append(str(head.val))
        head = head.next
    print(" -> ".join(vals))

head = Node(1, Node(2, Node(3, Node(4))))
print_list(head)
```

**Output:**
```
1 -> 2 -> 3 -> 4
```

# Reference

[Introduction to Algorithms (CLRS), Chapters 13-14](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
