# Successor and Predecessor

In-order successor is the next node in sorted order; predecessor is the previous.

```python
class Node:
    def __init__(self, key):
        self.key = key; self.left = self.right = None

def insert(root, key):
    if not root: return Node(key)
    if key < root.key: root.left = insert(root.left, key)
    else: root.right = insert(root.right, key)
    return root

def inorder(root):
    if root: yield from inorder(root.left); yield root.key; yield from inorder(root.right)

root = None
for k in [5,3,7,1,4,6,8]: root = insert(root, k)
print(list(inorder(root)))
```

**Output:**
```
[1, 3, 4, 5, 6, 7, 8]
```


# Reference

[Introduction to Algorithms (CLRS), Chapter 12](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
