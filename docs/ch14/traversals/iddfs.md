# Iterative Deepening

**Iterative Deepening** is an important concept in algorithm design and analysis.

$$T(V, E) = O(V + E)$$

```python
def dfs(graph, start):
    visited, stack = set(), [start]
    order = []
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            order.append(node)
            for neighbor in reversed(graph[node]):
                if neighbor not in visited:
                    stack.append(neighbor)
    return order

graph = {0:[1,2], 1:[3], 2:[3,4], 3:[], 4:[]}
print(f"DFS from 0: {dfs(graph, 0)}")
```

**Output:**
```
DFS from 0: [0, 1, 3, 2, 4]
```


# Reference

[Introduction to Algorithms (CLRS), Chapter 22](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
