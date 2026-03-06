# DFS Properties

```python
def dfs(node):
    stack = [node] 
    visited = set([node]) 
    
    while stack:
        cur = stack.pop()
        for neighbour in cur.neighbours:
            if neighbour not in visited:  
                stack.append(neighbour)
                visited.add(neighbour)
```

# Reference

[Depth First Search in Data Structure | Graph Traversal | DFS Algorithm | C++ Java Python](https://www.youtube.com/watch?v=tXSk6POIBJA&list=PL1w8k37X_6L9IfRTVvL-tKnrZ_F-8HJQt&index=4)
