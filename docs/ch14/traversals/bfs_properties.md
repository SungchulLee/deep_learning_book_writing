# BFS Properties

```python
def bfs(node):
    from collections import deque
    queue = deque([node]) 
    visited = set([node]) 
    
    while queue:
        cur = queue.popleft()
        for neighbour in cur.neighbours:
            if neighbour not in visited:  
                queue.append(neighbour)
                visited.add(neighbour)
```

# Reference

[Breadth First Search in Data Structure | Graph Traversal | BFS Algorithm | C++ Java Python](https://www.youtube.com/watch?v=6J50_2SD0C8&list=PL1w8k37X_6L9IfRTVvL-tKnrZ_F-8HJQt&index=3)
