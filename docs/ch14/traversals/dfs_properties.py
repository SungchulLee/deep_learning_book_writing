"""
DFS Properties
"""

def dfs(node):
    stack = [node] 
    visited = set([node]) 
    
    while stack:
        cur = stack.pop()
        for neighbour in cur.neighbours:
            if neighbour not in visited:  
                stack.append(neighbour)
                visited.add(neighbour)
