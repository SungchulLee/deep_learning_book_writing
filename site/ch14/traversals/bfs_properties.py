"""
BFS Properties
"""

# ======================================================================

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



# === Main ===
if __name__ == "__main__":
    pass
