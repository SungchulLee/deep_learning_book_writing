"""
DP on DAGs
"""

# Graph stored in the form of an adjacency Matrix 
INF = 999999999999 
graph = [
    [INF, 1, 2, 5, INF, INF, INF, INF],
    [INF, INF, INF, INF, 4, 11, INF, INF],
    [INF, INF, INF, INF, 9, 5, 16, INF],
    [INF, INF, INF, INF, INF, INF, 2, INF],
    [INF, INF, INF, INF, INF, INF, INF, 18],
    [INF, INF, INF, INF, INF, INF, INF, 13],
    [INF, INF, INF, INF, INF, INF, INF, 2],
    [INF, INF, INF, INF, INF, INF, INF, INF] 
]


def shortest_path(graph):
    N = len(graph)
  
    # dist[i] is going to store shortest 
    # distance from node i to node N-1. 
    dist = [0] * N 
  
    dist[N - 1] = 0
  
    # Calculating shortest path 
    # for rest of the nodes 
    for i in range(N - 2, -1, -1):
  
        # Initialize distance from  
        # i to destination (N-1) 
        dist[i] = INF 
  
        # Check all nodes of next stages 
        # to find shortest distance from 
        # i to N-1.
        for j in range(N):
              
            # Reject if no edge exists 
            if graph[i][j] == INF:
                continue
  
            # We apply recursive equation to 
            # distance to target through j. 
            # and compare with minimum 
            # distance so far. 
            dist[i] = min(dist[i], 
                          graph[i][j] + dist[j])
  
    return dist[0]


print(shortest_path(graph))
