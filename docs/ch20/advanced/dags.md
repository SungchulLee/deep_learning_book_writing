# DP on DAGs


<div align="center"><img src="https://media.geeksforgeeks.org/wp-content/uploads/multi-stage-graph.jpg" width="40%"></div>

[Multistage Graph (Shortest Path)](https://www.geeksforgeeks.org/multistage-graph-shortest-path/)


$$\begin{array}{lll}
\text{cost}(\text{edge}(i,j))&=&\text{Cost of travelling from $i$ to $j$}\\
\\
\text{cost}(i)&=&\text{Cost of travelling from $i$ to end}\\
\\
\text{cost}(i)&=&\min\left\{\text{cost}(\text{edge}(i,j))+\text{cost}(j)\right\}
\end{array}$$


```python
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
```


```python
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
```


```python
print(shortest_path(graph))
```

**Output:**
```
9
```


# Reference

[4.1 MultiStage Graph - Dynamic Programming](https://www.youtube.com/watch?v=9iE9Mj4m8jk&list=PLDN4rrl48XKpZkf03iYFl-O29szjTrs_O&index=47)

[4.1.1 MultiStage Graph (Program) - Dynamic Programming](https://www.youtube.com/watch?v=FcScLYJI42E&list=PLDN4rrl48XKpZkf03iYFl-O29szjTrs_O&index=48)

[Multistage graph](https://www.gdeepak.com/course/adslidesold/26ad.pdf)

[Multistage Graph (Shortest Path)](https://www.geeksforgeeks.org/multistage-graph-shortest-path/)
