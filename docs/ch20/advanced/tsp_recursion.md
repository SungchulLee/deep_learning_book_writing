# TSP - Recursion

Number the cities $1,2,\ldots,N$ and assume we start at city $1$, and the distance between city $i$ and city $j$ is $d_{i,j}$

# First Phase

Consider all the paths $\pi\in\Pi(i,S,j)$ starting from $i$ and ending at $j$
such that

$$\begin{array}{lll}
(1)&&\text{$\pi$ starts from $i$ and naver visits again}\\
(2)&&\text{$\pi$ visits all cities of $S$ ($i\notin S, j\notin S$) exactly once}\\
(3)&&\text{$\pi$ ends at $j$ and naver visited before}\\
\end{array}$$

$$

d(i,S,j)=\min_{\pi\in\Pi(i,S,j)}\left\{\text{cost}(\pi)\right\}

$$

If $S=\emptyset$,

$$

d(1,S,j)=d_{1,j}

$$

If $S\neq\emptyset$,

$$

d(1,S,j)=\min_{k\in S}\left\{d(1,S\setminus\{k\},k)+d_{k,j}\right\}

$$

# Second Phase

$$

\text{TSP COST} = \min_k\left\{d(1,\{1,2,3,4,\ldots,N\}\setminus\{1,k\},k)+d_{k,1}\right\}

$$

```python
def min_path_cost(start, mid_points, end, graph):
    """
    paths of interest : start ---> mid_points (set of vertices) ---> end
    """
    if len(mid_points) == 0:
        cost = graph[start][end]
    else:
        for i, mid_point in enumerate(mid_points):
            mid_points_ = mid_points - {mid_point}
            end_ = mid_point
            cost1 = min_path_cost(start, mid_points_, end_, graph)
            cost2 = graph[end_][end]
            if i == 0:
                cost = cost1 + cost2
            else:
                cost = min(cost, cost1 + cost2)
    return cost
```

```python
def tsp(graph):
    start = 0
    vertex = list(range(len(graph)))
    for i, end in enumerate(vertex[1:]):
        mid_points = set(vertex) - {start} - {end}
        cost = min_path_cost(start, mid_points, end, graph) + graph[end][start]
        if i == 0:
            tsp_cost = cost
        else:
            tsp_cost = min(tsp_cost, cost)
    return tsp_cost
```

```python
graph = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
]
```

```python
tsp_cost = tsp(graph)
print(f'tsp cost : {tsp_cost}')
```

**Output:**
```
tsp cost : 80
```

# Reference

[Held-Karp algorithm](https://en.wikipedia.org/wiki/Held%E2%80%93Karp_algorithm#Dynamic_programming_approach)

[4.7 [New] Traveling Salesman Problem - Dynamic Programming using Formula](https://www.youtube.com/watch?v=Q4zHb-Swzro&list=PLDN4rrl48XKpZkf03iYFl-O29szjTrs_O&index=58)

[Traveling Salesman Problem (TSP) Implementation](https://www.geeksforgeeks.org/traveling-salesman-problem-tsp-implementation/)
