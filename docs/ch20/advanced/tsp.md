# Traveling Salesman


```python
from itertools import permutations
```


```python
def cycle_cost(cycle, graph):
    cost = 0
    start = cycle[0]
    for end in cycle[1:]:
        cost += graph[start][end]
        start = end
    return cost
```


```python
def tsp(graph):
    start = 0
    vertex = list(range(len(graph)))[1:]
    for i, path in enumerate(permutations(vertex)):
        cycle = [start] + list(path) + [start] 
        current_cost = cycle_cost(cycle, graph)
        if i == 0:
            tsp_cost = current_cost
            tsp_cycle = cycle
        elif current_cost < tsp_cost:
            tsp_cost = current_cost
            tsp_cycle = cycle     
    return tsp_cost, tsp_cycle
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
tsp_cost, tsp_cycle = tsp(graph)
print(f'tsp cost  : {tsp_cost}')
print(f'tsp cycle : {tsp_cycle}')
```

**Output:**
```
tsp cost  : 80
tsp cycle : [0, 1, 3, 2, 0]
```


# Reference

[7.3 Traveling Salesman Problem - Branch and Bound](https://www.youtube.com/watch?v=1FEP_sNb62k&list=PLDN4rrl48XKpZkf03iYFl-O29szjTrs_O&index=71)

[Traveling Salesman Problem (TSP) Implementation](https://www.geeksforgeeks.org/traveling-salesman-problem-tsp-implementation/)
