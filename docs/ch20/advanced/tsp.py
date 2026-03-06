"""
Traveling Salesman
"""

from itertools import permutations


# === Functions ===

def cycle_cost(cycle, graph):
    cost = 0
    start = cycle[0]
    for end in cycle[1:]:
        cost += graph[start][end]
        start = end
    return cost


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


# === Main ===
if __name__ == "__main__":
    graph = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ]


    tsp_cost, tsp_cycle = tsp(graph)
    print(f'tsp cost  : {tsp_cost}')
    print(f'tsp cycle : {tsp_cycle}')
