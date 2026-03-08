"""
TSP - Recursion
"""

# ======================================================================

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


# === Main ===
if __name__ == "__main__":
    graph = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ]


    tsp_cost = tsp(graph)
    print(f'tsp cost : {tsp_cost}')
