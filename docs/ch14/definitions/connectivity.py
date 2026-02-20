"""
Connectivity
"""

from functools import lru_cache as cache


class Graph:
    
    def __init__(self, edges):
        self.edges = edges
        self.dict = self.edges_to_dict() 
        
    def edges_to_dict(self): 
        graph_dict = {}
        for start, end in self.edges:
            if start in graph_dict:
                graph_dict[start].append(end)
            else:
                graph_dict[start] = [end]
        return graph_dict 

    @cache
    def get_paths(self, start, end):
        if start == end: return [[start]]
        if start not in self.dict: return [[]]
        paths = []
        for node in self.dict[start]:
            new_paths = self.get_paths(node, end)
            for p in new_paths:
                paths.append([start]+p)
        return paths
    
    @cache
    def get_shortest_path(self, start, end):
        if start == end: return [start]
        if start not in self.dict: return []
        for idx, node in enumerate(self.dict[start]):
            path_segment = self.get_shortest_path(node, end)
            path_segment_length = len(path_segment)
            if idx==0:
                shortest_path_segment = path_segment
                shortest_path_segment_length = path_segment_length
            else:
                if path_segment_length < shortest_path_segment_length:
                    shortest_path_segment = path_segment
                    shortest_path_segment_length = path_segment_length
        return [start] + shortest_path_segment


if __name__ == '__main__':

    routes = [
        ("Mumbai", "Paris"),
        ("Mumbai", "Dubai"),
        ("Paris", "Dubai"),
        ("Paris", "New York"),
        ("Dubai", "New York"),
        ("New York", "Toronto"),
    ]
    
    route_graph = Graph(routes)

    start = "Mumbai"
    end = "New York"
    print(f"All paths between: {start} and {end}: ", route_graph.get_paths(start, end))
    print(f"Shortest path between {start} and {end}: ", route_graph.get_shortest_path(start,end))
