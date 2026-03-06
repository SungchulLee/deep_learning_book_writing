# Python for Algorithms

Python is widely used for algorithm study due to its readable syntax and rich standard library.

## Key Features for Algorithms

```python
from collections import defaultdict, deque, Counter
from heapq import heappush, heappop
from bisect import bisect_left, bisect_right
from itertools import permutations, combinations
from functools import lru_cache

def demo_collections():
    # defaultdict — automatic default values
    graph = defaultdict(list)
    graph[0].append(1)
    graph[0].append(2)
    print(f"Graph: {dict(graph)}")

    # Counter — frequency counting
    freq = Counter("abracadabra")
    print(f"Frequency: {freq.most_common(3)}")

    # deque — O(1) append/pop from both ends
    q = deque([1, 2, 3])
    q.appendleft(0)
    q.append(4)
    print(f"Deque: {list(q)}")

def demo_heap():
    heap = []
    for val in [5, 3, 8, 1, 9]:
        heappush(heap, val)
    result = [heappop(heap) for _ in range(len(heap))]
    print(f"Heap sort: {result}")

def demo_bisect():
    arr = [1, 3, 5, 7, 9]
    pos = bisect_left(arr, 5)
    print(f"bisect_left({arr}, 5) = {pos}")

def main():
    demo_collections()
    demo_heap()
    demo_bisect()

if __name__ == "__main__":
    main()
```

**Output:**
```
Graph: {0: [1, 2]}
Frequency: [('a', 5), ('b', 2), ('r', 2)]
Deque: [0, 1, 2, 3, 4]
Heap sort: [1, 3, 5, 8, 9]
bisect_left([1, 3, 5, 7, 9], 5) = 2
```

# Reference

[Python Documentation — Data Structures](https://docs.python.org/3/tutorial/datastructures.html)

[Python Collections Module](https://docs.python.org/3/library/collections.html)
