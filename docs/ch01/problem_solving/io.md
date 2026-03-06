# Input and Output

Algorithms transform **input** into **output**. Understanding the input/output relationship is crucial for algorithm design.

$$

f: \mathcal{I} \rightarrow \mathcal{O}

$$

where $\mathcal{I}$ is the set of valid inputs and $\mathcal{O}$ is the set of valid outputs.

## Input Types

| Type | Example | Typical Representation |
|---|---|---|
| Numbers | Find maximum | Array of integers |
| Strings | Pattern matching | Character array |
| Graphs | Shortest path | Adjacency list/matrix |
| Trees | LCA query | Parent array or nodes |

```python
import sys
from io import StringIO

def read_array(input_str):
    """Read array from string input."""
    data = input_str.strip().split()
    n = int(data[0])
    arr = list(map(int, data[1:n+1]))
    return arr

def main():
    test_input = "5\n3 1 4 1 5"
    arr = read_array(test_input.replace("\\n", "\n"))
    print(f"Read array: {arr}")
    print(f"Sum: {sum(arr)}")

if __name__ == "__main__":
    main()
```

**Output:**
```
Read array: [3, 1, 4, 1, 5]
Sum: 14
```

# Reference

[Competitive Programmer's Handbook - Chapter 1](https://cses.fi/book/book.pdf)
