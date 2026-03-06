# Algorithm Definition

An **algorithm** is a well-defined computational procedure that takes some value, or set of values, as **input** and produces some value, or set of values, as **output** in a finite number of steps.

$$

\text{Algorithm}: \text{Input} \rightarrow \text{Output}

$$

## Formal Definition

An algorithm must satisfy:

1. **Input**: Zero or more quantities are externally supplied
2. **Output**: At least one quantity is produced
3. **Definiteness**: Each instruction is clear and unambiguous
4. **Finiteness**: The algorithm terminates after a finite number of steps
5. **Effectiveness**: Every instruction is basic enough to be carried out

```python
def find_maximum(arr):
    """A simple algorithm: find the maximum element in an array."""
    if not arr:
        return None
    max_val = arr[0]
    for val in arr[1:]:
        if val > max_val:
            max_val = val
    return max_val

def main():
    arr = [3, 7, 2, 9, 1, 5]
    print(f"Array: {arr}")
    print(f"Maximum: {find_maximum(arr)}")

if __name__ == "__main__":
    main()
```

**Output:**
```
Array: [3, 7, 2, 9, 1, 5]
Maximum: 9
```

# Reference

[Introduction to Algorithms (CLRS), Chapter 1](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)

[What is an Algorithm? - Khan Academy](https://www.khanacademy.org/computing/computer-science/algorithms)
