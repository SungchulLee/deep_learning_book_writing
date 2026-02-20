# Problem Specification

A well-defined problem specification includes:

$$
\text{Problem} = (\text{Input}, \text{Output}, \text{Constraints})
$$

## Components

1. **Input description**: What data is given
2. **Output description**: What must be produced
3. **Constraints**: Bounds on input size, value ranges
4. **Preconditions**: What can be assumed about the input
5. **Postconditions**: What must be true about the output

## Example: Sorting Problem

- **Input**: A sequence of $n$ numbers $\langle a_1, a_2, \ldots, a_n \rangle$
- **Output**: A permutation $\langle a_1', a_2', \ldots, a_n' \rangle$ such that $a_1' \leq a_2' \leq \cdots \leq a_n'$
- **Constraint**: $1 \leq n \leq 10^6$

```python
def is_valid_sort(original, result):
    """Verify that result is a valid sorted version of original."""
    # Must be a permutation
    if sorted(original) != sorted(result):
        return False
    # Must be sorted
    for i in range(len(result) - 1):
        if result[i] > result[i + 1]:
            return False
    return True


def main():
    original = [3, 1, 4, 1, 5, 9]
    result = sorted(original)
    print(f"Original: {original}")
    print(f"Result:   {result}")
    print(f"Valid:    {is_valid_sort(original, result)}")


if __name__ == "__main__":
    main()
```

**Output:**
```
Original: [3, 1, 4, 1, 5, 9]
Result:   [1, 1, 3, 4, 5, 9]
Valid:    True
```


# Reference

[Introduction to Algorithms (CLRS), Chapter 1](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
