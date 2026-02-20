# Pigeonhole Principle

If $n$ items are placed into $m$ containers and $n > m$, then at least one container has more than one item.

$$
n \text{ pigeons into } m \text{ holes} \Rightarrow \exists \text{ hole with } \geq \lceil n/m \rceil \text{ pigeons}
$$

## Applications in Algorithms

- **Birthday paradox**: In a group of 23 people, there's a >50% chance two share a birthday
- **Hashing**: With $n > m$ keys and $m$ slots, collisions are guaranteed
- **Duplicate detection**: In an array of $n+1$ elements from $\{1, \ldots, n\}$, a duplicate must exist

```python
def find_duplicate(arr):
    """
    Find duplicate in array of n+1 integers in range [1, n].
    Uses Floyd's cycle detection (pigeonhole guarantees a duplicate).
    O(n) time, O(1) space.
    """
    slow = fast = 0
    while True:
        slow = arr[slow]
        fast = arr[arr[fast]]
        if slow == fast:
            break
    slow = 0
    while slow != fast:
        slow = arr[slow]
        fast = arr[fast]
    return slow


def main():
    # 5 elements in range [1, 4] — duplicate guaranteed
    arr = [1, 3, 4, 2, 2]
    print(f"Array: {arr}")
    print(f"Duplicate: {find_duplicate(arr)}")

    arr = [3, 1, 3, 4, 2]
    print(f"Array: {arr}")
    print(f"Duplicate: {find_duplicate(arr)}")


if __name__ == "__main__":
    main()
```

**Output:**
```
Array: [1, 3, 4, 2, 2]
Duplicate: 2
Array: [3, 1, 3, 4, 2]
Duplicate: 3
```


# Reference

[Discrete Mathematics and Its Applications (Rosen), Chapter 6](https://www.mhprofessional.com/discrete-mathematics-and-its-applications)
