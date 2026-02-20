# Pseudocode Conventions

Pseudocode provides a language-independent way to describe algorithms. It balances precision with readability.

## Common Conventions

$$
\begin{array}{ll}
\texttt{for } i = 1 \texttt{ to } n & \text{Loop from 1 to } n \\
\texttt{while } condition & \text{Loop while condition is true} \\
\texttt{if } condition & \text{Conditional execution} \\
\texttt{return } value & \text{Return a value} \\
A[i] & \text{Array access at index } i \\
\lfloor x \rfloor, \lceil x \rceil & \text{Floor and ceiling}
\end{array}
$$

## Example: Binary Search

```
BINARY-SEARCH(A, target)
  lo = 1
  hi = A.length
  while lo <= hi
    mid = floor((lo + hi) / 2)
    if A[mid] == target
      return mid
    else if A[mid] < target
      lo = mid + 1
    else
      hi = mid - 1
  return NIL
```

```python
def binary_search(a, target):
    lo, hi = 0, len(a) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if a[mid] == target:
            return mid
        elif a[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


def main():
    a = [2, 5, 8, 12, 16, 23, 38, 56, 72, 91]
    for target in [23, 50]:
        idx = binary_search(a, target)
        print(f"Search for {target}: index = {idx}")


if __name__ == "__main__":
    main()
```

**Output:**
```
Search for 23: index = 5
Search for 50: index = -1
```


# Reference

[Introduction to Algorithms (CLRS), Section 2.1](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
